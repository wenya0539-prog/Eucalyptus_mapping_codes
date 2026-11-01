# -*- coding: utf-8 -*-
"""
Temporal post-processing of annual Eucalyptus maps on a southern-China Albers Equal Area grid.

Implements the manuscript's explicit cumulative-occurrence rule:
  - pixels occurring as Eucalyptus in >= 3 years across 2001-2020 are temporally eligible;
  - for pixels occurring in only 1-2 years, relaxed rules apply only at the first/last
    two years of the full time series (2001-2002 and 2019-2020);
  - for each province-year, the eligible forest pixels with the highest RF probabilities
    are selected according to the provincial area prior.

Important assumptions:
  1. The area CSV's province columns are in 10,000 hectares (万公顷). If not, change
     AREA_VALUE_TO_M2 below.
  2. Probability rasters contain RF probabilities on a 0-1 scale, as described in the
     manuscript. Values outside [0, 1] are treated as invalid/NoData.
  3. Annual final maps use 1 = Eucalyptus, 0 = other forest, and other values/NoData
     = outside the mapped forest domain.

All yearly input rasters are aligned to a common grid covering the union of their
spatial extents. Probability ranking remains the primary criterion. Only candidates tied
at the K-th probability are resolved by spatial support from the same-year higher-probability
eligible pixels: first count in the 3x3 neighbourhood, then count in the outer ring of a
5x5 neighbourhood. If candidates remain indistinguishable after these contextual criteria,
a fixed year-specific random seed chooses among them without raster-order bias.

Outputs are written to a separate post-processing folder and do NOT overwrite the initial
prediction rasters. Run 001_reproject_to_south_china_albers.py first. Projected rasters are read from
D:\\Eucalyptus_mapping\\outputs\\albers\\<province> and checked for the Albers Equal Area CRS.
"""

from pathlib import Path
import gc
import math
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling, transform_bounds

# =============================================================================
# 1. Configuration
# =============================================================================
PROJECT_ROOT = Path("D:/Eucalyptus_mapping")
province = "chongqing"  # 每次运行一个省份；需与 001 中的省份保持一致
PROVINCE_NAMES = {
    "guangxi": "广西", "guangdong": "广东", "hainan": "海南",
    "yunnan": "云南", "sichuan": "四川", "guizhou": "贵州",
    "chongqing": "重庆", "fujian": "福建", "jiangxi": "江西",
    "hunan": "湖南", "zhejiang": "浙江",
}
if province not in PROVINCE_NAMES:
    raise ValueError(f"未知省份代码：{province}")
province_cn = PROVINCE_NAMES[province]
YEARS = list(range(2001, 2021))

base_dir = PROJECT_ROOT / "outputs" / "albers" / province
csv_path = PROJECT_ROOT / "data" / "input" / "Eucalyptus_area.csv"

PIXEL_SIZE_M = 30.0
PIXEL_AREA_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M
AREA_VALUE_TO_M2 = 100_000_000.0  # 万公顷 -> m²; change to 10_000 if the CSV is in ha
NODATA_OUT = 255
TEMP_NODATA = -9999.0

# All outputs from this comparison run are isolated from existing postprocessed files.
OUTPUT_SUBDIR = "postprocessed_equal_area_neighbor"
TIE_CHUNK_SIZE = 500_000
RANDOM_TIE_SEED_BASE = 20261015

# The manuscript describes RF probability values on a 0-1 scale. Values outside this
# range (including 255 used by some exports as NoData) are not eligible candidates.
PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0

# =============================================================================
# 2. Read and validate the area table
# =============================================================================
if not csv_path.exists():
    raise FileNotFoundError(f"面积统计表不存在：{csv_path}")

area_df = pd.read_csv(csv_path, encoding="GBK")
required_columns = {"Year", province_cn}
missing_columns = required_columns.difference(area_df.columns)
if missing_columns:
    raise ValueError(f"面积 CSV 缺少必要字段：{sorted(missing_columns)}")

area_df["Year"] = pd.to_numeric(area_df["Year"], errors="coerce")
area_df = area_df.loc[area_df["Year"].isin(YEARS)].copy()
if area_df["Year"].duplicated().any():
    duplicated = area_df.loc[area_df["Year"].duplicated(keep=False), "Year"].tolist()
    raise ValueError(f"面积 CSV 中存在重复年份：{duplicated}")
area_df = area_df.set_index("Year")
missing_years = sorted(set(YEARS).difference(area_df.index.astype(int).tolist()))
if missing_years:
    raise ValueError(f"面积 CSV 缺少这些年份：{missing_years}")

area_values = {}
for year in YEARS:
    value = pd.to_numeric(area_df.loc[year, province_cn], errors="coerce")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{year} 年 {province_cn} 面积无效：{value}")
    area_values[year] = float(value)

# =============================================================================
# 3. Define a common grid covering the union of annual final-map extents
# =============================================================================
final_paths = {
    year: base_dir / f"{province}_{year}" / f"{province}_euca_final_{year}.tif"
    for year in YEARS
}
missing_final = [str(p) for p in final_paths.values() if not p.exists()]
if missing_final:
    raise FileNotFoundError(
        "时序筛选需要完整的 2001-2020 年 final 栅格。缺少文件：\n" + "\n".join(missing_final)
    )

anchor_path = final_paths[YEARS[0]]
with rasterio.open(anchor_path) as anchor:
    if anchor.crs is None:
        raise ValueError(f"参考栅格没有 CRS：{anchor_path}")
    anchor_crs = anchor.crs
    anchor_transform = anchor.transform
    anchor_profile = anchor.profile.copy()

# The area formula below assumes projected coordinates in metres. This check does not
# establish equal-area properties; use check_raster_projection.py for ground-area distortion.
try:
    from pyproj import CRS as PyprojCRS
    anchor_pycrs = PyprojCRS.from_user_input(anchor_crs)
    if not anchor_pycrs.is_projected:
        raise ValueError(
            f"参考栅格 CRS 不是投影坐标系：{anchor_crs}。请先统一到以米为单位的投影网格。"
        )
    unit_factor = anchor_pycrs.axis_info[0].unit_conversion_factor
    if unit_factor is None or not math.isclose(float(unit_factor), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"参考栅格坐标单位不是米（unit conversion factor={unit_factor}）。"
            "当前 30-m/900-m² 假设不成立。"
        )
    operation = getattr(anchor_pycrs, "coordinate_operation", None)
    operation_name = (getattr(operation, "method_name", "") or "").lower()
    if "albers" not in operation_name or "equal area" not in operation_name:
        raise ValueError(
            "输入栅格不是预期的 Albers Equal Area 投影。"
            f"检测到的投影方法为：{getattr(operation, 'method_name', 'unknown')}。"
            "请先运行 001_reproject_to_south_china_albers.py，并检查 base_dir 指向统一输出目录。"
        )
except ImportError as exc:
    raise ImportError("需要 pyproj 来检查投影坐标单位，请在当前 Python 环境安装 pyproj。") from exc

if not (math.isclose(anchor_transform.b, 0.0) and math.isclose(anchor_transform.d, 0.0)):
    raise ValueError("参考栅格存在旋转/斜切仿射变换；请先统一到无旋转的 30-m 网格。")
if anchor_transform.a <= 0 or anchor_transform.e >= 0:
    raise ValueError("当前脚本要求北向上的栅格网格（x 像元尺寸为正、y 像元尺寸为负）。")

res_x = float(anchor_transform.a)
res_y = abs(float(anchor_transform.e))
if not (math.isclose(res_x, PIXEL_SIZE_M, rel_tol=0, abs_tol=0.05) and
        math.isclose(res_y, PIXEL_SIZE_M, rel_tol=0, abs_tol=0.05)):
    raise ValueError(
        f"参考栅格分辨率为 {res_x} × {res_y} m，不是预期的 30 m；请检查输入数据或修改 PIXEL_SIZE_M。"
    )

min_x = min_y = float("inf")
max_x = max_y = float("-inf")
for year, path in final_paths.items():
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"栅格没有 CRS：{path}")
        if src.crs == anchor_crs:
            left, bottom, right, top = (
                src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            )
        else:
            left, bottom, right, top = transform_bounds(
                src.crs, anchor_crs, *src.bounds, densify_pts=21
            )
        min_x = min(min_x, left)
        min_y = min(min_y, bottom)
        max_x = max(max_x, right)
        max_y = max(max_y, top)

# Snap the union extent to the anchor grid so all annual rasters share one alignment.
base_x0 = float(anchor_transform.c)
base_y0 = float(anchor_transform.f)
col_min = math.floor((min_x - base_x0) / res_x)
row_min = math.floor((base_y0 - max_y) / res_y)
grid_x0 = base_x0 + col_min * res_x
grid_y0 = base_y0 - row_min * res_y
grid_width = int(math.ceil((max_x - grid_x0) / res_x - 1e-9))
grid_height = int(math.ceil((grid_y0 - min_y) / res_y - 1e-9))
if grid_width <= 0 or grid_height <= 0:
    raise ValueError("根据年度栅格范围构建公共网格失败，请检查输入栅格的 CRS 和范围。")

grid_transform = Affine(res_x, 0.0, grid_x0, 0.0, -res_y, grid_y0)
output_profile = anchor_profile.copy()
output_profile.update(
    driver="GTiff",
    width=grid_width,
    height=grid_height,
    transform=grid_transform,
    crs=anchor_crs,
    count=1,
    dtype="uint8",
    nodata=NODATA_OUT,
    compress="lzw",
)

print(f"研究区：{province_cn}；时段：{YEARS[0]}–{YEARS[-1]}")
print(f"公共网格：{grid_width} 列 × {grid_height} 行；分辨率：{res_x:g} m")
print("年度栅格将统一对齐至所有年份 final 栅格范围的并集。")

# =============================================================================
# 4. Helpers: align categorical and probability rasters to the common grid
# =============================================================================
def read_final_aligned(path: Path) -> np.ndarray:
    """Read a 0/1 final map onto the common grid; return uint8 with 255 elsewhere."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"栅格没有 CRS：{path}")
        source = src.read(1)
        source_valid = np.isin(source, (0, 1))
        source_prepared = np.full(source.shape, TEMP_NODATA, dtype=np.float32)
        source_prepared[source_valid] = source[source_valid].astype(np.float32)

        destination = np.full((grid_height, grid_width), NODATA_OUT, dtype=np.uint8)
        reproject(
            source=source_prepared,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=TEMP_NODATA,
            dst_transform=grid_transform,
            dst_crs=anchor_crs,
            dst_nodata=NODATA_OUT,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )
    return destination


def read_probability_aligned(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 0-1 probability raster onto the common grid and return data plus valid mask."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"概率栅格没有 CRS：{path}")
        raw = src.read(1)
        source = raw.astype(np.float32, copy=False)
        source_valid = np.isfinite(source)

        if src.nodata is not None:
            try:
                nodata_is_nan = np.isnan(src.nodata)
            except TypeError:
                nodata_is_nan = False
            if nodata_is_nan:
                source_valid &= ~np.isnan(source)
            else:
                source_valid &= source != float(src.nodata)

        # Probabilities should be between 0 and 1. This also excludes unlabelled 255
        # fill values if a source raster did not define its NoData metadata correctly.
        source_valid &= (source >= PROBABILITY_MIN) & (source <= PROBABILITY_MAX)
        source_prepared = np.full(source.shape, TEMP_NODATA, dtype=np.float32)
        source_prepared[source_valid] = source[source_valid]

        destination = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        reproject(
            source=source_prepared,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=TEMP_NODATA,
            dst_transform=grid_transform,
            dst_crs=anchor_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )

    valid = np.isfinite(destination) & (destination >= PROBABILITY_MIN) & (destination <= PROBABILITY_MAX)
    return destination, valid

# =============================================================================
# 5. Tie-breaking helpers: use only neighbouring higher-probability candidates
# =============================================================================
def choose_tied_pixels_by_neighbour_support(
    tied_indices: np.ndarray,
    candidate_mask: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    required: int,
    width: int,
    height: int,
    year: int,
    chunk_size: int = TIE_CHUNK_SIZE,
) -> np.ndarray:
    """Choose exactly `required` threshold-tied pixels using local support.

    Primary RF ranking has already selected all eligible pixels with p > threshold.
    For pixels tied exactly at threshold, rank by:
      1) number of p > threshold eligible neighbours in the 8-connected 3x3 window;
      2) number of p > threshold eligible neighbours in the outer ring of a 5x5 window.
    This favours tied pixels adjacent to a high-confidence core without altering the
    probability ranking. Candidates equal on both scores are sampled with a fixed seed.
    The final seeded choice is only a neutral, reproducible tie resolution, not new evidence.
    """
    tied_indices = np.asarray(tied_indices, dtype=np.int64)
    if required < 0 or required > tied_indices.size:
        raise ValueError(
            f"阈值并列像元不足：需要 {required:,}，但只有 {tied_indices.size:,} 个。"
        )
    if required == 0:
        return np.empty(0, dtype=np.int64)
    if required == tied_indices.size:
        return tied_indices.copy()

    candidate_flat = candidate_mask.ravel()
    probability_flat = probability.ravel()
    # Bucket priority = immediate 3x3 support (0..8), then 5x5 outer-ring support (0..16).
    # Code max = 8*17 + 16 = 152; larger values mean stronger local support.
    bucket_chunks: list[list[np.ndarray]] = [[] for _ in range(153)]
    offsets_3x3 = [
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]
    offsets_outer = [
        (dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)
        if max(abs(dr), abs(dc)) == 2
    ]

    for start in range(0, tied_indices.size, chunk_size):
        idx = tied_indices[start:start + chunk_size]
        rows = idx // width
        cols = idx % width
        score3 = np.zeros(idx.size, dtype=np.uint8)
        score_outer = np.zeros(idx.size, dtype=np.uint8)

        for dr, dc in offsets_3x3 + offsets_outer:
            nr = rows + dr
            nc = cols + dc
            inside = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
            if not inside.any():
                continue
            positions = np.flatnonzero(inside)
            neighbour_idx = idx[positions] + dr * width + dc
            # Only candidates admitted by temporal, forest, and probability validity
            # filters count as core support; ties at the threshold do not count as core.
            core = candidate_flat[neighbour_idx] & (probability_flat[neighbour_idx] > threshold)
            if dr in (-1, 0, 1) and dc in (-1, 0, 1):
                score3[positions] += core.astype(np.uint8)
            else:
                score_outer[positions] += core.astype(np.uint8)

        priority = score3.astype(np.uint16) * 17 + score_outer.astype(np.uint16)
        for score in np.unique(priority):
            bucket_chunks[int(score)].append(idx[priority == score].copy())
        del idx, rows, cols, score3, score_outer, priority

    # Release the original tied-index array before concatenating the support buckets.
    del tied_indices
    rng = np.random.default_rng(RANDOM_TIE_SEED_BASE + year)
    remaining = required
    selected_parts: list[np.ndarray] = []

    for score in range(152, -1, -1):
        chunks = bucket_chunks[score]
        if not chunks:
            continue
        group = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        bucket_chunks[score] = []
        group_count = int(group.size)

        if group_count <= remaining:
            selected_parts.append(group)
            remaining -= group_count
        else:
            # This is the only random step, and it applies only where RF probability and
            # both neighbourhood scores are identical. A fixed seed makes the choice repeatable.
            selected_parts.append(rng.choice(group, size=remaining, replace=False))
            remaining = 0
            break

        if remaining == 0:
            break

    if remaining != 0:
        raise RuntimeError(f"邻域排序后仍有 {remaining} 个目标像元未能分配。")
    return np.concatenate(selected_parts).astype(np.int64, copy=False)


# =============================================================================
# 6. Pass 1: count cumulative occurrence and record the time-series edge years
# =============================================================================
print("开始统计每个像元在 2001-2020 年间的桉树累计出现年份……")
total_years_map = np.zeros((grid_height, grid_width), dtype=np.uint8)
edge_years_status = np.zeros((grid_height, grid_width), dtype=np.uint8)
edge_bits = {2001: 1, 2002: 2, 2019: 4, 2020: 8}

for year in YEARS:
    aligned = read_final_aligned(final_paths[year])
    is_euca = aligned == 1
    total_years_map += is_euca.astype(np.uint8)
    if year in edge_bits:
        edge_years_status[is_euca] |= edge_bits[year]
    del aligned, is_euca
    gc.collect()

# This mask is based on cumulative occurrence across the entire series, not a rolling
# three-year window. For one- or two-year detections, only full-series edge exceptions
# are retained: single detections in any edge year, or two detections occupying both
# years in either the first pair (2001-2002) or last pair (2019-2020).
first_pair_supported = (edge_years_status & 0b0011) == 0b0011
last_pair_supported = (edge_years_status & 0b1100) == 0b1100
any_edge_detection = (edge_years_status & 0b1111) != 0

valid_mask = (
    (total_years_map >= 3)
    | ((total_years_map == 2) & (first_pair_supported | last_pair_supported))
    | ((total_years_map == 1) & any_edge_detection)
)

# 新增：记录 2001-2020 年间至少出现过 1 次桉树的掩膜（用于候选不足时的二级优先补齐）
any_year_mask = total_years_map >= 1

print(
    "时序筛选完成：",
    f"累计出现 ≥3 年像元数={int(np.count_nonzero(total_years_map >= 3)):,}；",
    f"通过主时序掩膜像元数={int(np.count_nonzero(valid_mask)):,}；",
    f"累计出现 ≥1 年像元数={int(np.count_nonzero(any_year_mask)):,}",
)
del total_years_map, edge_years_status
# Keep valid_mask and any_year_mask for pass 2.
gc.collect()

# =============================================================================
# 7. Pass 2: area-constrained annual maps with neighbour-aware tie-breaking
# =============================================================================
threshold_records = []

for year in YEARS:
    print(f"\n处理 {year} 年……")
    final_path = final_paths[year]
    prob_path = base_dir / f"{province}_{year}" / f"{province}_euca_prob_{year}.tif"
    output_dir = base_dir / OUTPUT_SUBDIR / f"{province}_{year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{province}_euca_postprocessed_neighbor_{year}.tif"

    if not prob_path.exists() and area_values[year] > 0:
        raise FileNotFoundError(f"{year} 年目标面积大于 0，但概率图不存在：{prob_path}")

    target_area_m2 = area_values[year] * AREA_VALUE_TO_M2
    target_pixels = int(round(target_area_m2 / PIXEL_AREA_M2))
    if target_pixels < 0:
        raise ValueError(f"{year} 年目标像元数为负，请检查面积 CSV。")

    raw_data = read_final_aligned(final_path)
    forest_mask = np.isin(raw_data, (0, 1))
    post_final_map = np.full((grid_height, grid_width), NODATA_OUT, dtype=np.uint8)
    post_final_map[forest_mask] = 0
    # raw_data is needed only to establish the annual forest domain; free this large array.
    del raw_data

    candidate_count = 0
    tie_count = 0
    tie_selected_count = 0
    status = "target_area_zero" if target_pixels == 0 else "not_processed"

    if target_pixels == 0:
        print("  目标面积为 0：保留林地背景为 0，其余区域为 NoData。")
        threshold = np.nan
        selected_count = 0
        del forest_mask

    else:
        prob_data, prob_valid = read_probability_aligned(prob_path)
        valid_forest_prob = forest_mask & prob_valid
        del prob_valid, forest_mask

        candidate_mask = valid_mask & valid_forest_prob
        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_probs = prob_data.ravel()[candidate_indices]
        candidate_count = int(candidate_indices.size)

        if candidate_count == 0 and not np.any(valid_forest_prob):
            threshold = np.nan
            selected_count = 0
            status = "no_valid_candidates"
            print(
                f"  警告：没有任何有效的林地概率像元；"
                f"目标为 {target_pixels:,} 像元，无法满足目标面积。"
            )
            del valid_forest_prob
        elif candidate_count < target_pixels:
            # 1) 先保留全部主候选像元（Tier 1: valid_mask）
            post_final_map.ravel()[candidate_indices] = 1
            selected_count = candidate_count
            deficit = target_pixels - selected_count
            print(
                f"  提示：主时序候选像元 ({candidate_count:,}) 少于目标 ({target_pixels:,})，"
                f"差额 {deficit:,} 个像元启动阶梯回退补齐……"
            )

            # 2) 二级回退（Tier 2）：优先从“20年间至少出现过1-2次桉树”的林地像元中按高概率补齐
            tier2_mask = any_year_mask & (~valid_mask) & valid_forest_prob
            tier2_indices = np.flatnonzero(tier2_mask)
            tier2_count = int(tier2_indices.size)

            if tier2_count > 0 and deficit > 0:
                if tier2_count <= deficit:
                    post_final_map.ravel()[tier2_indices] = 1
                    selected_count += tier2_count
                    deficit -= tier2_count
                    threshold = float(prob_data.ravel()[tier2_indices].min())
                    status = "area_constrained_tier2_all_plus_tier3"
                else:
                    tier2_probs = prob_data.ravel()[tier2_indices]
                    kth = tier2_count - deficit
                    threshold = float(np.partition(tier2_probs, kth)[kth])
                    above = tier2_indices[tier2_probs > threshold]
                    tied = tier2_indices[tier2_probs == threshold]
                    rem = deficit - int(above.size)
                    tie_count = int(tied.size)

                    post_final_map.ravel()[above] = 1
                    if rem > 0:
                        # 邻域支持度：以已选中的 Tier 1 像元 + Tier 2 高于阈值的像元共同作为核心支持
                        support_mask = candidate_mask | tier2_mask
                        prob_for_tie = prob_data.copy()
                        prob_for_tie[candidate_mask] = 2.0  # 确保已选中的 Tier 1 全部作为核心支持
                        tied_choice = choose_tied_pixels_by_neighbour_support(
                            tied_indices=tied,
                            candidate_mask=support_mask,
                            probability=prob_for_tie,
                            threshold=threshold,
                            required=rem,
                            width=grid_width,
                            height=grid_height,
                            year=year,
                        )
                        post_final_map.ravel()[tied_choice] = 1
                        tie_selected_count = int(tied_choice.size)
                        del tied_choice, support_mask, prob_for_tie
                    else:
                        del tied

                    selected_count += int(above.size + tie_selected_count)
                    deficit = 0
                    status = "area_constrained_top_k_neighbour_tiebreak"
                    del tier2_probs, above
            del tier2_mask, tier2_indices

            # 3) 三级兜底（Tier 3）：若 Tier 1 + Tier 2 仍不够，从当年其余有效林地像元中按高概率补齐
            if deficit > 0:
                tier3_mask = valid_forest_prob & (post_final_map == 0)
                tier3_indices = np.flatnonzero(tier3_mask)
                tier3_count = int(tier3_indices.size)
                if tier3_count <= deficit:
                    post_final_map.ravel()[tier3_indices] = 1
                    selected_count += tier3_count
                    threshold = np.nan
                    status = "candidate_shortage_all_selected"
                else:
                    tier3_probs = prob_data.ravel()[tier3_indices]
                    kth = tier3_count - deficit
                    threshold = float(np.partition(tier3_probs, kth)[kth])
                    above = tier3_indices[tier3_probs > threshold]
                    tied = tier3_indices[tier3_probs == threshold]
                    rem = deficit - int(above.size)
                    tie_count = int(tied.size)

                    post_final_map.ravel()[above] = 1
                    if rem > 0:
                        support_mask = (post_final_map == 1) | tier3_mask
                        prob_for_tie = prob_data.copy()
                        prob_for_tie[post_final_map == 1] = 2.0
                        tied_choice = choose_tied_pixels_by_neighbour_support(
                            tied_indices=tied,
                            candidate_mask=support_mask,
                            probability=prob_for_tie,
                            threshold=threshold,
                            required=rem,
                            width=grid_width,
                            height=grid_height,
                            year=year,
                        )
                        post_final_map.ravel()[tied_choice] = 1
                        tie_selected_count = int(tied_choice.size)
                        del tied_choice, support_mask, prob_for_tie
                    else:
                        del tied

                    selected_count += int(above.size + tie_selected_count)
                    deficit = 0
                    status = "area_constrained_top_k_neighbour_tiebreak"
                    del tier3_probs, above
                del tier3_mask, tier3_indices

            del valid_forest_prob
        else:
            del valid_forest_prob
            status = "area_constrained_top_k_neighbour_tiebreak"
            # Probability remains the primary criterion: all pixels strictly above the
            # K-th probability are retained. Only pixels exactly tied at the cut-off are
            # ranked by neighbouring eligible pixels with probabilities above threshold.
            kth = candidate_count - target_pixels
            threshold = float(np.partition(candidate_probs, kth)[kth])
            above = candidate_indices[candidate_probs > threshold]
            tied = candidate_indices[candidate_probs == threshold]
            remaining = target_pixels - int(above.size)
            tie_count = int(tied.size)
            tie_selected_count = 0

            post_final_map.ravel()[above] = 1
            if remaining > 0:
                tied_choice = choose_tied_pixels_by_neighbour_support(
                    tied_indices=tied,
                    candidate_mask=candidate_mask,
                    probability=prob_data,
                    threshold=threshold,
                    required=remaining,
                    width=grid_width,
                    height=grid_height,
                    year=year,
                )
                post_final_map.ravel()[tied_choice] = 1
                tie_selected_count = int(tied_choice.size)
                del tied_choice
            else:
                del tied

            selected_count = int(above.size + tie_selected_count)
            del above

        # These arrays are no longer needed after threshold/tie selection.
        del prob_data, candidate_mask, candidate_indices, candidate_probs
    # Write metadata on the common grid so all annual outputs have the same extent,
    # CRS, pixel size, and alignment.
    with rasterio.open(out_path, "w", **output_profile) as dst:
        dst.write(post_final_map, 1)

    actual_area_m2 = selected_count * PIXEL_AREA_M2
    area_diff_m2 = actual_area_m2 - target_area_m2
    threshold_text = "NA" if not np.isfinite(threshold) else f"{threshold:.6f}"
    print(
        f"  目标像元数={target_pixels:,}；实际选中={selected_count:,}；"
        f"概率阈值={threshold_text}；像元数差={selected_count - target_pixels:+,}"
    )
    print(f"  已保存：{out_path}；面积差={area_diff_m2 / 10_000:.2f} ha")
    if tie_count > 0:
        print(
            f"  阈值并列像元={tie_count:,}；邻域支持筛选选中={tie_selected_count:,}；"
            "先比较 3×3 邻域，再比较 5×5 外环；最终完全并列才进行固定种子抽选。"
        )

    threshold_records.append({
        "Province": province_cn,
        "Year": year,
        "Target_area_10k_ha": area_values[year],
        "Target_pixels": target_pixels,
        "Candidate_pixels": candidate_count,
        "Selected_pixels": selected_count,
        "Threshold": threshold,
        "Threshold_tie_pixels": tie_count,
        "Tie_pixels_selected": tie_selected_count,
        "Tie_break_method": "3x3 higher-probability core support; then 5x5 outer-ring support; seeded residual tie" if tie_count else "not_needed",
        "Status": status,
    })

    del post_final_map
    gc.collect()

threshold_csv = base_dir / OUTPUT_SUBDIR / f"{province}_postprocessing_thresholds_neighbor.csv"
threshold_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(threshold_records).to_csv(threshold_csv, index=False, encoding="utf-8-sig")
print(f"\n全部年份后处理完成。阈值与面积诊断表已保存：{threshold_csv}")
print("输出位于 outputs/albers/<province>/postprocessed_equal_area_neighbor；原始预测结果不会被覆盖。")
print("请先结合投影检查结果，再比较新旧二值图的空间连通性与面积。")
