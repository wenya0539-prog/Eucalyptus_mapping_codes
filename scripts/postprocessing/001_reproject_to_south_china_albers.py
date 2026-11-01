# -*- coding: utf-8 -*-
"""
Reproject annual Eucalyptus input rasters to a southern-China Albers Equal Area grid.

Why this projection:
  - Existing audit for Guangxi reports 30 x 30 map-unit pixels (900 m² planar),
    but sampled ground areas have a median of about 883.35 m² (about -1.85%).
  - EPSG:32650 (WGS 84 / UTM zone 50N) is being used far beyond the central part of
    its zone across this broad study region. Its map-plane pixel area should not be
    assumed to equal 900 m² on the ground.
  - Albers Equal Area is appropriate for area-based mapping over a broad, predominantly
    east-west region. The custom parameters are centered on southern China and cover the
    manuscript's approximate latitude span (18-34 N).

Target CRS (custom; no EPSG code):
  +proj=aea +lat_1=18 +lat_2=34 +lat_0=26 +lon_0=110
  +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs

Input per year:
  outputs/raw_predictions/<province>/<province>_YYYY/<province>_euca_final_YYYY.tif
  outputs/raw_predictions/<province>/<province>_YYYY/<province>_euca_prob_YYYY.tif

Output per year (original files are not changed):
  outputs/albers/<province>/<province>_YYYY/<same filename>

All output grids are snapped to a fixed 30 m lattice at coordinate origin (0, 0), so
all provinces processed with this script share the same pixel alignment even though
file extents differ. The final maps use nearest-neighbour resampling. Probabilities use
bilinear resampling as a continuous variable. The post-processing and area thresholding
must be rerun on these projected rasters; do not reuse the old postprocessed rasters.

Dependencies: rasterio, numpy, pyproj
"""
from __future__ import annotations

import math
from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

# =============================================================================
# CONFIGURATION — change province and Chinese province name as needed
# =============================================================================
PROJECT_ROOT = Path("D:/Eucalyptus_mapping")
PROVINCE = "guizhou"  # 每次运行一个省份；11省分别修改此项
PROVINCE_NAMES = {
    "guangxi": "广西", "guangdong": "广东", "hainan": "海南",
    "yunnan": "云南", "sichuan": "四川", "guizhou": "贵州",
    "chongqing": "重庆", "fujian": "福建", "jiangxi": "江西",
    "hunan": "湖南", "zhejiang": "浙江",
}
if PROVINCE not in PROVINCE_NAMES:
    raise ValueError(f"未知省份代码：{PROVINCE}")
PROVINCE_CN = PROVINCE_NAMES[PROVINCE]
YEARS = list(range(2001, 2021))
SOURCE_ROOT = PROJECT_ROOT / "outputs" / "raw_predictions" / PROVINCE
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "albers" / PROVINCE
PIXEL_SIZE_M = 30.0

# Equal-area Albers projection tailored to the study region in southern China.
# Keep these parameters exactly the same for every province/year.
AEA_PROJ4 = (
    "+proj=aea +lat_1=18 +lat_2=34 +lat_0=26 +lon_0=110 "
    "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs"
)
TARGET_CRS = CRS.from_proj4(AEA_PROJ4)

NODATA_CLASS = 255
NODATA_PROB = float("nan")

# =============================================================================
# INPUT FILES AND SHARED OUTPUT GRID
# =============================================================================
def input_path(year: int, kind: str) -> Path:
    if kind == "final":
        name = f"{PROVINCE}_euca_final_{year}.tif"
    elif kind == "prob":
        name = f"{PROVINCE}_euca_prob_{year}.tif"
    else:
        raise ValueError(kind)
    return SOURCE_ROOT / f"{PROVINCE}_{year}" / name

files: list[tuple[int, str, Path]] = []
for year in YEARS:
    for kind in ("final", "prob"):
        p = input_path(year, kind)
        if not p.exists():
            raise FileNotFoundError(f"缺少输入文件：{p}")
        files.append((year, kind, p))

# Transform and combine all annual bounds, so all years of this province use one grid.
min_x = min_y = float("inf")
max_x = max_y = float("-inf")
source_crs_names = set()

for year, kind, path in files:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"输入栅格没有 CRS：{path}")
        source_crs_names.add(src.crs.to_string())
        bounds = transform_bounds(
            src.crs, TARGET_CRS,
            src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top,
            densify_pts=101,
        )
        left, bottom, right, top = bounds
        if not all(math.isfinite(v) for v in bounds):
            raise ValueError(f"无法将范围转换至目标 CRS：{path} -> {bounds}")
        min_x = min(min_x, left)
        min_y = min(min_y, bottom)
        max_x = max(max_x, right)
        max_y = max(max_y, top)

# Snap extent to the 30 m global lattice. This is crucial: all provinces must share
# origin alignment even if their bounds are different.
grid_left = math.floor(min_x / PIXEL_SIZE_M) * PIXEL_SIZE_M
grid_bottom = math.floor(min_y / PIXEL_SIZE_M) * PIXEL_SIZE_M
grid_right = math.ceil(max_x / PIXEL_SIZE_M) * PIXEL_SIZE_M
grid_top = math.ceil(max_y / PIXEL_SIZE_M) * PIXEL_SIZE_M
width = int(round((grid_right - grid_left) / PIXEL_SIZE_M))
height = int(round((grid_top - grid_bottom) / PIXEL_SIZE_M))
grid_transform = from_origin(grid_left, grid_top, PIXEL_SIZE_M, PIXEL_SIZE_M)

if width <= 0 or height <= 0:
    raise ValueError(f"目标网格尺寸异常：{width} × {height}")

print("=" * 72)
print(f"省份：{PROVINCE_CN} ({PROVINCE})；年份：{YEARS[0]}–{YEARS[-1]}")
print(f"输入 CRS：{', '.join(sorted(source_crs_names))}")
print("目标 CRS：自定义 WGS84 / South China Albers Equal Area")
print(f"PROJ：{AEA_PROJ4}")
print(f"统一网格：{width:,} 列 × {height:,} 行，像元 {PIXEL_SIZE_M:g} × {PIXEL_SIZE_M:g} m")
print(f"范围：left={grid_left:.2f}, bottom={grid_bottom:.2f}, right={grid_right:.2f}, top={grid_top:.2f}")
print(f"每个输出像元平面面积：{PIXEL_SIZE_M * PIXEL_SIZE_M:.2f} m²（Albers 等面积投影）")
print(f"输出目录：{OUTPUT_ROOT}")
print("原始目录不会被修改。")
print("=" * 72)

# =============================================================================
# REPROJECT EACH FINAL AND PROBABILITY RASTER
# =============================================================================
report_rows: list[dict[str, object]] = []

for year, kind, src_path in files:
    out_dir = OUTPUT_ROOT / f"{PROVINCE}_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / src_path.name

    with rasterio.open(src_path) as src:
        if kind == "final":
            dtype = "uint8"
            dst_nodata = NODATA_CLASS
            src_nodata = src.nodata if src.nodata is not None else NODATA_CLASS
            resampling = Resampling.nearest
        else:
            dtype = "float32"
            dst_nodata = NODATA_PROB
            # The audit indicates probability rasters use NaN NoData. If metadata has no
            # NoData value, explicitly treat NaNs as invalid; invalid values outside [0,1]
            # will also be excluded later by the post-processing script.
            src_nodata = src.nodata if src.nodata is not None else float("nan")
            resampling = Resampling.bilinear

        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "dtype": dtype,
            "crs": TARGET_CRS,
            "transform": grid_transform,
            "nodata": dst_nodata,
            "compress": "LZW",
            "predictor": 2 if dtype == "float32" else 1,
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }

        # Write through Rasterio bands instead of loading the full raster into RAM.
        with rasterio.open(out_path, "w", **profile) as dst:
            if kind == "final":
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=grid_transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                    init_dest_nodata=True,
                )
            else:
                # 概率图用数值型 -9999.0 替代 NaN 参与重投影，避免 GDAL Bilinear NaN 传染 Bug
                TEMP_ND = -9999.0
                src_arr = src.read(1).astype(np.float32)
                src_arr[~np.isfinite(src_arr) | (src_arr < 0.0) | (src_arr > 1.0)] = TEMP_ND
                dst_arr = np.full((height, width), TEMP_ND, dtype=np.float32)
                reproject(
                    source=src_arr,
                    destination=dst_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=TEMP_ND,
                    dst_transform=grid_transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=TEMP_ND,
                    resampling=resampling,
                    init_dest_nodata=True,
                )
                dst_arr[(dst_arr == TEMP_ND) | (dst_arr < 0.0) | (dst_arr > 1.0)] = np.nan
                dst.write(dst_arr, 1)

        report_rows.append({
            "Province": PROVINCE_CN,
            "Year": year,
            "Type": kind,
            "Input_file": str(src_path),
            "Output_file": str(out_path),
            "Input_CRS": src.crs.to_string(),
            "Output_CRS": "Custom WGS84 / South China Albers Equal Area",
            "Output_width": width,
            "Output_height": height,
            "Pixel_size_m": PIXEL_SIZE_M,
            "Pixel_area_map_m2": PIXEL_SIZE_M ** 2,
            "Resampling": resampling.name,
            "Output_NoData": "NaN" if isinstance(dst_nodata, float) and math.isnan(dst_nodata) else dst_nodata,
        })

    print(f"完成：{year} {kind} -> {out_path}")

# Write audit table and a CRS note next to outputs.
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
report_path = OUTPUT_ROOT / f"{PROVINCE}_albers_reprojection_report.csv"
with report_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
    writer.writeheader()
    writer.writerows(report_rows)

crs_note = OUTPUT_ROOT / "PROJECTION_README.txt"
crs_note.write_text(
    "Custom projection used for the annual Eucalyptus maps\n"
    "Name: WGS84 / South China Albers Equal Area (custom; no EPSG code)\n"
    f"PROJ: {AEA_PROJ4}\n"
    "Grid: 30 m x 30 m in the equal-area projected plane\n"
    "Map-plane pixel area: 900 m2\n"
    "The Albers Equal Area projection preserves area; distances and shapes are not globally preserved.\n"
    "Final categorical rasters were resampled by nearest neighbour; probability rasters by bilinear interpolation.\n"
    "IMPORTANT: rerun temporal post-processing and area-constrained thresholding on these projected rasters.\n"
    "Do not use the original postprocessed rasters as final outputs after reprojection.\n",
    encoding="utf-8",
)

print("\n重投影完成。")
print(f"重投影检查表：{report_path}")
print(f"投影说明：{crs_note}")
print("下一步：把后处理脚本的 base_dir 指向 OUTPUT_ROOT，再重新做时序筛选和面积约束。")
