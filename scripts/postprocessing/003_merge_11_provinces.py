# -*- coding: utf-8 -*-
"""
整理并生成论文发布栅格数据集至 outputs/release：
  1. 后处理二值图 (11省合并): Eucalyptus_binary/Eucalyptus_YYYY.tif
  2. 初步处理二值图 (11省合并): Eucalyptus_initial/Eucalyptus_initial_YYYY.tif
  3. 预测概率图 (分省整理重命名，不合并): Eucalyptus_probability/<Province>/Eucalyptus_prob_<Province>_YYYY.tif
"""

from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

# =============================================================================
# 1. 路径与基本配置
# =============================================================================
PROJECT_ROOT = Path("D:/Eucalyptus_mapping")
AEA_ROOT = PROJECT_ROOT / "outputs" / "albers"
OUT_ROOT = PROJECT_ROOT / "outputs" / "release"

YEARS = list(range(2001, 2021))
PIXEL_SIZE_M = 30.0
NODATA_UINT8 = 255
CHUNK_ROWS = 2048
SKIP_EXISTING = True  # 支持断点续跑：已存在的文件自动跳过

# 11 个省份配置：(原文件夹小写名, 中文名, 论文规范英文名, 所属子区域)
PROVINCES = [
    ("guangxi",   "广西", "Guangxi",   "SCH"),
    ("guangdong", "广东", "Guangdong", "SCH"),
    ("hainan",    "海南", "Hainan",    "SCH"),
    ("yunnan",    "云南", "Yunnan",    "SWC"),
    ("sichuan",   "四川", "Sichuan",   "SWC"),
    ("guizhou",   "贵州", "Guizhou",   "SWC"),
    ("chongqing", "重庆", "Chongqing", "SWC"),
    ("fujian",    "福建", "Fujian",    "CSEC"),
    ("jiangxi",   "江西", "Jiangxi",   "CSEC"),
    ("hunan",     "湖南", "Hunan",     "CSEC"),
    ("zhejiang",  "浙江", "Zhejiang",  "CSEC"),
]

# 三类核心数据输出目录
DIR_POST = OUT_ROOT / "Eucalyptus_binary"
DIR_INIT = OUT_ROOT / "Eucalyptus_initial"
DIR_PROB = OUT_ROOT / "Eucalyptus_probability"

for d in (OUT_ROOT, DIR_POST, DIR_INIT, DIR_PROB):
    d.mkdir(parents=True, exist_ok=True)


def get_src_path(prov_folder: str, year: int, kind: str) -> Path:
    if kind == "post":
        return (
            AEA_ROOT / prov_folder / "postprocessed_equal_area_neighbor"
            / f"{prov_folder}_{year}" / f"{prov_folder}_euca_postprocessed_neighbor_{year}.tif"
        )
    elif kind == "final":
        return AEA_ROOT / prov_folder / f"{prov_folder}_{year}" / f"{prov_folder}_euca_final_{year}.tif"
    elif kind == "prob":
        return AEA_ROOT / prov_folder / f"{prov_folder}_{year}" / f"{prov_folder}_euca_prob_{year}.tif"
    else:
        raise ValueError(f"未知类型: {kind}")


# =============================================================================
# 2. 计算 11 省二值合并图的统一网格 (Union Grid)
# =============================================================================
print("正在扫描 11 省范围以构建全域 30-m Albers 合并网格……")
min_x = min_y = float("inf")
max_x = max_y = float("-inf")
target_crs = None

for prov_folder, prov_cn, prov_en, _ in PROVINCES:
    p = get_src_path(prov_folder, YEARS[0], "post")
    if not p.exists():
        raise FileNotFoundError(f"找不到后处理文件：{p}")
    with rasterio.open(p) as src:
        if target_crs is None:
            target_crs = src.crs
        b = src.bounds
        min_x = min(min_x, b.left)
        min_y = min(min_y, b.bottom)
        max_x = max(max_x, b.right)
        max_y = max(max_y, b.top)

total_width = int(round((max_x - min_x) / PIXEL_SIZE_M))
total_height = int(round((max_y - min_y) / PIXEL_SIZE_M))
union_transform = from_origin(min_x, max_y, PIXEL_SIZE_M, PIXEL_SIZE_M)

print("=" * 76)
print(f"全域合并网格尺寸：{total_width:,} 列 × {total_height:,} 行 (像元 30 m × 30 m = 900 m²)")
print(f"范围 (m)：Left={min_x:.1f}, Bottom={min_y:.1f}, Right={max_x:.1f}, Top={max_y:.1f}")
print("=" * 76)


# =============================================================================
# 3. 合并函数：合成 11 省初步处理 (initial) 与后处理 (binary) 图层
# =============================================================================
def merge_11_provinces_binary(year: int, kind: str, out_path: Path) -> None:
    """分块无损合并 11 省 uint8 二值图 (1=桉树, 0=非桉树林地, 255=NoData)"""
    if SKIP_EXISTING and out_path.exists():
        print(f"  [跳过已存在] {out_path.name}")
        return

    profile = {
        "driver": "GTiff",
        "width": total_width,
        "height": total_height,
        "count": 1,
        "dtype": "uint8",
        "crs": target_crs,
        "transform": union_transform,
        "nodata": NODATA_UINT8,
        "compress": "LZW",
        "predictor": 1,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES",
    }

    with rasterio.open(out_path, "w+", **profile) as dst:
        for prov_folder, _, _, _ in PROVINCES:
            src_path = get_src_path(prov_folder, year, kind)
            if not src_path.exists():
                raise FileNotFoundError(f"缺少输入文件：{src_path}")

            with rasterio.open(src_path) as src:
                col_off = int(round((src.bounds.left - min_x) / PIXEL_SIZE_M))
                row_off = int(round((max_y - src.bounds.top) / PIXEL_SIZE_M))
                h, w = src.height, src.width

                for r0 in range(0, h, CHUNK_ROWS):
                    ch = min(CHUNK_ROWS, h - r0)
                    src_win = Window(0, r0, w, ch)
                    dst_win = Window(col_off, row_off + r0, w, ch)

                    s_arr = src.read(1, window=src_win)
                    if np.all(s_arr == NODATA_UINT8):
                        continue

                    d_arr = dst.read(1, window=dst_win)
                    mask_0 = (s_arr == 0) & (d_arr == NODATA_UINT8)
                    mask_1 = (s_arr == 1)
                    if np.any(mask_0) or np.any(mask_1):
                        d_arr[mask_0] = 0
                        d_arr[mask_1] = 1
                        dst.write(d_arr, 1, window=dst_win)


# =============================================================================
# 4. 执行第 1、2 类数据：逐年合并后处理二值图与初步处理二值图
# =============================================================================
print("\n>>> 第一阶段：合成 2001–2020 年 11 省【后处理二值图】与【初步处理二值图】……")
for year in YEARS:
    print(f"正在处理 {year} 年合并图层……")
    # 1) 后处理最终二值图: Eucalyptus_YYYY.tif
    out_post = DIR_POST / f"Eucalyptus_{year}.tif"
    merge_11_provinces_binary(year, "post", out_post)
    print(f"  [完成] 后处理合并图 -> {out_post}")

    # 2) 初步处理二值图: Eucalyptus_initial_YYYY.tif
    out_init = DIR_INIT / f"Eucalyptus_initial_{year}.tif"
    merge_11_provinces_binary(year, "final", out_init)
    print(f"  [完成] 初步处理合并图 -> {out_init}")


# =============================================================================
# 5. 执行第 3 类数据：按省份复制并科学重命名随机森林概率图 (不合并)
# =============================================================================
print("\n>>> 第二阶段：按省份复制并重命名 2001–2020 年【随机森林概率图】……")
for prov_folder, prov_cn, prov_en, _ in PROVINCES:
    prov_prob_dir = DIR_PROB / prov_en
    prov_prob_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0

    for year in YEARS:
        src_prob = get_src_path(prov_folder, year, "prob")
        if not src_prob.exists():
            raise FileNotFoundError(f"缺少概率图：{src_prob}")

        # 科学重命名格式: Eucalyptus_prob_<Province>_YYYY.tif
        dst_prob = prov_prob_dir / f"Eucalyptus_prob_{prov_en}_{year}.tif"
        if SKIP_EXISTING and dst_prob.exists():
            continue
        shutil.copy2(src_prob, dst_prob)
        copied_count += 1

    print(f"  [完成] {prov_cn} ({prov_en}): 20 年概率图已整理至 -> {prov_prob_dir} (本次新增复制 {copied_count} 张)")

print(f"\n全部三类栅格数据整理完毕！输出目录：{OUT_ROOT}")