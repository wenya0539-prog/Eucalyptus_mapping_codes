# -*- coding: utf-8 -*-
"""
保持原版 uint16 高精度 (0~10000, scale_factor=0.0001, 4位小数精度, NoData=65535)，
仅将每个省份原本分散的 20 个年度文件合成为 1 个三维时空 NetCDF-4 (.nc) 文件 (time=20, y, x)。
输出结构：
  outputs/release/Eucalyptus_probability_netcdf/Eucalyptus_prob_<Province>_2001_2020.nc (共 11 个省级三维 .nc 文件)
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset

# =============================================================================
# 1. 路径与基本参数配置
# =============================================================================
PROJECT_ROOT = Path("D:/Eucalyptus_mapping")
AEA_ROOT = PROJECT_ROOT / "outputs" / "albers"
OUT_ROOT = PROJECT_ROOT / "outputs" / "release" / "Eucalyptus_probability_netcdf"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2001, 2021))
ROW_BATCH = 512  # 每次流式读取 20 年的 512 行，防止大省份内存溢出
SKIP_EXISTING = True

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

AEA_PROJ4 = "+proj=aea +lat_1=18 +lat_2=34 +lat_0=26 +lon_0=110 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs"


def get_prob_path(prov_folder: str, year: int) -> Path:
    return AEA_ROOT / prov_folder / f"{prov_folder}_{year}" / f"{prov_folder}_euca_prob_{year}.tif"


# =============================================================================
# 2. 单省 20 年合并转换为 1 个三维 NetCDF-4 (.nc) [严格保持 uint16 精度]
# =============================================================================
def build_province_20yr_uint16_nc(prov_folder: str, prov_en: str, subregion: str, out_nc: Path) -> None:
    tif_paths = [get_prob_path(prov_folder, yr) for yr in YEARS]
    for p in tif_paths:
        if not p.exists():
            raise FileNotFoundError(f"缺少输入概率图: {p}")

    with rasterio.open(tif_paths[0]) as src0:
        h, w = src0.height, src0.width
        transform = src0.transform
        crs_wkt = src0.crs.to_wkt() if src0.crs else ""

    x_coords = transform.c + (np.arange(w, dtype=np.float64) + 0.5) * transform.a
    y_coords = transform.f + (np.arange(h, dtype=np.float64) + 0.5) * transform.e

    n_years = len(YEARS)
    # 关键：将 20 年时间轴全部纳入同一个三维 Chunk (20, 256, 256)，开启跨年三维压缩
    chunk_t = n_years
    chunk_y = min(256, h)
    chunk_x = min(256, w)

    with Dataset(out_nc, "w", format="NETCDF4") as nc:
        # --- 1) 全局元数据 (CF-1.8 规范) ---
        nc.Conventions = "CF-1.8"
        nc.title = f"30-m Eucalyptus plantation Random Forest probability stack for {prov_en} (2001-2020)"
        nc.institution = "College of Forestry, Guangxi University"
        nc.source = "Landsat 5/7/8 SR, NASADEM, GLCM textures, and Random Forest classification"
        nc.province = prov_en
        nc.subregion = subregion
        nc.temporal_coverage = "2001-2020 (20 annual layers)"
        nc.spatial_resolution = "30 m x 30 m"
        nc.proj4 = AEA_PROJ4

        # --- 2) 创建三维时空维度 (time=20, y, x) ---
        nc.createDimension("time", n_years)
        nc.createDimension("y", h)
        nc.createDimension("x", w)

        var_t = nc.createVariable("time", "i4", ("time",))
        var_t.standard_name = "time"
        var_t.long_name = "Year"
        var_t.units = "year"
        var_t.axis = "T"
        var_t[:] = np.array(YEARS, dtype=np.int32)

        var_y = nc.createVariable("y", "f8", ("y",))
        var_y.standard_name = "projection_y_coordinate"
        var_y.long_name = "y-coordinate in South China Albers Equal Area projection"
        var_y.units = "m"
        var_y.axis = "Y"
        var_y[:] = y_coords

        var_x = nc.createVariable("x", "f8", ("x",))
        var_x.standard_name = "projection_x_coordinate"
        var_x.long_name = "x-coordinate in South China Albers Equal Area projection"
        var_x.units = "m"
        var_x.axis = "X"
        var_x[:] = x_coords

        # --- 3) 创建 CF 标准 Albers 投影参考变量 ---
        var_crs = nc.createVariable("crs", "i4")
        var_crs.grid_mapping_name = "albers_conical_equal_area"
        var_crs.standard_parallel = np.array([18.0, 34.0], dtype=np.float64)
        var_crs.latitude_of_projection_origin = 26.0
        var_crs.longitude_of_central_meridian = 110.0
        var_crs.false_easting = 0.0
        var_crs.false_northing = 0.0
        var_crs.semi_major_axis = 6378137.0
        var_crs.inverse_flattening = 298.257223563
        var_crs.crs_wkt = crs_wkt
        var_crs.spatial_ref = crs_wkt
        var_crs.GeoTransform = f"{transform.c} {transform.a} 0.0 {transform.f} 0.0 {transform.e}"

        # --- 4) 创建三维概率变量 (严格保持 u2 即 uint16, 0~10000, scale_factor=0.0001) ---
        v_prob = nc.createVariable(
            "eucalyptus_prob", "u2", ("time", "y", "x"),
            zlib=True, complevel=9, shuffle=True,
            chunksizes=(chunk_t, chunk_y, chunk_x), fill_value=65535
        )
        v_prob.set_auto_maskandscale(False)
        v_prob.long_name = "Random Forest predicted probability of Eucalyptus occurrence (2001-2020)"
        v_prob.units = "1"
        v_prob.grid_mapping = "crs"
        v_prob.scale_factor = np.float32(0.0001)
        v_prob.add_offset = np.float32(0.0)
        v_prob.valid_range = np.array([0, 10000], dtype=np.uint16)

        # --- 5) 同时打开 20 年 TIF，按三维块流式写入 ---
        srcs = [rasterio.open(p) for p in tif_paths]
        try:
            for r0 in range(0, h, ROW_BATCH):
                ch = min(ROW_BATCH, h - r0)
                win = Window(0, r0, w, ch)

                # 初始化 (20, ch, w) 的 uint16 块，默认填充 65535 (NoData)
                block_3d = np.full((n_years, ch, w), 65535, dtype=np.uint16)

                for t_idx, src in enumerate(srcs):
                    arr = src.read(1, window=win)
                    if src.nodata is not None and not np.isnan(src.nodata):
                        nodata_mask = np.isnan(arr) | (arr == src.nodata) | (arr < 0.0)
                    else:
                        nodata_mask = np.isnan(arr) | (arr < 0.0)

                    valid = ~nodata_mask
                    if np.any(valid):
                        # 严格保持你原本的 0~10000 (0.0001 精度) 转换逻辑
                        block_3d[t_idx, valid] = np.rint(np.clip(arr[valid], 0.0, 1.0) * 10000.0).astype(np.uint16)

                # 一次性写入 20 年的三维块
                v_prob[:, r0:r0 + ch, :] = block_3d
        finally:
            for src in srcs:
                src.close()


# =============================================================================
# 3. 批量运行 11 省转换并实时统计总大小
# =============================================================================
if __name__ == "__main__":
    print(">>> 开始将 outputs/albers 中 11 省 2001–2020 年概率图合成为 11 个三维 NetCDF-4 (uint16, 0.0001精度) 文件……")
    total_bytes = 0

    for prov_folder, prov_cn, prov_en, subregion in PROVINCES:
        out_nc = OUT_ROOT / f"Eucalyptus_prob_{prov_en}_2001_2020.nc"
        if not (SKIP_EXISTING and out_nc.exists()):
            print(f"正在合成 {prov_cn} ({prov_en}) 2001–2020 三维 .nc 文件……")
            build_province_20yr_uint16_nc(prov_folder, prov_en, subregion, out_nc)

        f_size = out_nc.stat().st_size
        total_bytes += f_size
        print(f"  [完成] {out_nc.name} -> 大小: {f_size / (1024**3):.2f} GB ({f_size / (1024**2):.1f} MB)")

    print("=" * 72)
    print(f"全部 11 省三维 .nc 转换完毕！总体积: {total_bytes / (1024**3):.2f} GB")
    print("提示：请直接上传这 11 个 .nc 文件，无需再额外打包成 .zip！")