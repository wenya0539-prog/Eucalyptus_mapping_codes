# ============================================
# Extract predictor values from raster tiles under reference polygons.
# The EUCA response field is generated from the polygon class attribute.
# ============================================

# 1. 统一项目路径与输入输出配置
PROJECT_ROOT <- "D:/Eucalyptus_mapping"
province_pinyin <- "guangdong"  # 本次处理省份；11省分别运行并修改此项
sample_year <- 2020              # 参考样本特征提取年份

tif_dir <- file.path(PROJECT_ROOT, "data", "gee_features", province_pinyin, as.character(sample_year))
shp_file <- file.path(PROJECT_ROOT, "data", "reference_polygons", province_pinyin,
                      paste0(province_pinyin, ".shp"))
output_csv <- file.path(PROJECT_ROOT, "data", "samples", paste0(province_pinyin, ".csv"))
name_field <- "Name"             # 矢量面中存储类别编码的字段（0=非桉树，1=桉树）
temp_dir <- file.path(PROJECT_ROOT, "temp", "terra")

if (!dir.exists(tif_dir)) stop("找不到特征栅格目录：", tif_dir)
if (!file.exists(shp_file)) stop("找不到参考样本矢量：", shp_file)
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)

# 2. 加载包
library(sf)
library(terra)
library(data.table)

# 设置临时目录
if (!dir.exists(temp_dir)) dir.create(temp_dir, recursive = TRUE)
terraOptions(tempdir = temp_dir)

# ============================================
# 函数：去除Z坐标
remove_z <- function(sf_obj) {
  if (any(st_dimension(sf_obj) == "XYZ")) {
    sf_obj <- st_zm(sf_obj, drop = TRUE)
    message("已去除Z坐标")
  }
  return(sf_obj)
}

# ============================================
# 主程序
# ============================================

# 读取矢量面
message("1. 读取矢量面...")
v_raw <- st_read(shp_file, quiet = TRUE)
v_raw <- remove_z(v_raw)

# 检查字段
if (!name_field %in% names(v_raw)) {
  stop("字段 '", name_field, "' 不存在。可用字段：", 
       paste(names(v_raw), collapse = ", "))
}

# 剔除 name 为 "Untitled polygon" 的要素（无效面）
message("2. 剔除无效面 (name = 'Untitled polygon')...")
invalid_count <- sum(v_raw[[name_field]] == "Untitled polygon", na.rm = TRUE)
if (invalid_count > 0) {
  v_raw <- v_raw[v_raw[[name_field]] != "Untitled polygon", ]
  message("已剔除 ", invalid_count, " 个无效面")
} else {
  message("未发现无效面")
}

# 获取所有TIF文件并排序
tif_files <- list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE)
tif_files <- sort(tif_files)
if (length(tif_files) == 0) stop("未找到TIF文件")
message("共发现 ", length(tif_files), " 个分块文件")

# 以第一个TIF的坐标系为准，转换矢量
message("3. 统一坐标系...")
r_first <- rast(tif_files[1])
v_proj <- st_transform(v_raw, crs(r_first))

# 初始化CSV
if (file.exists(output_csv)) file.remove(output_csv)

# 逐块处理
first_chunk <- TRUE
total_valid_pixels <- 0

for (i in seq_along(tif_files)) {
  message("\n--- 处理第 ", i, "/", length(tif_files), " 块: ", basename(tif_files[i]), " ---")
  
  # 读取当前栅格块
  r <- rast(tif_files[i])
  v_this <- st_transform(v_proj, crs(r))
  v_spat <- vect(v_this)
  
  # 提取像元值
  e <- terra::extract(r, v_spat, ID = TRUE)
  
  if (nrow(e) == 0) {
    message("  该块无任何返回行（理论上不会发生）")
    next
  }
  
  setDT(e)
  
  # 过滤NA
  band_cols <- names(e)[-1]  
  na_rows <- apply(e[, ..band_cols], 1, function(x) all(is.na(x)))
  e <- e[!na_rows, ]
  
  if (nrow(e) == 0) {
    message("  该块无有效重叠像元")
    next
  }
  
  non_na <- sum(!is.na(e[[2]]))
  message("  有效像元数: ", nrow(e), "（其中非NA波段值: ", non_na, "）")
  
  # ==========================================
  # Assign the polygon class attribute to EUCA and convert it to numeric.
  # ==========================================
  v_this$temp_id <- seq_len(nrow(v_this))
  name_map <- setNames(v_this[[name_field]], v_this$temp_id)
  
  # Assign the polygon class label to EUCA as a numeric value.
  e[, EUCA := as.numeric(name_map[ID])]
  
  e[, ID := NULL]
  
  # Place EUCA first so the output is ready for model training.
  setcolorder(e, c("EUCA", setdiff(names(e), "EUCA")))
  # ==========================================
  
  # 写入CSV
  if (first_chunk) {
    fwrite(e, file = output_csv, col.names = TRUE)
    first_chunk <- FALSE
  } else {
    fwrite(e, file = output_csv, append = TRUE, col.names = FALSE)
  }
  
  total_valid_pixels <- total_valid_pixels + nrow(e)
  
  rm(e, r, v_this, v_spat)
  gc()
}

message("\n处理完成！")
message("总有效像元数: ", total_valid_pixels)
message("结果文件: ", output_csv)

# 统计验证
if (file.exists(output_csv)) {
  message("\n正在读取结果文件并验证 EUCA 字段...")
  # 这里也同步修改为读取 EUCA 列
  dt_names <- fread(output_csv, select = "EUCA")
  cat("\nEUCA 字段频数统计 (0=非桉树, 1=桉树)：\n")
  print(table(dt_names$EUCA, useNA = "ifany"))
  cat("\n总有效像元数：", nrow(dt_names), "\n")
}

unlink(list.files(temp_dir, full.names = TRUE, recursive = TRUE))