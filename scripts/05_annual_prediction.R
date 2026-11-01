# ==============================================================================
# Annual Eucalyptus probability mapping with a Ranger random forest model.
# Raster tiles and row chunks are processed sequentially to limit memory use.
# ==============================================================================

# -------------------- 【0. 环境初始化】 --------------------
# Avoid conflicts with system-level PROJ environment variables.
Sys.unsetenv("PROJ_LIB")   
Sys.unsetenv("PROJ_DATA")  

library(sf)
library(terra)
library(ranger)               
library(data.table)

# -------------------- 【1. 全局配置台】 --------------------

# -------------------- 统一项目路径与运行参数 --------------------
PROJECT_ROOT <- "D:/Eucalyptus_mapping"
province_pinyin <- "zhejiang"  # 每次运行一个省份；11省分别修改此项
province_cn <- "浙江"          # 与上面的英文省名对应

years_to_run <- 2001:2020      # 可按需改为单年，例如 2001

csv_path <- file.path(PROJECT_ROOT, "data", "input", "Eucalyptus_area.csv")
features_root <- file.path(PROJECT_ROOT, "data", "gee_features")
clcd_root <- file.path(PROJECT_ROOT, "data", "CLCD")
output_root <- file.path(PROJECT_ROOT, "outputs", "raw_predictions")
model_path <- file.path(PROJECT_ROOT, "models", "eucalyptus_rf_southern_china_2020.rds")

# --- 系统性能参数 ---
temp_dir <- file.path(PROJECT_ROOT, "temp", "terra")
n_cores <- 4
chunk_size <- 500

if (!file.exists(csv_path)) stop("找不到面积先验文件：", csv_path)
if (!file.exists(model_path)) stop("找不到随机森林模型：", model_path)
area_df <- read.csv(csv_path, check.names = FALSE, fileEncoding = "GBK")

# -------------------- 【2. 批量循环处理主体】 --------------------

for (current_year in years_to_run) {
  
  
  cat(sprintf("\n\n##################################################"))
  cat(sprintf("\n正在处理: %s (%d年)", province_cn, current_year))
  cat(sprintf("\n##################################################\n"))
  
  tif_dir <- file.path(features_root, province_pinyin, as.character(current_year))
  forest_mask_path <- file.path(clcd_root, province_pinyin,
                                sprintf("%s_%d_30m.tif", province_pinyin, current_year))
  output_dir <- file.path(output_root, province_pinyin,
                          sprintf("%s_%d", province_pinyin, current_year))
  
  prob_output_file <- sprintf("%s_euca_prob_%d.tif", province_pinyin, current_year)
  final_output_file <- sprintf("%s_euca_final_%d.tif", province_pinyin, current_year)
  
  if (!dir.exists(tif_dir)) { warning("找不到分块目录，跳过: ", tif_dir); next }
  if (!file.exists(forest_mask_path)) { warning("找不到掩膜文件，跳过: ", forest_mask_path); next }
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  if (!dir.exists(temp_dir)) dir.create(temp_dir, recursive = TRUE)
  terraOptions(tempdir = temp_dir)
  terra::tmpFiles(remove = TRUE)
  
  message("[1] 计算动态像元占比...")
  forest_mask <- rast(forest_mask_path)
  if (nlyr(forest_mask) > 1) forest_mask <- forest_mask[[1]]
  
  target_area_10k_ha <- area_df[area_df$Year == current_year, province_cn]
  target_euca_pixels <- (target_area_10k_ha * 100000000) / 900
  mask_notna_count <- as.numeric(global(forest_mask, "notNA")[,1])
  target_euca_ratio <- target_euca_pixels / mask_notna_count
  
  cat(sprintf(" -> 目标面积: %.4f 万公顷 | 林地基数: %d | 计算占比: %.6f\n", 
              target_area_10k_ha, mask_notna_count, target_euca_ratio))
  
  message("[2] 加载模型与分块预测...")
  rf_model <- readRDS(model_path) 
  
  tif_files <- sort(list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE))
  if (length(tif_files) == 0) stop("特征目录中没有 TIF 文件：", tif_dir)
  temp_files <- character(length(tif_files))
  
  ref_raster <- rast(tif_files[1])
  forest_mask <- project(forest_mask, crs(ref_raster), res = res(ref_raster), method = "near")
  
  # 特征名称必须与 02_GEE_feature_extraction.js 和训练模型完全一致。
  # NRGDI = (Red - Green) / (Red + Green)，不再进行 NGRDI/NRGDI 的二次改名。
  raster_layer_names <- names(ref_raster)
  model_features <- rf_model$forest$independent.variable.names
  if (!is.null(model_features)) {
    missing_features <- setdiff(model_features, raster_layer_names)
    if (length(missing_features) > 0) {
      stop("输入特征与模型不一致，缺少：", paste(missing_features, collapse = ", "),
           "。请检查 GEE 波段名称及模型是否使用 NRGDI。")
    }
  }
  
  for (i in seq_along(tif_files)) {
    r_block <- tryCatch(rast(tif_files[i]), error = function(e) NULL)
    if (is.null(r_block)) next
    
    e_block <- ext(r_block)
    if (is.null(intersect(e_block, ext(forest_mask)))) next
    
    mask_crop <- crop(forest_mask, e_block)
    # [提速]：直接跳过整块全为 NA 的区域，不浪费任何时间
    if (as.numeric(global(mask_crop, "notNA")[,1]) == 0) next
    if (!compareGeom(r_block, mask_crop, stopOnError = FALSE)) mask_crop <- resample(mask_crop, r_block, method = "near")
    
    temp_file <- file.path(output_dir, paste0("temp_v3_", i, ".tif"))
    if(file.exists(temp_file)) try(file.remove(temp_file), silent = TRUE)
    
    pred_block <- rast(r_block, nlyrs = 1)
    
    readStart(r_block); readStart(mask_crop)
    writeStart(pred_block, filename = temp_file, overwrite = TRUE, datatype = "FLT4S")
    
    row_starts <- seq(1, nrow(r_block), by = chunk_size)
    for (start_row in row_starts) {
      nrows_to_read <- min(chunk_size, nrow(r_block) - start_row + 1)
      mask_chunk <- readValues(mask_crop, row = start_row, nrows = nrows_to_read, mat = FALSE)
      valid_cols <- which(!is.na(mask_chunk))
      
      chunk_preds <- rep(NA, length(mask_chunk))
      if (length(valid_cols) > 0) {
        feat_chunk <- readValues(r_block, row = start_row, nrows = nrows_to_read, mat = FALSE)
        feat_mat <- matrix(feat_chunk, ncol = nlyr(r_block), byrow = FALSE)
        
        # [核心提速]：利用 data.table 原地无复制转换，打破 R 语言 data.frame 内存与速度瓶颈
        feat_dt <- as.data.table(feat_mat[valid_cols, , drop = FALSE])
        setnames(feat_dt, raster_layer_names) # 这里应用了修正后的正确变量名
        setDF(feat_dt) 
        
        # [提速]：重新开启多线程预测
        rf_pred_obj <- predict(rf_model, data = feat_dt, num.threads = n_cores, verbose = FALSE)
        chunk_preds[valid_cols] <- rf_pred_obj$predictions[, "1"]
        
        rm(feat_dt, rf_pred_obj); gc()
      }
      writeValues(pred_block, chunk_preds, start_row, nrows_to_read)
    }
    readStop(r_block); readStop(mask_crop); writeStop(pred_block)
    temp_files[i] <- temp_file
    rm(r_block, mask_crop, pred_block); gc()
  }
  
  # --- 步骤 C: VRT 虚拟栅格处理 (干掉硬盘合并瓶颈) ---
  message("[3] 构建 VRT 并执行全局重分类...")
  valid_temp <- temp_files[file.exists(temp_files)]
  
  # [提速]：不进行 merge，直接生成 .vrt 虚拟文件，省去数分钟的硬盘写入时间
  vrt_path <- file.path(output_dir, paste0(province_pinyin, "_temp_", current_year, ".vrt"))
  vrt_obj <- vrt(valid_temp, vrt_path, overwrite = TRUE)
  
  # 直接在 VRT 上计算分位数
  quantile_val <- global(vrt_obj, fun = quantile, probs = 1 - target_euca_ratio, na.rm = TRUE)
  dynamic_threshold <- as.numeric(quantile_val[1, 1])
  
  # 直接在 VRT 上重分类并输出最终结果图
  final_out_path <- file.path(output_dir, final_output_file)
  rcl_matrix <- matrix(c(-Inf, dynamic_threshold, 0, dynamic_threshold, Inf, 1), ncol = 3, byrow = TRUE)
  final_rast <- classify(vrt_obj, rcl = rcl_matrix, filename = final_out_path, overwrite = TRUE, datatype = "INT1U")
  
  # (可选) 如果你还需要输出概率图，这步会生成；如果不需要可以注释掉这行
  prob_out_path <- file.path(output_dir, prob_output_file)
  writeRaster(vrt_obj, prob_out_path, overwrite = TRUE, datatype = "FLT4S")
  
  # --- 步骤 D: 清理 ---
  message("[4] 清理内存与临时文件...")
  unlink(valid_temp)
  if(file.exists(vrt_path)) unlink(vrt_path)
  rm(forest_mask, ref_raster, final_rast, vrt_obj, rf_model); gc(); gc()
  terra::tmpFiles(remove = TRUE)
  
  cat(sprintf("%d年处理完成。结果目录：%s\n", current_year, output_dir))
}

message("\n所有选定年份均已处理完成。")