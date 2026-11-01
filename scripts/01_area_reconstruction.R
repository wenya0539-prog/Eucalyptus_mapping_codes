library(dplyr)
library(tidyr)

Sys.setlocale("LC_CTYPE", "zh_CN.UTF-8")

raw_data <- data.frame(
  Province = c("云南", "广东", "广西", "海南", "四川", "重庆", "浙江", "福建", "江西", "湖南", "贵州"),
  `Y1999_2003` = c(5.76, 40.77, 14.88, 16.67, 2.43, 0, 0, 1.68, 0, 0, 0),
  `Y2004_2008` = c(24.00, 132.41, 53.36, 19.30, 6.32, 0.48, 0, 15.87, 1.28, 1.60, 0),
  `Y2009_2013` = c(38.88, 171.27, 165.24, 14.62, 16.02, 4.65, 0, 27.14, 4.49, 2.25, 0.96),
  `Y2014_2018` = c(42.24, 186.65, 256.05, 12.94, 18.43, 3.54, 0.24, 20.88, 1.28, 1.92, 2.57)
)

anchor_years <- c(2001, 2006, 2011, 2016)
target_years <- 2000:2020
results_list <- list()

for (i in 1:nrow(raw_data)) {
  prov <- raw_data$Province[i]
  y_vals <- as.numeric(raw_data[i, 2:5])
  
  interp <- spline(x = anchor_years, y = y_vals, xout = target_years, method = "natural")
  interp_y <- pmax(interp$y, 0)
  
  for (j in seq_along(target_years)) {
    ty <- target_years[j]
    
    # Rule 1: prevent artificial early-period area where the first anchor is zero.
    if (ty < anchor_years[1] && y_vals[1] == 0) {
      interp_y[j] <- 0
    }
    
    # Rule 2: retain zero between two adjacent zero-area anchors.
    left_anchors <- anchor_years[anchor_years <= ty]
    right_anchors <- anchor_years[anchor_years >= ty]
    if (length(left_anchors) > 0 && length(right_anchors) > 0) {
      val_left <- y_vals[which(anchor_years == max(left_anchors))]
      val_right <- y_vals[which(anchor_years == min(right_anchors))]
      if (val_left == 0 && val_right == 0) {
        interp_y[j] <- 0
      }
    }
    
    # Rule 3: constrain extrapolation after the final anchor to avoid an abrupt decline.
    if (ty > anchor_years[4] && y_vals[4] > 0) {
      # Set the lower bound to 50% of the final anchor area.
      min_allowed <- y_vals[4] * 0.5 
      if (interp_y[j] < min_allowed) {
        # If the spline falls below the bound, apply a gradual decay.
        interp_y[j] <- min_allowed + (interp_y[j-1] - min_allowed) * 0.5
      }
    }
  }
  
  # 保留5位小数
  final_y_formatted <- sprintf("%.5f", interp_y)
  
  results_list[[prov]] <- data.frame(
    Province = prov,
    Year = target_years,
    Area = as.numeric(final_y_formatted) 
  )
}

annual_data <- bind_rows(results_list)

export_data <- annual_data %>%
  mutate(Area = sprintf("%.5f", Area)) %>%
  pivot_wider(names_from = Province, values_from = Area) %>%
  rename(Year = Year)

# 输出唯一的省级年度面积先验文件，供年度预测与时序后处理共同使用。
# 若已从论文数据记录下载 Eucalyptus_area.csv，请将其放入同一位置；默认不覆盖已有文件。
PROJECT_ROOT <- "D:/Eucalyptus_mapping"
AREA_CSV <- file.path(PROJECT_ROOT, "data", "input", "Eucalyptus_area.csv")
dir.create(dirname(AREA_CSV), recursive = TRUE, showWarnings = FALSE)

if (!file.exists(AREA_CSV)) {
  write.csv(export_data, AREA_CSV, row.names = FALSE, fileEncoding = "GBK")
  message("年度面积先验表已保存：", AREA_CSV)
} else {
  message("检测到已存在的 Eucalyptus_area.csv；为避免覆盖外部数据记录中的文件，本次未覆盖：", AREA_CSV)
}

# 在控制台查看结果
print(export_data, n = 21)