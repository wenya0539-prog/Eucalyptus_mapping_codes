#### 1. Environment and package setup ####
options(encoding = "UTF-8")
if (Sys.info()["sysname"] == "Windows") {
  Sys.setlocale(category = "LC_ALL", locale = "Chinese")
  tryCatch({
    Sys.setlocale(category = "LC_ALL", locale = "Chinese_China.UTF-8")
  }, error = function(e) {
    Sys.setlocale(category = "LC_ALL", locale = "chs")
  })
} else {
  Sys.setlocale(category = "LC_ALL", locale = "zh_CN.UTF-8")
}

required_packages <- c("data.table", "caret", "dplyr", "ranger", "pROC")
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

#### 2. Read provincial samples and split training/testing data ####
PROJECT_ROOT <- "D:/Eucalyptus_mapping"
base_dir <- file.path(PROJECT_ROOT, "data", "samples")
provinces <- c("guangxi", "guangdong", "yunnan", "fujian", "hainan", "sichuan","guizhou","chongqing","hunan","jiangxi","zhejiang")
train_test_split_ratio <- 0.7 
seed_value <- 234
set.seed(seed_value)

train_list <- list()
test_list <- list()
cat("\n[1] 开始读取数据并划分 70% 训练集 / 30% 测试集...\n")
for (prov in provinces) {
  file_path <- file.path(base_dir, paste0(prov, ".csv"))
  if (!file.exists(file_path)) stop("缺少省级样本 CSV：", file_path)
  prov_data <- fread(file_path)
  if("ID" %in% names(prov_data)) prov_data$ID <- NULL
  prov_data$Province <- prov
  
  y_vector <- as.numeric(prov_data$EUCA)
  # 这里已经严格按照70%训练/30%验证进行了划分
  train_index <- createDataPartition(y_vector, p = train_test_split_ratio, list = FALSE)
  
  train_list[[prov]] <- prov_data[train_index, ]
  test_list[[prov]] <- prov_data[-train_index, ]
}

global_train_data <- rbindlist(train_list)

#### 3. Combine the provincial test subsets ####
cat("\n[2] 开始合并全局测试集 (保留全部30%验证样本)...\n")
# 直接合并 test_list，不再做 5000 条的上限抽样限制
global_test_data <- rbindlist(test_list)

#### 4. Train the probability-based random forest model ####
cat("\n[3] 准备特征并开始训练随机森林模型...\n")
feature_cols <- setdiff(names(global_train_data), c("EUCA", "Province", "ID"))

# 转换标签为 Factor (Ranger 跑分类概率的必要条件)
train_df <- copy(global_train_data)
train_df$EUCA <- as.factor(train_df$EUCA)
test_df <- copy(global_test_data)
test_df$EUCA <- as.factor(test_df$EUCA)

train_cols <- c(feature_cols, "EUCA")

rf_model <- ranger(
  dependent.variable.name = "EUCA",
  data = train_df[, ..train_cols],
  num.trees = 500,          
  mtry = floor(sqrt(length(feature_cols))), 
  probability = TRUE,       # 强制输出概率，用于计算 AUC 和动态阈值
  importance = 'impurity',
  num.threads = parallel::detectCores() - 1, 
  seed = 123
)

#### 5. Evaluate classification accuracy by province ####
cat("\n[4] 提取概率、计算指标并生成学术评估面板...\n")

# 获取测试集预测概率
rf_preds_obj <- predict(rf_model, data = test_df[, ..feature_cols])
test_pred_probs <- rf_preds_obj$predictions[, "1"]
y_test_numeric <- as.numeric(as.character(test_df$EUCA))

# 初始化存放所有省份结果的列表
report_rows <- list()

for (p in unique(test_df$Province)) {
  # 提取单省份数据
  idx <- which(test_df$Province == p)
  prov_probs <- test_pred_probs[idx]
  prov_true <- y_test_numeric[idx]
  
  # 1. 计算 AUC 和 动态阈值 (面积约束)
  prov_roc <- roc(prov_true, prov_probs, quiet = TRUE)
  prov_auc <- as.numeric(auc(prov_roc))
  
  prov_ratio <- sum(prov_true == 1) / length(prov_true)
  prov_threshold <- as.numeric(quantile(prov_probs, probs = 1 - prov_ratio))
  
  # 2. 生成预测类别
  prov_pred <- ifelse(prov_probs >= prov_threshold, 1, 0)
  
  # 3. 提取混淆矩阵基本元素
  TP <- sum(prov_pred == 1 & prov_true == 1)  # 预测桉树，真实桉树
  FP <- sum(prov_pred == 1 & prov_true == 0)  # 预测桉树，真实非桉树
  FN <- sum(prov_pred == 0 & prov_true == 1)  # 预测非桉树，真实桉树
  TN <- sum(prov_pred == 0 & prov_true == 0)  # 预测非桉树，真实非桉树
  
  # 4. 计算各项精度百分比 (%)
  UA_euca <- round(TP / (TP + FP) * 100, 2)
  PA_euca <- round(TP / (TP + FN) * 100, 2)
  
  UA_noneuca <- round(TN / (TN + FN) * 100, 2)
  PA_noneuca <- round(TN / (TN + FP) * 100, 2)
  
  OA <- round((TP + TN) / length(prov_true) * 100, 2)
  
  # 计算 Kappa
  pe <- ((TP+FP)*(TP+FN) + (FN+TN)*(FP+TN)) / (length(prov_true)^2)
  po <- (TP + TN) / length(prov_true)
  Kappa <- round((po - pe) / (1 - pe), 4)
  
  # 5. Store the two class rows for the provincial accuracy table.
  row1 <- data.frame(
    省份 = p,
    类别 = "桉树",
    `桉树(像元)` = TP,
    `非桉树(像元)` = FP,
    `UA(%)` = sprintf("%.2f", UA_euca),
    `PA(%)` = sprintf("%.2f", PA_euca),
    `OA(%)` = sprintf("%.2f", OA),
    Kappa = sprintf("%.4f", Kappa),
    AUC = sprintf("%.4f", prov_auc),
    Threshold = sprintf("%.4f", prov_threshold),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  row2 <- data.frame(
    省份 = "",
    类别 = "非桉树",
    `桉树(像元)` = FN,
    `非桉树(像元)` = TN,
    `UA(%)` = sprintf("%.2f", UA_noneuca),
    `PA(%)` = sprintf("%.2f", PA_noneuca),
    `OA(%)` = "",
    Kappa = "",
    AUC = "",
    Threshold = "",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  report_rows[[p]] <- rbind(row1, row2)
}

# 组合全局表格并打印
final_report_table <- do.call(rbind, report_rows)
rownames(final_report_table) <- NULL

cat("\n", rep("=", 90), "\n")
cat("学术标准精度评估面板 (随机森林 + 面积约束阈值 - 全量验证集)\n")
cat(rep("-", 90), "\n")
print(final_report_table, right = FALSE, row.names = FALSE)
cat(rep("=", 90), "\n")

# -------------------- 6. 保存模型与特征重要性 (可跳过) --------------------
model_save_dir <- file.path(PROJECT_ROOT, "models")
if (!dir.exists(model_save_dir)) dir.create(model_save_dir, recursive = TRUE)

imp_values <- rf_model$variable.importance
if(length(imp_values) > 0) {
  imp_df <- data.frame(Feature = names(imp_values), Importance = as.numeric(imp_values))
  imp_df <- imp_df[order(-imp_df$Importance), ]
  write.csv(imp_df, file.path(model_save_dir, "rf_feature_importance.csv"), row.names = FALSE)
  cat("\n随机森林特征重要性已保存至:", file.path(model_save_dir, "rf_feature_importance.csv"), "\n")
}

saveRDS(rf_model, file.path(model_save_dir, "eucalyptus_rf_southern_china_2020.rds"))
cat("随机森林模型已保存至:", model_save_dir, "\n")