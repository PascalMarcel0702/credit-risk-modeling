# ==============================================================================
# Script: 03_model_performance.R
# Purpose: Out-of-sample testing, Risk Metrics (ROC/AUC, MSE)
# ==============================================================================

library(tidyverse)
library(rsample)
library(pROC)

# Create directory for performance metrics
dir.create("output/performance", recursive = TRUE, showWarnings = FALSE)

# 1. Data Preparation (Disaggregated Data for Classification) ------------------
credit <- read.table(file = "data/credit.txt", header = TRUE, sep = "")

credit$moral <- as.factor(credit$moral)
credit$laufkont <- as.factor(credit$laufkont)
credit$beruf <- as.factor(credit$beruf)

# 2. Stratified Train-Test Split -----------------------------------------------
set.seed(234)

# Stratified sampling ensures the exact class distribution (repayment rate) is maintained in both sets
data_split <- initial_split(credit, prop = 0.7, strata = kredit)
data_train <- training(data_split)
data_test  <- testing(data_split)

# 3. Model Training (In-Sample) ------------------------------------------------
model_train <- glm(
  kredit ~ laufzeit + moral + laufkont + alter, 
  data = data_train, 
  family = binomial(link = "logit")
)

# 4. Out-of-Sample Prediction & Evaluation -------------------------------------
phat_test <- predict(model_train, newdata = data_test, type = "response")
yhat_test <- ifelse(phat_test >= 0.5, 1, 0)

# 4.1 Confusion Matrix
conf_matrix <- table(Predicted = yhat_test, Actual = data_test$kredit)
print("Confusion Matrix (Test Data):")
print(conf_matrix)

# 4.2 ROC Curve and AUC (Area Under the Curve)
roc_obj <- roc(data_test$kredit, phat_test, quiet = TRUE)
auc_value <- auc(roc_obj)

# Export geometric interpretation of AUC
png("output/performance/roc_curve.png", width = 1800, height = 1500, res = 300)
plot(
  roc_obj, 
  main = "ROC Curve with AUC Area", 
  col = "darkblue", 
  lwd = 2, 
  print.auc = TRUE, 
  auc.polygon = TRUE, 
  auc.polygon.col = "lightblue", 
  print.thres = "best", 
  print.thres.pch = 19,         # Ausgefüllter, markanter Punkt
  print.thres.cex = 1.2,        # Größere Schrift und größerer Punkt
  print.thres.col = "darkred",  # Rote Farbe zur Hervorhebung
  print.thres.pattern = "Optimal Cut-Off: %.3f \n(Spec: %.3f, Sens: %.3f)" # Erklärender Text
)
dev.off()
# 4.3 Error Rates and Mean Squared Error (MSE)
test_accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
test_error <- 1 - test_accuracy
test_brier <- mean((phat_test - data_test$kredit)^2)

# Calculate in-sample baseline metrics for comparison
phat_train <- predict(model_train, type = "response")
yhat_train <- ifelse(phat_train >= 0.5, 1, 0)
train_error <- 1 - (sum(diag(table(Predicted = yhat_train, Actual = data_train$kredit))) / nrow(data_train))
train_brier <- mean((phat_train - data_train$kredit)^2)
roc_obj_train <- roc(data_train$kredit, phat_train, quiet = TRUE)
auc_value_train <- auc(roc_obj_train)

cat("\n--- Out-of-Sample vs In-Sample ---\n")
cat("AUC Value (Test):   ", round(auc_value, 4), "   | (Train):", round(auc_value_train, 4), "\n")
cat("Error Rate (Test):  ", round(test_error * 100, 2), "% | (Train):", round(train_error * 100, 2), "%\n")
cat("Brier Score (Test): ", round(test_brier, 4), "   | (Train):", round(train_brier, 4), "\n")

# 5. Export Metrics ------------------------------------------------------------
performance_metrics <- data.frame(
  Metric = c("Test AUC", "Train AUC", "Test Accuracy", "Test Error Rate", "Train Error Rate", "Test Brier Score", "Train Brier Score"),
  Value = c(auc_value, auc_value_train, test_accuracy, test_error, train_error, test_brier, train_brier)
)

write.csv(performance_metrics, "output/performance/classification_metrics.csv", row.names = FALSE)

conf_matrix_df <- as.data.frame(as.table(conf_matrix))
write.csv(conf_matrix_df, "output/performance/confusion_matrix.csv", row.names = FALSE)