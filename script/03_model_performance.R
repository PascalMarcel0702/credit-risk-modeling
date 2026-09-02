# ==============================================================================
# Script: 03_model_performance.R
# Purpose: Out-of-sample testing, Risk Metrics (ROC/AUC, MSE)
# ==============================================================================
library(dplyr)
library(tidyverse)
library(rsample)
library(pROC)

# Create directory for performance metrics
dir.create("output/performance", recursive = TRUE, showWarnings = FALSE)

# Load Data and Model -------------------------------------------------------
data_train <- readRDS("output/data_train.rds")
data_test  <- readRDS("output/data_test.rds")
model_train <- readRDS("output/model_main.rds")

# Calculate probabilities for Train and Test -------------------------------------
phat_train <- predict(model_train, newdata = data_train, type = "response")
phat_test  <- predict(model_train, newdata = data_test, type = "response")

# Determine optimal threshold based on training data

roc_train <- roc(data_train$kredit, phat_train, quiet = TRUE, ci = TRUE)
auc_train <- auc(roc_train) # 0.7511
ci_train <- ci(roc_train) # 95% CI: 0.7125-0.7897

roc_test <- roc(data_test$kredit, phat_test, quiet = TRUE, ci = TRUE)
auc_test <- auc(roc_test) # 0.8094
ci_test  <- ci(roc_test) # 95% CI: 0.7577-0.861

# Note: The higher test AUC relative to the train AUC is a statistical artifact driven by sampling variance in the test split, which is also indicated by the greater 95% confidence interval width.


# Empirical Cost Minimization ------------------------------------------------
# Goal: Calculate threshold that minimizes costs

# Note: The costs are asymmetrically distributed: Defaults are more costly than foregone interest rate margins due to rejection of loan
# Cost structure:
# loss of capital (lc) = credit amount * 0.7   -> loss quote of 0.7 due to debt collection, debt sell
# interest rate margin (irm) = credit amount * laufzeit / 12 * 0.04 -> assumption: assumption of 4 % credit margin for consumer credits
# cost ratio (cr) = lc / irm = 0.7 / [laufzeit / 12 * 0.04] = 210 / laufzeit

# Calculate meadian of laufzeit
median_laufzeit <- median(data_train$laufzeit) # 18 (months)

# Cost ratio = 210 / 18 = 11.67 \sim 12 (conservative estimation)
#Interpretation: Credit default costs 12 time more than forgone interest rate margine
cost_ratio <- ceiling(
  0.7 / (median_laufzeit / 12 * 0.04)
)

# Calculate repayment probabilities for all training data points yielding exhaustive list of possible thresholds for the underlying model
all_thresholds <- roc_train$thresholds

# Helper function to calculate empirical loss
calculate_empirical_loss <- function(thresh, probs, actuals, cost_ratio) {
  # Predictors
  yhat <- ifelse(probs >= thresh, 1, 0)
  # Ensure the confusion matrix always has 2x2 dimensions using factors
  cm <- table(Predicted = factor(yhat, levels = c(0, 1)), 
              Actual = factor(actuals, levels = c(0, 1)))
  
  # Extract FP and FN
  FN <- cm["0", "1"] # Model says 0 (Reject), but Actual is 1 (Repayment) -> Missed Opportunity (Cost = 1)
  FP <- cm["1", "0"] # Model says 1 (Approve), but Actual is 0 (Default) -> Capital Loss (Cost = 12)
  
  # Calculate Total Loss
  total_loss <- (FN * 1) + (FP * cost_ratio)
  return(total_loss)
}

# Calculate losses w.r.t. all thresholds
loss_values <- sapply(all_thresholds, function(t) {
  calculate_empirical_loss(thresh = t, probs = phat_train, actuals = data_train$kredit, cost_ratio = cost_ratio)
})

# Find minimum
best_index <- which.min(loss_values)
best_threshold <- all_thresholds[best_index]

# Validation: Analytical Cost-Sensitive Threshold Optimization (pROC coords)

# Prevalence of repayments (positive class) in train data
prevalence_train <- mean(data_train$kredit == 1) # approx. 70% defaults

# Calculate optimal threshold w.r.t. business case cost minimization
optimal_threshold_table_train <- coords(
  roc_train,
  best.weights = c(1 / cost_ratio, prevalence_train), # best.weights = c( Cost(FP) / Cost(FN), prevalence of positive class )
  "best",
  ret = c("threshold", "specificity", "sensitivity"),
  transpose = FALSE
)

print(best_threshold == optimal_threshold_table_train$threshold[1]) # TRUE
# Cross-validation confirms analytical and empirical cost minimizations converge to the identical global optimum, verifying implementation consistency.


# Empirical Profit Maximization --------------------------------------------------------
# Goal: Calculate threshold that maximizes profit
calculate_net_profit <- function(thresh, probs, data) {
  # Predictors
  yhat <- ifelse(probs >= thresh, 1, 0)
  # Profit structure
  interest_rate_margin <- data$hoehe * data$laufzeit / 12 * 0.04 # Amount * duration (in years) * interest rate ( = 4 % p.a.)
  loss_of_capital <- data$hoehe * 0.70 # In default case, 70 % of capital is lost
  
  profit_vector <- ifelse( yhat == 1 & data$kredit == 1, interest_rate_margin,
                   ifelse(  yhat == 1 & data$kredit == 0, -loss_of_capital,
                   ifelse(   yhat == 0 & data$kredit == 1, -interest_rate_margin, 0)))
 
  return(sum(profit_vector))
}


# Calculate losses w.r.t. all thresholds
profit_values <- sapply(all_thresholds, function(t) {
  calculate_net_profit(thresh = t, probs = phat_train, data = data_train)
})

max_index <- which.max(profit_values)
max_threshold <- all_thresholds[max_index]

# Note: No analytical Cost-Sensitive Threshold Optimization (pROC coords) possible


# Comparison: Threshold w.r.t. cost minimization vs. profit maximization ----------

# ROC - metrics and plot -------------------------------------------------------
# Sensitivity (true positive rate, i.e., portion of defaults detected by the model )
# Specificity (true negative, portion of repayments detected by the model)

# Training data

# Cost minimization
metrics_train <- coords(roc_train, x = best_threshold, input = "threshold", 
                        ret = c("specificity", "sensitivity"), transpose = FALSE)
specificity_train <- metrics_train$specificity[1]
sensitivity_train <- metrics_train$sensitivity[1]

# Profit maximization
metrics_profit_train <- coords(roc_train, x = max_threshold, input = "threshold", 
                        ret = c("specificity", "sensitivity"), transpose = FALSE)
specificity_profit_train <- metrics_profit_train$specificity[1]
sensitivity_profit_train <- metrics_profit_train$sensitivity[1]


# Test data
metrics_test <- coords(roc_test, x = best_threshold, input = "threshold", 
                       ret = c("specificity", "sensitivity"), transpose = FALSE)
specificity_test <- metrics_test$specificity[1]
sensitivity_test <- metrics_test$sensitivity[1]

# Profit maximization
metrics_profit_test <- coords(roc_test, x = max_threshold, input = "threshold", 
                       ret = c("specificity", "sensitivity"), transpose = FALSE)
specificity_profit_test <- metrics_profit_test$specificity[1]
sensitivity_profit_test <- metrics_profit_test$sensitivity[1]

# Generate ROC plot using ggroc
p_roc <- ggroc(roc_test, legacy.axes = TRUE, linewidth = 1.2) +
  aes(color = "Test", linetype = "Test") +
  # Filled AUC (Test)
  geom_ribbon(
    aes(x = 1 - specificity, ymin = 0, ymax = sensitivity), 
    fill = "lightblue", alpha = 0.4, 
    show.legend = FALSE
  ) +
  # Training curve
  geom_path(
    data = data.frame(
      fpr = 1 - roc_train$specificities, 
      tpr = roc_train$sensitivities
    ),
    aes(x = fpr, y = tpr, color = "Train", linetype = "Train"),
    linewidth = 1.2,
    alpha = 0.7
  ) +
  # Benchmark: Random guess
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "darkgray", linewidth = 1) +
# Optimal (Profit) threshold points
annotate("point", x = 1 - specificity_profit_test, y = sensitivity_profit_test, color = "darkgreen", size = 3, shape = 17) +
  annotate("point", x = 1 - specificity_profit_train, y = sensitivity_profit_train, color = "royalblue", size = 3.5, shape = 2, stroke = 1.2) +
  # Label for Green Triangle (Test)
  annotate(
    "text",
    x = 1 - specificity_profit_test, 
    y = sensitivity_profit_test,
    label = sprintf("(%.3f | %.3f)", 1- specificity_profit_test, sensitivity_profit_test),
    color = "darkgreen",
    hjust = -0.5, vjust = -0.1, size = 4, fontface = "italic"
  ) +
  # Label for Blue Triangle (Train)
  annotate(
    "text",
    x = 1 - specificity_profit_train, 
    y = sensitivity_profit_train,
    label = sprintf("(%.3f | %.3f)", 1 - specificity_profit_train, sensitivity_profit_train),
    color = "royalblue",
    hjust = 0, vjust = 1.5, size = 4, fontface = "italic"
  ) +
  # Optimal (Costs) threshold points
  annotate("point", x = 1 - specificity_test, y = sensitivity_test, color = "darkgreen", size = 3) +
  annotate("point", x = 1 - specificity_train, y = sensitivity_train, color = "royalblue", size = 3.5, shape = 1, stroke = 1.2) +
  # Label for Green Point (Test)
  annotate(
    "text",
    x = 1 - specificity_test, 
    y = sensitivity_test,
    label = sprintf("(%.3f | %.3f)", 1- specificity_test, sensitivity_test),
    color = "darkgreen",
    hjust = -0.3, vjust = -0.1, size = 4, fontface = "italic"
  ) +
  # Label for Blue Point (Train)
  annotate(
    "text",
    x = 1 - specificity_train, 
    y = sensitivity_train,
    label = sprintf("(%.3f | %.3f)", 1 - specificity_train, sensitivity_train),
    color = "royalblue",
    hjust = -0.1, vjust = 0.8, size = 4, fontface = "italic"
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  scale_color_manual(
    name = NULL,
    values = c("Test" = "midnightblue", "Train" = "royalblue"),
    labels = c(
      "Test" = sprintf("Test (AUC = %.3f)", auc_test),
      "Train" = sprintf("Training (AUC = %.3f)", auc_train)
    )
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c("Test" = "solid", "Train" = "dashed"),
    labels = c(
      "Test" = sprintf("Test (AUC = %.3f)", auc_test),
      "Train" = sprintf("Training (AUC = %.3f)", auc_train)
    )
  ) +
  # Axis Labels
  labs(
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  # Theme and Design
  theme_minimal(base_size = 14) +
  theme(
    plot.margin = margin(t = 10, r = 20, b = 10, l = 10),
    legend.position = "inside",
    legend.position.inside = c(0.00, 1.00),
    legend.justification = c(0, 1),
    legend.background = element_blank(),
    legend.key.width = unit(1.5, "cm"),
    legend.text = element_text(size = 11)
  )

# Interpretation: <To do: sum it up>
#Discriminatory Performance: Both training and test ROC curves remain positioned above the random-guess baseline, indicating the model's general capacity to rank credit risk profiles across datasets.
#Cost-Optimal Operating Points: The markers reflect the implementation of the 12:1 cost-ratio threshold, yielding low false positive rates (FPR $\approx 1.1\%$ for test, $2.9\%$ for training) aligned with the objective of limiting high-cost default exposure.
#Threshold Transferability: The spatial alignment between the training and test operating points indicates that the decision threshold derived from the training data maintains consistent positioning when applied to the test set distribution.
#Profit-Driven Operating Points: The triangular markers illustrate the profit-maximizing strategy. By shifting the threshold to prioritize net revenue and market share, the model accepts a moderate absolute increase in False Positive Rates (FPR $\approx 6.7\%$ for test, $10.0\%$ for training) to achieve a substantial gain in True Positive Rates (Sensitivity jumping from $\approx 20\%$ to $\approx 42\%$), thereby capturing significantly more profitable loans.
#Strategic Trade-Off & Stability: The steep initial slope of the ROC curve visually justifies this strategic shift: a marginal concession in risk tolerance (x-axis) yields disproportionately large gains in approved good customers (y-axis). Furthermore, the spatial proximity of the train and test profit markers confirms that this milder, value-driven threshold is out-of-sample stable and does not degrade on unseen data.


# Calibration Plot ------------------------------------------------------------

# Calibration plot w.r.t. risk classes "A (Prime)", "B (Very Good)", "C (Good)", "D (Acceptable)", "E (High Risk)", "F (Watchlist)", "G (Default Risk)"

calib_data <- tibble(
  predicted_default_prob = 1 - phat_test,
  empirical_default = 1 - data_test$kredit
)
calib_scale <-  mutate(calib_data,
    # rating classes for predited_default_prob
    rating_grade = cut(
      predicted_default_prob,
      breaks = c(-Inf, 0.005, 0.015, 0.05, 0.10, 0.20, 0.50, Inf),
      labels = c("A", "B", "C", "D", "E", "F", "G"),
      right = TRUE
    )
  ) %>% 
  # Calibration curve on level of ratings
  group_by(rating_grade) %>% 
  summarise(
    mean_predicted_default_prob =  mean(predicted_default_prob),
    mean_empirical_default =  mean(empirical_default),
    records =  n(),
    se =  sqrt((mean_empirical_default * (1 - mean_empirical_default)) / records), # standard error for weighted sum of defaults, i.e., 
    ci_lower =  pmax(0, mean_empirical_default - 1.96 * se),
    ci_upper =  pmin(1, mean_empirical_default + 1.96 * se),
    .groups = "drop"
  )

ggplot(calib_scale, aes(x = mean_predicted_default_prob, y = mean_empirical_default)) +
  # calibration line y = x
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "darkgray", linewidth = 1) + 
  # CI of rating bins
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.02, color = "darkgreen", alpha = 1) +
  # Linear Spline
  geom_line(color = "darkgreen", linewidth = 1.2) +
  # Coordinates of Bins
  geom_point(color = "darkgreen", size = 4) +
  # Labeling
  geom_text(
    aes(label = sprintf("%s (n = %d)", rating_grade, records)),
    hjust = 1, 
    nudge_x = -0.02,
    nudge_y = 0.05,
    size = 4, 
    fontface = "italic",
    color = "black"
  ) +
  labs(
    x = "predicted default rate",
    y = "empirical default rate",
    color = "Records\nper Grade"
  ) +
  scale_x_continuous(limits = c(0, NA)) +
  scale_y_continuous(limits = c(0, NA)) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none"
  )

# General interpretation:
# 1) The point lies below the dashed line (y < x: The model predicts a higher failure rate than actually occurs. The model overestimates the risk (it is pessimistic/conservative).
# 2) The point lies above the dashed line (y > x): The model predicts a lower failure rate than actually occurs. The model underestimates the risk (it is optimistic/aggressive).

# Interpretation: Grouped by risk classes
# For the good customers (Class D, bottom left): The point lies below the perfect calibration line, in particular, the model predicts an average default rate of \sim 8% (x-axis), but empirically, the default is approx. 2% (y-axis). This means, the model assesses these customers as worse/riskier than they actually are. It overestimates their risk.

# For the poor customers (Classes F and G, top right): The points lie almost perfectly on the dashed line (y = x), indicating solid calibration for high-risk profiles. 
# Note: the width of the CI for point G is large (approx. 0.28), since:
# 1. Small sample size: Class G contains only 44 records (the smallest bin in this plot).
# 2. Variance maximization: The empirical default rate is  approx. 63%. The variance of a binomial distribution p(1-p) is maximized by p = 0.5.
# Conclusion: Despite the large CI, the model is accurate in estimating expected defaults for high-risk customers.


# Calibration plot w.r.t. deciles (10 equal-sized bins) -----------------------------
calib_decile <- calib_data %>% 
  mutate(
    # decile classes for predicted_default_prob
    decile = ntile(predicted_default_prob, 10)
  ) %>% 
  # Calibration curve on level of deciles
  group_by(decile) %>% 
  summarise(
    mean_predicted_default_prob =  mean(predicted_default_prob),
    mean_empirical_default =  mean(empirical_default),
    records =  n(),
    se =  sqrt((mean_empirical_default * (1 - mean_empirical_default)) / records), # standard error for weighted sum of defaults
    ci_lower =  pmax(0, mean_empirical_default - 1.96 * se),
    ci_upper =  pmin(1, mean_empirical_default + 1.96 * se),
    .groups = "drop"
  )

# Plot of calibration curve
ggplot(calib_decile, aes(x = mean_predicted_default_prob, y = mean_empirical_default)) +
  # calibration line y = x
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "darkgray", linewidth = 1) + 
  # LOESS smoothing over ungrouped data
  geom_smooth(
    data = calib_data,
    aes(x = predicted_default_prob, y = empirical_default),
    method = "loess", 
    se = TRUE, 
    color = "royalblue", 
    fill = "lightblue", 
    alpha = 0.3,
    linewidth = 1
  ) +
  # CI of decile bins
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.015, color = "midnightblue", alpha = 1) +
  # Linear Spline
  geom_line(color = "midnightblue", linewidth = 1.2) +
  # Coordinates of Bins
  geom_point(color = "midnightblue", size = 4) +
  # Vertical lines for thresholds
  geom_vline(xintercept = 1 - best_threshold, linetype = "dotted", color = "darkgreen", linewidth = 0.75) + # profit maximization
  geom_vline(xintercept = 1 - max_threshold, linetype = "dotted", color = "darkorange", linewidth = 0.75) + # cost minimization
  # Labeling
  # Labeling
  labs(
    x = "predicted default rate",
    y = "empirical default rate"
  ) +
  scale_x_continuous(limits = c(0, NA)) +
  scale_y_continuous(limits = c(0, NA)) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none"
  )

# Interpretation: Grouped by deciles
# Structural behavior: The LOESS curve has a cubic polynomial shape w.r.t. the calibration line (y = x) as its vertical.

# 1. Low predicted default rates (x < 0.30):
# The LOESS curve and deciles lie below the dashed line (y < x), i.e., the model systematically overestimates the default risk in this region (conservative bias).

# 2. Medium to high predicted default rates (0.30 <= x <= 0.65):
# The LOESS curve and deciles cross the dashed line and lie above it (y > x), i.e., the model systematically underestimates the default risk in this region (optimistic bias).

# 3. Extreme high predicted default rates (x > 0.65):
# The highest decile point aligns perfectly with the dashed line (y = x).The LOESS curve drops with a large CI, which is an artifact of horizontal data sparsity: while the highest decile is stretched across a wide probability range, resulting in large estimated variance.



# Final Conclusion: Comparison of Master Scale vs. Decile Calibration

# Does the technical decile plot confirm the risk class plot? 
# Yes, it confirms the global boundaries, but it reveals a hidden local vulnerability.
# 1. Confirmations (Consistency across both views):
# - Low-Risk Conservative Bias: Both plots show that for low-risk customers (Class D / deciles with x < 0.30), the model overestimates default risk. The resulting foregone interest rate margins are not as expensive as wrongly predicting a happended default.
# - Extreme High-Risk Accuracy: Both plots confirm that the absolute highest risk tier (Class G / 10th decile) is well-calibrated, placing the point on the calibration line.

# - Hidden Optimistic Bias: The technical plot reveals a systematic underestimation of risk in the medium-to-high range (0.30 <= x <= 0.65). The largest Class F (n = 149) aggregates this entire region into a single point, averaging out the variance and making it appear perfectly calibrated. However, the model is actually too optimistic.
# All in all, the LOESS plot highlights that the model underestimates medium-risk applicants w.r.t. their true default probability. 

# Conclusion: Thresholds (Cost vs. Profit)

  # - The Actionable Gap: The space along the x-axis between the strict cost threshold (green) and the milder profit threshold (orange) represents the strategic "opportunity zone". Applicants falling into this probability range are rejected under cost-minimization but approved under profit-maximization.
  # - Risk Buffering for Business Expansion: Crucially, this entire expansion zone lies below the perfect calibration diagonal (y < x). This means that as the bank shifts its policy from the green to the orange line to capture more market share, the newly accepted applicants are systematically less risky in reality than their predicted probabilities suggest.
  # - Final Justification: Choose the profit-maximizing threshold: The "marginal risk" taken to increase the approval rate is buffered by the model's conservative bias.


# Decision of Strategy: Cost minimization vs. profit maximization ---------------------------

# Helper function to calculate strategic KPIs
evaluate_strategy <- function(strategy_name, thresh, probs, data) {
  # Predictors
  yhat <- ifelse(probs >= thresh, 1, 0)
  # Approval rate, note yhat == 1 \iff repayment expected
  approval_rate <- mean(yhat == 1, na.rm = TRUE)
  # #False positives
  fp_count <- sum(yhat == 1 & data$kredit == 0, na.rm = TRUE)
  # Margin and Costs
  irm <- data$hoehe * (data$laufzeit / 12) * 0.04
  lc  <- data$hoehe * 0.70
  
  profit_vector <- ifelse(yhat == 1 & data$kredit == 1, irm,
                          ifelse(yhat == 1 & data$kredit == 0, -lc,
                                 ifelse(yhat == 0 & data$kredit == 1, -irm, 0)))
  
  net_profit <- sum(profit_vector, na.rm = TRUE)
  
  # lent capital
  lent_capital <- sum(data$hoehe[yhat == 1], na.rm = TRUE)
  
  # Profit margin on lent capital
  profit_margin <- ifelse(lent_capital > 0, net_profit / lent_capital, 0)
  
  # Output
  tibble(
    'Strategy' = strategy_name,
    'Threshold' = round(thresh, 3),
    'Expected Approval Rate' = sprintf("%.1f %%", approval_rate * 100),
    'Wrongly predicted defaults' = fp_count,
    'Netto-Profit' = sprintf("%.2f", net_profit),
    'Profit Margine' = sprintf("%.2f %%", profit_margin * 100)
  )
}

# Metrics for cost minimization
row_cost_min <- evaluate_strategy(
  strategy_name = "Cost minimization", 
  thresh = best_threshold,
  probs = phat_train, 
  data = data_train
)

# Metrics for profit maximization 
row_profit_max <- evaluate_strategy(
  strategy_name = "Profitmaximierung", 
  thresh = max_threshold,        # Dein ermittelter Threshold aus der Profit-Schleife
  probs = phat_train, 
  data = data_train
)

# Create table
strategy_comparison_table <- bind_rows(row_cost_min, row_profit_max)
print(strategy_comparison_table)

# Conclusion:


# Calculate predicted values

yhat_test <- ifelse(phat_test >= optimal_threshold, 1, 0)
yhat_train <- ifelse(phat_train >= optimal_threshold, 1, 0)

# Confusion Matrix
conf_matrix <- table(Predicted = yhat_test, Actual = data_test$kredit)

TN <- conf_matrix[1, 1] # True Negatives (76) - loan was not repaid, model prediction was correct
FN <- conf_matrix[1, 2] # False Negatives (72) - loan was repaid, model prediction was incorrect (foregone opportunity costs)
FP <- conf_matrix[2, 1] # False Positives (14) - loan was not repaid, model prediction was incorrect (critical)
TP <- conf_matrix[2, 2] # True Positives (139) - loan was repaid, model prediction was correct

sensitivity <- TP / (TP + FN) # True Positive Rate           0.6587678 -> 65.88 %
specificity <- TN / (TN + FP) # True Negative Rate           0.8444444 -> 84.44 %
precision   <- TP / (TP + FP) # Positive Predictive Value    0.9084967 -> 90.85 %


# Conclusion: In total, the test data contain 301 records. Note, the credit risk is asymmetric - false positives are more costly than false negatives. Credit defaults were predicted correctly at a rate of 84.44%,leaving 14 false positives. On the other hand, there are 72 false negatives. This means in 34.12 % (100 % - 65.88 %) of the cases, the model causes foregone interest margins, but approves 65.88 % of good loans correctly. Last, when the model predicts a repayment, this is true in 90.85 % of the cases.




#-------------------------------------------------------------------------- Adjustment for below is needed



# 2.3 Error Rates and Mean Squared Error (MSE)
test_accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
test_error <- 1 - test_accuracy
test_brier <- mean((phat_test - data_test$kredit)^2)

# Calculate in-sample baseline metrics for comparison
phat_train <- predict(model_train, newdata = data_train, type = "response")
yhat_train <- ifelse(phat_train >= 0.5, 1, 0)
train_error <- 1 - (sum(diag(table(Predicted = yhat_train, Actual = data_train$kredit))) / nrow(data_train))
train_brier <- mean((phat_train - data_train$kredit)^2)
roc_obj_train <- roc(data_train$kredit, phat_train, quiet = TRUE)
auc_value_train <- auc(roc_obj_train)

cat("\n--- Out-of-Sample vs In-Sample ---\n")
cat("AUC Value (Test):   ", round(auc_value, 4), "   | (Train):", round(auc_value_train, 4), "\n")
cat("Error Rate (Test):  ", round(test_error * 100, 2), "% | (Train):", round(train_error * 100, 2), "%\n")
cat("Brier Score (Test): ", round(test_brier, 4), "   | (Train):", round(train_brier, 4), "\n")

# 3. Export Metrics ------------------------------------------------------------
performance_metrics <- data.frame(
  Metric = c("Test AUC", "Train AUC", "Test Accuracy", "Test Error Rate", "Train Error Rate", "Test Brier Score", "Train Brier Score"),
  Value = c(auc_value, auc_value_train, test_accuracy, test_error, train_error, test_brier, train_brier)
)

write.csv(performance_metrics, "output/performance/classification_metrics.csv", row.names = FALSE)

conf_matrix_df <- as.data.frame(as.table(conf_matrix))
write.csv(conf_matrix_df, "output/performance/confusion_matrix.csv", row.names = FALSE)