# ==============================================================================
# Script: 02_model_diagnostics.R
# Purpose: Functional form check, Goodness-of-Fit, and Regression Diagnostics
# ==============================================================================

library(tidyverse)

# 0. Load Data and Model -------------------------------------------------------
credit_agg <- readRDS("output/credit_agg.rds")

model_main <- readRDS("output/model_main.rds")

# 1. Functional Form Check (Partial Residuals for 'laufzeit)------------------
partial_residuals <- residuals(model_main, type = "partial")

partial_laufzeit <- data.frame(
  laufzeit = credit_agg$laufzeit,
  partial_residual = partial_residuals[, "laufzeit"]
)

p_laufzeit <- ggplot(partial_laufzeit, aes(x = laufzeit, y = partial_residual)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = TRUE, color = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_light() +
  labs(
    x = "Duration in Months (laufzeit)",
    y = "Partial Residual"
  )
ggsave("output/figures/partial_residual_laufzeit.png", plot = p_laufzeit, width = 8, height = 6, dpi = 300)
# Conclusion: The loess smoothing line approximately follows a horizontal straight trend, supporting a linear specification for 'laufzeit' in the logit model.
# Note, laufzeit is the only continuous covariate in this model

# 2. Goodness-of-Fit Assessment & Dispersion Check -----------------------------

# 2.1 Residual Deviance Test (Asymptotic global fit)
dev <- deviance(model_main) # 693.85
df_res <- df.residual(model_main) # 654
crit_val <- qchisq(0.95, df = df_res) # 705.19 > dev
gof_p <- pchisq(dev, df = df_res, lower.tail = FALSE) #  0.1 > 0.089 > 0.05 -> no formal statistical justification to reject model fit, but indication of slight excess variation.

# 2.2 Pearson Chi-Square Dispersion Check (Heterogeneity Factor)
pearson_residuals <- residuals(model_main, type = "pearson")
pearson_chi2 <- sum(pearson_residuals^2) # 647.12
dispersion_param <- pearson_chi2 / df_res # 1.003292 \sim 1

# Save combined GoF metrics to the tables directory
gof_results <- data.frame(
  Metric = c("Residual Deviance", "Degrees of Freedom", "Critical Chi-sq Value", 
             "p-value (Deviance)", "Pearson Chi-Square", "Dispersion Parameter (sigma^2)"),
  Value = c(dev, df_res, crit_val, gof_p, pearson_chi2, dispersion_param)
)
write.csv(gof_results, "output/tables/goodness_of_fit.csv", row.names = FALSE)

# Conclusion: 
# 1. Since dev < crit_val (and p-value > 0.05), there is no strong evidence of lack of fit.
# 2. A dispersion parameter (sigma^2) close to 1 indicating no strong evidence of overdispersion, since observed variance is approximately equal to the theoretical (binomial) variance. In particular, no strong evidence to use Beta-Binomial-Model instead.

# 3. Residual Analysis (Pearson, Deviance, Adjusted) ---------------------------
# Objective: Check for systematic lack of fit by plotting residuals against the fitted probabilities (predicted values).

# Interpretation (Residual Signs):
# - Negative Residual (Unexpected Default): Model predicted high repayment probability, but loan defaulted (y = 0).
# - Positive Residual (Unexpected Success): Model predicted high default risk, but loan was repaid (y = 1).

leverage <- hatvalues(model_main)
res_pearson <- residuals(model_main, type = "pearson")
res_dev <- residuals(model_main, type = "deviance")
res_adj <- res_pearson / sqrt(1 - leverage) # Adjusted to stabilize variance; Formula: e_i^a = e_i^P / sqrt(1 - h_ii^L)

# Predicted probabilities
fitted_probs <- fitted(model_main)

n_agg <- nrow(credit_agg)
df_residuals <- data.frame(
  Index = 1:n_agg,
  Fitted_Prob = fitted_probs,
  Pearson = res_pearson,
  Deviance = res_dev,
  Adjusted = res_adj
)

# Plot 1: Pearson Residuals
p_res_pearson <- ggplot(df_residuals, aes(x = Fitted_Prob, y = Pearson)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", se = FALSE, color = "darkgreen", linewidth = 1) +
  geom_hline(yintercept = 0, color = "green", linewidth = 0.5, linetype = "dashed") +
  theme_light() +
  labs( x = "Predicted Probability", y = "Pearson Residual")
ggsave("output/figures/residuals_pearson.png", plot = p_res_pearson, width = 8, height = 5, dpi = 300)
# Conclusion: The flat LOESS curve confirms the overall appropriateness of the model 
# and its link function. However, since raw Pearson residuals do not possess unit 
# variances, this plot cannot be used to reliably assess constant variance or true 
# outliers. For those checks, adjusted residuals are strictly required.


# Plot 2: Deviance Residuals
p_res_dev <- ggplot(df_residuals, aes(x = Fitted_Prob, y = Deviance)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "orange", linewidth = 0.5, linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "darkorange", linewidth = 1) +
  theme_light() +
  labs( x = "Predicted Probability", y = "Deviance Residual")
ggsave("output/figures/residuals_deviance.png", plot = p_res_dev, width = 8, height = 5, dpi = 300)
# Conclusion: The deviance residuals plotted against fitted values evaluate the overall 
# appropriateness of the model. The LOESS smooth shows only a negligible linear tilt 
# and lacks severe non-linear patterns, supporting the model structure. However, 
# because deviance residuals do not possess unit variances, this minor boundary 
# variation may be a structural artifact. Adjusted residuals must be used for final 
# variance assessment.


# Plot 3: Adjusted Pearson Residuals
p_res_adj <- ggplot(df_residuals, aes(x = Fitted_Prob, y = Adjusted)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "blue", linewidth = 0.5, linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "darkblue", linewidth = 1) +
  theme_light() +
  labs( x = "Predicted Probability", y = "Adjusted Residual")
ggsave("output/figures/residuals_adjusted.png", plot = p_res_adj, width = 8, height = 5, dpi = 300)

# Conclusion: The adjusted Pearson residuals, which correct for leverage and stabilize 
# variance, fluctuate symmetrically around zero. The LOESS curve is remarkably flat 
# across the predicted probability spectrum. Minor deviations at the extreme ends are 
# typical smoothing boundary artifacts. This firmly confirms the appropriateness of the 
# model and the correct specification of the logit link function.


# 4. Influential Observations (Leverage & Cook's Distance) ---------------------
p <- length(coef(model_main))

# R Standard
cooks_d <- cooks.distance(model_main)

# Approximative Calculation of Cooks Distance (D_i^a) by using second-order Taylor expansion avoiding refitting the model n times. 
# Formula: D_i^a = (e_i^P)^2 * [h_ii^L / (1 - h_ii^L)^2]
cooks_d_approx <- (res_pearson^2) * (leverage / (1 - leverage)^2)

# Thresholds for screening
theoretical_threshold <- 1 # critical threshold
practical_threshold <- 4 / n_agg # 0.006116208 for sensitive screening


df_influence <- data.frame(
  Index = 1:n_agg,
  Leverage = leverage,
  CooksD = cooks_d_approx,
  CooksDAnal = cooks_d,
  Adjusted_Residual = res_adj
)

# Plot 1: Approximate Cook's Distance Plot
p_cooks <- ggplot(df_influence, aes(x = Index, y = CooksD)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = practical_threshold, color = "orange", linetype = "dashed", linewidth = 1) +
  theme_light() +
  labs(x = "Observation Index", y = expression(Approximate~Cooks~Distance~(D[i]^a)))
# Conclusion: All approximate Cook's distances are well below the theoretical threshold of 1 (the maximum value is approximately 0.4). # Hence,no data point dominates the parameter estimation of the model


# Plot 2: Analytic Cook's Distance Plot (R Default)
p_cooks_anal <- ggplot(df_influence, aes(x = Index, y = CooksDAnal)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = practical_threshold, color = "red", linetype = "dashed") +
  theme_light() +
  labs(x = "Observation Index", y = "Cook's Distance (D_i)")
ggsave("output/figures/cooks_distance.png", plot = p_cooks_anal, width = 8, height = 5, dpi = 300)
# Conclusion: All approximate Cook's distances are well below the theoretical threshold of 1 (the maximum value is approximately 0.4). # Hence,no data point dominates the parameter estimation of the model.
# Note: cooks.distance() scales cooks_d_approx by 1 / p, whereby p = #parameters.

# Quantify how much (in \sigma^2) each regerssion coefficient changes if most influential observation is removed

# Identify index ob obs. with highest approx. Cook's Distance
top_cooks_index <- which.max(cooks_d_approx) # 164

# Calculate DFBETAs for all obs.
dfb <- dfbetas(model_main)

# Extract DFBETA for extracted index
top_obs_dfb <- dfb[top_cooks_index, ]

# Calculate screening threshold
dfbeta_threshold <- 2 / sqrt(n_agg) # 0.078
# Coefficients exceeding threshold: moral1 (0.42), laufkont2 (-0.14), laufkont3 (0.08), laufkont4 (-0.11)
# These shifts remain below critical threshold of 1. 

# Inspect the raw data of the top outlier to understand context
credit_agg[top_cooks_index, ]
# Conclusion: The top outlier (Index 164) represents two borrowers who repaid (each kredit=1) despite a severe high-risk profile: no current account (laufkont=1) and a critical credit history with external debts (moral=1). This discrepancy between low predicted probability and empirical success drives the maximum Cook's distance.

# Plot 3: Residuals vs Leverage (Combined Diagnostic Plot)
# Goal: Identify outliers (y-axis) and high-leverage points (x-axis)
leverage_threshold <- 2 * p / n_agg # 0.0275
p_res_lev <- ggplot(df_influence, aes(x = Leverage, y = Adjusted_Residual)) +
  geom_point(alpha = 0.7) +
  geom_vline(xintercept = leverage_threshold, color = "red", linetype = "dashed", linewidth = 0.5) +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed", linewidth = 0.5) +
  geom_smooth(method = "loess", se = FALSE, color = "darkblue", linewidth = 1) +
  theme_light() +
  labs(x = expression(Leverage~(h[ii]^L)), y = "Adjusted Pearson Residual")
ggsave("output/figures/residuals_vs_leverage_plot.png", plot = p_res_lev, width = 8, height = 5, dpi = 300)

# Verify that point on top-right corresponds to id top_cooks_index
top_right_point <- df_influence %>% 
  filter(Leverage > 0.07, Adjusted_Residual > 2) # 164 - true

# Conclusion: The plot simultaneously evaluates outliers (large adjusted residuals) and extreme covariate profiles (high leverage). 
# 1. Pure Outliers: Several points exhibit large negative residuals (near -4), indicating unexpected defaults  - but, their leverage is low, i.e., low influence on the model parameters.
# 2. High Leverage Points: Many observations on the right of horizontal red line (leverage threshold of 2p/n (0.0275)), but their residuals bounded near 0, i.e., the model prediction is approximately correct
# 3. Influential Point: The single observation in the top right (high leverage > 0.07 and high residual > 2) corresponds to identified top outlier (Index 164). DFBETAs analysis showed this unexpected success shifts specific parameters significantly - but these shifts remain below the critical threshold of 1. This obs. represents a valid extreme case, but does not yield strong evidence of misspecification. 