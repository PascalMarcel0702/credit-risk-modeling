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
ggsave("output/figures/partial_residual_laufzeit.png", plot = p_laufzeit, width = 8, height = 6, dpi = 300, bg = "white")
# Conclusion: The loess smoothing line approximately follows a horizontal straight trend. There is no strong indication that modelling 'laufzeit' as additive linear effect is not adequate.
# Note, laufzeit is the only continuous covariate in this model, thus no other partial residual plots are regarded for model diagnostic.

# 2. Goodness-of-Fit Assessment -----------------------------
# 2.1 Goodness-of-Fit Assessment
# Residual Deviance Test (Asymptotic global fit)
# For binomial data, the residual deviance is adequately approximated by a \chi^2 - distribution
dev <- deviance(model_main) # 693.85
df_res <- df.residual(model_main) # 645

gof_p <- pchisq(dev, df = df_res, lower.tail = FALSE) #  0.1 > 0.089 > 0.05 
# Interpretation: Residual deviance is greater than expected value ( = df), but p - value of 0.089 > 0.05 fails to reject Null hypothesis of adequate model fit.

# Pearson Chi-Square Test
pearson_residuals <- residuals(model_main, type = "pearson")
pearson_chi2 <- sum(pearson_residuals^2) # 647.12
gof_chi_p <- pchisq(pearson_chi2, df = df_res, lower.tail = FALSE) # 0.4690684 >> 0.05
# Interpretation: Reject null hypothesis that the model is correctly specified. In particular, pearson chi-squared statistic is the sum of squared pearson residuals and thus represents the ratio of the squared empirical deviation to the theoretical binomial variance. Hence, no strong indication that the observed and theoretical variance does not match approximately.

# 2.2 To-Do: Dispersion Check
# Idea: Develop Score Mulitplier Test (Dean 1992) for testing on overdispersion.
# In case, model random success probabilities via Beta distribution which results in a beta-binomial model.
# Literature: https://www.math.cit.tum.de/fileadmin/w00ccg/math/Forschung/forschungsgruppen/statistics/academics/lec5.pdf (pp. 8-10)
# Dean Test 1992 : https://www.tandfonline.com/doi/pdf/10.1080/01621459.1992.10475225

# Save combined GoF metrics to the tables directory
gof_results <- data.frame(
  Test = c("Residual Deviance", "Pearson Chi-Square"),
  Statistic = c(dev, pearson_chi2),
  DF = c(df_res, df_res),
  P_Value = c(gof_p, gof_chi_p)
)
write.csv(
  gof_results,
  "output/tables/goodness_of_fit.csv",
  row.names = FALSE
)


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

# Plot 1: Pearson Residuals (measures predicted error)
p_res_pearson <- ggplot(df_residuals, aes(x = Fitted_Prob, y = Pearson)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", se = FALSE, color = "darkgreen", linewidth = 1) +
  geom_hline(yintercept = 0, color = "green", linewidth = 0.5, linetype = "dashed") +
  theme_light() +
  labs( x = "Predicted Probability", y = "Pearson Residual")
ggsave("output/figures/residuals_pearson.png", plot = p_res_pearson, width = 8, height = 5, dpi = 300, bg = "white")
# Interpretation: The LOESS curve is flat, indicating no structural misspecification such as chosen link function or modeled covariates.
# Note: Raw pearson residuals have no unit variance and thus changes with predicted probability, hence artifacts can hide underlying problems. Therefore, adjusted residuals are needed.

# Plot 2: Deviance Residuals (measures contribution to deviance)
p_res_dev <- ggplot(df_residuals, aes(x = Fitted_Prob, y = Deviance)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "orange", linewidth = 0.5, linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "darkorange", linewidth = 1) +
  theme_light() +
  labs( x = "Predicted Probability", y = "Deviance Residual")
ggsave("output/figures/residuals_deviance.png", plot = p_res_dev, width = 8, height = 5, dpi = 300)
# Interpretation: The LOESS curve is flat, indicating no structural misspecification such as chosen link function or modeled covariates.
# Note: Deviance residuals also have no unit variance and change with predicted probability, hence artifacts can hide underlying problems. Therefore, adjusted residuals are needed.


# Plot 3: Adjusted Pearson Residuals
p_res_adj <- ggplot(df_residuals, aes(x = Fitted_Prob, y = Adjusted)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "blue", linewidth = 0.5, linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "darkblue", linewidth = 1) +
  theme_light() +
  labs( x = "Predicted Probability", y = "Adjusted Residual")
ggsave("output/figures/residuals_adjusted.png", plot = p_res_adj, width = 8, height = 5, dpi = 300, bg = "white")

# Conclusion: The LOESS curve is flat, indicating no structural misspecification such as chosen link function or modeled covariates.
# Note: The adjusted residuals have unit variance - the mathematical artifact driven by theoretical variance is removed.
# Note: The residual plot shows an asymmetric bounding (-4 for unexpected defaults vs. +2 for unexpected successes). This could be statistical artifact driven by the high repayment rate (approx. 70%) in data

# 4. Influential Observations (Leverage & Cook's Distance) ---------------------
p <- length(coef(model_main))

# R Standard
cooks_d <- cooks.distance(model_main)

# Approximate Calculation of Cooks Distance (D_i^a) by using second-order Taylor expansion avoiding refitting the model n times. 
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
ggsave("output/figures/cooks_distance.png", plot = p_cooks_anal, width = 8, height = 5, dpi = 300, bg = "white")
# Conclusion: All analytic Cook's distances are well below the theoretical threshold of 1 (the maximum value is approximately 0.04). # Hence,no data point dominates the parameter estimation.
# Note: cooks.distance() scales cooks_d_approx by 1 / p, whereby p = #parameters.

# Quantify how much (in SEs) each regression coefficient changes if most influential observation is removed

# Identify index of obs. with highest approx. Cook's Distance
top_cooks_index <- which.max(cooks_d_approx) # 164

# Calculate DFBETAs for all obs.
dfb <- dfbetas(model_main)

# Extract DFBETA for extracted index
top_obs_dfb <- dfb[top_cooks_index, ]

# Calculate screening threshold
dfbeta_threshold <- 2 / sqrt(n_agg) # 0.078
# Coefficients exceeding threshold: moral1 (0.42), laufkont2 (-0.14), laufkont3 (0.08), laufkont4 (-0.11)
# These shifts < 1 (critical threshold). 

# Understand context:
top_profile <- credit_agg[top_cooks_index, ]
# Interpretation: Top outlier (Index 164) represents two borrowers who repaid (each kredit=1) despite a high-risk profile: no current account (laufkont=1) and a critical credit history with external debts (moral=1).

# Extract predicted probability for top outlier:
prob_outlier <- fitted(model_main)[top_cooks_index]

#The discrepancy between low predicted probability and empirical success is responsible for the large cooks distance.

# Export DFBETAS of coefficients and profile context of top outlier

top_outlier_summary <- data.frame(
  Coefficient     = names(top_obs_dfb),
  DFBETA          = round(top_obs_dfb, 2),
  Threshold       = round(dfbeta_threshold, 3),
  Exceeds         = abs(top_obs_dfb) > dfbeta_threshold,
  Predicted_Prob  = round(prob_outlier, 4),
  Obs_Repayments  = top_profile$kredit,
  Obs_Total       = top_profile$kredit + top_profile$no_kredit
)

write.csv(top_outlier_summary, "output/tables/top_outlier_analysis.csv", row.names = FALSE)


# Plot 3: Residuals vs Leverage (Combined Diagnostic Plot)
# Goal: Identify outliers (y-axis) and high-leverage points (x-axis)
leverage_threshold <- 2 * p / n_agg # 0.0275
p_res_lev <- ggplot(df_influence, aes(x = Leverage, y = Adjusted_Residual)) +
  geom_point(alpha = 0.7) +
  geom_vline(xintercept = leverage_threshold, color = "red", linetype = "dashed", linewidth = 0.5) +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed", linewidth = 0.5) +
  theme_light() +
  labs(x = expression(Leverage~(h[ii]^L)), y = "Adjusted Pearson Residual")
ggsave("output/figures/residuals_vs_leverage_plot.png", plot = p_res_lev, width = 8, height = 5, dpi = 300, bg = "white")

# Verify that point on top-right corresponds to id top_cooks_index
top_right_point <- df_influence %>% 
  filter(Leverage > 0.07, Adjusted_Residual > 2) # 164 - true

# Conclusion (y-axis): The plot detects outliers (large adjusted residuals) and extreme covariate profiles (high leverage). 
# Pure Outliers: Some points show large residuals near -4 (unexpected default) but, their leverage is low.
# High Leverage Points (x-axis): Numerous observations exceed the leverage threshold of 0.0275, with residuals in range of [-2, 2]. The model handles the extreme covariate profiles adequately.
# Influential Point: The single observation in the top right (high leverage > 0.07 and high residual > 2) corresponds to identified top outlier (Index 164). As verified via DFBETAs, it represents a valid extreme case and does not indicate model misspecification.