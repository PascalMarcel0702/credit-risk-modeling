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
# Conclusion: The loess smoothing line approximately follows a horizontal straight path, 
# supporting a linear specification for 'laufzeit' in the logit model.
#Note, laufzeit is the only continous covariate in this model

# 2. Goodness-of-Fit Assessment & Dispersion Check -----------------------------

# 2.1 Residual Deviance Test (Asymptotic global fit)
dev <- deviance(model_main)
df_res <- df.residual(model_main)
crit_val <- qchisq(0.95, df = df_res)
gof_p <- pchisq(dev, df = df_res, lower.tail = FALSE)

# 2.2 Pearson Chi-Square Dispersion Check (Heterogeneity Factor)
pearson_residuals <- residuals(model_main, type = "pearson")
pearson_chi2 <- sum(pearson_residuals^2)
dispersion_param <- pearson_chi2 / df_res

# Print results to console
cat("--- Goodness-of-Fit & Dispersion ---\n")
cat("Residual Deviance:", dev, "\nDegrees of Freedom:", df_res, 
    "\nCritical Chi-sq Value:", crit_val, "\nGoodness-of-Fit p-value:", gof_p, "\n")
cat("Pearson Chi-Square:", pearson_chi2, 
    "\nEstimated Dispersion Parameter (sigma^2):", dispersion_param, "\n\n")

# Save combined GoF metrics to the tables directory
gof_results <- data.frame(
  Metric = c("Residual Deviance", "Degrees of Freedom", "Critical Chi-sq Value", 
             "p-value (Deviance)", "Pearson Chi-Square", "Dispersion Parameter (sigma^2)"),
  Value = c(dev, df_res, crit_val, gof_p, pearson_chi2, dispersion_param)
)
write.csv(gof_results, "output/tables/goodness_of_fit.csv", row.names = FALSE)

# Conclusion: 
# 1. Since dev < crit_val (and p-value > 0.05), there is no evidence of lack of fit.
# 2. A dispersion parameter (sigma^2) close to 1 indicates that the observed variance 
#    matches the theoretical binomial variance, formally ruling out severe overdispersion.

# 3. Residual Analysis (Pearson, Deviance, Adjusted) ---------------------------
# Objective: Check for systematic lack of fit across the probability spectrum 
# by plotting residuals against the fitted probabilities (predicted values).

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
  geom_hline(yintercept = 0, color = "green", linewidth = 0.5, linetype = "dashed")
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
# 4. Influential Observations (Leverage & Cook's Distance) ---------------------
p <- length(coef(model_main))
leverage_threshold <- 2 * p / n_agg

cooks_d <- cooks.distance(model_main)

# Approximative Calculation of Cooks Distance (D_i^a) by using second-order Taylor expansion 
# avoiding refitting the model n times. 
# Formula: D_i^a = (e_i^P)^2 * [h_ii^L / (1 - h_ii^L)^2]
cooks_d_approx <- (res_pearson^2) * (leverage / (1 - leverage)^2)

# Note: R's default cooks.distance() scales the value by the number of parameters (p).
# We strictly use the unscaled approximation D_i^a from the literature to correctly 
# apply the absolute threshold of 1.

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
  geom_hline(yintercept = 1, color = "red", linetype = "dashed", linewidth = 1) +
  theme_light() +
  labs(x = "Observation Index", y = expression(Approximate~Cooks~Distance~(D[i]^a)))
ggsave("output/figures/cooks_distance.png", plot = p_cooks, width = 8, height = 5, dpi = 300)
# Conclusion: All approximate Cook's distances are well below the critical 
# threshold of 1 (the maximum value is approximately 0.4). This confirms 
# that there are no highly influential observations dictating the model fit. 
# The estimated parameter coefficients are robust and not driven by 
# individual data points.

# Plot 2: Analytic Cook's Distance Plot (R Default)
p_cooks_anal <- ggplot(df_influence, aes(x = Index, y = CooksDAnal)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 1, color = "red", linetype = "dashed") +
  theme_light() +
  labs(x = "Observation Index", y = "Cook's Distance (D_i)")
ggsave("output/figures/anal_cooks_distance.png", plot = p_cooks_anal, width = 8, height = 5, dpi = 300)
# Conclusion: R's default Cook's distance yields a maximum value of roughly 0.1. 
# This value is lower than our manual approximation (0.4) strictly because R 
# automatically scales the output by the number of parameters (p). Note that the 
# literature actually warns that the unscaled approximation tends to underestimate 
# the true distance. Regardless of the scaling, all values remain safely below 
# the threshold of 1, confirming the complete absence of influential observations.

# Plot 3: Residuals vs Leverage (Combined Diagnostic Plot)
# Objective: Simultaneously identify outliers (y-axis) and high-leverage points (x-axis).
p_res_lev <- ggplot(df_influence, aes(x = Leverage, y = Adjusted_Residual)) +
  geom_point(alpha = 0.7) +
  geom_vline(xintercept = leverage_threshold, color = "red", linetype = "dashed", linewidth = 0.5) +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed", linewidth = 0.5) +
  geom_smooth(method = "loess", se = FALSE, color = "darkblue", linewidth = 1) +
  theme_light() +
  labs(x = expression(Leverage~(h[ii]^L)), y = "Adjusted Pearson Residual")
ggsave("output/figures/residuals_vs_leverage_plot.png", plot = p_res_lev, width = 8, height = 5, dpi = 300)
# Conclusion: The red dashed line marks the theoretical high-leverage threshold 
# of 2p/n. While several observations exceed this threshold (falling to the right), 
# their adjusted Pearson residuals remain moderate (mostly between -2 and +2). 
# The slight upward curve of the LOESS smooth in the high-leverage region is a 
# typical data-sparsity boundary effect. Crucially, because no points exhibit 
# simultaneously extreme leverage and extreme residuals, no single observation 
# exerts undue influence on the fitted model.