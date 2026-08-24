# ==============================================================================
# Script: 02_model_diagnostics.R
# Purpose: Functional form check, Goodness-of-Fit, and Regression Diagnostics
# ==============================================================================

library(tidyverse)

# 0. Load Data and Model -------------------------------------------------------
credit_agg <- readRDS("output/credit_agg.rds")

model_main <- readRDS("output/model_main.rds")

# 1. Functional Form Check (Partial Residuals for 'laufzeit') ------------------
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
    title = "Partial Residual Plot: laufzeit",
    x = "Duration in Months (laufzeit)",
    y = "Partial Residual"
  )
ggsave("output/figures/partial_residual_laufzeit.png", plot = p_laufzeit, width = 8, height = 6, dpi = 300)
# Conclusion: The loess smoothing line approximately follows a horizontal straight path, 
# supporting a linear specification for 'laufzeit' in the logit model.


# 2. Goodness-of-Fit Assessment (Residual Deviance Test) -----------------------
dev <- deviance(model_main)
df_res <- df.residual(model_main)
crit_val <- qchisq(0.95, df = df_res)
gof_p <- pchisq(dev, df = df_res, lower.tail = FALSE)

cat("Residual Deviance:", dev, "\nDegrees of Freedom:", df_res, 
    "\nCritical Chi-sq Value:", crit_val, "\nGoodness-of-Fit p-value:", gof_p, "\n")

# Save GoF metrics to the tables directory
gof_results <- data.frame(
  Metric = c("Residual Deviance", "Degrees of Freedom", "Critical Chi-sq Value", "p-value"),
  Value = c(dev, df_res, crit_val, gof_p)
)
write.csv(gof_results, "output/tables/goodness_of_fit.csv", row.names = FALSE)
# Conclusion: Since dev < crit_val (and p-value > 0.05), there is no evidence of 
# lack of fit or severe overdispersion based on the residual deviance.


# 3. Residual Analysis (Pearson, Deviance, Adjusted) ---------------------------
leverage <- hatvalues(model_main)
res_pearson <- residuals(model_main, type = "pearson")
res_dev <- residuals(model_main, type = "deviance")
res_adj <- res_pearson / sqrt(1 - leverage) # Adjusted to stabilize variance

n_agg <- nrow(credit_agg)
df_residuals <- data.frame(
  Index = 1:n_agg,
  Pearson = res_pearson,
  Deviance = res_dev,
  Adjusted = res_adj
)

# Plot 1: Pearson Residuals
p_res_pearson <- ggplot(df_residuals, aes(x = Index, y = Pearson)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, color = "green", linewidth = 1) +
  theme_light() +
  labs(title = "Pearson Residuals", x = "Observation Index", y = "Pearson Residual")
ggsave("output/figures/residuals_pearson.png", plot = p_res_pearson, width = 8, height = 5, dpi = 300)

# Plot 2: Deviance Residuals
p_res_dev <- ggplot(df_residuals, aes(x = Index, y = Deviance)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, color = "orange", linewidth = 1) +
  theme_light() +
  labs(title = "Deviance Residuals", x = "Observation Index", y = "Deviance Residual")
ggsave("output/figures/residuals_deviance.png", plot = p_res_dev, width = 8, height = 5, dpi = 300)

# Plot 3: Adjusted Pearson Residuals
p_res_adj <- ggplot(df_residuals, aes(x = Index, y = Adjusted)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, color = "blue", linewidth = 1) +
  theme_light() +
  labs(title = "Adjusted Pearson Residuals", x = "Observation Index", y = "Adjusted Residual")
ggsave("output/figures/residuals_adjusted.png", plot = p_res_adj, width = 8, height = 5, dpi = 300)
# Conclusion: The residuals oscillate symmetrically around zero with constant variance, 
# indicating no misspecification of the link function.


# 4. Influential Observations (Leverage & Cook's Distance) ---------------------
p <- length(coef(model_main))
leverage_threshold <- 2 * p / n_agg
cooks_d <- cooks.distance(model_main)

# 4.1 Leverage Plot
p_leverage <- ggplot(data.frame(Index = 1:n_agg, Leverage = leverage), aes(x = Index, y = Leverage)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = leverage_threshold, color = "red", linetype = "dashed") +
  theme_light() +
  labs(title = "Leverage Analysis", x = "Observation Index", y = "Hat Value (h_ii)")
ggsave("output/figures/leverage_plot.png", plot = p_leverage, width = 8, height = 5, dpi = 300)

# 4.2 Cook's Distance Plot
p_cooks <- ggplot(data.frame(Index = 1:n_agg, CooksD = cooks_d), aes(x = Index, y = CooksD)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 1, color = "red", linetype = "dashed") +
  theme_light() +
  labs(title = "Cook's Distance", x = "Observation Index", y = "Cook's Distance (D_i)")
ggsave("output/figures/cooks_distance.png", plot = p_cooks, width = 8, height = 5, dpi = 300)
# Conclusion: Although some points exceed the leverage threshold, all Cook's distances 
# are well below 1. Thus, there are no highly influential observations dictating the model.