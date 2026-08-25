# ==============================================================================
# Script: 01_data_prep_and_selection.R
# Purpose: Data inspection, functional form assessment (EDA, Partial Residuals),  
#          and model selection for credit repayment prediction.
# ==============================================================================

# 1. Setup ---------------------------------------------------------------------
library(tidyverse)
library(interactions)
library(gam)
library(rsample)
library(utf8)

# Create output directories if they do not exist
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables", recursive = TRUE, showWarnings = FALSE)

# 2. Data Import & Inspection --------------------------------------------------
credit <- read.table(file = "data/credit.txt", header = TRUE, sep = "")

str(credit)
summary(credit)

# Check for missing values
colSums(is.na(credit))

# Response distribution (The logistic model predicts P(kredit = 1), i.e., probability of proper repayment)
table(credit$kredit)
prop.table(table(credit$kredit))

# 3. Preparation ---------------------------------------------------------------
# Covariate scales:
# - kredit (Response): Dichotomous (0/1)
# - laufzeit, alter: Quantitative (months, years)
# - moral, laufkont, beruf: Ordinal categorical

credit_candidate <- credit[, c("kredit", "laufzeit", "moral", "laufkont", "alter", "beruf")]
credit_candidate$no_kredit <- 1 - credit_candidate$kredit

# Ordinal variables are modeled as unordered factors to avoid imposing a 
# strictly linear, equidistant effect across categories.
credit_candidate$moral <- as.factor(credit_candidate$moral)
credit_candidate$laufkont <- as.factor(credit_candidate$laufkont)
credit_candidate$beruf <- as.factor(credit_candidate$beruf)

# Document reference categories for correct odds ratio interpretation
levels(credit_candidate$moral)
levels(credit_candidate$laufkont)
levels(credit_candidate$beruf)

# 4. Train-Test Split & Data Aggregation ---------------------------------------
set.seed(234)

# 4.1 Split data to prevent data leakage during model selection
data_split <- initial_split(credit_candidate, prop = 0.7, strata = kredit)
data_train <- training(data_split)
data_test  <- testing(data_split)

# Export raw splits for out-of-sample evaluation in Script 3
saveRDS(data_train, "output/data_train.rds")
saveRDS(data_test, "output/data_test.rds")

# 4.2 Aggregate only the training data to perform goodness-of-fit tests and to analyze empirical logits
credit_agg <- aggregate(
  cbind(kredit, no_kredit) ~ laufzeit + moral + laufkont + alter + beruf, 
  data = data_train,
  FUN = sum
)

# Verify aggregation effect on training data
n_original_train <- nrow(data_train)
n_aggregated_train <- nrow(credit_agg)
n_original_train
n_aggregated_train

# Integrity check: Ensure no observations were lost during aggregation of the training set
stopifnot(sum(credit_agg$kredit) + sum(credit_agg$no_kredit) == nrow(data_train))
stopifnot(all(credit_agg$kredit >= 0), all(credit_agg$no_kredit >= 0))

# Export aggregated training dataset for subsequent diagnostic steps
saveRDS(credit_agg, "output/credit_agg.rds")

# 5. Exploratory Data Analysis (EDA) -------------------------------------------

# 5.1 Categorical Variables (Qualitative)
# Calculation of relative frequencies to identify sparse categories 
prop.table(table(credit$moral))
prop.table(table(credit$laufkont))
prop.table(table(credit$beruf))
#Interpretation: No empty categories. However, 'beruf' level 1 is quite rare (only 2.2%), 
# which will lead to high uncertainty (wide confidence intervals) for this specific group.

# Visualize the marginal effect of each categorical variable on the log-oddds -> To get idea of risk profile of each category
fit_moral <- glm(cbind(kredit, no_kredit) ~ moral, data = credit_agg, family = binomial(link = "logit"))
p_moral <- cat_plot(fit_moral, pred = moral, data = credit_agg, outcome.scale = "link", geom = "line", y.label = "Logit")
ggsave("output/figures/eda_catplot_moral.png", plot = p_moral, width = 6, height = 4, dpi = 300)

fit_laufkont <- glm(cbind(kredit, no_kredit) ~ laufkont, data = credit_agg, family = binomial(link = "logit"))
p_laufkont <- cat_plot(fit_laufkont, pred = laufkont, data = credit_agg, outcome.scale = "link", geom = "line", y.label = "Logit")
ggsave("output/figures/eda_catplot_laufkont.png", plot = p_laufkont, width = 6, height = 4, dpi = 300)

fit_beruf <- glm(cbind(kredit, no_kredit) ~ beruf, data = credit_agg, family = binomial(link = "logit"))
p_beruf <- cat_plot(fit_beruf, pred = beruf, data = credit_agg, outcome.scale = "link", geom = "line", y.label = "Logit")
ggsave("output/figures/eda_catplot_beruf.png", plot = p_beruf, width = 6, height = 4, dpi = 300)

# Interpretation of the categorical plots:
# - 'laufkont' shows a very clear, almost linear upward trend (better account status -> higher repayment odds).
# - 'moral' generally trends upwards but with slight non-monotonic jumps.
# - 'beruf' shows no clear directional trend, and the estimate for level 1 is highly uncertain 
#   due to the sparse data identified above.

# Clean up temporary GLM fits
rm(fit_moral, fit_laufkont, fit_beruf)


# 5.2 Continuous Variables (Quantitative) & Functional Form Assessment

# Descriptive statistics

continuous_summary <- tibble(
  Variable = c("laufzeit", "alter"),
  Min = c(min(credit$laufzeit), min(credit$alter)),
  Median = c(median(credit$laufzeit), median(credit$alter)),
  Mean = c(mean(credit$laufzeit), mean(credit$alter)),
  SD = c(sd(credit$laufzeit), sd(credit$alter)),
  Max = c(max(credit$laufzeit), max(credit$alter))
)

print(continuous_summary)


# Functional Form Assessment ---------------------------------------------------

# For continuous covariates (alter, laufzeit), we must verify if they have a 
# strictly linear relationship with the log-odds, or if they require transformation.
# We use Generalized Additive Models (GAMs) with smoothing splines s() to estimate
# the true, data-driven shape of the effect. 
# 
# Interpretation of the 'gam' package summary:
# We look at the "Anova for Nonparametric Effects" table.
# - The test statistic 'P(Chi)' tells us if the non-linear part of the spline is 
#   statistically significant. 
# - If significant (p < 0.05), a strictly linear term is insufficient. The variable 
#   should be transformed (e.g., using polynomials) or discretized before entering the GLM.


png("output/figures/gam_continuous_predictors.png", width = 2600, height = 1200, res = 300)
# 2. Set global graphical parameters for a modern look
# mfrow = c(1, 2) creates a 1x2 grid (side-by-side)
# bty = "l" removes the top and right box borders for a cleaner look
# las = 1 makes all axis labels horizontal and easier to read
par(mfrow = c(1, 2), mar = c(5, 5, 4, 2) + 0.1, las = 1, bty = "l", cex.main = 1.2, cex.lab = 1.1)

# Assess 'alter' (Age)
gam_alter <- gam(
  cbind(kredit, no_kredit) ~ s(alter),
  data = credit_agg,
  family = binomial(link = "logit")
)

plot(gam_alter, 
     se = TRUE, 
     main = "Functional Form: Alter (Age)", 
     xlab = "Age in Years", 
     ylab = "Partial Effect on Log-Odds",
     col = "#2c3e50",      # Modern dark slate blue
     lwd = 2.5            # Thicker line
     )

summary(gam_alter)
# Interpretation for 'alter':
# The plot shows a somewhat non-linear, cubic trend (risk increases until mid-30s, 
# then plateaus). However, the nonparametric test yields P(Chi) = 0.1095. Since this 
# is > 0.05, the non-linear deviation is not strictly significant. If we were to force 
# 'alter' into the final model, testing a discretized version (e.g., age groups) or a 
# polynomial might be required, but a linear baseline test is acceptable.

# Assess 'laufzeit' (Duration)
gam_laufzeit <- gam(
  cbind(kredit, no_kredit) ~ s(laufzeit),
  data = credit_agg,
  family = binomial(link = "logit")
)

plot(gam_laufzeit, 
     se = TRUE, 
     main = "Functional Form: Laufzeit (Duration)", 
     xlab = "Duration in Months", 
     ylab = "Partial Effect on Log-Odds",
     col = "#2c3e50", 
     lwd = 2.5)

dev.off()

summary(gam_laufzeit)
# Interpretation for 'laufzeit':
# The plot exhibits a very clear, strictly linear downward trend. The nonparametric 
# test confirms this with P(Chi) = 0.3467 (no significant non-linear effect). 
# This confirms that 'laufzeit' can be safely included as a standard linear main 
# effect in the logistic regression model.

# Clean up temporary GAM fits
rm(gam_alter, gam_laufzeit)

# 5.3 Exploratory Data Analysis for Interaction Effects-------------------------
# Objective: Check empirical logit plots for non-parallel trends to assess 
# the need for interaction terms. Instead of all 10 combinations, we analyze 
# the strongest main effects ('moral' and 'laufkont') interacting with 'laufzeit'.

# Calculate empirical logits with continuity correction to avoid log(0)
credit_agg$emp_logit <- log((credit_agg$kredit + 0.5) / (credit_agg$no_kredit + 0.5))

# Plot 1: Laufzeit vs. Moral
p_inter_laufzeit_moral <- ggplot(credit_agg, aes(x = laufzeit, y = emp_logit, color = moral)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "loess", se = FALSE, span = 0.9, linewidth = 1.2) +
  facet_wrap(~ moral) +
  theme_light() +
  labs(
    title = "Interaction Check: Laufzeit vs. Empirical Logits by Moral",
    x = "Duration in Months (laufzeit)",
    y = "Empirical Logit"
  ) +
  theme(legend.position = "none")

ggsave("output/figures/eda_interaction_laufzeit_moral.png", plot = p_inter_laufzeit_moral, width = 10, height = 6, dpi = 300)

# Interpretation (Moral):
# - Categories 2 and 4 cover the vast majority of data (82.3%) and show roughly parallel downward trends.
# - Deviations in categories 0 (4.0%), 1 (4.9%), and 3 (8.8%) stem from data sparsity and high variance.
# - Conclusion: Core population exhibits parallel trends; an additive main-effects framework is justified.


# Plot 2: Laufzeit vs. Laufkont
p_inter_laufzeit_laufkont <- ggplot(credit_agg, aes(x = laufzeit, y = emp_logit, color = laufkont)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "loess", se = FALSE, span = 0.9, linewidth = 1.2) +
  facet_wrap(~ laufkont) +
  theme_light() +
  labs(
    title = "Interaction Check: Laufzeit vs. Empirical Logits by Laufkont",
    x = "Duration in Months (laufzeit)",
    y = "Empirical Logit"
  ) +
  theme(legend.position = "none")

ggsave("output/figures/eda_interaction_laufzeit_laufkont.png", plot = p_inter_laufzeit_laufkont, width = 10, height = 6, dpi = 300)

# Interpretation (Laufkont):
# - Categories 1 (27.4%), 2 (26.9%), and 4 (39.4%) are well-represented.
# - Minor non-parallelisms (e.g., category 4 plateau between 20-30 months) trace back to boundary smoothing effects and sparse data.
# - Conclusion: No systemic interaction pattern across main groups; an additive model prevents overfitting.

# Clean up
credit_agg$emp_logit <- NULL


# 6. Model Specification -------------------------------------------------------

# 6.1 Null model (intercept only)
model_null <- glm(
  cbind(kredit, no_kredit) ~ 1, 
  data = credit_agg, 
  family = binomial(link = "logit")
)

# 6.2 # Baseline model
# The forward-selection procedure follows the predetermined assignment 
# specification: 'moral' is retained as the initial baseline covariate 
# and additional candidate covariates are evaluated based on AIC.
model_baseline <- glm(
  cbind(kredit, no_kredit) ~ moral, 
  data = credit_agg, 
  family = binomial(link = "logit")
)

# 6.3 Full model
model_full <- glm(
  cbind(kredit, no_kredit) ~ laufzeit + laufkont + alter + beruf + moral, 
  data = credit_agg, 
  family = binomial(link = "logit")
)

# 6.4 Forward selection (AIC)
model_stepwise_aic <- step(
  object = model_baseline, 
  direction = "forward", 
  scope = formula(model_full), 
  trace = 0
)

# 6.5 Forward selection (BIC) through setting k = log(n)
n_obs <- nrow(credit_agg)

model_stepwise_bic <- step(
  object = model_baseline, 
  direction = "forward", 
  scope = formula(model_full), 
  k = log(n_obs),
  trace = 0
)

# 6.6 Compare model complexity using both AIC and BIC
comparison_table <- data.frame(
  Model = c("Null", "Baseline", "AIC-selected", "BIC-selected", "Full"),
  AIC = c(AIC(model_null), AIC(model_baseline), AIC(model_stepwise_aic), AIC(model_stepwise_bic), AIC(model_full)),
  BIC = c(BIC(model_null), BIC(model_baseline), BIC(model_stepwise_aic), BIC(model_stepwise_bic), BIC(model_full))
)
print(comparison_table)
#Interpretation: model_stepwise_aic and model_stepwise_bic have same AIC and BIC and are thus equal

write.csv(
  comparison_table,
  "output/tables/model_comparison_aic_bic.csv",
  row.names = FALSE
)

# 7. Final model ---------------------------------------------------------------
# Extract the final mode. We proceed with BIC-selected model, since it is in general spareser than the AIC-selected model and thus easier to interpret.
model_main <- model_stepwise_bic #Covariates: Moral, Laufkont and Laufzeit - Alter and Beruf are excluded
saveRDS(
  model_main,
  "output/model_main.rds"
)

# Compare nested models using likelihood-ratio tests
anova(model_null, model_baseline, test = "Chisq") # Does moral add information? Answer: yes, with p-value of 1.254e-08
anova(model_baseline, model_main, test = "Chisq") # Do added variables improve fit? Answer: yes, with p-value of 3.023e-16

model_add_alter <- glm(
  cbind(kredit, no_kredit) ~ moral + laufkont + laufzeit + alter, 
  data = credit_agg, 
  family = binomial(link = "logit")
)

model_add_beruf <- glm(
  cbind(kredit, no_kredit) ~ moral + laufkont + laufzeit + beruf, 
  data = credit_agg, 
  family = binomial(link = "logit")
)

anova(model_main, model_add_alter, test = "Chisq") # Does 'alter' individually improve fit? Answer: no, with p-value of 0.2484
anova(model_main, model_add_beruf, test = "Chisq") # Does 'beruf' individually improve fit? Answer: no, with p-value of 0.7575

summary(model_main)

# 7.3 Model Parameters (Coefficients, Odds Ratios, Confidence Intervals)
or_table <- data.frame(
  term = names(coef(model_main)),
  estimate = coef(model_main),
  odds_ratio = exp(coef(model_main)),
  conf_low = exp(confint(model_main)[, 1]),
  conf_high = exp(confint(model_main)[, 2])
)
print(or_table)
write.csv(
  or_table,
  "output/tables/odds_ratios.csv",
  row.names = FALSE
)

# Odds ratios > 1 indicate higher odds of repayment (kredit = 1),
# whereas odds ratios < 1 indicate lower odds of repayment (higher risk).
# For factor variables, odds ratios are interpreted relative
# to the respective reference category documented above.