# ==============================================================================
# Script: 01_data_prep_and_selection.R
# Purpose: Data inspection, functional form assessment (EDA, Partial Residuals),  
#          and model selection for credit default prediction.
# ==============================================================================

# 1. Setup ---------------------------------------------------------------------
library(tidyverse)
library(interactions)
library(gam)

# 2. Data Import & Inspection --------------------------------------------------
credit <- read.table(file = "data/credit.txt", header = TRUE, sep = "")

str(credit)
summary(credit)

# Check for missing values
colSums(is.na(credit))

# Response distribution (The logistic model predicts P(kredit = 1))
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

# 4. Data Aggregation ----------------------------------------------------------
# Aggregate data to analyze empirical logits and perform goodness-of-fit tests
credit_agg <- aggregate(
  cbind(kredit, no_kredit) ~ laufzeit + moral + laufkont + alter + beruf, 
  data = credit_candidate,
  FUN = sum
)

# Verify aggregation effect (1000 individual obs. -> 229 unique combinations)
n_original <- nrow(credit)
n_aggregated <- nrow(credit_agg)
n_original
n_aggregated

# Integrity check: Ensure no observations were lost during aggregation
stopifnot(sum(credit_agg$kredit) + sum(credit_agg$no_kredit) == nrow(credit))
stopifnot(all(credit_agg$kredit >= 0), all(credit_agg$no_kredit >= 0))

# 5. Exploratory Data Analysis (EDA) -------------------------------------------

# 5.1 Categorical variables
prop.table(table(credit$moral))
prop.table(table(credit$laufkont))
prop.table(table(credit$beruf))

fit_moral <- glm(cbind(kredit, no_kredit) ~ moral, data = credit_agg, family = binomial(link = "logit"))
cat_plot(fit_moral, pred = moral, data = credit_agg, outcome.scale = "link", geom = "line", y.label = "Logit")

fit_laufkont <- glm(cbind(kredit, no_kredit) ~ laufkont, data = credit_agg, family = binomial(link = "logit"))
cat_plot(fit_laufkont, pred = laufkont, data = credit_agg, outcome.scale = "link", geom = "line", y.label = "Logit")

fit_beruf <- glm(cbind(kredit, no_kredit) ~ beruf, data = credit_agg, family = binomial(link = "logit"))
cat_plot(fit_beruf, pred = beruf, data = credit_agg, outcome.scale = "link", geom = "line", y.label = "Logit")

# 5.2 Quantitative variables & 5.3 Functional form assessment
# Functional form assessment via GAMs: An estimated degree of freedom (edf) 
# close to 1 supports a linear specification, whereas an edf significantly > 1 
# provides an indication of potential non-linearity (though not a formal proof).
gam_alter <- gam(cbind(kredit, no_kredit) ~ s(alter), data = credit_agg, family = binomial(link = "logit"))
plot(gam_alter, se = TRUE, main = "Smooth term for alter")
summary(gam_alter)

gam_laufzeit <- gam(cbind(kredit, no_kredit) ~ s(laufzeit), data = credit_agg, family = binomial(link = "logit"))
plot(gam_laufzeit, se = TRUE, main = "Smooth term for laufzeit")
summary(gam_laufzeit)

# Clean environment before formal selection
rm(fit_moral, fit_laufkont, fit_beruf, gam_alter, gam_laufzeit)

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

# 6.4 Forward selection
model_stepwise <- step(
  object = model_baseline, 
  direction = "forward", 
  scope = formula(model_full), 
  trace = 0
)

# Extract the final model selected by the step function
model_main <- model_stepwise

# 7. Final model ---------------------------------------------------------------

# Compare nested models using likelihood-ratio tests
anova(model_null, model_baseline, test = "Chisq") # Does moral add information?
anova(model_baseline, model_main, test = "Chisq") # Do added variables improve fit?
anova(model_main, model_full, test = "Chisq")     # Does the full model improve fit over the selected one?

# Compare model complexity using AIC
AIC(model_null, model_baseline, model_main, model_full)
# Forward selection drops 'beruf', as its inclusion does not improve model fit 
# enough to justify the added complexity (AIC penalty).

# 7.1 Summary
summary(model_main)

# 7.2 Coefficients, 7.3 Odds ratios, and 7.4 Confidence intervals
or_table <- data.frame(
  term = names(coef(model_main)),
  estimate = coef(model_main),
  odds_ratio = exp(coef(model_main)),
  conf_low = exp(confint(model_main)[, 1]),
  conf_high = exp(confint(model_main)[, 2])
)
print(or_table)
# Odds ratios > 1 indicate higher odds of kredit = 1,
# whereas odds ratios < 1 indicate lower odds of kredit = 1.
# For factor variables, odds ratios are interpreted relative
# to the respective reference category documented above.

# Functional Form Check (Partial Residuals for 'laufzeit')
partial_residuals <- residuals(model_main, type = "partial")

partial_laufzeit <- data.frame(
  laufzeit = credit_agg$laufzeit,
  partial_residual = partial_residuals[, "laufzeit"]
)

ggplot(partial_laufzeit, aes(x = laufzeit, y = partial_residual)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = TRUE, color = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_light() +
  labs(
    title = "Partial Residual Plot: laufzeit",
    x = "Duration in Months (laufzeit)",
    y = "Partial Residual"
  )
# The loess smoothing line approximately follows a straight path, supporting 
# a linear specification for the covariate 'laufzeit' in the multivariate model.

# 7.5 Goodness-of-Fit Assessment
# Approximate Chi-squared goodness-of-fit test based on residual deviance
dev <- deviance(model_main)
df_res <- df.residual(model_main)

dev
df_res

# Calculate critical Chi-squared value and p-value
crit_val <- qchisq(0.95, df = df_res)
gof_p <- pchisq(dev, df = df_res, lower.tail = FALSE)

cat("Residual Deviance:", dev, "\nDegrees of Freedom:", df_res, 
    "\nCritical Chi-sq Value:", crit_val, "\nGoodness-of-Fit p-value:", gof_p, "\n")

# A large p-value provides no evidence of lack of fit based on
# the residual deviance, but it does not prove that the model is
# correctly specified.