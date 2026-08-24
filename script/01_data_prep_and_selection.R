# ==============================================================================
# Script: 01_data_prep_and_selection.R
# Purpose: Data inspection, functional form assessment (EDA, Partial Residuals),  
#          and model selection for credit default prediction.
# ==============================================================================

# 1. Setup ---------------------------------------------------------------------
library(tidyverse)
library(interactions)
library(gam)

# Create output directories if they do not exist
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables", recursive = TRUE, showWarnings = FALSE)

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

# Export aggregated dataset for subsequent diagnostic steps
saveRDS(credit_agg, "output/credit_agg.rds")

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
gam_alter <- gam(
  cbind(kredit, no_kredit) ~ s(alter),
  data = credit_agg,
  family = binomial(link = "logit")
)

png("output/figures/gam_alter.png", width = 1000, height = 800, res = 300)
plot(gam_alter, se = TRUE, main = "Smooth term for alter")
dev.off()

summary(gam_alter)

gam_laufzeit <- gam(
  cbind(kredit, no_kredit) ~ s(laufzeit),
  data = credit_agg,
  family = binomial(link = "logit")
)

png("output/figures/gam_laufzeit.png", width = 1000, height = 800, res = 150)
plot(gam_laufzeit, se = TRUE, main = "Smooth term for laufzeit")
dev.off()

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
saveRDS(
  model_main,
  "output/model_main.rds"
)

# 7. Final model ---------------------------------------------------------------

# Compare nested models using likelihood-ratio tests
anova(model_null, model_baseline, test = "Chisq") # Does moral add information?
anova(model_baseline, model_main, test = "Chisq") # Do added variables improve fit?
anova(model_main, model_full, test = "Chisq")     # Does the full model improve fit over the selected one?

# Compare model complexity using AIC
aic_table <- AIC(
  model_null,
  model_baseline,
  model_main,
  model_full
)

print(aic_table)

write.csv(
  aic_table,
  "output/tables/model_comparison_aic.csv",
  row.names = FALSE
)
# Forward selection drops 'beruf', as its inclusion does not improve model fit 
# enough to justify the added complexity (AIC penalty).

# 7.1 Summary
summary(model_main)

# 7.2 Model Parameters (Coefficients, Odds Ratios, Confidence Intervals)
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
# Odds ratios > 1 indicate higher odds of kredit = 1,
# whereas odds ratios < 1 indicate lower odds of kredit = 1.
# For factor variables, odds ratios are interpreted relative
# to the respective reference category documented above.