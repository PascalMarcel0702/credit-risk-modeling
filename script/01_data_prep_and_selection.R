# ==============================================================================
# Script: 01_data_prep_and_selection.R
# Purpose: Data inspection, functional form assessment (EDA, Partial Residuals),  
#          and model selection for credit repayment prediction.
# ==============================================================================

# 1. Setup ---------------------------------------------------------------------
library(tidyverse)
library(interactions)
library(mgcv)
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

# Response distribution (The logistic model predicts P(kredit = 1), i.e., probability of proper repayment, which is given by 0.7)
table(credit$kredit)
prop.table(table(credit$kredit))
# Interpretation: 70% repay their loan.

#Feature Engineering-----------------------------------------
# The documentation (https://data.ub.uni-muenchen.de/23/1/DETAILS.html) defines discretizations for 
# 'alter' and 'laufzeit', but they are not pre-calculated in the raw data. 

credit <- credit %>%
  mutate(
    # Discretize laufzeit according to expert rules
    dlaufzeit = cut(laufzeit, 
                    breaks = c(-Inf, 6, 12, 18, 24, 30, 36, 42, 48, 54, Inf),
                    labels = c("<=6", "7-12", "13-18", "19-24", "25-30", 
                               "31-36", "37-42", "43-48", "49-54", ">54")),
    # Discretize alter according to expert rules
    dalter = cut(alter, 
                 breaks = c(-Inf, 25, 39, 59, 64, Inf),
                 labels = c("<=25", "26-39", "40-59", "60-64", ">=65"))
  )

# 3. Preparation ---------------------------------------------------------------
# Covariate scales:
# - kredit (Response): Dichotomous (0/1)
# - laufzeit, alter: Quantitative (months, years)
# - dlaufzeit, dalter: Ordinal categorical (expert-discretized bins, see documentation)
# - moral, laufkont, beruf: Ordinal categorical

credit_candidate <- credit[, c("kredit", "laufzeit", "dlaufzeit", "moral", "laufkont", "alter", "dalter", "beruf", "hoehe")]

credit_candidate <- credit_candidate %>%
  mutate(no_kredit = 1 - kredit)

# Categorical variables are modeled as unordered factors to avoid imposing a 
# strictly linear, equidistant effect across categories.
credit_candidate$moral <- as.factor(credit_candidate$moral)
credit_candidate$laufkont <- as.factor(credit_candidate$laufkont)
credit_candidate$beruf <- as.factor(credit_candidate$beruf)
credit_candidate$dalter <- as.factor(credit_candidate$dalter)
credit_candidate$dlaufzeit <- as.factor(credit_candidate$dlaufzeit)

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
saveRDS(data_train, "output/data_train.rds") # Adjustment needed: Save data at end
saveRDS(data_test, "output/data_test.rds")  # Adjustment needed: Save data at end

# 4.2 Aggregate only the training data to perform goodness-of-fit tests and to analyze empirical logits
# Note: We include both the continuous and their discrete counterparts here so they are available for both GAMs and EDA plots.
credit_agg <- aggregate(
  cbind(kredit, no_kredit) ~ laufzeit + dlaufzeit + moral + laufkont + alter + dalter + beruf, 
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
saveRDS(credit_agg, "output/credit_agg.rds") # Adjustment needed: Save data at end

# Pre-Check: Design Matrix Rank for Algorithmic Stability
# Create the full design matrix for all candidate variables
X_candidate <- model.matrix(~ laufzeit + dlaufzeit + moral + laufkont + alter + dalter + beruf, data = credit_agg)
# Calculating rank using QR decomposition
qr(X_candidate)$rank
# Calculating number of parameters
ncol(X_candidate)

#Interpretation: Rank (26) is equal to number of parameters, i.e., full rank.
# 5. Exploratory Data Analysis (EDA) -------------------------------------------

# 5.1 Categorical Variables (Qualitative)
# Calculation of relative frequencies to identify sparse categories 
prop.table(table(data_train$moral))
prop.table(table(data_train$laufkont))
prop.table(table(data_train$beruf))
prop.table(table(data_train$dalter))
prop.table(table(data_train$dlaufzeit))
# Interpretation:
# No empty categories exist. However, the data exhibits severe sparsity in specific groups:
# - 'beruf' level 1 is extremely rare (1.9%).
# - The expert-discretized continuous variables show heavy tail sparsity:
#   For 'dalter', age groups > 60 years account for only ~5.4% combined.
#   For 'dlaufzeit', durations > 36 months are highly fragmented, with bins like '49-54' 
#   containing less than 0.3% of the training observations.
# This sparsity will inevitably lead to high estimation variance and wide confidence 
# intervals for these specific categories during the exploratory data analysis (EDA).

# 5.1 Categorical Variables (Qualitative)
# Objective: Visualize the marginal effect of each categorical variable model-free 
# using empirical logits to assess risk profiles and identify high-variance sparse groups.

# Helper function to calculate and plot empirical logits for categorical variables
plot_emp_logit <- function(df, grouping_var, var_label) {
  df %>%
    group_by({{ grouping_var }}) %>%
    summarise(
      kredit = sum(kredit),
      no_kredit = sum(no_kredit),
      .groups = "drop"
    ) %>%
    mutate(
      emp_logit = log((kredit + 0.5) / (no_kredit + 0.5)),
      se_emp_logit = sqrt(1/(kredit + 0.5) + 1/(no_kredit + 0.5)),
      ci_lower = emp_logit - qnorm(0.975) * se_emp_logit,
      ci_upper = emp_logit + qnorm(0.975) * se_emp_logit
    ) %>%
    ggplot(aes(x = {{ grouping_var }}, y = emp_logit)) +
    geom_point(color = "#2c3e50", size = 3) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2, color = "#2c3e50") +
    theme_light() +
    labs(x = var_label, y = "Empirical Logit")+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Generate and save plots
p_moral <- plot_emp_logit(credit_agg, moral, "Payment History (moral)")
ggsave("output/figures/eda_emp_logit_moral.png", plot = p_moral, width = 6, height = 4, dpi = 300)

p_laufkont <- plot_emp_logit(credit_agg, laufkont, "Account Status (laufkont)")
ggsave("output/figures/eda_emp_logit_laufkont.png", plot = p_laufkont, width = 6, height = 4, dpi = 300)

p_beruf <- plot_emp_logit(credit_agg, beruf, "Occupation (beruf)")
ggsave("output/figures/eda_emp_logit_beruf.png", plot = p_beruf, width = 6, height = 4, dpi = 300)

# Evaluate expert-discretized continuous variables
p_dalter <- plot_emp_logit(credit_agg, dalter, "Discretized Age (dalter)")
ggsave("output/figures/eda_emp_logit_dalter.png", plot = p_dalter, width = 6, height = 4, dpi = 300)

p_dlaufzeit <- plot_emp_logit(credit_agg, dlaufzeit, "Discretized Duration (dlaufzeit)")
ggsave("output/figures/eda_emp_logit_dlaufzeit.png", plot = p_dlaufzeit, width = 6, height = 4, dpi = 300)


# Tabular Summary of Empirical Logits
# Goal: Create Table showing empirical logits, sample sizes (n), and 95% confidence intervals (L, U) for all discrete covariates.

# Helper function to generate and pivot the table for a single variable
generate_emp_logit_table <- function(df, grouping_var, var_name) {
  df %>%
    group_by({{ grouping_var }}) %>%
    summarise(
      n = sum(kredit) + sum(no_kredit),
      kredit = sum(kredit),
      no_kredit = sum(no_kredit),
      .groups = "drop"
    ) %>%
    mutate(
      `Empir. logits` = round(log((kredit + 0.5) / (no_kredit + 0.5)), 2),
      se_emp_logit = sqrt(1/(kredit + 0.5) + 1/(no_kredit + 0.5)),
      `L` = round(`Empir. logits` - qnorm(0.975) * se_emp_logit, 2),
      `U` = round(`Empir. logits` + qnorm(0.975) * se_emp_logit, 2)
    ) %>%
    # Select only the required columns
    select({{ grouping_var }}, `Empir. logits`, n, L, U) %>%
    # Format 'n' as numeric for consistency, others are rounded numerics
    mutate(n = as.numeric(n)) %>%
    # Pivot metrics to rows and categories to columns
    pivot_longer(
      cols = c(`Empir. logits`, n, L, U), 
      names_to = "Metric", 
      values_to = "Value"
    ) %>%
    pivot_wider(
      names_from = {{ grouping_var }}, 
      values_from = Value
    ) %>%
    # Add a column identifying the variable name
    mutate(Variable = var_name, .before = 1)
}

# Generate tables for all categorical and discretized variables
tab_moral     <- generate_emp_logit_table(credit_agg, moral, "Payment History (moral)")
tab_laufkont  <- generate_emp_logit_table(credit_agg, laufkont, "Account Status (laufkont)")
tab_beruf     <- generate_emp_logit_table(credit_agg, beruf, "Occupation (beruf)")
tab_dalter    <- generate_emp_logit_table(credit_agg, dalter, "Age Group (dalter)")
tab_dlaufzeit <- generate_emp_logit_table(credit_agg, dlaufzeit, "Duration Group (dlaufzeit)")

# Combine all individual tables into one master summary table
# bind_rows automatically aligns column names and fills non-matching categories with NA
eda_emp_logit_master <- bind_rows(
  tab_moral, 
  tab_laufkont, 
  tab_beruf, 
  tab_dalter, 
  tab_dlaufzeit
)

# Export the master table as a CSV for inclusion in the README or appendix
write.csv(
  eda_emp_logit_master, 
  "output/tables/eda_empirical_logits_summary.csv", 
  row.names = FALSE, 
  na = "" # Leaves NA cells blank for a cleaner look
)

print(eda_emp_logit_master)


# Interpretation of Empirical Logits through Logit Delta and CI separation

# 1. Primary Risk Drivers (High Delta, Non-Overlapping CIs):
# - 'dlaufzeit': Marginal effect size Delta = 2.17 (Extremes: <=6: 1.82 vs. 43-48: -0.35). Extremes exhibit no CI overlap. Bins >48 months and 37-42 exhibit CI widths > 2.00 due to data sparsity (n <= 10). Regarding this, the plot overall shows a monotonic downward trend.

# - 'moral': Marginal effect size Delta = 2.12 (Extremes: Level 4: 1.49 vs. Level 0: -0.63). Extremes exhibit no CI overlap.Bin 1 exhibits a CI width of 1.50 due to data sparsity (n = 27). The plot overall shows a monotonic upward trend.

#- 'laufkont': Marginal effect size Delta = 1.73 (Extremes: Level 4: 1.86 vs. Level 1: 0.13). Extremes exhibit no CI overlap. Bin 3 exhibits a CI width > 1.30 due to moderate data sparsity (n = 43).  The plot overall shows a monotonic upward trend.

# 2. Secondary Risk Driver (Moderate Delta, Structural Non-Linearity):
# - 'dalter': Concave, approximately quadratic structure (leading coefficient < 0). Marginal effect size Delta = 0.88 (Extremes: 60-64: 1.21 vs. <=25: 0.33). Extreme categories exhibit substantial CI overlap. Bins  60-64 (3,3%) and  >=65 (2,1%) exhibit CI widths > 1.75 due to data sparsity.

# 3. Weak Risk Driver (Low Delta, Severe CI Overlap):
# - 'beruf': Marginal effect size Delta = 0.49 (Extremes: Level 3: 0.93 vs. Level 1: 0.44). CI overlaps at all levels. Due to data sparsity, level 1 (1,8%) exhibit CI width > 2.00 and contains the CIs of the other levels. Therefore, no clear data is visible.

# 5.2 Continuous Variables (Quantitative) & Functional Form Assessment

# Descriptive statistics

continuous_summary <- tibble(
  Variable = c("laufzeit", "alter"),
  Min = c(min(data_train$laufzeit), min(data_train$alter)),
  Median = c(median(data_train$laufzeit), median(data_train$alter)),
  Mean = c(mean(data_train$laufzeit), mean(data_train$alter)),
  SD = c(sd(data_train$laufzeit), sd(data_train$alter)),
  Max = c(max(data_train$laufzeit), max(data_train$alter))
)

print(continuous_summary)


# Functional Form Assessment ---------------------------------------------------

# For continuous covariates (alter, laufzeit), we must verify if they have a 
# strictly linear relationship with the log-odds, or if they require transformation.
# We use Generalized Additive Models (GAMs) with smoothing splines s() to estimate
# the true, data-driven shape of the effect. 
# 
# Interpretation of the 'mgcv::gam' summary:
# - Check the Approximate significance of smooth terms.
# - A significant p-value (p < 0.05) combined with an Effective Degrees of Freedom 
#   (edf) > 1 suggests (a statistically significant) non-linear relationship.
# - If not significant, there is no evidence to reject the linear main effect.


png("output/figures/gam_continuous_predictors.png", width = 2450, height = 1200, res = 300)
# 2. Set global graphical parameters for a modern look
# mfrow = c(1, 2) creates a 1x2 grid (side-by-side)
# bty = "l" removes the top and right box borders for a cleaner look
# las = 1 makes all axis labels horizontal and easier to read
par(mfrow = c(1, 2), mar = c(5, 5, 4, 2) + 0.1, las = 1, bty = "l", cex.main = 1.2, cex.lab = 1.1)

# Assess 'alter' (Age)
gam_alter <- mgcv::gam(
  cbind(kredit, no_kredit) ~ s(alter),
  data = credit_agg,
  family = binomial(link = "logit")
)

plot(gam_alter, 
     se = TRUE, 
     xlab = "Age in Years", 
     ylab = "Partial Effect on Log-Odds",
     col = "#2c3e50",      # Modern dark slate blue
     lwd = 2.5             # Thicker line
)

summary(gam_alter)
# Interpretation for 'alter':
# The smooth term for 'alter' yields an estimated degree of freedom (edf) of 1.903 > 1 -> Indication of quadratic relationship by approximate significance test (p-value 0.0246 < 0.05)
# Conclusion: The GAM provides evidence for a nonlinear association between age and the log-odds.

# Assess 'laufzeit' (Duration)
gam_laufzeit <- mgcv::gam(
  cbind(kredit, no_kredit) ~ s(laufzeit),
  data = credit_agg,
  family = binomial(link = "logit")
)

plot(gam_laufzeit, 
     se = TRUE, 
     xlab = "Duration in Months", 
     ylab = "Partial Effect on Log-Odds",
     col = "#2c3e50", 
     lwd = 2.5)

dev.off()

summary(gam_laufzeit)
# Interpretation for 'laufzeit':
# The smooth term is shrinked to a straight line, with edf = 1.
# The plot shows a linear downward trend.
# Moreover, the p-value of 5.83e-07 < 0.05 indicates no non-linear transformation for 'laufzeit'
# Conclusion: The smooth term for 'duration'laufzeit' is highly significant, indicating a strong association with the response. However, its effective degrees of freedom is exactly 1, indicating that the estimated smooth is effectively linear rather than nonlinear

# Clean up temporary GAM fits
rm(gam_alter, gam_laufzeit)

# 5.3 Exploratory Data Analysis for Interaction Effects-------------------------
# Objective: Check empirical logit plots for non-parallel trends to visually assess the need for interaction terms.
# Instead of all 10 combinations, we analyze the strongest main effects driven by hypotheses


# 1. Helper function for Interaction Profile Plots
plot_interaction_profile <- function(df, x_var, group_var, x_label, legend_label) {
  
  # Step A: Aggregate raw counts and calculate metrics strictly per cell
  agg_data <- df %>%
    group_by({{ x_var }}, {{ group_var }}) %>%
    summarise(
      kredit = sum(kredit),
      no_kredit = sum(no_kredit),
      .groups = "drop"
    ) %>%
    mutate(
      emp_logit = log((kredit + 0.5) / (no_kredit + 0.5)),
      se_emp_logit = sqrt(1/(kredit + 0.5) + 1/(no_kredit + 0.5)),
      ci_lower = emp_logit - qnorm(0.975) * se_emp_logit,
      ci_upper = emp_logit + qnorm(0.975) * se_emp_logit
    )
  
  # Step B: Generate the overlaid profile plot
  pd <- position_dodge(width = 0.3) # Dodge width separates overlapping error bars
  
  ggplot(agg_data, aes(x = {{ x_var }}, y = emp_logit, color = {{ group_var }}, group = {{ group_var }})) +
    geom_point(size = 3, position = pd) +
    geom_line(linewidth = 1, position = pd) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2, position = pd) +
    theme_light() +
    labs(
      x = x_label,
      y = "Empirical Logit",
      color = legend_label
    ) +
    theme(
      legend.position = "right",
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1) # Rotated for cleanly displaying discrete bins
    )
}


# Plot 1: Laufzeit vs. Moral (The longer a loan runs (maturity), the higher the underlying risk of unforeseen life events (unemployment, illness))
# Q: Does a flawless credit history (moral = 4) provide better protection against this long-term risk than a critical history (moral = 0)?

p_inter_dlauf_moral <- plot_interaction_profile(
  credit_agg, dlaufzeit, moral, 
  "Discretized Duration (dlaufzeit)", "Payment Habits\n(moral)"
)
ggsave("output/figures/eda_interaction_laufzeit_moral.png", plot = p_inter_dlauf_moral, width = 8, height = 5, dpi = 300)


# Plot 2: Laufzeit vs. Laufkont (Account status proxies liquidity; duration defines time-at-risk)
# Q: Does high liquidity offset the risk of long-term loans, or do poor accounts amplify default risk over long durations (e.g., due to debt consolidation)?

p_inter_dlauf_laufkont <- plot_interaction_profile(
  credit_agg, dlaufzeit, laufkont, 
  "Discretized Duration (dlaufzeit)", "Account Status\n(laufkont)"
)
ggsave("output/figures/eda_interaction_laufzeit_laufkont.png", plot = p_inter_dlauf_laufkont, width = 8, height = 5, dpi = 300)


# Plot 3: Moral vs. Laufkont (Current account status represents immediate liquidity, whereas payment history reflects long-term behavioral reliability)
# Q: Can high current liquidity compensate for the default risk of a critical payment history, or does a flawless payer remain low-risk even when currently overdrawn?

p_inter_moral_laufkont <- plot_interaction_profile(
  credit_agg, moral, laufkont, 
  "Payment Habits (moral)", "Account Status\n(laufkont)"
)
ggsave("output/figures/eda_interaction_moral_laufkont.png", plot = p_inter_moral_laufkont, width = 8, height = 5, dpi = 300)


# Plot 4: Alter vs Laufzeit ( Risk exposure tends to vary over time depending on one's stage of life (young and volatile vs. middle-aged and financially stable))
# Q: Does the temporal risk of long-term borrowing scale additively across age groups or do extremely young/old borrowers face disproportionate default probabilities over long durations?

p_inter_alter_laufzeit <- plot_interaction_profile(
  credit_agg, dalter, dlaufzeit, 
  "Discretized Age (dalter)", "Discretized Duration (dlaufzeit)"
)
ggsave("output/figures/eda_interaction_alter_laufzeit.png", plot = p_inter_alter_laufzeit, width = 8, height = 5, dpi = 300)


# Table of Empirical Logits for Interactions
# Goal: Quantify the cross-tabulated empirical logits, cell sizes (n), and 95% CIs (L, U) for certain combinations the primary risk drivers to formally screen for interactions.

generate_interaction_table <- function(df, var1, var2, name_var1, name_var2) {
  df %>%
    group_by({{ var1 }}, {{ var2 }}) %>%
    summarise(
      n = sum(kredit) + sum(no_kredit),
      kredit = sum(kredit),
      no_kredit = sum(no_kredit),
      .groups = "drop"
    ) %>%
    # Calculate metrics with continuity correction
    mutate(
      `Empir. logits` = round(log((kredit + 0.5) / (no_kredit + 0.5)), 2),
      se = sqrt(1/(kredit + 0.5) + 1/(no_kredit + 0.5)),
      `L` = round(`Empir. logits` - qnorm(0.975) * se, 2),
      `U` = round(`Empir. logits` + qnorm(0.975) * se, 2)
    ) %>%
    select({{ var1 }}, {{ var2 }}, `Empir. logits`, n, L, U) %>%
    mutate(n = as.numeric(n)) %>%
    # Pivot metrics to rows
    pivot_longer(
      cols = c(`Empir. logits`, n, L, U), 
      names_to = "Metric", 
      values_to = "Value"
    ) %>%
    # Pivot var2 to columns
    pivot_wider(
      names_from = {{ var2 }}, 
      values_from = Value
    ) %>%
    # Enforce strict row order matching Table 4.8
    mutate(Metric = factor(Metric, levels = c("Empir. logits", "n", "L", "U"))) %>%
    arrange(Metric, {{ var1 }}) %>%
    rename(Category_Var1 = {{ var1 }}) %>%
    mutate(Interaction = paste0(name_var1, " vs ", name_var2), .before = 1)
}

# Generate interaction tables for the top 3 predictors
tab_inter_moral_dlauf <- generate_interaction_table(credit_agg, moral, dlaufzeit, "Moral", "Duration (dlaufzeit)")
tab_inter_laufk_dlauf <- generate_interaction_table(credit_agg, laufkont, dlaufzeit, "Account (laufkont)", "Duration (dlaufzeit)")
tab_inter_moral_laufk <- generate_interaction_table(credit_agg, moral, laufkont, "Moral", "Account (laufkont)")
tab_inter_alter_dlauf <- generate_interaction_table(credit_agg, dalter, dlaufzeit, "Age (dalter)", "Duration (dlaufzeit)")

# Combine into master table
eda_interaction_master <- bind_rows(
  tab_inter_moral_dlauf,
  tab_inter_laufk_dlauf,
  tab_inter_moral_laufk,
  tab_inter_alter_dlauf
)

# Export table
write.csv(
  eda_interaction_master, 
  "output/tables/eda_interaction_summary.csv", 
  row.names = FALSE, 
  na = ""
)

# Print for inspection
print(eda_interaction_master)


# Interpretation Plot 1: Discretized Duration (dlaufzeit) vs. Moral:
# All duration bins exhibit broadly parallel trajectories. The line intersections (e.g., 'moral' 0 crossing 'moral' 3 at 37-42 months) are driven by sparse cells (n = 1) with near-complete CI overlap. Visual deviations, such as the steep drop of 'moral' 1 at 25-30 months, are masked by CI widths > 6.00 logits (L: -4.30, U: 2.10, n = 1). Similarly, the spike of 'moral' 3 at 43-48 months corresponds to a CI width > 5.00 logits (L: -1.01, U: 4.91, n = 1). Data sparsity in duration bins >36 months yields overlapping CIs > 4.00 logits across categories. Non-parallelism is indistinguishable from estimation variance. Hence, no robust visual evidence for interaction.

# Interpretation Plot 2: Discretized Duration (dlaufzeit) vs. Laufkont:
# All duration bins exhibit broadly parallel downward trends. The line intersections (e.g., 'laufkont' 1 crossing 'laufkont' 2 at 37-42 months) are caused by sparse cells (n <= 3) and involve near-complete CI overlappings. Extreme visual deviations, such as the steep drop of 'laufkont' 3 at 25-30 months, are masked by CI widths > 5.00 logits (L: -4.30, U: 2.10, n = 1). Moreover, data sparsity in duration bins >36 months yields CI > 4.00 logits across all categories. This shows, non-parallelism is indistinguishable from estimation variance. Hence, no robust visual evidence for interaction.

# Interpretation Plot 3: Moral vs Laufkont:
# All categories exhibit broadly parallel upward trends. Apparent line intersections (e.g., 'laufkont' 1 crossing 'laufkont' 2 between 'moral' 3 and 4) involve sparse cells with substantial CI overlap. The visual deviation of 'laufkont' 1 at 'moral' 3 is masked by a CI width of 3.56 logits (n = 7). Extreme outliers, such as the spike of 'laufkont' 3 at 'moral' 1 (n = 2), exhibit CI widths > 6.00 logits. Apparent non-parallelism is driven entirely by estimation variance. No robust visual evidence for interaction.

# Interpretation Plot 4: Alter vs Laufzeit:
# The plot is characterized by high estimation variance. Some trajectories appear non-parallel (e.g., categories 43-48 and 49-54 months).The CIs exhibit near-complete overlap across all age groups and spanning > 5.00 logits. Structural differences are indistinguishable from statistical noise caused by data sparsity. No robust visual evidence for interaction.


# 5.4 Data Refinement: Category Merging based on EDA ---------------------------
# In case of data sparsity (e.g., dlaufzeit >36,dalter >=60, and beruf level 1), the variance of estimates rises which leads to large CIs. Moreover, if neighbooring groups in categories of covariates habe similar effects on response (similar empirical logits with overlapping CIs), the categories can be merged, if it makes sense in the context
# -> Bias-Variance-Trade-off: Loss of stability at boundaries against win of overall stability, in particular at dense regions.
# In particular, merging categories reduces the number of parameters leading to lower BIC in the model selection phase next chapter.

#Note on covariates moral and laufkont:
# 'laufkont' displays a strictly monotonic risk profile, and 'moral' captures crucial qualitative risk distinctiveness at its lower levels. Merging categories for these variables to fix cell sparsity (<5%) would destroy vital main effects and business logic. Hence, they remain unmerged.

# Define helper merging function to apply transformation on test- and training- data
apply_category_merging <- function(df) {
  df %>%
    mutate(
      # Duration Merging
      dlaufzeit_merged = ifelse(dlaufzeit %in% c("37-42", "43-48", "49-54", ">54"), ">36", as.character(dlaufzeit)),
      # Enforce correct logical ordering (instead of alphabetical default)
      dlaufzeit_merged = factor(dlaufzeit_merged, levels = c("<=6", "7-12", "13-18", "19-24", "25-30", "31-36", ">36")),
      
      # Age Merging
      dalter_merged = ifelse(dalter %in% c("60-64", ">=65"), ">=60", as.character(dalter)),
      # Enforce correct logical ordering
      dalter_merged = factor(dalter_merged, levels = c("<=25", "26-39", "40-59", ">=60")),
      
      # Occupation Merging
      beruf_merged = ifelse(beruf %in% c(1, 2), "1_2", as.character(beruf)),
      # Enforce correct logical ordering
      beruf_merged = factor(beruf_merged, levels = c("1_2", "3", "4"))
    )
}

# Application of transformation on test- and training- data
data_train <- apply_category_merging(data_train)
data_test  <- apply_category_merging(data_test)

# Re-aggregate training data
credit_agg <- aggregate(
  cbind(kredit, no_kredit) ~ laufzeit + dlaufzeit + dlaufzeit_merged + 
    moral + laufkont + 
    alter + dalter + dalter_merged + 
    beruf + beruf_merged, 
  data = data_train,
  FUN = sum
)

# Integrity check: Ensure no observations were lost during re-aggregation
stopifnot(sum(credit_agg$kredit) + sum(credit_agg$no_kredit) == nrow(data_train))
stopifnot(all(credit_agg$kredit >= 0), all(credit_agg$no_kredit >= 0))


# Pre-Check: Design Matrix Rank with respect to merged covariates for Algorithmic Stability # Adjustment needed
# Create the full design matrix for all candidate variables
X_candidate_merged <- model.matrix(~ laufzeit + dlaufzeit_merged + moral + laufkont + alter + dalter_merged + beruf_merged, data = credit_agg)
# Calculating rank using QR decomposition
qr(X_candidate_merged)$rank
# Calculating number of parameters
ncol(X_candidate_merged)
# Conclusion: full rank of 21


# Re-Evaluation of merged categories
# Goal: Verify the variance reduction and stabilization achieved by merging

# Calculation and comparison of relative frequencies
prop.table(table(data_train$beruf_merged))
prop.table(table(data_train$beruf))
prop.table(table(data_train$dalter_merged))
prop.table(table(data_train$dalter))
prop.table(table(data_train$dlaufzeit_merged))
prop.table(table(data_train$dlaufzeit))

# Interpretation: No empty categories (as before) and no sparse categories (each contains at least 5% of the training data)


# 5.4.1 Marginal Effects Comparison (Before vs. After)

# Generate plots for the merged variables using the helper function
library(patchwork)

p_dlaufzeit_merged <- plot_emp_logit(credit_agg, dlaufzeit_merged, "Merged Duration (dlaufzeit_merged)")
p_dalter_merged    <- plot_emp_logit(credit_agg, dalter_merged, "Merged Age (dalter_merged)")
p_beruf_merged     <- plot_emp_logit(credit_agg, beruf_merged, "Merged Occupation (beruf_merged)")

comparison_dlaufzeit <- p_dlaufzeit + p_dlaufzeit_merged
comparison_dalter <- p_dalter + p_dalter_merged
comparison_beruf <- p_beruf + p_beruf_merged

# Save the combined comparative plots
ggsave("output/figures/comparison_dlaufzeit.png", plot = comparison_dlaufzeit, width = 10, height = 4, dpi = 300)
ggsave("output/figures/comparison_beruf.png", plot = comparison_beruf, width = 10, height = 4, dpi = 300)
ggsave("output/figures/comparison_dalter.png", plot = comparison_dalter, width = 10, height = 4, dpi = 300)

# Tabular Comparison of Empirical Logits
tab_dlaufzeit_merged <- generate_emp_logit_table(credit_agg, dlaufzeit_merged, "Merged Duration (dlaufzeit_merged)")
tab_dalter_merged    <- generate_emp_logit_table(credit_agg, dalter_merged, "Merged Age (dalter_merged)")
tab_beruf_merged     <- generate_emp_logit_table(credit_agg, beruf_merged, "Merged Occupation (beruf_merged)")

# Combine raw and merged tables pairwise for direct comparison
eda_emp_logit_comparison <- bind_rows(
  tab_dlaufzeit,
  tab_dlaufzeit_merged,
  tab_dalter,
  tab_dalter_merged,
  tab_beruf,
  tab_beruf_merged
)

# Export the comparison table
write.csv(
  eda_emp_logit_comparison, 
  "output/tables/eda_empirical_logits_comparison.csv", 
  row.names = FALSE, 
  na = "" # Leaves NA cells blank for a cleaner look
)

print(eda_emp_logit_comparison)

# Interpretation:

# Laufzeit: 
# *Raw: Logit extremes ranging from 1.82 (<=6) to -0.35 (43-48). Due to data sparsity, in range of > 36 months, there are structural breaks in downward trend (e.g., spike to 1.73 at 37-42 months bin). Maximum CI width in category 49-54 given by 4.52 (L: -2.26, R: 2.26). 
# Merged (> 36): Aggregation of the four sparse categories (n = 60), the empirical logit is equal to 0. CI of right tail is given by 1.00 (L: -0.5, U: 0.5).
# * Conclusion: Estimation variance was stabilized

# Beruf:
#*Raw: Logit extremes given by 0.93 (3) to 0.44 (1). Due to data sparsity, category 1 (n = 13) exhibits a overlapping with other levels. Maximum CI width in category 1 given by 2.14 (L: -0.63, U: 1.51).
#* Merged (1_2): Aggregation of the sparse category 1 with category 2 (n = 153), the empirical logit is equal to 0.81. CI of the new group is given by 0.68 (L: 0.47, U: 1.15).
#* Conclusion: Estimation variance was stabilized and overlapping CI issues weakened.

# Alter:
#*Raw: Logit extremes ranging from 1.21 (60-64) to 0.33 (<=25). Due to data sparsity, in range of >= 60 years, there is a structural plateau with overlapping CIs. Maximum CI width in category >=65 given by 2.06 (L: -0.38, U: 1.68).
#* Merged (>= 60): Aggregation of the two sparse upper categories (n = 38), the empirical logit is equal to 1. CI of right tail is given by 1.42 (L: 0.29, U: 1.71).
#* Conclusion: Estimation variance was stabilized


# Interpretation of Empirical Logits for Merged Covariates (Post-Refinement)

# 1. Primary Risk Drivers (High Delta, Non-Overlapping CIs):
# - 'dlaufzeit_merged': Marginal effect size Delta = 1.82 (Extremes: <=6: 1.82 vs. >36: 0.00), adjusted from previous raw Delta = 2.17 (Extremes: <=6: 1.82 vs. 43-48: -0.35) due to outlier mitigation. Extremes exhibit no CI overlap. The structural breaks caused by previous tail sparsity are resolved, yielding a strictly monotonic downward trend. The maximum CI width is reduced to 1.28 (Bin 25-30), effectively eliminating estimation noise.

# - 'moral' (Unmerged): Marginal effect size Delta = 2.12 (Extremes: Level 4: 1.49 vs. Level 0: -0.63). Extremes exhibit no CI overlap. Bin 1 exhibits a CI width of 1.50 due to data sparsity (n = 27). The plot overall shows a strictly monotonic upward trend.

# - 'laufkont' (Unmerged): Marginal effect size Delta = 1.73 (Extremes: Level 4: 1.86 vs. Level 1: 0.13). Extremes exhibit no CI overlap. Bin 3 exhibits a CI width > 1.30 due to moderate data sparsity (n = 43). The plot overall shows a strictly monotonic upward trend.

# 2. Secondary Risk Driver (Moderate Delta, Structural Non-Linearity):
# - 'dalter_merged': Concave, approximately quadratic structure (leading coefficient < 0). Marginal effect size Delta = 0.69 (Extremes: 40-59: 1.02 vs. <=25: 0.33), reduced from previous raw Delta = 0.88 (Extremes: 60-64: 1.21 vs. <=25: 0.33). Extreme categories still exhibit substantial CI overlap. Merging the right tail (>=60) eliminated the noisy plateau and stabilized the maximum CI width to 1.42, preserving the structural shape while reducing dimensionality.

# 3. Weak Risk Driver (Low Delta, Severe CI Overlap):
# - 'beruf_merged': Marginal effect size Delta = 0.41 (Extremes: Level 3: 0.93 vs. Level 4: 0.52), adjusted from previous raw Delta = 0.49 (Extremes: Level 3: 0.93 vs. Level 1: 0.44). CI overlaps persist across all levels, indicating weak predictive discrimination. Nevertheless, merging levels 1 and 2 successfully eliminated the masking effect of the former level 1 (CI width > 2.00). The maximum CI width is now stabilized at 0.68 (Level 1_2).


# 5.4.2 Interaction Effects Comparison (Before vs. After)

# Generate Interaction Profile Plots, where merged covariates occured

p_inter_dlauf_merged_moral <- plot_interaction_profile(
  credit_agg, dlaufzeit_merged, moral, 
  "Merged Duration (dlaufzeit_merged)", "Payment Habits\n(moral)"
)


p_inter_dlauf_merged_laufkont <- plot_interaction_profile(
  credit_agg, dlaufzeit_merged, laufkont, 
  "Merged Duration (dlaufzeit_merged)", "Account Status\n(laufkont)"
)


p_inter_alter_merged_dlauf_merged <- plot_interaction_profile(
  credit_agg, dalter_merged, dlaufzeit_merged, 
  "Merged Age (dalter_merged)", "Merged Duration (dlaufzeit_merged)"
)

# Comparison: Laufzeit vs. Moral (Raw vs. Merged)
comparison_inter_dlauf_moral <- p_inter_dlauf_moral + p_inter_dlauf_merged_moral + theme(axis.title.y = element_blank()) +
  plot_layout(guides = "collect")

ggsave("output/figures/comparison_inter_dlauf_moral.png", plot = comparison_inter_dlauf_moral, width = 14, height = 6, dpi = 300)


# Comparison: Laufzeit vs. Laufkont (Raw vs. Merged)
comparison_inter_dlauf_laufkont <- p_inter_dlauf_laufkont + p_inter_dlauf_merged_laufkont + theme(axis.title.y = element_blank()) +
  plot_layout(guides = "collect")

ggsave("output/figures/comparison_inter_dlauf_laufkont.png", plot = comparison_inter_dlauf_laufkont, width = 14, height = 6, dpi = 300)


# Comparison: Alter vs. Laufzeit (Raw vs. Merged)
comparison_inter_alter_laufzeit <- p_inter_alter_laufzeit + p_inter_alter_merged_dlauf_merged + theme(axis.title.y = element_blank()) +
  plot_layout(guides = "collect")

ggsave("output/figures/comparison_inter_alter_laufzeit.png", plot = comparison_inter_alter_laufzeit, width = 14, height = 6, dpi = 300)


# Tabular Comparison of Empirical Logits for Interactions (Before vs. After)

tab_inter_moral_dlauf_merged <- generate_interaction_table(
  credit_agg, moral, dlaufzeit_merged, 
  "Moral", "Merged Duration (dlaufzeit_merged)"
)

tab_inter_laufk_dlauf_merged <- generate_interaction_table(
  credit_agg, laufkont, dlaufzeit_merged, 
  "Account (laufkont)", "Merged Duration (dlaufzeit_merged)"
)

tab_inter_alter_merged_dlauf_merged <- generate_interaction_table(
  credit_agg, dalter_merged, dlaufzeit_merged, 
  "Merged Age (dalter_merged)", "Merged Duration (dlaufzeit_merged)"
)

eda_interaction_comparison <- bind_rows(
  tab_inter_moral_dlauf,
  tab_inter_moral_dlauf_merged,
  
  tab_inter_laufk_dlauf,
  tab_inter_laufk_dlauf_merged,
  
  tab_inter_alter_dlauf,
  tab_inter_alter_merged_dlauf_merged
)

# Export the comparison table
write.csv(
  eda_interaction_comparison, 
  "output/tables/eda_interaction_comparison.csv", 
  row.names = FALSE, 
  na = "" # Leaves NA cells blank
)

# Print for inspection
print(eda_interaction_comparison)

# Interpretation: 
# Merged Duration (dlaufzeit_merged) vs. Moral:
# All duration bins exhibit broadly parallel trend in trajectories. The artificial line intersections and visual deviations in the right tail (previously driven by raw bins >36 months, such as the spike of 'moral' 3 at 43-48 with a CI width > 5.00 logits) where weakened by the new '>36' aggregate. The maximum CI width in this merged tail is now reduced to 3.82 logits for 'moral' 1 (L: -1.06, U: 2.76, n = 4). Remaining line crossings (e.g., 'moral' 1 crossing 'moral' 3) are driven by 2D combinatorial cell sparsity with near-complete CI overlap. Non-parallelism is indistinguishable from estimation variance. Hence, no robust visual evidence for interaction.

# Merged Duration (dlaufzeit_merged) vs. Laufkont:
# All duration bins exhibit broadly parallel downward trends. The CI widths in the new '>36' bin are substantially stabilized for dense categories (e.g., 'laufkont' 2 CI width is reduced to 1.52; L: -1.06, U: 0.46, n = 26). Extreme visual deviations are now strictly isolated to naturally sparse 2D cells (e.g., 'laufkont' 3 at '>36' months with n = 2 yields a masking CI width of 6.08 logits; L: -1.43, U: 4.65). Since structural trajectories follow nearly parallel trend and deviations are bounded by combinatorial variance, there is no robust visual evidence for interaction.

# Merged Age (dalter_merged) vs. Merged Duration (dlaufzeit_merged)
# The previously high estimation variance is reduced. By merging both covariates' sparse tails (Age >=60, Duration >36), the trajectories show nearly parallel trend. The CI width in the joint extreme tail (Age >=60 at Duration >36) is equal to 4.52 logits (L: -2.26, U: 2.26, n = 2), rather caused by combinatorial 2D sparsity than marginal instability. Structural differences are indistinguishable from statistical noise. No robust visual evidence for interaction.



# 6. Model Specification -------------------------------------------------------
# Main Goal: Predictive performance
# Method: AIC as primary model-selection model and BIC as sensitivity criterion

n_indiv <- nrow(data_train) # = 700

# Null model (intercept only)
model_null <- glm(
  cbind(kredit, no_kredit) ~ 1, 
  data = credit_agg, 
  family = binomial(link = "logit")
)

# Functional Form Selection: Laufzeit
mod_l_cont   <- glm(cbind(kredit, no_kredit) ~ laufzeit, data = credit_agg, family = binomial(link = "logit"))
mod_l_raw    <- glm(cbind(kredit, no_kredit) ~ dlaufzeit, data = credit_agg, family = binomial(link = "logit"))
mod_l_merged <- glm(cbind(kredit, no_kredit) ~ dlaufzeit_merged, data = credit_agg, family = binomial(link = "logit"))

AIC(mod_l_cont, mod_l_raw, mod_l_merged, k = log(n_indiv)) # 819.45, 863,75, 851.56 as BIC -> Decision for continous version of 'laufzeit'
AIC(mod_l_cont, mod_l_raw, mod_l_merged, k = 2) # 810.35, 818.25, 819.71 as AIC -> Decision for continous version of 'laufzeit'
# Interpretation: Consistent with the GAM results (strictly linear smooth), the continuous specification 'laufzeit' outperformes both discretizations in BIC and AIC.
# We proceed with 'laufzeit'.

# Functional Form Selection: Alter

mod_a_lin    <- glm(cbind(kredit, no_kredit) ~ alter, data = credit_agg, family = binomial(link = "logit"))
mod_a_quad   <- glm(cbind(kredit, no_kredit) ~ alter + I(alter^2), data = credit_agg, family = binomial(link = "logit"))
mod_a_raw    <- glm(cbind(kredit, no_kredit) ~ dalter, data = credit_agg, family = binomial(link = "logit"))
mod_a_merged <- glm(cbind(kredit, no_kredit) ~ dalter_merged, data = credit_agg, family = binomial(link = "logit"))

AIC(mod_a_lin, mod_a_quad , mod_a_raw, mod_a_merged, k = log(n_indiv)) # 841.38, 844.42, 856.00, 850.07 as BIC -> Decision for 
AIC(mod_a_lin, mod_a_quad , mod_a_raw, mod_a_merged, k = 2) # 832.29, 830.77, 833.25, 831,87 as AIC
# Interpretation:
# The linear form of 'alter' achieves the lowest BIC and the quadratic form achieves the lowest in AIC. The AIC difference of approx. 1.5 w.r.t. the linear and quadratic specification only indicates a tiny improvement in model fit. However, in 5.2, the GAM showed a significantly non-linear (quadratic) effect for age. Since the main goal is predictability, the result of AIC is used.
# We proceed with 'alter' (quadratic).

# Functional Form Selection: Beruf

mod_b_raw    <- glm(cbind(kredit, no_kredit) ~ beruf, data = credit_agg, family = binomial(link = "logit"))
mod_b_merged <- glm(cbind(kredit, no_kredit) ~ beruf_merged, data = credit_agg, family = binomial(link = "logit"))

AIC(mod_b_raw, mod_b_merged, k = log(n_indiv)) # 856.63, 850,47
AIC(mod_b_raw, mod_b_merged, k = 2)  # 838.43, 836.82 
# Interpretation: The merged form of 'beruf' achieves the lowest AIC and BIC. 
# We proceed with 'beruf_merged'.


# Full model (candidate 1)
model_full_raw <- glm(
  cbind(kredit, no_kredit) ~ laufzeit + laufkont + alter + I(alter^2) + beruf + moral, 
  data = credit_agg, 
  family = binomial(link = "logit")
)


# Model selection Forward selection (step()-function) via BIC and AIC

# BIC (forward- and backward- selection)
model_raw_stepwise_forward_bic <- step(
  object = model_null, 
  direction = "forward", 
  scope = formula(model_full_raw), 
  k = log(n_indiv),
  trace = 1
)
summary(model_raw_stepwise_forward_bic)

model_raw_stepwise_backward_bic <- step(
  object = model_full_raw, 
  direction = "backward",
  k = log(n_indiv),
  trace = 1
)
summary(model_raw_stepwise_backward_bic)
# Conclusion: Both methods yield same result - laufkont and laufzeit as only covariates


# AIC (forward- and backward- selection)
model_raw_stepwise_forward_aic <- step(
  object = model_null, 
  direction = "forward", 
  scope = formula(model_full_raw), 
  trace = 1
)
summary(model_raw_stepwise_forward_aic)
#Conclusion: laufkont, moral and laufzeit as covariates

model_raw_stepwise_backward_aic <- step(
  object = model_full_raw, 
  direction = "backward",
  trace = 1
)
summary(model_raw_stepwise_backward_aic)
# Conclusion: laufkont, moral,laufzeit and alter + I(alter^2) as covariates

# Conclusion (overall): Forward and backward AIC selection yield different specifications. Therefore, the role of the quadratic age effect is examined separately using a likelihood-ratio test.

#Question: Does quadratic effect of 'alter' provides significant information to the model with covariates laufkont, moral and laufzeit?
anova(model_raw_stepwise_forward_aic, model_raw_stepwise_backward_aic, test = "Chisq")
# Answer: No significant evidence that adding the linear and quadratic age terms improves model fit (p = 0.171) -> Proceed with covariates laufkont, moral and laufzeit

#Question: Does 'moral' provides significant information to the model with covariates laufkontand laufzeit?
anova(model_raw_stepwise_forward_bic, model_raw_stepwise_forward_aic, test = "Chisq")
# Answer: yes, with p-value of 0.0001258

# Comparison of AIC and BIC
# BIC
AIC(model_raw_stepwise_forward_bic, model_raw_stepwise_forward_aic, k = log(n_indiv)) # 768.34, 771.52 ->  \Delta = 3.18 (rel. small / moderate)
# AIC
AIC(model_raw_stepwise_forward_bic, model_raw_stepwise_forward_aic, k = 2) # 745.59, 730.57 -> Delta = 15.02 (large)


# Decision: Choose the covariates 'laufkont', 'laufzeit' and 'moral' for the model with raw data.
model_raw <-  model_raw_stepwise_forward_aic



# Full model (candidate 2) -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Expectation: According to EDA, beruf has only small influence on response, hence all 4 types of model selection wont let it enter the final model.
model_full_merged <- glm(
  cbind(kredit, no_kredit) ~ laufzeit + laufkont +  alter + I(alter^2) + beruf_merged + moral, 
  data = credit_agg, 
  family = binomial(link = "logit")
)

# BIC (forward- and backward- selection)
model_merged_stepwise_forward_bic <- step(
  object = model_null, 
  direction = "forward", 
  scope = formula(model_full_merged), 
  k = log(n_indiv),
  trace = 1
)
summary(model_merged_stepwise_forward_bic)

model_merged_stepwise_backward_bic <- step(
  object = model_full_merged, 
  direction = "backward",
  k = log(n_indiv),
  trace = 1
)
summary(model_merged_stepwise_backward_bic)
# Conclusion: Both methods yield same result - laufkont and laufzeit as only covariates - same result as for model_full_raw

# AIC (forward- and backward- selection)
model_merged_stepwise_forward_aic <- step(
  object = model_null, 
  direction = "forward", 
  scope = formula(model_full_merged), 
  trace = 1
)
summary(model_merged_stepwise_forward_aic)
#Conclusion: laufkont, moral and laufzeit as covariates

model_merged_stepwise_backward_aic <- step(
  object = model_full_merged, 
  direction = "backward",
  trace = 1
)
summary(model_merged_stepwise_backward_aic)
# Conclusion: laufkont, moral,laufzeit and alter + I(alter^2) as covariates

# Conclusion (overall): Same result as for model_full_raw

# Conclusion: Both candidates yield same model under forward / backward selection via AIC / BIC and Likelihood- Ratio- Tests.
model_final_without_interaction <- model_raw


# Interaction Effects --------------------------------------------------------------------------

# Define interaction models by adding single interaction term to model_final_without_interaction 

model_inter_laufzeit_moral <- update(model_final_without_interaction, . ~ . + laufzeit:moral)
model_inter_laufzeit_laufkont <- update(model_final_without_interaction, . ~ . + laufzeit:laufkont)
model_inter_moral_laufkont <- update(model_final_without_interaction, . ~ . + moral:laufkont)

# Q: Does any specific interaction term significantly reduce the residual deviance?

anova(model_final_without_interaction, model_inter_laufzeit_moral, test = "Chisq") # A: yes, by p-value of 0.02411
anova(model_final_without_interaction, model_inter_laufzeit_laufkont, test = "Chisq") # A: no, by p-value of 0.7274
anova(model_final_without_interaction, model_inter_moral_laufkont, test = "Chisq") # A: no, by p-value of 0.6231


# Comparison of AIC / BIC between final model with additional interaction covariate laufzeit \cdot moral and final model
# BIC
AIC(model_final_without_interaction, model_inter_laufzeit_moral, k = log(n_indiv)) # 771.52, 786.49 -> \Delta = 14.97
# AIC
AIC(model_final_without_interaction, model_inter_laufzeit_moral, k = 2) # 730.57, 727.35 -> \Delta = 3.22

# Final Conclusion on Interaction Effects:
# 1. Statistical Metrics: The LRT (p = 0.02411) and a minor AIC reduction (\Delta AIC = 3.22) provide weak support for the 'laufzeit:moral' interaction. Conversely, the BIC penalizes the parameter expansion, rejecting the interaction (\Delta BIC = +14.97).
# 2. EDA: The weak in-sample significance of the interaction term captures residual statistical noise, not a true structural effect. As discussed in the EDA-section, apparent non-parallel trajectories are strictly isolated to 2D combinatorial cell sparsity (n <= 2) and are masked by CI widths > 4.00 logits. 
# 3. Decision: To prevent overfitting to local noise artifacts and to maximize out-of-sample robustness, the strict parsimony of the BIC is prioritized. The interaction term is excluded.
# 
# Final Model: The main effects specification (laufzeit + laufkont + moral) is selected as the definitive predictive model.
model_main <- model_final_without_interaction
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