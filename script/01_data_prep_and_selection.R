# ==============================================================================
# Script: 01_data_prep_and_selection.R
# Purpose: Data inspection, functional form assessment EDA,  
#          and model selection for credit default prediction.
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

# Note: The covariate hoehe is added for section 3, model evaluation
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

# 4.2 Aggregate data
#  Group binary data to obtain binomial response. 
#  Note: Residual deviance in binary regression model cannot be used to evaluate
#  goodness of fit, since it does not approximately follow a \chi^2 distribution asymptotically. For binomial data, the residual deviance approx. follows a \chi^2 distribution asymptotically (under assumption of a correcet model), since the number of parameters is fixed relative to sample size


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
# No empty, but sparse categories exist:
# - 'beruf' level 1 (1.9%).
# - The discretized continuous variables:
#   'dalter': age groups > 60 years contain in total 5.4%.
#   'dlaufzeit': durations > 36 months are fragmented - bins like '49-54' 
#    contains 0.3% of obs.
# Consequence: High estimation variances and CI widths (high uncertainity)

# 5.1 Categorical Variables (Qualitative)
# Objective: Visualize the marginal effect on response by plotting empirical logits

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

# Helper function to generate table for a single variable
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
    # Select required columns
    select({{ grouping_var }}, `Empir. logits`, n, L, U) %>%
    # Format 'n' as numeric
    mutate(n = as.numeric(n)) %>%
    # Metrics as rows and categories as columns
    pivot_longer(
      cols = c(`Empir. logits`, n, L, U), 
      names_to = "Metric", 
      values_to = "Value"
    ) %>%
    pivot_wider(
      names_from = {{ grouping_var }}, 
      values_from = Value
    ) %>%
    # Add a column for variable name
    mutate(Variable = var_name, .before = 1)
}

# Generate tables
tab_moral     <- generate_emp_logit_table(credit_agg, moral, "Payment History (moral)")
tab_laufkont  <- generate_emp_logit_table(credit_agg, laufkont, "Account Status (laufkont)")
tab_beruf     <- generate_emp_logit_table(credit_agg, beruf, "Occupation (beruf)")
tab_dalter    <- generate_emp_logit_table(credit_agg, dalter, "Age Group (dalter)")
tab_dlaufzeit <- generate_emp_logit_table(credit_agg, dlaufzeit, "Duration Group (dlaufzeit)")

# Combine single tables into master table
eda_emp_logit_master <- bind_rows(
  tab_moral, 
  tab_laufkont, 
  tab_beruf, 
  tab_dalter, 
  tab_dlaufzeit
)

# Export the master table as CSV
write.csv(
  eda_emp_logit_master, 
  "output/tables/eda_empirical_logits_summary.csv", 
  row.names = FALSE, 
  na = "" # Leaves NA cells blank
)

print(eda_emp_logit_master)


# Interpretation of Empirical Logits by Logit-Delta and CI separation

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

# For continuous covariates (alter, laufzeit) investigate relationship with the log-odds
# Use Generalized Additive Models (GAMs) with smoothing splines s() to estimate
# data-driven shape of the covariate. 
# 
# Interpretation of the 'mgcv::gam' summary:
# - Check approximate significance of smooth terms.
# - A significant p-value (p < 0.05) combined with  'Effective Degrees of Freedom' 
#   (edf) > 1 indicates non-linear relationship.
# - If not significant, retain the linear main effect.

# Create plot for EDA
png("output/figures/gam_continuous_predictors.png", width = 2450, height = 1200, res = 300)
# Set global graphical parameters for base R plot
# mfrow = c(1, 2) creates a 1x2 grid (side-by-side)
# bty = "l" removes the top and right box borders
# las = 1 makes all axis labels horizontal
par(mfrow = c(1, 2), mar = c(5, 5, 4, 2) + 0.1, las = 1, bty = "l", cex.main = 1.2, cex.lab = 1.1)

# 'alter' (Age)
gam_alter <- mgcv::gam(
  cbind(kredit, no_kredit) ~ s(alter),
  data = credit_agg,
  family = binomial(link = "logit")
)

plot(gam_alter, 
     se = TRUE, 
     xlab = "Age in Years", 
     ylab = "Partial Effect on Log-Odds",
     col = "#2c3e50",
     lwd = 2.5        
)

summary(gam_alter)
# Interpretation for 'alter':
# The smooth term for 'alter' has estimated degree of freedom (edf) of 1.903 > 1.
# In addition: Approximate significance test with p-value 0.0246 < 0.05.
# Conclusion: Influence of quadratic (polynomial) transformation of 'alter' on response should be investigated in model selection phase.

# 'laufzeit' (Duration)
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
# The plot shows a linear downward trend with small CI - width in dense data regions.
# The smooth term is approx. a straight line, with edf = 1 with p-value of 5.83e-07 < 0.05 
# Conclusion: The linear main effect of 'laufzeit' on the response should be investigated in model selection phase.

# Clean up temporary GAM fits
rm(gam_alter, gam_laufzeit)

# 5.3 Exploratory Data Analysis for Interaction Effects-------------------------
# Objective: Check empirical logit plots for non-parallel trends to visually detect possible interaction terms.
# Instead of all possible combinations, analyze the strongest main effects driven by hypotheses


# Helper function for Interaction plots
plot_interaction_profile <- function(df, x_var, group_var, x_label, legend_label) {
  # Grouping and calculating metrics
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
  # Plot
  pd <- position_dodge(width = 0.3)
  
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
      axis.text.x = element_text(angle = 45, hjust = 1) # Rotated discrete bins on x- axis
    )
}


# Plot 1: Laufzeit vs. Moral (The longer a loan runs (maturity), the higher the underlying risk of unforeseen life events (unemployment, illness))
# Q: Does a excellent credit history (moral = 4) provide better protection against this long-term risk than a critical history (moral = 0)?

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


# Table of Empirical Logits for Interactions consisting of empirical logits, cell sizes (n), and 95% CIs (L, U)

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
    # Row order
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
# Data sparsity (e.g., dlaufzeit >36,dalter >=60, and beruf level 1), rises variance of estimates and thus leads to large CIs. If neighbooring bins of a covariate have similar effects on response (similar empirical logits with overlapping CIs), the categories can be merged, if it makes sense in the context
# -> Bias-Variance-Trade-off: Loss of stability at boundaries against win of overall stability, in particular at dense regions.
# Note: merging categories reduces the number of parameters leading to lower BIC in the model selection phase next chapter.

#Note on covariates moral and laufkont:
# 'laufkont' displays a strictly monotonic risk profile and thus no categories are merged as it would destroy vital main effects
# 'moral' captures crucial qualitative risk categories at its lower levels, which should not be merged from business context

# Define helper merging function
apply_category_merging <- function(df) {
  df %>%
    mutate(
      # laufzeit
      dlaufzeit_merged = ifelse(dlaufzeit %in% c("37-42", "43-48", "49-54", ">54"), ">36", as.character(dlaufzeit)),
      dlaufzeit_merged = factor(dlaufzeit_merged, levels = c("<=6", "7-12", "13-18", "19-24", "25-30", "31-36", ">36")),
      
      # alter
      dalter_merged = ifelse(dalter %in% c("60-64", ">=65"), ">=60", as.character(dalter)),
      dalter_merged = factor(dalter_merged, levels = c("<=25", "26-39", "40-59", ">=60")),
      
      # beruf
      beruf_merged = ifelse(beruf %in% c(1, 2), "1_2", as.character(beruf)),
      beruf_merged = factor(beruf_merged, levels = c("1_2", "3", "4"))
    )
}

# apply transformation on test- and training- data
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


# Pre-Check: Design Matrix Rank with respect to merged covariates for Algorithmic Stability
# Create the full design matrix for all candidate variables
X_candidate_merged <- model.matrix(~ laufzeit + dlaufzeit_merged + moral + laufkont + alter + dalter_merged + beruf_merged, data = credit_agg)
# Calculating rank using QR decomposition
qr(X_candidate_merged)$rank
# Calculating number of parameters
ncol(X_candidate_merged)
# Conclusion: full rank of 21


# Re-Evaluation of merged categories
# Goal: Verify the variance reduction achieved by merging

# Calculation and comparison of relative frequencies
prop.table(table(data_train$beruf_merged))
prop.table(table(data_train$beruf))
prop.table(table(data_train$dalter_merged))
prop.table(table(data_train$dalter))
prop.table(table(data_train$dlaufzeit_merged))
prop.table(table(data_train$dlaufzeit))

# Evaluation: No empty categories (as before) and no sparse categories (each contains at least 5% of the training data)


# 5.4.1 Marginal Effects Comparison (Before vs. After)

# Generate plots for the merged variables using the helper function
library(patchwork)

p_dlaufzeit_merged <- plot_emp_logit(credit_agg, dlaufzeit_merged, "Merged Duration (dlaufzeit_merged)")
p_dalter_merged    <- plot_emp_logit(credit_agg, dalter_merged, "Merged Age (dalter_merged)")
p_beruf_merged     <- plot_emp_logit(credit_agg, beruf_merged, "Merged Occupation (beruf_merged)")

comparison_dlaufzeit <- p_dlaufzeit + p_dlaufzeit_merged
comparison_dalter <- p_dalter + p_dalter_merged
comparison_beruf <- p_beruf + p_beruf_merged

# Save the combined plots
ggsave("output/figures/comparison_dlaufzeit.png", plot = comparison_dlaufzeit, width = 10, height = 4, dpi = 300)
ggsave("output/figures/comparison_beruf.png", plot = comparison_beruf, width = 10, height = 4, dpi = 300)
ggsave("output/figures/comparison_dalter.png", plot = comparison_dalter, width = 10, height = 4, dpi = 300)

# Tabular Comparison of Empirical Logits
tab_dlaufzeit_merged <- generate_emp_logit_table(credit_agg, dlaufzeit_merged, "Merged Duration (dlaufzeit_merged)")
tab_dalter_merged    <- generate_emp_logit_table(credit_agg, dalter_merged, "Merged Age (dalter_merged)")
tab_beruf_merged     <- generate_emp_logit_table(credit_agg, beruf_merged, "Merged Occupation (beruf_merged)")

# Combine raw and merged tables pairwise for comparison
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
  na = ""
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
#* Conclusion: Estimation variance was stabilized and reduction of overlapping CIs.

# Alter:
#*Raw: Logit extremes ranging from 1.21 (60-64) to 0.33 (<=25). Due to data sparsity, in range of >= 60 years, there is a structural plateau with overlapping CIs. Maximum CI width in category >=65 given by 2.06 (L: -0.38, U: 1.68).
#* Merged (>= 60): Aggregation of the two sparse upper categories (n = 38), empirical logit is equal to 1. CI of right tail is given by 1.42 (L: 0.29, U: 1.71).
#* Conclusion: Estimation variance was stabilized


# Interpretation of Empirical Logits for Merged Covariates (Post-Refinement)

# 1. Primary Risk Drivers (High Delta, Non-Overlapping CIs):
# - 'dlaufzeit_merged': Marginal effect size Delta = 1.82 (Extremes: <=6: 1.82 vs. >36: 0.00), adjusted from previous raw Delta = 2.17 (Extremes: <=6: 1.82 vs. 43-48: -0.35) due to integration of outliers in dense bins. Extremes show no CI overlap. The structural break caused by sparsity is weakend, yielding a strictly monotonic downward trend. The maximum CI width is reduced to 1.28 (Bin 25-30), eliminating estimation noise.

# - 'moral' (Unmerged): Marginal effect size Delta = 2.12 (Extremes: Level 4: 1.49 vs. Level 0: -0.63). Extremes exhibit no CI overlap. Bin 1 exhibits a CI width of 1.50 due to data sparsity (n = 27). The plot overall shows a strictly monotonic upward trend.

# - 'laufkont' (Unmerged): Marginal effect size Delta = 1.73 (Extremes: Level 4: 1.86 vs. Level 1: 0.13). Extremes exhibit no CI overlap. Bin 3 exhibits a CI width > 1.30 due to moderate data sparsity (n = 43). The plot overall shows a strictly monotonic upward trend.

# 2. Secondary Risk Driver (Moderate Delta, Structural Non-Linearity):
# - 'dalter_merged': Concave, approximately quadratic structure (leading coefficient < 0). Marginal effect size Delta = 0.69 (Extremes: 40-59: 1.02 vs. <=25: 0.33), reduced from previous raw Delta = 0.88 (Extremes: 60-64: 1.21 vs. <=25: 0.33). Extreme categories exhibit CI overlap. Merging right tail (>=60) weakend noisy plateau and reduced the maximum CI width to 1.42. 

# 3. Weak Risk Driver (Low Delta, Severe CI Overlap):
# - 'beruf_merged': Marginal effect size Delta = 0.41 (Extremes: Level 3: 0.93 vs. Level 4: 0.52), adjusted from previous raw Delta = 0.49 (Extremes: Level 3: 0.93 vs. Level 1: 0.44). CI overlaps persist across all levels, indicating weak predictive effect on response. Merging levels 1 and 2 weakened masking effect of level 1 bin (CI width > 2.00). The maximum CI width was reduced to 0.68 (Level 1_2).


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
  na = ""
)

print(eda_interaction_comparison)

# Interpretation: 
# Merged Duration (dlaufzeit_merged) vs. Moral:
# All duration bins exhibit nearly parallel trend in trajectories. The line intersections and deviations in the right tail (previously driven by raw bins >36 months, such as the spike of 'moral' 3 at 43-48 with a CI width > 5.00 logits) were weakened by the new '>36' aggregate. The maximum CI width in aggregated '> 36' bin is reduced to 3.82 logits for bin 1 w.r.t. 'moral' (L: -1.06, U: 2.76, n = 4). Remaining intersections of lines (e.g., 'moral' 1 crossing 'moral' 3) are driven by 2D combinatorial cell sparsity with near-complete CI overlap, which indicates high uncertainity. Non-parallelism is indistinguishable from estimation variance. Hence, no robust visual indication for interaction.

# Merged Duration (dlaufzeit_merged) vs. Laufkont:
# All duration bins show broadly parallel downward trends. The CI widths in the new '>36' bin are stabilized for dense categories (e.g., 'laufkont' 2 CI width is reduced to 1.52; L: -1.06, U: 0.46, n = 26). Extreme visual deviations are caused by sparse 2D cells (e.g., 'laufkont' 3 at '>36' months with n = 2 yields a masking CI width of 6.08 logits; L: -1.43, U: 4.65). All in all: Structural trajectories follow nearly parallel trend and deviations are bounded by combinatorial variance, there is no robust visual indication for interaction.

# Merged Age (dalter_merged) vs. Merged Duration (dlaufzeit_merged)
# The estimation variance is reduced. Merging sparse bins of the underlying covariates (Age >=60, Duration >36), yields that the trajectories show nearly parallel trend. The CI width in the joint extreme tail (Age >=60 at Duration >36) is equal to 4.52 logits (L: -2.26, U: 2.26, n = 2), rather caused by combinatorial 2D sparsity than marginal instability. Structural differences are indistinguishable from statistical noise. No robust visual indication for interaction.

# Note: The influence of the investigated interaction effects on the response is formally checked in the next chapter

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

AIC(mod_l_cont, mod_l_raw, mod_l_merged, k = log(n_indiv)) # 819.45, 863,75, 851.56 as BIC -> Decision for continuous version of 'laufzeit'
AIC(mod_l_cont, mod_l_raw, mod_l_merged, k = 2) # 810.35, 818.25, 819.71 as AIC -> Decision for continous version of 'laufzeit'
# Interpretation: Consistent with the GAM results (strictly linear smooth), the continuous specification 'laufzeit' outperforms both discretizations in BIC and AIC.
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
# 1. Statistical Metrics: The weak p-value p = 0.02411 in LRT and the minor AIC reduction (\Delta AIC = 3.22) provide weak support for the 'laufzeit:moral' interaction. Conversely, the BIC penalizes the parameter expansion, rejecting the interaction (\Delta BIC = +14.97).
# 2. EDA: The weak in-sample significance of the interaction term is likely driven by statistical noise rather than a true structural effect. As discussed in the EDA-section, non-parallel trajectories are strictly isolated to 2D combinatorial cell sparsity (n <= 2) and are masked by CI widths > 4.00 logits. 
# 3. Decision: To prevent overfitting and to maximize out-of-sample robustness, the interaction term 'laufzeit:moral' is excluded

# Final Model: The main effects specification (laufzeit + laufkont + moral) is selected as the predictive model.
model_main <- model_final_without_interaction
summary(model_main)

# Model Parameters (Coefficients, Odds Ratios, Confidence Intervals)
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

# Export final metrics
final_metrics <- data.frame(
  Model_Specification = c("Null Model", "Main Effects (BIC penalty)", "Main Effects (AIC penalty)"),
  Degrees_of_Freedom = c(model_null$rank, model_raw_stepwise_forward_bic$rank, model_main$rank),
  Residual_Deviance = c(model_null$deviance, model_raw_stepwise_forward_bic$deviance, model_main$deviance),
  AIC_Score = c(AIC(model_null), AIC(model_raw_stepwise_forward_bic), AIC(model_main)),
  BIC_Score = c(BIC(model_null), BIC(model_raw_stepwise_forward_bic), BIC(model_main))
)

write.csv(
  final_metrics, 
  "output/tables/final_model_selection_metrics.csv", 
  row.names = FALSE
)

# Export raw splits for out-of-sample evaluation
saveRDS(data_train, "output/data_train.rds")
saveRDS(data_test, "output/data_test.rds")
