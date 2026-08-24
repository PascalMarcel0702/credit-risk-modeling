# Credit Risk Modeling and Classification

This project develops an interpretable logistic regression model for credit risk prediction. The model estimates borrower-specific probabilities of repayment and thereby supports credit-risk differentiation rather than relying solely on binary classification. The workflow combines binomial GLMs, functional-form assessment, likelihood-based model selection, residual and influence diagnostics, and out-of-sample validation.

### Data & Variable Definitions

The dataset contains 1,000 credit observations and is documented by the Ludwig-Maximilians-Universität München. The original variable definitions and category descriptions are provided in the [dataset documentation](https://data.ub.uni-muenchen.de/23/1/DETAILS.html).

The model uses the following predictors:

| Variable | Description | Type |
|---|---|---|
| `laufzeit` | Credit duration in months | Numeric |
| `moral` | Previous payment behavior | Categorical |
| `laufkont` | Existing current account status | Categorical |
| `alter` | Borrower age in years | Numeric |
| `beruf` | Occupation | Categorical |

The target variable is:

| Variable | Description | Coding |
|---|---|---|
| `kredit` | Credit repayment status | `1` = correctly repaid, `0` = not correctly repaid |

Accordingly, `kredit = 1` represents the non-default class, while `kredit = 0` represents the default class throughout this project.

## Key Results

| Aspect | Result |
|---|---|
| Observations | 1,000 borrowers |
| Training profiles (aggregated)| 700 (654) |
| Selected predictors | `laufzeit`, `moral`, `laufkont` |
| Removed predictors | `beruf`, `alter` |
| Test AUC | 0.809 |
| Test Brier Score | 0.161 |
| Test Error Rate | 25.9% |

---

## Methodology

### Why Logistic Regression?
Logistic regression was chosen because it combines probabilistic prediction with direct coefficient and odds-ratio interpretation, making it particularly suitable for an interpretable credit-risk setting.

### Data Aggregation
Identical covariate profiles are aggregated into grouped binomial observations:

$$Y_j \sim \text{Binomial}(n_j,\pi_j)$$

where $n_j$ is the number of borrowers in profile $j$, $Y_j$ is the observed number of repayments, and $\pi_j$ is the profile-specific probability of repayment.

The aggregation preserves the binomial likelihood and provides the grouped structure used for residual and goodness-of-fit diagnostics.

### Mathematical Foundation
The linear predictor $\eta_i = \mathbf{x}_i^\top\boldsymbol{\beta}$ is mapped to the repayment probability via the inverse logit function:

$$\pi_i = \frac{1}{1+\exp(-\eta_i)}$$

Equivalently, the model estimates log-odds:

$$\log\left(\frac{\pi_i}{1-\pi_i}\right) = \mathbf{x}_i^\top\boldsymbol{\beta}$$

Here, $\pi_i = P(kredit_i = 1 \mid \mathbf{x}_i)$ denotes the conditional probability of repayment.
Coefficients are estimated by maximum likelihood:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i\log(\pi_i) + (1-y_i)\log(1-\pi_i) \right]$$

### Model Selection
Additional candidate variables are evaluated sequentially. AIC balances model fit against model complexity by penalizing the number of estimated parameters:

$$\text{AIC} = -2\ell(\hat{\boldsymbol{\beta}}) + 2k$$

The selected model is:

$$\text{logit}(\pi_i) = \beta_0 + \beta_1\,\text{laufzeit}_i + \beta_2\,\text{moral}_i + \beta_3\,\text{laufkont}_i$$

Categorical predictors are represented using indicator variables relative to their reference categories.

The variables `beruf` and `alter` were excluded because their inclusion did not reduce AIC sufficiently to justify the additional parameters.

### Model Comparison
Nested models are additionally compared using likelihood-ratio tests:

$$G^2 = 2\left[ \ell(\hat{\boldsymbol{\beta}}_{full}) - \ell(\hat{\boldsymbol{\beta}}_{reduced}) \right]$$

Likelihood-ratio tests assess whether additional predictors significantly improve model fit, while AIC provides the complementary complexity-adjusted selection criterion.

---

## Model Diagnostics

### Functional Form
**Question:** Is `laufzeit` adequately modeled as a linear predictor?

![Partial residual plot](output/figures/partial_residual_laufzeit.png)

**Method:** Partial residual analysis with LOESS.

**Evidence:** No pronounced systematic curvature is visible.

**Interpretation:** The plot provides no strong graphical evidence of systematic nonlinearity in the effect of `laufzeit` on the logit scale.

**Decision:** Retain `laufzeit` as a linear predictor.

### Goodness-of-Fit
**Question:** Does the model adequately describe the grouped data?

**Method:** Residual deviance test.

**Evidence:** The residual deviance is compared with its asymptotic $\chi^2_{df}$ reference distribution.

**Result:** Residual deviance = 693.8, df = 645, p = 0.089.

**Interpretation:** The residual deviance does not provide statistically significant evidence of lack of fit at the 5% level.

**Decision:** No evidence of lack of fit; retain the specification.

### Residual Structure
**Question:** Are there systematic residual patterns indicating model misspecification?

![Adjusted Pearson residuals](output/figures/residuals_adjusted.png)

**Method:** Leverage-adjusted Pearson residuals.

**Evidence:** Residuals fluctuate around zero without pronounced systematic patterns or clusters of extreme observations.

**Interpretation:** The residual structure provides no obvious evidence of systematic misspecification.

**Decision:** No evidence requiring a change in model specification.

### Influence Diagnostics
**Question:** Do individual covariate profiles exert disproportionate influence on the fitted model?

![Leverage](output/figures/leverage_plot.png)

![Cook's Distance](output/figures/cooks_distance.png)

**Method:** Leverage and Cook's Distance.

**Evidence:** Some profiles exceed the leverage reference threshold $2p/n$, while Cook's distances remain small.

**Interpretation:** Some profiles represent unusual predictor combinations, but none appears to exert disproportionate influence on the fitted coefficients.

**Decision:** Retain all observations.

---

## Validation & Risk Interpretation

### Out-of-Sample Performance
Stratification approximately preserves the class distribution across the training and test samples.

| Metric | Train | Test |
|---|---:|---:|
| AUC | 0.751 | 0.809 |
| Brier Score | 0.175 | 0.161 |
| Error Rate | 25.2% | 25.9% |

The similarity between training and test performance provides no pronounced evidence of overfitting on this hold-out sample.

### Discrimination
**Question:** Can the model distinguish higher-risk borrowers from lower-risk borrowers?

![ROC Curve](output/performance/roc_curve.png)

**Method:** ROC analysis and Area Under the Curve (AUC).

$$\text{AUC} = P(\hat p_{\text{repayment}} > \hat p_{\text{default}})$$

AUC measures how well the model distinguishes borrowers who repay their credit from borrowers who do not.

**Evidence:** The ROC curve lies above the random-classification benchmark, with an AUC of 0.811.

**Interpretation:** The model demonstrates good discrimination between repayment and default outcomes.

**Decision:** The model provides useful ranking information for risk differentiation.

### Probabilistic Accuracy
**Method:** Brier Score / MSE.

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat p_i-y_i)^2$$

The Brier Score measures the mean squared error of probabilistic predictions, with lower values indicating better probabilistic accuracy. It complements the threshold-independent discrimination measure AUC.

### Risk Interpretation
Odds ratios are defined as:

$$\text{OR}_j = e^{\beta_j}$$

An odds ratio above 1 indicates higher odds of repayment for a one-unit increase in the predictor, holding all other variables constant. For categorical variables, the odds ratio is interpreted relative to the reference category.

**Key Predictor Impacts:**
*   **Laufzeit:** An odds ratio of 0.966 means that a one-month increase in duration multiplies the odds of repayment by 0.966, holding all other predictors constant.
*   **Moral:** Category 4 has an odds ratio of 4.601 relative to the reference category, indicating substantially higher odds of repayment compared to the baseline moral category.

The displayed probability threshold represents an operating point selected according to the ROC criterion ($J = \text{Sensitivity} + \text{Specificity} - 1$). In a production credit-risk setting, the final decision threshold would additionally depend on asymmetric misclassification costs, risk appetite, and regulatory requirements.

---

## Limitations & Extensions

*   Single hold-out split (no repeated cross-validation).
*   No temporal validation.
*   No explicit cost-sensitive threshold optimization.

Natural extensions include repeated cross-validation, calibration analysis, interaction effects, nonlinear terms, and cost-sensitive decision thresholds.

---

## Repository Structure

```text
.
├── data/
│   └── credit.txt                              (Raw dataset)
├── output/
│   ├── figures/                                
│   │   ├── cooks_distance.png                  (Cook's distance analysis)
│   │   ├── gam_alter.png                       (GAM smooth term for age)
│   │   ├── gam_laufzeit.png                    (GAM smooth term for duration)
│   │   ├── leverage_plot.png                   (Leverage analysis)
│   │   ├── partial_residual_laufzeit.png       (Linearity check for duration)
│   │   ├── residuals_adjusted.png              (Adjusted Pearson residuals)
│   │   ├── residuals_deviance.png              (Deviance residuals)
│   │   └── residuals_pearson.png               (Pearson residuals)
│   ├── performance/                        
│   │   ├── classification_metrics.csv          (Test vs. Train risk metrics)
│   │   ├── confusion_matrix.csv                (Absolute prediction counts)
│   │   └── roc_curve.png                       (High-res ROC visualization)
│   ├── tables/                                 
│   │   ├── goodness_of_fit.csv                 (Residual deviance GoF test)
│   │   ├── model_comparison_aic.csv            (AIC stepwise selection steps)
│   │   └── odds_ratios.csv                     (Model coefficients and ORs)
│   ├── credit_agg.rds                          (Saved aggregated training dataset)
│   ├── data_test.rds                           (Saved hold-out test split)
│   ├── data_train.rds                          (Saved training split)
│   └── model_main.rds                          (Saved final GLM object)
├── script/
│   ├── 01_data_prep_and_selection.R            (Split, aggregation, and stepwise AIC selection)
│   ├── 02_model_diagnostics.R                  (Residual analysis and influence metrics)
│   └── 03_model_performance.R                  (Out-of-sample hold-out and ROC/Brier evaluation)
├── credit-risk-modeling.Rproj                  (RStudio project file)
└── README.md                                   (Project documentation)
