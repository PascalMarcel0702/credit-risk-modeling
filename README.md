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

The categorical variables show substantial differences in category frequencies:

| Variable | Category 0 | Category 1 | Category 2 | Category 3 | Category 4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `moral` | 4.0% | 4.9% | 53.0% | 8.8% | 29.3% |
| `laufkont` | - | 27.4% | 26.9% | 6.3% | 39.4% |
| `beruf` | - | 2.2% | 20.0% | 63.0% | 14.8% |

The categorical predictors are unevenly distributed, with particularly sparse groups in `moral` and `beruf`. These differences in category frequency are considered when interpreting uncertainty and apparent deviations in the subsequent EDA.
The continuous predictors are summarized using basic descriptive statistics:

| Variable | Min | Median | Mean | SD | Max |
|:---------|----:|-------:|-----:|----:|----:|
| `laufzeit` | 4 | 18 | 20.9 | 12.1 | 72 |
| `alter` | 19 | 33 | 35.5 | 11.4 | 75 |

The target variable is:

| Variable | Description | Coding |
|---|---|---|
| `kredit` | Credit repayment status | `1` = correctly repaid, `0` = not correctly repaid |

Accordingly, `kredit = 1` represents the non-default class, while `kredit = 0` represents the default class throughout this project.

### Key Results

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
The choice of a standard binomial logistic regression is grounded in the structural properties of the response variable and the absence of overdispersion:

*   **Probability Constraints:** The dependent variable is binary. Standard linear regression cannot constrain predicted values to the $[0, 1]$ interval. The logit link function naturally maps the unbounded linear predictor $\eta_i \in \mathbb{R}$ to valid conditional probabilities.
*   **Interpretability (Canonical Link):** While other cumulative distribution functions (e.g., Probit or Complementary log-log) could restrict predictions to valid probabilities, the logit link is the canonical link for the binomial distribution. It uniquely allows coefficients to be exponentiated into odds ratios, which is crucial for transparent risk differentiation in credit portfolios.
*   **Absence of Overdispersion:** Aggregated binomial profiles often exhibit variance greater than the theoretical binomial variance $np(1-p)$ due to unobserved heterogeneity, which would necessitate mixed models (e.g., Beta-Binomial regression). However, the diagnostic evaluation yielded a Pearson heterogeneity factor ($\hat{\sigma}^2$) close to 1. This indicates no severe overdispersion, confirming that the standard Binomial GLM is adequate and more complex models are not required.

### Data Aggregation
To prevent data leakage during model evaluation, identical covariate profiles within the training set are aggregated into grouped binomial observations:

$$Y_j \sim \text{Binomial}(n_j,\pi_j)$$

*   **$n_j$**: Number of borrowers sharing the identical covariate profile $j$.
*   **$Y_j$**: Observed number of proper loan repayments within profile $j$.
*   **$\pi_j$**: Profile-specific conditional probability of repayment.

Grouping binary responses into binomial counts is a mathematical prerequisite: it ensures the residual deviance and Pearson statistics follow an approximate $\chi^2$ distribution, which enables valid goodness-of-fit and overdispersion diagnostics.

### Mathematical Foundation
Let $\pi_j = P(\text{kredit}_j = 1 \mid \mathbf{x}_j)$ denote the conditional probability of proper repayment for a distinct covariate profile $j$. The model connects the linear predictor $\eta_j = \mathbf{x}_j^\top\boldsymbol{\beta}$ to this probability via the canonical logit link:

$$ \pi_j = \frac{1}{1+\exp(-\eta_j)} \iff \log\left(\frac{\pi_j}{1-\pi_j}\right) = \mathbf{x}_j^\top\boldsymbol{\beta} $$

This equivalence illustrates the dual structural of the chosen model: the left equation naturally bounds the predicted probabilities to the valid $(0, 1)$ interval, while the right equation guarantees a strict linear relationship between the predictors and the log-odds.

With the aggregated binomial data structure $Y_j \sim \text{Binomial}(n_j, \pi_j)$, the regression coefficients are estimated by maximizing the binomial log-likelihood:

$$ \ell(\boldsymbol{\beta}) = \sum_{j=1}^{J} \left[ Y_j \log(\pi_j) + (n_j - Y_j) \log(1 - \pi_j) \right] $$

---

## Exploratory Data Analysis & Functional Form Assessment
Before model selection, an exploratory analysis was conducted to assess category sparsity, functional forms of continuous predictors, and potential interaction effects.

#### Categorical Predictors
Marginal-effect plots on the log-odds scale were used to assess the functional relationships of the categorical predictors.
<p align="center">
  <img src="output/figures/eda_catplot_beruf.png" width="32%">
  <img src="output/figures/eda_catplot_laufkont.png" width="32%">
  <img src="output/figures/eda_catplot_moral.png" width="32%">
</p>

**Figure:** Marginal effects of the categorical predictors. `laufkont` shows a clear, approximately linear upward trend, indicating higher repayment odds for better account statuses. `moral` exhibits an overall upward trend with minor non-monotonic deviations, while `beruf` shows no clear directional pattern.

Given the sparse representation of several categories, particularly `beruf` category 1, deviations at the boundaries should be interpreted cautiously as they are associated with substantially higher estimation uncertainty. The three regarded covariates are therefore specified as linear main effects.

#### Continuous Predictors

The functional form of the continuous predictors was assessed using Generalized Additive Models (GAMs) with smoothing splines. The non-parametric ANOVA test evaluates whether the flexible GAM provides significant evidence of deviation from a linear relationship. The reported `P(Chi)` is the corresponding p-value.

| Predictor      | GAM assessment                      | `P(Chi)` | Decision             |
| :------------- | :---------------------------------- | :------: | :------------------- |
| **`alter`**    | Mildly non-linear visual pattern    |   0.110  | Linear term retained |
| **`laufzeit`** | Approximately linear downward trend |   0.347  | Linear term retained |

<p align="center">
  <img src="output/figures/gam_continuous_predictors.png" width="90%" alt="Functional Form of Continuous Predictors">
</p>

**Figure:** GAM smooths for `alter` and `laufzeit`. The estimated relationships do not provide statistically significant evidence for non-linearity.

With $\alpha = 0.05$, neither predictor shows statistically significant evidence of non-linearity. Both are therefore specified as linear main effects.

#### Interaction Diagnostics
Potential interactions were assessed using empirical logit plots with a continuity correction:

$$
\text{Empirical Logit}_i = \ln\left(\frac{y_i + 0.5}{n_i-y_i+0.5}\right)
$$

To evaluate whether the continuous variable `laufzeit` interacts with the strongest categorical main effects (`moral` and `laufkont`), the empirical logits were plotted across their respective categories.

*1. Interaction Check: Laufzeit vs. Moral*
<p align="center">
  <img src="output/figures/eda_interaction_laufzeit_moral.png" width="70%" alt="Interaction Check: Laufzeit vs. Moral">
</p>

* Core Population: Categories 2 and 4 represent the vast majority of the data (82.3% combined). Both panels exhibit consistent, roughly parallel downward trends.
* Sparse Categories: The erratic crossing patterns observed in categories 0 (4.0%), 1 (4.9%), and 3 (8.8%) are driven by data sparsity and high variance rather than systematic effects.

*2. Interaction Check: Laufzeit vs. Laufkont*
<p align="center">
  <img src="output/figures/eda_interaction_laufzeit_laufkont.png" width="70%" alt="Interaction Check: Laufzeit vs. Laufkont">
</p>

* Core Population: Categories 1 (27.4%), 2 (26.9%), and 4 (39.4%) are well-represented and generally follow a steady downward trend.
* Sparse Categories: Minor non-parallelisms, such as the slight plateau in category 4 between 20 and 30 months, trace back to boundary effects of the non-parametric smoother as data density decreases at higher durations.

**Conclusion**
Since the dominant subgroups across both key variables exhibit parallel trajectories, there is no systemic interaction pattern. Apparent deviations are strictly isolated to sparse data regions. Therefore, proceeding with a parsimonious, additive main-effects model is methodologically sound and effectively prevents overfitting.

#### Specification Decision
The EDA supports a parsimonious additive main-effects specification:

* `laufzeit` as a linear predictor
* `alter` as a linear predictor
* `moral`, `laufkont`, and `beruf` as categorical predictors
* No non-linear transformations
* No interaction terms

The EDA therefore defines the candidate model specification for the subsequent variable-selection procedure.

---

## Model Selection and Comparison

### AIC and BIC
Candidate variables are evaluated sequentially using both the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC). Both criteria balance model fit against complexity, with BIC applying a stricter penalty for the number of estimated parameters ($k$) based on the sample size ($n = 654$):

$$\text{AIC} = -2\ell(\hat{\boldsymbol{\beta}}) + 2k$$
$$\text{BIC} = -2\ell(\hat{\boldsymbol{\beta}}) + \ln(n)k$$

Both criteria unanimously selected the identical model:

$$\text{logit}(\pi_j) = \beta_0 + \beta_1\,\text{laufzeit}_j + \beta_2\,\text{moral}_j + \beta_3\,\text{laufkont}_j$$

Categorical predictors are represented using indicator variables relative to their reference categories. 

The candidate variables `beruf` and `alter` were excluded in both the AIC- and BIC-selected models, as their inclusion failed to improve the likelihood sufficiently to overcome either complexity penalty. This unanimous selection underscores the robustness of the retained predictors and their significant informational contribution to the model.

### Likelihood-Ration Tests
Nested models are additionally compared using sequential likelihood-ratio tests (implemented via Analysis of Deviance). In the context of GLMs, the test statistic $G^2$ is exactly equivalent to the difference in residual deviances ($\Delta D$) between the reduced and the full model:

$$G^2 = D_{\text{reduced}} - D_{\text{full}} = 2\left[ \ell(\hat{\boldsymbol{\beta}}_{\text{full}}) - \ell(\hat{\boldsymbol{\beta}}_{\text{reduced}}) \right]$$

Evaluated against a $\chi^2$ distribution, these partial deviance tests formally confirm the sequential variable selection:

*   **Null vs. Baseline (`moral`):** Adding the baseline predictor `moral` provides a highly significant improvement over the intercept-only model ($p < 0.001$).
*   **Baseline vs. Main (`moral`, `laufkont`, `laufzeit`):** The variables selected by AIC and BIC provide a further, highly significant improvement to the model fit ($p < 0.001$).
*   **Marginal Additions (`alter`, `beruf`):** Adding either `alter` ($p = 0.248$) or `beruf` ($p = 0.758$) individually to the main model yields no significant reduction in residual deviance. This validates the decision of the AIC/BIC selection to exclude both predictors.

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
*   **Laufzeit:** An odds ratio of 0.970 means that a one-month increase in duration multiplies the odds of repayment by 0.970, holding all other predictors constant.
*   **Moral:** The highest factor level (Category 4) has an odds ratio of 5.607 relative to the reference category, indicating substantially higher odds of repayment compared to the baseline moral category.
*   **Laufkont:** The highest factor level (Category 4) has an odds ratio of 5.510 relative to the reference category, indicating substantially higher odds of repayment compared to the baseline current account category.

The displayed probability threshold represents an operating point selected according to the ROC criterion ($J = \text{Sensitivity} + \text{Specificity} - 1$). In a production credit-risk setting, the final decision threshold would additionally depend on asymmetric misclassification costs, risk appetite, and regulatory requirements.

---

## Limitations & Extensions

*   **Model Specification:** The current specification strictly assumes additive main effects. A systematic evaluation of pairwise interaction effects via sequential analysis of deviance was not performed.
*   **Validation Strategy:** The out-of-sample evaluation relies on a single hold-out split without repeated cross-validation or temporal validation.
*   **Decision Thresholds:** No explicit cost-sensitive threshold optimization was applied for the classification cut-off.

Natural extensions include nested model selection incorporating interaction and nonlinear terms, probability calibration analysis, and asymmetric cost-sensitive decision thresholds.
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
