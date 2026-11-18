# Feature importance interpretation guide

This screener does not rely on a single importance score. It evaluates each feature across several complementary lenses so that a feature has to show consistent evidence before it is treated as a strong rare-event signal. This is especially important in highly imbalanced binary problems, where a weakly informative feature can appear deceptively strong when judged by only one metric.

The project reports and uses the following measures:

- Information Value (IV)
- KS statistic
- Univariate PR-AUC
- Permutation PR-AUC drop
- Positive-class SHAP attribution
- Consensus rank

Each measure captures a different notion of usefulness, and each has strengths and weaknesses. The best interpretation comes from looking across the collection rather than trusting one number in isolation.

## 1. Overview of the measures

### Information Value (IV)
What it measures:
- IV measures how much a feature separates the positive and negative classes after grouping values into bins.
- It is a classic scorecard/statistical credit-risk measure and is especially useful when the goal is to identify signal that shifts event odds across ranges.
- In this screener it is computed from binned feature values and captures how strongly the event rate differs across the feature distribution.

How to interpret it:
- A low IV means the feature does not separate the event and non-event distributions well.
- A moderate or high IV indicates the feature strongly changes the event rate across its values.
- In rare-event workflows, extremely large IV values can be a warning sign of leakage, target-proxy behavior, or a near-label feature.

Why it matters:
- IV is a strong summary of distributional separation, but it is not the same as model usefulness and it can be inflated by target leakage or over-binning.

### KS statistic
What it measures:
- KS measures the maximum vertical distance between the empirical cumulative distribution functions of the positive and negative classes.
- It tells us how different the feature distributions are for events and non-events.

How to interpret it:
- Higher KS means stronger distributional separation.
- It is easy to understand and useful for tail separation and skewed risk drivers.

Why it matters:
- KS is good for understanding separation in the raw feature distribution, but it is not a calibrated model measure and does not tell you whether the feature remains useful when combined with other features.

### Univariate PR-AUC
What it measures:
- This is the average precision of the feature alone when used to rank observations for the positive class.
- In imbalanced classification, PR-AUC is often more informative than ROC-AUC because it focuses on the minority class.

How to interpret it:
- A feature with PR-AUC above the baseline positive rate has useful ranking power.
- A very high value can indicate strong single-feature signal, but it can also be a sign of leakage or an overly informative proxy feature.

Why it matters:
- PR-AUC is especially relevant in rare-event problems because the positive class is sparse and the baseline is low.
- It captures ranking quality of the feature itself, but it does not tell you whether the feature helps a model once other variables are included.

### Permutation PR-AUC drop
What it measures:
- This is a model-based importance measure. It evaluates how much the model's validation PR-AUC drops when a feature is randomly shuffled.
- If the feature is truly useful, shuffling it should degrade the model's ability to separate events from non-events.

How to interpret it:
- Larger drops mean the feature is more important to the model.
- Near-zero or negative values suggest weak or unstable signal.

Why it matters:
- This measure captures conditional predictive value, not just raw marginal association.
- It is often more reliable than univariate statistics because it checks whether the feature actually helps the model under realistic data relationships.

### Positive-class SHAP attribution
What it measures:
- SHAP values attribute model predictions to individual features for each sample.
- In this screener the focus is on mean absolute SHAP contribution for positive-class rows, which is a way to identify features that push the model toward the event class.

How to interpret it:
- Larger absolute SHAP values mean the feature contributes more to positive-class predictions.
- This is helpful for understanding which variables are most influential when the event occurs, rather than only which variables are correlated with the target globally.

Why it matters:
- SHAP is model-based and local-by-sample, so it helps explain how the model uses the feature in context.
- It does not prove causal importance, and it can be sensitive to model specification and feature interactions.

### Consensus rank
What it measures:
- Consensus rank combines the ordering of the features across the other metrics.
- It is a summary view that reduces the chance that a feature wins on only one criterion.

How to interpret it:
- Lower consensus rank means the feature is consistently strong across multiple lenses.
- It is a prioritization metric, not a standalone statistical truth.

Why it matters:
- It helps aggregate evidence and is especially helpful when the metrics disagree.

## 2. Why a collection of measures is needed

No single metric is sufficient for a rare-event feature screener because each metric emphasizes a different notion of signal.

- IV tells us how strong the separation is across a feature's distribution.
- KS tells us how different the event and non-event cumulative distributions are.
- Univariate PR-AUC tells us whether the feature ranks positive cases well on its own.
- Permutation PR-AUC drop tells us whether the feature helps the model in a multivariate setting.
- SHAP tells us how model predictions are being driven on positive examples.

These metrics have complementary strengths and weaknesses:

- IV and KS are easy to interpret but can be inflated by leakage or target-proxy behavior.
- Univariate PR-AUC is especially useful under imbalance, but it ignores the other features and can overvalue a feature that only looks good by itself.
- Permutation importance is more realistic for model usefulness, but it can be noisy and is sensitive to model training choices.
- SHAP is explanatory and model-aware, but it is not a causal metric and can be difficult to compare across datasets or models.

Because of this, a feature should be considered strong only when it is supported across multiple measures rather than winning on a single metric alone.

This is exactly why the screener keeps multiple metrics and uses a consensus-based ranking rather than relying exclusively on any one score.

## 3. Interpretation bands and value ranges

The ranges below are practical heuristic bands for rare-event screening. They are not universal constants; they are designed to be useful in this project and in similar imbalanced classification settings.

### 3.1 Information Value bands
Suggested ranges:
- Less than 0.02: weak / not useful
- 0.02 to 0.10: weak-to-moderate
- 0.10 to 0.30: moderate
- 0.30 to 0.50: strong
- Greater than 0.50: very strong; investigate for leakage or target-proxy behavior

Interpretation:
- Low values generally indicate that the feature does not separate positive and negative classes well.
- Moderate values usually indicate a meaningfully useful rank-ordering feature.
- Very high values can be realistic for strong features, but in rare-event settings they should be audited because they may reflect a feature that is effectively behaving like a target label or a leakage proxy.

Reference basis:
- Siddiqi, N. (2006), Credit Risk Scorecards
- Hand and Henley (1997) on scorecard-style separation metrics

### 3.2 KS statistic bands
Suggested ranges:
- Less than 0.10: weak
- 0.10 to 0.20: modest
- 0.20 to 0.30: strong
- Greater than 0.30: very strong

Interpretation:
- KS is an intuitive measure of tail separation. A larger value means that the positive and negative class distributions are more distinct.
- A feature with weak KS is unlikely to be a strong standalone signal unless the interaction structure is very important.

Reference basis:
- Massey (1951), The Kolmogorov-Smirnov Test for Goodness of Fit
- Distributional scorecard and monitoring literature

### 3.3 Univariate PR-AUC bands
Suggested ranges:
- Less than 1.25 times the positive-rate baseline: weak
- 1.25x to 2x the baseline: modest
- 2x to 4x the baseline: meaningful
- Greater than 4x the baseline: strong
- Greater than 0.60: very strong / audit candidate

Interpretation:
- In a rare-event dataset, a feature should outperform the class prevalence baseline by a meaningful margin to be considered useful.
- If a single feature achieves very high PR-AUC, it may be genuinely strong, but it is also a place to watch for leakage or a near-perfect label proxy.

Reference basis:
- Davis and Goadrich (2006), Precision-Recall and ROC curves
- Saito and Rehmsmeier (2015), Precision-Recall plots for imbalanced datasets

### 3.4 Permutation PR-AUC drop bands
Suggested ranges:
- Less than 0.001: negligible contribution
- 0.001 to 0.005: small but marginal
- 0.005 to 0.020: meaningful contribution
- Greater than 0.020: strong contribution
- Less than 0: likely noise or unstable signal; treat cautiously

Interpretation:
- A feature causing a larger drop is more important to the predictive model.
- Small positive values can still be useful if the feature is part of a multivariate interaction pattern, but they generally do not justify a top-tier ranking.
- Negative values are not proof of a harmful feature; they usually indicate that the feature is not helping, or that the model is not using it in a stable way.

Reference basis:
- Breiman (2001), Random Forests
- Permutation importance methods in predictive modeling

### 3.5 Positive-class SHAP bands
Suggested ranges:
- Bottom quartile: weak positive-class contribution
- Middle quartiles: modest contribution
- Above 75th percentile: strong positive-class contribution
- Top decile: very strong / likely influential driver

Interpretation:
- SHAP helps answer: when the event happens, how strongly does this feature push the model toward the positive class?
- A feature may have modest raw distributional separation but still show meaningful SHAP if it matters for edge cases in the positive class.
- A feature with both high SHAP and high permutation lift is usually an especially credible signal.

Reference basis:
- Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions
- Molnar (2020), Interpretable Machine Learning

## 4. How to read a feature in practice

A feature should not be judged by just one number.

A strong feature typically shows a consistent pattern such as:
- moderate-to-high IV
- moderate-to-high KS
- PR-AUC meaningfully above the baseline
- positive permutation lift
- strong positive-class SHAP contribution
- tier classification as Anchor or Interaction

A weak feature usually looks like:
- low IV
- low KS
- PR-AUC near the positive-rate baseline
- small or negative permutation lift
- low SHAP attribution
- tier classification as Weak / Redundant or Noise / Drop

A suspicious feature often shows:
- extremely high IV or univariate PR-AUC
- a large gap between raw marginal separation and actual model contribution
- an implausibly strong signal for a single feature in a rare-event environment
- tier classification as Leakage Check

This is one of the main reasons the screener includes a leakage threshold and an explicit “Leakage Check” tier: some features may look good because they encode the target too directly or because of target leakage in the preprocessing stage.

## 5. Final guidance

The most robust interpretation is the combination of all metrics:

- IV and KS answer: does the feature separate classes?
- Univariate PR-AUC answers: can the feature rank positive cases by itself?
- Permutation PR-AUC drop answers: does the feature matter in a model?
- SHAP answers: how much does the model rely on it when events happen?
- Consensus rank answers: is the feature consistently strong across all of these views?

The screener is built around this philosophy: a feature is most credible when it wins across multiple lenses rather than when it only looks strong under one metric.

## 6. References

- Davis, J., and Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves.
- Hand, D. J., and Henley, W. E. (1997). Statistical Classification Methods in Consumer Credit Scoring: A Review.
- Lundberg, S. M., and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
- Massey, F. J. (1951). The Kolmogorov-Smirnov Test for Goodness of Fit.
- Molnar, C. (2020). Interpretable Machine Learning.
- Saito, T., and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.
- Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.
