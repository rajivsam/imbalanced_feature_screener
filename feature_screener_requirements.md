
# Imbalanced Feature Discovery Wrapper

## Requirements & Specification

## 1. Overview & Objective

A reusable, scikit-learn-compatible exploratory wrapper, `ImbalancedFeatureScreener`, designed to discover, rank, and visually diagnose discriminative features in highly imbalanced binary classification problems (1–2% positive class, near-anomaly regime).

Standard feature selection fails at low prevalence because the 98–99% majority class dominates variance, correlation, and default tree-split gains. This tool shifts evaluation to tail behavior, rank precision, and minority-class attribution.

## 2. Functional Requirements

### 2.1 Core API (`BaseEstimator`, `TransformerMixin`)

- `.fit(X, y)`: Runs univariate screening, out-of-fold multivariate evaluation, consensus ranking, and leakage detection. Stores structured diagnostics in `self.report_` and caches fitted binning tables in `self.binning_tables_`.
- `.transform(X)`: Filters input DataFrames down to `self.selected_features_` (top-k consensus features) for seamless pipeline integration.
- Data type handling: Automatically separates numerical and categorical columns, handling missing values and high-cardinality categoricals without manual preprocessing.

### 2.2 Multi-Lens Scoring Engine

To prevent blind spots between standalone tail signals and non-linear interactions, every feature must be scored across four complementary metrics:

1. Information Value (IV): Computed via `optbinning.OptimalBinning` with `min_bin_size=0.02` to capture non-linear tail concentration and log-odds separation.
2. Kolmogorov-Smirnov (KS) Statistic & Univariate PR-AUC: Measures maximum distributional separation and standalone precision-recall ranking power, evaluated bidirectionally.
3. Out-of-Fold PR-AUC Permutation Importance: Uses stratified cross-validation with a shallow, regularized gradient-boosted model (LightGBM), scored strictly on `average_precision` (PR-AUC) rather than ROC-AUC or accuracy.
4. Positive-Class SHAP Attribution (Mean `|SHAP|` on `Y=1`): Evaluates validation-fold SHAP values strictly on minority-class rows (`y == 1`) so majority-class baseline variance does not drown out rare-event drivers.

### 2.3 Guardrails for 1–2% Regimes

- Out-of-fold validation only: All tree-based metrics (permutation importance and SHAP) must be computed strictly on validation folds to prevent memorization of individual minority samples.
- Minimum leaf & bin support: Enforce minimum sample thresholds (`min_child_samples >= 0.5% of rows`, `min_bin_size >= 2%`) to prevent single noisy outliers from inflating scores.
- Automated target leakage flagging: Automatically flag features with `iv > 1.5` or `uni_pr_auc > 0.60` in `self.report_` with `leakage_flag = True` for manual inspection.

## 3. Visualization & Diagnostic Plot Requirements

### 3.1 Multi-Metric Leaderboard (`plot_summary(top_n=15)`)

Purpose: Compare univariate versus multivariate importance side-by-side to identify interaction-only features (low IV, high PR-AUC drop / positive SHAP).

Layout: 3-panel horizontal bar chart sharing the y-axis, sorted by `consensus_rank`:

- Panel 1: Information Value (`iv`) with vertical threshold reference lines at `0.1` (weak) and `0.3` (strong).
- Panel 2: Out-of-Fold PR-AUC Permutation Drop (`perm_pr_auc`).
- Panel 3: Mean `|SHAP|` evaluated strictly on `Y=1` (`pos_shap`).

Annotations: Display the overall dataset base event rate in the figure title.

### 3.2 Single-Feature Diagnostic Profile (`plot_feature(X, y, feature, log_x=False)`)

Purpose: Deep-dive inspection of tail separation and lift concentration for any candidate feature.

Layout: 2-panel side-by-side diagnostic figure:

- Left panel — Distributional separation:
  - Numeric features: Overlay empirical cumulative distribution functions (ECDFs) for `Y=0` (majority) versus `Y=1` (minority), supporting optional `log_x=True` scaling for heavy-tailed distributions and annotated with the KS statistic.
  - Categorical features: Grouped horizontal bar chart comparing class percentage distributions across the top 10 categories.
- Right panel — Optimal bins & event-rate lift:
  - Dual-axis chart built from the cached `OptimalBinning` table.
  - Primary y-axis (bars): Sample count per bin to verify statistical support.
  - Secondary y-axis (line + markers): Positive event rate (%) per bin alongside a dashed horizontal reference line marking the dataset's base event rate.

## 4. Technical Stack & Dependencies

| Package | Version constraint | Role in wrapper |
| --- | --- | --- |
| `numpy` | `>=1.23` | Array operations and ECDF calculations |
| `pandas` | `>=1.5` | DataFrame handling and diagnostic report generation |
| `scikit-learn` | `>=1.2` | `BaseEstimator`, `TransformerMixin`, `StratifiedKFold`, `permutation_importance`, `average_precision_score` |
| `lightgbm` | `>=4.0` | Shallow, regularized GBDT for out-of-fold interaction screening |
| `optbinning` | `>=0.19` | Constrained optimal binning, WoE, and Information Value (IV) |
| `shap` | `>=0.43` | `TreeExplainer` for minority-filtered (`Y=1`) feature attributions |
| `scipy` | `>=1.10` | `ks_2samp` distributional distance testing |
| `matplotlib` | `>=3.7` | Rendering `plot_summary()` and `plot_feature()` diagnostics |

## 5. Output Schema (`self.report_`)

| Column | Type | Description |
| --- | --- | --- |
| `feature` | `str` | Column name |
| `dtype` | `str` | `'numerical'` or `'categorical'` |
| `missing_rate` | `float` | Proportion of NaN values (0.0 to 1.0) |
| `iv` | `float` | Total Information Value from optimal binning |
| `ks_stat` | `float` | Two-sample KS statistic between `Y=1` and `Y=0` |
| `uni_pr_auc` | `float` | Best single-feature PR-AUC (forward or reverse direction) |
| `perm_pr_auc` | `float` | Mean out-of-fold PR-AUC drop upon feature permutation |
| `pos_shap` | `float` | Mean absolute SHAP value across `Y=1` validation samples |
| `consensus_rank` | `float` | Mean rank across `iv`, `uni_pr_auc`, `perm_pr_auc`, and `pos_shap` |
| `leakage_flag` | `bool` | `True` if `iv > 1.5` or `uni_pr_auc > 0.60` |

## Optional Next Step

A complete, copy-pasteable Python module (`screener.py`) can be generated alongside a runnable synthetic 1.5% imbalance demo script.
