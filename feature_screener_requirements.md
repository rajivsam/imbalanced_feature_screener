# Imbalanced Feature Discovery Wrapper — Updated Requirements & Specification (v2.0)

## 1. Overview & Objective

A reusable, `scikit-learn`-compatible exploratory wrapper (`ImbalancedFeatureScreener`) designed to discover discriminative features, estimate an optimal feature budget, prune noise and collinear twins, and visually diagnose tail separation in highly imbalanced binary classification problems (**1–2% positive class**, near-anomaly regime).
At 1–2% prevalence, effective sample size is constrained by the **minority class count (\(N_{\text{pos}}\))** rather than total rows. This wrapper prevents the curse of dimensionality by combining multi-lens signal scoring, an **Events-Per-Variable (EPV) budget**, and **configurable collinearity pruning**.

---

## 2. Functional Requirements

### 2.1 Core API (`BaseEstimator`, `TransformerMixin`)

- **`.fit(X, y)`**: Executes univariate tail screening, out-of-fold multivariate evaluation, consensus ranking, tier classification, configurable collinearity deduplication, and EPV budget selection. Populates `self.report_`, `self.selected_features_`, `self.dropped_collinear_`, and `self.binning_tables_`.
- **`.transform(X)`**: Filters input DataFrames down to `self.selected_features_` for seamless pipeline integration.
- **`.summary()`**: Prints an executive breakdown of sample prevalence, EPV feature budget, tier counts, and features demoted due to collinearity.

### 2.2 Multi-Lens Scoring Engine

Every feature is evaluated across four complementary metrics:

1. **Information Value (IV)**: Computed via `optbinning.OptimalBinning` (`min_bin_size=0.02`) to capture non-linear power-law tails and log-odds separation.
2. **Kolmogorov-Smirnov (KS) Statistic & Univariate PR-AUC**: Measures maximum ECDF vertical separation (`[0, 1]`) and bidirectional single-feature precision-recall ranking power.
3. **Out-of-Fold PR-AUC Permutation Importance (`perm_pr_auc`)**: Uses stratified cross-validation with a shallow, regularized `LightGBM` model scored strictly on `average_precision` (PR-AUC). Captures conditional interactions while flagging noisy/overfit splits (`perm_pr_auc < 0`).
4. **Positive-Class SHAP Attribution (`pos_shap`)**: Evaluates validation-fold `TreeExplainer` SHAP values strictly on minority rows (`y == 1`) so majority-class baseline variance does not obscure rare-event drivers.

### 2.3 Automated Tiering & EPV Feature Budgeting

Features are classified into five mutually exclusive tiers to establish a transparent pruning rubric:

- **`Leakage Check`**: `iv > leakage_iv_thresh` (default `1.50`) or `uni_pr_auc > 0.60`. Excluded by default (`include_leakage=False`) until manually audited.
- **`Anchor`**: Strong standalone predictors (`iv >= 0.10`, `ks_stat >= 0.15`, or `uni_pr_auc >= 3 * base_rate`).
- **`Interaction`**: Low univariate power, but proven multivariate lift (`perm_pr_auc >= 0.002` and top-quartile `pos_shap`).
- **`Weak / Redundant`**: Marginal positive signal or features demoted by collinearity pruning.
- **`Noise / Drop`**: Fails all minimum thresholds (`iv < 0.02`, `ks_stat < 0.08`, `perm_pr_auc <= 0.0005`) or exhibits negative validation lift (`perm_pr_auc < -0.005`).
**EPV Budget Enforcement**:

- Automatically computes the maximum recommended feature ceiling:
\(k_{\max }=\min \left(\text{max\_features},\;\max \left(5,\;\left\lfloor \frac{N_{\text{pos}}}{\text{min\_epv}}\right\rfloor \right)\right)\)
- Prioritizes `Anchor` and `Interaction` features first, filling any remaining budget slots with positive-lift `Weak / Redundant` features.

### 2.4 Configurable Collinearity Pruning (New in v2.0)

To avoid wasting scarce EPV budget slots on redundant twins (e.g., `amount_usd` vs. `amount_eur` or rolling 7d vs. 14d sums), the wrapper includes a greedy rank-ordered deduplication pass:

- **Configuration Parameters**:
- `prune_collinear: bool = True` — Master toggle to enable/disable collinearity pruning.
- `collinear_thresh: float = 0.85` — Absolute correlation threshold (`[0.0, 1.0]`) above which the lower-ranked feature is demoted.
- `collinear_method: Literal['spearman', 'pearson'] = 'spearman'` — Correlation metric (`spearman` recommended as default to catch monotonic power-law duplicates).
- **Deduplication Mechanics**:
- Iterates through features sorted by `consensus_rank` (highest signal first).
- For numerical pairs exceeding `|corr| >= collinear_thresh`, the higher-ranked feature is retained and the lower-ranked twin is demoted to `'Weak / Redundant'` with its `collinear_with` parent recorded in `self.report_`.
- Ensures distinct signal diversity within the final `selected_features_` set.

---

## 3. Visualization & Diagnostic Plot Requirements

### 3.1 Multi-Metric Leaderboard (`plot_summary(top_n=None)`)

- **Layout**: 3-panel horizontal bar chart (`iv`, `perm_pr_auc`, `pos_shap`) sorted by `consensus_rank`.
- **Visual Encodings**: Bars color-coded by assigned `tier`; horizontal dashed cutoff line marking the exact **EPV feature budget** boundary (`len(selected_features_)`).

### 3.2 Single-Feature Diagnostic Profile (`plot_feature(X, y, feature, log_x=False)`)

- **Left Panel (Distributional Separation)**: Side-by-side ECDF curves for `Y=0` vs. `Y=1` (with optional `log_x=True` for heavy-tailed/power-law features) annotated with the KS statistic (or top-10 category class breakdown for categoricals).
- **Right Panel (Optimal Bins & Lift)**: Dual-axis chart showing bin sample counts (bars) vs. positive event rate `%` (line) against the dashed dataset base rate reference line.

---

## 4. Updated Output Schema (`self.report_`)

| Column | Type | Description |
| --- | --- | --- |
| `feature` | `str` | Column name |
| `dtype` | `str` | `'numerical'` or `'categorical'` |
| `missing_rate` | `float` | Proportion of `NaN` values |
| `iv` | `float` | Total Information Value from constrained optimal binning |
| `ks_stat` | `float` | Two-sample KS statistic between `Y=1` and `Y=0` |
| `uni_pr_auc` | `float` | Best bidirectional single-feature PR-AUC |
| `perm_pr_auc` | `float` | Mean out-of-fold PR-AUC drop upon column permutation |
| `pos_shap` | `float` | Mean absolute SHAP value on the positive class |
| `consensus_rank` | `float` | Mean rank across the four core metrics |
| `tier` | `str` | Assigned category (`Anchor`, `Interaction`, `Weak / Redundant`, `Noise / Drop`, `Leakage Check`) |
| `collinear_with` | `str | None` | Name of higher-ranked feature that caused demotion (if `prune_collinear=True`), else `None` |
| `selected` | `bool` | `True` if retained in `self.selected_features_` within the EPV budget |

---

## 5. Configuration Parameters Reference

| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `min_epv` | `15` | `int` | Minimum positive events (`Y=1`) required per selected feature |
| `max_features` | `None` | `int | None` | Optional hard upper cap overriding EPV budget |
| `prune_collinear` | `True` | `bool` | Enable or disable rank-ordered collinearity demotion |
| `collinear_thresh` | `0.85` | `float` | Absolute correlation cutoff for demoting redundant features |
| `collinear_method` | `'spearman'` | `str` | Correlation method (`'spearman'` or `'pearson'`) |
| `iv_anchor_thresh` | `0.10` | `float` | Minimum IV required to qualify as an `Anchor` |
| `ks_anchor_thresh` | `0.15` | `float` | Minimum KS statistic required to qualify as an `Anchor` |
| `perm_lift_thresh` | `0.002` | `float` | Minimum PR-AUC permutation lift to qualify as `Interaction` |
| `leakage_iv_thresh` | `1.50` | `float` | IV threshold above which features are flagged for leakage |
| `include_leakage` | `False` | `bool` | Whether `Leakage Check` features are allowed into `selected_features_` |
| `n_splits` | `3` | `int` | Stratified K-Fold splits for out-of-fold LightGBM & SHAP evaluation |
| `random_state` | `42` | `int` | Random seed for reproducibility |

---

## 6. Technical Stack & Dependencies

| Package | Version constraint | Role in wrapper |
| --- | --- | --- |
| `numpy` | `>=1.23` | Array operations and ECDF calculations |
| `pandas` | `>=1.5` | DataFrame handling and diagnostic report generation |
| `scikit-learn` | `>=1.2` | `BaseEstimator`, `TransformerMixin`, `StratifiedKFold`, `average_precision_score` |
| `lightgbm` | `>=4.0` | Shallow, regularized GBDT for out-of-fold interaction screening |
| `optbinning` | `>=0.19` | Constrained optimal binning, WoE, and Information Value (IV) |
| `shap` | `>=0.43` | `TreeExplainer` for minority-filtered (`Y=1`) feature attributions |
| `scipy` | `>=1.10` | `ks_2samp` distributional distance testing |
| `matplotlib` | `>=3.7` | Rendering `plot_summary()` and `plot_feature()` diagnostics |

---

## Optional Next Step

A complete, copy-pasteable Python module (`screener.py`) can be generated alongside a runnable synthetic 1.5% imbalance demo script.
