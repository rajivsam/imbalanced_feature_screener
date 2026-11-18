# Imbalanced Feature Screener — workspace bootstrap

This repository is a Python project for screening features in highly imbalanced binary classification problems, with SBA loan distress as the sample use case.

## Project overview
- Package: `imb_feature_screener`
- Source tree: `src/imb_feature_screener/`
- Main implementation: `src/imb_feature_screener/feature_screener.py`
- Entrypoint: `src/imb_feature_screener/__main__.py`
- Evaluation script: `evaluate_feature_screener.py`
- Notebook work: `notebooks/`
- Data: `data/`
- Requirements update: `feature_screener_requirements.md` (v2.0, EPV feature budgeting + collinearity pruning)

## Updated requirements summary (v2.0)
This repo is aligned to the v2.0 design for a scikit-learn-compatible rare-event screener:

- Uses a multi-lens scoring model across IV, KS, univariate PR-AUC, permutation PR-AUC lift, and positive-class SHAP
- Applies tier assignments: `Anchor`, `Interaction`, `Weak / Redundant`, `Noise / Drop`, and `Leakage Check`
- Enforces an EPV-based feature budget using the minority-class count and `min_epv` with an optional `max_features` cap
- Performs greedy rank-ordered collinearity pruning via `prune_collinear`, `collinear_thresh`, and `collinear_method`
- Produces a richer `self.report_` schema with `tier`, `collinear_with`, `selected`, and the expected feature metrics
- Keeps the notebook workflow focused on displaying plot outputs inline for Jupyter/IPython sessions
- Includes a notebook-ready plotting workflow using `plot_summary(...)` and `plot_feature(...)` for visual inspection of ranked features and rare-event tail separation

The formal specification lives in `feature_screener_requirements.md`; keep implementation edits aligned with that source document.

## Current implementation checkpoints
- `ImbalancedFeatureScreener.fit(X, y)` populates `self.report_`, `self.selected_features_`, `self.dropped_collinear_`, and `self.binning_tables_`
- `ImbalancedFeatureScreener.transform(X)` returns only the retained feature subset selected under the EPV budget
- `ImbalancedFeatureScreener.summary()` prints the positive rate, EPV cap, tier counts, and collinearity demotions
- `plot_summary(top_n=...)` renders a ranked multi-metric leaderboard with the selected-cap boundary and tier colors
- `plot_feature(X, y, feature, log_x=False)` renders ECDF separation and bin-lift diagnostics for a chosen feature

## Startup validation sequence
When re-entering the repo, use this quick correctness check before broad experimentation:

```bash
cd /home/rajiv/programming/imbalanced_feature_screener
./imb_feature_screener/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/feature_screener_sba.ipynb
```

This confirms the package still fits, selects features under the EPV budget, and produces notebook-visible plots.

### Data leakage guard used in the active notebook
The notebook workflow explicitly drops leakage fields before modeling:

```python
leakage_columns = [
    'loanstatus',
    'paidinfulldate',
    'chargeoffdate',
    'grosschargeoffamount',
]
df = df.drop(columns=[c for c in leakage_columns if c in df.columns], errors='ignore')
df = df[df['loan_status_r'] != -1].copy()
df['target'] = df['loan_status_r'].astype(int)
```

This is required because these columns encode realized status or loss outcome information and would leak the target signal into the screening pipeline.

## Fast bootstrap
From the repo root:

```bash
cd /home/rajiv/programming/imbalanced_feature_screener
uv sync
```

Optional local environment activation:

```bash
source imb_feature_screener/bin/activate
```

## Core commands
Run the bundled evaluation:

```bash
uv run python evaluate_feature_screener.py
```

Run the package entrypoint:

```bash
uv run python -m imb_feature_screener
```

Or use the installed script name:

```bash
uv run imb-feature-screener
```

Start Jupyter for notebooks:

```bash
./imb_feature_screener/bin/python -m jupyter lab
```

Notebook execution check:

```bash
./imb_feature_screener/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/feature_screener_sba.ipynb
```

## Package usage
```python
import pandas as pd
from imb_feature_screener import ImbalancedFeatureScreener

# data-leakage guard: remove post-outcome fields before fitting
leakage_columns = [
    "loanstatus",
    "paidinfulldate",
    "chargeoffdate",
    "grosschargeoffamount",
]
X = df.drop(columns=["target", *leakage_columns], errors="ignore")
y = df["target"].astype(int)

screener = ImbalancedFeatureScreener(
    top_k=10,
    min_epv=15,
    prune_collinear=True,
    collinear_thresh=0.85,
    collinear_method="spearman",
)
screener.fit(X, y)
print(screener.report_.head())
print(screener.selected_features_)
screener.summary()
```

## Data + notebook context
- Notebook examples live in `notebooks/`
- The main analysis dataset is `data/sba_full_attrib_clean.csv`
- The screener workflow uses a cleaned binary target built from `loan_status_r` after filtering out `loan_status_r == -1`
- Before modeling, we explicitly drop leakage columns: `loanstatus`, `paidinfulldate`, `chargeoffdate`, and `grosschargeoffamount`
- These are outcome/post-outcome fields and would leak distress information into the feature set if left in the analysis
- The evaluation script reads the default dataset path from the repo root and expects a `target` column
- The reference notebook is `notebooks/feature_screener_sba.ipynb`

## Dependencies and environment
- Python requirement: `>=3.12`
- Dependency manager: `uv`
- Key libraries: `pandas`, `numpy`, `scikit-learn`, `shap`, `lightgbm`, `optbinning`, `scipy`, `matplotlib`

## Working conventions for this repo
- Prefer `uv run ...` for commands to ensure the project environment is used consistently.
- Keep changes aligned with the `src/` package layout rather than ad hoc scripts.
- The feature screener is designed for rare-event / imbalanced classification problems and outputs ranked feature importance metrics.
- Use the notebook in `notebooks/feature_screener_sba.ipynb` as the reference workflow when validating feature screening behavior.
- The screener is expected to expose `report_`, `selected_features_`, `dropped_collinear_`, and `binning_tables_` after `fit()`.

## Helpful reminders for future startup
- If imports fail, re-run `uv sync`.
- If the notebook kernel is missing, install it with:

```bash
./imb_feature_screener/bin/python -m ipykernel install --user --name imb-feature-screener
```

- When in doubt, validate via the notebook and/or the evaluation script before broader experiments.
- For rare-event datasets, watch EPV budget and collinearity pruning before trusting a large selected feature set.
