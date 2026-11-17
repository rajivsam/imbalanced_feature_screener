# Imbalanced Feature Screener


## Core value proposition

The core value proposition of the package is this: you have a rare-event dataset and you want a principled breakdown of the discriminative features with respect to the rare event. This is a common problem in enterprise data, including fraud detection, credit screening, and similar risk and anomaly settings. The screener is designed to be an exploratory and descriptive tool that ranks features by how strongly they distinguish the rare event from the majority class, while also surfacing the most informative, non-leaky signals in a way that remains interpretable and operationally useful.

A scikit-learn-compatible feature screening package for highly imbalanced binary classification problems, tuned for rare-event detection problems such as SBA loan distress.

## Installation

This project uses uv for project and dependency management:

```bash
uv sync
```

After installation, the package is available as:

```bash
uv run python -c "from imb_feature_screener import ImbalancedFeatureScreener; print(ImbalancedFeatureScreener)"
```

## Usage

```python
import pandas as pd
from imb_feature_screener import ImbalancedFeatureScreener

X = df.drop(columns=["target"])
y = df["target"].astype(int)

screener = ImbalancedFeatureScreener(top_k=10)
screener.fit(X, y)
print(screener.report_.head())
print(screener.selected_features_)
```

## Evaluation

```bash
uv run python evaluate_feature_screener.py
```

The screener ranks features using a combined score across information value, KS, univariate PR-AUC, permutation importance, and positive-class SHAP attribution.
