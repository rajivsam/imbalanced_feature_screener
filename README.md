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

### Categorical encoding options

Categorical columns are encoded with leakage-safe supervised encoders rather than arbitrary category codes. The default is `m_estimate`, which is a conservative choice for rare-event screening.

```python
from imb_feature_screener import ImbalancedFeatureScreener

# default: balanced M-estimation target encoding
screener_default = ImbalancedFeatureScreener(
    top_k=10,
    categorical_encoding="m_estimate",
    categorical_encoder_params={"m": 10.0, "sigma": 0.05},
)

# simpler smoothed target encoding
screener_target = ImbalancedFeatureScreener(
    top_k=10,
    categorical_encoding="target",
    categorical_encoder_params={"smoothing": 10.0, "min_samples_leaf": 20},
)

# advanced option: CatBoost-style regularized encoding
screener_catboost = ImbalancedFeatureScreener(
    top_k=10,
    categorical_encoding="catboost",
    categorical_encoder_params={"a": 1, "sigma": None},
)

# if you want to bypass the encoder and use the legacy category-code fallback
screener_raw = ImbalancedFeatureScreener(
    top_k=10,
    categorical_encoding="none",
)
```

The screener fits these encoders out-of-fold within each evaluation fold so category target means do not leak information from validation rows into the screening metrics.

## Evaluation

```bash
uv run python evaluate_feature_screener.py
```

The screener ranks features using a combined score across information value, KS, univariate PR-AUC, permutation importance, and positive-class SHAP attribution.
