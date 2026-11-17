from __future__ import annotations

import pandas as pd

try:
    from imb_feature_screener import ImbalancedFeatureScreener
except ImportError:  # pragma: no cover - compatibility for direct repo execution
    from feature_screener import ImbalancedFeatureScreener


def main() -> None:
    data_path = "/home/rajiv/programming/imbalanced_feature_screener/data/sba_loans_imb_learn.csv"
    target_col = "target"

    df = pd.read_csv(data_path, low_memory=False)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    screener = ImbalancedFeatureScreener(top_k=10)
    screener.fit(X, y)

    print(screener.report_.head(10)[["feature", "iv", "ks_stat", "uni_pr_auc", "perm_pr_auc", "pos_shap", "consensus_rank"]].to_string(index=False))
    print(f"\nSelected features: {screener.selected_features_}")


if __name__ == "__main__":
    main()
