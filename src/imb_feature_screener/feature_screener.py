from __future__ import annotations

from typing import Iterable

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=UserWarning, module="shap.explainers._tree")
from lightgbm import LGBMClassifier
from optbinning import OptimalBinning
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


class ImbalancedFeatureScreener(BaseEstimator, TransformerMixin):
    """Lightweight feature screener tuned for highly imbalanced binary problems."""

    def __init__(
        self,
        top_k: int = 15,
        random_state: int = 42,
        n_splits: int = 3,
        min_samples: int = 10,
        use_lightgbm: bool = True,
    ):
        self.top_k = top_k
        self.random_state = random_state
        self.n_splits = n_splits
        self.min_samples = min_samples
        self.use_lightgbm = use_lightgbm
        self.report_ = None
        self.selected_features_ = []
        self.feature_names_in_ = []

    def fit(self, X: pd.DataFrame, y: Iterable[int | float]):
        X = self._as_dataframe(X)
        y = pd.Series(y).reset_index(drop=True)
        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")

        if y.nunique() < 2:
            raise ValueError("Target must contain at least two classes.")

        self.feature_names_in_ = list(X.columns)
        processed = self._preprocess_features(X)

        rows = []
        for feature in processed.columns:
            values = processed[feature].to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue

            iv_score = self._information_value(values, y.to_numpy())
            ks_score = self._ks_statistic(values, y.to_numpy())
            pr_auc = self._univariate_pr_auc(values, y.to_numpy())
            perm_drop = self._permutation_ap_drop(processed, y.to_numpy(), feature)
            shap_score = self._positive_class_shap(processed, y.to_numpy(), feature)
            missing_rate = float(pd.isna(X[feature]).mean()) if feature in X.columns else 0.0

            rows.append(
                {
                    "feature": feature,
                    "dtype": self._infer_dtype(X[feature]),
                    "missing_rate": missing_rate,
                    "iv": iv_score,
                    "ks_stat": ks_score,
                    "uni_pr_auc": pr_auc,
                    "perm_pr_auc": perm_drop,
                    "pos_shap": shap_score,
                    "consensus_rank": float("nan"),
                    "leakage_flag": bool(iv_score > 1.5 or pr_auc > 0.60),
                }
            )

        self.report_ = pd.DataFrame(rows)
        if self.report_.empty:
            self.selected_features_ = []
            return self

        self.report_["consensus_rank"] = self._consensus_ranks(self.report_)
        self.report_ = self.report_.sort_values(["consensus_rank", "iv"], ascending=[True, False]).reset_index(drop=True)
        self.selected_features_ = self.report_.head(self.top_k)["feature"].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._as_dataframe(X)
        if self.report_ is None:
            raise ValueError("The screener has not been fitted yet.")
        missing = [col for col in self.selected_features_ if col not in X.columns]
        if missing:
            raise ValueError(f"Columns missing in transform input: {missing}")
        return X[self.selected_features_].copy()

    def plot_summary(self, top_n: int = 15):
        if self.report_ is None:
            raise ValueError("The screener has not been fitted yet.")
        summary = self.report_.head(top_n).copy()
        if summary.empty:
            raise ValueError("No features available to plot.")

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        x_labels = summary["feature"].tolist()
        axes[0].bar(x_labels, summary["iv"], color="steelblue")
        axes[0].axhline(0.1, color="darkorange", linestyle="--", linewidth=1)
        axes[0].axhline(0.3, color="crimson", linestyle="--", linewidth=1)
        axes[0].set_title("Information Value")

        axes[1].bar(x_labels, summary["perm_pr_auc"], color="seagreen")
        axes[1].set_title("Permutation PR-AUC Drop")

        axes[2].bar(x_labels, summary["pos_shap"], color="purple")
        axes[2].set_title("Mean |SHAP| on positive class")

        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    def plot_feature(self, X: pd.DataFrame, y: Iterable[int | float], feature: str, log_x: bool = False):
        X = self._as_dataframe(X)
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in X.")
        y = pd.Series(y).reset_index(drop=True)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        x_values = X[feature].to_numpy(dtype=float)
        y_values = y.to_numpy()
        pos = x_values[y_values == 1]
        neg = x_values[y_values == 0]

        if pd.api.types.is_numeric_dtype(X[feature]):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            if log_x:
                pos = np.log1p(np.abs(pos))
                neg = np.log1p(np.abs(neg))
            for label, values in [("Y=0", neg), ("Y=1", pos)]:
                vals = np.sort(values)
                if vals.size == 0:
                    continue
                ecdf = np.arange(1, vals.size + 1) / vals.size
                axes[0].plot(vals, ecdf, label=label)
            axes[0].set_title(f"ECDF for {feature}")
            axes[0].legend()

            all_vals = np.r_[pos, neg]
            if np.all(np.isclose(all_vals, all_vals[0])):
                axes[1].bar([0], [1], color="grey")
                axes[1].set_title(f"Positive rate by bin for {feature}")
                fig.tight_layout()
                return fig
            bins = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), 10)
            event_rate = []
            for left, right in zip(bins[:-1], bins[1:]):
                mask = (x_values >= left) & (x_values < right)
                if mask.sum() == 0:
                    event_rate.append(np.nan)
                else:
                    event_rate.append(float(y_values[mask].mean()))
            axes[1].plot(np.arange(len(event_rate)), event_rate, marker="o")
            axes[1].axhline(y_values.mean(), color="red", linestyle="--")
            axes[1].set_title(f"Positive rate by bin for {feature}")
            fig.tight_layout()
            return fig

        fig, ax = plt.subplots(figsize=(8, 4))
        counts = X[feature].astype(str).fillna("MISSING")
        by_label = pd.crosstab(counts, y_values)
        by_label.plot(kind="bar", ax=ax)
        ax.set_title(f"Category composition for {feature}")
        return fig

    def _as_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.copy()

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        processed = X.copy()
        for col in processed.columns:
            s = processed[col]
            if pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors="coerce")
                med = s.median()
                if pd.isna(med):
                    med = 0.0
                processed[col] = s.fillna(med)
            else:
                s = s.astype("string").fillna("MISSING")
                processed[col] = s.astype("category").cat.codes.astype(float)
        return processed

    def _infer_dtype(self, s: pd.Series) -> str:
        return "numerical" if pd.api.types.is_numeric_dtype(s) else "categorical"

    def _information_value(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        valid = np.isfinite(x)
        x, y = x[valid], y[valid]
        if len(x) == 0 or np.unique(x).size < 2:
            return 0.0

        try:
            min_bin_size = max(0.02, self.min_samples / max(len(x), 1))
            model = OptimalBinning(name="feature", dtype="numerical", min_bin_size=min_bin_size)
            model.fit(x, y)
            return float(model.iv)
        except Exception:
            bins = np.quantile(x, np.linspace(0, 1, 5))
            bins = np.unique(np.clip(bins, np.nanmin(x), np.nanmax(x)))
            if bins.size < 2:
                return 0.0
            binned = np.digitize(x, bins[1:-1], right=False)
            table = pd.DataFrame({"bin": binned, "target": y})
            total_pos = float(y.mean())
            total_neg = 1.0 - total_pos
            iv = 0.0
            for _, group in table.groupby("bin"):
                pos_rate = float(group["target"].mean())
                neg_rate_group = 1.0 - pos_rate
                if pos_rate <= 0 or neg_rate_group <= 0:
                    continue
                woe = np.log((pos_rate / max(total_pos, 1e-9)) / (neg_rate_group / max(total_neg, 1e-9)))
                iv += (pos_rate - total_pos) * woe
            return float(iv)

    def _ks_statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        valid = np.isfinite(x)
        x = x[valid]
        y = y[valid]
        if len(x) == 0 or np.unique(x).size < 2:
            return 0.0
        pos = x[y == 1]
        neg = x[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        return float(ks_2samp(pos, neg).statistic)

    def _univariate_pr_auc(self, x: np.ndarray, y: np.ndarray) -> float:
        valid = np.isfinite(x)
        x = x[valid]
        y = y[valid]
        if len(np.unique(x)) < 2:
            return 0.0
        return float(average_precision_score(y, x))

    def _permutation_ap_drop(self, X: pd.DataFrame, y: np.ndarray, feature: str) -> float:
        if X.shape[1] < 2:
            return 0.0
        if min(y.sum(), len(y) - y.sum()) < 2:
            return 0.0
        cv = StratifiedKFold(n_splits=min(self.n_splits, min(5, min(y.sum(), len(y) - y.sum()))))
        drops = []
        rng = np.random.default_rng(self.random_state)
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = LGBMClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                is_unbalance=True,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=1,
            )
            model.fit(X_train, y_train)
            base_prob = model.predict_proba(X_val)[:, 1]
            base_ap = average_precision_score(y_val, base_prob)

            X_perm = X_val.copy()
            X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
            perm_prob = model.predict_proba(X_perm)[:, 1]
            perm_ap = average_precision_score(y_val, perm_prob)
            drops.append(max(0.0, base_ap - perm_ap))

        return float(np.mean(drops)) if drops else 0.0

    def _positive_class_shap(self, X: pd.DataFrame, y: np.ndarray, feature: str) -> float:
        if X.shape[1] < 2:
            return 0.0
        if ((y == 1).sum() == 0) or ((y == 0).sum() == 0):
            return 0.0

        try:
            model = LGBMClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                is_unbalance=True,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=1,
            )
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            if hasattr(values, "ndim") and values.ndim == 3:
                values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
            feature_idx = X.columns.get_loc(feature)
            return float(np.abs(values[:, feature_idx]).mean())
        except Exception:
            return 0.0

    def _consensus_ranks(self, report: pd.DataFrame) -> np.ndarray:
        ranks = pd.DataFrame(index=report.index)
        ranks["iv"] = report["iv"].rank(method="average", ascending=False)
        ranks["ks_stat"] = report["ks_stat"].rank(method="average", ascending=False)
        ranks["uni_pr_auc"] = report["uni_pr_auc"].rank(method="average", ascending=False)
        ranks["perm_pr_auc"] = report["perm_pr_auc"].rank(method="average", ascending=False)
        ranks["pos_shap"] = report["pos_shap"].rank(method="average", ascending=False)
        return ranks.mean(axis=1).to_numpy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the imbalanced feature screener on a binary classification dataset.")
    parser.add_argument("--data", default="/home/rajiv/programming/imbalanced_feature_screener/data/sba_loans_imb_learn.csv", help="CSV file containing the target column.")
    parser.add_argument("--target", default="target", help="Target column name.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top features to retain.")
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' is missing from {args.data}")

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    screener = ImbalancedFeatureScreener(top_k=args.top_k)
    screener.fit(X, y)

    print(screener.report_.head(args.top_k)[["feature", "iv", "ks_stat", "uni_pr_auc", "perm_pr_auc", "pos_shap", "consensus_rank"]].to_string(index=False))
    print(f"\nSelected features: {screener.selected_features_}")


if __name__ == "__main__":
    main()
