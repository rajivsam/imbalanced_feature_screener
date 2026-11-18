from __future__ import annotations

import math
import os
import sys
import warnings
from typing import Iterable

import matplotlib

if "ipykernel" in sys.modules:
    matplotlib.use("module://matplotlib_inline.backend_inline")
elif os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from category_encoders import CatBoostEncoder, MEstimateEncoder, TargetEncoder
from lightgbm import LGBMClassifier
from optbinning import OptimalBinning
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning, module="shap.explainers._tree")


class ImbalancedFeatureScreener(BaseEstimator, TransformerMixin):
    """Lightweight feature screener tuned for highly imbalanced binary problems."""

    def __init__(
        self,
        top_k: int = 15,
        random_state: int = 42,
        n_splits: int = 3,
        min_samples: int = 10,
        use_lightgbm: bool = True,
        min_epv: int = 15,
        max_features: int | None = None,
        prune_collinear: bool = True,
        collinear_thresh: float = 0.85,
        collinear_method: str = "spearman",
        iv_anchor_thresh: float = 0.10,
        ks_anchor_thresh: float = 0.15,
        perm_lift_thresh: float = 0.002,
        leakage_iv_thresh: float = 1.50,
        include_leakage: bool = False,
        categorical_encoding: str = "m_estimate",
        categorical_encoder_params: dict | None = None,
        categorical_columns: list[str] | None = None,
    ):
        self.top_k = top_k
        self.random_state = random_state
        self.n_splits = n_splits
        self.min_samples = min_samples
        self.use_lightgbm = use_lightgbm
        self.min_epv = min_epv
        self.max_features = max_features
        self.prune_collinear = prune_collinear
        self.collinear_thresh = collinear_thresh
        self.collinear_method = collinear_method
        self.iv_anchor_thresh = iv_anchor_thresh
        self.ks_anchor_thresh = ks_anchor_thresh
        self.perm_lift_thresh = perm_lift_thresh
        self.leakage_iv_thresh = leakage_iv_thresh
        self.include_leakage = include_leakage
        self.categorical_encoding = categorical_encoding
        self.categorical_encoder_params = categorical_encoder_params or {}
        self.categorical_columns = categorical_columns

        self.report_ = None
        self.selected_features_ = []
        self.dropped_collinear_ = []
        self.binning_tables_ = {}
        self.feature_names_in_ = []
        self.n_pos_ = 0
        self.base_rate_ = 0.0
        self.categorical_encoders_ = {}
        self.categorical_columns_ = []

    def fit(self, X: pd.DataFrame, y: Iterable[int | float]):
        X = self._as_dataframe(X)
        y = pd.Series(y).reset_index(drop=True)
        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")
        if y.nunique() < 2:
            raise ValueError("Target must contain at least two classes.")

        self.feature_names_in_ = list(X.columns)
        self.n_pos_ = int(y.eq(1).sum())
        self.base_rate_ = float(y.mean())
        self.categorical_columns_ = self._resolve_categorical_columns(X)
        self.categorical_encoders_ = {}
        processed = self._preprocess_features(X, y)
        self.dropped_collinear_ = []
        self.binning_tables_ = {}

        rows = []
        for feature in processed.columns:
            values = processed[feature].to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue

            missing_rate = float(pd.isna(X[feature]).mean()) if feature in X.columns else 0.0
            iv_score = self._information_value(values, y.to_numpy())
            ks_score = self._ks_statistic(values, y.to_numpy())
            pr_auc = self._univariate_pr_auc(values, y.to_numpy())
            perm_drop = self._permutation_ap_drop(processed, y.to_numpy(), feature)
            shap_score = self._positive_class_shap(processed, y.to_numpy(), feature)

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
                    "tier": "Noise / Drop",
                    "collinear_with": None,
                    "selected": False,
                    "leakage_flag": bool(iv_score > self.leakage_iv_thresh or pr_auc > 0.60),
                }
            )

        self.report_ = pd.DataFrame(rows)
        if self.report_.empty:
            self.selected_features_ = []
            self.report_ = self.report_.copy()
            return self

        self.report_["consensus_rank"] = self._consensus_ranks(self.report_)
        self.report_ = self.report_.sort_values(["consensus_rank", "iv"], ascending=[True, False]).reset_index(drop=True)
        self._assign_tiers(y.to_numpy())
        self._prune_collinear(processed)
        self._apply_epv_budget(y.to_numpy())
        if self.categorical_encoding != "none":
            self._fit_categorical_encoders(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._as_dataframe(X)
        if self.report_ is None:
            raise ValueError("The screener has not been fitted yet.")
        missing = [col for col in self.selected_features_ if col not in X.columns]
        if missing:
            raise ValueError(f"Columns missing in transform input: {missing}")

        transformed = X[self.selected_features_].copy()
        if self.categorical_encoding != "none":
            for col in self.selected_features_:
                if col in self.categorical_encoders_:
                    encoded = self.categorical_encoders_[col].transform(X[[col]])
                    if isinstance(encoded, pd.DataFrame):
                        transformed[col] = encoded.iloc[:, 0].astype(float)
                    else:
                        transformed[col] = pd.Series(encoded, index=X.index).astype(float)
        return transformed

    def summary(self) -> None:
        if self.report_ is None:
            raise ValueError("The screener has not been fitted yet.")

        tier_counts = self.report_["tier"].value_counts().sort_index()
        recommended_cap = max(5, math.floor(self.n_pos_ / max(self.min_epv, 1)))
        if self.max_features is not None:
            recommended_cap = min(recommended_cap, self.max_features)
        print("Imbalanced Feature Screener summary")
        print("-" * 48)
        print(f"Positive count (n_pos): {self.n_pos_}")
        print(f"Positive rate (base rate): {self.base_rate_:.4%}")
        print(f"EPV feature budget: {min(self.top_k, recommended_cap)}")
        print(f"Selected features: {len(self.selected_features_)}")
        print(f"Tier counts: {dict(tier_counts.to_dict())}")
        if self.dropped_collinear_:
            print("Dropped due to collinearity:")
            for item in self.dropped_collinear_:
                print(f"  - {item['feature']} (collinear with {item['collinear_with']}, corr={item['corr']:.3f})")
        else:
            print("Dropped due to collinearity: none")

    def plot_summary(self, top_n: int | None = None):
        if self.report_ is None:
            raise ValueError("The screener has not been fitted yet.")

        summary = self.report_.sort_values("consensus_rank", ascending=True).copy()
        if top_n is not None:
            summary = summary.head(top_n).copy()
        if summary.empty:
            raise ValueError("No features available to plot.")

        color_map = {
            "Anchor": "#2ca02c",
            "Interaction": "#1f77b4",
            "Weak / Redundant": "#d89b1d",
            "Noise / Drop": "#7f7f7f",
            "Leakage Check": "#d62728",
        }
        summary["color"] = summary["tier"].map(color_map).fillna("#bbbbbb")

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
        selected_boundary = min(len(summary), len(self.selected_features_))
        for ax, metric in zip(axes, ["iv", "perm_pr_auc", "pos_shap"]):
            feature_labels = summary["feature"].tolist()
            colors = summary["color"].tolist()
            ax.barh(feature_labels, summary[metric].to_numpy(), color=colors, edgecolor="black", linewidth=0.25)
            ax.invert_yaxis()
            if selected_boundary > 0:
                ax.axvline(x=max(selected_boundary - 0.5, 0.0), color="black", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.set_title(metric.replace("_", " ").upper())
            ax.set_xlabel(metric.upper())
            ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
        axes[0].axhline(y=0.10, color="darkorange", linestyle="--", linewidth=1)
        axes[0].axhline(y=0.30, color="crimson", linestyle="--", linewidth=1)
        fig.tight_layout()
        plt.show()
        return fig

    def plot_feature(self, X: pd.DataFrame, y: Iterable[int | float], feature: str, log_x: bool = False):
        X = self._as_dataframe(X)
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in X.")
        y = pd.Series(y).reset_index(drop=True)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        x_values = X[feature].to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=int)
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
                axes[0].plot(vals, ecdf, label=label, linewidth=2)
            ks_stat = self._ks_statistic(x_values, y_values) if self.report_ is not None else 0.0
            axes[0].set_title(f"ECDF for {feature} (KS={ks_stat:.3f})")
            axes[0].set_xlabel(feature if not log_x else f"log1p({feature})")
            axes[0].set_ylabel("CDF")
            axes[0].legend()
            axes[0].grid(alpha=0.25)

            all_vals = np.r_[pos, neg]
            if np.all(np.isclose(all_vals, all_vals[0])):
                axes[1].bar([0], [1], color="grey")
                axes[1].set_title(f"Positive rate by bin for {feature}")
                fig.tight_layout()
                plt.show()
                return fig

            if np.all(np.isfinite(all_vals)) and np.nanmin(all_vals) < np.nanmax(all_vals):
                bins = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), 10)
            else:
                bins = np.unique(np.asarray(np.r_[all_vals[~np.isnan(all_vals)], np.nanmax(all_vals)]))
            counts = []
            event_rate = []
            for left, right in zip(bins[:-1], bins[1:]):
                mask = (x_values >= left) & (x_values < right)
                if mask.sum() == 0:
                    counts.append(0)
                    event_rate.append(np.nan)
                else:
                    counts.append(int(mask.sum()))
                    event_rate.append(float(y_values[mask].mean()))

            ax = axes[1]
            x_positions = np.arange(len(counts))
            ax.bar(x_positions, counts, alpha=0.5, color="steelblue", label="Bin count")
            ax2 = ax.twinx()
            ax2.plot(x_positions, event_rate, color="crimson", marker="o", label="Positive rate")
            ax2.axhline(y_values.mean(), color="black", linestyle="--", alpha=0.8, label="Base rate")
            ax.set_title(f"Optimal bins and lift for {feature}")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Sample count")
            ax2.set_ylabel("Positive rate")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            plt.show()
            return fig

        fig, ax = plt.subplots(figsize=(8, 4))
        counts = X[feature].astype(str).fillna("MISSING")
        by_label = pd.crosstab(counts, y_values)
        by_label.plot(kind="bar", ax=ax)
        ax.set_title(f"Category composition for {feature}")
        ax.legend(title="Target")
        plt.show()
        return fig

    def _as_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.copy()

    def _resolve_categorical_columns(self, X: pd.DataFrame) -> list[str]:
        if self.categorical_columns is not None:
            return [col for col in self.categorical_columns if col in X.columns]
        if self.categorical_encoding == "none":
            return []
        return [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

    def _build_categorical_encoder(self, column: str):
        params = dict(self.categorical_encoder_params)
        if self.categorical_encoding == "m_estimate":
            return MEstimateEncoder(
                cols=[column],
                m=params.get("m", 10.0),
                sigma=params.get("sigma", 0.05),
                handle_unknown=params.get("handle_unknown", "value"),
                handle_missing=params.get("handle_missing", "value"),
                random_state=self.random_state,
                return_df=True,
            )
        if self.categorical_encoding == "target":
            return TargetEncoder(
                cols=[column],
                min_samples_leaf=params.get("min_samples_leaf", 20),
                smoothing=params.get("smoothing", 10.0),
                handle_unknown=params.get("handle_unknown", "value"),
                handle_missing=params.get("handle_missing", "value"),
                return_df=True,
            )
        if self.categorical_encoding == "catboost":
            return CatBoostEncoder(
                cols=[column],
                a=params.get("a", 1),
                sigma=params.get("sigma"),
                random_state=self.random_state,
                handle_unknown=params.get("handle_unknown", "value"),
                handle_missing=params.get("handle_missing", "value"),
                return_df=True,
            )
        raise ValueError(f"Unsupported categorical encoding '{self.categorical_encoding}'")

    def _fit_categorical_encoders(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.categorical_encoders_ = {}
        for col in self.categorical_columns_:
            encoder = self._build_categorical_encoder(col)
            X_col = X[[col]].copy()
            X_col[col] = X_col[col].astype("string").fillna("MISSING")
            encoder.fit(X_col, y)
            self.categorical_encoders_[col] = encoder

    def _encode_oof_categorical(self, series: pd.Series, y: pd.Series, column: str) -> pd.Series:
        series = series.astype("string").fillna("MISSING")
        if series.empty:
            return series.astype(float)

        class_counts = pd.Series(y).value_counts()
        pos_count = int(class_counts.get(1, 0))
        neg_count = int(class_counts.get(0, 0))
        min_class_count = min(pos_count, neg_count)
        n_splits = min(self.n_splits, max(2, min_class_count))
        if n_splits > len(series):
            n_splits = len(series)
        if n_splits < 2:
            raise ValueError("At least two samples per class are required to cross-fit categorical encodings.")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        encoded = np.full(len(series), np.nan, dtype=float)
        for train_idx, val_idx in cv.split(series.to_numpy(), y.to_numpy()):
            train_df = pd.DataFrame({column: series.iloc[train_idx].to_numpy()}, index=np.arange(len(train_idx)))
            val_df = pd.DataFrame({column: series.iloc[val_idx].to_numpy()}, index=np.arange(len(val_idx)))
            y_train = y.iloc[train_idx].reset_index(drop=True)
            encoder = self._build_categorical_encoder(column)
            encoder.fit(train_df, y_train)
            transformed = encoder.transform(val_df)
            encoded[val_idx] = transformed.iloc[:, 0].to_numpy(dtype=float)

        if np.isnan(encoded).any():
            global_mean = float(np.mean(y.to_numpy()))
            encoded[np.isnan(encoded)] = global_mean
        return pd.Series(encoded, index=series.index, name=column).astype(float)

    def _preprocess_features(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
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
                if self.categorical_encoding == "none":
                    encoded = s.astype("string").fillna("MISSING")
                    processed[col] = encoded.astype("category").cat.codes.astype(float)
                elif y is not None:
                    processed[col] = self._encode_oof_categorical(s, y, col)
                else:
                    processed[col] = s.astype("string").fillna("MISSING")
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
            self.binning_tables_["__current__"] = model
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
        if min(int(y.sum()), len(y) - int(y.sum())) < 2:
            return 0.0
        cv = StratifiedKFold(n_splits=min(self.n_splits, min(5, min(int(y.sum()), len(y) - int(y.sum())))))
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

    def _assign_tiers(self, y: np.ndarray) -> None:
        if self.report_ is None or self.report_.empty:
            return

        base_rate = float(np.mean(y))
        shap_q75 = self.report_["pos_shap"].quantile(0.75)
        for idx, row in self.report_.iterrows():
            if row["iv"] > self.leakage_iv_thresh or row["uni_pr_auc"] > 0.60:
                tier = "Leakage Check"
            elif row["iv"] >= self.iv_anchor_thresh or row["ks_stat"] >= self.ks_anchor_thresh or row["uni_pr_auc"] >= max(3.0 * base_rate, 0.02):
                tier = "Anchor"
            elif row["perm_pr_auc"] >= self.perm_lift_thresh and row["pos_shap"] >= shap_q75:
                tier = "Interaction"
            elif row["perm_pr_auc"] < -0.005 or (row["iv"] < 0.02 and row["ks_stat"] < 0.08 and row["perm_pr_auc"] <= 0.0005):
                tier = "Noise / Drop"
            else:
                tier = "Weak / Redundant"
            self.report_.at[idx, "tier"] = tier

    def _prune_collinear(self, processed: pd.DataFrame) -> None:
        if not self.prune_collinear or self.report_ is None or self.report_.empty:
            return

        self.report_["collinear_with"] = None
        kept = []
        for idx, row in self.report_.iterrows():
            feature = row["feature"]
            if feature not in processed.columns:
                continue
            if not pd.api.types.is_numeric_dtype(processed[feature]):
                continue
            is_reduntant = False
            for prev_feature in kept:
                if prev_feature not in processed.columns:
                    continue
                corr = self._safe_corr(processed[prev_feature], processed[feature], method=self.collinear_method)
                if corr is None:
                    continue
                if abs(corr) >= self.collinear_thresh:
                    self.report_.at[idx, "tier"] = "Weak / Redundant"
                    self.report_.at[idx, "collinear_with"] = prev_feature
                    self.dropped_collinear_.append({
                        "feature": feature,
                        "collinear_with": prev_feature,
                        "corr": float(abs(corr)),
                    })
                    is_reduntant = True
                    break
            if not is_reduntant:
                kept.append(feature)

    def _safe_corr(self, left: pd.Series, right: pd.Series, method: str) -> float | None:
        left = pd.to_numeric(left, errors="coerce")
        right = pd.to_numeric(right, errors="coerce")
        valid = left.notna() & right.notna()
        left_valid = left[valid]
        right_valid = right[valid]
        if len(left_valid) < 2 or len(right_valid) < 2:
            return None
        if left_valid.nunique() <= 1 or right_valid.nunique() <= 1:
            return None
        try:
            corr = left_valid.corr(right_valid, method=method)
        except ValueError:
            return None
        if pd.isna(corr):
            return None
        return float(corr)

    def _apply_epv_budget(self, y: np.ndarray) -> None:
        if self.report_ is None or self.report_.empty:
            self.selected_features_ = []
            return

        n_pos = int(np.sum(y == 1))
        max_recommended = max(5, math.floor(n_pos / max(self.min_epv, 1)))
        if self.max_features is not None:
            max_recommended = min(max_recommended, self.max_features)
        budget = min(self.top_k, max_recommended)

        selected = []
        tier_order = ["Anchor", "Interaction", "Weak / Redundant"]
        if self.include_leakage:
            tier_order.append("Leakage Check")

        for tier in tier_order:
            tier_features = self.report_[self.report_["tier"] == tier].sort_values(["consensus_rank", "iv"], ascending=[True, False])
            for _, row in tier_features.iterrows():
                if len(selected) >= budget:
                    break
                selected.append(row["feature"])
            if len(selected) >= budget:
                break

        self.selected_features_ = selected
        self.report_["selected"] = self.report_["feature"].isin(selected)


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
