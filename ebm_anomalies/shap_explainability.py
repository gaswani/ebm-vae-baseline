# ebm_anomalies/shap_explainability.py
from __future__ import annotations

import os
import json
import time
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

import pandas as pd
import joblib

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt


@dataclass
class ShapConfig:
    """
    SHAP configuration for explaining VAE anomaly scores using a surrogate tree model.
    The VAE remains the anomaly detector (score + threshold). The surrogate exists only for explanation.
    """
    output_dir: str = "outputs/shap"
    max_anomalies: int = 100
    background_sample: int = 2000

    surrogate_model_path: str = "artifacts/shap_surrogate_lgbm.pkl"
    surrogate_meta_path: str = "artifacts/shap_surrogate_lgbm_meta.json"

    random_state: int = 42
    lgbm_params: Optional[dict] = None

    def __post_init__(self):
        if self.lgbm_params is None:
            self.lgbm_params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 50,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "random_state": self.random_state,
                "n_jobs": -1,
            }


def _ensure_dir(path: str) -> None:
    if path and path.strip():
        os.makedirs(path, exist_ok=True)


def _safe_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        preview = missing[:20]
        raise ValueError(
            f"Missing required feature columns (showing up to 20): {preview}"
            + (f" ... (+{len(missing)-20} more)" if len(missing) > 20 else "")
        )

    X = df.loc[:, feature_cols].copy()

    # Force numeric. If upstream outputs non-numeric, coerce -> NaN (LightGBM handles NaNs).
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X


def train_or_load_surrogate(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ShapConfig,
    force_retrain: bool = False
) -> lgb.LGBMRegressor:
    _ensure_dir(os.path.dirname(cfg.surrogate_model_path) or ".")
    _ensure_dir(os.path.dirname(cfg.surrogate_meta_path) or ".")

    if (not force_retrain) and os.path.exists(cfg.surrogate_model_path):
        return joblib.load(cfg.surrogate_model_path)

    model = lgb.LGBMRegressor(**cfg.lgbm_params)
    model.fit(X, y)

    joblib.dump(model, cfg.surrogate_model_path)

    meta = {
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target": str(getattr(y, "name", "reconstruction_error")),
        "model_path": cfg.surrogate_model_path,
        "lgbm_params": cfg.lgbm_params,
    }
    try:
        with open(cfg.surrogate_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    return model


def generate_shap_for_anomalies(
    df_scored: pd.DataFrame,
    feature_cols: Sequence[str],
    score_col: str = "reconstruction_error",
    anomaly_flag_col: str = "is_anomaly",
    id_cols: Optional[Sequence[str]] = None,
    cfg: Optional[ShapConfig] = None,
    force_retrain_surrogate: bool = False
) -> Dict[str, Any]:
    if cfg is None:
        cfg = ShapConfig()

    _ensure_dir(cfg.output_dir)

    if score_col not in df_scored.columns:
        raise ValueError(f"score_col '{score_col}' not found in df_scored.")
    if anomaly_flag_col not in df_scored.columns:
        raise ValueError(f"anomaly_flag_col '{anomaly_flag_col}' not found in df_scored.")

    df_non = df_scored[df_scored[anomaly_flag_col] == 0]
    df_an = df_scored[df_scored[anomaly_flag_col] == 1]

    if df_an.empty:
        return {"status": "no_anomalies", "message": "No anomalies found; no SHAP plots generated.", "output_dir": cfg.output_dir}

    # Surrogate training sample: mostly normal + some anomalies (keeps runtime manageable on big data)
    n_non = min(len(df_non), max(100_000, len(df_an) * 20))
    n_an = min(len(df_an), min(50_000, len(df_an)))

    parts = []
    if len(df_non) > 0 and n_non > 0:
        parts.append(df_non.sample(n=n_non, random_state=cfg.random_state))
    if len(df_an) > 0 and n_an > 0:
        parts.append(df_an.sample(n=n_an, random_state=cfg.random_state))

    df_train = pd.concat(parts, axis=0).sample(frac=1.0, random_state=cfg.random_state)
    X_train = _safe_feature_frame(df_train, feature_cols)
    y_train = df_train[score_col].astype(float)

    model = train_or_load_surrogate(X_train, y_train, cfg, force_retrain=force_retrain_surrogate)

    # Explain top anomalies
    df_focus = df_an.sort_values(score_col, ascending=False).head(cfg.max_anomalies).copy()
    X_focus = _safe_feature_frame(df_focus, feature_cols)

    # Background for SHAP (prefer normal behavior)
    df_bg = df_non if len(df_non) > 0 else df_scored
    df_bg = df_bg.sample(n=min(cfg.background_sample, len(df_bg)), random_state=cfg.random_state)
    X_bg = _safe_feature_frame(df_bg, feature_cols)

    explainer = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_focus)

    # Global summary plot
    summary_path = os.path.join(cfg.output_dir, "shap_summary_anomalies.png")
    plt.figure()
    shap.summary_plot(shap_values, X_focus, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=180)
    plt.close()

    # Per-record waterfall plots
    per_record_dir = os.path.join(cfg.output_dir, "anomaly_waterfalls")
    _ensure_dir(per_record_dir)

    def _row_id(row: pd.Series) -> str:
        if id_cols:
            parts = [str(row.get(c, "")).strip() for c in id_cols]
            joined = "_".join([p for p in parts if p])
            return joined if joined else str(row.name)
        return str(row.name)

    plot_paths = []
    for i in range(len(df_focus)):
        row = df_focus.iloc[i]
        rid = _row_id(row)
        rid_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rid)[:160]

        exp = shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=X_focus.iloc[i].values,
            feature_names=list(X_focus.columns),
        )

        out_path = os.path.join(per_record_dir, f"waterfall_{i+1:03d}_{rid_safe}.png")
        plt.figure()
        shap.plots.waterfall(exp, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        plot_paths.append(out_path)

    # Map plots to rows
    index_path = os.path.join(cfg.output_dir, "shap_plot_index.csv")
    out_map = df_focus.copy()
    out_map["_shap_plot_path"] = plot_paths
    out_map.to_csv(index_path, index=False)

    return {
        "status": "ok",
        "output_dir": cfg.output_dir,
        "summary_plot": summary_path,
        "plot_index_csv": index_path,
        "n_anomalies_explained": int(len(df_focus)),
        "surrogate_model_path": cfg.surrogate_model_path,
    }
