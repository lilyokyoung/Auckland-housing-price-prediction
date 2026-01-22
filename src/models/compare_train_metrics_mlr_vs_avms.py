from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def compute_metrics_log_and_level(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    # log-space
    m_log = regression_metrics(y_true_log, y_pred_log)

    # level-space (NZD)
    y_true_lvl = np.exp(y_true_log)
    y_pred_lvl = np.exp(y_pred_log)
    m_lvl = regression_metrics(y_true_lvl, y_pred_lvl)

    return {
        "RMSE_log": m_log["RMSE"],
        "MAE_log": m_log["MAE"],
        "R2_log": m_log["R2"],
        "RMSE_NZD": m_lvl["RMSE"],
        "MAE_NZD": m_lvl["MAE"],
        "R2_NZD": m_lvl["R2"],
    }


# ============================================================
# Config
# ============================================================
@dataclass
class CompareTrainConfig:
    # AVMs train dataset
    avms_train_path: Path

    # AVMs model files (best estimators)
    rf_model_path: Path
    xgb_model_path: Path
    svr_model_path: Path

    # MLR train dataset + model bundle
    mlr_train_path: Path
    mlr_results_path: Path
    mlr_feature_cols_path: Path

    out_dir: Path


# ============================================================
# Build X/y (AVMs)
# ============================================================
def build_X_y_avms(
    df: pd.DataFrame,
    target_col: str,
    drop_raw_cols: Tuple[str, ...] = (),
) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target not found: {target_col}")

    work = df.copy()
    if drop_raw_cols:
        work = work.drop(columns=[c for c in drop_raw_cols if c in work.columns], errors="ignore")

    y = work[target_col].astype(float).to_numpy()
    X = work.drop(columns=[target_col], errors="ignore")
    return X, y


# ============================================================
# MLR: load + predict on train
# ============================================================
def load_json_list(path: Path) -> list[str]:
    import json
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError("Feature cols JSON must be a list[str].")
    return cols


def load_pickle(path: Path):
    import pickle
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_X_y_mlr(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    id_cols: Tuple[str, ...] = ("Month", "District"),
) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target not found: {target_col}")

    X = df.drop(columns=[target_col] + [c for c in id_cols if c in df.columns], errors="ignore").copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce").values

    # Align columns exactly to training-selected columns
    X = X.reindex(columns=feature_cols, fill_value=0.0)
    X = sm.add_constant(X, has_constant="add")

    mask = np.isfinite(y) & np.isfinite(X.to_numpy()).all(axis=1)  # type: ignore
    return X.loc[mask], y[mask]  # type: ignore


# ============================================================
# Evaluate TRAIN (in-sample)
# ============================================================
def evaluate_avms_train(cfg: CompareTrainConfig, target_col: str = "log_Median_Price") -> pd.DataFrame:
    df = load_csv(cfg.avms_train_path)

    # drop rows with NA lags if any (safety)
    lag_cols = [c for c in df.columns if "_lag" in c]
    if lag_cols:
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

    rows = []

    # RF
    rf = joblib.load(cfg.rf_model_path)
    X_rf, y_rf = build_X_y_avms(df, target_col=target_col, drop_raw_cols=())
    yhat_rf = np.asarray(rf.predict(X_rf), dtype=float)
    rows.append({"model": "RandomForest", "split": "train_in_sample", **compute_metrics_log_and_level(y_rf, yhat_rf)})

    # XGB
    xgb = joblib.load(cfg.xgb_model_path)
    X_xgb, y_xgb = build_X_y_avms(df, target_col=target_col, drop_raw_cols=())
    yhat_xgb = np.asarray(xgb.predict(X_xgb), dtype=float)
    rows.append({"model": "XGBoost", "split": "train_in_sample", **compute_metrics_log_and_level(y_xgb, yhat_xgb)})

    # SVR (drop raw cols)
    svr = joblib.load(cfg.svr_model_path)
    X_svr, y_svr = build_X_y_avms(df, target_col=target_col, drop_raw_cols=("sales_count", "Consents"))
    yhat_svr = np.asarray(svr.predict(X_svr), dtype=float)
    rows.append({"model": "SVR", "split": "train_in_sample", **compute_metrics_log_and_level(y_svr, yhat_svr)})

    return pd.DataFrame(rows)


def evaluate_mlr_train(cfg: CompareTrainConfig, target_col: str = "log_Median_Price") -> pd.DataFrame:
    df = load_csv(cfg.mlr_train_path)

    # strict drop NA like OLS requires
    feature_cols_all = [c for c in df.columns if c != target_col]
    df = df.dropna(subset=[target_col] + feature_cols_all).reset_index(drop=True)

    cols = load_json_list(cfg.mlr_feature_cols_path)
    results = load_pickle(cfg.mlr_results_path)

    X, y = build_X_y_mlr(df, target_col=target_col, feature_cols=cols)
    yhat = np.asarray(results.predict(X), dtype=float)

    return pd.DataFrame([{"model": "MLR_baseline", "split": "train_in_sample", **compute_metrics_log_and_level(y, yhat)}])


def save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    csv_path = out_dir / "train_metrics_mlr_vs_avms.csv"
    txt_path = out_dir / "train_metrics_mlr_vs_avms.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "MLR vs AVMs — TRAIN metrics (in-sample; sorted by RMSE_NZD)",
        "===========================================================",
        "",
        "Note: Train metrics are in-sample diagnostics (will be better than TEST).",
        "",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']}: RMSE_NZD={r['RMSE_NZD']:.0f}, MAE_NZD={r['MAE_NZD']:.0f}, R2_NZD={r['R2_NZD']:.3f} | "
            f"RMSE_log={r['RMSE_log']:.4f}, MAE_log={r['MAE_log']:.4f}, R2_log={r['R2_log']:.3f}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {txt_path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import PREPROCESSED_DIR, MODEL_DIR

    cfg = CompareTrainConfig(
        avms_train_path=PREPROCESSED_DIR / "avms" / "train_for_AVMs.csv",

        rf_model_path=MODEL_DIR / "avms" / "rf_best_model" / "rf_best_model.joblib",
        xgb_model_path=MODEL_DIR / "avms" / "xgboost_best_model" / "xgb_best_model.joblib",
        svr_model_path=MODEL_DIR / "avms" / "svr_best_model" / "svr_best_model.joblib",

        mlr_train_path=PREPROCESSED_DIR / "train_for_mlr.csv",
        mlr_results_path=MODEL_DIR / "mlr" / "mlr_baseline_results.pkl",
        mlr_feature_cols_path=MODEL_DIR / "mlr" / "mlr_feature_cols.json",

        out_dir=MODEL_DIR / "compare",
    )

    df_avms = evaluate_avms_train(cfg, target_col="log_Median_Price")
    df_mlr = evaluate_mlr_train(cfg, target_col="log_Median_Price")

    df = pd.concat([df_mlr, df_avms], ignore_index=True)
    df = df.sort_values("RMSE_NZD", ascending=True).reset_index(drop=True)

    print("\n=== TRAIN metrics comparison (sorted by RMSE_NZD) ===")
    print(df)

    save_outputs(df, cfg.out_dir)


if __name__ == "__main__":
    main()
