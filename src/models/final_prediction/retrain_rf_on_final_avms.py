from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Config
# ============================================================
@dataclass
class RetrainRFConfig:
    final_data_path: Path
    rf_summary_path: Path
    out_dir: Path

    month_col: str = "Month"
    district_col: str = "District"
    target_col: str = "log_Median_Price"
    lag_suffix: str = "_lag"

    random_state: int = 42
    n_jobs: int = -1


# ============================================================
# I/O
# ============================================================
def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_rf_best_params(summary_path: Path) -> Dict[str, Any]:
    """
    Expect rf_summary.json contains something like:
      {
        "best_params": {...},
        ...
      }
    If not found, return empty dict.
    """
    if not summary_path.exists():
        print(f"[WARN] rf_summary.json not found: {summary_path} -> use defaults")
        return {}

    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    best_params = obj.get("best_params", {})
    if not isinstance(best_params, dict):
        print("[WARN] rf_summary.json best_params invalid -> use defaults")
        return {}

    # Normalize possible GridSearch style keys:
    # if stored like {"rf__n_estimators": 500, ...} strip "rf__"
    cleaned: Dict[str, Any] = {}
    for k, v in best_params.items():
        if isinstance(k, str) and k.startswith("rf__"):
            cleaned[k.replace("rf__", "", 1)] = v
        else:
            cleaned[k] = v
    return cleaned


# ============================================================
# Preprocessing
# ============================================================
def parse_month_to_month_start(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    if month_col not in df.columns:
        raise KeyError(f"Missing month column: {month_col}")
    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="raise")
    out[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    return out


def drop_rows_with_na_lags(df: pd.DataFrame, lag_suffix: str) -> pd.DataFrame:
    lag_cols = [c for c in df.columns if lag_suffix in c]
    if not lag_cols:
        raise ValueError(f"No lag columns found (suffix='{lag_suffix}').")
    return df.dropna(subset=lag_cols).reset_index(drop=True)


def drop_rows_with_any_na(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    feature_cols = [c for c in df.columns if c != target_col]
    return df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame, month_col: str, target_col: str) -> List[str]:
    """
    Features = all columns except target and Month.
    District is kept (categorical).
    """
    return [c for c in df.columns if c not in {target_col, month_col}]


# ============================================================
# Metrics
# ============================================================
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def compute_metrics_log_and_level(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    m_log = regression_metrics(y_true_log, y_pred_log)

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
# Modeling
# ============================================================
def build_rf_pipeline(
    feature_cols: List[str],
    district_col: str,
    rf_params: Dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> Pipeline:
    """
    RF pipeline:
      - OneHot encode District
      - Pass through numeric features
      - RandomForestRegressor
    """
    if district_col not in feature_cols:
        raise ValueError(f"District column '{district_col}' must be in features.")

    cat_cols = [district_col]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    # Set safe defaults, then override with best params if provided
    rf = RandomForestRegressor(
        n_estimators=int(rf_params.get("n_estimators", 500)),
        max_depth=rf_params.get("max_depth", None),
        min_samples_split=int(rf_params.get("min_samples_split", 2)),
        min_samples_leaf=int(rf_params.get("min_samples_leaf", 1)),
        max_features=rf_params.get("max_features", "auto"),  # sklearn may warn; ok
        bootstrap=bool(rf_params.get("bootstrap", True)),
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("rf", rf)])
    return pipe


def retrain_rf_on_final_dataset(cfg: RetrainRFConfig) -> Tuple[Pipeline, Dict[str, float], int]:
    df = load_table(cfg.final_data_path)

    # preprocessing (match your AVMs flow)
    df = parse_month_to_month_start(df, cfg.month_col)
    df = drop_rows_with_na_lags(df, cfg.lag_suffix)
    df = drop_rows_with_any_na(df, cfg.target_col)

    if cfg.target_col not in df.columns:
        raise KeyError(f"Target column not found: {cfg.target_col}")

    y = df[cfg.target_col].astype(float).to_numpy()

    feature_cols = get_feature_cols(df, cfg.month_col, cfg.target_col)
    X = df[feature_cols].copy()

    rf_params = load_rf_best_params(cfg.rf_summary_path)

    model = build_rf_pipeline(
        feature_cols=feature_cols,
        district_col=cfg.district_col,
        rf_params=rf_params,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    model.fit(X, y)

    # in-sample diagnostics (not CV)
    y_hat = np.asarray(model.predict(X), dtype=float)
    metrics = compute_metrics_log_and_level(y, y_hat)

    return model, metrics, len(df)


def save_outputs(
    model: Pipeline,
    metrics: Dict[str, float],
    rows_used: int,
    out_dir: Path,
    feature_cols: List[str],
) -> None:
    ensure_dir(out_dir)

    model_path = out_dir / "rf_final_model.joblib"
    joblib.dump(model, model_path)

    # metrics
    metrics_row = {"model": "RandomForest_final", "split": "train+test_final", "rows_used": rows_used, **metrics}
    pd.DataFrame([metrics_row]).to_csv(out_dir / "rf_final_metrics_train.csv", index=False, encoding="utf-8-sig")

    txt = (
        "RandomForest retrain on FINAL dataset (train_for_AVMs + test_for_AVMs)\n"
        "===============================================================\n"
        f"Rows used: {rows_used}\n\n"
        f"RMSE_NZD={metrics['RMSE_NZD']:.0f}, MAE_NZD={metrics['MAE_NZD']:.0f}, R2_NZD={metrics['R2_NZD']:.4f}\n"
        f"RMSE_log={metrics['RMSE_log']:.4f}, MAE_log={metrics['MAE_log']:.4f}, R2_log={metrics['R2_log']:.4f}\n"
    )
    (out_dir / "rf_final_metrics_train.txt").write_text(txt, encoding="utf-8")

    # feature cols for future consistency check
    (out_dir / "rf_final_feature_cols.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Saved final RF model   : {model_path}")
    print(f"[OK] Saved metrics (CSV)   : {out_dir / 'rf_final_metrics_train.csv'}")
    print(f"[OK] Saved metrics (TXT)   : {out_dir / 'rf_final_metrics_train.txt'}")
    print(f"[OK] Saved feature cols    : {out_dir / 'rf_final_feature_cols.json'}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import PREPROCESSED_DIR, MODEL_DIR

    cfg = RetrainRFConfig(
        final_data_path=PREPROCESSED_DIR / "avms" / "final_for_AVMs.csv",
        rf_summary_path=MODEL_DIR / "avms" / "rf_best_model" / "rf_summary.json",
        out_dir=MODEL_DIR / "avms" / "rf_final_model",
    )

    model, metrics, rows_used = retrain_rf_on_final_dataset(cfg)

    # feature columns must match what model saw
    df_tmp = load_table(cfg.final_data_path)
    df_tmp = parse_month_to_month_start(df_tmp, cfg.month_col)
    df_tmp = drop_rows_with_na_lags(df_tmp, cfg.lag_suffix)
    df_tmp = drop_rows_with_any_na(df_tmp, cfg.target_col)
    feature_cols = get_feature_cols(df_tmp, cfg.month_col, cfg.target_col)

    print("\n=== RF retrain on FINAL dataset finished ===")
    print(metrics)

    save_outputs(model, metrics, rows_used, cfg.out_dir, feature_cols)


if __name__ == "__main__":
    main()

