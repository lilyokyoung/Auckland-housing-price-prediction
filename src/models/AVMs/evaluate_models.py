from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Config
# ============================================================
@dataclass
class EvalConfig:
    test_path: Path
    out_dir: Path

    month_col: str = "Month"
    district_col: str = "District"
    target_col: str = "log_Median_Price"

    lag_suffix: str = "_lag"

    # model paths (all are sklearn Pipelines with OneHotEncoder inside)
    svr_model_path: Path = Path()
    rf_model_path: Path = Path()
    xgb_model_path: Path = Path()

    # SVR: drop raw columns (because log_* exists + SVR distance sensitivity)
    svr_drop_raw_cols: Tuple[str, ...] = ("sales_count", "Consents")


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


# ============================================================
# Preprocessing
# ============================================================
def parse_month_to_month_start(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """
    Normalize Month to month-start Timestamp.
    (Safe even if Month is already 'YYYY-MM' string.)
    """
    if month_col not in df.columns:
        raise KeyError(f"Missing month column: {month_col}")

    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="raise")
    out[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    return out


def drop_rows_with_na_lags(df: pd.DataFrame, lag_suffix: str = "_lag") -> pd.DataFrame:
    """Drop rows where any lag column is NA."""
    lag_cols = [c for c in df.columns if lag_suffix in c]
    if not lag_cols:
        raise ValueError(f"No lag columns found (suffix='{lag_suffix}').")
    return df.dropna(subset=lag_cols).reset_index(drop=True)


def drop_columns_if_present(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return df.drop(columns=present)


# ============================================================
# Metrics
# ============================================================
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
# Build X/y per model (keep Month & District for Pipeline)
# ============================================================
def build_X_y(
    df: pd.DataFrame,
    cfg: EvalConfig,
    drop_raw_cols: Tuple[str, ...] = (),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Return:
      X: all columns except target_col (Month & District kept for Pipeline / OneHot)
      y: target array

    drop_raw_cols:
      only used for SVR: ('sales_count', 'Consents')
    """
    if cfg.target_col not in df.columns:
        raise KeyError(f"Target column not found: {cfg.target_col}")

    work = df.copy()
    if drop_raw_cols:
        work = drop_columns_if_present(work, drop_raw_cols)

    y = work[cfg.target_col].astype(float).to_numpy()
    X = work.drop(columns=[cfg.target_col], errors="ignore")
    return X, y


# ============================================================
# Evaluation
# ============================================================
def evaluate_one_model(
    model_name: str,
    model_path: Path,
    X: pd.DataFrame,
    y_log: np.ndarray,
) -> Dict[str, float]:
    if not model_path.exists():
        raise FileNotFoundError(f"{model_name} model not found: {model_path}")

    model = joblib.load(model_path)
    y_pred_log = np.asarray(model.predict(X), dtype=float)

    m = compute_metrics_log_and_level(y_log, y_pred_log)
    return {"model": model_name, **m} # type: ignore
    


def evaluate_three_models_on_test(cfg: EvalConfig) -> pd.DataFrame:
    df = load_table(cfg.test_path)

    # Shared preprocessing
    df = parse_month_to_month_start(df, cfg.month_col)
    df = drop_rows_with_na_lags(df, cfg.lag_suffix)

    # --- SVR (drop raw cols) ---
    X_svr, y_svr = build_X_y(df, cfg, drop_raw_cols=cfg.svr_drop_raw_cols)
    res_svr = evaluate_one_model("SVR", cfg.svr_model_path, X_svr, y_svr)

    # --- RF (keep raw cols) ---
    X_rf, y_rf = build_X_y(df, cfg, drop_raw_cols=())
    res_rf = evaluate_one_model("RandomForest", cfg.rf_model_path, X_rf, y_rf)

    # --- XGB (keep raw cols) ---
    X_xgb, y_xgb = build_X_y(df, cfg, drop_raw_cols=())
    res_xgb = evaluate_one_model("XGBoost", cfg.xgb_model_path, X_xgb, y_xgb)

    out = (
        pd.DataFrame([res_svr, res_rf, res_xgb])
        .sort_values("RMSE_NZD", ascending=True)
        .reset_index(drop=True)
    )
    return out




def save_eval_outputs(df_metrics: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    csv_path = out_dir / "avms_test_metrics.csv"
    txt_path = out_dir / "avms_test_metrics.txt"

    df_metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "AVMs final evaluation on TEST (test_for_AVMs)",
        "============================================",
        "",
        "Primary metrics are in NZD (after exp back-transform).",
        "Lower RMSE_NZD is better.",
        "",
    ]

    for _, r in df_metrics.iterrows():
        lines.append(
            f"{r['model']}: "
            f"RMSE_NZD={r['RMSE_NZD']:.0f}, MAE_NZD={r['MAE_NZD']:.0f}, R2_NZD={r['R2_NZD']:.4f} | "
            f"RMSE_log={r['RMSE_log']:.4f}, MAE_log={r['MAE_log']:.4f}, R2_log={r['R2_log']:.4f}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Saved test metrics CSV: {csv_path}")
    print(f"[OK] Saved test metrics TXT: {txt_path}")




# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import PREPROCESSED_DIR, MODEL_DIR

    cfg = EvalConfig(
        test_path=PREPROCESSED_DIR / "avms" / "test_for_AVMs.csv",
        out_dir=MODEL_DIR / "avms" / "final_test_eval",

        svr_model_path=MODEL_DIR / "avms" / "svr_best_model" / "svr_best_model.joblib",
        rf_model_path=MODEL_DIR / "avms" / "rf_best_model" / "rf_best_model.joblib",
        xgb_model_path=MODEL_DIR / "avms" / "xgboost_best_model" / "xgb_best_model.joblib",
    )

    df_metrics = evaluate_three_models_on_test(cfg)

    print("\n=== TEST Evaluation Results (lower RMSE is better) ===")
    print(df_metrics)

    save_eval_outputs(df_metrics, cfg.out_dir)


if __name__ == "__main__":
    main()
