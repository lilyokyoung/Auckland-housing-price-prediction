from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Config
# =========================
ID_COLS = ["Month", "District"]
TARGET_COL = "log_Median_Price"


RAW_DROPPED_IN_TEST = ("sales_count", "Consents")


# =========================
# Helpers
# =========================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def drop_rows_with_any_na(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    feature_cols = [c for c in df.columns if c != target_col]
    return df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)


def build_X_y(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    # drop id + target
    X = df.drop(columns=[target_col] + [c for c in ID_COLS if c in df.columns], errors="ignore").copy()

    # numeric coercion
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce").values

    # align columns to training (missing -> 0)
    X = X.reindex(columns=feature_cols, fill_value=0.0)

    # add constant same as training
    X = sm.add_constant(X, has_constant="add")

    mask = np.isfinite(y) & np.isfinite(X.to_numpy()).all(axis=1) # type: ignore
    return X.loc[mask], y[mask] # type: ignore


# =========================
# IO: model bundle
# =========================
def load_results(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"MLR results not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_feature_cols(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature cols file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError("mlr_feature_cols.json has invalid format.")
    return cols


# =========================
# Runner
# =========================
@dataclass
class MLRTestConfig:
    test_for_mlr_path: Path
    model_results_path: Path
    feature_cols_path: Path
    out_dir: Path
    target_col: str = TARGET_COL


def evaluate_mlr_on_test(cfg: MLRTestConfig) -> dict[str, float]:
    df = load_csv(cfg.test_for_mlr_path)

    # drop raw columns if still exist
    present_raw = [c for c in RAW_DROPPED_IN_TEST if c in df.columns]
    if present_raw:
        df = df.drop(columns=present_raw)

    df_clean = drop_rows_with_any_na(df, target_col=cfg.target_col)

    feature_cols = load_feature_cols(cfg.feature_cols_path)
    results = load_results(cfg.model_results_path)

    X_test, y_test = build_X_y(df_clean, target_col=cfg.target_col, feature_cols=feature_cols)

    if len(X_test) == 0:
        raise ValueError("No usable rows left after dropping NA rows. Check lags / preprocessing.")

    # statsmodels predict
    y_pred = results.predict(X_test).values

    metrics = regression_metrics(y_test, y_pred)

    ensure_dir(cfg.out_dir)
    pd.DataFrame([metrics]).to_csv(cfg.out_dir / "mlr_test_metrics.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_df.to_csv(cfg.out_dir / "mlr_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("[RESULT] MLR test metrics:", metrics)
    print("[OK] Saved:", (cfg.out_dir / "mlr_test_metrics.csv").resolve())
    return metrics


def main() -> None:
    from src.config import PREPROCESSED_DIR, MODEL_DIR

    test_for_mlr_path = PREPROCESSED_DIR / "test_for_mlr.csv"

    mlr_dir = MODEL_DIR / "mlr"
    model_results_path = mlr_dir / "mlr_baseline_results.pkl"
    feature_cols_path = mlr_dir / "mlr_feature_cols.json"

    out_dir = mlr_dir / "mlr_test"

    cfg = MLRTestConfig(
        test_for_mlr_path=test_for_mlr_path,
        model_results_path=model_results_path,
        feature_cols_path=feature_cols_path,
        out_dir=out_dir,
        target_col="log_Median_Price",
    )

    evaluate_mlr_on_test(cfg)


if __name__ == "__main__":
    main()
