from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import pandas as pd
import statsmodels.api as sm


# =========================
# Config
# =========================
TARGET_COL = "log_Median_Price"
DROP_COLS = ["Month", "District"]


# =========================
# Helpers
# =========================
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def split_X_y(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")

    X = df.drop(columns=[target] + drop_cols, errors="ignore").copy()
    y = df[target].copy()

    # Ensure numeric (important if any column accidentally becomes object)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows with NA (required for OLS)
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if X.shape[0] == 0:
        raise ValueError("No training rows left after dropping NA. Check your lags / preprocessing.")

    return X, y


def fit_mlr(X: pd.DataFrame, y: pd.Series):
    # Keep constant consistent between train/test
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const)
    results = model.fit()
    return results


# =========================
# Main
# =========================
def main() -> None:
    from src.config import PREPROCESSED_DIR, MODEL_DIR

    train_path = PREPROCESSED_DIR / "train_for_mlr.csv"
    df = load_data(train_path)

    X, y = split_X_y(
        df,
        target=TARGET_COL,
        drop_cols=DROP_COLS,
    )

    results = fit_mlr(X, y)

    # =========================
    # Outputs
    # =========================
    out_dir = MODEL_DIR / "mlr"
    ensure_dir(out_dir)

    # 1) Full regression summary
    summary_path = out_dir / "mlr_baseline_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(results.summary().as_text())

    # 2) Coefficients table
    coef_df = pd.DataFrame(
        {
            "coef": results.params,
            "std_err": results.bse,
            "t_stat": results.tvalues,
            "p_value": results.pvalues,
        }
    )
    coef_path = out_dir / "mlr_baseline_coefficients.csv"
    coef_df.to_csv(coef_path, index=True, encoding="utf-8-sig")

    # 3) Save model results for prediction (pickle)
    model_path = out_dir / "mlr_baseline_results.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(results, f)

    # 4) Save feature column list (to align test columns)
    feature_cols_path = out_dir / "mlr_feature_cols.json"
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

    print("Baseline MLR fitted successfully.")
    print(results.summary())
    print(f"Saved summary to: {summary_path.resolve()}")
    print(f"Saved coefficients to: {coef_path.resolve()}")
    print(f"Saved model results to: {model_path.resolve()}")
    print(f"Saved feature cols to: {feature_cols_path.resolve()}")


if __name__ == "__main__":
    main()
