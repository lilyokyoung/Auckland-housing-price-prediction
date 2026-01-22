from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 0) Config
# ============================================================

BASE_COLS = [
    "Month",
    "District",
    "Median_Price",
    "sales_count",
    "OCR",
    "CGPI_Dwelling",
    "Consents",
    "Net_migration_monthly",
    "weeklyrent",
    "unemployment_rate",
    "RealMortgageRate",
]

LOG_COLS = [
    "Median_Price",
    "sales_count",
    "Consents"    
]


# ============================================================
# 1) I/O
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .parquet or .csv")


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    print(f"Saved: {out_path}")


# ============================================================
# 2) Core functions
# ============================================================

def select_base_variables(df: pd.DataFrame, base_cols: list[str] = BASE_COLS) -> pd.DataFrame:
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns: {missing}")
    return df[base_cols].copy()


def _ensure_strictly_positive(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), ["Month", "District", col]].head(10)
        raise ValueError(f"Column '{col}' has NaNs after numeric coercion.\nExamples:\n{bad}")

    if (df[col] <= 0).any():
        bad = df.loc[df[col] <= 0, ["Month", "District", col]].head(10)
        raise ValueError(
            f"log(x) not safe: column '{col}' contains <=0 values.\nExamples:\n{bad}"
        )


def add_log_features(
    df: pd.DataFrame,
    cols: list[str] = LOG_COLS,
    drop_raw: bool = True,
) -> pd.DataFrame:
    """
    Add log(x) columns:
      log_sales_count, log_Consents

    Optionally drop original raw columns.
    """
    out = df.copy()

    for c in cols:
        _ensure_strictly_positive(out, c)
        out[f"log_{c}"] = np.log(out[c])

    if drop_raw:
        out = out.drop(columns=cols)

    return out



def tidy_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep Month as a clean string 'YYYY-MM' for downstream consistency.
    Accepts Month as string/datetime/period.
    """
    out = df.copy()

    if pd.api.types.is_period_dtype(out["Month"]): # type: ignore
        out["Month"] = out["Month"].astype(str)
        return out

    dt = pd.to_datetime(out["Month"], errors="coerce", dayfirst=True)
    if dt.isna().any():
        # If already like '2018-08' it will parse fine;
        # otherwise raise for visibility.
        bad = out.loc[dt.isna(), "Month"].head(5).tolist()
        raise ValueError(f"Month parse failed, examples: {bad}")

    out["Month"] = dt.dt.to_period("M").astype(str)
    return out


def build_train_with_log_only(train_path: Path) -> pd.DataFrame:
    df = load_dataset(train_path)
    df = select_base_variables(df)
    df = add_log_features(df)
    df = tidy_month_column(df)

    # Sort for cleanliness
    df = df.sort_values(["Month", "District"]).reset_index(drop=True)
    return df


# ============================================================
# 3) main
# ============================================================

def main() -> None:
    from src.config import TRAIN_DIR, PREPROCESSED_DIR
    # Input: your split train file
    train_path = TRAIN_DIR / "train.parquet"
    if not train_path.exists():
        train_path = TRAIN_DIR / "train.csv"

    df_train_log = build_train_with_log_only(train_path)

    # Output: for the next step (log-lags)
    out_dir = PREPROCESSED_DIR / "LASSO"
    out_path = out_dir / "train_base_with_log.parquet"
    save_dataset(df_train_log, out_path)

    # optional csv for inspection
    save_dataset(df_train_log, out_dir / "train_base_with_log.csv")


if __name__ == "__main__":
    main()
