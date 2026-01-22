from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ============================================================
# Config (must match train_for_AVMs.py exactly)
# ============================================================
BASE_COLS: List[str] = [
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

LOG_BASE_COLS: List[str] = [
    "Median_Price",
    "sales_count",
    "Consents",
]

LAG_COLS: List[str] = [
    "log_sales_count",
    "OCR",
    "CGPI_Dwelling",
    "log_Consents",
    "unemployment_rate",
    "RealMortgageRate",
]

LAGS: List[int] = [1, 3, 6, 9, 12]


# ============================================================
# I/O
# ============================================================
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")


# ============================================================
# Helpers (same as train)
# ============================================================
def tidy_month_to_period_str(df: pd.DataFrame, month_col: str = "Month") -> pd.DataFrame:
    """Convert Month to 'YYYY-MM' string."""
    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="raise")
    out[month_col] = dt.dt.to_period("M").astype(str)
    return out


def select_base_variables(df: pd.DataFrame, base_cols: List[str] = BASE_COLS) -> pd.DataFrame:
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns: {missing}")
    return df[base_cols].copy()


def _ensure_strictly_positive(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), ["Month", "District", col]].head(10) # type: ignore
        raise ValueError(f"Column '{col}' has NaNs.\nExamples:\n{bad}")

    if (df[col] <= 0).any():
        bad = df.loc[df[col] <= 0, ["Month", "District", col]].head(10)
        raise ValueError(
            f"log(x) not safe: column '{col}' contains <= 0 values.\nExamples:\n{bad}"
        )


def add_log_columns_keep_raw(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Add log(x) columns:
      - Median_Price  -> log_Median_Price
      - sales_count   -> log_sales_count
      - Consents      -> log_Consents
    """
    out = df.copy()
    for c in cols:
        _ensure_strictly_positive(out, c)
        out[f"log_{c}"] = np.log(out[c])
    return out


def add_group_lags(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    lag_cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Create lag features within each District."""
    out = df.copy()
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)
    g = out.groupby(group_col, sort=False)

    for col in lag_cols:
        if col not in out.columns:
            raise ValueError(f"Lag source column not found: {col}")
        for l in lags:
            out[f"{col}_lag{l}"] = g[col].shift(l)

    return out

def drop_level_variables(
    df: pd.DataFrame,
    cols_to_drop: List[str] = ["sales_count", "Consents"],
) -> pd.DataFrame:
    """
    Drop raw level variables to avoid duplication with log / lag features.
    This is a SAFE operation (no leakage).
    """
    out = df.copy()
    existing = [c for c in cols_to_drop if c in out.columns]
    if existing:
        out = out.drop(columns=existing)
    return out
# ============================================================
# Pipeline (same as train, but DROP Median_Price)
# ============================================================
def build_test_for_avms(test_path: Path) -> pd.DataFrame:
    df = load_dataset(test_path)

    # 1) Keep base vars
    df = select_base_variables(df, BASE_COLS)

    # 2) Month → YYYY-MM
    df = tidy_month_to_period_str(df, "Month")

    # 3) Add log columns
    df = add_log_columns_keep_raw(df, LOG_BASE_COLS)

    # 
    df = df.drop(columns=["Median_Price"])

    # 4) Add lags within District
    df = add_group_lags(
        df,
        group_col="District",
        time_col="Month",
        lag_cols=LAG_COLS,
        lags=LAGS,
    )
    
    df = drop_level_variables(df, ["sales_count", "Consents"])

    df = df.sort_values(["Month", "District"]).reset_index(drop=True)
    return df

    

# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import TEST_DIR, PREPROCESSED_DIR

    test_path = TEST_DIR / "test.csv"
    if not test_path.exists():
        test_path = TEST_DIR / "test.parquet"

    df_out = build_test_for_avms(test_path)

    out_path = PREPROCESSED_DIR / "avms" / "test_for_AVMs.csv"
    save_csv(df_out, out_path)


if __name__ == "__main__":
    main()
