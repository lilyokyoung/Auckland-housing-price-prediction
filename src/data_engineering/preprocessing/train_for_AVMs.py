from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ============================================================
# Config
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
# Helpers
# ============================================================

def tidy_month_to_period_str(df: pd.DataFrame, month_col: str = "Month") -> pd.DataFrame:
    """
    Convert Month to 'YYYY-MM' string (month frequency, no day).
    """
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
        bad = df.loc[df[col].isna(), ["Month", "District", col]].head(10)
        raise ValueError(f"Column '{col}' has NaNs after numeric coercion.\nExamples:\n{bad}")

    if (df[col] <= 0).any():
        bad = df.loc[df[col] <= 0, ["Month", "District", col]].head(10)
        raise ValueError(
            f"log(x) not safe: column '{col}' contains <= 0 values.\nExamples:\n{bad}"
        )

def drop_raw_target(
    df: pd.DataFrame,
    raw_target: str = "Median_Price",
    log_target: str = "log_Median_Price",
) -> pd.DataFrame:
    """
    Drop raw target column after log transform.
    Ensures log target exists.
    """
    if log_target not in df.columns:
        raise ValueError(f"Expected log target not found: {log_target}")

    out = df.copy()
    if raw_target in out.columns:
        out = out.drop(columns=[raw_target])

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



def add_log_columns_keep_raw(df: pd.DataFrame, cols: List[str] = LOG_BASE_COLS) -> pd.DataFrame:
    """
    Add log(x) columns:
      - sales_count  -> log_sales_count
      - Consents     -> log_Consents
    Keep original raw columns.
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
    """
    Create lag features within each District, ordered by Month.
    """
    out = df.copy()

    # Ensure proper ordering for shifting
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)

    g = out.groupby(group_col, sort=False)

    for col in lag_cols:
        if col not in out.columns:
            raise ValueError(f"Lag source column not found: {col}")

        for l in lags:
            out[f"{col}_lag{l}"] = g[col].shift(l)

    return out


# ============================================================
# Pipeline
# ============================================================
def build_train_for_avms(train_path: Path) -> pd.DataFrame:
    df = load_dataset(train_path)

    # 1) Keep only base vars
    df = select_base_variables(df, BASE_COLS)

    # 2) Month to 'YYYY-MM' (no day) for consistency
    df = tidy_month_to_period_str(df, "Month")

    # 3) Add log columns (keep raw sales_count & Consents)
    df = add_log_columns_keep_raw(df, LOG_BASE_COLS)

    # 3.1) Drop raw Median_Price, keep log_Median_Price only
    df = drop_raw_target(
        df,
        raw_target="Median_Price",
        log_target="log_Median_Price",
    )

    df = drop_level_variables(df, ["sales_count", "Consents"])
    
    # 4) Add lags for specified columns, within each District
    df = add_group_lags(
        df,
        group_col="District",
        time_col="Month",
        lag_cols=LAG_COLS,
        lags=LAGS,
    )

    # Final sort for cleanliness
    df = df.sort_values(["Month", "District"]).reset_index(drop=True)
    return df


# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import TRAIN_DIR, PREPROCESSED_DIR

    train_path = TRAIN_DIR / "train.csv"
    if not train_path.exists():
        # fallback if you saved parquet
        train_path = TRAIN_DIR / "train.parquet"

    df_out = build_train_for_avms(train_path)

    out_path = PREPROCESSED_DIR / "avms" / "train_for_AVMs.csv"
    save_csv(df_out, out_path)


if __name__ == "__main__":
    main()
