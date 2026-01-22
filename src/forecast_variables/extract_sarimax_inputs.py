# src/forecast_variables/extract_sarimax_inputs.py
# Extract monthly Consents and Sales Count for SARIMAX forecasting

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import PROCESSED_DIR, FORECAST_DIR


# -----------------------------
# Config
# -----------------------------
@dataclass
class SARIMAXExtractConfig:
    input_path: Path = PROCESSED_DIR / "merged_dataset" / "merged_dataset1.csv"
    output_path: Path = (
        FORECAST_DIR / "sarimax_inputs" / "consents_sales_monthly.csv"
    )

    month_col: str = "Month"

    sarimax_cols: Tuple[str, ...] = (
        "Consents",
        "sales_count",
    )

    agg_method: str = "sum"   # important for counts


# -----------------------------
# Functions
# -----------------------------
def load_merged_dataset(path: Path) -> pd.DataFrame:
    """Load merged_dataset1.csv."""
    return pd.read_csv(path)


def standardize_month_column(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """Convert Month column to month-start timestamp."""
    df = df.copy()
    dt = pd.to_datetime(df[month_col], errors="coerce")

    if dt.isna().any():
        bad = df.loc[dt.isna(), month_col].head(5).tolist()
        raise ValueError(f"Unparseable Month values found: {bad}")

    df[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    return df.sort_values(month_col)


def extract_monthly_series(
    df_panel: pd.DataFrame,
    month_col: str,
    cols: Tuple[str, ...],
    agg_method: str,
) -> pd.DataFrame:
    """
    Aggregate panel data into monthly time series.
    """
    missing = [c for c in cols if c not in df_panel.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df_panel.copy()

    # ensure numeric
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if agg_method == "sum":
        out = df.groupby(month_col)[list(cols)].sum()
    elif agg_method == "mean":
        out = df.groupby(month_col)[list(cols)].mean()
    else:
        raise ValueError("agg_method must be 'sum' or 'mean'")

    out = out.sort_index()
    out.index.name = month_col
    return out


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


# -----------------------------
# main
# -----------------------------
def main() -> None:
    cfg = SARIMAXExtractConfig()

    print("\n=== Extract SARIMAX Inputs (Consents & Sales Count) ===")
    print("Input :", cfg.input_path.resolve())
    print("Output:", cfg.output_path.resolve())

    # 1) Load panel data
    df = load_merged_dataset(cfg.input_path)

    # 2) Standardize Month column
    df = standardize_month_column(df, cfg.month_col)

    # 3) Aggregate to monthly series
    df_monthly = extract_monthly_series(
        df_panel=df,
        month_col=cfg.month_col,
        cols=cfg.sarimax_cols,
        agg_method=cfg.agg_method,
    )

    # 4) Save
    save_output(df_monthly, cfg.output_path)

    print("\nPreview:")
    print(df_monthly.head())
    print("\nDone.")


if __name__ == "__main__":
    main()
