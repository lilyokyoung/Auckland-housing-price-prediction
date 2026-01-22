# src/data_engineering/extract_macro_variables.py
# Extract monthly macro variables (level) from merged_dataset1 (panel data)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR, FORECAST_DIR,FIGURE_DIR


# -----------------------------
# Config
# -----------------------------
@dataclass
class MacroExtractConfig:
    input_path: Path = PROCESSED_DIR / "merged_dataset" / "merged_dataset1.csv"
    output_path: Path = FORECAST_DIR / "macro_variables_level_monthly.csv"

    month_col: str = "Month"
    district_col: str = "District"

    macro_cols: Tuple[str, ...] = (
        "CPI",
        "CGPI_Dwelling",
        "OCR",
        "2YearFixedRate",
        "unemployment_rate",
    )


# -----------------------------
# Functions
# -----------------------------
def load_merged_dataset(path: Path) -> pd.DataFrame:
    """Load merged_dataset1.csv"""
    df = pd.read_csv(path)
    return df


def standardize_month_column(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """
    Convert Month column to month-start Timestamp (MS) and sort.
    Supports formats like '2018-08', '2018/08', '2018-08-01'.
    """
    df = df.copy()
    dt = pd.to_datetime(df[month_col], errors="coerce")

    if dt.isna().any():
        bad = df.loc[dt.isna(), month_col].head(5).tolist()
        raise ValueError(f"Unparseable Month values found: {bad}")

    df[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    df = df.sort_values(month_col).reset_index(drop=True)
    return df


def extract_monthly_macro_table(
    df_panel: pd.DataFrame,
    month_col: str,
    macro_cols: Tuple[str, ...],
    agg_method: str = "mean",
) -> pd.DataFrame:
    """
    Convert panel data (Month × District) into a monthly macro table (Month only).
    """
    # Column existence check
    missing = [c for c in macro_cols if c not in df_panel.columns]
    if missing:
        raise ValueError(
            f"Missing macro columns: {missing}\n"
            f"Available columns: {df_panel.columns.tolist()}"
        )

    # Ensure numeric
    df_panel = df_panel.copy()
    for col in macro_cols:
        df_panel[col] = pd.to_numeric(df_panel[col], errors="coerce")

    # Aggregate: one row per Month
    if agg_method == "mean":
        macro_df = df_panel.groupby(month_col, as_index=True)[list(macro_cols)].mean()
    elif agg_method == "first":
        macro_df = df_panel.groupby(month_col, as_index=True)[list(macro_cols)].first()
    else:
        raise ValueError("agg_method must be 'mean' or 'first'")

    macro_df = macro_df.sort_index()
    macro_df.index.name = month_col
    return macro_df


def save_macro_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

def plot_macro_series(
    df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "Level series",
) -> None:
    """
    Plot and save level macro time series for visual stationarity diagnostics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(df.index, df[col])
        plt.title(f"{title_prefix}: {col}")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.tight_layout()

        #  save figure
        fig_path = output_dir / f"{col}_level.png"
        plt.savefig(fig_path, dpi=300)

        plt.close()  # very important: avoid memory issues

        print(f"Saved plot: {fig_path}")

# -----------------------------
# main
# -----------------------------
def main() -> None:
    cfg = MacroExtractConfig()

    # 1. Load merged dataset
    df = load_merged_dataset(cfg.input_path)

    # 2. Standardize Month column
    df = standardize_month_column(df, cfg.month_col)

    # 3. Extract monthly macro variables (level)
    macro_monthly = extract_monthly_macro_table(
        df_panel=df,
        month_col=cfg.month_col,
        macro_cols=cfg.macro_cols,
        agg_method="mean",
    )

    #  4. Plot + save level macro series
    plot_macro_series(
    macro_monthly,
    output_dir=FIGURE_DIR / "bvar_forecast" / "macro_level_series"
)
    
    print("\n=== DEBUG PATHS ===")
    print("FIGURE_DIR =", FIGURE_DIR.resolve())
    print("Plot output dir =", FIGURE_DIR / "bvar_forecast" / "macro_level_series")
    print("===================\n")

    plot_macro_series(macro_monthly, output_dir=FIGURE_DIR / "bvar_forecast" / "macro_level_series") 
    # 5. Save output
    save_macro_table(macro_monthly, cfg.output_path)

    print("Macro variable extraction completed.")
    print(f"Output saved to: {cfg.output_path}")
if __name__ == "__main__":
    main()
