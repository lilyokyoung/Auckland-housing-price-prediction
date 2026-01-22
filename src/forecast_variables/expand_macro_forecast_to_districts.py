# src/forecast_variables/expand_macro_forecast_to_districts.py
# Expand monthly macro forecasts (VAR outputs) to a Month x District panel (7 districts)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import FORECAST_DIR


# -----------------------------
# Config
# -----------------------------
@dataclass
class ExpandConfig:
    input_path: Path = FORECAST_DIR / "var_macro" / "bvar_forecast_levels.csv"
    output_path: Path = FORECAST_DIR / "var_macro" / "bvar_forecast_levels_by_district.csv"

    month_col: str = "Month"
    district_col: str = "District"

    districts: Tuple[str, ...] = (
        "AucklandCity",
        "Franklin",
        "Manukau",
        "NorthShore",
        "Papakura",
        "Rodney",
        "Waitakere",
    )

    # keep only these columns (adjust if you want more/less)
    keep_cols: Tuple[str, ...] = (
        "CPI",
        "CGPI_Dwelling",
        "OCR",
        "2YearFixedRate",
        "unemployment_rate",
    )


# -----------------------------
# Functions
# -----------------------------
def load_monthly_macro_forecast(path: Path, month_col: str) -> pd.DataFrame:
    """
    Load monthly macro forecast table.
    Handles either:
    - Month as a normal column, or
    - Month saved as the first index column (e.g., 'Unnamed: 0')
    """
    df = pd.read_csv(path)

    # If Month isn't a column, try the first column (common when saving index=True)
    if month_col not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: month_col})

    df[month_col] = pd.to_datetime(df[month_col], errors="raise")
    df = df.sort_values(month_col).reset_index(drop=True)
    return df


def expand_to_district_panel(
    df_monthly: pd.DataFrame,
    month_col: str,
    district_col: str,
    districts: Tuple[str, ...],
    keep_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Cross-join monthly rows with the 7 districts to create a panel dataset."""
    missing = [c for c in keep_cols if c not in df_monthly.columns]
    if missing:
        raise ValueError(f"Missing columns in macro forecast: {missing}\nAvailable: {df_monthly.columns.tolist()}")

    df_m = df_monthly[[month_col] + list(keep_cols)].copy()

    df_d = pd.DataFrame({district_col: list(districts)})

    # Cross join (pandas 1.2+)
    df_m["_tmp"] = 1
    df_d["_tmp"] = 1
    out = df_m.merge(df_d, on="_tmp").drop(columns="_tmp")

    # Order columns
    out = out[[month_col, district_col] + list(keep_cols)]

    # Sort for neatness
    out = out.sort_values([month_col, district_col]).reset_index(drop=True)
    return out


def save_panel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -----------------------------
# main
# -----------------------------
def main() -> None:
    cfg = ExpandConfig()

    print("\n=== Expand Macro Forecast to 7 Districts ===")
    print("Input :", cfg.input_path.resolve())
    print("Output:", cfg.output_path.resolve())

    df_monthly = load_monthly_macro_forecast(cfg.input_path, cfg.month_col)
    df_panel = expand_to_district_panel(
        df_monthly=df_monthly,
        month_col=cfg.month_col,
        district_col=cfg.district_col,
        districts=cfg.districts,
        keep_cols=cfg.keep_cols,
    )

    save_panel(df_panel, cfg.output_path)

    print("\nPreview:")
    print(df_panel.head(10))
    print("\nDone.")


if __name__ == "__main__":
    main()
