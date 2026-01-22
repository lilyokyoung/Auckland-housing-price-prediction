from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import FORECAST_DIR


# ---------------------------------
# Config
# ---------------------------------
@dataclass
class RecoverLevelsConfig:
    # Historical level macro data (anchor)
    macro_level_path: Path = FORECAST_DIR / "macro_variables_level_monthly.csv"

    # BVAR forecast in transformed space
    bvar_forecast_path: Path = FORECAST_DIR / "var_macro" / "bvar_forecast.csv"

    # Output: recovered level forecast
    output_path: Path = FORECAST_DIR / "var_macro" / "bvar_forecast_levels.csv"

    month_col: str = "Month"

    # Variables to recover from dlog to level
    recover_map = {
        "dlog_CPI": "CPI",
        "dlog_CGPI_Dwelling": "CGPI_Dwelling",
    }

    # Level variables to carry forward directly
    carry_forward_vars: Tuple[str, ...] = (
        "OCR",
        "2YearFixedRate",
        "unemployment_rate",
    )


# ---------------------------------
# Functions
# ---------------------------------
def load_level_anchor(path: Path, month_col: str) -> pd.Series:
    df = pd.read_csv(path)
    df[month_col] = pd.to_datetime(df[month_col])
    df = df.sort_values(month_col)

    last_row = df.iloc[-1]
    return last_row


def load_bvar_forecast(path: Path, month_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[month_col] = pd.to_datetime(df[month_col])
    df = df.sort_values(month_col).set_index(month_col)
    return df


def recover_logdiff_series(
    dlog_series: pd.Series,
    last_level: float,
) -> pd.Series:
    """
    Recover level series from dlog forecast.
    """
    cum_log = dlog_series.cumsum()
    level_forecast = last_level * np.exp(cum_log)
    return level_forecast # type: ignore


# ---------------------------------
# main
# ---------------------------------
def main() -> None:
    cfg = RecoverLevelsConfig()

    # 1) Load anchor (last observed levels)
    last_levels = load_level_anchor(cfg.macro_level_path, cfg.month_col)

    # 2) Load BVAR forecast (dlog space)
    df_fc = load_bvar_forecast(cfg.bvar_forecast_path, cfg.month_col)

    out = pd.DataFrame(index=df_fc.index)

    # 3) Recover CPI & CGPI levels
    for dlog_var, level_var in cfg.recover_map.items():
        if dlog_var not in df_fc.columns:
            raise ValueError(f"{dlog_var} not found in BVAR forecast")

        if level_var not in last_levels.index:
            raise ValueError(f"{level_var} not found in historical macro table")

        out[level_var] = recover_logdiff_series(
            dlog_series=df_fc[dlog_var],
            last_level=last_levels[level_var],
        )

    # 4) Carry forward level variables directly
    for var in cfg.carry_forward_vars:
        if var not in df_fc.columns:
            raise ValueError(f"{var} not found in BVAR forecast")
        out[var] = df_fc[var]

    out.index.name = cfg.month_col

    # 5) Save
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.output_path, index=True)

    print("\nRecovered level forecast completed.")
    print("Saved to:", cfg.output_path)


if __name__ == "__main__":
    main()
