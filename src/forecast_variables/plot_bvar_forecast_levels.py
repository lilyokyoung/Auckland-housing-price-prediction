from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

from src.config import FORECAST_DIR,FIGURE_DIR


# ---------------------------------
# Config
# ---------------------------------
@dataclass
class PlotBVARConfig:
    # Historical level data
    historical_path: Path = FORECAST_DIR / "macro_variables_level_monthly.csv"

    # Recovered level forecast (from dlog -> level)
    forecast_level_path: Path = FORECAST_DIR / "var_macro" / "bvar_forecast_levels.csv"

    # Output directory for figures
    output_dir: Path = FIGURE_DIR / "bvar_forecast"

    month_col: str = "Month"

    plot_vars: Tuple[str, ...] = (
        "CPI",
        "CGPI_Dwelling",
    )


# ---------------------------------
# Functions
# ---------------------------------
def load_level_data(path: Path, month_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[month_col] = pd.to_datetime(df[month_col])
    df = df.sort_values(month_col).set_index(month_col)
    return df


def plot_history_with_forecast(
    hist: pd.Series,
    fc: pd.Series,
    var_name: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(8, 4))

    # Historical
    plt.plot(
        hist.index,
        hist.values, # type: ignore
        label="Historical",
        color="black",
        linewidth=2,
    )

    # Forecast
    plt.plot(
        fc.index,
        fc.values, # type: ignore
        label="BVAR Forecast (12 months)",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    # Vertical split line
    plt.axvline(hist.index.max(), color="grey", linestyle=":", linewidth=1)

    plt.title(f"BVAR Forecast for {var_name}")
    plt.xlabel("Time")
    plt.ylabel(var_name)
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"{var_name}_bvar_forecast.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved figure: {fig_path}")


# ---------------------------------
# main
# ---------------------------------
def main() -> None:
    cfg = PlotBVARConfig()

    # Load historical and forecast level data
    df_hist = load_level_data(cfg.historical_path, cfg.month_col)
    df_fc = load_level_data(cfg.forecast_level_path, cfg.month_col)

    for var in cfg.plot_vars:
        if var not in df_hist.columns or var not in df_fc.columns:
            raise ValueError(f"{var} not found in one of the datasets")

        plot_history_with_forecast(
            hist=df_hist[var],
            fc=df_fc[var],
            var_name=var,
            output_dir=cfg.output_dir,
        )

    print("\nBVAR level forecast plots completed.")


if __name__ == "__main__":
    main()
