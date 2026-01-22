# src/forecast_variables/sarimax_forecast.py
# SARIMAX forecasting for Consents and Sales Count (81-sample friendly)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import FORECAST_DIR


# -----------------------------
# Config
# -----------------------------
@dataclass
class SARIMAXConfig:
    input_path: Path = FORECAST_DIR / "sarimax_inputs" / "consents_sales_monthly.csv"
    out_dir: Path = FORECAST_DIR / "sarimax"

    date_col: str = "Month"
    horizon: int = 12

    # parameter grids (restricted for small sample)
    p_values: Tuple[int, ...] = (0, 1)
    q_values: Tuple[int, ...] = (0, 1)
    P_values: Tuple[int, ...] = (0, 1)
    Q_values: Tuple[int, ...] = (0, 1)

    d: int = 1
    D: int = 1
    m: int = 12


# -----------------------------
# Utilities
# -----------------------------
def load_data(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    return df.set_index(date_col).sort_index()


def sarimax_grid_search(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    cfg: SARIMAXConfig,
):
    best_res = None
    best_bic = np.inf
    best_order = None

    for p, q, P, Q in product(
        cfg.p_values, cfg.q_values, cfg.P_values, cfg.Q_values
    ):
        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=(p, cfg.d, q),
                seasonal_order=(P, cfg.D, Q, cfg.m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)

            if res.bic < best_bic: # type: ignore
                best_bic = res.bic # type: ignore
                best_res = res
                best_order = (p, cfg.d, q, P, cfg.D, Q)

        except Exception:
            continue

    return best_res, best_order, best_bic


def forecast_sarimax(
    res,
    steps: int,
    exog_future: Optional[pd.DataFrame],
) -> pd.Series:
    fc = res.get_forecast(steps=steps, exog=exog_future)
    return fc.predicted_mean


# -----------------------------
# main
# -----------------------------
def main() -> None:
    cfg = SARIMAXConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SARIMAX Forecasting ===")
    print("Input:", cfg.input_path.resolve())

    df = load_data(cfg.input_path, cfg.date_col)

    # --------------------------------------------------
    # 1) SARIMAX for Consents (no exog)
    # --------------------------------------------------
    print("\n[1] SARIMAX: Consents")

    y_consents = df["Consents"]

    res_c, order_c, bic_c = sarimax_grid_search(
        y=y_consents,
        exog=None,
        cfg=cfg,
    )

    print("Best order (p,d,q,P,D,Q):", order_c)
    print("BIC:", bic_c)

    fc_consents = forecast_sarimax(
        res_c,
        steps=cfg.horizon,
        exog_future=None,
    )

    # --------------------------------------------------
    # 2) SARIMAX for Sales Count (exog = Consents)
    # --------------------------------------------------
    print("\n[2] SARIMAX: Sales Count (exog = Consents)")

    y_sales = df["sales_count"]
    exog_sales = df[["Consents"]]

    res_s, order_s, bic_s = sarimax_grid_search(
        y=y_sales,
        exog=exog_sales,
        cfg=cfg,
    )

    print("Best order (p,d,q,P,D,Q):", order_s)
    print("BIC:", bic_s)

    # future exog = forecasted consents
    exog_future = fc_consents.to_frame(name="Consents")

    fc_sales = forecast_sarimax(
        res_s,
        steps=cfg.horizon,
        exog_future=exog_future,
    )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    future_idx = pd.date_range(
        start=df.index[-1] + pd.offsets.MonthBegin(1),
        periods=cfg.horizon,
        freq="MS",
    )

    out_df = pd.DataFrame(
        {
            "Consents_forecast": fc_consents.values,
            "sales_count_forecast": fc_sales.values,
        },
        index=future_idx,
    )
    out_df.index.name = cfg.date_col

    out_df.to_csv(cfg.out_dir / "sarimax_forecasts.csv")

    print("\nForecast preview:")
    print(out_df.head())
    print("\nSaved to:", (cfg.out_dir / "sarimax_forecasts.csv").resolve())


if __name__ == "__main__":
    main()
