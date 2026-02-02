# src/forecast_variables/sarimax_forecast.py
# SARIMAX forecasting for Consents and Sales Count (81-sample friendly)
# + Print BIC comparison table (Top N) for grid search

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Optional, Tuple, Any, Dict, List

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

    # how many BIC rows to print/save
    top_n: int = 10


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
) -> tuple[Any, Any, float, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      best_res, best_order, best_bic,
      results_df (all tried models incl. failures),
      top_df (Top N by BIC, successful fits only)
    """
    best_res = None
    best_bic = np.inf
    best_order = None

    rows: List[Dict[str, Any]] = []

    for p, q, P, Q in product(cfg.p_values, cfg.q_values, cfg.P_values, cfg.Q_values):
        order = (p, cfg.d, q)
        seasonal_order = (P, cfg.D, Q, cfg.m)

        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)

            bic = float(res.bic) # type: ignore
            aic = float(res.aic) # type: ignore
            hqic = float(res.hqic) # type: ignore

            rows.append(
                {
                    "pdq": str(order),
                    "pdqs": str(seasonal_order),
                    "bic": bic,
                    "aic": aic,
                    "hqic": hqic,
                    "status": "ok",
                }
            )

            if bic < best_bic:
                best_bic = bic
                best_res = res
                # keep your original combined order format (p,d,q,P,D,Q)
                best_order = (p, cfg.d, q, P, cfg.D, Q)

        except Exception as e:
            # Keep failure rows so you know what failed (optional but helpful)
            rows.append(
                {
                    "pdq": str(order),
                    "pdqs": str(seasonal_order),
                    "bic": np.nan,
                    "aic": np.nan,
                    "hqic": np.nan,
                    "status": "fail",
                    "error": type(e).__name__,
                }
            )
            continue

    results_df = pd.DataFrame(rows)

    # Top N by BIC (successful only)
    top_df = (
        results_df[results_df["status"] == "ok"]
        .dropna(subset=["bic"])
        .sort_values("bic", ascending=True)
        .head(cfg.top_n)
        .reset_index(drop=True)
    )

    return best_res, best_order, float(best_bic), results_df, top_df


def forecast_sarimax(
    res: Any,
    steps: int,
    exog_future: Optional[pd.DataFrame],
) -> pd.Series:
    fc = res.get_forecast(steps=steps, exog=exog_future)
    return fc.predicted_mean


def _print_top_table(title: str, top_df: pd.DataFrame) -> None:
    """Pretty console printing (no impact on Streamlit)."""
    print(f"\n--- {title}: Top models by BIC ---")
    if top_df.empty:
        print("(No successful fits)")
        return

    show_cols = ["pdq", "pdqs", "bic", "aic", "hqic"]
    # avoid scientific notation + keep it readable
    with pd.option_context("display.max_rows", 50, "display.max_columns", 20, "display.width", 140):
        print(top_df[show_cols].to_string(index=False))


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

    res_c, order_c, bic_c, all_c, top_c = sarimax_grid_search(
        y=y_consents,
        exog=None,
        cfg=cfg,
    )

    print("Best order (p,d,q,P,D,Q):", order_c)
    print("Best BIC:", bic_c)

    _print_top_table("Consents", top_c)

    # Save grid search tables (safe: does not affect Streamlit)
    top_c.to_csv(cfg.out_dir / "gridsearch_top_consents.csv", index=False)
    all_c.to_csv(cfg.out_dir / "gridsearch_all_consents.csv", index=False)

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

    res_s, order_s, bic_s, all_s, top_s = sarimax_grid_search(
        y=y_sales,
        exog=exog_sales,
        cfg=cfg,
    )

    print("Best order (p,d,q,P,D,Q):", order_s)
    print("Best BIC:", bic_s)

    _print_top_table("Sales Count", top_s)

    top_s.to_csv(cfg.out_dir / "gridsearch_top_sales.csv", index=False)
    all_s.to_csv(cfg.out_dir / "gridsearch_all_sales.csv", index=False)

    # future exog = forecasted consents
    exog_future = fc_consents.to_frame(name="Consents")

    fc_sales = forecast_sarimax(
        res_s,
        steps=cfg.horizon,
        exog_future=exog_future,
    )

    # --------------------------------------------------
    # Save outputs (KEEP SAME FILE NAME + COLUMNS -> Streamlit unaffected)
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

    out_path = cfg.out_dir / "sarimax_forecasts.csv"
    out_df.to_csv(out_path)

    print("\nForecast preview:")
    print(out_df.head())
    print("\nSaved forecasts to:", out_path.resolve())
    print("Saved BIC tables to:")
    print(" -", (cfg.out_dir / "gridsearch_top_consents.csv").resolve())
    print(" -", (cfg.out_dir / "gridsearch_top_sales.csv").resolve())
    print(" -", (cfg.out_dir / "gridsearch_all_consents.csv").resolve())
    print(" -", (cfg.out_dir / "gridsearch_all_sales.csv").resolve())


if __name__ == "__main__":
    main()
