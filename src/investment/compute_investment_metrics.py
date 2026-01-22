# src/investment/compute_investment_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import OUTPUT_DIR


@dataclass
class InvestmentMetricsConfig:
    # ✅ your actual forecast long file (Excel)
    forecast_path: Path = (
        OUTPUT_DIR / "models" / "rf_final_predictions" / "pred_all_scenarios_long.csv"
    )

    out_dir: Path = OUTPUT_DIR / "investment"
    horizon_months: int = 12

    # ✅ your actual column names
    col_month: str = "Month"
    col_district: str = "District"
    col_scenario: str = "Scenario"
    col_price: str = "pred_Median_Price"  # NZD level (investment uses level)

    # ✅ force a common anchor month for fair comparison across districts
    force_anchor_ym: str = "2025-07"  # set to None or "" to auto-anchor per district


def _read_forecast(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")

    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suf == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]
    return df


def _to_month_start(x: Any) -> pd.Timestamp:
    """
    Robust Month parsing and normalization to month-start.

    Accepts:
      - Excel/Datetime values
      - '1/07/2025' (dayfirst)
      - '2025-07-01'
      - '2025-07'
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NaT # type: ignore

    # Already datetime-like?
    if isinstance(x, pd.Timestamp):
        dt = x
    else:
        s = str(x).strip()
        if not s:
            return pd.NaT # type: ignore

        # If looks like d/m/Y (your file), parse with dayfirst=True
        if "/" in s and len(s.split("/")) == 3:
            dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        else:
            dt = pd.to_datetime(s, errors="coerce")

    if pd.isna(dt):
        return pd.NaT # type: ignore

    # Force month-start
    return dt.to_period("M").to_timestamp()


def _series_for(
    df: pd.DataFrame,
    scenario_name: str,
    month_col: str,
    price_col: str,
    scenario_col: str,
) -> Optional[pd.Series]:
    g = df[df[scenario_col].astype(str).str.lower() == scenario_name].sort_values(month_col)
    if g.empty:
        return None
    return g.set_index(month_col)[price_col]


def compute_metrics(df: pd.DataFrame, cfg: InvestmentMetricsConfig) -> pd.DataFrame:
    m, d, sc, p = cfg.col_month, cfg.col_district, cfg.col_scenario, cfg.col_price

    for c in [m, d, sc, p]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in forecast. Found: {list(df.columns)}")

    work = df[[m, d, sc, p]].copy()

    # Parse Month -> Timestamp at month start
    work[m] = work[m].apply(_to_month_start)
    work[p] = pd.to_numeric(work[p], errors="coerce")
    work = work.dropna(subset=[m, d, sc, p])

    # Normalize scenario labels: low/base/high
    work[sc] = work[sc].astype(str).str.strip().str.lower()

    # Decide anchor
    forced_anchor = None
    if cfg.force_anchor_ym and str(cfg.force_anchor_ym).strip():
        forced_anchor = pd.to_datetime(str(cfg.force_anchor_ym).strip() + "-01")

    out_rows = []

    for district, g_d in work.groupby(d):
        g_d = g_d.sort_values(m)

        # --- Anchor month selection (fair cross-district comparison) ---
        if forced_anchor is not None:
            anchor = forced_anchor
        else:
            # Auto: earliest base month if available, else earliest overall
            g_base = g_d[g_d[sc] == "base"]
            anchor = g_base[m].min() if not g_base.empty else g_d[m].min()

        if pd.isna(anchor):
            continue

        end = anchor + pd.DateOffset(months=cfg.horizon_months)

        # Keep only months within [anchor, end]
        g_win = g_d[(g_d[m] >= anchor) & (g_d[m] <= end)].copy()
        if g_win.empty:
            continue

        s_base = _series_for(g_win, "base", m, p, sc)
        s_low = _series_for(g_win, "low", m, p, sc)
        s_high = _series_for(g_win, "high", m, p, sc)

        # Need at least two points in base to compute returns/vol
        if s_base is None or len(s_base) < 2:
            continue

        # Current and horizon price under base
        p0 = float(s_base.iloc[0])
        pT = float(s_base.iloc[-1])

        expected_return = (pT - p0) / p0 if p0 != 0 else None

        # Volatility: std of monthly pct returns (base)
        r = s_base.pct_change().dropna()
        vol = float(r.std()) if len(r) >= 2 else None

        # Downside risk: max drawdown under low scenario (negative)
        downside = None
        if s_low is not None and len(s_low) >= 2:
            cummax = s_low.cummax()
            dd = (s_low / cummax - 1.0)
            downside = float(dd.min())

        # Scenario spread at horizon: high - low (NZD level)
        spread = None
        if s_high is not None and s_low is not None and len(s_high) >= 1 and len(s_low) >= 1:
            spread = float(s_high.iloc[-1] - s_low.iloc[-1])

        # Stability score: return / vol (simple risk-adjusted)
        stability = None
        if expected_return is not None and vol is not None and vol > 0:
            stability = float(expected_return / vol)

        out_rows.append(
            {
                "district": str(district),
                "anchor_month": anchor.strftime("%Y-%m"),
                "horizon_months": cfg.horizon_months,
                "current_price_base": p0,
                "forecast_price_base_h": pT,
                "expected_return_base": expected_return,
                "volatility_base": vol,
                "downside_max_drawdown_low": downside,
                "scenario_spread_high_minus_low": spread,
                "stability_score": stability,
            }
        )

    out = pd.DataFrame(out_rows)

    if out.empty:
        return out

    # Add percent-friendly columns
    out["expected_return_base_pct"] = (out["expected_return_base"] * 100.0).round(2)
    out["downside_max_drawdown_low_pct"] = (out["downside_max_drawdown_low"] * 100.0).round(2)

    # Sort: higher stability_score first, then higher expected return
    out = out.sort_values(["stability_score", "expected_return_base"], ascending=[False, False]).reset_index(drop=True)

    return out


def main() -> None:
    cfg = InvestmentMetricsConfig()

    df = _read_forecast(cfg.forecast_path)
    out = compute_metrics(df, cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / "metrics_by_district.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Wrote: {out_path}")
    if out.empty:
        print("[WARN] Output is empty. Check forecast coverage / columns.")
    else:
        print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
