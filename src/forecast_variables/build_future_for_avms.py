from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
LAGS: List[int] = [1, 3, 6, 9, 12]

LAG_COLS: List[str] = [
    "log_sales_count",
    "OCR",
    "CGPI_Dwelling",
    "log_Consents",
    "unemployment_rate",
    "RealMortgageRate",
]

# We will rename these if present
RENAME_MAP: Dict[str, str] = {
    "sales_count_forecast": "sales_count",
    "Consents_forecast": "Consents",
}


SCENARIO_CANDIDATES = ["Scenario"]
MONTH_CANDIDATES = ["Month"]
DISTRICT_CANDIDATES = ["District"]

CPI_CANDIDATES = ["CPI"]
RATE_CANDIDATES = ["2YearFixedRate"]

# If you used different names in your merged future table, add them above.


# ============================================================
# Helpers
# ============================================================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    """Pick the first existing column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Cannot find {label} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def tidy_month_to_period_str(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """Convert Month to 'YYYY-MM' string."""
    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="raise")
    out[month_col] = dt.dt.to_period("M").astype(str)
    return out


def ensure_strictly_positive(df: pd.DataFrame, col: str, id_cols: Tuple[str, str, str]) -> None:
    """Raise if non-positive values exist (log not defined)."""
    month_col, district_col, scenario_col = id_cols

    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), [scenario_col, month_col, district_col, col]].head(10)
        raise ValueError(f"Column '{col}' has NaNs after numeric coercion.\nExamples:\n{bad}")

    if (df[col] <= 0).any():
        bad = df.loc[df[col] <= 0, [scenario_col, month_col, district_col, col]].head(10)
        raise ValueError(
            f"log(x) not safe: column '{col}' contains <= 0 values.\nExamples:\n{bad}"
        )


def add_real_mortgage_rate(df: pd.DataFrame, cpi_col: str, rate_col: str) -> pd.DataFrame:
    """RealMortgageRate = 2YearFixedRate - CPI (must be in same units, e.g., both %)."""
    out = df.copy()
    out[cpi_col] = pd.to_numeric(out[cpi_col], errors="coerce")
    out[rate_col] = pd.to_numeric(out[rate_col], errors="coerce")
    if out[[cpi_col, rate_col]].isna().any().any():
        raise ValueError(f"NaNs found in {cpi_col} or {rate_col}; cannot compute RealMortgageRate.")
    out["RealMortgageRate"] = out[rate_col] - out[cpi_col]
    return out


def add_log_cols(df: pd.DataFrame, cols: List[str], id_cols: Tuple[str, str, str]) -> pd.DataFrame:
    """Create log_ columns and keep raw columns (you can drop raw later if you want)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Column not found for log transform: {c}")
        ensure_strictly_positive(out, c, id_cols=id_cols)
        out[f"log_{c}"] = np.log(out[c].astype(float).values)
    return out


def add_group_lags(
    df: pd.DataFrame,
    group_cols: List[str],
    time_col: str,
    lag_cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """
    Create lag features within each group (Scenario + District), ordered by Month.
    """
    out = df.copy()
    out = out.sort_values(group_cols + [time_col]).reset_index(drop=True)

    g = out.groupby(group_cols, sort=False)

    for col in lag_cols:
        if col not in out.columns:
            raise KeyError(f"Lag source column not found: {col}")
        for l in lags:
            out[f"{col}_lag{l}"] = g[col].shift(l)

    return out


def normalize_and_prepare_future(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str, str, str]:
    """
    Return prepared df with standardized Month to 'YYYY-MM', plus resolved key col names.
    """
    scenario_col = pick_col(df, SCENARIO_CANDIDATES, "scenario")
    month_col = pick_col(df, MONTH_CANDIDATES, "month")
    district_col = pick_col(df, DISTRICT_CANDIDATES, "district")
    cpi_col = pick_col(df, CPI_CANDIDATES, "CPI")
    rate_col = pick_col(df, RATE_CANDIDATES, "2YearFixedRate")

    out = df.copy()

    # Rename common forecast columns
    out = out.rename(columns={k: v for k, v in RENAME_MAP.items() if k in out.columns})

    # Month formatting
    out = tidy_month_to_period_str(out, month_col)

    # Derived variable
    out = add_real_mortgage_rate(out, cpi_col=cpi_col, rate_col=rate_col)

    # Drop CPI & 2YearFixedRate
    out = out.drop(columns=[cpi_col, rate_col], errors="ignore")

    # log transforms (after rename)
    id_cols = (month_col, district_col, scenario_col)
    out = add_log_cols(out, cols=["sales_count", "Consents"], id_cols=id_cols)

    # Lags (VERY IMPORTANT: group by scenario + district, no cross-scenario leakage)
    out = add_group_lags(
        out,
        group_cols=[scenario_col, district_col],
        time_col=month_col,
        lag_cols=LAG_COLS,
        lags=LAGS,
    )

    # Clean final sorting
    out = out.sort_values([scenario_col, month_col, district_col]).reset_index(drop=True)

    return out, scenario_col, month_col, district_col, cpi_col, rate_col


def split_and_save_scenarios(
    df: pd.DataFrame,
    scenario_col: str,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    # Normalize scenario labels to lower-case for matching
    scen_series = df[scenario_col].astype(str).str.strip().str.lower()

    mapping = {
        "base": ["base"],
        "low": ["low", "lower"],
        "high": ["high", "higher"],
    }

    for out_name, keys in mapping.items():
        mask = scen_series.isin(keys)
        df_s = df.loc[mask].copy()
        if df_s.empty:
            # fail loudly — you probably have different labels like "baseline" / "optimistic" etc.
            uniq = sorted(df[scenario_col].astype(str).unique())
            raise ValueError(
                f"No rows found for scenario='{out_name}'. "
                f"Your scenario labels might be different. Available: {uniq}"
            )

        out_path = out_dir / f"future_for_AVMs_{out_name}.csv"
        df_s.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved: {out_path}  (rows={len(df_s)})")


# ============================================================
# Main pipeline
# ============================================================
def build_future_for_avms(
    in_path: Path,
    out_dir: Path,
) -> None:
    df = load_csv(in_path)

    df_prepared, scenario_col, month_col, district_col, _, _ = normalize_and_prepare_future(df)

    # Quick sanity prints
    print("[INFO] Resolved columns:",
          f"scenario='{scenario_col}', month='{month_col}', district='{district_col}'")
    print("[INFO] Output columns include:", [c for c in ["RealMortgageRate", "log_sales_count", "log_Consents"] if c in df_prepared.columns])
    print("[INFO] Example lag columns:", [c for c in df_prepared.columns if c.endswith("_lag12")][:8])

    split_and_save_scenarios(df_prepared, scenario_col=scenario_col, out_dir=out_dir)


def main() -> None:
    from src.config import FORECAST_DIR

    in_path = FORECAST_DIR / "future_dataset_merged.csv"
    out_dir = FORECAST_DIR / "future datasets"

    build_future_for_avms(in_path=in_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
