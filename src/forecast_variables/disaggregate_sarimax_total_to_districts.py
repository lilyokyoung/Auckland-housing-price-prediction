from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.config import FORECAST_DIR


# -----------------------------
# Config
# -----------------------------
@dataclass
class DisaggConfig:
    # Inputs
    total_forecast_path: Path = FORECAST_DIR / "sarimax" / "sarimax_forecasts.csv"
    shares_path: Path = FORECAST_DIR / "sarimax" / "district_shares.csv"

    # Output
    output_path: Path = FORECAST_DIR / "sarimax" / "sarimax_forecasts_by_district.csv"

    # Column names
    month_col: str = "Month"
    district_col: str = "District"

    # Total forecast columns (from sarimax_forecasts.csv)
    consents_total_col: str = "Consents_forecast"
    sales_total_col: str = "sales_count_forecast"

    # Share columns (from district_shares.csv)
    consents_share_col: str = "consents_share"
    sales_share_col: str = "sales_share"


# -----------------------------
# Helpers
# -----------------------------
def standardize_month(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """
    Robust Month parsing:
    - Try multiple date parsing strategies
    - Choose the one that yields the most unique months
    - Convert to month-start Timestamp (MS)
    """
    df = df.copy()
    s = df[month_col].astype(str).str.strip()

    # Candidate parsers
    candidates = []

    # 1) Generic parse dayfirst=True
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    candidates.append(("dayfirst=True", dt1))

    # 2) Generic parse dayfirst=False
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    candidates.append(("dayfirst=False", dt2))

    # 3) Strict format: d/m/Y (Excel NZ style: 1/07/2025)
    dt3 = pd.to_datetime(s, errors="coerce", format="%d/%m/%Y")
    candidates.append(("format=%d/%m/%Y", dt3))

    # 4) Strict format: m/d/Y (US style)
    dt4 = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")
    candidates.append(("format=%m/%d/%Y", dt4))

    # Pick the best candidate: fewest NaT, then most unique months
    best_name, best_dt = None, None
    best_score = (-1, -1)  # (parsed_count, unique_months)

    for name, dt in candidates:
        parsed_count = int(dt.notna().sum())
        unique_months = int(dt.dropna().dt.to_period("M").nunique())
        score = (parsed_count, unique_months)
        if score > best_score:
            best_score = score
            best_name, best_dt = name, dt

    if best_dt is None or best_dt.isna().any():
        bad = s[best_dt.isna()].head(10).tolist() if best_dt is not None else s.head(10).tolist()
        raise ValueError(f"Unparseable Month values found (showing up to 10): {bad}")

    print(f"[DEBUG] Month parse chosen: {best_name}, parsed={best_score[0]}, unique_months={best_score[1]}")

    df[month_col] = best_dt.dt.to_period("M").dt.to_timestamp(how="start")
    return df


def load_total_forecast(cfg: DisaggConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.total_forecast_path)

    # strip hidden spaces in column names (Excel often adds them)
    df.columns = [c.strip() for c in df.columns]

    # ensure Month exists (sometimes saved as index)
    if cfg.month_col not in df.columns and df.columns[0] in ("Unnamed: 0", "index"):
        df = df.rename(columns={df.columns[0]: cfg.month_col})

    required = [cfg.month_col, cfg.consents_total_col, cfg.sales_total_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Total forecast is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"File: {cfg.total_forecast_path}"
        )

    df = standardize_month(df, cfg.month_col)

    # numeric
    df[cfg.consents_total_col] = pd.to_numeric(df[cfg.consents_total_col], errors="coerce")
    df[cfg.sales_total_col] = pd.to_numeric(df[cfg.sales_total_col], errors="coerce")

    if df[[cfg.consents_total_col, cfg.sales_total_col]].isna().any().any():
        raise ValueError("Total forecast contains non-numeric or NA values after conversion.")

    df = df.sort_values(cfg.month_col).reset_index(drop=True)
    return df


def load_shares(cfg: DisaggConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.shares_path)
    df.columns = [c.strip() for c in df.columns]

    required = [cfg.district_col, cfg.consents_share_col, cfg.sales_share_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Shares file missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"File: {cfg.shares_path}"
        )

    # numeric + sanity
    df[cfg.consents_share_col] = pd.to_numeric(df[cfg.consents_share_col], errors="coerce")
    df[cfg.sales_share_col] = pd.to_numeric(df[cfg.sales_share_col], errors="coerce")

    if df[[cfg.consents_share_col, cfg.sales_share_col]].isna().any().any():
        raise ValueError("Shares contain non-numeric or NA values after conversion.")

    cons_sum = float(df[cfg.consents_share_col].sum())
    sales_sum = float(df[cfg.sales_share_col].sum())

    # allow tiny floating error
    if not (0.99 <= cons_sum <= 1.01):
        print(f"[WARN] consents_share sum = {cons_sum:.6f} (expected ~ 1.0)")
    if not (0.99 <= sales_sum <= 1.01):
        print(f"[WARN] sales_share sum = {sales_sum:.6f} (expected ~ 1.0)")

    return df


def disaggregate_to_districts(
    total_fc: pd.DataFrame,
    shares: pd.DataFrame,
    cfg: DisaggConfig,
) -> pd.DataFrame:
    """
    Cross join Month x District then multiply by shares.
    Output columns:
      Month, District, Consents_forecast, sales_count_forecast
    """
    # cross join (pandas >= 1.2)
    out = total_fc.merge(shares, how="cross")

    out[cfg.consents_total_col] = out[cfg.consents_total_col] * out[cfg.consents_share_col]
    out[cfg.sales_total_col] = out[cfg.sales_total_col] * out[cfg.sales_share_col]

    out = out[[cfg.month_col, cfg.district_col, cfg.consents_total_col, cfg.sales_total_col]]
    out = out.sort_values([cfg.month_col, cfg.district_col]).reset_index(drop=True)
    return out


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -----------------------------
# main
# -----------------------------
def main() -> None:
    cfg = DisaggConfig()

    print("\n=== Disaggregate SARIMAX Total Forecasts to 7 Districts ===")
    print("Total forecast path:", cfg.total_forecast_path.resolve())
    print("Shares path       :", cfg.shares_path.resolve())
    print("Output path       :", cfg.output_path.resolve())

    total_fc = load_total_forecast(cfg)
    shares = load_shares(cfg)

    out = disaggregate_to_districts(total_fc, shares, cfg)

    save_output(out, cfg.output_path)

    print("\nPreview (first 14 rows):")
    print(out.head(14))

    print("\nDone. Saved to:", cfg.output_path.resolve())


if __name__ == "__main__":
    main()
