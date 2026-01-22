from pathlib import Path
import pandas as pd
import numpy as np
from src.config import PROCESSED_DIR, FORECAST_DIR

# =========================
# Config
# =========================
MERGED_DATA_PATH = PROCESSED_DIR / "merged_dataset" / "merged_dataset1.csv"

OUT_DIR = FORECAST_DIR / "future_scenarios" / "weeklyrent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIM_RATIO = 0.10          # 10% trimmed mean
HIGH_FACTOR = 1.10         # +10%
LOW_FACTOR = 0.90          # -10%
FORECAST_HORIZON = 12      # months

# =========================
# Helpers
# =========================
def normalize_month(df: pd.DataFrame, col: str = "Month") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), col].head(10).tolist()
        raise ValueError(f"Unparseable Month values: {bad}")
    df[col] = df[col].dt.to_period("M").dt.to_timestamp(how="start") # type: ignore
    return df


def extract_weeklyrent(merged_path: Path) -> pd.DataFrame:
    df = pd.read_csv(merged_path)
    df = normalize_month(df, "Month")

    if "weeklyrent" not in df.columns:
        raise KeyError("weeklyrent column not found in merged_dataset1")

    df = df[["Month", "weeklyrent"]].dropna()
    df = df.sort_values("Month").reset_index(drop=True)
    return df


def trimmed_mean(series: pd.Series, trim_ratio: float) -> float:
    lower = series.quantile(trim_ratio)
    upper = series.quantile(1 - trim_ratio)
    trimmed = series[(series >= lower) & (series <= upper)]

    if trimmed.empty:
        raise ValueError("Trimmed series is empty. Check trim_ratio or data distribution.")

    return float(trimmed.mean())


def build_weeklyrent_scenarios(
    hist_df: pd.DataFrame,
    future_months: pd.DatetimeIndex,
    trim_ratio: float,
    high_factor: float,
    low_factor: float,
) -> pd.DataFrame:
    base_value = trimmed_mean(hist_df["weeklyrent"], trim_ratio)

    scenarios = []
    for scenario, factor in {
        "Base": 1.0,
        "High": high_factor,
        "Low": low_factor,
    }.items():
        scenarios.append(
            pd.DataFrame(
                {
                    "Month": future_months,
                    "Scenario": scenario,
                    "weeklyrent": base_value * factor,
                }
            )
        )

    return pd.concat(scenarios, ignore_index=True)

# =========================
# Main
# =========================
def main() -> None:
    hist_rent = extract_weeklyrent(MERGED_DATA_PATH)

    last_month = hist_rent["Month"].max()
    future_months = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=FORECAST_HORIZON,
        freq="MS",
    )

    rent_scenarios = build_weeklyrent_scenarios(
        hist_df=hist_rent,
        future_months=future_months,
        trim_ratio=TRIM_RATIO,
        high_factor=HIGH_FACTOR,
        low_factor=LOW_FACTOR,
    )

    out_path = OUT_DIR / "weeklyrent_monthly_scenarios.csv"
    rent_scenarios.to_csv(out_path, index=False)

    print(f"Weekly rent scenarios saved to:\n{out_path}")
    print(rent_scenarios.head(9))


if __name__ == "__main__":
    main()
