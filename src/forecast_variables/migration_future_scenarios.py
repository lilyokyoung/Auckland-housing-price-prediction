from pathlib import Path
import pandas as pd
import numpy as np
from src.config import PROCESSED_DIR, FORECAST_DIR

# =========================
# Config
# =========================
MERGED_DATA_PATH = PROCESSED_DIR / "merged_dataset" / "merged_dataset1.csv"

OUT_DIR = FORECAST_DIR / "future_scenarios" / "migration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scenario parameters
WINSOR_LIMIT = 0.10      # 10% winsorization
SCENARIO_SHIFT = 0.10   # ±10% for High / Low
FORECAST_HORIZON = 12   # months

# =========================
# Helpers
# =========================
def normalize_month(df: pd.DataFrame, col: str = "Month") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), col].unique()[:10]
        raise ValueError(f"Unparseable Month values: {bad}")

    df[col] = df[col].dt.to_period("M").dt.to_timestamp(how="start") # type: ignore
    return df


def winsorized_mean(series: pd.Series, limit: float) -> float:
    """
    Compute winsorized mean (robust to extreme values).
    """
    lower = series.quantile(limit)
    upper = series.quantile(1 - limit)
    clipped = series.clip(lower=lower, upper=upper)
    return float(clipped.mean())


# =========================
# Core logic
# =========================
def extract_net_migration(merged_path: Path) -> pd.DataFrame:
    df = pd.read_csv(merged_path)
    df = normalize_month(df, "Month")

    keep = ["Month", "Net_migration_monthly"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in merged dataset: {missing}")

    df = (
        df[keep]
        .dropna()
        .sort_values("Month")
        .reset_index(drop=True)
    )
    return df


def compute_base_winsorized(
    hist_df: pd.DataFrame,
    winsor_limit: float,
) -> float:
    """
    Robust baseline migration level using winsorized mean.
    """
    return winsorized_mean(
        hist_df["Net_migration_monthly"],
        limit=winsor_limit,
    )


def build_net_migration_scenarios(
    hist_df: pd.DataFrame,
    future_months: pd.DatetimeIndex,
    winsor_limit: float,
    scenario_shift: float,
) -> pd.DataFrame:
    base_value = compute_base_winsorized(hist_df, winsor_limit)

    scenarios = {
        "Base": 1.0,
        "High": 1.0 + scenario_shift,
        "Low": 1.0 - scenario_shift,
    }

    out = []
    for name, factor in scenarios.items():
        out.append(
            pd.DataFrame(
                {
                    "Month": future_months,
                    "Scenario": name,
                    "Net_migration_monthly": base_value * factor,
                }
            )
        )

    return pd.concat(out, ignore_index=True)


# =========================
# Main
# =========================
def main() -> None:
    hist_net_mig = extract_net_migration(MERGED_DATA_PATH)

    last_month = hist_net_mig["Month"].max()
    future_months = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=FORECAST_HORIZON,
        freq="MS",
    )

    net_mig_scenarios = build_net_migration_scenarios(
        hist_df=hist_net_mig,
        future_months=future_months,
        winsor_limit=WINSOR_LIMIT,
        scenario_shift=SCENARIO_SHIFT,
    )

    out_path = OUT_DIR / "net_migration_monthly_scenarios.csv"
    net_mig_scenarios.to_csv(out_path, index=False)

    print(f"Net migration scenarios saved to: {out_path}")
    print(net_mig_scenarios.head(9))


if __name__ == "__main__":
    main()
