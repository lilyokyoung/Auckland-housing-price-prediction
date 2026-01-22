import pandas as pd
from src.config import FORECAST_DIR

# =========================
# Config
# =========================
IN_PATH = (
    FORECAST_DIR
    / "future_scenarios"
    / "weeklyrent"
    / "weeklyrent_monthly_scenarios.csv"
)

OUT_DIR = FORECAST_DIR / "future_scenarios" / "weeklyrent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISTRICTS = [
    "AucklandCity",
    "Franklin",
    "Manukau",
    "NorthShore",
    "Papakura",
    "Rodney",
    "Waitakere",
]


# =========================
# Helpers
# =========================
def normalize_month(df: pd.DataFrame, col: str = "Month") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), col].head(10).tolist()
        raise ValueError(f"Unparseable Month values (first 10): {bad}")
    df[col] = df[col].dt.to_period("M").dt.to_timestamp(how="start") # type: ignore
    return df


def expand_to_districts(
    scenario_df: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """
    Cross join: (Month, Scenario, weeklyrent) × Districts
    """
    scenario_df = normalize_month(scenario_df, "Month")
    district_df = pd.DataFrame({"District": districts})

    out = scenario_df.merge(district_df, how="cross")
    out = out[["Month", "Scenario", "District", "weeklyrent"]]
    return out.sort_values(["Scenario", "Month", "District"]).reset_index(drop=True)


# =========================
# Main
# =========================
def main() -> None:
    df = pd.read_csv(IN_PATH)
    expanded = expand_to_districts(df, DISTRICTS)

    out_path = OUT_DIR / "weeklyrent_monthly_scenarios_by_district.csv"
    expanded.to_csv(out_path, index=False)

    print(f"Saved weeklyrent scenarios by district to:\n{out_path}")
    print(expanded.head(12))


if __name__ == "__main__":
    main()
