from pathlib import Path
import pandas as pd
from src.config import FORECAST_DIR

# =========================
# Config
# =========================
IN_BVAR = FORECAST_DIR / "var_macro" / "bvar_forecast_levels_by_district.csv"
IN_SARIMAX = FORECAST_DIR / "sarimax" / "sarimax_forecasts_by_district.csv"
IN_MIG = FORECAST_DIR / "future_scenarios" / "migration" / "net_migration_monthly_scenarios_by_district.csv"
IN_RENT = FORECAST_DIR / "future_scenarios" / "weeklyrent" / "weeklyrent_monthly_scenarios_by_district.csv"

OUT_PATH = FORECAST_DIR / "future_dataset_merged.csv"

# =========================
# Helpers
# =========================
def normalize_month(df: pd.DataFrame, col: str = "Month") -> pd.DataFrame:
    """Unify Month to month-start Timestamp (MS)."""
    df = df.copy()
    dt = pd.to_datetime(df[col], errors="coerce")
    dt2 = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    dt = dt.fillna(dt2)

    if dt.isna().any():
        bad = df.loc[dt.isna(), col].head(10).tolist()
        raise ValueError(f"Unparseable Month values (first 10): {bad}")

    df[col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    return df


def assert_unique_keys(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(10)
        raise ValueError(
            f"[{name}] Duplicate keys found on {keys}. Sample:\n{sample}"
        )


def load_and_prepare(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path.resolve()}")
    df = pd.read_csv(path)
    if "Month" not in df.columns:
        raise KeyError(f"{name} missing Month column")
    df = normalize_month(df, "Month")
    return df


# =========================
# Core merge
# =========================
def merge_future_tables(
    bvar_df: pd.DataFrame,
    sarimax_df: pd.DataFrame,
    mig_df: pd.DataFrame,
    rent_df: pd.DataFrame,
) -> pd.DataFrame:

    # --- Required columns ---
    need_mig = {"Month", "District", "Scenario", "Net_migration_monthly"}
    need_rent = {"Month", "District", "Scenario", "weeklyrent"}
    need_sarimax = {"Month", "District"}   # 不强制 forecast 列名
    need_bvar = {"Month", "District"}

    for need, df, nm in [
        (need_mig, mig_df, "MIG"),
        (need_rent, rent_df, "RENT"),
        (need_sarimax, sarimax_df, "SARIMAX"),
        (need_bvar, bvar_df, "BVAR"),
    ]:
        miss = need - set(df.columns)
        if miss:
            raise KeyError(f"{nm} missing columns: {sorted(miss)}")

    # --- Uniqueness checks ---
    assert_unique_keys(mig_df, ["Month", "District", "Scenario"], "MIG")
    assert_unique_keys(rent_df, ["Month", "District", "Scenario"], "RENT")
    assert_unique_keys(sarimax_df, ["Month", "District"], "SARIMAX")
    assert_unique_keys(bvar_df, ["Month", "District"], "BVAR")

    # --- Step 1: skeleton = migration + rent ---
    base = mig_df.merge(
        rent_df,
        on=["Month", "District", "Scenario"],
        how="inner",
        validate="one_to_one",
    )

    # --- Step 2: merge SARIMAX (broadcast to Scenario) ---
    base = base.merge(
        sarimax_df,
        on=["Month", "District"],
        how="left",
        validate="many_to_one",
    )

    # --- Step 3: merge BVAR (broadcast to Scenario) ---
    base = base.merge(
        bvar_df,
        on=["Month", "District"],
        how="left",
        validate="many_to_one",
    )

    # --- Final ordering ---
    base = base.sort_values(
        ["Scenario", "Month", "District"]
    ).reset_index(drop=True)

    return base


# =========================
# Main
# =========================
def main() -> None:
    bvar = load_and_prepare(IN_BVAR, "BVAR")
    sarimax = load_and_prepare(IN_SARIMAX, "SARIMAX")
    mig = load_and_prepare(IN_MIG, "MIG")
    rent = load_and_prepare(IN_RENT, "RENT")

    merged = merge_future_tables(bvar, sarimax, mig, rent)

    merged.to_csv(OUT_PATH, index=False)
    print(f"[OK] merged saved to: {OUT_PATH.resolve()}")
    print("Shape:", merged.shape)
    print(merged.head(8))


if __name__ == "__main__":
    main()
