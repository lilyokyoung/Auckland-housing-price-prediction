from __future__ import annotations

from pathlib import Path
from functools import reduce
import pandas as pd


# ============================================================
# 1) Global config (panel structure)
# ============================================================

DISTRICT_ORDER = [
    "AucklandCity",
    "Franklin",
    "Manukau",
    "NorthShore",
    "Papakura",
    "Rodney",
    "Waitakere",
]

KEY_COLS = ["Month", "District"]


# ============================================================
# 2) Core helper functions
# ============================================================

def standardize_panel(
    df: pd.DataFrame,
    value_cols: list[str],
    district_order: list[str] = DISTRICT_ORDER,
) -> pd.DataFrame:
    """
    Standardize a panel dataframe to ['Month','District', value_cols...]
    - Ensure Month is str
    - Ensure District is categorical with fixed order
    - Sort by Month, District
    """
    df = df.copy()

    df["Month"] = df["Month"].astype(str)
    df["District"] = df["District"].astype(str)

    df["District"] = pd.Categorical(
        df["District"],
        categories=district_order,
        ordered=True,
    )

    return (
        df[KEY_COLS + value_cols]
        .sort_values(KEY_COLS)
        .reset_index(drop=True)
    )


def load_processed_csv(
    processed_dir: Path,
    relative_path: list[str],
) -> pd.DataFrame:
    """
    Load a CSV from PROCESSED_DIR using a relative path list.
    """
    path = processed_dir.joinpath(*relative_path)
    return pd.read_csv(path)


def merge_panel_datasets(
    df_base: pd.DataFrame,
    dfs_to_merge: list[pd.DataFrame],
    key_cols: list[str] = KEY_COLS,
) -> pd.DataFrame:
    """
    Sequential left-merge of multiple panel datasets on key columns.
    """
    return reduce(
        lambda left, right: left.merge(right, on=key_cols, how="left"),
        [df_base] + dfs_to_merge,
    )


def trim_by_month(
    df: pd.DataFrame,
    end_month: str,
) -> pd.DataFrame:
    """
    Trim panel data to Month < end_month.
    """
    return df.loc[df["Month"] < end_month].reset_index(drop=True)


def save_merged_dataset(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved merged dataset to: {out_path}")


# ============================================================
# 3) Main pipeline function (Agent / API friendly)
# ============================================================

def build_merged_panel_dataset(
    processed_dir: Path,
    end_month: str = "2025-07",
) -> pd.DataFrame:
    """
    Build the final merged panel dataset for modelling.
    """

    # ---------- Base: housing price ----------
    df_price = load_processed_csv(
        processed_dir,
        ["housing_price", "HousingPrice_7districts.csv"],
    )
    df_base = standardize_panel(df_price, ["Median_Price"])

    # ---------- Other district-level variables ----------
    df_sales = standardize_panel(
        load_processed_csv(processed_dir, ["Sales_count", "Sales_Count_7districts.csv"]),
        ["sales_count"],
    )

    df_OCR = standardize_panel(
        load_processed_csv(processed_dir, ["OCR", "OCR_7districts.csv"]),
        ["OCR"],
    )

    df_MortgageRate = standardize_panel(
        load_processed_csv(processed_dir, ["MortgageInterestRate", "MortgageRate_7districts.csv"]),
        ["2YearFixedRate"],
    )

    df_CPI = standardize_panel(
        load_processed_csv(processed_dir, ["CPI", "CPI_7districts_monthly.csv"]),
        ["CPI"],
    )

    df_CGPI_Dwelling = standardize_panel(
        load_processed_csv(processed_dir, ["CGPI_Dwelling", "CGPI_Dwelling_monthly.csv"]),
        ["CGPI_Dwelling"],
    )

    df_consents = standardize_panel(
        load_processed_csv(processed_dir, ["Building_consents", "AllDistricts_consents.csv"]),
        ["Consents"],
    )

    df_immigration = standardize_panel(
        load_processed_csv(processed_dir, ["Net_immigration", "Net_migration_monthly.csv"]),
        ["Net_migration_monthly"],
    )

    df_weekly_rent = standardize_panel(
        load_processed_csv(processed_dir, ["Average_rent_income", "Monthly_Weeklyrent.csv"]),
        ["weeklyrent"],
    )

    df_unemployment = standardize_panel(
        load_processed_csv(processed_dir, ["unemployment_rate", "Unemployment_ratecl.csv"]),
        ["unemployment_rate"],
    )

    dfs_district = [
        df_sales,
        df_OCR,
        df_MortgageRate,
        df_CPI,
        df_CGPI_Dwelling,
        df_consents,
        df_immigration,
        df_weekly_rent,
        df_unemployment,
    ]

    # ---------- Merge ----------
    df_merged = merge_panel_datasets(df_base, dfs_district)

    # ---------- Trim time range ----------
    df_merged = trim_by_month(df_merged, end_month=end_month)

    return df_merged


# ============================================================
# 4) main (script entry point)
# ============================================================

def main() -> None:
    from src.config import PROCESSED_DIR

    df_merged = build_merged_panel_dataset(
        processed_dir=PROCESSED_DIR,
        end_month="2025-07",
    )

    print(df_merged.head(10).to_string(index=False))
    df_merged.info()

    output_path = PROCESSED_DIR / "merged_dataset" / "merged_dataset1.csv"
    save_merged_dataset(df_merged, output_path)


if __name__ == "__main__":
    main()





