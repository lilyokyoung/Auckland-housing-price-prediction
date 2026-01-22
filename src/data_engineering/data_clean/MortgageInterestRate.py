from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1) Core functions
# ============================================================

def load_mortgage_rate_monthly(
    xlsx_path: Path,
) -> pd.DataFrame:
    """
    Load mortgage interest rate data and convert Date to Month period.
    Return DataFrame with columns: ['Month', '2YearFixedRate']
    """
    df = pd.read_excel(xlsx_path)

    if not {"Date", "2YearFixedRate"}.issubset(df.columns):
        raise ValueError("Input file must contain 'Date' and '2YearFixedRate'")

    df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    df = df.drop(columns="Date")

    return df[["Month", "2YearFixedRate"]].copy()


def expand_monthly_to_districts(
    df_monthly: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """
    Expand a monthly mortgage rate series to all districts.
    Return columns: ['Month', 'District', '2YearFixedRate']
    """
    df_7 = (
        df_monthly
        .assign(key=1)
        .merge(pd.DataFrame({"District": districts, "key": 1}), on="key")
        .drop(columns="key")
        .sort_values(["Month", "District"])
        .reset_index(drop=True)
    )
    return df_7[["Month", "District", "2YearFixedRate"]].copy()


def build_mortgage_rate_7districts(
    xlsx_path: Path,
    districts: list[str],
) -> pd.DataFrame:
    """
    Full pipeline:
    xlsx -> monthly mortgage rate -> expand to 7 districts
    """
    df_monthly = load_mortgage_rate_monthly(xlsx_path)
    df_7 = expand_monthly_to_districts(df_monthly, districts)
    return df_7


def save_mortgage_rate(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) main
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "MortgageInterestRate"
    base_out = PROCESSED_DIR / "MortgageInterestRate"
    base_out.mkdir(parents=True, exist_ok=True)

    mortgage_rate_path = base_in / "MortgageInterestRate.xlsx"

    districts = [
        "AucklandCity",
        "Franklin",
        "Manukau",
        "NorthShore",
        "Papakura",
        "Rodney",
        "Waitakere",
    ]

    df_mortgage_7 = build_mortgage_rate_7districts(
        xlsx_path=mortgage_rate_path,
        districts=districts,
    )

    print(df_mortgage_7.head(10))

    output_path = base_out / "MortgageRate_7districts.csv"
    save_mortgage_rate(df_mortgage_7, output_path)


if __name__ == "__main__":
    main()
