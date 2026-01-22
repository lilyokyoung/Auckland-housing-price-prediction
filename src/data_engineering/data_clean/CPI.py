from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1) Core functions
# ============================================================

def load_cpi_quarterly(xlsx_path: Path) -> pd.DataFrame:
    """
    Load quarterly CPI data and convert 'Date' like '2006Q1'
    to quarter start Timestamp.
    Return columns: ['Quarter', 'CPI']
    """
    df = pd.read_excel(xlsx_path)

    if "Date" not in df.columns or "CPI" not in df.columns:
        raise ValueError("CPI file must contain 'Date' and 'CPI' columns")

    df["Quarter"] = pd.PeriodIndex(
        df["Date"].astype(str),
        freq="Q"
    ).to_timestamp(how="start")

    df = (
        df[["Quarter", "CPI"]]
        .sort_values("Quarter")
        .reset_index(drop=True)
    )
    return df


def expand_quarterly_to_districts(
    df_quarterly: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """
    Expand a quarterly CPI series to all districts.
    Return columns: ['Quarter', 'District', 'CPI']
    """
    df_7 = (
        df_quarterly.assign(key=1)
        .merge(pd.DataFrame({"District": districts, "key": 1}), on="key")
        .drop(columns="key")
        .sort_values(["Quarter", "District"])
        .reset_index(drop=True)
    )
    return df_7[["Quarter", "District", "CPI"]].copy()


def quarterly_to_monthly_interpolate(
    df_quarterly_7: pd.DataFrame,
    district_order: list[str],
) -> pd.DataFrame:
    """
    Convert quarterly CPI to monthly frequency using linear interpolation.
    Return columns: ['Month', 'District', 'CPI']
    """
    df_monthly = (
        df_quarterly_7
        .set_index("Quarter")
        .groupby("District")["CPI"]
        .resample("MS")
        .interpolate("linear")
        .reset_index()
    )

    df_monthly["Month"] = (
        df_monthly["Quarter"]
        .dt.to_period("M") # type: ignore
        .astype(str)  # type: ignore
    )

    df_monthly = df_monthly[["Month", "District", "CPI"]]
    df_monthly["CPI"] = df_monthly["CPI"].round(2)

    df_monthly["District"] = pd.Categorical(
        df_monthly["District"],
        categories=district_order,
        ordered=True,
    )

    df_monthly = (
        df_monthly
        .sort_values(["Month", "District"])
        .reset_index(drop=True)
    )
    return df_monthly


def build_cpi_monthly(
    xlsx_path: Path,
    districts: list[str],
    district_order: list[str],
) -> pd.DataFrame:
    """
    Full CPI pipeline:
    xlsx -> quarterly -> expand districts -> monthly interpolation
    """
    df_q = load_cpi_quarterly(xlsx_path)
    df_q7 = expand_quarterly_to_districts(df_q, districts)
    df_m = quarterly_to_monthly_interpolate(df_q7, district_order)
    return df_m


def save_cpi_monthly(df_monthly: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_monthly.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) main
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "CPI"
    base_out = PROCESSED_DIR / "CPI"
    base_out.mkdir(parents=True, exist_ok=True)

    cpi_path = base_in / "CPI_clean.xlsx"

    districts = [
        "AucklandCity",
        "Franklin",
        "Manukau",
        "NorthShore",
        "Papakura",
        "Rodney",
        "Waitakere",
    ]
    district_order = districts.copy()

    df_cpi_monthly = build_cpi_monthly(
        xlsx_path=cpi_path,
        districts=districts,
        district_order=district_order,
    )

    print(df_cpi_monthly.head(14))

    output_path = base_out / "CPI_7districts_monthly.csv"
    save_cpi_monthly(df_cpi_monthly, output_path)


if __name__ == "__main__":
    main()
