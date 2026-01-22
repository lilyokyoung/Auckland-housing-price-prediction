from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1) Core functions
# ============================================================

def load_cgpi_dwelling_quarterly(xlsx_path: Path) -> pd.DataFrame:
    """
    Load quarterly CGPI dwelling data and convert 'Date' like '2006Q1' into
    quarter start Timestamp. Return columns: ['Quarter', 'CGPI_Dwelling'].
    """
    df = pd.read_excel(xlsx_path)

    if "Date" not in df.columns:
        raise ValueError(f"Missing 'Date' column in {xlsx_path.name}")
    if "CGPI_Dwelling" not in df.columns:
        raise ValueError(f"Missing 'CGPI_Dwelling' column in {xlsx_path.name}")

    # Convert quarter string like "2006Q1" -> Timestamp (start of quarter)
    df["Quarter"] = pd.PeriodIndex(df["Date"].astype(str), freq="Q").to_timestamp(how="start")
    df = df[["Quarter", "CGPI_Dwelling"]].sort_values("Quarter").reset_index(drop=True)
    return df


def expand_quarterly_to_districts(
    df_quarterly: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """
    Expand a national quarterly series to 7 districts.
    Return columns: ['Quarter', 'District', 'CGPI_Dwelling'].
    """
    if "Quarter" not in df_quarterly.columns or "CGPI_Dwelling" not in df_quarterly.columns:
        raise ValueError("df_quarterly must contain ['Quarter', 'CGPI_Dwelling']")

    df_7 = (
        df_quarterly.assign(key=1)
        .merge(pd.DataFrame({"District": districts, "key": 1}), on="key")
        .drop(columns="key")
        .sort_values(["Quarter", "District"])
        .reset_index(drop=True)
    )
    return df_7[["Quarter", "District", "CGPI_Dwelling"]].copy()


def quarterly_to_monthly_interpolate(
    df_quarterly_7: pd.DataFrame,
    district_order: list[str],
) -> pd.DataFrame:
    """
    For each district's quarterly series, resample to month start (MS) and
    linear interpolate.
    Output columns: ['Month', 'District', 'CGPI_Dwelling'] with Month='YYYY-MM'.
    """
    required = {"Quarter", "District", "CGPI_Dwelling"}
    if not required.issubset(df_quarterly_7.columns):
        raise ValueError(f"df_quarterly_7 must contain columns: {required}")

    df_monthly = (
        df_quarterly_7
        .set_index("Quarter")
        .groupby("District")["CGPI_Dwelling"]
        .apply(lambda s: s.resample("MS").interpolate("linear"))
        .reset_index()
    )

    # produce YYYY-MM month string
    df_monthly["Month"] = df_monthly["Quarter"].dt.to_period("M").astype(str)  # type: ignore
    df_monthly = df_monthly[["Month", "District", "CGPI_Dwelling"]].copy()
    df_monthly["CGPI_Dwelling"] = df_monthly["CGPI_Dwelling"].round(2)

    df_monthly["District"] = pd.Categorical(
        df_monthly["District"],
        categories=district_order,
        ordered=True,
    )
    df_monthly = df_monthly.sort_values(["Month", "District"]).reset_index(drop=True)
    return df_monthly


def build_cgpi_dwelling_monthly(
    xlsx_path: Path,
    districts: list[str],
    district_order: list[str],
) -> pd.DataFrame:
    """
    Full pipeline:
    xlsx -> quarterly -> expand to districts -> monthly interpolation -> tidy monthly df.
    """
    df_q = load_cgpi_dwelling_quarterly(xlsx_path)
    df_q7 = expand_quarterly_to_districts(df_q, districts)
    df_m = quarterly_to_monthly_interpolate(df_q7, district_order)
    return df_m


def save_cgpi_dwelling_monthly(df_monthly: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_monthly.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) main
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "CGPI_Dwelling"
    base_out = PROCESSED_DIR / "CGPI_Dwelling"
    base_out.mkdir(parents=True, exist_ok=True)

    xlsx_path = base_in / "CGPI_Dwelling_units.xlsx"

    districts = [
        "AucklandCity",
        "Franklin",
        "Manukau",
        "NorthShore",
        "Papakura",
        "Rodney",
        "Waitakere",
    ]
    district_order = [
        "AucklandCity",
        "Franklin",
        "Manukau",
        "NorthShore",
        "Papakura",
        "Rodney",
        "Waitakere",
    ]

    df_monthly = build_cgpi_dwelling_monthly(
        xlsx_path=xlsx_path,
        districts=districts,
        district_order=district_order,
    )

    print(df_monthly.head(14))

    out_path = base_out / "CGPI_Dwelling_monthly.csv"
    save_cgpi_dwelling_monthly(df_monthly, out_path)


if __name__ == "__main__":
    main()
