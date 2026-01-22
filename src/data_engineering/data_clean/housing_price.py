from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1) Core functions
# ============================================================

def build_7districts_housing_price(
    xlsx_path: Path,
    district_map: dict[str, str],
) -> pd.DataFrame:
    """
    Build monthly median housing prices for 7 districts.
    Return columns: ['Month', 'District', 'Median_Price']
    """
    df = pd.read_excel(xlsx_path)

    if not {"Date", "Area", "Median_Price"}.issubset(df.columns):
        raise ValueError("Input file must contain Date, Area, Median_Price")

    df["Month"] = df["Date"].dt.to_period("M").astype(str)  # type: ignore
    df = df.rename(columns={"Area": "District"})

    df = (
        df[["Month", "District", "Median_Price"]]
        .sort_values(["Month", "District"])
        .reset_index(drop=True)
    )

    df["District"] = df["District"].replace(district_map)
    return df


def build_auckland_region_price(
    xlsx_path: Path,
) -> pd.DataFrame:
    """
    Build monthly median housing prices for Auckland region (overall).
    Return columns: ['Month', 'Area', 'Median_Price']
    """
    df = pd.read_excel(xlsx_path)

    if not {"Date", "Median_Price"}.issubset(df.columns):
        raise ValueError("Input file must contain Date, Median_Price")

    df["Month"] = df["Date"].dt.to_period("M").astype(str)  # type: ignore
    df["Area"] = "AucklandRegion"

    df = (
        df[["Month", "Area", "Median_Price"]]
        .sort_values(["Month"])
        .reset_index(drop=True)
    )
    return df


def save_dataframe(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) main
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "housing_price"
    base_out = PROCESSED_DIR / "housing_price"
    base_out.mkdir(parents=True, exist_ok=True)

    # ---------- 7 districts ----------
    housing_price_path = base_in / "7districts_price.xlsx"

    district_map = {
        "Auckland_City": "AucklandCity",
        "Franklin_District": "Franklin",
        "Manukau_City": "Manukau",
        "NorthShore_City": "NorthShore",
        "Papakura_District": "Papakura",
        "Rodney_District": "Rodney",
        "Waitakere_City": "Waitakere",
    }

    df_7districts = build_7districts_housing_price(
        xlsx_path=housing_price_path,
        district_map=district_map,
    )

    print(df_7districts.head())

    output_7districts_path = base_out / "HousingPrice_7districts.csv"
    save_dataframe(df_7districts, output_7districts_path)

    # ---------- Auckland region overall ----------
    auckland_region_path = base_in / "Aucklandregion_price.xlsx"

    df_auckland_region = build_auckland_region_price(
        xlsx_path=auckland_region_path,
    )

    print(df_auckland_region.head())

    output_region_path = base_out / "AucklandRegion_Price.csv"
    save_dataframe(df_auckland_region, output_region_path)


if __name__ == "__main__":
    main()
