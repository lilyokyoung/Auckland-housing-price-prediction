from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1) Core functions
# ============================================================

def build_7districts_sales_count(
    xlsx_path: Path,
    district_map: dict[str, str],
) -> pd.DataFrame:
    """
    Build monthly sales_count for 7 districts.
    Return columns: ['Month', 'District', 'sales_count']
    """
    df = pd.read_excel(xlsx_path)

    required = {"Date", "Area", "Sales_Count"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required}")

    df["Month"] = df["Date"].dt.to_period("M").astype(str)  # type: ignore
    df = df.rename(columns={"Area": "District"})

    df_out = (
        df[["Month", "District", "Sales_Count"]]
        .rename(columns={"Sales_Count": "sales_count"})
        .sort_values(["Month", "District"])
        .reset_index(drop=True)
    )

    df_out["District"] = df_out["District"].replace(district_map)
    return df_out


def build_auckland_region_sales_count(
    xlsx_path: Path,
) -> pd.DataFrame:
    """
    Build monthly sales_count for Auckland region (overall).
    Return columns: ['Month', 'Area', 'sales_count']
    """
    df = pd.read_excel(xlsx_path)

    required = {"Date", "Sales_Count"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required}")

    df["Month"] = df["Date"].dt.to_period("M").astype(str)  # type: ignore
    df["Area"] = "AucklandRegion"

    df_out = (
        df[["Month", "Area", "Sales_Count"]]
        .rename(columns={"Sales_Count": "sales_count"})
        .sort_values(["Month"])
        .reset_index(drop=True)
    )
    return df_out


def save_dataframe(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) main
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "Sales_count"
    base_out = PROCESSED_DIR / "Sales_count"
    base_out.mkdir(parents=True, exist_ok=True)

    # ---------- 7 districts ----------
    sales_7_path = base_in / "7districts_price.xlsx"

    district_map = {
        "Auckland_City": "AucklandCity",
        "Franklin_District": "Franklin",
        "Manukau_City": "Manukau",
        "NorthShore_City": "NorthShore",
        "Papakura_District": "Papakura",
        "Rodney_District": "Rodney",
        "Waitakere_City": "Waitakere",
    }

    df_7 = build_7districts_sales_count(
        xlsx_path=sales_7_path,
        district_map=district_map,
    )
    print(df_7.head())

    out_7 = base_out / "Sales_Count_7districts.csv"
    save_dataframe(df_7, out_7)

    # ---------- Auckland region ----------
    sales_region_path = base_in / "Aucklandregion_price.xlsx"

    df_region = build_auckland_region_sales_count(
        xlsx_path=sales_region_path,
    )
    print(df_region.head())

    out_region = base_out / "AucklandRegion_Salescl.csv"
    save_dataframe(df_region, out_region)


if __name__ == "__main__":
    main()
