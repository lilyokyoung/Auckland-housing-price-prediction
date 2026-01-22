from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd


# ============================================================
# 1. Low-level helpers
# ============================================================

def mean_boards_from_excel(
    xlsx_path: Path,
    year_col: str,
    board_cols: List[str],
    out_col: str,
) -> pd.DataFrame:
    """
    Read an Excel file and compute the mean across specified local-board columns.
    Return DataFrame with columns: [Year, out_col]
    """
    df = pd.read_excel(xlsx_path)

    missing = [c for c in [year_col, *board_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {xlsx_path.name}")

    df[out_col] = df[board_cols].mean(axis=1)
    return df[[year_col, out_col]].copy()


def single_district_from_excel(
    xlsx_path: Path,
    year_col: str,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Read an Excel file with a single district column (Franklin, Papakura, Rodney).
    Return DataFrame with columns: [Year, out_col]
    """
    df = pd.read_excel(xlsx_path)

    missing = [c for c in [year_col, value_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {xlsx_path.name}")

    return (
        df[[year_col, value_col]]
        .rename(columns={value_col: out_col})
        .copy()
    )


# ============================================================
# 2. Core business logic
# ============================================================

def build_weeklyrent_yearly_long(
    raw_dir: Path,
    processed_dir: Path,
    save_intermediate_csv: bool = True,
) -> pd.DataFrame:
    """
    Build yearly weekly rent (long format):
    [Year, District, weeklyrent]
    """
    out_folder = processed_dir / "Average_rent_income"
    out_folder.mkdir(parents=True, exist_ok=True)

    year_col = "Year"
    value_col = "weeklyrent"

    # -------- Auckland City --------
    auckland_city_boards = [
        "Waitemata",
        "AlbertEden",
        "Puketapapa",
        "MaungakiekieTamaki",
        "Whau",
        "Orakei",
    ]
    df_auck = mean_boards_from_excel(
        raw_dir / "Average_rent_income" / "Rentincome_AucklandCity.xlsx",
        year_col,
        auckland_city_boards,
        "AucklandCity",
    )

    # -------- Manukau --------
    manukau_boards = [
        "Howick",
        "MangereOtahuhu",
        "OtaraPapatoetoe",
        "Manurewa",
    ]
    df_manu = mean_boards_from_excel(
        raw_dir / "Average_rent_income" / "Rentincome_Manukau.xlsx",
        year_col,
        manukau_boards,
        "Manukau",
    )

    # -------- North Shore --------
    nshore_boards = [
        "Hibiscus_and_Bays",
        "UpperHarbour",
        "Kaipatiki",
        "DevonportTakapuna",
    ]
    df_ns = mean_boards_from_excel(
        raw_dir / "Average_rent_income" / "Rentincome_NorthShore.xlsx",
        year_col,
        nshore_boards,
        "NorthShore",
    )

    # -------- Waitakere --------
    waitakere_boards = [
        "HendersonMassey",
        "WaitakereRanges",
    ]
    df_wk = mean_boards_from_excel(
        raw_dir / "Average_rent_income" / "Rentincome_Waitakere.xlsx",
        year_col,
        waitakere_boards,
        "Waitakere",
    )

    # -------- Single-column districts --------
    df_fr = single_district_from_excel(
        raw_dir / "Average_rent_income" / "Rentincome_Franklin.xlsx",
        year_col,
        "Franklin",
        "Franklin",
    )
    df_pk = single_district_from_excel(
        raw_dir / "Average_rent_income" / "Rentincome_Papakura.xlsx",
        year_col,
        "Papakura",
        "Papakura",
    )
    df_rd = single_district_from_excel(
        raw_dir / "Average_rent_income" / "Rentincome_Rodney.xlsx",
        year_col,
        "Rodney",
        "Rodney",
    )

    # -------- Optional: save intermediate CSVs（和你原来一致，便于核对） --------
    if save_intermediate_csv:
        df_auck.to_csv(out_folder / "Auckrentcl.csv", index=False, float_format="%.2f")
        df_manu.to_csv(out_folder / "Manukaurentcl.csv", index=False, float_format="%.2f")
        df_ns.to_csv(out_folder / "NShorerentcl.csv", index=False, float_format="%.2f")
        df_wk.to_csv(out_folder / "Waitakererentcl.csv", index=False, float_format="%.2f")

    # -------- Convert to long format --------
    def to_long(df: pd.DataFrame, district: str, col: str) -> pd.DataFrame:
        out = df[[year_col, col]].rename(columns={col: value_col}).copy()
        out["District"] = district
        return out

    df_long = pd.concat(
        [
            to_long(df_auck, "AucklandCity", "AucklandCity"),
            to_long(df_fr, "Franklin", "Franklin"),
            to_long(df_manu, "Manukau", "Manukau"),
            to_long(df_ns, "NorthShore", "NorthShore"),
            to_long(df_pk, "Papakura", "Papakura"),
            to_long(df_rd, "Rodney", "Rodney"),
            to_long(df_wk, "Waitakere", "Waitakere"),
        ],
        ignore_index=True,
    )

    district_order = [
        "AucklandCity",
        "Franklin",
        "Manukau",
        "NorthShore",
        "Papakura",
        "Rodney",
        "Waitakere",
    ]

    df_long["Year"] = df_long["Year"].astype(int)
    df_long["District"] = pd.Categorical(
        df_long["District"],
        categories=district_order,
        ordered=True,
    )

    df_long = (
        df_long
        .sort_values(["Year", "District"])
        .reset_index(drop=True)
        [["Year", "District", value_col]]
    )

    df_long.to_csv(out_folder / "Weeklyrent_merged.csv", index=False)
    return df_long


def expand_yearly_to_monthly(
    df_yearly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand yearly weekly rent into monthly frequency.
    Return DataFrame: [Month, District, weeklyrent]
    """
    df = df_yearly.copy()

    df["Month"] = df["Year"].map(
        lambda y: pd.period_range(f"{y}-01", f"{y}-12", freq="M").tolist()
    )

    df = df.explode("Month", ignore_index=True)
    df["Month"] = df["Month"].astype(str)
    df["weeklyrent"] = df["weeklyrent"].round(2)

    district_order = [
        "AucklandCity",
        "Franklin",
        "Manukau",
        "NorthShore",
        "Papakura",
        "Rodney",
        "Waitakere",
    ]
    df["District"] = pd.Categorical(
        df["District"],
        categories=district_order,
        ordered=True,
    )

    return (
        df[["Month", "District", "weeklyrent"]]
        .sort_values(["Month", "District"])
        .reset_index(drop=True)
    )


def save_monthly_weeklyrent(
    df_monthly: pd.DataFrame,
    processed_dir: Path,
) -> Path:
    out_folder = processed_dir / "Average_rent_income"
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / "Monthly_Weeklyrent.csv"
    df_monthly.to_csv(out_path, index=False)
    return out_path


# ============================================================
# 3. main（只负责 orchestration & I/O）
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    df_yearly = build_weeklyrent_yearly_long(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        save_intermediate_csv=True,
    )

    df_monthly = expand_yearly_to_monthly(df_yearly)
    out_path = save_monthly_weeklyrent(df_monthly, PROCESSED_DIR)

    print(df_yearly.head())
    print(df_monthly.head())
    print(f"Saved monthly weekly rent file to: {out_path}")


if __name__ == "__main__":
    main()
