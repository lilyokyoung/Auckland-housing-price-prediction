from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


DISTRICT_ORDER = [
    "AucklandCity",
    "Franklin",
    "Manukau",
    "NorthShore",
    "Papakura",
    "Rodney",
    "Waitakere",
]


# ============================================================
# Common utilities
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_excel(xlsx_path: Path, header: int = 1) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, header=header)


def save_csv(df: pd.DataFrame, out_path: Path, float_format: str | None = None) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, float_format=float_format)
    print(f"Saved file to: {out_path}")


def clean_year_from_date_column(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    For files where 'Date' may include non-year rows, keep only 4-digit years.
    """
    df = df.copy()
    mask = df[date_col].astype(str).str.match(r"^\d{4}$", na=False)
    df = df.loc[mask].copy()
    df[date_col] = df[date_col].astype(int)
    df = df.rename(columns={date_col: "Year"})
    return df


def standardize_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    For files where 'Date' should be treated as year but may include NaNs / messy strings.
    """
    df = df.copy()
    df = df.rename(columns={"Date": "Year"})
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    return df


def convert_comma_number_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert columns like '12,345' to float.
    """
    df = df.copy()
    for c in cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    return df


def sum_boards(df: pd.DataFrame, boards: list[str], out_col: str) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df[boards].sum(axis=1)
    return df


def to_long_annual(df: pd.DataFrame, value_col: str, district: str) -> pd.DataFrame:
    out = df[["Year", value_col]].rename(columns={value_col: "population"}).copy()
    out["District"] = district
    return out[["Year", "District", "population"]]


def sort_district_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["District"] = pd.Categorical(df["District"].astype(str), categories=DISTRICT_ORDER, ordered=True)
    return df.sort_values(["Year", "District"]).reset_index(drop=True)


def sort_district_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["District"] = pd.Categorical(df["District"].astype(str), categories=DISTRICT_ORDER, ordered=True)
    return df.sort_values(["Month", "District"]).reset_index(drop=True)


# ============================================================
# District builders (annual cleaned)
# ============================================================

def build_aucklandcity_annual(base_in: Path, base_out: Path) -> Path:
    boards = ["Waitemata", "Whau", "AlbertEden", "Puketapapa", "Orakei", "MaungakiekieTamaki"]
    in_path = base_in / "AucklandCity_Population.xlsx"
    out_path = base_out / "AucklandCity_populationcl.csv"

    df = load_excel(in_path, header=1)
    print(df.head())

    df = standardize_year_column(df)
    df = convert_comma_number_cols(df, boards)
    df = sum_boards(df, boards, out_col="AucklandCity")

    print(df.head())
    save_csv(df, out_path)
    return out_path


def build_franklin_annual(base_in: Path, base_out: Path) -> Path:
    in_path = base_in / "Franklin_District_population.xlsx"
    out_path = base_out / "Franklin_populationcl.csv"

    df = load_excel(in_path, header=1)
    print(df.head())
    print(df.info())

    df = clean_year_from_date_column(df, date_col="Date")

    print(df)
    save_csv(df, out_path)
    return out_path


def build_manukau_annual(base_in: Path, base_out: Path) -> Path:
    boards = ["Howick", "MangereOtahuhu", "OtaraPapatoetoe", "Manurewa"]
    in_path = base_in / "Manukau_City_Population.xlsx"
    out_path = base_out / "Manukau_populationcl.csv"

    df = load_excel(in_path, header=1)
    print(df.head())

    df = standardize_year_column(df)
    print(df.info())

    df = sum_boards(df, boards, out_col="Manukau")
    print(df.head())

    save_csv(df, out_path)
    return out_path


def build_northshore_annual(base_in: Path, base_out: Path) -> Path:
    boards = ["Hibiscus_and_Bays", "UpperHarbour", "Kaipatiki", "DevonportTakapuna"]
    in_path = base_in / "North_Shore_City population.xlsx"
    out_path = base_out / "NorthShore_populationcl.csv"

    df = load_excel(in_path, header=1)
    print(df.head())

    df = standardize_year_column(df)
    print(df.info())

    df = sum_boards(df, boards, out_col="NorthShore")
    print(df.head())

    save_csv(df, out_path)
    return out_path


def build_papakura_annual(base_in: Path, base_out: Path) -> Path:
    in_path = base_in / "Papakura_District_Population.xlsx"
    out_path = base_out / "Papakura_populationcl.csv"

    df = load_excel(in_path, header=1)
    print(df.head())

    df = clean_year_from_date_column(df, date_col="Date")
    print(df)

    save_csv(df, out_path)
    return out_path


def build_rodney_annual(base_in: Path, base_out: Path) -> Path:
    in_path = base_in / "Rodney_District_Population.xlsx"
    out_path = base_out / "Rodney_populationcl.csv"

    df = load_excel(in_path, header=1)
    print(df.head())

    df = clean_year_from_date_column(df, date_col="Date")
    print(df)

    save_csv(df, out_path)
    return out_path


def build_waitakere_annual(base_in: Path, base_out: Path) -> Path:
    boards = ["HendersonMassey", "WaitakereRanges"]
    in_path = base_in / "Waitakere_City_population.xlsx"
    out_path = base_out / "Waitakere_populationcl.csv"

    df = load_excel(in_path, header=1)
    print(df.head())

    df = standardize_year_column(df)
    print(df.info())

    df = sum_boards(df, boards, out_col="Waitakere")
    print(df.head())

    save_csv(df, out_path)
    return out_path


# ============================================================
# Merge annual -> long -> monthly interpolation
# ============================================================

def build_population_long_from_cleaned_csvs(base_out: Path) -> pd.DataFrame:
    """
    Read the cleaned annual CSVs (saved in base_out) and build a unified long table:
    Year, District, population
    """
    paths = {
        "AucklandCity": base_out / "AucklandCity_populationcl.csv",
        "Franklin": base_out / "Franklin_populationcl.csv",
        "Manukau": base_out / "Manukau_populationcl.csv",
        "NorthShore": base_out / "NorthShore_populationcl.csv",
        "Papakura": base_out / "Papakura_populationcl.csv",
        "Rodney": base_out / "Rodney_populationcl.csv",
        "Waitakere": base_out / "Waitakere_populationcl.csv",
    }

    # value column names in each cleaned file
    value_cols = {
        "AucklandCity": "AucklandCity",
        "Franklin": "Franklin",
        "Manukau": "Manukau",
        "NorthShore": "NorthShore",
        "Papakura": "Papakura",
        "Rodney": "Rodney",
        "Waitakere": "Waitakere",
    }

    long_parts: list[pd.DataFrame] = []
    for district, p in paths.items():
        df = pd.read_csv(p)
        long_parts.append(to_long_annual(df, value_col=value_cols[district], district=district))

    merged = pd.concat(long_parts, ignore_index=True)
    merged = sort_district_year(merged)
    return merged


def save_population_annual_merged(df_long: pd.DataFrame, base_out: Path) -> Path:
    out_path = base_out / "population_mergedcl.csv"
    save_csv(df_long, out_path)
    print(df_long.head())
    return out_path


def annual_to_monthly_linear(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each district: Yearly -> Month start (MS) with linear interpolation.
    """
    monthly_parts: list[pd.DataFrame] = []

    # Use string for District categories; keep order stable
    districts = [d for d in DISTRICT_ORDER if d in df_long["District"].astype(str).unique().tolist()]

    for district in districts:
        df_d = df_long.loc[df_long["District"].astype(str) == district, ["Year", "population"]].copy()
        df_d["Date"] = pd.to_datetime(df_d["Year"].astype(int).astype(str) + "-01-01")
        df_d = df_d.set_index("Date").sort_index()

        df_m = df_d.resample("MS").interpolate(method="linear").reset_index()
        df_m["Month"] = df_m["Date"].dt.to_period("M").astype(str)  # type: ignore
        df_m["District"] = district

        monthly_parts.append(df_m[["Month", "District", "population"]])

    out = pd.concat(monthly_parts, ignore_index=True)
    out = sort_district_month(out)
    return out


def save_population_monthly(df_monthly: pd.DataFrame, base_out: Path) -> Path:
    out_path = base_out / "population_monthlycl.csv"
    save_csv(df_monthly, out_path, float_format="%.2f")
    print(df_monthly.head())
    return out_path


# ============================================================
# High-level runner
# ============================================================

def run_population_pipeline(base_in: Path, base_out: Path) -> dict[str, Path]:
    """
    1) Build cleaned annual CSV for each district
    2) Merge annual -> long
    3) Annual long -> monthly linear interpolation
    """
    ensure_dir(base_out)

    outputs: dict[str, Path] = {}
    outputs["aucklandcity_annual"] = build_aucklandcity_annual(base_in, base_out)
    outputs["franklin_annual"] = build_franklin_annual(base_in, base_out)
    outputs["manukau_annual"] = build_manukau_annual(base_in, base_out)
    outputs["northshore_annual"] = build_northshore_annual(base_in, base_out)
    outputs["papakura_annual"] = build_papakura_annual(base_in, base_out)
    outputs["rodney_annual"] = build_rodney_annual(base_in, base_out)
    outputs["waitakere_annual"] = build_waitakere_annual(base_in, base_out)

    df_long = build_population_long_from_cleaned_csvs(base_out)
    outputs["population_annual_merged"] = save_population_annual_merged(df_long, base_out)

    df_monthly = annual_to_monthly_linear(df_long)
    outputs["population_monthly"] = save_population_monthly(df_monthly, base_out)

    return outputs


# ============================================================
# main
# ============================================================

def main() -> None:
    base_in = RAW_DIR / "population"
    base_out = PROCESSED_DIR / "population"

    outputs = run_population_pipeline(base_in=base_in, base_out=base_out)

    for k, p in outputs.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()

