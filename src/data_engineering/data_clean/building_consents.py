from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ============================================================
# 1) Helpers：通用清洗（Date: 2020M01 -> Month: 2020-01）
# ============================================================

def date_to_month_col(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Keep only rows with Date like 'YYYYMmm', convert to Month='YYYY-MM',
    drop Date column, sort by Month.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' column")

    mask = df[date_col].astype(str).str.match(r"^\d{4}M\d{2}$", na=False)
    df = df.loc[mask].copy()

    tmp = df[date_col].astype(str).str.replace("M", "-", regex=False)
    months: pd.Series = pd.to_datetime(tmp, format="%Y-%m")
    df["Month"] = months.dt.strftime("%Y-%m")  # type: ignore

    df = df.sort_values("Month").reset_index(drop=True)
    df = df.drop(columns=[date_col])
    return df


def save_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) District-level builders：各区处理（sum boards or take single col）
# ============================================================

def build_sum_boards_consents(
    xlsx_path: Path,
    boards: List[str],
    out_col: str,
) -> pd.DataFrame:
    """
    Read xlsx, convert Date->Month, then sum specified board columns into out_col.
    Return columns: ['Month', out_col]
    """
    df = pd.read_excel(xlsx_path)
    df = date_to_month_col(df, date_col="Date")

    missing = [c for c in boards if c not in df.columns]
    if missing:
        raise ValueError(f"Missing board columns {missing} in {xlsx_path.name}")

    df[out_col] = df[boards].sum(axis=1)
    return df[["Month", out_col]].copy()


def build_single_col_consents(
    xlsx_path: Path,
    value_col: str,
) -> pd.DataFrame:
    """
    Read xlsx, convert Date->Month, then keep columns ['Month', value_col]
    """
    df = pd.read_excel(xlsx_path)
    df = date_to_month_col(df, date_col="Date")

    if value_col not in df.columns:
        raise ValueError(f"Missing '{value_col}' in {xlsx_path.name}")

    return df[["Month", value_col]].copy()


# ============================================================
# 3) Main pipeline functions：清洗各区 + 保存 + 合并
# ============================================================

def build_and_save_district_consents(
    base_in: Path,
    base_out: Path,
) -> Dict[str, Path]:
    """
    Build each district consents CSV and save to processed dir.
    Return mapping: District -> cleaned_csv_path
    """
    base_out.mkdir(parents=True, exist_ok=True)

    # Auckland City
    auckland_city_boards = ["Waitemata", "Whau", "AlbertEden", "Puketapapa", "Orakei", "MaungakiekieTamaki"]
    df_auck = build_sum_boards_consents(
        xlsx_path=base_in / "Auckland_city_consents.xlsx",
        boards=auckland_city_boards,
        out_col="AucklandCity_consents",
    )
    auck_path = base_out / "AucklandCity_consentscl.csv"
    save_df(df_auck, auck_path)

    # Franklin
    df_fr = build_single_col_consents(
        xlsx_path=base_in / "Franklin_consents.xlsx",
        value_col="Franklin",
    )
    fr_path = base_out / "Franklin_consentscl.csv"
    save_df(df_fr, fr_path)

    # Manukau
    manukau_boards = ["Howick", "MangereOtahuhu", "OtaraPapatoetoe", "Manurewa"]
    df_mk = build_sum_boards_consents(
        xlsx_path=base_in / "Manukau_Consents.xlsx",
        boards=manukau_boards,
        out_col="Manukau_consents",
    )
    mk_path = base_out / "Manukau_consentscl.csv"
    save_df(df_mk, mk_path)

    # North Shore
    northshore_boards = ["Hibiscus_and_Bays", "UpperHarbour", "Kaipatiki", "DevonportTakapuna"]
    df_ns = build_sum_boards_consents(
        xlsx_path=base_in / "North_Shore_consents.xlsx",
        boards=northshore_boards,
        out_col="NorthShore_consents",
    )
    ns_path = base_out / "NorthShore_consentscl.csv"
    save_df(df_ns, ns_path)

    # Waitakere
    waitakere_boards = ["HendersonMassey", "WaitakereRanges"]
    df_wk = build_sum_boards_consents(
        xlsx_path=base_in / "Waitakere_consents.xlsx",
        boards=waitakere_boards,
        out_col="Waitakere_consents",
    )
    wk_path = base_out / "Waitakere_consentscl.csv"
    save_df(df_wk, wk_path)

    # Rodney
    df_rd = build_single_col_consents(
        xlsx_path=base_in / "Rodney_consents.xlsx",
        value_col="Rodney",
    )
    rd_path = base_out / "Rodney_consentscl.csv"
    save_df(df_rd, rd_path)

    # Papakura
    df_pk = build_single_col_consents(
        xlsx_path=base_in / "Papakura_consents.xlsx",
        value_col="Papakura",
    )
    pk_path = base_out / "Papakura_consentscl.csv"
    save_df(df_pk, pk_path)

    return {
        "AucklandCity": auck_path,
        "Franklin": fr_path,
        "Manukau": mk_path,
        "NorthShore": ns_path,
        "Papakura": pk_path,
        "Rodney": rd_path,
        "Waitakere": wk_path,
    }


def merge_all_districts_consents(
    cleaned_files: Dict[str, Path],
    out_path: Path,
) -> pd.DataFrame:
    """
    Read each district cleaned CSV, rename value column to 'Consents',
    add District column, and concat into:
    ['Month', 'District', 'Consents']
    """
    dfs: List[pd.DataFrame] = []

    for district, fpath in cleaned_files.items():
        df = pd.read_csv(fpath)

        # value col is the only non-Month column
        value_cols = [c for c in df.columns if c != "Month"]
        if len(value_cols) != 1:
            raise ValueError(f"Unexpected columns in {fpath.name}: {df.columns.tolist()}")

        value_col = value_cols[0]
        df = df.rename(columns={value_col: "Consents"})
        df["District"] = district
        df = df[["Month", "District", "Consents"]]
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True).sort_values(["Month", "District"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(merged.head(10))

    return merged


# ============================================================
# 4) main：只负责 orchestration（读 config + 调函数）
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "Building_consents"
    base_out = PROCESSED_DIR / "Building_consents"
    base_out.mkdir(parents=True, exist_ok=True)

    cleaned_files = build_and_save_district_consents(base_in=base_in, base_out=base_out)

    merged_out = base_out / "AllDistricts_consents.csv"
    merge_all_districts_consents(cleaned_files=cleaned_files, out_path=merged_out)


if __name__ == "__main__":
    main()



