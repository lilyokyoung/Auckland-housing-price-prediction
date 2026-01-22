from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1) Core functions
# ============================================================

def load_net_migration_yearly(
    xlsx_path: Path,
    district_map: dict[str, str],
) -> pd.DataFrame:
    """
    Load yearly net migration data, rename Area->District, keep ['Year','District','Net_migration'],
    and map district names.
    """
    df = pd.read_excel(xlsx_path)

    # Basic validation
    if "Area" in df.columns and "District" not in df.columns:
        df = df.rename(columns={"Area": "District"})

    required = {"Year", "District", "Net_migration"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required}. Got: {df.columns.tolist()}")

    df_clean = df[["Year", "District", "Net_migration"]].copy()
    df_clean["District"] = df_clean["District"].replace(district_map)

    return df_clean


def save_yearly_clean(df_yearly: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_yearly.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


def apportion_yearly_to_monthly(df_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yearly net migration to monthly by equal apportionment (divide by 12),
    and expand Year -> 12 months.
    Output: ['Month','District','Net_migration_monthly']
    """
    df = df_yearly.copy()

    # Ensure Year is int-like
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Expand to months
    df["Month"] = df["Year"].apply(lambda y: list(pd.period_range(f"{y}-01", f"{y}-12", freq="M")))
    df_m = df.explode("Month").reset_index(drop=True)

    df_m["Net_migration_monthly"] = df_m["Net_migration"] / 12
    df_m["Month"] = df_m["Month"].astype(str)

    df_out = (
        df_m[["Month", "District", "Net_migration_monthly"]]
        .sort_values(["Month", "District"])
        .reset_index(drop=True)
    )
    df_out["Net_migration_monthly"] = df_out["Net_migration_monthly"].round(2)

    return df_out


def save_monthly(df_monthly: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_monthly.to_csv(out_path, index=False)
    print(f"Saved monthly file to: {out_path}")


# ============================================================
# 2) main
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "Net_immigration"
    base_out = PROCESSED_DIR / "Net_immigration"
    base_out.mkdir(parents=True, exist_ok=True)

    net_immigration_path = base_in / "Net_migration.xlsx"

    district_map = {
        "Auckland_City": "AucklandCity",
        "Franklin_District": "Franklin",
        "Manukau_City": "Manukau",
        "NorthShore_City": "NorthShore",
        "Papakura_District": "Papakura",
        "Rodney_District": "Rodney",
        "Waitakere_City": "Waitakere",
    }

    df_yearly = load_net_migration_yearly(
        xlsx_path=net_immigration_path,
        district_map=district_map,
    )
    print(df_yearly.head())

    yearly_out = base_out / "Net_migrationcl.csv"
    save_yearly_clean(df_yearly, yearly_out)

    df_monthly = apportion_yearly_to_monthly(df_yearly)
    print(df_monthly.head())

    monthly_out = base_out / "Net_migration_monthly.csv"
    save_monthly(df_monthly, monthly_out)


if __name__ == "__main__":
    main()
