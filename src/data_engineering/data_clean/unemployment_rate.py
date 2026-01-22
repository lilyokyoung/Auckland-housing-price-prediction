from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1) Core functions
# ============================================================

def load_unemployment_quarterly(
    xlsx_path: Path,
) -> pd.DataFrame:
    """
    Load quarterly unemployment rate for Auckland.
    Expect columns: ['Quarter', 'Auckland']
    Return columns: ['Quarter', 'unemployment_rate']
    """
    df = pd.read_excel(xlsx_path)

    required = {"Quarter", "Auckland"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required}. Got: {df.columns.tolist()}")

    df_out = df[["Quarter", "Auckland"]].rename(columns={"Auckland": "unemployment_rate"}).copy()

    # Make Quarter datetime (you used dayfirst=True, keep consistent)
    df_out["Quarter"] = pd.to_datetime(df_out["Quarter"], dayfirst=True, errors="coerce")

    # Drop bad rows if any
    df_out = df_out.dropna(subset=["Quarter"]).sort_values("Quarter").reset_index(drop=True)
    return df_out


def expand_quarterly_to_districts(
    df_quarterly: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """
    Cross join unemployment quarterly series to 7 districts.
    Return columns: ['Quarter', 'District', 'unemployment_rate']
    """
    df_districts = pd.DataFrame({"District": districts})

    df_7 = (
        df_quarterly
        .merge(df_districts, how="cross")
        .sort_values(["Quarter", "District"])
        .reset_index(drop=True)
    )
    return df_7[["Quarter", "District", "unemployment_rate"]].copy()


def interpolate_quarterly_to_monthly_by_district(
    df_quarterly_7: pd.DataFrame,
    district_order: list[str],
) -> pd.DataFrame:
    """
    For each district:
    Quarterly -> Monthly ('MS') with linear interpolation.
    Output columns: ['Month','District','unemployment_rate']
    """
    required = {"Quarter", "District", "unemployment_rate"}
    if not required.issubset(df_quarterly_7.columns):
        raise ValueError(f"df_quarterly_7 must contain columns: {required}")

    def _interp_one_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Quarter")
        out = (
            g.set_index("Quarter")["unemployment_rate"]
            .resample("MS")
            .interpolate("linear")
            .reset_index()
        )
        out["District"] = g["District"].iloc[0]  # keep district safely
        return out

    df_monthly = (
        df_quarterly_7
        .groupby("District", group_keys=False)
        .apply(_interp_one_group)
        .reset_index(drop=True)
    )

    df_monthly["Month"] = df_monthly["Quarter"].dt.to_period("M").astype(str)  # type: ignore
    df_monthly = df_monthly[["Month", "District", "unemployment_rate"]].copy()
    df_monthly["unemployment_rate"] = df_monthly["unemployment_rate"].round(2)

    df_monthly["District"] = pd.Categorical(
        df_monthly["District"],
        categories=district_order,
        ordered=True,
    )

    df_monthly = df_monthly.sort_values(["Month", "District"]).reset_index(drop=True)
    return df_monthly


def build_unemployment_monthly_7districts(
    xlsx_path: Path,
    districts: list[str],
    district_order: list[str],
) -> pd.DataFrame:
    """
    Full pipeline:
    xlsx -> quarterly (Auckland) -> cross join districts -> monthly interpolation
    """
    df_q = load_unemployment_quarterly(xlsx_path)
    df_q7 = expand_quarterly_to_districts(df_q, districts)
    df_m = interpolate_quarterly_to_monthly_by_district(df_q7, district_order)
    return df_m


def save_unemployment_monthly(
    df_monthly: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_monthly.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) main(orchestration)
# ============================================================

def main() -> None:
    from src.config import RAW_DIR, PROCESSED_DIR

    base_in = RAW_DIR / "unemployment_rate"
    base_out = PROCESSED_DIR / "unemployment_rate"
    base_out.mkdir(parents=True, exist_ok=True)

    unemployment_rate_path = base_in / "Unemployment_rate.xlsx"

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

    df_unemp_monthly = build_unemployment_monthly_7districts(
        xlsx_path=unemployment_rate_path,
        districts=districts,
        district_order=district_order,
    )

    print(df_unemp_monthly.head(10))

    output_path = base_out / "Unemployment_ratecl.csv"
    save_unemployment_monthly(df_unemp_monthly, output_path)


if __name__ == "__main__":
    main()




