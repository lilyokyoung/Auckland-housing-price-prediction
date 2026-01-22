from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import FEATURE_DIR


# ============================================================
# 1) Load processed inputs (CSV)
# ============================================================

def load_mortgage_rate_7districts(csv_path: Path) -> pd.DataFrame:
    """
    Load processed mortgage rate (monthly, 7 districts) from CSV.
    Expected columns: ['Month', 'District', '2YearFixedRate']
    """
    df = pd.read_csv(csv_path)

    required = {"Month", "District", "2YearFixedRate"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mortgage CSV must contain columns: {required}")

    df["Month"] = pd.PeriodIndex(df["Month"].astype(str), freq="M")
    df["2YearFixedRate"] = pd.to_numeric(df["2YearFixedRate"], errors="coerce")

    return df[["Month", "District", "2YearFixedRate"]].copy()


def load_cpi_monthly_7districts(csv_path: Path) -> pd.DataFrame:
    """
    Load processed CPI (monthly, 7 districts) from CSV.
    Expected columns: ['Month', 'District', 'CPI']
    """
    df = pd.read_csv(csv_path)

    required = {"Month", "District", "CPI"}
    if not required.issubset(df.columns):
        raise ValueError(f"CPI CSV must contain columns: {required}")

    df["Month"] = pd.PeriodIndex(df["Month"].astype(str), freq="M")
    df["CPI"] = pd.to_numeric(df["CPI"], errors="coerce")

    return df[["Month", "District", "CPI"]].copy()


# ============================================================
# 2) CPI YoY + Real mortgage rate
# ============================================================

def compute_cpi_yoy(df_cpi: pd.DataFrame) -> pd.DataFrame:
    """
    CPI YoY inflation rate (%) at monthly frequency:
    (CPI_t - CPI_{t-12}) / CPI_{t-12} * 100
    """
    df = df_cpi.sort_values(["District", "Month"]).copy()

    cpi_lag12 = df.groupby("District")["CPI"].shift(12)
    df["CPI_YoY"] = (df["CPI"] - cpi_lag12) / cpi_lag12 * 100

    return df


def build_real_mortgage_rate(
    mortgage_csv: Path,
    cpi_csv: Path,
) -> pd.DataFrame:
    """
    Merge mortgage rate and CPI, compute CPI YoY and RealMortgageRate:
    RealMortgageRate = 2YearFixedRate - CPI_YoY
    """
    df_m = load_mortgage_rate_7districts(mortgage_csv)
    df_c = load_cpi_monthly_7districts(cpi_csv)
    df_c = compute_cpi_yoy(df_c)

    df = (
        df_m.merge(df_c, on=["Month", "District"], how="inner")
        .sort_values(["Month", "District"])
        .reset_index(drop=True)
    )

    df["RealMortgageRate"] = df["2YearFixedRate"] - df["CPI_YoY"]

    df["CPI_YoY"] = df["CPI_YoY"].round(4)
    df["RealMortgageRate"] = df["RealMortgageRate"].round(4)

    return df[["Month", "District", "2YearFixedRate", "CPI", "CPI_YoY", "RealMortgageRate"]].copy()


# ============================================================
# 2.5) Filter sample window (ADD THIS)
# ============================================================

def filter_sample_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep 2018-08 to 2025-09 (inclusive),
    and drop specific months: 2018-11, 2020-07
    """
    df = df.copy()

    # Ensure Period[M]
    df["Month"] = pd.PeriodIndex(df["Month"].astype(str), freq="M")

    start = pd.Period("2018-08", freq="M")
    end = pd.Period("2025-09", freq="M")
    df = df.loc[(df["Month"] >= start) & (df["Month"] <= end)]

    drop_months = [
        pd.Period("2018-11", freq="M"),
        pd.Period("2020-07", freq="M"),
    ]
    df = df.loc[~df["Month"].isin(drop_months)]

    return df.reset_index(drop=True)


def save_real_mortgage_rate(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out["Month"] = df_out["Month"].astype(str)

    df_out.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 3) main
# ============================================================

def main() -> None:
    from src.config import PROCESSED_DIR

    mortgage_csv = PROCESSED_DIR / "MortgageInterestRate" / "MortgageRate_7districts.csv"
    cpi_csv = PROCESSED_DIR / "CPI" / "CPI_7districts_monthly.csv"

    df_real = build_real_mortgage_rate(
        mortgage_csv=mortgage_csv,
        cpi_csv=cpi_csv,
    )

    # ✅ apply your requested window + drops
    df_real = filter_sample_window(df_real)

    # quick sanity check
    print(df_real.head(20))
    print("Month range:", df_real["Month"].min(), "to", df_real["Month"].max())

    out_path = FEATURE_DIR / "DerivedVariable" / "RealMortgageRate_7districts.csv"
    save_real_mortgage_rate(df_real, out_path)


if __name__ == "__main__":
    main()

