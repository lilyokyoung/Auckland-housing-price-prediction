from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import FEATURE_DIR, PROCESSED_DIR


# ============================================================
# 1) Core functions
# ============================================================

def load_merged_dataset1(merged1_path: Path) -> pd.DataFrame:
    df = pd.read_csv(merged1_path)

    required = {"Month", "District"}
    if not required.issubset(df.columns):
        raise ValueError(f"merged_dataset1 must contain columns: {required}")

    df["Month"] = pd.PeriodIndex(df["Month"].astype(str), freq="M")
    return df


def load_real_mortgage_rate(real_rate_path: Path) -> pd.DataFrame:
    df = pd.read_csv(real_rate_path)

    required = {"Month", "District", "RealMortgageRate"}
    if not required.issubset(df.columns):
        raise ValueError(f"RealMortgageRate file must contain columns: {required}")

    df = df[["Month", "District", "RealMortgageRate"]].copy()
    df["Month"] = pd.PeriodIndex(df["Month"].astype(str), freq="M")
    df["RealMortgageRate"] = pd.to_numeric(df["RealMortgageRate"], errors="coerce")

    return df


def build_merged_dataset2(
    df_merged1: pd.DataFrame,
    df_real: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    if drop_cols is None:
        drop_cols = ["CPI", "2YearFixedRate"]

    df_merged2 = df_merged1.merge(
        df_real,
        on=["Month", "District"],
        how="left",
        validate="many_to_one",  # each Month+District in df_real should be unique
    )

    df_merged2 = df_merged2.drop(columns=[c for c in drop_cols if c in df_merged2.columns])

    return df_merged2


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out["Month"] = df_out["Month"].astype(str)
    df_out.to_csv(out_path, index=False)

    print(f"Saved merged_dataset2 to: {out_path}")


# ============================================================
# 2) main
# ============================================================

def main() -> None:
    merged1_path = PROCESSED_DIR / "merged_dataset" / "merged_dataset1.csv"
    real_rate_path = FEATURE_DIR / "DerivedVariable" / "RealMortgageRate_7districts.csv"
    out_path = PROCESSED_DIR / "merged_dataset" / "merged_dataset2.csv"

    df_merged1 = load_merged_dataset1(merged1_path)
    df_real = load_real_mortgage_rate(real_rate_path)

    df_merged2 = build_merged_dataset2(
        df_merged1=df_merged1,
        df_real=df_real,
        drop_cols=["CPI", "2YearFixedRate"],
    )

    # ---- optional sanity checks (recommended) ----
    missing = df_merged2["RealMortgageRate"].isna().sum()
    print("RealMortgageRate missing count:", missing)
    print("Month range:", df_merged2["Month"].min(), "to", df_merged2["Month"].max())
    print(df_merged2[["Month", "District", "RealMortgageRate"]].head(10))

    save_dataset(df_merged2, out_path)


if __name__ == "__main__":
    main()


