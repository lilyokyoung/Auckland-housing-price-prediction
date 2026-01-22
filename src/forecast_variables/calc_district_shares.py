from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DIR, FORECAST_DIR


# -----------------------------
# Config
# -----------------------------
@dataclass
class ShareConfig:
    # panel history
    merged_panel_path: Path = (
        PROCESSED_DIR / "merged_dataset" / "merged_dataset1.csv"
    )

    # output
    output_path: Path = (
        FORECAST_DIR / "sarimax" / "district_shares.csv"
    )

    month_col: str = "Month"
    district_col: str = "District"

    consents_col: str = "Consents"
    sales_col: str = "sales_count"

    # how many recent months to use
    window_months: int = 12


# -----------------------------
# Functions
# -----------------------------
def load_panel(path: Path, month_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # robust Month parsing
    df[month_col] = pd.to_datetime(
        df[month_col],
        errors="coerce",
        dayfirst=True
    )

    if df[month_col].isna().any():
        bad = df.loc[df[month_col].isna(), month_col].head()
        raise ValueError(f"Unparseable Month values found:\n{bad}")

    return df


def restrict_recent_window(
    df: pd.DataFrame,
    month_col: str,
    window_months: int,
) -> pd.DataFrame:
    last_month = df[month_col].max()
    start_month = last_month - pd.DateOffset(months=window_months - 1)

    return df.loc[df[month_col] >= start_month].copy()


def compute_shares(
    df: pd.DataFrame,
    district_col: str,
    value_col: str,
    share_name: str,
) -> pd.DataFrame:
    agg = (
        df.groupby(district_col, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "total"})
    ) # type: ignore

    total_sum = agg["total"].sum()
    agg[share_name] = agg["total"] / total_sum

    return agg[[district_col, share_name]]


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -----------------------------
# main
# -----------------------------
def main() -> None:
    cfg = ShareConfig()

    print("\n=== Calculate District Shares (Consents & Sales) ===")
    print("Panel path :", cfg.merged_panel_path.resolve())
    print("Output path:", cfg.output_path.resolve())

    # 1) load panel
    panel = load_panel(cfg.merged_panel_path, cfg.month_col)

    # 2) recent window
    panel_recent = restrict_recent_window(
        panel,
        cfg.month_col,
        cfg.window_months,
    )

    # 3) compute shares
    consents_shares = compute_shares(
        panel_recent,
        cfg.district_col,
        cfg.consents_col,
        "consents_share",
    )

    sales_shares = compute_shares(
        panel_recent,
        cfg.district_col,
        cfg.sales_col,
        "sales_share",
    )

    # 4) merge
    shares = consents_shares.merge(
        sales_shares,
        on=cfg.district_col,
        how="inner",
    )

    # sanity check
    print("\nCheck sums:")
    print("consents_share sum =", shares["consents_share"].sum())
    print("sales_share sum   =", shares["sales_share"].sum())

    # 5) save
    save_output(shares, cfg.output_path)

    print("\nPreview:")
    print(shares)

    print("\nDone.")


if __name__ == "__main__":
    main()
