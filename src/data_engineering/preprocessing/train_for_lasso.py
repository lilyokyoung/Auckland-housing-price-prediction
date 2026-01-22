from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import TRAIN_DIR


# ============================================================
# 0) Config
# ============================================================

LAG_COLS = [
    "log_sales_count",
    "OCR",
    "CGPI_Dwelling",
    "log_Consents",
    "unemployment_rate",
    "RealMortgageRate",
]

LAGS = [1, 3, 6, 9, 12]


# ============================================================
# 1) I/O helpers
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    elif out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError("Output must be .csv or .parquet")
    print(f"Saved: {out_path}")


# ============================================================
# 2) Core functions
# ============================================================

def coerce_month_to_period_m(df: pd.DataFrame, month_col: str = "Month") -> pd.DataFrame:
    """
    Ensure Month is Period[M] for correct lagging.
    Accepts 'YYYY-MM', datetime, or other parseable strings.
    """
    out = df.copy()

    if pd.api.types.is_period_dtype(out[month_col]): # type: ignore
        return out

    dt = pd.to_datetime(out[month_col], errors="coerce", dayfirst=True)
    if dt.isna().any():
        bad = out.loc[dt.isna(), month_col].head(5).tolist()
        raise ValueError(f"Month parse failed, examples: {bad}")

    out[month_col] = dt.dt.to_period("M")
    return out


def check_required_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_group_lags(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Add lag features for selected columns within each group.
    """
    out = df.copy()
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)

    for c in cols:
        for k in lags:
            out[f"{c}_lag{k}"] = out.groupby(group_col)[c].shift(k)

    return out


def finalize_month_as_string(df: pd.DataFrame, month_col: str = "Month") -> pd.DataFrame:
    """
    Convert Month back to 'YYYY-MM' string for saving / ML.
    """
    out = df.copy()
    if pd.api.types.is_period_dtype(out[month_col]): # type: ignore
        out[month_col] = out[month_col].astype(str)
    return out


def build_train_for_lasso(
    in_path: Path,
    lag_cols: list[str] = LAG_COLS,
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    # load
    df = load_csv(in_path)

    # sanity checks
    check_required_columns(df, ["Month", "District"] + lag_cols)

    # normalize Month
    df = coerce_month_to_period_m(df, "Month")

    # add lags
    df = add_group_lags(
        df=df,
        group_col="District",
        time_col="Month",
        cols=lag_cols,
        lags=lags,
    )

    # tidy Month for output
    df = finalize_month_as_string(df, "Month")

    # final sort
    df = df.sort_values(["Month", "District"]).reset_index(drop=True)

    return df


# ============================================================
# 3) main
# ============================================================

def main() -> None:
    from src.config import PREPROCESSED_DIR  # adjust if your paths differ

    in_path = PREPROCESSED_DIR / "LASSO" / "train_base_with_log.csv"
    out_dir = PREPROCESSED_DIR / "LASSO"

    df_lasso = build_train_for_lasso(
        in_path=in_path,
        lag_cols=LAG_COLS,
        lags=LAGS,
    )

    # save outputs
    save_dataset(df_lasso, out_dir / "train_for_lasso.csv")
    # optional parquet
    save_dataset(df_lasso, out_dir / "train_for_lasso.parquet")

    print("Done: train_for_lasso generated with lag 1/3/6/9/12.")


if __name__ == "__main__":
    main()
