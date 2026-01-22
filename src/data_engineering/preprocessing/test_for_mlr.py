from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

BASE_COLS: List[str] = [
    "Month",
    "District",
    "Median_Price",
    "sales_count",
    "OCR",
    "CGPI_Dwelling",
    "Consents",
    "Net_migration_monthly",
    "weeklyrent",
    "unemployment_rate",
    "RealMortgageRate",
]

# Log-transform these base variables, then drop raw columns
LOG_COLS: List[str] = ["Median_Price","sales_count", "Consents"]

# Lag settings (same as your AVMs pipeline)
LAGS: List[int] = [1, 3, 6, 9, 12]

# Create lags for these variables (after log variables exist)
LAG_COLS: List[str] = [
    "log_sales_count",
    "OCR",
    "CGPI_Dwelling",
    "log_Consents",
    "unemployment_rate",
    "RealMortgageRate",
]


# ============================================================
# I/O
# ============================================================

def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def save_csv_month_yyyymm(df: pd.DataFrame, path: Path, month_col: str = "Month") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if month_col in out.columns:
        dt = pd.to_datetime(out[month_col], errors="raise")
        out[month_col] = dt.dt.to_period("M").astype(str)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


# ============================================================
# Preprocessing
# ============================================================

def normalize_month_to_month_start(df: pd.DataFrame, month_col: str = "Month") -> pd.DataFrame:
    if month_col not in df.columns:
        raise KeyError(f"Missing '{month_col}' column.")
    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="raise")
    out[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    return out


def select_base_columns(df: pd.DataFrame, base_cols: Sequence[str] = BASE_COLS) -> pd.DataFrame:
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns: {missing}")
    return df[list(base_cols)].copy()


def _ensure_positive_numeric(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), ["Month", "District", col]].head(10)
        raise ValueError(f"Column '{col}' has NaNs after numeric coercion.\nExamples:\n{bad}")
    if (df[col] <= 0).any():
        bad = df.loc[df[col] <= 0, ["Month", "District", col]].head(10)
        raise ValueError(
            f"log(x) not safe: column '{col}' contains <=0 values.\nExamples:\n{bad}"
        )


def add_log_and_drop_raw(df: pd.DataFrame, cols: Sequence[str] = LOG_COLS) -> pd.DataFrame:
    """
    Create log_{col} = log(col) and drop the original raw column.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Missing column for log transform: '{c}'")
        _ensure_positive_numeric(out, c)
        out[f"log_{c}"] = np.log(out[c].astype(float))
        out = out.drop(columns=[c])
    return out


def add_panel_lags(
    df: pd.DataFrame,
    group_col: str = "District",
    time_col: str = "Month",
    cols: Sequence[str] = LAG_COLS,
    lags: Sequence[int] = LAGS,
    lag_suffix: str = "_lag",
) -> pd.DataFrame:
    """
    Create lags within each District over time order.
    """
    out = df.copy()

    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Missing lag source column: '{c}'")

    out = out.sort_values([group_col, time_col]).reset_index(drop=True)
    g = out.groupby(group_col, sort=False)

    for c in cols:
        for L in lags:
            out[f"{c}{lag_suffix}{L}"] = g[c].shift(L)

    return out


# ============================================================
# LASSO selected features
# ============================================================

def load_lasso_selected_features(path: Path) -> List[str]:
    """
    Reads lasso_selected_features.csv and returns a list of selected feature names.
    Supports common column names; falls back to the first column.
    """
    if not path.exists():
        raise FileNotFoundError(f"LASSO selected features file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"LASSO selected features file is empty: {path}")

    for col in ["feature", "features", "selected_feature", "selected_features", "variable", "variables"]:
        if col in df.columns:
            feats = df[col].dropna().astype(str).tolist()
            return [f.strip() for f in feats if f.strip()]

    first = df.columns[0]
    feats = df[first].dropna().astype(str).tolist()
    return [f.strip() for f in feats if f.strip()]


def filter_to_selected(
    df: pd.DataFrame,
    selected: Sequence[str],
    keep_cols: Tuple[str, ...] = ("Month", "District", "log_Median_Price"),
) -> pd.DataFrame:
    """
    Keep Month/District/target + selected predictors that exist in df.
    Do NOT drop NA rows here (as requested).
    """
    keep = [c for c in keep_cols if c in df.columns]
    present_selected = [c for c in selected if c in df.columns]

    missing_selected = [c for c in selected if c not in df.columns]
    if missing_selected:
        print("[WARN] Selected features not found in the test table (skipped):")
        print("       ", missing_selected[:30], "..." if len(missing_selected) > 30 else "")

    cols = keep + present_selected
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


# ============================================================
# Diagnostics (NA lags)
# ============================================================

def report_lag_na(df: pd.DataFrame, lag_suffix: str = "_lag") -> None:
    lag_cols = [c for c in df.columns if lag_suffix in c]
    if not lag_cols:
        print("[INFO] No lag columns found for NA diagnostics.")
        return

    na_counts = df[lag_cols].isna().sum().sort_values(ascending=False)
    top = na_counts.head(12)

    print("\n[DIAG] NA counts in lag columns (top 12):")
    print(top.to_string())

    if "Month" in df.columns:
        dt = pd.to_datetime(df["Month"], errors="coerce")
        if dt.notna().any():
            earliest = df.loc[df[lag_cols].notna().all(axis=1), "Month"]
            if len(earliest) > 0:
                print("\n[DIAG] Earliest month with ALL lag features available:")
                print(pd.to_datetime(earliest).min())
            else:
                print("\n[DIAG] No rows have ALL lag features available (within this dataset window).")


# ============================================================
# Runner
# ============================================================

@dataclass
class TestForMLRConfig:
    test_path: Path
    lasso_selected_path: Path
    out_path: Path

    month_col: str = "Month"
    district_col: str = "District"
    target_col: str = "log_Median_Price"


def build_test_for_mlr(cfg: TestForMLRConfig) -> pd.DataFrame:
    df = load_table(cfg.test_path)

    df = select_base_columns(df, BASE_COLS)
    df = normalize_month_to_month_start(df, cfg.month_col)

    # log(base) then drop raw variables
    df = add_log_and_drop_raw(df, LOG_COLS)

    # create lags (DO NOT drop NA rows)
    df = add_panel_lags(
        df,
        group_col=cfg.district_col,
        time_col=cfg.month_col,
        cols=LAG_COLS,
        lags=LAGS,
        lag_suffix="_lag",
    )

    # diagnostics
    report_lag_na(df, lag_suffix="_lag")

    # select features by LASSO list
    selected = load_lasso_selected_features(cfg.lasso_selected_path)
    df = filter_to_selected(
        df,
        selected=selected,
        keep_cols=(cfg.month_col, cfg.district_col, cfg.target_col),
    )

    # stable order
    df = df.sort_values([cfg.month_col, cfg.district_col]).reset_index(drop=True)
    return df


def main() -> None:
    from src.config import TEST_DIR, PREPROCESSED_DIR, TABLE_DIR

    test_path = TEST_DIR / "test.csv"
    if not test_path.exists():
        test_path = TEST_DIR / "test.parquet"

    lasso_selected_path = TABLE_DIR / "LASSO_summary" / "lasso_selected_features.csv"
    out_path = PREPROCESSED_DIR / "test_for_mlr.csv"

    cfg = TestForMLRConfig(
        test_path=test_path,
        lasso_selected_path=lasso_selected_path,
        out_path=out_path,
    )

    df_out = build_test_for_mlr(cfg)
    save_csv_month_yyyymm(df_out, cfg.out_path, month_col=cfg.month_col)

    print("\n[INFO] test_for_mlr shape:", df_out.shape)
    print("[INFO] Month min/max:", df_out[cfg.month_col].min(), "->", df_out[cfg.month_col].max())


if __name__ == "__main__":
    main()

