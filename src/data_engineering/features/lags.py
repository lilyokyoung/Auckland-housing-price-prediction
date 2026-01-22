from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import FEATURE_DIR, PROCESSED_DIR

# ============================================================
# 1) Load / Save
# ============================================================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"Month", "District"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}")

    # Month as Period[M] for correct ordering/lagging
    df["Month"] = pd.PeriodIndex(df["Month"].astype(str), freq="M")

    # Stable ordering for lag
    df = df.sort_values(["District", "Month"]).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out["Month"] = df_out["Month"].astype(str)

    df_out.to_csv(out_path, index=False)
    print(f"Saved file to: {out_path}")


# ============================================================
# 2) Lag feature engineering
# ============================================================

def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    cols_to_lag: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Add lag features for specified columns.
    Lags are computed within each group (e.g., District) ordered by time_col.
    """
    missing_cols = [c for c in cols_to_lag if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")

    df = df.sort_values([group_col, time_col]).copy()
    g = df.groupby(group_col, sort=False)

    for col in cols_to_lag:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for k in lags:
            df[f"{col}_lag{k}"] = g[col].shift(k)

    return df


def reorder_lag_columns(
    df: pd.DataFrame,
    cols_with_lags: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Reorder columns so that for each base column:
    base, base_lag1, base_lag3, ..., base_lagK appear consecutively.
    Other columns keep their relative order.
    """
    current_cols = list(df.columns)
    new_cols: list[str] = []
    used: set[str] = set()

    for col in current_cols:
        if col in used:
            continue

        if col in cols_with_lags:
            # base column
            new_cols.append(col)
            used.add(col)

            # its lag columns
            for k in lags:
                lag_col = f"{col}_lag{k}"
                if lag_col in current_cols:
                    new_cols.append(lag_col)
                    used.add(lag_col)
        else:
            # non-lagged column
            new_cols.append(col)
            used.add(col)

    return df[new_cols]


# ============================================================
# 3) main
# ============================================================

def main() -> None:
    in_path = PROCESSED_DIR / "merged_dataset" / "merged_dataset2.csv"
    out_path = FEATURE_DIR / "features" / "merged_dataset2_lags.csv"

    df = load_dataset(in_path)

    cols_to_lag = [
        "sales_count",
        "OCR",
        "CGPI_Dwelling",
        "Consents",
        "unemployment_rate",
        "RealMortgageRate",
    ]
    lags = [1, 3, 6, 9, 12]

    df_lagged = add_lag_features(
        df=df,
        group_col="District",
        time_col="Month",
        cols_to_lag=cols_to_lag,
        lags=lags,
    )

    # Reorder: base column followed by its lags
    df_lagged = reorder_lag_columns(
        df=df_lagged,
        cols_with_lags=cols_to_lag,
        lags=lags,
    )

    # Optional quick check
    print(df_lagged[["Month", "District"] + [f"sales_count_lag{k}" for k in lags]].head(20))

    save_dataset(df_lagged, out_path)


if __name__ == "__main__":
    main()
