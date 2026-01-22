from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd


# ============================================================
# Config
# ============================================================
TARGET_COL = "log_Median_Price"   # train 
ID_COLS = {"Month", "District"}   # Pipeline 

# ============================================================
# Helpers
# ============================================================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def get_feature_cols(df: pd.DataFrame, has_target: bool) -> Set[str]:
    cols = set(df.columns)

    if has_target and TARGET_COL in cols:
        cols.remove(TARGET_COL)

    # Month / District 
    return cols


def compare_columns(
    train_cols: Set[str],
    future_cols: Set[str],
) -> None:
    missing_in_future = sorted(train_cols - future_cols)
    extra_in_future = sorted(future_cols - train_cols)

    print("\n================ COLUMN CHECK =================")
    print(f"Train feature count : {len(train_cols)}")
    print(f"Future feature count: {len(future_cols)}")

    if not missing_in_future and not extra_in_future:
        print("\n✅ PERFECT MATCH: Future columns exactly match train columns.")
        return

    if missing_in_future:
        print("\n❌ Columns MISSING in future (model will CRASH):")
        for c in missing_in_future:
            print("  -", c)

    if extra_in_future:
        print("\n⚠️ Extra columns in future (will be ignored only if Pipeline drops them):")
        for c in extra_in_future:
            print("  +", c)


# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import PREPROCESSED_DIR, FORECAST_DIR

    train_path = PREPROCESSED_DIR / "avms" / "train_for_AVMs.csv"
    future_dir = FORECAST_DIR / "future datasets"

    train_df = load_csv(train_path)
    train_X_cols = get_feature_cols(train_df, has_target=True)

    print("\n=== Checking future datasets against TRAIN ===")

    for scen in ["base", "low", "high"]:
        future_path = future_dir / f"future_for_AVMs_{scen}.csv"
        future_df = load_csv(future_path)
        future_X_cols = get_feature_cols(future_df, has_target=False)

        print(f"\n--- Scenario: {scen.upper()} ---")
        compare_columns(train_X_cols, future_X_cols)


if __name__ == "__main__":
    main()
