from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 0) Config
# ============================================================

TARGET_COL = "log_Median_Price"
ID_COLS = ["Month", "District"]


# ============================================================
# 1) I/O helpers
# ============================================================

def load_csv_or_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError("File must be .csv or .parquet")


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# ============================================================
# 2) Core logic
# ============================================================

def extract_selected_feature_names(lasso_selected_path: Path) -> list[str]:
    """
    Read lasso_selected_features.csv and return list of feature names.
    """
    df_sel = pd.read_csv(lasso_selected_path)

    if "feature" not in df_sel.columns:
        raise ValueError("lasso_selected_features.csv must contain column 'feature'")

    features = df_sel["feature"].astype(str).tolist()

    if len(features) == 0:
        raise ValueError("No features found in LASSO selection file")

    return features


def build_train_for_model(
    train_for_lasso_path: Path,
    lasso_selected_path: Path,
) -> pd.DataFrame:
    """
    Build final modeling dataset:
    [Month, District, log_Median_Price, <selected features>]
    """
    df = load_csv_or_parquet(train_for_lasso_path)

    selected_features = extract_selected_feature_names(lasso_selected_path)

    # Required columns
    required_cols = ID_COLS + [TARGET_COL] + selected_features

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in train_for_lasso: {missing}")

    df_out = df[required_cols].copy()

    # Optional: sort for safety
    df_out = df_out.sort_values(["Month", "District"]).reset_index(drop=True)

    return df_out


# ============================================================
# 3) main
# ============================================================

def main() -> None:
    from src.config import PREPROCESSED_DIR, TABLE_DIR

    # Input paths
    train_for_lasso_csv = PREPROCESSED_DIR / "LASSO" / "train_for_lasso.csv"
    train_for_lasso_parquet = PREPROCESSED_DIR / "LASSO" / "train_for_lasso.parquet"
    train_for_lasso_path = (
        train_for_lasso_csv
        if train_for_lasso_csv.exists()
        else train_for_lasso_parquet
    )

    lasso_selected_path = TABLE_DIR / "LASSO_summary" / "lasso_selected_features.csv"

    # Build dataset
    df_model = build_train_for_model(
        train_for_lasso_path=train_for_lasso_path,
        lasso_selected_path=lasso_selected_path,
    )

    # Output
    out_path = PREPROCESSED_DIR / "train_for_mlr.csv"
    save_csv(df_model, out_path)

    print("\n=== train_for_mlr summary ===")
    print(f"Rows: {len(df_model)}")
    print(f"Columns ({len(df_model.columns)}):")
    print(df_model.columns.tolist())
    print(df_model.info())

if __name__ == "__main__":
    main()

