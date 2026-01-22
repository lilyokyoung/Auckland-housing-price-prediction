from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# I/O
# ============================================================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Core logic
# ============================================================
def merge_train_test_for_final_avms(
    train_path: Path,
    test_path: Path,
) -> pd.DataFrame:
    """
    Merge train_for_AVMs and test_for_AVMs into one final dataset
    for retraining the best-performing AVM.
    """
    df_train = load_csv(train_path)
    df_test = load_csv(test_path)

    # ---- sanity check: columns must match ----
    if set(df_train.columns) != set(df_test.columns):
        raise ValueError(
            "Train and test AVMs tables have different columns.\n"
            f"Only in train: {set(df_train.columns) - set(df_test.columns)}\n"
            f"Only in test : {set(df_test.columns) - set(df_train.columns)}"
        )

    # ---- concat ----
    df_final = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # ---- drop lag NA again (safe) ----
    lag_cols = [c for c in df_final.columns if "_lag" in c]
    if lag_cols:
        df_final = df_final.dropna(subset=lag_cols).reset_index(drop=True)

    # ---- sort for time-series consistency ----
    sort_cols = [c for c in ["Month", "District"] if c in df_final.columns]
    if sort_cols:
        df_final = df_final.sort_values(sort_cols).reset_index(drop=True)

    return df_final


# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import PREPROCESSED_DIR

    avms_dir = PREPROCESSED_DIR / "avms"

    train_path = avms_dir / "train_for_AVMs.csv"
    test_path = avms_dir / "test_for_AVMs.csv"
    out_path = avms_dir / "final_for_AVMs.csv"

    df_final = merge_train_test_for_final_avms(train_path, test_path)

    ensure_dir(out_path.parent)
    df_final.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[OK] Final AVMs dataset created for retraining")
    print(f"Saved to: {out_path}")
    print(f"Rows: {len(df_final)} | Columns: {df_final.shape[1]}")


if __name__ == "__main__":
    main()
