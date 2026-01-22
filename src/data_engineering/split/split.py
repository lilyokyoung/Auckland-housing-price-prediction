# src/data_engineering/split/split.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# ============================================================
# Config
# ============================================================
@dataclass
class SplitConfig:
    input_path: Path
    train_dir: Path
    test_dir: Path

    month_col: str = "Month"

    # time split (month-level, inclusive)
    train_start: str = "2018-08"
    train_end: str = "2023-12"
    test_start: str = "2024-01"
    test_end: str = "2025-06"

    # outputs
    write_parquet: bool = True
    write_csv: bool = True


# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def month_ts(yyyymm: str) -> pd.Timestamp:
    """
    Convert 'YYYY-MM' or 'YYYY-MM-DD' -> month-start Timestamp.
    Example: '2024-01-15' -> 2024-01-01
    """
    return pd.Timestamp(yyyymm).to_period("M").to_timestamp(how="start")


def load_split_source(input_path: Path, month_col: str = "Month") -> pd.DataFrame:
    """
    Load dataset (.csv or .parquet) and normalize Month to month-start Timestamp.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Use .csv or .parquet")

    if month_col not in df.columns:
        raise KeyError(f"Column '{month_col}' not found in input dataset.")

    dt = pd.to_datetime(df[month_col], errors="raise")
    df[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")  # month-start

    df = df.sort_values(month_col).reset_index(drop=True)
    return df


def time_split_train_test(
    df: pd.DataFrame,
    month_col: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inclusive month ranges.
    """
    train = df[(df[month_col] >= train_start) & (df[month_col] <= train_end)].copy()
    test = df[(df[month_col] >= test_start) & (df[month_col] <= test_end)].copy()
    return train, test


def safe_write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    df.to_parquet(path, index=False)
    return path


def safe_write_csv(df: pd.DataFrame, path: Path, month_col: str = "Month") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    out = df.copy()
    # write Month as YYYY-MM (no day)
    out[month_col] = pd.to_datetime(out[month_col]).dt.strftime("%Y-%m")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_split_datasets_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: SplitConfig,
) -> Dict[str, Path]:
    ensure_dir(cfg.train_dir)
    ensure_dir(cfg.test_dir)

    outputs: Dict[str, Path] = {}

    if cfg.write_parquet:
        outputs["train_parquet"] = safe_write_parquet(train, cfg.train_dir / "train.parquet")
        outputs["test_parquet"] = safe_write_parquet(test, cfg.test_dir / "test.parquet")

    if cfg.write_csv:
        outputs["train_csv"] = safe_write_csv(train, cfg.train_dir / "train.csv", month_col=cfg.month_col)
        outputs["test_csv"] = safe_write_csv(test, cfg.test_dir / "test.csv", month_col=cfg.month_col)

    return outputs


def _summarize(df: pd.DataFrame, month_col: str) -> Dict[str, object]:
    if df.empty:
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": int(len(df)),
        "start": str(pd.to_datetime(df[month_col]).min().date()),
        "end": str(pd.to_datetime(df[month_col]).max().date()),
    }


# ============================================================
# Pipeline
# ============================================================
def run_split_pipeline(cfg: SplitConfig) -> Dict[str, object]:
    df = load_split_source(cfg.input_path, month_col=cfg.month_col)

    train_start_ts = month_ts(cfg.train_start)
    train_end_ts = month_ts(cfg.train_end)
    test_start_ts = month_ts(cfg.test_start)
    test_end_ts = month_ts(cfg.test_end)

    train, test = time_split_train_test(
        df,
        month_col=cfg.month_col,
        train_start=train_start_ts,
        train_end=train_end_ts,
        test_start=test_start_ts,
        test_end=test_end_ts,
    )

    saved_paths = save_split_datasets_train_test(train, test, cfg)

    return {
        "full": _summarize(df, cfg.month_col),
        "train": _summarize(train, cfg.month_col),
        "test": _summarize(test, cfg.month_col),
        "outputs": {k: str(v) for k, v in saved_paths.items()},
    }


# ============================================================
# Main
# ============================================================
def main() -> None:
    # IMPORTANT: use src.config (not "config")
    from src.config import FEATURE_DIR, TRAIN_DIR, TEST_DIR

    cfg = SplitConfig(
        input_path=FEATURE_DIR / "features" / "merged_dataset2_lags.csv",
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        train_start="2018-08",
        train_end="2023-12",
        test_start="2024-01",
        test_end="2025-06",
    )

    result = run_split_pipeline(cfg)

    print("\n=== Split pipeline diagnostics ===")
    print("FULL :", result["full"])
    print("TRAIN:", result["train"])
    print("TEST :", result["test"])
    print("OUTPUTS:", result["outputs"])


if __name__ == "__main__":
    main()
