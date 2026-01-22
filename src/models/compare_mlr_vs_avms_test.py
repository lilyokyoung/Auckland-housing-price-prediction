from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # RMSE, MAE, R2
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # R2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def compute_metrics_log_and_level(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    m_log = regression_metrics(y_true_log, y_pred_log)

    y_true_lvl = np.exp(y_true_log)
    y_pred_lvl = np.exp(y_pred_log)
    m_lvl = regression_metrics(y_true_lvl, y_pred_lvl)

    return {
        "RMSE_log": m_log["RMSE"],
        "MAE_log": m_log["MAE"],
        "R2_log": m_log["R2"],
        "RMSE_NZD": m_lvl["RMSE"],
        "MAE_NZD": m_lvl["MAE"],
        "R2_NZD": m_lvl["R2"],
    }


# ============================================================
# Config
# ============================================================
@dataclass
class CompareConfig:
    # already-computed metrics
    avms_metrics_csv: Path
    mlr_metrics_csv: Path

    # to compute NZD metrics for MLR we need predictions
    mlr_preds_csv: Path

    out_dir: Path


# ============================================================
# Load + Build comparison table
# ============================================================
def load_avms_metrics(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    # expected columns:
    # model, RMSE_log, MAE_log, R2_log, RMSE_NZD, MAE_NZD, R2_NZD
    required = {"model", "RMSE_log", "MAE_log", "R2_log", "RMSE_NZD", "MAE_NZD", "R2_NZD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AVMs metrics missing columns: {sorted(missing)}")
    return df.copy()


def build_mlr_metrics_row(mlr_metrics_csv: Path, mlr_preds_csv: Path) -> pd.DataFrame:
    """
    Your current MLR script saves log-space metrics only (RMSE/MAE/R2) into mlr_test_metrics.csv.
    Here we recompute BOTH log-space and NZD-space metrics using mlr_test_predictions.csv.
    """
    _ = load_csv(mlr_metrics_csv)  # existence check; not strictly needed further

    preds = load_csv(mlr_preds_csv)
    if not {"y_true", "y_pred"}.issubset(preds.columns):
        raise ValueError("MLR predictions file must contain columns: y_true, y_pred")

    y_true_log = preds["y_true"].astype(float).to_numpy()
    y_pred_log = preds["y_pred"].astype(float).to_numpy()

    m = compute_metrics_log_and_level(y_true_log, y_pred_log)

    row = {
        "model": "MLR_baseline",
        **m,
    }
    return pd.DataFrame([row])


def make_comparison_table(cfg: CompareConfig) -> pd.DataFrame:
    avms_df = load_avms_metrics(cfg.avms_metrics_csv)
    mlr_df = build_mlr_metrics_row(cfg.mlr_metrics_csv, cfg.mlr_preds_csv)

    out = pd.concat([mlr_df, avms_df], ignore_index=True)

    # sort by NZD RMSE (primary)
    out = out.sort_values("RMSE_NZD", ascending=True).reset_index(drop=True)

    # nice rounding for display (keep raw in csv if you want; here we keep as float)
    return out


def save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    csv_path = out_dir / "test_metrics_mlr_vs_avms.csv"
    txt_path = out_dir / "test_metrics_mlr_vs_avms.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "MLR vs AVMs — TEST metrics (sorted by RMSE_NZD)",
        "================================================",
        "",
        "Primary metrics: NZD (after exp back-transform from log target).",
        "",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']}: "
            f"RMSE_NZD={r['RMSE_NZD']:.0f}, MAE_NZD={r['MAE_NZD']:.0f}, R2_NZD={r['R2_NZD']:.3f} | "
            f"RMSE_log={r['RMSE_log']:.4f}, MAE_log={r['MAE_log']:.4f}, R2_log={r['R2_log']:.3f}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Saved comparison CSV: {csv_path}")
    print(f"[OK] Saved comparison TXT: {txt_path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import MODEL_DIR

    cfg = CompareConfig(
        avms_metrics_csv=MODEL_DIR / "avms" / "final_test_eval" / "avms_test_metrics.csv",
        mlr_metrics_csv=MODEL_DIR / "mlr" / "mlr_test" / "mlr_test_metrics.csv",
        mlr_preds_csv=MODEL_DIR / "mlr" / "mlr_test" / "mlr_test_predictions.csv",
        out_dir=MODEL_DIR / "compare",
    )

    df = make_comparison_table(cfg)

    print("\n=== MLR vs AVMs (TEST) — sorted by RMSE_NZD ===")
    print(df)

    save_outputs(df, cfg.out_dir)


if __name__ == "__main__":
    main()
