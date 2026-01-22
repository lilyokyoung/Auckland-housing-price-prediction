# src/agent/narratives/generalisation_narrative.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class GeneralisationNarrativeConfig:
    test_metrics_path: Path
    train_metrics_path: Path
    baseline_model: str = "MLR_baseline"
    prefer_unit: str = "NZD"  # "NZD" or "log"
    gap_warn_threshold: float = 80.0
    gap_info_threshold: float = 30.0


def _read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _pick_cols(unit: str) -> tuple[str, str, str]:
    if unit.upper() == "NZD":
        return "RMSE_NZD", "MAE_NZD", "R2_NZD"
    return "RMSE_log", "MAE_log", "R2_log"


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v
    except Exception:
        return None


def build_generalisation_meta(cfg: GeneralisationNarrativeConfig) -> Dict[str, Any]:
    df_test = _read_metrics(cfg.test_metrics_path)
    df_train = _read_metrics(cfg.train_metrics_path)

    rmse_col, mae_col, r2_col = _pick_cols(cfg.prefer_unit)

    # basic validation (soft)
    for c in ["model", rmse_col, mae_col, r2_col]:
        if c not in df_test.columns:
            raise KeyError(f"Test metrics missing column: {c}")
        if c not in df_train.columns:
            raise KeyError(f"Train metrics missing column: {c}")

    # best model by test RMSE
    df_test2 = df_test[["model", rmse_col, mae_col, r2_col]].copy()
    df_test2[rmse_col] = pd.to_numeric(df_test2[rmse_col], errors="coerce")
    df_test2 = df_test2.sort_values(rmse_col, ascending=True).reset_index(drop=True)
    best_model = str(df_test2.loc[0, "model"])

    # pull train metrics for best model
    df_train2 = df_train[df_train["model"].astype(str) == best_model]
    train_rmse = _to_float(df_train2[rmse_col].iloc[0]) if len(df_train2) else None
    test_rmse = _to_float(df_test2.loc[0, rmse_col])

    # gap %
    gap_pct = None
    if train_rmse and train_rmse > 0 and test_rmse is not None:
        gap_pct = (test_rmse / train_rmse - 1.0) * 100.0

    # baseline improvement
    baseline_row = df_test[df_test["model"].astype(str) == cfg.baseline_model]
    baseline_rmse = _to_float(baseline_row[rmse_col].iloc[0]) if len(baseline_row) else None
    improvement_pct = None
    if baseline_rmse and baseline_rmse > 0 and test_rmse is not None:
        improvement_pct = (1.0 - (test_rmse / baseline_rmse)) * 100.0  # positive = better

    return {
        "prefer_unit": cfg.prefer_unit,
        "rmse_col": rmse_col,
        "best_model": best_model,
        "test_rmse": test_rmse,
        "train_rmse": train_rmse,
        "gap_pct": gap_pct,
        "baseline_model": cfg.baseline_model,
        "baseline_rmse": baseline_rmse,
        "improvement_pct": improvement_pct,
        "thresholds": {
            "info": cfg.gap_info_threshold,
            "warn": cfg.gap_warn_threshold,
        },
    }


def build_generalisation_narrative(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {"tone": "success"|"info"|"warning", "text": "..."}
    """
    unit = str(meta.get("prefer_unit") or "NZD")
    best_model = str(meta.get("best_model") or "—")
    gap_pct = meta.get("gap_pct", None)
    improvement_pct = meta.get("improvement_pct", None)

    info_th = float((meta.get("thresholds") or {}).get("info", 30.0))
    warn_th = float((meta.get("thresholds") or {}).get("warn", 80.0))

    # base phrase about out-of-sample
    improv_phrase = ""
    if improvement_pct is not None:
        if improvement_pct >= 0:
            improv_phrase = f" Out-of-sample RMSE is **{improvement_pct:.1f}% lower** than the baseline."
        else:
            improv_phrase = f" Out-of-sample RMSE is **{abs(improvement_pct):.1f}% higher** than the baseline."

    if gap_pct is None:
        tone = "info"
        text = (
            f"Generalisation assessment: {best_model} is selected on the basis of **out-of-sample** performance."
            f"{improv_phrase} A train–test gap could not be computed due to missing train metrics."
        )
        return {"tone": tone, "text": text}

    # Carefully worded, not “model failed”
    if gap_pct <= info_th:
        tone = "success"
        text = (
            f"Generalisation assessment: the train→test RMSE difference is **{gap_pct:.1f}%**, "
            "which is broadly consistent with stable out-of-sample behaviour under time-based splitting."
            f"{improv_phrase}"
        )
        return {"tone": tone, "text": text}

    if gap_pct <= warn_th:
        tone = "info"
        text = (
            f"Generalisation assessment: a **{gap_pct:.1f}%** train→test RMSE difference is observed. "
            "This suggests performance varies across periods, which is common in **non-stationary housing markets** "
            "and motivates interpreting results primarily through **out-of-sample** metrics."
            f"{improv_phrase}"
        )
        return {"tone": tone, "text": text}

    tone = "warning"
    text = (
        f"Generalisation assessment: a **{gap_pct:.1f}%** train→test RMSE difference is observed. "
        "This likely reflects **structural differences between historical and recent market regimes** "
        "(i.e., distribution shift) rather than pure model error; results are therefore interpreted "
        "primarily using **out-of-sample** performance."
        f"{improv_phrase}"
    )
    return {"tone": tone, "text": text}
