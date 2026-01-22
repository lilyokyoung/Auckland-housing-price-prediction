# src/agent/narratives/evaluation_narrative.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class EvaluationNarrativeConfig:
    test_metrics_path: Path   # outputs/models/compare/test_metrics_mlr_vs_avms.csv
    train_metrics_path: Optional[Path] = None  # optional: outputs/models/compare/train_metrics_mlr_vs_avms.csv


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError("Metrics file must include column: model")
    return df


def build_evaluation_meta(cfg: EvaluationNarrativeConfig) -> Dict[str, Any]:
    test_df = _load_metrics(cfg.test_metrics_path)

    # primary: RMSE_NZD if exists else RMSE_log
    primary = "RMSE_NZD" if "RMSE_NZD" in test_df.columns else ("RMSE_log" if "RMSE_log" in test_df.columns else None)
    if primary is None:
        return {"ok": False, "error": "No RMSE column found (need RMSE_NZD or RMSE_log)."}

    test_df = test_df.sort_values(primary, ascending=True).reset_index(drop=True)

    best = test_df.iloc[0].to_dict()
    worst = test_df.iloc[-1].to_dict()

    # baseline row
    baseline = None
    base_mask = test_df["model"].astype(str).str.contains("mlr", case=False, na=False)
    if base_mask.any():
        baseline = test_df.loc[base_mask].iloc[0].to_dict()

    train_df = None
    if cfg.train_metrics_path and cfg.train_metrics_path.exists():
        train_df = pd.read_csv(cfg.train_metrics_path)

    return {
        "ok": True,
        "primary_metric": primary,
        "best": best,
        "worst": worst,
        "baseline": baseline,
        "test_table": test_df.to_dict(orient="records"),
        "has_train": train_df is not None,
    }


def build_evaluation_narrative(meta: Dict[str, Any]) -> str:
    if not meta.get("ok"):
        return f"Evaluation summary unavailable: {meta.get('error', 'unknown error')}"

    primary = meta["primary_metric"]
    best = meta["best"]
    baseline = meta.get("baseline")

    best_model = best.get("model", "—")
    best_rmse = best.get(primary)

    lines = []
    lines.append(f"**Out-of-sample selection (primary = {primary})** identifies **{best_model}** as the top performer.")

    if best_rmse is not None:
        lines.append(f"Best {primary}: **{best_rmse:.4f}**" if isinstance(best_rmse, (float, int)) else f"Best {primary}: **{best_rmse}**")

    if baseline is not None and isinstance(best_rmse, (float, int)) and isinstance(baseline.get(primary), (float, int)):
        b = float(baseline[primary])
        improv = (b - float(best_rmse)) / b * 100 if b != 0 else None
        if improv is not None:
            lines.append(f"Relative improvement vs MLR baseline: **{improv:.1f}%** lower {primary}.")

    lines.append(
        "Models are compared under the same time-based evaluation protocol to avoid look-ahead bias; "
        "reported results reflect out-of-sample generalization rather than in-sample fit."
    )
    return "\n\n".join(lines)
