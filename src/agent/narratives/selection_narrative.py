# src/agent/narratives/selection_narrative.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class SelectionNarrativeConfig:
    test_metrics_path: Path
    train_metrics_path: Optional[Path] = None
    baseline_model: str = "MLR_baseline"
    prefer_unit: str = "NZD"  # "NZD" or "log"

    # NEW: keep selection narrative clean; let /generalisation handle gap wording
    include_generalisation_bullet: bool = False


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def _detect_metric_cols(df: pd.DataFrame, prefer_unit: str) -> tuple[str, str, str, str]:
    cols = set(df.columns)

    if prefer_unit.upper() == "NZD" and {"RMSE_NZD", "MAE_NZD", "R2_NZD"}.issubset(cols):
        return "RMSE_NZD", "MAE_NZD", "R2_NZD", "NZD"
    if {"RMSE_log", "MAE_log", "R2_log"}.issubset(cols):
        return "RMSE_log", "MAE_log", "R2_log", "log"
    if {"RMSE_NZD", "MAE_NZD", "R2_NZD"}.issubset(cols):
        return "RMSE_NZD", "MAE_NZD", "R2_NZD", "NZD"

    raise KeyError(f"Metric columns not found. Available: {list(df.columns)}")


def build_selection_meta(cfg: SelectionNarrativeConfig) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"ok": True}

    try:
        df_test = _strip_cols(pd.read_csv(cfg.test_metrics_path))
        if "model" not in df_test.columns:
            raise KeyError("test metrics missing column: 'model'")

        rmse_col, mae_col, r2_col, unit = _detect_metric_cols(df_test, cfg.prefer_unit)

        for c in (rmse_col, mae_col, r2_col):
            df_test[c] = pd.to_numeric(df_test[c], errors="coerce")

        best = df_test.sort_values(rmse_col, ascending=True).iloc[0]
        best_model = str(best["model"])
        best_rmse_test = float(best[rmse_col])

        meta.update(
            {
                "unit": unit,
                "rmse_col": rmse_col,
                "mae_col": mae_col,
                "r2_col": r2_col,
                "best_model": best_model,
                "best_rmse_test": best_rmse_test,
                "include_generalisation_bullet": bool(cfg.include_generalisation_bullet),
            }
        )

        # baseline improvement
        base_mask = df_test["model"].astype(str).str.lower() == cfg.baseline_model.lower()
        if base_mask.any():
            base_rmse = float(df_test.loc[base_mask, rmse_col].iloc[0])
            meta["baseline_model"] = cfg.baseline_model
            meta["baseline_rmse_test"] = base_rmse
            meta["improvement_pct_vs_baseline"] = (
                (base_rmse - best_rmse_test) / base_rmse * 100.0 if base_rmse > 0 else None
            )
        else:
            meta["baseline_model"] = cfg.baseline_model
            meta["baseline_rmse_test"] = None
            meta["improvement_pct_vs_baseline"] = None

        # generalisation gap (train->test) for best model (optional)
        meta["best_rmse_train"] = None
        meta["rmse_gap_pct"] = None

        if cfg.train_metrics_path and cfg.train_metrics_path.exists():
            df_train = _strip_cols(pd.read_csv(cfg.train_metrics_path))
            if "model" in df_train.columns and rmse_col in df_train.columns:
                df_train[rmse_col] = pd.to_numeric(df_train[rmse_col], errors="coerce")
                m = df_train["model"].astype(str).str.lower() == best_model.lower()
                if m.any():
                    best_rmse_train = float(df_train.loc[m, rmse_col].iloc[0])
                    meta["best_rmse_train"] = best_rmse_train
                    meta["rmse_gap_pct"] = (
                        (best_rmse_test / best_rmse_train - 1.0) * 100.0 if best_rmse_train > 0 else None
                    )

    except Exception as e:
        meta["ok"] = False
        meta["error_type"] = type(e).__name__
        meta["error"] = str(e)

    return meta


def build_selection_narrative(meta: Dict[str, Any]) -> str:
    if not meta.get("ok"):
        return "Final model selection rationale unavailable (selection meta failed)."

    unit = meta.get("unit", "NZD")
    best_model = meta.get("best_model", "—")
    best_rmse_test = meta.get("best_rmse_test", None)
    imp = meta.get("improvement_pct_vs_baseline", None)
    base_model = meta.get("baseline_model", "MLR_baseline")
    gap = meta.get("rmse_gap_pct", None)
    include_gap = bool(meta.get("include_generalisation_bullet", False))

    def fmt_metric(x: Any) -> str:
        if x is None:
            return "—"
        return f"{float(x):,.0f} {unit}" if unit == "NZD" else f"{float(x):.4f} {unit}"

    lines: list[str] = []
    lines.append("The final AVM is selected based on **out-of-sample performance** and **robustness**:\n")
    lines.append(f"- **Selected model**: **{best_model}** (lowest test RMSE)")
    lines.append(f"- **Primary criterion**: best test RMSE ({fmt_metric(best_rmse_test)})")

    if imp is not None:
        lines.append(f"- **Relative improvement vs {base_model} baseline**: {imp:.1f}% lower RMSE")

    # IMPORTANT: do NOT use negative phrasing. Prefer to keep this out of selection narrative entirely.
    # If included, phrase it as market non-stationarity / regime change, not "model failure".
    if include_gap and gap is not None:
        if gap > 80:
            lines.append(
                f"- **Generalisation assessment**: a sizeable train→test RMSE difference is observed "
                f"({gap:.1f}%), which is consistent with **non-stationary housing market dynamics**; "
                "conclusions are therefore based primarily on **out-of-sample** performance."
            )
        elif gap > 30:
            lines.append(
                f"- **Generalisation assessment**: train→test RMSE differs by {gap:.1f}%, "
                "suggesting performance varies across periods under time-based splitting."
            )
        else:
            lines.append(
                f"- **Generalisation assessment**: train→test RMSE differs by {gap:.1f}%, "
                "which is broadly consistent with stable out-of-sample behaviour."
            )

    lines.append("- **Secondary checks**: MAE stability and consistency across districts (robustness)")
    lines.append("- **Practicality**: supports explanation (e.g., SHAP drivers) and deployment constraints")

    return "\n".join(lines)
