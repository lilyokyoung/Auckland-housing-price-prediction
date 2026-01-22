# src/agent/narratives/modeling_narrative.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


# ============================================================
# Config
# ============================================================
@dataclass
class ModelingNarrativeConfig:
    """
    Configuration for automatic modeling methodology narration.
    """
    model_dir: Path


# ============================================================
# Narrative builder
# ============================================================
def build_modeling_narrative(cfg: ModelingNarrativeConfig) -> str:
    """
    Automatically generate modeling methodology narration
    based on available trained models in the project directory.
    """

    models: List[str] = []

    # ---------- linear baseline ----------
    if (cfg.model_dir / "mlr").exists():
        models.append("Linear Regression (MLR / LASSO baseline)")

    # ---------- machine learning models ----------
    if (cfg.model_dir / "avms" / "rf_best_model").exists():
        models.append("Random Forest")

    if (cfg.model_dir / "avms" / "xgboost_best_model").exists():
        models.append("XGBoost")

    if (cfg.model_dir / "avms" / "svr_best_model").exists():
        models.append("Support Vector Regression (SVR)")

    model_list = ", ".join(models) if models else "multiple candidate models"

    return f"Available trained models: {model_list}."
