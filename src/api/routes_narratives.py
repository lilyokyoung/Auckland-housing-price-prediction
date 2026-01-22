# src/api/routes_narratives.py
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from src.config import PROJECT_ROOT, PREPROCESSED_DIR

from src.agent.narratives.dataset_narrative import (
    DatasetNarrativeConfig,
    build_dataset_meta,
    build_dataset_narrative,
)

from src.agent.narratives.modeling_narrative import (
    ModelingNarrativeConfig,
    build_modeling_narrative,
)

from src.agent.narratives.evaluation_narrative import (
    EvaluationNarrativeConfig,
    build_evaluation_meta,
    build_evaluation_narrative,
)

from src.agent.narratives.selection_narrative import (
    SelectionNarrativeConfig,
    build_selection_meta,
    build_selection_narrative,
)

from src.agent.narratives.generalisation_narrative import (
    GeneralisationNarrativeConfig,
    build_generalisation_meta,
    build_generalisation_narrative,
)

router = APIRouter(prefix="/narratives", tags=["narratives"])


@router.get("")
def get_methodology_narratives() -> Dict[str, Any]:
    # -----------------------
    # Dataset
    # -----------------------
    dataset_cfg = DatasetNarrativeConfig(
        modeling_table_path=PREPROCESSED_DIR / "avms" / "final_for_AVMs.csv"
    )
    ds_meta = build_dataset_meta(dataset_cfg)
    ds_text = build_dataset_narrative(ds_meta)

    # -----------------------
    # Modeling
    # -----------------------
    modeling_cfg = ModelingNarrativeConfig(
        model_dir=PROJECT_ROOT / "outputs" / "models"
    )
    modeling_text = build_modeling_narrative(modeling_cfg)

    # -----------------------
    # Evaluation
    # -----------------------
    eval_cfg = EvaluationNarrativeConfig(
        test_metrics_path=PROJECT_ROOT / "outputs" / "models" / "compare" / "test_metrics_mlr_vs_avms.csv",
        train_metrics_path=PROJECT_ROOT / "outputs" / "models" / "compare" / "train_metrics_mlr_vs_avms.csv",
    )
    eval_meta = build_evaluation_meta(eval_cfg)
    eval_text = build_evaluation_narrative(eval_meta)

    # -----------------------
    # Selection
    # -----------------------
    sel_cfg = SelectionNarrativeConfig(
        test_metrics_path=PROJECT_ROOT / "outputs" / "models" / "compare" / "test_metrics_mlr_vs_avms.csv",
        train_metrics_path=PROJECT_ROOT / "outputs" / "models" / "compare" / "train_metrics_mlr_vs_avms.csv",
        baseline_model="MLR_baseline",
        prefer_unit="NZD",
        include_generalisation_bullet=False,   # 
    )

    sel_meta = build_selection_meta(sel_cfg)
    sel_text = build_selection_narrative(sel_meta)

    # -----------------------
    # Generalisation (NEW)
    # -----------------------
    gen_cfg = GeneralisationNarrativeConfig(
        test_metrics_path=PROJECT_ROOT / "outputs" / "models" / "compare" / "test_metrics_mlr_vs_avms.csv",
        train_metrics_path=PROJECT_ROOT / "outputs" / "models" / "compare" / "train_metrics_mlr_vs_avms.csv",
        baseline_model="MLR_baseline",
        prefer_unit="NZD",
    )
    gen_meta = build_generalisation_meta(gen_cfg)
    gen_block = build_generalisation_narrative(gen_meta)

    return {
        "dataset": {"meta": ds_meta, "text": ds_text},
        "modeling": {"text": modeling_text},
        "evaluation": {"meta": eval_meta, "text": eval_text},
        "selection": {"meta": sel_meta, "text": sel_text},
        "generalisation": {"meta": gen_meta, **gen_block},  # <-- tone + text
    }
