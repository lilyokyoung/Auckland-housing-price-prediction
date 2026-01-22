# src/api/routes_predict.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    FORECAST_FUTURE_DATASETS,
    PREPROCESSED_DIR,
    MODEL_DIR,
)

# ✅ 重要：两者选其一（按你最终决定的脚本位置）
from src.models.final_prediction.predict_future_with_rf_final import (
    PredictConfig,
    run_predict_future,
)

router = APIRouter(tags=["predict"])


# =========================
# Schemas
# =========================
class PredictRequest(BaseModel):
    scenario: str = Field(default="base", description="base | low | high")
    # ⚠️ Swagger 会给 List[str] 自动填 ["string"]，所以我们做防呆：传 None 或删除该字段即可表示 all
    districts: Optional[List[str]] = Field(
        default=None,
        description="Optional. If omitted or null => all districts. "
                    "Examples: ['AucklandCity'] or ['Auckland City'] (space is ok).",
    )
    preview_rows: int = Field(default=50, ge=1, le=20000, description="Rows to return for preview")


class PredictResponse(BaseModel):
    scenario: str
    rows: int
    preview: List[Dict[str, Any]]


# =========================
# Helpers
# =========================
def _norm_scenario(s: str) -> str:
    return str(s).strip().lower()


def _norm_key(s: str) -> str:
    """
    Loose match key:
    - lower
    - remove spaces/underscores/hyphens/punctuation
    - keep only alnum
    So "Auckland City" == "AucklandCity" == "auckland_city".
    """
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _clean_districts(ds: Optional[List[str]]) -> Optional[List[str]]:
    """
    - None => None
    - Remove empty strings
    - Remove Swagger placeholder "string"
    - De-duplicate preserving order
    """
    if not ds:
        return None

    out: List[str] = []
    seen = set()
    for d in ds:
        dd = str(d).strip()
        if not dd:
            continue
        if dd.lower() == "string":  # Swagger placeholder
            continue
        if dd not in seen:
            out.append(dd)
            seen.add(dd)

    return out or None


# =========================
# Route
# =========================
@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    scenario_req = _norm_scenario(req.scenario)
    districts_req = _clean_districts(req.districts)

    if scenario_req not in {"base", "low", "high"}:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use base | low | high")

    future_dir = FORECAST_FUTURE_DATASETS

    cfg = PredictConfig(
        rf_final_model_path=MODEL_DIR / "avms" / "rf_final_model" / "rf_final_model.joblib",
        future_base_path=future_dir / "future_for_AVMs_base.csv",
        future_low_path=future_dir / "future_for_AVMs_low.csv",
        future_high_path=future_dir / "future_for_AVMs_high.csv",
        historical_path=PREPROCESSED_DIR / "avms" / "final_for_AVMs.csv",
        out_dir=MODEL_DIR / "rf_final_predictions",
        drop_feature_cols=["Scenario"],
        last_history_months=36,
        month_to_month_start=True,
        make_plots=False,
        write_csv=False,
    )

    try:
        df_all: pd.DataFrame = run_predict_future(cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    # ---- required columns check (fail fast) ----
    required_cols = {"Scenario", "District"}
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction output missing required columns.",
                "missing_columns": missing,
                "available_columns": df_all.columns.tolist(),
            },
        )

    # ---- normalize columns used for filtering ----
    df_all = df_all.copy()
    df_all["Scenario"] = df_all["Scenario"].astype(str).str.strip().str.lower()
    df_all["District"] = df_all["District"].astype(str).str.strip()

    available_cols = df_all.columns.tolist()
    available_scenarios = sorted(df_all["Scenario"].dropna().unique().tolist())
    available_districts = sorted(df_all["District"].dropna().unique().tolist())

    # ---- filter scenario ----
    df = df_all[df_all["Scenario"] == scenario_req].copy()

    # ---- optional filter districts (loose match) ----
    if districts_req:
        df["_district_key"] = df["District"].map(_norm_key)
        req_keys = {_norm_key(x) for x in districts_req}
        df = df[df["_district_key"].isin(req_keys)].copy()
        df.drop(columns=["_district_key"], inplace=True)

    # ---- Month -> string for JSON stability ----
    if "Month" in df.columns:
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ---- still empty? return helpful 404 ----
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No rows after filtering. Check scenario/district naming.",
                "requested": {"scenario": scenario_req, "districts": districts_req},
                "available_columns": available_cols,
                "available_scenarios": available_scenarios,
                "available_districts": available_districts,
                "hint": "If using Swagger, remove the default ['string'] placeholder in districts, or set districts=null.",
            },
        )

    # ---- build response ----
    preview_n = min(int(req.preview_rows), int(len(df)))
    preview: List[Dict[str, Any]] = (
    df.head(preview_n).to_dict(orient="records")  # type: ignore
)  

    return PredictResponse(
        scenario=scenario_req,
        rows=int(len(df)),
        preview=preview,
    )
