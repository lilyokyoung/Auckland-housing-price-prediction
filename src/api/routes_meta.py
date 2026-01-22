# src/api/routes_meta.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re

import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.config import PROCESSED_DIR

router = APIRouter(tags=["meta"])

# =========================
# Project constants
# =========================
HISTORY_CSV = PROCESSED_DIR / "avms" / "final_for_AVMs.csv"

MONTH_COL = "Month"
DISTRICT_COL = "District"
TARGET_COL = "Median_Price"

DEFAULT_DISTRICTS = [
    "Auckland City",
    "Franklin",
    "Manukau",
    "North Shore",
    "Papakura",
    "Rodney",
    "Waitakere",
]


# =========================
# Helpers (server-side only)
# =========================
def _safe_read_csv(path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if usecols is None:
        return pd.read_csv(path)
    return pd.read_csv(path, usecols=usecols)


def _standardize_month(df: pd.DataFrame, month_col: str = MONTH_COL) -> pd.DataFrame:
    df = df.copy()
    if month_col in df.columns:
        df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
        # normalize to month start for consistent monthly indexing
        df[month_col] = df[month_col].dt.to_period("M").dt.to_timestamp() # type: ignore
    return df


def _read_month_range(path) -> Dict[str, Optional[str]]:
    if not path.exists():
        return {"min": None, "max": None}

    df = _safe_read_csv(path, usecols=[MONTH_COL])
    df = _standardize_month(df)
    df = df.dropna(subset=[MONTH_COL])

    if df.empty:
        return {"min": None, "max": None}

    return {
        "min": df[MONTH_COL].min().strftime("%Y-%m"),
        "max": df[MONTH_COL].max().strftime("%Y-%m"),
    }


def _read_districts(path) -> List[str]:
    if not path.exists():
        return DEFAULT_DISTRICTS

    try:
        df = _safe_read_csv(path, usecols=[DISTRICT_COL])
        districts = sorted(df[DISTRICT_COL].dropna().unique().tolist())
        return districts or DEFAULT_DISTRICTS
    except Exception:
        return DEFAULT_DISTRICTS


def _infer_frequency(df: pd.DataFrame) -> str:
    if MONTH_COL not in df.columns:
        return "Unknown"
    months = df[MONTH_COL].dropna().drop_duplicates().sort_values()
    if len(months) < 3:
        return "Unknown"
    diffs = months.diff().dropna()
    med_days = diffs.dt.days.median() # type: ignore
    if med_days is not None and 25 <= med_days <= 35:
        return "Monthly"
    return "Irregular"


def _missing_months_from_month_series(months: pd.Series) -> Tuple[Optional[str], Optional[str], List[str], int]:
    months = pd.to_datetime(months, errors="coerce").dropna()
    if months.empty:
        return None, None, [], 0

    months = months.dt.to_period("M").dt.to_timestamp() # type: ignore
    months = months.drop_duplicates().sort_values()

    start = months.min()
    end = months.max()

    full = pd.date_range(start=start, end=end, freq="MS")
    missing = sorted(set(full) - set(months))

    return (
        start.strftime("%Y-%m"),
        end.strftime("%Y-%m"),
        [d.strftime("%Y-%m") for d in missing],
        int(months.shape[0]),
    )


def _infer_max_lag_from_columns(columns: List[str]) -> int:
    """
    Detect max lag from column names like:
      - OCR_lag1, OCR_lag3, ...
      - x_lag12
      - lag_1 (optional)
    """
    lags: List[int] = []
    for c in columns:
        s = c.lower()
        # match "...lag9" or "...lag_9"
        m = re.search(r"lag[_\-]?(\d+)", s)
        if m:
            try:
                lags.append(int(m.group(1)))
            except Exception:
                pass
    return max(lags) if lags else 0


def _missingness_top(df: pd.DataFrame, top_n: int = 12) -> List[Dict[str, Any]]:
    miss = df.isna().mean().sort_values(ascending=False).head(top_n)
    return [{"column": k, "missing_rate_pct": round(float(v) * 100, 2)} for k, v in miss.items()]


# =========================
# API endpoints
# =========================
@router.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "auckland-housing-api",
        "time": datetime.utcnow().isoformat(),
    }


@router.get("/metadata")
def metadata() -> Dict[str, Any]:
    month_range = _read_month_range(HISTORY_CSV)
    districts = _read_districts(HISTORY_CSV)

    return {
        "project": "Auckland Housing Price Forecast",
        "data": {
            "history_path": str(HISTORY_CSV),
            "month_col": MONTH_COL,
            "district_col": DISTRICT_COL,
            "target_col": TARGET_COL,
            "start_month": month_range["min"],
            "end_month": month_range["max"],
            "districts": districts,
        },
        "scenarios": ["base", "low", "high"],
        "models": {
            "baseline": "MLR",
            "avms": ["RF", "XGBoost"],
        },
        "api": {"version": "0.1.0"},
    }


class DataSummaryResponse(BaseModel):
    rows: int
    cols: int
    missing_rate: Dict[str, float]


@router.get("/data/summary", response_model=DataSummaryResponse)
def data_summary() -> DataSummaryResponse:
    if not HISTORY_CSV.exists():
        return DataSummaryResponse(rows=0, cols=0, missing_rate={})

    df = pd.read_csv(HISTORY_CSV)

    return DataSummaryResponse(
        rows=df.shape[0],
        cols=df.shape[1],
        missing_rate=df.isna().mean().round(4).to_dict(),
    )


# =========================
# NEW: Dataset summary for Streamlit Data & Features page
# =========================
@router.get("/dataset/summary")
def dataset_summary(
    path: Optional[str] = Query(
        None,
        description="Optional CSV path override. If omitted, uses server default HISTORY_CSV.",
    ),
    raw_start: Optional[str] = Query(
        None,
        description="Optional raw sample start (YYYY-MM). Used only for explanation display.",
    ),
    raw_end: Optional[str] = Query(
        None,
        description="Optional raw sample end (YYYY-MM). Used only for explanation display.",
    ),
    raw_missing: Optional[str] = Query(
        None,
        description="Optional comma-separated raw missing months (e.g., '2018-11,2020-07'). Used only for explanation display.",
    ),
) -> Dict[str, Any]:
    """
    Returns a compact, UI-friendly dataset summary including:
      - effective time span derived from this CSV
      - missing months derived from this CSV
      - inferred max lag from column names
      - lightweight explanation for lag-induced truncation
      - top missingness columns
    """
    csv_path = HISTORY_CSV if not path else pd.io.common.get_handle(path, "r").handle.name  # type: ignore[attr-defined]
    # The get_handle trick resolves relative paths in some environments; but we still must validate with Path.
    from pathlib import Path as _Path

    csv_path = _Path(csv_path)

    if not csv_path.exists():
        return {
            "ok": False,
            "error": f"Dataset not found: {str(csv_path)}",
            "default_path": str(HISTORY_CSV),
        }

    df = pd.read_csv(csv_path)
    df = _standardize_month(df)

    # effective coverage based on actual csv
    eff_start, eff_end, eff_missing, eff_unique_months = _missing_months_from_month_series(df[MONTH_COL]) if MONTH_COL in df.columns else (None, None, [], 0)
    frequency = _infer_frequency(df)
    max_lag = _infer_max_lag_from_columns(list(df.columns))

    # basic stats
    rows = int(df.shape[0])
    cols = int(df.shape[1])
    districts = int(df[DISTRICT_COL].nunique()) if DISTRICT_COL in df.columns else 0

    # build explanation text
    explanation_parts: List[str] = []
    if max_lag > 0:
        explanation_parts.append(
            f"This dataset contains lagged features up to lag-{max_lag}. "
            f"Lag construction requires historical observations, so the effective sample is truncated."
        )
    if raw_start and raw_end:
        explanation_parts.append(
            f"Raw coverage (before feature engineering): {raw_start} → {raw_end}."
        )
    if raw_missing:
        explanation_parts.append(
            f"Original missing months in raw data: {raw_missing}."
        )
    if eff_start and eff_end:
        explanation_parts.append(
            f"Effective coverage (after feature engineering / NA dropping): {eff_start} → {eff_end}."
        )
    if eff_missing:
        explanation_parts.append(
            "Missing months shown here reflect gaps in the effective modeling table, "
            "which may include lag-induced NAs and any explicit hold-out/forecast periods."
        )

    return {
        "ok": True,
        "dataset": {
            "path": str(csv_path),
            "rows": rows,
            "cols": cols,
            "districts": districts,
            "unique_months": eff_unique_months,
            "frequency": frequency,
            "effective_span": {"start": eff_start, "end": eff_end},
            "effective_missing_months": eff_missing,
            "max_lag_inferred": max_lag,
        },
        "sources": ["REINZ", "Stats NZ", "RBNZ"],
        "missingness_top": _missingness_top(df, top_n=12),
        "explanation": " ".join(explanation_parts).strip(),
    }
@router.get("/meta/narratives")
def meta_narratives() -> Dict[str, Any]:
    """
    One-stop endpoint for auto methodology narration used by Streamlit pages.
    """
    from src.config import PROJECT_ROOT  # if you have it; otherwise compute from this file path
    from src.agent.narratives.dataset_narrative import DatasetNarrativeConfig, build_dataset_meta, build_dataset_narrative
    from src.agent.narratives.evaluation_narrative import EvaluationNarrativeConfig, build_evaluation_meta, build_evaluation_narrative

    modeling_table = PROCESSED_DIR / "avms" / "final_for_AVMs.csv"
    test_metrics = PROJECT_ROOT / "outputs" / "models" / "compare" / "test_metrics_mlr_vs_avms.csv"
    train_metrics = PROJECT_ROOT / "outputs" / "models" / "compare" / "train_metrics_mlr_vs_avms.csv"

    ds_meta = build_dataset_meta(DatasetNarrativeConfig(modeling_table_path=modeling_table))
    ds_text = build_dataset_narrative(ds_meta)

    ev_meta = build_evaluation_meta(EvaluationNarrativeConfig(test_metrics_path=test_metrics, train_metrics_path=train_metrics))
    ev_text = build_evaluation_narrative(ev_meta)

    return {
        "dataset": {"meta": ds_meta, "text": ds_text},
        "evaluation": {"meta": ev_meta, "text": ev_text},
    }
