# src/api/routes_forecast_agent.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.simple_agent import SimpleForecastAgent, SimpleAgentConfig
from src.config import DATA_DIR, OUTPUT_DIR

router = APIRouter(tags=["forecast_agent"])


# ============================================================
# Directories
# ============================================================
MODEL_DIR = OUTPUT_DIR / "models" / "rf_final_predictions"
PROCESSED_DIR = DATA_DIR / "processed" / "merged_dataset"
SHAP_DIR = OUTPUT_DIR / "shap_forecast"


# ============================================================
# Schemas
# ============================================================
class ForecastAgentRequest(BaseModel):
    scenario: str = Field(..., description="base|low|high")
    district: str = Field(..., description="One of the 7 Auckland districts")
    month: str = Field(..., description="YYYY-MM")
    top_k: int = Field(8, ge=1, le=50)


class ForecastAgentResponse(BaseModel):
    ok: bool = True
    prediction: Dict[str, Any] = Field(default_factory=dict)
    shap: Dict[str, Any] = Field(default_factory=dict)
    narrative: str = ""
    agent: Dict[str, Any] = Field(default_factory=dict)
    debug: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Helpers: normalizers
# ============================================================
def _clean_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _norm_key(x: Any) -> str:
    """Lower + remove non-alnum.  'North Shore' == 'NorthShore' == 'north_shore'."""
    s = _clean_str(x).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _norm_scenario(x: Any) -> str:
    s = _clean_str(x).lower()
    return s if s in {"base", "low", "high"} else "base"


def _to_month_str(x: Any) -> Optional[str]:
    """
    Convert various formats to 'YYYY-MM'.
    Accept:
      - '2026-06' / '2026-06-01' / '2026/06'
      - datetime/date
      - excel date-like strings '1/07/2025'
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m")

    s = _clean_str(x)
    if not s:
        return None

    s2 = s.replace("/", "-")

    m = re.search(r"\b(20\d{2})-(\d{2})(?:-(\d{2}))?\b", s2)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return None
        return pd.Timestamp(dt).strftime("%Y-%m")
    except Exception:
        return None


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _pick_preferred_or_latest(folder: Path, contains: str, exts: Tuple[str, ...]) -> Optional[Path]:
    """Prefer a filename containing `contains`, else fallback to latest."""
    if not folder.exists():
        return None
    cands = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not cands:
        return None
    preferred = [p for p in cands if contains.lower() in p.name.lower()]
    if preferred:
        preferred.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return preferred[0]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


# ============================================================
# Loaders
# ============================================================
def _load_predictions() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load long predictions table (base/high/low).
    Prefer file containing 'pred_all_scenarios_long', else latest.
    """
    debug: Dict[str, Any] = {"model_dir": str(MODEL_DIR)}

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")

    pred_path = _pick_preferred_or_latest(MODEL_DIR, "pred_all_scenarios_long", (".csv", ".xlsx"))
    if pred_path is None:
        raise FileNotFoundError(f"No prediction table found in {MODEL_DIR}")

    debug["pred_path"] = str(pred_path)
    df = _read_table(pred_path)

    month_col = _find_col(df, ["Month", "month", "Date", "date"])
    district_col = _find_col(df, ["District", "district"])
    scenario_col = _find_col(df, ["Scenario", "scenario"])
    pred_col = _find_col(df, ["pred_Median_Price", "pred_median_price", "prediction", "pred", "yhat", "y_pred"])

    debug["columns"] = list(df.columns)
    debug["detected"] = {
        "month_col": month_col,
        "district_col": district_col,
        "scenario_col": scenario_col,
        "pred_col": pred_col,
    }

    if not (month_col and district_col and scenario_col and pred_col):
        raise ValueError(
            "Prediction table missing required columns. "
            f"Detected: {debug['detected']}. "
            "Expected at least Month/District/Scenario/pred_Median_Price."
        )

    df = df.copy()
    df["_month"] = df[month_col].map(_to_month_str)
    df["_district_key"] = df[district_col].map(_norm_key)
    df["_scenario"] = df[scenario_col].map(_norm_scenario)
    df["_pred"] = pd.to_numeric(df[pred_col], errors="coerce")

    debug["shape_before_dropna"] = list(df.shape)
    debug["na_pred"] = int(df["_pred"].isna().sum())

    # keep valid rows
    df = df.dropna(subset=["_month", "_district_key", "_scenario", "_pred"])
    debug["shape_after_dropna"] = list(df.shape)

    return df, debug


def _load_history() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load history table from PROCESSED_DIR.
    Prefer filename containing 'merged_dataset', else latest.
    """
    debug: Dict[str, Any] = {"processed_dir": str(PROCESSED_DIR)}

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"PROCESSED_DIR not found: {PROCESSED_DIR}")

    hist_path = _pick_preferred_or_latest(PROCESSED_DIR, "merged_dataset", (".csv", ".xlsx"))
    if hist_path is None:
        raise FileNotFoundError(f"No history file found in {PROCESSED_DIR}")

    debug["hist_path"] = str(hist_path)
    df = _read_table(hist_path)

    month_col = _find_col(df, ["Month", "month", "Date", "date"])
    district_col = _find_col(df, ["District", "district"])
    price_col = _find_col(df, ["Median_Price", "median_price", "MedianPrice", "price"])

    debug["columns"] = list(df.columns)
    debug["detected"] = {"month_col": month_col, "district_col": district_col, "price_col": price_col}

    if not (month_col and district_col and price_col):
        raise ValueError(
            "History table missing required columns. "
            f"Detected: {debug['detected']}. Expected Month/District/Median_Price."
        )

    df = df.copy()
    df["_month"] = df[month_col].map(_to_month_str)
    df["_district_key"] = df[district_col].map(_norm_key)
    df["_price"] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=["_month", "_district_key", "_price"])
    debug["shape"] = list(df.shape)

    return df, debug


def _load_shap() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load shap long table from SHAP_DIR.
    Prefer file containing 'forecast_shap_long', else latest.
    """
    debug: Dict[str, Any] = {"shap_dir": str(SHAP_DIR)}

    if not SHAP_DIR.exists():
        raise FileNotFoundError(f"SHAP_DIR not found: {SHAP_DIR}")

    cands = [p for p in SHAP_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".parquet"}]
    if not cands:
        raise FileNotFoundError(f"No SHAP file found in {SHAP_DIR}")

    preferred = [p for p in cands if "forecast_shap_long" in p.name.lower()]
    shap_path = preferred[0] if preferred else sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    debug["shap_path"] = str(shap_path)

    if shap_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(shap_path)
    else:
        df = _read_table(shap_path)

    month_col = _find_col(df, ["Month", "month", "Date", "date"])
    district_col = _find_col(df, ["District", "district"])
    scenario_col = _find_col(df, ["Scenario", "scenario"])
    feat_col = _find_col(df, ["feature", "Feature"])
    shap_col = _find_col(df, ["shap_value", "SHAP", "shap", "value"])

    debug["columns"] = list(df.columns)
    debug["detected"] = {
        "month_col": month_col,
        "district_col": district_col,
        "scenario_col": scenario_col,
        "feat_col": feat_col,
        "shap_col": shap_col,
    }

    missing = [k for k, v in debug["detected"].items() if v is None]
    if missing:
        raise ValueError(f"SHAP table missing columns: {missing}. Got columns: {list(df.columns)}")

    df = df.copy()
    df["_month"] = df[month_col].map(_to_month_str)
    df["_district_key"] = df[district_col].map(_norm_key)
    df["_scenario"] = df[scenario_col].map(_norm_scenario)
    df["_feature"] = df[feat_col].astype(str)
    df["_shap"] = pd.to_numeric(df[shap_col], errors="coerce")

    df = df.dropna(subset=["_month", "_district_key", "_scenario", "_feature", "_shap"])
    debug["shape"] = list(df.shape)

    return df, debug


# ============================================================
# Core compute (align with Streamlit: anchor before forecast_start)
# ============================================================
def _get_forecast_start_month(
    preds: pd.DataFrame,
    district_key: str,
    scenario: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    sub = preds[(preds["_district_key"] == district_key) & (preds["_scenario"] == scenario)].copy()
    dbg["pred_rows_for_start"] = int(len(sub))
    if sub.empty:
        return None, dbg

    months = sorted([m for m in sub["_month"].dropna().unique().tolist() if isinstance(m, str)])
    if not months:
        return None, dbg

    dbg["forecast_start_month"] = months[0]
    dbg["forecast_end_month"] = months[-1]
    return months[0], dbg


def _get_current_price_before_forecast(
    hist: pd.DataFrame,
    district_key: str,
    forecast_start_month: str,
) -> Tuple[Optional[float], Dict[str, Any]]:
    dbg: Dict[str, Any] = {"forecast_start_month": forecast_start_month}
    sub = hist[hist["_district_key"] == district_key].copy()
    dbg["hist_rows"] = int(len(sub))
    if sub.empty:
        dbg["error"] = "no_history_rows_for_district"
        return None, dbg

    sub["_month_dt"] = pd.to_datetime(sub["_month"].astype(str) + "-01", errors="coerce")
    sub = sub.dropna(subset=["_month_dt"]).sort_values("_month_dt")
    if sub.empty:
        dbg["error"] = "history_month_parse_failed"
        return None, dbg

    fs_dt = pd.to_datetime(forecast_start_month + "-01", errors="coerce")
    before = sub[sub["_month_dt"] < fs_dt]

    if not before.empty:
        current = float(before["_price"].iloc[-1])
        dbg["current_month"] = str(before["_month"].iloc[-1])
        dbg["anchor_rule"] = "last_history_before_forecast_start"
        return current, dbg

    # fallback: no history before forecast start
    current = float(sub["_price"].iloc[-1])
    dbg["current_month"] = str(sub["_month"].iloc[-1])
    dbg["anchor_rule"] = "fallback_last_history_value"
    return current, dbg


def _get_future_price(
    preds: pd.DataFrame,
    district_key: str,
    month: str,
    scenario: str,
) -> Tuple[Optional[float], Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    sub = preds[
        (preds["_district_key"] == district_key)
        & (preds["_month"] == month)
        & (preds["_scenario"] == scenario)
    ].copy()
    dbg["pred_rows_for_month"] = int(len(sub))
    if sub.empty:
        return None, dbg
    val = float(sub["_pred"].mean())
    return val, dbg


def _get_shap_drivers(
    shap_df: pd.DataFrame,
    district_key: str,
    month: str,
    scenario: str,
    top_k: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    sub = shap_df[
        (shap_df["_district_key"] == district_key)
        & (shap_df["_month"] == month)
        & (shap_df["_scenario"] == scenario)
    ].copy()

    dbg["shap_rows"] = int(len(sub))
    if sub.empty:
        return {"drivers": {"up": [], "down": [], "all": []}}, dbg

    g = (
        sub.groupby("_feature")["_shap"]
        .agg(mean_shap="mean", mean_abs_shap=lambda x: float(np.mean(np.abs(x))))
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    up_df = g[g["mean_shap"] > 0].head(top_k)
    down_df = g[g["mean_shap"] < 0].sort_values("mean_shap", ascending=True).head(top_k)

    up = [
        {"feature": r["_feature"], "mean_shap": float(r["mean_shap"]), "mean_abs_shap": float(r["mean_abs_shap"])}
        for _, r in up_df.iterrows()
    ]
    down = [
        {"feature": r["_feature"], "mean_shap": float(r["mean_shap"]), "mean_abs_shap": float(r["mean_abs_shap"])}
        for _, r in down_df.iterrows()
    ]
    all_list = [
        {"feature": r["_feature"], "mean_shap": float(r["mean_shap"]), "mean_abs_shap": float(r["mean_abs_shap"])}
        for _, r in g.iterrows()
    ]

    return {"drivers": {"up": up, "down": down, "all": all_list}}, dbg


# ============================================================
# Route
# ============================================================
@router.post("/forecast_agent", response_model=ForecastAgentResponse)
def forecast_agent(req: ForecastAgentRequest) -> ForecastAgentResponse:
    """
    POST /api/forecast_agent

    Uses:
      - MODEL_DIR (pred_all_scenarios_long) -> future_price
      - PROCESSED_DIR merged_dataset -> current_price (anchor BEFORE forecast_start)
      - SHAP_DIR forecast_shap_long -> drivers

    Then uses SimpleForecastAgent to generate narrative & structured agent output.
    """
    scenario = _norm_scenario(req.scenario)
    month = _to_month_str(req.month)
    district_raw = _clean_str(req.district)
    top_k = int(req.top_k)

    if not month:
        raise HTTPException(status_code=422, detail="Invalid month. Expect 'YYYY-MM'.")

    district_key = _norm_key(district_raw)

    debug: Dict[str, Any] = {
        "parsed": {
            "scenario": scenario,
            "district": district_raw,
            "district_key": district_key,
            "month": month,
            "top_k": top_k,
        }
    }

    try:
        preds, pred_dbg = _load_predictions()
        hist, hist_dbg = _load_history()
        shap_df, shap_dbg = _load_shap()

        debug["paths"] = {
            "MODEL_DIR": str(MODEL_DIR),
            "PROCESSED_DIR": str(PROCESSED_DIR),
            "SHAP_DIR": str(SHAP_DIR),
            "pred_source": pred_dbg.get("pred_path"),
            "hist_source": hist_dbg.get("hist_path"),
            "shap_source": shap_dbg.get("shap_path"),
        }
        debug["shapes"] = {
            "pred": pred_dbg.get("shape_after_dropna") or pred_dbg.get("shape_before_dropna"),
            "hist": hist_dbg.get("shape"),
            "shap": shap_dbg.get("shape"),
        }
        debug["detected_columns"] = {
            "pred": pred_dbg.get("detected"),
            "hist": hist_dbg.get("detected"),
            "shap": shap_dbg.get("detected"),
        }

        # District validity check (more robust than only checking preds)
        available_pred = set(preds["_district_key"].dropna().unique().tolist())
        available_hist = set(hist["_district_key"].dropna().unique().tolist())
        debug["available_districts"] = {
            "pred_count": int(len(available_pred)),
            "hist_count": int(len(available_hist)),
        }

        if district_key not in available_pred and district_key not in available_hist:
            raise HTTPException(status_code=422, detail=f"Unknown district: {district_raw}")

        # forecast start month (for anchor)
        forecast_start_month, fs_dbg = _get_forecast_start_month(preds, district_key, scenario)
        debug["forecast_start_debug"] = fs_dbg

        if not forecast_start_month:
            raise HTTPException(
                status_code=422,
                detail="No forecast rows found for this district+scenario (cannot infer forecast_start).",
            )

        current_price, cur_dbg = _get_current_price_before_forecast(hist, district_key, forecast_start_month)
        future_price, fut_dbg = _get_future_price(preds, district_key, month, scenario)
        shap_pack, shap_slice_dbg = _get_shap_drivers(shap_df, district_key, month, scenario, top_k)

        debug["current_price_debug"] = cur_dbg
        debug["future_price_debug"] = fut_dbg
        debug["shap_slice_debug"] = shap_slice_dbg

        # pct_change in percent (align with Streamlit UI)
        pct_change_pct: Optional[float] = None
        if current_price is not None and future_price is not None and current_price != 0:
            pct_change_pct = (future_price / current_price - 1.0) * 100.0

        agent = SimpleForecastAgent(SimpleAgentConfig(top_n=top_k))
        agent_out = agent.run(
            district=district_raw,
            scenario=scenario,
            month=month,
            current_price=current_price,
            future_price=future_price,
            pct_change=pct_change_pct,  # percent
            shap_up=shap_pack["drivers"]["up"],
            shap_down=shap_pack["drivers"]["down"],
        )

        shap_for_ui = {
            "drivers": {
                "up": [x["feature"] for x in shap_pack["drivers"]["up"]],
                "down": [x["feature"] for x in shap_pack["drivers"]["down"]],
            },
            "details": shap_pack["drivers"],
        }

        return ForecastAgentResponse(
            ok=True,
            prediction={
                "scenario": scenario,
                "district": district_raw,
                "month": month,
                "forecast_start_month": forecast_start_month,
                "current_month": cur_dbg.get("current_month"),
                "current_price": current_price,
                "future_price": future_price,
                "pct_change": pct_change_pct,  # percent
            },
            shap=shap_for_ui,
            narrative=str(agent_out.get("narrative") or agent_out.get("headline") or ""),
            agent=agent_out,
            debug=debug,
        )

    except HTTPException:
        raise
    except Exception as e:
        debug["error"] = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=debug)
