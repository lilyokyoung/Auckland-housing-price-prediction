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

# 建议你在 src/config.py 里已经有 DATA_DIR / OUTPUT_DIR
from src.config import DATA_DIR, OUTPUT_DIR

router = APIRouter(tags=["forecast_agent"])


# ============================================================
# Directories (match your description)
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

    # already timestamp?
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m")

    s = _clean_str(x)
    if not s:
        return None

    # quick normalize
    s2 = s.replace("/", "-")

    # match YYYY-MM or YYYY-MM-DD
    m = re.search(r"\b(20\d{2})-(\d{2})(?:-(\d{2}))?\b", s2)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # fallback: try parse by pandas
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return None
        return pd.Timestamp(dt).strftime("%Y-%m")
    except Exception:
        return None


def _pick_latest_file(folder: Path, exts: Tuple[str, ...] = (".csv", ".xlsx")) -> Optional[Path]:
    if not folder.exists():
        return None
    cands = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


# ============================================================
# Loaders
# ============================================================
def _load_predictions() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load long predictions table (base/high/low).
    You said: pred_all_scenarios_long (Excel) under MODEL_DIR.
    """
    debug: Dict[str, Any] = {"model_dir": str(MODEL_DIR)}

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")

    # pick best candidate
    # prefer file containing 'pred_all_scenarios_long' else latest
    cands = [p for p in MODEL_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}]
    if not cands:
        raise FileNotFoundError(f"No prediction table found in {MODEL_DIR}")

    preferred = [p for p in cands if "pred_all_scenarios_long" in p.name.lower()]
    pred_path = preferred[0] if preferred else sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    debug["pred_path"] = str(pred_path)

    df = _read_table(pred_path)

    # detect columns
    month_col = _find_col(df, ["Month", "month", "Date", "date"])
    district_col = _find_col(df, ["District", "district"])
    scenario_col = _find_col(df, ["Scenario", "scenario"])
    pred_col = _find_col(df, ["pred_Median_Price", "prediction", "pred", "yhat", "y_pred"])

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

    debug["shape"] = list(df.shape)
    debug["na_pred"] = int(df["_pred"].isna().sum())

    return df, debug


def _load_history() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load history table from PROCESSED_DIR.
    You said: price col = Median_Price, no lags.
    We'll pick latest merged_dataset*.csv/xlsx.
    """
    debug: Dict[str, Any] = {"processed_dir": str(PROCESSED_DIR)}

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"PROCESSED_DIR not found: {PROCESSED_DIR}")

    hist_path = _pick_latest_file(PROCESSED_DIR, (".csv", ".xlsx"))
    if hist_path is None:
        raise FileNotFoundError(f"No history file found in {PROCESSED_DIR}")

    debug["hist_path"] = str(hist_path)

    df = _read_table(hist_path)

    month_col = _find_col(df, ["Month", "month", "Date", "date"])
    district_col = _find_col(df, ["District", "district"])
    price_col = _find_col(df, ["Median_Price", "median_price", "price", "MedianPrice"])

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

    # keep valid rows
    df = df.dropna(subset=["_month", "_district_key", "_price"])
    debug["shape"] = list(df.shape)

    return df, debug


def _load_shap() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load shap long table from SHAP_DIR/forecast_shap_long.(csv|xlsx|parquet).
    Your file has columns: Month, District, Scenario, feature, shap_value
    """
    debug: Dict[str, Any] = {"shap_dir": str(SHAP_DIR)}

    if not SHAP_DIR.exists():
        raise FileNotFoundError(f"SHAP_DIR not found: {SHAP_DIR}")

    # prefer 'forecast_shap_long'
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
# Core compute
# ============================================================
def _get_current_price(hist: pd.DataFrame, district_key: str) -> Tuple[Optional[float], Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    sub = hist[hist["_district_key"] == district_key].copy()
    dbg["hist_rows"] = int(len(sub))
    if sub.empty:
        return None, dbg

    # sort by month
    sub["_month_dt"] = pd.to_datetime(sub["_month"].astype(str) + "-01", errors="coerce")
    sub = sub.dropna(subset=["_month_dt"]).sort_values("_month_dt")
    if sub.empty:
        return None, dbg

    current = float(sub["_price"].iloc[-1])
    dbg["current_month"] = sub["_month"].iloc[-1]
    return current, dbg


def _get_future_price(
    preds: pd.DataFrame, district_key: str, month: str, scenario: str
) -> Tuple[Optional[float], Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    sub = preds[
        (preds["_district_key"] == district_key)
        & (preds["_month"] == month)
        & (preds["_scenario"] == scenario)
    ].copy()
    dbg["pred_rows"] = int(len(sub))

    if sub.empty:
        return None, dbg

    # if multiple rows, take mean
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

    # aggregate to produce mean_shap + mean_abs_shap (fix your earlier missing_columns issue)
    g = (
        sub.groupby("_feature")["_shap"]
        .agg(mean_shap="mean", mean_abs_shap=lambda x: float(np.mean(np.abs(x))))
        .reset_index()  # make _feature a column so sort_values accepts 'by'
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # split
    up_df = g[g["mean_shap"] > 0].head(top_k)
    down_df = g[g["mean_shap"] < 0].head(top_k)

    up = [
        {"feature": r["_feature"], "mean_shap": float(r["mean_shap"]), "mean_abs_shap": float(r["mean_abs_shap"])}
        for _, r in up_df.iterrows()
    ]
    down = [
        {"feature": r["_feature"], "mean_shap": float(r["mean_shap"]), "mean_abs_shap": float(r["mean_abs_shap"])}
        for _, r in down_df.iterrows()
    ]

    # keep full list for debug / agent
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
      - PROCESSED_DIR merged_dataset -> current_price
      - SHAP_DIR forecast_shap_long -> drivers (aggregated from shap_value)

    Then uses SimpleForecastAgent to generate narrative & structured agent output.
    """
    scenario = _norm_scenario(req.scenario)
    month = _to_month_str(req.month)
    district_raw = _clean_str(req.district)

    if not month:
        raise HTTPException(status_code=422, detail="Invalid month. Expect 'YYYY-MM'.")

    district_key = _norm_key(district_raw)
    top_k = int(req.top_k)

    debug: Dict[str, Any] = {"parsed": {"scenario": scenario, "district": district_raw, "month": month, "top_k": top_k}}
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
            "pred": pred_dbg.get("shape"),
            "hist": hist_dbg.get("shape"),
            "shap": shap_dbg.get("shape"),
        }
        debug["detected_columns"] = {
            "pred": pred_dbg.get("detected"),
            "hist": hist_dbg.get("detected"),
            "shap": shap_dbg.get("detected"),
        }

        current_price, cur_dbg = _get_current_price(hist, district_key)
        future_price, fut_dbg = _get_future_price(preds, district_key, month, scenario)
        shap_pack, shap_slice_dbg = _get_shap_drivers(shap_df, district_key, month, scenario, top_k)

        debug["current_price_debug"] = cur_dbg
        debug["future_price_debug"] = fut_dbg
        debug["shap_slice_debug"] = shap_slice_dbg

        pct_change = None
        if current_price is not None and future_price is not None and current_price != 0:
            pct_change = (future_price - current_price) / current_price

        # Build narrative using SimpleAgent
        agent = SimpleForecastAgent(SimpleAgentConfig(top_n=top_k))
        agent_out = agent.run(
            district=district_raw,
            scenario=scenario,
            month=month,
            current_price=current_price,
            future_price=future_price,
            pct_change=pct_change,
            shap_up=shap_pack["drivers"]["up"],
            shap_down=shap_pack["drivers"]["down"],
        )

        # Provide UI-friendly driver lists (feature names)
        shap_for_ui = {
            "drivers": {
                "up": [x["feature"] for x in shap_pack["drivers"]["up"]],
                "down": [x["feature"] for x in shap_pack["drivers"]["down"]],
            },
            # keep details for debug/agent
            "details": shap_pack["drivers"],
        }

        return ForecastAgentResponse(
            ok=True,
            prediction={
                "scenario": scenario,
                "district": district_raw,
                "month": month,
                "current_price": current_price,
                "future_price": future_price,
                "pct_change": pct_change,
            },
            shap=shap_for_ui,
            narrative=str(agent_out.get("narrative") or agent_out.get("headline") or ""),
            agent=agent_out,
            debug=debug,
        )

    except Exception as e:
        # 保持 500，但给更多 debug 线索（你 Streamlit debug 面板会显示 raw_text=Internal Server Error）
        debug["error"] = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=debug["error"])
