# src/api/routes_explain.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.config import OUTPUT_DIR  # ensure OUTPUT_DIR exists

# main.py uses: app.include_router(explain_router, prefix="/api")
# final routes:
#   /api/explain/auto_summary
#   /api/explain/forecast_summary
#   /api/explain/reload_shap
router = APIRouter(prefix="/explain", tags=["explain"])


# ============================================================
# Paths (UPDATED: forecast SHAP long)
# ============================================================
SHAP_DIR = OUTPUT_DIR / "shap_forecast"
SHAP_LONG_PATH = SHAP_DIR / "forecast_shap_long.parquet"


# ============================================================
# Normalization helpers
# ============================================================
def _clean_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _validate_top_k(top_k: int) -> int:
    if top_k <= 0:
        return 8
    return min(int(top_k), 50)


def _norm_month_to_yyyy_mm(month: str) -> str:
    """
    Normalize inputs like:
      - YYYY-MM
      - YYYY-MM-DD
      - YYYY/MM
      - pandas datetime / excel-like dates
    into 'YYYY-MM' if parseable; otherwise returns stripped original.
    """
    s = _clean_str(month, "—")
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return dt.strftime("%Y-%m")

    # fallback slice for strings like "YYYY-MM-DD"
    if len(s) >= 7:
        s2 = s[:7].replace("/", "-")
        if len(s2) == 7 and s2[4] == "-":
            return s2
    return s


def _norm_scenario(scenario: str) -> str:
    return _clean_str(scenario, "base").lower()


def _norm_district(district: str) -> str:
    """
    Unify district naming between UI and SHAP files.

    - UI may send "Auckland City"
    - SHAP file may store "AucklandCity" or "Auckland City"

    We normalize by removing spaces, '-' and '_' and lowering.
    """
    s = _clean_str(district, "Unknown district")
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    return s.lower()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _format_feature_name(s: str) -> str:
    # Keep as-is for now; you can add a mapping later.
    return str(s)


# ============================================================
# Load + cache SHAP long
# ============================================================
@lru_cache(maxsize=1)
def _load_shap_long() -> pd.DataFrame:
    """
    Cache SHAP long table in memory for fast API responses.

    Expected columns:
      - feature, shap_value, abs_shap
      - identifiers: Month, District, Scenario
    """
    if not SHAP_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Missing SHAP long file: {SHAP_LONG_PATH}. "
            "Run: python -m src.explain.compute_shap_rf"
        )

    df = pd.read_parquet(SHAP_LONG_PATH)

    required = {"feature", "shap_value", "abs_shap"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"SHAP long file missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # ---- normalize identifiers (important for filtering) ----
    if "Scenario" in df.columns:
        df["Scenario"] = df["Scenario"].astype(str).str.strip().str.lower()

    if "District" in df.columns:
        # normalize same way as _norm_district: remove separators and lower
        df["District"] = (
            df["District"]
            .astype(str)
            .str.strip()
            .str.replace(" ", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace("_", "", regex=False)
            .str.lower()
        )

    if "Month" in df.columns:
        m = pd.to_datetime(df["Month"], errors="coerce")
        if m.notna().any():
            df["Month"] = m.dt.strftime("%Y-%m")
        else:
            df["Month"] = df["Month"].astype(str).str.strip().str.slice(0, 7).str.replace("/", "-", regex=False)

    # ensure numeric
    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce")
    df["abs_shap"] = pd.to_numeric(df["abs_shap"], errors="coerce")

    # direction (if missing)
    if "direction" not in df.columns:
        df["direction"] = df["shap_value"].apply(
            lambda v: "up" if _safe_float(v) is not None and float(v) >= 0 else "down"
        )

    # drop bad rows
    df = df.dropna(subset=["feature", "shap_value", "abs_shap"]).copy()
    df["feature"] = df["feature"].astype(str).str.strip()

    return df


def _clear_shap_cache() -> None:
    _load_shap_long.cache_clear()


# ============================================================
# Filtering + aggregation
# ============================================================
def _filter_slice(df: pd.DataFrame, district_norm: str, scenario_norm: str, month_yyyy_mm: str) -> pd.DataFrame:
    out = df
    if "District" in out.columns:
        out = out[out["District"] == district_norm]
    if "Scenario" in out.columns:
        out = out[out["Scenario"] == scenario_norm]
    if "Month" in out.columns:
        out = out[out["Month"] == month_yyyy_mm]
    return out


def _aggregate_drivers(df_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per feature within the slice.
    Output columns:
      - feature
      - mean_shap
      - mean_abs_shap
      - direction (based on mean_shap sign)
    """
    g = (
        df_slice.groupby("feature", as_index=False)
        .agg(mean_shap=("shap_value", "mean"), mean_abs_shap=("abs_shap", "mean"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    g["direction"] = np.where(g["mean_shap"] >= 0, "up", "down")
    return g


def _pack_driver_list(df_drivers: pd.DataFrame, top_k: int, direction: str) -> List[Dict[str, Any]]:
    d = df_drivers[df_drivers["direction"] == direction].head(top_k).copy()
    out: List[Dict[str, Any]] = []
    for _, row in d.iterrows():
        out.append(
            {
                "feature": _format_feature_name(row["feature"]),
                "mean_shap": float(row["mean_shap"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
            }
        )
    return out


def _pack_items(df_drivers: pd.DataFrame, top_k: int) -> List[Dict[str, Any]]:
    d = df_drivers.head(top_k).copy()
    items: List[Dict[str, Any]] = []
    for _, row in d.iterrows():
        items.append(
            {
                "feature": _format_feature_name(row["feature"]),
                "direction": row["direction"],
                "shap_value": float(row["mean_shap"]),
                "abs_shap": float(row["mean_abs_shap"]),
                "feature_raw": None,
            }
        )
    return items


def _build_narrative(
    district_ui: str,
    scenario: str,
    month: str,
    top_up: List[Dict[str, Any]],
    top_down: List[Dict[str, Any]],
) -> str:
    def _names(xs: List[Dict[str, Any]], n: int = 2) -> str:
        feats = [str(x.get("feature", "")) for x in xs[:n] if x.get("feature")]
        return ", ".join(feats)

    up2 = _names(top_up, 2)
    down2 = _names(top_down, 2)

    parts: List[str] = []
    parts.append(
        f"For **{district_ui}** under the **{scenario}** scenario in **{month}**, "
        "the model’s month-specific drivers are estimated from aggregated SHAP contributions."
    )

    if up2 and down2:
        parts.append(
            f"Upward pressure is mainly associated with **{up2}**, while downward pressure is mainly linked to **{down2}**."
        )
    elif up2:
        parts.append(
            f"Upward pressure is mainly associated with **{up2}**, while negative drivers appear relatively weaker in this month."
        )
    elif down2:
        parts.append(
            f"Downward pressure is mainly linked to **{down2}**, while positive drivers appear relatively weaker in this month."
        )
    else:
        parts.append("Top drivers are not available for this selection (insufficient SHAP records).")

    return " ".join(parts).strip()


def _debug_available_values(shap_long: pd.DataFrame) -> Dict[str, Any]:
    detail: Dict[str, Any] = {"available_columns": shap_long.columns.tolist()}
    if "District" in shap_long.columns:
        detail["available_districts"] = sorted(shap_long["District"].dropna().unique().tolist())
    if "Scenario" in shap_long.columns:
        detail["available_scenarios"] = sorted(shap_long["Scenario"].dropna().unique().tolist())
    if "Month" in shap_long.columns:
        ms = sorted(shap_long["Month"].dropna().unique().tolist())
        detail["available_months_head"] = ms[:12]
        detail["available_months_tail"] = ms[-12:] if len(ms) > 12 else ms
    return detail


# ============================================================
# Routes
# ============================================================
@router.get("/reload_shap")
def reload_shap() -> Dict[str, Any]:
    """
    Clear SHAP cache so the next request reloads the parquet.
    Useful after regenerating forecast_shap_long.parquet without restarting FastAPI.
    """
    _clear_shap_cache()
    return {"ok": True, "message": "SHAP cache cleared. Next call will reload parquet.", "source": str(SHAP_LONG_PATH)}


@router.get("/auto_summary")
def auto_summary(
    district: str = Query(..., description="District name, e.g., Auckland City"),
    scenario: str = Query(..., description="Scenario: base/low/high"),
    month: str = Query(..., description="Month in YYYY-MM or YYYY-MM-DD"),
    top_k: int = Query(8, ge=1, le=50, description="Top-K drivers to return"),
) -> Dict[str, Any]:
    """
    Narrative (auto) explanation endpoint (SHAP-based).

    Returns fields compatible with Streamlit UI:
      - narrative: str
      - drivers: { up: [...], down: [...] }
      - meta: echo inputs
    """
    district_ui = _clean_str(district, "Unknown district")
    district_norm = _norm_district(district_ui)
    scenario_norm = _norm_scenario(scenario)
    month_norm = _norm_month_to_yyyy_mm(month)
    top_k = _validate_top_k(int(top_k))

    try:
        shap_long = _load_shap_long()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    df_slice = _filter_slice(shap_long, district_norm=district_norm, scenario_norm=scenario_norm, month_yyyy_mm=month_norm)

    if df_slice.empty:
        detail: Dict[str, Any] = {
            "message": "No SHAP records found for this (district, scenario, month).",
            "requested": {
                "district_ui": district_ui,
                "district_norm": district_norm,
                "scenario": scenario_norm,
                "month": month_norm,
            },
            "source": str(SHAP_LONG_PATH),
        }
        detail.update(_debug_available_values(shap_long))
        raise HTTPException(status_code=404, detail=detail)

    drivers_df = _aggregate_drivers(df_slice)
    up_list = _pack_driver_list(drivers_df, top_k=top_k, direction="up")
    down_list = _pack_driver_list(drivers_df, top_k=top_k, direction="down")

    narrative = _build_narrative(
        district_ui=district_ui,
        scenario=scenario_norm,
        month=month_norm,
        top_up=up_list,
        top_down=down_list,
    )

    return {
        "narrative": narrative,
        "drivers": {"up": up_list, "down": down_list},
        "meta": {
            "district": district_ui,
            "district_norm": district_norm,
            "scenario": scenario_norm,
            "month": month_norm,
            "top_k": top_k,
            "mode": "auto_summary",
            "source": str(SHAP_LONG_PATH),
            "aggregation": "groupby(feature): mean(shap_value), mean(abs_shap)",
        },
    }


@router.get("/forecast_summary")
def forecast_summary(
    district: str = Query(..., description="District name, e.g., Auckland City"),
    scenario: str = Query(..., description="Scenario: base/low/high"),
    month: str = Query(..., description="Month in YYYY-MM or YYYY-MM-DD"),
    top_k: int = Query(8, ge=1, le=50, description="Top-K items to return"),
) -> Dict[str, Any]:
    """
    Forecast SHAP endpoint (SHAP-based), returns a ranked table-like list.

    Streamlit UI fields:
      - items: list[dict]
      - meta: echo inputs
    """
    district_ui = _clean_str(district, "Unknown district")
    district_norm = _norm_district(district_ui)
    scenario_norm = _norm_scenario(scenario)
    month_norm = _norm_month_to_yyyy_mm(month)
    top_k = _validate_top_k(int(top_k))

    try:
        shap_long = _load_shap_long()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    df_slice = _filter_slice(shap_long, district_norm=district_norm, scenario_norm=scenario_norm, month_yyyy_mm=month_norm)

    if df_slice.empty:
        detail: Dict[str, Any] = {
            "message": "No SHAP records found for this (district, scenario, month).",
            "requested": {
                "district_ui": district_ui,
                "district_norm": district_norm,
                "scenario": scenario_norm,
                "month": month_norm,
            },
            "source": str(SHAP_LONG_PATH),
        }
        detail.update(_debug_available_values(shap_long))
        raise HTTPException(status_code=404, detail=detail)

    drivers_df = _aggregate_drivers(df_slice)
    items = _pack_items(drivers_df, top_k=top_k)

    return {
        "items": items,
        "meta": {
            "district": district_ui,
            "district_norm": district_norm,
            "scenario": scenario_norm,
            "month": month_norm,
            "top_k": top_k,
            "mode": "forecast_summary",
            "source": str(SHAP_LONG_PATH),
            "aggregation": "groupby(feature): mean(shap_value), mean(abs_shap)",
        },
    }
