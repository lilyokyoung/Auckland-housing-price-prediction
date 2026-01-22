# src/api/main.py
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI

# -------------------------
# config checks (fail fast)
# -------------------------
from src.config import (
    verify_project_layout,
    FORECAST_FUTURE_DATASETS,
    PREPROCESSED_DIR,
    MODEL_DIR,
    SHAP_DIR,
)

# -------------------------
# import routers
# -------------------------
from src.api.routes_meta import router as meta_router
from src.api.routes_predict import router as predict_router
from src.api.routes_spatial import router as spatial_router
from src.api.routes_explain import router as explain_router
from src.api.routes_narratives import router as narratives_router
from src.api.routes_agent import router as agent_router

import importlib
_forecast_agent_module = importlib.import_module("src.api.routes_forecast_agent")
forecast_agent_router = getattr(_forecast_agent_module, "router")



# =====================================================
# Startup checks
# =====================================================
def _fmt_paths(paths: List[Path]) -> str:
    return "\n".join(f"- {p}" for p in paths)


def _fail_if_missing(required: List[Path], title: str) -> None:
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = (
            f"{title}\n"
            f"Missing required files:\n{_fmt_paths(missing)}\n\n"
            f"Tip: check your repo layout, file names, and src/config.py paths."
        )
        raise FileNotFoundError(msg)


def _warn_optional(optional: List[Path]) -> List[Path]:
    """Return list of missing optional files (for logging/diagnostics if needed)."""
    return [p for p in optional if not p.exists()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) Check core directory layout
    verify_project_layout(strict=True)

    # 2) Check critical files used by /api/predict
    required_predict_files = [
        MODEL_DIR / "avms" / "rf_final_model" / "rf_final_model.joblib",
        PREPROCESSED_DIR / "avms" / "final_for_AVMs.csv",
        FORECAST_FUTURE_DATASETS / "future_for_AVMs_base.csv",
        FORECAST_FUTURE_DATASETS / "future_for_AVMs_low.csv",
        FORECAST_FUTURE_DATASETS / "future_for_AVMs_high.csv",
    ]
    _fail_if_missing(required_predict_files, title="[Startup] Predict pipeline prerequisites not found.")

    # 3) Optional artifacts used by /api/explain (do NOT block startup)
    optional_explain_files = [
        SHAP_DIR / "forecast_shap_long.csv",
        SHAP_DIR / "feature_name_map.csv",
        SHAP_DIR / "shap_importance_rf.xlsx",
    ]
    missing_optional = _warn_optional(optional_explain_files)

    # Save diagnostics into app.state (UI/debug panel could read these later)
    app.state.startup_diagnostics = {
        "missing_optional_explain_files": [str(p) for p in missing_optional],
    }

    # If you want to surface this in logs, you can uncomment:
    # if missing_optional:
    #     print("[Startup] Optional explain artifacts missing:\n" + _fmt_paths(missing_optional))

    yield

    # shutdown (optional)
    # nothing to clean up


# =====================================================
# create app
# =====================================================
app = FastAPI(
    title="Auckland Housing Price AVM API",
    description="AI-powered property valuation and forecasting system for Auckland districts",
    version="0.1.0",
    lifespan=lifespan,
)

# -------------------------
# register routers
# -------------------------
app.include_router(meta_router)                     # /health, /capabilities
app.include_router(predict_router, prefix="/api")   # /api/predict
app.include_router(spatial_router, prefix="/api")   # /api/spatial/*
app.include_router(explain_router, prefix="/api")   # /api/explain/*
app.include_router(narratives_router, prefix="/api") 
app.include_router(agent_router, prefix="/api")  # -> /api/agent
app.include_router(forecast_agent_router, prefix="/api")  # /api/forecast_agent
# -------------------------
# root endpoint (optional but nice)
# -------------------------
@app.get("/")
def root():
    diag = getattr(app.state, "startup_diagnostics", {}) or {}
    missing_optional = diag.get("missing_optional_explain_files", [])

    return {
        "service": "Auckland Housing Price AVM API",
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "diagnostics": {
            "missing_optional_explain_files": missing_optional,
        },
    }

