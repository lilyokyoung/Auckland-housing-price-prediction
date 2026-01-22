# src/ui/api_client.py
from __future__ import annotations

from typing import Any, Dict, Optional
import requests


def _join_url(api_base: str, path: str) -> str:
    """
    Join api_base and path safely.

    - Strips leading/trailing whitespace
    - Removes accidental spaces inside the path (e.g., "auto_ summary")
    - Ensures exactly one "/" between base and path
    """
    api_base = (api_base or "").strip().rstrip("/")

    # IMPORTANT: clean path to avoid 'auto_ summary' -> 'auto_%20summary' -> 404
    path = (path or "").strip().replace(" ", "")

    if not path.startswith("/"):
        path = "/" + path

    return api_base + path


def get_json(
    api_base: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Send GET request to FastAPI and return JSON response."""
    url = _join_url(api_base, path)
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_json(
    api_base: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Send POST request to FastAPI and return JSON response."""
    url = _join_url(api_base, path)
    r = requests.post(url, json=(payload or {}), timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_health(api_base: str) -> Dict[str, Any]:
    return get_json(api_base, "/health", timeout=15)


# ============================================================
# Predict
# ============================================================
def post_predict(
    api_base: str,
    scenario: str,
    districts: Optional[list[str]] = None,
    preview_rows: int = 500,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "scenario": str(scenario),
        "districts": districts,
        "preview_rows": int(preview_rows),
    }
    return post_json(api_base, "/api/predict", payload=payload, timeout=600)


# ============================================================
# Explain
# ============================================================
def get_explain_auto_summary(
    api_base: str,
    district: str,
    scenario: str,
    month: str,
    top_k: int = 8,
) -> Dict[str, Any]:
    return get_json(
        api_base,
        "/api/explain/auto_summary",
        params={
            "district": str(district),
            "scenario": str(scenario),
            "month": str(month),
            "top_k": int(top_k),
        },
        timeout=60,
    )


# ============================================================
# Agent  ✅ NEW (fix 422 by enforcing schema)
# ============================================================
def post_agent(
    api_base: str,
    user_query: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    POST /api/agent
    Enforce AgentRequest schema exactly:
      { "user_query": <str>, "context": <dict> }

    This prevents Streamlit pages from accidentally sending:
    - wrong field names (query/message/text)
    - non-dict context
    - Swagger's default "additionalProp1" junk
    """
    ctx: Dict[str, Any] = {}
    if isinstance(context, dict):
        ctx = context
    elif context is None:
        ctx = {}
    else:
        # If someone passes non-dict, coerce to dict safely
        ctx = {"_raw_context": str(context)}

    payload: Dict[str, Any] = {
        "user_query": str(user_query),
        "context": ctx,
    }
    return post_json(api_base, "/api/agent", payload=payload, timeout=timeout)
