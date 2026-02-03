# src/api/routes_agent.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional, List

import requests
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.investment.investment_narrative import generate_investment_insight

router = APIRouter(tags=["agent"])

_DISTRICTS = [
    "Auckland City",
    "Franklin",
    "Manukau",
    "North Shore",
    "Papakura",
    "Rodney",
    "Waitakere",
]


# ============================================================
# Schemas
# ============================================================
class AgentRequest(BaseModel):
    user_query: str = Field(..., description="User question/query in natural language")
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    ok: bool
    mode: str
    result: Dict[str, Any]


# ============================================================
# Helpers
# ============================================================
def _clean_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _find_month(text: str) -> Optional[str]:
    """
    Match YYYY-MM or YYYY/MM, optionally with a day part (YYYY-MM-DD).
    Returns YYYY-MM.
    """
    m = re.search(r"\b(20\d{2})[-/](\d{2})(?:[-/]\d{2})?\b", text or "")
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def _norm_scenario(s: Any) -> str:
    ss = _clean_str(s).lower()
    return ss if ss in {"base", "low", "high"} else "base"


def _find_scenario(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\blow\b", t):
        return "low"
    if re.search(r"\bhigh\b", t):
        return "high"
    if re.search(r"\bbase\b", t):
        return "base"
    return "base"


def _norm_month(s: Any) -> Optional[str]:
    ss = _clean_str(s)
    if not ss:
        return None
    ss = ss.replace("/", "-")
    if len(ss) >= 7 and ss[:4].isdigit():
        return ss[:7]
    return _find_month(ss)


def _norm_key(x: Any) -> str:
    t = _clean_str(x).lower()
    return re.sub(r"[^a-z0-9]", "", t)


def _find_district(text: str) -> Optional[str]:
    t = (text or "").lower()

    # exact contains
    for d in _DISTRICTS:
        if d.lower() in t:
            return d

    # loose
    t2 = _norm_key(t)
    for d in _DISTRICTS:
        if _norm_key(d) in t2:
            return d

    return None

def _find_districts(text: str) -> List[str]:
    """
    Try to extract multiple districts from a query.
    Returns a de-duplicated list in appearance order.
    """
    t = (text or "").lower()
    found: List[str] = []

    # (1) exact contains
    for d in _DISTRICTS:
        if d.lower() in t and d not in found:
            found.append(d)

    # (2) loose matching (normalized)
    if not found:
        t2 = _norm_key(t)
        for d in _DISTRICTS:
            if _norm_key(d) in t2 and d not in found:
                found.append(d)

    # Special-case: user says "Auckland" (often means Auckland City in your districts list)
    # Only add Auckland City if no other Auckland-specific district already found.
    if re.search(r"\bauckland\b", t) and "Auckland City" not in found:
        found.insert(0, "Auckland City")

    # De-dupe while preserving order
    out: List[str] = []
    for d in found:
        if d not in out:
            out.append(d)
    return out


def _is_compare_intent(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    cues = ["compare", "vs", "versus", "difference", "differences", "between", "which is higher", "which is lower"]
    return any(c in t for c in cues)


def _is_investment_intent(text: str) -> bool:
    """
    Detect whether the user is asking for investment stability / risk-return advice.
    Keep it intentionally broad, but require a "choice/compare" cue to reduce false positives.
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    triggers = [
        "invest",
        "investment",
        "investing",
        "risk",
        "return",
        "risk-return",
        "drawdown",
        "volatility",
        "stable",
        "stability",
        "safe",
        "safer",
        "district",
        "ranking",
        "rank",
        "recommend",
        "recommendation",
        "portfolio",
    ]

    if not any(k in t for k in triggers):
        return False

    choose_cues = [
        "which",
        "choose",
        "compare",
        "ranking",
        "rank",
        "recommend",
        "best",
        "safer",
        "stable",
        "stability",
    ]
    return any(c in t for c in choose_cues)


def _parse_investment_params(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Investment narrative controls (client overrides via context).
    """
    tone = _clean_str(ctx.get("tone") or "cautious").lower()
    if tone not in {"neutral", "cautious", "confident"}:
        tone = "cautious"

    try:
        top_k = int(ctx.get("top_k", 3))
    except Exception:
        top_k = 3
    top_k = max(1, min(top_k, 10))

    include_profiles = bool(ctx.get("include_profiles", True))

    return {"tone": tone, "top_k": top_k, "include_profiles": include_profiles}


# ============================================================
# Route
# ============================================================
@router.post("/agent", response_model=AgentResponse)
def agent_chat(req: AgentRequest) -> AgentResponse:
    """
    POST /api/agent

    Modes:
    - investment: returns automated investment insight narrative (no district/month required)
    - forecast_agent: calls /api/forecast_agent with parsed district/month/scenario
    - help: guidance for missing params or downstream errors
    """
    q = _clean_str(req.user_query)
    ctx = req.context or {}

    # (1) Investment intent -> return investment narrative directly
    if _is_investment_intent(q):
        p = _parse_investment_params(ctx)
        out = generate_investment_insight(
            tone=p["tone"],
            top_k=p["top_k"],
            include_profiles=p["include_profiles"],
        )
        return AgentResponse(
            ok=True,
            mode="investment",
            result={
                "text": out.get("text", ""),
                "tone": out.get("tone", "info"),
                "meta": out.get("meta", {}),
                "profiles": out.get("profiles", {}),
                "source": out.get("source", ""),
                "received": {"user_query": q, "context": ctx},
                "parsed": {
                    "tone": p["tone"],
                    "top_k": p["top_k"],
                    "include_profiles": p["include_profiles"],
                },
            },
        )

    # (2) Otherwise -> forecast_agent routing (single or compare)
    districts = _find_districts(q)

    # context override support
    ctx_district = ctx.get("district")
    ctx_districts = ctx.get("districts")

    if not districts:
        if isinstance(ctx_districts, list):
            districts = [str(x).strip() for x in ctx_districts if str(x).strip()]
        elif ctx_district:
            d = _clean_str(ctx_district)
            if d:
                districts = [d]

    compare_intent = _is_compare_intent(q)

    # ✅ month handling:
    # - if user explicitly writes month in query -> use it
    # - if compare intent and no explicit month -> horizon (month=None), ignore ctx default month
    q_month = _find_month(q)
    if compare_intent and q_month is None:
        month = None
    else:
        month = q_month or _norm_month(ctx.get("month"))

    scenario = _norm_scenario(_find_scenario(q) or ctx.get("scenario"))


    try:
        top_k = int(ctx.get("top_k", 8))
    except Exception:
        top_k = 8
    top_k = max(1, min(top_k, 50))

    api_base = _clean_str(ctx.get("api_base") or "http://127.0.0.1:8000").rstrip("/")
    url = f"{api_base}/api/forecast_agent"

    compare_intent = _is_compare_intent(q)

    # ---- Compare mode: call forecast_agent twice ----
    if compare_intent and len(districts) >= 2:
        picked = districts[:2]  # compare first two
        outputs: Dict[str, Any] = {}
        errors: Dict[str, Any] = {}

        for d in picked:
            payload = {"scenario": scenario, "district": d, "top_k": top_k}
            if month:
                payload["month"] = month

            try:
                r = requests.post(url, json=payload, timeout=120)
                status = r.status_code
                try:
                    body = r.json()
                except Exception:
                    body = {"raw_text": r.text}

                if status >= 400:
                    errors[d] = {
                        "status_code": status,
                        "request_payload": payload,
                        "response_body": body,
                    }
                else:
                    outputs[d] = body

            except Exception as e:
                errors[d] = {
                    "error": f"{type(e).__name__}: {e}",
                    "request_payload": payload,
                }

        if errors:
            return AgentResponse(
                ok=False,
                mode="help",
                result={
                    "message": "Compare request failed when calling /api/forecast_agent. Open debug panel to see details.",
                    "downstream": {"url": url, "errors": errors, "success": list(outputs.keys())},
                    "received": {"user_query": q, "context": ctx},
                    "parsed": {"districts": picked, "month": month, "scenario": scenario, "top_k": top_k},
                    "partial": outputs,
                },
            )

        # simple comparison summary using pct_change if present
        def _pct_from_body(b: Dict[str, Any]) -> Optional[float]:
            try:
                val = (b.get("prediction") or {}).get("pct_change")
                if val is None:
                    return None
                if isinstance(val, str):
                    s = val.strip().replace("%", "").replace("+", "")
                    if s == "":
                        return None
                    return float(s)
                return float(val)
            except Exception:
                return None

        d1, d2 = picked[0], picked[1]
        p1, p2 = _pct_from_body(outputs[d1]), _pct_from_body(outputs[d2])

        if p1 is not None and p2 is not None:
            winner = d1 if p1 > p2 else d2
            loser = d2 if winner == d1 else d1

            win_val = (p1 if winner == d1 else p2)
            lose_val = (p2 if loser == d2 else p1)

            # ✅ wording depends on sign
            if win_val < 0 and lose_val < 0:
                summary = (
                    f"Comparison ({scenario}): {winner} has a smaller decline "
                    f"({win_val:+.2f}%) than {loser} ({lose_val:+.2f}%)."
                )
            elif win_val > 0 and lose_val > 0:
                summary = (
                    f"Comparison ({scenario}): {winner} has a larger increase "
                    f"({win_val:+.2f}%) than {loser} ({lose_val:+.2f}%)."
                )
            else:
                summary = (
                    f"Comparison ({scenario}): {winner} performs better "
                    f"({win_val:+.2f}%) than {loser} ({lose_val:+.2f}%)."
                )
        else:
            summary = f"Comparison ({scenario}): retrieved reports for {d1} and {d2} (pct_change missing in one/both)."


        return AgentResponse(
            ok=True,
            mode="forecast_compare",
            result={
                "summary": summary,
                "districts": picked,
                "scenario": scenario,
                "month": month,  # may be None => horizon
                "reports": outputs,
                "received": {"user_query": q, "context": ctx},
                "parsed": {"districts": picked, "month": month, "scenario": scenario, "top_k": top_k},
            },
        )

    # ---- Single mode fallback (call forecast_agent once) ----
    district = districts[0] if districts else None

    if not district:
        return AgentResponse(
            ok=True,
            mode="help",
            result={
                "message": (
                    "Please provide a district, e.g. 'Explain North Shore forecast', "
                    "or ask 'Compare North Shore and Manukau'. "
                    "Month is optional now (if omitted, the agent summarises the full forecast horizon)."
                ),
                "accepted_context_keys": [
                    "district",
                    "districts",
                    "month",
                    "scenario",
                    "top_k",
                    "api_base",
                    "tone",
                    "include_profiles",
                ],
                "received": {"user_query": q, "context": ctx},
                "parsed": {"district": district, "month": month, "scenario": scenario, "top_k": top_k},
            },
        )

    payload = {"scenario": scenario, "district": district, "top_k": top_k}
    if month:
        payload["month"] = month

    try:
        r = requests.post(url, json=payload, timeout=120)
        status = r.status_code

        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text}

        if status >= 400:
            return AgentResponse(
                ok=False,
                mode="help",
                result={
                    "message": "Downstream /api/forecast_agent call failed. Open debug panel to see details.",
                    "downstream": {
                        "url": url,
                        "status_code": status,
                        "request_payload": payload,
                        "response_body": body,
                    },
                    "received": {"user_query": q, "context": ctx},
                    "parsed": {
                        "district": district,
                        "month": month,
                        "scenario": scenario,
                        "top_k": top_k,
                    },
                },
            )

        return AgentResponse(
            ok=True,
            mode="forecast_agent",
            result=body if isinstance(body, dict) else {"raw": body},
        )

    except Exception as e:
        return AgentResponse(
            ok=False,
            mode="help",
            result={
                "message": "Could not call downstream /api/forecast_agent endpoint.",
                "error": f"{type(e).__name__}: {e}",
                "downstream": {"url": url, "request_payload": payload},
                "received": {"user_query": q, "context": ctx},
                "parsed": {
                    "district": district,
                    "month": month,
                    "scenario": scenario,
                    "top_k": top_k,
                },
            },
        )
