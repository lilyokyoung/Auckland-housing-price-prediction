# pages/9_AI_Agent_Optional.py
from __future__ import annotations
import os

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# Minimal bootstrap so "import src.*" works
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="AI Agent (Optional) | Auckland Housing Price AVM", layout="wide")
st.markdown("## Agent Chat")
st.caption("Chat with your AVM agent (narrative, metrics, SHAP, investment insight, etc.).")


# ============================================================
# Global shared state
# ============================================================
DEFAULT_API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
if "api_base" not in st.session_state:
    st.session_state["api_base"] = DEFAULT_API_BASE
if "district" not in st.session_state:
    st.session_state["district"] = "Auckland City"
if "pred_scenario" not in st.session_state:
    st.session_state["pred_scenario"] = "base"
if "explain_month" not in st.session_state:
    st.session_state["explain_month"] = "2026-06"

if "agent_messages" not in st.session_state:
    # each message: {"role":"user"|"assistant", "content": str, "raw": Optional[dict]}
    st.session_state["agent_messages"] = []


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    # =========================
    # (A) Demo controls (keep)
    # =========================
    st.markdown("### Investment controls")
    st.selectbox(
        "Tone",
        options=["cautious", "neutral", "confident"],
        index=0,
        help="Controls wording style for investment insight narrative.",
        key="inv_tone_select",
    )
    st.toggle(
        "Include investor profiles",
        value=True,
        help="Include conservative / balanced / growth paragraphs.",
        key="inv_profiles_toggle",
    )
    st.slider(
        "Top-K districts",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="inv_top_k_slider",
    )

    # =========================
    # (B) Debug toggle (optional)
    # =========================
    show_debug = st.toggle(
        "Show debug panels",
        value=False,
        help="Show raw payloads for debugging.",
        key="show_debug_panels_toggle",
    )

    # Always define safe defaults (so code below never breaks)
    endpoint = "/api/agent"
    timeout_s = int(st.session_state.get("agent_timeout", 300))

    # Keep api_base internal only (no UI exposure)
    api_base = (st.session_state.get("api_base", DEFAULT_API_BASE) or "").strip()

    # =========================
    # (C) Debug-only controls (hide in demo)
    # =========================
    if show_debug:
        st.markdown("---")
        st.markdown("### Debug controls")

        endpoint = st.selectbox(
            "Agent endpoint",
            options=["/api/agent", "/api/forecast_agent"],
            index=0,
            help="Use /api/agent for natural language chat. /api/forecast_agent is one-shot structured report.",
            key="agent_endpoint_select",
        )

        timeout_s = st.slider(
            "Timeout (seconds)",
            min_value=30,
            max_value=600,
            value=int(st.session_state.get("agent_timeout", 300)),
            step=10,
            key="agent_timeout_slider",
        )
        st.session_state["agent_timeout"] = int(timeout_s)

        def _test_connection() -> None:
            try:
                r = requests.get(f"{api_base.rstrip('/')}/health", timeout=10)
                if r.status_code == 200:
                    st.success("API reachable: /health OK")
                else:
                    st.warning(f"API reachable but /health returned {r.status_code}")
            except Exception as e:
                st.error(f"Could not reach API: {type(e).__name__}: {e}")

        st.button("Test agent connection", use_container_width=True, on_click=_test_connection)


# ============================================================
# Formatting helpers
# ============================================================
def _fmt_money(x: Any) -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        return f"{v:,.0f}"
    except Exception:
        return "—"


def _fmt_pct(x: Any) -> str:
    try:
        if x is None:
            return "—"
        v = float(x) * 100.0
        return f"{v:.2f}%"
    except Exception:
        return "—"


def _emoji_for_district(name: str) -> str:
    s = (name or "").lower().replace(" ", "")
    if "aucklandcity" in s or ("city" in s and "auckland" in s):
        return "🏙️"
    if "northshore" in s or "shore" in s:
        return "🌊"
    if "rodney" in s:
        return "🌿"
    if "franklin" in s:
        return "🌾"
    if "waitakere" in s:
        return "🏞️"
    if "manukau" in s:
        return "🚆"
    if "papakura" in s:
        return "🏡"
    return "📍"

def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has duplicate column names, coalesce them into one column by taking
    the first non-null value across duplicates (row-wise), then drop extras.
    This prevents df["col"] returning a DataFrame (2D) which breaks pd.to_numeric.
    """
    if df.empty:
        return df

    cols = list(df.columns)
    dup_names = [c for c in pd.Index(cols) if cols.count(c) > 1]
    if not dup_names:
        return df

    # process each duplicated name once
    for name in sorted(set(dup_names)):
        block = df.loc[:, name]  # this is a DataFrame when duplicates exist
        if isinstance(block, pd.DataFrame):
            # take first non-null across duplicates for each row
            df[name] = block.bfill(axis=1).iloc[:, 0] # type: ignore

    # keep first occurrence of each column name
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df



# ============================================================
# Investment metrics extraction + charts
# ============================================================
_EXPECTED_METRIC_COLS = [
    "district",
    "base_return",
    "volatility",
    "low_scenario_drawdown",
    "scenario_spread",
    "stability_score",
]

# Default local metrics file (relative path, per your request)
DEFAULT_INVESTMENT_SOURCE = str((PROJECT_ROOT / "outputs" / "investment" / "metrics_by_district.csv").resolve())


def _is_path_like(x: Any) -> bool:
    """Return True only for safe, local file-like paths (not URLs)."""
    if x is None:
        return False
    if isinstance(x, Path):
        return True
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return False
        # reject URLs
        if s.lower().startswith(("http://", "https://")):
            return False
        return True
    return False


def _resolve_source_path(source: Any) -> Optional[Path]:
    """
    Resolve 'source' to a readable local file Path.
    Security/robustness:
    - Reject URLs
    - If absolute path is outside PROJECT_ROOT, ignore it (use default)
    """
    if not _is_path_like(source):
        source = DEFAULT_INVESTMENT_SOURCE

    s = str(source).strip().strip('"').strip("'")

    # if still looks like URL, ignore
    if s.lower().startswith(("http://", "https://")):
        s = DEFAULT_INVESTMENT_SOURCE

    p = Path(s)

    # If relative, resolve under PROJECT_ROOT
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        # If absolute but NOT under PROJECT_ROOT, treat as unsafe and fallback
        try:
            proj = str(PROJECT_ROOT.resolve())
            if not str(p).startswith(proj):
                p = Path(DEFAULT_INVESTMENT_SOURCE)
        except Exception:
            p = Path(DEFAULT_INVESTMENT_SOURCE)

    return p


def _load_metrics_from_source_path(source: Any) -> pd.DataFrame:
    """
    Load metrics_by_district.csv into a standardised DataFrame.
    Returns empty df if file doesn't exist or cannot be read.
    IMPORTANT: Never crash the whole page due to PermissionError.
    """
    p = _resolve_source_path(source)
    if p is None:
        return pd.DataFrame()

    # ✅ SAFE exists/is_file checks (avoid PermissionError crashing app)
    try:
        if (not p.exists()) or (not p.is_file()):
            return pd.DataFrame()
    except PermissionError:
        return pd.DataFrame()
    except OSError:
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Normalise district string
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip()
        df = df[df["district"].astype(bool)]

    # Map your actual csv columns -> UI columns
    rename_map = {
        "District": "district",
        "district_name": "district",
        "name": "district",
        "expected_return_base": "base_return",
        "expected_return_base_pct": "base_return",
        "return": "base_return",
        "baseExpectedReturn": "base_return",
        "base_expected_return": "base_return",
        "expected_volatility_base": "volatility",
        "volatility_base": "volatility",
        "vol": "volatility",
        "risk": "volatility",
        "downside_max_drawdown_low": "low_scenario_drawdown",
        "downside_max_drawdown_low_pct": "low_scenario_drawdown",
        "drawdown": "low_scenario_drawdown",
        "low_drawdown": "low_scenario_drawdown",
        "scenario_spread_high_minus_low": "scenario_spread",
        "spread": "scenario_spread",
        "stability_score": "stability_score",
        "stability": "stability_score",
        "score": "stability_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df = _coalesce_duplicate_columns(df)

    # If base_return came from *_pct column, it might be 3.04 not 0.0304. Detect and convert to ratio.
    if "base_return" in df.columns:
        br = pd.to_numeric(df["base_return"], errors="coerce")
        if br.notna().any() and br.median(skipna=True) > 0.5:
            df["base_return"] = br / 100.0
        else:
            df["base_return"] = br

    # If drawdown came from *_pct column, it might be -7.38 not -0.0738. Detect similarly.
    if "low_scenario_drawdown" in df.columns:
        dd = pd.to_numeric(df["low_scenario_drawdown"], errors="coerce")
        if dd.notna().any() and dd.abs().median(skipna=True) > 0.5:
            df["low_scenario_drawdown"] = dd / 100.0
        else:
            df["low_scenario_drawdown"] = dd

    for col in ["volatility", "scenario_spread", "stability_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in _EXPECTED_METRIC_COLS if c in df.columns]
    if keep:
        df = df[keep].copy()

    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip()
        df = df[df["district"].astype(bool)]

    return df.reset_index(drop=True)



def _coerce_metrics_df(meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Try to extract structured district metrics from meta, if backend sends list-of-dicts.
    """
    if not isinstance(meta, dict):
        return pd.DataFrame()

    candidates = [
        meta.get("metrics"),
        meta.get("district_metrics"),
        meta.get("investment_metrics"),
        meta.get("rows"),
        meta.get("table"),
        meta.get("data"),
    ]

    rows = None
    for c in candidates:
        if isinstance(c, list) and c and all(isinstance(x, dict) for x in c):
            rows = c
            break

    if rows is None:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()

    rename_map = {
        "District": "district",
        "district_name": "district",
        "name": "district",
        "return": "base_return",
        "expected_return_base": "base_return",
        "expected_return_base_pct": "base_return",
        "baseExpectedReturn": "base_return",
        "base_expected_return": "base_return",
        "vol": "volatility",
        "volatility_base": "volatility",
        "expected_volatility_base": "volatility",
        "risk": "volatility",
        "drawdown": "low_scenario_drawdown",
        "downside_max_drawdown_low": "low_scenario_drawdown",
        "downside_max_drawdown_low_pct": "low_scenario_drawdown",
        "low_drawdown": "low_scenario_drawdown",
        "spread": "scenario_spread",
        "scenario_spread_high_minus_low": "scenario_spread",
        "stability": "stability_score",
        "score": "stability_score",
        "stability_score": "stability_score",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    
    df = _coalesce_duplicate_columns(df)


    # numeric conversions
    if "base_return" in df.columns:
        br = pd.to_numeric(df["base_return"], errors="coerce")
        if br.notna().any() and br.median(skipna=True) > 0.5:
            df["base_return"] = br / 100.0
        else:
            df["base_return"] = br

    if "low_scenario_drawdown" in df.columns:
        dd = pd.to_numeric(df["low_scenario_drawdown"], errors="coerce")
        if dd.notna().any() and dd.abs().median(skipna=True) > 0.5:
            df["low_scenario_drawdown"] = dd / 100.0
        else:
            df["low_scenario_drawdown"] = dd

    for col in ["volatility", "scenario_spread", "stability_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip()
        df = df[df["district"].astype(bool)]

    keep_cols = [c for c in _EXPECTED_METRIC_COLS if c in df.columns]
    if keep_cols:
        df = df[keep_cols]

    return df.reset_index(drop=True)


# --- Fallback regex parser from narrative text ---
_LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z\s]+?)\s*:\s*"
    r"base return\s*([+-]?\d+(?:\.\d+)?)%\s*,\s*"
    r"volatility\s*([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"low-scenario drawdown\s*([+-]?\d+(?:\.\d+)?)%\s*,\s*"
    r"scenario spread\s*([\d,]+)\s*,\s*"
    r"stability score\s*([+-]?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def _parse_metrics_from_text(full_text: str) -> pd.DataFrame:
    """
    Fallback: parse metrics from narrative bullets under 'District highlights'.
    Works when backend does NOT provide meta.metrics and CSV not readable.
    """
    if not full_text or not isinstance(full_text, str):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for raw_line in full_text.splitlines():
        line = raw_line.strip().lstrip("-•").strip()
        m = _LINE_RE.match(line)
        if not m:
            continue

        district = m.group(1).strip()
        base_return = float(m.group(2)) / 100.0
        volatility = float(m.group(3))
        low_dd = float(m.group(4)) / 100.0
        spread = float(m.group(5).replace(",", ""))
        score = float(m.group(6))

        rows.append(
            {
                "district": district,
                "base_return": base_return,
                "volatility": volatility,
                "low_scenario_drawdown": low_dd,
                "scenario_spread": spread,
                "stability_score": score,
            }
        )

    return pd.DataFrame(rows)


def _extract_investment_metrics(source: Any, meta: Dict[str, Any], text: str) -> pd.DataFrame:
    """
    Priority:
      1) load from result["source"] (relative or absolute)
      2) load from default outputs/investment/metrics_by_district.csv
      3) parse from meta structured list (meta.metrics etc.)
      4) parse from narrative text bullets
      5) last fallback: meta.highlights list-of-dicts (often truncated)
    """
    # 1) try explicit source
    df = _load_metrics_from_source_path(source)
    if not df.empty:
        return df

    # 2) default
    df = _load_metrics_from_source_path(DEFAULT_INVESTMENT_SOURCE)
    if not df.empty:
        return df

    # 3) meta list-of-dicts (if backend ever sends it)
    df = _coerce_metrics_df(meta)
    if not df.empty:
        return df

    # 4) text bullets
    df = _parse_metrics_from_text(text)
    if not df.empty:
        return df

    # 5) meta.highlights (may be only 5)
    highs = meta.get("highlights")
    if isinstance(highs, list) and highs and all(isinstance(x, dict) for x in highs):
        dfh = pd.DataFrame(highs).copy()
        dfh = dfh.rename(
            columns={
                "district": "district",
                "expected_return_base": "base_return",
                "volatility_base": "volatility",
                "downside_max_drawdown_low": "low_scenario_drawdown",
                "scenario_spread_high_minus_low": "scenario_spread",
                "stability_score": "stability_score",
                "expected_return_base_pct": "base_return",
                "downside_max_drawdown_low_pct": "low_scenario_drawdown",
            }
        )
        # conversions
        if "base_return" in dfh.columns:
            br = pd.to_numeric(dfh["base_return"], errors="coerce")
            if br.notna().any() and br.median(skipna=True) > 0.5:
                dfh["base_return"] = br / 100.0
            else:
                dfh["base_return"] = br
        if "low_scenario_drawdown" in dfh.columns:
            dd = pd.to_numeric(dfh["low_scenario_drawdown"], errors="coerce")
            if dd.notna().any() and dd.abs().median(skipna=True) > 0.5:
                dfh["low_scenario_drawdown"] = dd / 100.0
            else:
                dfh["low_scenario_drawdown"] = dd
        for col in ["volatility", "scenario_spread", "stability_score"]:
            if col in dfh.columns:
                dfh[col] = pd.to_numeric(dfh[col], errors="coerce")
        keep = [c for c in _EXPECTED_METRIC_COLS if c in dfh.columns]
        if keep:
            dfh = dfh[keep]
        if "district" in dfh.columns:
            dfh["district"] = dfh["district"].astype(str).str.strip()
            dfh = dfh[dfh["district"].astype(bool)]
        return dfh.reset_index(drop=True)

    return pd.DataFrame()


def _render_investment_charts(df: pd.DataFrame) -> None:
    """Render 1–2 charts from investment metrics."""
    if df.empty:
        return

    st.markdown("### Investment metrics visualisations")

    c1, c2 = st.columns([1, 1])

    with c1:
        if {"base_return", "volatility", "district"}.issubset(df.columns):
            st.markdown("#### Risk vs Return (Base scenario)")
            fig = px.scatter(
                df,
                x="volatility",
                y="base_return",
                hover_name="district",
                text="district",
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Scatter chart needs: district, base_return, volatility.")

    with c2:
        if {"district", "stability_score"}.issubset(df.columns):
            st.markdown("#### Stability score ranking")
            df2 = df.sort_values("stability_score", ascending=False).copy()
            fig2 = px.bar(df2, x="district", y="stability_score")
            fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Ranking chart needs: district, stability_score.")

    with st.expander("Show investment metrics table", expanded=False):
        st.dataframe(df, use_container_width=True)


def _render_district_bullets(df: pd.DataFrame) -> None:
    """District highlights as emoji bullets (computed from metrics table)."""
    if df.empty or "district" not in df.columns:
        return

    st.markdown("### District highlights")
    df_show = df.sort_values("stability_score", ascending=False) if "stability_score" in df.columns else df.copy()

    for _, r in df_show.iterrows():
        d = str(r.get("district", "")).strip()
        emj = _emoji_for_district(d)

        parts: List[str] = []
        if "base_return" in df_show.columns:
            parts.append(f"base return {_fmt_pct(r.get('base_return'))}")
        if "volatility" in df_show.columns and pd.notna(r.get("volatility")):
            parts.append(f"volatility {float(r.get('volatility')):.4f}")  # type: ignore
        if "low_scenario_drawdown" in df_show.columns:
            parts.append(f"low-scenario drawdown {_fmt_pct(r.get('low_scenario_drawdown'))}")
        if "scenario_spread" in df_show.columns:
            parts.append(f"scenario spread {_fmt_money(r.get('scenario_spread'))}")
        if "stability_score" in df_show.columns and pd.notna(r.get("stability_score")):
            parts.append(f"stability score {float(r.get('stability_score')):.2f}")  # type: ignore

        st.markdown(f"- {emj} **{d}**: " + ", ".join(parts))


# ============================================================
# Agent payload normalization
# ============================================================
def _extract_agent_payload(agent_resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize /api/agent output into a consistent structure.

    Expected /api/agent response:
      { ok: bool, mode: str, result: dict }

    Supported modes:
      - help
      - forecast_agent
      - investment
    """
    mode = agent_resp.get("mode", "") or "unknown"
    result = agent_resp.get("result", {}) if isinstance(agent_resp.get("result"), dict) else {}

    if mode == "help":
        return {
            "mode": "help",
            "message": result.get("message", "Please provide more information."),
            "raw": agent_resp,
        }

    if mode == "investment":
        text = (result.get("text") or "").strip()
        tone = (result.get("tone") or "info").strip()
        profiles = result.get("profiles", {}) if isinstance(result.get("profiles"), dict) else {}
        meta = result.get("meta", {}) if isinstance(result.get("meta"), dict) else {}
        source = result.get("source")  # ✅ source is in result layer

        return {
            "mode": "investment",
            "text": text,
            "tone": tone,
            "profiles": profiles,
            "meta": meta,
            "source": source,
            "raw": agent_resp,
        }

    if mode == "forecast_agent":
        narrative = ""
        prediction: Dict[str, Any] = {}
        shap: Dict[str, Any] = {}
        drivers: Dict[str, Any] = {"up": [], "down": []}

        narrative = result.get("narrative") or ""
        if isinstance(result.get("prediction"), dict):
            prediction = result["prediction"]
        if isinstance(result.get("shap"), dict):
            shap = result["shap"]

        if not narrative and isinstance(result.get("result"), dict):
            narrative = result["result"].get("narrative") or narrative
            if not prediction and isinstance(result["result"].get("prediction"), dict):
                prediction = result["result"]["prediction"]
            if not shap and isinstance(result["result"].get("shap"), dict):
                shap = result["result"]["shap"]

        if isinstance(shap.get("drivers"), dict):
            drivers = shap["drivers"]

        raw = result.get("raw") if isinstance(result.get("raw"), dict) else None

        return {
            "mode": "forecast_agent",
            "narrative": (narrative or "").strip(),
            "prediction": prediction,
            "drivers": drivers,
            "raw": raw or agent_resp,
        }

    return {"mode": mode, "raw": agent_resp}


def render_agent_response(agent_resp: Dict[str, Any], show_debug_panels: bool) -> None:
    """Render agent response cleanly (no raw JSON unless debug is enabled)."""
    norm = _extract_agent_payload(agent_resp)
    mode = norm.get("mode", "unknown")

    if mode == "help":
        st.info(norm.get("message", "Please provide more info."))
        if show_debug_panels:
            with st.expander("Debug (help payload)", expanded=False):
                st.json(norm.get("raw", agent_resp))
        return

    if mode == "investment":
        text = norm.get("text", "")
        tone = (norm.get("tone") or "info").lower()
        profiles = norm.get("profiles", {}) if isinstance(norm.get("profiles"), dict) else {}
        meta = norm.get("meta", {}) if isinstance(norm.get("meta"), dict) else {}
        source = norm.get("source")

        header = "📌 Investment insight (district stability vs upside)"
        if tone == "success":
            st.success(f"**{header}**\n\n{text if text else 'Investment insight returned an empty narrative.'}")
        elif tone == "warning":
            st.warning(f"**{header}**\n\n{text if text else 'Investment insight returned an empty narrative.'}")
        else:
            st.info(f"**{header}**\n\n{text if text else 'Investment insight returned an empty narrative.'}")

        # ✅ Extract metrics (source -> default -> meta -> text)
        df_metrics = _extract_investment_metrics(source=source, meta=meta, text=text)

        if not df_metrics.empty:
            _render_district_bullets(df_metrics)
            _render_investment_charts(df_metrics)
        else:
            st.caption("Could not extract investment metrics (CSV/meta/text all unavailable). Charts are skipped.")

        # Investor profiles default collapsed
        if profiles:
            with st.expander("Investor-style interpretations", expanded=False):
                for k, title in [("conservative", "Conservative"), ("balanced", "Balanced"), ("growth", "Growth")]:
                    if k in profiles and isinstance(profiles[k], str) and profiles[k].strip():
                        st.markdown(f"**{title}**")
                        st.markdown(profiles[k])

        if show_debug_panels:
            with st.expander("Debug (investment payload)", expanded=False):
                st.json(norm.get("raw", agent_resp))
                st.write("result.source:", source)
                st.write("default.source:", DEFAULT_INVESTMENT_SOURCE)

        return

    if mode == "forecast_agent":
        narrative = (norm.get("narrative") or "").strip()
        prediction = norm.get("prediction") if isinstance(norm.get("prediction"), dict) else {}
        drivers = norm.get("drivers") if isinstance(norm.get("drivers"), dict) else {"up": [], "down": []}

        if narrative:
            st.success(narrative)
        else:
            st.caption("No narrative returned (empty).")

        c1, c2, c3 = st.columns(3)
        pred = prediction if isinstance(prediction, dict) else {}
        with c1:
            st.metric("Current price", _fmt_money(pred.get("current_price")))
        with c2:
            st.metric("Future price", _fmt_money(pred.get("future_price")))
        with c3:
            st.metric("Pct change", _fmt_pct(pred.get("pct_change")))

        st.markdown("#### Key drivers")
        up_list = drivers.get("up", []) if isinstance(drivers, dict) and isinstance(drivers.get("up"), list) else []
        down_list = drivers.get("down", []) if isinstance(drivers, dict) and isinstance(drivers.get("down"), list) else []

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Upward**")
            if up_list:
                for x in up_list[:8]:
                    if isinstance(x, dict):
                        st.write(f"- ✅ {x.get('feature', '—')}")
                    else:
                        st.write(f"- ✅ {x}")
            else:
                st.write("—")

        with d2:
            st.markdown("**Downward**")
            if down_list:
                for x in down_list[:8]:
                    if isinstance(x, dict):
                        st.write(f"- ⚠️ {x.get('feature', '—')}")
                    else:
                        st.write(f"- ⚠️ {x}")
            else:
                st.write("—")

        if show_debug_panels:
            with st.expander("Debug (full agent payload)", expanded=False):
                st.json(norm.get("raw", agent_resp))
        return

    st.warning(f"Unknown agent mode: {mode}")
    if show_debug_panels:
        st.json(norm.get("raw", agent_resp))


# ============================================================
# API call helpers
# ============================================================
def post_agent(api_base: str, endpoint_path: str, user_query: str, context: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}{endpoint_path}"
    payload = {"user_query": user_query, "context": context}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ============================================================
# Context defaults (auto-inject from app state)
# ============================================================
def build_default_context() -> Dict[str, Any]:
    ctx = {
        "district": st.session_state.get("district", "Auckland City"),
        "scenario": st.session_state.get("pred_scenario", "base"),
        "month": st.session_state.get("explain_month", "2026-06"),
        "top_k": 8,
        "tone": st.session_state.get("inv_tone_select", "cautious"),
        "include_profiles": bool(st.session_state.get("inv_profiles_toggle", True)),
    }

    inv_top_k_val = st.session_state.get("inv_top_k_slider", 3)
    try:
        ctx["top_k"] = int(inv_top_k_val)
    except Exception:
        pass

    return ctx


# ============================================================
# Chat history display
# ============================================================
for msg in st.session_state["agent_messages"]:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    raw = msg.get("raw")

    with st.chat_message(role):
        if role == "assistant" and isinstance(raw, dict):
            render_agent_response(raw, show_debug)
        else:
            st.write(content)


# ============================================================
# Chat input
# ============================================================
user_text = st.chat_input("Ask the agent (e.g., 'Which district should I invest?')")

if user_text:
    st.session_state["agent_messages"].append({"role": "user", "content": user_text})

    ctx = build_default_context()

    try:
        resp_json = post_agent(
            api_base=st.session_state["api_base"],
            endpoint_path=endpoint,
            user_query=user_text,
            context=ctx,
            timeout=int(timeout_s),
        )
        st.session_state["agent_messages"].append({"role": "assistant", "content": "", "raw": resp_json})

    except requests.HTTPError as e:
        err_text = ""
        try:
            err_text = e.response.text if e.response is not None else str(e)
        except Exception:
            err_text = str(e)

        st.session_state["agent_messages"].append(
            {
                "role": "assistant",
                "content": (
                    f"POST failed: {st.session_state['api_base'].rstrip('/')}{endpoint}\n\n"
                    f"{type(e).__name__}: {e}\n\n{err_text}"
                ),
            }
        )
    except Exception as e:
        st.session_state["agent_messages"].append(
            {
                "role": "assistant",
                "content": f"Error: could not reach agent endpoint.\n\n{type(e).__name__}: {e}",
            }
        )

    st.rerun()
