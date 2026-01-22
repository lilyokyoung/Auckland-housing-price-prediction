# pages/3_Modelling_&_Evaluation.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Bootstrap: make project root importable so "import src.*" works
# pages/ is at PROJECT_ROOT/pages, so parents[1] is PROJECT_ROOT
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.style import apply_light_theme  # noqa: E402

# ============================================================
# Local fallback paths (used only if API is unavailable)
# ============================================================
MODEL_TABLE_CSV = PROJECT_ROOT / "data" / "preprocessed" / "avms" / "final_for_AVMs.csv"

TRAIN_METRICS_CSV = (
    PROJECT_ROOT / "outputs" / "models" / "compare" / "train_metrics_mlr_vs_avms.csv"
)
TEST_METRICS_CSV = (
    PROJECT_ROOT / "outputs" / "models" / "compare" / "test_metrics_mlr_vs_avms.csv"
)

# Optional robustness file
EVAL_BY_DISTRICT_CSV = PROJECT_ROOT / "outputs" / "evaluation" / "by_district.csv"

RAW_START = "2018-08"
RAW_END = "2025-06"
RAW_MISSING = ["2018-11", "2020-07"]


# ============================================================
# Style helpers
# ============================================================
def _apply_page_css() -> None:
    st.markdown(
        """
<style>
.block-container { padding-top: 2.0rem; padding-bottom: 2.0rem; }

/* Soft cards */
.soft-card {
  border: 1px solid rgba(120,120,120,0.18);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.55);
}

/* Subtle muted captions */
.small-muted { color: rgba(60,60,60,0.75); font-size: 0.92rem; }

/* Tighten expander spacing */
div[data-testid="stExpander"] details { border-radius: 12px; }
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# Data helpers
# ============================================================
def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


def _month_str(x: Any) -> str:
    if x is None:
        return "—"
    s = str(x).strip()
    if not s:
        return "—"
    # If it looks like 2024-01-01, show 2024-01
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:7]
    return s


def _safe_get_dataset_summary(api_base: str) -> Optional[Dict[str, Any]]:
    """
    Prefer backend dataset summary to keep pages consistent.
    Tries both /dataset/summary and /api/dataset/summary.
    Uses plain requests (avoids api_client path issues).
    """
    for path in ("/dataset/summary", "/api/dataset/summary"):
        url = api_base.rstrip("/") + path
        try:
            r = requests.get(
                url,
                params={
                    "raw_start": RAW_START,
                    "raw_end": RAW_END,
                    "raw_missing": ",".join(RAW_MISSING),
                    "path": str(MODEL_TABLE_CSV),
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, dict) else None
        except Exception:
            continue
    return None


def _infer_design_text(ds_summary: Optional[Dict[str, Any]]) -> str:
    if not ds_summary or not ds_summary.get("ok"):
        return (
            f"Raw coverage: {RAW_START} → {RAW_END} (missing: {', '.join(RAW_MISSING)}). "
            "Effective modelling coverage depends on feature engineering (e.g., lagged variables) "
            "and the time-based hold-out design."
        )

    d = ds_summary.get("dataset", {}) or {}
    eff = d.get("effective_span", {}) or {}
    eff_start = _month_str(eff.get("start"))
    eff_end = _month_str(eff.get("end"))
    max_lag = d.get("max_lag_inferred", None)

    max_lag_txt = "lagged features" if max_lag is None else f"lagged features up to lag-{int(max_lag)}"

    return (
        f"Raw coverage: {RAW_START} → {RAW_END} (missing: {', '.join(RAW_MISSING)}). "
        f"Effective modelling table spans {eff_start} → {eff_end} and includes {max_lag_txt}. "
        "All evaluation uses **time-based splitting** to avoid look-ahead bias."
    )


def _ensure_required_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")


def _build_train_test_table(df_train: pd.DataFrame, df_test: pd.DataFrame, unit: str) -> pd.DataFrame:
    """
    Merge train vs test metrics by model, compute gaps.
    Supports both NZD and log.
    """
    if unit == "NZD":
        rmse, mae, r2 = "RMSE_NZD", "MAE_NZD", "R2_NZD"
    else:
        rmse, mae, r2 = "RMSE_log", "MAE_log", "R2_log"

    _ensure_required_cols(df_train, ["model", rmse, mae, r2], "Train metrics")
    _ensure_required_cols(df_test, ["model", rmse, mae, r2], "Test metrics")

    t_train = df_train[["model", rmse, mae, r2]].copy().rename(
        columns={rmse: f"{rmse}_train", mae: f"{mae}_train", r2: f"{r2}_train"}
    )
    t_test = df_test[["model", rmse, mae, r2]].copy().rename(
        columns={rmse: f"{rmse}_test", mae: f"{mae}_test", r2: f"{r2}_test"}
    )

    merged = pd.merge(t_test, t_train, on="model", how="left")

    # numeric
    for c in merged.columns:
        if c != "model":
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # gaps
    merged["RMSE_gap_pct"] = (merged[f"{rmse}_test"] / merged[f"{rmse}_train"] - 1.0) * 100.0
    merged["MAE_gap_pct"] = (merged[f"{mae}_test"] / merged[f"{mae}_train"] - 1.0) * 100.0
    merged["R2_delta"] = merged[f"{r2}_test"] - merged[f"{r2}_train"]

    merged = merged.sort_values(f"{rmse}_test", ascending=True).reset_index(drop=True)
    return merged


# ============================================================
# Formatting
# ============================================================
def _fmt_level(x: Any) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "—"


def _fmt_log(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "—"


def _fmt_r2(x: Any) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"


def _get_narr_text(n: Dict[str, Any], block: str) -> str:
    """
    block in {'modeling','evaluation','selection','generalisation'}.
    tolerate 'modelling' spelling.
    """
    if not isinstance(n, dict):
        return ""
    if block == "modeling":
        b = n.get("modeling") or n.get("modelling") or {}
        return str((b or {}).get("text") or "").strip()
    if block == "evaluation":
        b = n.get("evaluation") or {}
        return str((b or {}).get("text") or "").strip()
    if block == "selection":
        b = n.get("selection") or {}
        return str((b or {}).get("text") or "").strip()
    if block == "generalisation":
        b = n.get("generalisation") or {}
        return str((b or {}).get("text") or "").strip()
    return ""


def _get_generalisation_tone(n: Dict[str, Any]) -> str:
    if not isinstance(n, dict):
        return ""
    b = n.get("generalisation") or {}
    tone = str((b or {}).get("tone") or "").strip().lower()
    return tone


# ============================================================
# API fetch
# ============================================================
@st.cache_data(show_spinner=False)
def _fetch_narratives(api_base_in: str) -> Dict[str, Any]:
    """
    Fetch narratives from API.
    Always returns a dict. If failed, returns {"_ok": False, "_error": "..."}.
    """
    url = api_base_in.rstrip("/") + "/api/narratives"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data["_ok"] = True
            data["_url"] = url
            return data
        return {"_ok": False, "_url": url, "_error": f"Unexpected JSON type: {type(data)}"}
    except Exception as e:
        return {"_ok": False, "_url": url, "_error": f"{type(e).__name__}: {e}"}


# ============================================================
# Page
# ============================================================
st.set_page_config(page_title="Modelling & Evaluation | Auckland", layout="wide")
apply_light_theme()
_apply_page_css()

try:
    st.markdown("## Modelling & Evaluation")
    st.caption("Compare candidate models using time-based evaluation and select the final AVM.")
    st.divider()

    api_base = st.session_state.get("api_base", "http://127.0.0.1:8000")

    # Sidebar settings
    st.sidebar.markdown("### Settings")
    show_debug = st.sidebar.toggle(
        "Show debug panels",
        value=False,
        help="Show raw payloads and intermediate tables for debugging",
    )

    # ============================================================
    # Auto methodology narration
    # ============================================================
    st.markdown("### 🧠 Automated modelling methodology summary")
    st.caption(
    "AI-generated narrative that summarises the modelling pipeline and evaluation results "
    "to support reproducibility and reporting."
    )

    col_btn, col_src = st.columns([0.25, 0.75], vertical_alignment="center")
    with col_btn:
        refresh = st.button("Refresh narration", use_container_width=True)
    with col_src:
        if show_debug:
           st.caption(f"Source: `{api_base}/api/narratives`")

    if refresh:
        st.cache_data.clear()

    narratives = _fetch_narratives(api_base)

    if narratives.get("_ok") is True:
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Modelling narration (auto)**")
            txt = _get_narr_text(narratives, "modeling")
            st.write(txt if txt else "—")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Evaluation narration (auto)**")
            txt = _get_narr_text(narratives, "evaluation")
            st.write(txt if txt else "—")
            st.markdown("</div>", unsafe_allow_html=True)

        if show_debug:
            with st.expander("Debug: narratives payload", expanded=False):
                st.json(narratives)
    else:
        st.info("Narratives API not available (this is OK). The rest of the page uses local CSVs.")
        if show_debug:
            with st.expander("Debug: narratives fetch error", expanded=True):
                st.json(narratives)

    st.divider()

    # ============================================================
    # 1) Modelling overview
    # ============================================================
    st.markdown("### 1) Modelling overview")
    st.markdown(
        """
This project evaluates multiple model classes to benchmark predictive performance and robustness:

- **Linear baselines**: MLR / LASSO (interpretability + benchmark)
- **Kernel-based ML**: SVR (captures nonlinear patterns via kernels)
- **Tree-based ML**: Random Forest / XGBoost (nonlinear relationships)
- **Forecast pipeline**: Scenario-based inputs + explainability
"""
    )
    st.divider()

    # ============================================================
    # 2) Training & evaluation design
    # ============================================================
    st.markdown("### 2) Training and evaluation design")

    ds_summary = _safe_get_dataset_summary(api_base)
    st.info(_infer_design_text(ds_summary))

    with st.expander("Show effective sample details (dataset summary)", expanded=False):
        if ds_summary and ds_summary.get("ok"):
            st.json(ds_summary.get("dataset", {}))
        else:
            st.caption("Dataset summary unavailable from API. (You can still proceed.)")

    st.markdown(
        """
**Evaluation principle**
- Use **time-based splits** (not random) to avoid look-ahead bias.
- Report **out-of-sample** metrics (test/hold-out) as the primary selection criterion.

**Primary metrics**
- RMSE (primary)
- MAE (robustness)
- R² (supporting)
"""
    )
    st.divider()

    # ============================================================
    # 3) Model comparison
    # ============================================================
    st.markdown("### 3) Model comparison (train vs test)")

    df_train = _read_csv(TRAIN_METRICS_CSV)
    df_test = _read_csv(TEST_METRICS_CSV)

    if df_train is None or df_test is None:
        st.warning(
            "Metrics files not found. Please check paths:\n"
            f"- {TRAIN_METRICS_CSV}\n"
            f"- {TEST_METRICS_CSV}\n\n"
            "Tip: run your scripts that generate these compare outputs."
        )
    else:
        c_unit, c_hint = st.columns([0.35, 0.65], vertical_alignment="center")
        with c_unit:
            unit_choice = st.radio("Metric unit", ["NZD (level)", "log"], horizontal=True, index=0)
        with c_hint:
            st.caption("NZD is easier to interpret; log is useful for relative-error comparison across districts.")
        unit = "NZD" if unit_choice.startswith("NZD") else "log"

        merged = _build_train_test_table(df_train, df_test, unit=unit)
        best = merged.iloc[0].to_dict()

        # KPI row
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns([1.2, 1, 1, 1])
        with k1:
            st.metric("Best model (test)", str(best.get("model", "—")))
        if unit == "NZD":
            rmse_col, mae_col, r2_col = "RMSE_NZD", "MAE_NZD", "R2_NZD"
            with k2:
                st.metric("RMSE (NZD, test)", _fmt_level(best.get(f"{rmse_col}_test")))
            with k3:
                st.metric("MAE (NZD, test)", _fmt_level(best.get(f"{mae_col}_test")))
            with k4:
                st.metric("R² (test)", _fmt_r2(best.get(f"{r2_col}_test")))
        else:
            rmse_col, mae_col, r2_col = "RMSE_log", "MAE_log", "R2_log"
            with k2:
                st.metric("RMSE (log, test)", _fmt_log(best.get(f"{rmse_col}_test")))
            with k3:
                st.metric("MAE (log, test)", _fmt_log(best.get(f"{mae_col}_test")))
            with k4:
                st.metric("R² (test)", _fmt_r2(best.get(f"{r2_col}_test")))
        st.markdown("</div>", unsafe_allow_html=True)

        # Generalisation (AUTO from API if available; else fallback to neutral info)
        gen_txt = _get_narr_text(narratives, "generalisation") if narratives.get("_ok") else ""
        gen_tone = _get_generalisation_tone(narratives) if narratives.get("_ok") else ""

        if gen_txt:
            if gen_tone == "success":
                st.success(gen_txt)
            elif gen_tone == "info":
                st.info(gen_txt)
            else:
                st.warning(gen_txt)
        else:
            # Fallback: keep it neutral (avoid "model failed" tone)
            gap = float(best.get("RMSE_gap_pct") or 0.0)
            st.info(
                f"Generalisation assessment: a train→test RMSE difference of **{gap:.1f}%** is observed under time-based splitting; "
                "results are interpreted primarily using **out-of-sample** performance."
            )

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Summary view", "Full comparison table", "Robustness (optional)"])

        with tab1:
            show_cols = [
                "model",
                f"{rmse_col}_test",
                f"{mae_col}_test",
                f"{r2_col}_test",
                "RMSE_gap_pct",
                "MAE_gap_pct",
            ]
            view_top = merged[show_cols].copy().head(5)

            if unit == "NZD":
                for c in [f"{rmse_col}_test", f"{mae_col}_test"]:
                    view_top[c] = pd.to_numeric(view_top[c], errors="coerce").round(0)
            else:
                for c in [f"{rmse_col}_test", f"{mae_col}_test"]:
                    view_top[c] = pd.to_numeric(view_top[c], errors="coerce").round(4)

            view_top[f"{r2_col}_test"] = pd.to_numeric(view_top[f"{r2_col}_test"], errors="coerce").round(3)
            view_top["RMSE_gap_pct"] = pd.to_numeric(view_top["RMSE_gap_pct"], errors="coerce").round(1)
            view_top["MAE_gap_pct"] = pd.to_numeric(view_top["MAE_gap_pct"], errors="coerce").round(1)

            st.dataframe(view_top, use_container_width=True, hide_index=True)
            st.caption("Summary highlights the top-ranked models by test RMSE. Use the full table for diagnostics.")

        with tab2:
            show_cols = [
                "model",
                f"{rmse_col}_test",
                f"{mae_col}_test",
                f"{r2_col}_test",
                f"{rmse_col}_train",
                f"{mae_col}_train",
                f"{r2_col}_train",
                "RMSE_gap_pct",
                "MAE_gap_pct",
                "R2_delta",
            ]
            view = merged[show_cols].copy()

            for c in [f"{r2_col}_test", f"{r2_col}_train", "R2_delta"]:
                view[c] = pd.to_numeric(view[c], errors="coerce").round(3)

            if unit == "NZD":
                for c in [f"{rmse_col}_test", f"{mae_col}_test", f"{rmse_col}_train", f"{mae_col}_train"]:
                    view[c] = pd.to_numeric(view[c], errors="coerce").round(0)
            else:
                for c in [f"{rmse_col}_test", f"{mae_col}_test", f"{rmse_col}_train", f"{mae_col}_train"]:
                    view[c] = pd.to_numeric(view[c], errors="coerce").round(4)

            view["RMSE_gap_pct"] = pd.to_numeric(view["RMSE_gap_pct"], errors="coerce").round(1)
            view["MAE_gap_pct"] = pd.to_numeric(view["MAE_gap_pct"], errors="coerce").round(1)

            st.dataframe(view, use_container_width=True, hide_index=True)

        with tab3:
            df_by_dist = _read_csv(EVAL_BY_DISTRICT_CSV)
            if df_by_dist is None:
                st.caption("No robustness file found. (Optional) Provide `outputs/evaluation/by_district.csv`.")
                st.caption(f"Expected path: `{EVAL_BY_DISTRICT_CSV}`")
            else:
                st.markdown("**District-level performance snapshot (optional)**")
                st.dataframe(df_by_dist, use_container_width=True, hide_index=True)

        if show_debug:
            with st.expander("Debug: raw metrics CSVs", expanded=False):
                st.write("Train metrics path:", str(TRAIN_METRICS_CSV))
                st.write("Test metrics path:", str(TEST_METRICS_CSV))
                st.markdown("**Test metrics (raw)**")
                st.dataframe(df_test, use_container_width=True, hide_index=True)
                st.markdown("**Train metrics (raw)**")
                st.dataframe(df_train, use_container_width=True, hide_index=True)

    st.divider()

    # ============================================================
    # 4) Final model selection rationale
    # ============================================================
    st.markdown("### 4) Final model selection rationale")

    selection_text = _get_narr_text(narratives, "selection") if narratives.get("_ok") else ""
    if selection_text.strip():
        st.markdown(selection_text)
    else:
        st.markdown(
            """
The final AVM is selected based on **out-of-sample performance** and **robustness**:

- Primary criterion: best **test RMSE**
- Secondary: MAE stability across districts
- Practical: supports consistent explanation (e.g., SHAP drivers) and deployment constraints

Proceed to **Forecast & Explanation** to run scenario forecasts and view SHAP-based explanations.
"""
        )

    st.caption(f"API Base: `{api_base}`")

except Exception as e:
    st.error("This page crashed while rendering.")
    st.exception(e)
    st.stop()
