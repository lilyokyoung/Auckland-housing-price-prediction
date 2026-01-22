# pages/2_Data_&_Features.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------
# Minimal bootstrap
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.style import apply_light_theme  # noqa: E402
from src.ui.api_client import get_json  # noqa: E402


# ============================================================
# Config (your known RAW coverage)
# ============================================================
RAW_START = "2018-08"
RAW_END = "2025-06"
RAW_MISSING = ["2018-11", "2020-07"]

DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "avms" / "final_for_AVMs.csv"
MONTH_COL = "Month"
DISTRICT_COL = "District"


# ============================================================
# Page config + theme
# ============================================================
st.set_page_config(
    page_title="Data & Features | Auckland Housing Price AVM",
    layout="wide",
)
apply_light_theme()

# -------------------------------------------------
# Light CSS (consistent with your Overview page)
# -------------------------------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Card container */
.avm-card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 16px 16px 14px 16px;
    background: rgba(255,255,255,0.55);
}
.avm-muted { color: rgba(107,114,128,1); }
.avm-hr { margin: 1.2rem 0 1.0rem 0; border: none; height: 1px; background: rgba(0,0,0,0.08); }

/* tiny badge */
.avm-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(0,0,0,0.02);
    color: rgba(55, 65, 81, 0.95);
    margin-right: 6px;
    margin-bottom: 6px;
}

/* make dataframes look lighter */
div[data-testid="stDataFrame"] { border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="avm-card">
  <div>
    <span class="avm-badge">🧾 Data</span>
    <span class="avm-badge">🧱 Features</span>
    <span class="avm-badge">🧪 Quality checks</span>
  </div>
  <h2 style="margin: 0.35rem 0 0.2rem 0;">Data & Features</h2>
  <p class="avm-muted" style="font-size: 1.02rem; margin: 0.2rem 0 0.2rem 0;">
    Auto-generated dataset coverage, feature structure, and lightweight quality checks.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="avm-hr"/>', unsafe_allow_html=True)

api_base = st.session_state.get("api_base", "http://127.0.0.1:8000")


# ============================================================
# Helpers
# ============================================================
def _load_local_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if MONTH_COL in df.columns:
        df[MONTH_COL] = pd.to_datetime(df[MONTH_COL], errors="coerce")
        df[MONTH_COL] = df[MONTH_COL].dt.to_period("M").dt.to_timestamp()  # type: ignore
    return df


def _infer_frequency_from_month(df: pd.DataFrame) -> str:
    if MONTH_COL not in df.columns:
        return "Unknown"
    months = df[MONTH_COL].dropna().drop_duplicates().sort_values()
    if len(months) < 3:
        return "Unknown"
    diffs = months.diff().dropna()
    med_days = diffs.dt.days.median()  # type: ignore
    if med_days is not None and 25 <= med_days <= 35:
        return "Monthly"
    return "Irregular"


def _span_and_missing(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str], int]:
    if MONTH_COL not in df.columns:
        return None, None, [], 0
    tmp = df.dropna(subset=[MONTH_COL]).copy()
    if tmp.empty:
        return None, None, [], 0
    start = tmp[MONTH_COL].min()
    end = tmp[MONTH_COL].max()
    full = pd.date_range(start=start, end=end, freq="MS")
    have = tmp[MONTH_COL].drop_duplicates().sort_values()
    missing = sorted(set(full) - set(have))
    return start.strftime("%Y-%m"), end.strftime("%Y-%m"), [d.strftime("%Y-%m") for d in missing], int(have.shape[0])


def _infer_max_lag(cols: List[str]) -> int:
    lags: List[int] = []
    for c in cols:
        m = re.search(r"lag[_\-]?(\d+)", c.lower())
        if m:
            try:
                lags.append(int(m.group(1)))
            except Exception:
                pass
    return max(lags) if lags else 0


def _missingness_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    out = miss.reset_index()
    out.columns = ["column", "missing_rate"]
    out["missing_rate_pct"] = (out["missing_rate"] * 100).round(2)
    out = out.drop(columns=["missing_rate"]).head(top_n)
    return out


def _feature_groups(cols: List[str]) -> Dict[str, List[str]]:
    """
    Group columns into a readable schema summary.
    Adjust rules if you rename prefixes later.
    """
    groups: Dict[str, List[str]] = {
        "Target / ID": [],
        "Categorical (one-hot / dummies)": [],
        "Numeric (base features)": [],
        "Lagged features": [],
        "Other": [],
    }

    for c in cols:
        cl = c.lower()

        if c in {MONTH_COL, DISTRICT_COL, "Median_Price", "log_Median_Price"}:
            groups["Target / ID"].append(c)
            continue

        if "cat__" in cl or cl.startswith("cat_") or "district_" in cl:
            groups["Categorical (one-hot / dummies)"].append(c)
            continue

        if "lag" in cl:
            groups["Lagged features"].append(c)
            continue

        if cl.startswith("num__") or cl.startswith("num_") or cl.startswith("log_") or cl.startswith("real"):
            groups["Numeric (base features)"].append(c)
            continue

        groups["Other"].append(c)

    # sort inside groups
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k])
        if len(groups[k]) == 0:
            groups.pop(k, None)

    return groups


def _try_load_dataset_summary(api_base: str) -> Optional[Dict[str, Any]]:
    for path in ("/dataset/summary", "/api/dataset/summary"):
        try:
            return get_json(
                api_base,
                path,
                params={
                    "raw_start": RAW_START,
                    "raw_end": RAW_END,
                    "raw_missing": ",".join(RAW_MISSING),
                    "path": str(DATA_PATH),
                },
                timeout=60,
            )
        except Exception:
            continue
    return None


def _district_month_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """District x month coverage table (for quick sanity checks)."""
    if MONTH_COL not in df.columns or DISTRICT_COL not in df.columns:
        return pd.DataFrame()

    tmp = df.dropna(subset=[MONTH_COL, DISTRICT_COL]).copy()
    tmp[MONTH_COL] = pd.to_datetime(tmp[MONTH_COL], errors="coerce")
    tmp[MONTH_COL] = tmp[MONTH_COL].dt.to_period("M").dt.to_timestamp()  # type: ignore
    tmp = tmp.dropna(subset=[MONTH_COL])

    g = (
        tmp.groupby(DISTRICT_COL)[MONTH_COL]
        .nunique()
        .reset_index(name="unique_months")
        .sort_values("unique_months", ascending=False)
    )

    # also show start/end per district
    span = tmp.groupby(DISTRICT_COL)[MONTH_COL].agg(["min", "max"]).reset_index()
    span["min"] = span["min"].dt.strftime("%Y-%m") # type: ignore
    span["max"] = span["max"].dt.strftime("%Y-%m") # type: ignore

    out = g.merge(span, on=DISTRICT_COL, how="left")
    out.rename(columns={"min": "start", "max": "end"}, inplace=True)
    return out


# ============================================================
# Prefer backend summary; fallback to local
# ============================================================
df_local: Optional[pd.DataFrame] = None
summary = _try_load_dataset_summary(api_base)

use_api = bool(summary and summary.get("ok") is True)

if use_api:
    ds = summary.get("dataset", {}) # type: ignore
    sources = summary.get("sources", ["REINZ", "Stats NZ", "RBNZ"]) # type: ignore
    expl = summary.get("explanation", "") # type: ignore
    miss_top = summary.get("missingness_top", []) # type: ignore
    eff_start = ds.get("effective_span", {}).get("start")
    eff_end = ds.get("effective_span", {}).get("end")
    eff_missing = ds.get("effective_missing_months", [])
    freq = ds.get("frequency", "Monthly")
    n_rows = int(ds.get("rows", 0) or 0)
    n_cols = int(ds.get("cols", 0) or 0)
    n_dist = int(ds.get("districts", 0) or 0)
    n_months = int(ds.get("unique_months", 0) or 0)
    max_lag = int(ds.get("max_lag_inferred", 0) or 0)
else:
    # local fallback
    if not DATA_PATH.exists():
        st.error(f"Local dataset not found: `{DATA_PATH}`")
        st.stop()

    df_local = _load_local_df(DATA_PATH)
    eff_start, eff_end, eff_missing, n_months = _span_and_missing(df_local)
    n_rows, n_cols = df_local.shape
    n_dist = int(df_local[DISTRICT_COL].nunique()) if DISTRICT_COL in df_local.columns else 0
    freq = _infer_frequency_from_month(df_local)
    max_lag = _infer_max_lag(df_local.columns.tolist())
    sources = ["REINZ", "Stats NZ", "RBNZ"]
    expl = ""
    miss_top = []


# ============================================================
# Section 1: Overview (KPIs + explanation)
# ============================================================
st.markdown("### 1) Dataset overview")

top1, top2, top3, top4 = st.columns(4)
with top1:
    st.metric("Primary sources", " / ".join(sources))
with top2:
    st.metric("Effective time span", f"{eff_start} → {eff_end}" if eff_start and eff_end else "—")
with top3:
    st.metric("Spatial unit", f"{n_dist} districts" if n_dist else "—")
with top4:
    st.metric("Frequency", freq)

bot1, bot2, bot3, bot4 = st.columns(4)
with bot1:
    st.metric("Rows", f"{n_rows:,}")
with bot2:
    st.metric("Columns", f"{n_cols:,}")
with bot3:
    st.metric("Unique months", f"{n_months:,}" if n_months else "—")
with bot4:
    st.metric("Max lag (inferred)", str(max_lag) if max_lag else "0")

st.markdown("")

with st.expander("Why does the effective sample have more missing months than the raw data?", expanded=True):
    st.markdown(
        f"""
- **Raw data coverage (before feature engineering):** **{RAW_START} → {RAW_END}**  
- **Original missing months in raw data:** **{", ".join(RAW_MISSING)}**  
- The modeling table includes **lagged features (up to lag-{max_lag})**. Building lags requires historical observations,
  which creates **lag-induced NAs** at the beginning of the sample (and around original gaps).  
- Depending on your pipeline, you may also exclude/hold out later months (e.g., test/forecast horizon).  
"""
    )
    if expl:
        st.caption(expl)

# Missing months: show as a small table (better than a long sentence)
if eff_missing:
    miss_df = pd.DataFrame({"missing_months": eff_missing})
    st.warning("Missing months detected in the **effective modeling table**:")
    st.dataframe(miss_df, use_container_width=True, hide_index=True, height=180)
else:
    st.success("No missing months detected in the effective modeling table.")

st.caption(f"Dataset path (UI): `{DATA_PATH}`")

st.markdown('<hr class="avm-hr"/>', unsafe_allow_html=True)

# ============================================================
# Section 2: Feature schema (lightweight)
# ============================================================
st.markdown("### 2) Feature schema (lightweight)")
st.caption("This section helps you *see* what the model ingests, without diving into full EDA.")

# Load a local df for schema grouping if API was used (we still need columns to group)
if use_api:
    # try load local for schema view; if unavailable, just show a generic note
    if DATA_PATH.exists():
        df_schema = _load_local_df(DATA_PATH)
        cols_for_schema = df_schema.columns.tolist()
    else:
        df_schema = None
        cols_for_schema = []
else:
    df_schema = df_local
    cols_for_schema = df_local.columns.tolist() # type: ignore

if cols_for_schema:
    groups = _feature_groups(cols_for_schema)

    g1, g2 = st.columns([1.0, 1.0])
    with g1:
        st.markdown("**Grouped feature families**")
        group_rows = [{"group": k, "count": len(v)} for k, v in groups.items()]
        st.dataframe(pd.DataFrame(group_rows), use_container_width=True, hide_index=True)

    with g2:
        st.markdown("**Max lag inferred**")
        st.info(f"Detected up to **lag-{max_lag}** columns based on naming (e.g., `*_lag12`).")

    st.markdown("")
    with st.expander("Show full feature lists (by group)", expanded=False):
        for k, v in groups.items():
            st.markdown(f"**{k} ({len(v)})**")
            st.code("\n".join(v), language="text")
else:
    st.info("Feature schema view unavailable because the dataset file was not found locally.")

st.markdown('<hr class="avm-hr"/>', unsafe_allow_html=True)

# ============================================================
# Section 3: Data quality checks (lightweight)
# ============================================================
st.markdown("### 3) Data quality checks (lightweight)")

# Missingness table
st.markdown("**Top missing-rate columns**")
if use_api and isinstance(miss_top, list) and miss_top:
    miss_df = pd.DataFrame(miss_top)
    # normalize if backend uses different naming
    if "missing_rate_pct" not in miss_df.columns and "missing_rate" in miss_df.columns:
        miss_df["missing_rate_pct"] = (pd.to_numeric(miss_df["missing_rate"], errors="coerce") * 100).round(2)
    st.dataframe(miss_df, use_container_width=True, hide_index=True, height=320)
else:
    if df_schema is not None:
        st.dataframe(_missingness_table(df_schema, top_n=15), use_container_width=True, hide_index=True, height=320)
    else:
        st.caption("Missingness summary not available.")

st.markdown("")

# Optional: district coverage (toggle)
with st.expander("District coverage check (optional)", expanded=False):
    if df_schema is None:
        st.info("District coverage check requires local dataset access.")
    else:
        cov = _district_month_coverage(df_schema)
        if cov.empty:
            st.info("District coverage check not available (missing Month/District columns).")
        else:
            cA, cB = st.columns([1.0, 1.0])
            with cA:
                st.dataframe(cov, use_container_width=True, hide_index=True, height=260)
            with cB:
                fig = px.bar(cov, x=DISTRICT_COL, y="unique_months", title="Unique months by district")
                fig.update_layout(height=260, xaxis_title="District", yaxis_title="Unique months")
                st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="avm-hr"/>', unsafe_allow_html=True)

# ============================================================
# Debug (collapsed)
# ============================================================
with st.expander("Debug (paths & mode)", expanded=False):
    st.write(
        {
            "mode": "api_summary" if use_api else "local_fallback",
            "api_base": api_base,
            "data_path": str(DATA_PATH),
            "raw_coverage": {"start": RAW_START, "end": RAW_END, "missing": RAW_MISSING},
            "effective_span": {"start": eff_start, "end": eff_end},
            "effective_missing_months": eff_missing,
            "shape": {"rows": n_rows, "cols": n_cols},
            "max_lag_inferred": max_lag,
        }
    )
