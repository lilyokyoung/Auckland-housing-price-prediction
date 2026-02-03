# pages/4_Forecast_&_Explanation.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Bootstrap: make project root importable so "import src.*" works
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.style import apply_light_theme  # noqa: E402

try:
    from src.config import DATA_DIR, OUTPUT_DIR  # type: ignore  # noqa: E402
except Exception:
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"


apply_light_theme()
st.markdown("## Forecast & Explanation")
st.caption("Run district scenario forecasts and compare predicted values against historical prices (auto).")
st.divider()


# ============================================================
# Constants / paths
# ============================================================
MODEL_DIR = OUTPUT_DIR / "models" / "rf_final_predictions"
PRED_LONG_PATH = MODEL_DIR / "pred_all_scenarios_long.csv"

PROCESSED_DIR = DATA_DIR / "processed" / "merged_dataset"
HIST_CANDIDATES = [
    PROCESSED_DIR / "merged_dataset2.csv",
    PROCESSED_DIR / "merged_dataset2.xlsx",
    PROCESSED_DIR / "merged_dataset1.csv",
    PROCESSED_DIR / "merged_dataset1.xlsx",
]

# Auckland Region (overall) historical price – comparison only
# ------------------------------------------------------------
AUCKLAND_REGION_PRICE_CANDIDATES = [
    DATA_DIR / "processed" /"housing_price" / "AucklandRegion_Price.csv",

]
SHAP_DIR = OUTPUT_DIR / "shap_forecast"
SHAP_LONG_PATH = SHAP_DIR / "forecast_shap_long.csv"  # Month, District, Scenario, feature, shap_value

DISTRICTS = [
    "Auckland City",
    "Franklin",
    "Manukau",
    "North Shore",
    "Papakura",
    "Rodney",
    "Waitakere",
]
SCENARIOS = ["base", "low", "high"]

# --- Auckland district centroids (token-based; robust against naming variants) ---
# NOTE: These are approximate district "centroid" points for point-based comparison (not polygons).
DISTRICT_CENTROIDS_TOKEN: Dict[str, tuple[float, float]] = {
    "AucklandCity": (-36.8485, 174.7633),
    "Franklin": (-37.2000, 174.9000),
    "Manukau": (-36.9900, 174.8800),
    "NorthShore": (-36.8000, 174.7500),
    "Papakura": (-37.0600, 174.9500),
    "Rodney": (-36.4200, 174.6500),
    "Waitakere": (-36.9200, 174.6500),
}

# Map view tuning so Rodney (north) + Franklin (south) are visible in one frame
MAP_CENTER = {"lat": -36.95, "lon": 174.80}
MAP_ZOOM = 8  # was 9 -> too zoomed-in, hides far districts


# ============================================================
# Helpers: robust IO
# ============================================================
def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def _resolve_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _load_auckland_region_price(
    candidates: list[Path],
    month_col: str = "Month",
    price_col: str = "Median_Price",
) -> pd.DataFrame:
    p = _resolve_existing(candidates)
    if p is None:
        return pd.DataFrame()

    df = _read_table(p)

    if month_col not in df.columns or price_col not in df.columns:
        return pd.DataFrame()

    # \
    out = df[[month_col, price_col]].copy()
    out[month_col] = _parse_month(out[month_col])   # ✅ 用你现成的月度标准化函数
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=[month_col, price_col]).sort_values(month_col)

    # 如果有 Area，再尝试过滤；只有过滤后非空才覆盖 out
    if "Area" in df.columns:
        area_s = df["Area"].astype(str).str.strip().str.lower()
        mask = area_s.str.replace(" ", "", regex=False).eq("aucklandregion")
        tmp = df.loc[mask, [month_col, price_col]].copy()
        tmp[month_col] = _parse_month(tmp[month_col])
        tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce")
        tmp = tmp.dropna(subset=[month_col, price_col]).sort_values(month_col)

        if not tmp.empty:
            out = tmp

    return out


# ============================================================
# Helpers: normalization
# ============================================================
def _clean_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _norm_scenario(x: Any) -> str:
    s = _clean_str(x).lower()
    return s if s in {"base", "low", "high"} else "base"


def _district_to_token(d: str) -> str:
    """Canonical UI district name -> token like AucklandCity / NorthShore."""
    return _clean_str(d).replace(" ", "")


def _norm_district_value(x: Any) -> str:
    """
    Normalize district values like:
      AucklandCity / NorthShore / "North Shore" / "north_shore"
    to canonical UI names in DISTRICTS.
    """
    raw = _clean_str(x)
    if not raw:
        return raw

    explicit_map = {
        "AucklandCity": "Auckland City",
        "NorthShore": "North Shore",
        "Papakura": "Papakura",
        "Waitakere": "Waitakere",
        "Manukau": "Manukau",
        "Rodney": "Rodney",
        "Franklin": "Franklin",
        # tolerate case variants
        "aucklandcity": "Auckland City",
        "northshore": "North Shore",
    }
    if raw in explicit_map:
        return explicit_map[raw]

    import re

    t2 = re.sub(r"[^a-z0-9]", "", raw.lower())

    for d in DISTRICTS:
        d2 = re.sub(r"[^a-z0-9]", "", d.lower())
        if d2 == t2:
            return d

    for d in DISTRICTS:
        if d.lower() in raw.lower():
            return d

    return raw


def _parse_month(series: pd.Series) -> pd.Series:
    """
    Parse dates and normalize them to *month start* timestamps.

    Accepts:
      - "2026-06", "2026-06-01", "2026/06"
      - NZ Excel dd/mm/yyyy such as "1/07/2025"
      - "YYYY-MM-DD" strings
    Returns: Timestamp at month start (YYYY-MM-01).
    """
    s = series.astype(str).str.strip().replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})

    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    mask_slash = s.str.contains("/", na=False)
    if mask_slash.any():
        dt.loc[mask_slash] = pd.to_datetime(
            s.loc[mask_slash],
            errors="coerce",
            dayfirst=True,
        )

    mask_dash = s.str.contains("-", na=False)
    if mask_dash.any():
        dt.loc[mask_dash] = pd.to_datetime(
            s.loc[mask_dash].str.replace("/", "-", regex=False),
            errors="coerce",
            dayfirst=False,
        )

    # Normalize to month start
    return dt.dt.to_period("M").dt.to_timestamp()  # type: ignore


def _month_key_from_dt(dt: pd.Series) -> pd.Series:
    """Timestamp -> 'YYYY-MM' (string)."""
    return dt.dt.to_period("M").astype(str) # type: ignore


def _find_pred_col(df: pd.DataFrame) -> str:
    if "pred_Median_Price" in df.columns:
        return "pred_Median_Price"

    candidates = ["pred_median_price", "Pred_Median_Price", "prediction", "y_pred", "pred"]
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        "Prediction column not found. Expected 'pred_Median_Price'. "
        f"Available columns: {list(df.columns)}"
    )


def _pretty_feature_name(f: str) -> str:
    """Strip num__/cat__ prefixes for readability."""
    s = _clean_str(f)
    for p in ("num__", "cat__"):
        if s.startswith(p):
            s = s[len(p) :]
            break
    return s


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.markdown("### Controls")
    district = st.selectbox("District", DISTRICTS, index=0)
    scenario = st.selectbox("Scenario", SCENARIOS, index=0)
    preview_rows = st.number_input("Preview rows", min_value=50, max_value=5000, value=500, step=50)


# ============================================================
# Load prediction long
# ============================================================
pred_path = None
for cand in [
    PRED_LONG_PATH,
    PRED_LONG_PATH.with_suffix(".xlsx"),
    PRED_LONG_PATH.with_suffix(".parquet"),
]:
    if cand.exists():
        pred_path = cand
        break

if pred_path is None:
    st.error(f"Prediction long table not found under: {MODEL_DIR}")
    st.stop()

try:
    pred_all = _read_table(pred_path)
except Exception as e:
    st.error(f"Failed to load prediction table: {pred_path}\n\n{type(e).__name__}: {e}")
    st.stop()


# ============================================================
# Load history
# ============================================================
hist_path = _resolve_existing(HIST_CANDIDATES)
if hist_path is None:
    st.error("Historical table not found. Tried:\n- " + "\n- ".join(str(p) for p in HIST_CANDIDATES))
    st.stop()

try:
    hist_all = _read_table(hist_path)
except Exception as e:
    st.error(f"Failed to load historical table: {hist_path}\n\n{type(e).__name__}: {e}")
    st.stop()


# ============================================================
# Load shap (optional)
# ============================================================
shap_path = None
for cand in [
    SHAP_LONG_PATH,
    SHAP_LONG_PATH.with_suffix(".xlsx"),
    SHAP_LONG_PATH.with_suffix(".parquet"),
]:
    if cand.exists():
        shap_path = cand
        break

shap_all: Optional[pd.DataFrame] = None
if shap_path is not None:
    try:
        shap_all = _read_table(shap_path)
    except Exception:
        shap_all = None


# ============================================================
# Clean / normalize columns
# ============================================================
# --- Predictions ---
for col in ["Month", "District", "Scenario"]:
    if col not in pred_all.columns:
        st.error(f"Prediction table missing required column '{col}'. Columns: {list(pred_all.columns)}")
        st.stop()

pred_all = pred_all.copy()
pred_all["District"] = pred_all["District"].apply(_norm_district_value)
pred_all["Scenario"] = pred_all["Scenario"].apply(_norm_scenario)
pred_all["Month_dt"] = _parse_month(pred_all["Month"])          # month start ts
pred_all["MonthKey"] = _month_key_from_dt(pred_all["Month_dt"]) # 'YYYY-MM'
pred_all["District_token"] = pred_all["District"].apply(_district_to_token)

pred_col = _find_pred_col(pred_all)
pred_all[pred_col] = pd.to_numeric(pred_all[pred_col], errors="coerce")

pred_sel = pred_all[(pred_all["District"] == district) & (pred_all["Scenario"] == scenario)].copy()
pred_sel = pred_sel.dropna(subset=["Month_dt"]).sort_values("Month_dt")

if pred_sel.empty:
    st.warning("No prediction rows found for this district+scenario.")
    st.stop()

# month options for metrics & SHAP slice
month_options_dt = pd.Series(pred_sel["Month_dt"].dropna().unique()).sort_values()
month_options = month_options_dt.dt.strftime("%Y-%m").tolist()  # type: ignore

sel_month_str = st.selectbox(
    "Forecast month (for metrics & SHAP)",
    options=month_options,
    index=len(month_options) - 1,
)
sel_month = pd.to_datetime(sel_month_str + "-01")  # month start
sel_month_key = sel_month_str                      # 'YYYY-MM'


# --- History ---
hist = hist_all.copy()

hist_month_col = None
for c in ["Month", "month", "Date", "date"]:
    if c in hist.columns:
        hist_month_col = c
        break
if hist_month_col is None:
    st.error(f"Historical table missing a month column. Columns: {list(hist.columns)}")
    st.stop()

hist_district_col = None
for c in ["District", "district"]:
    if c in hist.columns:
        hist_district_col = c
        break
if hist_district_col is None:
    st.error(f"Historical table missing 'District' column. Columns: {list(hist.columns)}")
    st.stop()

if "Median_Price" not in hist.columns:
    st.error(f"Historical table missing 'Median_Price' column. Columns: {list(hist.columns)}")
    st.stop()

hist[hist_district_col] = hist[hist_district_col].apply(_norm_district_value)
hist["Month_dt"] = _parse_month(hist[hist_month_col])           # month start
hist["MonthKey"] = _month_key_from_dt(hist["Month_dt"])         # 'YYYY-MM'
hist["Median_Price"] = pd.to_numeric(hist["Median_Price"], errors="coerce")
hist["District_token"] = hist[hist_district_col].apply(_district_to_token)

hist_sel = hist[hist[hist_district_col] == district].copy()
hist_sel = hist_sel.dropna(subset=["Month_dt"]).sort_values("Month_dt")

if hist_sel.empty:
    st.warning("No historical rows found for this district.")
    st.stop()


# ============================================================
# Compute current_price & future_price & pct_change (selected district)
# ============================================================
forecast_start = pred_sel["Month_dt"].min()

hist_before = hist_sel[hist_sel["Month_dt"] < forecast_start].copy()
if not hist_before.empty:
    current_anchor_dt = hist_before["Month_dt"].max()
    current_price = float(
        hist_before.loc[hist_before["Month_dt"] == current_anchor_dt, "Median_Price"].iloc[-1]  # type: ignore
    )
else:
    current_anchor_dt = hist_sel["Month_dt"].max()
    current_price = float(hist_sel.loc[hist_sel["Month_dt"] == current_anchor_dt, "Median_Price"].iloc[-1])  # type: ignore

future_row = pred_sel[pred_sel["MonthKey"] == sel_month_key]
future_price = (
    float(future_row[pred_col].iloc[-1])
    if (not future_row.empty and pd.notna(future_row[pred_col].iloc[-1]))
    else None
)

pct_change = None
if future_price is not None and current_price and current_price != 0:
    pct_change = (future_price / current_price - 1.0) * 100.0


# ============================================================
# Build comparison chart: historical + forecast (month normalized)
# ============================================================
hist_plot = hist_sel[["Month_dt", "Median_Price"]].rename(
    columns={"Month_dt": "Month", "Median_Price": "Value"}
)
hist_plot["Series"] = "Historical"

pred_plot = pred_sel[["Month_dt", pred_col]].rename(columns={"Month_dt": "Month", pred_col: "Value"})
pred_plot["Series"] = "Forecast"

plot_df = pd.concat([hist_plot, pred_plot], ignore_index=True)
plot_df = plot_df.dropna(subset=["Month", "Value"]).sort_values("Month")

st.caption(f"Selected forecast month: {sel_month_str}")

fig = px.line(
    plot_df,
    x="Month",
    y="Value",
    color="Series",
    title=f"{district} — Historical vs Forecast ({scenario})",
)
fig.update_layout(
    height=420,
    legend_title_text="",
    xaxis_title="Month",
    yaxis_title="Median Price",
)

# ============================================================
# ✅ Mark the selected month on the line chart
# ============================================================


# 2) marker at selected month (prefer Forecast; fallback to Historical)
def _get_value_at_month(series_name: str) -> Optional[float]:
    s = plot_df[(plot_df["Month"] == sel_month) & (plot_df["Series"] == series_name)]
    if s.empty:
        return None
    v = s["Value"].iloc[-1]
    return float(v) if pd.notna(v) else None

y_forecast = _get_value_at_month("Forecast")
y_hist = _get_value_at_month("Historical")
y_mark = y_forecast if y_forecast is not None else y_hist

if y_mark is not None:
    fig.add_trace(
        go.Scatter(
            x=[sel_month],
            y=[y_mark],
            mode="markers",
            text=None,
            textposition="top center",
            marker=dict(
                size=6,                 
                color="red",
                symbol="circle",
            ),
            textfont=dict(size=11),
            showlegend=False,
            hovertemplate="Selected month: %{x}<br>Price: %{y:,.0f}<extra></extra>",
            name="Selected month",
        )
    )

df_region = _load_auckland_region_price(AUCKLAND_REGION_PRICE_CANDIDATES)

if not df_region.empty:
    # 
    hist_end = pd.to_datetime(current_anchor_dt)  # 
    df_region_plot = df_region[df_region["Month"] <= hist_end].copy()
    
    df_region = _load_auckland_region_price(AUCKLAND_REGION_PRICE_CANDIDATES)
    

    if not df_region.empty:
        hist_end = current_anchor_dt  
        df_region_plot = df_region[df_region["Month"] <= hist_end].copy()
        


    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter(
            x=df_region_plot["Month"],
            y=df_region_plot["Median_Price"],
            mode="lines",
            name="Auckland Region (Historical)",
            line=dict(dash="dot"),   
            hovertemplate="Month: %{x|%Y-%m}<br>Auckland Region: %{y:,.0f}<extra></extra>",
        )
    )

st.plotly_chart(fig, use_container_width=True)


st.caption(
    f"Current anchor (history): {current_anchor_dt.strftime('%Y-%m')}  |  "
    f"Forecast starts: {forecast_start.strftime('%Y-%m')}"
)


# ============================================================
# Metrics
# ============================================================
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Current price", f"{current_price:,.0f}" if current_price is not None else "—")
with c2:
    st.metric("Future price", f"{future_price:,.0f}" if future_price is not None else "—")
with c3:
    st.metric("Pct change", f"{pct_change:.2f}%" if pct_change is not None else "—")


# ============================================================
# SHAP drivers (month normalized)
# ============================================================
st.markdown("### Forecast SHAP drivers (+ upward / − downward)")

top_k = 8
up_list: List[str] = []
down_list: List[str] = []

if shap_all is None:
    st.info("SHAP file not found or failed to load. (Expected under outputs/shap_forecast/forecast_shap_long.*)")
else:
    required = {"Month", "District", "Scenario", "feature", "shap_value"}
    if not required.issubset(set(shap_all.columns)):
        st.info(
            "SHAP file loaded but columns don't match expected schema.\n\n"
            f"Expected: {sorted(list(required))}\n"
            f"Found: {list(shap_all.columns)}"
        )
    else:
        shap_df = shap_all.copy()
        shap_df["District"] = shap_df["District"].apply(_norm_district_value)
        shap_df["Scenario"] = shap_df["Scenario"].apply(_norm_scenario)
        shap_df["Month_dt"] = _parse_month(shap_df["Month"])
        shap_df["MonthKey"] = _month_key_from_dt(shap_df["Month_dt"])
        shap_df["shap_value"] = pd.to_numeric(shap_df["shap_value"], errors="coerce")

        slice_df = shap_df[
            (shap_df["District"] == district)
            & (shap_df["Scenario"] == scenario)
            & (shap_df["MonthKey"] == sel_month_key)
        ].dropna(subset=["shap_value", "feature"])

        if slice_df.empty:
            st.info("No SHAP slice found for this district + scenario + month.")
        else:
            agg = (
                slice_df.groupby("feature", as_index=False)["shap_value"]
                .mean()
                .sort_values("shap_value", ascending=False)
            )

            cur_token = _district_to_token(district)
            cur_dummy = f"cat__District_{cur_token}"

            is_district_dummy = agg["feature"].astype(str).str.startswith("cat__District_")
            keep_mask = (~is_district_dummy) | (agg["feature"].astype(str) == cur_dummy)
            agg = agg[keep_mask].copy()

            up = agg[agg["shap_value"] > 0].head(top_k)
            down = agg[agg["shap_value"] < 0].sort_values("shap_value", ascending=True).head(top_k)

            up_list = up["feature"].tolist()
            down_list = down["feature"].tolist()

            plot_parts = []
            if not up.empty:
                tmp = up.copy()
                tmp["direction"] = "Upward (+)"
                plot_parts.append(tmp)
            if not down.empty:
                tmp = down.copy()
                tmp["direction"] = "Downward (−)"
                plot_parts.append(tmp)

            if plot_parts:
                bar_df = pd.concat(plot_parts, ignore_index=True)
                bar_df["feature_pretty"] = bar_df["feature"].apply(_pretty_feature_name)
                bar_df = bar_df.sort_values("shap_value", ascending=True)

                fig2 = px.bar(
                    bar_df,
                    x="shap_value",
                    y="feature_pretty",
                    orientation="h",
                    title="Forecast SHAP drivers (+ upward / - downward)",
                    labels={"shap_value": "SHAP contribution", "feature_pretty": "Feature"},
                )
                fig2.update_layout(height=520, yaxis_title="", xaxis_title="SHAP contribution")
                fig2.add_vline(x=0)
                st.plotly_chart(fig2, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Upward (+)**")
                if up_list:
                    for f in up_list:
                        st.write(f"• {_pretty_feature_name(f)}")
                else:
                    st.write("—")
            with col_b:
                st.markdown("**Downward (−)**")
                if down_list:
                    for f in down_list:
                        st.write(f"• {_pretty_feature_name(f)}")
                else:
                    st.write("—")


st.info(
    f"Note: **current_price** is defined as the last historical **Median_Price** "
    f"before the forecast start month ({forecast_start.strftime('%Y-%m')}). "
    f"If history doesn’t exist before that, it falls back to the last available historical value."
)


# ============================================================
# Spatial differences (Auckland): MonthKey join + proper map view (FIXED)
# ============================================================
st.divider()
st.markdown("## Spatial differences (Auckland)")
st.caption(
    "Compare district-level forecast outcomes across Auckland for the selected scenario and forecast month. "
    "All time fields are normalized to monthly frequency (YYYY-MM)."
)


def _build_spatial_table(
    pred_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    scenario_name: str,
    month_key: str,
) -> pd.DataFrame:
    """
    Build per-district spatial comparison table for chosen scenario + month_key ('YYYY-MM').

    - Uses District_token to match districts robustly.
    - Uses MonthKey to avoid day-level mismatches.
    - Keeps all 7 districts; missing values carry a missing_reason.
    """
    rows: List[Dict[str, Any]] = []

    for d in DISTRICTS:
        token = _district_to_token(d)

        p = pred_df[
            (pred_df["District_token"] == token)
            & (pred_df["Scenario"] == scenario_name)
        ].dropna(subset=["Month_dt", "MonthKey"])

        latlon = DISTRICT_CENTROIDS_TOKEN.get(token)
        lat = latlon[0] if latlon else None
        lon = latlon[1] if latlon else None

        if p.empty:
            rows.append(
                {
                    "district": d,
                    "token": token,
                    "lat": lat,
                    "lon": lon,
                    "current_price": None,
                    "current_anchor": None,
                    "future_price": None,
                    "price_change": None,
                    "pct_change": None,
                    "forecast_start": None,
                    "missing_reason": "No matching prediction rows (district missing or name mismatch)",
                }
            )
            continue

        forecast_start_d = p["Month_dt"].min()
        forecast_start_key = forecast_start_d.to_period("M").strftime("%Y-%m")

        # current anchor from history: last history before forecast_start_d
        h = hist_df[hist_df["District_token"] == token].dropna(subset=["Month_dt"]).sort_values("Month_dt")
        cur_price = None
        cur_anchor_key = None
        if not h.empty:
            before = h[h["Month_dt"] < forecast_start_d]
            if not before.empty:
                anchor_dt = before["Month_dt"].max()
                cur_anchor_key = anchor_dt.to_period("M").strftime("%Y-%m")
                v = before.loc[before["Month_dt"] == anchor_dt, "Median_Price"].iloc[-1]
                cur_price = float(v) if pd.notna(v) else None
            else:
                anchor_dt = h["Month_dt"].max()
                cur_anchor_key = anchor_dt.to_period("M").strftime("%Y-%m")
                v = h.loc[h["Month_dt"] == anchor_dt, "Median_Price"].iloc[-1] # type: ignore
                cur_price = float(v) if pd.notna(v) else None

        # future price (MonthKey match)
        fr = p[p["MonthKey"] == month_key]
        fut_price = None
        if not fr.empty and pd.notna(fr[pred_col].iloc[-1]):
            fut_price = float(fr[pred_col].iloc[-1])

        delta = None
        pct = None
        if (cur_price is not None) and (fut_price is not None) and cur_price != 0:
            delta = fut_price - cur_price
            pct = (fut_price / cur_price - 1.0) * 100.0

        missing_reason = None
        if fut_price is None:
            missing_reason = f"No forecast value for MonthKey={month_key} (check prediction horizon / coverage)"

        rows.append(
            {
                "district": d,
                "token": token,
                "lat": lat,
                "lon": lon,
                "current_price": cur_price,
                "current_anchor": cur_anchor_key,
                "future_price": fut_price,
                "price_change": delta,
                "pct_change": pct,
                "forecast_start": forecast_start_key,
                "missing_reason": missing_reason,
            }
        )

    return pd.DataFrame(rows)


spatial_df = _build_spatial_table(pred_all, hist, scenario, sel_month_key)

if spatial_df.empty:
    st.warning("Spatial table is empty (unexpected).")
else:
    metric_key = st.selectbox(
        "Metric to compare across districts",
        options=[
            ("future_price", "Future price ($)"),
            ("pct_change", "12-month % change (%)"),
            ("price_change", "Price change ($)"),
        ],
        format_func=lambda x: x[1],
        index=0,  # default: Future price ($)
        key="spatial_metric_select",
    )[0]

    show_missing_as_grey = st.toggle(
        "Show districts with missing values as grey markers",
        value=True,
        help="Districts without a value for the selected metric are still shown as grey points.",
    )

    # ---- split for plotting ----
    valid_df = spatial_df.dropna(subset=["lat", "lon", metric_key]).copy()
    missing_df = spatial_df.dropna(subset=["lat", "lon"]).copy()
    missing_df = missing_df[missing_df[metric_key].isna()].copy()

    st.markdown("### Map view")

    # ---- One combined map (recommended): plot missing first, then valid on same axes ----
    # We build a single figure to avoid different viewports.
    base_df = spatial_df.dropna(subset=["lat", "lon"]).copy()

    if base_df.empty:
        st.info("No districts have valid lat/lon for mapping.")
    else:
        # prepare plot frame: metric might be missing; keep it but use separate marker styles
        plot_df = base_df.copy()
        plot_df["has_value"] = plot_df[metric_key].notna()
        plot_df["size_marker"] = 14
        plot_df["label"] = plot_df["district"]

        # For stable coloring: only color rows with value; others grey
        # We'll render two layers inside one fig by concatenating and using discrete styling
        fig_map = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            hover_name="district",
            hover_data={
                "current_price": True,
                "future_price": True,
                "price_change": True,
                "pct_change": True,
                "forecast_start": True,
                "current_anchor": True,
                "missing_reason": True,
                "lat": False,
                "lon": False,
                "has_value": False,
                "size_marker": False,
            },
            text="label",  # show labels so close points can be distinguished
            zoom=MAP_ZOOM,
            center=MAP_CENTER,
            height=540,
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig_map.update_traces(textposition="top center", marker={"size": 14, "opacity": 0.9})

        # If we have valid values, overlay a second trace with continuous color
        if not valid_df.empty:
            val = valid_df.copy()
            val["size_marker"] = 18  # slightly bigger
            val["label"] = val["district"]

            fig_val = px.scatter_mapbox(
                val,
                lat="lat",
                lon="lon",
                color=metric_key,
                size="size_marker",
                hover_name="district",
                hover_data={
                    "current_price": True,
                    "future_price": True,
                    "price_change": True,
                    "pct_change": True,
                    "forecast_start": True,
                    "current_anchor": True,
                    "missing_reason": True,
                    "size_marker": False,
                    "lat": False,
                    "lon": False,
                },
                text="label",
                zoom=MAP_ZOOM,
                center=MAP_CENTER,
                height=540,
            )
            fig_val.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            fig_val.update_traces(textposition="top center")

            # Merge traces: base labels + colored layer
            for tr in fig_val.data:
                fig_map.add_trace(tr)

        # If user doesn't want missing points, hide them by opacity
        if (not show_missing_as_grey) and (not missing_df.empty):
            # first trace is the base trace with all points; we want to keep only those with value
            # easiest: rebuild just with valid trace when toggle is off
            if valid_df.empty:
                st.info("No districts have values for this metric.")
            else:
                val = valid_df.copy()
                val["size_marker"] = 18
                val["label"] = val["district"]
                fig_only = px.scatter_mapbox(
                    val,
                    lat="lat",
                    lon="lon",
                    color=metric_key,
                    size="size_marker",
                    hover_name="district",
                    hover_data={
                        "current_price": True,
                        "future_price": True,
                        "price_change": True,
                        "pct_change": True,
                        "forecast_start": True,
                        "current_anchor": True,
                        "missing_reason": True,
                        "size_marker": False,
                        "lat": False,
                        "lon": False,
                    },
                    text="label",
                    zoom=MAP_ZOOM,
                    center=MAP_CENTER,
                    height=540,
                )
                fig_only.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
                fig_only.update_traces(textposition="top center")
                st.plotly_chart(fig_only, use_container_width=True)
        else:
            st.plotly_chart(fig_map, use_container_width=True)

    # Ranking bar (only valid values)
    st.markdown("### Ranking (same metric)")
    rank_df = spatial_df.dropna(subset=[metric_key]).copy()
    if rank_df.empty:
        st.info("No districts have values for this metric at the selected month.")
    else:
        rank_df = rank_df.sort_values(metric_key, ascending=False)
        fig_rank = px.bar(
            rank_df,
            x="district",
            y=metric_key,
            title=(
                f"District ranking — "
                f"{dict(future_price='Future price', pct_change='12-month % change', price_change='Price change')[metric_key]}"
            ),
        )
        fig_rank.update_layout(height=420, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_rank, use_container_width=True)

    with st.expander("Preview: spatial comparison table (all districts)", expanded=False):
        show_cols = [
            "district",
            "token",
            "forecast_start",
            "current_anchor",
            "current_price",
            "future_price",
            "price_change",
            "pct_change",
            "missing_reason",
        ]
        st.dataframe(spatial_df[show_cols], use_container_width=True)


# ============================================================
# Debug previews (collapsible)
# ============================================================
with st.expander("Preview: prediction rows (filtered)", expanded=False):
    st.dataframe(
        pred_sel[
            ["Month", "Month_dt", "MonthKey", "District", "District_token", "Scenario", pred_col]
        ].head(int(preview_rows)),
        use_container_width=True,
    )

with st.expander("Preview: historical rows (filtered)", expanded=False):
    st.dataframe(
        hist_sel[
            [hist_month_col, "Month_dt", "MonthKey", hist_district_col, "District_token", "Median_Price"]
        ].head(int(preview_rows)),
        use_container_width=True,
    )

