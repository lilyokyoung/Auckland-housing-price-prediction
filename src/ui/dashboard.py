from __future__ import annotations

import streamlit as st

from style import apply_light_theme

# Optional: import the page renderers (you can refactor each page into a render() function)
# For now, we will keep content inline by calling files' functions if you add them.
# If your current pages are standalone, you can migrate logic gradually.

from api_client import post_json, get_json 


# -----------------------------
# Page config + style
# -----------------------------
st.set_page_config(page_title="Auckland Housing Price Forecast", layout="wide")
apply_light_theme()

# -----------------------------
# Global state (auto-sync)
# -----------------------------
if "api_base" not in st.session_state:
    st.session_state["api_base"] = "http://127.0.0.1:8000"
if "district" not in st.session_state:
    st.session_state["district"] = "Auckland City"
if "start_month" not in st.session_state:
    st.session_state["start_month"] = ""
if "end_month" not in st.session_state:
    st.session_state["end_month"] = ""
if "include_history" not in st.session_state:
    st.session_state["include_history"] = True
if "chat" not in st.session_state:
    st.session_state["chat"] = []

# -----------------------------
# Header
# -----------------------------
st.markdown("# Auckland Housing Price Forecast <span class='badge'>Dashboard Pro</span>", unsafe_allow_html=True)
st.caption("Tabs navigation + auto-synced controls across modules.")

# -----------------------------
# Global controls row (like Pro dashboard top controls)
# -----------------------------
c1, c2, c3, c4, c5 = st.columns([1.6, 1.2, 1.2, 1.3, 1.4])

with c1:
    st.session_state["api_base"] = st.text_input("API Base", value=st.session_state["api_base"])

with c2:
    st.session_state["district"] = st.selectbox(
        "District",
        ["Auckland City", "Franklin", "Manukau", "North Shore", "Papakura", "Rodney", "Waitakere"],
        index=["Auckland City", "Franklin", "Manukau", "North Shore", "Papakura", "Rodney", "Waitakere"].index(
            st.session_state["district"]
        ),
    )

with c3:
    st.session_state["start_month"] = st.text_input("Start month (optional)", value=st.session_state["start_month"])

with c4:
    st.session_state["end_month"] = st.text_input("End month (optional)", value=st.session_state["end_month"])

with c5:
    st.session_state["include_history"] = st.checkbox(
        "Include history (actual)", value=st.session_state["include_history"]
    )

st.divider()

# -----------------------------
# Top Tabs (like your screenshot)
# -----------------------------
tab_data, tab_viz, tab_spatial, tab_shap, tab_forecast, tab_chat = st.tabs(
    ["📁 Data", "📊 Visualization", "🧭 Spatial", "🧠 SHAP", "📈 Forecast", "💬 Agent Chat"]
)

# -----------------------------
# Data tab (placeholder)
# -----------------------------
with tab_data:
    st.markdown("## Data")
    st.info("This tab can show dataset summaries, row counts, missing values, and latest refresh date.")
    # You can connect to your API later for data profiling endpoints.

# -----------------------------
# Visualization tab (placeholder)
# -----------------------------
with tab_viz:
    st.markdown("## Visualization")
    st.info("This tab can host exploratory plots (trend, seasonality, correlations).")

# -----------------------------
# Spatial tab
# -----------------------------
with tab_spatial:
    st.markdown("## Spatial Context")
    api_base = st.session_state["api_base"]
    if st.button("Load Spatial Context", type="primary"):
        with st.spinner("Loading spatial context..."):
            data = get_json(api_base, "/spatial_context")
        st.json(data)

# -----------------------------
# SHAP tab
# -----------------------------
with tab_shap:
    st.markdown("## SHAP Explanation")
    api_base = st.session_state["api_base"]
    if st.button("Load SHAP Metadata", type="primary"):
        with st.spinner("Loading SHAP metadata..."):
            data = get_json(api_base, "/shap")
        st.json(data)

# -----------------------------
# Forecast tab (core)
# -----------------------------
with tab_forecast:
    st.markdown("## Forecast (Low / Base / High)")

    api_base = st.session_state["api_base"]
    district = st.session_state["district"]
    start_month = st.session_state["start_month"] or None
    end_month = st.session_state["end_month"] or None
    include_history = bool(st.session_state["include_history"])

    run = st.button("Run 3-Scenario Forecast", type="primary")

    if run:
        payload = {
            "district": district,
            "start_month": start_month,
            "end_month": end_month,
            "include_history": include_history,
        }

        with st.spinner("Running forecast..."):
            data = post_json(api_base, "/dashboard/forecast", payload, timeout=600)

        # Show response quickly first (you can replace with chart rendering once API returns normalized rows)
        st.success("Forecast returned successfully.")
        st.json(data)

    else:
        st.info("Click **Run 3-Scenario Forecast** to generate outputs.")

# -----------------------------
# Agent chat tab
# -----------------------------
with tab_chat:
    st.markdown("## Agent Chat")

    api_base = st.session_state["api_base"]

    # show history
    for role, content in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(content)

    msg = st.chat_input("Ask the agent (e.g., 'metrics', 'shap', 'spatial context')")

    if msg:
        st.session_state["chat"].append(("user", msg))
        with st.chat_message("user"):
            st.markdown(msg)

        try:
            with st.spinner("Agent responding..."):
                data = post_json(api_base, "/agent", {"user_query": msg}, timeout=300)
            reply = data.get("narrative", "")
        except Exception as e:
            reply = f"Error: {e}"

        st.session_state["chat"].append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
