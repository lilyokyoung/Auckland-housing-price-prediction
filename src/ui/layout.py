# src/ui/layout.py
from __future__ import annotations
import streamlit as st
from ui.state import DISTRICTS

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Auckland Housing Price Forecast")
        st.caption("Global controls (shared across pages)")

        st.session_state["api_base"] = st.text_input(
            "API Base", value=st.session_state["api_base"]
        )
        st.session_state["district"] = st.selectbox(
            "District",
            DISTRICTS,
            index=DISTRICTS.index(st.session_state["district"]),
        )
        st.session_state["start_month"] = st.text_input(
            "Start month (optional)", value=st.session_state["start_month"]
        )
        st.session_state["end_month"] = st.text_input(
            "End month (optional)", value=st.session_state["end_month"]
        )
        st.session_state["include_history"] = st.checkbox(
            "Include history (actual)", value=st.session_state["include_history"]
        )
