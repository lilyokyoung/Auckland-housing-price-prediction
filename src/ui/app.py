# src/ui/app.py
from __future__ import annotations
import os
import sys
from pathlib import Path

import streamlit as st

# ============================================================
# Bootstrap: make project root importable so "import src.*" works
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.style import apply_light_theme  # noqa: E402
from src.ui.api_client import get_health  # noqa: E402


def main() -> None:
    # ============================================================
    # Page config + theme
    # ============================================================
    st.set_page_config(page_title="Auckland Housing Price AVM", layout="wide")
    apply_light_theme()

    # ============================================================
    # Global state (auto-sync across pages)
    # ============================================================
    # Local dev default: 127.0.0.1
    # Docker default: host.docker.internal (UI container -> host -> API)
    DEFAULT_API_BASE = os.getenv(
    "API_BASE_URL",
    "https://avm-api.delightfulstone-2a01eb1e.australiaeast.azurecontainerapps.io",
)


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

    if "api_base" not in st.session_state:
        st.session_state["api_base"] = DEFAULT_API_BASE
    if "district" not in st.session_state:
        st.session_state["district"] = "Auckland City"
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = "base"
    if "start_month" not in st.session_state:
        st.session_state["start_month"] = ""
    if "end_month" not in st.session_state:
        st.session_state["end_month"] = ""
    if "include_history" not in st.session_state:
        st.session_state["include_history"] = True
    if "api_health" not in st.session_state:
        st.session_state["api_health"] = None  # type: ignore

    # ============================================================
    # Sidebar controls (global)
    # ============================================================
    with st.sidebar:
        st.markdown("## Auckland Housing Price AVM")
        st.caption("Global controls (auto-sync across pages)")

        st.session_state["api_base"] = st.text_input(
            "API Base",
            value=st.session_state["api_base"],
            help="Example: http://127.0.0.1:8000 (local) or your Azure URL",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Check API"):
                try:
                    st.session_state["api_health"] = get_health(st.session_state["api_base"])
                    st.success("API is reachable ✅")
                except Exception as e:
                    st.session_state["api_health"] = {"error": f"{type(e).__name__}: {e}"}
                    st.error("API check failed ❌")
        with c2:
            if st.button("Reset API"):
                st.session_state["api_base"] = DEFAULT_API_BASE
                st.session_state["api_health"] = None

        if st.session_state.get("api_health"):
            with st.expander("Last health response", expanded=False):
                st.json(st.session_state["api_health"])

        st.divider()

        current_district = st.session_state.get("district", DISTRICTS[0])
        if current_district not in DISTRICTS:
            current_district = DISTRICTS[0]
            st.session_state["district"] = current_district

        st.session_state["district"] = st.selectbox(
            "District",
            DISTRICTS,
            index=DISTRICTS.index(current_district),
        )

        current_scenario = st.session_state.get("scenario", "base")
        if current_scenario not in SCENARIOS:
            current_scenario = "base"
            st.session_state["scenario"] = current_scenario

        st.session_state["scenario"] = st.selectbox(
            "Scenario",
            SCENARIOS,
            index=SCENARIOS.index(current_scenario),
        )

        st.divider()
        st.session_state["start_month"] = st.text_input("Start month (optional)", value=st.session_state["start_month"])
        st.session_state["end_month"] = st.text_input("End month (optional)", value=st.session_state["end_month"])
        st.session_state["include_history"] = st.checkbox("Include history (actual)", value=st.session_state["include_history"])

    # ============================================================
    # Main: minimal landing
    # ============================================================
    st.markdown("# Auckland Housing Price AVM")
    st.caption("Open pages from the Streamlit menu (top-left). Start with **Overview**.")
    st.info("If you still see an **app** page, it means you're running Streamlit from `src/ui/app.py` directly.")


# Allow running directly (optional)
if __name__ == "__main__":
    main()
