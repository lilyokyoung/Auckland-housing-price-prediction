from __future__ import annotations

import streamlit as st

# -------------------------------------------------
# Auto-redirect to Overview page
# -------------------------------------------------
st.set_page_config(
    page_title="Auckland Housing Price AVM",
    layout="wide",
)

# Immediately redirect to Overview
st.switch_page("pages/1_Overview.py")