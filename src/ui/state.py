# src/ui/state.py
from __future__ import annotations
import streamlit as st

DISTRICTS = [
    "Auckland City", "Franklin", "Manukau", "North Shore",
    "Papakura", "Rodney", "Waitakere"
]

def init_state() -> None:
    defaults = {
        "api_base": "http://127.0.0.1:8000",
        "district": "Auckland City",
        "start_month": "",
        "end_month": "",
        "include_history": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
