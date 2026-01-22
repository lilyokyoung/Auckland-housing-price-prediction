# ui/style.py
from __future__ import annotations
import streamlit as st

LIGHT_CSS = """
<style>
:root{
  --base-font: 17px;
  --text: #111827;
  --muted: #6b7280;
  --border: rgba(0,0,0,0.10);
}

/* Global */
html, body {
  font-size: var(--base-font);
  color: var(--text);
}

/* Main content */
section.main {
  padding-top: 0.5rem;
}

/* Headings */
h1 { font-size: 2.0rem !important; font-weight: 750; letter-spacing: -0.02em; }
h2 { font-size: 1.6rem !important; font-weight: 720; letter-spacing: -0.01em; }
h3 { font-size: 1.25rem !important; font-weight: 700; }
h4 { font-size: 1.10rem !important; font-weight: 680; }

/* Markdown body text */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
  line-height: 1.55;
}

/* Sidebar typography */
section[data-testid="stSidebar"] {
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
  font-size: 0.98rem !important;
}

/* Inputs */
input, textarea, select {
  font-size: 1.0rem !important;
}

/* Buttons */
.stButton > button {
  font-size: 1.0rem !important;
  padding: 0.55rem 1.0rem !important;
  border-radius: 10px !important;
}

/* Tabs */
button[data-baseweb="tab"]{
  font-size: 1.0rem !important;
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
}

/* Alerts */
div[data-testid="stAlert"] * {
  font-size: 1.0rem !important;
}

/* Expander */
div[data-testid="stExpander"] * {
  font-size: 1.0rem !important;
}

/* Code/JSON */
pre, code {
  font-size: 0.95rem !important;
}

/* DataFrame */
div[data-testid="stDataFrame"] * {
  font-size: 0.95rem !important;
}
div[data-testid="stDataFrame"] thead * {
  font-weight: 700 !important;
}

/* Divider */
hr { border-color: var(--border); }
</style>
"""

def apply_light_theme() -> None:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
