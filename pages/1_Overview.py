# pages/1_Overview.py
from __future__ import annotations

import streamlit as st

# -------------------------------------------------
# Unified bootstrap: make project root importable
# -------------------------------------------------
from src.ui.bootstrap import bootstrap

bootstrap()

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Overview | Auckland Housing Price AVM",
    layout="wide",
)

# -------------------------------------------------
# Minimal page-level styling (safe + lightweight)
# -------------------------------------------------
st.markdown(
    """
<style>
/* tighten default spacing a bit */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* card-like container */
.avm-card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 18px 18px 14px 18px;
    background: rgba(255,255,255,0.55);
}

/* subtle badge */
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

/* section title spacing */
.avm-section-title { margin-top: 0.2rem; margin-bottom: 0.4rem; }
.avm-muted { color: rgba(107,114,128,1); }
.avm-hr { margin: 1.2rem 0 1.0rem 0; border: none; height: 1px; background: rgba(0,0,0,0.08); }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# HERO (title + subtitle)
# -------------------------------------------------
st.markdown(
    """
<div class="avm-card">
  <div>
    <span class="avm-badge">🏘️ AVM</span>
    <span class="avm-badge">📈 Forecast</span>
    <span class="avm-badge">🧩 Explainable AI</span>
    <span class="avm-badge">🗺️ Auckland</span>
  </div>

  <h1 style="margin: 0.35rem 0 0.2rem 0;">Auckland Housing Price AVM</h1>
  <p class="avm-muted" style="font-size: 1.05rem; margin: 0.2rem 0 0.6rem 0;">
    Application of Artificial Intelligence (AI) for property valuation in Auckland —
    scenario-based forecasting with interpretable explanations (Narrative + SHAP).
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="avm-hr"/>', unsafe_allow_html=True)

# -------------------------------------------------
# KPI cards
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spatial unit", "7 districts")
c2.metric("Frequency", "Monthly")
c3.metric("Scenarios", "Low / Base / High")
c4.metric("Explainability", "Narrative + SHAP")

st.markdown('<hr class="avm-hr"/>', unsafe_allow_html=True)

# -------------------------------------------------
# System architecture (high-level)
# -------------------------------------------------
st.markdown("### 🧠 System Architecture", help="High-level workflow of the AVM system.")

a1, a2, a3, a4, a5 = st.columns([1.15, 0.2, 1.15, 0.2, 1.3])
with a1:
    st.markdown(
        """
<div class="avm-card">
  <h4 class="avm-section-title">1) Data Layer</h4>
  <div class="avm-muted">
    Housing market + macroeconomic + demographic indicators (monthly, district-level).
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with a2:
    st.markdown("### →")
with a3:
    st.markdown(
        """
<div class="avm-card">
  <h4 class="avm-section-title">2) Modeling Layer</h4>
  <div class="avm-muted">
    Automated Valuation Model (AVM) trained on engineered features (lags, transforms).
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with a4:
    st.markdown("### →")
with a5:
    st.markdown(
        """
<div class="avm-card">
  <h4 class="avm-section-title">3) Forecast + Explanation</h4>
  <div class="avm-muted">
    Scenario forecasts (Low/Base/High) with SHAP drivers and narrative interpretation.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("")

b1, b2 = st.columns([1.0, 1.0])
with b1:
    st.markdown(
        """
<div class="avm-card">
  <h4 class="avm-section-title">Outputs</h4>
  <ul style="margin: 0.2rem 0 0 1.1rem;">
    <li>District-level price forecasts (monthly horizon)</li>
    <li>Driver attribution (SHAP) per selected month</li>
    <li>Readable narrative summaries for transparency</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
with b2:
    st.markdown(
        """
<div class="avm-card">
  <h4 class="avm-section-title">Intended Use</h4>
  <ul style="margin: 0.2rem 0 0 1.1rem;">
    <li>Support valuation and market monitoring</li>
    <li>Compare scenario sensitivity (Low/Base/High)</li>
    <li>Explain model signals (directional, not causal)</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="avm-hr"/>', unsafe_allow_html=True)

# -------------------------------------------------
# Project background (keep your content, improved formatting)
# -------------------------------------------------
st.markdown("### 📌 Project Background")

st.markdown(
    """
This project develops an **AI-enabled Automated Valuation Model (AVM)** to support
**district-level house price valuation and forecasting** across Auckland.

The system integrates:
- **Housing-market activity** (e.g., sales / liquidity proxies)
- **Macroeconomic conditions** (e.g., OCR / interest-rate environment)
- **Demographic demand indicators** (e.g., migration-related measures)

and provides **interpretable explanations** (Narrative + SHAP) to improve transparency
for decision-making and academic analysis.
"""
)

# -------------------------------------------------
# Call to action
# -------------------------------------------------
st.success("🚀 Start from **Forecast & Explanation** to explore district forecasts and month-specific SHAP drivers.")
