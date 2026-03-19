## 🏠 Auckland Housing Price Prediction

An AI-driven application for property valuation across seven Auckland districts: Auckland City, Franklin, Manukau, North Shore, Papakura, Rodney, and Waitakere.


## 📌 Project Overview

This project was designed around three core questions:
- **What** to build — a multi-district, scenario-aware housing price forecast system
- **How** to build it — a multi-layer analytical framework combining econometrics, machine learning, and explainable AI
- **How to present it** — an interactive Streamlit web application backed by a deployed FastAPI service

---

## 🏗️ Framework Architecture

The project is structured into five layers:

### 🔹 1. Data Engineering
- **Target variable:** Monthly median housing price across 7 Auckland districts
- **Explanatory variables:**
  - *Macroeconomic:* OCR, mortgage interest rates, CGPI for dwelling units, unemployment rate
  - *Demographic:* Net migration
  - *Housing market dynamics:* Sales counts, building consents, average weekly rent
- Dataset spans **August 2018 – June 2025** (monthly frequency)
- Feature engineering includes: real mortgage rate derivation, lagged features, log transformation for skewed variables
- EDA conducted on training set only to prevent data leakage

### 🔹 2. Forecasting Layer
- **Macroeconomic variables:** Bayesian VAR model (captures dynamic interdependencies)
- **District-level variables:** SARIMAX models (handles time dependence and seasonality)
- **Annual variables (migration, rent):** Scenario-based approach — Base / High / Low scenarios
- Building consents forecasts used as exogenous inputs for sales count predictions

### 🔹 3. Modelling & Prediction
| Model | Description |
|---|---|
| Multiple Linear Regression + LASSO | Baseline with feature selection |
| SVR | Support Vector Regression |
| Random Forest ✅ | **Best performer** |
| XGBoost | Gradient boosting |

- **Best model:** Random Forest — RMSE: **11.10%**, R²: **0.532** on test set
- Time-series cross-validation applied to prevent temporal leakage
- Final model retrained on full dataset for future forecasting

### 🔹 4. Explainable Analytics
- **SHAP summary plots** used to identify key price drivers
- Reveals spatial differences across districts
- Improves transparency for stakeholders

### 🔹 5. AI Agent & Deployment
- **FastAPI** backend containerized with **Docker**, deployed on **Azure Cloud**
- **Streamlit** frontend with the following pages:
  - Project Overview
  - Data & Features
  - Modelling & Evaluation
  - Forecast & Explanation (July 2025 – June 2026)
  - AI Agent (investment & risk insights)
- **GitHub** for version control and CI/CD

---

## 📊 Key Insights

| District | Investment Profile |
|---|---|
| Auckland City | Most stable — suitable for conservative strategies |
| Rodney | High potential return, high risk |
| Manukau & Waitakere | Balanced risk–return profiles |
| North Shore | Relatively stable, moderate risk with modest return potential |
| Papakura & Franklin | Weaker performance under forecast horizon |

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML/Stats:** scikit-learn, XGBoost, statsmodels, SHAP
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Deployment:** Docker, Azure Cloud
- **Version Control:** GitHub

---

## 📁 Project Structure

```
├── data/               # Raw and processed datasets
├── outputs/            # Model outputs and forecast results
├── pages/              # Streamlit app pages
├── src/                # Core source code (models, forecasting, agents)
├── Home.py             # Streamlit entry point
├── Dockerfile.api      # Docker config for FastAPI backend
├── Dockerfile.ui       # Docker config for Streamlit frontend
├── requirements.txt    # UI dependencies
└── requirements.api.txt# API dependencies
```

---

## 🚀 Getting Started

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Streamlit UI
streamlit run Home.py

# Start the FastAPI backend (in a separate terminal)
uvicorn src.api.main:app --reload
```

### Deployment (Azure CLI + ACR)

```bash
# 1. Login to Azure

# 2. Login to Azure Container Registry

# 3. Build Docker images

# 4. Push images to ACR

# 5. Deploy to Azure Container Apps

```
