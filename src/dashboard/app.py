import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mlflow.sklearn
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.credit_risk.config import AppConfig
from src.credit_risk.utils import load_csv
from src.credit_risk.explainability import RiskExplainer


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

cfg = AppConfig()
API_URL = "http://localhost:8000/predict"


# ---------------------------------------------------
# Custom Styling (Professional Look)
# ---------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
h1 {
    font-weight: 700 !important;
}
.stMetric {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# Load Resources (Cached)
# ---------------------------------------------------
@st.cache_resource
def load_resources():
    df = load_csv(cfg.FINAL_DATA_PATH)

     # CLOUD DEPLOYMENT LOGIC:
    # Try to load the standalone file first (for Streamlit Cloud)
    # If not found, try to load from MLflow (for local dev)
    model_path = "models/credit_risk_model.joblib"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Fallback to MLflow for local development
        experiment = mlflow.get_experiment_by_name("Credit_Risk_Production_Training")
        runs = mlflow.search_runs(experiment.experiment_id, order_by=["metrics.roc_auc DESC"])
        best_run_id = runs.iloc[0].run_id
        model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    
    explainer = RiskExplainer(model, []) 
    return df, model, explainer


df, model, explainer = load_resources()


# ---------------------------------------------------
# Sidebar – Risk Simulator
# ---------------------------------------------------
st.sidebar.markdown("## Credit Risk Simulator")

def get_user_inputs():
    st.sidebar.markdown("### Financial Profile")
    total_amount = st.sidebar.number_input("Total Amount", value=5000.0)
    avg_amount = st.sidebar.number_input("Average Amount", value=500.0)

    st.sidebar.markdown("### Behavioral Profile")
    total_tx = st.sidebar.slider("Total Transactions", 1, 100, 10)
    active_days = st.sidebar.slider("Tenure (Days)", 1, 365, 30)
    recency = st.sidebar.slider("Recency (Days)", 0, 30, 5)

    st.sidebar.markdown("### Categorical Attributes")
    provider = st.sidebar.selectbox("Provider", df['ProviderId'].unique())
    category = st.sidebar.selectbox("Category", df['ProductCategory'].unique())
    channel = st.sidebar.selectbox("Channel", df['ChannelId'].unique())
    pricing = st.sidebar.selectbox("Pricing Strategy", df['PricingStrategy'].unique())

    payload = {
        "total_transactions": int(total_tx),
        "total_amount": float(total_amount),
        "avg_amount": float(avg_amount),
        "std_amount": float(avg_amount * 0.1),
        "avg_fee_paid": 10.0,
        "total_refunds_count": 0,
        "tx_hour_mean": 12.0,
        "tx_day_mean": 15.0,
        "active_days": int(active_days),
        "Recency": int(recency),
        "Frequency": int(total_tx),
        "Monetary": float(total_amount),
        "ProviderId": str(provider),
        "ProductCategory": str(category),
        "ChannelId": str(channel),
        "PricingStrategy": str(pricing)
    }

    return payload


payload = get_user_inputs()


# ---------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------
st.title("Credit Risk Intelligence Dashboard")
st.caption("Real-time credit risk scoring with model explainability")


# ---------------------------------------------------
# API Prediction
# ---------------------------------------------------
with st.spinner("Analyzing credit profile..."):
    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            score = result['credit_score']
            prob = result['risk_probability']
            is_high_risk = result['is_high_risk']
            api_status = "Connected"
        else:
            st.error(f"API Error: {response.status_code}")
            st.stop()

    except requests.exceptions.ConnectionError:
        st.error(
            "API is Offline. Please start the FastAPI server using:\n\n"
            "`uvicorn src.api.main:app --reload`"
        )
        st.stop()


# ---------------------------------------------------
# Results Section
# ---------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Decision Outcome")

    if is_high_risk == 0:
        st.success("Approved")
    else:
        st.error("Rejected")

    st.metric("Credit Score", f"{score}")
    st.metric("Risk Probability", f"{prob:.2%}")
    st.caption(f"API Status: {api_status}")


with col2:
    st.subheader("Model Explainability (SHAP Waterfall)")

    input_df = pd.DataFrame([payload])
    shap_vals, X_trans = explainer.generate_shap_values(input_df)

    base_val = explainer.explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=base_val,
        data=X_trans[0],
        feature_names=explainer.transformed_feature_names
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)


# ---------------------------------------------------
# Portfolio Overview
# ---------------------------------------------------
st.divider()
st.subheader("Portfolio Risk Distribution")
st.bar_chart(df['is_high_risk'].value_counts())
