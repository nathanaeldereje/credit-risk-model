import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mlflow.sklearn
import shap
from src.credit_risk.config import AppConfig
from src.credit_risk.utils import load_csv
from src.credit_risk.explainability import RiskExplainer

# --- Page Config ---
st.set_page_config(page_title="Bati Bank Risk Dashboard", layout="wide")
cfg = AppConfig()

# API Endpoint (Ensure your FastAPI is running on port 8000)
API_URL = "http://localhost:8000/predict"

# --- Load Resources (Cached) ---
@st.cache_resource
def load_resources():
    df = load_csv(cfg.FINAL_DATA_PATH)
    # We still load the model locally just to generate SHAP plots
    experiment = mlflow.get_experiment_by_name("Credit_Risk_Production_Training")
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["metrics.roc_auc DESC"])
    best_run_id = runs.iloc[0].run_id
    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    explainer = RiskExplainer(model, []) 
    return df, model, explainer

df, model, explainer = load_resources()

# --- Sidebar Inputs ---
st.sidebar.header("🔍 Credit Score Simulator")

def get_user_inputs():
    st.sidebar.subheader("Financials")
    total_amount = st.sidebar.number_input("Total Amount", value=5000.0)
    avg_amount = st.sidebar.number_input("Avg Amount", value=500.0)
    
    st.sidebar.subheader("Behavior")
    total_tx = st.sidebar.slider("Total Transactions", 1, 100, 10)
    active_days = st.sidebar.slider("Tenure (Days)", 1, 365, 30)
    recency = st.sidebar.slider("Recency (Days)", 0, 30, 5)
    
    st.sidebar.subheader("Categorical")
    provider = st.sidebar.selectbox("Provider", df['ProviderId'].unique())
    category = st.sidebar.selectbox("Category", df['ProductCategory'].unique())
    channel = st.sidebar.selectbox("Channel", df['ChannelId'].unique())
    pricing = st.sidebar.selectbox("Pricing Strategy", df['PricingStrategy'].unique())

    # Build payload exactly as the API (Pydantic) expects it
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

# --- Main Page ---
st.title("🏦 Bati Bank | Credit Risk Intelligence")

# 1. API CONNECTION LOGIC
with st.spinner("Requesting prediction from API..."):
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
        st.error("🚨 API is Offline. Please start the FastAPI server with `uvicorn src.api.main:app`.")
        st.stop()

# 2. DISPLAY RESULTS
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Decision Result")
    if is_high_risk == 0:
        st.success("✅ APPROVED")
        color = "green"
    else:
        st.error("❌ REJECTED")
        color = "red"

    st.metric("Credit Score", score)
    st.write(f"Risk Probability: {prob:.2%}")
    st.caption(f"Status: API {api_status}")

with col2:
    st.subheader("💡 Decision Transparency (SHAP)")
    # We use the local explainer to show the "Why"
    input_df = pd.DataFrame([payload])
    shap_vals, X_trans = explainer.generate_shap_values(input_df)
    
    base_val = explainer.explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)): base_val = base_val[1]
    
    fig, ax = plt.subplots()
    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=base_val,
        data=X_trans[0],
        feature_names=explainer.transformed_feature_names
    )
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(plt.gcf())

st.divider()
st.subheader("📈 Portfolio Distribution")
st.bar_chart(df['is_high_risk'].value_counts())