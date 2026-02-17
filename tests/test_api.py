import pytest
from fastapi.testclient import TestClient
import unittest.mock as mock
import pandas as pd
from src.api.main import app, load_production_model
import src.api.main as api_main

client = TestClient(app)

def test_health_check_success():
    """
    Tests the /health endpoint when the model is correctly loaded.
    Verifies that the API returns a 'ready' status.
    """
    with mock.patch("src.api.main.model", "not none"):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

def test_health_check_fail():
    """
    Tests the /health endpoint when the model failed to load.
    Ensures the API remains up but reports the correct 'model_not_loaded' status.
    """
    with mock.patch("src.api.main.model", None):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "model_not_loaded"

def test_predict_success():
    """
    Tests the full /predict logic using a mocked model.
    Verifies:
    1. Successful response for a valid payload.
    2. Correct calculation of the Credit Score (300-850 scale).
    3. Proper mapping of High-Risk probability.
    """
    mock_model = mock.MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.1, 0.9]] # 90% risk

    payload = {
        "total_transactions": 5, "total_amount": 100.0, "avg_amount": 20.0,
        "std_amount": 2.0, "avg_fee_paid": 1.0, "total_refunds_count": 0,
        "tx_hour_mean": 12.0, "tx_day_mean": 10.0, "active_days": 30,
        "Recency": 5, "Frequency": 5, "Monetary": 100.0,
        "ProviderId": "P1", "ProductCategory": "C1", "ChannelId": "CH1", "PricingStrategy": "1"
    }

    with mock.patch("src.api.main.model", mock_model):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_high_risk"] == 1
        # Expected Score: 850 - (0.9 * 550) = 355
        assert data["credit_score"] == 355

def test_predict_internal_error():
    """
    Tests the API's error handling when the model crashes during inference.
    Verifies that the API catches the exception and returns a 500 Internal Server Error.
    """
    mock_model = mock.MagicMock()
    mock_model.predict.side_effect = Exception("Internal Model Crash")
    
    with mock.patch("src.api.main.model", mock_model):
        payload = {
            "total_transactions": 5, "total_amount": 100.0, "avg_amount": 20.0,
            "std_amount": 2.0, "avg_fee_paid": 1.0, "total_refunds_count": 0,
            "tx_hour_mean": 12.0, "tx_day_mean": 10.0, "active_days": 30,
            "Recency": 5, "Frequency": 5, "Monetary": 100.0,
            "ProviderId": "P1", "ProductCategory": "C1", "ChannelId": "CH1", "PricingStrategy": "1"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 500

def test_load_model_joblib():
    """
    Tests the model loading logic to ensure it prioritizes local joblib files.
    This is essential for deployments on platforms like Render or Streamlit Cloud.
    """
    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("joblib.load", return_value="mocked_binary_model"):
        res = load_production_model()
        assert res == "mocked_binary_model"

def test_load_model_mlflow_fail():
    """
    Tests the fallback logic in the model loader.
    Ensures that if neither a local file nor a valid MLflow experiment 
    is found, the loader returns None safely.
    """
    with mock.patch("os.path.exists", return_value=False), \
         mock.patch("mlflow.get_experiment_by_name", return_value=None):
        res = load_production_model()
        assert res is None