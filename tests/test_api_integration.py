import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import unittest.mock as mock

client = TestClient(app)

# Integration Test: API + Model Logic
def test_predict_integration():
    """
    Test the full flow from API request to Prediction response.
    We mock the 'model' object to avoid needing a real MLflow server in CI,
    but we test the API's ability to handle the data.
    """
    # 1. Setup a dummy payload matching your Pydantic model
    payload = {
        "total_transactions": 5,
        "total_amount": 1000.0,
        "avg_amount": 200.0,
        "std_amount": 20.0,
        "avg_fee_paid": 5.0,
        "total_refunds_count": 0,
        "tx_hour_mean": 14.0,
        "tx_day_mean": 10.0,
        "active_days": 100,
        "Recency": 5,
        "Frequency": 5,
        "Monetary": 1000.0,
        "ProviderId": "ProviderId_1",
        "ProductCategory": "airtime",
        "ChannelId": "ChannelId_3",
        "PricingStrategy": "2"
    }

    # 2. Mock the model's predict and predict_proba methods
    # This simulates the model being loaded and working
    with mock.patch("src.api.main.model") as mocked_model:
        mocked_model.predict.return_value = [0]
        mocked_model.predict_proba.return_value = [[0.8, 0.2]] # 20% risk

        # 3. Call the API
        response = client.post("/predict", json=payload)

    # 4. Assertions
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert "credit_score" in data
    # Verify the credit score math (850 - 0.2 * 550 = 740)
    assert data["credit_score"] == 740
    assert data["is_high_risk"] == 0

def test_health_check_integration():
    """Test if the health endpoint correctly reports model status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()