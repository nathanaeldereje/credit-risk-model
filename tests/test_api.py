from fastapi.testclient import TestClient
from src.api.main import app
import unittest.mock as mock

client = TestClient(app)

def test_health_check():
    """Tests the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # Match the logic in your main.py
    assert "status" in response.json()

def test_predict_endpoint_validation():
    """
    Tests that the API returns 422 Unprocessable Entity 
    if we send an empty or bad payload (Pydantic validation).
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422