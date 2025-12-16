from fastapi.testclient import TestClient
from src.api.main import app
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk API is running"}

# Note: Testing /predict usually requires a mocked model to avoid loading MLflow in CI
# For simplicity in this assignment, we check basic startup. 
# In a real scenario, you would mock load_best_model.