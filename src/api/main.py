import sys
import os
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime


# Add src to path to allow imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse

# Global variable to hold the model
model = None

def load_best_model():
    """
    Finds the best run from the local MLflow experiment and loads the model.
    """
    try:
        experiment_name = "Credit_Risk_Model_Experiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        # Search for the best run based on ROC_AUC
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.roc_auc DESC"]
        )
        
        if runs.empty:
            raise ValueError("No runs found in the experiment.")
            
        best_run_id = runs.iloc[0]["run_id"]
        print(f"Loading best model from Run ID: {best_run_id}")
        
        # Load model using the specific run URI
        logged_model_uri = f"runs:/{best_run_id}/model"
        loaded_model = mlflow.sklearn.load_model(logged_model_uri)
        return loaded_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    model = load_best_model()
    if model is None:
        print("WARNING: Model failed to load.")
    yield
    # Clean up on shutdown (if needed)

app = FastAPI(title="Credit Risk Scoring API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Credit Risk API is running"}

@app.post("/predict", response_model=CreditRiskResponse)
def predict_risk(request: CreditRiskRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to DataFrame
        # We must wrap it in a list to create a single-row DataFrame
        input_data = pd.DataFrame([request.model_dump()])
        # Approximate snapshot logic (same idea as training)
        snapshot_date = pd.Timestamp(datetime.utcnow())

        # You MUST have last_transaction_time logic
        # Since API doesn't have history, we approximate recency using tx_day_mean
        input_data["Recency"] = 30 - input_data["tx_day_mean"]  # proxy
        input_data["Frequency"] = input_data["total_transactions"]
        input_data["Monetary"] = input_data["total_amount"]

        # Optional but expected
        input_data["active_days"] = input_data["total_transactions"]
        # Make Prediction
        # The pipeline inside the model handles scaling/encoding automatically
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # Probability of Class 1 (High Risk)
        
        # Calculate a simple Credit Score (300-850 scale)
        # Inverse of risk: Higher risk = Lower score
        base_score = 850
        risk_penalty = probability * 550 # If prob is 1.0, penalty is 550 -> score 300
        credit_score = int(base_score - risk_penalty)
        
        return {
            "risk_probability": round(probability, 4),
            "is_high_risk": int(prediction),
            "credit_score": credit_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)