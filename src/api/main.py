import os
import joblib
import pandas as pd
import mlflow.sklearn
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.credit_risk.config import AppConfig
from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the model
model = None

def load_production_model():
    """Finds the best model from the Production experiment."""
    if os.path.exists("models/credit_risk_model.joblib"):
        return joblib.load("models/credit_risk_model.joblib")
    else:
        try:
            # TARGET THE NEW EXPERIMENT
            experiment_name = "Credit_Risk_Production_Training"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                logger.error(f"Experiment '{experiment_name}' not found.")
                return None

            # Get best run based on ROC-AUC
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.roc_auc DESC"],
                max_results=1
            )
            
            if runs.empty:
                logger.error("No runs found in experiment.")
                return None
                
            best_run_id = runs.iloc[0]["run_id"]
            logger.info(f"Loading best model from Run ID: {best_run_id}")
            
            return mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_production_model()
    yield

app = FastAPI(title="Bati Bank Credit Risk API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ready" if model else "model_not_loaded"}

@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame for the model pipeline
        input_df = pd.DataFrame([request.model_dump()])
        
        # The pipeline handles all scaling and OHE
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # Financial Standard Credit Score (300-850)
        # Higher probability of risk = Lower credit score
        credit_score = int(850 - (probability * 550))
        
        return {
            "risk_probability": round(probability, 4),
            "is_high_risk": prediction,
            "credit_score": credit_score
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)