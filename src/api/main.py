import logging
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, ValidationError  # <-- Important fix
# from pyexpat import features, model  <-- Remove this line completely

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bati Bank Credit Risk API")

# Define the request model (this was missing!)
class PredictionRequest(BaseModel):
    # Replace these with your actual feature names and types
    Amount_abs: float
    fee: float
    tx_hour: int
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int
    # ... add all other features your model expects

    class Config:
        schema_extra = {
            "example": {
                "Amount_abs": 5000.0,
                "fee": 750.0,
                "tx_hour": 14,
                "ProductCategory": "financial_services",
                "ChannelId": "ChannelId_3",
                "PricingStrategy": 2
            }
        }

# Load your trained model globally (once at startup)
try:
    import joblib  # or pickle, or mlflow
    model = joblib.load("../models/best_model.pkl")  # adjust path
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model at startup: {e}")
    model = None  # will cause 503 if used

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Convert request to DataFrame (model expects this format)
        features_df = pd.DataFrame([request.dict()])
        
        # Make prediction
        probability = model.predict_proba(features_df)[0][1]
        
        return {"risk_probability": float(probability)}

    except ValidationError as e:
        # This won't actually trigger here — Pydantic handles it automatically
        raise HTTPException(status_code=400, detail=f"Invalid input: {e.errors()}")

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")