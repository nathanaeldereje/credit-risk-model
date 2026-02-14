import mlflow.sklearn
import joblib
import os
from src.credit_risk.config import AppConfig

def export():
    cfg = AppConfig()
    os.makedirs("models", exist_ok=True)
    
    # 1. Fetch best model from local MLflow
    experiment = mlflow.get_experiment_by_name("Credit_Risk_Production_Training")
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["metrics.roc_auc DESC"])
    best_run_id = runs.iloc[0].run_id
    
    print(f"Exporting model from run: {best_run_id}")
    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    
    # 2. Save as a standalone joblib file
    joblib.dump(model, "models/credit_risk_model.joblib")
    print("✅ Model saved to models/credit_risk_model.joblib")

if __name__ == "__main__":
    export()