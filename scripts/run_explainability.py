import mlflow
import numpy as np
import pandas as pd
import os
from src.credit_risk.config import AppConfig
from src.credit_risk.utils import setup_logging, load_csv
from src.credit_risk.explainability import RiskExplainer

def main():
    logger = setup_logging()
    cfg = AppConfig()
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)

    # 1. Load Data
    df = load_csv(cfg.FINAL_DATA_PATH)
    X = df.drop(columns=[c for c in cfg.COLS_TO_DROP + [cfg.TARGET_COL] if c in df.columns])
    
    # 2. Load Best Model from MLflow
    logger.info("Fetching best model from MLflow...")
    experiment = mlflow.get_experiment_by_name("Credit_Risk_Production_Training")
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["metrics.roc_auc DESC"])
    best_run_id = runs.iloc[0].run_id
    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    # 3. Initialize Explainer
    explainer = RiskExplainer(model, X.columns.tolist())

    # 4. Generate SHAP values for a sample (or full set)
    # We take a sample of 100 for speed, or more if needed
    X_sample = X.sample(100, random_state=cfg.RANDOM_STATE)
    shap_vals, X_trans = explainer.generate_shap_values(X_sample)

    # 5. Global Plot
    explainer.plot_global_importance(
        shap_vals, X_trans, f"{cfg.REPORTS_DIR}/shap_summary.png"
    )

    # 6. Local Plot (Find a High Risk customer to explain)
    # Get probabilities for Class 1
    y_probs = model.predict_proba(X_sample)[:, 1]
    
    # Find the index of the highest risk customer in our sample
    target_idx = np.argmax(y_probs)
    
    logger.info(f"Explaining customer at sample index {target_idx} with risk prob {y_probs[target_idx]:.4f}")
    
    explainer.plot_local_explanation(
        shap_vals, X_trans, target_idx, f"{cfg.REPORTS_DIR}/shap_local_high_risk.png"
    )

if __name__ == "__main__":
    main()