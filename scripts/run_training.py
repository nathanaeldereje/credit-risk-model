import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow

from src.credit_risk.config import AppConfig
from src.credit_risk.utils import setup_logging, load_csv
from src.credit_risk.processing import get_transformation_pipeline
from src.credit_risk.model import ModelTrainer, calculate_metrics

def main():
    logger = setup_logging()
    cfg = AppConfig()
    
    # 1. Load Data
    df = load_csv(cfg.FINAL_DATA_PATH)
    
    # 2. Prepare X and y
    X = df.drop(columns=[c for c in cfg.COLS_TO_DROP + [cfg.TARGET_COL] if c in df.columns])
    y = df[cfg.TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    
    # 3. Setup Pipeline & Trainer
    # Note: We pass CAT_COLS from config
    preprocessor = get_transformation_pipeline(X_train, cfg.CAT_COLS, cfg.TARGET_COL)
    trainer = ModelTrainer(preprocessor)
    
    # 4. MLflow Experiment
    mlflow.set_experiment("Credit_Risk_Production_Training")
    
    models_to_run = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight='balanced'), cfg.LR_PARAMS),
        ("RandomForest", RandomForestClassifier(class_weight='balanced'), cfg.RF_PARAMS)
    ]
    
    for name, model_obj, params in models_to_run:
        with mlflow.start_run(run_name=name):
            # Train
            best_model, best_params = trainer.train_with_grid_search(X_train, y_train, model_obj, params, name)
            
            # Predict
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            
            # Log
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model", signature=mlflow.models.infer_signature(X_test, y_pred))
            
            logger.info(f"Model: {name} | ROC-AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()