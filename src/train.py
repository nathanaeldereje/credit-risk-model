import sys
import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import get_transformation_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Loads the dataset with the target variable."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    return pd.read_csv(filepath)

def eval_metrics(y_actual, y_pred, y_pred_proba):
    """Calculates standard classification metrics."""
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred, zero_division=0)
    recall = recall_score(y_actual, y_pred, zero_division=0)
    f1 = f1_score(y_actual, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_actual, y_pred_proba)
    except ValueError:
        roc_auc = 0.0 # Handle edge cases where only one class is present
    return accuracy, precision, recall, f1, roc_auc

def train_and_evaluate(data_path):
    # 1. Load Data
    logging.info("Loading data...")
    df = load_data(data_path)
    
    # Define Features and Target
    target_col = 'is_high_risk'
    # Drop ID columns and target for X
    drop_cols = ['CustomerId', 'TransactionId', 'first_transaction_time', 'last_transaction_time', 
                 'rfm_cluster', target_col]
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    
    # 2. Train/Test Split
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Get Preprocessing Pipeline (from Task 3 logic)
    # We pass X_train so the function can identify num/cat columns
    preprocessor = get_transformation_pipeline(X_train)
    
    # 4. Define Models and Hyperparameters
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "params": {
                "classifier__C": [0.1, 1, 10],
                "classifier__solver": ['liblinear', 'lbfgs']
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "params": {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5]
            }
        }
    }
    
    # 5. MLflow Experiment Setup
    mlflow.set_experiment("Credit_Risk_Model_Experiment")
    
    best_overall_model = None
    best_overall_auc = -1
    
    for model_name, config in models.items():
        with mlflow.start_run(run_name=model_name):
            logging.info(f"Training {model_name}...")
            
            # Create Full Pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', config["model"])
            ])
            
            # Hyperparameter Tuning
            grid_search = GridSearchCV(
                pipeline, 
                config["params"], 
                cv=3, 
                scoring='roc_auc', 
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            if hasattr(best_model, "predict_proba"):
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = [0] * len(y_test)
                
            # Evaluation
            acc, prec, rec, f1, auc = eval_metrics(y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            # Log Model Artifact
            # Signature allows MLflow to know expected input format
            signature = mlflow.models.infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            
            logging.info(f"{model_name} Results - AUC: {auc:.4f}, F1: {f1:.4f}")
            
            # Track Global Best
            if auc > best_overall_auc:
                best_overall_auc = auc
                best_overall_model = best_model
                
    logging.info(f"Training Complete. Best Overall AUC: {best_overall_auc:.4f}")
    return best_overall_model

if __name__ == "__main__":
    DATA_PATH = 'data/processed/customer_features_with_target.csv'
    train_and_evaluate(DATA_PATH)