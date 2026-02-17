import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple, Any
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .utils import setup_logging

logger = setup_logging()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculates all relevant classification metrics for finance."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }

class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def train_with_grid_search(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        base_model: Any, 
        param_grid: Dict,
        model_name: str
    ) -> Any:
        """Runs GridSearchCV within a pipeline."""
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', base_model)
        ])
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        logger.info(f"Starting Grid Search for {model_name}...")
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_