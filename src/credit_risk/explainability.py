import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from .utils import setup_logging

logger = setup_logging()

class RiskExplainer:
    def __init__(self, model_pipeline, feature_names: list):
        self.pipeline = model_pipeline
        self.classifier = model_pipeline.named_steps['classifier']
        self.preprocessor = model_pipeline.named_steps['preprocessor']
        self.feature_names = feature_names
        self.transformed_feature_names = self.preprocessor.get_feature_names_out()
        
        # Initialize the Explainer
        self.explainer = shap.TreeExplainer(self.classifier)

    def generate_shap_values(self, X: pd.DataFrame):
        """Calculates SHAP values and handles the multi-class output matrix."""
        logger.info("Transforming data for SHAP analysis...")
        X_transformed = self.preprocessor.transform(X)
        
        logger.info("Calculating SHAP values...")
        shap_values = self.explainer.shap_values(X_transformed)
        
        # FIX: Handling SHAP's multi-output format
        # If the output is a list [class0, class1], pick class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        # If the output is a 3D array (samples, features, classes), pick class 1
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
            
        return shap_values, X_transformed

    def plot_global_importance(self, shap_values, X_transformed, save_path: str):
        """Creates a Summary Plot of feature importance."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X_transformed, 
            feature_names=self.transformed_feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Global SHAP plot saved to {save_path}")

    def plot_local_explanation(self, shap_values, X_transformed, index: int, save_path: str):
        """Explains a single prediction (Waterfall plot)."""
        # Get the base value (expected value) for Class 1
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
            base_val = base_val[1]

        # FIX: Ensure we only pass the vector for a single class
        current_shap_values = shap_values[index]
        if len(current_shap_values.shape) == 2:
            current_shap_values = current_shap_values[:, 1]

        plt.figure(figsize=(12, 8))
        
        # Create an Explanation object for a single row
        explanation = shap.Explanation(
            values=current_shap_values,
            base_values=base_val,
            data=X_transformed[index],
            feature_names=self.transformed_feature_names
        )
        
        shap.plots.waterfall(explanation, show=False)
        plt.gcf().set_size_inches(12, 8) # Force resize
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Local SHAP plot for index {index} saved to {save_path}")