import pytest
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.credit_risk.explainability import RiskExplainer

@pytest.fixture
def toy_pipeline() -> tuple:
    """
    Creates a simple, fitted pipeline for explainability unit testing.
    Provides:
        1. A Pipeline with a preprocessor and a RandomForestClassifier.
        2. A sample feature DataFrame (X).
    """
    X = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10)
    })
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['feature1', 'feature2'])
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline, X

def test_explainer_initialization(toy_pipeline: tuple):
    """
    Tests the RiskExplainer constructor.
    Ensures that classifier steps and feature names are correctly 
    mapped during initialization.
    """
    pipeline, X = toy_pipeline
    explainer = RiskExplainer(pipeline, X.columns.tolist())
    assert explainer.classifier is not None
    assert len(explainer.transformed_feature_names) == 2

def test_generate_shap_values(toy_pipeline: tuple):
    """
    Validates SHAP value generation for binary classification.
    Checks that the output matrix matches the sample size and feature count.
    """
    pipeline, X = toy_pipeline
    explainer = RiskExplainer(pipeline, X.columns.tolist())
    shap_values, X_trans = explainer.generate_shap_values(X.head(2))
    
    # Expect 2 samples, 2 features
    assert shap_values.shape == (2, 2)
    assert X_trans.shape == (2, 2)

def test_plot_generation(toy_pipeline: tuple, tmp_path):
    """
    Tests the persistence of explainability visualizations.
    Ensures that global summary and local waterfall plots are 
    successfully rendered to the file system.
    """
    pipeline, X = toy_pipeline
    explainer = RiskExplainer(pipeline, X.columns.tolist())
    shap_values, X_trans = explainer.generate_shap_values(X.head(2))
    
    # Test global importance plot
    global_path = tmp_path / "global.png"
    explainer.plot_global_importance(shap_values, X_trans, str(global_path))
    assert os.path.exists(global_path)
    
    # Test local waterfall explanation plot
    local_path = tmp_path / "local.png"
    explainer.plot_local_explanation(shap_values, X_trans, 0, str(local_path))
    assert os.path.exists(local_path)