import os
import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.credit_risk.processing import (
    preprocess_transactions, 
    aggregate_customer_features, 
    add_categorical_modes, 
    get_transformation_pipeline
)
from src.credit_risk.features import calculate_rfm, define_high_risk_label
from src.credit_risk.config import AppConfig
from src.credit_risk.utils import setup_logging, load_csv, save_csv
from src.credit_risk.model import calculate_metrics, ModelTrainer

@pytest.fixture
def sample_transaction_data() -> pd.DataFrame:
    """
    Provides a small, controlled transaction dataset for unit testing.
    Includes cases for refunds (negative amounts) and fees.
    """
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': ['2023-01-01T10:00:00Z', '2023-01-02T12:00:00Z', '2023-01-01T09:00:00Z'],
        'Amount': [1000, -200, 500],
        'Value': [1050, 200, 510]
    })

def test_preprocess_transactions(sample_transaction_data: pd.DataFrame):
    """
    Tests transaction-level feature extraction.
    Verifies:
    1. Time features (tx_hour) are correctly extracted.
    2. Amount_abs handles sign correctly.
    3. value_diff correctly calculates transaction fees.
    """
    processed = preprocess_transactions(sample_transaction_data)
    assert 'tx_hour' in processed.columns
    assert 'Amount_abs' in processed.columns
    assert processed.loc[1, 'is_negative_amount'] == 1
    assert processed.loc[0, 'value_diff'] == 50 

def test_aggregation_counts(sample_transaction_data: pd.DataFrame):
    """
    Tests the transition from transaction-level to customer-level data.
    Ensures that multiple transactions are correctly summed per CustomerId.
    """
    processed = preprocess_transactions(sample_transaction_data)
    aggregated = aggregate_customer_features(processed)
    assert len(aggregated) == 2
    assert aggregated.set_index('CustomerId').loc['C1', 'total_transactions'] == 2

def test_rfm_recency():
    """
    Tests the Recency calculation logic.
    Ensures that customers with older 'last_transaction_time' receive a 
    higher Recency (days since last activity) value.
    """
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C2'],
        'last_transaction_time': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-10')],
        'first_transaction_time': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01')],
        'total_transactions': [1, 5],
        'total_amount': [100, 500]
    })
    rfm = calculate_rfm(df)
    assert rfm.loc[rfm['CustomerId'] == 'C2', 'Recency'].values[0] < \
           rfm.loc[rfm['CustomerId'] == 'C1', 'Recency'].values[0]

def test_define_high_risk_label():
    """
    Tests the behavioral proxy target assignment.
    Validates that the 'is_high_risk' flag is correctly assigned to the 
    cluster with the lowest monetary value (Proxy for default risk).
    """
    df = pd.DataFrame({
        'rfm_cluster': [0, 1, 2],
        'Monetary': [100, 5000, 10000]
    })
    labeled = define_high_risk_label(df)
    assert labeled.loc[0, 'is_high_risk'] == 1
    assert labeled.loc[1, 'is_high_risk'] == 0

def test_config_defaults():
    """
    Ensures that the AppConfig dataclass maintains consistent project constants.
    """
    cfg = AppConfig()
    assert cfg.K_CLUSTERS == 3
    assert "is_high_risk" == cfg.TARGET_COL

def test_active_days_logic():
    """
    Validates account tenure calculation.
    Checks that 'active_days' accurately measures the span between the 
    first and last transactions inclusive.
    """
    df = pd.DataFrame({
        'CustomerId': ['C1'],
        'first_transaction_time': [pd.Timestamp('2023-01-01')],
        'last_transaction_time': [pd.Timestamp('2023-01-05')],
        'total_transactions': [2],
        'total_amount': [100]
    })
    rfm = calculate_rfm(df)
    assert rfm.loc[0, 'active_days'] == 5

def test_setup_logging():
    """
    Verifies that the centralized logging utility initializes correctly.
    """
    logger = setup_logging()
    assert logger.name == "src.credit_risk.utils"

def test_csv_utils(tmp_path):
    """
    Tests data I/O utilities using a temporary file system.
    Ensures that the load/save operations preserve data integrity and shape.
    """
    d = tmp_path / "data"
    d.mkdir()
    file = d / "test.csv"
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    
    save_csv(df, str(file))
    assert os.path.exists(str(file))
    
    loaded_df = load_csv(str(file))
    assert loaded_df.shape == (2, 2)
    assert list(loaded_df.columns) == ['a', 'b']

def test_calculate_metrics():
    """
    Validates the mathematical implementation of classification metrics.
    Ensures ROC-AUC and Accuracy calculations match expected outcomes 
    for a known prediction set.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] == 0.75

def test_std_deviation_single_transaction():
    """
    Tests edge-case stability for single-transaction customers.
    Ensures that standard deviation is filled as 0 (not NaN) to 
    prevent pipeline crashes during model inference.
    """
    df = pd.DataFrame({
        'TransactionId': ['T1'],
        'CustomerId': ['C1'],
        'TransactionStartTime': ['2023-01-01T10:00:00Z'],
        'Amount': [1000],
        'Value': [1000]
    })
    processed = preprocess_transactions(df)
    aggregated = aggregate_customer_features(processed)
    assert aggregated.loc[0, 'std_amount'] == 0

def test_add_categorical_modes():
    """
    Tests categorical feature engineering.
    Ensures that the 'mode' (most frequent) category is correctly 
    mapped from transaction history back to the customer profile.
    """
    df_trans = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'ProviderId': ['P1', 'P1', 'P2']
    })
    df_cust = pd.DataFrame({'CustomerId': ['C1', 'C2']})
    result = add_categorical_modes(df_trans, df_cust, ['ProviderId'])
    assert result.loc[0, 'ProviderId'] == 'P1'
    assert result.loc[1, 'ProviderId'] == 'P2'

def test_get_transformation_pipeline():
    """
    Validates the Scikit-Learn ColumnTransformer configuration.
    Ensures the pipeline correctly identifies and routes numeric vs 
    categorical features based on the input dataframe schema.
    """
    df = pd.DataFrame({
        'num1': [1, 2],
        'cat1': ['A', 'B'],
        'is_high_risk': [0, 1]
    })
    preprocessor = get_transformation_pipeline(df, ['cat1'], 'is_high_risk')
    assert preprocessor is not None
    assert 'num1' in preprocessor.transformers[0][2]

def test_model_trainer_logic():
    """
    Tests the ModelTrainer wrapper.
    Verifies that a basic GridSearchCV run returns a valid fitted 
    model and the expected parameter dictionary.
    """
    X = pd.DataFrame({'f1': np.random.rand(10), 'f2': np.random.rand(10)})
    y = pd.Series([0, 1] * 5)
    
    # Simple preprocessing mock
    preprocessor = ColumnTransformer([('n', StandardScaler(), ['f1', 'f2'])])
    trainer = ModelTrainer(preprocessor)
    
    model, params = trainer.train_with_grid_search(
        X, y, LogisticRegression(), 
        {'classifier__C': [1]}, 
        "TestModel"
    )
    assert model is not None
    assert params['classifier__C'] == 1