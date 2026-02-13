import os
import pytest
import pandas as pd
import numpy as np
from src.credit_risk.processing import preprocess_transactions, aggregate_customer_features
from src.credit_risk.features import calculate_rfm, define_high_risk_label
from src.credit_risk.config import AppConfig
# NEW IMPORTS FOR COVERAGE
from src.credit_risk.utils import setup_logging, load_csv, save_csv
from src.credit_risk.model import calculate_metrics

@pytest.fixture
def sample_transaction_data():
    """Fixture to provide dummy transaction data."""
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': ['2023-01-01T10:00:00Z', '2023-01-02T12:00:00Z', '2023-01-01T09:00:00Z'],
        'Amount': [1000, -200, 500],
        'Value': [1050, 200, 510]
    })

# --- EXISTING TESTS ---

def test_preprocess_transactions(sample_transaction_data):
    processed = preprocess_transactions(sample_transaction_data)
    assert 'tx_hour' in processed.columns
    assert 'Amount_abs' in processed.columns
    assert processed.loc[1, 'is_negative_amount'] == 1
    assert processed.loc[0, 'value_diff'] == 50 

def test_aggregation_counts(sample_transaction_data):
    processed = preprocess_transactions(sample_transaction_data)
    aggregated = aggregate_customer_features(processed)
    assert len(aggregated) == 2
    assert aggregated.set_index('CustomerId').loc['C1', 'total_transactions'] == 2

def test_rfm_recency():
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
    df = pd.DataFrame({
        'rfm_cluster': [0, 1, 2],
        'Monetary': [100, 5000, 10000]
    })
    labeled = define_high_risk_label(df)
    assert labeled.loc[0, 'is_high_risk'] == 1
    assert labeled.loc[1, 'is_high_risk'] == 0

def test_config_defaults():
    cfg = AppConfig()
    assert cfg.K_CLUSTERS == 3
    assert "is_high_risk" == cfg.TARGET_COL

def test_active_days_logic():
    df = pd.DataFrame({
        'CustomerId': ['C1'],
        'first_transaction_time': [pd.Timestamp('2023-01-01')],
        'last_transaction_time': [pd.Timestamp('2023-01-05')],
        'total_transactions': [2],
        'total_amount': [100]
    })
    rfm = calculate_rfm(df)
    assert rfm.loc[0, 'active_days'] == 5

# --- NEW TESTS TO FIX COVERAGE GAPS ---

# TEST 7: Utilities - Logging (Fixes utils.py 0%)
def test_setup_logging():
    logger = setup_logging()
    assert logger.name == "src.credit_risk.utils"

# TEST 8: Utilities - CSV IO (Fixes utils.py 0%)
def test_csv_utils(tmp_path):
    # tmp_path is a built-in pytest fixture for temporary directories
    d = tmp_path / "data"
    d.mkdir()
    file = d / "test.csv"
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    
    # Test Save
    save_csv(df, str(file))
    assert os.path.exists(str(file))
    
    # Test Load
    loaded_df = load_csv(str(file))
    assert loaded_df.shape == (2, 2)
    assert list(loaded_df.columns) == ['a', 'b']

# TEST 9: Model Metrics (Fixes model.py 0%)
def test_calculate_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])  # One False Negative
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] == 0.75

# TEST 10: Edge Case - Standard Deviation for 1 transaction (Fixes processing.py gap)
def test_std_deviation_single_transaction():
    df = pd.DataFrame({
        'TransactionId': ['T1'],
        'CustomerId': ['C1'],
        'TransactionStartTime': ['2023-01-01T10:00:00Z'],
        'Amount': [1000],
        'Value': [1000]
    })
    processed = preprocess_transactions(df)
    aggregated = aggregate_customer_features(processed)
    # std_amount should be 0, not NaN
    assert aggregated.loc[0, 'std_amount'] == 0