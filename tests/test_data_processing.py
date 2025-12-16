import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import preprocess_transactions, aggregate_customer_features

@pytest.fixture
def sample_transaction_data():
    """Creates a small dummy dataframe for testing."""
    data = {
        'TransactionId': ['T1', 'T2', 'T3'],
        'BatchId': ['B1', 'B1', 'B2'],
        'AccountId': ['A1', 'A1', 'A2'],
        'SubscriptionId': ['S1', 'S1', 'S2'],
        'CustomerId': ['C1', 'C1', 'C2'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX'],
        'CountryCode': [256, 256, 256],
        'ProviderId': ['P1', 'P1', 'P2'],
        'ProductId': ['Prod1', 'Prod2', 'Prod1'],
        'ProductCategory': ['Cat1', 'Cat1', 'Cat2'],
        'ChannelId': ['Ch1', 'Ch1', 'Ch2'],
        'Amount': [1000, -500, 2000],
        'Value': [1000, 500, 2100],
        'TransactionStartTime': ['2023-01-01T10:00:00Z', '2023-01-02T11:00:00Z', '2023-01-03T12:00:00Z'],
        'PricingStrategy': ['PS1', 'PS1', 'PS2'],
        'FraudResult': [0, 0, 0]
    }
    return pd.DataFrame(data)

def test_preprocess_transactions_columns(sample_transaction_data):
    """Test if preprocessing adds the expected time and interaction columns."""
    df_processed = preprocess_transactions(sample_transaction_data)
    
    expected_cols = ['tx_hour', 'tx_day', 'tx_month', 'tx_year', 'Amount_abs', 'value_diff', 'is_negative_amount']
    for col in expected_cols:
        assert col in df_processed.columns, f"Column {col} missing from processed data"
        
    # Check logic specific: Value Diff for 3rd row (2100 Value - 2000 Amount = 100)
    assert df_processed.loc[2, 'value_diff'] == 100

def test_aggregate_customer_features_shape(sample_transaction_data):
    """Test if aggregation returns one row per customer."""
    df_processed = preprocess_transactions(sample_transaction_data)
    df_agg = aggregate_customer_features(df_processed)
    
    # We have 2 unique customers (C1, C2)
    assert len(df_agg) == 2
    
    # Check if index is CustomerId
    assert df_agg.index.name == 'CustomerId'
    
    # Check aggregation logic (C1 had 2 transactions)
    # Note: access via index 'C1'
    assert df_agg.loc['C1', 'total_transactions'] == 2