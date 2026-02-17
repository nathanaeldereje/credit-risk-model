import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and time feature extraction."""
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['tx_hour'] = df['TransactionStartTime'].dt.hour
    df['tx_day'] = df['TransactionStartTime'].dt.day
    df['tx_month'] = df['TransactionStartTime'].dt.month
    df['tx_year'] = df['TransactionStartTime'].dt.year
    
    df['Amount_abs'] = df['Amount'].abs()
    df['value_diff'] = df['Value'] - df['Amount_abs']
    df['is_negative_amount'] = (df['Amount'] < 0).astype(int)
    return df

def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transaction data to customer level."""
    aggs = {
        'TransactionId': 'count',
        'Amount_abs': ['sum', 'mean', 'std'],
        'value_diff': ['mean'],
        'is_negative_amount': 'sum',
        'tx_hour': 'mean',
        'tx_day': 'mean',
        'TransactionStartTime': ['max', 'min']
    }
    customer_df = df.groupby('CustomerId').agg(aggs)
    customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
    
    rename_map = {
        'TransactionId_count': 'total_transactions',
        'Amount_abs_sum': 'total_amount',
        'Amount_abs_mean': 'avg_amount',
        'Amount_abs_std': 'std_amount',
        'value_diff_mean': 'avg_fee_paid',
        'is_negative_amount_sum': 'total_refunds_count',
        'TransactionStartTime_max': 'last_transaction_time',
        'TransactionStartTime_min': 'first_transaction_time'
    }
    customer_df.rename(columns=rename_map, inplace=True)
    customer_df['std_amount'] = customer_df['std_amount'].fillna(0)
    return customer_df.reset_index()

def add_categorical_modes(df_trans: pd.DataFrame, df_cust: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Adds mode category per customer."""
    for col in cat_cols:
        if col in df_trans.columns:
            mode_series = df_trans.groupby('CustomerId')[col].agg(lambda x: x.mode().iloc[0])
            df_cust = df_cust.merge(mode_series, on='CustomerId', how='left')
    return df_cust

def get_transformation_pipeline(df: pd.DataFrame, cat_cols: List[str], target_col: str) -> ColumnTransformer:
    """Builds the Scikit-Learn transformation pipeline."""
    numeric_features = [col for col in df.select_dtypes(include=np.number).columns 
                        if col not in ['CustomerId', target_col] and col not in cat_cols]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, cat_cols)
    ], verbose_feature_names_out=False)