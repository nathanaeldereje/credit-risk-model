import pandas as pd
import numpy as np
import logging
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------
# STAGE 1: Data Loading & Preprocessing
# ---------------------------------------------------------

def load_data(filepath):
    """Loads raw CSV data."""
    logging.info(f"Loading raw data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    
    df = pd.read_csv(filepath)
    logging.info(f"Raw data loaded. Shape: {df.shape}")
    return df

def preprocess_transactions(df):
    """Clean and extract features at transaction level."""
    logging.info("Preprocessing transaction data...")
    
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extract Time Features
    df['tx_hour'] = df['TransactionStartTime'].dt.hour
    df['tx_day'] = df['TransactionStartTime'].dt.day
    df['tx_month'] = df['TransactionStartTime'].dt.month
    df['tx_year'] = df['TransactionStartTime'].dt.year
    
    # Interaction Features
    df['Amount_abs'] = df['Amount'].abs()
    df['value_diff'] = df['Value'] - df['Amount_abs']
    df['is_negative_amount'] = (df['Amount'] < 0).astype(int)
    
    return df

# ---------------------------------------------------------
# STAGE 2: Aggregation (The RFM Base)
# ---------------------------------------------------------

def aggregate_customer_features(df):
    """Aggregates transactions to customer level."""
    logging.info("Aggregating data to Customer Level...")
    
    aggs = {
        'TransactionId': 'count',
        'Amount_abs': ['sum', 'mean', 'std'],
        'value_diff': ['mean'],
        'is_negative_amount': 'sum',
        'tx_hour': 'mean',
        'tx_day': 'mean',
        'TransactionStartTime': 'max'
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
        'TransactionStartTime_max': 'last_transaction_time'
    }
    customer_df.rename(columns=rename_map, inplace=True)
    customer_df['std_amount'] = customer_df['std_amount'].fillna(0)
    
    return customer_df

def add_categorical_modes(df_transactions, df_customers):
    """Adds mode (most frequent) category per customer."""
    logging.info("Calculating categorical modes...")
    cat_cols = ['ProviderId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    
    for col in cat_cols:
        if col in df_transactions.columns:
            mode_series = df_transactions.groupby('CustomerId')[col].agg(lambda x: x.mode().iloc[0])
            df_customers = df_customers.merge(mode_series, on='CustomerId', how='left')
            
    return df_customers

# ---------------------------------------------------------
# STAGE 3: Scaling & Encoding (The Scikit-Learn Pipeline)
# ---------------------------------------------------------

def get_transformation_pipeline(df):
    """
    Builds the Sklearn Pipeline for Scaling and Encoding.
    Returns the preprocessor object.
    """
    logging.info("Building Transformation Pipeline...")
    
    # 1. Identify Columns
    # Numeric: All number columns except CustomerId (index) and the target
    numeric_features = [col for col in df.select_dtypes(include=np.number).columns 
                        if col not in ['CustomerId', 'is_high_risk']]
    
    # Categorical: Object columns
    categorical_features = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                            if col not in ['CustomerId']]

    logging.info(f"Numeric features to scale: {len(numeric_features)}")
    logging.info(f"Categorical features to encode: {len(categorical_features)}")

    # 2. Define Transformers
    # Numeric: Impute Median -> Standardize
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute 'Missing' -> OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for easier DF conversion
    ])

    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False # Keeps column names clean in newer sklearn versions
    )
    
    return preprocessor

# ---------------------------------------------------------
# STAGE 4: WoE and IV (Weight of Evidence)
# ---------------------------------------------------------

def calculate_woe_iv(df, target_col, feature_cols):
    """
    Calculates WoE and Information Value.
    Only runs if the target column exists (Task 4 dependency).
    """
    if target_col not in df.columns:
        logging.warning(f"Target column '{target_col}' not found. Skipping WoE/IV calculation.")
        return df, None

    logging.info("Calculating WoE and IV...")
    
    # Initialize WoE Transformer
    # xverse handles binning for numericals and grouping for categoricals automatically
    clf = WOE()
    
    # Fit & Transform
    try:
        clf.fit(df[feature_cols], df[target_col])
        woe_df = clf.transform(df[feature_cols])
        iv_df = clf.iv_df
        logging.info("WoE calculation successful.")
        print("Top 5 Features by Information Value (IV):")
        print(iv_df.head(5))
        return woe_df, iv_df
    except Exception as e:
        logging.error(f"WoE calculation failed: {e}")
        return df, None

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def process_data(raw_filepath, output_filepath):
    # 1. Load & Preprocess
    df_raw = load_data(raw_filepath)
    df_trans = preprocess_transactions(df_raw)
    
    # 2. Aggregate
    df_cust = aggregate_customer_features(df_trans)
    df_cust = add_categorical_modes(df_trans, df_cust)
    
    # 3. SAVE INTERMEDIATE FILE (Critical for Task 4 - Clustering)
    # We need the raw monetary values for RFM, not the scaled ones.
    df_cust.to_csv(output_filepath, index=True)
    logging.info(f"Raw Aggregated Data saved to {output_filepath} (Ready for Task 4)")

    # -----------------------------------------------------
    # DEMONSTRATION OF TASK 3 REQUIREMENTS (Pipeline & WoE)
    # -----------------------------------------------------
    # Even though we need Task 4 to fully utilize these, we define and test them here.
    
    # A. Test Pipeline Construction
    try:
        pipeline = get_transformation_pipeline(df_cust)
        # Note: We don't fit_transform here yet because we want to split Train/Test first (Task 5)
        # But we prove it works:
        logging.info("Scikit-Learn Pipeline constructed successfully.")
    except Exception as e:
        logging.error(f"Pipeline construction failed: {e}")

    # B. Test WoE Functionality (Placeholder)
    # This will log a warning that 'is_high_risk' is missing, which is expected behavior for now.
    calculate_woe_iv(df_cust, target_col='is_high_risk', feature_cols=df_cust.columns.tolist())
    
    return df_cust

if __name__ == "__main__":
    RAW_PATH = 'data/raw/data.csv'  
    OUTPUT_PATH = 'data/processed/customer_features.csv'
    
    process_data(RAW_PATH, OUTPUT_PATH)