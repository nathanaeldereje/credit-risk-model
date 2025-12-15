import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Loads the customer features."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    return pd.read_csv(filepath)

def calculate_rfm(df):
    """
    Calculates Recency, Frequency, and Monetary metrics.
    """
    logging.info("Calculating RFM metrics...")
    
    # Ensure datetime
    df['last_transaction_time'] = pd.to_datetime(df['last_transaction_time'])
    
    # 1. Recency
    # Snapshot date = last date in data + 1 day
    snapshot_date = df['last_transaction_time'].max() + pd.Timedelta(days=1)
    df['Recency'] = (snapshot_date - df['last_transaction_time']).dt.days
    
    # 2. Frequency
    df['Frequency'] = df['total_transactions']
    
    # 3. Monetary
    df['Monetary'] = df['total_amount']
    
    logging.info("RFM calculation complete.")
    return df

def perform_clustering(df):
    """
    Performs K-Means clustering to segment customers.
    """
    logging.info("Preparing data for clustering...")
    
    # Select RFM columns
    rfm_data = df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Handle Skewness (Log Transform)
    # Add constant to prevent log(0)
    rfm_log = np.log1p(rfm_data)
    
    # Scale Data (StandardScaler)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # K-Means Clustering (k=3)
    logging.info("Running K-Means (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['rfm_cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return df

def define_high_risk_label(df):
    """
    Analyzes clusters and assigns 'is_high_risk' label.
    Risk Logic: High Recency + Low Frequency + Low Monetary = High Risk.
    """
    logging.info("Analyzing clusters to define risk...")
    df['first_transaction_time'] = pd.to_datetime(df['first_transaction_time'])
    df['last_transaction_time'] = pd.to_datetime(df['last_transaction_time'])
    
    df['active_days'] = (df['last_transaction_time'] - df['first_transaction_time']).dt.days + 1
    # Calculate mean RFM per cluster
    cluster_summary = df.groupby('rfm_cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    print("\n--- Cluster Profiles ---")
    print(cluster_summary)
    
    # Scoring to find the "worst" cluster
    # We want the cluster with Highest Recency and Lowest Freq/Monetary
    
    # Rank clusters (1=Best, 3=Worst logic)
    # Recency: Higher is worse
    r_rank = cluster_summary['Recency'].rank(ascending=False) 
    # Freq/Monetary: Lower is worse
    f_rank = cluster_summary['Frequency'].rank(ascending=True)
    m_rank = cluster_summary['Monetary'].rank(ascending=True)
    
    # Combine ranks (higher score = better customer)
    # We want the lowest score
    # Actually, simpler logic:
    # High Risk Cluster = The one with Max Recency OR Min Monetary
    
    # Let's pick the cluster with the LOWEST 'Monetary' value as the anchor for High Risk
    # (Usually correlates perfectly with low frequency and high recency in financial data)
    high_risk_cluster_id = cluster_summary['Monetary'].idxmin()
    
    logging.info(f"Identified Cluster {high_risk_cluster_id} as High Risk (Lowest Monetary Value).")
    
    # Create Label
    df['is_high_risk'] = (df['rfm_cluster'] == high_risk_cluster_id).astype(int)
    # After creating is_high_risk
    print("\n--- Risk Distribution ---")
    print(f"High-risk customers: {df['is_high_risk'].sum()} ({df['is_high_risk'].mean():.2%}) out of {len(df)} total customers")
    # Validation
    risk_counts = df['is_high_risk'].value_counts(normalize=True)
    print("\n--- Risk Distribution ---")
    print(risk_counts)
    
    return df

if __name__ == "__main__":
    INPUT_PATH = 'data/processed/customer_features.csv'
    OUTPUT_PATH = 'data/processed/customer_features_with_target.csv'
    
    # 1. Load
    df = load_data(INPUT_PATH)
    
    # 2. RFM
    df = calculate_rfm(df)
    
    # 3. Cluster
    df = perform_clustering(df)
    
    # 4. Label
    df = define_high_risk_label(df)
    
    # 5. Save
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Labeled data saved to {OUTPUT_PATH}")