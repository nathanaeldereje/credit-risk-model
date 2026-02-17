import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates RFM metrics for clustering."""
    df = df.copy()
    df['last_transaction_time'] = pd.to_datetime(df['last_transaction_time'])
    df['active_days'] = (df['last_transaction_time'] - df['first_transaction_time']).dt.days + 1
    snapshot_date = df['last_transaction_time'].max() + pd.Timedelta(days=1)
    
    df['Recency'] = (snapshot_date - df['last_transaction_time']).dt.days
    df['Frequency'] = df['total_transactions']
    df['Monetary'] = df['total_amount']
    return df

def perform_clustering(df: pd.DataFrame, n_clusters: int, random_state: int, n_init: int) -> pd.DataFrame:
    """Segments customers using K-Means."""
    rfm_data = df[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_log = np.log1p(rfm_data)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    df['rfm_cluster'] = kmeans.fit_predict(rfm_scaled)
    return df

def define_high_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns the binary proxy target based on cluster characteristics."""
    # Find cluster with lowest monetary value
    cluster_summary = df.groupby('rfm_cluster')['Monetary'].mean()
    high_risk_cluster_id = cluster_summary.idxmin()
    
    df['is_high_risk'] = (df['rfm_cluster'] == high_risk_cluster_id).astype(int)
    return df