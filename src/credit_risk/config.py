# src/credit_risk/config.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class AppConfig:
    """Project-wide configuration and paths."""
    # Paths
    RAW_DATA_PATH: str = "data/raw/data.csv"
    PROCESSED_DATA_PATH: str = "data/processed/customer_features.csv"
    FINAL_DATA_PATH: str = "data/processed/customer_features_with_target.csv"
    
    # Column Definitions
    CAT_COLS: List[str] = field(default_factory=lambda: [
        'ProviderId', 'ProductCategory', 'ChannelId', 'PricingStrategy'
    ])
    TARGET_COL: str = "is_high_risk"
    
    # RFM & Clustering Settings
    K_CLUSTERS: int = 3
    RANDOM_STATE: int = 42
    N_INIT: int = 10



 
    # Training Settings
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Columns to drop before training (IDs and non-feature dates)
    COLS_TO_DROP: List[str] = field(default_factory=lambda: [
        'CustomerId', 'first_transaction_time', 'last_transaction_time', 
        'rfm_cluster'
    ])

    # Model Grids
    LR_PARAMS: Dict = field(default_factory=lambda: {
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ['liblinear', 'lbfgs']
    })
    
    RF_PARAMS: Dict = field(default_factory=lambda: {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [5, 10, None]
    })


    # For shap
    REPORTS_DIR: str = "reports/figures"