import os
import sys

from src.credit_risk.config import AppConfig
from src.credit_risk.utils import setup_logging, load_csv, save_csv
from src.credit_risk.processing import (
    preprocess_transactions, aggregate_customer_features, add_categorical_modes
)
from src.credit_risk.features import (
    calculate_rfm, perform_clustering, define_high_risk_label
)

def main():
    logger = setup_logging()
    cfg = AppConfig()

    # 1. Load and Trans-level processing
    df_raw = load_csv(cfg.RAW_DATA_PATH)
    df_trans = preprocess_transactions(df_raw)

    # 2. Aggregation
    df_cust = aggregate_customer_features(df_trans)
    df_cust = add_categorical_modes(df_trans, df_cust, cfg.CAT_COLS)

    # 3. Feature/Target Engineering
    df_cust = calculate_rfm(df_cust)
    df_cust = perform_clustering(df_cust, cfg.K_CLUSTERS, cfg.RANDOM_STATE, cfg.N_INIT)
    df_final = define_high_risk_label(df_cust)

    # 4. Save
    save_csv(df_final, cfg.FINAL_DATA_PATH)
    logger.info(f"Preprocessing successful. Final shape: {df_final.shape}")
    logger.info(f"Target Distribution: {df_final[cfg.TARGET_COL].mean():.2%}")

if __name__ == "__main__":
    main()