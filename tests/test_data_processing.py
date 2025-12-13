import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(raw_path: str) -> pd.DataFrame:
    try:
        if not Path(raw_path).exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        
        df = pd.read_csv(raw_path)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
        logging.info(f"Successfully loaded {len(df)} rows from {raw_path}")
        return df
    
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise