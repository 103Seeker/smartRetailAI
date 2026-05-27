import pandas as pd
import numpy as np
import logging
import os

from preprocess import preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_CANDIDATES = [
    "data/raw/sales_transactions.csv",
    "data/raw/Walmart.csv",
    "data/raw/walmart.csv",
]
STAGED_PATH = "data/staged/cleaned_sales.parquet"
CURATED_PATH = "data/curated/analytics_ready.parquet"


def find_raw_path() -> str:
    """Return the first supported raw CSV path that exists."""
    for path in RAW_CANDIDATES:
        if os.path.exists(path):
            return path
    expected = ", ".join(RAW_CANDIDATES)
    raise FileNotFoundError(f"Raw sales CSV not found. Put your Kaggle file in one of: {expected}")


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Load the raw Kaggle sales CSV."""
    raw_path = path or find_raw_path()
    logger.info(f"Loading raw data from {raw_path}")
    return pd.read_csv(raw_path)


def save_staged(df: pd.DataFrame, path: str = STAGED_PATH) -> None:
    """Save cleaned staged data as parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Staged data saved to {path}")


def save_curated(df: pd.DataFrame, path: str = CURATED_PATH) -> None:
    """Save feature-engineered data as parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Curated data saved to {path}")


def run_feature_engineering() -> pd.DataFrame:
    """Run full feature engineering: raw CSV -> staged parquet -> curated parquet."""
    df = load_raw()
    df = preprocess(df)
    save_staged(df)
    save_curated(df)
    return df


if __name__ == "__main__":
    run_feature_engineering()
