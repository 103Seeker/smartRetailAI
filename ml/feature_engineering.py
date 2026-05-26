import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGED_PATH = "data/staged/cleaned_sales.parquet"
CURATED_PATH = "data/curated/analytics_ready.parquet"


def load_staged(path: str = STAGED_PATH) -> pd.DataFrame:
    """Load cleaned staged data."""
    logger.info(f"Loading staged data from {path}")
    return pd.read_parquet(path)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date-based features from the date column."""
    logger.info("Adding date features...")
    df["day_of_week"] = df["date"].dt.dayofweek       # 0=Monday, 6=Sunday
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["date"].dt.day
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling average and lag sales features."""
    logger.info("Adding lag and rolling features...")
    df = df.sort_values("date")

    # 7-day rolling average sales
    df["rolling_avg_7"] = (
        df["sales"].rolling(window=7, min_periods=1).mean().round(2)
    )

    # Lag features: sales from 1 day ago and 7 days ago
    df["lag_1"] = df["sales"].shift(1).fillna(0)
    df["lag_7"] = df["sales"].shift(7).fillna(0)

    return df


def add_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target:
    1 = High sales day (above median), 0 = Low sales day.
    This is what the Random Forest model will predict.
    """
    logger.info("Adding target label (high_sales)...")
    median_sales = df["sales"].median()
    df["high_sales"] = (df["sales"] > median_sales).astype(int)
    logger.info(f"Median sales threshold: {median_sales:.2f}")
    logger.info(f"High sales days: {df['high_sales'].sum()} / {len(df)}")
    return df


def save_curated(df: pd.DataFrame, path: str = CURATED_PATH) -> None:
    """Save feature-engineered data as parquet."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Curated data saved to {path}")


def run_feature_engineering() -> pd.DataFrame:
    """Run full feature engineering: staged -> curated."""
    df = load_staged()
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_target_label(df)
    save_curated(df)
    return df


if __name__ == "__main__":
    run_feature_engineering()
