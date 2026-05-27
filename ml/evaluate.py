import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging

sys.path.insert(0, os.path.abspath('.'))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH   = "ml/saved_models/sales_classifier.pkl"
CURATED_PATH = "data/curated/analytics_ready.parquet"
RAW_CANDIDATES = [
    "data/raw/sales_transactions.csv",
    "data/raw/Walmart.csv",
    "data/raw/walmart.csv",
]
TARGET_COL   = "high_sales"


def find_raw_path() -> str | None:
    """Return the first supported raw CSV path that exists."""
    for path in RAW_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def load_or_create_curated_data() -> pd.DataFrame:
    """Load curated data, or rebuild it from the raw sales CSV if available."""
    if os.path.exists(CURATED_PATH):
        logger.info(f"Loading data from {CURATED_PATH}")
        return pd.read_parquet(CURATED_PATH)

    raw_path = find_raw_path()
    if raw_path is None:
        expected = ", ".join(RAW_CANDIDATES)
        raise FileNotFoundError(
            f"Missing {CURATED_PATH}. To evaluate the model, add the raw dataset at "
            f"one of: {expected}. Then rerun this script."
        )

    logger.info(f"{CURATED_PATH} not found. Rebuilding from {raw_path}")
    from preprocess import preprocess

    df = pd.read_csv(raw_path)
    df = preprocess(df)
    os.makedirs(os.path.dirname(CURATED_PATH), exist_ok=True)
    df.to_parquet(CURATED_PATH, index=False)
    logger.info(f"Curated data saved to {CURATED_PATH}")
    return df


def evaluate_model() -> dict:
    """Load saved model and evaluate on test set."""

    # Load model bundle
    logger.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Load curated data
    df = load_or_create_curated_data()

    # Keep only features model was trained on
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df[TARGET_COL]

    # Same split as training — random_state=42 ensures same test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
    }

    # Feature importances
    importances = dict(zip(
        available,
        [round(x, 4) for x in model.feature_importances_]
    ))
    importances = dict(sorted(
        importances.items(), key=lambda x: x[1], reverse=True
    ))

    # Print results
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"{k.upper():<12}: {v}")
    logger.info("\nCLASSIFICATION REPORT:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Low Sales', 'High Sales'])}")
    logger.info("CONFUSION MATRIX:")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
    logger.info("\nFEATURE IMPORTANCES (top to bottom):")
    for feat, score in importances.items():
        logger.info(f"  {feat:<20}: {score}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
