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
TARGET_COL   = "high_sales"


def evaluate_model() -> dict:
    """Load saved model and evaluate on test set."""

    # Load model bundle
    logger.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Load curated data
    logger.info(f"Loading data from {CURATED_PATH}")
    df = pd.read_parquet(CURATED_PATH)

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