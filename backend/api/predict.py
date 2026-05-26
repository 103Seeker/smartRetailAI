import pickle
import numpy as np
from fastapi import APIRouter, HTTPException
from backend.models.schemas import PredictRequest, PredictResponse
from backend.database import log_prediction
from backend.config import MODEL_PATH
from backend.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Load model once at startup - not on every request
_model = None
_feature_cols = None


def get_model():
    """Lazy-load the model bundle from disk."""
    global _model, _feature_cols
    if _model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            _model        = bundle["model"]
            _feature_cols = bundle["feature_cols"]
            logger.info(f"Model loaded from {MODEL_PATH}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {MODEL_PATH}. Run train_classifier.py first.")
            raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return _model, _feature_cols


@router.post("/predict", response_model=PredictResponse)
def predict_sales(payload: PredictRequest):
    """
    API 2 - ML Prediction.
    Takes sales features and returns High/Low sales prediction with confidence.
    """
    try:
        model, feature_cols = get_model()

        # Build feature dict from request
        input_dict = {
            "day_of_week":   payload.day_of_week,
            "month":         payload.month,
            "quarter":       payload.quarter,
            "week_of_year":  payload.week_of_year,
            "holiday_flag":  payload.holiday_flag,
            "temperature":   payload.temperature,
            "fuel_price":    payload.fuel_price,
            "cpi":           payload.cpi,
            "unemployment":  payload.unemployment,
            "rolling_avg_7": payload.rolling_avg_7,
            "lag_1":         payload.lag_1,
            "lag_4":         payload.lag_4,
        }

        # Keep only features the model was trained on, in correct order
        features = np.array([[input_dict[f] for f in feature_cols]])

        # Predict
        prediction  = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][prediction])
        label       = "High Sales Day" if prediction == 1 else "Low Sales Day"

        # Log prediction to MongoDB
        log_prediction(
            input_data=input_dict,
            prediction=prediction,
            confidence=round(probability, 4),
        )

        logger.info(f"Prediction: {label} ({probability:.2%} confidence)")

        return PredictResponse(
            prediction=prediction,
            label=label,
            confidence=round(probability, 4),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")