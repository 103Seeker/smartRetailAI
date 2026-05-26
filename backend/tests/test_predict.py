from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from backend.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "day_of_week":   2,
    "month":         6,
    "quarter":       2,
    "is_weekend":    0,
    "day_of_month":  15,
    "rolling_avg_7": 320.5,
    "lag_1":         310.0,
    "lag_7":         295.0,
}


def test_predict_high_sales():
    """Should return prediction 1 (High Sales) with valid input."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    feature_cols = list(VALID_PAYLOAD.keys())

    with patch("backend.api.predict.get_model", return_value=(mock_model, feature_cols)):
        with patch("backend.api.predict.get_db") as mock_db:
            mock_db.return_value = iter([MagicMock()])
            response = client.post("/api/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["label"] == "High Sales Day"
    assert response.json()["confidence"] == 0.8


def test_predict_low_sales():
    """Should return prediction 0 (Low Sales) with valid input."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])
    feature_cols = list(VALID_PAYLOAD.keys())

    with patch("backend.api.predict.get_model", return_value=(mock_model, feature_cols)):
        with patch("backend.api.predict.get_db") as mock_db:
            mock_db.return_value = iter([MagicMock()])
            response = client.post("/api/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediction"] == 0
    assert response.json()["label"] == "Low Sales Day"


def test_predict_missing_fields():
    """Missing required fields should return 422."""
    response = client.post("/api/predict", json={"day_of_week": 2})
    assert response.status_code == 422
