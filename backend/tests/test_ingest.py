from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from backend.main import app

client = TestClient(app)


def test_ingest_success():
    """Valid records should return 200 with correct count."""
    payload = {
        "records": [
            {"date": "2024-01-01", "product_id": "P001", "sales": 100.0, "store_id": "S01"},
            {"date": "2024-01-02", "product_id": "P002", "sales": 200.0, "store_id": "S01"},
        ]
    }
    with patch("backend.api.ingest.get_db") as mock_db:
        mock_session = MagicMock()
        mock_db.return_value = iter([mock_session])
        response = client.post("/api/ingest", json=payload)

    assert response.status_code == 200
    assert response.json()["records_saved"] == 2


def test_ingest_empty_records():
    """Empty records list should still return 200 with 0 saved."""
    payload = {"records": []}
    with patch("backend.api.ingest.get_db") as mock_db:
        mock_session = MagicMock()
        mock_db.return_value = iter([mock_session])
        response = client.post("/api/ingest", json=payload)

    assert response.status_code == 200
    assert response.json()["records_saved"] == 0


def test_ingest_missing_fields():
    """Missing required fields should return 422 validation error."""
    payload = {"records": [{"date": "2024-01-01"}]}  # missing product_id and sales
    response = client.post("/api/ingest", json=payload)
    assert response.status_code == 422
