from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app

client = TestClient(app)


def test_search_returns_results():
    """Valid query should return list of results."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        "Product A: High demand in Q4.",
        "Product B: Low stock alert.",
    ]

    with patch("backend.api.search.get_retriever", return_value=mock_retriever):
        response = client.post("/api/search", json={"query": "top products", "top_k": 2})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert response.json()["query"] == "top products"


def test_search_missing_query():
    """Missing query field should return 422."""
    response = client.post("/api/search", json={})
    assert response.status_code == 422
