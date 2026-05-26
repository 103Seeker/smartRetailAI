from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app

client = TestClient(app)


def test_agent_forecast_routing():
    """Message about sales should route to forecast agent."""
    mock_orch = MagicMock()
    mock_orch.route.return_value = ("forecast_agent", "Next week sales are predicted to be high.")

    with patch("backend.api.agent.get_orchestrator", return_value=mock_orch):
        response = client.post("/api/agent", json={"message": "What will sales look like next week?"})

    assert response.status_code == 200
    assert response.json()["agent_used"] == "forecast_agent"
    assert "sales" in response.json()["response"].lower()


def test_agent_missing_message():
    """Missing message should return 422."""
    response = client.post("/api/agent", json={})
    assert response.status_code == 422


def test_agent_empty_message():
    """Empty string message should still process."""
    mock_orch = MagicMock()
    mock_orch.route.return_value = ("forecast_agent", "Please provide more details.")

    with patch("backend.api.agent.get_orchestrator", return_value=mock_orch):
        response = client.post("/api/agent", json={"message": ""})

    assert response.status_code == 200
