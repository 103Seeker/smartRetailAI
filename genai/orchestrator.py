from genai.agents.forecast_agent import ForecastAgent
from genai.agents.product_agent import ProductAgent
from genai.agents.anomaly_agent import AnomalyAgent
from backend.logger import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Multi-Agent Orchestrator.
    Classifies incoming user message by intent and routes
    to the correct specialized agent.

    Routing logic:
    - Keywords about sales, forecast, demand, prediction  → ForecastAgent
    - Keywords about products, stock, catalog, inventory  → ProductAgent
    - Keywords about anomaly, unusual, spike, drop, alert → AnomalyAgent
    - Default fallback                                    → ForecastAgent
    """

    def __init__(self):
        self.forecast_agent = ForecastAgent()
        self.product_agent  = ProductAgent()
        self.anomaly_agent  = AnomalyAgent()
        logger.info("AgentOrchestrator initialized with 3 agents.")

    def _classify_intent(self, message: str) -> str:
        """
        Keyword-based intent classifier.
        Returns agent name as string.
        """
        msg = message.lower()

        forecast_keywords = [
            "forecast", "predict", "sales", "demand",
            "next week", "next month", "trend", "revenue",
            "will sell", "how much", "projection"
        ]

        product_keywords = [
            "product", "stock", "inventory", "catalog",
            "available", "price", "category", "sku",
            "how many", "in stock", "variant", "item"
        ]

        anomaly_keywords = [
            "anomaly", "unusual", "spike", "drop", "alert",
            "abnormal", "sudden", "unexpected", "fell", "jumped",
            "why did", "what happened"
        ]

        forecast_score = sum(1 for kw in forecast_keywords if kw in msg)
        product_score  = sum(1 for kw in product_keywords  if kw in msg)
        anomaly_score  = sum(1 for kw in anomaly_keywords  if kw in msg)

        scores = {
            "forecast_agent": forecast_score,
            "product_agent":  product_score,
            "anomaly_agent":  anomaly_score,
        }

        best = max(scores, key=scores.get)

        # If all scores are 0, default to forecast agent
        if scores[best] == 0:
            best = "forecast_agent"

        logger.info(f"Intent scores: {scores} → routed to: {best}")
        return best

    def route(self, message: str) -> tuple[str, str]:
        """
        Route message to the correct agent.
        Returns (agent_name, response_text).
        """
        agent_name = self._classify_intent(message)

        if agent_name == "forecast_agent":
            response = self.forecast_agent.run(message)

        elif agent_name == "product_agent":
            response = self.product_agent.run(message)

        elif agent_name == "anomaly_agent":
            response = self.anomaly_agent.run(message)

        else:
            response = self.forecast_agent.run(message)
            agent_name = "forecast_agent"

        return agent_name, response