from fastapi import APIRouter, HTTPException
from backend.models.schemas import AgentRequest, AgentResponse
from backend.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-load orchestrator
_orchestrator = None


def get_orchestrator():
    """Load agent orchestrator once at startup."""
    global _orchestrator
    if _orchestrator is None:
        try:
            from genai.orchestrator import AgentOrchestrator
            _orchestrator = AgentOrchestrator()
            logger.info("Agent orchestrator loaded.")
        except Exception as e:
            logger.error(f"Orchestrator load failed: {e}")
            raise HTTPException(status_code=503, detail="Agent orchestrator not ready.")
    return _orchestrator


@router.post("/agent", response_model=AgentResponse)
def agent_chat(payload: AgentRequest):
    """
    API 4 — Agent Interaction.
    Routes the user message to the correct agent (forecast, product, anomaly)
    via the orchestrator and returns a natural language response.
    """
    try:
        orchestrator = get_orchestrator()
        agent_name, response = orchestrator.route(message=payload.message)

        logger.info(f"Message routed to: {agent_name}")

        return AgentResponse(
            agent_used=agent_name,
            response=response,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
