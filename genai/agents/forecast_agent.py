import os
from openai import AzureOpenAI
from backend.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)
from backend.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATH = "genai/prompts/forecast_prompt.txt"


class ForecastAgent:
    """
    Agent 1 — Forecast Agent.
    Handles questions about sales trends, demand predictions,
    and future sales outlook using the ML model output as context.
    """

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.deployment = AZURE_OPENAI_DEPLOYMENT

        # Load system prompt from file
        with open(PROMPT_PATH, "r") as f:
            self.system_prompt = f.read().strip()

        logger.info("ForecastAgent initialized.")

    def run(self, message: str, context: str = "") -> str:
        """
        Run the forecast agent on a user message.
        Optionally pass ML prediction output as context.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # If ML context is available, inject it before the user message
        if context:
            messages.append({
                "role": "system",
                "content": f"Current ML prediction context:\n{context}"
            })

        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=300,
            temperature=0.3,   # low temperature = consistent, factual answers
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"ForecastAgent response generated ({len(answer)} chars)")
        return answer