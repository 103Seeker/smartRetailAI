import pandas as pd
import numpy as np
from openai import AzureOpenAI
from backend.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    CURATED_PATH,
)
from backend.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATH = "genai/prompts/anomaly_prompt.txt"


class AnomalyAgent:
    """
    Agent 3 — Anomaly Agent.
    Detects unusual sales using Z-score on the curated data,
    then explains them in plain business language via the LLM.
    No separate ML model needed — Z-score logic lives here.
    """

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.deployment = AZURE_OPENAI_DEPLOYMENT

        with open(PROMPT_PATH, "r") as f:
            self.system_prompt = f.read().strip()

        logger.info("AnomalyAgent initialized.")

    def detect_anomalies(self, threshold: float = 2.0) -> str:
        """
        Load curated sales data, compute Z-scores,
        and return a summary of anomalous rows as a string.
        """
        try:
            df = pd.read_parquet(CURATED_PATH)

            sales_col = "weekly_sales" if "weekly_sales" in df.columns else "sales"

            if sales_col not in df.columns:
                return "No sales or weekly_sales column found in curated data."

            mean   = df[sales_col].mean()
            std    = df[sales_col].std()
            df["z_score"] = (df[sales_col] - mean) / std

            anomalies = df[df["z_score"].abs() > threshold].copy()

            if anomalies.empty:
                return "No significant anomalies detected in recent sales data."

            # Build a readable summary of top 5 anomalies
            anomalies = anomalies.sort_values("z_score", key=abs, ascending=False).head(5)
            lines = []
            for _, row in anomalies.iterrows():
                direction = "spike" if row["z_score"] > 0 else "drop"
                lines.append(
                    f"Date: {row.get('date', 'N/A')} | "
                    f"Product: {row.get('product_id', 'N/A')} | "
                    f"Sales: {row[sales_col]:.1f} | "
                    f"Z-score: {row['z_score']:.2f} ({direction})"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return f"Could not load sales data for anomaly analysis: {str(e)}"

    def run(self, message: str) -> str:
        """
        Detect anomalies from data, then ask the LLM to explain them.
        """
        # Step 1 — Get anomaly data
        anomaly_summary = self.detect_anomalies()

        # Step 2 — LLM explains the anomalies in business language
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Anomaly data detected:\n{anomaly_summary}"},
            {"role": "user",   "content": message},
        ]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=350,
            temperature=0.3,
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"AnomalyAgent response generated ({len(answer)} chars)")
        return answer
        
