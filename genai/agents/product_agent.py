from openai import AzureOpenAI
from genai.rag.retriever import RetailRetriever
from backend.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)
from backend.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATH = "genai/prompts/product_prompt.txt"


class ProductAgent:
    """
    Agent 2 - Product Agent.
    Answers questions about products, stock, and catalog
    using RAG over the product catalog vector store.
    """

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.retriever  = RetailRetriever()

        with open(PROMPT_PATH, "r") as f:
            self.system_prompt = f.read().strip()

        logger.info("ProductAgent initialized.")

    def run(self, message: str) -> str:
        """
        Retrieve relevant catalog chunks for the query,
        then generate a grounded answer using the LLM.
        """
        # Step 1 - Retrieve top 3 relevant catalog entries
        retrieved_chunks = self.retriever.search(query=message, top_k=3)
        context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

        # Step 2 - Generate answer grounded in retrieved context
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Product catalog context:\n{context}"},
            {"role": "user",   "content": message},
        ]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=300,
            temperature=0.2,   # very low - stay grounded in the catalog
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"ProductAgent response generated ({len(answer)} chars)")
        return answer