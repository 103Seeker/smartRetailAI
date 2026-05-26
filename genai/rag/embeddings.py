import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import pickle
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

from backend.logger import get_logger
logger = get_logger(__name__)

CATALOG_PATH    = "data/raw/product_catalog.csv"
EMBEDDINGS_PATH = "genai/rag/embeddings.pkl"

# Read directly from environment to avoid any config caching issues
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_DEPLOYMENT     = "text-embedding-ada-002"

logger.info(f"Endpoint loaded: {AZURE_OPENAI_ENDPOINT}")
logger.info(f"Key loaded: {'YES' if AZURE_OPENAI_API_KEY else 'NO - CHECK .env FILE'}")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01",
)


def load_catalog(path: str = CATALOG_PATH) -> list:
    logger.info(f"Loading catalog from {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    chunks = []
    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        chunks.append(text)
    logger.info(f"Created {len(chunks)} text chunks from catalog")
    return chunks


def generate_embeddings(chunks: list) -> list:
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model=EMBEDDING_DEPLOYMENT
        )
        embeddings.append(response.data[0].embedding)
        if (i + 1) % 10 == 0:
            logger.info(f"  Embedded {i + 1}/{len(chunks)} chunks")
    logger.info("Embedding generation complete.")
    return embeddings


def save_embeddings(chunks: list, embeddings: list, path: str = EMBEDDINGS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {"chunks": chunks, "embeddings": embeddings}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Embeddings saved to {path}")


def run_embeddings():
    chunks     = load_catalog()
    embeddings = generate_embeddings(chunks)
    save_embeddings(chunks, embeddings)
    return chunks, embeddings


if __name__ == "__main__":
    run_embeddings()