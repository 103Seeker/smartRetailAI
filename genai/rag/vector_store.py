import os
import pickle
import numpy as np
import faiss
from backend.logger import get_logger

logger = get_logger(__name__)

EMBEDDINGS_PATH = "genai/rag/embeddings.pkl"
INDEX_PATH      = "genai/rag/faiss_index"


def build_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    """
    Build a FAISS flat L2 index from embedding vectors.
    FlatL2 = exact nearest neighbour search — simple and good enough for small catalogs.
    """
    vectors = np.array(embeddings, dtype="float32")
    dim     = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatL2, path: str = INDEX_PATH):
    """Save FAISS index to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    faiss.write_index(index, path)
    logger.info(f"FAISS index saved to {path}")


def load_index(path: str = INDEX_PATH) -> faiss.IndexFlatL2:
    """Load FAISS index from disk."""
    index = faiss.read_index(path)
    logger.info(f"FAISS index loaded: {index.ntotal} vectors")
    return index


def build_and_save():
    """Load embeddings pickle → build FAISS index → save."""
    with open(EMBEDDINGS_PATH, "rb") as f:
        bundle = pickle.load(f)

    embeddings = bundle["embeddings"]
    index      = build_index(embeddings)
    save_index(index)
    return index


if __name__ == "__main__":
    build_and_save()