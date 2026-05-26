from fastapi import APIRouter, HTTPException
from backend.models.schemas import SearchRequest, SearchResponse
from backend.config import VECTOR_STORE
from backend.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-load retriever
_retriever = None


def get_retriever():
    """Load FAISS retriever once at startup."""
    global _retriever
    if _retriever is None:
        try:
            from genai.rag.retriever import RetailRetriever
            _retriever = RetailRetriever(index_path=VECTOR_STORE)
            logger.info("RAG retriever loaded.")
        except Exception as e:
            logger.error(f"Retriever load failed: {e}")
            raise HTTPException(status_code=503, detail="RAG retriever not ready.")
    return _retriever


@router.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest):
    """
    API 3 — Document Search (RAG).
    Takes a natural language query and returns relevant product/policy results
    retrieved from the FAISS vector store.
    """
    try:
        retriever = get_retriever()
        results   = retriever.search(query=payload.query, top_k=payload.top_k)

        logger.info(f"Search query: '{payload.query}' → {len(results)} results")

        return SearchResponse(
            query=payload.query,
            results=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
