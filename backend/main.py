from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import ingest, predict, search, agent
from backend.database import init_db
from backend.logger import get_logger
from backend.config import APP_HOST, APP_PORT

logger = get_logger(__name__)

# --- App Init ---
app = FastAPI(
    title="Smart Retail AI Platform",
    description="Multi-Agent AI Platform for demand forecasting, product Q&A, and anomaly detection.",
    version="1.0.0",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register Routes ---
app.include_router(ingest.router, prefix="/api", tags=["Data Ingestion"])
app.include_router(predict.router, prefix="/api", tags=["ML Prediction"])
app.include_router(search.router,  prefix="/api", tags=["Document Search"])
app.include_router(agent.router,   prefix="/api", tags=["Agent Interaction"])


# --- Startup ---
@app.on_event("startup")
def on_startup():
    logger.info("Starting Smart Retail AI Platform...")
    init_db()
    logger.info("Server ready.")


# --- Health Check ---
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Smart Retail AI Platform"}


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
