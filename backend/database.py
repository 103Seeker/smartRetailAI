from pymongo import MongoClient
from datetime import datetime
from backend.config import DATABASE_URL
from backend.logger import get_logger

logger = get_logger(__name__)

# MongoDB connection
try:
    client = MongoClient(DATABASE_URL, serverSelectionTimeoutMS=5000)
    db     = client["smart_retail_db"]

    # Collections (MongoDB equivalent of SQL tables)
    sales_collection       = db["sales_records"]
    predictions_collection = db["prediction_logs"]
    logger.info("MongoDB client initialized.")
except Exception as e:
    logger.warning(f"MongoDB client init failed: {e}")
    client                 = None
    db                     = None
    sales_collection       = None
    predictions_collection = None


def insert_sales_records(records: list) -> int:
    """Insert a list of sales record dicts into MongoDB."""
    if not records:
        return 0
    if sales_collection is None:
        logger.warning("MongoDB unavailable — skipping insert_sales_records.")
        return 0
    try:
        for record in records:
            record["created_at"] = datetime.utcnow()
        result = sales_collection.insert_many(records)
        logger.info(f"Inserted {len(result.inserted_ids)} sales records.")
        return len(result.inserted_ids)
    except Exception as e:
        logger.warning(f"insert_sales_records failed: {e}")
        return 0


def log_prediction(input_data: dict, prediction: int, confidence: float):
    """Log a prediction result to MongoDB."""
    if predictions_collection is None:
        logger.warning("MongoDB unavailable — skipping log_prediction.")
        return
    try:
        doc = {
            "input_data": input_data,
            "prediction": prediction,
            "confidence": confidence,
            "created_at": datetime.utcnow(),
        }
        predictions_collection.insert_one(doc)
        logger.info(f"Prediction logged: {prediction} ({confidence:.2%})")
    except Exception as e:
        logger.warning(f"log_prediction failed: {e}")


def get_db():
    """
    Returns the MongoDB database object.
    Kept so FastAPI routes don't need to change.
    """
    return db


def init_db():
    """Create indexes for faster queries."""
    if sales_collection is None:
        logger.warning("MongoDB unavailable — skipping index creation.")
        return
    try:
        sales_collection.create_index("date")
        sales_collection.create_index("product_id")
        predictions_collection.create_index("created_at")
        logger.info("MongoDB indexes initialized.")
    except Exception as e:
        logger.warning(f"MongoDB index creation failed: {e}")


if __name__ == "__main__":
    init_db()