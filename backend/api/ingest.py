from fastapi import APIRouter, HTTPException
from backend.models.schemas import IngestRequest, IngestResponse
from backend.database import insert_sales_records
from backend.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_sales(payload: IngestRequest):
    """
    API 1 - Data Ingestion.
    Accepts a list of Walmart sales records and saves them to MongoDB.
    """
    try:
        records = [
            {
                "store":        row.store,
                "date":         row.date,
                "weekly_sales": row.weekly_sales,
                "holiday_flag": row.holiday_flag,
                "temperature":  row.temperature,
                "fuel_price":   row.fuel_price,
                "cpi":          row.cpi,
                "unemployment": row.unemployment,
            }
            for row in payload.records
        ]

        saved = insert_sales_records(records)
        logger.info(f"Ingested {saved} sales records.")

        return IngestResponse(
            message="Records ingested successfully.",
            records_saved=saved,
        )

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")