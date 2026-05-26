from pydantic import BaseModel, Field
from typing import Optional, List


class SalesRow(BaseModel):
    store:        int   = Field(..., example=1)
    date:         str   = Field(..., example="2024-01-15")
    weekly_sales: float = Field(..., example=52000.0)
    holiday_flag: int   = Field(..., example=0)
    temperature:  float = Field(..., example=38.5)
    fuel_price:   float = Field(..., example=3.1)
    cpi:          float = Field(..., example=210.5)
    unemployment: float = Field(..., example=7.2)


class IngestRequest(BaseModel):
    records: List[SalesRow]


class IngestResponse(BaseModel):
    message:       str
    records_saved: int


class PredictRequest(BaseModel):
    day_of_week:   int
    month:         int
    quarter:       int
    week_of_year:  int
    holiday_flag:  int
    temperature:   float
    fuel_price:    float
    cpi:           float
    unemployment:  float
    rolling_avg_7: float
    lag_1:         float
    lag_4:         float


class PredictResponse(BaseModel):
    prediction: int
    label:      str
    confidence: float


class SearchResult(BaseModel):
    content: str
    source:  str


class SearchRequest(BaseModel):
    query: str = Field(..., example="Show me top selling products this week")
    top_k: int = Field(3, ge=1, le=10)


class SearchResponse(BaseModel):
    query:   str
    results: List[SearchResult]


class AgentRequest(BaseModel):
    message:         str = Field(..., example="Why did sales drop on Monday?")
    conversation_id: Optional[str] = None


class AgentResponse(BaseModel):
    agent_used: str
    response:   str