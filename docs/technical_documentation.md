# Smart Retail AI Platform — Technical Documentation

## 1. Project Overview
An end-to-end Multi-Agent AI Platform built for retail demand forecasting, product Q&A, and sales anomaly detection. Built with FastAPI, LangChain, scikit-learn, and deployed on Azure.

---

## 2. Architecture

```
User / Evaluator
      │
      ▼
FastAPI Backend (4 REST APIs)
      │
      ├── /api/ingest   → SQLite Database (SalesRecord table)
      ├── /api/predict  → ML Model (sales_classifier.pkl)
      ├── /api/search   → RAG Retriever → FAISS Vector Store
      └── /api/agent    → Agent Orchestrator
                              ├── ForecastAgent  → Azure OpenAI
                              ├── ProductAgent   → RAG + Azure OpenAI
                              └── AnomalyAgent   → Z-score + Azure OpenAI

Data Pipeline
      Raw CSV → Azure Data Factory → Staged Parquet
             → Databricks PySpark  → Curated Delta Table
             → Power BI Dashboard
```

---

## 3. Data Flow

| Layer    | Storage             | Tool                  |
|----------|---------------------|-----------------------|
| Raw      | CSV                 | Azure Data Factory    |
| Staged   | Parquet             | Databricks PySpark    |
| Curated  | Delta / Parquet     | Spark SQL             |
| Serving  | Power BI            | DirectQuery           |

---

## 4. Machine Learning Model

- **Type:** Binary Classification
- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Target:** `high_sales` — 1 if daily sales above median, 0 otherwise
- **Features:** day_of_week, month, quarter, is_weekend, day_of_month, rolling_avg_7, lag_1, lag_7
- **Persistence:** `ml/saved_models/sales_classifier.pkl`
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score

---

## 5. GenAI & Multi-Agent System

| Agent          | Trigger Keywords                        | Technology         |
|----------------|-----------------------------------------|--------------------|
| ForecastAgent  | sales, forecast, demand, predict        | Azure OpenAI GPT-4o|
| ProductAgent   | product, stock, inventory, catalog      | RAG + FAISS        |
| AnomalyAgent   | anomaly, spike, drop, unusual, why did  | Z-score + GPT-4o   |

**RAG Pipeline:**
1. Product catalog rows embedded with `text-embedding-ada-002`
2. Stored in FAISS flat L2 index
3. Query embedded at runtime → top-3 chunks retrieved → passed to LLM as context

---

## 6. REST APIs

| Method | Route          | Description                        |
|--------|----------------|------------------------------------|
| POST   | /api/ingest    | Ingest sales records into database |
| POST   | /api/predict   | Predict high/low sales day         |
| POST   | /api/search    | RAG search over product catalog    |
| POST   | /api/agent     | Multi-agent natural language chat  |
| GET    | /health        | Health check                       |

Full interactive docs available at `/docs` (Swagger UI).

---

## 7. Azure Components

| Component         | Purpose                                  |
|-------------------|------------------------------------------|
| Azure OpenAI      | LLM for all agents + embeddings          |
| Azure Web App     | Hosts FastAPI backend                    |
| Azure AI Search   | Production vector store (RAG)            |
| Azure Key Vault   | Secure secret management                 |
| Azure Data Factory| Raw data ingestion pipeline              |
| Azure Databricks  | PySpark data transformation              |

---

## 8. Security
- All secrets stored in Azure Key Vault
- Environment variables injected at runtime via Key Vault references
- No secrets committed to GitHub
- `.env` excluded via `.gitignore`

---

## 9. How to Run Locally

```bash
# 1. Clone repo
git clone <repo-url>
cd smart-retail-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Fill in your Azure OpenAI key and endpoint in .env

# 4. Train the ML model
python ml/preprocess.py
python ml/feature_engineering.py
python ml/train_classifier.py
python ml/evaluate.py

# 5. Build RAG index
python genai/rag/embeddings.py
python genai/rag/vector_store.py

# 6. Start API server
uvicorn backend.main:app --reload

# 7. Open Swagger UI
# http://localhost:8000/docs

# 8. Run tests
pytest backend/tests/ -v
```
