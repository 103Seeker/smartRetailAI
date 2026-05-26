# Smart Retail AI Platform

Multi-Agent AI Platform for retail demand forecasting, product Q&A, and anomaly detection.
Built as part of the Capgemini Left Shift Program 2026 — Data & AI (T5).

---

## Tech Stack
- **Backend:** FastAPI, SQLAlchemy, SQLite
- **ML:** scikit-learn (Random Forest), pandas, numpy
- **GenAI:** LangChain, Azure OpenAI (GPT-4o), FAISS
- **Data Engineering:** Azure Data Factory, Databricks (PySpark)
- **Visualization:** Power BI
- **Deployment:** Azure Web App, Docker, GitHub Actions

---

## Project Structure
```
smart-retail-ai/
├── backend/        FastAPI app — 4 REST APIs + DB
├── ml/             ML pipeline — preprocess, train, evaluate
├── genai/          3 agents + RAG + orchestrator
├── data_engineering/ ADF config, PySpark, SQL
├── data/           raw / staged / curated data layers
├── deployment/     Dockerfile, CI/CD, Key Vault setup
├── docs/           Architecture, technical docs, reflection
└── powerbi/        Power BI dashboard
```

---

## Quick Start
```bash
pip install -r requirements.txt
cp .env.example .env          # add your Azure OpenAI key
python ml/preprocess.py
python ml/feature_engineering.py
python ml/train_classifier.py
python genai/rag/embeddings.py
python genai/rag/vector_store.py
uvicorn backend.main:app --reload
```
API docs: http://localhost:8000/docs

---

## Run Tests
```bash
pytest backend/tests/ -v
```
