# Reflection Note — Smart Retail AI Platform

## Challenges Faced

**1. Multi-agent routing accuracy**
The keyword-based orchestrator works well for clear queries but struggles with ambiguous messages like "tell me about last Monday." I addressed this by scoring all keyword categories and routing to the highest match, with a fallback to the ForecastAgent as the most commonly needed agent.

**2. RAG grounding**
Initially the ProductAgent would sometimes hallucinate product details not in the catalog. Fixing the temperature to 0.2 and explicitly instructing the model to say "I could not find that" when context is insufficient resolved most cases.

**3. Feature alignment between training and inference**
The ML model must receive features in the exact same order and names as during training. I solved this by saving the feature column list inside the model pickle bundle alongside the model itself, so inference always uses the correct features.

**4. Azure deployment cold start**
The first API call after deployment had a delay because models load lazily. Added startup logging to make this visible and documented it as expected behavior for evaluators.

---

## Learnings

- RAG is significantly easier to maintain than fine-tuning for domain-specific Q&A — updating the catalog CSV and rebuilding the index takes minutes
- Saving model metadata (feature names, thresholds) alongside the model weights prevents silent failures in production
- FastAPI's automatic Swagger UI at `/docs` is invaluable for demos — no separate frontend needed
- Z-score anomaly detection inside the agent (rather than a separate ML model) keeps the architecture simpler without losing explainability

---

## What I Would Improve With More Time

1. **Replace keyword routing with LLM-based intent classification** — more accurate for complex or multi-intent queries
2. **Add real-time streaming** with Azure Event Hub so anomalies trigger instant Power BI alerts
3. **A/B test the ML model** by deploying a second model (XGBoost) in shadow mode and comparing predictions
4. **Add conversation memory** to the agents so follow-up questions maintain context across multiple turns
5. **Automate retraining** — trigger a new model training run via Azure ML pipeline whenever new sales data exceeds a volume threshold
