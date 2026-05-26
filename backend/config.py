import os
from dotenv import load_dotenv

load_dotenv()

# ── Key Vault support ──────────────────────────────────────────
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

def get_secret(name: str, fallback_env: str) -> str:
    kv_url = os.getenv('AZURE_KEY_VAULT_URL', '')
    if kv_url and os.getenv('APP_ENV') == 'production':
        try:
            cred   = DefaultAzureCredential()
            client = SecretClient(vault_url=kv_url, credential=cred)
            return client.get_secret(name).value
        except Exception:
            pass
    return os.getenv(fallback_env, '')

# ── Azure OpenAI ───────────────────────────────────────────────
AZURE_OPENAI_API_KEY     = get_secret('AzureOpenAIKey', 'AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT    = os.getenv('AZURE_OPENAI_ENDPOINT', '')
AZURE_OPENAI_DEPLOYMENT  = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')

# ── Azure AI Search ────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT', '')
AZURE_SEARCH_KEY      = get_secret('AzureSearchKey', 'AZURE_SEARCH_KEY')

# ── Azure Key Vault ────────────────────────────────────────────
AZURE_KEY_VAULT_URL = os.getenv('AZURE_KEY_VAULT_URL', '')

# ── Azure ML (skipped due to student quota — kept for future) ──
# AZURE_ML_ENDPOINT = os.getenv('AZURE_ML_ENDPOINT', '')
# AZURE_ML_API_KEY  = get_secret('AzureMLApiKey', 'AZURE_ML_API_KEY')

# ── Database ───────────────────────────────────────────────────
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./retail.db')

# ── Paths ──────────────────────────────────────────────────────
MODEL_PATH   = os.getenv('MODEL_PATH',   'ml/saved_models/sales_classifier.pkl')
CURATED_PATH = os.getenv('CURATED_PATH', 'data/curated/analytics_ready.parquet')
VECTOR_STORE = os.getenv('VECTOR_STORE', 'genai/rag/faiss_index')

# ── App ────────────────────────────────────────────────────────
APP_ENV  = os.getenv('APP_ENV',  'development')
APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('APP_PORT', 8000))