import os
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from backend.config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY
)
class RetailRetriever:
    def __init__(self, index_path=None):
        self.openai = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.search_client = SearchClient(      # ← renamed to search_client
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name='retail-knowledge',
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )

    def _embed(self, text):
        resp = self.openai.embeddings.create(
            input=text,
            model='text-embedding-ada-002'
        )
        return resp.data[0].embedding

    def search(self, query, top_k=5):           # ← method stays as search
        vector = self._embed(query)
        vq = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top_k,
            fields='content_vector'
        )
        results = self.search_client.search(    # ← uses search_client now
            search_text=None,
            vector_queries=[vq]
        )
        return [{'content': r['content'], 'source': r['source']} for r in results]