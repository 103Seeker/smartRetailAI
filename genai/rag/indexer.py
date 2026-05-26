import os, uuid, pandas as pd
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()

openai_client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)

search_client = SearchClient(
    endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    index_name='retail-knowledge',
    credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
)

df = pd.read_csv('data/raw/product_catalog.csv')

def get_embedding(text):
    resp = openai_client.embeddings.create(
        input=text,
        model='text-embedding-ada-002'   # your embedding deployment name
    )
    return resp.data[0].embedding

docs = []
for _, row in df.iterrows():
    text = str(row.to_dict())
    docs.append({
        'id': str(uuid.uuid4()),
        'content': text,
        'source': 'product_catalog',
        'content_vector': get_embedding(text)
    })

search_client.upload_documents(documents=docs)
print(f'Uploaded {len(docs)} documents to Azure AI Search.')
