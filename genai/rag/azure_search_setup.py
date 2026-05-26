from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SimpleField, SearchableField
)
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
load_dotenv()

endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
key      = os.getenv('AZURE_SEARCH_KEY')

client = SearchIndexClient(endpoint, AzureKeyCredential(key))

fields = [
    SimpleField(name='id', type=SearchFieldDataType.String, key=True),
    SearchableField(name='content', type=SearchFieldDataType.String),
    SimpleField(name='source', type=SearchFieldDataType.String),
    SearchField(
        name='content_vector',
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name='myHnswProfile'
    ),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name='myHnsw')],
    profiles=[VectorSearchProfile(name='myHnswProfile', algorithm_configuration_name='myHnsw')]
)

index = SearchIndex(name='retail-knowledge', fields=fields, vector_search=vector_search)
client.create_or_update_index(index)
print('Index created successfully.')

