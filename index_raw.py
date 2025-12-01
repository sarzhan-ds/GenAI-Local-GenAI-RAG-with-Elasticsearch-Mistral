import os
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

ES_HOST = os.environ.get("ELASTICSEARCH_HOST")

es_vector_store = ElasticsearchStore(
    index_name="cvs",              # ← Виправлено
    vector_field='embedding',       # ← Виправлено
    text_field='text',              # ← Виправлено
    es_url=ES_HOST or "http://elasticsearch:9200"
)
