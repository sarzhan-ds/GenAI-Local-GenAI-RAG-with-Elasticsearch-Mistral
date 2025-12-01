from fastapi import FastAPI
from pydantic import BaseModel
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from elasticsearch import AsyncElasticsearch

# --- Config ---
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
EMBED_MODEL = "nomic-embed-text"

print(f"🔧 ES_HOST: {ES_HOST}")
print(f"🔧 OLLAMA_HOST: {OLLAMA_HOST}")
print(f"🔧 LLM_MODEL: {LLM_MODEL}")
print(f"🔧 EMBED_MODEL: {EMBED_MODEL}")

# --- Elasticsearch client ---
es_client = AsyncElasticsearch([ES_HOST])

# --- Elasticsearch vector store ---
es_vector_store = ElasticsearchStore(
    index_name="cvs",
    vector_field="embedding",
    text_field="text",
    es_client=es_client
)

# --- LLM + Embeddings ---
Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_HOST,
    request_timeout=300.0
)

Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_HOST
)

# --- FastAPI app ---
app = FastAPI()

class IndexRequest(BaseModel):
    directory: str = "/app/resumes"

@app.post("/index")
async def index_documents(request: IndexRequest = None):
    """Index all PDF files from the specified directory."""
    if request is None:
        request = IndexRequest()
    
    try:
        print(f"📂 Loading documents from: {request.directory}")
        
        # Load documents from directory
        documents = SimpleDirectoryReader(
            input_dir=request.directory,
            required_exts=[".pdf"]
        ).load_data()
        
        print(f"📄 Found {len(documents)} documents")
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=es_vector_store
        )
        
        # Create index and store in Elasticsearch
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        
        print(f"✅ Indexed {len(documents)} documents")
        
        return {
            "status": "success",
            "message": f"Indexed {len(documents)} documents from {request.directory}"
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "llm": LLM_MODEL, "embed": EMBED_MODEL}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FastAPI RAG Service",
        "endpoints": {
            "POST /index": "Index PDF documents",
            "GET /health": "Health check"
        }
    }
