import json, os
import fitz
import re
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from index_raw import es_vector_store  # Тепер правильний імпорт
from ollama import chat
from ollama import ChatResponse

OLLAMA_HOST = os.environ.get("OLLAMA_HOST")
MODEL_NAME = os.environ.get("OLLAMA_MODEL")
ES_HOST = os.environ.get("ELASTICSEARCH_HOST")

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        text += page_text
    print(text)
    return text

def prepare_text_to_json(text_to_summarize):
    instruction_template = (
        'You are a lossless extractor. Extract the name and text from the following document and output it in the format {"name": string, "text": string}. '
        'Only provide the response in this format, with no extra explanation. Do not omit, shorten, or summarize any content. '
        'Copy every word, bullet point, list item, heading, character, and space exactly as in the document; preserve line breaks as \\n and tabs as \\t. '
        'Do not insert or replace with ellipses or placeholders; NEVER output ..., …, unless they appear in the source. '
        'Use double quotes for all JSON strings. If you cannot include every character for any reason, return exactly {"error":"TOO_LONG"} instead of partial content.'
    )
    
    response: ChatResponse = chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': instruction_template + text_to_summarize}]
    )
    
    response_text = response['message']['content']
    
    # Очищаємо контрольні символи (крім \n, \t, \r)
    cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', response_text)
    
    # Витягуємо JSON з markdown блоку, якщо він є
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    
    print(".....Prepared this json.....\n")
    print(cleaned)
    return cleaned

local_llm = Ollama(model=MODEL_NAME)

def process_pdf(pdf_path):
    ollama_embedding = OllamaEmbedding(MODEL_NAME, base_url=OLLAMA_HOST)
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=200, chunk_overlap=50),
            ollama_embedding,
        ],
        vector_store=es_vector_store
    )
    
    extracted = extract_text_from_pdf(pdf_path)
    prepped_json = json.loads(prepare_text_to_json(extracted))
    
    documents = [Document(text=prepped_json['text'], metadata={"name": prepped_json['name']})]
    
    pipeline.run(documents=documents)
    print(".....Done running pipeline.....\n")

if __name__ == "__main__":
    process_pdf('Jeffrey_Lebowski_CV.pdf')
