# chatbot/query_engine.py
import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from chatbot.document_loader import load_documents
from transformers import AutoModelForSeq2SeqLM  # Importieren Sie die spezifische AutoModel-Klasse

# Lokales Embedding-Modell
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Lokales LLM initialisieren (passen Sie model_name an Ihr heruntergeladenes Modell an)
llm = HuggingFaceLLM(
    model_name="google/flan-t5-base",
    tokenizer_name="google/flan-t5-base",
    context_window=2048,
    max_new_tokens=256,
    model=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base"), # Überschreiben Sie AutoModel
    device_map="auto" # Versuchen Sie device_map hier wieder
    # model_kwargs={"model_type": "seq2seq"}, # Entfernen Sie dies
)

_documents = load_documents()
_index = VectorStoreIndex.from_documents(_documents)
_query_engine = _index.as_query_engine(llm=llm)

def ask_question(question: str) -> str:
    response = _query_engine.query(question)
    return str(response)