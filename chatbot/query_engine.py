import os
import logging
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Optional

# Richten Sie das Logging ein
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents(data_dir: str = "./data") -> List[Document]:
    """Lädt Dokumente aus dem angegebenen Verzeichnis.

    Args:
        data_dir: Der Pfad zum Verzeichnis mit den Dokumenten.

    Returns:
        Eine Liste von Dokumenten.
    """
    # Erweitern Sie den Pfad nicht erneut, da er bereits absolut ist
    logger.info(f"Lade Dokumente aus: {data_dir}")
    # Der ursprüngliche Code hatte eine race condition, da er os.makedirs und os.listdir in separaten Schritten aufrief.
    # Verwenden Sie stattdessen eine einzige list comprehension.
    try:
        filenames = [
            os.path.join(data_dir, filename)
            for filename in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, filename))
        ]
    except FileNotFoundError:
        logger.error(f"Verzeichnis nicht gefunden: {data_dir}")
        return []  # Rückgabe einer leeren Liste, um einen Fehler zu vermeiden

    documents = []
    for filename in filenames:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(text=text))
            logger.info(f"Dokument geladen: {filename}")
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Datei {filename}: {e}")
    return documents



def ask_question(question: str, query_engine) -> str:
    """Beantwortet eine Frage anhand der Dokumente.

    Args:
        question: Die Frage, die gestellt werden soll.
        query_engine: Die Llama Index-Abfrage-Engine.

    Returns:
        Die Antwort auf die Frage.
    """
    try:
        response = query_engine.query(question)
        logger.info(f"Frage beantwortet: {question}")
        return str(response)  # Stellen Sie sicher, dass die Antwort ein String ist
    except Exception as e:
        logger.error(f"Fehler bei der Abfrage: {e}")
        return "Ich konnte die Frage nicht beantworten."



def main(data_dir: str = "./data",
         embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
         llm_model_name: str = "google/flan-t5-base") -> None:
    """Initialisiert die Chatbot-Komponenten und führt eine Abfrage durch.

    Args:
        data_dir: Der Pfad zum Verzeichnis mit den Dokumenten.
        embed_model_name: Der Name des zu verwendenden Embedding-Modells.
        llm_model_name: Der Name des zu verwendenden LLM-Modells.
    """

    # Laden Sie die Dokumente nur einmal
    documents = load_documents(data_dir)
    if not documents:
        logger.warning("Keine Dokumente zum Verarbeiten gefunden. Die Anwendung wird beendet.")
        return  # Beenden Sie die Funktion, wenn keine Dokumente vorhanden sind

    # Lokales Embedding-Modell
    try:
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        Settings.embed_model = embed_model
        logger.info(f"Embedding-Modell geladen: {embed_model_name}")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Embedding-Modells: {e}")
        return

    # Lokales LLM initialisieren
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        llm = HuggingFaceLLM(
            model_name=llm_model_name,
            tokenizer_name=llm_model_name,
            context_window=2048,
            max_new_tokens=256,
            model=model,
            device_map="auto",
        )
        logger.info(f"LLM-Modell geladen: {llm_model_name}")
    except Exception as e:
        logger.error(f"Fehler beim Laden des LLM-Modells: {e}")
        return

    try:
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        query_engine = index.as_query_engine(llm=llm)
        logger.info("Index und Abfrage-Engine erstellt.")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Index oder der Abfrage-Engine: {e}")
        return


    # Beispielabfrage
    frage = "Was ist die Antwort auf alles?"
    antwort = ask_question(frage, query_engine) # Übergabe der Instanz von query_engine
    print(f"Frage: {frage}")
    print(f"Antwort: {antwort}")



if __name__ == "__main__":
    main()
