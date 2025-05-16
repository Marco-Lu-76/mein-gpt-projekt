from llama_index.core import SimpleDirectoryReader
import os

def load_documents():
    # Stellen Sie sicher, dass der Ordner 'data' im Hauptprojekt existiert
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Ordner '{data_dir}' wurde erstellt. Legen Sie Ihre Textdateien dort ab.")
        return []

    reader = SimpleDirectoryReader(input_dir=data_dir)
    documents = reader.load_data()
    return documents

if __name__ == "__main__":
    documents = load_documents()
    print(f"Anzahl der geladenen Dokumente: {len(documents)}")