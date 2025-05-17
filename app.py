import os
import nltk

# Setze den NLTK-Datenpfad explizit, BEVOR Llama Index importiert wird.
nltk.data.path.append("/tmp")

import streamlit as st
from chatbot.query_engine import ask_question

# Stelle sicher, dass das Verzeichnis existiert
os.makedirs("/tmp", exist_ok=True)

st.title("Mein lokaler GPT-Chatbot")
frage = st.text_input("Stelle eine Frage basierend auf deinen Dokumenten:")

if frage:
    antwort = ask_question(frage)
    st.write(antwort)

# Download der benötigten NLTK-Daten (optional, falls nicht bereits vorhanden)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir="/tmp")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', download_dir="/tmp")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir="/tmp")