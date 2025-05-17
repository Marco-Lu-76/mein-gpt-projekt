import streamlit as st
from chatbot.query_engine import ask_question

st.title("Mein lokaler GPT-Chatbot")
frage = st.text_input("Stelle eine Frage basierend auf deinen Dokumenten:")

if frage:
    antwort = ask_question(frage)
    st.write(antwort)
