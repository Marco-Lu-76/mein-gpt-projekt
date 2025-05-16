<<<<<<< HEAD
import streamlit as st
from chatbot.query_engine import ask_question

st.title("Mein lokaler GPT-Chatbot")
frage = st.text_input("Stelle eine Frage basierend auf deinen Dokumenten:")

if frage:
    antwort = ask_question(frage)
=======
import streamlit as st
from chatbot.query_engine import ask_question

st.title("Mein lokaler GPT-Chatbot")
frage = st.text_input("Stelle eine Frage basierend auf deinen Dokumenten:")

if frage:
    antwort = ask_question(frage)
>>>>>>> 95e5c8e5dc2ad04320ed412926951b66a1e3b252
    st.write(antwort)