import streamlit as st
from rag_chain import rag_answer

st.set_page_config(
    page_title="Enterprise RAG Bot",
    layout="wide"
)

st.title("Enterprise RAG Chatbot")
st.write("Ask questions grounded in enterprise documents")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Generating response..."):
        response = rag_answer(query)
        st.markdown("### Answer")
        st.write(response)
