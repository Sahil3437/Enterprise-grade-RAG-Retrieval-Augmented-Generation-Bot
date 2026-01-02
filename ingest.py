import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"

documents = []

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)

    if file.lower().endswith(".pdf"):
        documents.extend(PyPDFLoader(path).load())

    elif file.lower().endswith(".docx"):
        documents.extend(Docx2txtLoader(path).load())

    elif file.lower().endswith(".txt"):
        documents.extend(TextLoader(path, encoding="utf-8").load())

if not documents:
    raise ValueError("No documents were loaded. Check the data folder.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("vectorstore")

print("Document ingestion completed successfully.")
