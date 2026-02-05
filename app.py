import streamlit as st

from src.infrastructure.pdf_loader import load_pdf
from src.infrastructure.embedding_model import EmbeddingModel
from src.infrastructure.vector_store import VectorStore
from src.infrastructure.llm_model import LLM
from src.pipeline.rag_pipeline import RAGPipeline
from src.config.settings import *

st.title("📘 Student Notes Q&A (RAG)")

uploaded = st.file_uploader("Upload PDF", type="pdf")

if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())

    pages = load_pdf("temp.pdf")

    embedder = EmbeddingModel(EMBEDDING_MODEL_NAME)
    vector_store = VectorStore(dim=384)
    llm = LLM(LLM_MODEL_NAME)

    rag = RAGPipeline(embedder, vector_store, llm)
    rag.ingest(pages)

    question = st.text_input("Ask a question")
    if question:
        answer = rag.answer(question)
        st.subheader("Answer")
        st.write(answer)
