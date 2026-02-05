import streamlit as st
from src.infrastructure.pdf_loader import load_pdf
from src.infrastructure.embedding_model import EmbeddingModel
from src.infrastructure.vector_store import VectorStore
from src.infrastructure.llm_model import LLM
from src.pipeline.rag_pipeline import RAGPipeline
from src.config.settings import *

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Student Notes Q&A",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- HEADER ----------
st.title(" Student Notes Q&A (RAG)")
st.markdown(
    "Upload your lecture notes (PDF) and ask questions. "
    "Powered by a Retrieval-Augmented Generation pipeline."
)

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader("Upload PDF lecture notes", type="pdf")

if uploaded:
    st.info(" File is being uploaded...")

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())

    st.success("File uploaded successfully!")
    
    # ---------- PDF LOADING ----------
    with st.spinner(" Processing PDF..."):
        pages = load_pdf("temp.pdf")
        st.info(f"Loaded {len(pages)} pages from PDF.")

    # ---------- INIT ML MODELS ----------
    with st.spinner(" Initializing models..."):
        embedder = EmbeddingModel(EMBEDDING_MODEL_NAME)
        vector_store = VectorStore(dim=384)
        llm = LLM(LLM_MODEL_NAME)
        rag = RAGPipeline(embedder, vector_store, llm)
        rag.ingest(pages)
    st.success(" PDF processed and RAG pipeline ready!")

    # ---------- QUESTION INPUT ----------
    st.markdown("---")
    st.subheader("Ask a Question")
    question = st.text_input("Type your question here:")

    if question:
        with st.spinner("⏳ Generating answer..."):
            answer = rag.answer(question)
        st.subheader("Answer")
        st.write(answer)
