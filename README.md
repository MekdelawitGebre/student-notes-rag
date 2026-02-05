# 📘 Student Notes Q&A (RAG)

**Live App:**  [student-notes-rag.streamlit.app](https://student-notes-rag.streamlit.app)  

---

##  Overview

**Student Notes Q&A** is an **Retrieval-Augmented Generation (RAG) application** that allows users to upload **PDF lecture notes** and ask **natural language questions** about them.

The system retrieves relevant content using **vector similarity search (FAISS)** and generates precise answers using a **Transformer-based LLM (FLAN-T5)**.

---

##  Features

- Upload any PDF lecture notes  
- Automatic text chunking with overlap  
- Semantic embeddings using Sentence Transformers  
- Fast similarity search with FAISS  
- Intelligent answers via FLAN-T5  
- Real-time UI feedback (uploading, processing, generating)  

---

## How It Works — RAG Pipeline

```text
PDF Upload
   ↓
PDF Loader (PyMuPDF)
   ↓
Text Chunking
   ↓
Embedding Model (MiniLM)
   ↓
Vector Store (FAISS)
   ↓
Retriever (Top-K chunks)
   ↓
Prompt Construction
   ↓
LLM (FLAN-T5)
   ↓
Answer Displayed in Streamlit UI
```

---

##  Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python 3.10 |
| **UI** | Streamlit |
| **LLM** | google/flan-t5-base |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector DB** | FAISS (CPU) |
| **PDF Parsing** | PyMuPDF |
| **ML Framework** | PyTorch |
| **Deployment** | Streamlit Cloud |

---

## Project Structure

```yaml
student-notes-rag/
│
├── app.py                     # Streamlit UI
├── requirements.txt            # Dependencies
├── README.md
│
├── src/
│   ├── config/
│   │   └── settings.py         # Model & pipeline configs
│   │
│   ├── core/
│   │   ├── chunking.py         # Text chunking logic
│   │   └── prompts.py          # Prompt templates
│   │
│   ├── infrastructure/
│   │   ├── pdf_loader.py       # PDF text extraction
│   │   ├── embedding_model.py  # Embedding model wrapper
│   │   ├── vector_store.py     # FAISS abstraction
│   │   └── llm_model.py        # LLM wrapper
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py     # End-to-end RAG logic
│   │
│   └── utils/
│       └── logger.py           # Logging utilities
```

---

##  Setup Instructions (Local)

### 1️ Clone Repository
```bash
git clone https://github.com/MekdelawitGebre/student-notes-rag.git
cd student-notes-rag
```

### 2️Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# OR
.venv\Scripts\activate      # Windows
```

### 3️Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️ Run the App
```bash
streamlit run app.py
```

Then open the link displayed (usually):  
 **http://localhost:8501**

---

##  Example Usage

**Question:**
```text
What is gradient descent?
```

**Answer:**
```text
Gradient descent is an optimization algorithm used to minimize a function
by iteratively moving in the direction of the negative gradient.
```

**UI Feedback:**
```
 File is being uploaded...
 Processing PDF...
 Initializing models...
 Generating answer...
 Done!
```
