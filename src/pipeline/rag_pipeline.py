from src.core.chunking import chunk_text
from src.core.prompts import build_prompt
from src.config.settings import *

class RAGPipeline:
    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def ingest(self, pages):
        chunks = []
        for page in pages:
            chunks.extend(chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP))
        embeddings = self.embedder.encode(chunks)
        self.vector_store.add(embeddings, chunks)

    def answer(self, question: str):
        query_embedding = self.embedder.encode([question])
        retrieved = self.vector_store.search(query_embedding, TOP_K)
        context = "\n".join(retrieved)
        prompt = build_prompt(context, question)
        return self.llm.generate(prompt, MAX_TOKENS)

