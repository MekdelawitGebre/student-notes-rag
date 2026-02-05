import faiss

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, k: int):
        _, indices = self.index.search(query_embedding, k)
        return [self.texts[i] for i in indices[0]]
