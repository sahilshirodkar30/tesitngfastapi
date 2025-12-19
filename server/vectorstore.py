import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(embeddings.astype("float32"))
        self.texts.extend(texts)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), k
        )
        return [self.texts[i] for i in indices[0]]
