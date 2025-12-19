from sentence_transformers import SentenceTransformer

class Embedder:
    _model = None

    @classmethod
    def load(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
        return cls._model

    @classmethod
    def embed(cls, texts: list[str]):
        model = cls.load()
        return model.encode(
            texts,
            batch_size=8,
            convert_to_numpy=True,
            show_progress_bar=False
        )
