from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:

    def __init__(self):

        self.model = SentenceTransformer(MODEL_NAME)

    def embed_documents(
        self,
        texts: list[str]
    ):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def embed_query(
        self,
        query: str
    ):

        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )