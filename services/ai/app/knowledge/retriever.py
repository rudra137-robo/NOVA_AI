from pathlib import Path

from app.knowledge.embedder import EmbeddingEngine
from app.knowledge.vector_store import VectorStore


class KnowledgeRetriever:

    def __init__(self, index_directory: Path):

        self.embedder = EmbeddingEngine()

        self.store = VectorStore.load(
            index_directory
        )

    def search(
        self,
        question: str,
        top_k: int = 3
    ):

        if not question.strip():
            return []

        query_vector = self.embedder.embed_query(
            question
        )

        return self.store.search(
            query_vector,
            top_k=top_k
        )