import json
import faiss
import numpy as np

from pathlib import Path


class VectorStore:

    def __init__(self, dimension: int):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.metadata = []

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[dict]
    ):

        vectors = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        if len(vectors) != len(chunks):
            raise ValueError("Embedding and metadata counts differ")

        if len(vectors) == 0:
            return

        self.index.add(vectors)

        self.metadata.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ):

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError("Query dimension mismatch")

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query,
            k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            results.append({
                **self.metadata[int(idx)],
                "score": float(score)
            })

        return results

    def save(self, directory: Path):

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(directory / "knowledge.faiss")
        )

        with (
            directory / "metadata.json"
        ).open("w", encoding="utf-8") as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, directory: Path):

        index = faiss.read_index(
            str(directory / "knowledge.faiss")
        )

        instance = cls(index.d)

        instance.index = index

        with (
            directory / "metadata.json"
        ).open("r", encoding="utf-8") as file:

            instance.metadata = json.load(file)

        if instance.index.ntotal != len(instance.metadata):
            raise ValueError("Index and metadata counts differ")

        return instance