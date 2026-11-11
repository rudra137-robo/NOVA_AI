from pathlib import Path

from app.knowledge.retriever import KnowledgeRetriever
from app.providers.base import LLMProvider


class RAGService:

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        llm_provider: LLMProvider
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider

    def build_prompt(self, question: str, documents: list[dict]) -> str:

        context_parts = []

        for index, document in enumerate(documents, start=1):

            source = document["source"]
            page = document.get("page")

            location = (
                f"Page {page}" if page is not None
                else "Page not specified"
            )

            context_parts.append(
                f"[{index}] Source: {source} | {location}\n"
                f"{document['content']}"
            )

        context = "\n\n".join(context_parts)

        return f"""
Answer the student's question using the context below.

STUDENT QUESTION:
{question}

UNIVERSITY CONTEXT:
{context}

INSTRUCTIONS:
1. Use the context as the source of university-specific facts.
2. Do not invent missing information.
3. If the context is insufficient, say so clearly.
4. Cite supporting passages using their reference numbers,
   for example [1] or [2].
5. Keep the answer clear and student-friendly.
"""

    def ask(self, question: str, top_k: int = 3) -> dict:

        documents = self.retriever.search(
            question,
            top_k=top_k
        )

        if not documents:
            return {
                "answer": (
                    "I couldn't find relevant information in "
                    "the available university documents."
                ),
                "sources": []
            }

        prompt = self.build_prompt(question, documents)

        answer = self.llm_provider.generate(prompt)

        sources = []

        for index, document in enumerate(documents, start=1):

            sources.append({
                "reference": index,
                "source": document["source"],
                "page": document.get("page"),
                "chunk_index": document["chunk_index"],
                "similarity_score": document["score"]
            })

        return {
            "answer": answer,
            "sources": sources
        }