from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.knowledge.retriever import KnowledgeRetriever
from app.schemas.knowledge import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)

router = APIRouter(
    prefix="/api/v1/knowledge",
    tags=["Knowledge"]
)

ROOT = Path(__file__).resolve().parents[5]

INDEX_DIR = (
    ROOT / "data" / "index" / "vector_store"
)


@lru_cache(maxsize=1)
def get_retriever():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            "Knowledge index not found. Build the index first."
        )

    return KnowledgeRetriever(INDEX_DIR)


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty."
        )

    try:
        retriever = get_retriever()

        results = retriever.search(
            question,
            top_k=request.top_k
        )

        formatted_results = [
            SearchResult(**result)
            for result in results
        ]

        return SearchResponse(
            question=question,
            total_results=len(formatted_results),
            results=formatted_results
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge search failed: {error}"
        )