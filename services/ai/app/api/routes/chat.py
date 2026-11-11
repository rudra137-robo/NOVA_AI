from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.core.rag_service import RAGService
from app.knowledge.retriever import KnowledgeRetriever
from app.providers.factory import get_llm_provider
from app.schemas.chat import ChatRequest, ChatResponse

from pathlib import Path

router = APIRouter(
    prefix="/api/v1",
    tags=["Chat"]
)

ROOT = Path(__file__).resolve().parents[5]

INDEX_DIR = ROOT / "data" / "index" / "vector_store"


@lru_cache(maxsize=1)
def get_rag_service():

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            "Knowledge index not found. Build the index first."
        )

    retriever = KnowledgeRetriever(INDEX_DIR)
    provider = get_llm_provider()

    return RAGService(
        retriever=retriever,
        llm_provider=provider
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty."
        )

    try:
        rag = get_rag_service()

        result = rag.ask(
            question=question,
            top_k=request.top_k
        )

        return ChatResponse(**result)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation failed: {error}"
        )