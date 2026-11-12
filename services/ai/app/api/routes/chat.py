import logging

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.core.orchestrator import generate_answer
from app.core.rag_service import RAGService
from app.knowledge.retriever import KnowledgeRetriever
from app.memory.sqlite_memory import (
    get_history,
    save_messages,
)
from app.providers.factory import get_llm_provider
from app.schemas.chat import ChatRequest, ChatResponse


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Chat"],
)

ROOT = Path(__file__).resolve().parents[5]

INDEX_DIR = (
    ROOT / "data" / "index" / "vector_store"
)


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
        llm_provider=provider,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    conversation_id = (
        request.conversation_id or str(uuid4())
    )

    try:

        sources = []

        # MODE 1: DOCUMENT KNOWLEDGE
        if request.use_knowledge:

            rag = get_rag_service()

            result = rag.ask(
                question=question,
                top_k=request.top_k,
            )

            answer = result["answer"]

            sources = result.get("sources", [])

        # MODE 2: GENERAL NOVA CHAT
        else:

            history = get_history(conversation_id)

            answer = generate_answer(
                history=history,
                user_message=question,
            )

        # SAVE BOTH USER AND ASSISTANT MESSAGES
        save_messages(
            conversation_id,
            [
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ],
        )

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception("NOVA chat generation failed")

        raise HTTPException(
            status_code=500,
            detail="NOVA encountered an error while generating a response.",
        )