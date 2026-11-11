from pyexpat.errors import messages
from urllib import response
import uuid
from app.orchestrator import generate_answer
from app.research import search_web

from app.model_client import (
    MODEL,
    OLLAMA_URL,
    get_available_models,
    chat_with_model,
)
from app.memory import (
    init_db,
    get_history,
    save_messages,
    delete_conversation,
)
from fastapi import FastAPI, HTTPException, requests
from pydantic import BaseModel, Field
from app.tools import TOOL_DEFINITIONS, execute_tool
from app.tiny_model import generate_text, get_model_info




app = FastAPI(
    title="NOVA AI",
    description="Our own AI assistant backend",
    version="0.1.0",
)
init_db()  # Ensure the database is initialized when the app starts.


SYSTEM_PROMPT = """
You are NOVA, a helpful general-purpose AI assistant.
Answer clearly and naturally.
Explain complex concepts step by step when appropriate.
Never pretend to have used a tool or verified information
when you have not. Admit uncertainty when necessary.
"""

# Temporary conversation storage; resets when the server restarts.
conversations = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    conversation_id: str | None = None

class TinyGenerateRequest(BaseModel):
    prompt: str = "hello "
    max_new_tokens: int = 100


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    model: str

class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health():
    try:
        models = get_available_models()

        return {
            "status": "ok",
            "ollama": "connected",
            "model": MODEL,
            "model_available": any(
                name == MODEL or name.startswith(MODEL + ":")
                for name in models
            ),
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama: {exc}",
        ) from exc

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama: {exc}",
        )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    print("NOVA RECEIVED:", repr(body.message))
    conversation_id = body.conversation_id or str(uuid.uuid4())
    history = get_history(conversation_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": body.message},
    ]

   
    try:
        answer = generate_answer(
            history=history,
            user_message=body.message,
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="NOVA could not communicate with its model.",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="NOVA encountered an invalid model response.",
        ) from exc

    save_messages(conversation_id, [
        {"role": "user", "content": body.message},
        {"role": "assistant", "content": answer},
    ])

    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        model=MODEL,
    )


@app.delete("/conversations/{conversation_id}")
def reset_conversation(conversation_id: str):
    delete_conversation(conversation_id)

    return {
        "deleted": True,
        "conversation_id": conversation_id,
    }


@app.post("/experimental/generate")
def experimental_generate(request: TinyGenerateRequest):
    if not 1 <= request.max_new_tokens <= 300:
        raise HTTPException(
            status_code=400,
            detail="max_new_tokens must be between 1 and 300",
        )

    try:
        generated_text = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )

        return {
            "model": "nova-tiny-transformer",
            "generated_text": generated_text,
            "checkpoint": get_model_info(),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Tiny Transformer checkpoint is unavailable.",
        ) from exc


@app.post("/research")
def research(body: ResearchRequest):
    try:
        results = search_web(
            query=body.query,
            max_results=body.max_results,
        )

        return {
            "query": body.query,
            "result_count": len(results),
            "sources": results,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Web search failed. Check your connection and try again.",
        ) from exc

