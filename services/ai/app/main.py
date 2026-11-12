from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.memory.sqlite_memory import init_db


app = FastAPI(
    title="NOVA AI",
    description="Independent Modular AI Assistant Platform",
    version="0.1.0",
)


@app.on_event("startup")
async def initialize_database():
    init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


app.include_router(health_router)
app.include_router(knowledge_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "project": "NOVA AI",
        "message": "Independent AI Assistant Platform",
        "status": "running",
    }