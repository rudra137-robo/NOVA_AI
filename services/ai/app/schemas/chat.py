from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str

    conversation_id: str | None = None

    use_knowledge: bool = False

    top_k: int = Field(default=3, ge=1, le=10)


class SourceReference(BaseModel):
    reference: int
    source: str
    page: int | None = None
    chunk_index: int
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str

    conversation_id: str

    sources: list[SourceReference] = Field(
        default_factory=list
    )