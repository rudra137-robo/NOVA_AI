from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    source: str
    page: int | None
    chunk_index: int
    content: str
    score: float


class SearchResponse(BaseModel):
    question: str
    total_results: int
    results: list[SearchResult]