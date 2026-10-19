"""Pydantic models for DocuMindRAG API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    chunk_text: str
    char_start: int
    char_end: int
    retrieval_score: float


class GroundednessScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Overall groundedness 0-1")
    citation_precision: float = Field(..., ge=0.0, le=1.0)
    context_coverage: float = Field(..., ge=0.0, le=1.0)
    has_hedging: bool
    verdict: str = Field(..., pattern="^(grounded|partially_grounded|ungrounded|not_applicable)$")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    doc_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    groundedness: GroundednessScore
    timing_ms: float
    model_used: str


class UploadResponse(BaseModel):
    doc_id: str
    status: DocumentStatus
    name: str
    page_count: int
    chunk_count: int


class DocumentInfo(BaseModel):
    id: str
    name: str
    page_count: int
    chunk_count: int
    status: DocumentStatus
    created_at: str
    error_message: Optional[str] = None


class DeleteResponse(BaseModel):
    status: str = "deleted"
    doc_id: str


class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant: str = "connected"
    version: str = "0.1.0"
    demo_mode: bool = False
    llm_calls_left_today: int | None = None


class FeedbackRequest(BaseModel):
    query_log_id: int = Field(..., gt=0)
    rating: str = Field(..., pattern="^(thumbs_up|thumbs_down)$")
    comment: Optional[str] = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str = "saved"


class AnalyticsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    avg_groundedness: float
    thumbs_up: int
    thumbs_down: int
    feedback_rate: float
    recent_queries: list[dict]
    verdicts: dict
    models_used: dict


class EvalRunRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)


class EvalResult(BaseModel):
    query: str
    answer: str
    groundedness_score: float
    groundedness_verdict: str
    citation_count: int
    latency_ms: float
    model_used: str


class EvalRunResponse(BaseModel):
    total_questions: int
    results: list[EvalResult]
    avg_groundedness: float
    avg_latency_ms: float
    avg_citations: float
