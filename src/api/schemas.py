"""Pydantic models for FastAPI request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request ──

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str = Field(default="default")


class AgenticQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_rounds: int = Field(default=10, ge=1, le=20)
    collection: str = Field(default="default")


# ── Response ──

class CitationItem(BaseModel):
    index: int
    source: str
    score: float
    text_snippet: str
    page: Optional[int] = None


class ImageItem(BaseModel):
    image_id: str
    base64: str
    mime_type: str = "image/png"


class TraceRound(BaseModel):
    round: int
    thought: str
    action: str
    observation: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationItem] = []
    images: List[ImageItem] = []
    mode: str = "traditional"
    elapsed_ms: float


class AgenticQueryResponse(BaseModel):
    answer: str
    citations: List[CitationItem] = []
    trace: List[TraceRound] = []
    total_rounds: int
    mode: str = "agentic"
    elapsed_ms: float


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    status: str


class DeleteResponse(BaseModel):
    doc_id: str
    status: str


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CollectionListResponse(BaseModel):
    collections: List[CollectionInfo]


class TraceInfo(BaseModel):
    trace_id: str
    trace_type: str
    metadata: Dict[str, Any]
    stages: List[Dict[str, Any]]
    elapsed_ms: float


class TraceListResponse(BaseModel):
    traces: List[TraceInfo]


class IngestProgressEvent(BaseModel):
    stage: str
    progress: float
    message: str
    doc_id: Optional[str] = None
