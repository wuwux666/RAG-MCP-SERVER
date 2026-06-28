"""FastAPI application entry point.

Usage: uvicorn src.api.server:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Modular RAG API",
    description="REST API for Modular RAG MCP Server — query, agentic search, document management",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routers.query import router as query_router
from src.api.routers.collections import router as collections_router
from src.api.routers.documents import router as documents_router
from src.api.routers.traces import router as traces_router

app.include_router(query_router)
app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(traces_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
