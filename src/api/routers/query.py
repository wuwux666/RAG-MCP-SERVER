"""Query endpoints: traditional hybrid search + agentic ReAct."""

from __future__ import annotations

import asyncio
import time
from fastapi import APIRouter

from src.api.schemas import (
    QueryRequest, QueryResponse, AgenticQueryRequest, AgenticQueryResponse,
    CitationItem, TraceRound,
)
from src.core.settings import load_settings

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Traditional hybrid search — fast, single-pass retrieval."""
    settings = load_settings()

    from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
    tool = QueryKnowledgeHubTool(settings=settings)

    t0 = time.monotonic()
    response = await tool.execute(query=req.query, top_k=req.top_k, collection=req.collection)
    elapsed_ms = (time.monotonic() - t0) * 1000

    citations = [
        CitationItem(
            index=c.index,
            source=c.source,
            score=c.score,
            text_snippet=c.text_snippet,
            page=getattr(c, "page", None),
        )
        for c in response.citations
    ]

    return QueryResponse(
        answer=response.content,
        citations=citations,
        mode="traditional",
        elapsed_ms=round(elapsed_ms, 1),
    )


@router.post("/agentic-query", response_model=AgenticQueryResponse)
async def agentic_query(req: AgenticQueryRequest):
    """Agentic RAG with ReAct loop — for complex multi-hop questions."""
    settings = load_settings()

    from src.agentic.mcp_client import MCPClient
    from src.agentic.react_loop import ReActLoop
    from src.libs.llm.llm_factory import LLMFactory

    llm = LLMFactory.create(settings)
    client = MCPClient()
    await client.connect()

    try:
        loop = ReActLoop(mcp_client=client, llm_client=llm, max_rounds=req.max_rounds)
        t0 = time.monotonic()
        result = await loop.run(req.query)
        elapsed_ms = (time.monotonic() - t0) * 1000

        trace = [
            TraceRound(round=t.round_number, thought=t.thought, action=t.action, observation=t.observation)
            for t in result.trace
        ]

        # Build answer: main answer text + citations block.
        # result.citations are formatted strings like "[1] docs/a.md", not CitationItem objects.
        answer = result.answer
        if result.citations:
            citations_block = "\n\n---\n**引用来源:**\n" + "\n".join(result.citations)
            answer = answer + citations_block

        return AgenticQueryResponse(
            answer=answer,
            citations=[],  # citation tracker doesn't produce structured items — use answer text references
            trace=trace,
            total_rounds=len(result.trace),
            mode="agentic",
            elapsed_ms=round(elapsed_ms, 1),
        )
    finally:
        await client.close()
