"""Traces endpoints: list query and ingestion trace records."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Query

from src.api.schemas import TraceInfo, TraceListResponse

router = APIRouter(prefix="/api/traces", tags=["traces"])

TRACE_FILE = "logs/traces.jsonl"


def _read_traces(trace_type: str, limit: int) -> list[TraceInfo]:
    """Read the last N trace lines matching a given trace_type."""
    if not os.path.exists(TRACE_FILE):
        return []

    # Work relative to repo root (where logs/ lives)
    trace_path = Path(TRACE_FILE)
    if not trace_path.is_absolute():
        # Resolve relative to cwd (project root when launched from there)
        trace_path = Path.cwd() / TRACE_FILE

    traces: list[TraceInfo] = []
    try:
        with open(trace_path, encoding="utf-8") as f:
            lines = f.readlines()
        # Read from the end, collect up to `limit` matching lines
        for line in reversed(lines):
            if len(traces) >= limit:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("trace_type") != trace_type:
                continue
            traces.append(
                TraceInfo(
                    trace_id=data.get("trace_id", ""),
                    trace_type=trace_type,
                    metadata=data.get("metadata", {}),
                    stages=data.get("stages", []),
                    # The JSONL uses "total_elapsed_ms", not "elapsed_ms"
                    elapsed_ms=data.get("total_elapsed_ms", 0),
                )
            )
        # Reverse back so newest-first order is preserved in output (we
        # iterated reversed, so records are oldest-first right now).
        traces.reverse()
    except Exception:
        return []

    return traces


@router.get("/query", response_model=TraceListResponse)
async def list_query_traces(limit: int = Query(default=50, le=200)):
    """List recent query traces."""
    return TraceListResponse(traces=_read_traces("query", limit))


@router.get("/ingestion", response_model=TraceListResponse)
async def list_ingestion_traces(limit: int = Query(default=20, le=200)):
    """List recent ingestion traces."""
    return TraceListResponse(traces=_read_traces("ingestion", limit))
