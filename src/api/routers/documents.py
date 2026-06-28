"""Documents endpoints: upload, list, delete ingested documents."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from src.api.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
)
from src.core.settings import load_settings, resolve_path

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("default"),
):
    """Upload and ingest a document through the full pipeline."""
    settings = load_settings()

    # Write upload to temp file
    suffix = f"_{file.filename}" if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(settings, collection=collection)
        result = pipeline.run(tmp_path)
        pipeline.close()

        return UploadResponse(
            doc_id=result.doc_id or Path(tmp_path).name,
            filename=file.filename or "unknown",
            chunks=result.chunk_count,
            status="ingested" if result.success else f"failed: {result.error}",
        )
    except Exception as exc:
        return UploadResponse(
            doc_id=Path(tmp_path).name,
            filename=file.filename or "unknown",
            chunks=0,
            status=f"error: {exc}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all successfully ingested documents from ingestion history."""
    try:
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker

        db_path = str(resolve_path("data/db/ingestion_history.db"))
        checker = SQLiteIntegrityChecker(db_path)
        rows = checker.list_processed()

        documents = [
            DocumentInfo(
                doc_id=row.get("file_hash", ""),
                filename=Path(row.get("file_path", "")).name or row.get("file_hash", ""),
                chunks=0,  # chunk count not stored in integrity DB
                ingested_at=row.get("processed_at", ""),
            )
            for row in rows
        ]
        return DocumentListResponse(documents=documents)
    except Exception:
        return DocumentListResponse(documents=[])


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the system."""
    settings = load_settings()
    errors: list[str] = []

    # 1. Remove from ingestion history (SQLite)
    try:
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker

        db_path = str(resolve_path("data/db/ingestion_history.db"))
        checker = SQLiteIntegrityChecker(db_path)
        deleted = checker.remove_record(doc_id)
        if not deleted:
            errors.append("not found in ingestion history")
    except Exception as exc:
        errors.append(f"history error: {exc}")

    # 2. Remove chunks from ChromaDB (by source_hash metadata)
    try:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        store = VectorStoreFactory.create(settings)
        if hasattr(store, "delete_by_metadata"):
            # ChromaStore supports metadata-based deletion
            deleted_count = store.delete_by_metadata({"source_hash": doc_id})
        elif hasattr(store, "delete"):
            # Fallback: attempt direct ID deletion (may not match)
            store.delete([doc_id])
    except Exception as exc:
        errors.append(f"vector store error: {exc}")

    # 3. Remove BM25 index file if it exists
    try:
        for collection_name in ("default",):
            bm25_dir = resolve_path(f"data/db/bm25/{collection_name}")
            bm25_index = bm25_dir / f"{doc_id}.json"
            if bm25_index.exists():
                bm25_index.unlink()
    except Exception as exc:
        errors.append(f"bm25 cleanup error: {exc}")

    status = "deleted" if not errors else f"partial: {'; '.join(errors)}"
    return DeleteResponse(doc_id=doc_id, status=status)
