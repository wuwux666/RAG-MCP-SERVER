"""Collections endpoints: list available ChromaDB collections with counts."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import CollectionInfo, CollectionListResponse
from src.core.settings import load_settings

router = APIRouter(prefix="/api", tags=["collections"])


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """List all ChromaDB collections with document counts."""
    try:
        settings = load_settings()
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        store = VectorStoreFactory.create(settings)

        collections: list[CollectionInfo] = []
        if hasattr(store, "client"):
            collection_names = store.client.list_collections()
            for name in collection_names:
                try:
                    coll = store.client.get_collection(name=str(name))
                    count = coll.count()
                except Exception:
                    count = 0
                collections.append(
                    CollectionInfo(name=str(name), document_count=count)
                )

        return CollectionListResponse(collections=collections)
    except Exception as e:
        # Return empty list on error rather than crashing
        return CollectionListResponse(collections=[])
