"""MCP Tool: query_knowledge_hub

This tool provides knowledge retrieval capabilities through the MCP protocol.
It combines HybridSearch (Dense + Sparse + RRF Fusion) with optional Reranking
to find relevant documents and return formatted results with citations.

Usage via MCP:
    Tool name: query_knowledge_hub
    Input schema:
        - query (string, required): The search query
        - top_k (integer, optional): Number of results to return (default: 5)
        - collection (string, optional): Limit search to specific collection
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

from src.core.response.response_builder import ResponseBuilder, MCPToolResponse
from src.core.settings import load_settings, resolve_path, Settings
from src.core.trace import TraceContext, TraceCollector
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import CoreReranker

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = """Search the knowledge base for relevant documents.

This tool uses hybrid search (semantic + keyword) to find the most relevant
documents matching your query. Results include source citations for reference.

Supports agentic mode for complex multi-hop questions — the tool automatically
detects question complexity and performs iterative retrieval when needed.

Parameters:
- query: Your search question or keywords
- top_k: Maximum number of results (default: 5)
- collection: Limit search to a specific document collection
- agentic: "auto" (default) for automatic routing, "off" for traditional search,
  "on" to force multi-step iterative retrieval
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question to find relevant documents for.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to limit the search scope.",
        },
        "agentic": {
            "type": "string",
            "description": "Agentic mode: 'auto' (smart routing, default), 'off' (traditional single-shot), 'on' (force multi-step).",
            "enum": ["auto", "off", "on"],
            "default": "auto",
        },
    },
    "required": ["query"],
}


@dataclass
class QueryKnowledgeHubConfig:
    """Configuration for query_knowledge_hub tool.

    Attributes:
        default_top_k: Default number of results if not specified
        max_top_k: Maximum allowed top_k value
        default_collection: Default collection if not specified
        enable_rerank: Whether to apply reranking
        agentic_mode: Agentic routing mode ("auto", "off", "on")
        max_agentic_rounds: Max iterative rounds in agentic mode
    """
    default_top_k: int = 5
    max_top_k: int = 20
    default_collection: str = "default"
    enable_rerank: bool = True
    agentic_mode: str = "auto"
    max_agentic_rounds: int = 5


class QueryKnowledgeHubTool:
    """MCP Tool for knowledge base queries.
    
    This class encapsulates the query_knowledge_hub tool logic,
    coordinating HybridSearch and Reranker to produce formatted results.
    
    Design Principles:
    - Lazy initialization: Components created on first use
    - Error resilience: Graceful handling of search/rerank failures
    - Configurable: All parameters from settings.yaml
    
    Example:
        >>> tool = QueryKnowledgeHubTool(settings)
        >>> result = await tool.execute(query="Azure 配置", top_k=5)
        >>> print(result.content)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[QueryKnowledgeHubConfig] = None,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[CoreReranker] = None,
        response_builder: Optional[ResponseBuilder] = None,
    ) -> None:
        """Initialize QueryKnowledgeHubTool.
        
        Args:
            settings: Application settings. If None, loaded from default path.
            config: Tool configuration. If None, uses defaults.
            hybrid_search: Optional pre-configured HybridSearch instance.
            reranker: Optional pre-configured CoreReranker instance.
            response_builder: Optional pre-configured ResponseBuilder instance.
        """
        self._settings = settings
        self.config = config or QueryKnowledgeHubConfig()
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._embedding_client = None
        self._response_builder = response_builder or ResponseBuilder()

        # Track initialization state
        self._initialized = False
        self._current_collection: Optional[str] = None
    
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings
    
    def _ensure_initialized(self, collection: str) -> None:
        """Ensure search components are initialized for the given collection.
        
        Caching strategy (balances speed vs freshness):
        - **Fully cached** (stateless, never go stale): embedding client,
          reranker, query processor, settings.
        - **Cached until collection changes**: vector store (ChromaDB
          PersistentClient reads from SQLite — sees data written by other
          processes), dense retriever, hybrid search.
        - **Auto-refreshes on every query**: BM25 sparse index — the
          ``SparseRetriever._ensure_index_loaded()`` always reloads from
          disk, so the cached SparseRetriever object is fine.
        
        Only when *collection* changes do we tear down and rebuild.
        
        Args:
            collection: Target collection name.
        """
        # Always rebuild vector_store and retriever components so that
        # data ingested by other processes (e.g. Dashboard) is visible
        # immediately without requiring an MCP Server restart.
        
        logger.info(f"Initializing query components for collection: {collection}")
        
        # Import here to avoid circular imports and allow lazy loading
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        
        # === Fully cached components (stateless, never go stale) ===

        if self._reranker is None:
            self._reranker = create_core_reranker(settings=self.settings)

        if self._embedding_client is None:
            self._embedding_client = EmbeddingFactory.create(self.settings)
        
        if self._reranker is None:
            self._reranker = create_core_reranker(settings=self.settings)
        
        # === Rebuild for new collection ===
        # ChromaDB PersistentClient uses SQLite under the hood —
        # concurrent readers see committed writes from other processes
        # (dashboard ingestion), so caching the client is safe.
        vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        
        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )
        
        # BM25Indexer just holds the index dir path; the SparseRetriever
        # calls _ensure_index_loaded() on every search, which always
        # reloads from disk — so it picks up dashboard-written data.
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        
        query_processor = QueryProcessor()
        self._hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        
        self._current_collection = collection
        self._initialized = True
        logger.info(f"Query components initialized for collection: {collection}")
    
    async def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        agentic: str = "auto",
    ) -> MCPToolResponse:
        """Execute the query_knowledge_hub tool.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            collection: Target collection name.
            agentic: Routing mode — "auto" (smart routing), "off" (traditional),
                     "on" (force multi-step iterative retrieval).

        Returns:
            MCPToolResponse with formatted content and citations.

        Raises:
            ValueError: If query is empty or invalid.
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Apply defaults
        effective_top_k = min(
            top_k or self.config.default_top_k,
            self.config.max_top_k
        )
        effective_collection = collection or self.config.default_collection

        logger.info(
            f"Executing query_knowledge_hub: query='{query[:50]}...', "
            f"top_k={effective_top_k}, collection={effective_collection}, "
            f"agentic={agentic}"
        )

        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["top_k"] = effective_top_k
        trace.metadata["collection"] = effective_collection
        trace.metadata["agentic"] = agentic
        trace.metadata["source"] = "mcp"

        try:
            import time as _time
            # Initialize components for collection
            _init_t0 = _time.monotonic()
            await asyncio.to_thread(self._ensure_initialized, effective_collection)
            _init_elapsed = (_time.monotonic() - _init_t0) * 1000.0
            trace.record_stage("initialization", {
                "collection": effective_collection,
                "cold_start": _init_elapsed > 500,
            }, elapsed_ms=_init_elapsed)

            # ── Agentic routing ──────────────────────────────────────────
            should_use_agentic = agentic == "on"

            if agentic == "auto":
                _t0 = _time.monotonic()
                should_use_agentic = await asyncio.to_thread(
                    self._complexity_check, query
                )
                _elapsed = (_time.monotonic() - _t0) * 1000.0
                trace.record_stage("complexity_check", {
                    "is_complex": should_use_agentic,
                }, elapsed_ms=_elapsed)
                logger.info(
                    f"Complexity check: complex={should_use_agentic}"
                    f" ({_elapsed:.0f}ms)"
                )

            if should_use_agentic:
                response = await self._agentic_search(
                    query=query,
                    top_k=effective_top_k,
                    collection=effective_collection,
                    trace=trace,
                )
                response.metadata["agentic_routing"] = "agentic"
            else:
                response = await self._traditional_search(
                    query=query,
                    top_k=effective_top_k,
                    collection=effective_collection,
                    trace=trace,
                )
                response.metadata["agentic_routing"] = "traditional"
            # ─────────────────────────────────────────────────────────────

            logger.info(
                f"query_knowledge_hub completed: "
                f"agentic={should_use_agentic}, "
                f"is_empty={response.is_empty}"
            )

            TraceCollector().collect(trace)
            return response

        except Exception as e:
            logger.exception(f"query_knowledge_hub failed: {e}")
            TraceCollector().collect(trace)
            return self._build_error_response(query, effective_collection, str(e))
    
    def _perform_search(
        self,
        query: str,
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid search.
        
        Args:
            query: Search query.
            top_k: Maximum results.
            trace: Optional TraceContext for observability.
            
        Returns:
            List of RetrievalResult.
        """
        if self._hybrid_search is None:
            raise RuntimeError("HybridSearch not initialized")
        
        # Use a larger initial retrieval for reranking
        initial_top_k = top_k * 2 if self.config.enable_rerank else top_k
        
        try:
            results = self._hybrid_search.search(
                query=query,
                top_k=initial_top_k,
                filters=None,
                trace=trace,
                return_details=False,
            )
            return results if isinstance(results, list) else results.results
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            return []
    
    def _apply_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Apply reranking to search results.
        
        Args:
            query: Original query.
            results: Search results to rerank.
            top_k: Final number of results.
            trace: Optional TraceContext for observability.
            
        Returns:
            Reranked results (or original if reranking fails).
        """
        if self._reranker is None or not self._reranker.is_enabled:
            return results[:top_k]
        
        try:
            rerank_result = self._reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
                trace=trace,
            )
            
            if rerank_result.used_fallback:
                logger.warning(
                    f"Reranker fallback: {rerank_result.fallback_reason}"
                )
            
            return rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results[:top_k]
    
    def _build_error_response(
        self,
        query: str,
        collection: str,
        error_message: str,
    ) -> MCPToolResponse:
        """Build error response.
        
        Args:
            query: Original query.
            collection: Target collection.
            error_message: Error description.
            
        Returns:
            MCPToolResponse indicating error.
        """
        content = f"## 查询失败\n\n"
        content += f"查询: **{query}**\n"
        content += f"集合: `{collection}`\n\n"
        content += f"**错误信息:** {error_message}\n\n"
        content += "请检查:\n"
        content += "- 数据库连接是否正常\n"
        content += "- 集合是否已创建并包含数据\n"
        content += "- 配置文件是否正确\n"
        
        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={
                "query": query,
                "collection": collection,
                "error": error_message,
            },
            is_empty=True,
        )

    # ── Agentic routing helpers ──────────────────────────────────────────

    def _get_llm(self):
        """Get or create an LLM client for complexity check + decomposition.

        Cached per instance to avoid repeated factory calls.
        """
        if not hasattr(self, "_llm_client") or self._llm_client is None:
            from src.libs.llm.llm_factory import LLMFactory
            self._llm_client = LLMFactory.create(self.settings)
        return self._llm_client

    def _complexity_check(self, query: str) -> bool:
        """Quick LLM-based complexity check for routing decisions.

        Returns True if the question is complex (needs multi-hop), False
        if it's a simple single-shot query. Uses a minimal prompt (~50
        output tokens) to keep latency and cost low.

        Args:
            query: User's question.

        Returns:
            True for complex questions, False for simple ones.
        """
        from src.libs.llm.base_llm import Message
        llm = self._get_llm()

        prompt = (
            "判断以下问题是简单还是复杂。\n"
            "简单 = 单一事实查询，一次检索即可回答，如'什么是X'、'怎么配置Y'。\n"
            "复杂 = 需要对比多个信息源、多步推理、因果分析，如'对比A和B的差异'、"
            "'为什么X导致Y'、'综合多个来源分析Z'。\n\n"
            f"问题：{query}\n\n"
            "只回答一个字：简单 或 复杂。"
        )

        try:
            response = llm.chat([
                Message(role="user", content=prompt),
            ])
            result = response.content.strip()
            is_complex = "复杂" in result
            logger.info(
                f"Complexity check: '{query[:60]}...' → "
                f"'{result[:20]}' → complex={is_complex}"
            )
            return is_complex
        except Exception as e:
            logger.warning(f"Complexity check LLM call failed: {e}, "
                           "falling back to traditional search")
            return False

    async def _traditional_search(
        self,
        query: str,
        top_k: int,
        collection: str,
        trace: Any,
    ) -> MCPToolResponse:
        """Traditional single-shot hybrid search (existing behavior).

        Args:
            query: Search query.
            top_k: Maximum results.
            collection: Target collection.
            trace: TraceContext for observability.

        Returns:
            MCPToolResponse with formatted content and citations.
        """
        # Perform hybrid search (blocking: embedding API + DB queries)
        results = await asyncio.to_thread(
            self._perform_search, query, top_k * 2, trace,
        )

        # Apply reranking if enabled (may call LLM API)
        if self.config.enable_rerank and results:
            results = await asyncio.to_thread(
                self._apply_rerank, query, results, top_k, trace,
            )

        # Build response
        response = self._response_builder.build(
            results=results[:top_k],
            query=query,
            collection=collection,
        )

        # Store final results in trace for dashboard display
        trace.metadata["final_results"] = [
            {
                "chunk_id": r.chunk_id,
                "score": round(r.score, 4),
                "text": r.text or "",
                "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                "title": r.metadata.get("title", ""),
            }
            for r in results[:top_k]
        ]

        return response

    async def _agentic_search(
        self,
        query: str,
        top_k: int,
        collection: str,
        trace: Any,
    ) -> MCPToolResponse:
        """Multi-hop iterative search for complex questions.

        Decomposes the query, searches each sub-query, evaluates
        coverage, and optionally performs additional rounds if
        the LLM determines information is insufficient.

        Limited to ``max_agentic_rounds`` decomposition rounds for
        latency control. Each round: decompose → parallel search →
        evaluate coverage → merge.

        Args:
            query: Complex user question.
            top_k: Maximum total results.
            collection: Target collection.
            trace: TraceContext for observability.

        Returns:
            MCPToolResponse with merged results and citations.
        """
        from src.libs.llm.base_llm import Message
        llm = self._get_llm()
        all_results: Dict[str, RetrievalResult] = {}  # chunk_id → result
        search_history: List[str] = []

        # Round 0: decompose original question
        decompose_prompt = (
            "你是一个查询分析专家。将下面的复杂问题分解为 2-4 个独立的检索子问题，"
            "每个子问题应该简洁、独立，适合在知识库中搜索。\n\n"
            f"问题：{query}\n\n"
            "只输出子问题，每行一个，不要编号。"
        )

        try:
            decompose_resp = llm.chat([
                Message(role="user", content=decompose_prompt),
            ])
            sub_queries = [
                q.strip() for q in decompose_resp.content.strip().split("\n")
                if q.strip() and len(q.strip()) > 2
            ]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, "
                           "falling back to original query")
            sub_queries = [query]

        if not sub_queries:
            sub_queries = [query]

        sub_queries = sub_queries[:4]  # max 4 sub-queries
        logger.info(f"Agentic search: decomposed into {len(sub_queries)} sub-queries")

        for round_num in range(self.config.max_agentic_rounds):
            round_queries = sub_queries if round_num == 0 else []

            # If not first round, ask LLM if more searching is needed
            if round_num > 0:
                coverage_prompt = (
                    f"原始问题：{query}\n\n"
                    f"已检索的信息：\n{_summarize_for_coverage(all_results.values())}\n\n"
                    "判断：以上信息是否足以完整回答问题?"
                    "如果足够，回复'足够'。"
                    "如果不够，给出 1-2 个新的检索子问题(每行一个)。"
                )
                try:
                    coverage_resp = llm.chat([
                        Message(role="user", content=coverage_prompt),
                    ])
                    coverage_text = coverage_resp.content.strip()
                    if "足够" in coverage_text and "不" not in coverage_text:
                        logger.info(
                            f"Agentic search: coverage sufficient after "
                            f"round {round_num}"
                        )
                        break
                    round_queries = [
                        q.strip() for q in coverage_text.split("\n")
                        if q.strip() and len(q.strip()) > 2
                        and q.strip() not in search_history
                    ][:2]
                except Exception as e:
                    logger.warning(f"Coverage check failed: {e}")
                    break

            if not round_queries:
                break

            # Parallel search for this round's sub-queries
            for sub_q in round_queries:
                if sub_q in search_history:
                    continue
                search_history.append(sub_q)

                try:
                    results = await asyncio.to_thread(
                        self._perform_search, sub_q, top_k, trace,
                    )
                    for r in results:
                        if r.chunk_id not in all_results:
                            all_results[r.chunk_id] = r
                except Exception as e:
                    logger.warning(f"Sub-query search failed: '{sub_q}': {e}")
                    continue

            logger.info(
                f"Agentic search round {round_num + 1}: "
                f"{len(round_queries)} sub-queries, "
                f"{len(all_results)} unique results so far"
            )

        # Merge and rerank all collected results
        merged = sorted(
            all_results.values(),
            key=lambda r: r.score,
            reverse=True,
        )

        # Apply reranking if enabled
        if self.config.enable_rerank and merged:
            merged = await asyncio.to_thread(
                self._apply_rerank, query, merged, top_k, trace,
            )

        trace.metadata["agentic_rounds"] = round_num + 1
        trace.metadata["agentic_sub_queries"] = len(search_history)

        # Build response
        response = self._response_builder.build(
            results=merged[:top_k],
            query=query,
            collection=collection,
        )

        # Store final results in trace
        trace.metadata["final_results"] = [
            {
                "chunk_id": r.chunk_id,
                "score": round(r.score, 4),
                "text": r.text or "",
                "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                "title": r.metadata.get("title", ""),
            }
            for r in merged[:top_k]
        ]

        return response


def _summarize_for_coverage(
    results: Any,  # Iterable[RetrievalResult]
    max_items: int = 6,
) -> str:
    """Build a short summary of retrieved results for coverage check."""
    items = list(results)[:max_items]
    lines = []
    for i, r in enumerate(items, 1):
        text = (r.text or "")[:120].replace("\n", " ")
        source = r.metadata.get("source_path", r.metadata.get("source", "?"))
        lines.append(f"[{i}] {text}... (来源: {source})")
    if len(list(results)) > max_items:
        lines.append(f"... 还有 {len(list(results)) - max_items} 条结果")
    return "\n".join(lines) if lines else "(暂无检索结果)"


# Module-level tool instance (lazy-initialized)
_tool_instance: Optional[QueryKnowledgeHubTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> QueryKnowledgeHubTool:
    """Get or create the tool instance.
    
    Args:
        settings: Optional settings to use for initialization.
        
    Returns:
        QueryKnowledgeHubTool instance.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = QueryKnowledgeHubTool(settings=settings)
    return _tool_instance


async def query_knowledge_hub_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
    agentic: str = "auto",
) -> types.CallToolResult:
    """Handler function for MCP tool registration.

    This function is registered with the ProtocolHandler and called
    when the MCP client invokes the query_knowledge_hub tool.

    Supports multimodal responses - if search results contain images,
    the response will include ImageContent blocks alongside TextContent.

    Args:
        query: Search query string.
        top_k: Maximum number of results.
        collection: Optional collection name.
        agentic: Routing mode — "auto" (smart routing), "off" (traditional),
                 "on" (force multi-step).

    Returns:
        MCP CallToolResult with content blocks (text and optionally images).
    """
    tool = get_tool_instance()
    
    try:
        response = await tool.execute(
            query=query,
            top_k=top_k,
            collection=collection,
            agentic=agentic,
        )
        
        # Use to_mcp_content() which handles multimodal (text + images)
        content_blocks = response.to_mcp_content()
        
        return types.CallToolResult(
            content=content_blocks,
            isError=response.is_empty and "error" in response.metadata,
        )
        
    except ValueError as e:
        # Invalid parameters
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"参数错误: {e}",
                )
            ],
            isError=True,
        )
    except Exception as e:
        # Internal error
        logger.exception(f"query_knowledge_hub handler error: {e}")
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"内部错误: 查询处理失败",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register query_knowledge_hub tool with the protocol handler.
    
    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=query_knowledge_hub_handler,
    )
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
