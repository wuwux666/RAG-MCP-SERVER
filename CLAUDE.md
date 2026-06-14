# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all unit tests (fast, no external deps)
pytest tests/unit/ -v

# Run tests excluding slow/LLM tests
pytest -m "not llm and not slow" -v

# Run a single test file
pytest tests/unit/test_hybrid_search.py -v

# Run a single test function
pytest tests/unit/test_hybrid_search.py::test_fusion_with_two_lists -v

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Ingest a document
python scripts/ingest.py --path data/documents/your_file.pdf --collection default

# Query from CLI
python scripts/query.py --query "your question" --top-k 5 --collection default

# Start Dashboard (browser at localhost:8501)
streamlit run src/observability/dashboard/app.py

# Start MCP Server (for Agent connection — hangs waiting for stdin, that's normal)
python -m src.mcp_server.server

# Run evaluation
python scripts/evaluate.py
```

## Architecture

This is a **modular RAG system exposed as an MCP server** over stdio transport. All backends (LLM, Embedding, Reranker, VectorStore, Splitter, Evaluator) are pluggable via factory patterns — change providers by editing `config/settings.yaml`, zero code changes.

### Three-layer architecture

| Layer | Path | Role |
|-------|------|------|
| MCP Protocol | `src/mcp_server/` | JSON-RPC over stdio, exposes 3 tools to AI agents |
| Core Business | `src/core/` | Query engine, response builder, types, settings, traces |
| Pluggable Backends | `src/libs/` | LLM, Embedding, Reranker, VectorStore, Splitter, Loader, Evaluator |

### Two main data flows

**Ingestion (6-stage pipeline, entry: `src/ingestion/pipeline.py`):**
`PDF → Load(MarkItDown+PyMuPDF) → Split(LangChain RecursiveCharacterTextSplitter) → Transform(Refine/Enrich/Caption, each with LLM fallback to rule-based) → Dual Encode(Dense+Sparse) → Store(ChromaDB+BM25 JSON+SQLite)`

**Query (entry: `src/mcp_server/tools/query_knowledge_hub.py`):**
`QueryProcessor(jieba tokenize) → Parallel{DenseRetriever + SparseRetriever} → RRFFusion(k=60) → CoreReranker(optional, with fallback) → ResponseBuilder(citations+images)`

### Key design patterns

- **Factory pattern everywhere**: All 6 provider families use the same pattern — `_PROVIDERS` registry dict, `register_provider()`, `create(settings)`. To add a new provider: inherit the Base class → implement interface → register in Factory. No existing code changes needed.
- **Graceful degradation**: LLM failures in transform stages fall back to rule-based processing. HybridSearch falls back to single retriever if the other fails. Reranker failure returns original ordering. Only dual-retriever failure raises an error.
- **Idempotent operations**: SHA256 file dedup (`ingestion_history.db`), deterministic chunk IDs via `hash(source_path + index + content_hash)`, upsert semantics in ChromaDB.
- **Four persistent stores** (all local, no external DB servers): ChromaDB (vectors+metadata), BM25 JSON files (inverted indexes), SQLite `ingestion_history.db` (SHA256 records), SQLite `image_index.db` (image path mapping).
- **Stdio isolation**: stdout is reserved for JSON-RPC protocol messages; all logging goes exclusively to stderr. Heavy imports (chromadb) are preloaded in the main thread to prevent import-lock deadlocks with asyncio worker threads.

### Configuration is the single source of truth

`config/settings.yaml` drives everything. Key gotchas for new users:
- `vision_llm` section **must** include an `enabled: true/false` field, otherwise `Settings.from_dict()` raises `Missing required field: vision_llm.enabled`
- `embedding.provider` only supports `openai`, `azure`, `ollama` — third-party OpenAI-compatible APIs (Qwen DashScope, etc.) use `provider: "openai"` with a custom `base_url`
- `rerank.enabled: false` and `evaluation.enabled: false` by default

### Core types (lingua franca of the entire codebase)

`src/core/types.py` defines `Document` → `Chunk` → `ChunkRecord` → `RetrievalResult` → `ProcessedQuery`. Every layer imports from here; all data flows through these dataclasses.

### MCP tools exposed

| Tool | Params | What it does |
|------|--------|-------------|
| `query_knowledge_hub` | `query`(req), `top_k`(opt,5), `collection`(opt) | Full hybrid search pipeline |
| `list_collections` | `include_stats`(opt,true) | Lists ChromaDB collections |
| `get_document_summary` | `doc_id`(req), `collection`(opt) | Document metadata + preview |

### Test organization (68 files)

- `tests/unit/` — 51 files. All external deps mocked. Fast, no network.
- `tests/integration/` — 13 files. Real ChromaDB, PDF loading, MCP subprocess.
- `tests/e2e/` — 4 files. Full MCP client-server roundtrip.
- Pytest markers: `unit`, `integration`, `e2e`, `llm` (needs API key), `slow`.
- `tests/conftest.py` adds project root to `sys.path`. All `from src.xxx` imports work in tests without `pip install -e .`.

### Provider registration is static (not dynamic discovery)

New provider classes must be imported and registered in their factory's module-level `_register_builtin_providers()` function. There's no plugin auto-discovery — if you forget to add the registration call, the factory won't find the provider even if the class file exists.