# FastAPI + Vue3 (Sakura UI) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** 新增 REST API 层和 Sakura 主题 Vue3 前端，提供一个完整的产品级 RAG Web 应用。

**Architecture:** FastAPI 后端 (`src/api/`) 通过 HTTP/WS 暴露核心功能，Vue3 前端 (`frontend/`) 通过 Vite 独立开发运行。后端全部复用现有 `src/core/` + `src/agentic/`，零侵入。

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, Pydantic v2, Vue 3 + Composition API, TypeScript, Vite, Vue Router 4, axios, Tailwind CSS, lucide-vue-next

## Global Constraints

- 不修改 `src/mcp_server/`、`src/core/`、`src/agentic/` 任何文件
- FastAPI 端口: 8000
- Vue3 开发服务器: 5173
- CORS 允许 origin: `http://localhost:5173`
- LLM 调用全部通过 `asyncio.to_thread()` 避免阻塞
- 文件上传使用 FastAPI `UploadFile` 流式处理
- 前端暂不要求自动化测试
- API 测试使用 `httpx` + FastAPI `TestClient`

---

### Task 1: FastAPI 骨架 + Schemas

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/server.py`
- Create: `src/api/schemas.py`
- Create: `tests/unit/api/__init__.py`
- Create: `tests/unit/api/test_server.py`

**Interfaces:**
- Produces: FastAPI app, Pydantic request/response models

- [ ] **Step 1: Install deps**

```bash
pip install fastapi uvicorn httpx python-multipart
```

- [ ] **Step 2: Create schemas**

`src/api/schemas.py`:

```python
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
```

- [ ] **Step 3: Create server skeleton**

`src/api/server.py`:

```python
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


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Test**

`tests/unit/api/test_server.py`:

```python
"""Smoke test for FastAPI server."""

from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

- [ ] **Step 5: Run + Commit**

```bash
pytest tests/unit/api/test_server.py -v
git add src/api/ tests/unit/api/
git commit -m "feat: add FastAPI server skeleton + Pydantic schemas"
```

---

### Task 2: Query Router (传统 + Agentic)

**Files:**
- Create: `src/api/routers/__init__.py`
- Create: `src/api/routers/query.py`
- Create: `tests/unit/api/test_query_router.py`

- [ ] **Step 1: Write router**

`src/api/routers/query.py`:

```python
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

        return AgenticQueryResponse(
            answer=result.answer,
            citations=[],  # citation tracker doesn't produce structured items — use answer text references
            trace=trace,
            total_rounds=len(result.trace),
            mode="agentic",
            elapsed_ms=round(elapsed_ms, 1),
        )
    finally:
        await client.close()
```

- [ ] **Step 2: Register in server.py**

```python
from src.api.routers.query import router as query_router
app.include_router(query_router)
```

- [ ] **Step 3: Write test**

`tests/unit/api/test_query_router.py`:

```python
"""Tests for query endpoints using TestClient + mocked internals."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


@patch("src.api.routers.query.QueryKnowledgeHubTool")
def test_query_endpoint(mock_tool_cls):
    mock_tool = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "检索结果"
    mock_response.citations = []
    mock_response.is_empty = False
    mock_tool.execute = AsyncMock(return_value=mock_response)
    mock_tool_cls.return_value = mock_tool

    resp = client.post("/api/query", json={"query": "test", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "traditional"
    assert "answer" in data
```

- [ ] **Step 4: Run + Commit**

```bash
pytest tests/unit/api/test_query_router.py -v
git add src/api/routers/ tests/unit/api/
git commit -m "feat: add query router — traditional + agentic endpoints"
```

---

### Task 3: Collections + Documents + Traces Routers

**Files:**
- Create: `src/api/routers/collections.py`
- Create: `src/api/routers/documents.py`
- Create: `src/api/routers/traces.py`

一次性实现三个简单 CRUD 路由，每个都复用现有 MCP 工具。

`src/api/routers/collections.py`:

```python
from fastapi import APIRouter
from src.api.schemas import CollectionListResponse, CollectionInfo

router = APIRouter(prefix="/api", tags=["collections"])

@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    from src.core.settings import load_settings
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    settings = load_settings()
    store = VectorStoreFactory.create(settings)
    # ChromaDB list_collections
    names = store.client.list_collections() if hasattr(store, 'client') else []
    collections = [CollectionInfo(name=str(n), document_count=0) for n in names]
    return CollectionListResponse(collections=collections)
```

`src/api/routers/documents.py`:

```python
from fastapi import APIRouter, UploadFile, File, Form
from src.api.schemas import UploadResponse, DocumentListResponse, DocumentInfo, DeleteResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), collection: str = Form("default")):
    import tempfile, os
    from src.ingestion.pipeline import IngestionPipeline
    from src.core.settings import load_settings

    settings = load_settings()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = IngestionPipeline(settings=settings)
        result = await pipeline.ingest_file(tmp_path, collection_name=collection)
        return UploadResponse(
            doc_id=result.doc_id or os.path.basename(tmp_path),
            filename=file.filename or "unknown",
            chunks=result.chunk_count,
            status="ingested",
        )
    finally:
        os.unlink(tmp_path)


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    from src.ingestion.storage.ingestion_history import IngestionHistory
    history = IngestionHistory()
    docs = history.list_all()
    return DocumentListResponse(documents=[
        DocumentInfo(doc_id=d.doc_id, filename=d.filename or d.doc_id, chunks=d.chunk_count, ingested_at=d.ingested_at or "")
        for d in docs
    ])


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    from src.ingestion.storage.document_manager import DocumentManager
    from src.core.settings import load_settings

    settings = load_settings()
    mgr = DocumentManager(settings=settings)
    await asyncio.to_thread(mgr.delete, doc_id)
    return DeleteResponse(doc_id=doc_id, status="deleted")
```

`src/api/routers/traces.py`:

```python
from fastapi import APIRouter, Query
from src.api.schemas import TraceListResponse, TraceInfo

router = APIRouter(prefix="/api/traces", tags=["traces"])

@router.get("/query", response_model=TraceListResponse)
async def list_query_traces(limit: int = Query(default=50, le=200)):
    import json, os
    trace_file = "logs/traces.jsonl"
    traces = []
    if os.path.exists(trace_file):
        with open(trace_file, encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                try:
                    data = json.loads(line)
                    if data.get("trace_type") == "query":
                        traces.append(TraceInfo(
                            trace_id=data.get("trace_id", ""),
                            trace_type="query",
                            metadata=data.get("metadata", {}),
                            stages=data.get("stages", []),
                            elapsed_ms=data.get("elapsed_ms", 0),
                        ))
                except json.JSONDecodeError:
                    continue
    return TraceListResponse(traces=traces)


@router.get("/ingestion", response_model=TraceListResponse)
async def list_ingestion_traces(limit: int = Query(default=20, le=200)):
    import json, os
    trace_file = "logs/traces.jsonl"
    traces = []
    if os.path.exists(trace_file):
        with open(trace_file, encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                try:
                    data = json.loads(line)
                    if data.get("trace_type") == "ingestion":
                        traces.append(TraceInfo(
                            trace_id=data.get("trace_id", ""),
                            trace_type="ingestion",
                            metadata=data.get("metadata", {}),
                            stages=data.get("stages", []),
                            elapsed_ms=data.get("elapsed_ms", 0),
                        ))
                except json.JSONDecodeError:
                    continue
    return TraceListResponse(traces=traces)
```

Register in `server.py`:
```python
from src.api.routers.collections import router as collections_router
from src.api.routers.documents import router as documents_router
from src.api.routers.traces import router as traces_router
app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(traces_router)
```

- [ ] **Run + Commit**

```bash
git add src/api/routers/
git commit -m "feat: add collections, documents, traces routers"
```

---

### Task 4: Vue3 项目脚手架 + Sakura 主题

**Files:**
- Create: `frontend/` (Vite + Vue3 + TypeScript scaffold)

- [ ] **Step 1: Scaffold project**

```bash
npm create vite@latest frontend -- --template vue-ts
cd frontend
npm install
npm install vue-router@4 axios lucide-vue-next
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Configure Tailwind with Sakura palette**

`tailwind.config.js`:

```js
export default {
  content: ["./index.html", "./src/**/*.{vue,ts,js}"],
  theme: {
    extend: {
      colors: {
        sakura: {
          pink: '#FF91A4',
          light: '#FFB6C1',
          bg: '#FFF5F7',
          text: '#4A3040',
          muted: '#C9A0B0',
          accent: '#FF85A2',
        }
      },
      borderRadius: {
        'xl': '16px',
        '2xl': '20px',
      },
      boxShadow: {
        'sakura': '0 4px 15px rgba(255, 145, 164, 0.2)',
      },
      fontFamily: {
        sans: ['"Nunito"', '"PingFang SC"', '"Microsoft YaHei"', 'sans-serif'],
      }
    }
  }
}
```

- [ ] **Step 3: Global CSS**

`frontend/src/style.css`:

```css
@import "tailwindcss";

body {
  background-color: #FFF5F7;
  color: #4A3040;
  font-family: 'Nunito', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

/* 自定义滚动条 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #FFF5F7; }
::-webkit-scrollbar-thumb { background: #FFB6C1; border-radius: 3px; }

/* 选中颜色 */
::selection { background: #FFB6C1; color: white; }
```

- [ ] **Step 4: Setup Vue Router**

`frontend/src/router/index.ts`:

```typescript
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', name: 'chat', component: () => import('../views/ChatView.vue') },
    { path: '/documents', name: 'documents', component: () => import('../views/DocumentsView.vue') },
    { path: '/traces', name: 'traces', component: () => import('../views/TracesView.vue') },
    { path: '/evaluation', name: 'evaluation', component: () => import('../views/EvaluationView.vue') },
  ]
})

export default router
```

- [ ] **Step 5: App.vue + NavBar**

`frontend/src/App.vue`:

```vue
<script setup lang="ts">
import NavBar from './components/NavBar.vue'
</script>

<template>
  <div class="min-h-screen bg-sakura-bg">
    <NavBar />
    <main class="max-w-5xl mx-auto px-4 py-6">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>
```

`frontend/src/components/NavBar.vue`:

```vue
<script setup lang="ts">
import { useRoute } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()
const links = [
  { to: '/chat', label: '💬 Chat', icon: '🌸' },
  { to: '/documents', label: '📁 文档', icon: '📚' },
  { to: '/traces', label: '🔍 追踪', icon: '🐾' },
  { to: '/evaluation', label: '📊 评估', icon: '⭐' },
]
</script>

<template>
  <nav class="bg-white/80 backdrop-blur-sm border-b border-sakura-light/30 sticky top-0 z-50">
    <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
      <router-link to="/chat" class="flex items-center gap-2 text-sakura-pink font-bold text-lg no-underline">
        <span>🌸</span>
        <span>Sakura RAG</span>
      </router-link>
      <div class="flex gap-1">
        <router-link
          v-for="link in links"
          :key="link.to"
          :to="link.to"
          class="px-3 py-1.5 rounded-xl text-sm transition-all duration-200 no-underline"
          :class="route.path === link.to
            ? 'bg-sakura-pink text-white'
            : 'text-sakura-muted hover:bg-sakura-light/20 hover:text-sakura-text'"
        >
          {{ link.icon }} {{ link.label }}
        </router-link>
      </div>
    </div>
  </nav>
</template>
```

- [ ] **Step 6: Create API wrapper**

`frontend/src/api/index.ts`:

```typescript
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 60000,
})

export async function query(params: { query: string; top_k?: number; collection?: string }) {
  const { data } = await api.post('/query', params)
  return data
}

export async function agenticQuery(params: { query: string; max_rounds?: number; collection?: string }) {
  const { data } = await api.post('/agentic-query', params)
  return data
}

export async function listCollections() {
  const { data } = await api.get('/collections')
  return data
}

export async function uploadDocument(file: File, collection: string = 'default') {
  const form = new FormData()
  form.append('file', file)
  form.append('collection', collection)
  const { data } = await api.post('/documents/upload', form)
  return data
}

export async function listDocuments() {
  const { data } = await api.get('/documents')
  return data
}

export async function deleteDocument(docId: string) {
  const { data } = await api.delete(`/documents/${docId}`)
  return data
}

export async function listQueryTraces(limit: number = 50) {
  const { data } = await api.get('/traces/query', { params: { limit } })
  return data
}

export async function listIngestionTraces(limit: number = 20) {
  const { data } = await api.get('/traces/ingestion', { params: { limit } })
  return data
}
```

- [ ] **Step 7: Commit**

```bash
cd frontend && git add -A && git commit -m "feat: scaffold Vue3 project with Sakura theme, router, API layer"
```

---

### Task 5: ChatView + 核心组件

**Files:**
- Create: `frontend/src/views/ChatView.vue`
- Create: `frontend/src/components/ChatMessage.vue`
- Create: `frontend/src/components/CitationCard.vue`
- Create: `frontend/src/components/QueryInput.vue`
- Create: `frontend/src/components/AgenticTrace.vue`

核心实现：消息列表 + 发送查询 + 显示结果 + Agentic 推理展开。

`ChatView.vue` 关键逻辑：

```vue
<script setup lang="ts">
import { ref } from 'vue'
import { query, agenticQuery } from '../api'
import ChatMessage from '../components/ChatMessage.vue'
import QueryInput from '../components/QueryInput.vue'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: any[]
  trace?: any[]
  mode?: string
  loading?: boolean
}

const messages = ref<Message[]>([])
const agenticMode = ref(false)
const loading = ref(false)

async function sendQuery(text: string) {
  messages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'assistant', content: '', loading: true })
  loading.value = true

  try {
    const fn = agenticMode.value ? agenticQuery : query
    const result = await fn({ query: text })

    const idx = messages.value.length - 1
    messages.value[idx] = {
      role: 'assistant',
      content: result.answer,
      citations: result.citations,
      trace: result.trace,
      mode: result.mode,
    }
  } catch (e: any) {
    messages.value[messages.value.length - 1] = {
      role: 'assistant',
      content: `❌ 出错了: ${e.message}`,
    }
  } finally {
    loading.value = false
  }
}
</script>
```

`ChatMessage.vue` 气泡样式：

```vue
<template>
  <div class="flex gap-3 mb-4" :class="msg.role === 'user' ? 'flex-row-reverse' : ''">
    <div class="w-8 h-8 rounded-full flex items-center justify-center text-lg shrink-0"
         :class="msg.role === 'user' ? 'bg-sakura-light' : 'bg-sakura-pink'">
      {{ msg.role === 'user' ? '👤' : '🤖' }}
    </div>
    <div class="max-w-[75%] rounded-2xl px-4 py-3 shadow-sakura"
         :class="msg.role === 'user'
           ? 'bg-white text-sakura-text rounded-tr-md'
           : 'bg-sakura-pink/10 text-sakura-text rounded-tl-md'">
      <!-- loading -->
      <div v-if="msg.loading" class="flex gap-1 py-2">
        <span class="w-2 h-2 bg-sakura-pink rounded-full animate-bounce" style="animation-delay: 0s"></span>
        <span class="w-2 h-2 bg-sakura-pink rounded-full animate-bounce" style="animation-delay: 0.15s"></span>
        <span class="w-2 h-2 bg-sakura-pink rounded-full animate-bounce" style="animation-delay: 0.3s"></span>
      </div>
      <!-- markdown content -->
      <div v-else class="prose prose-sm max-w-none" v-html="renderedContent"></div>
      <!-- citations -->
      <CitationCard v-if="msg.citations?.length" :citations="msg.citations" />
      <!-- agentic trace -->
      <AgenticTrace v-if="msg.trace?.length" :trace="msg.trace" />
    </div>
  </div>
</template>
```

- [ ] **Commit**

```bash
git add frontend/src/views/ChatView.vue frontend/src/components/
git commit -m "feat: add ChatView with agentic toggle, citations, trace expand"
```

---

### Task 6: DocumentsView + TracesView + EvaluationView

**Files:**
- Create: `frontend/src/views/DocumentsView.vue`
- Create: `frontend/src/views/TracesView.vue`
- Create: `frontend/src/views/EvaluationView.vue`
- Create: `frontend/src/components/DocUploader.vue`
- Create: `frontend/src/components/TraceTimeline.vue`

3 个辅助页面，功能简单：列表 + 基础操作。

- `DocumentsView` — 拖拽上传区域 + 文档卡片列表 + 删除按钮，SakuraCard 样式
- `TracesView` — 两个 Tab (Query / Ingestion)，TraceTimeline 展示各阶段
- `EvaluationView` — 指标卡片（Hit Rate、MRR、Faithfulness），从配置读取

- [ ] **Commit**

```bash
git add frontend/src/views/DocumentsView.vue frontend/src/views/TracesView.vue frontend/src/views/EvaluationView.vue frontend/src/components/DocUploader.vue frontend/src/components/TraceTimeline.vue
git commit -m "feat: add Documents, Traces, Evaluation views with Sakura styling"
```

---

### Task 7: 集成测试 + 全量回归

- [ ] **Step 1: 启动 FastAPI 集成测试**

```bash
# 启动 Server（后台）
uvicorn src.api.server:app --port 8000 &
# 测试全部 API 端点
pytest tests/unit/api/ -v
# 清理
kill %1
```

- [ ] **Step 2: 确认 Vue3 可启动**

```bash
cd frontend && npm run dev
# 手动验证 localhost:5173 可以加载
```

- [ ] **Step 3: 全量单元测试 + 确认零回归**

```bash
pytest tests/unit/ -v -m "not llm and not slow"
```

- [ ] **Step 4: 最终提交**

```bash
git add -A && git commit -m "chore: finalize FastAPI + Vue3 Sakura — all tests passing"
```

---

## 运行方式总结

```bash
# 终端 1: FastAPI
uvicorn src.api.server:app --reload --port 8000

# 终端 2: Vue3
cd frontend && npm run dev

# 终端 3 (可选): MCP Server
python -m src.mcp_server.server

# 终端 4 (可选): Streamlit
streamlit run src/observability/dashboard/app.py
```

浏览器打开 `http://localhost:5173` 即可使用 Sakura RAG 界面。
