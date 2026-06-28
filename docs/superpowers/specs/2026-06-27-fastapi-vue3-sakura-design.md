# FastAPI + Vue3 (Sakura UI) 设计文档

**日期**: 2026-06-27
**状态**: 待实现
**范围**: 新增 REST API 层和现代化 Web 前端，不改动现有代码

---

## 1. 目标与非目标

### 目标

- 提供 REST API，让 Web/移动客户端可通过 HTTP 调用 RAG 系统
- 提供 Agentic RAG 的 HTTP 端点，支持前端实时展示 ReAct 推理过程
- 用 Vue3 构建现代化管理界面，替代 Streamlit 作为主要人机交互界面
- 可爱少女风 UI (Sakura Theme)

### 非目标

- 不修改 `src/mcp_server/`、`src/core/`、`src/agentic/` 任何代码
- 不删除 Streamlit Dashboard（保留共存）
- 不做用户认证/多租户
- 不做 SSR/SEO

---

## 2. 架构

```
┌──────────────────────────────────────────┐
│  Vue3 前端 (frontend/)          :5173    │
│  Vite + TypeScript + Vue Router           │
│  Sakura 主题 (粉白 + 圆角 + 猫爪)         │
│                                            │
│  页面: /chat, /documents, /traces, /eval  │
└──────────────────┬───────────────────────┘
                   │ HTTP REST + WebSocket
                   │ CORS: localhost:5173
                   ▼
┌──────────────────────────────────────────┐
│  FastAPI 后端 (src/api/)        :8000    │
│                                            │
│  routers/: query, collections, docs,      │
│            traces                         │
│  websocket/: ingest_progress              │
│  schemas.py: Pydantic models              │
│                                            │
│  全部复用 src/core/ + src/agentic/        │
└──────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  src/core/ + src/agentic/ (不动)          │
└──────────────────────────────────────────┘
```

---

## 3. API 契约

### POST /api/query

```
Request:  { "query": "...", "top_k": 5, "collection": "default" }
Response: {
  "answer": "markdown content",
  "citations": [{ "index": 1, "source": "...", "score": 0.95, "text_snippet": "..." }],
  "images": [{ "image_id": "...", "base64": "..." }],
  "mode": "traditional",
  "elapsed_ms": 342
}
```

### POST /api/agentic-query

```
Request:  { "query": "...", "max_rounds": 10, "collection": "default" }
Response: {
  "answer": "综合答案",
  "citations": [...],
  "trace": [
    { "round": 1, "thought": "...", "action": "...", "observation": "..." }
  ],
  "total_rounds": 3,
  "mode": "agentic",
  "elapsed_ms": 2847
}
```

### POST /api/documents/upload

```
Request:  multipart/form-data { file: binary, collection: "default" }
Response: { "doc_id": "...", "filename": "...", "chunks": 15, "status": "ingested" }
```

### GET /api/documents

```
Response: { "documents": [{ "doc_id": "...", "filename": "...", "chunks": 15, "ingested_at": "..." }] }
```

### DELETE /api/documents/{doc_id}

```
Response: { "doc_id": "...", "status": "deleted" }
```

### GET /api/collections

```
Response: { "collections": [{ "name": "default", "document_count": 12 }] }
```

### GET /api/traces/query?limit=50

```
Response: { "traces": [{ "trace_id": "...", "query": "...", "stages": [...], "elapsed_ms": 342 }] }
```

### GET /api/traces/ingestion?limit=20

```
Response: { "traces": [{ "trace_id": "...", "filename": "...", "stages": [...], "elapsed_ms": 5234 }] }
```

### WS /ws/ingest-progress

```
Server → Client: { "stage": "loading", "progress": 0.1, "message": "解析 PDF..." }
Server → Client: { "stage": "splitting", "progress": 0.3, "message": "切分为 15 chunks" }
Server → Client: { "stage": "embedding", "progress": 0.7, "message": "向量化..." }
Server → Client: { "stage": "done", "progress": 1.0, "message": "完成", "doc_id": "abc123" }
```

---

## 4. 前端设计

### 4.1 技术选型

| 项 | 选择 |
|----|------|
| 框架 | Vue 3 + Composition API |
| 语言 | TypeScript |
| 构建 | Vite |
| 路由 | Vue Router 4 |
| HTTP | axios |
| UI 组件 | 手写 (Sakura Theme)，辅助 lucide-vue-next 图标 |
| CSS | Tailwind CSS (自定义 sakura 色板) 或纯 CSS 变量 |

### 4.2 Sakura 主题色板

```css
:root {
  --sakura-pink: #FF91A4;
  --sakura-light: #FFB6C1;
  --sakura-bg: #FFF5F7;
  --sakura-white: #FFFFFF;
  --sakura-text: #4A3040;
  --sakura-muted: #C9A0B0;
  --sakura-shadow: rgba(255, 145, 164, 0.2);
  --sakura-accent: #FF85A2;
  --radius-lg: 20px;
  --radius-md: 14px;
}
```

### 4.3 页面结构

```
/chat          → ChatView.vue         (默认页)
/documents     → DocumentsView.vue
/traces        → TracesView.vue
/evaluation    → EvaluationView.vue
```

### 4.4 ChatView 核心交互

```
┌──────────────────────────────────────────────────┐
│  🌸 Chat           [Agentic 🌸]  [普通模式]      │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │ 🤖 根据检索结果，项目使用了...               │  │
│  │    📎 [1] docs/arch.md  [2] docs/api.md     │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌──────────────────────────┐                    │
│  │             对比A和B？ 👤 │                    │
│  └──────────────────────────┘                    │
│                                                   │
│  [比较A和B的技术差异__________]  [🌸 发送]       │
└──────────────────────────────────────────────────┘
```

Agentic 模式开启后，消息下方展开 ReAct 步骤：

```
┌─────────────────────────────────────────────┐
│ 🤖 综合分析如下...                           │
│    📎 [1] doc-a.md  [2] doc-b.md            │
│                                              │
│  ▼ 推理过程                                  │
│  ┌──────────────────────────────────────┐   │
│  │ 🔄 Round 1: 检索"项目A技术栈" → 3条  │   │
│  │ 🔄 Round 2: 检索"项目B技术栈" → 2条  │   │
│  │ ✅ Final: 综合对比答案               │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 4.5 组件树

```
App.vue
├── NavBar.vue           (🌸 logo + 导航链接)
├── router-view
│   ├── ChatView.vue
│   │   ├── ChatMessage.vue     (消息气泡)
│   │   │   └── CitationCard.vue (引用卡片)
│   │   ├── AgenticTrace.vue    (推理步骤展开)
│   │   └── QueryInput.vue      (输入框 + toggle)
│   ├── DocumentsView.vue
│   │   ├── DocUploader.vue     (拖拽上传)
│   │   └── DocCard.vue         (文档卡片)
│   ├── TracesView.vue
│   │   └── TraceTimeline.vue   (追踪时间线)
│   └── EvaluationView.vue
│       └── MetricCard.vue      (指标卡片)
└── Footer.vue            (🌸 可爱页脚)

公共组件:
├── SakuraButton.vue      (粉色圆角按钮)
├── SakuraCard.vue         (粉色卡片容器)
├── SakuraInput.vue        (粉色输入框)
├── LoadingSpinner.vue     (樱花 loading)
└── Toast.vue              (粉色提示)
```

---

## 5. 测试策略

- FastAPI 端点用 `TestClient` (Starlette) 测试
- 集成测试用真实 Server 子进程
- 前端暂不做自动化测试（后续可加 Vitest + Playwright）

---

## 6. 风险

| 风险 | 缓解 |
|------|------|
| FastAPI 阻塞事件循环 | LLM/Embedding 调用通过 `asyncio.to_thread` |
| 文件上传大文件 OOM | FastAPI `UploadFile` 流式写入临时文件 |
| WebSocket 连接泄漏 | 超时自动断开 + 客户端心跳 |
| Vue3 构建产物体积 | Vite code-splitting + lazy import |
