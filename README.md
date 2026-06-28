# Modular RAG MCP Server

> 可插拔、可观测的模块化 RAG 系统。三种接入方式：Web 界面、REST API、MCP 协议——改一行 YAML 即可切换 LLM / Embedding / VectorStore 后端。

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/wuwux666/RAG-MCP-SERVER.git && cd MODULAR-RAG-MCP-SERVER
pip install -e ".[dev]"
```

### 2. 配置

编辑 `config/settings.yaml`，最少配 LLM + Embedding：

```yaml
llm:
  provider: "deepseek"
  model: "deepseek-v4-pro"
  api_key: "sk-xxx"

embedding:
  provider: "openai"
  model: "text-embedding-v3"
  api_key: "sk-xxx"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 3. 摄入文档

```bash
python scripts/ingest.py --file data/documents/your_doc.pdf --collection default
```

### 4. 启动服务

```bash
# Web 界面 + API（推荐）
uvicorn src.api.server:app --reload --port 8000
cd frontend && npm install && npm run dev       # 浏览器打开 localhost:5173

# 或 MCP Server（供 AI Agent 调用）
python -m src.mcp_server.server
```

---

## 三种接入方式

| 方式 | 入口 | 适用场景 |
|------|------|---------|
| 🌸 **Web 界面** | `localhost:5173` | 人在浏览器中搜索、管理文档、查看追踪 |
| 🔗 **REST API** | `localhost:8000/api/*` | 外部应用/脚本通过 HTTP 调用 |
| 🤖 **MCP Server** | stdio JSON-RPC | Claude Desktop / Copilot 等 AI Agent 直接调用 |

三种方式共享同一套检索核心，互不冲突，可同时运行。

### REST API 端点

```
POST /api/query           传统混合检索（快）
POST /api/agentic-query   ReAct 智能路由（复杂多跳问题）
GET  /api/collections     集合列表
POST /api/documents/upload  文档上传
GET  /api/documents          文档列表
DELETE /api/documents/{id}   删除文档
GET  /api/traces/query       查询追踪
GET  /api/traces/ingestion   摄取追踪
```

### MCP Tools

| Tool | 功能 |
|------|------|
| `query_knowledge_hub` | 混合检索（Dense + Sparse + RRF + Rerank），支持 `agentic` 参数自动路由 |
| `list_collections` | 列出 ChromaDB 集合及统计 |
| `get_document_summary` | 文档元数据与内容预览 |

---

## 架构

```
┌─────────────────────────────────────────────────┐
│  Vue3 前端 (Sakura 🌸)         :5173            │
│  Chat · 文档管理 · 追踪 · 评估                    │
└──────────────────┬──────────────────────────────┘
                   │ HTTP
┌──────────────────┴──────────────────────────────┐
│  FastAPI REST API              :8000            │
│  /api/query · /api/agentic-query · /api/docs    │
└──────────────────┬──────────────────────────────┘
                   │ 复用
┌──────────────────┴──────────────────────────────┐
│  src/core/  +  src/agentic/                     │
│  HybridSearch · ReAct Loop · Ingestion Pipeline │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────┐
│  MCP Server (stdio)                             │
│  query_knowledge_hub · list_collections · ...   │
└─────────────────────────────────────────────────┘
         ▲
         │ MCP 协议
    ┌────┴────┐
    │ Claude  │  (外部 AI Agent)
    └─────────┘
```

**检索链路**：`Query → QueryProcessor(jieba) → Dense + Sparse 并行 → RRF Fusion → Rerank(可选) → Response + Citations`

**摄入链路**：`PDF → Load(MarkItDown) → Split(Recursive) → Transform(LLM Refine/Enrich/Caption) → Dual Encode → Store(ChromaDB + BM25)`

**Agentic 智能路由**：统一入口 `query_knowledge_hub`，LLM 自动判断问题复杂度——简单问题走传统混合检索，复杂多跳问题触发 ReAct 循环（分解→搜索→评估→改写→综合）。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 🔌 **全链路可插拔** | LLM / Embedding / VectorStore / Reranker 等 6 大组件工厂模式 + YAML 驱动，改配置零代码切换 |
| 🔀 **混合检索** | Dense（语义向量）+ Sparse（BM25 关键词）+ RRF 融合 + Cross-Encoder / LLM 精排，每层独立 fallback |
| 🤖 **Agentic RAG** | ReAct 推理循环，自主分解问题、调用多工具、迭代检索，支持复杂多跳对比/因果/广度类问题 |
| 🖼️ **多模态** | PDF 图片 → Vision LLM 生成描述 → 缝合进文本检索链路，"搜文出图"无需独立图片索引 |
| 📊 **白盒可观测** | Ingestion 5 阶段 + Query 5 阶段全链路 Trace JSONL，Streamlit Dashboard 逐阶段展开 |
| ✅ **自动化评估** | Ragas LLM-as-Judge + Custom 指标 + Golden Test Set 回归测试 |
| 🧪 **工程化** | TDD 开发，1200+ 测试（Unit / Integration / E2E），零外部数据库依赖 |

---

## 项目结构

```
MODULAR-RAG-MCP-SERVER/
├── config/settings.yaml          # 全局配置
├── scripts/                      # CLI 工具
│   ├── query.py, ingest.py       # 命令行查询/摄入
│   └── agentic_query.py          # ReAct Agent CLI
├── frontend/                     # Vue3 前端 (Sakura UI)
│   └── src/views/                # Chat, Documents, Traces, Evaluation
├── src/
│   ├── api/                      # FastAPI REST Server (新增)
│   │   ├── server.py             # 入口 :8000
│   │   ├── schemas.py            # Pydantic 模型
│   │   └── routers/              # query, collections, docs, traces
│   ├── agentic/                  # Agentic RAG 模块 (新增)
│   │   ├── react_loop.py         # ReAct 循环状态机
│   │   ├── mcp_client.py         # MCP 子进程客户端
│   │   └── ...                   # parser, tracker, registry, prompts
│   ├── core/                     # 核心层：类型 · 配置 · 检索引擎 · 响应
│   ├── libs/                     # 可插拔后端：LLM · Embedding · VectorStore · Reranker · ...
│   ├── ingestion/                # 摄入管道：Load → Split → Transform → Encode → Store
│   ├── mcp_server/               # MCP Server (stdio)
│   └── observability/            # Dashboard · 评估 · Trace
└── tests/                        # Unit / Integration / E2E
```

---

## 技术栈

Python · FastAPI · Vue3 · Vite · TypeScript · MCP Protocol · ChromaDB · BM25 · jieba · LangChain · DeepSeek · OpenAI · Streamlit · Ragas · pytest · SQLite

---

## CLI 速览

```bash
# 传统检索
python scripts/query.py --query "Azure OpenAI 怎么配置？" --top-k 5

# Agentic 多跳推理
python scripts/agentic_query.py --query "对比项目A和项目B的技术选型差异" --verbose

# 摄入文档
python scripts/ingest.py --file data/documents/report.pdf --collection default

# 启动 Dashboard
streamlit run src/observability/dashboard/app.py

# 启动 MCP Server
python -m src.mcp_server.server
```
