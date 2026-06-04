# Modular RAG MCP Server

> 可插拔、可观测的模块化 RAG 系统，通过 MCP 协议对外暴露检索工具，任何 MCP Client 可直接调用。全链路配置驱动——改一行 YAML 即可切换 LLM、Embedding、VectorStore、Reranker 后端。

---

## 本地部署

### 环境要求

- Python 3.12+
- Windows / macOS / Linux

### 1. 克隆并创建虚拟环境

```bash
git clone <repo-url>
cd Modular-RAG-MCP-Server
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

### 2. 配置 settings.yaml

编辑 `config/settings.yaml`，最少需要配 3 项：

```yaml
llm:
  provider: "deepseek"          # 或 openai / azure / ollama
  model: "deepseek-v4-pro"
  api_key: "sk-xxx"             # 你的 API Key

embedding:
  provider: "openai"
  model: "text-embedding-v3"
  api_key: "sk-xxx"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

vector_store:
  provider: "chroma"
  persist_directory: "./data/db/chroma"
```

> 完整配置项见 `config/settings.yaml`。

### 3. 摄入文档

```bash
python scripts/ingest.py --file data/documents/your_doc.pdf --collection default
```

### 4. 查询

```bash
python scripts/query.py --query "你的问题" --top-k 5
```

### 5. 启动 Dashboard

```bash
streamlit run src/observability/dashboard/app.py
```

浏览器打开 `http://localhost:8501`。

### 6. 作为 MCP Server 启动（供 Agent 调用）

```bash
python -m src.mcp_server.server
```

配置到 Claude Desktop / Cursor / Continue.dev 的 MCP 配置文件中即可被 Agent 发现和调用。

---

## 项目概览

一个从 PDF 摄入到智能检索的完整 RAG 系统，核心链路：

```
PDF → Load → Chunk → Transform → Dual Encode → Store
                                                    ↓
                                               ChromaDB + BM25
                                                    ↓
                          Query → IntentRoute → HybridSearch → Rerank → Response
                                                    ↓
                                          Dashboard + Trace 可观测
```

**一句话**：把非结构化文档变成可语义搜索的知识库，通过 MCP 协议让任何 AI Agent 都能调用。

### 核心模块

| 模块 | 能力 |
|---|---|
| **Ingestion Pipeline** | PDF → Markdown → 语义分块 → LLM 精炼/元数据增强 → Dense+Sparse 双路编码 → 存储 |
| **Hybrid Search** | Dense（语义向量）+ Sparse（BM25 关键词）+ RRF 融合 + 可选 Rerank |
| **Intent Router** 🆕 | LLM 驱动的查询意图分类（chat/search/filter/compare），闲聊零开销短路 |
| **MCP Server** | 标准 MCP 协议，暴露 `query_knowledge_hub` 等 3 个 Tool |
| **Dashboard** | Streamlit 6 页面：概览/数据浏览/摄入管理/摄入追踪/查询追踪/评估面板 |
| **Evaluation** | Ragas LLM-as-Judge + Custom 评估 + Golden Test Set 回归测试 |
| **Observability** | 全链路 Trace（ingestion 5 阶段 + query 5 阶段），Dashboard 可逐阶段展开 |

---

## 关键创新点

### 1. 全链路可插拔工厂架构

LLM / Embedding / Reranker / Splitter / VectorStore 均定义抽象基类 + 工厂模式，通过 `settings.yaml` 一键切换后端，**零代码改动**。

```
LLMFactory._PROVIDERS = {openai, azure, deepseek, ollama}
EmbeddingFactory._PROVIDERS = {openai, azure, ollama}
VectorStoreFactory._PROVIDERS = {chroma, ...}
→ 改一行 YAML 即可切 Provider
```

### 2. 查询意图路由器（IntentRouter）

在检索前插入 LLM 零样本分类层，识别 4 种意图并选择最优链路：

| 意图 | 策略 | 效果 |
|---|---|---|
| chat | 短路，LLM 直接回答 | 零 Embedding 调用 |
| search | 正常 HybridSearch + Rerank | 当前逻辑 |
| filter | LLM 提取 metadata 过滤条件注入检索 | 精准命中 |
| compare | LLM 拆分 2-3 个子查询合并召回 | 覆盖多角度 |

规则层（高频问候、元问题）命中时 < 1ms，LLM 分类 ~200ms。分类失败自动降级为 search——**用户零感知**。

### 3. 双路检索 + 优雅降级

- **Dense**：Embedding 语义匹配，解决同义词/跨语言
- **Sparse**：BM25 关键词匹配，解决专有名词/精确匹配
- **RRF Fusion**：两者融合排序，互补长短
- **Rerank**（可选）：Cross-Encoder 或 LLM 精排

每一环都有独立 fallback：Dense 挂了 Sparse 顶上，Rerank 挂了保持原始排序，HybridSearch 挂了返回空列表——**整体不崩**。

### 4. 多模态 Image Captioning

PDF 中的图片先提取 → Vision LLM 生成文字描述 → 缝合进 Chunk 文本。复用纯文本 RAG 链路即可实现"搜文字出图"，不需要独立的图片向量索引。

### 5. 全链路白盒可观测

每次 Ingestion（5 阶段）和 Query（5 阶段 + Intent Routing）的中间状态、耗时、数据全部写入 JSONL Trace。Dashboard 可逐阶段展开查看——chunk 拆分前后的文本对比、Dense/Sparse 各自的检索结果、RRF 融合前后的分数变化。



## 目录与架构

```
MODULAR-RAG-MCP-SERVER/
├── config/
│   └── settings.yaml              # 全局配置（LLM/Embedding/VectorStore/Ingestion/Rerank）
├── scripts/
│   ├── query.py                   # CLI 查询脚本（含 IntentRouter）
│   ├── ingest.py                  # CLI 摄入脚本
│   └── evaluate.py                # CLI 评估脚本
├── tests/
│   ├── unit/                      # 单元测试（test_intent_router 等）
│   ├── integration/               # 集成测试
│   └── fixtures/                  # 测试数据（golden_test_set.json）
├── src/
│   ├── core/                      # 核心层：类型定义 + 配置 + 检索引擎 + 响应构建
│   │   ├── types.py               # Document / Chunk / RetrievalResult 等核心数据结构
│   │   ├── settings.py            # 配置加载与解析
│   │   ├── trace/                 # Trace 上下文与收集器
│   │   ├── query_engine/          # 检索引擎
│   │   │   ├── intent_router.py   # 🆕 查询意图路由器
│   │   │   ├── hybrid_search.py   # Dense + Sparse + RRF 融合
│   │   │   ├── dense_retriever.py # 向量检索
│   │   │   ├── sparse_retriever.py# BM25 检索
│   │   │   ├── fusion.py          # RRF 融合算法
│   │   │   ├── reranker.py        # 重排序
│   │   │   └── query_processor.py # 查询预处理（分词/关键词提取）
│   │   └── response/              # MCP 响应构建
│   │       └── response_builder.py
│   ├── libs/                      # 可插拔库层：工厂模式 + 抽象基类 + 各 Provider 实现
│   │   ├── llm/                   # LLM Provider（OpenAI/Azure/DeepSeek/Ollama + Vision）
│   │   │   ├── base_llm.py        # BaseLLM 抽象类
│   │   │   ├── llm_factory.py     # LLMFactory（双注册表：文本 + 视觉）
│   │   │   ├── deepseek_llm.py
│   │   │   ├── openai_llm.py
│   │   │   ├── azure_llm.py
│   │   │   └── ollama_llm.py
│   │   ├── embedding/             # Embedding Provider
│   │   │   ├── base_embedding.py
│   │   │   └── embedding_factory.py
│   │   ├── vector_store/          # VectorStore Provider（ChromaDB 等）
│   │   ├── splitter/              # 文本分割器（Recursive 等）
│   │   ├── loader/                # 文档加载器 + 文件完整性检查
│   │   ├── reranker/              # 重排序器工厂
│   │   └── evaluator/             # 评估器工厂
│   ├── ingestion/                 # 摄入管道
│   │   ├── pipeline.py            # 🎯 6 阶段管道编排器
│   │   ├── chunking/              # 文档分块
│   │   ├── transform/             # Chunk 精炼/元数据增强/图片描述
│   │   ├── embedding/             # Dense/Sparse 编码 + 批处理
│   │   └── storage/               # BM25 索引 / ChromaDB Upsert / 图片存储
│   ├── mcp_server/                # MCP Server
│   │   ├── server.py              # stdio 启动入口
│   │   ├── protocol_handler.py    # JSON-RPC 协议处理 + Tool 注册
│   │   └── tools/                 # MCP Tool 实现
│   │       ├── query_knowledge_hub.py  # 🎯 核心检索 Tool
│   │       ├── list_collections.py
│   │       └── get_document_summary.py
│   └── observability/             # 可观测性
│       ├── dashboard/             # Streamlit Dashboard
│       │   ├── app.py             # 导航入口（6 页面注册）
│       │   ├── pages/             # 6 个页面实现
│       │   └── services/          # ConfigService / DataService / TraceService
│       └── evaluation/            # Ragas 评估器 + EvalRunner
└── data/                          # 运行时数据（gitignore）
    ├── db/chroma/                 # ChromaDB 持久化
    ├── db/bm25/                   # BM25 索引文件
    └── images/                    # 提取的图片
```

---

## 核心流程

### 写入链路：Ingestion Pipeline（6 阶段）

```
PDF 文件
  │
  ▼
Stage 1: File Integrity Check    ← SHA256 hash → SQLite 查重，已处理过的跳过
  │
  ▼
Stage 2: Document Loading        ← PDF → Markdown（markitdown），提取图片
  │                                 输出: Document(text + metadata.images)
  ▼
Stage 3: Chunking                ← Recursive 分块，800 chars + 150 overlap
  │                                 输出: List[Chunk]（ID 格式: doc_hash_index_contenthash）
  ▼
Stage 4: Transform Pipeline      ← 3 个子步骤串联
  │  4a: Chunk Refinement        ← LLM 优化文本 / Rule 规则清理
  │  4b: Metadata Enrichment     ← LLM 提取标题/标签/摘要 / Rule 正则提取
  │  4c: Image Captioning        ← Vision LLM 生成图片描述，缝合进 Chunk.text
  ▼
Stage 5: Encoding                ← Dense（Embedding API → float vector）
  │                                + Sparse（本地 BM25 词频统计）
  ▼
Stage 6: Storage                 ← 3 路并行写入
     6a: ChromaDB Upsert         ← 向量 + metadata
     6b: BM25 Index              ← Whoosh 倒排索引（chunk_id 与 Chroma 对齐）
     6c: Image Storage Index     ← 图片元数据注册
```

### 读取链路：Query Pipeline（5 阶段 + IntentRoute）

```
用户查询: "Azure OpenAI 怎么配置？"
  │
  ▼
IntentRouter.classify(query)      ← 🆕 LLM 零样本分类（chat/search/filter/compare）
  │                                  chat → 短路，LLM 直接回答
  │                                  filter → 提取 metadata 过滤条件
  │                                  compare → 拆分子查询
  ▼
QueryProcessor                    ← jieba 分词 + 关键词提取 + 停用词过滤
  │
  ├── Dense Retrieval             ← Embedding API → ChromaDB 向量相似度查询（top 20）
  │
  └── Sparse Retrieval            ← BM25 关键词查询 → Whoosh 倒排索引（top 20）
  │
  ▼
RRF Fusion                        ← k=60，合并排序 → top 10
  │
  ▼
Reranker（可选，默认关闭）         ← Cross-Encoder 或 LLM 精排 → top 5
  │
  ▼
ResponseBuilder                   ← 格式化结果 + citations + MCP 协议包装
```

### IntentRouter 决策流程

```
query → _pre_check(query)         ← 规则层（< 1ms）
  ├── 命中（问候/元问题）         → chat, confidence=0.99
  └── 未命中                      → _llm_classify(query)  ← LLM 零样本（~200ms）
                                       ├── 成功              → 返回意图 + 置信度
                                       └── 失败              → fallback search
```

---

## 技术栈

| 类别 | 技术 |
|---|---|
| **语言** | Python 3.12+ |
| **LLM** | DeepSeek / OpenAI / Azure OpenAI / Ollama（工厂模式切换） |
| **Embedding** | OpenAI text-embedding-v3 / Azure / Ollama |
| **向量存储** | ChromaDB（PersistentClient，SQLite 引擎） |
| **稀疏检索** | BM25（Whoosh 引擎） |
| **融合算法** | RRF（Reciprocal Rank Fusion，k=60） |
| **重排序** | Cross-Encoder / LLM Rerank（可选） |
| **分词** | jieba（中文分词 + 关键词提取） |
| **协议** | MCP（Model Context Protocol），JSON-RPC 2.0，stdio transport |
| **Dashboard** | Streamlit（6 页面） |
| **评估** | Ragas（LLM-as-Judge）+ Custom + Golden Test Set |
| **可观测** | JSONL Trace（ingestion 5 阶段 + query 5 阶段） |
| **图片处理** | Vision LLM（Qwen-omni-flash）+ PIL |
| **文档解析** | markitdown（PDF → Markdown） |
| **测试** | pytest（Unit / Integration / E2E） |

---

## 环境变量

| 变量 | 用途 | 必须 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（`settings.llm.api_key` 也可） | 使用 DeepSeek 时 |
| `OPENAI_API_KEY` | OpenAI API Key | 使用 OpenAI 时 |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API Key | 使用 Azure 时 |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI Endpoint | 使用 Azure 时 |
| `OLLAMA_BASE_URL` | Ollama 服务地址（默认 `http://localhost:11434`） | 使用 Ollama 时 |

> 所有 API Key 支持 3 层优先级：**显式传参 > settings.yaml > 环境变量**（`DeepSeekLLM` 已修复对齐，`AzureLLM` 和 `OpenAILLM` 天然支持）。

---

## API 速览

MCP Server 对外暴露 3 个 Tool，每个 Tool 内部经历多条处理管道。

---

### Tool 1: `query_knowledge_hub`

**Input Schema:**

```json
{
  "name": "query_knowledge_hub",
  "parameters": {
    "query":       { "type": "string",  "required": true },
    "top_k":       { "type": "integer", "default": 5, "minimum": 1, "maximum": 20 },
    "collection":  { "type": "string",  "default": "default" }
  }
}
```

**内部管道（8 步）：**

```
MCP Client 调用 query_knowledge_hub
  │
  ▼
1. 参数校验
   └── query 不能为空，top_k 限制在 1-20，collection 默认 "default"

2. 懒加载组件初始化（首次查询，后续缓存命中跳过）
   └── asyncio.to_thread(_ensure_initialized, collection)
       ├── EmbeddingFactory.create(settings)        ← 永久缓存，不重建
       ├── create_core_reranker(settings)           ← 永久缓存，不重建
       ├── VectorStoreFactory.create(collection)    ← 每次重建（读写入进程数据）
       ├── create_dense_retriever(embedding, store)
       ├── BM25Indexer + create_sparse_retriever    ← 每次 reload Whoosh 索引
       ├── QueryProcessor                          ← jieba 分词 + 停用词过滤
       └── create_hybrid_search(dense, sparse, processor)

3. 🆕 意图路由（IntentRouter.classify）
   └── asyncio.to_thread(router.classify, query)
       ├── 规则预检（_pre_check）→ 问候/元问题 → chat，< 1ms
       ├── LLM 零样本分类（_llm_classify）→ 结构化 JSON 输出，~200ms
       └── 路由分支：
           ├── chat    → 短路，直接返回 build_chat_response()
           ├── filter  → 提取 metadata 过滤条件，注入 query 上下文
           ├── compare → 拆分子查询，拼接增强 query
           └── search  → 不做干预，走原有检索流程

4. 混合搜索（HybridSearch.search）
   └── asyncio.to_thread(_perform_search, query, top_k*2)
       ├── Dense Retrieval（并行）
       │   ├── QueryProcessor 分词 + 提取关键词
       │   ├── Embedding API 编码 query → float vector
       │   └── ChromaDB 向量相似度查询 → top 20 结果
       ├── Sparse Retrieval（并行）
       │   ├── BM25Indexer._ensure_index_loaded() → 从磁盘加载 Whoosh 索引
       │   ├── BM25 词频匹配 → top 20 结果
       │   └── 通过 chunk_id 从 ChromaDB 回填完整 metadata
       └── RRF Fusion (k=60)
           └── 合并两个排名列表，按 1/(k+rank) 加权求和，取 top 10

5. 重排序（可选，config.enable_rerank 控制）
   └── asyncio.to_thread(_apply_rerank, query, results, top_k)
       ├── Cross-Encoder 逐对打分（query, chunk_text）→ 精排
       ├── LLM Rerank（备选，按 settings.rerank.provider 切换）
       └── 失败降级：取原始排序的前 top_k 结果

6. 响应构建（ResponseBuilder.build）
   └── 遍历 results → 格式化 Markdown 文本
       ├── 每个 chunk: 分数 + 来源文件 + chunk 索引 + 文本预览
       ├── citations 列表: {title, source_path, chunk_id, score}
       └── 图片 chunk → ImageContent 块

7. Trace 收集
   └── trace.metadata["final_results"] = [每个结果的关键字段]
   └── TraceCollector().collect(trace) → 写入 logs/traces.jsonl

8. MCP 协议包装
   └── response.to_mcp_content() → 分离 TextContent + ImageContent
   └── types.CallToolResult(content=content_blocks, isError=False)
```

**涉及的核心类/模块：**

| 步骤 | 类/模块 | 文件 |
|---|---|---|
| 初始化 | `QueryKnowledgeHubTool._ensure_initialized()` | `src/mcp_server/tools/query_knowledge_hub.py` |
| 意图路由 | `IntentRouter.classify()` | `src/core/query_engine/intent_router.py` |
| 混合搜索 | `HybridSearch.search()` | `src/core/query_engine/hybrid_search.py` |
| 向量检索 | `DenseRetriever.search()` | `src/core/query_engine/dense_retriever.py` |
| 关键词检索 | `SparseRetriever.search()` | `src/core/query_engine/sparse_retriever.py` |
| 融合 | `RRFFusion.fuse()` | `src/core/query_engine/fusion.py` |
| 重排序 | `CoreReranker.rerank()` | `src/core/query_engine/reranker.py` |
| 分词 | `QueryProcessor.process()` | `src/core/query_engine/query_processor.py` |
| 响应 | `ResponseBuilder.build()` | `src/core/response/response_builder.py` |
| 追踪 | `TraceContext / TraceCollector` | `src/core/trace/` |

---

### Tool 2: `list_collections`

**Input Schema:**

```json
{
  "name": "list_collections",
  "parameters": {}
}
```

**内部管道：**

```
MCP Client 调用 list_collections
  │
  ▼
1. 获取所有 collection
   └── VectorStoreFactory.create(settings) → 获取 ChromaDB PersistentClient
   └── client.list_collections() → [(name, count), ...]

2. 收集每个 collection 的统计
   ├── collection.count() → chunk 总数
   └── 查询 BM25 索引目录 data/db/bm25/{name}/ → 是否有索引

3. 格式化响应
   └── Markdown 表格: | Name | Chunks | BM25 Index |
   └── types.CallToolResult(content=[TextContent])
```

---

### Tool 3: `get_document_summary`

**Input Schema:**

```json
{
  "name": "get_document_summary",
  "parameters": {
    "source_hash": { "type": "string", "required": true },
    "collection":  { "type": "string", "default": "default" }
  }
}
```

**内部管道：**

```
MCP Client 调用 get_document_summary
  │
  ▼
1. 参数解析
   └── source_hash: 文件 SHA256，来自 Ingestion Pipeline Stage 1

2. 查 ChromaDB
   └── collection.get(where={"doc_hash": source_hash})
       ├── 返回该文档所有 chunk（ids + documents + metadatas）
       └── 汇总统计: chunk_count, total_chars, avg_chunk_size

3. 查 Ingestion 历史
   └── SQLiteIntegrityChecker.get_record(file_hash)
       ├── source_path（原始文件路径）
       ├── processed_at（摄入时间）
       └── status（success / failed）

4. 查图片记录
   └── ImageStorage.list_images(doc_hash=source_hash)
       └── 返回图片列表: [{image_id, file_path, page_num}, ...]

5. 格式化响应
   └── Markdown: 文档来源 + 摄入时间 + chunk 统计 + 图片预览
   └── types.CallToolResult(content=[TextContent])
```

**涉及的核心类/模块：**

| 步骤 | 类/模块 | 文件 |
|---|---|---|
| 查询 | `VectorStore.collection.get()` | `src/libs/vector_store/` |
| 历史 | `SQLiteIntegrityChecker` | `src/libs/loader/file_integrity.py` |
| 图片 | `ImageStorage.list_images()` | `src/ingestion/storage/image_storage.py` |

---

## 流式输出与实时检索过程 — 技术细节

### 当前实现

`query_knowledge_hub` 采用**全量返回**模式：完整走完 IntentRoute → Search → Rerank 后，一次性返回所有结果。每一步的状态写入 Trace（JSONL），可在 Dashboard 的 Query Traces 页逐阶段展开查看。

### 流式输出（规划中）

MCP Server stdio transport 基于 asyncio，天然支持流式。计划实现的流式方案：

```
客户端发起查询
  →
  ① IntentRoute 完成 → 推送 {"stage": "routing", "intent": "search", "elapsed_ms": 200}
  ② Dense 检索完成  → 推送 {"stage": "dense", "hits": 20, "elapsed_ms": 350}
  ③ Sparse 检索完成 → 推送 {"stage": "sparse", "hits": 20, "elapsed_ms": 120}
  ④ RRF Fusion 完成 → 推送 {"stage": "fusion", "hits": 10, "elapsed_ms": 50}
  ⑤ Rerank 完成     → 推送 {"stage": "rerank", "hits": 5, "elapsed_ms": 300}
  ⑥ 最终结果流式返回，每个 chunk 附 citations
```

MCP 协议支持 `CallToolResult` 返回多个 `TextContent`，每个 chunk 可作为独立的 content block 逐步返回：

```python
async def query_knowledge_hub_streaming_handler(query, top_k, collection):
    for stage_event in pipeline.stream_execute(query, top_k, collection):
        yield types.CallToolResult(
            content=[types.TextContent(type="text", text=stage_event.to_json())]
        )
```

### RRF 融合算法

```
RRF_score(chunk) = Σ (1 / (k + rank_i))
  其中 k = 60, rank_i = chunk 在第 i 个来源列表中的排名

示例：
  chunk_A 在 Dense 排名第 2，Sparse 排名第 5
  RRF_score = 1/(60+2) + 1/(60+5) = 0.0161 + 0.0154 = 0.0315
```

### 冷启动说明

首次查询时 `_ensure_initialized()` 需要创建 ChromaDB 连接、加载 jieba 词典、初始化 Embedding 客户端，耗时约 500ms-2s。后续查询复用缓存组件，延迟降至 200ms-800ms（取决于 Embedding API 响应速度）。
