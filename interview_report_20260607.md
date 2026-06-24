# 模拟面试报告

**项目**：Modular RAG MCP Server
**面试时间**：2026-06-07 22:30
**面试官风格**：🎲 MIX（随机混搭型）
**掷骰结果**：4
**评分**：5/10

---

## 一、面试记录

> ✅ 答对核心要点 | ⚠️ 方向正确但细节缺失 | ❌ 未能答出或方向错误

### 方向 1：项目综述

| 轮次 | 风格 | 问题 | 候选人回答摘要 | 评估 |
|-----|------|------|-------------|------|
| 1 | FAST | 这个系统里有哪几类存储？各自负责什么？为什么不能只用一个？ | chromaDB 存文本向量，JSON 存倒排索引 | ⚠️ |
| 2 | DEEP | 从用户发起一次查询到拿到结果，整个链路经过哪些组件？ | 意图识别 → jirab 分词 → 双路检索 → RRF 融合 → 精排 → 五条结果发给 LLM 生成答案 | ⚠️ |
| 3 | CODE | `query_knowledge_hub` 的 register_tool 函数在哪个文件？接收哪几个参数？ | 不太清楚（现场查看代码后答出） | ❌ |
| 4 | HARD | BM25 和 Dense 分数量纲不同，怎么公平合并？RRF 公式是什么？k 值含义？ | RRF 只看排名不看分数，k 是常数防止分数膨胀，设为 60；公式忘记了 | ⚠️ |

### 方向 2：简历深挖

| 轮次 | 风格 | 问题 | 候选人回答摘要 | 评估 | 露馅 |
|-----|------|------|-------------|------|-----|
| 1 | DEEP | Vision LLM 生成的 Caption 怎么进入检索链路？存在 metadata 还是别处？ | 摄入时扫描占位符，缝合进 chunk，Embedding 一起编码 | ⚠️ | 否 |
| 2 | HARD | 新增一个 Embedding Provider（如本地 BGE）要改哪几个文件？每一步做什么？ | 不用改文件，直接注册工厂 | ❌ | 是 |

### 方向 3：技术深挖

| 轮次 | 风格 | 问题 | 候选人回答摘要 | 评估 |
|-----|------|------|-------------|------|
| 1 | CODE | Stage 4 Transform 3 个子步骤分别是什么？Vision LLM 挂了后面受影响吗？ | ①去噪去乱码 ②生成标题/简介/关键词 ③Vision LLM 生成图片描述；不影响，直接跳过 | ✅ |
| 2 | CODE | BM25 和 Dense 检索是串行还是并行？代码怎么实现的？ | 并行；具体实现忘记了（现场查看后答出 ThreadPoolExecutor） | ⚠️ |

---

## 二、参考答案

### <a id="a-四类存储"></a>Q: 这个系统里有哪几类存储？各自负责什么？

**参考答案**：
系统有 **4 类存储**，缺一不可：

| 存储 | 技术 | 存什么 | 为什么不能删 |
|------|------|--------|-------------|
| 向量库 | ChromaDB（SQLite 引擎） | 每个 chunk 的 Dense Embedding 向量 + metadata | 语义检索的唯一来源 |
| 倒排索引 | BM25 JSON（Whoosh 引擎） | 每个 chunk 的词频统计 + 倒排表 | 关键词精确匹配的唯一来源 |
| 摄入历史 | SQLite（ingestion_history.db） | 文件 SHA256 + 摄入状态（success/failed） | 幂等性：已处理文件跳过，失败文件自动重试 |
| 图片索引 | SQLite（image_index.db） | image_id → 文件路径映射 | 检索命中时找到图片文件并返回 |

**为什么不能只用一个**：BM25 需要词频倒排结构，ChromaDB 存的是向量浮点数组，两者数据结构根本不同。仅用向量库 → 专有名词匹配差；仅用 BM25 → 语义匹配差。SQLite 记录文件处理历史，是幂等性的基础——删了就无法防止重复摄入。

---

### <a id="a-查询全链路"></a>Q: 从用户发起一次查询到拿到结果，整个链路经过哪些组件？

**参考答案**：

```
用户查询 "Azure OpenAI 怎么配置"
  │
  ▼
① QueryProcessor（jieba 分词 + 关键词提取 + 停用词过滤）
  │
  ├──② Dense Retrieval（Embedding API 编码 → ChromaDB 向量查询 → top 20）
  │
  └──③ Sparse Retrieval（BM25 词频匹配 → Whoosh 倒排索引 → top 20）
  │
  ▼
④ RRF Fusion（k=60，合并两个排名 → top 10）
  │
  ▼
⑤ Reranker（可选，Cross-Encoder 逐对精排 → top 5）
  │
  ▼
⑥ ResponseBuilder（格式化 Markdown + citations → MCP CallToolResult）
```

注意：Dense 和 Sparse 是并行执行的（`ThreadPoolExecutor(max_workers=2)`），不是串行。

---

### <a id="a-register-tool"></a>Q: `register_tool` 在哪个文件？接收哪些参数？

**参考答案**：
`src/mcp_server/tools/query_knowledge_hub.py` 第 514-524 行：

```python
def register_tool(protocol_handler) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,           # "query_knowledge_hub"
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=query_knowledge_hub_handler,  # 异步 handler 函数
    )
```

`register_tool` 是一个模块级函数，接收 1 个参数 `protocol_handler`，内部调用它的 `register_tool()` 方法传入 4 个参数：name、description、input_schema、handler。

---

### <a id="a-rrf"></a>Q: RRF 融合公式是什么？k 值含义？为什么不用线性加权？

**参考答案**：

$$Score_{RRF}(d) = \frac{1}{k + Rank_{Dense}(d)} + \frac{1}{k + Rank_{Sparse}(d)}$$

- **k 的含义**：平滑因子，防止排名头部的文档分数被过度高估。k = 60 是 Cormack et al. 2009 论文的标准推荐值。调大 → 更均匀，调小 → 头部优势更明显。
- **为什么不用线性加权**：BM25 分数无上界，余弦相似度在 [-1, 1]，量纲不同。线性加权必须先归一化，引入额外超参。RRF 只依赖排名序数，天然无需归一化，鲁棒性更强。
- **RRF 的局限**：不感知排名的质量落差（Dense 第 1 名和第 10 名在 RRF 里差异是 1/61 vs 1/71，差距极小）。本项目通过 RRF 后进行 Cross-Encoder 精排来补足，让交互式模型感知真实质量差异。

---

### <a id="a-image-caption"></a>Q: Vision LLM 生成的 Caption 怎么进入检索链路？

**参考答案**：
**不是在检索时读取 metadata，而是在摄入时缝合进 `chunk.text`。**

```
Stage 2: PdfLoader → Document.text 含 [IMAGE: img_001] 占位符
Stage 3: Chunking → chunk.text 继承了占位符
Stage 4c: ImageCaptioner.transform()
    ├── 正则扫描 chunk.text 中的 [IMAGE: xxx]
    ├── Vision LLM.chat_with_image(prompt, image_path) → caption
    └── 把 caption 缝合进 chunk.text:
        "[IMAGE: img_001]" → "[IMAGE: img_001]\n(Description: 系统架构图，展示了三层结构...)"
Stage 5: DenseEncoder.encode(chunk.text)  ← Caption 文本一起编码为向量
Stage 6: ChromaDB 存储 ← 向量 via chunk.text

检索时: 用户搜"系统架构图" → Embedding 编码 → 和 chunk.text（含 caption）语义匹配 → 命中
```

**为什么选这个方案**：如果把 caption 只存在 `chunk.metadata` 里，Embedding 只读 `chunk.text`，用户搜"系统架构图"就搜不到这张图。缝合进 text 后，Embedding API 编码时 caption 的语义被编码进向量，检索就能匹配了。

---

### <a id="a-可插拔三步骤"></a>Q: 新增一个 Embedding Provider 需要改哪几个文件？

**参考答案**：
严格需要 **3 步**，已有代码零修改：

| 步骤 | 做什么 | 文件 |
|------|--------|------|
| 1 | **新建 Provider 类**，继承 `BaseEmbedding`，实现 `embed()` 方法 | `src/libs/embedding/bge_embedding.py`（新建） |
| 2 | **注册到工厂**：`EmbeddingFactory.register_provider("bge", BGEEmbedding)` | `src/libs/embedding/embedding_factory.py` |
| 3 | **切配置**：`embedding.provider: "bge"` | `config/settings.yaml` |

项目里 6 类组件（LLM、Embedding、Reranker、VectorStore、Splitter、Loader）全部遵循同一套三步流程。

**常见误区**：说"直接注册就行"——注册的前提是有一个类可注册，写 Provider 类（继承 + 实现接口方法）是第一步，不能跳过。

---

### <a id="a-transform"></a>Q: Stage 4 Transform 的 3 个子步骤？

**参考答案**：

| 顺序 | 类名 | 做的事 | 失败行为 |
|------|------|--------|---------|
| 4a | `ChunkRefiner` | 规则清理（HTML 标签、页眉页脚、多余空格）+ LLM 文本精炼（可选） | LLM 失败 → fallback 到规则结果 |
| 4b | `MetadataEnricher` | 规则提取 + LLM 生成 title/summary/tags | LLM 失败 → fallback 到规则结果 |
| 4c | `ImageCaptioner` | 扫描 [IMAGE: id] 占位符 → Vision LLM 生成描述 → 缝合进 chunk.text | Vision LLM 失败 → 原 chunk 不变，管道继续 |

三个步骤在 `pipeline.py` 第 329-351 行顺序调用，前一步的输出是后一步的输入。每个步骤独立容错——单个 chunk 处理失败保留原 chunk，不影响其他 chunk。

---

### <a id="a-parallel-retrieval"></a>Q: Dense 和 Sparse 检索是串行还是并行？代码怎么实现？

**参考答案**：
**并行执行**，代码在 `src/core/query_engine/hybrid_search.py` 第 402-449 行：

```python
# 第 402 行：判断是否并行（配置项 parallel_retrieval 默认 True）
if self.config.parallel_retrieval:
    dense_results, sparse_results = self._run_parallel_retrievals(...)

# 第 447 行：实现
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(self._dense_search, ...): "dense",
        executor.submit(self._sparse_search, ...): "sparse",
    }
    for future in as_completed(futures):
        # 哪个先返回就先拿，不阻塞另一个
```

`ThreadPoolExecutor`，最多 2 个 worker 线程。Dense（Embedding API ~350ms）慢于 Sparse（BM25 磁盘读取 ~120ms），并行让总耗时 = max(稠密, 稀疏)，而非两者相加。

---

## 三、简历包装点评

### 包装合理 ✅
- **"BM25 + Dense Embedding 双路召回，RRF 融合后可经 CrossEncoder 精排"**：能说清 RRF 解决量纲问题的原理、能讲 k 值的作用、能讲 Cross-Encoder 补足 RRF 局限。链条完整。
- **"Vision LLM 生成图片描述并融合进 Chunk"**：核心机制（摄入时缝合进 text）答对了，知道 Embedding 会一起编码。

### 露馅点 ❌
- **"可插拔架构…YAML 配置一键切换 Provider，零代码修改"** → 被问到新增 Provider 具体步骤时说"不用改文件，直接注册"。**严重性：高**。暴露了简历只写了"架构亮点"但没亲自走过新增 Provider 的完整流程。简历写"零代码修改已有代码"，但新增 Provider 本身需要写 Provider 类——这个第一步被漏掉了。
- **"意图识别"出现在查询链路描述中** → 追问后承认"暂时还没实现"。**严重性：中**。简历没写但口头答了未实现的功能——面试官听起来就是"简历上有但讲不清楚"。

### 改进建议
- 亲手走一遍新增 Provider：在项目中新加一个 MockLLM → 注册 → 改 YAML → 跑 `scripts/query.py` 验证。把"3 步流程"背下来。
- 把题干里的"意图识别"从回答里去掉了——用标准术语"QueryProcessor（jieba 分词）"代替，和代码对齐。
- 背下 RRF 公式（就 4 个符号：`1/(k+rank)` 两项相加），这是面试里最高频的追问。

---

## 四、综合评价

**优势**：
- Q4（RRF 局限 → Cross-Encoder 补足）：能主动说出 Cross-Encoder 引入 query-chunk 交互打分来弥补 RRF 只看排名的局限，设计思维连贯
- Q6（Transform 3 步 + Vision LLM 挂了的处理）：三个步骤和容错行为全部答对，理解最扎实的一题
- 整体回答简洁，不绕弯，比"扯一堆不知道怎么停"好

**薄弱点**：
- **算法公式记忆不足**：RRF 公式记不清，k 值含义只说对一半（"防止分数膨胀"对，"防止差值过小"不准确——k 增大反而让差值变小）
- **实现细节代码定位不准**：`register_tool` 不知道在哪、并行检索实现不知道是什么机制
- **可插拔架构停留在"概念"**：知道工厂模式，但说新增 Provider"不用改文件"暴露了没有动手走过流程
- **口头说了未实现功能**：链路里提到"意图识别"，追问又承认没做——面试里这种矛盾很致命

**面试官建议**：
1. 花 30 分钟动手新增一个 MockLLM Provider——跟着三步（写类→注册→改 YAML）完整走一遍，比看 3 遍文档有用
2. 背下 RRF 公式，对着 `fusion.py` 看它怎么实现的，理解 `for rank, item in enumerate(results)` 那一行
3. 打开 `hybrid_search.py` 的 `_run_parallel_retrievals` 方法，看 `ThreadPoolExecutor` 那几行——并行是面试高频题，3 分钟就能看明白

---

## 五、评分

| 维度 | 分数 | 评分依据 |
|-----|------|---------|
| 项目架构掌握 | 5/10 | 知道主链路组件和存储类型，但 transform 子步骤类名不熟，register_tool 位置不知道 |
| 简历真实性 | 5/10 | RRF 和 Caption 机制能自圆其说，但可插拔架构被追问穿帮，意图识别自爆未实现 |
| 算法理论深度 | 4/10 | 选对了 RRF 方案但公式写不出，k 值理解正确但不完整，未提线性加权的问题 |
| 实现细节掌握 | 4/10 | register_tool 文件/参数不清楚，并行实现机制不知道，需现场查代码 |
| 表达清晰度 | 6/10 | 回答简练主干清晰，但技术术语不够精确（jirab→jieba，类名替代描述性表述） |
| **综合** | **5/10** | 架构理解有骨架但缺少血肉，简历包装度不高但有 1 处露馅，需要补代码细节和数学公式 |