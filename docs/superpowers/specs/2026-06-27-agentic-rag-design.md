# Agentic RAG 设计文档

**日期**: 2026-06-27
**状态**: 待实现
**范围**: 在现有 MCP RAG Server 之上新增 Agent 编排层，不改动现有代码

---

## 1. 目标与非目标

### 目标

- 支持复杂多跳问题：一个问题需要多次检索不同来源，组合后综合回答
- Agent 拥有完整推理循环：分解 → 检索 → 评估 → 改写 → 综合 → 自检
- 通过 MCP 协议复用现有 3 个工具，不动现有 Server 代码

### 非目标

- 不修改 `src/mcp_server/`、`src/core/` 任何代码
- 不做对话式多轮（每个问题是独立的复杂查询）
- 不做 Agent 间协作 / 多 Agent 编排

---

## 2. 架构

新增模块 `src/agentic/`，通过 MCP 子进程连接现有 Server：

```
scripts/agentic_query.py (CLI 入口)
        │
        ▼
src/agentic/ (新增，不改动现有代码)
├── react_loop.py          # ReAct 循环状态机
├── mcp_client.py          # MCP JSON-RPC 子进程客户端
├── tool_registry.py       # 工具定义 + 参数解析
├── prompt_manager.py      # System prompt + few-shot 示例
├── citation_tracker.py    # 跨轮次引用追踪
└── response_parser.py     # 解析 LLM 输出

        │ MCP 协议 (stdio/subprocess)
        ▼
src/mcp_server/ (现有，不改)
    query_knowledge_hub | list_collections | get_document_summary
```

关键决策：

- MCP 通信：复用 `mcp` 库 `ClientSession` + `stdio_client`，启动子进程，纯本地
- LLM 配置：复用 `config/settings.yaml` 现有 `llm` 段落
- 不使用 HTTP，保持 stdio 传输

---

## 3. ReAct 循环

### 状态机

```
用户问题
  │
  ▼
System Prompt (含 3 个 few-shot 示例)
  │
  ▼
┌──────────────────────────────────────┐
│  while round < max_rounds (10):      │
│                                      │
│  1. call_llm(messages)               │
│     → LLM 返回 Thought + Action      │
│       或 Final Answer                │
│                                      │
│  2. response_parser.parse(response)  │
│     ├─ Final Answer → 跳出循环       │
│     └─ Action → 继续                 │
│                                      │
│  3. tool_registry.execute(action)    │
│     → Observation                    │
│                                      │
│  4. citation_tracker.add(results)    │
│                                      │
│  5. trim_context(messages)           │
│                                      │
│  6. round += 1                       │
└──────────────────────────────────────┘
  │
  ▼
AgenticResult(answer, citations, trace)
```

### 安全约束

| 机制 | 参数 | 说明 |
|------|------|------|
| 最大轮次 | 10 轮 | 超出则强制基于已有信息生成答案 |
| 上下文裁剪 | 近 3 轮完整 Observation | 每轮结果截断到 500 字/条 |
| 循环检测 | 连续 2 次相同调用 | 警告并建议 LLM 换策略 |
| 引用追踪 | 持久化每轮检索结果 | 最终答案标记 `[1] source_path` |

### LLM 输出格式

LLM 只需输出两种格式：

```
Thought: 需要先调查项目 A 使用什么技术栈
Action: query_knowledge_hub(query="项目A 技术选型", top_k=5)
```

或：

```
Final Answer: 项目 A 使用 React + Go，项目 B 使用 Vue + Python...
[来源: docs/projectA.md, docs/projectB.md]
```

---

## 4. 文件结构

### src/agentic/react_loop.py

核心调度器，`ReActLoop` 类：

```python
@dataclass
class AgenticResult:
    answer: str
    citations: List[Citation]
    trace: List[RoundTrace]  # 每轮 Thought/Action/Observation 记录

class ReActLoop:
    max_rounds: int = 10
    context_window: int = 3

    def __init__(self, mcp_client, tool_registry, llm_client, ...):
        ...

    async def run(self, question: str) -> AgenticResult:
        # 构建初始 messages (system + user question)
        # while 循环:
        #   response = await call_llm(messages)
        #   parsed = response_parser.parse(response)
        #   if parsed.is_final: break
        #   observation = await tool_registry.execute(parsed.action)
        #   self.citation_tracker.add(observation.results)
        #   messages = trim_context(messages, observation)
        # 循环结束后:
        #   citations = self.citation_tracker.format()
        #   return AgenticResult(answer=parsed.answer, citations=citations, ...)
```

### src/agentic/mcp_client.py

MCP 子进程客户端，封装 3 个工具调用：

```python
class MCPClient:
    async def connect(self): ...   # 启动子进程，initialize 握手
    async def call_tool(self, name, arguments) -> ToolResult: ...
    async def close(self): ...
```

### src/agentic/tool_registry.py

工具定义与执行调度：

```python
TOOLS = {
    "query_knowledge_hub": ToolDef(
        params={"query": str, "top_k": int, "collection": Optional[str]},
        description="...",
    ),
    "list_collections": ToolDef(...),
    "get_document_summary": ToolDef(...),
}

class ToolRegistry:
    def parse_action(self, llm_text: str) -> ParsedAction: ...
    async def execute(self, action: ParsedAction) -> Observation: ...
```

### src/agentic/prompt_manager.py

- System prompt：角色设定 + 工具说明 + 行为约束
- 3 个 few-shot 示例（多跳场景）：
  1. 对比类（"A 和 B 的技术差异"）
  2. 因果类（"为什么 A 导致了 B"）
  3. 需要多次检索的广度类（"列出所有使用了 X 的项目"）

### src/agentic/citation_tracker.py

```python
class CitationTracker:
    def add(self, results: List[RetrievalResult]): ...  # 去重合并（按 chunk_id）
    def format(self) -> str: ...                         # 从追踪池生成引用表 [1] source
    @property
    def all_sources(self) -> List[str]: ...              # 去重后的 source_path 列表
```

注意：引用表基于 `add()` 实际积累的检索结果生成，不依赖 LLM 在 Final Answer 中自报的来源，防止引用幻觉。

### src/agentic/response_parser.py

正则 + 状态机解析 LLM 原始输出：

```python
@dataclass
class ParsedResponse:
    is_final: bool
    thought: Optional[str]
    action_name: Optional[str]
    action_params: Optional[Dict]
    answer: Optional[str]
    sources: List[str]

def parse(llm_raw: str) -> ParsedResponse: ...
```

### scripts/agentic_query.py

CLI 入口：

```
用法: python scripts/agentic_query.py --query "对比A和B的技术选型"
可选: --verbose (打印每轮推理), --max-rounds 10
```

---

## 5. 错误处理

| 场景 | 处理 |
|------|------|
| MCP Server 启动失败 | 重试 1 次，仍失败则提示用户手动启动 |
| 工具调用超时 (30s) | Observation = "(工具调用超时)"，LLM 自行决定重试或换策略 |
| LLM API 错误 | 重试 1 次，连续 3 次失败则终止并输出已有信息 |
| LLM 输出无法解析 | 将原始输出回传，提示"请使用 Thought:/Action: 格式" |
| 检索返回 0 结果 | Observation 如实报告，LLM 自行改写查询 |
| 工具幻觉 | 返回 "Unknown tool: xxx"，LLM 自行纠正 |
| 用户 Ctrl+C | 捕获信号，输出已有信息 + 引用 |

---

## 6. 可观测性

- `--verbose`：打印完整 Thought/Action/Observation
- 默认模式：进度概括 `[1/5] 检索"项目A技术栈" → 3 条结果`
- 最终引用表：`[1] docs/projectA.md` `[2] docs/projectB.md`
- 复用现有 `TraceCollector`，每轮检索产生独立 trace

---

## 7. 测试策略

- **单元测试** (`tests/unit/agentic/`)：response_parser、citation_tracker、prompt_manager 纯逻辑测试
- **集成测试** (`tests/integration/agentic/`)：mock MCP Server，测试完整 ReAct 循环
- **E2E** (`tests/e2e/`)：真实 MCP 子进程，真实 LLM（标记 `@pytest.mark.llm`）

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LLM 循环过长 / 重复 | 10 轮上限 + 循环检测 |
| token 消耗过高 | 上下文裁剪 + 结果截断 |
| 子进程 MCP 通信不稳定 | 超时 + 重试 + 显式错误提示 |
| 引用幻觉 | citation_tracker 只记录实际检索结果，不做 LLM 生成 |