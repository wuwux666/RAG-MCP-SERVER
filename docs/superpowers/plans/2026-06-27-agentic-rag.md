# Agentic RAG 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 MCP RAG Server 之上构建 Agent 编排层，支持复杂多跳问题的 ReAct 循环检索。

**Architecture:** 新增 `src/agentic/` 模块（7 个文件），通过 MCP 子进程连接现有 Server。Agent 以 Thought→Action→Observation 循环运行，LLM 驱动推理，最大 10 轮，含上下文裁剪和循环检测。现有代码零改动。

**Tech Stack:** Python 3.10+, `mcp` SDK (ClientSession + stdio_client), 现有 `BaseLLM.chat()` (同步), `asyncio.to_thread` 模式

## Global Constraints

- 不修改 `src/mcp_server/`、`src/core/` 任何文件
- LLM 配置复用 `config/settings.yaml` 的 `llm` 段落
- 每轮检索结果截断到 500 字/条
- 最大 ReAct 轮次: 10
- 上下文窗口: 保留近 3 轮完整 Observation
- 连续 2 次相同工具调用 = 循环检测警告
- 引用表基于 `add()` 实际积累的检索结果，不依赖 LLM 自报来源
- 所有 LLM 调用通过 `asyncio.to_thread()` 包装（现有同步接口）
- 测试放在 `tests/unit/agentic/` 和 `tests/integration/agentic/`

---

### Task 1: response_parser — 解析 LLM 输出

**Files:**
- Create: `src/agentic/__init__.py`
- Create: `src/agentic/response_parser.py`
- Create: `tests/unit/agentic/__init__.py`
- Create: `tests/unit/agentic/test_response_parser.py`

**Interfaces:**
- Produces: `ParsedResponse` dataclass, `parse_response(raw: str) -> ParsedResponse`

- [ ] **Step 1: Create `src/agentic/__init__.py`**

```python
"""Agentic RAG module — ReAct agent that orchestrates multi-hop retrieval 
over the existing MCP RAG Server without modifying it.
"""
```

- [ ] **Step 2: Create `tests/unit/agentic/__init__.py`**

```python
"""Unit tests for agentic RAG module."""
```

- [ ] **Step 3: Write the failing test**

`tests/unit/agentic/test_response_parser.py`:

```python
"""Tests for response_parser — parsing LLM ReAct output."""

import pytest
from src.agentic.response_parser import parse_response, ParsedResponse


class TestParseAction:
    """Parse Thought + Action format."""

    def test_parse_simple_action(self):
        raw = (
            "Thought: 需要先查询项目 A 的技术栈\n"
            'Action: query_knowledge_hub(query="项目A 技术选型", top_k=5)'
        )
        result = parse_response(raw)
        assert result.is_final is False
        assert "查询项目 A" in result.thought
        assert result.action_name == "query_knowledge_hub"
        assert result.action_params["query"] == "项目A 技术选型"
        assert result.action_params["top_k"] == 5

    def test_parse_action_with_collection(self):
        raw = (
            "Thought: 在 api-docs 集合中搜索\n"
            'Action: query_knowledge_hub(query="REST API", top_k=3, collection="api-docs")'
        )
        result = parse_response(raw)
        assert result.action_name == "query_knowledge_hub"
        assert result.action_params["query"] == "REST API"
        assert result.action_params["top_k"] == 3
        assert result.action_params["collection"] == "api-docs"

    def test_parse_list_collections(self):
        raw = (
            "Thought: 先看看有哪些可用的集合\n"
            "Action: list_collections()"
        )
        result = parse_response(raw)
        assert result.action_name == "list_collections"
        assert result.action_params == {}

    def test_parse_get_document_summary(self):
        raw = (
            "Thought: 需要查看这个文档的摘要\n"
            'Action: get_document_summary(doc_id="doc_abc123")'
        )
        result = parse_response(raw)
        assert result.action_name == "get_document_summary"
        assert result.action_params["doc_id"] == "doc_abc123"

    def test_parse_multiline_thought(self):
        raw = (
            "Thought: 这个问题比较复杂。\n"
            "需要先查 A 的信息，\n"
            "然后再对比 B 的情况。\n"
            'Action: query_knowledge_hub(query="A", top_k=5)'
        )
        result = parse_response(raw)
        assert result.is_final is False
        assert "复杂" in result.thought
        assert result.action_name == "query_knowledge_hub"


class TestParseFinalAnswer:
    """Parse Final Answer format."""

    def test_parse_final_answer(self):
        raw = (
            "Final Answer: 项目 A 使用 React + Go 技术栈，"
            "项目 B 使用 Vue + Python。\n"
            "主要差异在于前端框架和后端语言的选择。"
        )
        result = parse_response(raw)
        assert result.is_final is True
        assert "React + Go" in result.answer
        assert "Vue + Python" in result.answer

    def test_parse_final_answer_multiline(self):
        raw = (
            "Final Answer: 综合分析：\n"
            "1. 项目 A 采用 React 前端\n"
            "2. 项目 B 采用 Vue 前端\n"
            "两者在生态和性能上有明显差异。"
        )
        result = parse_response(raw)
        assert result.is_final is True
        assert "React" in result.answer
        assert "Vue" in result.answer


class TestParseEdgeCases:
    """Edge cases and error handling."""

    def test_unknown_tool_returns_parsed_anyway(self):
        raw = (
            "Thought: 试试这个\n"
            'Action: unknown_tool(param="value")'
        )
        result = parse_response(raw)
        assert result.is_final is False
        assert result.action_name == "unknown_tool"

    def test_empty_input(self):
        result = parse_response("")
        assert result.is_final is False
        assert result.thought == ""

    def test_no_thought_no_action_no_final(self):
        raw = "这是一段随意的文本，没有遵守格式。"
        result = parse_response(raw)
        # should not crash; returned as unparseable
        assert result.is_final is False
        assert result.action_name is None
```

- [ ] **Step 4: Run tests — expected FAIL**

```bash
pytest tests/unit/agentic/test_response_parser.py -v
```
Expected: all fail — `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 5: Write minimal implementation**

`src/agentic/response_parser.py`:

```python
"""Parse LLM ReAct output into structured ParsedResponse.

Supports two formats:
- Thought: ... \\n Action: tool_name(key=value, ...)
- Final Answer: ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ParsedResponse:
    """Parsed ReAct response from LLM.

    Attributes:
        is_final: True if this is a Final Answer (loop should stop).
        thought: Reasoning text from Thought: section.
        action_name: Tool name from Action: section (None if final).
        action_params: Tool parameters parsed from Action: section.
        answer: Answer text from Final Answer: section (None if action).
    """

    is_final: bool = False
    thought: Optional[str] = None
    action_name: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None


def parse_response(raw: str) -> ParsedResponse:
    """Parse LLM raw output into a ParsedResponse.

    Args:
        raw: Raw text output from the LLM.

    Returns:
        ParsedResponse with structured fields.
    """
    if not raw or not raw.strip():
        return ParsedResponse()

    text = raw.strip()

    # Check for Final Answer first
    final_match = re.match(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return ParsedResponse(
            is_final=True,
            answer=final_match.group(1).strip(),
        )

    # Parse Thought: section (everything before Action:)
    thought: Optional[str] = None
    thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Parse Action: section
    action_match = re.search(
        r"Action:\s*(\w+)\((.*?)\)",
        text,
        re.DOTALL,
    )
    if action_match:
        action_name = action_match.group(1).strip()
        params_str = action_match.group(2).strip()
        action_params = _parse_params(params_str)
        return ParsedResponse(
            is_final=False,
            thought=thought,
            action_name=action_name,
            action_params=action_params,
        )

    # Neither final answer nor action found — return raw as thought
    return ParsedResponse(
        is_final=False,
        thought=text,
        action_name=None,
    )


def _parse_params(params_str: str) -> Dict[str, Any]:
    """Parse action parameters from LLM-generated string.

    Handles: key="value", key=123, key="escaped \\" quote"
    Returns a dict of parsed parameters.
    """
    if not params_str.strip():
        return {}

    params: Dict[str, Any] = {}

    # Match key=value pairs where value is quoted string or unquoted number
    pattern = r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|(\d+))'
    for match in re.finditer(pattern, params_str):
        key = match.group(1)
        str_val = match.group(2)
        int_val = match.group(3)

        if str_val is not None:
            params[key] = str_val.replace('\\"', '"').replace("\\\\", "\\")
        elif int_val is not None:
            params[key] = int(int_val)

    return params
```

- [ ] **Step 6: Run tests — expected PASS**

```bash
pytest tests/unit/agentic/test_response_parser.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentic/__init__.py src/agentic/response_parser.py tests/unit/agentic/
git commit -m "feat: add response_parser — parse LLM ReAct output (Thought/Action/Final)"
```

---

### Task 2: citation_tracker — 跨轮次引用追踪

**Files:**
- Create: `src/agentic/citation_tracker.py`
- Create: `tests/unit/agentic/test_citation_tracker.py`

**Interfaces:**
- Produces: `CitationTracker` class with `add(results)`, `format() -> str`, `all_sources -> List[str]`

- [ ] **Step 1: Write the failing test**

`tests/unit/agentic/test_citation_tracker.py`:

```python
"""Tests for citation_tracker — cross-round citation dedup and formatting."""

import pytest
from src.agentic.citation_tracker import CitationTracker
from src.core.types import RetrievalResult


class TestCitationTrackerAdd:
    """Test adding results and dedup."""

    def test_add_single_result(self):
        tracker = CitationTracker()
        results = [
            RetrievalResult(
                chunk_id="doc1_chunk_001",
                score=0.95,
                text="Azure OpenAI 配置...",
                metadata={"source_path": "docs/azure-guide.pdf", "title": "Azure Guide"},
            )
        ]
        tracker.add(results)
        assert len(tracker.all_sources) == 1
        assert "docs/azure-guide.pdf" in tracker.all_sources

    def test_add_dedup_same_chunk_id(self):
        tracker = CitationTracker()
        r1 = RetrievalResult(
            chunk_id="doc1_chunk_001",
            score=0.95,
            text="Azure config...",
            metadata={"source_path": "docs/azure-guide.pdf"},
        )
        r2 = RetrievalResult(
            chunk_id="doc1_chunk_001",  # same chunk_id
            score=0.90,
            text="Azure config...",
            metadata={"source_path": "docs/azure-guide.pdf"},
        )
        tracker.add([r1])
        tracker.add([r2])
        assert len(tracker.all_sources) == 1

    def test_add_multiple_rounds_across_sources(self):
        tracker = CitationTracker()
        tracker.add([
            RetrievalResult(
                chunk_id="doc1_chunk_001", score=0.95, text="...",
                metadata={"source_path": "docs/projectA.md"},
            )
        ])
        tracker.add([
            RetrievalResult(
                chunk_id="doc2_chunk_003", score=0.88, text="...",
                metadata={"source_path": "docs/projectB.md"},
            )
        ])
        sources = tracker.all_sources
        assert len(sources) == 2
        assert "docs/projectA.md" in sources
        assert "docs/projectB.md" in sources

    def test_add_empty_list(self):
        tracker = CitationTracker()
        tracker.add([])
        assert len(tracker.all_sources) == 0


class TestCitationTrackerFormat:
    """Test citation table formatting."""

    def test_format_single(self):
        tracker = CitationTracker()
        tracker.add([
            RetrievalResult(
                chunk_id="doc1_chunk_001", score=0.95, text="...",
                metadata={"source_path": "docs/report.pdf", "title": "Annual Report"},
            )
        ])
        formatted = tracker.format()
        assert "[1]" in formatted
        assert "docs/report.pdf" in formatted

    def test_format_multiple_ordered(self):
        tracker = CitationTracker()
        tracker.add([
            RetrievalResult(
                chunk_id="doc1_chunk_001", score=0.95, text="...",
                metadata={"source_path": "docs/first.pdf"},
            )
        ])
        tracker.add([
            RetrievalResult(
                chunk_id="doc2_chunk_001", score=0.80, text="...",
                metadata={"source_path": "docs/second.pdf"},
            )
        ])
        formatted = tracker.format()
        lines = formatted.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("[1]")
        assert lines[1].startswith("[2]")
        assert "docs/first.pdf" in lines[0]
        assert "docs/second.pdf" in lines[1]

    def test_format_empty(self):
        tracker = CitationTracker()
        assert tracker.format() == ""
```

- [ ] **Step 2: Run tests — expected FAIL**

```bash
pytest tests/unit/agentic/test_citation_tracker.py -v
```
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/agentic/citation_tracker.py`:

```python
"""Cross-round citation tracker for ReAct agent.

Tracks retrieval results across multiple rounds, deduplicates by chunk_id,
and formats a numbered citation table for the final answer.
"""

from __future__ import annotations

from typing import Dict, List

from src.core.types import RetrievalResult


class CitationTracker:
    """Accumulates retrieval results across ReAct rounds.

    Deduplicates by chunk_id and provides ordered source listing
    for citation table generation. Citations are based ONLY on
    actual retrieval results, not LLM self-reported sources —
    this prevents citation hallucination.
    """

    def __init__(self) -> None:
        self._seen_ids: Dict[str, str] = {}  # chunk_id -> source_path
        self._ordered_sources: List[str] = []

    def add(self, results: List[RetrievalResult]) -> None:
        """Add retrieval results from one round.

        Deduplicates by chunk_id — if the same chunk is retrieved
        in multiple rounds, its source only appears once.

        Args:
            results: List of RetrievalResult from a tool call.
        """
        for r in results:
            if r.chunk_id not in self._seen_ids:
                source = r.metadata.get("source_path", r.chunk_id)
                self._seen_ids[r.chunk_id] = source
                if source not in self._ordered_sources:
                    self._ordered_sources.append(source)

    @property
    def all_sources(self) -> List[str]:
        """Deduplicated list of source paths in order of first appearance.

        Returns:
            List of unique source_path strings.
        """
        return list(self._ordered_sources)

    def format(self) -> str:
        """Generate a numbered citation table.

        Returns:
            Multi-line string with [1] source_path format,
            or empty string if no citations accumulated.
        """
        if not self._ordered_sources:
            return ""

        lines = []
        for i, source in enumerate(self._ordered_sources, start=1):
            lines.append(f"[{i}] {source}")

        return "\n".join(lines)
```

- [ ] **Step 4: Run tests — expected PASS**

```bash
pytest tests/unit/agentic/test_citation_tracker.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic/citation_tracker.py tests/unit/agentic/test_citation_tracker.py
git commit -m "feat: add citation_tracker — cross-round dedup and citation table generation"
```

---

### Task 3: prompt_manager — System Prompt + Few-shot

**Files:**
- Create: `src/agentic/prompt_manager.py`
- Create: `tests/unit/agentic/test_prompt_manager.py`

**Interfaces:**
- Produces: `PromptManager.build_system_prompt() -> str`, `build_user_message(question) -> Message`

- [ ] **Step 1: Write the failing test**

`tests/unit/agentic/test_prompt_manager.py`:

```python
"""Tests for prompt_manager — system prompt and few-shot example generation."""

import pytest
from src.agentic.prompt_manager import PromptManager


class TestSystemPrompt:
    """Test system prompt content requirements."""

    def test_prompt_contains_tool_descriptions(self):
        prompt = PromptManager.build_system_prompt()
        assert "query_knowledge_hub" in prompt
        assert "list_collections" in prompt
        assert "get_document_summary" in prompt

    def test_prompt_contains_output_format(self):
        prompt = PromptManager.build_system_prompt()
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Final Answer:" in prompt

    def test_prompt_contains_behaviour_constraints(self):
        prompt = PromptManager.build_system_prompt()
        assert "10" in prompt  # max rounds reference

    def test_prompt_contains_few_shot_examples(self):
        prompt = PromptManager.build_system_prompt()
        assert "技术选型" in prompt or "technology" in prompt.lower()


class TestUserMessage:
    """Test user message construction."""

    def test_build_user_message(self):
        msg = PromptManager.build_user_message("对比 A 和 B 的技术差异")
        assert msg.role == "user"
        assert "对比 A 和 B" in msg.content

    def test_system_message(self):
        msg = PromptManager.build_system_message()
        assert msg.role == "system"
        assert len(msg.content) > 100
```

- [ ] **Step 2: Run tests — expected FAIL**

```bash
pytest tests/unit/agentic/test_prompt_manager.py -v
```
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/agentic/prompt_manager.py`:

```python
"""System prompt and message builder for ReAct agent.

Constructs the system prompt with tool descriptions, output format
instructions, behaviour constraints, and 3 few-shot examples covering
the main multi-hop patterns: comparison, causal, and breadth.
"""

from __future__ import annotations

from src.libs.llm.base_llm import Message


class PromptManager:
    """Builds prompts for the ReAct agent LLM calls."""

    @staticmethod
    def build_system_prompt() -> str:
        """Build the full system prompt.

        Returns:
            System prompt string with tool descriptions, format,
            constraints, and few-shot examples.
        """
        return _SYSTEM_PROMPT

    @staticmethod
    def build_system_message() -> Message:
        """Build the system Message for the LLM chat.

        Returns:
            Message with role='system' and the full system prompt.
        """
        return Message(role="system", content=_SYSTEM_PROMPT)

    @staticmethod
    def build_user_message(question: str) -> Message:
        """Build a user Message for the given question.

        Args:
            question: The user's complex question.

        Returns:
            Message with role='user' and the question as content.
        """
        return Message(role="user", content=question)


_SYSTEM_PROMPT = """你是一个知识检索 Agent，可以访问知识库来回答复杂问题。

## 可用工具

1. **query_knowledge_hub(query, top_k, collection?)** — 在知识库中搜索相关文档。返回文档片段及来源引用。
2. **list_collections()** — 列出所有可用的文档集合。
3. **get_document_summary(doc_id, collection?)** — 获取某个文档的元数据和内容摘要。

## 工作方式

对于复杂问题，你需要：
1. 分解问题 — 拆成子问题逐步解决
2. 检索信息 — 使用 query_knowledge_hub 搜索每个子问题
3. 评估结果 — 信息不够就换个角度重新搜索
4. 综合回答 — 整理所有来源的信息，给出完整答案

## 输出格式

如果需要执行操作，输出：
```
Thought: <你的推理过程>
Action: <tool_name>(<params>)
```

如果已有足够信息回答问题，输出：
```
Final Answer: <基于检索结果的综合答案>
```

## 重要规则

- 每轮最多 10 轮，请在信息足够时尽早输出 Final Answer
- 如果检索结果为空，请尝试改写查询关键词再搜
- 最终答案必须基于实际检索到的内容，不能编造
- 最终答案需要标注信息来源文档

## 示例 1: 对比类问题

用户: "对比项目 A 和项目 B 的技术选型差异"

Thought: 需要分别检索两个项目的技术栈信息
Action: query_knowledge_hub(query="项目A 技术选型 技术栈", top_k=5)

--- 收到检索结果 ---

Thought: 已获取项目 A 的技术栈信息，现在查项目 B
Action: query_knowledge_hub(query="项目B 技术选型 技术栈", top_k=5)

--- 收到检索结果 ---

Final Answer: 项目 A 使用 React + Go 技术栈，项目 B 使用 Vue + Python 技术栈。主要差异：前端框架（React vs Vue）、后端语言（Go vs Python）。

## 示例 2: 因果类问题

用户: "为什么选择了 ChromaDB 作为向量数据库？"

Thought: 需要搜索关于 ChromaDB 选型原因的相关文档
Action: query_knowledge_hub(query="ChromaDB 选型 原因 优势", top_k=5)

--- 收到检索结果 ---

Final Answer: 选择 ChromaDB 的主要原因是：1) 本地部署无需外部服务，2) 基于 SQLite 的持久化存储，3) 支持嵌入式向量检索...

## 示例 3: 广度类问题

用户: "知识库中有哪些关于性能优化的文档？"

Thought: 先列出所有集合了解文档范围
Action: list_collections()

--- 收到检索结果 ---

Thought: 知道了有哪些集合，现在在各个集合中搜索性能优化相关内容
Action: query_knowledge_hub(query="性能优化 best practices", top_k=5)

--- 收到检索结果 ---

Final Answer: 知识库中涉及性能优化的文档包括：[列出文档]...
"""
```

- [ ] **Step 4: Run tests — expected PASS**

```bash
pytest tests/unit/agentic/test_prompt_manager.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic/prompt_manager.py tests/unit/agentic/test_prompt_manager.py
git commit -m "feat: add prompt_manager — system prompt with 3 few-shot examples"
```

---

### Task 4: mcp_client — MCP 子进程客户端

**Files:**
- Create: `src/agentic/mcp_client.py`
- Create: `tests/integration/agentic/__init__.py`
- Create: `tests/integration/agentic/test_mcp_client.py`

**Interfaces:**
- Produces: `MCPClient` class with `connect()`, `call_tool(name, arguments) -> ToolResult`, `close()`

- [ ] **Step 1: Write the integration test**

`tests/integration/agentic/test_mcp_client.py`:

```python
"""Integration tests for MCPClient — connects to real MCP server subprocess."""

import pytest
from src.agentic.mcp_client import MCPClient


@pytest.fixture
async def mcp_client():
    """Create and connect an MCPClient to the real server subprocess."""
    client = MCPClient()
    await client.connect()
    yield client
    await client.close()


class TestMCPClientConnect:
    """Test connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        client = MCPClient()
        try:
            await client.connect()
            assert client._session is not None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_double_connect_is_idempotent(self, mcp_client):
        await mcp_client.connect()
        assert mcp_client._session is not None


class TestMCPClientCallTool:
    """Test calling real MCP tools."""

    @pytest.mark.asyncio
    async def test_list_collections(self, mcp_client):
        result = await mcp_client.call_tool("list_collections", {})
        assert result.name == "list_collections"
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_query_knowledge_hub(self, mcp_client):
        result = await mcp_client.call_tool(
            "query_knowledge_hub",
            {"query": "测试查询", "top_k": 2},
        )
        assert result.name == "query_knowledge_hub"
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mcp_client):
        result = await mcp_client.call_tool(
            "nonexistent_tool",
            {"param": "value"},
        )
        assert result.is_error is True
```

- [ ] **Step 2: Run tests — expected FAIL**

```bash
pytest tests/integration/agentic/test_mcp_client.py -v
```
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/agentic/mcp_client.py`:

```python
"""MCP stdio subprocess client.

Spawns the existing MCP server as a subprocess and communicates
via JSON-RPC over stdio. The agentic layer uses this as its only
channel to the RAG backend — same protocol as any external MCP client.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

TOOL_CALL_TIMEOUT = 30.0
CONNECT_TIMEOUT = 15.0


@dataclass
class ToolResult:
    """Result from a single MCP tool call."""

    name: str
    content: str
    is_error: bool = False
    raw: Optional[types.CallToolResult] = None


class MCPClient:
    """MCP stdio client that talks to the RAG server subprocess.

    Usage:
        client = MCPClient()
        await client.connect()
        result = await client.call_tool("query_knowledge_hub", {...})
        await client.close()
    """

    def __init__(self) -> None:
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._connected = False

    async def connect(self) -> None:
        """Start the MCP server subprocess and initialize the session.

        Retries once on failure.

        Raises:
            RuntimeError: If server subprocess fails to start after retry.
        """
        if self._connected and self._session is not None:
            logger.debug("MCPClient already connected, skipping")
            return

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server.server"],
            env=None,
        )

        for attempt in range(2):
            try:
                async with asyncio.timeout(CONNECT_TIMEOUT):
                    self._read, self._write = await stdio_client(server_params)
                    self._session = ClientSession(self._read, self._write)
                    await self._session.initialize()
                    self._connected = True
                    logger.info("MCPClient connected to server subprocess")
                    return
            except Exception as e:
                logger.warning(
                    f"MCP connection attempt {attempt + 1}/2 failed: {e}"
                )
                if attempt == 1:
                    raise RuntimeError(
                        "无法启动 MCP Server 子进程。\n"
                        f"错误: {e}\n"
                        "请手动启动: python -m src.mcp_server.server"
                    ) from e
                await asyncio.sleep(1.0)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool on the MCP server.

        Args:
            name: Tool name (e.g. 'query_knowledge_hub').
            arguments: Tool parameters as a dict.

        Returns:
            ToolResult with flattened text content and error status.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._connected or self._session is None:
            raise RuntimeError("MCPClient not connected. Call connect() first.")

        try:
            async with asyncio.timeout(TOOL_CALL_TIMEOUT):
                result: types.CallToolResult = await self._session.call_tool(
                    name, arguments
                )
        except asyncio.TimeoutError:
            return ToolResult(
                name=name,
                content="(工具调用超时)",
                is_error=True,
            )

        text_parts: List[str] = []
        is_error = getattr(result, "isError", False)

        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "data"):
                text_parts.append(f"[image data: {len(block.data)} bytes]")

        return ToolResult(
            name=name,
            content="\n".join(text_parts) if text_parts else "(empty response)",
            is_error=is_error,
            raw=result,
        )

    async def close(self) -> None:
        """Close the MCP session and terminate the subprocess."""
        if not self._connected:
            return

        try:
            if self._session is not None:
                pass
        except Exception as e:
            logger.debug(f"Error during session cleanup: {e}")
        finally:
            self._connected = False
            self._session = None
            self._read = None
            self._write = None
            logger.info("MCPClient disconnected")
```

- [ ] **Step 4: Run tests — expected PASS**

```bash
pytest tests/integration/agentic/test_mcp_client.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic/mcp_client.py tests/integration/agentic/
git commit -m "feat: add mcp_client — MCP stdio subprocess client for server communication"
```

---

### Task 5: tool_registry — 工具定义与执行

**Files:**
- Create: `src/agentic/tool_registry.py`
- Create: `tests/unit/agentic/test_tool_registry.py`

**Interfaces:**
- Consumes: `MCPClient` (from Task 4)
- Produces: `ToolRegistry` with `parse_action(text) -> ParsedAction`, `execute(action) -> Observation`

- [ ] **Step 1: Write the failing test**

`tests/unit/agentic/test_tool_registry.py`:

```python
"""Tests for tool_registry — tool definitions and action parsing."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agentic.tool_registry import ToolRegistry, ParsedAction, Observation


@pytest.fixture
def mock_mcp_client():
    client = MagicMock()
    client.call_tool = AsyncMock()
    return client


@pytest.fixture
def registry(mock_mcp_client):
    return ToolRegistry(mock_mcp_client)


class TestParseAction:
    """Test action text parsing into ParsedAction."""

    def test_parse_query_with_all_params(self, registry):
        action = registry.parse_action(
            'query_knowledge_hub(query="test", top_k=5, collection="docs")'
        )
        assert action.name == "query_knowledge_hub"
        assert action.params["query"] == "test"
        assert action.params["top_k"] == 5
        assert action.params["collection"] == "docs"

    def test_parse_query_minimal(self, registry):
        action = registry.parse_action(
            'query_knowledge_hub(query="hello")'
        )
        assert action.name == "query_knowledge_hub"
        assert action.params["query"] == "hello"

    def test_parse_list_collections(self, registry):
        action = registry.parse_action("list_collections()")
        assert action.name == "list_collections"
        assert action.params == {}

    def test_parse_get_document_summary(self, registry):
        action = registry.parse_action(
            'get_document_summary(doc_id="abc123")'
        )
        assert action.name == "get_document_summary"
        assert action.params["doc_id"] == "abc123"

    def test_parse_unknown_tool_still_works(self, registry):
        action = registry.parse_action(
            'unknown_tool(param="value")'
        )
        assert action.name == "unknown_tool"
        assert action.params["param"] == "value"


class TestExecute:
    """Test tool execution delegation to MCPClient."""

    @pytest.mark.asyncio
    async def test_execute_calls_mcp_client(self, registry, mock_mcp_client):
        mock_mcp_client.call_tool.return_value = MagicMock(
            name="query_knowledge_hub",
            content="test result content",
            is_error=False,
        )
        action = ParsedAction(
            name="query_knowledge_hub",
            params={"query": "test", "top_k": 5},
        )
        obs = await registry.execute(action)
        mock_mcp_client.call_tool.assert_called_once_with(
            "query_knowledge_hub", {"query": "test", "top_k": 5}
        )
        assert "test result content" in obs.text

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self, registry):
        action = ParsedAction(name="nonexistent_tool", params={})
        obs = await registry.execute(action)
        assert obs.is_error is True
        assert "Unknown tool" in obs.text

    @pytest.mark.asyncio
    async def test_execute_timeout(self, registry, mock_mcp_client):
        import asyncio
        mock_mcp_client.call_tool.side_effect = asyncio.TimeoutError()
        action = ParsedAction(name="query_knowledge_hub", params={"query": "x"})
        obs = await registry.execute(action)
        assert "timeout" in obs.text.lower() or "超时" in obs.text
```

- [ ] **Step 2: Run tests — expected FAIL**

```bash
pytest tests/unit/agentic/test_tool_registry.py -v
```
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/agentic/tool_registry.py`:

```python
"""Tool definitions, action parsing, and execution delegation for ReAct agent.

Maps LLM-generated Action strings to MCP tool calls via MCPClient.
3 tools: query_knowledge_hub, list_collections, get_document_summary.
"""

from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.agentic.mcp_client import MCPClient, ToolResult

logger = logging.getLogger(__name__)

_TOOL_DEFS: Dict[str, Dict[str, Any]] = {
    "query_knowledge_hub": {
        "description": "在知识库中搜索相关文档。返回文档片段及来源引用。",
    },
    "list_collections": {
        "description": "列出所有可用的文档集合。",
    },
    "get_document_summary": {
        "description": "获取某个文档的元数据和内容摘要。",
    },
}


@dataclass
class ParsedAction:
    """A parsed tool invocation from LLM output."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Observation returned to the LLM after a tool call."""

    text: str
    is_error: bool = False
    results: List[RetrievalResult] = field(default_factory=list)


class ToolRegistry:
    """Manages tool definitions and executes actions via MCPClient."""

    def __init__(self, mcp_client: MCPClient) -> None:
        self._mcp_client = mcp_client

    def parse_action(self, action_text: str) -> ParsedAction:
        """Parse LLM-generated action string into a ParsedAction.

        Args:
            action_text: The action portion of LLM output
                         e.g. 'tool_name(key="value", key2=123)'.

        Returns:
            ParsedAction with tool name and parameters.
        """
        match = re.match(r"(\w+)\((.*)\)", action_text.strip(), re.DOTALL)
        if not match:
            return ParsedAction(name=action_text.strip(), params={})

        name = match.group(1).strip()
        params_str = match.group(2).strip()
        params = self._parse_params(params_str)

        return ParsedAction(name=name, params=params)

    async def execute(self, action: ParsedAction) -> Observation:
        """Execute a parsed action via MCPClient.

        Args:
            action: ParsedAction to execute.

        Returns:
            Observation with result text, error status, and parsed results.
        """
        if action.name not in _TOOL_DEFS:
            known = ", ".join(sorted(_TOOL_DEFS.keys()))
            return Observation(
                text=f"Unknown tool: {action.name}. Available tools: {known}",
                is_error=True,
            )

        try:
            result: ToolResult = await self._mcp_client.call_tool(
                action.name, action.params
            )
        except Exception as e:
            return Observation(
                text=f"Tool call failed: {e}",
                is_error=True,
            )

        parsed_results: List[RetrievalResult] = []
        if action.name == "query_knowledge_hub" and not result.is_error:
            parsed_results = self._extract_results(result.content)

        truncated = self._truncate_content(result.content, max_chars=500)

        return Observation(
            text=truncated,
            is_error=result.is_error,
            results=parsed_results,
        )

    @staticmethod
    def _parse_params(params_str: str) -> Dict[str, Any]:
        """Parse key=value parameters from LLM action string."""
        if not params_str.strip():
            return {}

        params: Dict[str, Any] = {}
        pattern = r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|(\d+))'
        for match in re.finditer(pattern, params_str):
            key = match.group(1)
            str_val = match.group(2)
            int_val = match.group(3)

            if str_val is not None:
                params[key] = str_val.replace('\\"', '"').replace("\\\\", "\\")
            elif int_val is not None:
                params[key] = int(int_val)

        return params

    @staticmethod
    def _extract_results(content: str) -> List[RetrievalResult]:
        """Extract RetrievalResult list from query_knowledge_hub response JSON."""
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group(1))
            final_results = data.get("metadata", {}).get("final_results", [])

            results: List[RetrievalResult] = []
            for item in final_results:
                results.append(RetrievalResult(
                    chunk_id=item.get("chunk_id", ""),
                    score=item.get("score", 0.0),
                    text=item.get("text", ""),
                    metadata={
                        "source_path": item.get("source", ""),
                        "title": item.get("title", ""),
                    },
                ))
            return results
        except (json.JSONDecodeError, KeyError):
            return []

    @staticmethod
    def _truncate_content(text: str, max_chars: int = 500) -> str:
        """Truncate text content to max characters at word boundary."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars].rsplit(" ", 1)[0]
        return truncated + "..."
```

- [ ] **Step 4: Run tests — expected PASS**

```bash
pytest tests/unit/agentic/test_tool_registry.py -v
```
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic/tool_registry.py tests/unit/agentic/test_tool_registry.py
git commit -m "feat: add tool_registry — action parsing, validation, and MCP delegation"
```

---

### Task 6: react_loop — ReAct 循环状态机

**Files:**
- Create: `src/agentic/react_loop.py`
- Create: `tests/unit/agentic/test_react_loop.py`

**Interfaces:**
- Consumes: All previous tasks + `BaseLLM.chat()` (existing)
- Produces: `ReActLoop`, `AgenticResult`, `RoundTrace`

- [ ] **Step 1: Write the failing test**

`tests/unit/agentic/test_react_loop.py`:

```python
"""Tests for react_loop — the ReAct reasoning loop."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agentic.react_loop import ReActLoop, AgenticResult, RoundTrace
from src.libs.llm.base_llm import ChatResponse


@pytest.fixture
def mock_mcp_client():
    client = MagicMock()
    client.call_tool = AsyncMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_llm():
    return MagicMock()


def make_response(content: str) -> ChatResponse:
    return ChatResponse(content=content, model="test-model")


class TestReActLoopRun:
    """Test the core ReAct loop execution."""

    @pytest.mark.asyncio
    async def test_single_hop_question(self, mock_mcp_client, mock_llm):
        mock_llm.chat.side_effect = [
            make_response(
                'Thought: 需要查一下\n'
                'Action: query_knowledge_hub(query="test query", top_k=3)'
            ),
            make_response("Final Answer: 根据检索结果，答案是 X。"),
        ]

        mock_mcp_client.call_tool.return_value = MagicMock(
            name="query_knowledge_hub",
            content="模拟的检索结果内容...",
            is_error=False,
        )

        loop = ReActLoop(
            mcp_client=mock_mcp_client,
            llm_client=mock_llm,
            max_rounds=10,
            context_window=3,
        )

        result = await loop.run("test query")

        assert isinstance(result, AgenticResult)
        assert "X" in result.answer
        assert mock_mcp_client.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_multi_hop_question(self, mock_mcp_client, mock_llm):
        mock_llm.chat.side_effect = [
            make_response(
                'Thought: 第一步\n'
                'Action: query_knowledge_hub(query="first", top_k=5)'
            ),
            make_response(
                'Thought: 还需要更多\n'
                'Action: query_knowledge_hub(query="second", top_k=5)'
            ),
            make_response("Final Answer: 综合两次检索的回答。"),
        ]

        mock_mcp_client.call_tool.return_value = MagicMock(
            name="query_knowledge_hub",
            content="检索结果...",
            is_error=False,
        )

        loop = ReActLoop(
            mcp_client=mock_mcp_client,
            llm_client=mock_llm,
            max_rounds=10,
            context_window=3,
        )

        result = await loop.run("complex question")
        assert result.answer
        assert mock_mcp_client.call_tool.call_count == 2
        assert len(result.trace) == 2

    @pytest.mark.asyncio
    async def test_max_rounds_forced_stop(self, mock_mcp_client, mock_llm):
        responses = [
            make_response(
                f'Thought: 第{i}轮\n'
                f'Action: query_knowledge_hub(query="search{i}", top_k=3)'
            )
            for i in range(10)
        ]
        mock_llm.chat.side_effect = responses
        mock_mcp_client.call_tool.return_value = MagicMock(
            name="query_knowledge_hub",
            content="result",
            is_error=False,
        )

        loop = ReActLoop(
            mcp_client=mock_mcp_client,
            llm_client=mock_llm,
            max_rounds=3,
            context_window=3,
        )

        result = await loop.run("test")
        assert mock_mcp_client.call_tool.call_count <= 3
        assert result.answer

    @pytest.mark.asyncio
    async def test_context_trimming(self, mock_mcp_client, mock_llm):
        calls = []
        for i in range(5):
            calls.append(make_response(
                f'Thought: round{i}\n'
                f'Action: query_knowledge_hub(query="q{i}", top_k=2)'
            ))
        calls.append(make_response("Final Answer: done."))
        mock_llm.chat.side_effect = calls
        mock_mcp_client.call_tool.return_value = MagicMock(
            name="query_knowledge_hub",
            content="result...",
            is_error=False,
        )

        loop = ReActLoop(
            mcp_client=mock_mcp_client,
            llm_client=mock_llm,
            max_rounds=10,
            context_window=2,
        )

        result = await loop.run("test")
        assert result.answer

    @pytest.mark.asyncio
    async def test_tool_error_observed(self, mock_mcp_client, mock_llm):
        mock_llm.chat.side_effect = [
            make_response(
                'Thought: try bad tool\n'
                'Action: unknown_tool(param="x")'
            ),
            make_response(
                'Thought: 改用正确的工具\n'
                'Action: query_knowledge_hub(query="real", top_k=3)'
            ),
            make_response("Final Answer: recovered."),
        ]

        async def call_tool_side_effect(name, args):
            result = MagicMock()
            result.name = name
            result.is_error = (name == "unknown_tool")
            result.content = "Unknown tool" if name == "unknown_tool" else "good"
            return result

        mock_mcp_client.call_tool.side_effect = call_tool_side_effect

        loop = ReActLoop(
            mcp_client=mock_mcp_client,
            llm_client=mock_llm,
            max_rounds=10,
            context_window=3,
        )

        result = await loop.run("test")
        assert result.answer is not None
```

- [ ] **Step 2: Run tests — expected FAIL**

```bash
pytest tests/unit/agentic/test_react_loop.py -v
```
Expected: all fail with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

`src/agentic/react_loop.py`:

```python
"""ReAct (Reasoning + Acting) loop for agentic RAG.

Core state machine that orchestrates the Thought → Action → Observation
cycle, with safety constraints: max rounds, loop detection, and context
trimming.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.agentic.citation_tracker import CitationTracker
from src.agentic.prompt_manager import PromptManager
from src.agentic.response_parser import parse_response, ParsedResponse
from src.agentic.tool_registry import ToolRegistry
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message

if TYPE_CHECKING:
    from src.agentic.mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class RoundTrace:
    """A single round of the ReAct loop for observability."""

    round_number: int
    thought: str
    action: str
    observation: str


@dataclass
class AgenticResult:
    """Final result from the ReAct agent."""

    answer: str
    citations: List[str] = field(default_factory=list)
    trace: List[RoundTrace] = field(default_factory=list)


class ReActLoop:
    """ReAct reasoning loop for multi-hop knowledge retrieval.

    Example:
        loop = ReActLoop(mcp_client, llm, max_rounds=10)
        result = await loop.run("对比项目 A 和 B 的技术差异")
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        llm_client: BaseLLM,
        max_rounds: int = 10,
        context_window: int = 3,
    ) -> None:
        self._mcp_client = mcp_client
        self._llm = llm_client
        self.max_rounds = max_rounds
        self.context_window = context_window

        self._tool_registry = ToolRegistry(mcp_client)
        self._citation_tracker = CitationTracker()
        self._trace: List[RoundTrace] = []

    async def run(self, question: str) -> AgenticResult:
        """Run the ReAct loop for a given question."""
        system_msg = PromptManager.build_system_message()
        user_msg = PromptManager.build_user_message(question)
        messages: List[Message] = [system_msg, user_msg]

        last_action_key: Optional[str] = None
        repeat_count: int = 0
        final_answer: Optional[str] = None

        for round_num in range(1, self.max_rounds + 1):
            # 1. Call LLM
            response: ChatResponse = await asyncio.to_thread(
                self._llm.chat, messages
            )

            # 2. Parse response
            parsed: ParsedResponse = parse_response(response.content)

            # 3. Check for final answer
            if parsed.is_final:
                final_answer = parsed.answer or response.content
                messages.append(Message(role="assistant", content=response.content))
                logger.info(f"ReAct loop finished at round {round_num}")
                break

            # 4. Execute action
            action = parsed.action_name
            params = parsed.action_params
            action_text = (
                f"{action}({_format_params(params)})" if action else "(no action)"
            )

            if action is None:
                observation_text = (
                    "无法解析你的输出。请使用以下格式：\n"
                    "Thought: <你的推理>\n"
                    'Action: tool_name(key="value", ...)\n\n'
                    "或者：\n"
                    "Final Answer: <你的答案>"
                )
                messages.append(Message(role="user", content=_format_obs(observation_text)))
                continue

            parsed_action = self._tool_registry.parse_action(action_text)
            observation = await self._tool_registry.execute(parsed_action)

            # 5. Track citations
            if observation.results:
                self._citation_tracker.add(observation.results)

            # 6. Loop detection
            current_key = f"{parsed_action.name}:{_params_key(parsed_action.params)}"
            if current_key == last_action_key:
                repeat_count += 1
                if repeat_count >= 2:
                    observation_text = (
                        f"{observation.text}\n\n"
                        "[警告] 连续 2 次相同调用，请换策略或输出 Final Answer。"
                    )
            else:
                repeat_count = 0
            last_action_key = current_key

            # 7. Append to messages
            messages.append(Message(role="assistant", content=response.content))
            messages.append(Message(role="user", content=_format_obs(observation.text)))

            # 8. Record trace
            self._trace.append(RoundTrace(
                round_number=round_num,
                thought=parsed.thought or "",
                action=action_text,
                observation=observation.text,
            ))

            # 9. Trim context
            messages = self._trim_context(messages)

        # Force answer if loop exhausted
        if final_answer is None:
            logger.warning(f"ReAct loop hit max_rounds ({self.max_rounds})")
            force_prompt = "已达到最大轮次。请基于以上检索结果给出最佳答案。"
            messages.append(Message(role="user", content=force_prompt))
            response = await asyncio.to_thread(self._llm.chat, messages)
            final_answer = response.content

        citation_text = self._citation_tracker.format()
        citations = citation_text.split("\n") if citation_text else []

        return AgenticResult(
            answer=final_answer,
            citations=citations,
            trace=list(self._trace),
        )

    def _trim_context(self, messages: List[Message]) -> List[Message]:
        """Trim messages to keep system prompt + recent context."""
        system = messages[0] if messages else None
        non_system = messages[1:]
        keep_count = self.context_window * 2
        if len(non_system) > keep_count:
            non_system = non_system[-keep_count:]
        result = [system] if system else []
        result.extend(non_system)
        return result


def _format_obs(text: str) -> str:
    return f"--- 工具返回结果 ---\n{text}\n--- 结束 ---"


def _format_params(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _params_key(params: Dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))
```

- [ ] **Step 4: Run tests — expected PASS**

```bash
pytest tests/unit/agentic/test_react_loop.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic/react_loop.py tests/unit/agentic/test_react_loop.py
git commit -m "feat: add react_loop — ReAct reasoning loop with constraints and trace"
```

---

### Task 7: agentic_query.py — CLI 入口

**Files:**
- Create: `scripts/agentic_query.py`

**Interfaces:**
- Consumes: `ReActLoop`, `MCPClient`, `LLMFactory`, `load_settings`

- [ ] **Step 1: Write the CLI script**

`scripts/agentic_query.py`:

```python
#!/usr/bin/env python3
"""Agentic RAG Query CLI — ReAct agent for complex multi-hop questions.

Usage:
    python scripts/agentic_query.py --query "对比A和B的技术差异"
    python scripts/agentic_query.py --query "..." --verbose --max-rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from src.agentic.mcp_client import MCPClient
from src.agentic.react_loop import ReActLoop, AgenticResult
from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG — ReAct agent for multi-hop knowledge retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/agentic_query.py --query "对比项目A和项目B的技术选型差异"
  python scripts/agentic_query.py --query "为什么选择了这个架构？" --verbose
        """,
    )
    parser.add_argument(
        "--query", "-q", required=True, help="The complex question to answer",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full Thought/Action/Observation per round",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=10,
        help="Maximum ReAct rounds (default: 10)",
    )
    return parser.parse_args()


async def run_agent(
    question: str,
    max_rounds: int,
    verbose: bool,
) -> AgenticResult:
    """Initialize components and run the ReAct loop."""
    settings = load_settings()
    llm = LLMFactory.create(settings)

    client = MCPClient()
    await client.connect()

    try:
        loop = ReActLoop(
            mcp_client=client,
            llm_client=llm,
            max_rounds=max_rounds,
            context_window=3,
        )

        if verbose:
            print(f"问题: {question}", file=sys.stderr)
            print(f"最大轮次: {max_rounds}", file=sys.stderr)
            print("-" * 50, file=sys.stderr)

        result = await loop.run(question)

        if verbose:
            _print_verbose_trace(result)

        return result

    finally:
        await client.close()


def _print_verbose_trace(result: AgenticResult) -> None:
    """Print full ReAct trace to stderr."""
    for t in result.trace:
        print(f"\n[Round {t.round_number}]", file=sys.stderr)
        print(f"  Thought: {t.thought}", file=sys.stderr)
        print(f"  Action:  {t.action}", file=sys.stderr)
        obs = t.observation[:200] + "..." if len(t.observation) > 200 else t.observation
        print(f"  Obs:     {obs}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)


def _print_result(result: AgenticResult, verbose: bool) -> None:
    """Print final result to stdout."""
    print(result.answer)
    print()

    if result.citations:
        print("---")
        print("引用来源:")
        for citation in result.citations:
            print(f"  {citation}")

    if not verbose and result.trace:
        print(f"\n(共 {len(result.trace)} 轮检索)")


async def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result: Optional[AgenticResult] = None

    try:
        result = await run_agent(
            question=args.query,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\n用户中断。", file=sys.stderr)
        if result is not None:
            _print_result(result, args.verbose)
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        if result is not None:
            _print_result(result, args.verbose)
        sys.exit(1)

    _print_result(result, args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/agentic_query.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/agentic_query.py
git commit -m "feat: add agentic_query CLI — ReAct agent entry point"
```

---

### Task 8: Integration Test — 完整 ReAct 循环

**Files:**
- Create: `tests/integration/agentic/test_react_loop_integration.py`

- [ ] **Step 1: Write the integration test**

`tests/integration/agentic/test_react_loop_integration.py`:

```python
"""Integration tests: full ReAct loop with real MCP server + LLM.

Marked 'llm' and 'slow' to allow exclusion in CI without API keys.
"""

import pytest
from src.agentic.mcp_client import MCPClient
from src.agentic.react_loop import ReActLoop, AgenticResult
from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory


@pytest.fixture
async def react_loop():
    """Create a ReActLoop with real MCP client and LLM."""
    settings = load_settings()
    llm = LLMFactory.create(settings)
    client = MCPClient()
    await client.connect()

    loop = ReActLoop(
        mcp_client=client,
        llm_client=llm,
        max_rounds=5,
        context_window=3,
    )

    yield loop

    await client.close()


class TestReActLoopIntegration:
    """Integration tests requiring real MCP server + LLM."""

    @pytest.mark.llm
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_simple_query_produces_answer(self, react_loop):
        """A simple query should produce a non-empty answer."""
        result = await react_loop.run("什么是知识库？")
        assert isinstance(result, AgenticResult)
        assert result.answer
        assert len(result.answer) > 10

    @pytest.mark.llm
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_result_has_trace(self, react_loop):
        """Result should contain round traces."""
        result = await react_loop.run("列出知识库中的文档")
        assert isinstance(result, AgenticResult)
        assert len(result.trace) >= 1

    @pytest.mark.llm
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_no_hallucinated_citations(self, react_loop):
        """Citations should be properly formatted."""
        result = await react_loop.run("搜索一些技术文档")
        for citation in result.citations:
            if citation.strip():
                assert citation.strip().startswith("["), (
                    f"Citation not properly formatted: {citation}"
                )
```

- [ ] **Step 2: Verify tests skip without LLM**

```bash
pytest tests/integration/agentic/test_react_loop_integration.py -v -m "not llm"
```
Expected: 0 tests collected (all skipped).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/agentic/test_react_loop_integration.py
git commit -m "test: add integration tests for full ReAct loop (llm-marked)"
```

---

### Task 9: Run Full Test Suite

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/unit/agentic/ -v
```
Expected: all unit tests PASS (response_parser 9, citation_tracker 7, prompt_manager 6, tool_registry 8, react_loop 5 = 35 tests).

- [ ] **Step 2: Run integration tests (exclude LLM)**

```bash
pytest tests/integration/agentic/ -v -m "not llm"
```
Expected: MCP client tests PASS.

- [ ] **Step 3: Run full test suite — verify no regressions**

```bash
pytest tests/unit/ -v -m "not llm and not slow"
```
Expected: existing tests still PASS, no regressions.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: finalize agentic RAG — all tests passing, no regressions"
```