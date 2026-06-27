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
