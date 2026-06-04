"""Query Intent Router — classify queries and route to optimal retrieval strategy.

Design principle:
- LLM as lightweight classifier, NOT as answer generator
- Zero-shot classification with structured JSON output
- Fallback to "search" on any failure (never block the user)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.libs.llm.base_llm import Message

if TYPE_CHECKING:
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    CHAT = "chat"          # 闲聊 / 元问题："你好", "你能做什么"
    SEARCH = "search"      # 知识检索："Azure 怎么配置", "RAG 的原理"
    FILTER = "filter"      # 带过滤条件的检索："2024年的合同", "王工写的报告"
    COMPARE = "compare"    # 对比/多步推理："A方案和B方案有什么区别"


@dataclass
class RoutingDecision:
    intent: QueryIntent
    confidence: float
    extracted_filters: Dict[str, Any] = field(default_factory=dict)
    sub_queries: List[str] = field(default_factory=list)
    reasoning: str = ""


class IntentRouter:
    """LLM-based query intent classifier.

    Uses a lightweight LLM call (1-2 tokens of thinking + short JSON output)
    to classify the user's query intent.  On any failure, gracefully falls
    back to ``QueryIntent.SEARCH`` so the user never sees an error.

    Args:
        llm: An LLM instance from LLMFactory.  The router only needs a
            small/capable model — it never generates long answers.
    """

    # ── Classifier system prompt ──────────────────────────────────
    SYSTEM_PROMPT = """你是一个查询意图分类器。分析用户查询，返回JSON格式的分类结果。

## 意图类型 (intent):
- "chat": 闲聊、问候、元问题（"你好"、"你能做什么"、"今天星期几"）
- "search": 知识检索 → 需要从文档中找到信息
- "filter": 知识检索 + 元数据过滤 → 查询中包含时间、作者、文档类型等筛选条件
  - 示例："2024年的合同"、"张三写的报告"、"最近的会议纪要"
- "compare": 对比分析 → 需要比较两个或多个事物
  - 示例："RAG和微调有什么区别"、"方案A和方案B哪个更好"

## 过滤条件提取 (extracted_filters):
仅当 intent="filter" 时填写。支持的过滤键:
- date_range: "2024" | "2025-Q1" | "最近三个月" | null
- author: 作者名 | null
- doc_type: "合同" | "报告" | "会议纪要" | null
- collection: 集合名 | null

## 子查询拆分 (sub_queries):
仅当 intent="compare" 时填写。将对比问题拆成2-3个独立查询。
- 示例: "RAG和微调有什么区别" → ["RAG技术详解", "微调技术详解", "RAG与微调的对比"]

## 规则:
1. 默认 intent="search"，除非明确匹配其他意图
2. confidence 表示你的确信度 (0.0-1.0)
3. 只返回JSON，不要额外解释

## 返回JSON格式:
{
  "intent": "chat|search|filter|compare",
  "confidence": 0.85,
  "extracted_filters": {},
  "sub_queries": [],
  "reasoning": "一句话说明分类理由"
}"""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def classify(self, query: str) -> RoutingDecision:
        """Classify a single query and return routing decision.

        Args:
            query: The user's raw query string.

        Returns:
            RoutingDecision with intent, confidence, filters.
            Defaults to SEARCH on any failure.
        """
        # Pre-check: obvious chat queries don't need LLM
        obvious_chat = self._pre_check(query)
        if obvious_chat is not None:
            return obvious_chat

        try:
            return self._llm_classify(query)
        except Exception as exc:
            logger.warning("Intent classification failed, falling back to search: %s", exc)
            return RoutingDecision(
                intent=QueryIntent.SEARCH,
                confidence=0.3,
                reasoning=f"Fallback due to error: {exc}",
            )

    def _pre_check(self, query: str) -> Optional[RoutingDecision]:
        """Rule-based fast path for obvious cases.

        Avoids an LLM call for queries that are clearly chat.
        Returns None if rules can't decide — caller should fall through to LLM.
        """
        q = query.strip().lower()

        # Chinese greetings
        if q in ("你好", "嗨", "hello", "hi", "在吗", "谢谢", "再见"):
            return RoutingDecision(
                intent=QueryIntent.CHAT,
                confidence=0.99,
                reasoning="Obvious greeting",
            )

        # Meta questions about the system itself
        meta_patterns = ("你能做什么", "你能干什么", "你会什么", "你是谁",
                         "what can you do", "who are you")
        if any(p in q for p in meta_patterns):
            return RoutingDecision(
                intent=QueryIntent.CHAT,
                confidence=0.95,
                reasoning="Meta question about capabilities",
            )

        # Very short queries that end with ?  → likely search
        # Very long queries → likely search or compare
        # Return None to let LLM decide when unsure
        return None

    def _llm_classify(self, query: str) -> RoutingDecision:
        """Call LLM for zero-shot classification."""
        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=f"查询: {query}"),
        ]

        response = self.llm.chat(messages, temperature=0.0, max_tokens=256)

        # Parse JSON from response
        content = response.content.strip()
        # Remove markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]

        data = json.loads(content)

        intent_str = data.get("intent", "search")
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.SEARCH

        return RoutingDecision(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            extracted_filters=data.get("extracted_filters", {}),
            sub_queries=data.get("sub_queries", []),
            reasoning=data.get("reasoning", ""),
        )
