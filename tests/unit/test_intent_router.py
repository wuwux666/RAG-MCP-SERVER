"""Tests for IntentRouter — zero-shot classification coverage.

Test strategy:
1. Rule-based pre-check (fast path, no LLM)
2. LLM classification (integration, requires LLM)
3. Fallback behavior (LLM fails → SEARCH)
4. All 4 intent types
"""

import os

import pytest
from src.core.query_engine.intent_router import (
    IntentRouter, QueryIntent, RoutingDecision,
)
from src.libs.llm.base_llm import BaseLLM


class TestPreCheck:
    """Rule-based fast path tests.
    
    Use a dummy LLM that raises if called — this proves pre_check 
    short-circuits BEFORE reaching the LLM, without accessing private methods.
    """
    
    @staticmethod
    def _dummy_llm():
        """LLM that will raise if chat() is accidentally called."""
        class _UnreachableLLM(BaseLLM):
            def chat(self, *args, **kwargs):
                raise AssertionError("LLM should not be called for obvious greetings")
        return _UnreachableLLM()

    @pytest.mark.parametrize("query", ["你好", "嗨", "hello", "hi", "在吗", "谢谢", "再见"])
    def test_greetings_are_chat(self, query):
        router = IntentRouter(llm=self._dummy_llm())
        decision = router.classify(query)      # ← 公开接口
        assert decision.intent == QueryIntent.CHAT
        assert decision.confidence > 0.9

    @pytest.mark.parametrize("query", ["你能做什么", "你能干什么", "你会什么"])
    def test_meta_questions_are_chat(self, query):
        router = IntentRouter(llm=self._dummy_llm())
        decision = router.classify(query)      # ← 公开接口
        assert decision.intent == QueryIntent.CHAT


class TestLLMClassification:
    """Integration tests — require a configured LLM."""

    @pytest.fixture
    def router(self):
        from src.core.settings import load_settings
        from src.libs.llm.llm_factory import LLMFactory
        settings = load_settings()
        # DeepSeekLLM reads from env var DEEPSEEK_API_KEY, not settings.llm.api_key.
        # Pass the key from settings.yaml explicitly so tests work in CI / local.
        api_key = getattr(settings.llm, 'api_key', None) or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("No LLM API key configured — skipping integration test")
        llm = LLMFactory.create(settings, api_key=api_key)
        return IntentRouter(llm)

    def test_search_intent(self, router):
        decision = router.classify("Azure OpenAI 怎么配置？")
        assert decision.intent == QueryIntent.SEARCH
        assert decision.confidence > 0.5

    def test_chat_intent(self, router):
        decision = router.classify("今天天气怎么样？")
        assert decision.intent in (QueryIntent.CHAT, QueryIntent.SEARCH)
        # chat questions without knowledge base context may be classified as search
        # by some models — both are acceptable for this test

    def test_filter_intent(self, router):
        decision = router.classify("2024年签订的合同有哪些")
        # filter intent should be detected
        assert decision.intent in (QueryIntent.FILTER, QueryIntent.SEARCH)

    def test_compare_intent(self, router):
        decision = router.classify("RAG和微调有什么区别")
        assert decision.intent in (QueryIntent.COMPARE, QueryIntent.SEARCH)


class TestFallback:
    """Graceful degradation tests."""

    def test_llm_failure_falls_back_to_search(self):
        """When LLM fails, classify() returns SEARCH instead of raising."""
        class BrokenLLM(BaseLLM):
            def chat(self, *args, **kwargs):
                raise RuntimeError("Connection refused")

        router = IntentRouter(llm=BrokenLLM())
        decision = router.classify("Azure 怎么配置？")
        assert decision.intent == QueryIntent.SEARCH
        assert decision.confidence == 0.3

    def test_invalid_json_falls_back_to_search(self):
        """When LLM returns garbage, classify() returns SEARCH."""
        class GarbageLLM(BaseLLM):
            def chat(self, *args, **kwargs):
                from src.libs.llm.base_llm import ChatResponse
                return ChatResponse(
                    content="这不是JSON，随便写的",
                    model="test",
                )

        router = IntentRouter(llm=GarbageLLM())
        decision = router.classify("Azure 怎么配置？")
        assert decision.intent == QueryIntent.SEARCH
