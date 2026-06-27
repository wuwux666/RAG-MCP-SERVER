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
