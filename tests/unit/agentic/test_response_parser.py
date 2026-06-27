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
