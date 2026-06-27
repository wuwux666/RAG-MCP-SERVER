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
