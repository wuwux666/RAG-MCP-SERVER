"""Integration tests for MCPClient — connects to real MCP server subprocess."""

import pytest
import pytest_asyncio
from src.agentic.mcp_client import MCPClient


@pytest_asyncio.fixture
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
