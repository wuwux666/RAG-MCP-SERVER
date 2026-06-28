"""Tests for query endpoints using TestClient + mocked internals."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


@patch("src.mcp_server.tools.query_knowledge_hub.QueryKnowledgeHubTool")
def test_query_endpoint(mock_tool_cls):
    mock_tool = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "检索结果"
    mock_response.citations = []
    mock_response.is_empty = False
    mock_tool.execute = AsyncMock(return_value=mock_response)
    mock_tool_cls.return_value = mock_tool

    resp = client.post("/api/query", json={"query": "test", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "traditional"
    assert "answer" in data
    assert data["answer"] == "检索结果"
    assert data["elapsed_ms"] >= 0
