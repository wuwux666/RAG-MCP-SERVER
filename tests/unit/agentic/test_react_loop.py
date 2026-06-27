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
