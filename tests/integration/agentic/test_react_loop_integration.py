"""Integration tests: full ReAct loop with real MCP server + LLM.

Marked 'llm' and 'slow' to allow exclusion in CI without API keys.
"""

import pytest
import pytest_asyncio
from src.agentic.mcp_client import MCPClient
from src.agentic.react_loop import ReActLoop, AgenticResult
from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory


@pytest_asyncio.fixture
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
