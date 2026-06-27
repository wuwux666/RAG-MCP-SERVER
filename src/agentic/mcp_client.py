"""MCP stdio subprocess client.

Spawns the existing MCP server as a subprocess and communicates
via JSON-RPC over stdio. The agentic layer uses this as its only
channel to the RAG backend — same protocol as any external MCP client.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

TOOL_CALL_TIMEOUT = 30.0
CONNECT_TIMEOUT = 15.0


@dataclass
class ToolResult:
    """Result from a single MCP tool call."""

    name: str
    content: str
    is_error: bool = False
    raw: Optional[types.CallToolResult] = None


class MCPClient:
    """MCP stdio client that talks to the RAG server subprocess.

    Usage:
        client = MCPClient()
        await client.connect()
        result = await client.call_tool("query_knowledge_hub", {...})
        await client.close()
    """

    def __init__(self) -> None:
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._connected = False
        self._stdio_ctx = None
        self._session_ctx = None

    async def connect(self) -> None:
        """Start the MCP server subprocess and initialize the session.

        Retries once on failure.

        Raises:
            RuntimeError: If server subprocess fails to start after retry.
        """
        if self._connected and self._session is not None:
            logger.debug("MCPClient already connected, skipping")
            return

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server.server"],
            env=None,
        )

        for attempt in range(2):
            try:
                async with asyncio.timeout(CONNECT_TIMEOUT):
                    self._stdio_ctx = stdio_client(server_params)
                    self._read, self._write = await self._stdio_ctx.__aenter__()
                    self._session_ctx = ClientSession(self._read, self._write)
                    self._session = await self._session_ctx.__aenter__()
                    await self._session.initialize()
                    self._connected = True
                    logger.info("MCPClient connected to server subprocess")
                    return
            except Exception as e:
                logger.warning(
                    f"MCP connection attempt {attempt + 1}/2 failed: {e}"
                )
                if attempt == 1:
                    raise RuntimeError(
                        "无法启动 MCP Server 子进程。\n"
                        f"错误: {e}\n"
                        "请手动启动: python -m src.mcp_server.server"
                    ) from e
                await asyncio.sleep(1.0)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool on the MCP server.

        Args:
            name: Tool name (e.g. 'query_knowledge_hub').
            arguments: Tool parameters as a dict.

        Returns:
            ToolResult with flattened text content and error status.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._connected or self._session is None:
            raise RuntimeError("MCPClient not connected. Call connect() first.")

        try:
            async with asyncio.timeout(TOOL_CALL_TIMEOUT):
                result: types.CallToolResult = await self._session.call_tool(
                    name, arguments
                )
        except asyncio.TimeoutError:
            return ToolResult(
                name=name,
                content="(工具调用超时)",
                is_error=True,
            )

        text_parts: List[str] = []
        is_error = getattr(result, "isError", False)

        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "data"):
                text_parts.append(f"[image data: {len(block.data)} bytes]")

        return ToolResult(
            name=name,
            content="\n".join(text_parts) if text_parts else "(empty response)",
            is_error=is_error,
            raw=result,
        )

    async def close(self) -> None:
        """Close the MCP session and terminate the subprocess."""
        if not self._connected:
            return

        try:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(None, None, None)
                self._session_ctx = None
        except Exception as e:
            logger.debug(f"Error during session cleanup: {e}")
        finally:
            self._connected = False
            self._session = None

        try:
            if self._stdio_ctx is not None:
                await self._stdio_ctx.__aexit__(None, None, None)
                self._stdio_ctx = None
        except Exception as e:
            logger.debug(f"Error during stdio cleanup: {e}")
        finally:
            self._read = None
            self._write = None
            logger.info("MCPClient disconnected")
