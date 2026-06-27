#!/usr/bin/env python3
"""Agentic RAG Query CLI — ReAct agent for complex multi-hop questions.

Usage:
    python scripts/agentic_query.py --query "对比A和B的技术差异"
    python scripts/agentic_query.py --query "..." --verbose --max-rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from src.agentic.mcp_client import MCPClient
from src.agentic.react_loop import ReActLoop, AgenticResult
from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG — ReAct agent for multi-hop knowledge retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/agentic_query.py --query "对比项目A和项目B的技术选型差异"
  python scripts/agentic_query.py --query "为什么选择了这个架构？" --verbose
        """,
    )
    parser.add_argument(
        "--query", "-q", required=True, help="The complex question to answer",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full Thought/Action/Observation per round",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=10,
        help="Maximum ReAct rounds (default: 10)",
    )
    return parser.parse_args()


async def run_agent(
    question: str,
    max_rounds: int,
    verbose: bool,
) -> AgenticResult:
    """Initialize components and run the ReAct loop."""
    settings = load_settings()
    llm = LLMFactory.create(settings)

    client = MCPClient()
    await client.connect()

    try:
        loop = ReActLoop(
            mcp_client=client,
            llm_client=llm,
            max_rounds=max_rounds,
            context_window=3,
        )

        if verbose:
            print(f"问题: {question}", file=sys.stderr)
            print(f"最大轮次: {max_rounds}", file=sys.stderr)
            print("-" * 50, file=sys.stderr)

        result = await loop.run(question)

        if verbose:
            _print_verbose_trace(result)

        return result

    finally:
        await client.close()


def _print_verbose_trace(result: AgenticResult) -> None:
    """Print full ReAct trace to stderr."""
    for t in result.trace:
        print(f"\n[Round {t.round_number}]", file=sys.stderr)
        print(f"  Thought: {t.thought}", file=sys.stderr)
        print(f"  Action:  {t.action}", file=sys.stderr)
        obs = t.observation[:200] + "..." if len(t.observation) > 200 else t.observation
        print(f"  Obs:     {obs}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)


def _print_result(result: AgenticResult, verbose: bool) -> None:
    """Print final result to stdout."""
    print(result.answer)
    print()

    if result.citations:
        print("---")
        print("引用来源:")
        for citation in result.citations:
            print(f"  {citation}")

    if not verbose and result.trace:
        print(f"\n(共 {len(result.trace)} 轮检索)")


async def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result: Optional[AgenticResult] = None

    try:
        result = await run_agent(
            question=args.query,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\n用户中断。", file=sys.stderr)
        if result is not None:
            _print_result(result, args.verbose)
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        if result is not None:
            _print_result(result, args.verbose)
        sys.exit(1)

    _print_result(result, args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
