"""ReAct (Reasoning + Acting) loop for agentic RAG.

Core state machine that orchestrates the Thought → Action → Observation
cycle, with safety constraints: max rounds, loop detection, and context
trimming.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.agentic.citation_tracker import CitationTracker
from src.agentic.prompt_manager import PromptManager
from src.agentic.response_parser import parse_response, ParsedResponse
from src.agentic.tool_registry import ToolRegistry
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message

if TYPE_CHECKING:
    from src.agentic.mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class RoundTrace:
    """A single round of the ReAct loop for observability."""

    round_number: int
    thought: str
    action: str
    observation: str


@dataclass
class AgenticResult:
    """Final result from the ReAct agent."""

    answer: str
    citations: List[str] = field(default_factory=list)
    trace: List[RoundTrace] = field(default_factory=list)


class ReActLoop:
    """ReAct reasoning loop for multi-hop knowledge retrieval.

    Example:
        loop = ReActLoop(mcp_client, llm, max_rounds=10)
        result = await loop.run("对比项目 A 和 B 的技术差异")
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        llm_client: BaseLLM,
        max_rounds: int = 10,
        context_window: int = 3,
    ) -> None:
        self._mcp_client = mcp_client
        self._llm = llm_client
        self.max_rounds = max_rounds
        self.context_window = context_window

        self._tool_registry = ToolRegistry(mcp_client)
        self._citation_tracker = CitationTracker()
        self._trace: List[RoundTrace] = []

    async def run(self, question: str) -> AgenticResult:
        """Run the ReAct loop for a given question."""
        system_msg = PromptManager.build_system_message()
        user_msg = PromptManager.build_user_message(question)
        messages: List[Message] = [system_msg, user_msg]

        last_action_key: Optional[str] = None
        repeat_count: int = 0
        final_answer: Optional[str] = None

        for round_num in range(1, self.max_rounds + 1):
            # 1. Call LLM
            response: ChatResponse = await asyncio.to_thread(
                self._llm.chat, messages
            )

            # 2. Parse response
            parsed: ParsedResponse = parse_response(response.content)

            # 3. Check for final answer
            if parsed.is_final:
                final_answer = parsed.answer or response.content
                messages.append(Message(role="assistant", content=response.content))
                logger.info(f"ReAct loop finished at round {round_num}")
                break

            # 4. Execute action
            action = parsed.action_name
            params = parsed.action_params
            action_text = (
                f"{action}({_format_params(params)})" if action else "(no action)"
            )

            if action is None:
                observation_text = (
                    "无法解析你的输出。请使用以下格式：\n"
                    "Thought: <你的推理>\n"
                    'Action: tool_name(key="value", ...)\n\n'
                    "或者：\n"
                    "Final Answer: <你的答案>"
                )
                messages.append(Message(role="user", content=_format_obs(observation_text)))
                continue

            parsed_action = self._tool_registry.parse_action(action_text)
            observation = await self._tool_registry.execute(parsed_action)

            # 5. Track citations
            if observation.results:
                self._citation_tracker.add(observation.results)

            # 6. Loop detection
            current_key = f"{parsed_action.name}:{_params_key(parsed_action.params)}"
            if current_key == last_action_key:
                repeat_count += 1
            else:
                repeat_count = 0
            last_action_key = current_key

            obs_content = observation.text
            if repeat_count >= 2:
                obs_content = (
                    f"{obs_content}\n\n"
                    "[警告] 连续 2 次相同调用，请换策略或输出 Final Answer。"
                )

            # 7. Append to messages
            messages.append(Message(role="assistant", content=response.content))
            messages.append(Message(role="user", content=_format_obs(obs_content)))

            # 8. Record trace
            self._trace.append(RoundTrace(
                round_number=round_num,
                thought=parsed.thought or "",
                action=action_text,
                observation=obs_content,
            ))

            # 9. Trim context
            messages = self._trim_context(messages)

        # Force answer if loop exhausted
        if final_answer is None:
            logger.warning(f"ReAct loop hit max_rounds ({self.max_rounds})")
            force_prompt = "已达到最大轮次。请基于以上检索结果给出最佳答案。"
            messages.append(Message(role="user", content=force_prompt))
            response = await asyncio.to_thread(self._llm.chat, messages)
            final_answer = response.content

        citation_text = self._citation_tracker.format()
        citations = citation_text.split("\n") if citation_text else []

        return AgenticResult(
            answer=final_answer,
            citations=citations,
            trace=list(self._trace),
        )

    def _trim_context(self, messages: List[Message]) -> List[Message]:
        """Trim messages to keep system prompt + recent context."""
        system = messages[0] if messages else None
        non_system = messages[1:]
        keep_count = self.context_window * 2
        if len(non_system) > keep_count:
            non_system = non_system[-keep_count:]
        result = [system] if system else []
        result.extend(non_system)
        return result


def _format_obs(text: str) -> str:
    return f"--- 工具返回结果 ---\n{text}\n--- 结束 ---"


def _format_params(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _params_key(params: Dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))
