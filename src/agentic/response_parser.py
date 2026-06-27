"""Parse LLM ReAct output into structured ParsedResponse.

Supports two formats:
- Thought: ... \\n Action: tool_name(key=value, ...)
- Final Answer: ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ParsedResponse:
    """Parsed ReAct response from LLM.

    Attributes:
        is_final: True if this is a Final Answer (loop should stop).
        thought: Reasoning text from Thought: section.
        action_name: Tool name from Action: section (None if final).
        action_params: Tool parameters parsed from Action: section.
        answer: Answer text from Final Answer: section (None if action).
    """

    is_final: bool = False
    thought: Optional[str] = None
    action_name: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None


def parse_response(raw: str) -> ParsedResponse:
    """Parse LLM raw output into a ParsedResponse.

    Args:
        raw: Raw text output from the LLM.

    Returns:
        ParsedResponse with structured fields.
    """
    if not raw or not raw.strip():
        return ParsedResponse(thought="")

    text = raw.strip()

    # Check for Final Answer first
    final_match = re.match(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return ParsedResponse(
            is_final=True,
            answer=final_match.group(1).strip(),
        )

    # Parse Thought: section (everything before Action:)
    thought: Optional[str] = None
    thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Parse Action: section
    action_match = re.search(
        r"Action:\s*(\w+)\((.*?)\)",
        text,
        re.DOTALL,
    )
    if action_match:
        action_name = action_match.group(1).strip()
        params_str = action_match.group(2).strip()
        action_params = _parse_params(params_str)
        return ParsedResponse(
            is_final=False,
            thought=thought,
            action_name=action_name,
            action_params=action_params,
        )

    # Neither final answer nor action found — return raw as thought
    return ParsedResponse(
        is_final=False,
        thought=text,
        action_name=None,
    )


def _parse_params(params_str: str) -> Dict[str, Any]:
    """Parse action parameters from LLM-generated string.

    Handles: key="value", key=123, key="escaped \\" quote"
    Returns a dict of parsed parameters.
    """
    if not params_str.strip():
        return {}

    params: Dict[str, Any] = {}

    # Match key=value pairs where value is quoted string or unquoted number
    pattern = r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|(\d+))'
    for match in re.finditer(pattern, params_str):
        key = match.group(1)
        str_val = match.group(2)
        int_val = match.group(3)

        if str_val is not None:
            params[key] = str_val.replace('\\"', '"').replace("\\\\", "\\")
        elif int_val is not None:
            params[key] = int(int_val)

    return params
