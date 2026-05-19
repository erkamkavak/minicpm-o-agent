"""Helpers for MiniCPM/Qwen-style tool call markup."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict]]:
    """Extract ``<tool_call>`` JSON blocks from generated text."""
    tool_calls: List[Dict] = []

    def _replace(match: re.Match) -> str:
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return match.group(0)

        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if name:
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
        return ""

    cleaned = _TOOL_CALL_PATTERN.sub(_replace, text or "").strip()
    return cleaned, tool_calls


def build_tool_instruction_block(tools: List[Dict] | None) -> str:
    """Build the system-prompt tool block used by the tokenizer chat template."""
    if not tools:
        return ""

    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    lines.extend(json.dumps(tool, ensure_ascii=False) for tool in tools)
    lines.extend(
        [
            "</tools>",
            "",
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name": <function-name>, "arguments": <args-json-object>}',
            "</tool_call>",
        ]
    )
    return "\n".join(lines)
