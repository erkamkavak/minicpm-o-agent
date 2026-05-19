import json
import sys
import types
from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: str):
    spec = spec_from_file_location(name, ROOT / path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("minicpmo_demo", types.ModuleType("minicpmo_demo"))
sys.modules.setdefault("minicpmo_demo.core", types.ModuleType("minicpmo_demo.core"))
sys.modules.setdefault("minicpmo_demo.core.schemas", types.ModuleType("minicpmo_demo.core.schemas"))

common = _load_module("minicpmo_demo.core.schemas.common", "src/minicpmo_demo/core/schemas/common.py")
chat = _load_module("minicpmo_demo.core.schemas.chat", "src/minicpmo_demo/core/schemas/chat.py")
tool_calls = _load_module("minicpmo_demo.core.tool_calls", "src/minicpmo_demo/core/tool_calls.py")

ChatRequest = chat.ChatRequest
Message = common.Message
Role = common.Role
build_tool_instruction_block = tool_calls.build_tool_instruction_block
parse_tool_calls = tool_calls.parse_tool_calls


def test_chat_schema_accepts_tools_and_tool_messages():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "switch_agent",
                "description": "Switch persona",
                "parameters": {"type": "object", "properties": {"persona": {"type": "string"}}},
            },
        }
    ]
    request = ChatRequest(
        messages=[
            Message(role=Role.USER, content="Switch to poet"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    {
                        "type": "function",
                        "function": {"name": "switch_agent", "arguments": {"persona": "poet"}},
                    }
                ],
            ),
            Message(role=Role.TOOL, name="switch_agent", tool_call_id="call_1", content="ok"),
        ],
        tools=tools,
    )

    dumped = request.model_dump()
    assert dumped["tools"] == tools
    assert dumped["messages"][1]["tool_calls"][0]["function"]["name"] == "switch_agent"
    assert dumped["messages"][2]["role"] == "tool"


def test_tool_prompt_block_matches_qwen_style_markers():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

    block = build_tool_instruction_block(tools)

    assert "# Tools" in block
    assert "<tools>" in block
    assert "</tools>" in block
    assert "<tool_call>" in block
    assert json.dumps(tools[0], ensure_ascii=False) in block


def test_parse_tool_calls_removes_structured_call_from_text():
    text = 'Before<tool_call>\n{"name": "get_weather", "arguments": {"city": "Istanbul"}}\n</tool_call>After'

    cleaned, tool_calls = parse_tool_calls(text)

    assert cleaned == "BeforeAfter"
    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Istanbul"},
            },
        }
    ]
