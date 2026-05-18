# Update System Prompt Mid-Session (Duplex Mode)

## Goal

Allow changing the system instruction (system prompt) in the middle of an active duplex
session, **without** resetting the KV cache or losing conversation history. The units that
have been prefilled and generated should remain in the cache — only the system portion
at the beginning of the cache is replaced.

## Current Architecture

### Cache Layout (Context-Preserving Sliding Window)

After `prepare()` followed by N units of `prefill → generate → finalize`:

```
[prefix] [previous_content] [suffix] [unit_0] [unit_1] ... [unit_N]
  ^-fix     ^-dynamic            ^-fix    ^-- units to preserve --^
```

Tracking variables in `StreamDecoder`:

| Variable | Meaning |
|---|---|
| `_preserve_prefix_length` | Token count of prefix (system instruction + ref audio) |
| `_previous_content_length` | Token count of the summary text between prefix and suffix |
| `_suffix_token_ids` | Token IDs of the suffix (e.g. `<|im_end|>`) |
| `_system_preserve_length` | Total protected length = prefix + previous_content + suffix |
| `_unit_history` | List of `{unit_id, length, type, generated_tokens, ...}` |
| `cache` | The actual KV cache (DynamicCache or TupleCache) |

### Who owns what

- **StreamDecoder** (`src/minicpmo_demo/model/runtime/stream_decoder.py`): Owns the KV cache, unit history, system
  prompt protection boundaries, and cache manipulation primitives (`_slice_cache`,
  `_concat_caches`, `_reindex_rope_for_cache`, `_rebuild_cache_with_previous`).
- **DuplexCapability** (`src/minicpmo_demo/model/capabilities/duplex.py`): Owns the duplex
  orchestration (prepare/prefill/generate/finalize), TTS state, and audio processing.
  Delegates cache management to `StreamDecoder`.
- **MiniCPMO** (`src/minicpmo_demo/model/modeling_minicpmo_unified.py`): Top-level model. Core
  chat/streaming wrapper; operational helpers and duplex facade methods live in
  `src/minicpmo_demo/model/services/`.

### Existing building blocks in StreamDecoder

Mostly in `src/minicpmo_demo/model/runtime/stream_decoder.py`, with shared cache helpers in `src/minicpmo_demo/model/runtime/cache.py`:

| Method | Purpose |
|---|---|
| `_slice_cache(start, end)` | Extract a contiguous slice of the KV cache |
| `_concat_caches(cache1, cache2)` | Concatenate two KV caches |
| `_reindex_rope_for_cache(cache, old_start, new_start, length)` | Re-index RoPE positions when cache layout changes (needed because position-dependent RoPE must match the new position) |
| `_rebuild_cache_with_previous(new_previous_tokens, ...)` | Full rebuild: re-embed + forward pass + reindex RoPE + concat. Currently only replaces the `previous_content` region between prefix and suffix. |
| `embed_tokens(token_ids)` | Token → embedding lookup |
| `self.m(...)` (the LLM) | Forward pass; pass `inputs_embeds` + `past_key_values` to get new KV cache |
| `register_system_prompt()` | Records `_system_preserve_length = current cache length` |
| `register_system_prompt_with_context(...)` | Records `_preserve_prefix_length`, `_suffix_token_ids`, `_system_preserve_length` |

## Required Changes

### Phase 1: StreamDecoder (`src/minicpmo_demo/model/runtime/stream_decoder.py`)

#### 1.1 Add `update_system_prompt(prefix, suffix, ref_audio_embeds)` method

```python
def update_system_prompt(
    self,
    new_prefix_token_ids: List[int],
    new_suffix_token_ids: List[int],
    new_ref_audio_embeds: Optional[torch.Tensor] = None,
) -> bool:
    """Replace the system prompt portion of the KV cache mid-session.

    The old cache layout:
        [old_prefix] [old_prev_content] [old_suffix] [units...]

    Becomes:
        [new_prefix] [old_prev_content] [new_suffix] [units...]
                           ^--- same, not re-embedded

    If new_ref_audio_embeds is provided, the layout becomes:
        [new_prefix] [new_ref_audio] [old_prev_content] [new_suffix] [units...]

    Returns True on success.

    Limitations:
        - Must not be called while _pending_finalize is active
        - TTS state is NOT managed here (caller must handle)
        - If new prefix/suffix changes total system length, units cache RoPE is reindexed
    """
```

**Implementation steps:**

1. **Extract the units cache** (everything after old suffix):
   ```python
   old_system_end = (self._preserve_prefix_length
                     + self._previous_content_length
                     + len(self._suffix_token_ids))
   total = self.get_cache_length()
   units_len = total - old_system_end
   if units_len > 0:
       units_cache = self._slice_cache(old_system_end, total)
   else:
       units_cache = None
   ```

2. **Build new system tokens**:
   ```python
   new_system_tokens = list(new_prefix_token_ids)
   if new_ref_audio_embeds is not None:
       # ref_audio_embeds will be fed separately (not token-based)
       pass
   new_system_tokens.extend(self._previous_token_ids)  # preserve previous
   new_system_tokens.extend(new_suffix_token_ids)
   ```

3. **Forward pass to generate new system KV cache**:
   ```python
   new_system_embeds = self.embed_tokens(new_system_tokens)
   if new_ref_audio_embeds is not None:
       new_system_embeds = torch.cat([new_system_embeds, new_ref_audio_embeds], dim=0)
   new_system_len = new_system_embeds.shape[0]
   position_ids = torch.arange(0, new_system_len, device=device).unsqueeze(0)
   outputs = self.m(
       inputs_embeds=new_system_embeds.unsqueeze(0),
       position_ids=position_ids,
       past_key_values=None,
       use_cache=True,
       return_dict=True,
   )
   new_system_cache = outputs.past_key_values
   ```

4. **Reindex RoPE for units cache**:
   ```python
   if units_cache is not None and units_len > 0:
       units_cache = self._reindex_rope_for_cache(
           units_cache, old_system_end, new_system_len, units_len
       )
   ```

5. **Concatenate**:
   ```python
   if units_cache is not None:
       self.cache = self._concat_caches(new_system_cache, units_cache)
   else:
       self.cache = new_system_cache
   ```

6. **Update tracking variables**:
   ```python
   # ref_audio_embeds count = new_system_len - len(new_system_tokens)
   ref_audio_len = (new_system_len - len(new_system_tokens))
   self._preserve_prefix_length = len(new_prefix_token_ids) + ref_audio_len
   self._suffix_token_ids = list(new_suffix_token_ids)
   self._system_preserve_length = (
       self._preserve_prefix_length
       + self._previous_content_length
       + len(self._suffix_token_ids)
   )
   ```

#### 1.2 Edge cases to handle

- **Empty cache** (before any prefill): Just reset and re-prepare, no need for this method.
- **No units yet** (after prepare but before first prefill): The cache only contains the
  system prompt. This method works the same — just no units_cache to reindex.
- **Pending finalize**: Must call `finalize_unit()` first. The cache has incomplete
  state during pending finalize.
- **TTS in progress**: The TTS subsystem has its own KV cache and state. Updating the
  system prompt does NOT affect TTS — but the caller should decide whether to reset TTS.
- **RoPE position offset**: `_reindex_rope_for_cache` already handles position offsets.
  When cache layout changes but cached positions don't match actual positions, keys need
  rotary interpolation. The `_rope_inv_freq_cache` dict is used to share precomputed
  inv_freq between calls.
- **Very long sessions**: If the unit cache is large, `_slice_cache` clones all layer
  tensors, which temporarily doubles memory. Consider using non-cloning slice for
  performance-sensitive paths.

### Phase 2: DuplexCapability (`src/minicpmo_demo/model/capabilities/duplex.py`)

#### 2.1 Add `update_system_prompt()` method

```python
def update_system_prompt(
    self,
    prefix_system_prompt: Optional[str] = None,
    suffix_system_prompt: Optional[str] = None,
    ref_audio: Optional[np.ndarray] = None,
) -> bool:
    """Replace the system prompt mid-session, preserving conversation history.

    This is the duplex-level wrapper around StreamDecoder.update_system_prompt().
    It handles:
    1. Tokenizing the new prefix and suffix text
    2. Re-embedding reference audio (if any)
    3. Calling decoder.update_system_prompt()
    4. Resetting TTS state if needed

    Args:
        prefix_system_prompt: New prefix text (e.g. "<|im_start|>system\n...")
        suffix_system_prompt: New suffix text (e.g. "<|audio_end|><|im_end|>")
        ref_audio: New reference audio embedding (16kHz mono ndarray)

    Returns:
        True if the update was applied successfully.
    """
```

**Implementation steps:**

1. **Flush pending finalize** if any:
   ```python
   if self.needs_finalize:
       logger.warning("update_system_prompt: flushing pending finalize")
       self.finalize_unit()
   ```

2. **Tokenize the new prompts**:
   ```python
   new_prefix_ids = (self.tokenizer.encode(prefix_system_prompt, add_special_tokens=False)
                     if prefix_system_prompt else [])
   new_suffix_ids = (self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)
                     if suffix_system_prompt else [])
   ```

3. **Re-embed reference audio** if provided:
   ```python
   ref_embed = None
   if ref_audio is not None:
       data = self.processor.process_audio([ref_audio])
       embeds_nested = self.model.get_audio_embedding(
           data, chunk_length=self.model.config.audio_chunk_length
       )
       ref_embed = (torch.cat([t for g in embeds_nested for t in g], dim=0)
                    if embeds_nested else None)
   ```

4. **Call decoder**:
   ```python
   return self.decoder.update_system_prompt(
       new_prefix_token_ids=new_prefix_ids,
       new_suffix_token_ids=new_suffix_ids,
       new_ref_audio_embeds=ref_embed,
   )
   ```

5. **Handle TTS**:
   After the system prompt changes, any previously generated TTS text is stale.
   Consider resetting TTS state:
   ```python
   self._reset_token2wav_for_new_turn()
   self.tts_past_key_values = None
   self.tts_text_start_pos = 0
   ```

#### 2.2 Expose on DuplexCapability

```python
# In __init__ or as a property
@property
def supports_system_prompt_update(self) -> bool:
    """Whether the current session supports mid-session system prompt update."""
    return self.decoder.get_cache_length() > 0
```

### Phase 3: MiniCPMO (convenience method)

```python
def duplex_update_system_prompt(
    self,
    prefix_system_prompt: Optional[str] = None,
    suffix_system_prompt: Optional[str] = None,
    ref_audio: Optional[np.ndarray] = None,
) -> bool:
    """Update system prompt mid-duplex-session (convenience method).

    Requires init_unified() to have been called and duplex mode to be active.

    Example:
        model.duplex_update_system_prompt(
            prefix_system_prompt="<|im_start|>system\nNow you are a poet.\n<|audio_start|>",
        )
    """
    if self.duplex is None:
        raise RuntimeError("duplex not initialized. Call init_unified() first.")
    if self._current_mode != ProcessorMode.DUPLEX:
        logger.warning("Not in duplex mode, but proceeding anyway")
    return self.duplex.update_system_prompt(
        prefix_system_prompt=prefix_system_prompt,
        suffix_system_prompt=suffix_system_prompt,
        ref_audio=ref_audio,
    )
```

### Phase 4: Gateways / API (optional)

If the system prompt update should be triggerable from an external API (e.g., as a
WebSocket message in the duplex session), add a message type:

```json
{
    "type": "update_system_prompt",
    "prefix": "<|im_start|>system\nNew persona.\n<|audio_start|>",
    "suffix": "<|audio_end|><|im_end|>"
}
```

## Testing Plan

### Unit tests (StreamDecoder)

1. **Basic swap**: Prepare session, prefill 3 units, call `update_system_prompt` with
   new prefix. Verify cache length changes correctly and units are preserved.

2. **RoPE correctness**: After update, generate a new token and verify the logits are
   consistent (no sudden distribution changes from incorrect position encoding).

3. **Empty units**: Call `update_system_prompt` before any unit prefill. Should work and
   just replace the system portion.

4. **Ref audio**: Verify that including ref_audio_embeds inserts them at the right
   position and the length tracking is correct.

5. **No-op**: Calling with the exact same prefix/suffix should be a no-op (detect by
   comparing token IDs; early return to avoid forward pass).

6. **Large delta**: If new system is much longer/shorter than old, verify the units
   cache reindex handles the shift correctly.

### Integration tests (DuplexCapability)

1. **End-to-end**: Full duplex session → update system prompt → continue session.
   Verify that the model uses the new instruction.

2. **TTS continuity**: Update system prompt while TTS is generating. Verify graceful
   handling (either TTS continues or is cleanly reset).

3. **Edge: pending finalize**: Call update while finalize is pending. Should flush
   and proceed.

### Validation tests

1. **Generate with old system prompt** → record output
2. **Update system prompt** to something contradictory (e.g. "always answer in French")
3. **Generate with new system prompt** → verify output follows the new instruction
4. **The model should NOT "remember" the old system prompt in its generated text**

## Alternatives Considered

### A. Full session reset

```python
model.duplex._reset_streaming_state()
model.duplex.decoder.reset()
model.duplex.prepare(new_system_prompt, ...)
```

**Pros**: Simple, no new code needed.
**Cons**: Loses ALL conversation history. Would need to re-feed every old unit.

### B. Rebuild entire cache from scratch

Serialize the entire session to messages, inject new system prompt, re-prefill everything.

**Pros**: Clean, no cache manipulation.
**Cons**: Extremely expensive (re-encode all audio/video), slow.

### C. Inject system prompt as a new "unit" (not replacing old)

Insert a fake unit at the beginning position with the new instruction.

**Pros**: Simple insertion logic.
**Cons**: Cache grows unboundedly. Old system prompt still consumed by model, which may
create conflicts. The model sees TWO system prompts.

### D. The chosen approach (replace in cache)

**Pros**:
- Conversation history preserved at KV cache level (no re-encoding needed)
- Only the system portion is re-embedded (few tokens, fast)
- Leverages existing `_rebuild_cache_with_previous` infrastructure
- O(prefix_length) forward pass only (typically < 100 tokens, ~a few ms)

**Cons**:
- Requires careful RoPE reindexing
- Cache slicing doubles memory temporarily
- Must ensure units_cache length calculation is correct


---

# Tools / Function Calling: Current State & Required Changes

## Background: Tools are system-prompt-dependent

In the Qwen3 chat template (inherited by MiniCPM-o), tool definitions are **not** a
separate mechanism. They are injected **into the system prompt text** by the
`apply_chat_template()` Jinja template when `tools=` is passed:

```
<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {"name": "get_weather", "description": "...", "parameters": {...}}}
</tools>

For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call><|im_end|>
```

This means **tools ARE part of the system prompt**. Updating tools mid-session is the
same problem as updating the system prompt — they are both embedded as KV cache tokens
at the beginning of the session.

## Current state: what works vs what's missing

| Layer | Tools support | What's missing |
|---|---|---|
| **Tokenizer** (`src/minicpmo_demo/model/hf_assets/tokenizer_config.json`) | ✅ Full support in chat_template | Nothing — Jinja template handles `tools`, `tool_calls`, `tool` role |
| **`model.chat()`** (`modeling_minicpmo_unified.py`) | ❌ `apply_chat_template()` called without `tools=` | Need to pass `tools` kwarg through |
| **`model.non_streaming_prefill()`** (`modeling_minicpmo_unified.py`) | ❌ Same issue | Same fix |
| **`model.streaming_prefill()`** (`modeling_minicpmo_unified.py`) | ❌ Same issue | Same fix |
| **`ChatRequest` schema** (`src/minicpmo_demo/core/schemas/chat.py`) | ❌ No `tools` field | Add `tools: Optional[List[dict]]` |
| **`Message` schema** (`src/minicpmo_demo/core/schemas/common.py`) | ❌ No `tool_calls` field, no `TOOL` role | Add `tool_calls` + `TOOL` role |
| **`ChatView.chat()`** (`src/minicpmo_demo/core/processors/unified.py`) | ❌ Doesn't pass tools | Forward from request → model |
| **Duplex mode** (duplex pipeline) | ❌ Tools frozen in KV cache | Same `update_system_prompt()` mechanism |

## Required Changes: Chat Mode

Chat mode is simpler because each `chat()` call is stateless — everything is processed
from scratch. Tools can be changed freely between calls once the plumbing exists.

### Phase A1: Add `tools` to `ChatRequest` schema

**File**: `src/minicpmo_demo/core/schemas/chat.py`

```python
class ChatRequest(BaseModel):
    # ... existing fields ...
    
    # ── NEW ──
    tools: Optional[List[Dict]] = Field(
        None,
        description=(
            "Tool/function definitions the model may call. "
            "Each entry follows the OpenAI function calling schema:\n"
            '{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}\n'
            "When provided, they are injected into the system prompt via the chat template."
        ),
    )
```

### Phase A2: Add `TOOL` role and `tool_calls` to `Message` schema

**File**: `src/minicpmo_demo/core/schemas/common.py`

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # ← NEW: role for tool response messages


class Message(BaseModel):
    role: Role
    content: Union[str, List["ContentItem"], None] = None
    
    # ── NEW ──
    tool_calls: Optional[List[Dict]] = Field(
        None,
        description=(
            "Tool calls made by the assistant. "
            "Each entry: {\"id\": ..., \"type\": \"function\", "
            "\"function\": {\"name\": ..., \"arguments\": ...}}\n"
            "Only used when role=ASSISTANT."
        ),
    )
    
    # ── NEW ──
    tool_call_id: Optional[str] = Field(
        None,
        description="Tool call ID this message is responding to (only when role=TOOL)",
    )
    
    # ── NEW ──
    name: Optional[str] = Field(
        None,
        description="Name of the tool that produced this result (only when role=TOOL)",
    )
```

### Phase A3: Wire `tools` through `model.chat()`

**File**: `src/minicpmo_demo/model/modeling_minicpmo_unified.py`

In the `chat()` method, add `tools` parameter and pass it to `apply_chat_template`:

```python
def chat(
    self,
    image=None,
    msgs=None,
    # ... existing params ...
    tools: Optional[List[Dict]] = None,  # ← NEW
    **kwargs,
):
    # ... existing code ...
    
    prompts_lists.append(
        self.processor.tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=False if teacher_forcing else True,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            tools=tools,  # ← NEW: pass tool definitions
        )
    )
```

**Important**: The same change must be made in every `apply_chat_template()` call site:

1. `chat()` method (line ~2137) — the main chat path
2. `non_streaming_prefill()` method (line ~3129) — the non-streaming prefill path
3. `streaming_prefill()` method (line ~3472) — the old streaming prefill path

### Phase A4: Wire `tools` through `ChatView.chat()`

**File**: `src/minicpmo_demo/core/processors/unified.py`

In `_chat_impl()`, pass `tools` from the request to the model:

```python
def _chat_impl(self, request, ...):
    # ... existing code ...
    
    # ── NEW: extract tools from request ──
    tools = getattr(request, 'tools', None)
    
    result = self._model.chat(
        msgs=msgs,
        tools=tools,  # ← NEW
        # ... existing params ...
    )
```

### Phase A5: Parse `tool_calls` from model output

**File**: `src/minicpmo_demo/model/modeling_minicpmo_unified.py` (in `_decode_text()` or the caller)

The model outputs tool calls as `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`
in the generated text. The caller must parse these and convert them to structured data.

Add a helper:

```python
import re
import json

_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{.*?\})\s*\}\s*</tool_call>',
    re.DOTALL,
)

@staticmethod
def parse_tool_calls(text: str) -> Tuple[str, List[Dict]]:
    """Parse <tool_call> tags from generated text.

    Returns:
        (cleaned_text, list_of_tool_calls)
        where cleaned_text has the tool_call tags removed.
    """
    tool_calls = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            arguments = match.group(2)  # keep as raw string
        tool_calls.append({
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        })
    cleaned = _TOOL_CALL_PATTERN.sub("", text).strip()
    return cleaned, tool_calls
```

## Required Changes: Streaming / Duplex Mode

### Tools in Old Streaming Mode

In `streaming_prefill()`, the first prefill uses `apply_chat_template()` with all
messages. Adding `tools=tools` here would embed tool definitions at the start.

However, since the code manually constructs prompts for subsequent prefills (without
`apply_chat_template`), tools are **not** re-evaluated mid-stream. This is the same
limitation as the system prompt.

**The reset-strategy workaround**:

```python
# 1. Snapshot current session
session_msgs = model._omni_chunk_history  # partial
# 2. Reset
model.reset_session()
# 3. Re-prefill with new tools
model.streaming_prefill(
    msgs=[{"role": "system", "content": "..."}, ...],
    tools=new_tools,
)
```

### Tools in Duplex Mode

Since tools are part of the system prompt text (inside `prefix_system_prompt`), the
`update_system_prompt()` method from Phase 1-3 **automatically handles tool updates**.

To change tools mid-duplex-session, include them in the new prefix:

```python
# Build a new system prompt with different tools
new_prefix = (
    "<|im_start|>system\n"
    "You are now a poet. Write beautiful poetry.\n\n"
    "# Tools\n"
    "<tools>\n"
    + json.dumps(new_tool_definitions) + "\n"
    "</tools>\n"
    "<|audio_start|>"
)

model.duplex_update_system_prompt(
    prefix_system_prompt=new_prefix,
    suffix_system_prompt="<|audio_end|><|im_end|>",
)
```

## The "Agent Switch" pattern (Tool-Triggered Persona Change)

This is the scenario you described: one tool call triggers a complete agent
reconfiguration mid-conversation.

### Flow

```
┌─────────────────────────────────────────────────────┐
│ 1. Session starts with Agent A                      │
│    System: "You are a helpful math tutor."          │
│    Tools:  [calculate, draw_graph, switch_agent]    │
├─────────────────────────────────────────────────────┤
│ 2. User: "Teach me calculus, then switch to poet."  │
├─────────────────────────────────────────────────────┤
│ 3. Model teaches calculus (uses calculate tool)     │
├─────────────────────────────────────────────────────┤
│ 4. Model calls: switch_agent(persona="poet", ...)   │
├─────────────────────────────────────────────────────┤
│ 5. [Application handles the switch_agent call]      │
│    a. Parse arguments                               │
│    b. Build new system prompt + new tools           │
│    c. Call update_system_prompt()                   │
├─────────────────────────────────────────────────────┤
│ 6. Model now behaves as Agent B (poet)              │
│    - New persona, new tools                         │
│    - Still remembers earlier conversation           │
│      (units before the switch remain in cache)      │
│    - But follows the new system instruction         │
└─────────────────────────────────────────────────────┘
```

### Implementation sketch

```python
# 1. Define the switch_agent function (server-side)
SWITCH_AGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "switch_agent",
        "description": "Switch the AI agent to a different persona with different capabilities.",
        "parameters": {
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": "The new persona description",
                    "enum": ["poet", "scientist", "teacher", "coder"]
                },
                "reason": {
                    "type": "string",
                    "description": "Why the switch is needed"
                }
            },
            "required": ["persona"]
        }
    }
}

# 2. When the model calls it, handle the switch
AGENT_CONFIGS = {
    "poet": {
        "system_prompt": "You are a romantic poet. Speak in verse.",
        "tools": [{"type": "function", ...}],
    },
    "scientist": {
        "system_prompt": "You are a严谨 scientist. Be precise.",
        "tools": [calculate_tool, search_tool],
    },
}

def handle_switch_agent(model, tool_call_args):
    persona = tool_call_args["persona"]
    config = AGENT_CONFIGS[persona]
    
    new_prefix = (
        "<|im_start|>system\n"
        f"{config['system_prompt']}\n\n"
        "# Tools\n<tools>\n"
        + json.dumps(config["tools"]) + "\n"
        "</tools>\n<|audio_start|>"
    )
    
    model.duplex_update_system_prompt(
        prefix_system_prompt=new_prefix,
        suffix_system_prompt="<|audio_end|><|im_end|>",
    )
```

### Important considerations

1. **Tool definitions must be comprehensive at the start**: The model can only call
   tools it knows about. `switch_agent` must be available in the **initial** tool set.

2. **After the switch, the old tools are gone**: The new system prompt replaces the old
   one entirely, including tool definitions. The model can no longer call the old tools.

3. **Conversation memory preserved**: Units before the switch remain in the KV cache.
   The model can refer to earlier conversation. But `switch_agent` is no longer
   available after the switch (unless included in the new tool set).

4. **Security boundary**: The application (not the model) executes the switch. The
   model only requests it via a tool call. The application validates and controls
   which persona transitions are allowed.

5. **TTS continuity**: If the model was speaking when the switch occurs, the TTS
   subsystem should be reset (current speech output stops, new speech uses new persona).

## Summary: Changes needed for tool support

| Change | File(s) | Effort |
|---|---|---|
| Add `tools` field to `ChatRequest` | `src/minicpmo_demo/core/schemas/chat.py` | ~10 lines |
| Add `TOOL` role, `tool_calls` to `Message` | `src/minicpmo_demo/core/schemas/common.py` | ~20 lines |
| Pass `tools` to `apply_chat_template()` | `src/minicpmo_demo/model/modeling_minicpmo_unified.py` (3 call sites) | ~5 lines each |
| Wire through `ChatView.chat()` | `src/minicpmo_demo/core/processors/unified.py` | ~5 lines |
| Parse `<tool_call>` from output | `src/minicpmo_demo/model/modeling_minicpmo_unified.py` | ~30 lines helper |
| `update_system_prompt()` for duplex tools | `src/minicpmo_demo/model/runtime/stream_decoder.py` + `src/minicpmo_demo/model/capabilities/duplex.py` | Already covered in Phase 1-3 |

**Total for basic chat-mode tools**: ~1-2 hours of focused work.
**Total for full duplex tool switching**: Already covered by the `update_system_prompt()` implementation (~1 day).
