# `fluid/regular_interface/interface.py`

The core agent loop. `RegularInterface` is the central class that ties together generation, tool execution, skill management, context retrieval, and conversation flow.

## Class: `RegularInterface`

### Constructor

```python
RegularInterface(
    messages,          # list — conversation messages in OpenAI format
    skills=[],         # list — skill configuration dicts
    max_new_skill=2,   # int  — max non-default skills the agent can activate
    sys_messages=[],   # list — custom system prompt strings
    srcf=None,         # SRCF instance or None
    all_tools=[],      # list — tool metadata dicts (from discover_tools)
    max_turns=20,      # int  — max LLM call cycles
)
```

### What Happens at Init

1. **Skill partitioning** — `skills` are split into `default_skills` and `non_default_skills`.
2. **Tool filtering** — `all_tools` are split into `on_tools` (belonging to default skills) and `off_tools` (belonging to non-default skills or no skill).
3. **Context retrieval** — If default skills have vector collections, they are queried with the last user message via `retriving_pool`.
4. **System message construction** — `BuildSystemMessages` formats skill info and retrieved context into system prompts.
5. **SRCF** — If provided, conversation history is filtered for relevance.
6. **Tool conversion** — Active tools are converted to OpenAI tool format. Skill manipulation tools are added if skills are configured.

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `agent_messages` | `list` | The full message list sent to the LLM |
| `agent_tools` | `list` | Currently active tool metadata |
| `agent_skills` | `list` | Currently active skill configs |
| `openai_agent_tools` | `list` | Tools in OpenAI API format |
| `default_skills` | `list` | Skills marked as default |
| `non_default_skills` | `list` | Skills available but not active |
| `on_tools` | `list` | Tools from active skills |
| `off_tools` | `list` | Tools from inactive skills |
| `content_buffer` | `str` | Accumulated text from current LLM turn |
| `reasoning_buffer` | `str` | Accumulated reasoning tokens |
| `tool_calls_buffer` | `list` | Accumulated tool call deltas |

### Method: `run_agent(...)`

The main agent loop. Returns a generator that yields structured events.

```python
for event in ri.run_agent(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:14b",
):
    ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | required | LLM API endpoint |
| `api_key` | `str` | required | API key |
| `model` | `str` | required | Model name |
| `tool_choice` | `str` | `"auto"` | Tool selection strategy |
| `timeout` | `int \| None` | `None` | HTTP timeout |
| `max_retires` | `int` | `5` | Max retries for LLM calls |
| `http_client` | `object \| None` | `None` | Custom HTTP client |
| `default_headers` | `dict \| None` | `None` | Extra headers |
| `max_complition_tokens` | `int` | `3000` | Max tokens per turn |
| `temperature` | `float \| None` | `None` | Sampling temperature |

### Event Types

Every yielded event is a dict:

```python
{
    "type":      str,       # event type
    "content":   str|None,  # text token
    "reasoning": str|None,  # reasoning token
    "tool_call": { ... },   # tool execution details
    "turn":      int,       # current turn (1-indexed)
}
```

| Type | When |
|------|------|
| `"content"` | A text token arrives from the model |
| `"reasoning"` | A reasoning/thinking token arrives |
| `"tool_start"` | A tool execution begins |
| `"tool_update"` | Live stdout/stderr from a running tool |
| `"tool_timeout"` | A tool's timeout expired, awaiting decision |
| `"tool_done"` | A tool finished successfully |
| `"tool_error"` | A tool had a JSON parse error or crash |
| `"error"` | The LLM stream itself errored |
| `"done"` | The model finished or max turns reached |

### Tool Call Fields

The `tool_call` dict inside events has:

| Field | Description |
|-------|-------------|
| `tool_call_id` | OpenAI tool call ID |
| `process_id` | PID of the child process (external tools) |
| `tool_name` | Name of the tool |
| `tool_args` | Arguments passed to the tool |
| `tool_comment` | Short comment from the model about why it's calling this |
| `timeout` | Timeout value in seconds |
| `stdout` | Captured stdout |
| `stderr` | Captured stderr |
| `result` | Final result or status message |
| `status` | `"started"`, `"running"`, `"timeout"`, `"finished"`, `"error"` |

### `finish_reason` Handling

| Reason | Behavior |
|--------|----------|
| `"stop"` / `None` | Model is done. Yields `"done"` event and returns. |
| `"length"` | Token limit hit. Appends partial content and continues to next turn. |
| `"tool_calls"` | Parses tool calls, executes them, feeds results back, continues. |

### Tool Execution Flow

1. **JSON parse error** — If tool call arguments are invalid JSON, the error is fed back to the model to retry.
2. **Inline tools** — `add_new_skill`, `remove_skill`, `switch_skill`, `tool_run_tool` run directly on the `RegularInterface` instance (no subprocess).
3. **External tools** — Spawned as child processes via `multiprocessing.Process`. The worker uses `start_process` from `tool_runner.py` and communicates through queues.

### Timeout Decision Flow

When an external tool exceeds its timeout:

1. The worker sends an `awaiting_decision` update.
2. If `next_input` is already set (from a prior `tool_run_tool` call), that value is used immediately.
3. Otherwise, `_ask_timeout_decision` makes a separate LLM call with only `tool_run_tool` available, showing the tool's current output.
4. The LLM decides to extend (positive seconds) or terminate (0).
5. If the LLM doesn't call `tool_run_tool`, retries up to `_TIMEOUT_DECISION_MAX_RETRIES` (3) times.
6. Defaults to terminate after all retries.

### Skill Manipulation Methods

#### `add_new_skill(skill_name)`

Activates a non-default skill and moves its tools from `off_tools` to `agent_tools`. Validates against duplicates, unknown names, and the `max_new_skill` limit.

#### `remove_skill(skill_name)`

Deactivates a non-default skill and moves its tools back to `off_tools`. Default skills cannot be removed.

#### `switch_skill(skill_name, to_which)`

Removes one active non-default skill and adds another in a single operation.

### Process Management

#### `stop_all()`

Stops the agent loop, signals the OpenAI stream to stop, and terminates all tracked subprocesses. Thread-safe.

#### `terminate_process(pid)`

Terminates a single subprocess by PID.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `_INLINE_TOOLS` | `{"add_new_skill", "remove_skill", "switch_skill", "tool_run_tool"}` | Tools handled in-process |
| `_DEFAULT_TIMEOUT` | `30` | Default tool timeout in seconds |
| `_MIN_TIMEOUT` | `5` | Minimum allowed timeout |
| `_MAX_TIMEOUT` | `300` | Maximum allowed timeout |
| `_TIMEOUT_DECISION_MAX_RETRIES` | `3` | Retries for timeout decisions |

### Logging

All operations log to `logs/interface.log`.
