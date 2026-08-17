# `fluid/generation/openai.py`

Thread-safe OpenAI-compatible streaming client for chat completions and embeddings.

## Class: `OpenAICom`

### Purpose

Wraps the `openai` Python client to provide streaming chat completions and embedding requests. Manages per-stream state (buffers, stop events) so multiple streams can run concurrently without interference.

### Concurrency Model

- A threading `Lock` protects shared dictionaries (`_buffers_by_id`, `_stop_events_by_id`).
- Each stream gets its own `buffer_id` (randomly generated) and a dedicated `Event` for cancellation.
- A global `stop_event` exists for stopping all active streams at once.

### Methods

#### `__init__()`

Initializes:
- `_lock` — threading lock for concurrent access.
- `_buffers_by_id` — `dict[str, list]`, stores streamed chunks per buffer ID.
- `_stop_events_by_id` — `dict[str, Event]`, per-stream stop signals.
- `stop_event` — global stop signal (stops every stream).

#### `openai(messages, base_url, api_key, model, ...)`

Streams a chat completion. Returns a generator.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list` | required | Chat messages in OpenAI format |
| `base_url` | `str` | required | API endpoint URL |
| `api_key` | `str` | required | API key |
| `model` | `str` | required | Model identifier |
| `tools` | `list \| None` | `None` | Tool definitions |
| `tool_choice` | `str` | `"auto"` | Tool selection strategy |
| `parallel_tool_calls` | `bool` | `False` | Allow parallel tool calls |
| `timeout` | `int \| None` | `None` | Request timeout |
| `max_retries` | `int` | `0` | Retry count on failure |
| `http_client` | `object \| None` | `None` | Custom HTTP client |
| `default_headers` | `dict \| None` | `None` | Extra HTTP headers |
| `response_format` | `dict \| None` | `None` | Response format (defaults to `{"type": "text"}`) |
| `max_completion_tokens` | `int \| None` | `None` | Max tokens to generate |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `top_p` | `float \| None` | `None` | Nucleus sampling |
| `presence_penalty` | `float \| None` | `None` | Presence penalty |
| `frequency_penalty` | `float \| None` | `None` | Frequency penalty |

**Yields:** `dict` with keys:
- `buffer_id` — stream identifier
- `chunk` — the raw chunk object from the API
- `error` — error string (only on failure, `chunk` will be `None`)

**Validation:** Required params are validated _before_ the generator starts, so errors surface immediately rather than mid-iteration.

**Error handling:** Operational failures (bad API key, network drop) yield an error chunk instead of raising.

#### `openai_emb(model, input, base_url, api_key, ...)`

Requests embeddings. Returns the response object on success, `None` on failure.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Embedding model name |
| `input` | `str \| list` | required | Text(s) to embed |
| `base_url` | `str` | required | API endpoint URL |
| `api_key` | `str` | required | API key |
| `timeout` | `int \| None` | `None` | Request timeout |
| `max_retries` | `int` | `0` | Retry count |
| `http_client` | `object \| None` | `None` | Custom HTTP client |
| `default_headers` | `dict \| None` | `None` | Extra HTTP headers |

#### `stop(buffer_id)`

Signals a single stream to stop by setting its per-stream event.

#### `get_buffer(buffer_id)`

Returns a copy of the chunks collected so far for a stream.

### Logging

All operations log to `logs/openai_compatible.log` via a rotating file handler.
