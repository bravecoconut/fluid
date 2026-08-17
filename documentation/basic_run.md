# `_basic_run/basic_run.py`

A working demo that shows how to wire up Fluid to build an autonomous OS agent.

## What It Contains

### Example Tools

Six tool functions decorated with `@tool(skill="OS")`:

| Tool | Description | `usually_takes` |
|------|-------------|-----------------|
| `download_file` | Downloads a file from a URL | 30s |
| `run_shell_command` | Runs a shell command and returns structured output | — |
| `compute_file_hash` | Computes a file's hash digest (md5, sha1, sha256, sha512) | — |
| `write_text_file` | Writes text content to a file | — |
| `zip_directory` | Compresses a directory into a .zip | 5s |
| `read_text_file` | Reads and returns file contents (with truncation) | — |

All tools are self-contained — standard library only (except `download_file` which uses `requests`).

### Agent Configuration

The `__main__` block demonstrates:

1. **System prompts** — Two prompts defining the agent's role as an OS agent.
2. **Skill configuration** — A single default skill `"OS"` with:
   - A ChromaDB persistent backend for context retrieval
   - OpenAI-compatible embedding setup (Ollama in this example)
   - Collection-level template for injecting context
3. **Messages** — A multi-step task asking the agent to download, read, summarize, and zip files.
4. **Agent instantiation** — Creates `RegularInterface` with all components.
5. **Event loop** — (Commented out) Shows how to iterate events and handle each type.

### Event Handling Pattern

```python
for event in ri.run_agent(base_url=..., api_key=..., model=...):
    etype = event["type"]

    if etype == "content":
        print(event["content"], end="", flush=True)
    elif etype == "reasoning":
        print(event["reasoning"], end="", flush=True)
    elif etype == "tool_start":
        tc = event["tool_call"]
        print(f"🔧 TOOL START: {tc['tool_name']}({tc['tool_args']})")
    elif etype == "tool_update":
        tc = event["tool_call"]
        print(f"   ↻ UPDATE: pid={tc['process_id']}")
    elif etype == "tool_timeout":
        tc = event["tool_call"]
        print(f"   ⏰ TIMEOUT: {tc['tool_name']}")
    elif etype == "tool_done":
        tc = event["tool_call"]
        print(f"   ✓ DONE: {tc['tool_name']} → {tc.get('result', '')[:200]}")
    elif etype == "tool_error":
        tc = event["tool_call"]
        print(f"   ✗ ERROR: {tc['tool_name']}")
    elif etype == "done":
        print(f"--- DONE (turn {event['turn']}) ---")
    elif etype == "error":
        print(f"!!! ERROR: {event['content']}")
```

### Key Note

The comment "running model's parallel tool calls sequentially is a feature, not a bug — just to save API requests" is a design note: tools are executed one at a time within a turn to minimize the number of API calls.
