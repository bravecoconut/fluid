# Fluid Tutorial — Getting Started

This tutorial walks you through building an agent with Fluid, from zero to a fully operational autonomous agent with tools, skills, and context retrieval.

## Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM endpoint (Ollama, vLLM, OpenRouter, etc.)
- (Optional) A ChromaDB instance for vector retrieval

## Installation

```bash
pip install openai chromadb sentence_transformers
```

## 1. Your First Agent — No Tools, No Skills

The simplest agent is just an LLM loop. No tools, no skills, no retrieval.

```python
from fluid.regular_interface.interface import RegularInterface

messages = [
    {"role": "user", "content": "What is the capital of France?"},
]

ri = RegularInterface(
    messages=messages,
    sys_messages=["You are a helpful assistant."],
)

for event in ri.run_agent(
    base_url="http://localhost:11434/v1",  # your LLM endpoint
    api_key="ollama",
    model="qwen3:14b",
):
    if event["type"] == "content":
        print(event["content"], end="", flush=True)
    elif event["type"] == "done":
        print("\n--- Done ---")
```

This creates an agent that sends the messages to the LLM and streams the response. No tool calling, no context retrieval.

---

## 2. Adding Tools

Tools are Python functions decorated with `@tool`. Create a file with your tool functions:

```python
# my_tools.py
from fluid.toolkit.tool_registry import tool

@tool(skill="general")
def get_weather(city: str) -> str:
    '''Get the current weather for a city.'''
    # In reality, call a weather API
    return f"The weather in {city} is sunny, 25°C."

@tool(skill="general", usually_takes=5)
def search_web(query: str, max_results: int = 3) -> str:
    '''Search the web for a query and return results.'''
    return f"Search results for '{query}': [result1, result2, result3]"
```

Now discover and use them:

```python
from fluid.regular_interface.interface import RegularInterface
from fluid.toolkit.tool_registry import discover_tools

# Discover all @tool-decorated functions in the current directory
all_tools = discover_tools(root_dir=".")

messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"},
]

ri = RegularInterface(
    messages=messages,
    sys_messages=["You are a helpful assistant with access to tools."],
    all_tools=all_tools,
)

for event in ri.run_agent(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:14b",
):
    etype = event["type"]

    if etype == "content":
        print(event["content"], end="", flush=True)
    elif etype == "tool_start":
        tc = event["tool_call"]
        print(f"\n🔧 Calling {tc['tool_name']}({tc['tool_args']})")
    elif etype == "tool_done":
        tc = event["tool_call"]
        print(f"   ✓ Result: {tc['result'][:200]}")
    elif etype == "done":
        print(f"\n--- Done (turn {event['turn']}) ---")
```

### How tools work under the hood

1. `discover_tools()` scans a directory for Python files, imports them, and collects metadata from `@tool`-decorated functions.
2. Tool metadata is converted to OpenAI tool format during `RegularInterface` initialization.
3. When the LLM calls a tool, the agent spawns a subprocess that imports the tool's source file and calls the function.
4. Stdout from the subprocess is captured and fed back to the LLM as the tool result.

### The `@tool` decorator

```python
@tool(skill="skill_name", usually_takes=30)
def my_function(param1: str, param2: int = 10) -> str:
    '''This docstring becomes the tool description shown to the LLM.'''
    return "result"
```

- `skill` — groups this tool under a skill name. Used for skill-based activation/deactivation.
- `usually_takes` — estimated execution time in seconds. Shown to the LLM as a timeout hint.
- Type hints are used to generate JSON schema for the OpenAI API.
- The docstring is the tool description.

---

## 3. Using Skills

Skills group tools and add context retrieval. A skill can be **default** (always active) or **non-default** (available but must be activated by the agent).

```python
from fluid.regular_interface.interface import RegularInterface
from fluid.toolkit.tool_registry import discover_tools

all_tools = discover_tools(root_dir=".")

skills = [
    {
        "name": "general",
        "default": True,  # always active
        "template": "Context for general tasks:\n{placeholder}",
        "collections": [
            {
                "name": "knowledge_base",
                "max_result": 5,
                "threshold": 1.0,
                "collec_template": "Relevant knowledge:\n{placeholder}",
                "em_setup": {
                    "hf_local": "path/to/embedding/model",
                },
            },
        ],
        "backend": {
            "persistent_backend": {
                "path": "path/to/chromadb/directory",
            },
        },
    },
    {
        "name": "coding",
        "default": False,  # must be activated by the agent
        "template": "Coding skill context:\n{placeholder}",
        "collections": [...],
        "backend": {...},
    },
]

messages = [
    {"role": "user", "content": "Help me write a Python script."},
]

ri = RegularInterface(
    messages=messages,
    sys_messages=["You are an autonomous assistant."],
    skills=skills,
    max_new_skill=2,  # agent can activate up to 2 non-default skills
    all_tools=all_tools,
)

for event in ri.run_agent(...):
    # handle events
    ...
```

### What happens with skills

1. **Default skills** — Their tools are immediately available. If they have collections, those are queried for relevant context using the last user message.

2. **Non-default skills** — Listed in a system message as "available but not active." The agent gets `add_new_skill`, `remove_skill`, and `switch_skill` tools to manage them.

3. **Skill manipulation** — The agent can call `add_new_skill("coding")` to activate the coding skill and gain access to its tools. It can `switch_skill("coding", "debugging")` to swap one for another.

---

## 4. Configuring Vector Retrieval

Skills can have associated ChromaDB collections for context retrieval. This is how you give your agent domain knowledge.

### Embedding Options

You have two choices for embeddings:

**Option A: Local HuggingFace model**

```python
"em_setup": {
    "hf_local": "path/to/local/model",
    # or a HuggingFace model name:
    # "hf_local": "BAAI/bge-small-en-v1.5",
}
```

**Option B: OpenAI-compatible API**

```python
"em_setup": {
    "openai_embed": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "znbang/bge:large-en-v1.5-f16",
    },
}
```

### Backend Options

**Persistent (on-disk):**

```python
"backend": {
    "persistent_backend": {
        "path": "/path/to/chromadb/data",
    },
}
```

**HTTP (remote server):**

```python
"backend": {
    "http_backend": {
        "host": "chroma.example.com",
        "port": 8000,
        "ssl": True,
        "headers": {"Authorization": "Bearer ..."},
    },
}
```

### Collection Configuration

```python
{
    "name": "my_collection",       # ChromaDB collection name
    "max_result": 8,               # max results per query
    "threshold": 0.5,              # max embedding distance (lower = stricter)
    "meta": None,                  # metadata filter (ChromaDB where clause)
    "collec_template": "Relevant documents:\n{placeholder}",
    "em_setup": {...},             # embedding config
}
```

### Templates

Templates use `{placeholder}` to inject content:

- **Skill template** — wraps all collections' context for that skill
- **Collection template** — wraps context from a specific collection

They are nested: collection context goes into collection template, then collection templates go into skill template.

---

## 5. SRCF — Conversation Memory

For long conversations, SRCF keeps token usage manageable by filtering out irrelevant old messages.

```python
from fluid.regular_interface.srcf import SRCF

srcf = SRCF(
    srcf_percent=80,        # consider the oldest 80% for filtering
    srcf_last_n=5,          # use last 5 messages as the relevance query
    srcf_threshold=0.323,   # max distance to keep (lower = stricter)
    srcf_em_model="path/to/embedding/model",
)

ri = RegularInterface(
    messages=long_conversation,
    srcf=srcf,
    ...
)
```

### When to use SRCF

- Conversations with 50+ messages
- Multi-turn tasks where older context becomes noise
- When you're hitting token limits

### When NOT to use SRCF

- Short conversations (< 20 messages)
- Tasks where every message is critical
- When you want deterministic behavior (SRCF uses embedding similarity)

### How it filters

1. Groups messages into turns (user message + responses).
2. Takes the oldest 80% of turns.
3. Embeds them and queries with the most recent messages.
4. Keeps only turns with distance ≤ threshold.
5. Always keeps the newest 20% of turns unfiltered.

---

## 6. Handling Events

The `run_agent()` generator yields events for every phase of the agent's work. Here's a complete event handler:

```python
for event in ri.run_agent(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:14b",
    max_complition_tokens=4000,
    temperature=0.7,
):
    etype = event["type"]
    tc = event["tool_call"]

    if etype == "content":
        # Text token from the model
        print(event["content"], end="", flush=True)

    elif etype == "reasoning":
        # Reasoning/thinking token (if the model supports it)
        print(f"[think] {event['reasoning']}", end="", flush=True)

    elif etype == "tool_start":
        # A tool is about to be executed
        print(f"\n🔧 Starting {tc['tool_name']}({tc['tool_args']})")
        print(f"   Comment: {tc['tool_comment']}")
        print(f"   Timeout: {tc['timeout']}s")

    elif etype == "tool_update":
        # Live output from a running tool
        if tc["stdout"]:
            print(f"   stdout: {tc['stdout'][-100:]}")

    elif etype == "tool_timeout":
        # Tool exceeded its timeout, agent is deciding what to do
        print(f"   ⏰ Timeout hit for {tc['tool_name']} (pid={tc['process_id']})")

    elif etype == "tool_done":
        # Tool finished
        print(f"   ✓ {tc['tool_name']} completed: {tc['result'][:200]}")

    elif etype == "tool_error":
        # Tool had a JSON parse error
        print(f"   ✗ {tc['tool_name']} error: {tc['result'][:200]}")

    elif etype == "error":
        # LLM stream error
        print(f"\n!!! Error: {event['content']}")

    elif etype == "done":
        # Agent finished
        print(f"\n--- Done at turn {event['turn']} ---")
        if event["content"]:
            print(f"    {event['content']}")
```

---

## 7. Stopping the Agent

### From code

```python
ri.stop_all()
```

This:
1. Sets the `_stopped` flag (agent loop exits at next check)
2. Signals the LLM stream to stop
3. Terminates all running tool subprocesses

Thread-safe — can be called from any thread (e.g., a signal handler or a UI button).

### Killing a specific tool

```python
ri.terminate_process(pid=12345)
```

---

## 8. Timeouts and Tool Lifecycle

When a tool runs longer than its timeout:

1. The agent receives a `tool_timeout` event with the tool's current stdout/stderr.
2. A separate LLM call asks the agent to decide: extend or terminate.
3. The agent calls `tool_run_tool(next_timeout_window=N)`:
   - `N > 0` → extend by N seconds
   - `N = 0` → terminate the process

The timeout is clamped between 5 and 300 seconds. The default is 30 seconds.

### Timeout hints

Tools can declare `usually_takes` in the `@tool` decorator:

```python
@tool(skill="downloads", usually_takes=30)
def download_file(url: str) -> str:
    ...
```

This adds a hint to the tool description, helping the model set an appropriate `_venus_timeout`.

---

## 9. Full Example — Autonomous OS Agent

Here's a complete working example:

```python
from fluid.regular_interface.interface import RegularInterface
from fluid.toolkit.tool_registry import tool, discover_tools
import subprocess
from pathlib import Path

# --- Define tools ---

@tool(skill="OS")
def run_shell_command(cmd: str, timeout: int = 30) -> str:
    '''Run a shell command and return its output.'''
    result = subprocess.run(cmd, shell=True, capture_output=True,
                          text=True, timeout=timeout)
    return f"RETURNCODE: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

@tool(skill="OS")
def write_text_file(path: str, content: str) -> str:
    '''Write text content to a file.'''
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {p.resolve()}"

@tool(skill="OS")
def read_text_file(path: str) -> str:
    '''Read and return file contents.'''
    return Path(path).read_text(encoding="utf-8")[:5000]

# --- Configure and run ---

all_tools = discover_tools(root_dir=".")

ri = RegularInterface(
    messages=[
        {"role": "user", "content": "List files in /tmp and create a summary.txt there."},
    ],
    sys_messages=[
        "You are an autonomous OS agent. Execute tasks using your tools.",
    ],
    all_tools=all_tools,
    max_turns=10,
)

for event in ri.run_agent(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:14b",
):
    etype = event["type"]
    if etype == "content":
        print(event["content"], end="", flush=True)
    elif etype == "tool_start":
        print(f"\n🔧 {event['tool_call']['tool_name']}")
    elif etype == "tool_done":
        print(f"   ✓ Done")
    elif etype == "done":
        print(f"\n--- Complete ---")
```

---

## 10. Configuration Reference

### `RegularInterface` constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list` | required | Conversation in OpenAI message format |
| `skills` | `list` | `[]` | Skill configurations |
| `max_new_skill` | `int` | `2` | Max non-default skills agent can activate |
| `sys_messages` | `list` | `[]` | Custom system prompt strings |
| `srcf` | `SRCF \| None` | `None` | SRCF instance for message filtering |
| `all_tools` | `list` | `[]` | Tool metadata from `discover_tools()` |
| `max_turns` | `int` | `20` | Max LLM call cycles before stopping |

### `run_agent()` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | required | LLM API endpoint |
| `api_key` | `str` | required | API key |
| `model` | `str` | required | Model name |
| `tool_choice` | `str` | `"auto"` | Tool selection strategy |
| `timeout` | `int \| None` | `None` | HTTP request timeout |
| `max_retires` | `int` | `5` | Max retries for failed LLM calls |
| `max_complition_tokens` | `int` | `3000` | Max tokens per LLM turn |
| `temperature` | `float \| None` | `None` | Sampling temperature |

### Skill configuration

```python
{
    "name": "skill_name",            # required
    "default": True,                 # True = always active, False = must be activated
    "template": "{placeholder}",     # optional, wraps all context for this skill
    "device": "cpu",                 # optional, device for embedding model
    "collections": [...],           # required if using retrieval
    "backend": {
        "persistent_backend": {"path": "..."},
        # OR
        "http_backend": {"host": "...", "port": 8000},
    },
}
```

### SRCF configuration

```python
SRCF(
    srcf_percent=80,      # % of old messages to consider for filtering
    srcf_last_n=5,        # recent messages used as relevance query
    srcf_threshold=0.323, # max embedding distance to keep
    srcf_em_model="...",  # path to embedding model
)
```
