# Fluid — Framework Overview

Fluid is a Python agent framework for building LLM-powered agents that can discover, manage, and execute tools at runtime. It uses an OpenAI-compatible API for generation, ChromaDB for vector retrieval, and a skill-based architecture that lets agents dynamically load and unload capabilities.

## Core Concepts

### Skills

A **skill** is a named group of related tools and an optional knowledge base. Skills can be:

- **Default** — always active. The agent has permanent access to their tools and retrieves relevant context from their vector collections at the start of each run.
- **Non-default** — available but inactive. The agent can activate them mid-conversation using built-in skill manipulation tools (`add_new_skill`, `remove_skill`, `switch_skill`).

### Tools

A **tool** is a Python function decorated with `@tool()`. Tools are discovered automatically by scanning a directory, and each tool declares which skill it belongs to. At runtime, tools are executed in isolated child processes with configurable timeouts.

### SRCF (Self-Referential Context Filtering)

SRCF is an optional conversation memory system. When conversations grow long, SRCF uses embedding similarity to keep only the messages most relevant to the current task, reducing token usage while preserving important context.

### Vector Retrieval

Skills can have associated ChromaDB collections. When a user message comes in, Fluid queries these collections in parallel (across worker processes) to retrieve relevant context, which is then injected into the agent's system prompt.

## Architecture

```
fluid/
├── generation/
│   └── openai.py            # OpenAI-compatible streaming client
├── regular_interface/
│   ├── interface.py          # Core agent loop (RegularInterface)
│   ├── srcf.py               # Self-Referential Context Filtering
│   ├── build_system_messages/
│   │   ├── _main.py          # System message builder
│   │   └── system_message.py # Prompt templates
│   ├── embedding_models/     # Local embedding model storage
│   └── system_tools/
│       └── agent_inbuild_tools.py  # Skill manipulation tool definitions
├── toolkit/
│   ├── tool_registry.py      # @tool decorator and discovery
│   └── tool_runner.py        # Subprocess-based tool execution
├── vector_retrive/
│   ├── retrive.py            # ChromaDB query functions
│   └── retriveing_pool.py    # Parallel retrieval pool
└── utill.py                  # Shared utilities and helpers
```

## How a Request Flows

1. **Initialization** — `RegularInterface` receives messages, skill configs, and discovered tools.
2. **Skill filtering** — Skills are split into default and non-default. Tools are matched to their skills.
3. **Context retrieval** — Default skills' ChromaDB collections are queried in parallel for relevant context.
4. **System message construction** — `BuildSystemMessages` formats retrieved context and non-default skill info into system prompts.
5. **SRCF (optional)** — If enabled, older messages are filtered by embedding similarity to keep only relevant history.
6. **Agent loop** — The agent calls the LLM, streams the response, and processes tool calls turn by turn.
7. **Tool execution** — External tools run in child processes with timeouts. The agent can extend or terminate timed-out tools.
8. **Repeat** — Results feed back into the message history, and the loop continues until the model says "stop" or max turns are reached.

## Dependencies

- `openai` — OpenAI-compatible API client
- `chromadb` — Vector database for retrieval
- `sentence_transformers` — Local embedding models (HuggingFace)
