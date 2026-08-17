# Fluid — Agent Framework

Fluid is a Python agent framework for building LLM-powered agents that can discover, manage, and execute tools at runtime. It uses an OpenAI-compatible API for generation, ChromaDB for vector retrieval, and a skill-based architecture that lets agents dynamically load and unload capabilities.

## Core Concepts

* **Skills**: A named group of related tools and an optional knowledge base. Can be default (always active) or non-default (activated mid-conversation).
* **Tools**: Python functions decorated with `@tool()`, discovered automatically and executed in isolated child processes with timeouts.
* **SRCF (Self-Referential Context Filtering)**: Optional conversation memory system for long conversations, filtering messages by embedding similarity to save tokens.
* **Vector Retrieval**: Skills query associated ChromaDB collections in parallel to inject relevant context into the agent's system prompt.

## Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM endpoint (Ollama, vLLM, OpenRouter, etc.)
- (Optional) A ChromaDB instance for vector retrieval

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Fluid.git
cd Fluid
pip install -r requirements.txt
```

*(Alternatively, you can just install the core dependencies: `pip install openai chromadb sentence_transformers`)*

## Quick Start: Your First Agent

The simplest agent is just an LLM loop without tools, skills, or retrieval:

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

For more advanced features like tools, skills, vector retrieval, and SRCF, please check out the comprehensive documentation in the `documentation/` folder, and the step-by-step tutorial in the `tutorial/` folder.

## Architecture

Fluid is built with a modular structure:

```
fluid/
├── generation/           # OpenAI-compatible streaming client
├── regular_interface/    # Core agent loop, memory filtering (SRCF), tools
├── toolkit/              # Tool registry and child-process runners
├── vector_retrive/       # ChromaDB interactions and parallel retrieval
└── utill.py              # Shared utilities
```

## Documentation

Full documentation is available in the [`documentation/`](./documentation/) directory, including:
- Interface & core agent loops (`interface.md`)
- Toolkit and tool execution (`toolkit.md`)
- Vector retrieval details (`vector_retrive.md`)
- SRCF and memory filtering (`srcf.md`)

Step-by-step guides and examples can be found in the [`tutorial/`](./tutorial/) directory.
# fluid
# fluid
