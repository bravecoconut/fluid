# `fluid/regular_interface/build_system_messages/_main.py`

Constructs system messages that inform the agent about available skills and retrieved context.

## Class: `BuildSystemMessages`

### Purpose

After skills are partitioned and context is retrieved, this class formats everything into two system message strings:

1. **Non-default skills and their tools** — Tells the agent what skills are _available but inactive_, including tool names, descriptions, and estimated execution times. The agent can use skill manipulation tools to activate them.

2. **Default skills and their contexts** — Injects retrieved context from vector collections into the system prompt, using configurable templates at both the skill and collection level.

### Constructor

```python
BuildSystemMessages(
    default_skills,       # list — default skill configs
    non_default_skills,   # list — non-default skill configs
    grouped_contexts,     # dict — retrieved context grouped by skill
    off_tools,            # list — tools belonging to inactive skills
)
```

### Methods

#### `build_skills_and_tools()`

Pairs each non-default skill with the tools that belong to it (matched by `tool["skill"] == skill["name"]`).

Returns a list of `{"skill": {...}, "tools": [...]}` dicts.

#### `build_skills_and_context()`

Pairs each default skill with its retrieved context (matched by `skill_name` against `grouped_contexts`).

Returns a list of `{"skill": {...}, "contexts": [...]}` dicts.

#### `format_them()`

Main method. Calls both builders above and formats the results into two system message strings.

**Returns:** `(non_default_sys_message, default_sys_message)` — two strings.

### Template System

Skills and collections support `{placeholder}` templates:

- **Skill template** — Wraps all context for that skill. Example: `"Your OS tools context:\n{placeholder}"`
- **Collection template** — Wraps context from a specific collection. Example: `"Relevant docs:\n{placeholder}"`

Templates use Python's `.format(placeholder=...)`. If formatting fails (bad template syntax), the raw content is used as a fallback.

### Output Format

**Non-default skills message** (example):
```
THESE SKILLS ARE AVAILABLE, BUT NOT ACTIVE.
USE SKILL MANIPULATION TO ACTIVE THEM IF NECESSARY.

    - SKILL NAME: web_search
        - TOOL NAME: google_search
        - TIME NEEDED: usually takes 5 seconds
        - DESCRIPTION: Search the web using Google...
```

**Default skills message** (example):
```
SKILL NAME: OS

    - Your OS tools context:
        - Relevant docs:
            - <retrieved document content>
```

---

# `fluid/regular_interface/build_system_messages/system_message.py`

String constants used as prompt templates.

### `SYSTEM_MESSAGE`

Prepended when SRCF is active. Explains to the LLM that some messages may be missing due to pruning, and that retained context is sufficient.

### `TOOL_INPUT_REQ`

Used during timeout decisions. Instructs the LLM to call `tool_run_tool` with either `0` (terminate) or a positive number (extend).

**Placeholders:** `{tool_name}`, `{tool_id}`

### `TOOL_UPDATE`

Shows the current state of a running tool during timeout decisions.

**Placeholders:** `{tool_id}`, `{tool_name}`, `{pid}`, `{c_outputs}`, `{error}`
