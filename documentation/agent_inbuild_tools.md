# `fluid/regular_interface/system_tools/agent_inbuild_tools.py`

Defines the OpenAI-format tool schemas for the agent's built-in tools. These are not discovered via `@tool` — they are hardcoded and handled inline by `RegularInterface`.

## `SKILL_MANIPULATION_TOOLS`

A list of three tool definitions added to the agent when skills are configured:

### `add_new_skill`

Activates a non-default skill and its tools.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `skill_name` | `string` | yes | Exact name of the skill to activate |
| `_tool_comment` | `string` | no | Brief reason for calling this tool |

### `remove_skill`

Deactivates a non-default skill and revokes its tool access.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `skill_name` | `string` | yes | Exact name of the skill to remove |
| `_tool_comment` | `string` | no | Brief reason for calling this tool |

### `switch_skill`

Swaps one active non-default skill for another.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `skill_name` | `string` | yes | Skill to remove |
| `to_which` | `string` | yes | Skill to activate in its place |
| `_tool_comment` | `string` | no | Brief reason for calling this tool |

## `TOOL_FOR_TOOL`

A single-element list containing `tool_run_tool`. This is used during timeout decisions — the agent is called with _only_ this tool available, forcing it to make a timeout decision.

### `tool_run_tool`

Controls a running tool's timeout.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `next_timeout_window` | `number` | yes | `0` = terminate now. Positive = extend by N seconds. |
| `_tool_comment` | `string` | no | Brief reason for calling this tool |

## `_tool_comment`

All built-in tools accept a `_tool_comment` parameter. This is a short (3-10 word) description of _why_ the agent is calling the tool. It gets extracted during argument parsing and included in event updates for observability.
