# `fluid/utill.py`

Shared utility functions used across the framework. Contains ID generation, skill/tool filtering, message processing, type mapping, tool call parsing, and safety wrappers.

## ID Generation

### `generate_buffer_id(buffers, length)`

Generates a random alphanumeric string of `length` characters that doesn't collide with existing IDs in `buffers`.

### `mid(mids, length)`

Same as `generate_buffer_id` but used for collection/task identifiers.

## Skill and Tool Filtering

### `_filter_default_and_non_default_skills(locations)`

Splits a list of skill location dicts into `(default_skills, non_default_skills)` based on the `"default"` key.

- Missing `"default"` key → treated as default (`True`)
- Non-boolean `"default"` value → warned, treated as default

### `filter_tools_by_default_skills(default_skills, all_tools)`

Splits `all_tools` into `(matching_tools, other_tools)` based on whether each tool's `"skill"` matches a default skill's `"name"`.

A tool's `"skill"` can be a string, a list of strings, or missing. Matches if _any_ skill name is in the default set.

### `_find_skill_index(skill_list, skill_name)`

Returns the index of a skill by name, or `None`.

### `_tool_matches_skill(tool, skill_name)`

Returns `True` if a tool's `"skill"` field includes the given name.

### `_pop_tools_for_skill(skill_name, source_list)`

Removes and returns all tools matching a skill from `source_list` in place. Iterates backward to avoid index-shifting bugs.

## Message Processing

### `_split_system_messages(messages)`

Separates system messages from non-system messages without mutating the input.

Returns `(system_messages, other_messages)`.

### `_group_into_turns(messages)`

Groups messages into turns — a user message plus everything after it (assistant replies, tool calls, tool results) until the next user message. Guarantees tool_use and tool_result are never separated.

### `trim_messages(grouped_messages, percent)`

Splits grouped messages by percentage. Returns `(first_portion, last_portion)`.

## Tool Format Conversion

### `convert_to_openai_tools(raw_tools_list)`

Converts internal tool metadata dicts into OpenAI-compatible `tools` format.

For each tool, it:
- Maps Python types to JSON schema types (`str` → `"string"`, `int` → `"integer"`, etc.)
- Adds a `_venus_timeout` parameter (lets the model specify a timeout hint)
- Adds a `_tool_comment` parameter (lets the model describe what it's doing)
- Appends a timeout hint to the description based on `usually_takes`

### `_map_type(type_str)`

Maps Python type representations to JSON schema types:

| Python | JSON Schema |
|--------|-------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |
| `tuple` | `array` |
| `NoneType` | `None` |
| anything else | `string` |

### `_build_timeout_note(usually_takes)`

Returns a timeout hint string. If `usually_takes` is `None`, suggests 10-15s.

## Tool Call Parsing

### `return_completed_tool_calls(tool_calls_buffer)`

Assembles streamed tool call deltas into complete tool calls.

1. Groups deltas by `index`.
2. Accumulates `id`, `name`, and `arguments` fragments.
3. Parses the concatenated arguments as JSON.
4. Extracts `_venus_timeout` and `_tool_comment` from arguments.
5. On JSON parse failure, returns an error entry with the raw buffer preserved.

**Returns:** list of dicts:

```python
{
    "id": "call_abc123",
    "name": "tool_name",
    "tool_timeout": 30,      # extracted from _venus_timeout
    "tool_comment": "...",   # extracted from _tool_comment
    "arguments": {...},      # parsed args (without _venus_timeout/_tool_comment)
    "error": None,           # or "_args_error"
    "content": None,         # or error message for the model
    "raw_args": "...",       # only present on error
}
```

## Safety Wrappers

Defensive utility functions that prevent crashes from bad data:

| Function | Purpose |
|----------|---------|
| `_safe_get(obj, key, default)` | Dict get that handles non-dict inputs |
| `_safe_iter(obj)` | Returns an iterable; None → `[]` |
| `_safe_format(template, **kwargs)` | `.format()` that never raises |
| `safe_get(source, key, default, label)` | Verbose version with logging |
| `safe_list(value, label)` | Ensures value is a list |
| `safe_dict_items(value, label)` | Ensures value is a dict, returns `.items()` |
| `safe_strip(text, default, label)` | `.strip()` that handles None |
| `safe_format(template, label, **kwargs)` | Verbose `.format()` with logging |

## Other

### `json_ser(obj)`

Simple JSON serializer fallback — just calls `str(obj)`.

### `load_and_save_em_models(em_name, dir)`

Downloads a HuggingFace model via `snapshot_download` and saves it locally.

## Logging

Uses `_build_logger(log_dir, log_file)` to create a rotating file handler. Safe to call multiple times — only adds a handler once per process.

Logs to `logs/utilss.log`.
