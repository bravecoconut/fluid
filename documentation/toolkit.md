# `fluid/toolkit/tool_registry.py`

Tool discovery and registration system. Provides the `@tool` decorator and the `discover_tools()` function.

## Decorator: `@tool(skill=None, usually_takes=None)`

Marks a function as a tool and registers it in the global `TOOL_REGISTRY`.

```python
@tool(skill="math", usually_takes="5")
def add(a: int, b: int) -> int:
    '''Add two numbers together.'''
    return a + b
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `skill` | `str \| None` | Which skill group this tool belongs to |
| `usually_takes` | `str \| None` | Estimated execution time in seconds |

The decorator sets three attributes on the function:
- `func._is_tool = True`
- `func._skill = skill`
- `func._usually_takes = usually_takes`

## Function: `discover_tools(root_dir=None)`

Scans a directory tree for Python files, imports each one (triggering `@tool` decorators), and returns metadata for all discovered tools.

**Parameters:**
- `root_dir` — directory to scan. Defaults to `os.getcwd()`.

**Returns:** `list[dict]` — one dict per tool, with this shape:

```python
{
    "name": "add",                    # function name
    "skill": "math",                  # from @tool(skill=...)
    "usually_takes": "5",             # from @tool(usually_takes=...)
    "description": "Add two numbers", # from the docstring
    "parameters": [                   # from function signature + type hints
        {
            "name": "a",
            "type": "<class 'int'>",
            "required": True,
            "default": None,
        },
        ...
    ],
    "return_type": "<class 'int'>",
    "module": "add_module",
    "source_file": "/path/to/file.py",
}
```

### How Discovery Works

1. `_find_python_files(root_dir)` walks the directory tree, collecting `.py` files. It skips:
   - Virtual environments (`.venv`, `venv`, `env`, `.env`)
   - Caches (`__pycache__`, `.mypy_cache`, `.pytest_cache`)
   - Version control (`.git`, `.hg`, `.svn`)
   - Package directories (`node_modules`, `site-packages`)
   - Build artifacts (`build`, `dist`)
   - Hidden directories (anything starting with `.`)
   - The file `tool_registry.py` itself

2. `_import_module_from_path(file_path)` imports each file using `importlib.util`, which executes the file's top-level code and triggers any `@tool` decorators.

3. `_describe_tool(func)` inspects each registered function to extract its metadata (signature, type hints, docstring, source file).

### Important Notes

- `TOOL_REGISTRY` is cleared on each call to `discover_tools()`, so calling it twice doesn't produce duplicates.
- Files that fail to import are skipped with a warning, not a crash.
- Type hints are resolved via `typing.get_type_hints()`. If resolution fails, hints fall back to empty.

---

# `fluid/toolkit/tool_runner.py`

Executes tool functions in isolated subprocesses with live output streaming and timeout management.

## Function: `start_process(...)`

```python
start_process(
    task_id,       # str — identifier for this task
    tool_name,     # str — name of the tool function
    tool_args,     # dict — keyword arguments
    avail_tools,   # list — tool metadata (from discover_tools)
    timeout,       # int — initial timeout in seconds
    in_q,          # Queue — for receiving "need_input" prompts
    out_q,         # Queue — for sending decision responses
)
```

A generator that drives a tool's execution and yields status updates.

### How It Works

1. **Lookup** — Finds `module_name` and `source_file` from `avail_tools`.

2. **Build command** — Generates a Python one-liner that:
   - Imports the module from its file path
   - Calls the function with the given arguments
   - Handles async functions via `asyncio.run()`
   - Prints the result

3. **Spawn** — Runs the command as a `subprocess.Popen` with `python3 -c`.

4. **Live capture** — Two background threads pump `stdout` and `stderr` into buffers.

5. **Monitor loop** — Polls every second:
   - If the process finishes → yields final output and returns.
   - If timeout expires → yields an `awaiting_decision` update, then waits for a decision via queues:
     - `≤ 0` → terminate the process
     - `> 0` → extend the deadline by that many seconds

### Yielded Updates

Each yielded dict has:

| Key | Description |
|-----|-------------|
| `process` | `{"pid": int, "stdout": str, "stderr": str}` |
| `call_reason` | Human-readable status message |
| `tool_name` | Name of the tool |
| `tool_args` | Arguments passed |
| `awaiting_decision` | `True` when timeout expired and waiting for input |
| `stated_messages` | `True` on the initial yield |

### Environment

- `PYTHONUNBUFFERED=1` is set so output arrives in real time rather than being buffered.

### Logging

Logs to `logs/tool_runner.log`.
