from logging.handlers import RotatingFileHandler
import logging
import os
import random
import string
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, List, Tuple
from huggingface_hub import snapshot_download
import json

LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "utilss.log")


def _build_logger(log_dir, log_file):
    """
    Sets up (once per process) a logger that writes to
    logs/vector_retrieve.log. Safe to call many times -- it only adds a
    handler the first time, so messages never get duplicated.
    """
    logger = logging.getLogger("vector_retrieve")

    if logger.handlers:
        return logger  # already configured in this process

    os.makedirs(log_dir, exist_ok=True)
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s: %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


logger = _build_logger(
    log_dir=LOG_DIR,
    log_file=LOG_FILE,
)


def generate_buffer_id(buffers: list, length: int):
    characters = string.ascii_letters + string.digits
    # `buffers` is a flat list of already-used ID strings (e.g. from dict.keys()).
    used = set(buffers)
    while True:
        buffer_id = "".join(random.choice(characters) for _ in range(length))
        if buffer_id not in used:
            return buffer_id


def mid(mids: list, length: int) -> str:
    characters = string.ascii_letters + string.digits
    used = set(mids)  # mids is already a flat list/set of used ids
    while True:
        _mid = "".join(random.choice(characters) for _ in range(length))
        if _mid not in used:
            return _mid


# ---------------------------------------------------------------------------
# LOGGING SETUP
#
# Every call to chroma_db_persistent / chroma_db_http writes a line to
# logs/vector_retrieve.log -- what was queried, whether it succeeded, and
# any error that happened. This means if something goes wrong hours later
# (or in a background worker you can't see the terminal for), you can just
# open the log file and see exactly what happened.
#
# NOTE for multiprocessing: each worker process sets up its OWN logger and
# they all write to the same file. For short log lines this works fine in
# practice, but it's not perfectly "safe" the way a database write is --
# under heavy concurrent load, lines could theoretically interleave. If you
# ever need bulletproof multi-process logging, look into Python's
# `logging.handlers.QueueHandler`. For a project this size, this is fine.
# ---------------------------------------------------------------------------
def _filter_default_and_non_default_skills(locations):
    """
    Splits `locations` into two lists based on their 'default' value.

    Returns (default_skills, non_default_skills):
      - default_skills:     locations with default=True, or missing/invalid 'default'
                             (missing/invalid is treated as default -- see notes below)
      - non_default_skills: locations with default=False, exactly

    A location missing the 'default' key entirely is treated as default
    (True), not non-default. Flip the `location.get("default", True)` line
    below to `False` if you want the opposite assumption.
    """
    if not isinstance(locations, list):
        raise ValueError(f"'locations' must be a list, got: {type(locations).__name__}")

    default_skills = []
    non_default_skills = []

    for index, location in enumerate(locations):
        if not isinstance(location, dict):
            logger.warning(
                f"Skipping location #{index}: expected a dict, got {type(location).__name__}"
            )
            continue

        is_default = location.get("default", True)  # missing key -> treated as default

        if not isinstance(is_default, bool):
            logger.warning(
                f"Location {location.get('name', f'#{index}')!r} has a non-boolean "
                f"'default' value ({is_default!r}) -- treating it as default."
            )
            is_default = True

        if is_default is False:
            non_default_skills.append(location)
        else:
            default_skills.append(location)

    return default_skills, non_default_skills


def filter_tools_by_default_skills(default_skills, all_tools):
    """
    Splits `all_tools` into two lists based on whether each tool's "skill"
    matches the "name" of one of the locations in `default_skills`.

    A tool's "skill" can be:
      - a single string, e.g. "coding"
      - a list of strings, e.g. ["coding", "text"]  -- matches if ANY are in default_skills
      - missing, None, or empty (""/[]) -- always treated as non-matching

    Returns (matching_tools, other_tools):
      - matching_tools: tools with at least one "skill" that IS a default skill name
      - other_tools:     every other tool (no skill matched, or skill was missing/malformed)
    """
    if not isinstance(default_skills, list):
        raise ValueError(
            f"'default_skills' must be a list, got: {type(default_skills).__name__}"
        )

    if not isinstance(all_tools, list):
        raise ValueError(f"'all_tools' must be a list, got: {type(all_tools).__name__}")

    skill_names = set()

    for index, skill in enumerate(default_skills):
        if not isinstance(skill, dict) or "name" not in skill:
            logger.warning(
                f"default_skills[{index}] is missing a 'name' key or isn't a dict "
                f"-- skipping it: {skill!r}"
            )
            continue
        skill_names.add(skill["name"])

    matching_tools = []
    other_tools = []

    for index, tool in enumerate(all_tools):
        if not isinstance(tool, dict):
            logger.warning(
                f"all_tools[{index}] isn't a dict -- treating it as non-matching: {tool!r}"
            )
            other_tools.append(tool)
            continue

        tool_skill = tool.get("skill")

        # normalize into a list of candidate skill names, regardless of
        # whether the tool defined "skill" as a single string or a list
        if tool_skill is None:
            candidates = []
        elif isinstance(tool_skill, str):
            candidates = [tool_skill]
        elif isinstance(tool_skill, list):
            candidates = [s for s in tool_skill if isinstance(s, str)]
            if len(candidates) != len(tool_skill):
                logger.warning(
                    f"Tool {tool.get('name', f'#{index}')!r} has non-string entries in "
                    f"its 'skill' list -- ignoring those, keeping the rest: {tool_skill!r}"
                )
        else:
            logger.warning(
                f"Tool {tool.get('name', f'#{index}')!r} has an unsupported 'skill' type "
                f"({type(tool_skill).__name__}) -- treating it as non-matching: {tool_skill!r}"
            )
            candidates = []

        if not candidates:
            # covers: missing, None, "", [], or a fully-invalid list/type
            other_tools.append(tool)
            continue

        # matches if ANY of the tool's skills is a known default skill name
        if any(name in skill_names for name in candidates):
            matching_tools.append(tool)
        else:
            other_tools.append(tool)

    return matching_tools, other_tools


def json_ser(obj):
    return str(obj)


# ---------------------------------------------------------------------------
# Small helpers, kept outside the class so they're easy to test on their
# own and reused by all three methods -- no duplicated matching logic.
# ---------------------------------------------------------------------------
def _find_skill_index(skill_list, skill_name):
    """Returns the index of the skill named `skill_name` in `skill_list`, or None."""
    for i, skill in enumerate(skill_list):
        if isinstance(skill, dict) and skill.get("name") == skill_name:
            return i
    return None


def _tool_matches_skill(tool, skill_name):
    """
    True if `tool`'s "skill" field includes `skill_name`. Handles "skill"
    being a single string, a list of strings, or missing/empty -- same
    normalization used in filter_tools_by_default_skills.
    """
    if not isinstance(tool, dict):
        return False

    tool_skill = tool.get("skill")

    if isinstance(tool_skill, str):
        return tool_skill == skill_name
    if isinstance(tool_skill, list):
        return skill_name in tool_skill
    return False  # None, empty, or any other type -> no match


def _pop_tools_for_skill(skill_name, source_list):
    """
    Removes and returns ALL tools in `source_list` matching `skill_name`,
    mutating `source_list` in place. Walks backward so popping doesn't
    shift the index of items not yet checked -- this is what makes it
    safe to remove multiple matches in one pass instead of just the
    first one.
    """
    moved = []
    for i in range(len(source_list) - 1, -1, -1):
        if _tool_matches_skill(source_list[i], skill_name):
            moved.append(source_list.pop(i))
    moved.reverse()  # restore original order, since we collected back-to-front
    return moved


from typing import Any, Dict, List, Tuple

Message = Dict[str, Any]


def _split_system_messages(
    messages: List[Message],
) -> Tuple[List[Message], List[Message]]:
    """Pull out system messages without mutating the input list."""
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]
    return system_messages, other_messages


def _group_into_turns(messages: List[Message]) -> List[List[Message]]:
    """
    Group messages into "turns": a user message plus everything that
    follows it (assistant replies, tool calls, tool results) up to the
    next user message. This guarantees trimming later never separates
    a tool_use call from its tool_result, or a user question from its
    answer.
    """
    turns: List[List[Message]] = []
    current_turn: List[Message] = []

    for message in messages:
        is_new_user_turn = message.get("role") == "user" and current_turn
        if is_new_user_turn:
            turns.append(current_turn)
            current_turn = [message]
        else:
            current_turn.append(message)

    if current_turn:
        turns.append(current_turn)

    return turns


def trim_messages(grouped_messages, percent):
    rounded_pg = round(len(grouped_messages) * percent / 100)
    first_m = grouped_messages[:rounded_pg]
    last_m = grouped_messages[rounded_pg:]

    return first_m, last_m


def load_and_save_em_models(
    em_name="BAAI/bge-small-en-v1.5",
    dir="modules/regular_interface/embedding_models",
):
    snapshot_download(
        repo_id=em_name,
        local_dir=f"{dir}/{em_name.replace("/","_")}",
    )


def _safe_get(obj, key, default=None):
    """Get `key` from obj if obj is a dict, else return default. Never raises."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _safe_iter(obj):
    """Return an iterable version of obj. None/non-iterable -> empty list."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return obj
    if isinstance(obj, dict):
        # dicts iterate over keys by default; usually not what we want here,
        # but we don't silently swallow real dicts either.
        return obj
    # Anything else (str, int, unexpected type) - treat as not iterable-safe
    return []


def _safe_format(template, **kwargs):
    """Format a template string, never raising. Falls back to raw template on failure."""
    if not isinstance(template, str):
        return ""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError) as e:
        print(
            f"[WARN] template.format() failed ({e!r}); using raw template. Template: {template!r}"
        )
        return template


# =============================================================================
# HELPER FUNCTIONS
#
# These are small, reusable "safety wrapper" functions. Instead of letting
# the program crash when data is missing, wrong-typed, or None, they print
# a clear warning and return a sensible default value instead.
# =============================================================================


def safe_get(source, key, default=None, label="value"):
    """
    Safely get `key` from a dictionary called `source`.
    - If `source` isn't a dict, or is None -> warn and return default.
    - If `key` is missing -> warn and return default.
    - If the value stored is None -> warn and return default.
    """
    print(f"[CHECK] Looking for key '{key}' in {label}...")

    if source is None:
        print(
            f"[WARNING] {label} is None. Cannot get key '{key}'. Using default: {default!r}"
        )
        return default

    if not isinstance(source, dict):
        print(
            f"[WARNING] {label} is not a dictionary (got {type(source).__name__}). Using default: {default!r}"
        )
        return default

    if key not in source:
        print(f"[WARNING] Key '{key}' not found in {label}. Using default: {default!r}")
        return default

    value = source[key]

    if value is None:
        print(f"[WARNING] Key '{key}' in {label} is None. Using default: {default!r}")
        return default

    print(f"[OK] Found '{key}' in {label}: {value!r}")
    return value


def safe_list(value, label="list"):
    """
    Make sure `value` is something we can safely loop over.
    - If it's None or not a list/tuple -> warn and return an empty list.
    """
    if value is None:
        print(f"[WARNING] {label} is None. Treating it as an empty list.")
        return []

    if not isinstance(value, (list, tuple)):
        print(
            f"[WARNING] {label} is not a list (got {type(value).__name__}). Treating it as an empty list."
        )
        return []

    print(f"[OK] {label} has {len(value)} item(s).")
    return value


def safe_dict_items(value, label="dictionary"):
    """
    Make sure `value` is a usable dictionary and return its .items().
    - If it's None or not a dict -> warn and return an empty list of items.
    """
    if value is None:
        print(f"[WARNING] {label} is None. Treating it as an empty dictionary.")
        return []

    if not isinstance(value, dict):
        print(
            f"[WARNING] {label} is not a dictionary (got {type(value).__name__}). Treating it as an empty dictionary."
        )
        return []

    print(f"[OK] {label} has {len(value)} key(s).")
    return list(value.items())


def safe_strip(text, default="", label="text"):
    """
    Safely call .strip() on something that is supposed to be text.
    - If it's None -> return default.
    - If .strip() fails for any reason -> print the error and return default.
    """
    if text is None:
        print(f"[WARNING] {label} is None, cannot strip it. Using default: {default!r}")
        return default
    try:
        return str(text).strip()
    except Exception as error:
        print(f"[ERROR] Could not strip {label} ({text!r}): {error}")
        return default


def safe_format(template, label="template", **kwargs):
    """
    Safely call .format(**kwargs) on a template string.
    - If template is empty/None -> warn and return "".
    - If formatting fails (e.g. a placeholder is missing) -> print the error
      and return "" instead of crashing.
    """
    if not template:
        print(
            f"[WARNING] {label} is empty or None. Cannot format it. Returning empty string."
        )
        return ""
    try:
        return template.format(**kwargs)
    except Exception as error:
        print(f"[ERROR] Failed to format {label} ({template!r}) with {kwargs}: {error}")
        return ""


import re


def _map_type(type_str: str) -> str:
    """
    Map a Python type repr (e.g. "<class 'str'>") or a bare type name
    (e.g. "Any", "int") to a JSON schema type. Falls back to "string"
    for anything unrecognized/dynamic (like "Any").
    """
    match = re.match(r"<class '(.+)'>", type_str)
    py_type = match.group(1) if match else type_str

    mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "tuple": "array",
        "NoneType": "None",
    }
    return mapping.get(py_type, "string")  # "Any" and unknowns -> "string"


def _build_timeout_note(usually_takes) -> str:
    if usually_takes is None:
        return "[timeout hint: completes quickly, use 10-15s]"
    return f"[timeout hint: usually takes ~{usually_takes} seconds, set _venus_timeout accordingly]"


def convert_to_openai_tools(raw_tools_list: list[dict]) -> list[dict]:
    """
    Convert a list of internal skill/function definitions into an
    OpenAI-compatible `tools` list.
    """
    tools = []

    for skill in raw_tools_list:
        properties = {}
        required = []

        for param in skill.get("parameters", []):
            prop = {"type": _map_type(param.get("type", "Any"))}

            if param.get("default") is not None:
                prop["default"] = param["default"]

            properties[param["name"]] = prop

            if param.get("required"):
                required.append(param["name"])

        # extra param: lets the caller specify a timeout (in seconds) for this tool call
        properties["_venus_timeout"] = {
            "type": "number",
            "description": (
                "Timeout in seconds for this tool call. "
                "Set based on the timeout hint in the tool description. "
                "Example: 10 for quick tasks, 30+ for downloads/network."
            ),
        }

        properties["_tool_comment"] = {
            "type": "string",
            "description": "describe what you are doing or why you wants to run it [under 3-10 words].",
        }

        timeout_note = _build_timeout_note(skill.get("usually_takes"))
        description = f"{skill.get('description', '').rstrip()} {timeout_note}"

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": skill["name"],
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )

    return tools


import json


def return_completed_tool_calls(tool_calls_buffer):
    grouped = {}  # index -> {"id":..., "name":..., "args_buffer":...}

    for tc in tool_calls_buffer:
        idx = tc["index"]
        entry = grouped.setdefault(idx, {"id": None, "name": None, "args_buffer": ""})

        if tc.get("id"):
            entry["id"] = tc["id"]

        fn = tc.get("function") or {}
        if fn.get("name"):
            entry["name"] = fn["name"]
        if fn.get("arguments"):
            entry["args_buffer"] += fn["arguments"]

    tool_calls = []
    for idx in sorted(grouped):
        entry = grouped[idx]

        try:
            args = json.loads(entry["args_buffer"])
            if not isinstance(args, dict):
                raise ValueError("arguments parsed but is not a JSON object")
        except (json.JSONDecodeError, ValueError):
            tool_calls.append(
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "tool_timeout": None,
                    "tool_comment": None,
                    "arguments": None,
                    "error": "_args_error",
                    "content": (
                        f"Your arguments for tool '{entry['name']}' were not valid JSON "
                        "and could not be parsed. Please resend this tool call with "
                        "corrected, valid JSON arguments."
                    ),
                    # Keep the raw buffer so the assistant message can
                    # include it (OpenAI requires all tool_calls to have
                    # an "arguments" string, even broken ones).
                    "raw_args": entry["args_buffer"],
                }
            )
            continue

        tool_calls.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "tool_timeout": args.pop("_venus_timeout", None),
                "tool_comment": args.pop("_tool_comment", None),
                "arguments": args,
                "error": None,
                "content": None,
            }
        )

    return tool_calls

