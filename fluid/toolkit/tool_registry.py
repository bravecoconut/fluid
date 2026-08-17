"""
tool_registry.py
-----------------
This file gives you two things:

1. A `@tool` decorator you stick on any function you want to be discoverable.
2. A `discover_tools()` function that walks a directory (including subfolders),
   imports every .py file it finds, and collects metadata about every
   function that was decorated with @tool.

No external libraries needed — just Python's built-in `os`, `importlib`,
and `inspect` modules.
"""

import os
import importlib.util
import inspect
from typing import get_type_hints

# This list holds every function that gets decorated with @tool,
# no matter which file it lives in. It fills up as files get imported.
TOOL_REGISTRY = []


def tool(skill=None, usually_takes=None):
    """
    Decorator: mark a function as a "tool" and register it.

    Usage:
        @tool(skill="math", usually_takes="5")
        def add(a: int, b: int) -> int:
            '''Add two numbers together.'''
            return a + b

    `skill` is just a free-text label you can use to group tools later
    (e.g. "math", "files", "search"). It's optional.

    `usually_takes` is time, that function usually takes.
    """
    def decorator(func):
        func._is_tool = True      # tag the function so we can recognize it
        func._skill = skill       # remember which skill/group it belongs to
        func._usually_takes = usually_takes
        TOOL_REGISTRY.append(func)
        return func
    return decorator


# Folder names to never walk into. These are either huge (virtual envs,
# node_modules), machine-generated (__pycache__), or full of third-party
# code that isn't safe to import as a bare file (it can break relative
# imports, or in rare cases — like numpy's f2py — even run code as a
# side effect of being imported).
EXCLUDED_DIRS = {
    ".venv", "venv", "env", ".env",
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", "site-packages",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".tox",
}


def _find_python_files(root_dir):
    """Walk root_dir (and all subfolders) and return every .py file path,
    skipping virtual envs, caches, and other non-project directories."""
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modifying dirnames in place tells os.walk not to descend into
        # these folders at all — so it's fast, not just filtered after.
        # This also skips ALL hidden folders (anything starting with "."),
        # not just the ones named in EXCLUDED_DIRS.
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            if filename.endswith(".py") and filename != "tool_registry.py":
                py_files.append(os.path.join(dirpath, filename))
    return py_files


def _import_module_from_path(file_path): # read
    """
    Import a .py file by its file path (not by package name).
    This is what lets us load files scattered in random subfolders,
    without needing them to be a proper installed package.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        # Actually running the file's top-level code is what triggers
        # the @tool decorators inside it, adding them to TOOL_REGISTRY.
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"⚠️  Skipping {file_path}: could not import it ({e})")
        return None
    return module


def _describe_tool(func):
    """Turn one decorated function into a plain dictionary of metadata."""
    sig = inspect.signature(func)

    # get_type_hints resolves annotations like `a: int` into real types.
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    parameters = []
    for name, param in sig.parameters.items():
        parameters.append({
            "name": name,
            "type": f"{hints.get(name, "Any")}",
            "required": param.default is inspect.Parameter.empty,
            "default": None if param.default is inspect.Parameter.empty else param.default,
        })

    return {
        "name": func.__name__,
        "skill": getattr(func, "_skill", None),
        "usually_takes": getattr(func, "_usually_takes", None),
        "description": inspect.getdoc(func) or "No description provided.",
        "parameters": parameters,
        "return_type": hints.get("return", "Any"),
        "module": func.__module__,
        "source_file": inspect.getfile(func),
    }


def discover_tools(root_dir=None):
    """
    The main entry point.

    Scans root_dir for .py files, imports each one (which triggers any
    @tool decorators inside them), and returns a list of metadata
    dictionaries — one per discovered tool.

    If root_dir is not given, it defaults to the current working directory —
    i.e. whatever folder you were standing in when you ran `python3 ...`.
    This is deliberately NOT the folder that contains the calling file,
    because that file might live nested inside a subpackage (e.g.
    `ignore/g.py`) while you still want the scan to start at the project
    root you launched python from (e.g. `venus/`).
    """
    if root_dir is None:
        root_dir = os.getcwd()

    TOOL_REGISTRY.clear()  # reset, in case this runs more than once

    for file_path in _find_python_files(root_dir):
        _import_module_from_path(file_path)

    return [_describe_tool(func) for func in TOOL_REGISTRY]