import textwrap
import subprocess
import threading
import time
import logging
import os

# Logger for tool_runner — writes to the same logs/ directory.
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "tool_runner.log")

def _setup_logger():
    """Set up a file logger for tool_runner (safe to call multiple times)."""
    _logger = logging.getLogger("tool_runner")
    if _logger.handlers:
        return _logger
    os.makedirs(LOG_DIR, exist_ok=True)
    _logger.setLevel(logging.INFO)
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(processName)s: %(message)s"
        )
    )
    _logger.addHandler(fh)
    return _logger

logger = _setup_logger()


def _pump(pipe, buffer, lock):
    try:
        for line in iter(pipe.readline, ""):
            with lock:
                buffer.append(line)
    finally:
        pipe.close()


def start_process(task_id, tool_name, tool_args, avail_tools, timeout, in_q, out_q):
    tool_params = ""
    process = None

    logger.info(
        f"start_process: starting tool '{tool_name}' | "
        f"task_id={task_id} timeout={timeout}s args={tool_args}"
    )

    try:
        module_name = source_file = None
        for tool in avail_tools:
            if tool["name"] == tool_name:
                module_name = tool["module"]
                source_file = tool["source_file"]

        if module_name is None or source_file is None:
            logger.error(
                f"start_process: tool '{tool_name}' not found in "
                f"{len(avail_tools)} available tools"
            )
            yield {
                "process": None,
                "call_reason": f"ERROR: tool '{tool_name}' not found",
                "tool_name": tool_name,
                "tool_args": tool_args,
            }
            return

        if tool_args:
            for arg_name, arg_value in tool_args.items():
                tool_params += f"{arg_name}={arg_value!r},"

        function_call = f"{tool_name}({tool_params})"

        logger.info(
            f"start_process: running {function_call} from "
            f"{source_file}"
        )

        command = textwrap.dedent(f"""
            import importlib.util, asyncio, inspect
            spec = importlib.util.spec_from_file_location({module_name!r}, {source_file!r})
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result = mod.{function_call}
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            print(result)
        """)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            ["python3", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        logger.info(
            f"start_process: spawned subprocess pid={process.pid} "
            f"for '{tool_name}'"
        )

        # live-capture buffers, filled by background threads
        stdout_lines, stderr_lines = [], []
        lock = threading.Lock()
        out_thread = threading.Thread(
            target=_pump, args=(process.stdout, stdout_lines, lock), daemon=True
        )
        err_thread = threading.Thread(
            target=_pump, args=(process.stderr, stderr_lines, lock), daemon=True
        )
        out_thread.start()
        err_thread.start()

        def snapshot():
            with lock:
                return "".join(stdout_lines), "".join(stderr_lines)

        start_time = time.time()
        deadline = start_time + timeout

        yield {
            "process": {"pid": process.pid},
            "call_reason": "task started, waiting for completion or timeout",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "stated_messages":True,
        }

        while True:
            if process.poll() is not None:
                out_thread.join(timeout=2)
                err_thread.join(timeout=2)
                stdout, stderr = snapshot()
                elapsed = time.time() - start_time
                logger.info(
                    f"start_process: '{tool_name}' (pid={process.pid}) "
                    f"finished in {elapsed:.1f}s | "
                    f"returncode={process.returncode}"
                )
                yield {
                    "process": {"pid": process.pid, "stdout": stdout, "stderr": stderr},
                    "call_reason": f"finished, total {elapsed:.1f}s",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                }
                return

            if time.time() >= deadline:
                stdout, stderr = snapshot()
                elapsed = time.time() - start_time
                logger.info(
                    f"start_process: '{tool_name}' (pid={process.pid}) "
                    f"timeout reached at {elapsed:.1f}s, "
                    f"awaiting decision"
                )
                yield {
                    "process": {"pid": process.pid, "stdout": stdout, "stderr": stderr},
                    "call_reason": f"timeout window reached "
                    f"(total {elapsed:.1f}s). "
                    f"send seconds to extend, or <=0 to terminate.",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "awaiting_decision": True,
                }
                in_q.put(("need_input", task_id, f"Enter value for task {task_id}: "))

                # Safely get and convert the decision value.
                raw_decision = out_q.get()
                try:
                    next_decision = int(float(raw_decision))
                except (ValueError, TypeError):
                    logger.warning(
                        f"start_process: bad decision value "
                        f"{raw_decision!r}, defaulting to 0 (terminate)"
                    )
                    next_decision = 0

                if next_decision is None or next_decision <= 0:
                    logger.info(
                        f"start_process: terminating '{tool_name}' "
                        f"(pid={process.pid})"
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"start_process: '{tool_name}' "
                            f"(pid={process.pid}) didn't terminate, "
                            f"killing"
                        )
                        process.kill()
                        process.wait()
                    out_thread.join(timeout=2)
                    err_thread.join(timeout=2)
                    stdout, stderr = snapshot()
                    yield {
                        "process": {
                            "pid": process.pid,
                            "stdout": stdout,
                            "stderr": stderr,
                        },
                        "call_reason": f"terminated after {time.time() - start_time:.1f}s",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                    }
                    return
                else:
                    logger.info(
                        f"start_process: extending '{tool_name}' "
                        f"(pid={process.pid}) by {next_decision}s"
                    )
                    deadline = time.time() + next_decision
            else:
                time.sleep(min(1, deadline - time.time()))

    except Exception as e:
        # Guard against process being None (e.g. Popen itself failed).
        pid = process.pid if process else None
        logger.error(
            f"start_process: exception running '{tool_name}' "
            f"(pid={pid}): {e}",
            exc_info=True,
        )
        yield {
            "process": {"pid": pid, "stdout": None, "stderr": None},
            "call_reason": f"ERROR running {tool_name}({tool_params}): {e}",
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

