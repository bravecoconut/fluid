from fluid.generation.openai import OpenAICom
from fluid.regular_interface.build_system_messages._main import BuildSystemMessages
from fluid.vector_retrive.retriveing_pool import retriving_pool
from fluid.toolkit.tool_registry import discover_tools
from fluid.regular_interface.srcf import SRCF
from fluid.toolkit.tool_runner import start_process
from multiprocessing import Manager, Process
import traceback
import time
import os
import json
import logging
from fluid.utill import (
    filter_tools_by_default_skills,
    _filter_default_and_non_default_skills,
    json_ser,
    _pop_tools_for_skill,
    _build_logger,
    _find_skill_index,
    _tool_matches_skill,
    convert_to_openai_tools,
    return_completed_tool_calls,
)
from fluid.regular_interface.system_tools.agent_inbuild_tools import (
    SKILL_MANIPULATION_TOOLS,
    TOOL_FOR_TOOL,
)
from fluid.regular_interface.build_system_messages.system_message import (
    SYSTEM_MESSAGE,
    TOOL_INPUT_REQ,
    TOOL_UPDATE,
)

LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "interface.log")
logger = _build_logger(
    log_dir=LOG_DIR,
    log_file=LOG_FILE,
)


class RegularInterface:
    def __init__(
        self,
        messages,
        skills=[],
        max_new_skill=2,
        sys_messages=[],
        srcf=None,  # pass SRCF()
        all_tools=[],
        max_turns=20,
    ):
        self.messages = messages
        self.skills = skills
        self.max_new_skill = max_new_skill
        self.sys_messages = sys_messages
        self.srcf = srcf
        self.all_tools = all_tools
        self.max_turns = max_turns

        self.agent_history = []

        self.system_messages = []
        self.agent_messages = []
        self.openai_agent_tools = []

        self.agent_skills = []
        self.agent_tools = []
        self.default_skills = []
        self.non_default_skills = []
        self.on_tools = []
        self.off_tools = []
        self.raw_contexts = None
        self.contexts = None
        self.n_ds_and_ot_sys_m = None
        self.ds_and_c_sys_m = None

        self.relevent_messages = None
        self.leftover_messages = None

        if self.all_tools and not skills:
            self.agent_tools += self.all_tools

        if skills:
            self.default_skills, self.non_default_skills = (
                _filter_default_and_non_default_skills(locations=skills)
            )

            if SKILL_MANIPULATION_TOOLS:
                self.openai_agent_tools += SKILL_MANIPULATION_TOOLS

        if self.default_skills and messages:
            self.raw_contexts, self.contexts = retriving_pool(
                query=messages[-1]["content"],
                locations=self.default_skills,
            )

        self.on_tools, self.off_tools = filter_tools_by_default_skills(
            default_skills=self.default_skills,
            all_tools=self.all_tools,
        )

        if self.default_skills and self.on_tools:
            self.agent_skills += list(self.default_skills)
            self.agent_tools += list(self.on_tools)

        if (self.default_skills and self.contexts) or (
            self.non_default_skills and self.off_tools
        ):
            self.n_ds_and_ot_sys_m, self.ds_and_c_sys_m = BuildSystemMessages(
                default_skills=self.default_skills,
                non_default_skills=self.non_default_skills,
                grouped_contexts=self.contexts,
                off_tools=self.off_tools,
            ).format_them()

        if self.srcf and messages:
            self.relevent_messages, self.leftover_messages = self.srcf.srcf_go(
                messages=messages
            )

        if self.sys_messages:
            for m in self.sys_messages:
                self.system_messages.append({"role": "system", "content": m})

        if self.srcf:
            self.agent_messages += [{"role": "system", "content": SYSTEM_MESSAGE}]

        if self.n_ds_and_ot_sys_m:
            self.system_messages.append(
                {"role": "system", "content": self.n_ds_and_ot_sys_m}
            )

        if self.ds_and_c_sys_m:
            self.system_messages.append(
                {"role": "system", "content": self.ds_and_c_sys_m}
            )

        self.agent_messages += self.system_messages
        self.openai_agent_tools += convert_to_openai_tools(
            raw_tools_list=self.agent_tools
        )

        if self.srcf:
            self.agent_messages += self.relevent_messages + self.leftover_messages

        else:
            self.agent_messages += messages

        self._openai = OpenAICom()

        self.content_buffer = ""
        self.reasoning_buffer = ""
        self.tool_calls_buffer = []  # list of tool_call delta dicts from the stream

        self.in_q = None
        self.out_q = None
        self.update_q = None

        self.next_input = None

        # Tracks every subprocess spawned by tool calls: {pid: subprocess.Popen}
        # Used by stop_all() and terminate_process() to kill running tools.
        self._active_processes = {}

        # When set to True, the agent loop exits at the next safe point.
        self._stopped = False

    # ------------------------------------------------------------------
    # Names of tools that are handled directly by calling methods on
    # this class, NOT by spawning a subprocess. If the LLM calls one
    # of these, we run `self.<method>(...)` and return the result string
    # immediately — no child process, no timeout logic.
    # ------------------------------------------------------------------
    _INLINE_TOOLS = {"add_new_skill", "remove_skill", "switch_skill", "tool_run_tool"}

    # Default timeout (seconds) when the model doesn't specify one.
    _DEFAULT_TIMEOUT = 30

    # Clamp range for tool timeouts — prevents the model from setting
    # absurdly short (1s) or absurdly long (1h) timeouts.
    _MIN_TIMEOUT = 5
    _MAX_TIMEOUT = 300

    # How many times to retry the LLM call when it fails to call
    # tool_run_tool during a timeout decision.
    _TIMEOUT_DECISION_MAX_RETRIES = 3

    @staticmethod
    def _safe_timeout(raw_value, default=30):
        """
        Robustly convert whatever the model sent for _venus_timeout into
        a usable integer, clamped between _MIN_TIMEOUT and _MAX_TIMEOUT.

        Handles: int, float, str ("10", "10.0"), None, garbage strings.
        Returns `default` if conversion fails entirely.
        """
        if raw_value is None:
            return default
        try:
            value = int(float(raw_value))
        except (ValueError, TypeError):
            return default
        if value <= 0:
            return default
        # Clamp to safe range.
        return max(5, min(value, 300))

    def run_agent(
        self,
        base_url,
        api_key,
        model,
        tool_choice="auto",
        timeout=None,
        max_retires=5,
        http_client=None,
        default_headers=None,
        max_complition_tokens=3000,
        temperature=None,
    ):
        """
        The main agent loop.  Yields updates one at a time so the caller
        can print / stream / log them however they want.

        Every yielded value is a dict with this shape:

            {
                "type":      str,           # see below
                "content":   str | None,    # text token
                "reasoning": str | None,    # reasoning token
                "tool_call": { ... },       # tool info (see below)
                "turn":      int,           # current turn (1-indexed)
            }

        Possible "type" values:
            "content"      - a text token from the model
            "reasoning"    - a reasoning/thinking token
            "tool_start"   - a tool execution just started
            "tool_update"  - live stdout/stderr from a running tool
            "tool_timeout" - a tool's timeout expired, agent is deciding
            "tool_done"    - a tool finished (result is in tool_call.result/stdout)
            "tool_error"   - a tool had a JSON-parse error or crashed
            "error"        - the LLM stream itself errored
            "done"         - the model said stop, agent loop is over
        """

        logger.info(
            f"run_agent: starting | model={model!r} max_turns={self.max_turns} "
            f"tools={len(self.openai_agent_tools)} skills={len(self.agent_skills)}"
        )

        # ---- helper: build a clean yield dict ----
        def _make_event(
            event_type,
            turn,
            content=None,
            reasoning=None,
            tool_call_id=None,
            process_id=None,
            tool_name=None,
            tool_args=None,
            tool_comment=None,
            tool_timeout=None,
            stdout=None,
            stderr=None,
            result=None,
            status=None,
        ):
            return {
                "type": event_type,
                "content": content,
                "reasoning": reasoning,
                "tool_call": {
                    "tool_call_id": tool_call_id,
                    "process_id": process_id,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_comment": tool_comment,
                    "timeout": tool_timeout,
                    "stdout": stdout,
                    "stderr": stderr,
                    "result": result,
                    "status": status,
                },
                "turn": turn,
            }

        # ---- agent loop: call LLM → process response → run tools → repeat ----
        for turn in range(1, self.max_turns + 1):

            # Check if stop was requested between turns.
            if self._stopped:
                logger.info(f"run_agent: stopped by user at turn {turn}")
                yield _make_event("done", turn, content="Agent stopped by user.")
                return

            # Reset per-turn buffers.
            self.tool_calls_buffer = []
            self.content_buffer = ""
            self.reasoning_buffer = ""

            logger.info(
                f"run_agent: turn {turn}/{self.max_turns} | "
                f"messages={len(self.agent_messages)}"
            )

            # ---- 1. call the LLM ----
            try:
                openai_stream = self._openai.openai(
                    base_url=base_url,
                    api_key=api_key,
                    messages=self.agent_messages,
                    tools=self.openai_agent_tools if self.openai_agent_tools else None,
                    model=model,
                    tool_choice=tool_choice,
                    timeout=timeout,
                    max_retries=max_retires,
                    http_client=http_client,
                    default_headers=default_headers,
                    max_completion_tokens=max_complition_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(
                    f"run_agent: LLM call failed at turn {turn}: {e}",
                    exc_info=True,
                )
                yield _make_event("error", turn, content=f"LLM call failed: {e}")
                return

            finish_reason = None

            # ---- 2. stream chunks from LLM ----
            try:
                for chunk in openai_stream:

                    # The stream can yield an error dict (no "chunk" key)
                    # when the API call itself fails.
                    if chunk.get("error"):
                        logger.error(
                            f"run_agent: stream error at turn {turn}: "
                            f"{chunk['error']}"
                        )
                        yield _make_event("error", turn, content=chunk["error"])
                        return

                    raw = chunk["chunk"]

                    # raw can be a Pydantic model or already a dict
                    my_chunk = (
                        raw.model_dump() if hasattr(raw, "model_dump") else raw
                    )
                    choice = my_chunk["choices"][0]
                    delta = choice.get("delta") or {}

                    # --- content token ---
                    if delta.get("content"):
                        self.content_buffer += delta["content"]
                        yield _make_event(
                            "content", turn, content=delta["content"]
                        )

                    # --- reasoning token ---
                    if delta.get("reasoning"):
                        self.reasoning_buffer += delta["reasoning"]
                        yield _make_event(
                            "reasoning", turn, reasoning=delta["reasoning"]
                        )

                    # --- tool_call deltas (accumulate, don't yield yet) ---
                    if delta.get("tool_calls"):
                        self.tool_calls_buffer.extend(delta["tool_calls"])

                    # --- finish_reason lives on choice, NOT on delta ---
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

            except Exception as e:
                logger.error(
                    f"run_agent: exception while streaming at turn {turn}: {e}",
                    exc_info=True,
                )
                yield _make_event(
                    "error", turn,
                    content=f"Stream processing failed: {e}",
                )
                return

            logger.info(
                f"run_agent: turn {turn} stream done | "
                f"finish_reason={finish_reason!r} "
                f"content_len={len(self.content_buffer)} "
                f"tool_calls={len(self.tool_calls_buffer)}"
            )

            # ---- 3. decide what to do based on finish_reason ----

            if finish_reason == "stop" or finish_reason is None:
                # Model is done talking — exit the agent loop.
                logger.info(f"run_agent: model finished at turn {turn}")
                yield _make_event("done", turn)
                return

            if finish_reason == "length":
                # Model ran out of tokens mid-generation. Append whatever
                # partial content it produced as an assistant message so
                # the context is preserved, then continue to the next turn
                # so the model can keep going.
                logger.warning(
                    f"run_agent: turn {turn} hit token limit "
                    f"(finish_reason='length'), continuing"
                )
                if self.content_buffer:
                    self.agent_messages.append(
                        {"role": "assistant", "content": self.content_buffer}
                    )
                continue

            if finish_reason == "tool_calls":
                # Parse the accumulated tool_call deltas into complete calls.
                comp_tool_calls = return_completed_tool_calls(
                    tool_calls_buffer=self.tool_calls_buffer,
                )

                logger.info(
                    f"run_agent: turn {turn} has {len(comp_tool_calls)} "
                    f"tool call(s): "
                    f"{[tc['name'] for tc in comp_tool_calls]}"
                )

                # Build the assistant message that contains the tool_calls,
                # exactly how OpenAI expects it in the next request.
                #
                # IMPORTANT: we include ALL tool calls here — even ones
                # with _args_error. OpenAI requires every "role: tool"
                # response to have a matching tool_call in the assistant
                # message. If we skip errored calls, the API rejects them.
                assistant_tool_calls_msg = []
                for tc in comp_tool_calls:
                    # For errored calls, we still need them in the
                    # assistant message. Use raw args_buffer as arguments.
                    if tc["error"]:
                        assistant_tool_calls_msg.append(
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc.get("raw_args", "{}"),
                                },
                            }
                        )
                    else:
                        assistant_tool_calls_msg.append(
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"]),
                                },
                            }
                        )

                # Append the assistant message with tool_calls.
                if assistant_tool_calls_msg:
                    self.agent_messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                self.content_buffer
                                if self.content_buffer
                                else None
                            ),
                            "tool_calls": assistant_tool_calls_msg,
                        }
                    )

                # ---- 4. execute each tool call ----
                for tool_call in comp_tool_calls:
                    tc_id = tool_call["id"]
                    tc_name = tool_call["name"]
                    tc_timeout = tool_call["tool_timeout"]
                    tc_comment = tool_call["tool_comment"]
                    tc_args = tool_call["arguments"]
                    tc_error = tool_call["error"]
                    tc_content = tool_call["content"]

                    # --- 4a. JSON parse error → tell the model to retry ---
                    if tc_error == "_args_error":
                        logger.warning(
                            f"run_agent: tool '{tc_name}' (id={tc_id}) has "
                            f"malformed JSON arguments, asking model to fix"
                        )
                        yield _make_event(
                            "tool_error",
                            turn,
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                            result=tc_content,
                            status="error",
                        )
                        self.agent_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": tc_content,
                            }
                        )
                        continue

                    # --- 4b. inline tools (skill manipulation + tool_run_tool) ---
                    if tc_name in self._INLINE_TOOLS:
                        logger.info(
                            f"run_agent: executing inline tool "
                            f"'{tc_name}' (id={tc_id})"
                        )
                        # Yield start so the caller sees every tool's lifecycle.
                        yield _make_event(
                            "tool_start",
                            turn,
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                            tool_args=tc_args,
                            tool_comment=tc_comment,
                            status="started",
                        )

                        result_str = self._handle_inline_tool(
                            tc_name, tc_args or {}
                        )

                        logger.info(
                            f"run_agent: inline tool '{tc_name}' done | "
                            f"result={str(result_str)[:100]}"
                        )

                        yield _make_event(
                            "tool_done",
                            turn,
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                            tool_args=tc_args,
                            tool_comment=tc_comment,
                            result=result_str,
                            status="finished",
                        )
                        self.agent_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": str(result_str),
                            }
                        )

                        # After inline skill changes, rebuild the openai
                        # tools list so the next LLM call sees updated set.
                        if tc_name in (
                            "add_new_skill",
                            "remove_skill",
                            "switch_skill",
                        ):
                            self.openai_agent_tools = convert_to_openai_tools(
                                raw_tools_list=self.agent_tools
                            )
                            # Re-add the skill manipulation tools.
                            if self.skills and SKILL_MANIPULATION_TOOLS:
                                self.openai_agent_tools = (
                                    list(SKILL_MANIPULATION_TOOLS)
                                    + self.openai_agent_tools
                                )
                            logger.info(
                                f"run_agent: rebuilt tool list after "
                                f"'{tc_name}' | "
                                f"total_tools="
                                f"{len(self.openai_agent_tools)}"
                            )
                        continue

                    # --- 4c. external tool → subprocess via _worker ---
                    # Robustly coerce the timeout the model sent.
                    tc_timeout = self._safe_timeout(
                        tc_timeout, default=self._DEFAULT_TIMEOUT
                    )

                    logger.info(
                        f"run_agent: spawning external tool '{tc_name}' "
                        f"(id={tc_id}) | timeout={tc_timeout}s "
                        f"args={tc_args}"
                    )

                    # Yield a "started" event before spawning.
                    yield _make_event(
                        "tool_start",
                        turn,
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                        tool_args=tc_args,
                        tool_comment=tc_comment,
                        tool_timeout=tc_timeout,
                        status="started",
                    )

                    # Spawn the worker in a child process.
                    final_stdout = ""
                    final_stderr = ""

                    try:
                        with Manager() as manager:
                            self.in_q = manager.Queue()
                            self.out_q = manager.Queue()
                            self.update_q = manager.Queue()

                            worker = Process(
                                target=self._worker,
                                args=(
                                    tc_id,
                                    tc_name,
                                    tc_args,
                                    self.all_tools,
                                    tc_timeout,
                                    self.in_q,
                                    self.out_q,
                                    self.update_q,
                                ),
                            )
                            worker.start()
                            logger.info(
                                f"run_agent: worker process started "
                                f"for '{tc_name}' (worker_pid="
                                f"{worker.pid})"
                            )

                            # Read updates from the worker until it sends
                            # None (done).
                            while True:
                                update = self.update_q.get()
                                if update is None:
                                    break

                                proc = update.get("process") or {}
                                u_pid = proc.get("pid")
                                u_stdout = proc.get("stdout")
                                u_stderr = proc.get("stderr")

                                # Keep track of the latest stdout/stderr.
                                if u_stdout is not None:
                                    final_stdout = u_stdout
                                if u_stderr is not None:
                                    final_stderr = u_stderr

                                # Yield a live update to the caller.
                                yield _make_event(
                                    "tool_update",
                                    turn,
                                    tool_call_id=tc_id,
                                    process_id=u_pid,
                                    tool_name=tc_name,
                                    tool_args=tc_args,
                                    tool_comment=tc_comment,
                                    tool_timeout=tc_timeout,
                                    stdout=u_stdout,
                                    stderr=u_stderr,
                                    result=update.get("call_reason"),
                                    status="running",
                                )

                                # --- timeout decision ---
                                if update.get("awaiting_decision"):
                                    logger.info(
                                        f"run_agent: timeout hit for "
                                        f"'{tc_name}' (pid={u_pid}), "
                                        f"requesting decision"
                                    )
                                    msg_type, task_id, prompt = (
                                        self.in_q.get()
                                    )
                                    if msg_type != "need_input":
                                        logger.error(
                                            f"run_agent: unexpected msg_type "
                                            f"from worker: {msg_type!r}"
                                        )

                                    if self.next_input is not None:
                                        # User/agent already set a value
                                        # via tool_run_tool.
                                        decision = int(self.next_input)
                                        self.next_input = None
                                        logger.info(
                                            f"run_agent: using pre-set "
                                            f"decision={decision}"
                                        )
                                        self.out_q.put(decision)
                                    else:
                                        # Ask the LLM to decide: extend
                                        # or terminate?
                                        yield _make_event(
                                            "tool_timeout",
                                            turn,
                                            tool_call_id=tc_id,
                                            process_id=u_pid,
                                            tool_name=tc_name,
                                            tool_args=tc_args,
                                            tool_comment=tc_comment,
                                            tool_timeout=tc_timeout,
                                            stdout=u_stdout,
                                            stderr=u_stderr,
                                            result=(
                                                "Timeout expired, asking "
                                                "agent for decision."
                                            ),
                                            status="timeout",
                                        )

                                        yield from (
                                            self._ask_timeout_decision(
                                                base_url=base_url,
                                                api_key=api_key,
                                                model=model,
                                                turn=turn,
                                                tc_id=tc_id,
                                                tc_name=tc_name,
                                                u_pid=u_pid,
                                                u_stdout=u_stdout,
                                                u_stderr=u_stderr,
                                                _make_event=_make_event,
                                                timeout=timeout,
                                                max_retires=max_retires,
                                                http_client=http_client,
                                                default_headers=(
                                                    default_headers
                                                ),
                                                temperature=temperature,
                                            )
                                        )

                                        # _ask_timeout_decision sets
                                        # self.next_input
                                        decision = (
                                            int(self.next_input)
                                            if self.next_input is not None
                                            else 0
                                        )
                                        self.next_input = None
                                        logger.info(
                                            f"run_agent: timeout decision "
                                            f"for '{tc_name}': {decision}"
                                        )
                                        self.out_q.put(decision)

                            # Wait for the worker to finish. If it hangs,
                            # terminate it so we don't get zombie processes.
                            worker.join(timeout=10)
                            if worker.is_alive():
                                logger.warning(
                                    f"run_agent: worker for '{tc_name}' "
                                    f"didn't exit in 10s, terminating"
                                )
                                worker.terminate()
                                worker.join(timeout=5)
                                if worker.is_alive():
                                    logger.error(
                                        f"run_agent: worker for "
                                        f"'{tc_name}' still alive "
                                        f"after terminate, killing"
                                    )
                                    worker.kill()
                                    worker.join(timeout=3)

                    except Exception as e:
                        logger.error(
                            f"run_agent: exception in tool '{tc_name}' "
                            f"execution: {e}",
                            exc_info=True,
                        )
                        final_stderr = (
                            f"{final_stderr}\n[agent error: {e}]"
                            if final_stderr
                            else f"[agent error: {e}]"
                        )

                    # Build the tool result to feed back to the LLM.
                    tool_output = (
                        final_stdout or final_stderr or "(no output)"
                    )

                    logger.info(
                        f"run_agent: tool '{tc_name}' finished | "
                        f"output_len={len(tool_output)}"
                    )

                    yield _make_event(
                        "tool_done",
                        turn,
                        tool_call_id=tc_id,
                        process_id=None,
                        tool_name=tc_name,
                        tool_args=tc_args,
                        tool_comment=tc_comment,
                        tool_timeout=tc_timeout,
                        stdout=final_stdout,
                        stderr=final_stderr,
                        result=tool_output,
                        status="finished",
                    )

                    # Append the tool result so the LLM sees it next turn.
                    self.agent_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": tool_output,
                        }
                    )

                # After processing ALL tool calls for this turn, the for
                # loop continues to the next iteration → calls the LLM
                # again with updated messages that now include tool results.
                continue

            # Any other unknown finish_reason — log and treat as done.
            logger.warning(
                f"run_agent: unknown finish_reason={finish_reason!r} "
                f"at turn {turn}"
            )
            yield _make_event(
                "done",
                turn,
                content=f"Finished with reason: {finish_reason}",
            )
            return

        # Reached max_turns without the model saying "stop".
        logger.info(
            f"run_agent: reached max_turns ({self.max_turns}), stopping"
        )
        yield _make_event(
            "done",
            self.max_turns,
            content=f"Reached maximum turns ({self.max_turns}).",
        )

    # ------------------------------------------------------------------
    # Timeout decision — ask the LLM whether to extend or kill a tool
    # ------------------------------------------------------------------
    def _ask_timeout_decision(
        self,
        base_url,
        api_key,
        model,
        turn,
        tc_id,
        tc_name,
        u_pid,
        u_stdout,
        u_stderr,
        _make_event,
        timeout=None,
        max_retires=5,
        http_client=None,
        default_headers=None,
        temperature=None,
    ):
        """
        Makes a quick LLM call to ask the agent whether to extend the
        timeout or terminate a running tool.

        How it works:
          1. Adds a system message with TOOL_UPDATE (stdout/stderr/pid)
             + TOOL_INPUT_REQ so the agent sees everything transparently.
          2. Calls the LLM with only TOOL_FOR_TOOL (tool_run_tool) available.
          3. Streams the response, yielding content/reasoning tokens.
          4. Parses the tool_run_tool call and sets self.next_input.
          5. If the LLM doesn't call tool_run_tool, RETRIES up to
             _TIMEOUT_DECISION_MAX_RETRIES times with a stronger prompt.
          6. Only defaults to 0 (terminate) after all retries are exhausted.
        """
        logger.info(
            f"_ask_timeout_decision: starting for tool '{tc_name}' "
            f"(id={tc_id}, pid={u_pid})"
        )

        # Give the agent the same TOOL_UPDATE it normally sees, plus the
        # timeout-decision prompt — full transparency.
        timeout_prompt = TOOL_UPDATE.format(
            tool_id=tc_id,
            tool_name=tc_name,
            pid=u_pid,
            c_outputs=u_stdout or "(none yet)",
            error=u_stderr or "(none)",
        )
        timeout_prompt += "\n" + TOOL_INPUT_REQ.format(
            tool_name=tc_name,
            tool_id=tc_id,
        )

        # Retry loop — re-call the LLM if it doesn't use tool_run_tool.
        for attempt in range(1, self._TIMEOUT_DECISION_MAX_RETRIES + 1):

            logger.info(
                f"_ask_timeout_decision: attempt {attempt}/"
                f"{self._TIMEOUT_DECISION_MAX_RETRIES} "
                f"for '{tc_name}'"
            )

            # Build messages for this attempt. On retries, add a nudge
            # telling the model it MUST call the tool.
            decision_messages = list(self.agent_messages) + [
                {"role": "system", "content": timeout_prompt}
            ]

            if attempt > 1:
                # Stronger nudge on retry — the model didn't call
                # tool_run_tool last time.
                decision_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You MUST call tool_run_tool now. "
                            "Pass next_timeout_window=0 to terminate, "
                            "or a positive number (e.g. 15) to extend. "
                            "Do NOT reply with text — call the tool."
                        ),
                    }
                )

            # Quick LLM call with only tool_run_tool available.
            try:
                decision_stream = self._openai.openai(
                    base_url=base_url,
                    api_key=api_key,
                    messages=decision_messages,
                    tools=TOOL_FOR_TOOL,
                    model=model,
                    tool_choice="auto",
                    timeout=timeout,
                    max_retries=max_retires,
                    http_client=http_client,
                    default_headers=default_headers,
                    max_completion_tokens=300,  # small — just a decision
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(
                    f"_ask_timeout_decision: LLM call failed on attempt "
                    f"{attempt}: {e}",
                    exc_info=True,
                )
                # On error, default to terminate.
                self.next_input = 0
                return

            decision_tool_calls = []
            got_tool_call = False

            try:
                for chunk in decision_stream:
                    if chunk.get("error"):
                        logger.error(
                            f"_ask_timeout_decision: stream error on "
                            f"attempt {attempt}: {chunk['error']}"
                        )
                        # On error, default to terminate.
                        self.next_input = 0
                        return

                    raw = chunk["chunk"]
                    my_chunk = (
                        raw.model_dump()
                        if hasattr(raw, "model_dump")
                        else raw
                    )
                    choice = my_chunk["choices"][0]
                    delta = choice.get("delta") or {}

                    # Stream content/reasoning to the caller so they
                    # see the agent thinking.
                    if delta.get("content"):
                        yield _make_event(
                            "content", turn, content=delta["content"]
                        )
                    if delta.get("reasoning"):
                        yield _make_event(
                            "reasoning", turn, reasoning=delta["reasoning"]
                        )

                    if delta.get("tool_calls"):
                        decision_tool_calls.extend(delta["tool_calls"])

                    # Check if the model finished with a tool call.
                    finish = choice.get("finish_reason")
                    if finish == "tool_calls" and decision_tool_calls:
                        comp = return_completed_tool_calls(
                            decision_tool_calls
                        )
                        for tc in comp:
                            if (
                                tc["name"] == "tool_run_tool"
                                and not tc["error"]
                            ):
                                val = tc["arguments"].get(
                                    "next_timeout_window", 0
                                )
                                try:
                                    self.next_input = int(float(val))
                                except (ValueError, TypeError):
                                    logger.warning(
                                        f"_ask_timeout_decision: bad value "
                                        f"for next_timeout_window: {val!r}, "
                                        f"defaulting to 0"
                                    )
                                    self.next_input = 0

                                logger.info(
                                    f"_ask_timeout_decision: got decision="
                                    f"{self.next_input} on attempt {attempt}"
                                )
                                got_tool_call = True
                                return

            except Exception as e:
                logger.error(
                    f"_ask_timeout_decision: exception streaming on "
                    f"attempt {attempt}: {e}",
                    exc_info=True,
                )
                self.next_input = 0
                return

            if got_tool_call:
                return

            # Model didn't call tool_run_tool — log and retry.
            logger.warning(
                f"_ask_timeout_decision: model did NOT call tool_run_tool "
                f"on attempt {attempt}, "
                f"{'retrying' if attempt < self._TIMEOUT_DECISION_MAX_RETRIES else 'giving up'}"
            )

        # All retries exhausted — default to terminate.
        logger.warning(
            f"_ask_timeout_decision: all {self._TIMEOUT_DECISION_MAX_RETRIES} "
            f"attempts exhausted for '{tc_name}', defaulting to terminate"
        )
        if self.next_input is None:
            self.next_input = 0

    # ------------------------------------------------------------------
    # Inline tool dispatcher
    # ------------------------------------------------------------------
    def _handle_inline_tool(self, tool_name, tool_args):
        """
        Calls one of the agent's built-in methods (add_new_skill, etc.)
        and returns its result as a string.  Never raises — bad tool
        names return an error string instead.
        """
        logger.info(
            f"_handle_inline_tool: dispatching '{tool_name}' "
            f"with args={tool_args}"
        )
        try:
            if tool_name == "add_new_skill":
                result = self.add_new_skill(tool_args.get("skill_name", ""))
            elif tool_name == "remove_skill":
                result = self.remove_skill(tool_args.get("skill_name", ""))
            elif tool_name == "switch_skill":
                result = self.switch_skill(
                    tool_args.get("skill_name", ""),
                    tool_args.get("to_which", ""),
                )
            elif tool_name == "tool_run_tool":
                result = self.tool_run_tool(
                    tool_args.get("next_timeout_window", 0),
                )
            else:
                result = f"Unknown inline tool: {tool_name}"
                logger.warning(
                    f"_handle_inline_tool: unknown tool '{tool_name}'"
                )
            return result
        except Exception as e:
            logger.error(
                f"_handle_inline_tool: error running '{tool_name}': {e}",
                exc_info=True,
            )
            return f"Error running {tool_name}: {e}"

    # ------------------------------------------------------------------
    # Worker (runs in child process)
    # ------------------------------------------------------------------
    @staticmethod
    def _worker(
        task_id, tool_name, tool_args, avail_tools, timeout,
        in_q, out_q, update_q,
    ):
        """Runs in a child process: drives the start_process generator to
        completion and forwards every yielded update to update_q. When the
        generator blocks on out_q.get() (waiting for a timeout decision),
        that block happens here, in the child - not in the main process.

        Always sends None as a final sentinel so the parent process never
        hangs waiting on update_q.get().
        """
        try:
            gen = start_process(
                task_id=task_id,
                tool_name=tool_name,
                tool_args=tool_args,
                avail_tools=avail_tools,
                timeout=timeout,
                in_q=in_q,
                out_q=out_q,
            )
            for update in gen:
                update_q.put(update)
        except Exception as e:
            # Send the error as a final update so the parent knows what
            # went wrong, rather than silently dying.
            update_q.put(
                {
                    "process": {"pid": None, "stdout": None, "stderr": None},
                    "call_reason": f"WORKER ERROR: {e}",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                }
            )
        finally:
            # Always send sentinel — even on crash — so the parent's
            # `while True: update = update_q.get()` never hangs.
            update_q.put(None)

    # ------------------------------------------------------------------
    # tool_run_tool — called by the LLM to set the next timeout or kill
    # ------------------------------------------------------------------
    def tool_run_tool(self, next_timeout_window: int):
        """
        Sets the next timeout window for a running external tool.
        Pass 0 to terminate the tool immediately.

        Returns a confirmation string (fed back to the LLM as the tool result).
        """
        logger.info(f"tool_run_tool: next_timeout_window={next_timeout_window}")
        self.next_input = next_timeout_window
        if next_timeout_window <= 0:
            return "Tool will be terminated."
        return f"Timeout extended by {next_timeout_window} seconds."

    # ------------------------------------------------------------------
    # stop_all — kill everything: stream + all running subprocesses
    # ------------------------------------------------------------------
    def stop_all(self):
        """
        Immediately stops the agent:
          1. Sets the _stopped flag so the agent loop exits.
          2. Signals the OpenAI stream to stop.
          3. Terminates every tracked subprocess (tools).

        Safe to call from any thread.
        """
        self._stopped = True
        self._openai.stop_event.set()

        for pid, proc in list(self._active_processes.items()):
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            self._active_processes.pop(pid, None)

    # ------------------------------------------------------------------
    # terminate_process — kill one specific subprocess by PID
    # ------------------------------------------------------------------
    def terminate_process(self, pid: int):
        """
        Terminates a single running tool process by its PID.
        Pass timeout=0 in tool_run_tool to let the agent itself kill a tool,
        or call this directly from the outside.

        Returns a status string.
        """
        proc = self._active_processes.get(pid)
        if proc is None:
            return f"No tracked process with PID {pid}."

        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

        self._active_processes.pop(pid, None)
        return f"Process {pid} terminated."

    def _debug_agent(self):
        # print("*" * 20, "messages", "*" * 20)
        # print(json.dumps(self.messages, indent=4, default=json_ser))

        # print("*" * 20, "default skills", "*" * 20)
        # print(json.dumps(self.default_skills, indent=4, default=json_ser))

        # print("*" * 20, "non default skills", "*" * 20)
        # print(json.dumps(self.non_default_skills, indent=4, default=json_ser))

        print("*" * 20, "raw contexts", "*" * 20)
        print(json.dumps(self.raw_contexts, indent=4, default=json_ser))

        print("*" * 20, "grouped contexts (with templates)", "*" * 20)
        print(json.dumps(self.contexts, indent=4, default=json_ser))

        # print("*" * 20, "off tools", "*" * 20)
        # print(json.dumps(self.off_tools, indent=4, default=json_ser))

        # print("*" * 20, "on tools", "*" * 20)
        # print(json.dumps(self.on_tools, indent=4, default=json_ser))

        # print("*" * 20, "non default skills and their tools sys m", "*" * 20)
        # print(self.n_ds_and_ot_sys_m)

        # print("*" * 20, "default skills and contexts sys m", "*" * 20)
        # print(self.ds_and_c_sys_m)

        # print("*" * 20, "agent tools", "*" * 20)
        # print(json.dumps(self.agent_tools, indent=4, default=json_ser))

        # print("*" * 20, "agent skills", "*" * 20)
        # print(json.dumps(self.agent_skills, indent=4, default=json_ser))

        print("*" * 20, "relevent messages", "*" * 20)
        print(json.dumps(self.relevent_messages, indent=4, default=json_ser))

        print("*" * 20, "leftover messages", "*" * 20)
        print(json.dumps(self.leftover_messages, indent=4, default=json_ser))

        print("*" * 20, "system messages", "*" * 20)
        print(json.dumps(self.system_messages, indent=4, default=json_ser))

        print("*" * 20, "agent messages", "*" * 20)
        print(json.dumps(self.agent_messages, indent=4, default=json_ser))

        print("*" * 20, "openai agent tools", "*" * 20)
        print(json.dumps(self.openai_agent_tools, indent=4, default=json_ser))

    def add_new_skill(self, skill_name):
        """
        Adds a non-default skill (and its tools) to the agent's active set.
        Always returns a single string message -- never raises, so a bad
        LLM tool call can't crash the agent loop.
        """
        if not isinstance(skill_name, str) or not skill_name:
            return "skill_name must be a non-empty string."

        # check: is this already a default skill? Agent has permanent access already.
        if _find_skill_index(self.default_skills, skill_name) is not None:
            return (
                f"'{skill_name}' is a default skill -- you already have permanent "
                "access to it and its tools. No action needed."
            )

        # check: is it already active from an earlier add_new_skill call?
        if _find_skill_index(self.agent_skills, skill_name) is not None:
            return f"'{skill_name}' is already active. No action needed."

        # check: does this skill actually exist in the available catalog?
        catalog_index = _find_skill_index(self.non_default_skills, skill_name)
        if catalog_index is None:
            return (
                f"'{skill_name}' was not found in any available skill. "
                "Check the spelling and try again."
            )

        # check: would this exceed the max-new-skills limit?
        # NOTE: this is a behavior fix vs. the original -- it now uses ">="
        # instead of ">", closing an off-by-one that let one extra skill
        # slip in past the stated limit.
        active_non_default = [
            s
            for s in self.agent_skills
            if isinstance(s, dict) and not s.get("default", True)
        ]
        if len(active_non_default) >= self.max_new_skill:
            active_names = ", ".join(s.get("name", "?") for s in active_non_default)
            return (
                f"Can't add '{skill_name}' -- you're at your limit of "
                f"{self.max_new_skill} extra skill(s): {active_names}. "
                "Try switch_skill to swap one out instead."
            )

        # all checks passed -- add the skill and move ALL of its tools
        # (not just the first one) from off_tools into on_tools/agent_tools
        new_skill = self.non_default_skills[catalog_index]
        self.agent_skills.append(new_skill)

        moved_tools = _pop_tools_for_skill(skill_name, self.off_tools)
        self.on_tools.extend(moved_tools)
        self.agent_tools.extend(moved_tools)

        logger.info(
            f"add_new_skill: added {skill_name!r} ({len(moved_tools)} tool(s) activated)"
        )

        return (
            f"'{skill_name}' and its tools are now active. Use them whenever "
            "they're the right fit for the task."
        )

    def remove_skill(self, skill_name):
        """
        Removes an active non-default skill AND moves its tools back to
        off_tools, so tool access is actually revoked, not just the
        skill entry.
        """
        if not isinstance(skill_name, str) or not skill_name:
            return "skill_name must be a non-empty string."

        if _find_skill_index(self.default_skills, skill_name) is not None:
            return (
                f"'{skill_name}' is a default skill -- it can't be removed. "
                "Default skills define your core identity and purpose."
            )

        index = _find_skill_index(self.agent_skills, skill_name)
        if index is None:
            return (
                f"'{skill_name}' is not currently active, so there's nothing to remove."
            )

        del self.agent_skills[index]

        moved_tools = _pop_tools_for_skill(skill_name, self.agent_tools)
        _pop_tools_for_skill(
            skill_name, self.on_tools
        )  # keep on_tools in sync, discard its own copy
        self.off_tools.extend(moved_tools)

        logger.info(
            f"remove_skill: removed {skill_name!r} ({len(moved_tools)} tool(s) deactivated)"
        )

        return (
            f"'{skill_name}' and its tools have been removed from your active access."
        )

    def switch_skill(self, skill_name, to_which):
        """
        Removes `skill_name` and adds `to_which` in one call -- a proper
        swap. (The previous version validated `to_which` but never
        actually looked it up or added it; this version fixes that.)
        """
        if not isinstance(skill_name, str) or not skill_name:
            return "skill_name must be a non-empty string."
        if not isinstance(to_which, str) or not to_which:
            return "to_which must be a non-empty string."
        if skill_name == to_which:
            return (
                f"'{skill_name}' and to_which are the same skill -- nothing to switch."
            )

        if _find_skill_index(self.default_skills, skill_name) is not None:
            return f"'{skill_name}' is a default skill -- it can't be switched out."

        old_index = _find_skill_index(self.agent_skills, skill_name)
        if old_index is None:
            return f"'{skill_name}' is not currently active, so there's nothing to switch out."

        if _find_skill_index(self.default_skills, to_which) is not None:
            return f"'{to_which}' is already a default skill -- you already have permanent access to it."

        if _find_skill_index(self.agent_skills, to_which) is not None:
            return f"'{to_which}' is already active -- nothing to switch to."

        catalog_index = _find_skill_index(self.non_default_skills, to_which)
        if catalog_index is None:
            return (
                f"'{to_which}' was not found in any available skill. "
                "Check the spelling and try again."
            )

        new_skill = self.non_default_skills[catalog_index]

        # remove the old skill + move its tools out
        del self.agent_skills[old_index]
        old_tools = _pop_tools_for_skill(skill_name, self.agent_tools)
        _pop_tools_for_skill(skill_name, self.on_tools)
        self.off_tools.extend(old_tools)

        # add the new skill + move its tools in
        self.agent_skills.append(new_skill)
        new_tools = _pop_tools_for_skill(to_which, self.off_tools)
        self.on_tools.extend(new_tools)
        self.agent_tools.extend(new_tools)

        logger.info(
            f"switch_skill: {skill_name!r} -> {to_which!r} ({len(new_tools)} tool(s) activated)"
        )

        return (
            f"Switched '{skill_name}' out for '{to_which}'. Its tools are now active."
        )
