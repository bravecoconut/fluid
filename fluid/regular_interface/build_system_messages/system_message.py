# fluid/regular_interface/build_system_messages/system_message.py

SYSTEM_MESSAGE = """
## Notice: Pruned Conversation History

Some messages in this conversation may appear to be missing or out of
sequence. This is expected -- older or low-signal messages are periodically
pruned to reduce noise, not lost.

Even when a message isn't directly visible, the key facts, decisions, and
context it contained have been preserved and are available to you through
the surrounding conversation. Do not assume a gap means missing
information -- treat the retained context as complete enough to reason
over, and proceed accordingly.

If you genuinely need a detail that seems unavailable, ask the user to
clarify rather than assuming it was never provided.
"""

TOOL_INPUT_REQ = """
## ACTION REQUIRED: Timeout Decision

Tool '{tool_name}' (id: {tool_id}) has exceeded its timeout window.
The process is still running and you must decide what to do.

You MUST call the `tool_run_tool` function with ONE of these options:
  - next_timeout_window = 0     → TERMINATE the tool immediately
  - next_timeout_window = N     → EXTEND the timeout by N more seconds (e.g. 15, 30, 60)

Choose based on the tool's current output:
  - If the output shows progress (data downloading, processing), EXTEND it.
  - If there's no output or only errors, TERMINATE it.
  - If you're unsure, extend by 15 seconds to check again.

DO NOT reply with text. You MUST call tool_run_tool now.
"""


TOOL_UPDATE = """
## Tool Status Update

Tool call: {tool_id}
Tool name: {tool_name}
Process ID: {pid}

### Current Output:
{c_outputs}

### Errors (if any):
{error}
"""