# fluid/regular_interface/system_tools/agent_inbuild_tools.py

SKILL_MANIPULATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_new_skill",
            "description": "Add a non-default skill (and all of its tools) to the agent's active set. Fails gracefully with an explanatory message if the skill is already a default skill, already active, not found in the catalog, or if adding it would exceed the max-new-skills limit",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The exact name of the non-default skill to activate, as it appears in the skill catalog.",
                    },
                    "_tool_comment": {
                        "type": "string",
                        "description": "describe what you are doing or why you wants to run it [under 3-10 words].",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_skill",
            "description": "Remove an active non-default skill and revoke access to its tools by moving them back to the inactive tool set. Default skills cannot be removed. Fails gracefully with an explanatory message if the skill is a default skill or is not currently active.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The exact name of the currently active non-default skill to remove.",
                    },
                    "_tool_comment": {
                        "type": "string",
                        "description": "describe what you are doing or why you wants to run it [under 3-10 words].",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_skill",
            "description": "Swap one active non-default skill for another in a single call: removes skill_name and its tools, then adds to_which and its tools. Fails gracefully with an explanatory message if skill_name and to_which are the same, skill_name is a default skill or not currently active, to_which is already a default skill or already active, or to_which is not found in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The exact name of the currently active non-default skill to remove.",
                    },
                    "to_which": {
                        "type": "string",
                        "description": "The exact name of the non-default skill from the catalog to activate in its place.",
                    },
                    "_tool_comment": {
                        "type": "string",
                        "description": "describe what you are doing or why you wants to run it [under 3-10 words].",
                    },
                },
                "required": ["skill_name", "to_which"],
            },
        },
    },
]


TOOL_FOR_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "tool_run_tool",
            "description": (
                "Control a running tool's timeout. "
                "Pass next_timeout_window=0 to TERMINATE the tool immediately. "
                "Pass a positive number (e.g. 15, 30) to EXTEND the timeout "
                "by that many seconds. You MUST call this when a tool times out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "next_timeout_window": {
                        "type": "number",
                        "description": (
                            "0 = terminate the tool now. "
                            "Positive number = extend timeout by this many seconds "
                            "(e.g. 15 means wait 15 more seconds)."
                        ),
                    },
                    "_tool_comment": {
                        "type": "string",
                        "description": "describe what you are doing or why you wants to run it [under 3-10 words].",
                    },
                },
                "required": ["next_timeout_window"],
            },
        },
    },
]
