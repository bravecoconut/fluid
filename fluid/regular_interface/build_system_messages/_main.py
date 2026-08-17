# fluid/regular_interface/build_system_messages/_main.py

from fluid.utill import json_ser
import json


class BuildSystemMessages:
    def __init__(
        self,
        default_skills,
        non_default_skills,
        grouped_contexts,
        off_tools,
    ):
        self.non_default_skills = non_default_skills
        self.off_tools = off_tools
        self.non_default_skills_and_their_tools = []
        self.non_default_skills_and_their_tools_sys_m = ""

        self.default_skills = default_skills
        self.grouped_contexts = grouped_contexts
        self.default_skills_and_contexts = []
        self.default_skills_and_contexts_sys_m = ""

    def build_skills_and_tools(self):
        """
        For every non-default skill, collect the tools that belong to it
        (a tool belongs to a skill when tool["skill"] == skill["name"]).
        """

        # Fall back to an empty list if the attribute is missing or None,
        # so the for-loops below never blow up on `None`.
        non_default_skills = self.non_default_skills or []
        off_tools = self.off_tools or []

        # Make sure the result container actually exists.
        if getattr(self, "non_default_skills_and_their_tools", None) is None:
            self.non_default_skills_and_their_tools = []

        for skill in non_default_skills:
            # Skip anything that isn't a dict - we can't safely read from it.
            if not isinstance(skill, dict):
                continue

            skill_name = skill.get("name")
            # A skill with no name can never match a tool, so skip it.
            if not skill_name:
                continue

            new_el = {"skill": skill, "tools": []}

            for tool in off_tools:
                if not isinstance(tool, dict):
                    continue

                tool_skill = tool.get("skill")
                if not tool_skill:
                    continue

                if skill_name == tool_skill:
                    new_el["tools"].append(tool)

            self.non_default_skills_and_their_tools.append(new_el)

        return self.non_default_skills_and_their_tools

    def build_skills_and_context(self):
        # default

        default_skills = self.default_skills or []
        grouped_contexts = self.grouped_contexts or {}

        # Make sure the result list actually exists before we append to it.
        if getattr(self, "default_skills_and_contexts", None) is None:
            self.default_skills_and_contexts = []

        for skill in default_skills:
            # Skip anything that isn't a dict - can't safely read from it.
            if not isinstance(skill, dict):
                continue

            skill_name = skill.get("name")
            # A skill with no name can never match a context group, so skip it.
            if not skill_name:
                continue

            new_el = {"skill": skill, "contexts": None}

            # grouped_contexts should be a dict of {id: {...}} - guard in case it's
            # some other type (e.g. a list) that would break .items().
            if isinstance(grouped_contexts, dict):
                for _key_contexts, _values in grouped_contexts.items():
                    if not isinstance(_values, dict):
                        continue

                    group_skill_name = _values.get("skill_name")
                    if not group_skill_name:
                        continue

                    if skill_name == group_skill_name:
                        # NOTE: db name is actually skill name!
                        new_el["contexts"] = _values.get("db_collections")
                        break  # found the match, no need to keep scanning

            self.default_skills_and_contexts.append(new_el)

        return self.default_skills_and_contexts

    def format_them(self):
        """
        non default skills: skills info + thier tools
        default skills: skills info + context with templates.
        """

        # non-default skills system message
        # Make sure the attribute exists and isn't None before looping.
        non_default_skills_and_their_tools = self.build_skills_and_tools() or []

        # Make sure the target string attribute exists before we += onto it.
        if getattr(self, "non_default_skills_and_their_tools_sys_m", None) is None:
            self.non_default_skills_and_their_tools_sys_m = ""

        _skill_buff = ""

        for skill_tools_pair in non_default_skills_and_their_tools:
            # Skip anything that isn't a dict shape we expect.
            if not isinstance(skill_tools_pair, dict):
                continue

            skill = skill_tools_pair.get("skill")
            tools = skill_tools_pair.get("tools")

            # Need a proper skill dict and a name to show anything useful.
            if not isinstance(skill, dict):
                continue

            skill_name = skill.get("name")
            if not skill_name:
                continue

            # tools should be a list; if missing/None/empty, skip this skill entirely.
            if not tools or not isinstance(tools, list):
                continue

            _tools_buff = ""

            for tool in tools:
                # Skip malformed tool entries instead of crashing.
                if not isinstance(tool, dict):
                    continue

                tool_name = tool.get("name")
                if not tool_name:
                    # A tool without a name isn't useful to show, skip it.
                    continue

                usually_takes = tool.get("usually_takes")
                tool_usually_takes = (
                    f"usually takes {usually_takes} seconds"
                    if usually_takes
                    else "usually not takes time"
                )

                description = tool.get("description")
                tool_description = (
                    description
                    if description
                    else "no description is available. it would be better to ignore using it to prevent mistakes!"
                )

                # str(...) protects .strip() in case a value isn't already a string
                # (e.g. tool_name accidentally being a number).
                _tools_buff += (
                    f"\n\t\t\t- TOOL NAME: {str(tool_name).strip()} "
                    f"\n\t\t\t- TIME NEEDED: {str(tool_usually_takes).strip()} "
                    f"\n\t\t\t- DESCRIPTION: {str(tool_description).strip().replace("\n","\n\t\t\t")}\n"
                )

            # If every tool in this skill was malformed/skipped, _tools_buff is empty -
            # don't bother adding an empty skill section.
            if not _tools_buff:
                continue

            _skill_buff += (
                f"\n\t\t- SKILL NAME: {str(skill_name).strip()}\n{_tools_buff}\n"
            )

        if _skill_buff:
            self.non_default_skills_and_their_tools_sys_m += (
                f"\n\tTHESE SKILLS ARE AVAILABLE, BUT NOT ACTIVE. "
                f"USE SKILL MANIPULATION TO ACTIVE THEM IF NECESSARY.\n{_skill_buff}"
            )

        # print("*" * 20, "non default skills and their tools sys m", "*" * 20)
        # print(self.non_default_skills_and_their_tools_sys_m)

        # default skills system message
        default_skills_and_contexts = self.build_skills_and_context() or []

        # Make sure the target string attribute exists before we += onto it.
        if getattr(self, "default_skills_and_contexts_sys_m", None) is None:
            self.default_skills_and_contexts_sys_m = ""

        for skill_context_pair in default_skills_and_contexts:
            if not isinstance(skill_context_pair, dict):
                continue

            skill = skill_context_pair.get("skill")
            contexts = skill_context_pair.get("contexts")

            if not isinstance(skill, dict):
                continue

            skill_name = skill.get("name")
            if not skill_name:
                # No name means we can't label this skill meaningfully, skip it.
                continue

            # Fall back to a plain "{placeholder}" template so .format() still works
            # even if the real template is missing.
            skill_template = skill.get("template") or "{placeholder}"

            collections = skill.get("collections")
            if not isinstance(collections, list):
                collections = []

            # contexts can legitimately be None (set that way when no match was found
            # earlier), so always normalize it to a list before looping.
            if not isinstance(contexts, list):
                contexts = []

            skill_buff = ""

            for collection in collections:
                if not isinstance(collection, dict):
                    continue

                collection_template = (
                    collection.get("collec_template") or "{placeholder}"
                )

                context_buff = ""

                for context in contexts:
                    if not isinstance(context, dict):
                        continue

                    results_contexts = context.get("results_contexts")
                    if not isinstance(results_contexts, list) or not results_contexts:
                        continue

                    for _context in results_contexts:
                        if not isinstance(_context, dict):
                            continue

                        document = _context.get("document")
                        if not document:
                            # Nothing useful to show for this entry, skip it.
                            continue

                        # Store in a variable first, then use it in the f-string.
                        # (Nesting double quotes inside an f-string's {} like the
                        # original code did only works on Python 3.12+; this avoids
                        # that problem entirely and works on older versions too.)
                        context_buff += (
                            f"\n\t\t\t\t- {document.replace("\n","\n\t\t\t\t")}\n"
                        )

                # .format() can raise (e.g. if the template has a stray "{" or a
                # placeholder name that doesn't exist), so don't let one bad
                # template crash the whole loop.
                try:
                    formatted_collection = collection_template.format(
                        placeholder=context_buff
                    ) if context_buff else ''
                except (KeyError, IndexError, ValueError):
                    formatted_collection = context_buff  # fall back to raw content

                skill_buff += f"\n\t\t\t- {formatted_collection}\n" if formatted_collection else ''

            try:
                formatted_skill = skill_template.format(placeholder=skill_buff) if skill_buff else ''
            except (KeyError, IndexError, ValueError):
                formatted_skill = skill_buff  # fall back to raw content

            self.default_skills_and_contexts_sys_m += (
                f"\n\tSKILL NAME: {skill_name}\n\n\t\t- {formatted_skill}\n"
            ) if formatted_skill else ''

        # print("*" * 20, "default skills and contexts sys m", "*" * 20)
        # print(self.default_skills_and_contexts_sys_m)

        return (
            self.non_default_skills_and_their_tools_sys_m,
            self.default_skills_and_contexts_sys_m,
        )
