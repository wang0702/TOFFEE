
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from toffee.config import (
    D_ACTION,
    EFFORT_LIST,
    HISTORY_LIST,
    HISTORY_MODES,
    MODEL_LIST,
    MODEL_TIERS,
    N_EFFORTS,
    N_HISTORIES,
    N_MODELS,
    N_TOOLS,
    UNIFIED_TOOL_NAMES,
)
from toffee.utils import extract_json_string_field, is_metadata_sql_query


_CACHE_MIN_CHARS = 4000


def _maybe_cacheable_user(text: str) -> Dict[str, object]:
    if len(text) < _CACHE_MIN_CHARS:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [{
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }],
    }


GOAL_ANALYTICAL_INTENT = {
    "reconnaissance":  "Discover what data exists and form initial hypotheses",
    "transformation":  "Query data to test hypotheses -- look for specific numbers",
    "verification":    "Stress-test findings -- check edge cases, quantify uncertainty",
    "reporting":       "Synthesize evidence into structured answer with specific numbers",
}

_OPERATOR_GOAL_PHASE = {
    "SchemaScout":   "reconnaissance",
    "DataInspect":   "reconnaissance",
    "SQLDraft":      "transformation",
    "SQLRepair":     "transformation",
    "PythonCompute": "transformation",
    "SanityCheck":   "verification",
    "AnswerCompose": "reporting",
}


def _summarize_findings(state) -> str:
    parts = []

    if state.memory.discovered_tables:
        tables_str = ", ".join(sorted(state.memory.discovered_tables))
        parts.append(f"Tables in DB: [{tables_str}]")

    recent = []
    for msg in reversed(state.messages):
        if msg["role"] == "tool" and not msg.get("content", "").startswith("ERROR:"):
            content = msg["content"].strip()
            if len(content) > 10 and len(recent) < 2:
                recent.append(content[:150])
    recent.reverse()
    parts.extend(recent)
    if not parts:
        return "(First step — no prior results)"
    return " | ".join(parts)


def _schema_tools_already_done(state) -> List[str]:
    done = set()
    for idx, msg in enumerate(state.messages):
        if msg["role"] != "assistant":
            continue
        content = msg.get("content", "")
        match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', content)
        if not match:
            continue
        tool = match.group(1)
        if tool not in ("list_tables", "get_table_schema", "list_directory"):
            continue

        if idx + 1 < len(state.messages):
            nxt = state.messages[idx + 1]
            if nxt.get("role") == "tool" and not nxt.get("content", "").startswith("ERROR:"):
                done.add(tool)
    return sorted(done)


def _successful_data_steps(state) -> int:
    if getattr(state, "step_events", None):
        return sum(1 for ev in state.step_events if ev.substantive)

    count = 0
    for idx, msg in enumerate(state.messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if '"tool_name": "execute_sql"' not in content and '"tool_name": "execute_python"' not in content:
            continue
        if '"tool_name": "execute_sql"' in content:
            query = extract_json_string_field(content, "query")
            if is_metadata_sql_query(query):
                continue
        if idx + 1 < len(state.messages):
            next_msg = state.messages[idx + 1]
            if next_msg.get("role") == "tool" and not next_msg.get("content", "").startswith("ERROR:"):
                count += 1
    return count


OPERATOR_NAMES = [
    "SchemaScout",
    "DataInspect",
    "SQLDraft",
    "SQLRepair",
    "PythonCompute",
    "SanityCheck",
    "AnswerCompose",
]

_FILE_REF_RE = re.compile(r"\b[\w.-]+\.(?:md|txt|csv|tsv|json|jsonl|xlsx|xls)\b", re.IGNORECASE)


@dataclass
class Operator:
    name: str
    precondition: Callable
    tools: List[str]
    prompt_template: str = ""

    def is_feasible(self, state) -> bool:
        return self.precondition(state)

    def active_tools(self, state) -> List[str]:


        tools = list(self.tools)
        if "list_tables" in tools and "list_tables" in _schema_tools_already_done(state):
            tools = [t for t in tools if t != "list_tables"]
        return tools

    def build_prompt(
        self,
        state,
        history_mode: str = "mid",
        task_context: Optional[dict] = None,
        active_tool: str = "",
    ) -> List[Dict[str, str]]:


        max_chars = HISTORY_MODES.get(history_mode, 16000)
        history_text = ""
        for msg in state.messages:
            history_text += f"[{msg['role']}]: {msg['content']}\n"
        if len(history_text) > max_chars:
            history_text = "[... earlier history elided ...]\n" + history_text[-(max_chars - 50):]


        allowed = self.active_tools(state)
        if active_tool and active_tool in allowed:
            active_tools = [active_tool]
        else:
            active_tools = allowed
        tool_list = ", ".join(active_tools) if active_tools else "none (text-only response)"
        question = ""
        db_path = ""
        db_path_full = ""
        if task_context:
            question = task_context.get("question", task_context.get("objective", ""))
            env = task_context.get("env", {})
            db_path_full = env.get("db_path") or env.get("data_file", "")

            db_path = Path(db_path_full).name if db_path_full else ""

        phase_key = _OPERATOR_GOAL_PHASE.get(self.name, "transformation")
        analytical_phase = GOAL_ANALYTICAL_INTENT.get(phase_key, "Analyze the data")
        findings_summary = _summarize_findings(state)
        schema_done = _schema_tools_already_done(state)
        schema_done_str = ", ".join(schema_done) if schema_done else ""
        referenced_files = []
        if question:
            referenced_files = sorted(set(_FILE_REF_RE.findall(question)))
        missing_files = state.missing_referenced_files() if hasattr(state, "missing_referenced_files") else []

        system = (
            f"You are a senior data analyst. Answer the QUESTION by querying the DATABASE.\n\n"
            f"QUESTION: {question}\n"
            f"DATABASE: {db_path}\n"
            f"PHASE: {analytical_phase}\n"
            f"KEY FACTS: {findings_summary}\n"
            f"ALLOWED TOOLS: {tool_list}\n"
        )
        deliverable_txt = ""
        if task_context:
            deliverable_txt = str(task_context.get("deliverable", "")).strip()
        if deliverable_txt:
            system += f"DELIVERABLE: {deliverable_txt}\n"


        has_sql_tools = "execute_sql" in self.tools or "execute_python" in self.tools
        has_schema_tools = "list_tables" in self.tools or "get_table_schema" in self.tools

        if state.schema_discovered and has_sql_tools:
            phase_guidance = (
                f"Schema is ALREADY KNOWN. Write substantive data queries (SELECT with "
                f"GROUP BY, JOIN, aggregation) to answer the question."
            )
        elif state.schema_discovered and not has_sql_tools:
            phase_guidance = (
                f"Schema is already known. Use your allowed tools ({tool_list}) to "
                f"explore the data further."
            )
        elif has_schema_tools:
            if state.memory.discovered_tables:
                phase_guidance = (
                    f"Table names are already known; get_table_schema on one "
                    f"relevant table is the natural next step."
                )
            else:
                phase_guidance = (
                    f"Schema not yet discovered. Call list_tables first, then get_table_schema "
                    f"on relevant tables."
                )
        else:
            phase_guidance = (
                f"Explore the data environment using your allowed tools ({tool_list})."
            )
        if self.name == "DataInspect" and state.primary_source_kind in {"sqlite", "db"} and not state.has_aux_sources:
            phase_guidance = (
                "This is a database-only environment. Do NOT read the SQLite file as text or binary. "
                "Prefer list_tables, get_table_schema, and substantive SQL analysis instead."
            )
        elif self.name == "DataInspect" and referenced_files:
            phase_guidance = (
                f"The question explicitly references external files ({', '.join(referenced_files)}). "
                f"Read those files first and extract only the facts needed to drive later SQL/Python analysis."
            )


        error_guidance = ""
        if state.has_error or state.consecutive_errors > 0:
            error_guidance = (
                "\nERROR RECOVERY — Your last tool call FAILED. Before your next call:\n"
                "1. Re-read the ACTUAL table/column names from your previous SUCCESSFUL outputs.\n"
                "2. Use ONLY names that appeared verbatim in list_tables or get_table_schema results.\n"
                "3. Start simple: SELECT * FROM <known_table> LIMIT 5.\n"
                "4. Do NOT retry the exact same query or guess new table names.\n"
            )

        system += (
            f"\n{phase_guidance}\n"
            + error_guidance
            + (
                "\nGOOD NEXT-STEP EXAMPLE AFTER list_tables:\n"
                "<think>\n"
                "list_tables returned: Campuses, degrees, faculty.\n"
                "To compute the requested ratio, I need column names from degrees and faculty.\n"
                "The correct next step is get_table_schema on one of those tables.\n"
                "</think>\n"
                "I now need the schema of a key table before writing joins or aggregations.\n"
                "{\"tool_name\": \"get_table_schema\", \"arguments\": {\"table_name\": \"degrees\"}}\n"
            if self.name == "SchemaScout" and state.memory.discovered_tables and not state.schema_discovered else ""
            )
            + f"\nRESPONSE FORMAT (all three parts, in order):\n"
            f"1. <think>...</think> — MUST begin by quoting the actual result from the "
            f"PREVIOUS tool output (e.g., 'list_tables returned: channels, drivers, orders...'). "
            f"Then identify the analytical gap. Then plan the next step.\n"
            f"2. One sentence summarizing what you LEARNED and what you will investigate next. "
            f"Do NOT describe expected results — you have not run the tool yet.\n"
            f'3. {{"tool_name": "...", "arguments": {{...}}}}\n\n'
            f"GROUNDING RULES (critical):\n"
            f"- Use ONLY table names from list_tables output and column names from get_table_schema output.\n"
            f"- Do NOT infer or guess table/column names from the database filename or question text.\n"
            f"- Every number you cite must come from a previous tool output you received.\n"
            f"- If a table name is not in your list_tables result, it does not exist.\n\n"
            + (
                f"EXTERNAL FILES MENTIONED IN THE QUESTION: {', '.join(referenced_files)}.\n"
                f"Read those exact files rather than scanning unrelated files.\n\n"
                if referenced_files else ""
            )
            + (
                f"FILES NOT YET READ: {', '.join(missing_files)}.\n"
                f"Prioritize these before returning to more file discovery.\n\n"
                if missing_files else ""
            )
            +
            f"APPROACH: read the schema of each table the question touches, run one baseline aggregate over the full population, then drill in with targeted queries "
            f"(prefer ORDER BY over hardcoded ids, JOINs over assumed alignment). Cross-check at least one key finding a second way, and compose the answer only from "
            f"numbers that appeared in tool outputs. Use as many tool calls as the analysis honestly needs.\n\n"
            f"- execute_sql: arg='query'. sqlite3 dialect.\n"
            f"- execute_python: arg='code'. Use sqlite3.connect('{db_path_full}').\n"
        )

        if self.name == "AnswerCompose":

            system = (
                f"You are a senior data analyst writing a FINAL analytical report.\n\n"
                f"QUESTION: {question}\n"
                f"DATABASE: {db_path}\n"
                + (f"TABLES: {', '.join(sorted(state.memory.discovered_tables))}\n" if state.memory.discovered_tables else "") +
                f"\nYou have completed your analysis. Now write the deliverable.\n\n"
                f"IMPORTANT: Your report must be grounded in actual query results — "
                f"numbers that appeared in execute_sql or execute_python output. "
                f"Schema-only observations (table names, column types) are NOT evidence. "
                f"If your queries returned errors, acknowledge the specific errors and "
                f"explain what they reveal, but do NOT fabricate data.\n\n"
                f"FORMAT:\n"
                f"1. <think>...</think> — Synthesize ALL evidence (at least 300 words). "
                f"For each finding cite exact numbers from query results. Compare across groups. "
                f"Note what is surprising vs expected. Identify caveats about joins, "
                f"missing data, aggregation artifacts.\n\n"
                f"2. After </think>, write the COMPLETE report. DO NOT output any JSON or tool calls.\n\n"
                f"The report MUST have these sections:\n\n"
                f"## Evidence\n"
                f"Ranked list of concrete findings with specific numbers from your queries.\n"
                f"Example: '1. Crossfit_Hanna leads with EUR5.08M total volume (55,139 transactions), "
                f"nearly 2x Golfclub_Baron_Friso at EUR2.55M (27,748 transactions).'\n\n"
                f"## Interpretation\n"
                f"What the numbers mean: scale vs rate effects, which differences are large vs small, "
                f"comparisons across dimensions (merchant, category, account type, fee structure).\n\n"
                f"RULES:\n"
                f"- Length follows the deliverable: 100-200 words for a scalar or list answer, "
                f"200-350 for a comparative analysis, up to 500 for a multi-source report\n"
                f"- Every claim must cite a specific number from your queries\n"
                f"- Do NOT mention tools, operators, or workflow steps\n"
                f"- Do NOT output JSON\n"
                f"- Report the numeric facts you observed, in plain declarative sentences\n"
                f"- Write to completion — do not stop mid-sentence"
            )
            messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
            if history_text:
                messages.append(_maybe_cacheable_user(f"Analysis evidence collected:\n{history_text}"))
            messages.append(
                {"role": "user", "content": (
                    "Write the complete analytical report now. "
                    "Start with <think> to synthesize your findings, then write the full structured report."
                )}
            )
            return messages

        if self.name in ("SQLDraft", "PythonCompute", "SanityCheck"):
            system += (
                "\n\nUse this step for substantive analysis over the data itself, "
                "not schema-only SQL such as sqlite_master or PRAGMA."
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        if history_text:
            messages.append(_maybe_cacheable_user(f"Analysis history:\n{history_text}"))


        if self.name == "DataInspect" and referenced_files:
            user_msg = (
                f"The question depends on external files: {', '.join(referenced_files)}. "
                f"Read the missing referenced files directly: {', '.join(missing_files or referenced_files)}."
            )
        elif not state.result_exists and not state.schema_discovered and state.memory.discovered_tables:
            user_msg = (
                f"Table names are known: {', '.join(sorted(state.memory.discovered_tables))}. "
                f"Call get_table_schema on the 1-2 tables most relevant to the task."
            )
        elif not state.result_exists and not state.schema_discovered:
            user_msg = "Start the analysis. Call list_tables first."
        elif not state.result_exists and state.memory.discovered_tables:
            user_msg = (
                f"Schema is known. Tables: {', '.join(sorted(state.memory.discovered_tables))}. "
                f"Write a data query using these exact table names."
            )
        elif not state.result_exists:
            user_msg = "Schema is known. Write your first data query now."
        else:
            user_msg = (
                "Pick the next query that produces a genuinely new number: a different "
                "table, JOIN, aggregation, or verification. Stop only when the answer is "
                "fully grounded in the evidence you have collected."
            )
        messages.append({"role": "user", "content": user_msg})
        return messages


OPERATORS: Dict[str, Operator] = {
    "SchemaScout":   Operator(name="SchemaScout",   precondition=lambda s: True,                                tools=["list_tables", "get_table_schema"]),
    "DataInspect":   Operator(
        name="DataInspect",
        precondition=lambda s: s.has_aux_sources,
        tools=["read_file", "list_directory"],
    ),
    "SQLDraft":      Operator(name="SQLDraft",      precondition=lambda s: s.schema_discovered,                 tools=["execute_sql"]),
    "SQLRepair":     Operator(name="SQLRepair",     precondition=lambda s: s.has_error,                         tools=["execute_sql"]),
    "PythonCompute": Operator(name="PythonCompute", precondition=lambda s: s.schema_discovered,                     tools=["execute_python"]),
    "SanityCheck":   Operator(name="SanityCheck",   precondition=lambda s: s.result_exists,                     tools=["execute_sql", "execute_python"]),


    "AnswerCompose": Operator(
        name="AnswerCompose",
        precondition=lambda s: s.schema_discovered,
        tools=[],
    ),
}


@dataclass
class ActionConfig:

    operator_name: str
    model: str
    history: str = "mid"
    effort: str = "standard"
    tool_name: str = ""

    @property
    def key(self) -> str:
        return f"{self.operator_name}|{self.tool_name}|{self.model}|{self.history}|{self.effort}"

    def feature_vector(self) -> np.ndarray:
        z = np.zeros(D_ACTION, dtype=np.float64)
        idx = 0

        if self.tool_name in UNIFIED_TOOL_NAMES:
            z[idx + UNIFIED_TOOL_NAMES.index(self.tool_name)] = 1.0
        idx += N_TOOLS

        model_idx = MODEL_LIST.index(self.model) if self.model in MODEL_LIST else 0
        z[idx + model_idx] = 1.0;  idx += N_MODELS

        hist_idx = HISTORY_LIST.index(self.history) if self.history in HISTORY_LIST else 1
        z[idx + hist_idx] = 1.0;  idx += N_HISTORIES

        effort_idx = EFFORT_LIST.index(self.effort) if self.effort in EFFORT_LIST else 0
        z[idx + effort_idx] = 1.0;  idx += N_EFFORTS

        return z

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "model": self.model,
            "history": self.history,
            "effort": self.effort,
        }


def enumerate_feasible_actions(state) -> List[ActionConfig]:
    actions: List[ActionConfig] = []
    for op_name, op in OPERATORS.items():
        if not op.is_feasible(state):
            continue


        tools_for_op = op.active_tools(state) or [""]
        for tool in tools_for_op:
            for model in MODEL_LIST:
                for hist in HISTORY_LIST:
                    for effort in EFFORT_LIST:
                        actions.append(ActionConfig(
                            operator_name=op_name,
                            tool_name=tool,
                            model=model,
                            history=hist,
                            effort=effort,
                        ))
    return actions
