
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from toffee.config import MIN_ACCEPT_QUALITY, MODELS

log = logging.getLogger(__name__)


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "list_tables",
        "description": "List all tables in the database. Returns table names with row counts. Use this first to discover available data.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_table_schema",
        "description": "Get detailed schema for a table: column names, data types, nullable, primary keys, row count, and 3 sample rows. Essential before writing queries.",
        "parameters": {"type": "object",
                       "properties": {"table_name": {"type": "string", "description": "Name of the table"}},
                       "required": ["table_name"]},
    }},
    {"type": "function", "function": {
        "name": "execute_sql",
        "description": "Execute a SQL query (SELECT only). Returns up to 100 rows with column names. The result is also stored in 'data' variable for Python analysis.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "SQL SELECT query to execute"}},
                       "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "execute_python",
        "description": "Execute Python code for data analysis.\n\nAVAILABLE LIBRARIES: pandas (pd), numpy (np), json, math, statistics, re, datetime, Counter, defaultdict\n\nSPECIAL VARIABLES:\n- 'data': DataFrame containing the last SQL query result\n- Variables PERSIST across multiple calls - you can reuse previously defined variables\n\nOUTPUT: Use print() to display results, or assign to 'result' variable.\n\nEXAMPLE:\n```python\n# Analyze last SQL result\nprint(data.describe())\nfiltered = data[data['amount'] > 100]\nresult = filtered.groupby('category').sum()\n```",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                       "required": ["code"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file (JSON, CSV, MD, TXT, YAML, Excel). Use for context files containing domain knowledge, configuration, or additional data.",
        "parameters": {"type": "object",
                       "properties": {
                           "file_path": {"type": "string", "description": "Path to the file"},
                           "encoding": {"type": "string", "description": "File encoding (default: auto-detect)"},
                           "max_lines": {"type": "integer", "description": "Max lines to read (default: 500)"},
                       },
                       "required": ["file_path"]},
    }},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and directories. Use to discover available data files when you don't know what exists.",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string", "description": "Directory path (default: current directory)"},
                           "pattern": {"type": "string", "description": "Glob pattern filter (e.g., '*.csv', '*.json')"},
                           "recursive": {"type": "boolean", "description": "List subdirectories recursively (default: false)"},
                       },
                       "required": []},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file. Use to save analysis results, reports, or processed data.",
        "parameters": {"type": "object",
                       "properties": {
                           "file_path": {"type": "string", "description": "Path to write to"},
                           "content": {"type": "string", "description": "Content to write"},
                           "mode": {"type": "string", "description": "'w' (overwrite, default) or 'a' (append)"},
                       },
                       "required": ["file_path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Execute a bash command. Use for system operations, file management, or when other tools don't fit. Dangerous commands (rm -rf, sudo) are blocked.",
        "parameters": {"type": "object",
                       "properties": {
                           "command": {"type": "string", "description": "Bash command to execute"},
                           "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30, max: 120)"},
                           "cwd": {"type": "string", "description": "Working directory (optional)"},
                       },
                       "required": ["command"]},
    }},
]


_PLACEHOLDER_FA_RE = re.compile(
    r"^(analysis completed with \d+ steps\.?\s*|based on the analysis steps above\.?\s*)$",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r'(?:/[\w._-]+){3,}/([^/\s\'"\\)]+\.\w+)')
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _scrub_paths(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _PATH_RE.sub(r"\1", text)


def _wrap_output(tool_name: str, raw_output: str) -> str:
    if isinstance(raw_output, str) and raw_output.startswith("{") and '"success"' in raw_output:
        return raw_output
    is_error = isinstance(raw_output, str) and raw_output.startswith("ERROR:")
    return json.dumps(
        {"success": not is_error, "output": raw_output},
        ensure_ascii=False,
    )


def _filter_redundant_steps(steps: list) -> list:
    if len(steps) <= 3:
        return list(steps)
    seen: set = set()
    out = []
    for step in steps:
        tool = step.tool_name
        is_data = tool in ("execute_sql", "execute_python")
        out_str = step.real_output if isinstance(step.real_output, str) else str(step.real_output)
        if out_str.startswith("ERROR: Bad arguments:"):
            continue
        sig = f"{tool}|{json.dumps(step.arguments, sort_keys=True) if step.arguments else ''}"
        if sig in seen and not is_data:
            continue
        seen.add(sig)
        out.append(step)
    if len(out) != len(steps):
        log.info("filter: removed %d/%d duplicate non-data steps", len(steps) - len(out), len(steps))
    return out


def _build_user_message(task: dict, env: dict) -> str:
    data_file = Path(env.get("data_file", env.get("db_path", ""))).name
    question = task.get("question", task.get("objective", "")).strip()
    deliverable = task.get("deliverable", "Provide the analysis result").strip()
    parts = []
    if data_file:
        parts.append(f"Data file: {data_file}")
    body = f"Task: {question}"
    body += " Constraints: Base conclusions only on computed evidence."
    if deliverable:
        body += f" Deliverable: {deliverable}"
    parts.append(body)
    return "\n\n".join(parts)


def _assistant_content(reasoning: str, visible: str) -> str:
    reasoning = (reasoning or "").strip()
    visible = (visible or "").strip()
    visible = _THINK_RE.sub("", visible).strip()
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{visible}".rstrip()
    return visible


def _summarize_for_compose(steps: list, max_chars: int = 6000) -> str:
    chunks = []
    used = 0
    for i, s in enumerate(steps):
        out_str = s.real_output if isinstance(s.real_output, str) else str(s.real_output)
        block = (
            f"[Step {i+1}] tool={s.tool_name} args={json.dumps(s.arguments, ensure_ascii=False)}\n"
            f"output: {out_str[:600]}\n"
        )
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)
    return "".join(chunks)


def _compose_missing_answer(
    trajectory, task: dict, client, budget=None,
) -> tuple[str, str]:
    if client is None:
        return trajectory.final_answer or "", trajectory.final_reasoning or ""

    transcript = _summarize_for_compose(trajectory.steps)
    question = task.get("question", task.get("objective", ""))
    deliverable = task.get("deliverable", "")

    system = (
        "You are composing the final answer for a data-agent trajectory. "
        "Use only numbers and facts that already appear in the tool outputs above. "
        "Do not invent new data. Do not run new tools.\n"
        "Hard ban on hedging language. Do NOT write any of: "
        "'cannot be concluded', 'cannot be derived', 'cannot be computed', "
        "'cannot be established', 'cannot determine', 'unable to determine', "
        "'not part of evidence', 'not visible from', 'not present in the evidence', "
        "'is not established', 'evidence is insufficient', 'evidence is narrow', "
        "'evidence is limited', 'data is insufficient', 'do not have enough', "
        "'insufficient evidence', or any close paraphrase. "
        "If a deliverable component truly needs more data, just state numeric facts you "
        "have in plain declarative sentences and stop — do not announce what is missing.\n"
        'Return ONLY a JSON object: {"reasoning": "...", "answer": "..."}'
    )
    user = (
        f"Question: {question}\n"
        f"Deliverable: {deliverable}\n\n"
        f"Trajectory transcript:\n{transcript}\n\n"
        "Compose the final reasoning and answer."
    )
    try:
        content, usage = client.call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=MODELS["capable"],
            max_tokens=1024,
            temperature=0.0,
        )
        if budget is not None and usage is not None:
            budget.record(usage.cost, usage.prompt_tokens + usage.completion_tokens)
        m = re.search(r"\{.*\}", content or "", re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            answer = (parsed.get("answer", "") or "").strip()
            reasoning = (parsed.get("reasoning", "") or "").strip()
            if answer:
                return answer, reasoning
    except Exception as exc:
        log.warning("compose_missing_answer failed: %s", exc)

    return trajectory.final_answer or "", trajectory.final_reasoning or ""


def _final_answer_present(trajectory) -> bool:
    fa = (trajectory.final_answer or "").strip()
    if not fa:
        return False
    if _PLACEHOLDER_FA_RE.match(fa):
        return False
    return True


def assemble_sft(
    trajectory,
    task: dict,
    env: dict,
    output_path: str,
    round_idx: int = 0,
    client=None,
    mcts_quality: float = 0.0,
    budget=None,
) -> dict:
    result = {
        "ok": False,
        "validation_ok": True,
        "sample_judge_score": mcts_quality,
        "sample_judge_reason": "assembled",
        "extra_cost": 0.0,
        "extra_tokens": 0,
    }


    raw_steps = [
        s for s in trajectory.steps
        if s.operator_name != "AnswerCompose" and s.tool_name not in ("none", "AnswerCompose", "")
    ]
    steps = _filter_redundant_steps(raw_steps)

    messages: List[Dict[str, Any]] = []
    messages.append({"role": "user", "content": _scrub_paths(_build_user_message(task, env))})

    for i, step in enumerate(steps):
        assistant_content = _assistant_content(step.reasoning, step.visible_text or "")
        messages.append({"role": "assistant", "content": _scrub_paths(assistant_content)})

        tool_call_id = f"call_{i+1:03d}"
        tc_payload = {
            "name": step.tool_name,
            "arguments": step.arguments if isinstance(step.arguments, dict) else {},
        }
        messages.append({
            "role": "tool_call",
            "content": _scrub_paths(json.dumps(tc_payload, ensure_ascii=False)),
            "tool_call_id": tool_call_id,
        })

        wrapped = _wrap_output(step.tool_name, step.real_output)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": step.tool_name,
            "content": _scrub_paths(wrapped),
        })


    if _final_answer_present(trajectory):
        final_answer = trajectory.final_answer.strip()
        final_reasoning = (trajectory.final_reasoning or "").strip()
    else:
        final_answer, final_reasoning = _compose_missing_answer(
            trajectory, task, client, budget=budget,
        )
        if not final_answer:


            final_answer = (trajectory.final_reasoning or "").strip() or "Analysis summary based on the steps above."
            final_reasoning = ""

    closing = _assistant_content(final_reasoning, final_answer)
    messages.append({"role": "assistant", "content": _scrub_paths(closing)})

    sample = {
        "messages": messages,
        "tools": json.dumps(TOOL_SCHEMAS, ensure_ascii=False),
    }


    toffee_meta = {
        k: task[k] for k in ("level", "path", "hint", "source_span", "format_span", "question_type")
        if k in task and task.get(k) not in (None, "", [])
    }
    if toffee_meta:
        sample["toffee_meta"] = toffee_meta


    certificate = getattr(trajectory, "certificate", None)
    if certificate:
        sample["certificate"] = certificate

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False, indent=2)
    log.info("SFT sample saved to %s", output_path)


    swift_root = Path(output_path).parent
    while swift_root.name in ("_rejected", "openai"):
        swift_root = swift_root.parent
    swift_dir = swift_root / ("swift_rejected" if "_rejected" in str(output_path) else "swift")
    swift_dir.mkdir(parents=True, exist_ok=True)
    swift_path = swift_dir / Path(output_path).name
    with open(swift_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    log.info("Swift SFT sample saved to %s", swift_path)

    result["ok"] = mcts_quality >= MIN_ACCEPT_QUALITY
    return result
