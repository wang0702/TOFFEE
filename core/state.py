
from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from toffee.config import (
    D_STATE, MAX_CONSECUTIVE_ERRORS, MAX_STEP_COUNT, N_FORMATS,
)
from toffee.utils import is_metadata_sql_query, extract_json_string_field, sha256_short


_EXT_TO_FORMAT = {
    "sqlite": "sqlite_table", "db": "sqlite_table", "sqlite3": "sqlite_table",
    "csv": "csv_table", "tsv": "csv_table",
    "xlsx": "excel_sheet", "xls": "excel_sheet", "xlsm": "excel_sheet",
    "md": "md_struct", "markdown": "md_struct",
}


def _infer_format_span_from_env(env: dict) -> float:
    paths = []
    primary = env.get("data_file") or env.get("db_path") or ""
    if primary:
        paths.append(primary)
    for p in env.get("extra_sources") or []:
        if p:
            paths.append(p)
    formats = set()
    for p in paths:
        ext = str(p).rsplit(".", 1)[-1].lower() if "." in str(p) else ""
        fmt = _EXT_TO_FORMAT.get(ext)
        if fmt:
            formats.add(fmt)
    return (len(formats) / N_FORMATS) if formats else 0.0


SUBSTANTIVE_STEP_SCORE = 0.50
_REF_FILE_RE = re.compile(r"\b[\w.-]+\.(?:md|txt|csv|tsv|json|jsonl|xlsx|xls)\b", re.IGNORECASE)


def _normalize_unit_name(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_metadata_python(code: str) -> bool:
    text = (code or "").lower()
    markers = (
        "pragma table_info",
        "sqlite_master",
        ".tables",
        "table_info(",
        "cursor.description",
        "information_schema",
    )
    return any(marker in text for marker in markers)


@dataclass
class CompressedMemory:
    schema_digest: str = ""
    discovered_tables: List[str] = field(default_factory=list)
    error_traces: List[str] = field(default_factory=list)
    result_signatures: List[str] = field(default_factory=list)

    def digest(self) -> str:
        payload = (
            self.schema_digest
            + "|".join(sorted(self.discovered_tables))
            + "|".join(self.error_traces[-3:])
        )
        return sha256_short(payload)

    def add_error(self, err: str) -> None:
        self.error_traces.append(err[:500])
        if len(self.error_traces) > 10:
            self.error_traces = self.error_traces[-10:]

    def add_result(self, sig: str) -> None:
        self.result_signatures.append(sig[:200])
        if len(self.result_signatures) > 10:
            self.result_signatures = self.result_signatures[-10:]


@dataclass
class StepEvent:
    operator_name: str
    tool_name: str
    success: bool
    nonempty: bool
    metadata_only: bool = False
    judge_score: Optional[float] = None
    substantive: bool = False
    verification: bool = False


@dataclass
class AnalysisState:

    state_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    canonical_key: str = ""


    env_fingerprint: str = ""


    messages: List[Dict[str, str]] = field(default_factory=list)
    step_events: List[StepEvent] = field(default_factory=list)
    memory: CompressedMemory = field(default_factory=CompressedMemory)


    schema_discovered: bool = False
    result_exists: bool = False
    answer_drafted: bool = False
    verification_passed: bool = False
    has_error: bool = False
    consecutive_errors: int = 0
    pending_goal: str = "reconnaissance"


    last_tool: str = "other"
    last_result_nonempty: bool = False
    resolved_prior_error: bool = False


    step_count: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0


    visit_count: int = 0
    children: Dict[str, "AnalysisState"] = field(default_factory=dict)
    action_values: Dict[str, float] = field(default_factory=dict)
    action_visits: Dict[str, int] = field(default_factory=dict)
    action_cumulative: Dict[str, float] = field(default_factory=dict)
    priors: Dict[str, float] = field(default_factory=dict)
    sigmas: Dict[str, float] = field(default_factory=dict)
    action_sq_cumulative: Dict[str, float] = field(default_factory=dict)


    provenance_paths: List[str] = field(default_factory=list)
    reusable_summaries: List[str] = field(default_factory=list)


    level_1: float = 0.0
    level_2: float = 0.0
    level_3: float = 0.0
    source_span: float = 0.0
    format_span: float = 0.0
    entry_units: List[str] = field(default_factory=list)
    primary_source_kind: str = ""
    has_aux_sources: bool = False
    referenced_files: List[str] = field(default_factory=list)
    seen_files: List[str] = field(default_factory=list)


    parent_id: Optional[str] = None
    action_key: str = ""
    operator_name: str = ""
    config_model: str = ""
    config_history: str = ""
    config_effort: str = ""
    llm_content: str = ""
    tool_output: str = ""
    reasoning: str = ""
    step_judge_score: Optional[float] = None
    last_file_coverage_gain: float = 0.0


    def compute_canonical_key(self) -> str:
        raw = f"{self.env_fingerprint}|{self.memory.digest()}|{self.pending_goal}"
        self.canonical_key = sha256_short(raw)
        return self.canonical_key


    def feature_vector(self) -> np.ndarray:
        x = np.zeros(D_STATE, dtype=np.float64)
        x[0] = self.step_count / MAX_STEP_COUNT
        x[1] = float(self.schema_discovered)
        x[2] = float(self.result_exists)
        x[3] = float(self.has_error)
        x[4] = min(self.consecutive_errors, 5) / 5.0
        x[5] = min(self._count_data_steps() / MAX_STEP_COUNT, 1.0)
        from toffee import config as _cfg
        if _cfg.PROVENANCE_FEATURES:
            x[6] = self.level_1
            x[7] = self.level_2
            x[8] = self.level_3
            x[9] = max(0.0, min(1.0, self.source_span))
            x[10] = max(0.0, min(1.0, self.format_span))
        return x

    def _count_data_steps(self) -> int:
        if self.step_events:
            return sum(1 for ev in self.step_events if ev.substantive)


        count = 0
        for idx, msg in enumerate(self.messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if "execute_sql" not in content and "execute_python" not in content:
                continue
            if "execute_sql" in content:
                query = extract_json_string_field(content, "query")
                if is_metadata_sql_query(query):
                    continue
            if idx + 1 < len(self.messages):
                nxt = self.messages[idx + 1]
                if nxt.get("role") == "tool" and not nxt.get("content", "").startswith("ERROR:"):
                    count += 1
        return count

    def _refresh_phase_flags(self) -> None:
        self.result_exists = any(ev.substantive for ev in self.step_events)
        self.verification_passed = any(ev.verification for ev in self.step_events)
        if self.answer_drafted:
            self.pending_goal = "reporting"
        elif self.result_exists:
            self.pending_goal = "verification"
        elif self.schema_discovered:
            self.pending_goal = "transformation"
        else:
            self.pending_goal = "reconnaissance"

    def file_coverage_ratio(self) -> float:
        if not self.referenced_files:
            return 1.0
        seen = {Path(p).name for p in self.seen_files}
        target = {Path(p).name for p in self.referenced_files}
        return len(seen & target) / max(len(target), 1)

    def missing_referenced_files(self) -> List[str]:
        if not self.referenced_files:
            return []
        seen = {Path(p).name for p in self.seen_files}
        return [p for p in self.referenced_files if Path(p).name not in seen]

    def entry_units_touched(self) -> List[str]:
        if not self.entry_units:
            return []
        if not self.step_events:
            return []


        sql_text: List[str] = []
        file_args: List[str] = []
        list_dir_outputs: List[str] = []
        for ev in self.step_events:
            tool = getattr(ev, "tool_name", "") or ""
            args = getattr(ev, "arguments", None) or {}
            if not isinstance(args, dict):
                args = {}
            if tool in ("execute_sql", "execute_python"):
                for v in args.values():
                    sql_text.append(str(v).lower())
            elif tool == "get_table_schema":
                sql_text.append(str(args.get("table_name", "")).lower())
            elif tool in ("read_file", "list_directory"):
                for v in args.values():
                    file_args.append(str(v).lower())
            if tool == "list_directory":
                out = getattr(ev, "output_text", None) or ""
                list_dir_outputs.append(str(out).lower())

        sql_hay = "\n".join(sql_text)
        file_hay = "\n".join(file_args)
        listing_hay = "\n".join(list_dir_outputs)

        touched: List[str] = []
        for unit in self.entry_units:
            u = str(unit).strip().lower()
            if not u:
                continue
            base = u.rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0]
            has_ext = "." in base
            if has_ext:


                if base in file_hay or base in listing_hay:
                    touched.append(unit)
            else:


                if base in sql_hay or (len(stem) >= 3 and stem in sql_hay):
                    touched.append(unit)
        return touched

    def entry_units_coverage_ratio(self) -> float:
        if not self.entry_units:
            return 1.0
        return len(self.entry_units_touched()) / max(len(self.entry_units), 1)

    def apply_step_judge(self, score: Optional[float]) -> None:
        self.step_judge_score = score
        if not self.step_events:
            self._refresh_phase_flags()
            return

        event = self.step_events[-1]
        was_substantive = event.substantive
        event.judge_score = score

        is_analytical_op = event.operator_name in ("SQLDraft", "SQLRepair", "PythonCompute", "SanityCheck")
        qualifies = event.success and event.nonempty and not event.metadata_only and is_analytical_op
        if score is None:
            event.substantive = qualifies
        else:
            event.substantive = qualifies and score >= SUBSTANTIVE_STEP_SCORE
        event.verification = event.substantive and event.operator_name == "SanityCheck"

        if event.substantive and not was_substantive and self.tool_output:
            self.memory.add_result(self.tool_output[:200])

        self._refresh_phase_flags()

    @property
    def schema_coverage_ratio(self) -> float:
        if not self.memory.discovered_tables:
            return 0.0
        referenced = sum(
            1 for t in self.memory.discovered_tables
            if any(t.lower() in sig.lower() for sig in self.memory.result_signatures[-5:])
        )
        return referenced / len(self.memory.discovered_tables)


    @classmethod
    def from_task(cls, task: dict) -> "AnalysisState":
        env = task.get("env", {})
        db_path = env.get("db_path", env.get("data_file", ""))
        try:
            level = int(str(task.get("level", task.get("motif") or 0)).lstrip("Mm") or 0)
        except ValueError:
            level = 0
        state = cls(
            env_fingerprint=sha256_short(db_path),
            pending_goal="reconnaissance",
            provenance_paths=[task.get("task_id", uuid.uuid4().hex[:8])],
            level_1=1.0 if level == 1 else 0.0,
            level_2=1.0 if level == 2 else 0.0,
            level_3=1.0 if level == 3 else 0.0,
            source_span=float(task.get("source_span") or task.get("scope_span") or 0.0),
            format_span=float(
                task.get("format_span") or task.get("family_span")
                or _infer_format_span_from_env(env)
            ),
            entry_units=[
                _normalize_unit_name(u)
                for u in (task.get("hint", {}) or {}).get("entry_units", [])
                if str(u).strip()
            ],
            primary_source_kind=(
                (str(env.get("data_file", env.get("db_path", ""))).rsplit(".", 1)[-1].lower())
                if "." in str(env.get("data_file", env.get("db_path", ""))) else ""
            ),
            has_aux_sources=bool(env.get("extra_sources")),
            referenced_files=sorted(set(_REF_FILE_RE.findall(
                " ".join([
                    str(task.get("question", "")),
                    str(task.get("deliverable", "")),
                ])
            ))),
        )
        state.compute_canonical_key()
        return state

    def clone_with_new_provenance(self, task_id: str) -> "AnalysisState":
        new = copy.deepcopy(self)
        new.state_id = uuid.uuid4().hex[:12]
        new.provenance_paths = list(self.provenance_paths) + [task_id]
        new.visit_count = 0
        new.children = {}
        new.action_values = {}
        new.action_visits = {}
        new.action_cumulative = {}
        new.priors = {}
        new.sigmas = {}
        new.action_sq_cumulative = {}
        return new

    def expand(
        self,
        operator_name: str,
        config: dict,
        llm_content: str,
        tool_result: Any,
        usage: Any,
    ) -> "AnalysisState":
        child = copy.deepcopy(self)
        child.state_id = uuid.uuid4().hex[:12]
        child.parent_id = self.state_id


        child.children = {}
        child.action_values = {}
        child.action_visits = {}
        child.action_cumulative = {}
        child.priors = {}
        child.sigmas = {}
        child.action_sq_cumulative = {}
        child.visit_count = 0
        child.operator_name = operator_name
        child.config_model = config.get("model", "")
        child.config_history = config.get("history", "mid")
        child.config_effort = config.get("effort", "standard")
        child.llm_content = llm_content
        child.reasoning = ""
        child.step_judge_score = None
        child.last_file_coverage_gain = 0.0

        child.tool_output = getattr(tool_result, "output", str(tool_result))
        child.last_result_nonempty = getattr(tool_result, "nonempty", bool(child.tool_output.strip()))
        tool_success = getattr(tool_result, "success", True)
        tool_name = getattr(tool_result, "tool_name", "")
        metadata_only = False
        if tool_name == "execute_sql":
            query = extract_json_string_field(llm_content, "query")
            metadata_only = is_metadata_sql_query(query)
        elif tool_name == "execute_python":
            code = extract_json_string_field(llm_content, "code")
            metadata_only = _is_metadata_python(code)
        file_path = (
            extract_json_string_field(llm_content, "path")
            or extract_json_string_field(llm_content, "file_path")
        ).strip()

        had_error_before = child.has_error
        child.has_error = not tool_success
        if not tool_success:
            child.consecutive_errors += 1
            child.memory.add_error(getattr(tool_result, "stderr", "")[:500])
        else:
            child.resolved_prior_error = had_error_before and (not child.has_error)
            child.consecutive_errors = 0

        if operator_name == "SchemaScout" and tool_success:
            raw = getattr(tool_result, "stdout", "")
            if tool_name == "list_tables" and raw:
                tables = [t.strip() for t in raw.split() if t.strip()]
                child.memory.discovered_tables = list(set(child.memory.discovered_tables + tables))
                child.memory.schema_digest = sha256_short("|".join(sorted(child.memory.discovered_tables)))
            elif tool_name == "get_table_schema" and child.last_result_nonempty:
                child.schema_discovered = True
        elif operator_name == "AnswerCompose":
            child.answer_drafted = True

        if tool_name == "read_file" and tool_success and file_path:
            before_cov = self.file_coverage_ratio()
            normalized = Path(file_path).name
            if normalized and normalized not in child.seen_files:
                child.seen_files = child.seen_files + [normalized]
            child.last_file_coverage_gain = max(0.0, child.file_coverage_ratio() - before_cov)

        if operator_name in ("SQLDraft", "SQLRepair"):
            child.last_tool = "sql"
        elif operator_name == "PythonCompute":
            child.last_tool = "python"
        elif operator_name in ("SchemaScout", "DataInspect"):
            child.last_tool = "schema"
        else:
            child.last_tool = "other"

        child.step_count += 1
        if usage:
            child.total_cost += getattr(usage, "cost", 0.0)
            child.total_tokens += getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)

        child.messages = child.messages + [
            {"role": "assistant", "content": llm_content},
            {"role": "tool", "content": child.tool_output},
        ]
        child.step_events = child.step_events + [StepEvent(
            operator_name=operator_name,
            tool_name=tool_name,
            success=tool_success,
            nonempty=child.last_result_nonempty,
            metadata_only=metadata_only,
        )]

        action_key = f"{operator_name}|{config.get('model', '')}|{config.get('history', '')}|{config.get('effort', '')}"
        child.action_key = action_key
        child._refresh_phase_flags()
        child.compute_canonical_key()
        return child

    def is_terminal(self) -> bool:
        return (
            self.answer_drafted
            or self.step_count >= MAX_STEP_COUNT
            or self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS
        )

    def is_prunable(self, q_min: float) -> bool:
        if not self.action_values:
            return False
        return all(v < q_min for v in self.action_values.values())
