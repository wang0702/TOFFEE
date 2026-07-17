
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from toffee.config import (
    JUDGE_MODEL,
    MAX_STEP_COUNT,
    REWARD_LAMBDA,
)
from toffee.utils import (
    extract_answer_text,
    extract_json_string_field,
    is_metadata_sql_query,
)

log = logging.getLogger(__name__)

_NUMERIC_TOKEN_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_TABLE_NAME_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\b", re.IGNORECASE)
_JSON_SCORE_RE = re.compile(r"\{[\s\S]*?\}")
_JUDGE_CACHE: Dict[str, float] = {}


def _extract_visible_text(content: str) -> str:
    think_end = content.find("</think>")
    if think_end >= 0:
        return content[think_end + len("</think>"):].strip()
    return content.strip()


def _extract_tool_name(content: str) -> str:
    match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', content or "")
    return match.group(1) if match else "unknown"


def _extract_final_answer(state) -> str:


    for msg in reversed(state.messages):
        if msg.get("role") == "assistant":
            return extract_answer_text(msg.get("content", ""))
    return ""


def _classify_steps(state) -> Dict[str, Any]:
    total = len(getattr(state, "step_events", []))
    successful = 0
    failed = 0
    data_queries = 0
    schema_queries = 0
    distinct_tables: Set[str] = set()
    tool_outputs: List[str] = []

    if getattr(state, "step_events", None):
        successful = sum(1 for ev in state.step_events if ev.success)
        failed = total - successful
        data_queries = sum(1 for ev in state.step_events if ev.substantive)
        schema_queries = sum(
            1 for ev in state.step_events
            if ev.tool_name in ("list_tables", "get_table_schema", "read_file", "list_directory")
        )

    assistant_idx = -1
    for idx, msg in enumerate(state.messages):
        if msg.get("role") != "assistant":
            continue
        assistant_idx += 1
        content = msg.get("content", "")
        tool_name = _extract_tool_name(content)
        event = None
        if getattr(state, "step_events", None) and assistant_idx < len(state.step_events):
            event = state.step_events[assistant_idx]
        if not getattr(state, "step_events", None):
            total += 1

        is_schema = tool_name in ("list_tables", "get_table_schema", "list_directory")
        is_data = tool_name in ("execute_sql", "execute_python")
        is_metadata_query = False
        if tool_name == "execute_sql":
            query = extract_json_string_field(content, "query")
            is_metadata_query = is_metadata_sql_query(query)
            if query:
                _SQL_KEYWORDS = {
                    "select", "from", "where", "and", "or", "not", "in",
                    "join", "left", "right", "inner", "outer", "on", "as",
                    "group", "by", "order", "having", "limit", "offset",
                    "union", "all", "distinct", "count", "sum", "avg",
                    "min", "max", "null", "true", "false", "case", "when",
                    "then", "else", "end", "like", "between", "exists",
                    "insert", "update", "delete", "create", "drop", "alter",
                    "table", "index", "view", "pragma", "integer", "text",
                    "real", "blob", "varchar", "int", "float", "desc", "asc",
                }
                for table_match in _TABLE_NAME_RE.finditer(query):
                    candidate = table_match.group(1).lower()
                    if candidate not in _SQL_KEYWORDS:
                        distinct_tables.add(candidate)

        tool_success = False
        if idx + 1 < len(state.messages):
            nxt = state.messages[idx + 1]
            if nxt.get("role") == "tool":
                output = nxt.get("content", "")
                if output.startswith("ERROR:"):
                    if not getattr(state, "step_events", None):
                        failed += 1
                else:
                    tool_success = True
                    if not getattr(state, "step_events", None):
                        successful += 1
                    if event is None or event.substantive:
                        tool_outputs.append(output)

        if getattr(state, "step_events", None):
            continue

        if is_schema:
            schema_queries += 1
        elif is_data and not is_metadata_query:
            if tool_success:


                output_text = tool_outputs[-1] if tool_outputs else ""
                error_lines = sum(1 for line in output_text.split("\n") if "ERROR" in line.upper())
                total_lines = max(output_text.count("\n") + 1, 1)
                if error_lines / total_lines < 0.5:
                    data_queries += 1

    return {
        "total_steps": total,
        "successful_steps": successful,
        "failed_steps": failed,
        "data_queries": data_queries,
        "schema_queries": schema_queries,
        "distinct_tables_queried": distinct_tables,
        "tool_outputs": tool_outputs,
    }


def verify_answer(predicted: str, ground_truth: str) -> float:
    pred, gt = _normalise(predicted), _normalise(ground_truth)
    if pred == gt:
        return 1.0
    try:
        pf, gf = float(pred), float(gt)
        if abs(gf) > 1e-9 and abs(pf - gf) / abs(gf) < 0.01:
            return 0.95
        if abs(pf - gf) < 1e-6:
            return 0.95
    except (ValueError, TypeError):
        pass
    if gt in pred or pred in gt:
        return 0.6
    return 0.0


def _normalise(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    for ch in ['"', "'", "`"]:
        text = text.strip(ch)
    return text


def _ground_truth_is_answer(ground_truth: str) -> bool:
    raw = (ground_truth or "").strip().lower()
    if not raw:
        return False
    if raw.startswith(("select ", "with ", "pragma ", "explain ")):
        return False
    if "select " in raw and " from " in raw:
        return False
    return True


_FACT_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")


def _fact_close(x: float, y: float) -> bool:
    if abs(y) < 1e-9:
        return abs(x) < 1e-6
    return abs(x - y) / abs(y) <= 0.01


def _fact_nums(text: str) -> List[float]:
    out: List[float] = []
    for tok in _FACT_NUM_RE.findall(_THOUSANDS_RE.sub("", text or "")):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


_KEY_NUM_RE = re.compile(r"(?<![\d.,/-])\d[\d,]*(?:\.\d+)?(?![\d/-])")


def _key_numbers_with_pos(text: str) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for m in _KEY_NUM_RE.finditer(text or ""):
        try:
            out.append((m.start(), float(m.group(0).replace(",", ""))))
        except ValueError:
            continue
    return out


def _normalize_key_rows(provenance: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _push(label: Any, value: Any) -> None:
        lab = str(label) if label not in (None, "") else None
        val = float(value) if isinstance(value, (int, float)) else None
        if lab is None and val is None:
            return
        rows.append({"label": lab, "value": val})

    for item in provenance or []:
        if not isinstance(item, dict):
            continue
        if "value" in item or "raw" in item:
            _push(item.get("label"), item.get("value"))
            continue
        subrows = item.get("rows")
        if subrows:
            for r in subrows:
                labs = r.get("labels") or []
                nums = r.get("nums") or []
                _push(labs[0] if labs else None, nums[0] if nums else None)
            continue
        labs = item.get("labels") or []
        nums = item.get("nums") or []
        if labs and nums:
            _push(labs[0], nums[0])
        elif nums:
            for n in nums:
                _push(None, n)
        elif labs:
            for lab in labs:
                _push(lab, None)

    seen: Set[Tuple[Any, Any]] = set()
    uniq: List[Dict[str, Any]] = []
    for r in rows:
        k = (r["label"], r["value"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return uniq


def _clause_numbers(
    occ: List[Tuple[int, str]], nums: List[Tuple[int, float]], i: int, text_len: int,
) -> List[float]:
    lo = 0 if i == 0 else occ[i][0]
    hi = occ[i + 1][0] if i + 1 < len(occ) else text_len
    return [v for (q, v) in nums if lo <= q < hi]


def _correct(
    predicted: str, key_rows: List[Dict[str, Any]],
) -> Optional[List[Tuple[Optional[str], Optional[float]]]]:
    text = predicted or ""
    low = text.lower()
    nums = _key_numbers_with_pos(text)

    spans: List[Tuple[int, int, str]] = []
    for lab in {r["label"] for r in key_rows if r["label"]}:
        ll = lab.lower()
        start = 0
        while True:
            k = low.find(ll, start)
            if k < 0:
                break
            spans.append((k, k + len(ll), ll))
            start = k + max(1, len(ll))


    occ: List[Tuple[int, str]] = sorted(
        (s, ll) for s, e, ll in spans
        if not any(s2 <= s and e <= e2 and len(l2) > len(ll)
                   for s2, e2, l2 in spans))

    matched: List[Tuple[Optional[str], Optional[float]]] = []
    for r in key_rows:
        lab, val = r["label"], r["value"]
        if lab is None:
            if val is None:
                continue
            if not any(_fact_close(v, val) for _p, v in nums):
                return None
            matched.append((None, val))
            continue
        ll = lab.lower()
        if val is None:
            if ll not in low:
                return None
            matched.append((lab, None))
            continue
        covered = any(
            any(_fact_close(v, val) for v in _clause_numbers(occ, nums, i, len(low)))
            for i, (_p, ol) in enumerate(occ) if ol == ll)
        if not covered:
            return None
        matched.append((lab, val))
    return matched


def _reads_source(step: Dict[str, Any], source_names: Sequence[str]) -> bool:
    text = step.get("text")
    if text is None:
        return True
    tl = str(text).lower()
    names = [str(n).lower() for n in (source_names or []) if n]
    named = any(re.search(r"\b" + re.escape(n) + r"\b", tl) for n in names)
    if (step.get("tool") or "") == "execute_python":
        return named or any(t in tl for t in ("scratch", ".sqlite", ".db", ".csv", ".xlsx"))
    if not any(t in tl for t in (" from ", "\nfrom ", "from(", "from (")):
        return False
    return named


def _step_reads_unit(step: Dict[str, Any], unit: str) -> bool:
    text = step.get("text")
    if text is None:
        return False
    return bool(re.search(r"\b" + re.escape(str(unit).lower()) + r"\b", str(text).lower()))


_SQL_STRING_RE = re.compile(r"'(?:[^']|'')*'|\"[^\"]*\"")
_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_NUM_LITERAL_RE = re.compile(r"(?<![\w.\"'])(\d+(?:\.\d+)?)(?![\w.])")
_SQL_ROWCAP_RE = re.compile(r"\b(?:limit|offset)\s+\d+(?:\s*,\s*\d+)?", re.IGNORECASE)


def _sql_literal_values(sql: str) -> List[float]:
    """Numeric literal nodes of the statement's syntax tree. Parsed with
    sqlglot when available; otherwise read off the tokenized text after string
    literals, comments, and LIMIT/OFFSET row caps are removed, which yields
    the same literal set under SQLite's grammar."""
    try:
        import sqlglot
        from sqlglot import exp as _exp
        out: List[float] = []
        for stmt in sqlglot.parse(sql or "", read="sqlite"):
            if stmt is None:
                continue
            for lit in stmt.find_all(_exp.Literal):
                if not lit.is_number:
                    continue
                if isinstance(lit.parent, (_exp.Limit, _exp.Offset)):
                    continue
                try:
                    out.append(float(lit.this))
                except (TypeError, ValueError):
                    continue
        return out
    except ImportError:
        pass
    except Exception:
        pass
    body = _SQL_COMMENT_RE.sub(" ", sql or "")
    body = _SQL_STRING_RE.sub(" ", body)
    body = _SQL_ROWCAP_RE.sub(" ", body)
    out = []
    for m in _SQL_NUM_LITERAL_RE.finditer(body):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _python_literal_values(code: str) -> List[float]:
    """Numeric constant nodes of the Python syntax tree; the same taint rule
    as the SQL check. Falls back to the token scan when the code does not parse."""
    import ast as _pyast
    try:
        tree = _pyast.parse(code or "")
    except SyntaxError:
        return _sql_literal_values(code or "")
    vals: List[float] = []
    for node in _pyast.walk(tree):
        if isinstance(node, _pyast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            vals.append(float(node.value))
    return vals


def _step_literals(step: Dict[str, Any]) -> List[float]:
    text = step.get("text")
    if not text:
        return []
    if (step.get("tool") or "") == "execute_python":
        return _python_literal_values(str(text))
    return _sql_literal_values(str(text))


_SQL_SHAPE_TOKENS = (
    " where ", "\nwhere ", " join ", "\njoin ", " group by", " having ",
    " distinct ", "sum(", "count(", "avg(", "min(", "max(", "total(", " limit ",
)


def _output_row_count(output: str) -> int:
    lines = [ln for ln in (output or "").splitlines() if ln.strip()]
    return max(0, len(lines) - 2)


def _is_wide_dump(step: Dict[str, Any]) -> bool:
    """An unfiltered scan: no filter, join, or aggregation, and more output
    rows than WIDE_SCAN_ROW_LIMIT. Such a dump cannot back a stated value."""
    if (step.get("tool") or "") != "execute_sql":
        return False
    text = step.get("text")
    if not text:
        return False
    body = _SQL_COMMENT_RE.sub(" ", str(text).lower())
    body = " " + _SQL_STRING_RE.sub(" ", body) + " "
    if any(tok in body for tok in _SQL_SHAPE_TOKENS):
        return False
    from toffee.config import WIDE_SCAN_ROW_LIMIT
    return _output_row_count(step.get("output", "")) > WIDE_SCAN_ROW_LIMIT


def _computed_support(
    matched: List[Tuple[Optional[str], Optional[float]]],
    data_steps: List[Dict[str, Any]],
    source_names: Sequence[str],
) -> Optional[Dict[int, List[int]]]:
    """Map each stated value to the scoped data steps whose executed output
    backs it. A step counts only when it reads a source, is not an unfiltered
    scan, and does not state the value as a literal in its own text. Returns
    None when some stated value has no support."""
    scoped: List[Tuple[int, Dict[str, Any], List[float]]] = []
    for i, s in enumerate(data_steps):
        if not (s.get("output") or "").strip():
            continue
        if not _reads_source(s, source_names):
            continue
        if _is_wide_dump(s):
            continue
        scoped.append((i, s, _fact_nums(s.get("output", ""))))
    support: Dict[int, List[int]] = {}
    for mi, (_lab, val) in enumerate(matched):
        if val is None:
            continue
        backing = [
            i for i, s, nums in scoped
            if any(_fact_close(x, val) for x in nums)
            and not any(_fact_close(l, val) for l in _step_literals(s))
        ]
        if not backing:
            return None
        support[mi] = backing
    return support


_INTERVENED_DB_CACHE: Dict[Tuple[str, float, str], str] = {}


def _intervened_copy(db_path: str, unit: str) -> Optional[str]:
    """A copy of the environment database with the unit's rows removed (the
    emptied-source intervention). Copies are cached per (db, mtime, unit)."""
    import os as _os
    import shutil as _shutil
    import sqlite3 as _sq
    import tempfile as _tempfile
    try:
        key = (db_path, _os.path.getmtime(db_path), str(unit))
    except OSError:
        return None
    hit = _INTERVENED_DB_CACHE.get(key)
    if hit and _os.path.exists(hit):
        return hit
    fd, tmp = _tempfile.mkstemp(prefix="toffee_intervene_", suffix=".sqlite")
    _os.close(fd)
    try:
        _shutil.copyfile(db_path, tmp)
        conn = _sq.connect(tmp)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        low = str(unit).lower()
        target = next((t for t in tables if t.lower() == low), None)
        if target is None:
            target = next((t for t in tables if t.lower().startswith(low)
                           or low.startswith(t.lower())), None)
        if target is None:
            conn.close()
            _os.unlink(tmp)
            return None
        conn.execute(f'DELETE FROM "{target}"')
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("intervention copy failed for %s/%s: %s", db_path, unit, exc)
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        return None
    _INTERVENED_DB_CACHE[key] = tmp
    return tmp


def _replay_step(step: Dict[str, Any], intervened_db: str, original_db: str) -> Optional[str]:
    """Re-execute a recorded supporting query against the intervened copy.
    Only recorded queries are replayed; no model is called."""
    tool = step.get("tool") or ""
    text = step.get("text")
    if not text:
        return None
    if tool == "execute_sql":
        from toffee.core.executor import execute_sql
        return execute_sql(intervened_db, str(text)).output
    if tool == "execute_python" and original_db and original_db in str(text):
        from toffee.core.executor import execute_python
        return execute_python(str(text).replace(original_db, intervened_db)).output
    return None


def _norm_output(output: str) -> str:
    return "\n".join(ln.rstrip() for ln in (output or "").splitlines()).strip()


def _covered(
    support: Dict[int, List[int]],
    data_steps: List[Dict[str, Any]],
    required_sources: Sequence[str],
    db_path: Optional[str],
    detail: Dict[str, Any],
) -> bool:
    """The answer must depend on every source on the task's provenance path:
    the supporting queries together read each source, and re-executing them
    with that source emptied changes their output or fails them."""
    import os as _os
    sup_idxs = sorted({i for idxs in support.values() for i in idxs})
    replays = 0
    for unit in required_sources:
        readers = [i for i in sup_idxs if _step_reads_unit(data_steps[i], unit)]
        if not readers:
            detail["covered_fail"] = f"no supporting query reads {unit}"
            return False
        if not db_path or not _os.path.exists(db_path):
            continue
        tmp = _intervened_copy(db_path, unit)
        if tmp is None:
            continue
        unit_replays = 0
        changed = False
        for i in readers:
            out2 = _replay_step(data_steps[i], tmp, db_path)
            if out2 is None:
                continue
            unit_replays += 1
            replays += 1
            if _norm_output(out2) != _norm_output(data_steps[i].get("output", "")):
                changed = True
                break
        if unit_replays and not changed:
            detail["covered_fail"] = f"answer does not depend on {unit}"
            detail["intervention_replays"] = replays
            return False
    detail["intervention_replays"] = replays
    return True


def _column_at(header: str, rule: str, pos: int) -> Optional[str]:
    spans = [(m.start(), m.end()) for m in re.finditer(r"-+", rule or "")]
    for s, e in spans:
        if s <= pos <= e + 2:
            cell = header[s:e].strip()
            if cell:
                return cell
    return (header or "").strip() or None


def _locate_value(output: str, val: float) -> Tuple[Optional[str], Optional[str]]:
    lines = (output or "").splitlines()
    header = lines[0] if lines else ""
    rule = lines[1] if len(lines) > 1 else ""
    body = lines[2:] if len(lines) > 2 else lines
    for ln in body:
        for m in _KEY_NUM_RE.finditer(ln):
            try:
                x = float(m.group(0).replace(",", ""))
            except ValueError:
                continue
            if _fact_close(x, val):
                return ln.strip(), _column_at(header, rule, m.start())
    return None, None


def _lineage_records(
    matched: List[Tuple[Optional[str], Optional[float]]],
    support: Dict[int, List[int]],
    data_steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per stated pair: the supporting query, the output row, and the producing
    column expression. This record ships with every released trajectory."""
    recs: List[Dict[str, Any]] = []
    for mi, (lab, val) in enumerate(matched):
        if val is None or mi not in support:
            continue
        step = data_steps[support[mi][0]]
        row, col = _locate_value(step.get("output", ""), val)
        recs.append({
            "label": lab, "value": val,
            "query": (str(step.get("text") or ""))[:500],
            "output_row": row, "column": col,
        })
    return recs


def certify_answer(
    predicted: str, answer_key: str, provenance: list,
    data_steps: Optional[List[Dict[str, Any]]] = None,
    trace_outputs: Optional[List[str]] = None,
    source_names: Optional[Sequence[str]] = None,
    required_sources: Optional[Sequence[str]] = None,
    db_path: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """The certificate: Correct and Computed and Covered. All three lift V_q
    to the accept tier (>= 0.95, exact match 1.0); Correct alone caps V_q at
    0.5; otherwise V_q stays below 0.5."""
    detail: Dict[str, Any] = {"correct": False, "computed": False, "covered": None,
                              "lineage": [], "intervention_replays": 0}
    v = verify_answer(predicted, answer_key) if answer_key else 0.0
    key_rows = _normalize_key_rows(provenance)
    if not key_rows:
        return v, detail
    matched = _correct(predicted, key_rows)
    if matched is None:
        return min(v, 0.5), detail
    detail["correct"] = True
    steps = data_steps
    if steps is None and trace_outputs is not None:
        steps = [{"tool": "trace", "text": None, "output": o} for o in trace_outputs]
    if steps is None:
        return max(v, 0.95), detail
    support = _computed_support(matched, steps, source_names or [])
    if support is None:
        return 0.5, detail
    detail["computed"] = True
    detail["lineage"] = _lineage_records(matched, support, steps)
    required = [r for r in (required_sources or []) if r]
    if required:
        detail["covered"] = _covered(support, steps, required, db_path, detail)
        if not detail["covered"]:
            return 0.5, detail
    return max(v, 0.95), detail


def vq_score(
    predicted: str, answer_key: str, provenance: list,
    trace_outputs: Optional[List[str]] = None,
    data_steps: Optional[List[Dict[str, Any]]] = None,
    source_names: Optional[Sequence[str]] = None,
    required_sources: Optional[Sequence[str]] = None,
    db_path: Optional[str] = None,
) -> float:
    v, _detail = certify_answer(
        predicted, answer_key, provenance,
        data_steps=data_steps, trace_outputs=trace_outputs,
        source_names=source_names, required_sources=required_sources,
        db_path=db_path,
    )
    return v


def _tool_outputs(state) -> List[str]:
    return [
        m.get("content", "")
        for m in getattr(state, "messages", []) or []
        if m.get("role") == "tool"
        and m.get("content")
        and not m.get("content", "").startswith("ERROR:")
    ]


def _data_steps_from_state(state) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    msgs = getattr(state, "messages", []) or []
    for idx, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        tool = _extract_tool_name(content)
        if tool not in ("execute_sql", "execute_python"):
            continue
        text = (extract_json_string_field(content, "query")
                or extract_json_string_field(content, "sql")
                or extract_json_string_field(content, "code"))
        output = ""
        if idx + 1 < len(msgs) and msgs[idx + 1].get("role") == "tool":
            nxt = msgs[idx + 1].get("content", "")
            if not nxt.startswith("ERROR:"):
                output = nxt
        steps.append({"tool": tool, "text": text, "output": output})
    return steps


def _source_names_of_task(task: dict) -> List[str]:
    names: Set[str] = set()
    for f in ((task or {}).get("provenance") or (task or {}).get("facts") or []):
        for t in f.get("tables") or []:
            if t:
                names.add(str(t))
    for u in ((task or {}).get("hint") or {}).get("entry_units") or []:
        base = str(u).rsplit("/", 1)[-1]
        names.add(base)
        stem = base.rsplit(".", 1)[0]
        if stem:
            names.add(stem)
    return [n for n in names if n]


def _required_sources_of_task(task: dict) -> List[str]:
    """The source units on the task's provenance path, the set the Covered
    condition must span."""
    required: List[str] = []
    for f in ((task or {}).get("provenance") or (task or {}).get("facts") or []):
        for t in f.get("tables") or []:
            t = str(t)
            if t and t not in required:
                required.append(t)
    return required


def _db_path_of_task(task: dict) -> Optional[str]:
    env = (task or {}).get("env") or {}
    return env.get("db_path") or env.get("scratch_db") or None


_PHASE_ORDER = ("reconnaissance", "transformation", "verification", "reporting")


def _vh_terms(state, info: Dict) -> Dict[str, float]:


    progress = (
        _PHASE_ORDER.index(state.pending_goal) / (len(_PHASE_ORDER) - 1)
        if state.pending_goal in _PHASE_ORDER else 0.0
    )
    data_coverage = min(info["data_queries"] / MAX_STEP_COUNT, 1.0)

    total = max(info["total_steps"], 1)
    tool_success = info["successful_steps"] / total

    tool_arg_seen: set = set()
    redundant = 0
    for msg in state.messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        tool_name = _extract_tool_name(content)
        sig = f"{tool_name}|{content[:120]}"
        if sig in tool_arg_seen:
            redundant += 1
        tool_arg_seen.add(sig)
    non_redundancy = max(0.0, 1.0 - redundant / total)

    return {
        "progress": progress,
        "data_coverage": data_coverage,
        "tool_success": tool_success,
        "non_redundancy": non_redundancy,
    }


def heuristic_evaluate(state) -> float:
    if state.step_count == 0:
        return 0.0
    info = _classify_steps(state)
    terms = _vh_terms(state, info)
    from toffee.config import VH_MODE
    if VH_MODE == "success":

        return float(np.clip(terms["tool_success"], -1.0, 1.0))
    v_h = sum(terms.values()) / len(terms)
    log.debug("v_h=%.3f terms=%s data_q=%d",
              v_h, {k: round(v, 3) for k, v in terms.items()}, info["data_queries"])
    return float(np.clip(v_h, -1.0, 1.0))


def _summarize_history(state, max_messages: int = 10) -> str:
    lines = []
    for msg in state.messages[-max_messages:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "assistant":
            tool_name = _extract_tool_name(content)
            visible = _extract_visible_text(content)
            snippet = visible or content
            lines.append(f"[assistant/{tool_name}] {snippet[:240]}")
        elif role == "tool":
            status = "error" if content.startswith("ERROR:") else "ok"
            lines.append(f"[tool/{status}] {content[:240]}")
        else:
            lines.append(f"[{role}] {content[:240]}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = _JSON_SCORE_RE.search(text or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _judge_cache_key(state, task: dict, mode: str) -> str:
    question = task.get("question", task.get("objective", ""))
    history = _summarize_history(state, max_messages=12)
    final_answer = _extract_final_answer(state)
    raw = f"{mode}|{state.step_count}|{question}|{history}|{final_answer}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def llm_judge_score(
    state, task: dict, client, mode: str = "final",
    budget=None, metrics: Optional[Dict] = None,
) -> float:
    cache_key = _judge_cache_key(state, task, mode)
    if cache_key in _JUDGE_CACHE:
        return _JUDGE_CACHE[cache_key]

    question = task.get("question", task.get("objective", ""))
    final_answer = _extract_final_answer(state)
    history_summary = _summarize_history(state)

    dimensions = ["evidence_grounding", "analytical_depth", "coherence"]

    system = (
        "You are judging a data-agent trajectory for analytical correctness "
        "and evidence grounding.\n"
        "Score on whether the answer is correct, supported, and uses the "
        "operations the question requires; do not reward step count.\n\n"
        "Score three dimensions from 0.0 to 1.0:\n"
        "- evidence_grounding: Does the final answer cite specific numbers from actual "
        "data queries (execute_sql with SELECT/GROUP BY/JOIN, or execute_python)? "
        "Schema-only numbers do NOT count. Score 0.0 if no substantive data query ran.\n"
        "- analytical_depth: alignment with the operations the question demands "
        "(comparisons, joins, attribution, verification). Score by whether the "
        "required operations are performed and supported, not by raw step count: "
        "a single query is acceptable for a single-fact question, and additional "
        "steps are only credited when they execute a required operation.\n"
        "- coherence: Does each step logically follow from the previous result? "
        "Does the agent react to what it found (not just execute a pre-planned script)? "
        "Score 0.0 if the final answer does not follow from the gathered evidence.\n\n"
        "CRITICAL FAILURES (overall 0.0-0.1):\n"
        "- Report without any data queries\n"
        "- Hallucinated numbers (claimed before any tool returned them)\n"
        "- Same tool call repeated 3+ times\n\n"
        "Recoverable mistakes (syntax errors, wrong column) are fine if the agent recovers.\n"
        'Return ONLY JSON: {"overall": <0..1>, "evidence_grounding": <0..1>, '
        '"analytical_depth": <0..1>, "coherence": <0..1>, "reason": "..."}'
    )
    user = (
        f"Question: {question}\n\n"
        f"Trajectory:\n{history_summary}\n\n"
        f"Final answer:\n{final_answer[:1500]}\n"
    )
    try:
        content, usage = client.call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=JUDGE_MODEL,
            max_tokens=512,
            temperature=0.1,
        )
        if budget is not None and usage is not None:
            budget.record(
                usage.cost,
                usage.prompt_tokens + usage.completion_tokens,
            )
        if metrics is not None:
            metrics["total_llm_calls"] = metrics.get("total_llm_calls", 0) + 1
            metrics["judge_calls"] = metrics.get("judge_calls", 0) + 1
        parsed = _extract_json_object(content)
        if not parsed:
            return 0.0

        dimension_scores = []
        for key in dimensions:
            val = parsed.get(key)
            if isinstance(val, (int, float)):
                dimension_scores.append(float(np.clip(val, 0.0, 1.0)))

        overall = parsed.get("overall")
        if isinstance(overall, (int, float)):
            base = float(np.clip(overall, 0.0, 1.0))
        elif dimension_scores:
            base = float(np.mean(dimension_scores))
        else:
            return 0.0

        if parsed.get("reason"):
            log.debug("LLM judge: %.3f :: %s", base, str(parsed["reason"])[:200])
        _JUDGE_CACHE[cache_key] = base
        return base
    except Exception as exc:
        log.warning("LLM judge failed: %s", exc)
        _JUDGE_CACHE[cache_key] = 0.0
        return 0.0


def batch_final_judge(
    terminal_states: List, task: dict, client,
    budget=None, metrics: Optional[Dict] = None,
) -> List[float]:
    if not terminal_states or client is None:
        return [0.0] * len(terminal_states)

    question = task.get("question", task.get("objective", ""))[:300]
    scores: List[Optional[float]] = [None] * len(terminal_states)
    entries: List[Tuple[int, str, str, str]] = []

    for i, state in enumerate(terminal_states):
        cache_key = _judge_cache_key(state, task, "final")
        if cache_key in _JUDGE_CACHE:
            scores[i] = _JUDGE_CACHE[cache_key]
            continue
        final_answer = _extract_final_answer(state)[:1200]
        history_summary = _summarize_history(state, max_messages=10)
        entries.append((i, cache_key, final_answer, history_summary))

    if not entries:
        return [s if s is not None else 0.0 for s in scores]

    n = len(entries)
    system = (
        f"Rate {n} data-agent trajectories for analytical correctness and evidence grounding.\n"
        "Score on whether each answer is correct, supported, and uses the operations "
        "the question requires; do not reward step count.\n"
        "Score three dimensions from 0.0 to 1.0:\n"
        "- evidence_grounding: answer numbers trace to data queries (SELECT/JOIN/GROUP BY, "
        "execute_python); schema-only numbers do NOT count; 0.0 if no substantive data query ran.\n"
        "- analytical_depth: alignment with the operations the question demands "
        "(comparisons, joins, attribution, verification); a single query is acceptable for "
        "a single-fact question; additional steps count only when they execute a required operation.\n"
        "- coherence: each step follows from the previous result; 0.0 if the answer "
        "does not follow from the gathered evidence.\n"
        "Critical failures (overall 0.0-0.1): report with no data queries, hallucinated numbers, "
        "same tool repeated 3+ times.\n"
        f'Return ONLY JSON: {{"scores": [<o1>, <o2>, ...]}} with exactly {n} overall scores in [0,1].'
    )
    trajectory_blocks = []
    for idx, (_i, _ck, answer, history) in enumerate(entries):
        trajectory_blocks.append(
            f"=== Trajectory {idx+1} ===\n"
            f"History:\n{history}\n\n"
            f"Final answer:\n{answer}"
        )
    user = f"Question: {question}\n\n" + "\n\n".join(trajectory_blocks)

    try:
        content, usage = client.call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=JUDGE_MODEL,
            max_tokens=128 + n * 16,
            temperature=0.1,
        )
        if budget is not None and usage is not None:
            budget.record(usage.cost, usage.prompt_tokens + usage.completion_tokens)
        if metrics is not None:
            metrics["total_llm_calls"] = metrics.get("total_llm_calls", 0) + 1
            metrics["judge_calls"] = metrics.get("judge_calls", 0) + 1
        parsed = _extract_json_object(content)
        raw_scores = parsed.get("scores", []) if parsed else []
        for j, (orig_idx, cache_key, _a, _h) in enumerate(entries):
            if j < len(raw_scores) and isinstance(raw_scores[j], (int, float)):
                s = float(np.clip(raw_scores[j], 0.0, 1.0))
            else:
                s = 0.0
            scores[orig_idx] = s
            _JUDGE_CACHE[cache_key] = s
    except Exception as exc:
        log.warning("Batch final judge failed: %s", exc)
        for orig_idx, cache_key, _a, _h in entries:
            scores[orig_idx] = 0.0
            _JUDGE_CACHE[cache_key] = 0.0

    return [s if s is not None else 0.0 for s in scores]


def evaluate(
    state, task: dict, client=None,
    budget=None, metrics: Optional[Dict] = None,
    final_judge_val: Optional[float] = None,
) -> float:
    gt = task.get("ground_truth", task.get("answer", ""))
    gt_is_answer = _ground_truth_is_answer(gt)


    answer_key = str(task.get("answer_key", "") or "")
    key_rows = (task.get("answer_key_rows") or task.get("provenance")
                or task.get("answer_contract") or task.get("facts") or [])
    has_exec_evidence = bool(answer_key or key_rows)

    def _verified() -> float:
        if has_exec_evidence:
            return vq_score(
                _extract_final_answer(state), answer_key, key_rows,
                data_steps=_data_steps_from_state(state),
                source_names=_source_names_of_task(task),
                required_sources=_required_sources_of_task(task),
                db_path=_db_path_of_task(task))
        return verify_answer(_extract_final_answer(state), gt)

    log.debug("evaluate: step=%d terminal=%s", state.step_count, state.answer_drafted)


    if state.answer_drafted:
        if has_exec_evidence or gt_is_answer:
            return float(np.clip(_verified(), 0.0, 1.0))
        if client is None:
            return 0.0
        judge = (
            float(np.clip(final_judge_val, 0.0, 1.0))
            if final_judge_val is not None
            else llm_judge_score(state, task, client, mode="final",
                                 budget=budget, metrics=metrics)
        )
        return float(np.clip(judge, 0.0, 1.0))


    if state.step_count > 0:
        h = heuristic_evaluate(state)
        result = float(np.clip(h, -1.0, 1.0))
        log.debug("evaluate: non-terminal h=%.3f result=%.3f", h, result)
        return result

    return 0.0


def compute_reward(
    value: float,
    child_state,
    cost: float,
    cost_estimate: float,
    was_verified: bool = False,
    was_reused: bool = False,
) -> float:
    raw_ratio = cost / max(cost_estimate, 1e-6)
    return float(np.clip(value - REWARD_LAMBDA * min(raw_ratio, 1.0), -1.0, 1.0))
