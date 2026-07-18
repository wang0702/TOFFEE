
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from toffee.client.openrouter import OpenRouterClient
from toffee.config import (
    ADMISSION_REPLAY,
    B_SKEL_BASE,
    BRIDGE_MAX,
    MAX_UNITS_PER_ENV,
    MODELS,
    N_FORMATS,
    TAU_ND,
)
from toffee.core.executor import execute_sql, get_table_schema, list_tables
from toffee.generation.ingest import SourceUnit, ingest_environment

log = logging.getLogger(__name__)


@dataclass
class Anchor:
    a_id: str
    q_probe: str
    r: str
    r_rowcount: int
    r_signature: str
    c: Dict[str, Any]
    U_a: List[str]
    scope_id: str
    category: str
    numeric_variance: float = 0.0


@dataclass
class Synopsis:
    Ent: List[str]
    Kind: str
    Repr: List[str]
    WfPrior: Dict[str, Any]


@dataclass
class Scope:
    scope_id: str
    units: List[str]
    synopsis: Optional[Synopsis] = None


@dataclass
class Hierarchy:
    units: Dict[str, SourceUnit]
    scopes: Dict[str, Scope]
    anchors: Dict[str, Anchor]

    def anchors_in_scope(self, scope_id: str) -> List[Anchor]:
        return [a for a in self.anchors.values() if a.scope_id == scope_id]


@dataclass
class TaskPackage:
    x: str
    path: List[str]
    level: int
    hint: Dict[str, Any]
    question_type: str
    context: str = ""
    deliverable: str = ""
    ground_truth: str = ""
    source_span: float = 0.0
    format_span: float = 0.0


@dataclass
class SynthesizedTask:
    task_id: str
    question: str
    context: str
    deliverable: str
    env: Dict[str, Any]
    ground_truth: str = ""
    round_idx: int = 0
    question_type: str = ""

    level: int = 1
    path: List[str] = field(default_factory=list)


    edge_kinds: List[str] = field(default_factory=list)
    hint: Dict[str, Any] = field(default_factory=dict)
    source_span: float = 0.0
    format_span: float = 0.0


    answer_key: str = ""
    answer_key_rows: List[Dict[str, Any]] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnswerContract:
    """One executable source of truth for a synthesized task."""
    exec_db: str
    query_sql: str
    display_sql: str
    target_clause: str
    key_rows: List[Dict[str, Any]]
    key_stdout: str
    required_units: List[str]


def _fingerprint(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _nontrivial(output: str) -> Tuple[bool, int, float]:
    lines = [ln for ln in output.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False, 0, 0.0

    data_lines = lines[2:] if re.match(r"^-+\s", lines[1]) else lines[1:]
    if len(data_lines) < 2:
        return False, len(data_lines), 0.0


    if len(set(data_lines)) == 1:
        return False, len(data_lines), 0.0


    nums: List[float] = []
    for ln in data_lines:
        for tok in ln.split():
            try:
                nums.append(float(tok))
                break
            except ValueError:
                continue
    variance = 0.0
    if len(nums) >= 2:
        mean = sum(nums) / len(nums)
        variance = sum((x - mean) ** 2 for x in nums) / len(nums)

    if variance <= 0 and len(set(data_lines)) < 2:
        return False, len(data_lines), variance
    return True, len(data_lines), variance


def _probe_unit_ranking(u: SourceUnit) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for col, typ, _ in u.sch:
        col_stat = u.stat.get("cols", {}).get(col, {})
        if _is_category(col, typ, col_stat):
            q = (f'SELECT "{col}", COUNT(*) AS n FROM "{u.logical_table}" '
                 f'WHERE "{col}" IS NOT NULL GROUP BY "{col}" ORDER BY n DESC LIMIT 10;')
            out.append((q, {"tables_used": [u.logical_table], "col_used": [col]}))
            break
    return out


def _is_num_type(t: str) -> bool:
    t = (t or "").upper()
    return any(k in t for k in ("INT", "REAL", "FLOAT", "DOUBLE", "NUM", "DEC"))


def _is_date_col(col: str, t: str) -> bool:
    if "DATE" in (t or "").upper() or "TIME" in (t or "").upper():
        return True
    return bool(re.search(r"date|time|year|month|day", col or "", re.IGNORECASE))


def _is_category(col: str, typ: str, col_stat: Dict[str, Any]) -> bool:
    if _is_num_type(typ) or _is_date_col(col, typ):
        return False
    d = col_stat.get("distinct_ratio", 0.0) or 0.0
    return 0.0 < d < 0.5


def _probe_unit_temporal(u: SourceUnit) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    date_col = next((c for c, t, _ in u.sch if _is_date_col(c, t)), None)
    if not date_col:
        return out
    num_col = next(
        (c for c, t, _ in u.sch if _is_num_type(t) and c != date_col),
        None,
    )
    if num_col:
        q = (f'SELECT "{date_col}", SUM("{num_col}") AS total '
             f'FROM "{u.logical_table}" WHERE "{date_col}" IS NOT NULL '
             f'GROUP BY "{date_col}" ORDER BY "{date_col}" LIMIT 20;')
        out.append((q, {"tables_used": [u.logical_table], "col_used": [date_col, num_col]}))
    else:
        q = (f'SELECT "{date_col}", COUNT(*) AS n FROM "{u.logical_table}" '
             f'WHERE "{date_col}" IS NOT NULL GROUP BY "{date_col}" '
             f'ORDER BY "{date_col}" LIMIT 20;')
        out.append((q, {"tables_used": [u.logical_table], "col_used": [date_col]}))
    return out


def _probe_unit_association(u: SourceUnit) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    cats = [
        c for c, t, _ in u.sch
        if _is_category(c, t, u.stat.get("cols", {}).get(c, {}))
    ]
    nums = [c for c, t, _ in u.sch if _is_num_type(t)]
    if len(cats) >= 1 and len(nums) >= 1:
        c, n = cats[0], nums[0]
        q = (f'SELECT "{c}", AVG("{n}") AS avg_v '
             f'FROM "{u.logical_table}" WHERE "{c}" IS NOT NULL AND "{n}" IS NOT NULL '
             f'GROUP BY "{c}" ORDER BY avg_v DESC LIMIT 10;')
        out.append((q, {"tables_used": [u.logical_table], "col_used": [c, n]}))
    elif len(cats) >= 2:
        c1, c2 = cats[0], cats[1]
        q = (f'SELECT CAST("{c1}" AS TEXT) || \' / \' || CAST("{c2}" AS TEXT) '
             f'AS label, COUNT(*) AS n FROM "{u.logical_table}" '
             f'WHERE "{c1}" IS NOT NULL AND "{c2}" IS NOT NULL '
             f'GROUP BY "{c1}", "{c2}" ORDER BY n DESC LIMIT 10;')
        out.append((q, {"tables_used": [u.logical_table], "col_used": [c1, c2]}))
    return out


def _synopsize(scope: Scope, units_by_id: Dict[str, SourceUnit],
               anchors: List[Anchor]) -> Synopsis:
    ent_set: List[str] = []
    for uid in scope.units:
        ent_set.extend(units_by_id[uid].ent)
    ent = list(dict.fromkeys(ent_set))[:32]

    cat_counts: Dict[str, int] = {}
    for a in anchors:
        cat_counts[a.category] = cat_counts.get(a.category, 0) + 1
    if cat_counts:
        dominant = max(cat_counts.items(), key=lambda kv: kv[1])[0]
    else:
        dominant = "inspect"
    kind = {
        "ranking": "comparison",
        "temporal": "temporal",
        "association": "attribution",
        "anomaly": "narrative",
        "cross_source": "attribution",
    }.get(dominant, "comparison")

    repr_anchors = sorted(
        anchors,
        key=lambda a: (a.r_rowcount * (a.numeric_variance + 1.0)),
        reverse=True,
    )[:3]
    repr_ids = [a.a_id for a in repr_anchors]


    sorted_units = sorted(
        [units_by_id[uid] for uid in scope.units],
        key=lambda u: u.stat.get("n_rows", 0),
        reverse=True,
    )
    entry_units = [u.unit_id for u in sorted_units[:2]]
    first_format = sorted_units[0].format if sorted_units else ""
    first_tool = "list_tables" if first_format == "sqlite_table" else "get_table_schema"
    preferred_ops = {
        "comparison": ["inspect", "aggregate"],
        "temporal":   ["inspect", "aggregate"],
        "attribution": ["inspect", "join", "aggregate"],
        "narrative":  ["inspect", "verify"],
    }.get(kind, ["inspect", "aggregate"])
    wf_prior = {
        "entry_units": entry_units,
        "first_tool": first_tool,
        "preferred_ops": preferred_ops,
    }

    return Synopsis(Ent=ent, Kind=kind, Repr=repr_ids, WfPrior=wf_prior)


def _diversity_sample_units(units: List[SourceUnit], cap: int) -> List[SourceUnit]:
    if len(units) <= cap:
        return units

    by_rows_desc = sorted(units, key=lambda u: u.stat.get("n_rows", 0), reverse=True)
    large = by_rows_desc[:8]

    small_nontrivial = [
        u for u in by_rows_desc
        if 3 <= u.stat.get("n_rows", 0) < by_rows_desc[0].stat.get("n_rows", 0)
    ]
    small = sorted(small_nontrivial, key=lambda u: u.stat.get("n_rows", 0))[:8]

    family_cover: List[SourceUnit] = []
    seen_formats: set = set()
    for u in by_rows_desc:
        if u.format not in seen_formats:
            seen_formats.add(u.format)
            family_cover.append(u)

    picked: List[SourceUnit] = []
    seen_ids: set = set()
    for batch in (family_cover, large, small):
        for u in batch:
            if u.unit_id in seen_ids:
                continue
            seen_ids.add(u.unit_id)
            picked.append(u)
            if len(picked) >= cap:
                return picked

    for u in by_rows_desc:
        if len(picked) >= cap:
            break
        if u.unit_id in seen_ids:
            continue
        seen_ids.add(u.unit_id)
        picked.append(u)
    return picked


def _sources_of_path(H: Hierarchy, anchor_tuple: Tuple[Anchor, ...]) -> List[str]:
    uids: List[str] = []
    for a in anchor_tuple:
        uids.extend(a.U_a)

    for sid in {a.scope_id for a in anchor_tuple}:
        uids.extend(H.scopes[sid].units)
    return list(dict.fromkeys(uids))


def _admit_stable(H: Hierarchy, anchor_tuple: Tuple[Anchor, ...]) -> bool:
    for a in anchor_tuple:
        exec_db = a.c.get("exec_db") or H.units[a.U_a[0]].db_path
        replay_sql = str(a.c.get("stable_sql") or a.q_probe)
        expected_signature = str(
            a.c.get("stable_signature") or a.r_signature
        )
        expected_lines = int(
            a.c.get("stable_line_count", a.r.count("\n"))
        )
        res = execute_sql(exec_db, replay_sql)
        if not res.success:
            return False
        if abs(res.stdout.count("\n") - expected_lines) > 2:
            return False
        if _fingerprint(res.stdout) != expected_signature:
            return False
    return True


def _admit_accessible(H: Hierarchy, anchor_tuple: Tuple[Anchor, ...]) -> bool:
    uids = _sources_of_path(H, anchor_tuple)
    for uid in uids:
        unit = H.units.get(uid)
        if not unit:
            return False
        if not Path(unit.db_path).is_file():
            return False
        res = list_tables(unit.db_path)
        if not res.success:
            return False
        table_names = set(res.stdout.split())
        if unit.logical_table not in table_names:
            return False
    return True


_NUM_TOK = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for tok in _NUM_TOK.findall(text or ""):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _numeric_match(answer: str, anchor: Anchor) -> bool:


    a_nums = _extract_numbers(answer)
    r_nums = _extract_numbers(anchor.r)
    if not a_nums or not r_nums:
        return False
    for x in a_nums:
        for y in r_nums:
            if y == 0:
                if abs(x) < 1e-9:
                    return True
            elif abs(x - y) / max(abs(y), 1e-9) <= TAU_ND:
                return True
    return False


def _label_match(answer: str, anchor: Anchor) -> bool:
    for lab in _labels_from_result(anchor.r):
        if re.search(rf"\b{re.escape(lab)}\b", answer, re.IGNORECASE):
            return True
    return False


def _hint_blurb(task: TaskPackage) -> str:
    units = (task.hint or {}).get("entry_units") or []
    return ", ".join(str(u) for u in units)


def _admit_nontrivial(
    task: TaskPackage, H: Hierarchy, anchor_tuple: Tuple[Anchor, ...],
    client: OpenRouterClient,
) -> bool:


    blurb = _hint_blurb(task)
    if not blurb:
        synopses_blurb: List[str] = []
        for sid in sorted({a.scope_id for a in anchor_tuple}):
            syn = H.scopes[sid].synopsis
            if not syn:
                continue
            synopses_blurb.append(
                f"scope {sid}: kind={syn.Kind}, workflow_prior={syn.WfPrior}"
            )
        blurb = "\n".join(synopses_blurb)
    prompt = (
        "Answer the analytical question using ONLY the information below. "
        "If it is not enough, reply exactly with NONE.\n\n"
        f"Question: {task.x}\n\nData sources: " + blurb +
        "\n\nAnswer:"
    )


    content = None
    for attempt in (1, 2):
        try:
            content, _usage = client.call(
                [{"role": "user", "content": prompt}],
                model=MODELS["cost_effective"], max_tokens=256, temperature=0.0,
            )
            break
        except Exception as exc:
            if attempt == 2:
                log.warning("NonDeg probe failed twice (%s); rejecting", exc)
                return False
            log.warning("NonDeg probe failed (%s); retrying once", exc)
    content = (content or "").strip()
    if not content or content.upper().startswith("NONE"):
        return True
    for a in anchor_tuple:
        if _numeric_match(content, a):
            return False


        if not _extract_numbers(a.r) and _label_match(content, a):
            return False
    return True


def _admit_solvable(
    task: TaskPackage, H: Hierarchy, anchor_tuple: Tuple[Anchor, ...],
) -> bool:
    uids = _sources_of_path(H, anchor_tuple)
    units = [H.units[u] for u in uids if u in H.units]
    if not units or not anchor_tuple:
        return False
    b_skel = B_SKEL_BASE + len(units)
    steps = 0

    res = list_tables(units[0].db_path)
    steps += 1
    if not res.success:
        return False

    n_schema = min(2, len(units), max(0, b_skel - steps - len(anchor_tuple)))
    for u in units[:n_schema]:
        _ = get_table_schema(u.db_path, u.logical_table)
        steps += 1

    for a in anchor_tuple:
        if steps >= b_skel:
            return False
        exec_db = a.c.get("exec_db") or H.units[a.U_a[0]].db_path
        replay_sql = str(a.c.get("stable_sql") or a.q_probe)
        res = execute_sql(exec_db, replay_sql)
        steps += 1
        if not res.success or not res.stdout.strip():
            return False
    return True


def _scratch_db_for(env: Dict[str, Any]) -> str:
    db_path = env.get("db_path") or env.get("data_file") or ""
    if db_path and db_path.lower().endswith((".sqlite", ".db", ".sqlite3")):
        scratch_parent = Path(db_path).parent / ".toffee_scratch"
    else:
        scratch_parent = Path("/tmp/toffee_scratch")
    source_id = env.get("source_id") or "env"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    return str(scratch_parent / f"env_{source_id}.sqlite")


_SIBLING_EXTS = (".csv", ".tsv", ".xlsx", ".xls", ".xlsm", ".md", ".markdown")


def _collect_sources(env: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for key in ("db_path", "data_file"):
        v = env.get(key)
        if v and v not in paths:
            paths.append(v)
    for v in env.get("extra_sources") or []:
        if v and v not in paths:
            paths.append(v)


    primary = paths[0] if paths else None
    if primary:
        try:
            primary_dir = Path(primary).parent
            if primary_dir.is_dir():
                for child in sorted(primary_dir.iterdir()):
                    if not child.is_file():
                        continue
                    if child.suffix.lower() not in _SIBLING_EXTS:
                        continue
                    cp = str(child)
                    if cp not in paths:
                        paths.append(cp)
        except Exception:
            pass
    return paths


def _parse_llm_json(content: str) -> Optional[Dict[str, str]]:
    content = (content or "").strip()


    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if content.startswith("<think>"):
        brace = content.find("{")
        content = content[brace:] if brace >= 0 else content
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        log.warning("Depgraph realize JSON parse failed; first 200 chars: %r", content[:200])
        return None


def _qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_body(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def _teacher_sql_view(sql: str) -> str:
    """Return a query view with data literals and file paths removed."""
    view = re.sub(r"'(?:''|[^'])*'", "'<value>'", sql or "")
    view = re.sub(
        r"\bIN\s*\((?!\s*SELECT\b)[^)]*\)", "IN (<values>)", view,
        flags=re.IGNORECASE,
    )
    view = re.sub(
        r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])",
        "<number>", view,
    )
    return view


def _humanize_identifier(name: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name or ""))
    text = re.sub(r"[_\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _probe_target_clause(u: SourceUnit, sql: str, meta: Dict[str, Any]) -> str:
    cols = list((meta or {}).get("col_used") or [])
    first = _humanize_identifier(cols[0]) if cols else "group"
    second = _humanize_identifier(cols[1]) if len(cols) > 1 else "records"
    low = (sql or "").lower()
    if "avg(" in low:
        return f"report the average {second} for every {first}"
    if "sum(" in low:
        return f"report the total {second} for every {first}"
    if len(cols) >= 2 and "||" in low:
        return f"report the record count for every {first} and {second} pair"
    return f"report the record count for every {first}"


def _mk_path_anchor(
    sql: str, exec_db: str, unit_ids: List[str], category: str,
    units_by_id: Dict[str, SourceUnit], *, answer_sql: Optional[str] = None,
    display_sql: Optional[str] = None, target_clause: str = "",
    composed: bool = False,
) -> Optional[Anchor]:
    res = execute_sql(exec_db, sql)
    if not res.success or not res.stdout.strip():
        return None
    tables = [units_by_id[u].logical_table for u in unit_ids if u in units_by_id]
    _keep, rowcount, variance = _nontrivial(res.stdout)
    a_id = "a_" + hashlib.sha1(sql.encode()).hexdigest()[:10]
    contract_meta = {
        "exec_db": exec_db,
        "category": category,
        "tables_used": tables,
        "answer_sql": answer_sql or sql,
        "display_sql": display_sql or _teacher_sql_view(sql),
        "target_clause": target_clause,
        "composed": composed,
    }
    return Anchor(
        a_id=a_id, q_probe=sql, r=res.stdout[:2000], r_rowcount=rowcount,
        r_signature=_fingerprint(res.stdout),
        c=contract_meta,
        U_a=unit_ids, scope_id="g00", category=category, numeric_variance=variance,
    )


def _chain_queries(
    edge, units_by_id: Dict[str, SourceUnit],
) -> Optional[Tuple[str, str, str, str]]:
    src = units_by_id.get(edge.src)
    dst = units_by_id.get(edge.dst)
    if not src or not dst or src.db_path != dst.db_path:
        return None
    probe = _sql_body(edge.meta.get("probe_sql", ""))
    bridge_col = edge.meta.get("bridge_col")
    target_col = edge.meta.get("target_col")
    if not probe or not bridge_col or not target_col:
        return None
    prefix = (
        f"WITH bridge AS ({probe}) "
        f"SELECT t.{_qident(target_col)} AS {_qident(target_col)}, "
        f"COUNT(*) AS matching_rows FROM {_qident(dst.logical_table)} AS t "
        f"JOIN bridge AS b ON t.{_qident(target_col)} = b.{_qident(bridge_col)} "
        f"GROUP BY t.{_qident(target_col)} "
        f"ORDER BY matching_rows DESC, t.{_qident(target_col)}"
    )
    run_sql = prefix + " LIMIT 20;"
    answer_sql = prefix + ";"
    target = (
        f"report the matching {dst.title} row count for every group selected "
        "by the upstream computation"
    )
    return run_sql, answer_sql, _teacher_sql_view(run_sql), target


def _execute_edge(edge, units_by_id: Dict[str, SourceUnit]) -> Optional[List[Anchor]]:
    if edge.kind == "join":
        src, dst = units_by_id.get(edge.src), units_by_id.get(edge.dst)
        if not src or not dst:
            return None
        target = "report the matching row count for every shared group"
        a = _mk_path_anchor(edge.meta["join_sql"], edge.meta["exec_db"],
                            [edge.src, edge.dst], "join", units_by_id,
                            answer_sql=_strip_row_cap(edge.meta["join_sql"]),
                            target_clause=target)
        return [a] if a else None
    compiled = _chain_queries(edge, units_by_id)
    if compiled is None:
        return None
    run_sql, answer_sql, display_sql, target = compiled
    src = units_by_id[edge.src]
    probe_meta = dict(edge.meta.get("probe_meta") or {})
    probe = _mk_path_anchor(edge.meta["probe_sql"], edge.meta["probe_exec_db"],
                            [edge.src], "probe", units_by_id,
                            answer_sql=_strip_row_cap(edge.meta["probe_sql"]),
                            target_clause=_probe_target_clause(
                                src, edge.meta["probe_sql"], probe_meta))
    verify = _mk_path_anchor(
        run_sql, src.db_path, [edge.src, edge.dst], "chain", units_by_id,
        answer_sql=answer_sql, display_sql=display_sql, target_clause=target,
        composed=True,
    )
    if not probe or not verify:
        return None
    return [probe, verify]


def _labels_from_result(text: str) -> List[str]:
    from toffee.generation.depgraph import parse_column_output
    try:
        _header, rows = parse_column_output(text or "")
    except Exception:
        return []
    labels: List[str] = []
    seen = set()
    for row in rows:
        for cell in row:
            cell = (cell or "").strip()
            if not cell or len(cell) > 40:
                continue
            try:
                float(cell.replace(",", ""))
                continue
            except ValueError:
                pass
            key = cell.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(cell)
            if len(labels) >= 8:
                return labels
    return labels


def _rows_from_result(text: str) -> List[Dict[str, Any]]:
    from toffee.generation.depgraph import parse_column_output
    try:
        _header, rows = parse_column_output(text or "")
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for row in rows[:12]:
        labels: List[str] = []
        nums: List[float] = []
        for cell in row:
            cell = (cell or "").strip()
            if not cell or len(cell) > 40:
                continue
            try:
                nums.append(float(cell.replace(",", "")))
            except ValueError:
                labels.append(cell)
        if labels or nums:
            out.append({"labels": labels[:4], "nums": nums[:8]})
    return out


def _provenance_from_anchors(anchors: Sequence[Anchor]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for a in anchors:
        facts.append({
            "anchor": a.a_id,
            "tables": list(a.c.get("tables_used", [])),
            "unit_ids": list(a.U_a),
            "query": str(a.c.get("display_sql") or _teacher_sql_view(a.q_probe)),
            "depends_on": list(a.c.get("tables_used", [])),
            "composed": bool(a.c.get("composed")),
            "nums": _extract_numbers(a.r)[:12],
            "labels": _labels_from_result(a.r),
            "rows": _rows_from_result(a.r),
        })
    return facts


_KEY_TOL = 0.01


_KEY_ROW_CAP = 2000
_LIMIT_RE = re.compile(r"\blimit\s+\d+(\s*,\s*\d+)?(\s+offset\s+\d+)?", re.IGNORECASE)


def _strip_row_cap(sql: str) -> str:
    if _LIMIT_RE.search(sql or ""):
        return _LIMIT_RE.sub(f"LIMIT {_KEY_ROW_CAP}", sql or "", count=1)
    return sql or ""


def _key_rows_from_stdout(stdout: str) -> List[Dict[str, Any]]:
    from toffee.generation.depgraph import parse_column_output
    try:
        _header, rows = parse_column_output(stdout or "")
    except Exception:
        return []
    if len(_header) not in (1, 2):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows[:_KEY_ROW_CAP]:
        if not row:
            continue
        label: Optional[str] = str(row[0]).strip() or None
        value: Optional[float] = None
        if len(_header) == 2:
            if len(row) < 2 or not str(row[1]).strip():
                return []
            try:
                value = float(str(row[1]).strip().replace(",", ""))
            except ValueError:
                return []
        if label is not None or value is not None:
            out.append({"label": label, "value": value,
                        "raw": " ".join(c.strip() for c in row if c.strip())})
    return out


def _final_exec_db(final_anchor: Anchor, units_by_id: Dict[str, SourceUnit]) -> str:
    db = final_anchor.c.get("exec_db")
    if db:
        return db
    for uid in final_anchor.U_a:
        u = units_by_id.get(uid)
        if u:
            return u.db_path
    return ""


def _compile_answer_key(
    final_anchor: Anchor, units_by_id: Dict[str, SourceUnit],
) -> Tuple[List[Dict[str, Any]], str]:
    exec_db = _final_exec_db(final_anchor, units_by_id)
    capfree = str(final_anchor.c.get("answer_sql") or
                  _strip_row_cap(final_anchor.q_probe))
    stdout = final_anchor.r
    truncated = False
    if exec_db:
        res = execute_sql(exec_db, capfree)
        if res.success and res.stdout.strip():
            stdout = res.stdout
            truncated = res.truncated
    rows = _key_rows_from_stdout(stdout)
    # The key must hold the full result. A re-execution that hits the
    # operational row bound or the output cap is truncated, not full, so the
    # path is dropped (the empty key_rows fails the guessable-key check).
    if truncated or len(rows) >= _KEY_ROW_CAP:
        return [], stdout
    return rows, stdout


def _compile_answer_contract(
    final_anchor: Anchor, units_by_id: Dict[str, SourceUnit],
) -> Optional[AnswerContract]:
    key_rows, key_stdout = _compile_answer_key(final_anchor, units_by_id)
    if not key_rows:
        return None
    query_sql = str(final_anchor.c.get("answer_sql") or final_anchor.q_probe)
    display_sql = str(final_anchor.c.get("display_sql") or
                      _teacher_sql_view(query_sql))
    target_clause = str(final_anchor.c.get("target_clause") or "").strip()
    if not target_clause:
        return None
    # Admission must replay the same complete query that produced the key, not
    # the row-limited construction preview stored in ``q_probe``.
    final_anchor.c["stable_sql"] = query_sql
    final_anchor.c["stable_signature"] = _fingerprint(key_stdout)
    final_anchor.c["stable_line_count"] = key_stdout.count("\n")
    return AnswerContract(
        exec_db=_final_exec_db(final_anchor, units_by_id),
        query_sql=query_sql,
        display_sql=display_sql,
        target_clause=target_clause,
        key_rows=key_rows,
        key_stdout=key_stdout,
        required_units=list(dict.fromkeys(final_anchor.U_a)),
    )


_GUESSABLE_LABELS = {"yes", "no", "true", "false", "none", "null", "0", "1"}


def _key_guessable(key_rows: List[Dict[str, Any]]) -> bool:
    """A key that could be matched by coincidence, one whose values are only
    0 or 1 or whose labels are bare boolean tokens, is dropped."""
    if not key_rows:
        return True
    vals = [r.get("value") for r in key_rows if r.get("value") is not None]
    labels = [str(r.get("label") or "").strip().lower()
              for r in key_rows if r.get("label")]
    if vals:
        return all(v in (0.0, 1.0) for v in vals)
    return bool(labels) and all(l in _GUESSABLE_LABELS for l in labels)


def _hidden_key_values(key_rows: Sequence[Dict[str, Any]]) -> List[str]:
    hidden: List[str] = []
    for row in key_rows:
        label = str(row.get("label") or "").strip()
        if label:
            hidden.append(label)
        value = row.get("value")
        if value is None:
            continue
        number = float(value)
        hidden.append(str(number))
        if number.is_integer():
            hidden.append(str(int(number)))
    return list(dict.fromkeys(hidden))


def _margin_ok(final_anchor: Anchor, key_stdout: str) -> bool:
    if final_anchor.c.get("composed"):
        return True
    m = re.search(r"order\s+by\b.*\blimit\s+(\d+)", final_anchor.q_probe or "",
                  re.IGNORECASE | re.DOTALL)
    if not m:
        return True
    k = int(m.group(1))
    vals = [r["value"] for r in _key_rows_from_stdout(key_stdout)
            if r["value"] is not None]
    if len(vals) <= k:
        return True
    vk, vk1 = vals[k - 1], vals[k]
    denom = max(abs(vk), abs(vk1), 1e-9)
    return abs(vk - vk1) / denom > _KEY_TOL


def _applicable_question_types(path, anchors: Sequence[Anchor], graph) -> List[str]:
    fams: List[str] = []
    final = anchors[-1]
    final_rows = _rows_from_result(final.r)
    n_groups = sum(1 for r in final_rows if r.get("labels"))
    has_measure = any(r.get("nums") for r in final_rows)
    if has_measure and n_groups >= 2:
        fams.append("comparison")
    if has_measure:
        fams.append("verification")
    return list(dict.fromkeys(fams))


def _decoy_reproducible(key_rows: List[Dict[str, Any]], path, graph) -> bool:
    if not key_rows:
        return False
    touched = set(getattr(path, "nodes", []) or [])
    label_samples: Set[str] = set()
    value_samples: List[float] = []
    for uid, u in graph.units.items():
        if uid in touched:
            continue
        for _col, st in (u.stat.get("cols", {}) or {}).items():
            for lab in st.get("top_categories", []) or []:
                label_samples.add(str(lab).lower())
            for sv in st.get("sample_values", []) or []:
                try:
                    value_samples.append(float(str(sv).replace(",", "")))
                except ValueError:
                    continue
    if not label_samples and not value_samples:
        return False
    for r in key_rows:
        lab, val = r.get("label"), r.get("value")
        if lab is not None and str(lab).lower() not in label_samples:
            return False
        if val is not None and not any(
                abs(val - s) <= _KEY_TOL * max(abs(val), abs(s), 1e-9)
                for s in value_samples):
            return False
    return True


def _hint_for_path(path, graph) -> Dict[str, Any]:
    entry_titles = [graph.units[uid].title for uid in dict.fromkeys(path.nodes)
                    if uid in graph.units]
    return {"entry_units": entry_titles}


def _join_endpoint_cols(edge) -> Optional[Tuple[str, str]]:
    endpoints = dict(edge.meta.get("endpoint_cols") or {})
    if edge.src in endpoints and edge.dst in endpoints:
        return str(endpoints[edge.src]), str(endpoints[edge.dst])
    pair = list(edge.meta.get("col_pair") or [])
    if len(pair) == 2:
        return str(pair[0]), str(pair[1])
    return None


def _level3_queries(
    path, graph, empty_unit: Optional[str] = None,
) -> Optional[Tuple[str, str, str, str, str]]:
    """Compile A->B->C into one query whose terminal edge consumes a CTE."""
    if path.level != 3 or len(path.nodes) != 3 or len(path.edges) != 2:
        return None
    if len(set(path.nodes)) != 3:
        return None
    first, terminal = path.edges
    if terminal.kind != "chain" or first.dst != terminal.src:
        return None
    units = graph.units
    if any(uid not in units for uid in path.nodes):
        return None
    a, b, c = (units[uid] for uid in path.nodes)
    if len({a.db_path, b.db_path, c.db_path}) != 1:
        return None

    def table_ref(u: SourceUnit, alias: str) -> str:
        table = _qident(u.logical_table)
        if empty_unit == u.unit_id:
            return f"(SELECT * FROM {table} WHERE 0) AS {alias}"
        return f"{table} AS {alias}"

    second_bridge = terminal.meta.get("bridge_col")
    final_col = terminal.meta.get("target_col")
    if not second_bridge or not final_col:
        return None

    ctes: List[str] = []
    if first.kind == "join":
        join_cols = _join_endpoint_cols(first)
        if join_cols is None:
            return None
        a_col, b_col = join_cols
        bridge_sql = (
            f"SELECT b.{_qident(second_bridge)} AS k, COUNT(*) AS support "
            f"FROM {table_ref(a, 'a')} JOIN {table_ref(b, 'b')} "
            f"ON a.{_qident(a_col)} = b.{_qident(b_col)} "
            f"WHERE b.{_qident(second_bridge)} IS NOT NULL "
            f"GROUP BY b.{_qident(second_bridge)} "
            f"ORDER BY support DESC, k LIMIT {BRIDGE_MAX}"
        )
    elif first.kind == "chain":
        upstream_probe = _sql_body(first.meta.get("probe_sql", ""))
        upstream_bridge = first.meta.get("bridge_col")
        middle_col = first.meta.get("target_col")
        if not upstream_probe or not upstream_bridge or not middle_col:
            return None
        if empty_unit == a.unit_id:
            # Recompile the known probe against an empty view of A.  All probe
            # templates name A once in their FROM clause.
            table_pat = re.compile(
                rf'(?i)(\bFROM\s+){re.escape(_qident(a.logical_table))}'
            )
            upstream_probe, n_sub = table_pat.subn(
                rf'\1(SELECT * FROM {_qident(a.logical_table)} WHERE 0)',
                upstream_probe, count=1,
            )
            if n_sub != 1:
                return None
        ctes.append(f"upstream AS ({upstream_probe})")
        bridge_sql = (
            f"SELECT b.{_qident(second_bridge)} AS k, COUNT(*) AS support "
            f"FROM {table_ref(b, 'b')} JOIN upstream AS u "
            f"ON b.{_qident(middle_col)} = u.{_qident(upstream_bridge)} "
            f"WHERE b.{_qident(second_bridge)} IS NOT NULL "
            f"GROUP BY b.{_qident(second_bridge)} "
            f"ORDER BY support DESC, k LIMIT {BRIDGE_MAX}"
        )
    else:
        return None

    ctes.append(f"bridge AS ({bridge_sql})")
    prefix = (
        f"WITH {', '.join(ctes)} "
        f"SELECT c.{_qident(final_col)} AS {_qident(final_col)}, "
        f"COUNT(*) AS matching_rows FROM {table_ref(c, 'c')} "
        f"JOIN bridge AS x ON c.{_qident(final_col)} = x.k "
        f"GROUP BY c.{_qident(final_col)} "
        f"ORDER BY matching_rows DESC, c.{_qident(final_col)}"
    )
    run_sql = prefix + " LIMIT 20;"
    answer_sql = prefix + ";"
    target = (
        f"report the matching {c.title} row count for every group selected "
        "by the preceding computation"
    )
    return a.db_path, run_sql, answer_sql, _teacher_sql_view(run_sql), target


def _level3_source_dependence_ok(path, graph, normal_output: str) -> bool:
    normal = "\n".join(line.rstrip() for line in normal_output.splitlines()).strip()
    for uid in path.nodes:
        compiled = _level3_queries(path, graph, empty_unit=uid)
        if compiled is None:
            return False
        exec_db, run_sql, _answer_sql, _display_sql, _target = compiled
        result = execute_sql(exec_db, run_sql)
        if not result.success:
            return False
        changed = "\n".join(
            line.rstrip() for line in result.stdout.splitlines()
        ).strip()
        if changed == normal:
            return False
    return True


def _execute_path(path, graph) -> Optional[List[Anchor]]:
    units_by_id = graph.units
    if path.level == 1:
        from toffee.generation.depgraph import _cheapest_probe
        u = units_by_id.get(path.nodes[0])
        if not u:
            return None
        probe = _cheapest_probe(u)
        if not probe:
            return None
        a = _mk_path_anchor(
            probe[0], u.db_path, [u.unit_id], "probe", units_by_id,
            answer_sql=_strip_row_cap(probe[0]),
            target_clause=_probe_target_clause(u, probe[0], probe[1]),
        )
        return [a] if a else None
    if path.level == 3:
        lead = _execute_edge(path.edges[0], units_by_id)
        compiled = _level3_queries(path, graph)
        if not lead or compiled is None:
            return None
        exec_db, run_sql, answer_sql, display_sql, target = compiled
        final = _mk_path_anchor(
            run_sql, exec_db, list(path.nodes), "composed", units_by_id,
            answer_sql=answer_sql, display_sql=display_sql,
            target_clause=target, composed=True,
        )
        if final is None or not _level3_source_dependence_ok(path, graph, final.r):
            return None
        return lead + [final]
    anchors: List[Anchor] = []
    for edge in path.edges:
        step = _execute_edge(edge, units_by_id)
        if not step:
            return None
        anchors.extend(step)
    return anchors or None


def _hierarchy_for_path(path, graph, anchors: List[Anchor]) -> Hierarchy:
    node_ids = list(dict.fromkeys(path.nodes))
    scope = Scope(scope_id="g00", units=node_ids)
    scope.synopsis = _synopsize(scope, graph.units, anchors)
    return Hierarchy(
        units=graph.units,
        scopes={"g00": scope},
        anchors={a.a_id: a for a in anchors},
    )


_TYPE_TEMPLATES: Dict[str, str] = {
    "comparison": "Compare the same quantitative measure across the returned "
                  "groups and report each value.",
    "verification": "Report the computed measure so it can be checked against "
                    "an operational expectation.",
}


_FRAMING_OPTIONS: Dict[str, Tuple[str, str]] = {
    "operations_planning": ("the operations team", "operational planning"),
    "resource_allocation": ("the planning team", "resource allocation"),
    "performance_review": ("the management team", "a performance review"),
    "risk_review": ("the risk team", "a risk review"),
    "financial_review": ("the finance team", "a financial review"),
    "program_review": ("the program team", "a program review"),
    "policy_review": ("the policy team", "a policy review"),
    "research_review": ("the research team", "a research review"),
    "service_planning": ("the service team", "service planning"),
    "quality_review": ("the quality team", "a quality review"),
}


def _framing_menu() -> str:
    return "\n".join(
        f"- {frame_id}: audience={audience}; decision={decision}"
        for frame_id, (audience, decision) in _FRAMING_OPTIONS.items()
    )

_DEPGRAPH_REALIZE_PROMPT = """You are a senior analytics lead drafting a high-stakes question over a data environment.

**Question type ({question_type}):** {op_template}
**Analysis path the answer must follow (structure only, intermediate values withheld):**
{path_desc}

**Value-free query view (the authoritative query, with no result values):**
{final_step}

**Required analytical target:** {target_clause}

**Tables on the path (name and columns):**
{schema_blurb}

Choose the ONE audience-and-decision frame below that best fits the fixed
computation. You choose only a frame; the system writes the question and
inserts the analytical target and source names itself.

**Allowed frames (return the id exactly):**
{framing_menu}

Hard rules:
- Do not write question text, background facts, values, or column names.
- Do not add, restate, or replace the analytical target.
- Do NOT mention any specific intermediate key value, id, code, or the final
  numeric answer. The bridge values that link the steps must be discovered by
  the analyst, never stated in the question.

If no coherent {question_type} question fits this path -- the operation cannot be
posed over these sources without inventing facts -- do NOT force one. Respond
with exactly: {{"decline": true}}

Otherwise respond with strict JSON:
{{
  "frame_id": "one_allowed_id"
}}
"""


def _realize_is_decline(parsed: Optional[Dict[str, Any]]) -> bool:
    if not parsed:
        return False
    if parsed.get("decline") in (True, "true", "True"):
        return True
    if str(parsed.get("frame_id", "") or "").strip():
        return False
    q = str(parsed.get("question_template", parsed.get("question", "")) or "").strip()
    return q == "" or q.upper() == "NONE"


def _source_phrase(path, graph) -> str:
    names: List[str] = []
    for uid in dict.fromkeys(getattr(path, "nodes", []) or []):
        unit = graph.units.get(uid)
        if not unit:
            continue
        name = re.sub(r"\s+", " ", str(unit.title or "")).strip()
        if name:
            names.append("`" + name.replace("`", "'") + "`")
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    if len(names) > 2:
        return ", ".join(names[:-1]) + f", and {names[-1]}"
    return ""


def _realize_depgraph(
    path, graph, client: OpenRouterClient, question_type: str,
    contract: AnswerContract,
) -> Optional[Dict[str, Any]]:
    from toffee.generation.depgraph import describe_path
    schema_lines: List[str] = []
    for uid in dict.fromkeys(path.nodes):
        u = graph.units.get(uid)
        if not u:
            continue
        cols = ", ".join(c for c, _t, _n in u.sch)
        schema_lines.append(f"  - {u.title}: {cols}")
    prompt = _DEPGRAPH_REALIZE_PROMPT.format(
        question_type=question_type,
        op_template=_TYPE_TEMPLATES.get(question_type, ""),
        path_desc=describe_path(path, graph),
        final_step=contract.display_sql,
        target_clause=contract.target_clause,
        schema_blurb="\n".join(schema_lines),
        framing_menu=_framing_menu(),
    )
    try:
        from toffee.config import TASK_REALIZE_MODEL
        content, _usage = client.call(
            [{"role": "user", "content": prompt}],
            model=TASK_REALIZE_MODEL, max_tokens=8192, temperature=0.7,
        )
    except Exception as exc:
        log.warning("Depgraph realize call failed: %s", exc)
        return None
    parsed = _parse_llm_json(content)
    if parsed is not None and not _realize_is_decline(parsed):
        frame_id = str(parsed.get("frame_id", "") or "").strip()
        frame = _FRAMING_OPTIONS.get(frame_id)
        sources = _source_phrase(path, graph)
        if frame is None or not sources:
            return None
        audience, decision = frame
        question = (
            f"For {audience}, {contract.target_clause} using {sources}. "
            f"The result will support {decision}."
        )
        return {
            "question": question,
            "context": "",
            "deliverable": contract.target_clause[:1].upper()
                           + contract.target_clause[1:] + ".",
            "_question_type": question_type,
            "_frame_id": frame_id,
        }
    return parsed


def _admission_battery(
    task: TaskPackage, H: Hierarchy, anchor_tuple: Tuple[Anchor, ...],
    client, stats: Dict[str, int], replay: bool = False,
) -> bool:
    """The four admission checks, run as one battery. Admission ends with an
    independent replay of the whole battery (replay=True): every path query
    re-executes and the probe is called afresh, so a task admits only when
    the battery passes twice."""
    if not _admit_stable(H, anchor_tuple):
        if not replay:
            stats["rej_stable"] += 1
        return False
    if not _admit_accessible(H, anchor_tuple):
        if not replay:
            stats["rej_accessible"] += 1
        return False
    if not _admit_nontrivial(task, H, anchor_tuple, client):
        if not replay:
            stats["rej_nontrivial"] += 1
        return False
    if not _admit_solvable(task, H, anchor_tuple):
        if not replay:
            stats["rej_solvable"] += 1
        return False
    return True


def _synthesize_tasks_depgraph(
    env: Dict[str, Any], client: OpenRouterClient, tasks_per_env: int,
    seed: Optional[int],
) -> List[SynthesizedTask]:
    from toffee.generation import depgraph as dg

    sources = _collect_sources(env)
    if not sources:
        log.warning("No sources in env; skipping synthesis")
        return []
    scratch_db = _scratch_db_for(env)
    source_id = env.get("source_id", "")

    log.info("Depgraph build start: %s (%d source files)", source_id, len(sources))
    units = ingest_environment(sources, scratch_db)
    if not units:
        log.info("No units for %s; skipping depgraph synthesis", source_id)
        return []
    if len(units) > MAX_UNITS_PER_ENV:
        units = _diversity_sample_units(units, MAX_UNITS_PER_ENV)

    graph = dg.build_dependency_graph(units)
    log.info("Depgraph %s: %s", source_id, graph.stats)
    if not graph.nodes:
        return []
    levels = dg.supported_levels(graph)

    rng = random.Random(seed) if seed is not None else random.Random()
    stats = {"sampled": 0, "realized": 0, "rej_decline": 0, "rej_leak": 0,
             "rej_guessable": 0, "rej_margin": 0, "rej_decoy": 0,
             "rej_stable": 0, "rej_accessible": 0, "rej_nontrivial": 0,
             "rej_solvable": 0, "rej_battery_replay": 0, "admitted": 0}
    out: List[SynthesizedTask] = []
    attempts = 0
    max_attempts = max(tasks_per_env * 4, 12)
    while len(out) < tasks_per_env and attempts < max_attempts:
        attempts += 1

        level = rng.choice((1, 2, 3))
        if level not in levels:
            level = 1
        path = dg.sample_path(graph, level, rng)
        if path is None:
            path = dg.sample_path(graph, 1, rng)
        if path is None:
            continue
        stats["sampled"] += 1
        anchors = _execute_path(path, graph)
        if not anchors:
            continue


        contract = _compile_answer_contract(anchors[-1], graph.units)
        if contract is None:
            continue
        key_rows, key_stdout = contract.key_rows, contract.key_stdout
        if _key_guessable(key_rows):
            stats["rej_guessable"] += 1
            continue
        if not _margin_ok(anchors[-1], key_stdout):
            stats["rej_margin"] += 1
            continue
        if _decoy_reproducible(key_rows, path, graph):
            stats["rej_decoy"] += 1
            continue

        applicable = _applicable_question_types(path, anchors, graph) or ["verification"]
        question_type = rng.choice(applicable)
        realized = _realize_depgraph(path, graph, client, question_type, contract)
        if _realize_is_decline(realized):
            remaining = [t for t in applicable if t != question_type]
            if not remaining:
                stats["rej_decline"] += 1
                continue
            question_type = rng.choice(remaining)
            realized = _realize_depgraph(path, graph, client, question_type, contract)
            if _realize_is_decline(realized):
                stats["rej_decline"] += 1
                continue
        if not realized or not realized.get("question"):
            continue
        stats["realized"] += 1
        question = str(realized.get("question", "")).strip()


        hidden_vals = dg.collect_bridge_values(path) + _hidden_key_values(key_rows)
        visible_payload = "\n".join([
            question,
            str(realized.get("context", "") or ""),
            str(realized.get("deliverable", "") or ""),
        ])
        if dg.question_leaks(visible_payload, hidden_vals):
            stats["rej_leak"] += 1
            continue

        H = _hierarchy_for_path(path, graph, anchors)
        anchor_tuple = tuple(anchors)
        qtype = realized.get("_question_type") or question_type
        task = TaskPackage(
            x=question,
            path=[n for n in path.nodes] + [a.a_id for a in anchors],
            level=path.level,
            hint=_hint_for_path(path, graph),
            question_type=qtype,
            context=str(realized.get("context", "")).strip(),
            deliverable=str(realized.get("deliverable", "")).strip() or "Provide the analysis result",
            ground_truth=contract.query_sql,
            source_span=len(set(path.nodes)) / 3.0,
            format_span=len({graph.units[uid].format for uid in path.nodes
                             if uid in graph.units}) / N_FORMATS,
        )

        if not _admission_battery(task, H, anchor_tuple, client, stats):
            continue
        if ADMISSION_REPLAY and not _admission_battery(
                task, H, anchor_tuple, client, stats, replay=True):
            stats["rej_battery_replay"] += 1
            continue

        stats["admitted"] += 1
        file_sources = [
            src for src in sources
            if Path(src).suffix.lower() not in (".sqlite", ".db", ".sqlite3")
        ]
        visible_files = list(dict.fromkeys(
            list(env.get("extra_sources") or []) + file_sources
        ))
        out.append(SynthesizedTask(
            task_id=f"depgraph_{uuid.uuid4().hex[:8]}",
            question=task.x,
            context=task.context,
            deliverable=task.deliverable,
            env={
                "db_path": scratch_db,
                "data_file": sources[0],
                "extra_sources": visible_files,
                "working_dir": str(Path(sources[0]).resolve().parent),
                "source_files": list(sources),
                "source_id": source_id,
                "scratch_db": scratch_db,
            },
            ground_truth=task.ground_truth,
            question_type=task.question_type,
            level=path.level,
            path=task.path,
            edge_kinds=path.edge_kinds(),
            hint=task.hint,
            source_span=task.source_span,
            format_span=task.format_span,


            answer_key=key_stdout[:800],
            answer_key_rows=key_rows,
            provenance=_provenance_from_anchors(anchors),
        ))
    log.info("Depgraph admission for %s: %s", source_id, stats)
    return out


def synthesize_tasks(
    env: Dict[str, Any], client: OpenRouterClient, tasks_per_env: int = 5,
    seed: Optional[int] = None,
) -> List[SynthesizedTask]:
    return _synthesize_tasks_depgraph(env, client, tasks_per_env, seed)


__all__ = [
    "SourceUnit", "Anchor", "Synopsis", "Scope", "Hierarchy",
    "TaskPackage", "SynthesizedTask",
    "synthesize_tasks",
]
