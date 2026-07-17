
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
        q = (f'SELECT "{c}", AVG("{n}") AS avg_v, COUNT(*) AS k '
             f'FROM "{u.logical_table}" WHERE "{c}" IS NOT NULL AND "{n}" IS NOT NULL '
             f'GROUP BY "{c}" ORDER BY avg_v DESC LIMIT 10;')
        out.append((q, {"tables_used": [u.logical_table], "col_used": [c, n]}))
    elif len(cats) >= 2:
        c1, c2 = cats[0], cats[1]
        q = (f'SELECT "{c1}", "{c2}", COUNT(*) AS n FROM "{u.logical_table}" '
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
        res = execute_sql(exec_db, a.q_probe)
        if not res.success:
            return False
        if abs(res.stdout.count("\n") - a.r.count("\n")) > 2:
            return False
        if _fingerprint(res.stdout) != a.r_signature:
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
        res = execute_sql(exec_db, a.q_probe)
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


def _mk_path_anchor(
    sql: str, exec_db: str, unit_ids: List[str], category: str,
    units_by_id: Dict[str, SourceUnit],
) -> Optional[Anchor]:
    res = execute_sql(exec_db, sql)
    if not res.success or not res.stdout.strip():
        return None
    tables = [units_by_id[u].logical_table for u in unit_ids if u in units_by_id]
    _keep, rowcount, variance = _nontrivial(res.stdout)
    a_id = "a_" + hashlib.sha1(sql.encode()).hexdigest()[:10]
    return Anchor(
        a_id=a_id, q_probe=sql, r=res.stdout[:2000], r_rowcount=rowcount,
        r_signature=_fingerprint(res.stdout),
        c={"exec_db": exec_db, "category": category, "tables_used": tables},
        U_a=unit_ids, scope_id="g00", category=category, numeric_variance=variance,
    )


def _execute_edge(edge, units_by_id: Dict[str, SourceUnit]) -> Optional[List[Anchor]]:
    if edge.kind == "join":
        a = _mk_path_anchor(edge.meta["join_sql"], edge.meta["exec_db"],
                            [edge.src, edge.dst], "join", units_by_id)
        return [a] if a else None
    probe = _mk_path_anchor(edge.meta["probe_sql"], edge.meta["probe_exec_db"],
                            [edge.src], "probe", units_by_id)
    verify = _mk_path_anchor(edge.meta["verify_sql"], edge.meta["verify_exec_db"],
                             [edge.dst], "chain", units_by_id)
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
    out: List[Dict[str, Any]] = []
    for row in rows[:_KEY_ROW_CAP]:
        label: Optional[str] = None
        value: Optional[float] = None
        for cell in row:
            cell = (cell or "").strip()
            if not cell:
                continue
            try:
                v = float(cell.replace(",", ""))
                if value is None:
                    value = v
            except ValueError:
                if label is None and len(cell) <= 60:
                    label = cell
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
    capfree = _strip_row_cap(final_anchor.q_probe)
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


def _margin_ok(final_anchor: Anchor, key_stdout: str) -> bool:
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
    measure_vals = [r["nums"][0] for r in final_rows if r.get("nums")]

    if has_measure and n_groups >= 2:
        fams.append("comparison")
    if len(measure_vals) >= 2 and (max(measure_vals) - min(measure_vals)) > 0.0:
        fams.append("anomaly")
    if any(e.kind == "chain" for e in getattr(path, "edges", []) or []):
        for a in anchors:
            if a.category == "chain" and len(_rows_from_result(a.r)) >= 2:
                fams.append("diagnosis")
                break
    dq = False
    for uid in dict.fromkeys(getattr(path, "nodes", []) or []):
        u = graph.units.get(uid)
        if not u:
            continue
        for _col, st in (u.stat.get("cols", {}) or {}).items():
            if float(st.get("null_rate", 0.0) or 0.0) > 0.0:
                dq = True
            if st.get("role") == "identifier" and 0.0 < float(
                    st.get("distinct_ratio", 1.0) or 1.0) < 1.0:
                dq = True
    if dq:
        fams.append("data_quality")
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
        a = _mk_path_anchor(probe[0], u.db_path, [u.unit_id], "probe", units_by_id)
        return [a] if a else None
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
    "comparison": "Compare a quantitative measure across groups, periods, or "
                  "sources and quantify the gap.",
    "diagnosis": "Explain an observed change or outcome by locating the "
                 "segment or factor that drives it.",
    "verification": "Check a stated expectation against the data and "
                    "quantify any discrepancy.",
    "anomaly": "Find the segment, period, or record that deviates most from "
               "the rest and size the deviation.",
    "data_quality": "Assess missing, duplicated, or inconsistent values and "
                    "quantify their extent or impact.",
}

_DEPGRAPH_REALIZE_PROMPT = """You are a senior analytics lead drafting a high-stakes question over a data environment.

**Question type ({question_type}):** {op_template}
**Analysis path the answer must follow (structure only, intermediate values withheld):**
{path_desc}

**Final step of the path (result values withheld):**
{final_step}

**Tables on the path (name and columns):**
{schema_blurb}

Write ONE focused, deeply analytical question whose answer requires following
the path above end to end. The path is multi-step by construction: an early
result selects rows in a later table, so a one-shot lookup cannot answer it.

Hard rules:
- The question must ask for exactly the quantity the final step computes:
  the same measure over the same grouping. Do NOT substitute a metric the
  path does not compute (e.g., never ask about revenue when the final step
  counts rows).
- Do NOT mention any specific intermediate key value, id, code, or the final
  numeric answer. The bridge values that link the steps must be discovered by
  the analyst, never stated in the question.
- Name the data sources the analyst should draw on (their table or file
  names) naturally inside the question or context, the way a real request
  names its data. Never name the columns that join or link them: finding
  those is the analyst's job.
- Commit to one quantitative target (rate, ratio, gap, share, growth,
  concentration, etc.), not "investigate" or "explore".
- Frame it around a concrete decision-maker facing a real operational choice.
- Deliverable: ONE concise phrase, <= 30 words, one report shape and one
  evidence type. No colon-lists, no "and also", no parenthetical sub-items.

If no coherent {family} question fits this path -- the operation cannot be
posed over these sources without inventing facts -- do NOT force one. Respond
with exactly: {{"decline": true}}

Otherwise respond with strict JSON:
{{
  "question": "...",
  "context": "...",
  "deliverable": "...",
  "ground_truth_query": "..."
}}
"""


def _realize_is_decline(parsed: Optional[Dict[str, Any]]) -> bool:
    if not parsed:
        return False
    if parsed.get("decline") in (True, "true", "True"):
        return True
    q = str(parsed.get("question", "") or "").strip()
    return q == "" or q.upper() == "NONE"


def _realize_depgraph(
    path, graph, client: OpenRouterClient, question_type: str,
    final_anchor: Optional[Anchor] = None,
) -> Optional[Dict[str, Any]]:
    from toffee.generation.depgraph import describe_path
    schema_lines: List[str] = []
    for uid in dict.fromkeys(path.nodes):
        u = graph.units.get(uid)
        if not u:
            continue
        cols = ", ".join(c for c, _t, _n in u.sch)
        schema_lines.append(f"  - {u.title}: {cols}")
    final_step = "(not available)"
    if final_anchor is not None:
        header = (final_anchor.r or "").strip().splitlines()
        final_step = (
            f"  query: {final_anchor.q_probe}\n"
            f"  result columns: {header[0] if header else ''}"
        )
    prompt = _DEPGRAPH_REALIZE_PROMPT.format(
        question_type=question_type,
        op_template=_TYPE_TEMPLATES.get(question_type, ""),
        path_desc=describe_path(path, graph),
        final_step=final_step,
        schema_blurb="\n".join(schema_lines),
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
        parsed["_question_type"] = question_type
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


        applicable = _applicable_question_types(path, anchors, graph) or ["verification"]
        question_type = rng.choice(applicable)
        realized = _realize_depgraph(path, graph, client, question_type, anchors[-1])
        if _realize_is_decline(realized):
            remaining = [t for t in applicable if t != question_type]
            if not remaining:
                stats["rej_decline"] += 1
                continue
            question_type = rng.choice(remaining)
            realized = _realize_depgraph(path, graph, client, question_type, anchors[-1])
            if _realize_is_decline(realized):
                stats["rej_decline"] += 1
                continue
        if not realized or not realized.get("question"):
            continue
        stats["realized"] += 1
        question = str(realized.get("question", "")).strip()


        bridge_vals = dg.collect_bridge_values(path)
        if dg.question_leaks(question, bridge_vals):
            stats["rej_leak"] += 1
            continue


        key_rows, key_stdout = _compile_answer_key(anchors[-1], graph.units)
        if _key_guessable(key_rows):
            stats["rej_guessable"] += 1
            continue
        if not _margin_ok(anchors[-1], key_stdout):
            stats["rej_margin"] += 1
            continue
        if _decoy_reproducible(key_rows, path, graph):
            stats["rej_decoy"] += 1
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
            ground_truth=str(realized.get("ground_truth_query", "")).strip(),
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
        out.append(SynthesizedTask(
            task_id=f"depgraph_{uuid.uuid4().hex[:8]}",
            question=task.x,
            context=task.context,
            deliverable=task.deliverable,
            env={
                "db_path": sources[0],
                "data_file": sources[0],
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
