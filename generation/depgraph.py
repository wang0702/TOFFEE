
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from toffee.config import (
    BRIDGE_MAX,
    CHAIN_VERIFY_BUDGET,
    JOIN_FANOUT_MAX,
    JOIN_VERIFY_BUDGET,
    ROLE_GATE,
    VALUE_SAMPLE_N,
)
from toffee.core.executor import execute_sql
import os as _os
import time as _time


DEPGRAPH_BUDGET_S = float(_os.environ.get("TOFFEE_DEPGRAPH_BUDGET_S", "1200"))


from toffee.generation.bottomup import (
    _is_date_col,
    _is_num_type,
    _nontrivial,
    _probe_unit_association,
    _probe_unit_ranking,
    _probe_unit_temporal,
)
from toffee.generation.ingest import SourceUnit

log = logging.getLogger(__name__)


_MAX_PROPOSAL_COLS = 8


_JOIN_DECOY_MAX = 4


@dataclass
class DepEdge:
    src: str
    dst: str
    kind: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DepGraph:
    nodes: List[str]
    edges: List[DepEdge]
    units: Dict[str, SourceUnit] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)

    def out_edges(self, node: str) -> List[DepEdge]:
        out: List[DepEdge] = []
        for e in self.edges:
            if e.src == node:
                out.append(e)
            elif e.kind == "join" and e.dst == node:
                meta = dict(e.meta)
                pair = list(meta.get("col_pair") or [])
                if len(pair) == 2:
                    meta["col_pair"] = [pair[1], pair[0]]
                tables = list(meta.get("tables") or [])
                if len(tables) == 2:
                    meta["tables"] = [tables[1], tables[0]]
                out.append(DepEdge(src=node, dst=e.src, kind="join", meta=meta))
        return out


@dataclass
class Path:
    level: int
    nodes: List[str]
    edges: List[DepEdge] = field(default_factory=list)

    def edge_kinds(self) -> List[str]:
        return [e.kind for e in self.edges]


def parse_column_output(stdout: str) -> Tuple[List[str], List[List[str]]]:
    lines = stdout.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 2 or not re.match(r"^-+", lines[1]):
        return [], []
    sep = lines[1]
    spans = [(m.start(), m.end()) for m in re.finditer(r"-+", sep)]
    header = [lines[0][s:e].strip() for s, e in spans]
    rows: List[List[str]] = []
    for ln in lines[2:]:
        if not ln.strip():
            continue
        rows.append([ln[s:e].strip() for s, e in spans])
    return header, rows


def _first_column_values(stdout: str, cap: int) -> List[str]:
    _header, rows = parse_column_output(stdout)
    out: List[str] = []
    for r in rows:
        if r and r[0] != "":
            out.append(r[0])
        if len(out) >= cap:
            break

    return list(dict.fromkeys(out))


def sample_column_values(db_path: str, table: str, col: str, n: int = VALUE_SAMPLE_N) -> Set[str]:
    q = f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {int(n)};'
    res = execute_sql(db_path, q)
    if not res.success:
        return set()
    _header, rows = parse_column_output(res.stdout)
    return {r[0] for r in rows if r and r[0] != ""}


def containment(a: Set[str], b: Set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _get_sample(
    samples: Dict[Tuple[str, str], Set[str]], u: SourceUnit, col: str,
) -> Set[str]:
    key = (u.unit_id, col)
    if key not in samples:
        samples[key] = sample_column_values(u.db_path, u.logical_table, col)
    return samples[key]


def _col_class(col: str, typ: str) -> str:
    return "num" if _is_num_type(typ) else "text"


def _role_of(u: SourceUnit, col: str) -> str:
    st = u.stat.get("cols", {}).get(col, {})
    role = st.get("role")
    if role:
        return str(role)
    typ = next((t for c, t, _ in u.sch if c == col), "")
    dr = float(st.get("distinct_ratio", 0.0) or 0.0)
    if _is_date_col(col, typ):
        return "time"
    from toffee.generation.ingest import _KEY_NAME_PATTERNS
    if _KEY_NAME_PATTERNS.search(col or ""):
        if _is_num_type(typ) or dr >= 0.5:
            return "identifier"
    if dr >= 0.95:
        return "identifier"
    if _is_num_type(typ):
        return "measure"
    if 0.0 < dr < 0.5:
        return "category"
    return "entity"


def _proposal_columns(u: SourceUnit) -> List[Tuple[str, str]]:
    keys = [(c, t) for c, t, _ in u.sch if c in set(u.key)]
    cols_stat = u.stat.get("cols", {})
    rest = [
        (c, t) for c, t, _ in u.sch
        if c not in set(u.key)
    ]
    rest.sort(key=lambda ct: cols_stat.get(ct[0], {}).get("distinct_ratio", 0.0), reverse=True)
    ordered = keys + rest
    return ordered[:_MAX_PROPOSAL_COLS]


_GENERIC_KEY_TOKENS = ("id", "code", "key", "no", "num", "number", "pk", "fk",
                       "idx", "index", "name")


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _name_corresponds(cu: str, cv: str) -> bool:
    a, b = _norm_col(cu), _norm_col(cv)
    if not a or a != b:
        return False
    stem = a
    for tok in _GENERIC_KEY_TOKENS:
        if stem.endswith(tok):
            stem = stem[: -len(tok)]
            break
    return len(stem) >= 3


def _name_similarity(cu: str, cv: str) -> float:
    if _name_corresponds(cu, cv):
        return 1.0
    ta = set(re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cu or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cv or "").lower()))
    ta -= set(_GENERIC_KEY_TOKENS)
    tb -= set(_GENERIC_KEY_TOKENS)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _join_sql(u: SourceUnit, v: SourceUnit, cu: str, cv: str) -> Tuple[str, str]:
    if u.db_path == v.db_path:
        sql = (f'SELECT a."{cu}", COUNT(*) AS n '
               f'FROM "{u.logical_table}" a JOIN "{v.logical_table}" b '
               f'ON a."{cu}" = b."{cv}" '
               f'GROUP BY a."{cu}" ORDER BY n DESC LIMIT 10;')
        return sql, u.db_path

    scratch = getattr(u, "scratch_db_path", "") or getattr(v, "scratch_db_path", "")
    if v.db_path == scratch and scratch:
        host, host_tbl, host_col = u.db_path, u.logical_table, cu
        other_db, other_tbl, other_col = v.db_path, v.logical_table, cv
    elif u.db_path == scratch and scratch:
        host, host_tbl, host_col = v.db_path, v.logical_table, cv
        other_db, other_tbl, other_col = u.db_path, u.logical_table, cu
    else:
        host, host_tbl, host_col = u.db_path, u.logical_table, cu
        other_db, other_tbl, other_col = v.db_path, v.logical_table, cv

    attach = f"ATTACH DATABASE '{other_db}' AS other;"
    join_q = (f'SELECT a."{host_col}", COUNT(*) AS n '
              f'FROM "{host_tbl}" a JOIN other."{other_tbl}" b '
              f'ON a."{host_col}" = b."{other_col}" '
              f'GROUP BY a."{host_col}" ORDER BY n DESC LIMIT 10;')
    return f"{attach}\n{join_q}", host


def _best_join_pair(
    u: SourceUnit, v: SourceUnit, samples: Dict[Tuple[str, str], Set[str]],
) -> Optional[Tuple[str, str, float]]:
    best = None
    ukeys, vkeys = set(u.key), set(v.key)
    for cu, tu in _proposal_columns(u):
        su = samples.get((u.unit_id, cu), set())
        if not su:
            continue
        for cv, tv in _proposal_columns(v):
            if _col_class(cu, tu) != _col_class(cv, tv):
                continue


            if ROLE_GATE and ("measure" in (_role_of(u, cu), _role_of(v, cv))):
                continue
            sv = samples.get((v.unit_id, cv), set())
            if not sv:
                continue

            cont = max(containment(su, sv), containment(sv, su))
            if cont <= 0.0:
                continue

            rank = _name_similarity(cu, cv) + cont
            if cu in ukeys and cv in vkeys:
                rank += 1e-6
            if best is None or rank > best[0]:
                best = (rank, cont, cu, cv)
    if best is None:
        return None
    return best[2], best[3], best[1]


def _max_fanout(stdout: str) -> int:
    _header, rows = parse_column_output(stdout)
    for r in rows:
        if len(r) >= 2:
            try:
                return int(float(r[1]))
            except ValueError:
                continue
    return 0


def declared_fks(units: List[SourceUnit]) -> Set[Tuple[str, str, str, str, str]]:
    out: Set[Tuple[str, str, str, str, str]] = set()
    seen: Set[Tuple[str, str]] = set()
    for u in units:
        if u.format != "sqlite_table":
            continue
        recorded = list((u.stat or {}).get("declared_fks") or [])
        if recorded:
            for fk in recorded:
                if fk.get("table") and fk.get("from") and fk.get("to"):
                    out.add((u.db_path, u.logical_table, str(fk["from"]),
                             str(fk["table"]), str(fk["to"])))
            continue
        key = (u.db_path, u.logical_table)
        if key in seen:
            continue
        seen.add(key)
        res = execute_sql(u.db_path, f"PRAGMA foreign_key_list('{u.logical_table}');")
        if not res.success:
            continue
        _header, rows = parse_column_output(res.stdout)

        for r in rows:
            if len(r) >= 5 and r[2] and r[3] and r[4]:
                out.add((u.db_path, u.logical_table, r[3], r[2], r[4]))
    return out


def _declared_pair_for(
    u: SourceUnit, v: SourceUnit, fkset: Set[Tuple[str, str, str, str, str]],
) -> Optional[Tuple[str, str]]:
    if u.db_path != v.db_path:
        return None
    for db, ct, cc, pt, pc in fkset:
        if db != u.db_path:
            continue
        if ct == u.logical_table and pt == v.logical_table:
            return cc, pc
        if ct == v.logical_table and pt == u.logical_table:
            return pc, cc
    return None


def counterfactual_join_ok(
    edge: DepEdge, u: SourceUnit, v: SourceUnit,
) -> bool:
    cu, cv = edge.meta["col_pair"]
    cls = _col_class(cu, dict((c, t) for c, t, _ in u.sch).get(cu, ""))
    decoys: List[Tuple[str, str]] = []
    for cvp, tvp in _proposal_columns(v):
        if cvp != cv and _col_class(cvp, tvp) == cls:
            decoys.append((cu, cvp))
    for cup, tup in _proposal_columns(u):
        if cup != cu and _col_class(cup, tup) == cls:
            decoys.append((cup, cv))
    if not decoys:


        return False
    for cup, cvp in decoys[:_JOIN_DECOY_MAX]:
        sql, exec_db = _join_sql(u, v, cup, cvp)
        res = execute_sql(exec_db, sql)
        if res.success and _nontrivial(res.stdout)[0]:
            return False
    return True


def _build_join_edges(
    units: List[SourceUnit], samples: Dict[Tuple[str, str], Set[str]],
    units_by_id: Dict[str, SourceUnit],
    fkset: Set[Tuple[str, str, str, str, str]],
) -> Tuple[List[DepEdge], Dict[str, int]]:

    _deadline = _time.monotonic() + DEPGRAPH_BUDGET_S
    proposals: List[Tuple[int, int, float, SourceUnit, SourceUnit, str, str]] = []
    for i in range(len(units)):
        if _time.monotonic() > _deadline:
            log.warning("depgraph budget exhausted during proposal at unit %d/%d; keeping partial graph", i, len(units))
            break
        for j in range(i + 1, len(units)):
            u, v = units[i], units[j]
            dfk = _declared_pair_for(u, v, fkset)
            if dfk is not None:
                cu, cv = dfk
                cont = containment(_get_sample(samples, u, cu), _get_sample(samples, v, cv))
                proposals.append((1, int(_name_corresponds(cu, cv)), cont, u, v, cu, cv))
                continue
            bp = _best_join_pair(u, v, samples)
            if bp is None:
                continue
            cu, cv, cont = bp
            proposals.append((0, int(_name_corresponds(cu, cv)), cont, u, v, cu, cv))


    proposals.sort(key=lambda p: (p[0], p[1], p[2]), reverse=True)

    stats = {"verified": 0, "kept": 0, "rej_degenerate": 0, "rej_nondiscriminative": 0,
             "rej_fanout": 0,
             "kept_declared": 0, "kept_name": 0, "kept_counterfactual": 0}
    edges: List[DepEdge] = []
    for is_declared, _corr, cont, u, v, cu, cv in proposals:
        if _time.monotonic() > _deadline:
            log.warning("depgraph budget exhausted during verification; keeping partial graph")
            break
        if stats["verified"] >= JOIN_VERIFY_BUDGET:
            break
        sql, exec_db = _join_sql(u, v, cu, cv)
        res = execute_sql(exec_db, sql)
        stats["verified"] += 1
        if not res.success or not res.stdout.strip():
            continue


        if not is_declared and not _nontrivial(res.stdout)[0]:
            stats["rej_degenerate"] += 1
            continue


        fu = _max_fanout(res.stdout)
        fv = 0
        if ROLE_GATE and not is_declared and fu > 1:
            m_sql, m_db = _join_sql(v, u, cv, cu)
            m_res = execute_sql(m_db, m_sql)
            fv = _max_fanout(m_res.stdout) if m_res.success else 0
            if min(fu, fv) > 1 and fu * fv > JOIN_FANOUT_MAX:
                stats["rej_fanout"] += 1
                continue
        edge = DepEdge(
            src=u.unit_id, dst=v.unit_id, kind="join",
            meta={
                "col_pair": [cu, cv],
                "endpoint_cols": {u.unit_id: cu, v.unit_id: cv},
                "containment": round(cont, 4),
                "declared_fk": bool(is_declared),
                "name_corresponds": _name_corresponds(cu, cv),
                "join_sql": sql,
                "exec_db": exec_db,
                "tables": [u.logical_table, v.logical_table],
                "fanout": [fu, fv],
            },
        )


        if is_declared:
            admitted_by = "declared"
        elif _name_corresponds(cu, cv):
            admitted_by = "name"
        elif counterfactual_join_ok(edge, u, v):
            admitted_by = "counterfactual"
        else:
            stats["rej_nondiscriminative"] += 1
            continue
        edge.meta["admitted_by"] = admitted_by
        edges.append(edge)
        stats["kept"] += 1
        stats["kept_" + admitted_by] += 1
    return edges, stats


def _cheapest_probe(u: SourceUnit) -> Optional[Tuple[str, Dict[str, Any]]]:
    for fn in (_probe_unit_ranking, _probe_unit_temporal, _probe_unit_association):
        probes = fn(u)
        if probes:
            return probes[0]
    return None


_BRIDGE_ROLE_PREF = {"identifier": 0, "entity": 1, "category": 2, "time": 3}


def _bridge_probe(u: SourceUnit) -> Optional[Tuple[str, Dict[str, Any], str]]:
    best: Optional[Tuple[int, int, str, Dict[str, Any], str]] = None
    for order, fn in enumerate(
            (_probe_unit_ranking, _probe_unit_temporal, _probe_unit_association)):
        probes = fn(u)
        if not probes:
            continue
        sql, meta = probes[0]
        cols = meta.get("col_used") or []
        bridge_col = cols[0] if cols else ""
        role = _role_of(u, bridge_col) if bridge_col else "entity"
        if not ROLE_GATE:
            return sql, meta, bridge_col
        if role == "measure":
            continue
        pref = _BRIDGE_ROLE_PREF.get(role, 4)
        if best is None or (pref, order) < (best[0], best[1]):
            best = (pref, order, sql, meta, bridge_col)
    if best is None:
        return None
    return best[2], best[3], best[4]


def _bridge_is_numeric(vals: Sequence[str]) -> bool:
    if not vals:
        return False
    for x in vals:
        try:
            float(str(x))
        except ValueError:
            return False
    return True


_CHAIN_ROLE_COMPAT = {
    "identifier": {"identifier"},
    "entity": {"entity", "identifier", "category"},
    "category": {"category", "entity"},
    "time": {"time"},
}


def _chain_role_ok(
    bridge_col: str, bridge_role: str, vals: Sequence[str],
    v: SourceUnit, cv: str,
) -> bool:
    target_role = _role_of(v, cv)
    if target_role == "measure":
        return False
    if target_role not in _CHAIN_ROLE_COMPAT.get(bridge_role, set()):
        return False
    if _bridge_is_numeric(vals):
        return (bridge_role == "identifier" and target_role == "identifier"
                and _name_corresponds(bridge_col, cv))
    return True


def _chain_verify_sql(v: SourceUnit, cv: str, bridge: Sequence[str]) -> str:
    vals = ", ".join("'" + str(b).replace("'", "''") + "'" for b in bridge)
    return (f'SELECT "{cv}", COUNT(*) AS n FROM "{v.logical_table}" '
            f'WHERE "{cv}" IN ({vals}) GROUP BY "{cv}" ORDER BY n DESC, "{cv}" LIMIT 20;')


_DATE_VALUE_RE = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}|\s*$)")


def _bridge_is_date_only(vals: Sequence[str]) -> bool:
    return bool(vals) and all(_DATE_VALUE_RE.match(str(x)) for x in vals)


def _build_chain_edges(
    units: List[SourceUnit], samples: Dict[Tuple[str, str], Set[str]],
) -> Tuple[List[DepEdge], Dict[str, int]]:
    stats = {"probed": 0, "verified": 0, "kept": 0, "cf_pass": 0, "cf_reject": 0,
             "rej_role": 0}


    bridges: Dict[str, Tuple[str, List[str], SourceUnit, str, Dict[str, Any]]] = {}
    for u in units:
        probe = _bridge_probe(u)
        if not probe:
            continue
        sql, probe_meta, bridge_col = probe
        res = execute_sql(u.db_path, sql)
        if not res.success:
            continue
        vals = _first_column_values(res.stdout, BRIDGE_MAX)
        if not vals:
            continue
        stats["probed"] += 1
        bridges[u.unit_id] = (sql, vals, u, bridge_col, dict(probe_meta))


    candidates: List[Tuple[
        float, SourceUnit, List[str], str, SourceUnit, str, str, Dict[str, Any]
    ]] = []
    for uid, (probe_sql, vals, u, bridge_col, probe_meta) in bridges.items():
        bridge_set = set(vals)
        bridge_role = _role_of(u, bridge_col) if bridge_col else "entity"
        for v in units:
            if v.unit_id == uid:
                continue
            for cv, tv in _proposal_columns(v):
                sv = samples.get((v.unit_id, cv), set())
                if not sv:
                    continue
                cont = containment(bridge_set, sv)
                if cont <= 0.0:
                    continue
                if ROLE_GATE and not _chain_role_ok(bridge_col, bridge_role, vals, v, cv):
                    stats["rej_role"] += 1
                    continue
                candidates.append((cont, u, vals, probe_sql, v, cv,
                                   bridge_col, probe_meta))


    candidates.sort(key=lambda c: (not _bridge_is_date_only(c[2]), c[0]), reverse=True)

    edges: List[DepEdge] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for cont, u, vals, probe_sql, v, cv, bridge_col, probe_meta in candidates:
        if stats["verified"] >= CHAIN_VERIFY_BUDGET:
            break
        pair = (u.unit_id, v.unit_id)
        if pair in seen_pairs:
            continue
        verify_sql = _chain_verify_sql(v, cv, vals)
        res = execute_sql(v.db_path, verify_sql)
        stats["verified"] += 1
        seen_pairs.add(pair)
        if not res.success:
            continue
        _header, rows = parse_column_output(res.stdout)
        if not rows:
            continue
        edge = DepEdge(
            src=u.unit_id, dst=v.unit_id, kind="chain",
            meta={
                "probe_sql": probe_sql,
                "probe_meta": probe_meta,
                "probe_exec_db": u.db_path,
                "bridge_values": list(vals),
                "bridge_col": bridge_col,
                "bridge_role": _role_of(u, bridge_col) if bridge_col else "entity",
                "target_col": cv,
                "target_role": _role_of(v, cv),
                "selected_by": "role_pref" if ROLE_GATE else "unrestricted",
                "verify_sql": verify_sql,
                "verify_exec_db": v.db_path,
                "containment": round(cont, 4),
            },
        )
        if not bridge_unique_ok(edge):
            continue
        if counterfactual_ok(edge, v, samples.get((v.unit_id, cv), set())):
            stats["cf_pass"] += 1
            edges.append(edge)
        else:
            stats["cf_reject"] += 1
    stats["kept"] = len(edges)
    return edges, stats


def bridge_unique_ok(edge: DepEdge) -> bool:
    if edge.kind != "chain":
        return True
    n = len(edge.meta.get("bridge_values", []))
    return 1 <= n <= BRIDGE_MAX


def counterfactual_ok(
    edge: DepEdge, v: SourceUnit, col_sample: Set[str],
) -> bool:
    if edge.kind != "chain":
        return True
    bridge = set(edge.meta.get("bridge_values", []))
    cv = edge.meta["target_col"]
    decoys = [x for x in col_sample if x not in bridge]
    if not decoys:
        return False
    decoys = decoys[:max(1, len(bridge))]
    bridge_res = execute_sql(edge.meta["verify_exec_db"], edge.meta["verify_sql"])
    decoy_sql = _chain_verify_sql(v, cv, decoys)
    decoy_res = execute_sql(edge.meta["verify_exec_db"], decoy_sql)
    if not bridge_res.success or not decoy_res.success:
        return False
    _bh, bridge_rows = parse_column_output(bridge_res.stdout)
    _dh, decoy_rows = parse_column_output(decoy_res.stdout)

    return sorted(map(tuple, bridge_rows)) != sorted(map(tuple, decoy_rows))


def build_dependency_graph(units: List[SourceUnit], per_env_budget: Optional[int] = None) -> DepGraph:
    units = [u for u in units if u.stat.get("n_rows", 0) > 0 and u.sch]
    units_by_id = {u.unit_id: u for u in units}


    samples: Dict[Tuple[str, str], Set[str]] = {}
    for u in units:
        for col, _typ in _proposal_columns(u):
            samples[(u.unit_id, col)] = sample_column_values(u.db_path, u.logical_table, col)

    fkset = declared_fks(units)
    join_edges, join_stats = _build_join_edges(units, samples, units_by_id, fkset)
    chain_edges, chain_stats = _build_chain_edges(units, samples)

    stats = {
        "n_units": len(units),
        "declared_fks": len(fkset),
        "join_verified": join_stats["verified"],
        "join_kept": join_stats["kept"],
        "join_kept_declared": join_stats["kept_declared"],
        "join_kept_name": join_stats["kept_name"],
        "join_kept_counterfactual": join_stats["kept_counterfactual"],
        "join_rej_degenerate": join_stats["rej_degenerate"],
        "join_rej_nondiscriminative": join_stats["rej_nondiscriminative"],
        "join_rej_fanout": join_stats["rej_fanout"],
        "chain_rej_role": chain_stats["rej_role"],
        "chain_probed": chain_stats["probed"],
        "chain_verified": chain_stats["verified"],
        "chain_kept": chain_stats["kept"],
        "cf_pass": chain_stats["cf_pass"],
        "cf_reject": chain_stats["cf_reject"],
    }
    return DepGraph(
        nodes=[u.unit_id for u in units],
        edges=join_edges + chain_edges,
        units=units_by_id,
        stats=stats,
    )


def _composable_continuations(graph: DepGraph, first: DepEdge) -> List[DepEdge]:
    """Chain edges that consume a result produced over ``first.dst``."""
    return [
        edge for edge in graph.out_edges(first.dst)
        if edge.kind == "chain"
        and edge.dst not in (first.src, first.dst)
    ]


def _oriented_first_edges(graph: DepGraph) -> List[DepEdge]:
    edges = list(graph.edges)
    for edge in graph.edges:
        if edge.kind == "join":
            edges.extend(
                candidate for candidate in graph.out_edges(edge.dst)
                if candidate.kind == "join" and candidate.dst == edge.src
            )
    return edges


def supported_levels(graph: DepGraph) -> List[int]:
    levels = [1] if graph.nodes else []
    if graph.edges:
        levels.append(2)


        for e in _oriented_first_edges(graph):
            if _composable_continuations(graph, e):
                levels.append(3)
            if 3 in levels:
                break
    return levels


def sample_path(graph: DepGraph, level: int, rng: random.Random) -> Optional[Path]:
    if not graph.nodes:
        return None
    if level <= 1:
        return Path(level=1, nodes=[rng.choice(graph.nodes)], edges=[])
    if not graph.edges:
        return None
    if level == 2:
        e = rng.choice(graph.edges)
        return Path(level=2, nodes=[e.src, e.dst], edges=[e])


    edges = _oriented_first_edges(graph)
    rng.shuffle(edges)
    for e1 in edges:
        conts = _composable_continuations(graph, e1)
        if conts:
            e2 = rng.choice(conts)
            return Path(level=3, nodes=[e1.src, e1.dst, e2.dst], edges=[e1, e2])
    return None


_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or "
    "that the their this to was what when where which with over per each "
    "most least total number show find give list".split()
)


def _question_tokens(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def _unit_tokens(u: SourceUnit) -> Set[str]:
    parts = [u.title or "", u.logical_table or ""] + [c for c, _t, _n in u.sch]
    toks: Set[str] = set()
    for p in parts:
        p = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(p))
        toks.update(w for w in re.findall(r"[a-z0-9]+", p.lower()) if len(w) >= 3)
    return toks - _STOPWORDS


def candidate_paths_for_question(
    question: str, graph: DepGraph, cap: int = 6,
) -> List[Path]:
    q_toks = _question_tokens(question)
    if not q_toks or not graph.nodes:
        return []
    unit_toks: Dict[str, Set[str]] = {}
    matched: Set[str] = set()
    for uid in graph.nodes:
        u = graph.units.get(uid)
        if not u:
            continue
        toks = _unit_tokens(u)
        unit_toks[uid] = toks
        if q_toks & toks:
            matched.add(uid)
    if not matched:
        return []

    def coverage(nodes: Sequence[str]) -> int:
        joint: Set[str] = set()
        for n in nodes:
            joint |= unit_toks.get(n, set())
        return len(q_toks & joint)

    cand: List[Tuple[int, int, Path]] = []
    for uid in matched:
        cand.append((coverage([uid]), -1, Path(level=1, nodes=[uid], edges=[])))
    for e in graph.edges:
        if e.src in matched or e.dst in matched:
            cand.append((coverage([e.src, e.dst]), -2,
                         Path(level=2, nodes=[e.src, e.dst], edges=[e])))
    for e1 in _oriented_first_edges(graph):
        for e2 in _composable_continuations(graph, e1):
            nodes = [e1.src, e1.dst, e2.dst]
            if not (set(nodes) & matched):
                continue
            cand.append((coverage(nodes), -3,
                         Path(level=3, nodes=nodes, edges=[e1, e2])))
    cand.sort(key=lambda c: (c[0], c[1]), reverse=True)
    out: List[Path] = []
    seen: Set[Tuple] = set()
    for _cov, _neg, p in cand:
        key = (tuple(p.nodes), tuple(p.edge_kinds()))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= cap:
            break
    return out


def collect_bridge_values(path: Path) -> List[str]:
    vals: List[str] = []
    for e in path.edges:
        if e.kind == "chain":
            vals.extend(str(x) for x in e.meta.get("bridge_values", []))
    return vals


def question_leaks(question: str, bridge_values: Sequence[str], min_len: int = 2) -> bool:
    q = (question or "").lower()
    for b in bridge_values:
        b = str(b).strip()
        if not b:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", b):
            if re.search(rf"(?<![\w.]){re.escape(b)}(?![\w.])", q):
                return True
        elif len(b) >= min_len:
            if re.search(rf"(?<!\w){re.escape(b.lower())}(?!\w)", q):
                return True
    return False


def describe_path(path: Path, graph: DepGraph) -> str:
    def title(uid: str) -> str:
        u = graph.units.get(uid)
        return u.title if u else uid

    if path.level == 1:
        return f"level 1: a single-table analysis over {title(path.nodes[0])}."
    parts: List[str] = [f"level {path.level}:"]
    prev = path.nodes[0]
    for e in path.edges:
        if e.kind == "join":
            parts.append(
                f"join {title(e.src)} with {title(e.dst)} on their shared key, "
                f"then aggregate the joined rows;"
            )
        else:


            parts.append(
                f"compute a result over {title(e.src)}; its key values select "
                f"matching rows in {title(e.dst)};"
            )
        prev = e.dst
    parts.append("the final result is the answer key.")
    return " ".join(parts)


__all__ = [
    "DepEdge", "DepGraph", "Path",
    "parse_column_output", "sample_column_values", "containment",
    "build_dependency_graph", "sample_path", "supported_levels",
    "counterfactual_ok", "counterfactual_join_ok", "bridge_unique_ok",
    "declared_fks", "describe_path", "collect_bridge_values", "question_leaks",
    "candidate_paths_for_question",
]
