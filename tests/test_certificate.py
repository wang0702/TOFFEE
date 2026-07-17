
from toffee.generation import bottomup as B
from toffee.search.evaluator import vq_score
from toffee.search.mcts import Trajectory, TrajectoryStep, _acceptance_check
from toffee import config as _cfg


def _check(name, cond):
    if not cond:
        raise AssertionError("FAILED: " + name)
    print("ok -", name)


def _step(output, tool="execute_sql",
          query="SELECT region, revenue FROM orders GROUP BY region"):
    return TrajectoryStep(
        operator_name="SQLQuery", tool_name=tool, arguments={"query": query},
        real_output=output, reasoning="", visible_text="", model="m", cost=0.0,
    )


def _traj(outputs, answer):
    return Trajectory(
        steps=[_step(o) for o in outputs], quality=0.0, total_cost=0.0,
        final_answer=answer, ended_with_answer=True,
        successful_data_steps=len(outputs), error_steps=0,
    )


_SRC_HINT = {"entry_units": ["orders"]}


_SQL_OUT_1 = ("region      revenue   \n"
              "----------  ----------\n"
              "EU          1200      \n")
_SQL_OUT_2 = ("region      share     \n"
              "----------  ----------\n"
              "NA          17.5      \n")

FACTS = [{"anchor": "a1", "nums": [1200.0], "labels": ["EU"]},
         {"anchor": "a2", "nums": [17.5], "labels": ["NA"]}]


def test_labels_from_result():
    labels = B._labels_from_result(_SQL_OUT_1)
    _check("row label extracted", labels == ["EU"])
    cat = "region    \n----------\nEU        \nNA        \n"
    _check("categorical rows keep all labels",
           B._labels_from_result(cat) == ["EU", "NA"])
    _check("numeric cells are not labels",
           B._labels_from_result("n         \n----------\n1,200     \n") == [])


KEY_ROWS = [{"label": "EU", "value": 1200.0, "raw": "EU 1200"},
            {"label": "NA", "value": 17.5, "raw": "NA 17.5"}]
_ORDERS_SQL = ("SELECT region, SUM(revenue) FROM orders "
               "WHERE region IN (SELECT region FROM marketing) GROUP BY region")


def test_verifier_answer_side():


    ans = "EU revenue fell to 1,200 while NA spend share was 17.5%."
    _check("every key_rows row stated near its label -> certified",
           vq_score(ans, "", KEY_ROWS) >= 0.95)
    _check("per-step fact records normalize to the same key_rows",
           vq_score(ans, "", FACTS) >= 0.95)
    _check("bare number dump (no labels) fails coverage",
           vq_score("Values: 1200 and 17.5.", "", KEY_ROWS) == 0.0)
    _check("partial coverage fails (NA row missing)",
           vq_score("EU revenue was 1200.", "", KEY_ROWS) == 0.0)
    cat = [{"label": "APAC", "value": None, "raw": "APAC"}]
    _check("label-only row stated by naming its label",
           vq_score("APAC deviates most.", "", cat) >= 0.95)


def test_verifier_computed_gate():
    ans = "EU revenue is 1200 and NA share is 17.5."
    src = ["orders", "marketing"]
    nested = [{"tool": "execute_sql", "text": _ORDERS_SQL,
               "output": "EU 1200\nNA 17.5"}]
    _check("one nested query over a source certifies (path freedom)",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=nested, source_names=src) >= 0.95)
    literal = [{"tool": "execute_sql", "text": "SELECT 'EU', 1200, 'NA', 17.5",
                "output": "EU 1200 NA 17.5"}]
    _check("literal echo reads no source -> Computed fails, capped 0.5",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=literal, source_names=src) == 0.5)
    _check("Correct but no data query -> capped 0.5",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=[], source_names=src) == 0.5)
    _check("cross-label mispairing rejected",
           vq_score("NA revenue is 1200 and EU share is 17.5.",
                                    "", KEY_ROWS, data_steps=nested,
                                    source_names=src) == 0.0)
    _check("value computed but not stated by the source query -> capped 0.5",
           vq_score(ans, "", KEY_ROWS, source_names=src, data_steps=[
               {"tool": "execute_sql", "text": "SELECT region FROM orders",
                "output": "EU\nNA"}]) == 0.5)


def test_verifier_trace_side():


    ans = "EU revenue fell to 1,200 while NA spend share was 17.5%."
    _check("stated + present in outputs -> certified",
           vq_score(
               ans, "", KEY_ROWS, trace_outputs=[_SQL_OUT_1, _SQL_OUT_2]) >= 0.95)
    _check("a value absent from every output -> capped 0.5",
           vq_score(
               ans, "", KEY_ROWS, trace_outputs=[_SQL_OUT_1]) == 0.5)
    _check("outputs need the values only, not the labels",
           vq_score(
               ans, "", KEY_ROWS, trace_outputs=["1200", "17.5"]) >= 0.95)
    _check("no executed outputs -> Computed fails, capped 0.5",
           vq_score(ans, "", KEY_ROWS, trace_outputs=[]) == 0.5)


_SQL_OUT_TWO_ROWS = ("region      revenue   \n"
                     "----------  ----------\n"
                     "EU          1200      \n"
                     "NA          900       \n")

ROW_CONTRACT = [{"label": "EU", "value": 1200.0, "raw": "EU 1200"},
                {"label": "NA", "value": 900.0, "raw": "NA 900"}]


def test_contract_coverage_and_crosslabel():
    _check("rows extracted from executed output",
           B._rows_from_result(_SQL_OUT_TWO_ROWS) ==
           [{"labels": ["EU"], "nums": [1200.0]},
            {"labels": ["NA"], "nums": [900.0]}])
    _check("both rows stated with their own labels -> certified",
           vq_score(
               "EU revenue is 1200 and NA revenue is 900.", "", ROW_CONTRACT,
               trace_outputs=[_SQL_OUT_TWO_ROWS]) >= 0.95)
    _check("cross-label pairing rejected (NA's name on EU's value)",
           vq_score(
               "NA revenue is 1200.", "", ROW_CONTRACT,
               trace_outputs=[_SQL_OUT_TWO_ROWS]) == 0.0)
    _check("fully swapped values rejected",
           vq_score(
               "EU revenue is 900 and NA revenue is 1200.", "", ROW_CONTRACT,
               trace_outputs=[_SQL_OUT_TWO_ROWS]) == 0.0)
    _check("partial coverage (one of two rows) rejected",
           vq_score(
               "EU revenue is 1200.", "", ROW_CONTRACT,
               trace_outputs=[_SQL_OUT_TWO_ROWS]) == 0.0)
    _check("exact key match cannot bypass missing coverage",
           vq_score(
               "1200", "1200", ROW_CONTRACT,
               trace_outputs=[_SQL_OUT_TWO_ROWS]) <= 0.5)


def test_acceptance_recomputes_certificate():
    task = {"answer_key": "1200", "facts": FACTS, "hint": _SRC_HINT}
    old_rule = _cfg.ACCEPT_RULE
    _cfg.ACCEPT_RULE = "fact"
    try:
        good = _traj([_SQL_OUT_1, _SQL_OUT_2],
                     "EU revenue is 1200 and NA share is 17.5.")
        good.quality = 0.0
        _check("certified trajectory accepted regardless of search score",
               _acceptance_check(good, task, quality=0.0)[0])

        dump = _traj([_SQL_OUT_1, _SQL_OUT_2], "Values: 1200, 17.5.")
        dump.quality = 1.0
        _check("number dump rejected regardless of search score",
               not _acceptance_check(dump, task, quality=1.0)[0])


        lazy = _traj([_SQL_OUT_1],
                     "EU revenue is 1200 and NA share is 17.5.")
        _check("composed answer without executed evidence rejected",
               not _acceptance_check(lazy, task, quality=0.95)[0])


        no_query = Trajectory(
            steps=[_step("", tool="list_tables", query="")],
            quality=0.0, total_cost=0.0,
            final_answer="EU revenue is 1200 and NA share is 17.5.",
            ended_with_answer=True)
        _check("answer with no data query rejected",
               not _acceptance_check(no_query, task, quality=0.95)[0])

        _check("no-evidence task falls back to search quality",
               _acceptance_check(_traj(["x"], "done"), {}, quality=0.95)[0]
               and not _acceptance_check(_traj(["x"], "done"), {}, quality=0.5)[0])
    finally:
        _cfg.ACCEPT_RULE = old_rule


def test_prefix_outputs_count_for_certificate():


    task = {"answer_key": "1200", "facts": FACTS, "hint": _SRC_HINT}
    old_rule = _cfg.ACCEPT_RULE
    _cfg.ACCEPT_RULE = "fact"
    try:
        t = _traj([_SQL_OUT_2], "EU revenue is 1200 and NA share is 17.5.")
        _check("collapsed prefix without outputs is rejected",
               not _acceptance_check(t, task)[0])
        t.tool_outputs = [_SQL_OUT_1, _SQL_OUT_2]
        _check("prefix-carried outputs witness the key_rows",
               _acceptance_check(t, task)[0])
    finally:
        _cfg.ACCEPT_RULE = old_rule


def test_nondeg_fail_closed():
    anchor = B.Anchor(
        a_id="a1", q_probe="SELECT 1", r=_SQL_OUT_1, r_rowcount=1,
        r_signature="s", c={"tables_used": ["orders"]}, U_a=["u1"],
        scope_id="g00", category="aggregate",
    )
    tp = B.TaskPackage(x="q?", path=["u1"], level=1,
                       hint={"entry_units": ["orders"], "columns": []},
                       question_type="comparison")

    class DownClient:
        calls = 0
        def call(self, messages, **kw):
            DownClient.calls += 1
            raise RuntimeError("503")

    _check("probe failure rejects (fail-closed)",
           not B._admit_nontrivial(tp, None, (anchor,), DownClient()))
    _check("probe retried once before rejecting", DownClient.calls == 2)

    class GuessClient:
        def call(self, messages, **kw):
            return "roughly 1200", {}

    _check("numeric guess still rejected",
           not B._admit_nontrivial(tp, None, (anchor,), GuessClient()))

    cat_anchor = B.Anchor(
        a_id="a2", q_probe="SELECT region", r="region    \n----------\nEU        \n",
        r_rowcount=1, r_signature="s", c={"tables_used": ["orders"]},
        U_a=["u1"], scope_id="g00", category="lookup",
    )

    class LabelClient:
        def call(self, messages, **kw):
            return "It is EU.", {}

    _check("categorical guess rejected on label",
           not B._admit_nontrivial(tp, None, (cat_anchor,), LabelClient()))


def test_hint_is_sources_only():
    class _Unit:
        def __init__(self, title):
            self.title = title
            self.format = "sqlite_table"

    class _Edge:
        kind = "join"
        meta = {"col_pair": ["region", "region"]}
        src, dst = "u1", "u2"

    class _Graph:
        units = {"u1": _Unit("orders"), "u2": _Unit("marketing")}

    class _Path:
        nodes = ["u1", "u2"]
        edges = [_Edge()]
        level = 2

    h = B._hint_for_path(_Path(), _Graph())
    _check("hint carries the source names",
           h["entry_units"] == ["orders", "marketing"])
    _check("hint carries nothing else", set(h) == {"entry_units"})


_RANK_TIE = ("region      n         \n"
             "----------  ----------\n"
             "A           100       \n"
             "B           90        \n"
             "C           90        \n")
_RANK_SEP = ("region      n         \n"
             "----------  ----------\n"
             "A           100       \n"
             "B           90        \n"
             "C           50        \n")


def _anchor(r, q_probe="", category="probe", U_a=("u1",)):
    return B.Anchor(
        a_id="x", q_probe=q_probe, r=r, r_rowcount=0, r_signature="",
        c={}, U_a=list(U_a), scope_id="g00", category=category)


def test_margin_gate():
    a = _anchor("", q_probe="SELECT region, n FROM orders ORDER BY n DESC LIMIT 2")
    _check("near-tie at the rank-k boundary rejected", not B._margin_ok(a, _RANK_TIE))
    _check("separated rank-k boundary passes", B._margin_ok(a, _RANK_SEP))
    plain = _anchor("", q_probe="SELECT region, SUM(n) FROM orders GROUP BY region")
    _check("non-selection query skips the gate", B._margin_ok(plain, _RANK_TIE))


def test_family_preconditions():
    class _U:
        stat = {"cols": {}}

    class _G:
        units = {"u1": _U()}

    class _P1:
        edges = []
        nodes = ["u1"]
        level = 1

    one_group = _anchor("grp    val   \n-----  -----\nA      5     \n")
    types1 = B._applicable_question_types(_P1(), [one_group], _G())
    _check("comparison not drawn on a single-group path", "comparison" not in types1)
    _check("anomaly not drawn on a single-group path", "anomaly" not in types1)
    _check("verification applies to any numeric result", "verification" in types1)

    two_groups = _anchor("grp    val   \n-----  -----\nA      5     \nB      9     \n")
    types2 = B._applicable_question_types(_P1(), [two_groups], _G())
    _check("comparison applies to a multi-group measure", "comparison" in types2)
    _check("anomaly applies when dispersion is nonzero", "anomaly" in types2)


def test_teacher_decline_parse():
    _check("explicit decline parsed", B._realize_is_decline({"decline": True}))
    _check("empty question is a decline", B._realize_is_decline({"question": ""}))
    _check("NONE question is a decline", B._realize_is_decline({"question": "NONE"}))
    _check("a real question is not a decline",
           not B._realize_is_decline({"question": "What drove the gap?"}))
    _check("no parse is not a decline", not B._realize_is_decline(None))


def test_contract_compilation_from_stdout():
    rows = B._key_rows_from_stdout(_SQL_OUT_TWO_ROWS)
    _check("first text cell is the label, first numeric cell the value",
           rows == [{"label": "EU", "value": 1200.0, "raw": "EU 1200"},
                    {"label": "NA", "value": 900.0, "raw": "NA 900"}])
    _check("row cap raised on the final query",
           B._strip_row_cap("SELECT x FROM t ORDER BY x LIMIT 10")
           == f"SELECT x FROM t ORDER BY x LIMIT {B._KEY_ROW_CAP}")
    _check("no LIMIT left untouched",
           B._strip_row_cap("SELECT x FROM t") == "SELECT x FROM t")


def test_answer_extraction_unified():
    from toffee.utils import extract_answer_text
    json_reply = ('<think>compute</think>```json\n'
                  '{"answer": "EU revenue is 1200 and NA share is 17.5.",'
                  ' "reasoning": "joined 3 tables, 42 rows"}\n```')
    ans = extract_answer_text(json_reply)
    _check("json reply yields the answer field",
           ans == "EU revenue is 1200 and NA share is 17.5.")
    _check("reasoning numbers do not count as stated evidence", "42" not in ans)
    _check("bare scalar reply falls back to raw text",
           extract_answer_text("42") == "42")
    _check("plain reply passes through after think block",
           extract_answer_text("<think>x</think>The gap is 30%.") == "The gap is 30%.")
    _check("trailing tool-call payload stripped",
           extract_answer_text('Done: 1200 rows. {"tool_name": "execute_sql"}')
           == "Done: 1200 rows.")


def test_literal_taint():
    ans = "EU revenue is 1200 and NA share is 17.5."
    src = ["orders", "marketing"]
    projected = [{"tool": "execute_sql",
                  "text": "SELECT region, 1200 AS revenue, 17.5 AS share "
                          "FROM orders WHERE region = 'EU'",
                  "output": "EU 1200 17.5"}]
    _check("literal projection over a source rejected by the taint check",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=projected, source_names=src) == 0.5)
    const_join = [{"tool": "execute_sql",
                   "text": "SELECT o.region, 1200 AS revenue, 17.5 AS share "
                           "FROM orders o JOIN marketing m ON o.region = m.region",
                   "output": "EU 1200 17.5"}]
    _check("constant projected over a join of required sources rejected",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=const_join, source_names=src) == 0.5)
    py_const = [{"tool": "execute_python",
                 "text": "import sqlite3\nconn = sqlite3.connect('scratch.sqlite')\n"
                         "print('EU', 1200)\nprint('NA', 17.5)",
                 "output": "EU 1200\nNA 17.5"}]
    _check("Python constant caught by the syntax-tree taint rule",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=py_const, source_names=src) == 0.5)
    computed = [{"tool": "execute_sql", "text": _ORDERS_SQL,
                 "output": "EU 1200\nNA 17.5"}]
    _check("genuinely computed values untouched by the taint rule",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=computed, source_names=src) >= 0.95)
    limit_only = [{"tool": "execute_sql",
                   "text": _ORDERS_SQL + " LIMIT 1200",
                   "output": "EU 1200\nNA 17.5"}]
    _check("a LIMIT row cap is not a value literal",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=limit_only, source_names=src) >= 0.95)


def test_wide_dump_rule():
    ans = "EU revenue is 1200 and NA share is 17.5."
    src = ["orders", "marketing"]
    body = "\n".join(f"row{i}       {i}" for i in range(150))
    dump_out = ("region      revenue   \n----------  ----------\n"
                "EU          1200      \nNA          17.5      \n" + body)
    dump = [{"tool": "execute_sql", "text": "SELECT region, revenue FROM orders",
             "output": dump_out}]
    _check("an unfiltered scan over 100 rows cannot back an answer",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=dump, source_names=src) == 0.5)
    filtered = [{"tool": "execute_sql",
                 "text": "SELECT region, revenue FROM orders "
                         "WHERE region IN (SELECT region FROM marketing)",
                 "output": dump_out}]
    _check("the same output behind a filter still counts",
           vq_score(ans, "", KEY_ROWS,
                                    data_steps=filtered, source_names=src) >= 0.95)


def _covered_fixture():
    import os
    import sqlite3
    import tempfile
    from toffee.core.executor import execute_sql
    fd, db = tempfile.mkstemp(prefix="toffee_cov_", suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (region TEXT, revenue REAL)")
    conn.execute("CREATE TABLE marketing (region TEXT, spend REAL)")
    conn.executemany("INSERT INTO orders VALUES (?, ?)",
                     [("EU", 700.0), ("EU", 500.0), ("NA", 900.0)])
    conn.executemany("INSERT INTO marketing VALUES (?, ?)",
                     [("EU", 40.0)])
    conn.commit()
    conn.close()

    def run(sql):
        res = execute_sql(db, sql)
        return {"tool": "execute_sql", "text": sql, "output": res.output}

    return db, run


def test_covered_intervention_and_lineage():
    import os
    from toffee.search.evaluator import certify_answer
    db, run = _covered_fixture()
    key_rows = [{"label": "EU", "value": 1200.0, "raw": "EU 1200"}]
    src = ["orders", "marketing"]
    required = ["orders", "marketing"]
    ans = "EU revenue is 1200."
    try:
        dependent = [run("SELECT region, SUM(revenue) FROM orders "
                         "WHERE region IN (SELECT region FROM marketing) "
                         "GROUP BY region")]
        v, detail = certify_answer(ans, "", key_rows, data_steps=dependent,
                                   source_names=src, required_sources=required,
                                   db_path=db)
        _check("dependency-checked answer certifies", v >= 0.95)
        _check("intervention replays were executed",
               detail["intervention_replays"] >= 1)
        _check("lineage records the supporting query",
               detail["lineage"] and "orders" in detail["lineage"][0]["query"])
        _check("lineage records the output row",
               detail["lineage"][0]["output_row"] is not None)

        decorative = [run("SELECT o.region, SUM(o.revenue) FROM orders o "
                          "LEFT JOIN marketing m ON m.region = 'NOPE' "
                          "GROUP BY o.region")]
        v2, detail2 = certify_answer(ans, "", key_rows, data_steps=decorative,
                                     source_names=src, required_sources=required,
                                     db_path=db)
        _check("a decorative read fails the intervention test", v2 == 0.5)
        _check("the failing source is named",
               "marketing" in detail2.get("covered_fail", ""))

        orders_only = [run("SELECT region, SUM(revenue) FROM orders "
                           "GROUP BY region")]
        v3, detail3 = certify_answer(ans, "", key_rows, data_steps=orders_only,
                                     source_names=src, required_sources=required,
                                     db_path=db)
        _check("a required source never read fails Covered", v3 == 0.5)
        _check("read-coverage failure is reported",
               "marketing" in detail3.get("covered_fail", ""))

        v4, _d4 = certify_answer(ans, "", key_rows, data_steps=dependent,
                                 source_names=src, required_sources=["orders"],
                                 db_path=db)
        _check("single-source requirement passes on its own read", v4 >= 0.95)
    finally:
        os.unlink(db)


def main():
    test_labels_from_result()
    test_verifier_answer_side()
    test_verifier_computed_gate()
    test_verifier_trace_side()
    test_contract_coverage_and_crosslabel()
    test_acceptance_recomputes_certificate()
    test_prefix_outputs_count_for_certificate()
    test_margin_gate()
    test_family_preconditions()
    test_teacher_decline_parse()
    test_contract_compilation_from_stdout()
    test_nondeg_fail_closed()
    test_hint_is_sources_only()
    test_answer_extraction_unified()
    test_literal_taint()
    test_wide_dump_rule()
    test_covered_intervention_and_lineage()
    print("\nAll certificate unit checks passed.")


if __name__ == "__main__":
    main()
