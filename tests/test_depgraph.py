
import random

from toffee.generation import depgraph as D
from toffee.generation.depgraph import DepEdge, DepGraph, Path


def _check(name, cond):
    if not cond:
        raise AssertionError("FAILED: " + name)
    print("ok -", name)


def test_containment():
    _check("containment full", D.containment({"a", "b"}, {"a", "b", "c"}) == 1.0)
    _check("containment half", D.containment({"a", "x"}, {"a", "b"}) == 0.5)
    _check("containment none", D.containment({"x"}, {"a", "b"}) == 0.0)
    _check("containment empty guard", D.containment(set(), {"a"}) == 0.0)


def test_parse_column_output():
    out = ("Name         Population\n"
           "-----------  ----------\n"
           "Aruba        103000    \n"
           "Afghanistan  22720000  \n")
    header, rows = D.parse_column_output(out)
    _check("parse header", header == ["Name", "Population"])
    _check("parse rows", rows == [["Aruba", "103000"], ["Afghanistan", "22720000"]])

    spaced = ("Name                  Code\n"
              "--------------------  ----\n"
              "United Arab Emirates  ARE \n")
    h2, r2 = D.parse_column_output(spaced)
    _check("parse value-with-spaces", r2 == [["United Arab Emirates", "ARE"]])

    _check("parse empty", D.parse_column_output("") == ([], []))


def _toy_graph():

    units = {u: type("U", (), {"title": u})() for u in ("a", "b", "c", "d")}
    e_join = DepEdge("a", "b", "join", {"col_pair": ["k", "k"], "tables": ["a", "b"]})
    e_chain = DepEdge("b", "c", "chain",
                      {"bridge_values": ["X1", "Y2", "Z3"], "target_col": "col",
                       "verify_sql": "", "verify_exec_db": ""})
    return DepGraph(nodes=["a", "b", "c", "d"], edges=[e_join, e_chain], units=units)


def test_sample_path_shapes():
    g = _toy_graph()
    rng = random.Random(1)
    p1 = D.sample_path(g, 1, rng)
    _check("L1 single node", p1.level == 1 and len(p1.nodes) == 1 and p1.edges == [])

    p2 = D.sample_path(g, 2, rng)
    _check("L2 one edge two nodes", p2.level == 2 and len(p2.edges) == 1 and len(p2.nodes) == 2)


    got3 = False
    for _ in range(50):
        p3 = D.sample_path(g, 3, random.Random(_))
        if p3 is not None:
            got3 = True
            assert p3.level == 3 and len(p3.nodes) == 3 and len(p3.edges) == 2
            assert len(set(p3.nodes)) == 3
    _check("L3 three nodes / two edges / simple path", got3)

    _check("supported levels", D.supported_levels(g) == [1, 2, 3])


    g0 = DepGraph(nodes=["a"], edges=[], units={"a": type("U", (), {"title": "a"})()})
    _check("no-edge L2 None", D.sample_path(g0, 2, rng) is None)
    _check("no-edge levels", D.supported_levels(g0) == [1])


def test_bridge_and_leak():
    good = DepEdge("u", "v", "chain", {"bridge_values": ["X1", "Y2"]})
    empty = DepEdge("u", "v", "chain", {"bridge_values": []})
    too_many = DepEdge("u", "v", "chain", {"bridge_values": [str(i) for i in range(20)]})
    _check("bridge cap ok", D.bridge_unique_ok(good))
    _check("bridge empty rejected", not D.bridge_unique_ok(empty))
    _check("bridge overflow rejected", not D.bridge_unique_ok(too_many))
    _check("join edge always unique-ok", D.bridge_unique_ok(DepEdge("u", "v", "join", {})))

    bridges = ["CHN", "USA", "42"]
    _check("leak detects verbatim",
           D.question_leaks("Which region drove growth in CHN last year?", bridges))
    _check("leak ignores short values",
           not D.question_leaks("The answer is 42 units", ["42"]))
    _check("no leak when absent",
           not D.question_leaks("Rank regions by growth", bridges))

    p = Path(level=2, nodes=["u", "v"],
             edges=[DepEdge("u", "v", "chain", {"bridge_values": ["CHN"], "target_col": "c"})])
    _check("collect_bridge_values", D.collect_bridge_values(p) == ["CHN"])


def test_name_correspondence():

    _check("AlbumId corresponds", D._name_corresponds("AlbumId", "AlbumId"))
    _check("CustomerId corresponds", D._name_corresponds("CustomerId", "CustomerId"))
    _check("FirstName corresponds", D._name_corresponds("FirstName", "FirstName"))
    _check("bare id does not correspond", not D._name_corresponds("id", "id"))
    _check("bare Name does not correspond", not D._name_corresponds("Name", "Name"))
    _check("different names do not correspond",
           not D._name_corresponds("AlbumId", "EmployeeId"))
    _check("case-insensitive", D._name_corresponds("customerid", "CustomerID"))
    _check("name_similarity exact=1", D._name_similarity("AlbumId", "AlbumId") == 1.0)
    _check("name_similarity disjoint=0", D._name_similarity("AlbumId", "Price") == 0.0)


def test_declared_fk():


    import sqlite3, tempfile, os
    d = tempfile.mkdtemp(prefix="depgraph_fk_")
    db = os.path.join(d, "fk.sqlite")
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("CREATE TABLE parent(pid INTEGER PRIMARY KEY, label TEXT)")
    cur.execute("CREATE TABLE child(cid INTEGER PRIMARY KEY, ref INTEGER, "
                "amt INTEGER, FOREIGN KEY(ref) REFERENCES parent(pid))")
    cur.executemany("INSERT INTO parent VALUES(?,?)",
                    [(1, "a"), (2, "b"), (3, "c"), (4, "d")])
    cur.executemany("INSERT INTO child VALUES(?,?,?)",
                    [(10, 1, 5), (11, 2, 6), (12, 1, 7), (13, 3, 8)])
    con.commit(); con.close()

    from toffee.generation.ingest import ingest_environment
    units = ingest_environment([db], os.path.join(d, "scratch.sqlite"))
    fks = D.declared_fks(units)
    _check("declared FK discovered",
           any(ct == "child" and cc == "ref" and pt == "parent" and pc == "pid"
               for _db, ct, cc, pt, pc in fks))

    child = next(u for u in units if u.title == "child")
    parent = next(u for u in units if u.title == "parent")
    _check("declared pair oriented child->parent",
           D._declared_pair_for(child, parent, fks) == ("ref", "pid"))
    _check("declared pair oriented parent->child",
           D._declared_pair_for(parent, child, fks) == ("pid", "ref"))

    g = D.build_dependency_graph(units)
    join_by = [e.meta.get("admitted_by") for e in g.edges if e.kind == "join"]
    _check("declared FK join admitted as declared", "declared" in join_by)
    _check("graph stats count declared", g.stats["join_kept_declared"] >= 1)


def test_date_bridge():
    _check("timestamp bridge is date-only",
           D._bridge_is_date_only(["2008-04-30 00:00:00"]))
    _check("date bridge is date-only", D._bridge_is_date_only(["2008-04-30", "2009/01/02"]))
    _check("mixed bridge not date-only",
           not D._bridge_is_date_only(["2008-04-30", "USD"]))
    _check("value bridge not date-only", not D._bridge_is_date_only(["USD", "EUR"]))
    _check("empty bridge not date-only", not D._bridge_is_date_only([]))


def test_join_counterfactual():

    import sqlite3, tempfile, os
    d = tempfile.mkdtemp(prefix="depgraph_cf_")
    db = os.path.join(d, "t.sqlite")
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("CREATE TABLE dim(code TEXT, region TEXT, misc INTEGER)")
    cur.executemany("INSERT INTO dim VALUES(?,?,?)",
                    [("US", "NA", 1), ("GB", "EU", 2), ("DE", "EU", 3), ("FR", "EU", 4)])
    cur.execute("CREATE TABLE fact(code TEXT, region TEXT, amt INTEGER)")
    cur.executemany("INSERT INTO fact VALUES(?,?,?)",
                    [("US", "NA", 10), ("GB", "EU", 20), ("US", "NA", 30), ("DE", "EU", 5)])
    con.commit(); con.close()

    from toffee.generation.ingest import ingest_environment
    scratch = os.path.join(d, "scratch.sqlite")
    units = ingest_environment([db], scratch)
    ubyid = {u.unit_id: u for u in units}
    fact = next(u for u in units if u.title == "fact")
    dim = next(u for u in units if u.title == "dim")


    e_code = D.DepEdge(fact.unit_id, dim.unit_id, "join", {"col_pair": ["code", "code"]})
    res = D.counterfactual_join_ok(e_code, fact, dim)
    _check("join counterfactual returns bool", isinstance(res, bool))

    e_solo = D.DepEdge(fact.unit_id, dim.unit_id, "join", {"col_pair": ["amt", "misc"]})

    _check("no-decoy join rejected", D.counterfactual_join_ok(e_solo, fact, dim) is False)


def _mk_unit(title, cols):
    return type("U", (), {
        "title": title, "logical_table": title,
        "sch": [(c, "TEXT", "") for c in cols],
    })()


def test_candidate_paths():
    orders = _mk_unit("orders", ["region", "revenue", "quarter"])
    marketing = _mk_unit("marketing", ["region", "spend", "campaign"])
    demo = _mk_unit("demographics", ["city", "population"])
    units = {"orders": orders, "marketing": marketing, "demographics": demo}
    e_join = DepEdge("orders", "marketing", "join",
                     {"col_pair": ["region", "region"], "tables": ["orders", "marketing"]})
    e_chain = DepEdge("marketing", "demographics", "chain",
                      {"bridge_values": ["N"], "target_col": "city"})
    g = DepGraph(nodes=list(units), edges=[e_join, e_chain], units=units)


    cands = D.candidate_paths_for_question(
        "How does the revenue decline relate to marketing spend by region?", g)
    _check("two-source question has candidates", len(cands) >= 1)
    _check("two-source question -> level-2 join path first",
           cands[0].level == 2 and set(cands[0].nodes) == {"orders", "marketing"})


    cands1 = D.candidate_paths_for_question("Total population by city", g)
    _check("single-source question -> level-1 first",
           cands1[0].level == 1 and cands1[0].nodes == ["demographics"])


    _check("unrelated question -> no path",
           D.candidate_paths_for_question("thermal conductivity of copper", g) == [])


def test_verify_facts():
    from toffee.search.evaluator import vq_score
    key_rows = [{"label": "EU", "value": 1200.0, "raw": "EU 1200"},
                {"label": "NA", "value": 17.5, "raw": "NA 17.5"}]
    _check("every row stated near its label -> certified",
           vq_score(
               "EU fell to 1200 while NA share was 17.5%", "", key_rows) >= 0.95)
    _check("1% tolerance witnesses", vq_score(
        "EU about 1195 and NA 17.4", "", key_rows) >= 0.95)
    _check("one row missing -> rejected",
           vq_score("EU fell to 1200.", "", key_rows) == 0.0)
    _check("nothing stated -> rejected",
           vq_score("It went down a lot.", "", key_rows) == 0.0)
    _check("no key_rows falls back to the answer key",
           vq_score("42", "42", []) == 1.0)
    _check("answer-key match cannot bypass missing coverage",
           vq_score("42", "42", key_rows) == 0.5)


def test_path_admission_pipeline():


    import sqlite3, tempfile, os
    d = tempfile.mkdtemp(prefix="depgraph_pipe_")
    db = os.path.join(d, "t.sqlite")
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("CREATE TABLE orders(region TEXT, revenue INTEGER)")
    cur.executemany("INSERT INTO orders VALUES(?,?)",
                    [("NA", 10), ("EU", 20), ("NA", 30), ("APAC", 5)])
    cur.execute("CREATE TABLE marketing(region TEXT, spend INTEGER)")
    cur.executemany("INSERT INTO marketing VALUES(?,?)",
                    [("NA", 7), ("EU", 9), ("EU", 4)])
    con.commit(); con.close()

    from toffee.generation.ingest import ingest_environment
    from toffee.generation import bottomup as B
    units = ingest_environment([db], os.path.join(d, "scratch.sqlite"))
    g = D.build_dependency_graph(units)
    joins = [e for e in g.edges if e.kind == "join"]
    _check("join edge discovered on real db", len(joins) >= 1)
    e = joins[0]
    p = Path(level=2, nodes=[e.src, e.dst], edges=[e])

    hint = B._hint_for_path(p, g)
    _check("hint names sources", len(hint["entry_units"]) == 2)
    _check("hint carries nothing else", set(hint) == {"entry_units"})

    anchors = B._execute_path(p, g)
    _check("path executes to anchors", anchors is not None and len(anchors) >= 1)
    facts = B._provenance_from_anchors(anchors)
    _check("facts carry step numerics", facts and facts[-1]["nums"])
    _check("answer key non-empty", bool(anchors[-1].r.strip()))

    H = B._hierarchy_for_path(p, g, anchors)
    tp = B.TaskPackage(x="q", path=list(p.nodes), level=2,
                       hint=hint, question_type="comparison")
    _check("solvable skeleton passes on live path",
           B._admit_solvable(tp, H, tuple(anchors)))
    _check("replay passes on live path", B._admit_stable(H, tuple(anchors)))


def main():
    test_containment()
    test_parse_column_output()
    test_sample_path_shapes()
    test_bridge_and_leak()
    test_name_correspondence()
    test_join_counterfactual()
    test_declared_fk()
    test_date_bridge()
    test_candidate_paths()
    test_verify_facts()
    test_path_admission_pipeline()
    print("\nAll depgraph unit checks passed.")


if __name__ == "__main__":
    main()
