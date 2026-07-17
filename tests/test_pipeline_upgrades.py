
from toffee import config as _cfg
from toffee.core.state import AnalysisState
from toffee.generation import bottomup as B
from toffee.utils import BudgetTracker


def _check(name, cond):
    if not cond:
        raise AssertionError("FAILED: " + name)
    print("ok -", name)


def test_execution_metering():
    budget = BudgetTracker(max_dollars=1.00)
    cost = budget.record_execution(3600.0)
    _check("one vCPU-hour bills the configured rate",
           abs(cost - _cfg.EXEC_VCPU_RATE_PER_HOUR) < 1e-9)
    _check("execution charges against the same budget",
           abs(budget.spent_dollars - cost) < 1e-9)
    _check("execution spend tracked separately",
           abs(budget.spent_exec_dollars - cost) < 1e-9)
    _check("execution count and seconds recorded",
           budget.exec_count == 1 and abs(budget.exec_seconds - 3600.0) < 1e-9)
    budget.record(0.10, 100)
    _check("model and execution spend share one total",
           abs(budget.spent_dollars - (cost + 0.10)) < 1e-9)
    _check("a typical sub-second execution costs well under a cent",
           BudgetTracker().record_execution(0.14) < 0.0001)


def test_observable_state_features():
    task = {"task_id": "t1", "question": "q", "level": 3,
            "source_span": 1.0, "format_span": 0.75,
            "hint": {"entry_units": ["orders"]}, "env": {}}
    state = AnalysisState.from_task(task)
    x = state.feature_vector()
    _check("provenance dims zeroed by default (observable-only LCM)",
           all(abs(v) < 1e-12 for v in x[6:11]))
    old = _cfg.PROVENANCE_FEATURES
    _cfg.PROVENANCE_FEATURES = True
    try:
        x2 = state.feature_vector()
        _check("+provenance ablation arm restores the dims",
               x2[8] == 1.0 and x2[9] == 1.0 and x2[10] == 0.75)
    finally:
        _cfg.PROVENANCE_FEATURES = old
    _check("execution-state dims unaffected by the gate",
           list(x[:6]) == list(state.feature_vector()[:6]))


def test_admission_battery_replay():
    orig = (B._admit_stable, B._admit_accessible, B._admit_nontrivial, B._admit_solvable)
    calls = {"nondeg": 0}
    B._admit_stable = lambda H, a: True
    B._admit_accessible = lambda H, a: True
    B._admit_solvable = lambda t, H, a: True

    def flaky_nondeg(t, H, a, c):
        calls["nondeg"] += 1
        return calls["nondeg"] == 1

    B._admit_nontrivial = flaky_nondeg
    try:
        stats = {k: 0 for k in ("rej_stable", "rej_accessible", "rej_nontrivial",
                                "rej_solvable", "rej_battery_replay")}
        _check("first battery passes",
               B._admission_battery(None, None, (), None, stats))
        _check("the closing replay reruns every check and can reject",
               not B._admission_battery(None, None, (), None, stats, replay=True))
        _check("probe called afresh on the replay", calls["nondeg"] == 2)
        _check("replay rejection not double-counted in first-run stats",
               stats["rej_nontrivial"] == 0)
        _check("admission replay on by default", _cfg.ADMISSION_REPLAY)
    finally:
        (B._admit_stable, B._admit_accessible,
         B._admit_nontrivial, B._admit_solvable) = orig


def test_guessable_key_dropped():
    _check("all-boolean values dropped",
           B._key_guessable([{"label": "flagged", "value": 1.0},
                             {"label": "rest", "value": 0.0}]))
    _check("bare yes/no labels dropped",
           B._key_guessable([{"label": "Yes", "value": None}]))
    _check("empty key_rows dropped", B._key_guessable([]))
    _check("a real measure kept",
           not B._key_guessable([{"label": "EU", "value": 1200.0}]))
    _check("mixed values kept when any is informative",
           not B._key_guessable([{"label": "EU", "value": 1.0},
                                 {"label": "NA", "value": 900.0}]))


def test_baseline_strategy_surface():
    from toffee.search import baselines as BL
    _check("rejection sampling implemented",
           callable(getattr(BL, "rejection_sampling_search", None)))
    _check("LATS (adapted) implemented",
           callable(getattr(BL, "lats_search", None)))
    _check("every baseline accepts the shared prefix cache",
           all("prefix_cache" in f.__code__.co_varnames for f in (
               BL.single_pass_search, BL.react_search, BL.greedy_search,
               BL.best_of_n_search, BL.beam_search,
               BL.rejection_sampling_search, BL.lats_search)))
    _check("every baseline accepts the +LCM arm",
           all("lcm" in f.__code__.co_varnames for f in (
               BL.single_pass_search, BL.react_search, BL.greedy_search,
               BL.best_of_n_search, BL.beam_search,
               BL.rejection_sampling_search, BL.lats_search)))


def test_lcm_pick_for_baselines():
    from toffee.search.baselines import _lcm_pick
    task = {"task_id": "t2", "question": "q", "env": {},
            "hint": {"entry_units": ["orders"]}}
    state = AnalysisState.from_task(task)

    class FakeLCM:
        def rank_actions(self, x, acts):
            return [(1.0 if a.model.endswith("gpt-5.4") else 0.2, 0.1, a)
                    for a in acts]

    picked = _lcm_pick(state, "SchemaScout", FakeLCM())
    _check("+LCM arm picks the top-ranked configuration",
           picked is not None and picked.model.endswith("gpt-5.4"))
    _check("infeasible operator yields no pick",
           _lcm_pick(state, "NoSuchOp", FakeLCM()) is None)


def main():
    test_execution_metering()
    test_observable_state_features()
    test_admission_battery_replay()
    test_guessable_key_dropped()
    test_baseline_strategy_surface()
    test_lcm_pick_for_baselines()
    print("\nAll pipeline-upgrade unit checks passed.")


if __name__ == "__main__":
    main()
