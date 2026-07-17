
from toffee import cli
from toffee import config as _cfg


def _check(name, cond):
    if not cond:
        raise AssertionError("FAILED: " + name)
    print("ok -", name)


def test_task_level():
    _check("level 1", cli._task_level({"level": 1}) == 1)
    _check("level 2", cli._task_level({"level": 2}) == 2)
    _check("level 3", cli._task_level({"level": 3}) == 3)
    _check("missing level -> 1", cli._task_level({}) == 1)
    _check("out-of-range level -> 1", cli._task_level({"level": 4}) == 1)
    _check("non-numeric level -> 1", cli._task_level({"level": "weird"}) == 1)
    _check("legacy M2 key still read", cli._task_level({"motif": "M2"}) == 2)


def test_multiplier_and_effective_budget():
    orig = _cfg.LEVEL_BUDGET_SCALING
    try:

        _cfg.LEVEL_BUDGET_SCALING = False
        _check("flag off level 3 -> x1", cli._level_budget_multiplier({"level": 3}) == 1)
        _check("flag off level 1 -> x1", cli._level_budget_multiplier({"level": 1}) == 1)


        _cfg.LEVEL_BUDGET_SCALING = True
        _check("flag on level 1 -> x1", cli._level_budget_multiplier({"level": 1}) == 1)
        _check("flag on level 2 -> x2", cli._level_budget_multiplier({"level": 2}) == 2)
        _check("flag on level 3 -> x3", cli._level_budget_multiplier({"level": 3}) == 3)
        _check("flag on missing level -> x1", cli._level_budget_multiplier({}) == 1)


        base_budget, base_iters = 0.30, 10
        for level, exp in ((1, 1), (2, 2), (3, 3), (None, 1)):
            task = {"level": level} if level else {}
            m = cli._level_budget_multiplier(task)
            _check(f"eff budget level {level or '-'}",
                   base_budget * m == 0.30 * exp)
            _check(f"eff iters level {level or '-'}",
                   base_iters * m == 10 * exp)
    finally:
        _cfg.LEVEL_BUDGET_SCALING = orig


def main():
    test_task_level()
    test_multiplier_and_effective_budget()
    print("\nAll level-budget unit checks passed.")


if __name__ == "__main__":
    main()
