
from __future__ import annotations

import logging
import time as _time
from typing import Dict, List, Optional, Tuple

from toffee.config import EFFORT_LEVELS, MAX_CONSECUTIVE_ERRORS, MAX_STEP_COUNT
from toffee.core.executor import ToolResult, execute_tool
from toffee.core.operators import OPERATORS, ActionConfig, _successful_data_steps
from toffee.core.state import AnalysisState
from toffee.search.evaluator import evaluate, heuristic_evaluate
from toffee.search.mcts import (
    SearchOutcome,
    Trajectory,
    _acceptance_check,
    _parse_tool_call,
    _trajectory_rank_key,
)
from toffee.utils import BudgetTracker

log = logging.getLogger(__name__)


def _get_baseline_model():
    import os as _os
    from toffee import config
    override = _os.environ.get("TOFFEE_BASELINE_MODEL")
    if override:
        return config.MODELS.get(override, override)
    return config.MODELS.get("capable", "anthropic/claude-sonnet-4.6")
BASELINE_HISTORY = "long"
BASELINE_EFFORT = "extended"
BASELINE_TEMP = 0.4
DIVERSITY_TEMP = 0.7


_OP_MAX_TOKENS = {
    "SchemaScout": 1024, "DataInspect": 1024,
    "SQLDraft": 2048, "SQLRepair": 2048,
    "PythonCompute": 4096, "SanityCheck": 2048,
    "AnswerCompose": 8192,
}


def _select_next_operator(state: AnalysisState, allow_repair: bool = True) -> Optional[str]:
    data_steps = _successful_data_steps(state)


    if allow_repair and state.has_error and state.consecutive_errors < MAX_CONSECUTIVE_ERRORS:
        if OPERATORS["SQLRepair"].is_feasible(state):
            return "SQLRepair"


    if not state.schema_discovered:
        return "SchemaScout"


    if data_steps < 3:
        if OPERATORS["SQLDraft"].is_feasible(state):
            return "SQLDraft"


    if data_steps < 5 and state.step_count < MAX_STEP_COUNT - 3:
        if data_steps % 2 == 1 and OPERATORS["PythonCompute"].is_feasible(state):
            return "PythonCompute"
        if OPERATORS["SQLDraft"].is_feasible(state):
            return "SQLDraft"


    if (state.result_exists and data_steps >= 2
            and not state.verification_passed
            and OPERATORS["SanityCheck"].is_feasible(state)):
        return "SanityCheck"


    if OPERATORS["AnswerCompose"].is_feasible(state):
        return "AnswerCompose"


    if OPERATORS["SQLDraft"].is_feasible(state):
        return "SQLDraft"
    if OPERATORS["PythonCompute"].is_feasible(state):
        return "PythonCompute"

    return None


def _prefix_root(task: dict, prefix_cache=None, metrics: Optional[Dict] = None) -> AnalysisState:
    """Root state with prefix-cache reuse. The prefix cache serves every
    method: repeated schema steps within an environment reuse it regardless
    of the search structure."""
    root = AnalysisState.from_task(task)
    if prefix_cache is None:
        return root
    prefix_state = prefix_cache.find_reusable_prefix(root)
    if prefix_state and not prefix_state.is_terminal():
        root = prefix_state.clone_with_new_provenance(task.get("task_id", "?"))
        if metrics is not None:
            metrics["prefix_reused"] = True
            metrics["prefix_depth"] = root.step_count
        log.info("Reusing cached prefix (depth=%d) for task %s",
                 root.step_count, task.get("task_id", "?"))
    return root


def _lcm_pick(state: AnalysisState, operator_name: str, lcm):
    """Rank the feasible configurations of the chosen operator with the \\lcm,
    for the '+ LCM' arms that give a baseline the same configuration policy."""
    from toffee.core.operators import enumerate_feasible_actions
    acts = [a for a in enumerate_feasible_actions(state)
            if a.operator_name == operator_name]
    if not acts:
        return None
    ranked = lcm.rank_actions(state.feature_vector(), acts)
    if not ranked:
        return None
    return max(ranked, key=lambda r: r[0])[2]


def _execute_step(
    state: AnalysisState,
    operator_name: str,
    task: dict,
    client,
    budget: BudgetTracker,
    temperature: float = BASELINE_TEMP,
    history: str = BASELINE_HISTORY,
    effort: str = BASELINE_EFFORT,
    model: str = None,
    prefix_cache=None,
    lcm=None,
) -> Optional[AnalysisState]:
    picked = None
    x_state = None
    if model is None and lcm is not None:
        picked = _lcm_pick(state, operator_name, lcm)
        if picked is not None:
            x_state = state.feature_vector()
            model, history, effort = picked.model, picked.history, picked.effort
    if model is None:
        model = _get_baseline_model()
    if budget.exhausted():
        return None

    operator = OPERATORS[operator_name]
    messages = operator.build_prompt(state, history, task_context=task)
    effort_tokens = EFFORT_LEVELS.get(effort, 8192)
    effort_tokens = min(effort_tokens, _OP_MAX_TOKENS.get(operator_name, 4096))

    try:
        t0 = _time.perf_counter()
        content, usage = client.call(
            messages, model=model,
            max_tokens=effort_tokens, temperature=temperature,
        )
        llm_elapsed = _time.perf_counter() - t0
    except Exception as exc:
        log.warning("LLM call failed for %s: %s", operator_name, exc)
        return None

    budget.record(usage.cost, usage.prompt_tokens + usage.completion_tokens)

    tool_name, arguments = _parse_tool_call(
        content, operator_name, state=state, fallback_tool=operator.tools[0] if operator.tools else ""
    )

    env_context = task.get("env", {})
    if operator_name == "AnswerCompose":
        tool_result = ToolResult(tool_name="none", success=True, stdout=content)
    else:
        tool_result = execute_tool(tool_name, arguments, env_context)
        budget.record_execution(tool_result.elapsed_s)

    config_dict = {"model": model, "history": history, "effort": effort}
    child = state.expand(
        operator_name=operator_name,
        config=config_dict,
        llm_content=content,
        tool_result=tool_result,
        usage=usage,
    )
    if operator_name != "AnswerCompose":
        child.apply_step_judge(None)
    action_key = f"{operator_name}|{tool_name}|{model}|{history}|{effort}"
    child.action_key = action_key
    state.children[action_key] = child
    if prefix_cache is not None:
        prefix_cache.register(child)
    if picked is not None and x_state is not None:
        from toffee.config import MCTS_MAX_ITERATIONS
        from toffee.search.evaluator import compute_reward, heuristic_evaluate
        c_tilde = budget.max_dollars / max(1, MCTS_MAX_ITERATIONS)
        step_reward = compute_reward(
            value=heuristic_evaluate(child), child_state=child,
            cost=usage.cost, cost_estimate=c_tilde,
        )
        lcm.update(x_state, picked.feature_vector(), step_reward,
                      op_name=operator_name)

    log.info("baseline step: op=%s tool=%s success=%s steps=%d cost=$%.4f %.1fs",
             operator_name, tool_name, tool_result.success, child.step_count,
             budget.spent_dollars, llm_elapsed)

    return child


def _forward_pass(
    task: dict,
    client,
    budget: BudgetTracker,
    max_steps: int = MAX_STEP_COUNT,
    temperature: float = BASELINE_TEMP,
    allow_repair: bool = True,
    label: str = "",
    prefix_cache=None,
    lcm=None,
) -> Tuple[AnalysisState, AnalysisState, Dict]:
    task_id = task.get("task_id", "?")
    metrics: Dict = {
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "total_expansions": 0,
        "llm_time_s": 0.0,
        "operator_counts": {},
    }
    root = _prefix_root(task, prefix_cache, metrics)
    state = root

    for step in range(max_steps):
        if state.is_terminal() or budget.exhausted():
            break

        op_name = _select_next_operator(state, allow_repair=allow_repair)
        if op_name is None:
            log.info("[%s%s] no feasible operator at step %d", task_id, label, step)
            break

        t0 = _time.perf_counter()
        child = _execute_step(
            state, op_name, task, client, budget, temperature=temperature,
            prefix_cache=prefix_cache, lcm=lcm,
        )
        elapsed = _time.perf_counter() - t0

        if child is None:
            break

        metrics["total_llm_calls"] += 1
        metrics["total_tool_calls"] += 1
        metrics["total_expansions"] += 1
        metrics["llm_time_s"] += elapsed
        metrics["operator_counts"][op_name] = metrics["operator_counts"].get(op_name, 0) + 1
        state = child

    return root, state, metrics


def _compose_answer_if_needed(
    leaf: AnalysisState,
    task: dict,
    client,
    budget: BudgetTracker,
    metrics: Dict,
    model: Optional[str] = None,
) -> AnalysisState:
    if leaf is None:
        return leaf
    if leaf.is_terminal() or leaf.answer_drafted or not leaf.result_exists:
        return leaf
    if budget.exhausted():
        return leaf
    if not OPERATORS["AnswerCompose"].is_feasible(leaf):
        return leaf
    if "AnswerCompose" in {k.split("|")[0] for k in leaf.children}:
        return leaf

    if model is None:
        model = _get_baseline_model()
    op = OPERATORS["AnswerCompose"]
    msgs = op.build_prompt(leaf, BASELINE_HISTORY, task_context=task)
    max_tokens = min(EFFORT_LEVELS.get(BASELINE_EFFORT, 8192),
                     _OP_MAX_TOKENS.get("AnswerCompose", 8192))
    try:
        t0 = _time.perf_counter()
        content, ac_usage = client.call(
            msgs, model=model, max_tokens=max_tokens, temperature=BASELINE_TEMP,
        )
        metrics["llm_time_s"] = metrics.get("llm_time_s", 0.0) + (_time.perf_counter() - t0)
    except Exception as exc:
        log.warning("baseline AnswerCompose wrapper failed: %s", exc)
        return leaf

    budget.record(ac_usage.cost, ac_usage.prompt_tokens + ac_usage.completion_tokens)
    ac_result = ToolResult(tool_name="none", success=True, stdout=content)
    config_dict = {"model": model, "history": BASELINE_HISTORY, "effort": BASELINE_EFFORT}
    ac_child = leaf.expand(
        operator_name="AnswerCompose",
        config=config_dict,
        llm_content=content,
        tool_result=ac_result,
        usage=ac_usage,
    )
    action_key = f"AnswerCompose|none|{model}|{BASELINE_HISTORY}|{BASELINE_EFFORT}"
    ac_child.action_key = action_key
    leaf.children[action_key] = ac_child

    metrics["total_llm_calls"] = metrics.get("total_llm_calls", 0) + 1
    metrics["total_expansions"] = metrics.get("total_expansions", 0) + 1
    metrics.setdefault("operator_counts", {})
    metrics["operator_counts"]["AnswerCompose"] = metrics["operator_counts"].get("AnswerCompose", 0) + 1
    return ac_child


def _finalize(
    root: AnalysisState,
    leaf: AnalysisState,
    task: dict,
    client,
    budget: BudgetTracker,
    metrics: Dict,
    strategy: str,
) -> SearchOutcome:
    task_id = task.get("task_id", "?")
    leaf = _compose_answer_if_needed(leaf, task, client, budget, metrics)
    traj = Trajectory.extract(root, leaf)
    _before_judge = budget.spent_dollars
    quality = evaluate(leaf, task, client, budget=budget, metrics=metrics)
    metrics["judge_cost"] = metrics.get("judge_cost", 0.0) + max(
        0.0, budget.spent_dollars - _before_judge,
    )
    traj.quality = quality
    accepted, reason = _acceptance_check(traj, task, quality=quality)

    metrics.update({
        "strategy": strategy,
        "task_id": task_id,
        "model_counts": {_get_baseline_model(): metrics["total_llm_calls"]},
        "budget_spent": round(budget.spent_dollars, 4),
    })

    log.info("[%s] %s done: q=%.3f steps=%d data=%d accepted=%s $%.4f",
             task_id, strategy, quality, len(traj.steps), traj.successful_data_steps,
             accepted, budget.spent_dollars)

    return SearchOutcome(
        accepted_trajectory=traj if accepted else None,
        best_trajectory=traj,
        best_quality=quality,
        reject_reason="" if accepted else reason,
        metadata=metrics,
    )


def single_pass_search(
    task: dict,
    client,
    budget: BudgetTracker,
    max_steps: int = MAX_STEP_COUNT,
    prefix_cache=None,
    lcm=None,
) -> SearchOutcome:
    t0 = _time.perf_counter()
    root, leaf, metrics = _forward_pass(
        task, client, budget, max_steps,
        temperature=BASELINE_TEMP, allow_repair=False, prefix_cache=prefix_cache,
        lcm=lcm,
    )
    metrics["search_time_s"] = round(_time.perf_counter() - t0, 2)
    return _finalize(root, leaf, task, client, budget, metrics, "single_pass")


def greedy_search(
    task: dict,
    client,
    budget: BudgetTracker,
    max_steps: int = MAX_STEP_COUNT,
    k: int = 3,
    prefix_cache=None,
    lcm=None,
) -> SearchOutcome:
    from toffee.search.evaluator import heuristic_evaluate

    t0 = _time.perf_counter()
    task_id = task.get("task_id", "?")
    metrics: Dict = {
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "total_expansions": 0,
        "llm_time_s": 0.0,
        "operator_counts": {},
    }
    root = _prefix_root(task, prefix_cache, metrics)
    state = root

    for _step in range(max_steps):
        if state.is_terminal() or budget.exhausted():
            break
        op_name = _select_next_operator(state, allow_repair=True)
        if op_name is None:
            break
        best_child, best_v = None, -2.0
        for _ in range(k):
            if budget.exhausted():
                break
            t1 = _time.perf_counter()
            child = _execute_step(
                state, op_name, task, client, budget,
                temperature=DIVERSITY_TEMP, prefix_cache=prefix_cache, lcm=lcm,
            )
            metrics["llm_time_s"] += _time.perf_counter() - t1
            if child is None:
                continue
            metrics["total_llm_calls"] += 1
            metrics["total_tool_calls"] += 1
            metrics["total_expansions"] += 1
            metrics["operator_counts"][op_name] = (
                metrics["operator_counts"].get(op_name, 0) + 1
            )
            v = heuristic_evaluate(child)
            if v > best_v:
                best_child, best_v = child, v
        if best_child is None:
            break
        state = best_child

    metrics["search_time_s"] = round(_time.perf_counter() - t0, 2)
    log.info("[%s] greedy: committed %d steps", task_id, state.step_count)
    return _finalize(root, state, task, client, budget, metrics, "greedy")


def react_search(
    task: dict,
    client,
    budget: BudgetTracker,
    max_steps: int = MAX_STEP_COUNT,
    prefix_cache=None,
    lcm=None,
) -> SearchOutcome:
    t0 = _time.perf_counter()
    root, leaf, metrics = _forward_pass(
        task, client, budget, max_steps,
        temperature=BASELINE_TEMP, allow_repair=True, prefix_cache=prefix_cache,
        lcm=lcm,
    )
    metrics["search_time_s"] = round(_time.perf_counter() - t0, 2)
    return _finalize(root, leaf, task, client, budget, metrics, "react")


def best_of_n_search(
    task: dict,
    client,
    budget: BudgetTracker,
    n: int = 5,
    max_steps: int = MAX_STEP_COUNT,
    prefix_cache=None,
    lcm=None,
) -> SearchOutcome:
    task_id = task.get("task_id", "?")
    t0 = _time.perf_counter()

    candidates: List[Tuple[Trajectory, float, Dict]] = []

    for i in range(n):
        if budget.exhausted():
            log.info("[%s] best_of_n: budget exhausted after %d/%d runs", task_id, i, n)
            break

        root, leaf, run_metrics = _forward_pass(
            task, client, budget, max_steps,
            temperature=DIVERSITY_TEMP, allow_repair=True,
            label=f" run{i+1}/{n}", prefix_cache=prefix_cache, lcm=lcm,
        )
        leaf = _compose_answer_if_needed(leaf, task, client, budget, run_metrics)
        traj = Trajectory.extract(root, leaf)
        _before_judge = budget.spent_dollars
        quality = evaluate(leaf, task, client, budget=budget, metrics=run_metrics)
        run_metrics["judge_cost"] = run_metrics.get("judge_cost", 0.0) + max(
            0.0, budget.spent_dollars - _before_judge,
        )
        traj.quality = quality
        candidates.append((traj, quality, run_metrics))

        log.info("[%s] best_of_n run %d/%d: q=%.3f steps=%d data=%d",
                 task_id, i + 1, n, quality, len(traj.steps), traj.successful_data_steps)

    if not candidates:
        return SearchOutcome(
            accepted_trajectory=None, best_trajectory=None,
            best_quality=0.0, reject_reason="no trajectories produced",
            metadata={"strategy": "best_of_n", "task_id": task_id, "n_runs": 0},
        )


    best_traj, best_quality, _ = max(
        candidates, key=lambda c: _trajectory_rank_key(c[0], c[1]),
    )
    accepted, reason = _acceptance_check(best_traj, task, quality=best_quality)

    elapsed = _time.perf_counter() - t0
    total_llm = sum(m["total_llm_calls"] for _, _, m in candidates)
    total_ops = {}
    for _, _, m in candidates:
        for op, cnt in m.get("operator_counts", {}).items():
            total_ops[op] = total_ops.get(op, 0) + cnt

    metrics = {
        "strategy": "best_of_n",
        "task_id": task_id,
        "n_runs": len(candidates),
        "n_target": n,
        "per_run_quality": [round(q, 4) for _, q, _ in candidates],
        "total_llm_calls": total_llm,
        "total_tool_calls": total_llm,
        "total_expansions": total_llm,
        "operator_counts": total_ops,
        "model_counts": {_get_baseline_model(): total_llm},
        "search_time_s": round(elapsed, 2),
        "budget_spent": round(budget.spent_dollars, 4),
    }

    log.info("[%s] best_of_n done: best_q=%.3f from %d runs, accepted=%s $%.4f %.1fs",
             task_id, best_quality, len(candidates), accepted, budget.spent_dollars, elapsed)

    return SearchOutcome(
        accepted_trajectory=best_traj if accepted else None,
        best_trajectory=best_traj,
        best_quality=best_quality,
        reject_reason="" if accepted else reason,
        metadata=metrics,
    )


def beam_search(
    task: dict,
    client,
    budget: BudgetTracker,
    beam_width: int = 3,
    max_depth: int = MAX_STEP_COUNT,
    prefix_cache=None,
    lcm=None,
) -> SearchOutcome:
    task_id = task.get("task_id", "?")
    t0 = _time.perf_counter()

    metrics: Dict = {
        "strategy": "beam_search",
        "task_id": task_id,
        "beam_width": beam_width,
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "total_expansions": 0,
        "llm_time_s": 0.0,
        "operator_counts": {},
        "depth_reached": 0,
    }
    root = _prefix_root(task, prefix_cache, metrics)
    beams: List[Tuple[AnalysisState, AnalysisState]] = [(root, root)]
    terminal_pool: List[Tuple[AnalysisState, AnalysisState]] = []

    for depth in range(max_depth):
        if budget.exhausted() or not beams:
            break

        candidates: List[Tuple[AnalysisState, AnalysisState, float]] = []

        for beam_root, state in beams:
            if state.is_terminal():
                terminal_pool.append((beam_root, state))
                continue


            op_name = _select_next_operator(state, allow_repair=False)
            if op_name is None:
                terminal_pool.append((beam_root, state))
                continue

            produced = 0
            for _ in range(beam_width):
                if budget.exhausted():
                    break
                child = _execute_step(
                    state, op_name, task, client, budget,
                    temperature=BASELINE_TEMP, prefix_cache=prefix_cache, lcm=lcm,
                )
                if child is None:
                    continue

                produced += 1
                metrics["total_llm_calls"] += 1
                metrics["total_tool_calls"] += 1
                metrics["total_expansions"] += 1
                metrics["operator_counts"][op_name] = metrics["operator_counts"].get(op_name, 0) + 1

                score = heuristic_evaluate(child)
                candidates.append((beam_root, child, score))

            if produced == 0:
                terminal_pool.append((beam_root, state))

        if not candidates:
            break


        candidates.sort(key=lambda c: c[2], reverse=True)
        beams = [(r, s) for r, s, _ in candidates[:beam_width]]
        metrics["depth_reached"] = depth + 1

        log.info("[%s] beam depth %d: %d candidates, top score=%.3f, terminals=%d",
                 task_id, depth + 1, len(candidates),
                 candidates[0][2] if candidates else 0.0, len(terminal_pool))


    for beam_root, state in beams:
        terminal_pool.append((beam_root, state))


    best_traj: Optional[Trajectory] = None
    best_quality = -float("inf")

    for beam_root, leaf in terminal_pool:
        leaf = _compose_answer_if_needed(leaf, task, client, budget, metrics)
        traj = Trajectory.extract(beam_root, leaf)
        _before_judge = budget.spent_dollars
        quality = evaluate(leaf, task, client, budget=budget, metrics=metrics)
        metrics["judge_cost"] = metrics.get("judge_cost", 0.0) + max(
            0.0, budget.spent_dollars - _before_judge,
        )
        traj.quality = quality

        if _trajectory_rank_key(traj, quality) > _trajectory_rank_key(best_traj, best_quality):
            best_traj = traj
            best_quality = quality

    accepted = False
    reason = "no trajectory"
    if best_traj is not None:
        accepted, reason = _acceptance_check(best_traj, task, quality=best_quality)

    elapsed = _time.perf_counter() - t0
    metrics.update({
        "search_time_s": round(elapsed, 2),
        "model_counts": {_get_baseline_model(): metrics["total_llm_calls"]},
        "budget_spent": round(budget.spent_dollars, 4),
        "terminal_count": len(terminal_pool),
    })

    log.info("[%s] beam_search done: q=%.3f depth=%d terminals=%d accepted=%s $%.4f %.1fs",
             task_id, best_quality if best_quality > -float("inf") else 0.0,
             metrics["depth_reached"], len(terminal_pool),
             accepted, budget.spent_dollars, elapsed)

    return SearchOutcome(
        accepted_trajectory=best_traj if accepted else None,
        best_trajectory=best_traj,
        best_quality=max(best_quality, 0.0),
        reject_reason="" if accepted else reason,
        metadata=metrics,
    )


def rejection_sampling_search(
    task: dict,
    client,
    budget: BudgetTracker,
    max_steps: int = MAX_STEP_COUNT,
    max_draws: int = 8,
    prefix_cache=None,
    lcm=None,
) -> SearchOutcome:
    """Rejection Sampling: draw independent passes and redraw until the
    verifier accepts or the per-task cap is spent."""
    task_id = task.get("task_id", "?")
    t0 = _time.perf_counter()

    draws: List[Tuple[Trajectory, float, Dict]] = []
    accepted_traj: Optional[Trajectory] = None
    for i in range(max_draws):
        if budget.exhausted():
            log.info("[%s] rejection_sampling: cap spent after %d draws", task_id, i)
            break
        root, leaf, run_metrics = _forward_pass(
            task, client, budget, max_steps,
            temperature=DIVERSITY_TEMP, allow_repair=True,
            label=f" draw{i+1}/{max_draws}", prefix_cache=prefix_cache, lcm=lcm,
        )
        leaf = _compose_answer_if_needed(leaf, task, client, budget, run_metrics)
        traj = Trajectory.extract(root, leaf)
        _before_judge = budget.spent_dollars
        quality = evaluate(leaf, task, client, budget=budget, metrics=run_metrics)
        run_metrics["judge_cost"] = run_metrics.get("judge_cost", 0.0) + max(
            0.0, budget.spent_dollars - _before_judge,
        )
        traj.quality = quality
        draws.append((traj, quality, run_metrics))
        ok, _reason = _acceptance_check(traj, task, quality=quality)
        log.info("[%s] rejection_sampling draw %d/%d: q=%.3f accepted=%s",
                 task_id, i + 1, max_draws, quality, ok)
        if ok:
            accepted_traj = traj
            break

    if not draws:
        return SearchOutcome(
            accepted_trajectory=None, best_trajectory=None,
            best_quality=0.0, reject_reason="no trajectories produced",
            metadata={"strategy": "rejection_sampling", "task_id": task_id,
                      "n_draws": 0},
        )

    best_traj, best_quality, _ = max(
        draws, key=lambda c: _trajectory_rank_key(c[0], c[1]),
    )
    if accepted_traj is not None:
        best_traj, best_quality = accepted_traj, accepted_traj.quality
        reason = ""
    else:
        _ok, reason = _acceptance_check(best_traj, task, quality=best_quality)

    total_llm = sum(m["total_llm_calls"] for _, _, m in draws)
    total_ops: Dict = {}
    for _, _, m in draws:
        for op, cnt in m.get("operator_counts", {}).items():
            total_ops[op] = total_ops.get(op, 0) + cnt
    metrics = {
        "strategy": "rejection_sampling",
        "task_id": task_id,
        "n_draws": len(draws),
        "per_draw_quality": [round(q, 4) for _, q, _ in draws],
        "total_llm_calls": total_llm,
        "total_tool_calls": total_llm,
        "total_expansions": total_llm,
        "operator_counts": total_ops,
        "model_counts": {_get_baseline_model(): total_llm},
        "search_time_s": round(_time.perf_counter() - t0, 2),
        "budget_spent": round(budget.spent_dollars, 4),
    }
    log.info("[%s] rejection_sampling done: q=%.3f draws=%d accepted=%s $%.4f",
             task_id, best_quality, len(draws), accepted_traj is not None,
             budget.spent_dollars)
    return SearchOutcome(
        accepted_trajectory=accepted_traj,
        best_trajectory=best_traj,
        best_quality=best_quality,
        reject_reason="" if accepted_traj is not None else reason,
        metadata=metrics,
    )


_LATS_VALUE_PROMPT = (
    "You are evaluating a partial data-analysis trajectory. Given the task "
    "and the executed steps with their observations, rate how promising this "
    "state is for reaching a correct final answer. Respond with JSON only: "
    '{"score": <float between 0 and 1>}'
)


def _lats_state_value(state: AnalysisState, task: dict, client, budget) -> float:
    """LATS scores each executed node with an LLM value model reading the
    executed observations."""
    from toffee.search.evaluator import _extract_json_object
    history = []
    for msg in state.messages[-12:]:
        role = msg.get("role", "")
        content = (msg.get("content") or "")[:800]
        history.append(f"[{role}] {content}")
    user = (f"Task: {task.get('question', '')}\n\nExecuted steps:\n"
            + "\n".join(history))
    try:
        content, usage = client.call(
            [{"role": "system", "content": _LATS_VALUE_PROMPT},
             {"role": "user", "content": user}],
            model=_get_baseline_model(), max_tokens=128, temperature=0.1,
        )
        if usage is not None:
            budget.record(usage.cost, usage.prompt_tokens + usage.completion_tokens)
        parsed = _extract_json_object(content)
        if parsed and isinstance(parsed.get("score"), (int, float)):
            return max(0.0, min(1.0, float(parsed["score"])))
    except Exception as exc:
        log.warning("LATS value model failed: %s", exc)
    return 0.5


def lats_search(
    task: dict,
    client,
    budget: BudgetTracker,
    n_expand: int = 2,
    max_iterations: int = 25,
    c_uct: float = 1.0,
    prefix_cache=None,
    lcm=None,
) -> SearchOutcome:
    """LATS, adapted to the shared tool interface and caps: a tree search over
    the same tool steps whose executed nodes are scored by an LLM value model.
    Selection follows UCT over those values; terminal answers pass through the
    same AnswerCompose step and acceptance check as every other method."""
    import math

    task_id = task.get("task_id", "?")
    t0 = _time.perf_counter()

    metrics: Dict = {
        "strategy": "lats",
        "task_id": task_id,
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "total_expansions": 0,
        "llm_time_s": 0.0,
        "operator_counts": {},
        "value_calls": 0,
        "iterations": 0,
    }
    root = _prefix_root(task, prefix_cache, metrics)

    visits: Dict[int, int] = {id(root): 0}
    values: Dict[int, float] = {id(root): 0.0}
    kids: Dict[int, List[AnalysisState]] = {id(root): []}
    parent: Dict[int, Optional[AnalysisState]] = {id(root): None}

    def _backprop(node: Optional[AnalysisState], value: float) -> None:
        while node is not None:
            visits[id(node)] = visits.get(id(node), 0) + 1
            values[id(node)] = values.get(id(node), 0.0) + value
            node = parent.get(id(node))

    def _select() -> AnalysisState:
        node = root
        while kids.get(id(node)):
            n_parent = max(visits.get(id(node), 1), 1)
            best, best_score = None, -float("inf")
            for ch in kids[id(node)]:
                n_ch = visits.get(id(ch), 0)
                if n_ch == 0:
                    return ch
                mean = values.get(id(ch), 0.0) / n_ch
                score = mean + c_uct * math.sqrt(math.log(n_parent) / n_ch)
                if score > best_score:
                    best, best_score = ch, score
            node = best
        return node

    best_traj: Optional[Trajectory] = None
    best_quality = -float("inf")

    for it in range(max_iterations):
        if budget.exhausted():
            break
        leaf = _select()
        if leaf.is_terminal() or leaf.answer_drafted:
            _before_judge = budget.spent_dollars
            quality = evaluate(leaf, task, client, budget=budget, metrics=metrics)
            metrics["judge_cost"] = metrics.get("judge_cost", 0.0) + max(
                0.0, budget.spent_dollars - _before_judge,
            )
            traj = Trajectory.extract(root, leaf)
            traj.quality = quality
            if _trajectory_rank_key(traj, quality) > _trajectory_rank_key(best_traj, best_quality):
                best_traj, best_quality = traj, quality
            _backprop(leaf, quality)
            metrics["iterations"] = it + 1
            continue

        op_name = _select_next_operator(leaf, allow_repair=True)
        if op_name is None:
            _backprop(leaf, 0.0)
            continue
        expanded = 0
        for _ in range(n_expand):
            if budget.exhausted():
                break
            child = _execute_step(
                leaf, op_name, task, client, budget,
                temperature=DIVERSITY_TEMP, prefix_cache=prefix_cache, lcm=lcm,
            )
            if child is None:
                continue
            expanded += 1
            metrics["total_llm_calls"] += 1
            metrics["total_tool_calls"] += 1
            metrics["total_expansions"] += 1
            metrics["operator_counts"][op_name] = (
                metrics["operator_counts"].get(op_name, 0) + 1
            )
            parent[id(child)] = leaf
            kids.setdefault(id(leaf), []).append(child)
            kids.setdefault(id(child), [])
            value = _lats_state_value(child, task, client, budget)
            metrics["value_calls"] += 1
            metrics["total_llm_calls"] += 1
            _backprop(child, value)
        if expanded == 0:
            _backprop(leaf, 0.0)
        metrics["iterations"] = it + 1

    leaf_for_answer = None
    if best_traj is None or not best_traj.ended_with_answer:
        node_best, node_val = None, -float("inf")
        for node_list in kids.values():
            for ch in node_list:
                n_ch = visits.get(id(ch), 0)
                if n_ch <= 0 or not ch.result_exists:
                    continue
                mean = values.get(id(ch), 0.0) / n_ch
                if mean > node_val:
                    node_best, node_val = ch, mean
        leaf_for_answer = node_best

    if leaf_for_answer is not None and not budget.exhausted():
        leaf_for_answer = _compose_answer_if_needed(
            leaf_for_answer, task, client, budget, metrics)
        _before_judge = budget.spent_dollars
        quality = evaluate(leaf_for_answer, task, client, budget=budget, metrics=metrics)
        metrics["judge_cost"] = metrics.get("judge_cost", 0.0) + max(
            0.0, budget.spent_dollars - _before_judge,
        )
        traj = Trajectory.extract(root, leaf_for_answer)
        traj.quality = quality
        if _trajectory_rank_key(traj, quality) > _trajectory_rank_key(best_traj, best_quality):
            best_traj, best_quality = traj, quality

    accepted = False
    reason = "no trajectory"
    if best_traj is not None:
        accepted, reason = _acceptance_check(best_traj, task, quality=best_quality)

    metrics.update({
        "search_time_s": round(_time.perf_counter() - t0, 2),
        "model_counts": {_get_baseline_model(): metrics["total_llm_calls"]},
        "budget_spent": round(budget.spent_dollars, 4),
    })
    log.info("[%s] lats done: q=%.3f iters=%d value_calls=%d accepted=%s $%.4f",
             task_id, best_quality if best_quality > -float("inf") else 0.0,
             metrics["iterations"], metrics["value_calls"], accepted,
             budget.spent_dollars)
    return SearchOutcome(
        accepted_trajectory=best_traj if accepted else None,
        best_trajectory=best_traj,
        best_quality=max(best_quality, 0.0),
        reject_reason="" if accepted else reason,
        metadata=metrics,
    )
