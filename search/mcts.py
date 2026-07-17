
from __future__ import annotations

import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from toffee.config import (
    ALPHA_B,
    EFFORT_LEVELS,
    K_MAX,
    LCM_ENABLED,
    C_PUCT,
    MCTS_MAX_ITERATIONS,
    MIN_ACCEPT_QUALITY,
    MODEL_LIST,
    MODELS,
    ROUTER_MODE,
)
from toffee.core.executor import execute_tool
from toffee.core.operators import OPERATORS, ActionConfig, enumerate_feasible_actions
from toffee.core.state import AnalysisState
from toffee.client.openrouter import OpenRouterClient
from toffee.search.lcm import FactoredLinUCB
from toffee.search.evaluator import (
    _tool_outputs,
    batch_final_judge,
    compute_reward,
    evaluate,
)
from toffee.search.prefix_cache import PrefixCache
from toffee.utils import (
    BudgetTracker,
    extract_answer_text,
    extract_json_string_field,
    is_metadata_sql_query,
)

log = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    operator_name: str
    tool_name: str
    arguments: dict
    real_output: str
    reasoning: str
    visible_text: str
    model: str
    cost: float


@dataclass
class Trajectory:
    steps: List[TrajectoryStep]
    quality: float
    total_cost: float
    final_reasoning: str = ""
    final_answer: str = ""
    ended_with_answer: bool = False
    successful_data_steps: int = 0
    error_steps: int = 0


    tool_outputs: List[str] = field(default_factory=list)


    certificate: Optional[dict] = None

    @classmethod
    def extract(cls, root: AnalysisState, leaf: AnalysisState) -> "Trajectory":
        path_states = _collect_path_to_root(root, leaf)
        steps: List[TrajectoryStep] = []
        successful_data_steps = 0
        error_steps = 0
        for s in path_states:
            if s.parent_id is None:
                continue
            if s.operator_name == "AnswerCompose":
                continue
            fallback_tool = s.step_events[-1].tool_name if s.step_events else ""
            tool_name, arguments = _parse_tool_call(
                s.llm_content, s.operator_name, fallback_tool=fallback_tool
            )
            real_output = s.tool_output
            step_event = s.step_events[-1] if s.step_events else None
            if step_event is not None and not step_event.success:
                error_steps += 1
            if step_event is not None and step_event.substantive:
                successful_data_steps += 1
            steps.append(TrajectoryStep(
                operator_name=s.operator_name,
                tool_name=tool_name,
                arguments=arguments,
                real_output=real_output,
                reasoning=s.reasoning or _extract_reasoning(s.llm_content),
                visible_text=_extract_visible(s.llm_content),
                model=s.config_model,
                cost=0.0,
            ))

        final_answer = final_reasoning = ""
        if steps:
            final_reasoning = _extract_think_block(leaf.llm_content)


            final_answer = extract_answer_text(leaf.llm_content)

            if not final_reasoning:
                try:
                    parsed = json.loads(_strip_code_fences(leaf.llm_content))
                    if isinstance(parsed, dict):
                        final_reasoning = parsed.get("reasoning", "")
                except (json.JSONDecodeError, TypeError):
                    pass

            if not final_answer and steps[-1].visible_text:
                final_answer = steps[-1].visible_text
            if not final_reasoning and steps[-1].reasoning:
                final_reasoning = steps[-1].reasoning
            if not final_answer:
                final_answer = f"Analysis completed with {len(steps)} steps."
            if not final_reasoning:
                final_reasoning = "Based on the analysis steps above."

        return cls(
            steps=steps, quality=0.0, total_cost=leaf.total_cost,
            final_reasoning=final_reasoning, final_answer=final_answer,
            ended_with_answer=leaf.answer_drafted,
            successful_data_steps=successful_data_steps,
            error_steps=error_steps,
            tool_outputs=_tool_outputs(leaf),
        )


@dataclass
class SearchOutcome:
    accepted_trajectory: Optional[Trajectory]
    best_trajectory: Optional[Trajectory]
    best_quality: float
    reject_reason: str = ""
    metadata: Dict = field(default_factory=dict)


def _collect_path_to_root(root: AnalysisState, leaf: AnalysisState) -> List[AnalysisState]:
    all_states: Dict[str, AnalysisState] = {}
    _index_tree(root, all_states)
    path: List[AnalysisState] = []
    current: Optional[AnalysisState] = leaf
    while current is not None:
        path.append(current)
        if current.state_id == root.state_id:
            break
        current = all_states.get(current.parent_id)
    path.reverse()
    return path


def _index_tree(state: AnalysisState, index: Dict[str, AnalysisState], _depth: int = 0) -> None:
    if _depth > 50 or state.state_id in index:
        return
    index[state.state_id] = state
    for child in state.children.values():
        _index_tree(child, index, _depth + 1)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


_OPERATOR_DEFAULT_TOOL = {
    "SchemaScout": "list_tables",
    "DataInspect": "list_directory",
    "SQLDraft": "execute_sql",
    "SQLRepair": "execute_sql",
    "PythonCompute": "execute_python",
    "SanityCheck": "execute_sql",
    "AnswerCompose": None,
}

_VALID_TOOLS = {"list_tables", "get_table_schema", "execute_sql", "execute_python",
                "read_file", "list_directory", "write_file", "run_bash"}

_TOOL_ACCEPTED_ARGS: Dict[str, set] = {
    "list_tables": set(),
    "get_table_schema": {"table_name", "table"},
    "execute_sql": {"query", "sql"},
    "execute_python": {"code"},
    "read_file": {"file_path", "path", "encoding", "max_lines"},
    "list_directory": {"path", "pattern", "recursive"},
    "write_file": {"file_path", "path", "content", "mode"},
    "run_bash": {"command", "timeout", "cwd"},
}

_SCHEMA_FORBIDDEN_TABLES = {"sqlite_master", "sqlite_temp_master", "sqlite_sequence"}


def _extract_think_block(content: str) -> str:
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_json_block(content: str) -> Optional[dict]:
    marker = content.rfind('"tool_name"')
    if marker >= 0:
        start = content.rfind("{", 0, marker)
        if start >= 0:
            depth = 0
            for i in range(start, len(content)):
                ch = content[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = content[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict) and "tool_name" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            break
    try:
        parsed = json.loads(_strip_code_fences(content))
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_tool_call(
    llm_content: str,
    operator_name: str,
    state: Optional[AnalysisState] = None,
    fallback_tool: str = "",
) -> Tuple[str, dict]:
    allowed_tools = set()
    if operator_name in OPERATORS:
        if state is not None:
            allowed_tools = set(OPERATORS[operator_name].active_tools(state))
        else:
            allowed_tools = set(OPERATORS[operator_name].tools)
    parsed = _extract_json_block(llm_content)
    if parsed:
        tool_name = parsed.get("tool_name", "")
        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        if operator_name == "AnswerCompose":
            return "none", {}
        if (
            not tool_name
            or tool_name not in _VALID_TOOLS
            or (allowed_tools and tool_name not in allowed_tools)
        ):
            tool_name = fallback_tool or _OPERATOR_DEFAULT_TOOL.get(operator_name, operator_name)
            arguments = {}
        accepted = _TOOL_ACCEPTED_ARGS.get(tool_name)
        if accepted is not None:
            arguments = {k: v for k, v in arguments.items() if k in accepted}
        if tool_name == "get_table_schema":
            table_name = str(arguments.get("table_name", arguments.get("table", ""))).strip().lower()
            if table_name in _SCHEMA_FORBIDDEN_TABLES:
                return "list_tables", {}
        return tool_name or operator_name, arguments
    if operator_name == "AnswerCompose":
        return "none", {}
    tool_name = fallback_tool or _OPERATOR_DEFAULT_TOOL.get(operator_name, operator_name)
    return tool_name or operator_name, {}


def _extract_reasoning(content: str) -> str:
    think = _extract_think_block(content)
    if think:
        return think
    try:
        return json.loads(_strip_code_fences(content)).get("reasoning", "")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def _extract_visible(content: str) -> str:
    think_end = content.find("</think>")
    if think_end >= 0:
        after_think = content[think_end + len("</think>"):].strip()
        json_marker = after_think.find('{"tool_name"')
        if json_marker < 0:
            json_marker = after_think.find('```')
        if json_marker > 0:
            visible = after_think[:json_marker].strip()
        else:
            visible = after_think.strip()
        if visible:
            return visible
    try:
        return json.loads(_strip_code_fences(content)).get("visible_text", "")
    except (json.JSONDecodeError, TypeError, AttributeError):

        text = content.strip()
        json_marker = text.find('{"tool_name"')
        if json_marker > 0:
            text = text[:json_marker].strip()
        return text


def _per_tool_best(
    ranked: List[Tuple[float, float, ActionConfig]],
) -> List[Tuple[float, float, ActionConfig]]:
    best: Dict[str, Tuple[float, float, ActionConfig]] = {}
    for ucb, sigma, action in ranked:
        key = action.tool_name or "__compose__"
        if key not in best:
            best[key] = (ucb, sigma, action)
    return sorted(best.values(), key=lambda x: -x[0])


def _uniform_fixed_arms(
    feasible_actions: List[ActionConfig],
) -> List[Tuple[float, float, ActionConfig]]:
    tier = os.environ.get("TOFFEE_FIXED_TIER", "")
    mid_model = MODELS[tier] if tier in MODELS else MODEL_LIST[1]
    return [
        (0.0, 0.0, a) for a in feasible_actions
        if a.model == mid_model and a.history == "mid" and a.effort == "moderate"
    ]


_RULE_ROUTER_TIER = {


    "SchemaScout": "cost_effective",
    "DataInspect": "cost_effective",
    "SanityCheck": "cost_effective",
    "SQLDraft": "capable",
    "PythonCompute": "capable",
    "AnswerCompose": "capable",
    "SQLRepair": "premium",
}


def _rule_router_arms(
    feasible_actions: List[ActionConfig],
) -> List[Tuple[float, float, ActionConfig]]:
    chosen = {}
    for a in feasible_actions:
        tier = _RULE_ROUTER_TIER.get(a.operator_name, "capable")
        if a.model != MODELS[tier] or a.history != "mid" or a.effort != "moderate":
            continue
        key = (a.operator_name, a.tool_name)
        if key not in chosen:
            chosen[key] = (0.0, 0.0, a)
    return list(chosen.values())


def _softmax_probs(values: List[float], temperature: float = 1.0) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float64) / max(temperature, 1e-6)
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    denom = np.sum(probs)
    if denom <= 0:
        return [1.0 / len(values)] * len(values)
    return (probs / denom).tolist()


def _plausible_winner_count(
    candidates: List[Tuple[float, float, ActionConfig]], alpha: float,
) -> int:
    if len(candidates) <= 1:
        return len(candidates)
    lowers = [ucb - 2.0 * alpha * sigma for ucb, sigma, _a in candidates]
    thresh = max(lowers)
    return sum(1 for ucb, _sigma, _a in candidates if ucb >= thresh - 1e-12)


def _acceptance_check(
    trajectory: Optional[Trajectory],
    task: Optional[dict] = None,
    quality: float = 0.0,
) -> tuple[bool, str]:
    if trajectory is None or not trajectory.steps:
        return False, "no trajectory"
    from toffee.config import ACCEPT_RULE, ACCEPT_CUTS
    cut = ACCEPT_CUTS[ACCEPT_RULE]
    answer_key = str((task or {}).get("answer_key", "") or "")
    key_rows = ((task or {}).get("answer_key_rows") or (task or {}).get("provenance")
                or (task or {}).get("answer_contract") or (task or {}).get("facts") or [])
    if answer_key or key_rows:
        from toffee.search.evaluator import (
            certify_answer, _db_path_of_task, _required_sources_of_task,
            _source_names_of_task)


        data_steps: List[dict] = []
        step_outputs: set = set()
        for s in trajectory.steps:
            if s.tool_name in ("execute_sql", "execute_python"):
                args = s.arguments or {}
                text = args.get("query") or args.get("sql") or args.get("code") or ""
                data_steps.append({"tool": s.tool_name, "text": text,
                                   "output": s.real_output})
            if s.real_output:
                step_outputs.add(s.real_output)
        for o in (trajectory.tool_outputs or []):
            if o and o not in step_outputs:
                data_steps.append({"tool": "prefix", "text": None, "output": o})
        v_q, cert_detail = certify_answer(
            trajectory.final_answer, answer_key, key_rows,
            data_steps=data_steps,
            source_names=_source_names_of_task(task or {}),
            required_sources=_required_sources_of_task(task or {}),
            db_path=_db_path_of_task(task or {}),
        )
        cert_detail["v_q"] = round(v_q, 4)
        trajectory.certificate = cert_detail
        if v_q + 1e-9 < cut:
            return False, f"certificate V_q {v_q:.3f} below {ACCEPT_RULE} cut {cut}"
        return True, "accepted"
    if quality + 1e-9 < cut:
        return False, f"quality {quality:.3f} below {ACCEPT_RULE} cut {cut}"
    elif quality < MIN_ACCEPT_QUALITY:
        return False, f"quality {quality:.3f} < {MIN_ACCEPT_QUALITY}"
    return True, "accepted"


def _trajectory_rank_key(
    trajectory: Optional[Trajectory],
    quality: float,
) -> tuple[int, float, int, int]:
    if trajectory is None:
        return (-1, -1.0, 0, 0)
    did_data = 1 if trajectory.successful_data_steps >= 1 else 0
    return (
        did_data,
        float(quality),
        int(trajectory.ended_with_answer),
        len(trajectory.steps),
    )


def _has_untried_operators(node: AnalysisState) -> bool:
    tried_ops = {k.split("|")[0] for k in node.children}
    for op_name, op in OPERATORS.items():
        if op_name not in tried_ops and op.is_feasible(node):
            return True
    return False


def select_path(
    root: AnalysisState, lcm: FactoredLinUCB, budget: BudgetTracker,
) -> Tuple[List[AnalysisState], AnalysisState]:
    path = [root]
    node = root

    while node.children and not node.is_terminal():
        n_children = len(node.children)


        if (node.visit_count >= 3 * n_children + 2
                and n_children < 4
                and _has_untried_operators(node)):
            break

        best_key: Optional[str] = None
        best_score = -float("inf")
        n_children = max(len(node.children), 1)
        for action_key, child in node.children.items():
            q = node.action_values.get(action_key, 0.0)
            n_parent = max(node.visit_count, 1)
            n_child = node.action_visits.get(action_key, 0)


            prior = node.priors.get(action_key, 1.0 / n_children)
            exploration = C_PUCT * prior * math.sqrt(n_parent) / (1 + n_child)
            score = q + exploration
            if score > best_score:
                best_score = score
                best_key = action_key
        if best_key is None:
            break
        node = node.children[best_key]
        path.append(node)

    return path, node


def backpropagate(path: List[AnalysisState], child: AnalysisState, reward: float) -> None:
    child.visit_count += 1
    full_path = path + [child]
    for idx in range(len(full_path) - 2, -1, -1):
        node = full_path[idx]
        next_node = full_path[idx + 1]
        action_key = next_node.action_key
        node.visit_count += 1
        node.action_visits[action_key] = node.action_visits.get(action_key, 0) + 1
        node.action_cumulative[action_key] = node.action_cumulative.get(action_key, 0.0) + reward
        node.action_sq_cumulative[action_key] = node.action_sq_cumulative.get(action_key, 0.0) + reward ** 2
        n = node.action_visits[action_key]
        node.action_values[action_key] = node.action_cumulative[action_key] / n


def backpropagate_terminal(path: List[AnalysisState], reward: float) -> None:
    if not path:
        return
    path[-1].visit_count += 1
    for idx in range(len(path) - 2, -1, -1):
        node = path[idx]
        child = path[idx + 1]
        action_key = child.action_key
        node.visit_count += 1
        node.action_visits[action_key] = node.action_visits.get(action_key, 0) + 1
        node.action_cumulative[action_key] = node.action_cumulative.get(action_key, 0.0) + reward
        node.action_sq_cumulative[action_key] = node.action_sq_cumulative.get(action_key, 0.0) + reward ** 2
        n = node.action_visits[action_key]
        node.action_values[action_key] = node.action_cumulative[action_key] / n


def search_task(
    task: dict,
    client: OpenRouterClient,
    prefix_cache: PrefixCache,
    lcm: FactoredLinUCB,
    budget: BudgetTracker,
    max_iterations: int = MCTS_MAX_ITERATIONS,
    use_prefix_cache: bool = True,
) -> SearchOutcome:
    import time as _time
    search_t0 = _time.perf_counter()

    env_context = task.get("env", {})
    task_id = task.get("task_id", "?")
    root = AnalysisState.from_task(task)


    c_tilde = budget.max_dollars / max(1, max_iterations)


    metrics: Dict = {
        "task_id": task_id,
        "total_iterations": 0,
        "total_expansions": 0,
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "llm_time_s": 0.0,
        "tool_time_s": 0.0,
        "judge_calls": 0,
        "judge_cost": 0.0,
        "expansion_widths": [],
        "quality_curve": [],
        "operator_counts": {},
        "model_counts": {},
        "prefix_reused": False,
        "prefix_depth": 0,
        "use_prefix_cache": use_prefix_cache,
        "first_answer_iter": None,
        "convergence_iter": None,
    }

    prefix_state = None
    if use_prefix_cache:
        prefix_state = prefix_cache.find_reusable_prefix(root)
        if prefix_state and not prefix_state.is_terminal():
            root = prefix_state.clone_with_new_provenance(task_id)
            metrics["prefix_reused"] = True
            metrics["prefix_depth"] = root.step_count
            log.info("Reusing cached prefix (depth=%d) for task %s", root.step_count, task_id)
        elif prefix_state:
            log.warning("Discarding terminal memo prefix for task %s (answer=%s, steps=%d)",
                         task_id, prefix_state.answer_drafted, prefix_state.step_count)
    else:
        log.info("[%s] prefix cache disabled — running independent MCTS from scratch", task_id)

    best_trajectory: Optional[Trajectory] = None
    best_leaf: Optional[AnalysisState] = None
    best_quality = -float("inf")
    reject_reason = "no trajectory"
    last_improvement_iter = 0

    effective_iterations = 0
    for iteration in range(max_iterations * 3):
        if effective_iterations >= max_iterations:
            break
        if budget.exhausted():
            reject_reason = (
                f"budget exhausted at iteration {effective_iterations + 1} "
                f"(spent ${budget.spent_dollars:.4f}, tokens={budget.spent_tokens})"
            )
            log.info("[%s] %s", task_id, reject_reason)
            break

        path, leaf = select_path(root, lcm, budget)
        log.info(
            "[%s] iter %d/%d leaf(step=%d, goal=%s, schema=%s, result=%s, answer=%s) "
            "best_q=%.3f spent=$%.4f",
            task_id, effective_iterations + 1, max_iterations, leaf.step_count, leaf.pending_goal,
            leaf.schema_discovered, leaf.result_exists, leaf.answer_drafted,
            best_quality if best_quality > -float("inf") else 0.0, budget.spent_dollars,
        )

        if leaf.is_terminal():
            _before_judge = budget.spent_dollars
            value = evaluate(leaf, task, client, budget=budget, metrics=metrics)
            metrics["judge_cost"] += max(0.0, budget.spent_dollars - _before_judge)
            traj = Trajectory.extract(root, leaf)
            traj.quality = value
            reward = compute_reward(
                value=value, child_state=leaf, cost=0.0, cost_estimate=c_tilde,
                was_verified=leaf.verification_passed, was_reused=prefix_state is not None,
            )
            backpropagate_terminal(path, reward)
            if _trajectory_rank_key(traj, value) > _trajectory_rank_key(best_trajectory, best_quality):
                best_quality = value
                best_trajectory = traj
                best_leaf = leaf
                _accepted, reject_reason = _acceptance_check(best_trajectory, task, quality=best_quality)
            log.info("[%s] terminal leaf value=%.3f", task_id, value)
            continue

        feasible_actions = enumerate_feasible_actions(leaf)
        if not feasible_actions:
            reject_reason = f"no feasible actions at step {leaf.step_count}"
            continue

        x_state = leaf.feature_vector()
        if ROUTER_MODE == "rule":
            per_tool = _rule_router_arms(feasible_actions)
        elif LCM_ENABLED:
            ranked_all = lcm.rank_actions(x_state, feasible_actions)

            per_tool = _per_tool_best(ranked_all)
        else:
            per_tool = _uniform_fixed_arms(feasible_actions)
        budget_ratio = budget.remaining_dollar_ratio()
        k_ci = _plausible_winner_count(per_tool, ALPHA_B)
        k_budget = max(1, math.ceil(K_MAX * max(0.0, min(1.0, budget_ratio))))
        K = max(1, min(k_ci, k_budget))
        ranked = per_tool[:K]


        log.info(
            "[%s] iter %d/%d K=%d (budget=%.0f%%): %s",
            task_id, effective_iterations + 1, max_iterations, len(ranked),
            budget_ratio * 100,
            ", ".join(f"{a.operator_name}/{a.tool_name or 'compose'}"
                      for _, _, a in ranked),
        )


        expand_jobs: List[Tuple[ActionConfig, list, int]] = []

        ucb_scores = [s for s, _, _ in ranked]
        prior_probs = _softmax_probs(ucb_scores, temperature=1.0)
        for (_ucb_score, sigma, action_config), prior_p in zip(ranked, prior_probs):
            leaf.priors[action_config.key] = max(0.05, prior_p)
            leaf.sigmas[action_config.key] = sigma
            operator = OPERATORS[action_config.operator_name]
            messages = operator.build_prompt(
                leaf,
                action_config.history,
                task_context=task,
                active_tool=action_config.tool_name,
            )


            effort_tokens = EFFORT_LEVELS.get(action_config.effort, 2048)
            expand_jobs.append((action_config, messages, effort_tokens))

        def _expand_one(job):
            action_config, messages, effort_tokens = job
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            log.info("expand_one START: %s model=%s effort=%d prompt=%dch",
                     action_config.operator_name, action_config.model, effort_tokens, prompt_chars)
            try:
                t_llm = _time.perf_counter()
                content, usage = client.call(
                    messages, model=action_config.model,
                    max_tokens=effort_tokens, temperature=0.4,
                )
                llm_elapsed = _time.perf_counter() - t_llm
                log.info("expand_one LLM done: %s %.1fs tokens=%d+%d",
                         action_config.operator_name, llm_elapsed,
                         usage.prompt_tokens, usage.completion_tokens)
            except Exception as exc:
                log.warning("LLM call failed for %s after %.1fs: %s",
                            action_config.key, _time.perf_counter() - t_llm, exc)
                return None
            budget.record(usage.cost, usage.prompt_tokens + usage.completion_tokens)
            tool_name, arguments = _parse_tool_call(
                content,
                action_config.operator_name,
                leaf,
                fallback_tool=action_config.tool_name,
            )
            t_tool = _time.perf_counter()
            if action_config.operator_name == "AnswerCompose":
                from toffee.core.executor import ToolResult
                tool_result = ToolResult(tool_name="none", success=True, stdout=content)
                exec_cost = 0.0
            else:
                tool_result = execute_tool(tool_name, arguments, env_context)
                exec_cost = budget.record_execution(tool_result.elapsed_s)
            tool_elapsed = _time.perf_counter() - t_tool
            return (action_config, content, tool_result, usage, llm_elapsed, tool_elapsed, exec_cost)

        expansion_results = []
        if len(expand_jobs) == 1:
            r = _expand_one(expand_jobs[0])
            if r is not None:
                expansion_results.append(r)
        else:
            with ThreadPoolExecutor(max_workers=len(expand_jobs)) as pool:
                futures = {pool.submit(_expand_one, job): job for job in expand_jobs}
                try:
                    for fut in as_completed(futures, timeout=180):
                        try:
                            r = fut.result(timeout=10)
                        except Exception:
                            r = None
                        if r is not None:
                            expansion_results.append(r)
                except TimeoutError:
                    log.warning("[%s] expansion timeout — using %d/%d results",
                                task_id, len(expansion_results), len(expand_jobs))

        children: List[Tuple[AnalysisState, ActionConfig, object]] = []
        if not expansion_results:

            continue
        effective_iterations += 1
        for result_tuple in expansion_results:
            action_config, content, tool_result, usage, llm_elapsed, tool_elapsed, exec_cost = result_tuple
            child = leaf.expand(
                operator_name=action_config.operator_name,
                config=action_config.to_dict(),
                llm_content=content,
                tool_result=tool_result,
                usage=usage,
            )
            child.action_key = action_config.key
            leaf.children[action_config.key] = child
            children.append((child, action_config, usage, exec_cost))

            metrics["total_expansions"] += 1
            metrics["total_llm_calls"] += 1
            metrics["total_tool_calls"] += 1
            metrics["llm_time_s"] += llm_elapsed
            metrics["tool_time_s"] += tool_elapsed
            op = action_config.operator_name
            metrics["operator_counts"][op] = metrics["operator_counts"].get(op, 0) + 1
            tier = action_config.model.split("/")[-1] if "/" in action_config.model else action_config.model
            metrics["model_counts"][tier] = metrics["model_counts"].get(tier, 0) + 1


        child_states = [child for child, _, _, _ in children]
        for child, action_config, _usage, _exec_cost in children:
            if action_config.operator_name != "AnswerCompose":
                child.apply_step_judge(None)


        terminal_idxs = [i for i, ch in enumerate(child_states) if ch.answer_drafted]
        fj_scores: List[Optional[float]] = [None] * len(children)
        _judge_is_fallback = bool(
            task.get("answer_key") or task.get("answer_key_rows") or task.get("provenance")
            or task.get("answer_contract") or task.get("facts"))
        if terminal_idxs and client is not None and not _judge_is_fallback:
            _before_judge = budget.spent_dollars
            fj_list = batch_final_judge(
                [child_states[i] for i in terminal_idxs],
                task, client, budget=budget, metrics=metrics,
            )
            metrics["judge_cost"] += max(0.0, budget.spent_dollars - _before_judge)
            for pos, i in enumerate(terminal_idxs):
                fj_scores[i] = fj_list[pos] if pos < len(fj_list) else 0.0

        eval_results = []
        for idx, (child, action_config, usage, exec_cost) in enumerate(children):
            log.info("[%s] evaluating child %d/%d op=%s terminal=%s",
                     task_id, idx + 1, len(children), action_config.operator_name, child.answer_drafted)
            _before_judge = budget.spent_dollars
            value = evaluate(
                child, task, client,
                final_judge_val=fj_scores[idx],
                budget=budget, metrics=metrics,
            )
            metrics["judge_cost"] += max(0.0, budget.spent_dollars - _before_judge)
            log.info("[%s] child %d/%d value=%.3f", task_id, idx + 1, len(children), value)
            eval_results.append((child, action_config, usage, exec_cost, value))

        for child, action_config, usage, exec_cost, value in eval_results:
            reward = compute_reward(
                value=value, child_state=child, cost=usage.cost + exec_cost, cost_estimate=c_tilde,
                was_verified=child.verification_passed, was_reused=prefix_state is not None,
            )
            backpropagate(path, child, reward)
            if LCM_ENABLED:
                z_action = action_config.feature_vector()
                lcm.update(x_state, z_action, reward, op_name=action_config.operator_name)

            traj = Trajectory.extract(root, child)
            traj.quality = value
            if _trajectory_rank_key(traj, value) > _trajectory_rank_key(best_trajectory, best_quality):
                best_quality = value
                best_trajectory = traj
                best_leaf = child
                _accepted, reject_reason = _acceptance_check(best_trajectory, task, quality=best_quality)


        already_answered = best_trajectory is not None and best_trajectory.ended_with_answer
        for child, action_config, usage, _exec_cost, value in eval_results:
            if already_answered and value <= best_quality:
                continue
            if (child.result_exists
                    and not child.answer_drafted
                    and not child.is_terminal()
                    and "AnswerCompose" not in {k.split("|")[0] for k in child.children}
                    and not budget.exhausted()):
                ac_feasible = OPERATORS["AnswerCompose"].is_feasible(child)
                if not ac_feasible:
                    continue
                from toffee.core.operators import ActionConfig as _AC
                ac = _AC(operator_name="AnswerCompose",
                         model=action_config.model, history="long", effort="extended")
                op = OPERATORS["AnswerCompose"]
                msgs = op.build_prompt(child, "long", task_context=task)
                try:
                    t0 = _time.perf_counter()
                    content, ac_usage = client.call(msgs, model=ac.model, max_tokens=8192, temperature=0.4)
                    metrics["llm_time_s"] += _time.perf_counter() - t0
                    metrics["total_llm_calls"] += 1
                except Exception:
                    continue
                budget.record(ac_usage.cost, ac_usage.prompt_tokens + ac_usage.completion_tokens)
                from toffee.core.executor import ToolResult as _TR
                ac_result = _TR(tool_name="none", success=True, stdout=content)
                ac_child = child.expand(
                    operator_name="AnswerCompose", config=ac.to_dict(),
                    llm_content=content, tool_result=ac_result, usage=ac_usage,
                )
                ac_child.action_key = ac.key
                child.children[ac.key] = ac_child
                _before_judge = budget.spent_dollars
                ac_value = evaluate(ac_child, task, client, budget=budget, metrics=metrics)
                metrics["judge_cost"] += max(0.0, budget.spent_dollars - _before_judge)
                ac_reward = compute_reward(
                    value=ac_value, child_state=ac_child, cost=ac_usage.cost,
                    cost_estimate=c_tilde,
                    was_verified=False, was_reused=prefix_state is not None,
                )
                backpropagate(path + [child], ac_child, ac_reward)
                if LCM_ENABLED:
                    lcm.update(child.feature_vector(), ac.feature_vector(), ac_reward, op_name="AnswerCompose")
                metrics["total_expansions"] += 1
                metrics["operator_counts"]["AnswerCompose"] = metrics["operator_counts"].get("AnswerCompose", 0) + 1
                ac_traj = Trajectory.extract(root, ac_child)
                ac_traj.quality = ac_value
                if _trajectory_rank_key(ac_traj, ac_value) > _trajectory_rank_key(best_trajectory, best_quality):
                    best_quality = ac_value
                    best_trajectory = ac_traj
                    best_leaf = ac_child
                    _accepted, reject_reason = _acceptance_check(best_trajectory, task, quality=best_quality)
                log.info("[%s] greedy AnswerCompose: value=%.3f (best_q=%.3f)", task_id, ac_value, best_quality)
                break

        if use_prefix_cache:
            for child, _, _, _ in children:
                prefix_cache.register(child)

        metrics["total_iterations"] = effective_iterations
        metrics["total_attempts"] = iteration + 1
        metrics["expansion_widths"].append(len(ranked))
        metrics["quality_curve"].append(round(best_quality, 4) if best_quality > -float("inf") else 0.0)

        _accepted, reject_reason = _acceptance_check(best_trajectory, task, quality=best_quality)
        if best_trajectory is not None:
            prev_best = metrics["quality_curve"][-2] if len(metrics["quality_curve"]) > 1 else -1
            if best_quality > prev_best:
                last_improvement_iter = effective_iterations
            if best_trajectory.ended_with_answer and metrics["first_answer_iter"] is None:
                metrics["first_answer_iter"] = effective_iterations
            log.info(
                "[%s] iter %d best so far: q=%.3f steps=%d data=%d errors=%d reject=%s",
                task_id, effective_iterations, best_quality, len(best_trajectory.steps),
                best_trajectory.successful_data_steps, best_trajectory.error_steps, reject_reason,
            )


        _stop_accepted, _ = _acceptance_check(best_trajectory, task, quality=best_quality)
        if (best_trajectory is not None
                and best_trajectory.ended_with_answer
                and _stop_accepted):
            log.info(
                "[%s] early-stop: accepted q=%.3f at iter %d (steps=%d)",
                task_id, best_quality,
                effective_iterations, len(best_trajectory.steps),
            )
            metrics["early_stopped"] = True
            metrics["early_stop_iter"] = effective_iterations
            break


    metrics["search_time_s"] = round(_time.perf_counter() - search_t0, 2)
    metrics["convergence_iter"] = last_improvement_iter + 1
    metrics["budget_spent"] = round(budget.spent_dollars, 4)
    metrics["budget_remaining_pct"] = round(budget.remaining_dollar_ratio() * 100, 1)
    metrics["exec_cost"] = round(budget.spent_exec_dollars, 6)
    metrics["exec_count"] = budget.exec_count
    metrics["exec_seconds"] = round(budget.exec_seconds, 2)
    metrics["avg_expansion_width"] = round(sum(metrics["expansion_widths"]) / max(len(metrics["expansion_widths"]), 1), 2)
    log.info(
        "[%s] search complete: iters=%d expansions=%d llm_calls=%d "
        "llm_time=%.1fs tool_time=%.1fs judge_calls=%d converged_at=%d "
        "ops=%s models=%s",
        task_id, metrics["total_iterations"], metrics["total_expansions"],
        metrics["total_llm_calls"], metrics["llm_time_s"], metrics["tool_time_s"],
        metrics["judge_calls"], metrics["convergence_iter"],
        metrics["operator_counts"], metrics["model_counts"],
    )


    if use_prefix_cache and prefix_state is not None and best_trajectory is not None:
        prefix_cache.propagate_trajectory_reward(
            root.env_fingerprint, best_quality,
        )


    if (best_trajectory is not None
            and not best_trajectory.ended_with_answer
            and best_trajectory.successful_data_steps >= 1):
        if best_leaf is not None and not best_leaf.answer_drafted:
            log.info(
                "[%s] budget-exhausted without answer — running post-search "
                "AnswerCompose on best leaf (q=%.3f, steps=%d)",
                task_id, best_quality, len(best_trajectory.steps),
            )
            try:
                from toffee.core.operators import ActionConfig as _AC
                ac = _AC(operator_name="AnswerCompose", model=MODELS["premium"],
                         history="long", effort="extended")
                op = OPERATORS["AnswerCompose"]
                msgs = op.build_prompt(best_leaf, "long", task_context=task)
                content, ac_usage = client.call(
                    msgs, model=ac.model, max_tokens=8192, temperature=0.4,
                )
                budget.record(ac_usage.cost, ac_usage.prompt_tokens + ac_usage.completion_tokens)
                metrics["total_llm_calls"] += 1
                from toffee.core.executor import ToolResult as _TR
                ac_result = _TR(tool_name="none", success=True, stdout=content)
                ac_child = best_leaf.expand(
                    operator_name="AnswerCompose", config=ac.to_dict(),
                    llm_content=content, tool_result=ac_result, usage=ac_usage,
                )
                ac_child.action_key = ac.key
                best_leaf.children[ac.key] = ac_child
                ac_traj = Trajectory.extract(root, ac_child)
                if (ac_traj.ended_with_answer
                        and (ac_traj.final_answer or "").strip()):


                    ac_traj.quality = best_quality
                    best_trajectory = ac_traj
                    log.info("[%s] post-search AnswerCompose attached: steps=%d answer_chars=%d",
                             task_id, len(best_trajectory.steps),
                             len(best_trajectory.final_answer or ""))
                else:
                    log.info("[%s] post-search AnswerCompose produced empty answer — keeping raw trajectory",
                             task_id)
            except Exception as exc:
                log.warning("[%s] post-search AnswerCompose failed: %s", task_id, exc)

    accepted, final_reason = _acceptance_check(best_trajectory, task, quality=best_quality)
    if accepted:
        log.info("[%s] returning best trajectory q=%.3f steps=%d data=%d (%s)",
                 task_id, best_quality,
                 len(best_trajectory.steps) if best_trajectory else 0,
                 best_trajectory.successful_data_steps if best_trajectory else 0,
                 final_reason)
        return SearchOutcome(
            accepted_trajectory=best_trajectory, best_trajectory=best_trajectory,
            best_quality=best_quality, reject_reason="", metadata=metrics,
        )

    if best_trajectory:
        log.info("[%s] best trajectory rejected: %s (q=%.3f)", task_id, final_reason, best_quality)
    else:
        log.warning("No trajectory found for task %s", task_id)

    return SearchOutcome(
        accepted_trajectory=None, best_trajectory=best_trajectory,
        best_quality=max(best_quality, 0.0), reject_reason=final_reason,
        metadata=metrics,
    )
