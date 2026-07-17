#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from toffee import config as _cfg
from toffee.config import MCTS_MAX_ITERATIONS
from toffee.client.openrouter import OpenRouterClient
from toffee.generation.assembler import assemble_sft
from toffee.generation.bottomup import synthesize_tasks
from toffee.search.lcm import FactoredLinUCB
from toffee.search.mcts import search_task
from toffee.search.prefix_cache import PrefixCache
from toffee.utils import BudgetTracker, ensure_dir, load_catalog, setup_logging

log = logging.getLogger(__name__)

_SIBLING_STRUCT_EXTS = (
    ".md", ".markdown", ".csv", ".tsv", ".xlsx", ".xls", ".xlsm",
)


def _discover_siblings(data_file: str, explicit: List[str] | None = None) -> List[str]:
    result: List[str] = [p for p in (explicit or []) if p]
    if not data_file:
        return result
    try:
        parent = Path(data_file).parent
        if not parent.is_dir():
            return result
        for child in sorted(parent.iterdir()):
            if not child.is_file():
                continue
            if child.suffix.lower() not in _SIBLING_STRUCT_EXTS:
                continue
            p = str(child)
            if p != data_file and p not in result:
                result.append(p)
    except Exception:
        pass
    return result


def parse_mix(mix_str: str) -> Dict[str, float]:
    mix: Dict[str, float] = {}
    for part in mix_str.split(","):
        part = part.strip()
        if not part:
            continue
        key, val = part.split(":")
        mix[key.strip()] = float(val.strip())
    return mix


def synthesize_task_pool(
    catalog: List[Dict[str, Any]],
    mix: Dict[str, float],
    total: int,
    client: OpenRouterClient,
    tasks_per_env: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for src in catalog:
        group = src.get("source_group", src.get("benchmark", "other"))
        groups.setdefault(group, []).append(src)

    selected: List[Dict[str, Any]] = []
    for group, weight in mix.items():
        pool = list(groups.get(group, []))
        if not pool:
            log.warning("No sources found for group '%s'", group)
            continue
        rng.shuffle(pool)
        want = max(1, int(round(total * weight)))
        selected.extend(pool[:want])
    if not selected:
        selected = list(catalog)
        rng.shuffle(selected)

    per_env = max(1, min(tasks_per_env, -(-total // len(selected))))
    tasks: List[Dict[str, Any]] = []
    for src in selected:
        if len(tasks) >= total:
            break
        data_file = src.get("data_file", "")
        if not data_file:
            continue
        env = {
            "db_path": data_file,
            "data_file": data_file,
            "extra_sources": _discover_siblings(
                data_file, src.get("extra_sources", []) or [],
            ),
            "source_id": src.get("source_id", ""),
            "benchmark": src.get("benchmark", ""),
        }
        try:
            synthesized = synthesize_tasks(
                env=env, client=client, tasks_per_env=per_env, seed=seed,
            )
        except Exception as exc:
            log.warning("Task synthesis failed for source %s: %s",
                        env["source_id"], exc)
            continue
        for st in synthesized:
            tasks.append({
                "task_id": st.task_id,
                "source_id": st.env.get("source_id", ""),
                "question": st.question,
                "context": st.context,
                "deliverable": st.deliverable,
                "env": st.env,
                "ground_truth": st.ground_truth,
                "round_idx": st.round_idx,
                "level": st.level,
                "path": st.path,
                "hint": st.hint,
                "source_span": st.source_span,
                "format_span": st.format_span,
                "question_type": st.question_type,
                "answer_key": st.answer_key,
                "answer_key_rows": st.answer_key_rows,
                "provenance": st.provenance,
            })
        log.info("Synthesized %d admitted tasks from source %s",
                 len(synthesized), env["source_id"])
    rng.shuffle(tasks)
    return tasks[:total]


def _task_level(task: Dict[str, Any]) -> int:
    try:
        level = int(str(task.get("level", task.get("motif") or 0)).lstrip("Mm") or 0)
    except ValueError:
        level = 0
    return level if level in (1, 2, 3) else 1


def _level_budget_multiplier(task: Dict[str, Any]) -> int:
    if not _cfg.LEVEL_BUDGET_SCALING:
        return 1
    return _task_level(task)


def run_one_task(
    task: Dict[str, Any],
    client: OpenRouterClient,
    lcm: FactoredLinUCB,
    prefix_cache: PrefixCache,
    output_dir: Path,
    budget_per_task: float,
    max_iterations: int,
    use_prefix_cache: bool = True,
    strategy: str = "mcts",
) -> Dict[str, Any]:
    task_id = task["task_id"]


    existing_path = output_dir / "openai" / f"{task_id}.json"
    if existing_path.is_file():
        return {"task_id": task_id, "status": "resumed",
                "quality": 0.0, "cost": 0.0, "elapsed_s": 0.0,
                "output_path": str(existing_path)}


    level_mult = _level_budget_multiplier(task)
    eff_budget_per_task = budget_per_task * level_mult
    eff_max_iterations = max_iterations * level_mult
    if _cfg.LEVEL_BUDGET_SCALING:
        log.info("Submit %s: level x%d -> budget $%.4f, iters %d",
                 task_id, level_mult,
                 eff_budget_per_task, eff_max_iterations)

    budget = BudgetTracker(max_dollars=eff_budget_per_task)
    t0 = time.perf_counter()


    baseline_memo = prefix_cache if use_prefix_cache else None
    baseline_bandit = lcm if os.environ.get("TOFFEE_BASELINE_LCM", "0") == "1" else None
    try:
        try:
            if strategy == "mcts":
                search_outcome = search_task(
                    task=task, client=client, prefix_cache=prefix_cache,
                    lcm=lcm, budget=budget, max_iterations=eff_max_iterations,
                    use_prefix_cache=use_prefix_cache,
                )
            elif strategy == "single_pass":
                from toffee.search.baselines import single_pass_search
                search_outcome = single_pass_search(
                    task=task, client=client, budget=budget,
                    prefix_cache=baseline_memo, lcm=baseline_bandit)
            elif strategy == "react":
                from toffee.search.baselines import react_search
                search_outcome = react_search(
                    task=task, client=client, budget=budget,
                    prefix_cache=baseline_memo, lcm=baseline_bandit)
            elif strategy == "greedy":
                from toffee.search.baselines import greedy_search
                search_outcome = greedy_search(
                    task=task, client=client, budget=budget,
                    prefix_cache=baseline_memo, lcm=baseline_bandit)
            elif strategy == "best_of_n":
                from toffee.search.baselines import best_of_n_search
                search_outcome = best_of_n_search(
                    task=task, client=client, budget=budget, n=5,
                    prefix_cache=baseline_memo, lcm=baseline_bandit)
            elif strategy == "beam_search":
                from toffee.search.baselines import beam_search
                search_outcome = beam_search(
                    task=task, client=client, budget=budget, beam_width=3,
                    prefix_cache=baseline_memo, lcm=baseline_bandit)
            elif strategy == "rejection_sampling":
                from toffee.search.baselines import rejection_sampling_search
                search_outcome = rejection_sampling_search(
                    task=task, client=client, budget=budget,
                    prefix_cache=baseline_memo, lcm=baseline_bandit)
            elif strategy == "lats":
                from toffee.search.baselines import lats_search
                search_outcome = lats_search(
                    task=task, client=client, budget=budget,
                    max_iterations=eff_max_iterations,
                    prefix_cache=baseline_memo, lcm=baseline_bandit)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        except Exception as exc:
            log.exception("%s search failed for %s: %s", strategy, task_id, exc)
            return {"task_id": task_id, "status": "search_error", "error": str(exc),
                    "cost": budget.spent_dollars, "elapsed_s": time.perf_counter() - t0}

        elapsed = time.perf_counter() - t0
        accepted = search_outcome.accepted_trajectory is not None
        trajectory = search_outcome.accepted_trajectory or search_outcome.best_trajectory

        if trajectory is None:
            return {"task_id": task_id, "status": "no_trajectory",
                    "quality": search_outcome.best_quality,
                    "reject_reason": search_outcome.reject_reason,
                    "cost": budget.spent_dollars, "elapsed_s": elapsed}

        openai_dir = output_dir / ("openai" if accepted else "openai/_rejected")
        openai_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(openai_dir / f"{task_id}.json")
        try:
            assembly = assemble_sft(
                trajectory=trajectory,
                task=task,
                env=task.get("env", {}),
                output_path=output_path,
                round_idx=task.get("round_idx", 0),
                client=client,
                mcts_quality=search_outcome.best_quality,
            )
        except Exception as exc:
            log.error("SFT assembly failed for %s: %s", task_id, exc)
            return {"task_id": task_id, "status": "assembly_error", "error": str(exc),
                    "quality": trajectory.quality, "cost": budget.spent_dollars, "elapsed_s": elapsed}

        budget.record(assembly.get("extra_cost", 0.0), assembly.get("extra_tokens", 0))
        if not accepted:
            status = "quality_gate_failed"
        elif assembly.get("ok"):
            status = "success"
        elif assembly.get("sample_judge_reason"):
            status = "quality_gate_failed"
        else:
            status = "validation_failed"
        return {"task_id": task_id, "status": status,
                "quality": trajectory.quality, "cost": budget.spent_dollars,
                "exec_cost": round(budget.spent_exec_dollars, 6),
                "exec_count": budget.exec_count,
                "exec_seconds": round(budget.exec_seconds, 2),
                "steps": len(trajectory.steps),
                "output_path": output_path if (accepted and assembly.get("ok")) else None,
                "sample_judge_score": assembly.get("sample_judge_score"),
                "sample_judge_reason": assembly.get("sample_judge_reason"),
                "accepted": accepted,
                "reject_reason": "" if accepted else search_outcome.reject_reason,
                "elapsed_s": elapsed,
                "search_metadata": search_outcome.metadata}
    finally:
        if strategy == "mcts":
            lcm.increment_task_count()


def main() -> None:
    parser = argparse.ArgumentParser(description="TOFFEE Synthesis Pipeline")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--sources", default="auto")
    parser.add_argument("--auto-large-mix", default="",
                        help="source_group:weight,... (default: uniform over the catalog)")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--budget-per-task", type=float, default=1.00)
    parser.add_argument("--mcts-iterations", type=int, default=MCTS_MAX_ITERATIONS)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks-per-env", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--early-abort-threshold", type=float, default=0.6)
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable cross-task prefix-cache reuse; each task runs a fully "
                             "independent MCTS search from scratch")
    parser.add_argument("--strategy",
                        choices=["mcts", "single_pass", "react", "greedy", "best_of_n",
                                 "beam_search", "rejection_sampling", "lats"],
                        default="mcts",
                        help="Search strategy (default: mcts)")
    parser.add_argument("--backend", choices=["openrouter"], default=None,
                        help="Override TOFFEE_BACKEND env var")
    args = parser.parse_args()

    import toffee.config as _cfg

    if args.backend:
        os.environ["TOFFEE_BACKEND"] = args.backend
        _cfg.BACKEND = args.backend

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "runs" / args.run_name
    ensure_dir(output_dir)
    setup_logging(args.log_level, log_file=str(output_dir / "run.log"))
    log.info("Output directory: %s", output_dir)

    log.info("Backend: %s, Models: %s", _cfg.BACKEND, dict(_cfg.MODELS))

    client = OpenRouterClient()
    lcm = FactoredLinUCB()
    prefix_cache = PrefixCache()

    catalog = load_catalog(None if args.sources == "auto" else args.sources)
    mix = parse_mix(args.auto_large_mix)
    _tasks_file = os.environ.get("TOFFEE_TASKS_FILE", "")
    if _tasks_file:
        with open(_tasks_file) as _tf:
            tasks = json.load(_tf)
        log.info("Loaded %d tasks from %s (synthesis skipped)", len(tasks), _tasks_file)
    else:
        tasks = synthesize_task_pool(
            catalog, mix, args.total, client,
            tasks_per_env=args.tasks_per_env, seed=args.seed,
        )

    log.info("Total tasks to process: %d", len(tasks))
    if not _tasks_file:
        _dump_path = str(output_dir / "tasks_admitted.json")
        try:
            with open(_dump_path, "w") as _df:
                json.dump(tasks, _df, ensure_ascii=False, default=str)
            log.info("Admitted tasks dumped to %s", _dump_path)
        except Exception as _exc:
            log.warning("Task dump failed: %s", _exc)
    if os.environ.get("TOFFEE_TASKS_ONLY", "") == "1":
        log.info("TOFFEE_TASKS_ONLY=1 — exiting after task synthesis")
        return

    results: List[Dict[str, Any]] = []
    successes = failures = 0

    from toffee.generation.bottomup import _scratch_db_for, _collect_sources
    from toffee.generation.ingest import ingest_environment

    def _materialize_env(task):
        env = task.get("env") or {}
        db = (env.get("db_path") or "").lower()
        if db.endswith((".sqlite", ".db", ".sqlite3")) and os.path.exists(env["db_path"]):
            return
        scratch = _scratch_db_for(env)
        if not os.path.exists(scratch):
            ingest_environment(_collect_sources(env), scratch)
        env["db_path"] = scratch
        task["env"] = env

    for _t in tasks:
        _materialize_env(_t)

    use_prefix_cache = not args.no_prefix_cache
    strategy = args.strategy
    if not use_prefix_cache:
        log.info("Prefix cache disabled: all tasks will run independent MCTS searches")
    if strategy != "mcts":
        from toffee.search.baselines import _get_baseline_model
        log.info("Strategy: %s (baselines use model=%s)",
                 strategy, _get_baseline_model())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_task, t, client, lcm, prefix_cache, output_dir,
                        args.budget_per_task, args.mcts_iterations, use_prefix_cache,
                        strategy=strategy): t
            for t in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"task_id": task["task_id"], "status": "exception", "error": str(exc)}

            results.append(result)
            if result.get("status") in ("success", "resumed"):
                successes += 1
            else:
                failures += 1

            total_done = successes + failures
            log.info("[%d/%d] %s — %s (q=%.2f, $%.4f, %.1fs)",
                     total_done, len(tasks), result.get("task_id", "?"),
                     result.get("status"), result.get("quality", 0),
                     result.get("cost", 0), result.get("elapsed_s", 0))

            if total_done >= 10 and failures / total_done > args.early_abort_threshold:
                log.error("Early abort: failure rate %.1f%% after %d tasks",
                          100 * failures / total_done, total_done)
                for f in futures:
                    f.cancel()
                break

    manifest = {
        "run_name": args.run_name, "strategy": strategy,
        "total_tasks": len(tasks),
        "completed": len(results), "successes": successes, "failures": failures,
        "success_rate": successes / max(len(results), 1),
        "total_cost": sum(r.get("cost", 0) for r in results),
        "total_exec_cost": round(sum(r.get("exec_cost", 0) for r in results), 4),
        "total_exec_count": sum(r.get("exec_count", 0) for r in results),
        "use_prefix_cache": use_prefix_cache,
        "prefix_cache_stats": prefix_cache.stats() if use_prefix_cache else {"disabled": True},
        "bandit_tasks": lcm.task_count if strategy == "mcts" else 0,
        "results": results,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


    swift_dir = output_dir / "swift"
    if swift_dir.is_dir():
        jsonl_path = output_dir / f"{args.run_name}_swift.jsonl"
        swift_files = sorted(swift_dir.glob("*.json"))
        with open(jsonl_path, "w", encoding="utf-8") as fout:
            for sf in swift_files:
                with open(sf, encoding="utf-8") as fin:
                    fout.write(fin.read().replace("\n", " ").strip() + "\n")
        log.info("Merged %d swift samples -> %s", len(swift_files), jsonl_path)

    log.info("=" * 60)
    log.info("Run: %s", args.run_name)
    log.info("Success: %d/%d (%.1f%%)", successes, len(results), 100 * successes / max(len(results), 1))
    log.info("Total cost: $%.4f", manifest["total_cost"])
    if use_prefix_cache:
        log.info("Prefix cache hit rate: %.1f%%", 100 * prefix_cache.hit_rate)
    else:
        log.info("Prefix cache: disabled (--no-dag mode)")
    log.info("=" * 60)
