
from __future__ import annotations

import copy
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from toffee.core.state import AnalysisState, CompressedMemory

log = logging.getLogger(__name__)


@dataclass
class MemoNode:

    canonical_key: str
    depth: int
    env_fingerprint: str


    assistant_content: str
    tool_content: str


    operator_name: str
    schema_discovered: bool
    result_exists: bool
    answer_drafted: bool
    has_error: bool
    consecutive_errors: int
    pending_goal: str
    last_tool: str
    last_result_nonempty: bool
    resolved_prior_error: bool
    total_cost: float
    total_tokens: int
    memory: CompressedMemory


    parent: Optional[MemoNode]
    observed_units: List[str] = field(default_factory=list)
    children: Dict[str, "MemoNode"] = field(default_factory=dict)


    usage_count: int = 0
    cumulative_reward: float = 0.0
    best_downstream_reward: float = -1.0
    last_used_ts: float = field(default_factory=time.time)

    @property
    def avg_reward(self) -> float:
        return self.cumulative_reward / self.usage_count if self.usage_count > 0 else 0.0


class PrefixCache:

    _ID_MAP_CAP = 10_000

    def __init__(self, max_nodes: int = 2000):

        self._nodes: Dict[str, MemoNode] = {}


        self._roots: Dict[str, MemoNode] = {}


        self._id_to_canonical: Dict[str, str] = {}
        self._id_insertion_order: List[str] = []


        self._last_prefix_key: Dict[str, str] = {}

        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._merges = 0
        self._max_nodes = max_nodes


    def _get_or_create_root(self, env_fp: str) -> MemoNode:
        if env_fp not in self._roots:
            self._roots[env_fp] = MemoNode(
                canonical_key=f"ROOT|{env_fp}",
                depth=0, env_fingerprint=env_fp,
                assistant_content="", tool_content="",
                operator_name="", schema_discovered=False,
                result_exists=False, answer_drafted=False,
                has_error=False, consecutive_errors=0,
                pending_goal="reconnaissance",
                last_tool="other", last_result_nonempty=False,
                resolved_prior_error=False,
                total_cost=0.0, total_tokens=0,
                memory=CompressedMemory(),
                observed_units=[],
                parent=None,
            )
        return self._roots[env_fp]

    def _record_id(self, state_id: str, canonical_key: str) -> None:
        self._id_to_canonical[state_id] = canonical_key
        self._id_insertion_order.append(state_id)
        if len(self._id_to_canonical) > self._ID_MAP_CAP:

            drop = len(self._id_insertion_order) // 2
            for sid in self._id_insertion_order[:drop]:
                self._id_to_canonical.pop(sid, None)
            self._id_insertion_order = self._id_insertion_order[drop:]

    @staticmethod
    def _passive_reward(state: AnalysisState) -> float:
        if state.answer_drafted and state.step_count >= 7:
            return 0.7
        if state.result_exists and not state.answer_drafted and state.step_count >= 4:
            return 0.3
        if state.schema_discovered and state.step_count >= 2:
            return 0.1
        return 0.0

    def _backpropagate_reward(self, node: MemoNode, reward: float) -> None:
        current: Optional[MemoNode] = node
        while current is not None and current.depth > 0:
            current.cumulative_reward += reward
            current.best_downstream_reward = max(
                current.best_downstream_reward, reward,
            )
            current = current.parent

    def _materialize(self, node: MemoNode) -> AnalysisState:

        path: List[MemoNode] = []
        cur: Optional[MemoNode] = node
        while cur is not None and cur.depth > 0:
            path.append(cur)
            cur = cur.parent
        path.reverse()


        messages: List[Dict[str, str]] = []
        for n in path:
            messages.append({"role": "assistant", "content": n.assistant_content})
            messages.append({"role": "tool", "content": n.tool_content})

        state = AnalysisState(
            env_fingerprint=node.env_fingerprint,
            messages=messages,
            memory=copy.deepcopy(node.memory),
            schema_discovered=node.schema_discovered,
            result_exists=node.result_exists,
            answer_drafted=node.answer_drafted,
            has_error=node.has_error,
            consecutive_errors=node.consecutive_errors,
            pending_goal=node.pending_goal,
            last_tool=node.last_tool,
            last_result_nonempty=node.last_result_nonempty,
            resolved_prior_error=node.resolved_prior_error,
            step_count=node.depth,
            total_cost=node.total_cost,
            total_tokens=node.total_tokens,
            operator_name=node.operator_name,
            config_model="",
            llm_content=node.assistant_content,
            tool_output=node.tool_content,
        )
        state.compute_canonical_key()
        return state

    def _selection_score(
        self, node: MemoNode, parent_total: int, entry_units: List[str],
    ) -> float:
        if node.usage_count == 0:
            return float("inf")
        exploit = node.avg_reward
        explore = math.sqrt(
            2.0 * math.log(max(parent_total, 1)) / node.usage_count
        )
        return exploit + explore

    def _evict(self) -> None:
        now = time.time()
        leaves: List[tuple] = []
        for key, node in self._nodes.items():
            if node.children:
                continue
            recency = now - node.last_used_ts
            quality = node.avg_reward
            eviction_score = recency - 100.0 * quality
            leaves.append((eviction_score, key))

        leaves.sort(reverse=True)
        n_evict = max(1, len(leaves) // 4)
        for _, key in leaves[:n_evict]:
            node = self._nodes.pop(key, None)
            if node is None:
                continue
            if node.parent is not None and key in node.parent.children:
                del node.parent.children[key]


    def find_reusable_prefix(
        self, new_state: AnalysisState,
    ) -> Optional[AnalysisState]:
        env_fp = new_state.env_fingerprint

        with self._lock:
            sentinel = self._roots.get(env_fp)
            if sentinel is None or not sentinel.children:
                self._misses += 1
                log.info("Prefix cache MISS for db=%s: no prefix DAG yet (nodes=%d)",
                         env_fp[:8], len(self._nodes))
                return None


            path: List[MemoNode] = [sentinel]
            current = sentinel
            entry_units = list(new_state.entry_units)
            while current.children:
                parent_total = sum(
                    c.usage_count for c in current.children.values()
                )
                best_child: Optional[MemoNode] = None
                best_score = -float("inf")
                for child in current.children.values():
                    if child.consecutive_errors >= 3:
                        continue
                    if child.answer_drafted or child.result_exists:
                        continue
                    score = self._selection_score(child, parent_total, entry_units)
                    if score > best_score:
                        best_score = score
                        best_child = child
                if best_child is None:
                    break
                path.append(best_child)
                current = best_child


            selected: Optional[MemoNode] = None
            for node in reversed(path):
                if node is sentinel:
                    continue
                if node.answer_drafted or node.result_exists:
                    continue
                if node.schema_discovered:
                    selected = node
                    break

            if selected is None:
                self._misses += 1
                schema_flags = [n.schema_discovered for n in path if n is not sentinel]
                log.info("Prefix cache MISS for db=%s: UCB1 path depth=%d, schema_flags=%s",
                         env_fp[:8], len(path) - 1, schema_flags)
                return None

            self._hits += 1
            selected.usage_count += 1
            selected.last_used_ts = time.time()
            self._last_prefix_key[env_fp] = selected.canonical_key
            result = self._materialize(selected)

        log.info(
            "Prefix cache HIT for db=%s: reusing prefix depth=%d tables=%d "
            "avg_reward=%.3f usage=%d",
            env_fp[:8], selected.depth,
            len(selected.memory.discovered_tables),
            selected.avg_reward, selected.usage_count,
        )
        return result

    def register(self, state: AnalysisState) -> None:
        if state.step_count == 0:
            return

        key = state.canonical_key
        if not key:
            key = state.compute_canonical_key()

        with self._lock:

            self._record_id(state.state_id, key)


            if not state.schema_discovered:
                return
            if state.answer_drafted or state.result_exists:
                return


            if key in self._nodes:
                existing = self._nodes[key]
                existing.usage_count += 1
                existing.last_used_ts = time.time()
                self._merges += 1

                reward = self._passive_reward(state)
                if reward > 0:
                    self._backpropagate_reward(existing, reward)
                return


            parent_node: Optional[MemoNode] = None
            if state.parent_id and state.parent_id in self._id_to_canonical:
                parent_key = self._id_to_canonical[state.parent_id]
                parent_node = self._nodes.get(parent_key)

            if parent_node is None:

                prefix_key = self._last_prefix_key.get(state.env_fingerprint)
                if prefix_key:
                    parent_node = self._nodes.get(prefix_key)

            if parent_node is None:
                parent_node = self._get_or_create_root(state.env_fingerprint)


            assistant_content = ""
            tool_content = ""
            if len(state.messages) >= 2:
                assistant_content = state.messages[-2].get("content", "")
                tool_content = state.messages[-1].get("content", "")

            node = MemoNode(
                canonical_key=key,
                depth=state.step_count,
                env_fingerprint=state.env_fingerprint,
                assistant_content=assistant_content,
                tool_content=tool_content,
                operator_name=state.operator_name,
                schema_discovered=state.schema_discovered,
                result_exists=state.result_exists,
                answer_drafted=state.answer_drafted,
                has_error=state.has_error,
                consecutive_errors=state.consecutive_errors,
                pending_goal=state.pending_goal,
                last_tool=state.last_tool,
                last_result_nonempty=state.last_result_nonempty,
                resolved_prior_error=state.resolved_prior_error,
                total_cost=state.total_cost,
                total_tokens=state.total_tokens,
                memory=copy.deepcopy(state.memory),
                observed_units=[t.strip().lower() for t in state.memory.discovered_tables if t.strip()],
                parent=parent_node,
                usage_count=1,
            )
            parent_node.children[key] = node
            self._nodes[key] = node


            reward = self._passive_reward(state)
            if reward > 0:
                self._backpropagate_reward(node, reward)


            if len(self._nodes) > self._max_nodes:
                self._evict()

    def propagate_trajectory_reward(
        self,
        env_fingerprint: str,
        trajectory_reward: float,
        beta: float = 0.5,
    ) -> None:
        with self._lock:
            prefix_key = self._last_prefix_key.get(env_fingerprint)
            if prefix_key is None:
                return
            node = self._nodes.get(prefix_key)
            if node is None:
                return

            current: Optional[MemoNode] = node
            while current is not None and current.depth > 0:
                passive = self._passive_reward_from_node(current)
                blended = beta * trajectory_reward + (1.0 - beta) * passive
                current.cumulative_reward += blended
                current.usage_count += 1
                current.best_downstream_reward = max(
                    current.best_downstream_reward, trajectory_reward,
                )
                current = current.parent

    @staticmethod
    def _passive_reward_from_node(node: MemoNode) -> float:
        if node.answer_drafted and node.depth >= 7:
            return 0.7
        if node.result_exists and not node.answer_drafted and node.depth >= 4:
            return 0.3
        if node.schema_discovered and node.depth >= 2:
            return 0.1
        return 0.0


    @property
    def frontier_size(self) -> int:
        with self._lock:
            return len(self._nodes)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        with self._lock:
            max_depth = 0
            for node in self._nodes.values():
                if node.depth > max_depth:
                    max_depth = node.depth
            return {
                "frontier_size": len(self._nodes),
                "db_count": len(self._roots),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
                "merges": self._merges,
                "dag_depth": max_depth,
            }
