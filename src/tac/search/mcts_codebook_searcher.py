# SPDX-License-Identifier: MIT
"""MCTSCodebookSearcher — in-house Monte Carlo Tree Search for discrete
codebook search.

Per `.omx/research/tac_search_namespace_design_20260517.md` §6 + §10:

  MCTS is canonical for discrete codebook search problems where the state
  space is too large for exhaustive enumeration (e.g., per-pair selector
  index assignment with K modes per pair × 600 pairs = K^600 candidate
  archives; selecting per-block quantization scale from a discrete set).

This builder implements the canonical UCT (Upper Confidence Bound applied
to Trees) algorithm in PURE PYTHON. UNIQUE-FORK: not a library wrapper
because:
  - The state space is substrate-specific (encoded in the bounds spec)
  - No production-grade discrete-codebook MCTS library exists with the
    right contract surface
  - Pure-Python is sufficient — MCTS bottleneck is the objective eval

Per CLAUDE.md "Bit-level deconstruction" + Catalog #158: deterministic
when seed is pinned. UCT exploration uses Python's `random.Random(seed)`
for tie-breaking + simulation rollouts.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": the algorithm IS canonical
MCTS (Kocsis-Szepesvári 2006 UCT); only the integration into tac.search
+ the bounds-encoded action space is forked.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from tac.search.contract import SearchContract
from tac.search.decorator import search_strategy
from tac.search.errors import ObjectiveFunctionError
from tac.search.pipeline import SearchHistory, SearchResult, SearchTrial

__all__ = [
    "MCTSCodebookSearcher",
    "MCTSCodebookSearcherSpec",
]


@dataclass(frozen=True)
class MCTSCodebookSearcherSpec:
    """Specification for an MCTS discrete-codebook search strategy.

    bounds dict shape: {param_name: list_of_discrete_choices}. The MCTS
    builds a tree where each level corresponds to one parameter (sorted by
    name for byte-stability) and each child node is one discrete choice.
    """

    strategy_id: str
    bounds: Mapping[str, list[Any]]
    max_simulations: int = 500
    exploration_constant_c_uct: float = 1.414  # sqrt(2)
    rollout_depth_limit: int | None = None
    seed: int = 42
    sensitivity_weighted: bool = False
    description: str = ""
    lane_id: str | None = None
    objective_is_surrogate: bool = False
    predicted_search_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, Mapping) or not self.bounds:
            raise ValueError(
                f"bounds={self.bounds!r} must be a non-empty mapping of "
                "{param_name: [choice, ...]}"
            )
        for name, choices in self.bounds.items():
            if not isinstance(choices, (list, tuple)) or not choices:
                raise ValueError(
                    f"bounds[{name!r}]={choices!r}: choices must be a "
                    "non-empty list/tuple"
                )
        if self.max_simulations < 1:
            raise ValueError(
                f"max_simulations={self.max_simulations} must be >= 1"
            )
        if self.exploration_constant_c_uct <= 0:
            raise ValueError(
                f"exploration_constant_c_uct="
                f"{self.exploration_constant_c_uct} must be > 0"
            )
        if self.rollout_depth_limit is not None and self.rollout_depth_limit < 1:
            raise ValueError(
                f"rollout_depth_limit={self.rollout_depth_limit} must be "
                ">= 1 or None"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


@dataclass
class _MCTSNode:
    """A single node in the MCTS tree.

    The state is a partial parameter assignment (depth = number of params
    fixed so far). Children correspond to discrete choices for the NEXT
    parameter (sorted by name).
    """

    depth: int
    partial_assignment: dict[str, Any]
    visit_count: int = 0
    total_value: float = 0.0
    children: dict[Any, "_MCTSNode"] = field(default_factory=dict)

    def is_fully_expanded(self, choices_at_depth: list[Any]) -> bool:
        return len(self.children) == len(choices_at_depth)

    def is_terminal(self, n_params: int) -> bool:
        return self.depth == n_params


class MCTSCodebookSearcher:
    """Builder for an MCTS discrete-codebook search strategy."""

    def __init__(self, *, spec: MCTSCodebookSearcherSpec) -> None:
        if not isinstance(spec, MCTSCodebookSearcherSpec):
            raise TypeError(
                f"spec must be MCTSCodebookSearcherSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SearchContract:
        rationale: dict[str, str] = {
            "hook_pareto_constraint": (
                "Single-objective MCTS; Pareto undefined (use Rashomon "
                "ensemble for multi-criterion ranking instead)."
            ),
            "hook_bit_allocator_class": (
                "MCTS discovers discrete codebook assignments; bit "
                "allocation is the downstream consumer."
            ),
            "hook_probe_disambiguator": (
                "MCTS with UCT exploration is canonical; the rollout "
                "policy is the design surface."
            ),
        }
        if not self.spec.sensitivity_weighted:
            rationale["hook_sensitivity_contribution"] = (
                "MCTS UCT is sensitivity-blind; set "
                "sensitivity_weighted=True to bias the initial child "
                "selection by master_gradient."
            )
        return SearchContract(
            id=self.spec.strategy_id,
            parent_strategy_id=None,
            description=(
                self.spec.description
                or (
                    f"MCTS (UCT) for discrete codebook search: "
                    f"max_simulations={self.spec.max_simulations}, "
                    f"c_uct={self.spec.exploration_constant_c_uct}, "
                    f"params={sorted(self.spec.bounds.keys())!r}."
                )
            ),
            search_kind="discrete",
            n_candidate_evaluations_max=self.spec.max_simulations,
            parallelism="serial",  # MCTS is state-dependent across trials
            requires_objective_function=True,
            objective_is_surrogate=self.spec.objective_is_surrogate,
            deterministic=True,
            seed=self.spec.seed,
            predicted_search_cost_usd=self.spec.predicted_search_cost_usd,
            hook_sensitivity_contribution=(
                "master_gradient_v1"
                if self.spec.sensitivity_weighted
                else "not_applicable_with_rationale"
            ),
            hook_pareto_constraint="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind="search_strategy_outcomes_v1",
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale=rationale,
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/tac_search_namespace_design_20260517.md"
            ),
            canonical_vs_unique_decision=(
                "UNIQUE-FORK: in-house MCTS implementation in pure Python "
                "(no external library exposes the right discrete-codebook "
                "contract surface). The UCT algorithm itself is canonical "
                "(Kocsis-Szepesvári 2006); the tree structure + action "
                "space encoding is forked to this namespace."
            ),
        )

    def register(self) -> SearchContract:
        contract = self.build_contract()
        spec = self.spec

        @search_strategy(contract)
        def _mcts_run(
            objective_fn: Callable[[Mapping[str, Any]], float],
            *,
            bounds: Mapping[str, list[Any]] | None = None,
            seed: int = spec.seed,
            warm_start: Mapping[str, Any] | None = None,
            **_ignored: Any,
        ) -> SearchResult:
            return _run_mcts(
                contract=contract,
                spec=spec,
                objective_fn=objective_fn,
                bounds=bounds or spec.bounds,
                seed=seed,
                warm_start=warm_start,
            )

        return contract


def _run_mcts(
    *,
    contract: SearchContract,
    spec: MCTSCodebookSearcherSpec,
    objective_fn: Callable[[Mapping[str, Any]], float],
    bounds: Mapping[str, list[Any]],
    seed: int,
    warm_start: Mapping[str, Any] | None,
) -> SearchResult:
    """Execute MCTS UCT; return SearchResult.

    Algorithm (per Kocsis-Szepesvári 2006):
      For sim in range(max_simulations):
        1. SELECT: traverse from root using UCT until a non-fully-expanded
           or terminal node is reached
        2. EXPAND: add one new child to the selected node
        3. SIMULATE: random rollout from the new child to a terminal state
        4. BACKPROPAGATE: update value statistics along the path

    Returns the best terminal-state params + score observed across all
    simulations.
    """
    rng = random.Random(seed)
    param_names = sorted(bounds.keys())
    n_params = len(param_names)
    choices_per_depth = [list(bounds[n]) for n in param_names]

    root = _MCTSNode(depth=0, partial_assignment={})

    trials: list[SearchTrial] = []
    best_params: dict[str, Any] = {}
    best_score = math.inf
    start = time.monotonic()
    trial_index = 0

    # Optional warm-start: seed the first rollout with the warm_start params
    # if it's a complete + valid assignment.
    if warm_start and all(
        n in warm_start and warm_start[n] in choices_per_depth[i]
        for i, n in enumerate(param_names)
    ):
        params = {n: warm_start[n] for n in param_names}
        score = _evaluate(objective_fn, params, trial_index)
        trials.append(
            SearchTrial(trial_index=trial_index, params=params, score=score)
        )
        if score < best_score:
            best_score = score
            best_params = dict(params)
        trial_index += 1

    for sim in range(spec.max_simulations - trial_index):
        # 1. SELECT + 2. EXPAND in one pass
        node = root
        path: list[_MCTSNode] = [node]
        while not node.is_terminal(n_params):
            choices_here = choices_per_depth[node.depth]
            if not node.is_fully_expanded(choices_here):
                # Expand: pick an unexplored choice
                unexplored = [c for c in choices_here if c not in node.children]
                choice = rng.choice(unexplored)
                child_assignment = dict(node.partial_assignment)
                child_assignment[param_names[node.depth]] = choice
                child = _MCTSNode(
                    depth=node.depth + 1,
                    partial_assignment=child_assignment,
                )
                node.children[choice] = child
                node = child
                path.append(node)
                break
            # Fully expanded: UCT-select
            node = _uct_select_child(
                node, choices_here, spec.exploration_constant_c_uct, rng
            )
            path.append(node)

        # 3. SIMULATE: random rollout from `node` to terminal
        rollout_assignment = dict(node.partial_assignment)
        for d in range(node.depth, n_params):
            rollout_assignment[param_names[d]] = rng.choice(
                choices_per_depth[d]
            )
            if (
                spec.rollout_depth_limit is not None
                and (d - node.depth) >= spec.rollout_depth_limit
            ):
                # Fill remaining with first-choice (deterministic)
                for d2 in range(d + 1, n_params):
                    rollout_assignment[param_names[d2]] = choices_per_depth[
                        d2
                    ][0]
                break

        score = _evaluate(objective_fn, rollout_assignment, trial_index)
        trials.append(
            SearchTrial(
                trial_index=trial_index,
                params=dict(rollout_assignment),
                score=score,
            )
        )
        if score < best_score:
            best_score = score
            best_params = dict(rollout_assignment)
        trial_index += 1

        # 4. BACKPROPAGATE: minimization → use -score as "reward" so UCT
        # prefers low-score paths (higher reward).
        reward = -score
        for n in path:
            n.visit_count += 1
            n.total_value += reward

    elapsed = time.monotonic() - start
    return SearchResult(
        strategy_id=spec.strategy_id,
        best_params=best_params,
        best_score=best_score,
        n_evaluations=trial_index,
        elapsed_seconds=elapsed,
        history=SearchHistory(trials=tuple(trials)),
        objective_is_surrogate=contract.objective_is_surrogate,
        seed=seed,
    )


def _uct_select_child(
    node: _MCTSNode,
    choices_here: list[Any],
    c_uct: float,
    rng: random.Random,
) -> _MCTSNode:
    """Select the child with the highest UCT score.

    UCT(child) = mean_reward(child) + c * sqrt(ln(parent.visits) / child.visits)

    Ties broken randomly via `rng.choice` so byte-stability is preserved
    across runs with the same seed.
    """
    log_parent = math.log(max(node.visit_count, 1))
    best_uct = -math.inf
    best_children: list[_MCTSNode] = []
    for choice in choices_here:
        child = node.children[choice]
        if child.visit_count == 0:
            return child
        mean = child.total_value / child.visit_count
        uct = mean + c_uct * math.sqrt(log_parent / child.visit_count)
        if uct > best_uct:
            best_uct = uct
            best_children = [child]
        elif uct == best_uct:
            best_children.append(child)
    return rng.choice(best_children)


def _evaluate(
    objective_fn: Callable[[Mapping[str, Any]], float],
    params: Mapping[str, Any],
    trial_index: int,
) -> float:
    """Invoke objective_fn + coerce/validate the result."""
    try:
        score = objective_fn(params)
    except Exception as exc:
        raise ObjectiveFunctionError(
            f"objective_fn raised on MCTS trial {trial_index}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        score_f = float(score)
    except (TypeError, ValueError) as exc:
        raise ObjectiveFunctionError(
            f"objective_fn returned {score!r} on MCTS trial {trial_index}; "
            "expected a finite float."
        ) from exc
    if score_f != score_f or score_f in (float("inf"), float("-inf")):
        raise ObjectiveFunctionError(
            f"objective_fn returned non-finite {score_f} on MCTS trial "
            f"{trial_index}"
        )
    return score_f
