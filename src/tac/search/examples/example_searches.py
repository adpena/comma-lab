# SPDX-License-Identifier: MIT
"""Canonical example search strategies — one per builder.

Five minimal strategies that exercise the decorator + composition API:

  1. ``random_search_baseline_example``       : in-line bare @search_strategy
  2. ``cma_es_over_x_example``                : CMAESCandidateSearcher
  3. ``tpe_mixed_x_y_example``                : OptunaTPESampler
  4. ``gp_bo_over_x_example``                 : BayesianOptimizationGP
  5. ``mcts_codebook_example``                : MCTSCodebookSearcher
  6. ``rashomon_committee_example``           : RashomonEnsembleCommittee

The example bodies are TOY (synthetic objective or fixed score) so the
strategies are testable without GPU + without the external libraries
installed (for the 3 wrappers the registration call works; the .run()
will raise SearchEngineNotInstalledError if `cma` / `optuna` / `skopt`
are absent, which the test suite exercises explicitly).

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — every claim in
the docstrings is backed by an executable body.
"""

from __future__ import annotations

from typing import Any, Mapping

from tac.search.bayesian_optimization_gp import (
    BayesianOptimizationGP,
    BayesianOptimizationGPSpec,
)
from tac.search.cma_es_searcher import (
    CMAESCandidateSearcher,
    CMAESCandidateSearcherSpec,
)
from tac.search.contract import SearchContract
from tac.search.decorator import search_strategy
from tac.search.mcts_codebook_searcher import (
    MCTSCodebookSearcher,
    MCTSCodebookSearcherSpec,
)
from tac.search.optuna_tpe_sampler import (
    OptunaTPESampler,
    OptunaTPESamplerSpec,
)
from tac.search.pipeline import SearchHistory, SearchResult, SearchTrial
from tac.search.rashomon_ensemble_committee import (
    RashomonEnsembleCommittee,
    RashomonEnsembleCommitteeSpec,
)

_EXAMPLE_LANE_ID = (
    "lane_tac_search_namespace_decorator_api_20260517"
)
_EXAMPLE_DESIGN_MEMO = (
    ".omx/research/tac_search_namespace_design_20260517.md"
)


# ---------------------------------------------------------------------------
# 1. RandomSearch baseline — in-line @search_strategy
# ---------------------------------------------------------------------------


@search_strategy(
    SearchContract(
        id="random_search_baseline_example",
        description=(
            "Toy RandomSearch baseline: draws `n_candidate_evaluations_max` "
            "samples uniformly from `bounds` and returns the best."
        ),
        search_kind="continuous",
        n_candidate_evaluations_max=20,
        parallelism="process_pool",
        requires_objective_function=True,
        objective_is_surrogate=False,
        deterministic=True,
        seed=42,
        predicted_search_cost_usd=0.0,
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_pareto_constraint="not_applicable_with_rationale",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="cathedral_autopilot_v1",
        hook_continual_learning_anchor_kind="search_strategy_outcomes_v1",
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": (
                "RandomSearch is sensitivity-blind by design."
            ),
            "hook_pareto_constraint": (
                "Single-objective; Pareto undefined."
            ),
            "hook_bit_allocator_class": (
                "RandomSearch discovers parameter values, not bit allocations."
            ),
            "hook_probe_disambiguator": (
                "Single canonical uniform-sampling interpretation."
            ),
        },
        lane_id=_EXAMPLE_LANE_ID,
        design_memo=_EXAMPLE_DESIGN_MEMO,
        canonical_vs_unique_decision=(
            "ADOPT_CANONICAL_BECAUSE_SERVES: baseline / warm-start for "
            "subsequent more sophisticated strategies."
        ),
    )
)
def random_search_baseline_example(
    objective_fn,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    seed: int = 42,
    warm_start: Mapping[str, Any] | None = None,
    **_ignored: Any,
) -> SearchResult:
    """Toy RandomSearch: 20 uniform samples; return the best."""
    import random
    import time

    rng = random.Random(seed)
    bounds = bounds or {"x": (0.0, 1.0)}
    trials: list[SearchTrial] = []
    best_params: dict[str, Any] = {}
    best_score = float("inf")
    start = time.monotonic()
    for i in range(20):
        params = {n: rng.uniform(*bounds[n]) for n in bounds}
        score = float(objective_fn(params))
        trials.append(
            SearchTrial(trial_index=i, params=params, score=score)
        )
        if score < best_score:
            best_score = score
            best_params = params
    elapsed = time.monotonic() - start
    return SearchResult(
        strategy_id="random_search_baseline_example",
        best_params=best_params,
        best_score=best_score,
        n_evaluations=20,
        elapsed_seconds=elapsed,
        history=SearchHistory(trials=tuple(trials)),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 2-6. Builder-driven examples — registered lazily so the contracts exist
# in the registry the moment example_searches is imported.
# ---------------------------------------------------------------------------


def _register_cma_es_example() -> SearchContract:
    return CMAESCandidateSearcher(
        spec=CMAESCandidateSearcherSpec(
            strategy_id="cma_es_over_x_example",
            bounds={"x": (-5.0, 5.0)},
            population_size=4,
            sigma_init=1.0,
            max_evaluations=20,
            seed=42,
            description="Toy CMA-ES example; minimizes a 1-D quadratic.",
            lane_id=_EXAMPLE_LANE_ID,
        )
    ).register()


def _register_optuna_tpe_example() -> SearchContract:
    return OptunaTPESampler(
        spec=OptunaTPESamplerSpec(
            strategy_id="tpe_mixed_x_y_example",
            bounds={
                "x": ("float", -5.0, 5.0),
                "y": ("int", 0, 10),
                "use_offset": ("bool",),
            },
            n_trials=20,
            n_startup_trials=5,
            seed=42,
            description="Toy Optuna TPE example over mixed types.",
            lane_id=_EXAMPLE_LANE_ID,
        )
    ).register()


def _register_gp_bo_example() -> SearchContract:
    return BayesianOptimizationGP(
        spec=BayesianOptimizationGPSpec(
            strategy_id="gp_bo_over_x_example",
            bounds={"x": (-5.0, 5.0)},
            n_initial_points=4,
            n_calls=12,
            seed=42,
            description="Toy GP-BO example; 12 calls on a 1-D quadratic.",
            lane_id=_EXAMPLE_LANE_ID,
        )
    ).register()


def _register_mcts_example() -> SearchContract:
    return MCTSCodebookSearcher(
        spec=MCTSCodebookSearcherSpec(
            strategy_id="mcts_codebook_example",
            bounds={
                "K": [4, 8, 16, 32],
                "lambda_log10": [-3, -2, -1, 0],
            },
            max_simulations=50,
            seed=42,
            description="Toy MCTS example over discrete codebook choices.",
            lane_id=_EXAMPLE_LANE_ID,
        )
    ).register()


def _register_rashomon_committee_example() -> SearchContract:
    return RashomonEnsembleCommittee(
        spec=RashomonEnsembleCommitteeSpec(
            strategy_id="rashomon_committee_example",
            candidate_pool=[
                {"candidate_id": f"c{i}", "K": 4 + (i * 4)}
                for i in range(8)
            ],
            ensemble_size=4,  # toy: 4 instead of canonical 8 for speed
            bootstrap_seed_base=42,
            sparsity_target=3,
            integer_coefficient_bound=10,
            description="Toy Rashomon committee example over K=8 candidates.",
            lane_id=_EXAMPLE_LANE_ID,
        )
    ).register()


def _register_random_search_baseline() -> SearchContract:
    """Re-register the in-line RandomSearch baseline after a registry clear.

    The original ``random_search_baseline_example`` is registered at
    module-import via ``@search_strategy``; once a test fixture clears
    the registry the function loses its contract. This helper re-creates
    + re-registers via the same canonical contract so post-clear
    callers can still invoke the strategy by id.
    """
    from tac.search.decorator import _REGISTERED_STRATEGIES

    strategy_id = "random_search_baseline_example"
    if strategy_id in _REGISTERED_STRATEGIES:
        return _REGISTERED_STRATEGIES[strategy_id]
    # Re-build the contract (must mirror the @search_strategy contract
    # above exactly so behavior is preserved).
    contract = SearchContract(
        id=strategy_id,
        description=(
            "Toy RandomSearch baseline: draws "
            "`n_candidate_evaluations_max` samples uniformly from `bounds` "
            "and returns the best."
        ),
        search_kind="continuous",
        n_candidate_evaluations_max=20,
        parallelism="process_pool",
        requires_objective_function=True,
        objective_is_surrogate=False,
        deterministic=True,
        seed=42,
        predicted_search_cost_usd=0.0,
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_pareto_constraint="not_applicable_with_rationale",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="cathedral_autopilot_v1",
        hook_continual_learning_anchor_kind="search_strategy_outcomes_v1",
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": (
                "RandomSearch is sensitivity-blind by design."
            ),
            "hook_pareto_constraint": (
                "Single-objective; Pareto undefined."
            ),
            "hook_bit_allocator_class": (
                "RandomSearch discovers parameter values, not bit allocations."
            ),
            "hook_probe_disambiguator": (
                "Single canonical uniform-sampling interpretation."
            ),
        },
        lane_id=_EXAMPLE_LANE_ID,
        design_memo=_EXAMPLE_DESIGN_MEMO,
        canonical_vs_unique_decision=(
            "ADOPT_CANONICAL_BECAUSE_SERVES: baseline / warm-start for "
            "subsequent more sophisticated strategies."
        ),
    )
    search_strategy(contract)(random_search_baseline_example)
    return contract


def register_example_searches() -> list[SearchContract]:
    """Register all 6 example strategies (random + 5 builder) + return contracts.

    Idempotent at the id level: if a strategy with the example id is
    already registered the existing contract is returned (the registry
    is the single source of truth; a second `register()` would build a
    fresh contract object that is value-equal but identity-not-equal,
    which would trigger the decorator's duplicate-id detector).
    """
    from tac.search.decorator import _REGISTERED_STRATEGIES

    builders = [
        ("random_search_baseline_example", _register_random_search_baseline),
        ("cma_es_over_x_example", _register_cma_es_example),
        ("tpe_mixed_x_y_example", _register_optuna_tpe_example),
        ("gp_bo_over_x_example", _register_gp_bo_example),
        ("mcts_codebook_example", _register_mcts_example),
        ("rashomon_committee_example", _register_rashomon_committee_example),
    ]
    contracts: list[SearchContract] = []
    for strategy_id, builder_fn in builders:
        if strategy_id in _REGISTERED_STRATEGIES:
            contracts.append(_REGISTERED_STRATEGIES[strategy_id])
        else:
            contracts.append(builder_fn())
    return contracts


# Eagerly register on import so the registry is populated when consumers
# import the examples module. Per CLAUDE.md "Beauty, simplicity, and
# developer experience": predictable side-effect at import time.
_BUILDER_CONTRACTS = register_example_searches()
