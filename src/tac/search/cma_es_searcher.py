# SPDX-License-Identifier: MIT
"""CMAESCandidateSearcher — covariance matrix adaptation evolutionary
strategy wrapper.

Per `.omx/research/tac_search_namespace_design_20260517.md` §6 + §10:

  CMA-ES is the SOTA for non-convex continuous optimization with
  population-based search. The canonical Python library `cma`
  (https://github.com/CMA-ES/pycma) implements the algorithm; this
  builder WRAPS it (canonical-vs-unique decision: UNIQUE-FORK because
  external library wrappers are by design — re-implementing CMA-ES would
  be cargo-cult per the operator's "consolidate everything into canonical
  helpers" directive).

The external library is lazy-imported INSIDE the strategy's run-time
function so the namespace is importable without `cma` installed. The
strategy's `.run()` raises ``SearchEngineNotInstalledError`` when `cma`
is absent, with a suggested install command.

Per CLAUDE.md "Bit-level deconstruction" + Catalog #158: the strategy
is deterministic when seed is pinned (CMA-ES uses a deterministic
sampler seeded by `numpy.random.seed(seed)` + the `cma.CMAEvolutionStrategy`
``seed`` option).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from tac.search.contract import SearchContract
from tac.search.decorator import search_strategy
from tac.search.errors import (
    ObjectiveFunctionError,
    SearchBudgetExceededError,
    SearchEngineNotInstalledError,
)
from tac.search.pipeline import SearchHistory, SearchResult, SearchTrial

__all__ = [
    "CMAESCandidateSearcher",
    "CMAESCandidateSearcherSpec",
]


@dataclass(frozen=True)
class CMAESCandidateSearcherSpec:
    """Specification for a CMA-ES search strategy.

    Frozen so spec composition is structurally immutable. The CMA-ES
    parameters (population_size, sigma_init, max_evaluations) are pinned
    at decoration time for byte-stable reproducibility per Catalog #158.
    """

    strategy_id: str
    bounds: Mapping[str, tuple[float, float]]
    population_size: int = 12
    sigma_init: float = 1.0
    max_evaluations: int = 200
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
                "{param_name: (low, high)}"
            )
        for name, b in self.bounds.items():
            if (
                not isinstance(b, tuple)
                or len(b) != 2
                or not all(isinstance(x, (int, float)) for x in b)
                or b[0] >= b[1]
            ):
                raise ValueError(
                    f"bounds[{name!r}]={b!r} must be (low: float, "
                    "high: float) with low < high"
                )
        if self.population_size < 1:
            raise ValueError(
                f"population_size={self.population_size} must be >= 1"
            )
        if self.sigma_init <= 0:
            raise ValueError(f"sigma_init={self.sigma_init} must be > 0")
        if self.max_evaluations < 1:
            raise ValueError(
                f"max_evaluations={self.max_evaluations} must be >= 1"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class CMAESCandidateSearcher:
    """Builder for a CMA-ES search strategy contract + run-time function.

    Usage::

        from tac.search import (
            CMAESCandidateSearcher, CMAESCandidateSearcherSpec,
        )

        searcher = CMAESCandidateSearcher(
            spec=CMAESCandidateSearcherSpec(
                strategy_id="cma_es_over_fec6_k_palette",
                bounds={"K": (4.0, 64.0)},
                population_size=12,
                sigma_init=8.0,
                max_evaluations=200,
                seed=42,
                lane_id="lane_my_substrate_20260601",
            )
        )
        searcher.register()  # registers via @search_strategy(contract)

    The `register()` method creates the contract + binds the run-time
    function into the namespace registry. After registration the strategy
    is invocable via ``run_search_over_pipeline`` or directly via
    ``get_strategy_function(strategy_id)``.
    """

    def __init__(self, *, spec: CMAESCandidateSearcherSpec) -> None:
        if not isinstance(spec, CMAESCandidateSearcherSpec):
            raise TypeError(
                f"spec must be CMAESCandidateSearcherSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SearchContract:
        """Build the SearchContract for this CMA-ES strategy."""
        rationale: dict[str, str] = {
            "hook_pareto_constraint": (
                "Single-objective minimization; Pareto undefined."
            ),
            "hook_bit_allocator_class": (
                "Search strategies discover parameter values, not bit "
                "allocations."
            ),
            "hook_probe_disambiguator": (
                "CMA-ES has a single canonical population-based interpretation."
            ),
        }
        if not self.spec.sensitivity_weighted:
            rationale["hook_sensitivity_contribution"] = (
                "CMA-ES is sensitivity-blind by design (no master_gradient "
                "consumption); set sensitivity_weighted=True to opt in."
            )
        return SearchContract(
            id=self.spec.strategy_id,
            parent_strategy_id=None,
            description=(
                self.spec.description
                or (
                    f"CMA-ES wrapping the `cma` library: "
                    f"pop={self.spec.population_size}, "
                    f"sigma_init={self.spec.sigma_init}, "
                    f"max_evals={self.spec.max_evaluations}, "
                    f"bounds={dict(self.spec.bounds)!r}."
                )
            ),
            search_kind="continuous",
            n_candidate_evaluations_max=self.spec.max_evaluations,
            parallelism="vectorized",
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
                "UNIQUE-FORK: external library wrapper (cma); the CMA-ES "
                "algorithm is canonical; the wrapper integrates it into "
                "the tac.search registry + observability surface."
            ),
        )

    def register(self) -> SearchContract:
        """Register the strategy + run-time function into the registry."""
        contract = self.build_contract()
        spec = self.spec

        @search_strategy(contract)
        def _cma_es_run(
            objective_fn: Callable[[Mapping[str, Any]], float],
            *,
            bounds: Mapping[str, tuple[float, float]] | None = None,
            seed: int = spec.seed,
            warm_start: Mapping[str, Any] | None = None,
            **_ignored: Any,
        ) -> SearchResult:
            """Run CMA-ES against `objective_fn` via the lazy-imported `cma`."""
            try:
                import cma as _cma_module  # type: ignore[import-not-found]
            except ImportError as exc:
                raise SearchEngineNotInstalledError(
                    "CMA-ES strategy requires the `cma` library "
                    "(https://github.com/CMA-ES/pycma). Install via "
                    "`uv pip install cma` and retry. Original ImportError: "
                    f"{exc}"
                ) from exc

            return _run_cma_es(
                cma_module=_cma_module,
                contract=contract,
                spec=spec,
                objective_fn=objective_fn,
                bounds=bounds or spec.bounds,
                seed=seed,
                warm_start=warm_start,
            )

        return contract


# ---------------------------------------------------------------------------
# Internal runner (kept module-level for testability)
# ---------------------------------------------------------------------------


def _run_cma_es(
    *,
    cma_module: Any,
    contract: SearchContract,
    spec: CMAESCandidateSearcherSpec,
    objective_fn: Callable[[Mapping[str, Any]], float],
    bounds: Mapping[str, tuple[float, float]],
    seed: int,
    warm_start: Mapping[str, Any] | None,
) -> SearchResult:
    """Execute CMA-ES; return SearchResult.

    Bounds are extracted to a sorted-name list so the CMA-ES x-vector
    ordering is deterministic across runs. Each candidate evaluation
    converts the x-vector back to a {name: value} dict before calling
    objective_fn.
    """
    import time as _time

    param_names = sorted(bounds.keys())
    lower = [float(bounds[n][0]) for n in param_names]
    upper = [float(bounds[n][1]) for n in param_names]
    centroid = (
        [float(warm_start[n]) for n in param_names]
        if warm_start
        and all(isinstance(warm_start.get(n), (int, float)) for n in param_names)
        else [(lo + hi) / 2.0 for lo, hi in zip(lower, upper)]
    )

    options = {
        "bounds": [lower, upper],
        "popsize": spec.population_size,
        "seed": seed + 1,  # cma seed must be > 0
        "maxfevals": spec.max_evaluations,
        "verbose": -9,  # silent
    }
    es = cma_module.CMAEvolutionStrategy(centroid, spec.sigma_init, options)

    trials: list[SearchTrial] = []
    best_x: list[float] = centroid
    best_score = float("inf")
    n_evaluations = 0
    start = _time.monotonic()

    while not es.stop() and n_evaluations < spec.max_evaluations:
        solutions = es.ask()
        scores: list[float] = []
        for x in solutions:
            if n_evaluations >= spec.max_evaluations:
                break
            params = {n: float(x[i]) for i, n in enumerate(param_names)}
            t0 = _time.monotonic()
            try:
                score_val = objective_fn(params)
            except Exception as exc:
                raise ObjectiveFunctionError(
                    f"objective_fn raised on trial {n_evaluations}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            elapsed_this = _time.monotonic() - t0
            try:
                score_f = float(score_val)
            except (TypeError, ValueError) as exc:
                raise ObjectiveFunctionError(
                    f"objective_fn returned {score_val!r} on trial "
                    f"{n_evaluations}; expected a finite float."
                ) from exc
            if score_f != score_f or score_f in (float("inf"), float("-inf")):
                raise ObjectiveFunctionError(
                    f"objective_fn returned non-finite {score_f} on trial "
                    f"{n_evaluations}"
                )
            trials.append(
                SearchTrial(
                    trial_index=n_evaluations,
                    params=params,
                    score=score_f,
                    elapsed_seconds=elapsed_this,
                )
            )
            scores.append(score_f)
            if score_f < best_score:
                best_score = score_f
                best_x = list(x)
            n_evaluations += 1
        if not scores:
            break
        es.tell(solutions[: len(scores)], scores)

    if n_evaluations > spec.max_evaluations:
        raise SearchBudgetExceededError(
            f"CMA-ES strategy {spec.strategy_id!r} exceeded "
            f"max_evaluations={spec.max_evaluations} with "
            f"n_evaluations={n_evaluations}"
        )

    elapsed = _time.monotonic() - start
    best_params = {n: float(best_x[i]) for i, n in enumerate(param_names)}
    return SearchResult(
        strategy_id=spec.strategy_id,
        best_params=best_params,
        best_score=best_score,
        n_evaluations=n_evaluations,
        elapsed_seconds=elapsed,
        history=SearchHistory(trials=tuple(trials)),
        objective_is_surrogate=contract.objective_is_surrogate,
        seed=seed,
    )
