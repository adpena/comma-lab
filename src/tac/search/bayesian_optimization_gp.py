# SPDX-License-Identifier: MIT
"""BayesianOptimizationGP — Gaussian Process Bayesian Optimization wrapper.

Per `.omx/research/tac_search_namespace_design_20260517.md` §6 + §10:

  GP Bayesian Optimization is canonical for continuous parameter
  manifolds with EXPENSIVE objectives (≤ 100 trials). The GP kernel
  learns the objective's correlation structure; the acquisition function
  (default Expected Improvement) balances exploration vs exploitation.

This builder prefers `scikit-optimize` (skopt) because (a) it's pure
Python (BoTorch requires PyTorch, which is already a project dep but
adds startup latency); (b) skopt's API is more declarative; (c) the
sister `tac.preflight_rudin_daubechies` already uses skopt-style code
in `compressive_coverage_estimator`. If skopt is absent, the runner
falls back to `botorch` lazy import.

UNIQUE-FORK: external library wrapper. Re-implementing GP-BO would be
cargo-cult; skopt + botorch are production-grade.

Per CLAUDE.md "Bit-level deconstruction" + Catalog #158: deterministic
when seed is pinned (skopt + botorch both honor `random_state`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from tac.search.contract import SearchContract
from tac.search.decorator import search_strategy
from tac.search.errors import (
    ObjectiveFunctionError,
    SearchEngineNotInstalledError,
)
from tac.search.pipeline import SearchHistory, SearchResult, SearchTrial

__all__ = [
    "BayesianOptimizationGP",
    "BayesianOptimizationGPSpec",
]


_LEGAL_ACQUISITION_FNS: frozenset[str] = frozenset({"EI", "PI", "LCB"})


@dataclass(frozen=True)
class BayesianOptimizationGPSpec:
    """Specification for a GP Bayesian Optimization search strategy."""

    strategy_id: str
    bounds: Mapping[str, tuple[float, float]]
    n_initial_points: int = 10
    n_calls: int = 50
    acquisition_function: str = "EI"
    kernel: str = "matern52"
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
        if self.n_initial_points < 1:
            raise ValueError(
                f"n_initial_points={self.n_initial_points} must be >= 1"
            )
        if self.n_calls <= self.n_initial_points:
            raise ValueError(
                f"n_calls={self.n_calls} must be > n_initial_points="
                f"{self.n_initial_points}"
            )
        if self.acquisition_function not in _LEGAL_ACQUISITION_FNS:
            raise ValueError(
                f"acquisition_function={self.acquisition_function!r} not "
                f"in {sorted(_LEGAL_ACQUISITION_FNS)}"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class BayesianOptimizationGP:
    """Builder for a GP Bayesian Optimization search strategy."""

    def __init__(self, *, spec: BayesianOptimizationGPSpec) -> None:
        if not isinstance(spec, BayesianOptimizationGPSpec):
            raise TypeError(
                f"spec must be BayesianOptimizationGPSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SearchContract:
        rationale: dict[str, str] = {
            "hook_pareto_constraint": (
                "Single-objective GP-BO; Pareto undefined."
            ),
            "hook_bit_allocator_class": (
                "GP-BO discovers parameter values, not bit allocations."
            ),
            "hook_probe_disambiguator": (
                f"GP-BO with {self.spec.acquisition_function!r} acquisition "
                "is canonical; the kernel choice is the design surface."
            ),
        }
        if not self.spec.sensitivity_weighted:
            rationale["hook_sensitivity_contribution"] = (
                "GP-BO learns the correlation structure from observations; "
                "no external sensitivity prior. Set sensitivity_weighted=True "
                "to add a mean-function prior from master_gradient."
            )
        return SearchContract(
            id=self.spec.strategy_id,
            parent_strategy_id=None,
            description=(
                self.spec.description
                or (
                    f"GP Bayesian Optimization wrapping scikit-optimize / "
                    f"botorch: kernel={self.spec.kernel!r}, "
                    f"acq={self.spec.acquisition_function!r}, "
                    f"n_initial_points={self.spec.n_initial_points}, "
                    f"n_calls={self.spec.n_calls}, "
                    f"bounds={dict(self.spec.bounds)!r}."
                )
            ),
            search_kind="continuous",
            n_candidate_evaluations_max=self.spec.n_calls,
            parallelism="serial",
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
                "UNIQUE-FORK: external library wrapper (skopt preferred; "
                "botorch fallback). GP-BO is canonical; the wrapper "
                "integrates it into the tac.search registry."
            ),
        )

    def register(self) -> SearchContract:
        contract = self.build_contract()
        spec = self.spec

        @search_strategy(contract)
        def _gp_bo_run(
            objective_fn: Callable[[Mapping[str, Any]], float],
            *,
            bounds: Mapping[str, tuple[float, float]] | None = None,
            seed: int = spec.seed,
            warm_start: Mapping[str, Any] | None = None,
            **_ignored: Any,
        ) -> SearchResult:
            try:
                from skopt import gp_minimize as _gp_minimize  # type: ignore[import-not-found]
                from skopt.space import Real as _Real  # type: ignore[import-not-found]
            except ImportError:
                try:
                    import botorch  # type: ignore[import-not-found]  # noqa: F401
                except ImportError as exc:
                    raise SearchEngineNotInstalledError(
                        "BayesianOptimizationGP strategy requires either "
                        "`scikit-optimize` or `botorch`. Install via "
                        "`uv pip install scikit-optimize` and retry. "
                        f"Original ImportError: {exc}"
                    ) from exc
                # If skopt absent but botorch present, raise a non-engine
                # error directing the operator at skopt (BoTorch backend
                # is intentionally not implemented in this builder to keep
                # the wrapper narrow per design memo §6).
                raise SearchEngineNotInstalledError(
                    "BayesianOptimizationGP currently prefers "
                    "`scikit-optimize`; the botorch fallback is not yet "
                    "implemented. Install scikit-optimize via "
                    "`uv pip install scikit-optimize`."
                )

            return _run_gp_bo(
                gp_minimize=_gp_minimize,
                Real=_Real,
                contract=contract,
                spec=spec,
                objective_fn=objective_fn,
                bounds=bounds or spec.bounds,
                seed=seed,
                warm_start=warm_start,
            )

        return contract


def _run_gp_bo(
    *,
    gp_minimize: Any,
    Real: Any,
    contract: SearchContract,
    spec: BayesianOptimizationGPSpec,
    objective_fn: Callable[[Mapping[str, Any]], float],
    bounds: Mapping[str, tuple[float, float]],
    seed: int,
    warm_start: Mapping[str, Any] | None,
) -> SearchResult:
    """Execute GP-BO via skopt; return SearchResult."""
    import time as _time

    param_names = sorted(bounds.keys())
    space = [
        Real(float(bounds[n][0]), float(bounds[n][1]), name=n)
        for n in param_names
    ]
    trials: list[SearchTrial] = []
    state = {"trial_index": 0}

    def _objective(x_list):
        params = {n: float(x_list[i]) for i, n in enumerate(param_names)}
        t0 = _time.monotonic()
        try:
            score = objective_fn(params)
        except Exception as exc:
            raise ObjectiveFunctionError(
                f"objective_fn raised on GP-BO trial "
                f"{state['trial_index']}: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed_this = _time.monotonic() - t0
        try:
            score_f = float(score)
        except (TypeError, ValueError) as exc:
            raise ObjectiveFunctionError(
                f"objective_fn returned {score!r} on GP-BO trial "
                f"{state['trial_index']}; expected a finite float."
            ) from exc
        if score_f != score_f or score_f in (float("inf"), float("-inf")):
            raise ObjectiveFunctionError(
                f"objective_fn returned non-finite {score_f} on GP-BO "
                f"trial {state['trial_index']}"
            )
        trials.append(
            SearchTrial(
                trial_index=state["trial_index"],
                params=params,
                score=score_f,
                elapsed_seconds=elapsed_this,
            )
        )
        state["trial_index"] += 1
        return score_f

    x0 = None
    y0 = None
    if warm_start:
        try:
            x0 = [float(warm_start[n]) for n in param_names]
            y0 = float(_objective(x0))
        except (KeyError, TypeError, ValueError):
            x0 = None
            y0 = None

    acq_map = {"EI": "EI", "PI": "PI", "LCB": "LCB"}
    start = _time.monotonic()
    result = gp_minimize(
        _objective,
        space,
        n_calls=spec.n_calls,
        n_initial_points=spec.n_initial_points,
        acq_func=acq_map[spec.acquisition_function],
        random_state=seed,
        x0=x0,
        y0=y0,
    )
    elapsed = _time.monotonic() - start

    best_params = {n: float(result.x[i]) for i, n in enumerate(param_names)}
    return SearchResult(
        strategy_id=spec.strategy_id,
        best_params=best_params,
        best_score=float(result.fun),
        n_evaluations=len(trials),
        elapsed_seconds=elapsed,
        history=SearchHistory(trials=tuple(trials)),
        objective_is_surrogate=contract.objective_is_surrogate,
        seed=seed,
    )
