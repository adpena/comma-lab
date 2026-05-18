# SPDX-License-Identifier: MIT
"""OptunaTPESampler — Tree-structured Parzen estimator wrapper.

Per `.omx/research/tac_search_namespace_design_20260517.md` §6 + §10:

  Optuna TPE handles mixed continuous + discrete parameter spaces
  gracefully with pruning support. The canonical Python library `optuna`
  (https://github.com/optuna/optuna) implements the sampler; this builder
  WRAPS it (UNIQUE-FORK because re-implementing TPE would be cargo-cult).

The external library is lazy-imported INSIDE the strategy's run-time
function so the namespace is importable without `optuna` installed.

Per CLAUDE.md "Bit-level deconstruction" + Catalog #158: deterministic
when seed is pinned (Optuna uses a NumPy random state seeded by the
provided seed; same seed → same trial sequence).
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
    "OptunaTPESampler",
    "OptunaTPESamplerSpec",
]


# Legal bounds spec: (low, high, kind) where kind ∈ {"int", "float",
# "log_float", "categorical", "bool"}
_LEGAL_BOUNDS_KINDS: frozenset[str] = frozenset(
    {"int", "float", "log_float", "categorical", "bool"}
)


@dataclass(frozen=True)
class OptunaTPESamplerSpec:
    """Specification for an Optuna TPE search strategy.

    bounds dict shape per parameter:
      - ("int", low, high): integer in [low, high]
      - ("float", low, high): float in [low, high]
      - ("log_float", low, high): log-uniform float in [low, high]
      - ("bool",): bool
      - ("categorical", [choice1, choice2, ...]): discrete choice
    """

    strategy_id: str
    bounds: Mapping[str, tuple[Any, ...]]
    n_trials: int = 100
    n_startup_trials: int = 10
    multivariate: bool = True
    seed: int = 42
    sensitivity_weighted: bool = False
    description: str = ""
    lane_id: str | None = None
    objective_is_surrogate: bool = False
    predicted_search_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, Mapping) or not self.bounds:
            raise ValueError(
                f"bounds={self.bounds!r} must be a non-empty mapping"
            )
        for name, b in self.bounds.items():
            if not isinstance(b, tuple) or not b:
                raise ValueError(
                    f"bounds[{name!r}]={b!r} must be a non-empty tuple"
                )
            kind = b[0]
            if kind not in _LEGAL_BOUNDS_KINDS:
                raise ValueError(
                    f"bounds[{name!r}][0]={kind!r} not in "
                    f"{sorted(_LEGAL_BOUNDS_KINDS)}"
                )
            if kind in ("int", "float", "log_float"):
                if len(b) != 3:
                    raise ValueError(
                        f"bounds[{name!r}]={b!r}: "
                        f"({kind!r}, low, high) required"
                    )
                low, high = b[1], b[2]
                if not all(isinstance(x, (int, float)) for x in (low, high)):
                    raise ValueError(
                        f"bounds[{name!r}]={b!r}: low/high must be numeric"
                    )
                if low >= high:
                    raise ValueError(
                        f"bounds[{name!r}]={b!r}: low must be < high"
                    )
            elif kind == "categorical":
                if len(b) != 2 or not isinstance(b[1], (list, tuple)) or not b[1]:
                    raise ValueError(
                        f"bounds[{name!r}]={b!r}: "
                        f"('categorical', [choice, ...]) required with "
                        "non-empty choices"
                    )
            elif kind == "bool":
                if len(b) != 1:
                    raise ValueError(
                        f"bounds[{name!r}]={b!r}: ('bool',) required (no "
                        "extra args)"
                    )
        if self.n_trials < 1:
            raise ValueError(f"n_trials={self.n_trials} must be >= 1")
        if self.n_startup_trials < 0:
            raise ValueError(
                f"n_startup_trials={self.n_startup_trials} must be >= 0"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class OptunaTPESampler:
    """Builder for an Optuna TPE search strategy contract + run-time function."""

    def __init__(self, *, spec: OptunaTPESamplerSpec) -> None:
        if not isinstance(spec, OptunaTPESamplerSpec):
            raise TypeError(
                f"spec must be OptunaTPESamplerSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SearchContract:
        rationale: dict[str, str] = {
            "hook_pareto_constraint": (
                "Single-objective TPE; Pareto undefined (use Optuna's "
                "NSGA-II sampler for multi-objective)."
            ),
            "hook_bit_allocator_class": (
                "Search strategies discover parameter values, not bit "
                "allocations."
            ),
            "hook_probe_disambiguator": (
                "TPE has a single canonical Bayesian-bandit interpretation."
            ),
        }
        if not self.spec.sensitivity_weighted:
            rationale["hook_sensitivity_contribution"] = (
                "TPE is sensitivity-blind by design; set "
                "sensitivity_weighted=True to add a master_gradient "
                "prior to the sampler."
            )
        return SearchContract(
            id=self.spec.strategy_id,
            parent_strategy_id=None,
            description=(
                self.spec.description
                or (
                    f"Optuna TPE wrapping the `optuna` library: "
                    f"n_trials={self.spec.n_trials}, "
                    f"startup={self.spec.n_startup_trials}, "
                    f"multivariate={self.spec.multivariate}, "
                    f"bounds={dict(self.spec.bounds)!r}."
                )
            ),
            search_kind="mixed",
            n_candidate_evaluations_max=self.spec.n_trials,
            parallelism="serial",  # Optuna native is serial; parallel via DB
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
                "UNIQUE-FORK: external library wrapper (optuna); TPE is "
                "canonical; the wrapper integrates it into the tac.search "
                "registry + observability surface."
            ),
        )

    def register(self) -> SearchContract:
        contract = self.build_contract()
        spec = self.spec

        @search_strategy(contract)
        def _optuna_tpe_run(
            objective_fn: Callable[[Mapping[str, Any]], float],
            *,
            bounds: Mapping[str, tuple[Any, ...]] | None = None,
            seed: int = spec.seed,
            warm_start: Mapping[str, Any] | None = None,
            **_ignored: Any,
        ) -> SearchResult:
            try:
                import optuna as _optuna  # type: ignore[import-not-found]
            except ImportError as exc:
                raise SearchEngineNotInstalledError(
                    "Optuna TPE strategy requires the `optuna` library. "
                    "Install via `uv pip install optuna` and retry. "
                    f"Original ImportError: {exc}"
                ) from exc
            return _run_optuna_tpe(
                optuna=_optuna,
                contract=contract,
                spec=spec,
                objective_fn=objective_fn,
                bounds=bounds or spec.bounds,
                seed=seed,
                warm_start=warm_start,
            )

        return contract


def _run_optuna_tpe(
    *,
    optuna: Any,
    contract: SearchContract,
    spec: OptunaTPESamplerSpec,
    objective_fn: Callable[[Mapping[str, Any]], float],
    bounds: Mapping[str, tuple[Any, ...]],
    seed: int,
    warm_start: Mapping[str, Any] | None,
) -> SearchResult:
    """Execute Optuna TPE; return SearchResult."""
    import time as _time

    # Suppress Optuna's INFO chatter + the multivariate ExperimentalWarning
    # so test + production logs stay readable. Operators who want the chatter
    # can re-raise the logger level after import.
    import logging
    import warnings
    logging.getLogger("optuna").setLevel(logging.WARNING)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*multivariate.*experimental.*",
            category=getattr(optuna.exceptions, "ExperimentalWarning", Warning),
        )
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=spec.n_startup_trials,
            multivariate=spec.multivariate,
        )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if warm_start:
        try:
            study.enqueue_trial(dict(warm_start))
        except Exception:  # noqa: BLE001 — defensive; warm_start optional
            pass

    trials: list[SearchTrial] = []

    def _suggest(trial: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, b in bounds.items():
            kind = b[0]
            if kind == "int":
                params[name] = trial.suggest_int(name, b[1], b[2])
            elif kind == "float":
                params[name] = trial.suggest_float(name, b[1], b[2])
            elif kind == "log_float":
                params[name] = trial.suggest_float(name, b[1], b[2], log=True)
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, list(b[1]))
            elif kind == "bool":
                params[name] = trial.suggest_categorical(name, [False, True])
        return params

    start = _time.monotonic()

    def _wrapped(trial: Any) -> float:
        params = _suggest(trial)
        t0 = _time.monotonic()
        try:
            score = objective_fn(params)
        except Exception as exc:
            raise ObjectiveFunctionError(
                f"objective_fn raised on trial {trial.number}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        elapsed_this = _time.monotonic() - t0
        try:
            score_f = float(score)
        except (TypeError, ValueError) as exc:
            raise ObjectiveFunctionError(
                f"objective_fn returned {score!r} on trial {trial.number}; "
                "expected a finite float."
            ) from exc
        if score_f != score_f or score_f in (float("inf"), float("-inf")):
            raise ObjectiveFunctionError(
                f"objective_fn returned non-finite {score_f} on trial "
                f"{trial.number}"
            )
        trials.append(
            SearchTrial(
                trial_index=trial.number,
                params=params,
                score=score_f,
                elapsed_seconds=elapsed_this,
            )
        )
        return score_f

    study.optimize(_wrapped, n_trials=spec.n_trials, show_progress_bar=False)

    elapsed = _time.monotonic() - start
    return SearchResult(
        strategy_id=spec.strategy_id,
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        n_evaluations=len(study.trials),
        elapsed_seconds=elapsed,
        history=SearchHistory(trials=tuple(trials)),
        objective_is_surrogate=contract.objective_is_surrogate,
        seed=seed,
    )
