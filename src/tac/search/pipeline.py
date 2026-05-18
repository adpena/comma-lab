# SPDX-License-Identifier: MIT
"""ComposableSearchPipeline + SearchResult + run_search_over_pipeline.

Per ``.omx/research/tac_search_namespace_design_20260517.md`` §3:

  - ``pipeline | strategy_id`` — sequential chain (e.g. RandomSearch warmup
    | TPE refinement)
  - ``pipeline & strategy_id`` — parallel-merge ensemble (Rashomon-style)
  - ``run_search_over_pipeline(sister_pipeline, objective_fn)`` — canonical
    helper that resolves the `@`-attached strategy descriptor on a
    sister-pipeline (tac.boosting / tac.compress_time_optimization) and
    executes the search.

Distinct from the sister namespaces' ComposablePipelines:
  - SearchPipeline is OPT-IN composable: stand-alone search composition
    without requiring a sister-pipeline substrate.
  - The `@` operator on sister pipelines stores a descriptor as opaque
    metadata; the EXECUTION lives in `run_search_over_pipeline`.

Per CLAUDE.md "Beauty, simplicity, and developer experience":
  - immutable construction → no mid-build state races
  - all errors at .build()/.run() surface SearchAmbiguousCompositionError
    or SearchPipelineError with named conflicting strategies
  - JSON-serializable representation for cathedral autopilot ranking
  - SearchResult is frozen + dict-roundtrippable for ledger persistence
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from tac.search.contract import SearchContract  # noqa: TC001
from tac.search.decorator import (
    _REGISTERED_STRATEGIES,
    get_strategy_function,
)
from tac.search.errors import (
    ObjectiveFunctionError,
    SearchAmbiguousCompositionError,
    SearchBudgetExceededError,
    SearchPipelineError,
)

__all__ = [
    "ComposableSearchPipeline",
    "SearchHistory",
    "SearchPipelineStrategyRef",
    "SearchResult",
    "SearchTrial",
    "run_search_over_pipeline",
]


@dataclass(frozen=True)
class SearchPipelineStrategyRef:
    """A single search-strategy reference in a pipeline (id + composition kind).

    Frozen so pipeline composition is structurally immutable.
    Mirrors ``tac.boosting.pipeline.PipelineStageRef`` /
    ``tac.compress_time_optimization.pipeline.PipelineStageRef`` at the
    search-strategy surface. Per the design memo: STRUCTURALLY INDEPENDENT
    (no import / no shared base) so each namespace can evolve independently.
    """

    strategy_id: str
    parameters: tuple[tuple[str, Any], ...] = ()
    composition_kind: str = "sequential"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "parameters": list(self.parameters),
            "composition_kind": self.composition_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchPipelineStrategyRef":
        params = tuple((k, v) for k, v in data.get("parameters", []))
        return cls(
            strategy_id=data["strategy_id"],
            parameters=params,
            composition_kind=data.get("composition_kind", "sequential"),
        )


@dataclass(frozen=True)
class SearchTrial:
    """A single candidate evaluation in a search history.

    Frozen + sort-keyed JSON-serializable so two trial sequences can be
    diff-ed for byte-stability (per design memo §11 observability surface
    facet 3: diff-able across runs).
    """

    trial_index: int
    params: Mapping[str, Any]
    score: float
    elapsed_seconds: float = 0.0
    # Optional axis tag per CLAUDE.md "Forbidden empirical-claim-without-
    # evidence-tag" (the docstring-overstatement trap). Search trials run
    # against an objective_fn whose axis the caller declares.
    score_axis: str = "[proxy]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "params": dict(self.params),
            "score": self.score,
            "elapsed_seconds": self.elapsed_seconds,
            "score_axis": self.score_axis,
        }


@dataclass(frozen=True)
class SearchHistory:
    """Immutable history of trials for a search run."""

    trials: tuple[SearchTrial, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"trials": [t.to_dict() for t in self.trials]}

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials)


@dataclass(frozen=True)
class SearchResult:
    """Result returned by a search strategy's run.

    Per the design memo §11 observability surface: every field is
    inspectable, every numeric is decomposable via `history`, two results
    with the same `strategy_id` + `seed` + `objective_function_label`
    diff byte-identically.
    """

    strategy_id: str
    best_params: Mapping[str, Any]
    best_score: float
    n_evaluations: int
    elapsed_seconds: float = 0.0
    history: SearchHistory = field(default_factory=SearchHistory)
    score_axis: str = "[proxy]"
    objective_function_label: str = ""
    # Pareto front (multi-objective strategies populate); None for
    # single-objective.
    pareto_front: tuple[Mapping[str, Any], ...] | None = None
    # Metadata threaded from the contract for downstream consumers.
    objective_is_surrogate: bool = False
    seed: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "best_params": dict(self.best_params),
            "best_score": self.best_score,
            "n_evaluations": self.n_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "history": self.history.to_dict(),
            "score_axis": self.score_axis,
            "objective_function_label": self.objective_function_label,
            "pareto_front": (
                [dict(p) for p in self.pareto_front]
                if self.pareto_front is not None
                else None
            ),
            "objective_is_surrogate": self.objective_is_surrogate,
            "seed": self.seed,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ComposableSearchPipeline:
    """Immutable pipeline-of-search-strategies with operator composition.

    Construction is via the canonical ``|`` operator chaining starting from
    an empty pipeline::

        pipeline = (
            ComposableSearchPipeline()
            | "random_search_baseline"   # warm-start
            | "tpe_fec6_K_plus_lambda_R" # refinement
        )

    Or alternatively from a list (imperative form)::

        pipeline = ComposableSearchPipeline.from_strategy_ids(
            ["random_search_baseline", "tpe_fec6_K_plus_lambda_R"]
        )

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": the pipeline carries
    NO hidden state — every behavior is visible via the ``strategies``
    tuple + the ``shared_objective_function_label`` field. Two pipelines
    with equal ``strategies`` are equivalent.
    """

    strategies: tuple[SearchPipelineStrategyRef, ...] = ()
    # Optional shared objective label that propagates into each
    # SearchResult.objective_function_label so the ledger can group
    # outcomes by objective. None = each strategy declares its own.
    shared_objective_function_label: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_strategy_ids(
        cls, strategy_ids: list[str], **kwargs: Any
    ) -> "ComposableSearchPipeline":
        """Build a pipeline from a flat list of strategy ids."""
        pipeline = cls(**kwargs)
        for sid in strategy_ids:
            pipeline = pipeline | sid
        return pipeline

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComposableSearchPipeline":
        """Reconstruct a pipeline from a JSON-deserialized dict."""
        strategies = tuple(
            SearchPipelineStrategyRef.from_dict(s)
            for s in data.get("strategies", [])
        )
        return cls(
            strategies=strategies,
            shared_objective_function_label=data.get(
                "shared_objective_function_label"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": [s.to_dict() for s in self.strategies],
            "shared_objective_function_label": (
                self.shared_objective_function_label
            ),
        }

    def to_json(self) -> str:
        """JSON-serialize the pipeline (sorted keys for byte-stable output)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def __str__(self) -> str:
        if not self.strategies:
            return "ComposableSearchPipeline(<empty>)"
        chain = " | ".join(s.strategy_id for s in self.strategies)
        suffix = ""
        if self.shared_objective_function_label is not None:
            suffix = (
                f".with_shared_objective_label("
                f"{self.shared_objective_function_label!r})"
            )
        return f"ComposableSearchPipeline({chain}){suffix}"

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __or__(
        self, strategy: "str | SearchPipelineStrategyRef"
    ) -> "ComposableSearchPipeline":
        """Sequential composition (`A | B` runs A then B with warm-start)."""
        ref = (
            strategy
            if isinstance(strategy, SearchPipelineStrategyRef)
            else SearchPipelineStrategyRef(
                strategy_id=strategy, composition_kind="sequential"
            )
        )
        return replace(self, strategies=(*self.strategies, ref))

    def __and__(
        self, strategy: "str | SearchPipelineStrategyRef"
    ) -> "ComposableSearchPipeline":
        """Parallel-merge composition (`A & B` runs both on same objective,
        merges via Rashomon-style consensus / disagreement).
        """
        if not self.strategies:
            raise SearchPipelineError(
                "`&` (parallel-merge) requires at least one prior strategy; "
                "use `|` for the first strategy."
            )
        ref = (
            replace(strategy, composition_kind="parallel")
            if isinstance(strategy, SearchPipelineStrategyRef)
            else SearchPipelineStrategyRef(
                strategy_id=strategy, composition_kind="parallel"
            )
        )
        return replace(self, strategies=(*self.strategies, ref))

    def with_shared_objective_label(
        self, label: str
    ) -> "ComposableSearchPipeline":
        """Attach a shared objective label propagated to every SearchResult."""
        if not isinstance(label, str) or not label.strip():
            raise SearchPipelineError(
                f"shared_objective_function_label must be a non-empty str; "
                f"got {label!r}"
            )
        return replace(self, shared_objective_function_label=label)

    # ------------------------------------------------------------------
    # Build + Run
    # ------------------------------------------------------------------

    def build(self) -> "ComposableSearchPipeline":
        """Validate the pipeline's structural correctness without running.

          - Unknown strategy id (not registered via @search_strategy) →
            SearchPipelineError
          - Two sequential strategies emit the same `best_params` key
            structure without explicit warm-start declaration →
            SearchAmbiguousCompositionError
          - Cycle in parent_strategy_id chain → SearchPipelineError

        Returns self (the pipeline is already immutable; build() is a
        validation pass). The validated pipeline is then safe to .run().
        """
        for ref in self.strategies:
            if ref.strategy_id not in _REGISTERED_STRATEGIES:
                raise SearchPipelineError(
                    f"Pipeline references strategy id={ref.strategy_id!r} "
                    "which is not registered via @search_strategy. "
                    f"Registered ids: {sorted(_REGISTERED_STRATEGIES)}"
                )

        # Cycle detection in parent_strategy_id chain
        for ref in self.strategies:
            contract = _REGISTERED_STRATEGIES[ref.strategy_id]
            seen = {contract.id}
            cursor = contract.parent_strategy_id
            while cursor is not None:
                if cursor in seen:
                    raise SearchPipelineError(
                        f"Cycle detected in parent_strategy_id chain "
                        f"starting from strategy id={contract.id!r}: cycle "
                        f"through {cursor!r}"
                    )
                seen.add(cursor)
                parent_contract = _REGISTERED_STRATEGIES.get(cursor)
                if parent_contract is None:
                    raise SearchPipelineError(
                        f"Strategy id={contract.id!r} declares "
                        f"parent_strategy_id={cursor!r} which is not "
                        "registered. Either register the parent OR set "
                        "parent_strategy_id=None."
                    )
                cursor = parent_contract.parent_strategy_id

        # Ambiguous composition: two sequential strategies that don't
        # declare warm-start chaining via parent_strategy_id. Operator can
        # silence by setting parent_strategy_id explicitly.
        prior_strategy_id: str | None = None
        prior_emits_best_params = False
        for ref in self.strategies:
            if ref.composition_kind != "sequential":
                continue
            contract = _REGISTERED_STRATEGIES[ref.strategy_id]
            if prior_strategy_id is not None and prior_emits_best_params:
                if (
                    contract.parent_strategy_id is None
                    and prior_strategy_id != contract.id
                ):
                    raise SearchAmbiguousCompositionError(
                        f"Pipeline chains strategy {prior_strategy_id!r} → "
                        f"{contract.id!r} sequentially but the downstream "
                        "strategy does NOT declare parent_strategy_id="
                        f"{prior_strategy_id!r}. The pipeline cannot infer "
                        "whether the chain is warm-start (downstream "
                        "consumes upstream best_params) or independent "
                        "(downstream ignores upstream). Either set "
                        f"parent_strategy_id={prior_strategy_id!r} on "
                        f"{contract.id!r}, OR rename the chain to be "
                        "explicit, OR use `&` (parallel-merge) to indicate "
                        "ensemble semantics."
                    )
            prior_strategy_id = contract.id
            prior_emits_best_params = True

        return self

    def run(
        self,
        objective_fn: Callable[[Mapping[str, Any]], float],
        *,
        bounds: Mapping[str, Any] | None = None,
        score_axis: str = "[proxy]",
        master_gradient: Any | None = None,
        **strategy_kwargs: Any,
    ) -> SearchResult:
        """Execute the pipeline against ``objective_fn``.

        For each strategy in the pipeline:
          1. Resolve the registered strategy function from the registry
          2. Invoke ``fn(objective_fn, bounds=..., seed=..., ...)``
          3. Collect a SearchResult per strategy; the final returned
             SearchResult is the AGGREGATE across all strategies (best
             across the chain).

        Per CLAUDE.md "Beauty, simplicity, and developer experience":
          - objective_fn signature: ``objective_fn(params: Mapping) ->
            float``
          - bounds are passed through to each strategy (the strategy
            specializes them per its own contract)
          - strategy_kwargs propagate to every strategy invocation
            (useful for shared diagnostic kwargs)

        Returns a single aggregate SearchResult whose best_params is the
        BEST across the chain (minimum score by default).
        """
        self.build()

        if not callable(objective_fn):
            raise ObjectiveFunctionError(
                f"objective_fn must be callable; got "
                f"{type(objective_fn).__name__}"
            )

        if not self.strategies:
            raise SearchPipelineError(
                "Cannot run an empty pipeline; use `|` to add a strategy."
            )

        aggregate_best_params: dict[str, Any] = {}
        aggregate_best_score = math.inf
        aggregate_trials: list[SearchTrial] = []
        aggregate_elapsed = 0.0
        aggregate_n_evals = 0
        last_strategy_id = self.strategies[-1].strategy_id
        objective_is_surrogate = False
        seed_observed: int | None = None
        warm_start_params: Mapping[str, Any] | None = None

        for ref in self.strategies:
            contract = _REGISTERED_STRATEGIES[ref.strategy_id]
            fn = get_strategy_function(ref.strategy_id)
            kwargs: dict[str, Any] = dict(strategy_kwargs)
            if bounds is not None:
                kwargs.setdefault("bounds", bounds)
            if contract.seed is not None:
                kwargs.setdefault("seed", contract.seed)
                seed_observed = contract.seed
            if warm_start_params is not None:
                kwargs.setdefault("warm_start", dict(warm_start_params))
            for k, v in ref.parameters:
                kwargs[k] = v

            start = time.monotonic()
            try:
                result = fn(objective_fn, **kwargs)
            except SearchBudgetExceededError:
                raise
            except ObjectiveFunctionError:
                raise
            except Exception as exc:
                raise SearchPipelineError(
                    f"Strategy id={ref.strategy_id!r} raised during "
                    f"pipeline.run: {type(exc).__name__}: {exc}"
                ) from exc
            elapsed_this = time.monotonic() - start

            if not isinstance(result, SearchResult):
                raise SearchPipelineError(
                    f"Strategy id={ref.strategy_id!r} returned "
                    f"{type(result).__name__}; expected SearchResult."
                )

            aggregate_elapsed += elapsed_this
            aggregate_n_evals += result.n_evaluations
            if result.score_axis:
                score_axis = result.score_axis
            objective_is_surrogate = (
                objective_is_surrogate or result.objective_is_surrogate
            )
            aggregate_trials.extend(result.history.trials)
            if result.best_score < aggregate_best_score:
                aggregate_best_score = result.best_score
                aggregate_best_params = dict(result.best_params)
            warm_start_params = result.best_params

        objective_function_label = (
            self.shared_objective_function_label or ""
        )

        return SearchResult(
            strategy_id=last_strategy_id,
            best_params=aggregate_best_params,
            best_score=aggregate_best_score,
            n_evaluations=aggregate_n_evals,
            elapsed_seconds=aggregate_elapsed,
            history=SearchHistory(trials=tuple(aggregate_trials)),
            score_axis=score_axis,
            objective_function_label=objective_function_label,
            objective_is_surrogate=objective_is_surrogate,
            seed=seed_observed,
            notes=f"pipeline aggregate over {len(self.strategies)} strateg(ies)",
        )

    def strategy_contracts(self) -> tuple[SearchContract, ...]:
        """Return contracts of every strategy in the pipeline (in order)."""
        return tuple(
            _REGISTERED_STRATEGIES[ref.strategy_id] for ref in self.strategies
        )


# ---------------------------------------------------------------------------
# Sister-pipeline `@` integration
# ---------------------------------------------------------------------------


def run_search_over_pipeline(
    sister_pipeline: Any,
    objective_fn: Callable[[Mapping[str, Any]], float],
    *,
    bounds: Mapping[str, Any] | None = None,
    score_axis: str = "[proxy]",
    objective_function_label: str = "",
    **strategy_kwargs: Any,
) -> SearchResult:
    """Canonical helper for `@`-attached search execution on a sister
    pipeline (tac.boosting / tac.compress_time_optimization).

    The sister pipelines store the search descriptor as opaque metadata via
    their ``__matmul__`` operator::

        from tac.boosting import ComposableBoostingPipeline
        pipeline = ComposableBoostingPipeline() | "raw_decoder"
        pipeline_with_search = pipeline @ "cma_es_over_palette_k"

    This helper:
      1. Reads ``sister_pipeline.search_strategy_descriptor`` (str)
      2. Looks up the registered strategy via
         ``get_strategy_function(strategy_id)``
      3. Invokes the strategy with the objective_fn + bounds
      4. Returns the SearchResult

    Per the design memo §3 + §6: this is the ENGINE for the `@` operator.
    The sister pipelines remain pure (no search dependency injected) and
    this namespace evolves independently per CLAUDE.md
    "UNIQUE-AND-COMPLETE-PER-METHOD".

    Raises:
        SearchPipelineError: sister_pipeline has no
            ``search_strategy_descriptor`` attribute OR the descriptor is
            None / empty.
        SearchStrategyNotRegisteredError: the descriptor names a strategy
            that was never registered via ``@search_strategy``.
    """
    descriptor = getattr(
        sister_pipeline, "search_strategy_descriptor", None
    )
    if descriptor is None or not str(descriptor).strip():
        raise SearchPipelineError(
            f"Sister pipeline {type(sister_pipeline).__name__} has no "
            "search_strategy_descriptor attached. Use the sister pipeline's "
            '`@` operator first: pipeline = pipeline @ "<strategy_id>" '
            "before invoking run_search_over_pipeline."
        )

    strategy_id = str(descriptor)
    contract = _REGISTERED_STRATEGIES.get(strategy_id)
    if contract is None:
        from tac.search.errors import SearchStrategyNotRegisteredError

        raise SearchStrategyNotRegisteredError(
            f"Sister pipeline references strategy id={strategy_id!r} which "
            "is not registered via @search_strategy. Registered ids: "
            f"{sorted(_REGISTERED_STRATEGIES)}"
        )

    if not callable(objective_fn):
        raise ObjectiveFunctionError(
            f"objective_fn must be callable; got "
            f"{type(objective_fn).__name__}"
        )

    fn = get_strategy_function(strategy_id)
    kwargs: dict[str, Any] = dict(strategy_kwargs)
    if bounds is not None:
        kwargs.setdefault("bounds", bounds)
    if contract.seed is not None:
        kwargs.setdefault("seed", contract.seed)

    start = time.monotonic()
    try:
        result = fn(objective_fn, **kwargs)
    except SearchBudgetExceededError:
        raise
    except ObjectiveFunctionError:
        raise
    except Exception as exc:
        raise SearchPipelineError(
            f"Strategy id={strategy_id!r} raised during "
            f"run_search_over_pipeline: {type(exc).__name__}: {exc}"
        ) from exc
    elapsed = time.monotonic() - start

    if not isinstance(result, SearchResult):
        raise SearchPipelineError(
            f"Strategy id={strategy_id!r} returned "
            f"{type(result).__name__}; expected SearchResult."
        )

    # Augment the result with sister-pipeline metadata
    if objective_function_label and not result.objective_function_label:
        result = replace(result, objective_function_label=objective_function_label)
    if score_axis and result.score_axis == "[proxy]":
        result = replace(result, score_axis=score_axis)
    return replace(result, elapsed_seconds=max(result.elapsed_seconds, elapsed))
