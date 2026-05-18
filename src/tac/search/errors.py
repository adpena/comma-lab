# SPDX-License-Identifier: MIT
"""Typed exceptions for the tac.search namespace.

Per CLAUDE.md "Beauty, simplicity, and developer experience" — narrow typed
exceptions raised at decoration / pipeline-build / runtime time so consumers
can distinguish failure classes structurally.

Mirrors ``tac.boosting.BoostingNamespaceError`` /
``tac.compress_time_optimization.CompressTimeOptimizationError`` (sister
namespace patterns). Every exception is a subclass of
``SearchNamespaceError`` so a single ``except`` clause catches all
namespace errors.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": this namespace does NOT
re-export sister namespace errors. Sister namespaces are structurally
independent; a consumer that imports both must catch each base separately.

Catalog #229 / #294 / #303 / #305 design-memo provenance lives in
``.omx/research/tac_search_namespace_design_20260517.md``.
"""

from __future__ import annotations

__all__ = [
    "ObjectiveFunctionError",
    "SearchAmbiguousCompositionError",
    "SearchBudgetExceededError",
    "SearchContractError",
    "SearchEngineNotInstalledError",
    "SearchLedgerCorruptError",
    "SearchNamespaceError",
    "SearchPipelineError",
    "SearchStrategyNotRegisteredError",
    "DeterminismViolation",
    "SeedRequiredViolation",
]


class SearchNamespaceError(Exception):
    """Root exception for the tac.search namespace.

    All typed errors below inherit from this so callers can write::

        try:
            run_search_over_pipeline(pipeline, objective_fn)
        except SearchNamespaceError as exc:
            ...

    and catch every namespace-level failure with a single clause.
    """


class SearchContractError(SearchNamespaceError, ValueError):
    """Raised at decoration time when a SearchContract is invalid.

    Sister of ``CompressTimePassContractError`` / ``BoostStageContractError``
    — same role at the search-strategy surface. Always raised from
    ``SearchContract.__post_init__`` or the ``@search_strategy``
    decorator wrapper.
    """


class DeterminismViolation(SearchContractError):
    """Raised at decoration time when a strategy claims ``deterministic=True``
    but its function signature includes a randomness parameter without an
    accompanying ``seed=`` kwarg.

    Mirrors ``tac.compress_time_optimization.DeterminismViolation`` at the
    search-strategy surface.

    Per CLAUDE.md "Bit-level deconstruction" + Catalog #158 deterministic-
    compiler discipline + the canonical search-strategy contract: every
    strategy that claims deterministic=True MUST produce byte-identical
    trial sequences for identical input. Strategies that use randomness
    MUST accept a ``seed=`` kwarg so the trial sequence is reproducible.
    """


class SeedRequiredViolation(SearchContractError):
    """Raised at decoration time when the strategy's contract declares
    ``deterministic=True`` but ``seed`` is None AND the function signature
    has no ``seed=`` parameter to derive reproducibility from.

    Sister of DeterminismViolation: this version catches the OTHER half of
    the bug class — a deterministic strategy whose function does NOT accept
    a seed AND whose contract does NOT pin one. A deterministic strategy
    MUST EITHER pin seed on the contract OR accept seed in the signature.

    Mirrors ``tac.compress_time_optimization.SeedRequiredViolation``.
    """


class SearchStrategyNotRegisteredError(SearchNamespaceError, LookupError):
    """Raised when a caller looks up a strategy id that was never registered
    via ``@search_strategy``.

    Distinguished from ``SearchContractError`` so callers can react to
    "strategy unknown" (probably a typo / out-of-order import) separately
    from "contract invalid" (probably a code change to the decorated
    function's signature).
    """


class SearchEngineNotInstalledError(SearchNamespaceError, RuntimeError):
    """Raised at run time when a builder's external library dependency
    (``cma`` / ``optuna`` / ``botorch`` / ``scikit-optimize``) is not
    installed in the current environment.

    The error message includes the missing import name + a suggested
    install command (typically ``uv pip install <lib>``) so the operator
    can recover without consulting docs.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": library
    absence is detected at ``.run()`` time, NOT at module import / class
    construction / contract decoration — so the namespace is importable
    without ``cma`` / ``optuna`` / etc. (lazy import inside ``.run()``).
    """


class SearchPipelineError(SearchNamespaceError):
    """Raised at pipeline-build time when a ComposableSearchPipeline
    composition is structurally invalid (unknown strategy id / cycle /
    type mismatch).

    Always raised from ``ComposableSearchPipeline.__or__`` /
    ``ComposableSearchPipeline.run`` / ``ComposableSearchPipeline.build``.
    """


class SearchAmbiguousCompositionError(SearchPipelineError):
    """Raised when two strategies in the pipeline emit the same key without
    explicit ordering / merge policy.

    The canonical case: chaining two single-objective strategies via ``|``
    without explicitly declaring that the second consumes the first's
    output (e.g. warm-start). Without an explicit consumer the pipeline
    cannot decide which best_params dict to forward to subsequent
    strategies.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — the ambiguity
    is surfaced at build time with a structured error message naming the
    conflicting strategies, NOT silently resolved by accidental ordering.
    """


class SearchBudgetExceededError(SearchPipelineError):
    """Raised at run time when a search strategy exceeds its declared
    ``n_candidate_evaluations_max`` budget.

    The error includes evaluation count + the budget cap so the operator
    can audit the breach. Distinguished from
    ``CompressTimeBudgetExceededError`` because the budget is evaluation-
    count, not wallclock (per the design memo §4 UNIQUE-FORK).
    """


class ObjectiveFunctionError(SearchNamespaceError, RuntimeError):
    """Raised at run time when the objective function:

    - Returns a non-finite score (NaN / inf)
    - Raises an exception
    - Returns a value that is not float-coercible
    - Returns a shape inconsistent with multi_objective declaration

    The error chains the underlying exception (via ``raise ... from exc``)
    so the operator can diagnose the objective's own failure.
    """


class SearchLedgerCorruptError(SearchNamespaceError):
    """Raised by ``persistence.load_search_outcomes_strict`` when the JSONL
    ledger contains a malformed line OR is otherwise structurally invalid.

    Sister of ``BoostingLedgerCorruptError`` /
    ``CompressTimeLedgerCorruptError`` — same fail-closed pattern (Catalog
    #138). On corruption the caller quarantines the file via
    ``persistence._quarantine_corrupt_ledger`` rather than silently
    overwriting (which would drop the corrupt evidence).
    """
