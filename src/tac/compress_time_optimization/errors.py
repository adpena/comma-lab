# SPDX-License-Identifier: MIT
"""Typed exceptions for the tac.compress_time_optimization namespace.

Per CLAUDE.md "Beauty, simplicity, and developer experience" — narrow typed
exceptions raised at decoration / pipeline-build / runtime time so consumers
can distinguish failure classes structurally.

Sister of ``tac.boosting.BoostingNamespaceError`` (the canonical META-layer
error model for boost stages). Every exception is a subclass of
``CompressTimeOptimizationError`` so a single ``except`` clause catches all
namespace errors.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": this namespace does NOT
re-export tac.boosting's errors. Sister namespaces are structurally
independent; a consumer that imports both must catch each base separately.
"""

from __future__ import annotations

__all__ = [
    "AmbiguousCompositionError",
    "CompressTimeBudgetExceededError",
    "CompressTimeLedgerCorruptError",
    "CompressTimeOptimizationError",
    "CompressTimePassContractError",
    "CompressTimePipelineError",
    "DeterminismViolation",
    "InflatePhaseForbiddenError",
    "RateBudgetViolation",
    "SeedRequiredViolation",
]


class CompressTimeOptimizationError(Exception):
    """Root exception for the tac.compress_time_optimization namespace.

    All typed errors below inherit from this so callers can write::

        try:
            pipeline.run(seed_state)
        except CompressTimeOptimizationError as exc:
            ...

    and catch every namespace-level failure with a single clause.
    """


class CompressTimePassContractError(CompressTimeOptimizationError, ValueError):
    """Raised at decoration time when a CompressTimePassContract is invalid.

    Sister of ``BoostStageContractError`` — same role at the compress-time
    pass surface. Always raised from
    ``CompressTimePassContract.__post_init__`` or from the
    ``@compress_time_pass`` decorator wrapper.
    """


class DeterminismViolation(CompressTimePassContractError):
    """Raised at decoration time when a pass claims ``deterministic=True``
    but its function signature includes a randomness parameter without an
    accompanying ``seed=`` kwarg.

    Per CLAUDE.md "Bit-level deconstruction" non-negotiable + Catalog #158
    deterministic-compiler discipline: every pass that emits archive bytes
    MUST produce byte-identical output for identical input. Stages that use
    randomness MUST accept a ``seed=`` kwarg so the byte stream is
    reproducible across runs.

    Distinguished from SeedRequiredViolation: this error fires when the
    pass DECLARES randomness in its signature WITHOUT seed. The
    SeedRequiredViolation fires when the pass declares deterministic=True
    AND seed=None on the contract while the function signature has no
    seed param.
    """


class SeedRequiredViolation(CompressTimePassContractError):
    """Raised at decoration time when the pass's contract declares
    ``deterministic=True`` but ``seed`` is None AND the function signature
    has no ``seed=`` parameter to derive reproducibility from.

    Sister of DeterminismViolation: this version catches the OTHER half of
    the bug class — a deterministic stage whose function does NOT accept
    a seed AND whose contract does NOT pin one. A deterministic stage must
    EITHER pin seed on the contract OR accept seed in the signature.
    """


class InflatePhaseForbiddenError(CompressTimePassContractError):
    """Raised at decoration time when a CompressTimePassContract declares
    ``stage_phase='inflate'``.

    Per CLAUDE.md "Strict scorer rule" non-negotiable + the namespace's
    spec scope: this namespace is COMPRESS-TIME ONLY. Inflate-time
    post-processing is the sister namespace tac.inflate_time_post_processing
    (per spec §5.2 build queue, deferred to a future subagent slot).

    The error message points the consumer to the sister namespace so they
    immediately know where the inflate-time analog lives.
    """


class CompressTimePipelineError(CompressTimeOptimizationError):
    """Raised at pipeline-build time when a composition is structurally
    invalid (cycle / missing producer / type mismatch / unknown pass id).

    Always raised from ``ComposableCompressPipeline.__or__`` /
    ``ComposableCompressPipeline.run`` /
    ``ComposableCompressPipeline.build``.
    """


class AmbiguousCompositionError(CompressTimePipelineError):
    """Raised when two passes in the pipeline emit the same key without
    explicit ordering / merge policy.

    The canonical case: ``A | B`` where both passes ``emit`` the key
    ``archive_bytes_v1``. Without an explicit merge policy (``&``
    parallel-merge operator or a downstream pass that consumes both via
    different keys), the pipeline cannot decide which value to forward to
    subsequent passes.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — the ambiguity
    is surfaced at build time with a structured error message naming the
    conflicting passes, NOT silently resolved by pass-ordering accident.
    """


class RateBudgetViolation(CompressTimePipelineError):
    """Raised at run time when a ``with_rate_budget(bytes=N)`` filter
    rejects a stage's contribution that would exceed the cumulative
    byte budget.

    The error includes the cumulative byte count BEFORE the rejected
    stage so the operator can audit which stage was the breaking point.
    """


class CompressTimeBudgetExceededError(CompressTimePipelineError):
    """Raised at run time when a ``with_wallclock_budget(seconds=N)``
    filter rejects a stage that would exceed the cumulative wallclock
    budget (per Catalog #167 smoke-before-full discipline).

    The error includes elapsed seconds + the offending stage id so the
    operator can audit the budget breach.
    """


class CompressTimeLedgerCorruptError(CompressTimeOptimizationError):
    """Raised by ``persistence.load_pass_outcomes_strict`` when the JSONL
    ledger contains a malformed line OR is otherwise structurally invalid.

    Sister of ``BoostingLedgerCorruptError`` in ``tac.boosting.persistence``
    — same fail-closed pattern (Catalog #138). On corruption the caller
    quarantines the file via ``persistence._quarantine_corrupt_ledger``
    rather than silently overwriting (which would drop the corrupt
    evidence).
    """
