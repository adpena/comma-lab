# SPDX-License-Identifier: MIT
"""Typed exceptions for the tac.inflate_time_post_processing namespace.

Per CLAUDE.md "Beauty, simplicity, and developer experience" — narrow typed
exceptions raised at decoration / pipeline-build / runtime time so consumers
can distinguish failure classes structurally.

Sister of ``tac.boosting.BoostingNamespaceError`` and
``tac.compress_time_optimization.CompressTimeOptimizationError`` — the
canonical META-layer error model applied to a different stage surface
(inflate-time post-processing). Every exception is a subclass of
``InflateTimePostProcessingError`` so a single ``except`` clause catches
all namespace errors.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this namespace does NOT
re-export tac.boosting / tac.compress_time_optimization errors. Sister
namespaces are structurally independent; a consumer that imports multiple
must catch each base separately.
"""

from __future__ import annotations

__all__ = [
    "AmbiguousCompositionError",
    "ArchiveBytesViolation",
    "CompressPhaseForbiddenError",
    "InflateBudgetExceededError",
    "InflateTimeLedgerCorruptError",
    "InflateTimePassContractError",
    "InflateTimePipelineError",
    "InflateTimePostProcessingError",
    "ScorerAccessForbiddenError",
    "SeedRequiredViolation",
    "WallclockBudgetRequiredError",
]


class InflateTimePostProcessingError(Exception):
    """Root exception for the tac.inflate_time_post_processing namespace.

    All typed errors below inherit from this so callers can write::

        try:
            pipeline.run(decoded_frames)
        except InflateTimePostProcessingError as exc:
            ...

    and catch every namespace-level failure with a single clause.
    """


class InflateTimePassContractError(InflateTimePostProcessingError, ValueError):
    """Raised at decoration time when an ``InflateTimePostProcessingContract``
    is invalid.

    Sister of ``BoostStageContractError`` /
    ``CompressTimePassContractError`` — same role at the inflate-time
    post-processing surface. Always raised from
    ``InflateTimePostProcessingContract.__post_init__`` or from the
    ``@inflate_time_post_filter`` decorator wrapper.
    """


class CompressPhaseForbiddenError(InflateTimePassContractError):
    """Raised at decoration time when an ``InflateTimePostProcessingContract``
    declares ``stage_phase='compress'`` (or sister forbidden values).

    Per CLAUDE.md "Strict scorer rule" non-negotiable + the namespace's
    spec scope: this namespace is INFLATE-TIME ONLY. Compress-time stages
    belong in the sister namespace ``tac.compress_time_optimization``.

    The error message points the consumer to the sister namespace so they
    immediately know where the compress-time analog lives. Mirrors the
    inverse pattern in ``tac.compress_time_optimization`` which raises
    ``InflatePhaseForbiddenError`` for the OPPOSITE wrong-namespace case.
    """


class ScorerAccessForbiddenError(InflateTimePassContractError):
    """Raised at decoration time when an ``InflateTimePostProcessingContract``
    declares ``scorer_free=False`` (or sister forbidden states).

    Per CLAUDE.md "Strict scorer rule" non-negotiable: NO loading PoseNet
    or SegNet at inflate time. The contest scorer (~73 MB) NEVER ships in
    the archive; loading it at inflate would inflate the rate term and
    catastrophically degrade the score.

    A Hinton-distilled scorer SURROGATE (per Catalog #527 — CPU-trained,
    small, baked into inflate.py) IS legal and is the canonical pattern
    used by MultiPassInflateRefinement to rank multiple inflate variants
    deterministically. The surrogate is NOT the contest scorer; the
    distinction is structural: the surrogate's weights are PART of the
    archive bytes (charged to rate); the contest scorer's weights are NOT.
    """


class ArchiveBytesViolation(InflateTimePassContractError):
    """Raised at decoration time when an ``InflateTimePostProcessingContract``
    declares ``archive_bytes_added > 0``.

    Per spec §G inflate-time + the namespace's structural invariant:
    inflate-time post-processing operates on FRAMES (after the renderer
    decodes them) — it does NOT mutate archive BYTES. Adding archive bytes
    is by definition a COMPRESS-time technique and belongs in
    ``tac.compress_time_optimization``.

    A learned post-filter MODEL (Catalog #146-compliant, baked into
    inflate.py per CLAUDE.md "Operator gates must be wired and used") IS
    legal — its weights are part of the archive bytes via the COMPRESS-time
    archive grammar, NOT added at inflate time. The decorator does not
    enforce this distinction; the upstream archive grammar does.
    """


class SeedRequiredViolation(InflateTimePassContractError):
    """Raised at decoration time when the pass's contract declares
    ``deterministic=True`` but ``seed`` is None AND the function signature
    has no ``seed=`` parameter to derive reproducibility from.

    Mirrors ``tac.compress_time_optimization.SeedRequiredViolation`` at the
    inflate-time-pass surface. Per CLAUDE.md "Bit-level deconstruction"
    non-negotiable + Catalog #158 deterministic-compiler discipline: every
    inflate pass that emits frame bytes MUST produce byte-identical output
    for identical input. Stages that use randomness MUST accept a ``seed=``
    kwarg so the frame stream is reproducible across runs (and across the
    contest scorer's per-pair eval).
    """


class WallclockBudgetRequiredError(InflateTimePassContractError):
    """Raised at decoration time when an ``InflateTimePostProcessingContract``
    declares ``max_wallclock_seconds=None``.

    Per spec §G + CLAUDE.md "Beauty, simplicity, and developer experience":
    inflate-time post-processing has a 30-MINUTE HARD CEILING on T4 (the
    contest scorer's wall budget). Unlike compress-time (unbounded), every
    inflate-time pass MUST declare its expected wallclock so the pipeline
    composer can sum per-pass budgets and refuse compositions that would
    exceed the ceiling at build time, not at the contest scorer's deadline.

    The contract field is NON-OPTIONAL — passes must commit to a budget at
    decoration time. The pipeline's ``with_inflate_compute_budget(seconds=N)``
    filter caps the cumulative total at N <= 1800.
    """


class InflateBudgetExceededError(InflateTimePostProcessingError):
    """Raised at run time when a pass's elapsed wallclock would push the
    cumulative inflate-time budget above the pipeline's
    ``inflate_compute_budget_seconds`` (or the global 1800.0 ceiling).

    Sister of ``CompressTimeBudgetExceededError`` — same fail-loud pattern
    at the inflate-time surface. Includes elapsed seconds + offending stage
    id so the operator can audit the budget breach.

    Distinguished from ``WallclockBudgetRequiredError`` (decoration-time
    contract violation): this error fires at RUN time when the CUMULATIVE
    budget would be exceeded, even if each individual pass declared a
    legitimate per-pass wallclock.
    """


class InflateTimePipelineError(InflateTimePostProcessingError):
    """Raised at pipeline-build time when a composition is structurally
    invalid (cycle / missing producer / type mismatch / unknown pass id).

    Always raised from ``ComposableInflatePipeline.__or__`` /
    ``ComposableInflatePipeline.run`` / ``ComposableInflatePipeline.build``.
    """


class AmbiguousCompositionError(InflateTimePipelineError):
    """Raised when two passes in the pipeline emit the same key without
    explicit ordering / merge policy.

    The canonical case: ``A | B`` where both passes ``emit`` the key
    ``frames_v1``. Without an explicit merge policy (``&`` parallel-merge
    operator or a downstream pass that consumes both via different keys),
    the pipeline cannot decide which value to forward to subsequent passes.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — the ambiguity
    is surfaced at build time with a structured error message naming the
    conflicting passes, NOT silently resolved by pass-ordering accident.
    """


class InflateTimeLedgerCorruptError(InflateTimePostProcessingError):
    """Raised by ``persistence.load_pass_outcomes_strict`` when the JSONL
    ledger contains a malformed line OR is otherwise structurally invalid.

    Sister of ``CompressTimeLedgerCorruptError`` /
    ``BoostingLedgerCorruptError`` — same fail-closed pattern (Catalog
    #138). On corruption the caller quarantines the file via
    ``persistence._quarantine_corrupt_ledger`` rather than silently
    overwriting (which would drop the corrupt evidence).
    """
