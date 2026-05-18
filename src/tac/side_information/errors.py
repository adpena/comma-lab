# SPDX-License-Identifier: MIT
"""Typed exceptions for the tac.side_information namespace.

Per CLAUDE.md "Beauty, simplicity, and developer experience" — narrow typed
exceptions raised at decoration / pipeline-build / runtime time so consumers
can distinguish failure classes structurally.

Sister of ``tac.boosting.BoostingNamespaceError`` and
``tac.compress_time_optimization.CompressTimeOptimizationError`` — the
canonical META-layer error model for side-information bakers. Every
exception is a subclass of ``SideInformationError`` so a single ``except``
clause catches all namespace errors.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" (Catalog #290 + standing
directive 2026-05-15): this namespace does NOT re-export sister errors.
Sister namespaces are structurally independent; a consumer that imports
two namespaces catches each base separately.

The naming centerpiece is ``NonReproducibleSideInfoViolation`` — every
piece of side information used by the contest decoder MUST be derivable
from public sources (Wyner-Ziv 1976 + contest rules); non-reproducible
side-info is a structural rule violation distinct from a generic contract
error.
"""

from __future__ import annotations

__all__ = [
    "AmbiguousCompositionError",
    "CanonicalComma2k19CacheRequiredViolation",
    "InflateRuntimeBudgetExceededError",
    "NonReproducibleSideInfoViolation",
    "SideInfoArchiveBudgetViolation",
    "SideInfoLedgerCorruptError",
    "SideInfoPipelineError",
    "SideInfoBakerContractError",
    "SideInformationError",
    "WynerZivCorrelationInvalidError",
]


class SideInformationError(Exception):
    """Root exception for the tac.side_information namespace.

    All typed errors below inherit from this so callers can write::

        try:
            pipeline.run(seed_state)
        except SideInformationError as exc:
            ...

    and catch every namespace-level failure with a single clause.
    """


class SideInfoBakerContractError(SideInformationError, ValueError):
    """Raised at decoration time when a SideInfoBakerContract is invalid.

    Sister of ``BoostStageContractError`` (tac.boosting) and
    ``CompressTimePassContractError`` (tac.compress_time_optimization) — same
    role at the side-info-baker surface. Always raised from
    ``SideInfoBakerContract.__post_init__`` or from the ``@side_info_baker``
    decorator wrapper.
    """


class NonReproducibleSideInfoViolation(SideInfoBakerContractError):
    """Raised at decoration time when a SideInfoBakerContract declares
    ``side_info_reproducible=False``.

    Per the contest rules + CLAUDE.md "Apples-to-apples evidence discipline"
    + Wyner-Ziv 1976 source-coding-with-side-information theorem: every
    piece of side information used by the contest decoder MUST be derivable
    from public sources. Non-reproducible side information (e.g. proprietary
    datasets, license-restricted statistics, secret pre-computed tables)
    violates contest reproducibility and is a structural rule violation
    distinguished from a generic SideInfoBakerContractError.

    The legal cases for side information per Wyner-Ziv + contest rules:
      - Frozen scorer weights (PoseNet + SegNet) — available at decode by
        contest contract.
      - Distilled palettes from PUBLIC dashcam / image datasets that the
        verifier can reproduce.
      - Public statistics (ImageNet means, dashcam class distributions).
      - Per-pair residuals encoded against a function of those legal sources.

    This error is intentionally STRUCTURAL — the decorator refuses to
    register a non-reproducible baker so it never reaches dispatch.
    """


class CanonicalComma2k19CacheRequiredViolation(SideInfoBakerContractError):
    """Raised at decoration time when a SideInfoBakerContract sets
    ``requires_canonical_comma2k19_cache=True`` but the canonical helper
    ``tac.substrates.pretrained_driving_prior.local_chunk_cache`` is not
    importable in the runtime environment.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
    against" + STRICT preflight Catalog #213 + the canonical helper
    discipline: bakers that distill priors from Comma2k19 chunks MUST route
    through ``Comma2k19LocalCache.fetch_chunk(...)`` and never bypass via
    direct downloads. This error fires at decoration time when the canonical
    cache module is missing (e.g. partial checkout) so the dispatch never
    fires a baker that would attempt a bare URL fetch.

    The error message names the canonical helper module so the operator
    knows exactly which import to repair.
    """


class WynerZivCorrelationInvalidError(SideInfoBakerContractError):
    """Raised at decoration time when ``wyner_ziv_correlation_estimate`` is
    set to a value outside the legal interval [0.0, 1.0] (or NaN).

    The Wyner-Ziv mutual-information ratio I(X;Y)/H(X) is by construction
    in [0, 1] — outside that interval the value cannot be a meaningful
    correlation estimate and the cathedral autopilot ranker would consume
    garbage. The decorator refuses such values at decoration time so a
    typo in the spec cannot corrupt the ranker's input.
    """


class SideInfoPipelineError(SideInformationError):
    """Raised at pipeline-build time when a composition is structurally
    invalid (cycle / missing producer / type mismatch / unknown baker id).

    Always raised from ``ComposableSideInfoPipeline.__or__`` /
    ``ComposableSideInfoPipeline.run`` /
    ``ComposableSideInfoPipeline.build``.
    """


class AmbiguousCompositionError(SideInfoPipelineError):
    """Raised when two bakers in the pipeline emit the same key without
    explicit ordering / merge policy.

    The canonical case: ``A | B`` where both bakers ``emit`` the key
    ``side_info_palette_v1``. Without an explicit merge policy (``&``
    parallel-merge operator or a downstream baker that consumes both via
    different keys), the pipeline cannot decide which value to forward to
    subsequent bakers.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — the ambiguity
    is surfaced at build time with a structured error message naming the
    conflicting bakers, NOT silently resolved by baker-ordering accident.
    """


class SideInfoArchiveBudgetViolation(SideInfoPipelineError):
    """Raised at run time when a ``with_archive_budget(bytes=N)`` filter
    rejects a baker's contribution that would exceed the cumulative
    archive-byte budget.

    The error includes the cumulative byte count BEFORE the rejected
    baker so the operator can audit which baker was the breaking point.
    """


class InflateRuntimeBudgetExceededError(SideInfoPipelineError):
    """Raised at run time when a ``with_inflate_runtime_budget(bytes=N)``
    filter rejects a baker whose precomputed inflate-runtime constants
    would push cumulative ``inflate_runtime_bytes_added`` past the budget.

    Per CLAUDE.md HNeRV parity discipline lesson 4 (≤ 100 LOC inflate
    budget) — inflate runtime constants count against the inflate.py size
    budget. The default budget is None (unbounded); operators that want
    to bound the inflate-runtime size opt in via ``with_inflate_runtime_budget``.
    """


class SideInfoLedgerCorruptError(SideInformationError):
    """Raised by ``persistence.load_baker_outcomes_strict`` when the JSONL
    ledger contains a malformed line OR is otherwise structurally invalid.

    Sister of ``BoostingLedgerCorruptError`` in
    ``tac.boosting.persistence`` and
    ``CompressTimeLedgerCorruptError`` in
    ``tac.compress_time_optimization.persistence`` — same fail-closed
    pattern (Catalog #138). On corruption the caller quarantines the file
    via ``persistence._quarantine_corrupt_ledger`` rather than silently
    overwriting (which would drop the corrupt evidence).
    """
