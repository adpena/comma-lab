# SPDX-License-Identifier: MIT
"""Typed exceptions for the tac.boosting namespace.

Per CLAUDE.md "Beauty, simplicity, and developer experience" — narrow
typed exceptions raised at decoration / pipeline-build / runtime time so
consumers can distinguish failure classes structurally.

Sister of ``tac.substrate_registry.SubstrateContractError`` (the canonical
META-layer error model). Every exception is a subclass of
``BoostingNamespaceError`` so a single ``except`` clause can catch all
namespace errors.
"""

from __future__ import annotations

__all__ = [
    "AmbiguousCompositionError",
    "BoostStageContractError",
    "BoostingLedgerCorruptError",
    "BoostingNamespaceError",
    "BoostingPipelineError",
    "DeterminismViolation",
    "ScorerFreedomViolation",
]


class BoostingNamespaceError(Exception):
    """Root exception for the tac.boosting namespace.

    All typed errors below inherit from this so callers can write::

        try:
            pipeline.run(seed_archive)
        except BoostingNamespaceError as exc:
            ...

    and catch every namespace-level failure with a single clause.
    """


class BoostStageContractError(BoostingNamespaceError, ValueError):
    """Raised at decoration time when a BoostStageContract is invalid.

    Sister of ``SubstrateContractError`` — same role at the boost-stage
    level. Always raised from ``BoostStageContract.__post_init__`` or from
    the ``@boost_stage`` decorator wrapper.
    """


class DeterminismViolation(BoostStageContractError):
    """Raised at decoration time when a stage claims ``deterministic=True``
    but its function signature includes randomness without a ``seed=`` kwarg.

    Per Catalog #158 deterministic-compiler discipline + CLAUDE.md
    "Bit-level deconstruction" — every stage that emits archive bytes
    must produce byte-identical output for identical input.

    The decorator inspects the wrapped function's signature for parameters
    named ``rng`` / ``random_state`` / ``noise`` without a paired ``seed=``
    parameter; if found, raises this exception so the violation surfaces
    at import time, NOT in a dispatch ten hours into a long-burn run.
    """


class ScorerFreedomViolation(BoostStageContractError):
    """Raised at decoration time when a stage with ``stage_phase="inflate"``
    declares a dependency on a scorer (``segnet`` / ``posenet`` / scorer
    weights).

    Per CLAUDE.md "Strict scorer rule" non-negotiable: inflate-time code
    paths MUST NOT load PoseNet/SegNet/scorer weights. Loading scorers at
    inflate destroys the rate term (~73 MB of weights inflates the archive)
    AND silently produces non-compliant artifacts.

    The decorator scans the contract's ``consumes`` set for forbidden tokens
    (``segnet`` / ``posenet`` / ``scorer_state`` / ``scorer_weights``); if
    found AND ``stage_phase == "inflate"``, raises this exception.
    """


class BoostingPipelineError(BoostingNamespaceError):
    """Raised at pipeline-build time when a composition is structurally
    invalid (cycle / missing producer / type mismatch / unknown stage id).

    Sister of ``DeterminismViolation`` / ``ScorerFreedomViolation`` — same
    role at the pipeline level. Always raised from
    ``ComposableBoostingPipeline.__or__`` /
    ``ComposableBoostingPipeline.run`` /
    ``ComposableBoostingPipeline.build``.
    """


class AmbiguousCompositionError(BoostingPipelineError):
    """Raised when two stages in the pipeline emit the same key without
    explicit ordering / merge policy.

    The canonical case: ``A | B`` where both stages ``emit`` the key
    ``frames_v1``. Without an explicit merge policy (``&`` parallel-merge
    operator or a downstream stage that consumes both via different keys),
    the pipeline cannot decide which value to forward to subsequent stages.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — the ambiguity
    is surfaced at build time with a structured error message naming the
    conflicting stages, NOT silently resolved by stage-ordering accident.
    """


class BoostingLedgerCorruptError(BoostingNamespaceError):
    """Raised by ``persistence.load_stage_outcomes_strict`` when the JSONL
    ledger contains a malformed line OR is otherwise structurally invalid.

    Sister of ``CallIdLedgerCorruptError`` in
    ``tac.deploy.modal.call_id_ledger`` — same fail-closed pattern (Catalog
    #138). On corruption the caller quarantines the file via
    ``persistence._quarantine_corrupt_ledger`` rather than silently
    overwriting (which would drop the corrupt evidence).
    """
