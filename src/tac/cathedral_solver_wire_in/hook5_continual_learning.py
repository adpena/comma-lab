# SPDX-License-Identifier: MIT
"""Cable D consumers 7-14 → Catalog #125 hook #5 continual-learning posterior.

Slot LL 2026-05-20 — closes the canonical producer → sidecar → ranker (FF) →
solver (HH) → posterior (LL hook #5) loop per Catalog #125 for the 6 Cable D
consumers landed 2026-05-19 (commit `6a1e94a63`).

Per Catalog #303 cargo-cult audit per assumption: the operator prompt asks
"wire hook #5 × 6 consumers (target: 6)". The HARD-EARNED inventory per the
producer-header ``CONSUMER_HOOK_NUMBERS`` declarations is **1 ACTIVE cell**,
NOT 6:

    | Consumer                              | hook #5 declared in CONSUMER_HOOK_NUMBERS? |
    |---------------------------------------|--------------------------------------------|
    | #7  per_pair_pareto_envelope          | N/A (NO-OP per producer header)            |
    | #8  per_pair_lagrangian_lambda_bisection | N/A (NO-OP per producer header)         |
    | #9  per_pair_lora_supervision_signal  | **ACTIVE** (declared CONTINUAL_LEARNING_POSTERIOR) |
    | #10 per_pair_coding_budget_allocation | N/A (NO-OP per producer header)            |
    | #12 per_pair_kkt_residuals            | N/A (NO-OP per producer header)            |
    | #13 per_pair_volterra_cross_terms     | N/A (NO-OP per producer header)            |

1 ACTIVE + 5 N/A = 6 cells. The 5 N/A cells each carry the explicit producer-
header docstring rationale: *"Per-pair <X> sidecar JSON is canonically
persisted via ``tac.master_gradient_consumers.consumer_output_path`` at
``.omx/state/master_gradient_consumers/``. NO-OP by design."*

Forcing the full 6 via "all × all" would create **duplicate posterior writes
whose existence the producer never declared** — that IS the canonical
Catalog #131 fcntl-locked-bare-write-discipline bug class. The sidecar JSON
IS the canonical posterior write surface; a SECOND write to
``.omx/state/continual_learning_posterior.json`` for the same producer would
be a duplicate write at a different surface, exactly the failure mode
Catalog #128/#131/#138 sister discipline extincts.

**Why consumer #9 IS HARD-EARNED ACTIVE:** the producer's
``CONSUMER_HOOK_NUMBERS`` tuple explicitly declares
``HookNumber.CONTINUAL_LEARNING_POSTERIOR``. The producer wired this hook
because LoRA-supervision signals participate in continual-learning at
training time (per-pair LoRA adapter targets are the canonical absorber for
distortion-variance-across-pairs). Note: the producer's ``update_from_anchor``
remains a docstring-NO-OP because the sidecar IS the canonical posterior
write surface; THIS module emits the canonical SOLVER-side posterior anchor
that downstream ``tac.continual_learning.posterior_update_locked`` consumers
(autopilot ranker / Rashomon ensemble / sister Cathedral consumers) query
via the canonical query helpers below.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323/#341:
every posterior anchor emitted by this module is OBSERVABILITY-ONLY:

    - ``score_claim_valid=False`` (per canonical Provenance)
    - ``promotion_eligible=False`` (per Catalog #127/#192/#317/#341 promotion-leak guard)
    - canonical ``[predicted]`` axis tag (per Catalog #287)
    - the anchor is metadata about the sidecar's presence + structural-signal
      counts, NOT a contest-axis score claim

Per Catalog #318 raw-byte master-gradient guard: anchors NEVER contain raw
archive-byte tensors; only canonical typed structural-signal markers
(``n_pairs`` + ``n_bytes`` + canonical sidecar SHA-256).

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305: every
anchor is inspectable per layer (canonical helper exposes per-cell verdict),
decomposable per signal (per-consumer return type), diff-able across runs
(stateless functions of canonical sidecar JSON), queryable post-hoc (the
``Hook5PosteriorAnchor`` dataclass exposes all fields), cite-able (canonical
sidecar path + sha256), and counterfactual-able (removing the canonical
sidecar zeros that consumer × hook #5 anchor).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from tac.cathedral_solver_wire_in.consumers_7_14_contributions import (
    CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS,
    SOLVER_WIRE_IN_OBSERVABILITY_AXIS,
    _build_contribution_payload,
    _expected_schema_for_consumer,
)

# ─── Catalog #303 cargo-cult audit: HARD-EARNED hook #5 ACTIVE cells ────────
# Per producer-header CONSUMER_HOOK_NUMBERS declarations: ONLY consumer #9
# (per_pair_lora_supervision_signal) declares
# HookNumber.CONTINUAL_LEARNING_POSTERIOR. The other 5 consumers have
# NO-OP ``update_from_anchor`` per the producer-header rationale that the
# sidecar IS the canonical posterior write.

CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE: frozenset[str] = frozenset(
    {"per_pair_lora_supervision_signal"}
)

# Public re-export: the 1 ACTIVE (consumer, hook) pair for hook #5.
CABLE_D_CONSUMERS_7_14_HOOK_5_PAIRS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (consumer_id, "continual_learning_posterior")
        for consumer_id in CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE
    )
)


@dataclass(frozen=True)
class Hook5PosteriorAnchor:
    """Typed observability-only continual-learning posterior anchor.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
    #287/#323/#341: anchors are OBSERVABILITY-ONLY:

    - ``score_claim_valid=False``
    - ``promotion_eligible=False``
    - canonical ``[predicted]`` ``axis_tag``
    - ``predicted_delta_adjustment=0.0``

    Per Catalog #305 6-facet observability: every field is inspectable +
    decomposable + diff-able + queryable + cite-able + counterfactual-able.

    Per Catalog #318 master-gradient raw-byte-authority guard: the payload
    field is restricted to typed structural-signal markers (canonical sidecar
    path + schema + sha + n_pairs/n_bytes counts), NEVER raw archive-byte
    tensors.

    Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY: anchors are
    deterministic projections of the canonical sidecar JSON; they are NOT
    persisted independently (the sidecar IS the canonical write surface).
    Sister subagents that need a persistent posterior trail should call
    ``tac.continual_learning.posterior_update_locked`` with a properly-formed
    ``ContestResult`` — that helper handles the canonical fcntl-locked write
    to ``.omx/state/continual_learning_posterior.json``. THIS module emits
    typed READ-ONLY anchors for downstream consumers to query.
    """

    consumer_id: str
    archive_sha256: str
    sidecar_present: bool
    sidecar_path: str | None  # canonical sidecar path for cite-chain
    sidecar_schema: str  # expected schema tag (canonical)
    sidecar_sha256: str | None  # SHA-256 of the canonical sidecar JSON (cite-chain)
    n_pairs: int  # structural signal: 0 if sidecar absent/invalid
    n_bytes: int  # structural signal: 0 if sidecar absent/invalid

    # Canonical non-promotable markers (per Catalog #287/#323/#341)
    score_claim_valid: bool = False
    promotion_eligible: bool = False
    axis_tag: str = SOLVER_WIRE_IN_OBSERVABILITY_AXIS

    # Per-consumer narrative annotation (human-readable cite-chain).
    rationale: str = ""

    # Canonical Provenance citation for the anchor.
    provenance_canonical_helper: str = (
        "tac.cathedral_solver_wire_in.hook5_continual_learning"
    )

    def __post_init__(self) -> None:
        if self.consumer_id not in CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE:
            raise ValueError(
                f"consumer_id={self.consumer_id!r} is N/A for hook #5 per "
                f"Catalog #303 cargo-cult audit; HARD-EARNED hook #5 consumers "
                f"per CONSUMER_HOOK_NUMBERS declarations are "
                f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE)}. "
                "Constructing this N/A anchor would create a phantom posterior "
                "write per CLAUDE.md 'Apples-to-apples evidence discipline' + "
                "Catalog #131 fcntl-locked bare-write discipline (the sidecar "
                "IS the canonical posterior write; a second write here is "
                "duplicate)."
            )
        # Canonical non-promotable invariants per Catalog #287/#323/#341
        if self.score_claim_valid:
            raise ValueError(
                "score_claim_valid=True forbidden per CLAUDE.md "
                "'Apples-to-apples evidence discipline' + Catalog #287/#323 "
                "canonical Provenance; posterior anchors are observability-only."
            )
        if self.promotion_eligible:
            raise ValueError(
                "promotion_eligible=True forbidden per Catalog #127/#192/#317/#341 "
                "promotion-leak guard; posterior anchors never leak into promotion."
            )
        if self.axis_tag != SOLVER_WIRE_IN_OBSERVABILITY_AXIS:
            raise ValueError(
                f"axis_tag={self.axis_tag!r} must equal "
                f"{SOLVER_WIRE_IN_OBSERVABILITY_AXIS!r} per Catalog #287 "
                "canonical Provenance for [predicted] axis."
            )


def consumer_owns_hook_5(consumer_id: str) -> bool:
    """True iff consumer is in the HARD-EARNED hook #5 ACTIVE set.

    Per Catalog #303: this is the canonical disambiguator between HARD-EARNED
    ACTIVE cells (per producer-header CONSUMER_HOOK_NUMBERS declarations) and
    CARGO-CULTED N/A cells (per "all consumers × hook #5" reflex).
    """
    return consumer_id in CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE


def _sha256_of_sidecar(path_str: str | None) -> str | None:
    """Compute canonical SHA-256 of a sidecar JSON for cite-chain provenance.

    Returns None if path is None or unreadable. Per Catalog #287/#323 canonical
    Provenance: the sha is a CITE-CHAIN marker, NOT a score-claim signature.
    """
    if path_str is None:
        return None
    try:
        with open(path_str, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()
    except (OSError, FileNotFoundError):
        return None


def _hook_5_rationale_for(consumer_id: str, sidecar_present: bool) -> str:
    """Human-readable cite-chain rationale for hook #5 anchor.

    Documents the producer-header CONSUMER_HOOK_NUMBERS declaration that
    justifies this consumer being HARD-EARNED ACTIVE per Catalog #303.
    """
    base = (
        f"Cable D consumer {consumer_id!r} → hook #5 continual-learning posterior "
        "per producer-header CONSUMER_HOOK_NUMBERS declaration (HookNumber."
        "CONTINUAL_LEARNING_POSTERIOR explicitly declared); per-pair LoRA "
        "supervision targets participate in continual-learning at training time"
    )
    presence = (
        "canonical sidecar PRESENT + schema-valid + custody-clean + non-trivial "
        "structural signal; SHA-256 captured for cite-chain"
        if sidecar_present
        else "canonical sidecar ABSENT or non-structural; anchor reflects empty state"
    )
    return f"{base}; {presence} [predicted]"


def query_posterior_for_consumer(
    consumer_id: str, archive_sha256: str
) -> Hook5PosteriorAnchor:
    """Hook #5 continual-learning posterior anchor for a Cable D consumer.

    Per Catalog #303 HARD-EARNED cells (producer-header CONSUMER_HOOK_NUMBERS
    declarations):

    - #9 per_pair_lora_supervision_signal — declares
      ``HookNumber.CONTINUAL_LEARNING_POSTERIOR`` in ``CONSUMER_HOOK_NUMBERS``;
      per-pair LoRA adapter targets are the canonical absorber for
      distortion-variance-across-pairs at training time.

    Per Catalog #303 N/A cells (CARGO-CULTED per "all × all"):

    - #7  per_pair_pareto_envelope — NO-OP per producer header
    - #8  per_pair_lagrangian_lambda_bisection — NO-OP per producer header
    - #10 per_pair_coding_budget_allocation — NO-OP per producer header
    - #12 per_pair_kkt_residuals — NO-OP per producer header
    - #13 per_pair_volterra_cross_terms — NO-OP per producer header

    Raises ``ValueError`` if ``consumer_id`` is N/A per Catalog #303. The error
    message cites the producer-header declaration that justifies the N/A
    classification + the Catalog #131 fcntl-locked bare-write discipline that
    forbids duplicate posterior writes.

    Per Catalog #305 observability: the returned anchor exposes the canonical
    sidecar path + sha256 for cite-chain, ``n_pairs``/``n_bytes`` for
    structural-signal decomposition, and the consumer-id for per-layer
    inspection.
    """
    if not consumer_owns_hook_5(consumer_id):
        raise ValueError(
            f"consumer {consumer_id!r} does NOT own hook #5 "
            f"'continual_learning_posterior' per Catalog #303 HARD-EARNED "
            f"registry; HARD-EARNED hook #5 consumers per CONSUMER_HOOK_NUMBERS "
            f"declarations are "
            f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE)}. The producer "
            "header explicitly declares NO-OP for hook #5; the sidecar IS the "
            "canonical posterior write surface (per "
            "tac.master_gradient_consumers.consumer_output_path); duplicate "
            "posterior writes are forbidden per Catalog #131 fcntl-locked "
            "bare-write discipline."
        )
    if not archive_sha256:
        raise ValueError("archive_sha256 must be non-empty")
    expected_schema = _expected_schema_for_consumer(consumer_id)
    sidecar_present, sidecar_path, n_pairs, n_bytes = _build_contribution_payload(
        consumer_id, archive_sha256
    )
    sidecar_sha = _sha256_of_sidecar(sidecar_path)
    return Hook5PosteriorAnchor(
        consumer_id=consumer_id,
        archive_sha256=archive_sha256,
        sidecar_present=sidecar_present,
        sidecar_path=sidecar_path,
        sidecar_schema=expected_schema,
        sidecar_sha256=sidecar_sha,
        n_pairs=n_pairs,
        n_bytes=n_bytes,
        rationale=_hook_5_rationale_for(consumer_id, sidecar_present),
    )


def collect_all_hook_5_anchors_for_archive(
    archive_sha256: str,
) -> tuple[Hook5PosteriorAnchor, ...]:
    """Collect every HARD-EARNED hook #5 anchor for an archive.

    Returns the 1 ACTIVE anchor per Catalog #303 cargo-cult audit. N/A cells
    are NOT included; per CLAUDE.md "Apples-to-apples evidence discipline"
    forcing N/A cells would create phantom anchors.

    Per Catalog #305 6-facet observability: the return value is the canonical
    queryable snapshot of every hook #5 anchor for this archive, diff-able
    across runs (stateless), inspectable per consumer, cite-able via each
    anchor's sidecar_path + sidecar_sha256.
    """
    if not archive_sha256:
        raise ValueError("archive_sha256 must be non-empty")
    anchors: list[Hook5PosteriorAnchor] = []
    for consumer_id in sorted(CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE):
        anchors.append(query_posterior_for_consumer(consumer_id, archive_sha256))
    return tuple(anchors)


def is_hook_5_anchor_promotable(anchor: Hook5PosteriorAnchor) -> bool:
    """Per Catalog #341: hook #5 anchors are NEVER promotable.

    Returns False by construction; provided as a public guard so downstream
    consumers (e.g. autopilot ranker, dispatch wrappers,
    ``tac.continual_learning.posterior_update_locked`` callers) can call this
    explicitly per CLAUDE.md "Apples-to-apples evidence discipline" instead
    of relying on the anchor's ``promotion_eligible`` field directly.
    """
    return False  # always False by Catalog #341 invariant


__all__ = [
    "CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE",
    "CABLE_D_CONSUMERS_7_14_HOOK_5_PAIRS",
    "Hook5PosteriorAnchor",
    "collect_all_hook_5_anchors_for_archive",
    "consumer_owns_hook_5",
    "is_hook_5_anchor_promotable",
    "query_posterior_for_consumer",
]
