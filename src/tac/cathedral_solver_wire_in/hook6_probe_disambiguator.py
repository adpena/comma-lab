# SPDX-License-Identifier: MIT
"""Cable D consumers 7-14 → Catalog #125 hook #6 probe-disambiguator.

Slot LL 2026-05-20 — closes the canonical producer → sidecar → ranker (FF) →
solver (HH) → posterior (LL hook #5) → disambiguator (LL hook #6) FULL 6-hook
loop per Catalog #125 for the 6 Cable D consumers landed 2026-05-19 (commit
`6a1e94a63`).

Per Catalog #303 cargo-cult audit per assumption: the operator prompt asks
"wire hook #6 × 6 consumers (target: 6)". The HARD-EARNED inventory per the
producer-header ``CONSUMER_HOOK_NUMBERS`` declarations is **1 ACTIVE cell**,
NOT 6:

    | Consumer                              | hook #6 declared in CONSUMER_HOOK_NUMBERS? |
    |---------------------------------------|--------------------------------------------|
    | #7  per_pair_pareto_envelope          | N/A (not declared)                         |
    | #8  per_pair_lagrangian_lambda_bisection | N/A (not declared)                      |
    | #9  per_pair_lora_supervision_signal  | N/A (not declared)                         |
    | #10 per_pair_coding_budget_allocation | N/A (not declared)                         |
    | #12 per_pair_kkt_residuals            | **ACTIVE** (declared PROBE_DISAMBIGUATOR)  |
    | #13 per_pair_volterra_cross_terms     | N/A (not declared)                         |

1 ACTIVE + 5 N/A = 6 cells. The HARD-EARNED ACTIVE consumer is #12
per_pair_kkt_residuals, whose producer-header explicitly declares
``HookNumber.PROBE_DISAMBIGUATOR`` because the per-pair KKT residual is
**the canonical stationarity certificate**: ``||dD/dθ + λ_R · dR/dθ||_2`` per
pair. HIGH residual = joint codec failing to balance distortion vs rate at
that pair; the per-pair residual rank IS the canonical disambiguator that
ranks dispatch targets by leverage (highest-residual pairs first).

Forcing the full 6 via "all × all" would create **phantom disambiguator
surfaces** for consumers whose signal type is NOT disambiguating —
Pareto envelope (#7) is unambiguous (rate/distortion frontier); λ_R bisection
(#8) is unambiguous when bisection converges; LoRA supervision (#9) targets
are derived, not disambiguating; coding-budget allocation (#10) is a
canonical primary decision, not a disambiguator; Volterra cross-terms (#13)
are one-shot characterization of interaction matrix, not a runtime ambiguity-
resolver. Each N/A is HARD-EARNED per the producer header NOT declaring the
hook in ``CONSUMER_HOOK_NUMBERS``.

Per CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator pattern":
*"when a design choice has 2+ defensible interpretations, ship BOTH modes
via callable interface + build ``tools/probe_<track>_disambiguator.py`` that
returns the regime-conditional verdict."* For Cable D, the regime-conditional
verdict is **per-pair stationarity rank** — the top-N pairs with highest KKT
residual identify where the joint codec needs the next dispatch's attention.

**Sister wiring to Catalog #313**: the probe-outcomes ledger at
``.omx/state/probe_outcomes.jsonl`` (canonical helper
``tac.probe_outcomes_ledger``) is the canonical persistence layer for probe
verdicts. THIS module emits typed READ-ONLY ``Hook6DisambiguatorVerdict``
anchors that downstream consumers can pass to
``tac.probe_outcomes_ledger.register_probe_outcome`` for persistence. Per
Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY: anchors are
deterministic projections of the canonical sidecar JSON; they are NOT
persisted independently by this module.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323/#341:
every disambiguator verdict emitted by this module is OBSERVABILITY-ONLY:

    - ``score_claim_valid=False`` (per canonical Provenance)
    - ``promotion_eligible=False`` (per Catalog #127/#192/#317/#341 promotion-leak guard)
    - canonical ``[predicted]`` axis tag (per Catalog #287)
    - the verdict is per-pair RANK metadata, NOT a contest-axis score claim

Per Catalog #318 raw-byte master-gradient guard: verdicts NEVER contain raw
archive-byte tensors; only canonical typed structural-signal markers
(``n_pairs`` + ``top_pair_indices`` + ``top_pair_residual_magnitudes``).

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305: every
verdict is inspectable per layer + decomposable per per-pair rank +
diff-able across runs (stateless functions of canonical sidecar JSON) +
queryable post-hoc + cite-able (canonical sidecar path + sha256) +
counterfactual-able (removing the canonical sidecar zeros the verdict).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from tac.cathedral_solver_wire_in.consumers_7_14_contributions import (
    CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS,
    SOLVER_WIRE_IN_OBSERVABILITY_AXIS,
    _build_contribution_payload,
    _expected_schema_for_consumer,
)

# ─── Catalog #303 cargo-cult audit: HARD-EARNED hook #6 ACTIVE cells ────────
# Per producer-header CONSUMER_HOOK_NUMBERS declarations: ONLY consumer #12
# (per_pair_kkt_residuals) declares HookNumber.PROBE_DISAMBIGUATOR.
# The other 5 consumers do NOT declare hook #6 because their signal types
# are not disambiguating (Pareto envelope = unambiguous frontier; λ_R bisection
# = unambiguous when converged; LoRA targets = derived; coding-budget =
# canonical primary decision; Volterra = one-shot characterization).

CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE: frozenset[str] = frozenset(
    {"per_pair_kkt_residuals"}
)

# Public re-export: the 1 ACTIVE (consumer, hook) pair for hook #6.
CABLE_D_CONSUMERS_7_14_HOOK_6_PAIRS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (consumer_id, "probe_disambiguator")
        for consumer_id in CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE
    )
)

# Default top-N pair count for per-pair stationarity rank. Operator-tunable
# via the ``top_n`` kwarg on ``disambiguate_per_pair_stationarity``.
DEFAULT_DISAMBIGUATOR_TOP_N: int = 16

# Canonical verdict taxonomy per probe-outcomes-ledger sister (Catalog #313).
# These map 1:1 to ``tac.probe_outcomes_ledger.VALID_VERDICTS``. The hook #6
# surface defaults to PARTIAL (per-pair rank is informational; full
# adjudication requires sister probe + council deliberation per CLAUDE.md
# "Forbidden premature KILL without research exhaustion").
HOOK_6_DEFAULT_VERDICT: str = "PARTIAL"


@dataclass(frozen=True)
class Hook6DisambiguatorVerdict:
    """Typed observability-only probe-disambiguator verdict.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
    #287/#323/#341: verdicts are OBSERVABILITY-ONLY:

    - ``score_claim_valid=False``
    - ``promotion_eligible=False``
    - canonical ``[predicted]`` ``axis_tag``

    The verdict carries the per-pair STATIONARITY RANK: ``top_pair_indices``
    is the indices of pairs with highest KKT residual; ``top_pair_residual_magnitudes``
    is the corresponding magnitudes. Downstream dispatch ranking consumes the
    top-N pairs as "where the next dispatch should focus its byte budget".

    Per Catalog #305 6-facet observability: every field is inspectable +
    decomposable per per-pair rank + diff-able + queryable + cite-able +
    counterfactual-able.

    Per Catalog #318 master-gradient raw-byte-authority guard: the payload
    field is restricted to typed structural-signal markers (canonical sidecar
    path + schema + sha + per-pair INDICES + per-pair MAGNITUDES), NEVER raw
    archive-byte tensors.

    Per Catalog #313 sister probe-outcomes ledger: downstream consumers can
    persist this verdict via ``tac.probe_outcomes_ledger.register_probe_outcome``
    with ``probe_kind="cable_d_per_pair_kkt_stationarity_rank"`` and
    ``verdict="PARTIAL"`` (default; informational per-pair rank does not
    block dispatch).
    """

    consumer_id: str
    archive_sha256: str
    sidecar_present: bool
    sidecar_path: str | None  # canonical sidecar path for cite-chain
    sidecar_schema: str  # expected schema tag (canonical)
    sidecar_sha256: str | None  # SHA-256 of the canonical sidecar JSON (cite-chain)
    n_pairs: int  # total pair count from sidecar (0 if absent)

    # Per-pair RANK signal (canonical disambiguator output):
    top_pair_indices: tuple[int, ...]  # indices of top-N pairs by residual
    top_pair_residual_magnitudes: tuple[float, ...]  # corresponding magnitudes
    top_n_requested: int  # N requested (may exceed n_pairs)

    # Canonical verdict taxonomy mirror per Catalog #313.
    verdict: str = HOOK_6_DEFAULT_VERDICT
    metric_name: str = "per_pair_kkt_residual_magnitude_l2"

    # Canonical non-promotable markers (per Catalog #287/#323/#341)
    score_claim_valid: bool = False
    promotion_eligible: bool = False
    axis_tag: str = SOLVER_WIRE_IN_OBSERVABILITY_AXIS

    # Per-consumer narrative annotation (human-readable cite-chain).
    rationale: str = ""

    # Canonical Provenance citation for the verdict.
    provenance_canonical_helper: str = (
        "tac.cathedral_solver_wire_in.hook6_probe_disambiguator"
    )

    def __post_init__(self) -> None:
        if self.consumer_id not in CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE:
            raise ValueError(
                f"consumer_id={self.consumer_id!r} is N/A for hook #6 per "
                f"Catalog #303 cargo-cult audit; HARD-EARNED hook #6 consumers "
                f"per CONSUMER_HOOK_NUMBERS declarations are "
                f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE)}. "
                "Constructing this N/A verdict would create a phantom "
                "disambiguator surface per CLAUDE.md "
                "'Anti-arbitrariness primitive: the probe-disambiguator pattern' "
                "+ CLAUDE.md 'Apples-to-apples evidence discipline'."
            )
        # Canonical non-promotable invariants per Catalog #287/#323/#341
        if self.score_claim_valid:
            raise ValueError(
                "score_claim_valid=True forbidden per CLAUDE.md "
                "'Apples-to-apples evidence discipline' + Catalog #287/#323 "
                "canonical Provenance; disambiguator verdicts are "
                "observability-only."
            )
        if self.promotion_eligible:
            raise ValueError(
                "promotion_eligible=True forbidden per Catalog #127/#192/#317/#341 "
                "promotion-leak guard; disambiguator verdicts never leak into "
                "promotion."
            )
        if self.axis_tag != SOLVER_WIRE_IN_OBSERVABILITY_AXIS:
            raise ValueError(
                f"axis_tag={self.axis_tag!r} must equal "
                f"{SOLVER_WIRE_IN_OBSERVABILITY_AXIS!r} per Catalog #287 "
                "canonical Provenance for [predicted] axis."
            )
        # Per Catalog #313 sister: verdict must be in canonical ledger taxonomy
        if self.verdict not in {
            "INDEPENDENT",
            "KILL",
            "DEFER",
            "PROMOTE",
            "PROCEED",
            "PARTIAL",
            "OPERATOR_REVIEW_REQUIRED",
        }:
            raise ValueError(
                f"verdict={self.verdict!r} not in canonical Catalog #313 "
                "VALID_VERDICTS set; the disambiguator verdict must map to "
                "the probe-outcomes ledger taxonomy."
            )
        # Per-pair rank invariant: indices and magnitudes must be parallel
        if len(self.top_pair_indices) != len(self.top_pair_residual_magnitudes):
            raise ValueError(
                f"top_pair_indices ({len(self.top_pair_indices)}) and "
                f"top_pair_residual_magnitudes "
                f"({len(self.top_pair_residual_magnitudes)}) must have equal "
                "length (parallel arrays for per-pair rank)."
            )
        if self.top_n_requested < 0:
            raise ValueError(
                f"top_n_requested={self.top_n_requested} must be non-negative"
            )


def consumer_owns_hook_6(consumer_id: str) -> bool:
    """True iff consumer is in the HARD-EARNED hook #6 ACTIVE set.

    Per Catalog #303: this is the canonical disambiguator between HARD-EARNED
    ACTIVE cells (per producer-header CONSUMER_HOOK_NUMBERS declarations) and
    CARGO-CULTED N/A cells (per "all consumers × hook #6" reflex).
    """
    return consumer_id in CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE


def _sha256_of_sidecar(path_str: str | None) -> str | None:
    """Compute canonical SHA-256 of a sidecar JSON for cite-chain provenance."""
    if path_str is None:
        return None
    try:
        with open(path_str, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()
    except (OSError, FileNotFoundError):
        return None


def _read_sidecar_payload(path_str: str | None) -> Mapping[str, Any] | None:
    """Read canonical sidecar JSON payload (read-only)."""
    if path_str is None:
        return None
    try:
        with open(path_str, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except (OSError, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _extract_top_n_pair_rank(
    payload: Mapping[str, Any] | None, top_n: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Extract per-pair stationarity rank from canonical KKT sidecar payload.

    Per the canonical
    ``master_gradient_consumer_per_pair_kkt_residuals_v1`` schema, the
    payload may carry one of:

    - ``per_pair_residual_magnitudes``: list[float] of per-pair |residual|
    - ``residual_magnitudes``: alias
    - ``per_pair_kkt_residual``: list[float] of per-pair signed residuals

    Returns ``(top_indices, top_magnitudes)`` sorted by descending magnitude.
    Returns ``((), ())`` if no recognizable per-pair rank signal present.
    """
    if payload is None:
        return ((), ())
    candidate_keys = (
        "per_pair_residual_magnitudes",
        "residual_magnitudes",
        "per_pair_kkt_residual",
        "per_pair_residuals",
    )
    magnitudes: list[float] | None = None
    for key in candidate_keys:
        raw = payload.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            try:
                magnitudes = [abs(float(x)) for x in raw]
                break
            except (TypeError, ValueError):
                continue
    if magnitudes is None:
        return ((), ())
    # Sort descending by magnitude; keep original indices.
    indexed = sorted(
        enumerate(magnitudes), key=lambda pair: pair[1], reverse=True
    )
    truncated = indexed[: max(0, top_n)]
    indices = tuple(i for i, _ in truncated)
    mags = tuple(m for _, m in truncated)
    return (indices, mags)


def _hook_6_rationale_for(
    consumer_id: str,
    sidecar_present: bool,
    n_pairs: int,
    n_top: int,
) -> str:
    """Human-readable cite-chain rationale for hook #6 verdict.

    Documents the producer-header CONSUMER_HOOK_NUMBERS declaration that
    justifies this consumer being HARD-EARNED ACTIVE per Catalog #303.
    """
    base = (
        f"Cable D consumer {consumer_id!r} → hook #6 probe-disambiguator per "
        "producer-header CONSUMER_HOOK_NUMBERS declaration (HookNumber."
        "PROBE_DISAMBIGUATOR explicitly declared); per-pair KKT residual "
        "||dD + λ_R·dR||_2 is the canonical per-pair stationarity certificate "
        "(HIGH residual = joint codec failing to balance distortion vs rate at "
        "that pair)"
    )
    if sidecar_present:
        presence = (
            f"canonical sidecar PRESENT + schema-valid + custody-clean; "
            f"top-{n_top} of {n_pairs} pairs ranked by residual magnitude; "
            "SHA-256 captured for cite-chain"
        )
    else:
        presence = (
            "canonical sidecar ABSENT or non-structural; verdict reflects "
            "empty per-pair rank"
        )
    return f"{base}; {presence} [predicted]"


def disambiguate_per_pair_stationarity(
    archive_sha256: str,
    *,
    top_n: int = DEFAULT_DISAMBIGUATOR_TOP_N,
    consumer_id: str = "per_pair_kkt_residuals",
) -> Hook6DisambiguatorVerdict:
    """Hook #6 canonical per-pair stationarity disambiguator for Cable D.

    Returns the typed verdict carrying the top-N pairs ranked by per-pair
    KKT residual magnitude. HIGH residual = joint codec failing to balance
    distortion vs rate at that pair; downstream dispatch ranking can consume
    the top-N pair indices to focus the next dispatch's byte budget.

    Per Catalog #303 ONLY consumer #12 (per_pair_kkt_residuals) owns hook #6
    per producer-header CONSUMER_HOOK_NUMBERS declaration. The ``consumer_id``
    kwarg defaults to the canonical consumer; passing any other consumer
    raises ``ValueError`` per Catalog #303 phantom-cell refusal.

    Per Catalog #313 sister probe-outcomes ledger: downstream consumers can
    pass this verdict's typed fields to
    ``tac.probe_outcomes_ledger.register_probe_outcome`` for canonical
    fcntl-locked persistence. The verdict defaults to "PARTIAL" because the
    per-pair rank is INFORMATIONAL (HIGH-residual pairs need attention but
    do NOT block dispatch); full adjudication (INDEPENDENT / KILL / DEFER /
    PROMOTE / PROCEED) requires sister probe + council deliberation per
    CLAUDE.md "Forbidden premature KILL without research exhaustion".
    """
    if not consumer_owns_hook_6(consumer_id):
        raise ValueError(
            f"consumer {consumer_id!r} does NOT own hook #6 "
            f"'probe_disambiguator' per Catalog #303 HARD-EARNED registry; "
            f"HARD-EARNED hook #6 consumers per CONSUMER_HOOK_NUMBERS "
            f"declarations are "
            f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE)}. Per-pair Pareto "
            "envelope / lambda bisection / LoRA supervision / coding budget / "
            "Volterra cross-terms do NOT carry per-pair ambiguity-resolution "
            "semantics; their producer headers do not declare hook #6."
        )
    if not archive_sha256:
        raise ValueError("archive_sha256 must be non-empty")
    if top_n < 0:
        raise ValueError(f"top_n={top_n} must be non-negative")
    expected_schema = _expected_schema_for_consumer(consumer_id)
    sidecar_present, sidecar_path, n_pairs, _n_bytes = _build_contribution_payload(
        consumer_id, archive_sha256
    )
    sidecar_sha = _sha256_of_sidecar(sidecar_path)
    payload = _read_sidecar_payload(sidecar_path) if sidecar_present else None
    top_indices, top_magnitudes = _extract_top_n_pair_rank(payload, top_n)
    return Hook6DisambiguatorVerdict(
        consumer_id=consumer_id,
        archive_sha256=archive_sha256,
        sidecar_present=sidecar_present,
        sidecar_path=sidecar_path,
        sidecar_schema=expected_schema,
        sidecar_sha256=sidecar_sha,
        n_pairs=n_pairs,
        top_pair_indices=top_indices,
        top_pair_residual_magnitudes=top_magnitudes,
        top_n_requested=top_n,
        rationale=_hook_6_rationale_for(
            consumer_id, sidecar_present, n_pairs, len(top_indices)
        ),
    )


def collect_all_hook_6_verdicts_for_archive(
    archive_sha256: str,
    *,
    top_n: int = DEFAULT_DISAMBIGUATOR_TOP_N,
) -> tuple[Hook6DisambiguatorVerdict, ...]:
    """Collect every HARD-EARNED hook #6 verdict for an archive.

    Returns the 1 ACTIVE verdict per Catalog #303 cargo-cult audit. N/A cells
    are NOT included; per CLAUDE.md "Apples-to-apples evidence discipline"
    forcing N/A cells would create phantom disambiguator surfaces.

    Per Catalog #305 6-facet observability: the return value is the canonical
    queryable snapshot of every hook #6 verdict for this archive.
    """
    if not archive_sha256:
        raise ValueError("archive_sha256 must be non-empty")
    verdicts: list[Hook6DisambiguatorVerdict] = []
    for consumer_id in sorted(CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE):
        verdicts.append(
            disambiguate_per_pair_stationarity(
                archive_sha256, top_n=top_n, consumer_id=consumer_id
            )
        )
    return tuple(verdicts)


def is_hook_6_verdict_promotable(verdict: Hook6DisambiguatorVerdict) -> bool:
    """Per Catalog #341: hook #6 verdicts are NEVER promotable.

    Returns False by construction; provided as a public guard so downstream
    consumers (e.g. autopilot ranker, dispatch wrappers,
    ``tac.probe_outcomes_ledger.register_probe_outcome`` callers) can call
    this explicitly per CLAUDE.md "Apples-to-apples evidence discipline"
    instead of relying on the verdict's ``promotion_eligible`` field directly.
    """
    return False  # always False by Catalog #341 invariant


__all__ = [
    "CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE",
    "CABLE_D_CONSUMERS_7_14_HOOK_6_PAIRS",
    "DEFAULT_DISAMBIGUATOR_TOP_N",
    "HOOK_6_DEFAULT_VERDICT",
    "Hook6DisambiguatorVerdict",
    "collect_all_hook_6_verdicts_for_archive",
    "consumer_owns_hook_6",
    "disambiguate_per_pair_stationarity",
    "is_hook_6_verdict_promotable",
]
