# SPDX-License-Identifier: MIT
"""Unified scorer-quotient candidate schema — the adjudication layer for the
distortion-closure / feasibility fleet (operator directive 2026-06-10).

Every score-mover (#63 hinge, #69 requant, #71 structural compression, #54
cross-pair corrector, #72 margin-conditional residual coder, #73 legal-frame
Dykstra feasibility) emits a normalized row in this family so heterogeneous
branches become directly comparable and the sub-0.15 firewall is enforced in
ONE place.

The two dual problems this fleet attacks (the operator's framing):
  * Problem 1 — quotient-target synthesis: find the desired SegNet argmax
    PARTITION + the 6 pooled PoseNet scalars (a low-dim non-smooth quotient).
  * Problem 2 — legal-preimage realization: find a CHEAP RGB frame whose
    evaluator projection IS that quotient (constraint satisfaction, not
    reconstruction). #73 is the first real attack on Problem 2.

The score is always recomputed from components (the rounded ``final_score``
field lies):  S = 100*d_seg + sqrt(10*d_pose) + 25*bytes / RATE_DENOM .

THE FIRREWALL (sub-0.15 non-negotiable + MPS-never + "Frontier scores are
pointer-only"): a candidate is ``pointer_update_eligible`` ONLY when its
authority is a contest-tier EXACT evaluate.py row AND its recomputed ΔS < 0.
Telemetry-proxy / MLX / local-CPU-advisory rows RANK and seed priors but NEVER
promote. A Dykstra projection that satisfies the linearized surrogate but lacks
an exact contest row is NOT a pointer move — projection residual is a surrogate,
exact evaluate.py is the authority.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RATE_DENOM = 37_545_489  # evaluate.py:64 verified fixed denominator

# --- controlled vocabularies ------------------------------------------------
CANDIDATE_KINDS = frozenset({
    "dykstra_feasible_frame",   # #73 legal-frame feasibility
    "margin_residual",          # #72 margin-conditional residual coder
    "cross_pair_corrector",     # #54 cross-pair waterfilled corrector
    "requant",                  # #69 score-aware weight re-quant
    "structural_compression",   # #71 factor/prune/share/distill
    "renderer_loss_hinge",      # #63 d_seg-loss conditioning
})
# Authority ladder (low -> high). Only the top two contest tiers can promote.
AUTHORITY_TIERS = ("telemetry_proxy", "exact_cpu_advisory", "contest_cpu", "contest_cuda")
_PROMOTING_AUTHORITY = frozenset({"contest_cpu", "contest_cuda"})
METRIC_FAMILIES = frozenset({"scorer_proxy", "exact_pair_scorer", "exact_evaluate"})
CANDIDATE_DECISIONS = frozenset({"accept", "reject", "continue", "scale", "defer"})
# The legal-frame feasibility decision LADDER (operator): each rung is strictly
# stronger; only the top rung at contest authority can move the pointer.
FEASIBILITY_DECISIONS = ("projection_only", "scorer_effect", "byte_real", "exact_candidate")


class SchemaError(ValueError):
    """Raised when a row violates the schema contract."""


def recompute_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """The ONLY authoritative score formula (evaluate.py). Never trust a stored
    ``final_score``; recompute it here."""
    if d_seg < 0 or d_pose < 0 or archive_bytes < 0:
        raise SchemaError("score components must be non-negative")
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / RATE_DENOM


@dataclass(frozen=True)
class ScorerQuotientCandidateRow:
    """``scorer_quotient_candidate_row.v1`` — the unified mover row."""

    lever_id: str
    candidate_kind: str
    base_archive_sha256: str
    bytes_before: int
    bytes_after: int
    d_seg_before: float
    d_seg_after: float
    d_pose_before: float
    d_pose_after: float
    authority_tier: str
    metric_family: str
    decision: str
    repaired_flips: int | None = None
    new_bad_flips: int | None = None
    pose_side_effect: float | None = None
    runtime_seconds: float | None = None
    first_failed_surface: str | None = None
    schema: str = "scorer_quotient_candidate_row.v1"

    # derived (computed in __post_init__ via object.__setattr__ since frozen)
    score_before: float = field(default=0.0)
    score_after: float = field(default=0.0)
    archive_bytes_delta: int = field(default=0)
    delta_score_total: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise SchemaError(f"unknown candidate_kind={self.candidate_kind!r}")
        if self.authority_tier not in AUTHORITY_TIERS:
            raise SchemaError(f"unknown authority_tier={self.authority_tier!r}")
        if self.metric_family not in METRIC_FAMILIES:
            raise SchemaError(f"unknown metric_family={self.metric_family!r}")
        if self.decision not in CANDIDATE_DECISIONS:
            raise SchemaError(f"unknown decision={self.decision!r}")
        sb = recompute_score(self.d_seg_before, self.d_pose_before, self.bytes_before)
        sa = recompute_score(self.d_seg_after, self.d_pose_after, self.bytes_after)
        object.__setattr__(self, "score_before", sb)
        object.__setattr__(self, "score_after", sa)
        object.__setattr__(self, "archive_bytes_delta", self.bytes_after - self.bytes_before)
        object.__setattr__(self, "delta_score_total", sa - sb)

    @property
    def net_repaired_flips(self) -> int | None:
        """The HONEST repair count: repaired - new_bad (the repaired count alone
        lies, per the #55 boundary-solver lesson)."""
        if self.repaired_flips is None or self.new_bad_flips is None:
            return None
        return self.repaired_flips - self.new_bad_flips

    @property
    def pointer_update_eligible(self) -> bool:
        """THE FIREWALL: promote ONLY a contest-tier exact-evaluate row that
        recomputes to a strictly lower score. Advisory/proxy NEVER promote."""
        return (
            self.authority_tier in _PROMOTING_AUTHORITY
            and self.metric_family == "exact_evaluate"
            and self.delta_score_total < 0.0
        )


@dataclass(frozen=True)
class LegalFrameFeasibilityTrace:
    """``legal_frame_feasibility_trace.v1`` — the #73-specific trace that
    enforces the Dykstra caveat: projection residual is a SURROGATE; only an
    ``exact_candidate`` at contest authority is a real pointer move."""

    base_candidate: str
    basis_used: str
    constraints_projected: tuple[str, ...]
    projection_residual: float
    margin_constraints_satisfied: bool
    pose_tube_surrogate_satisfied: bool
    exact_d_seg_after: float | None
    exact_d_pose_after: float | None
    bytes_estimate: int | None
    decision: str
    new_bad_flips: int | None = None
    runtime_seconds: float | None = None
    schema: str = "legal_frame_feasibility_trace.v1"

    def __post_init__(self) -> None:
        if self.decision not in FEASIBILITY_DECISIONS:
            raise SchemaError(f"unknown feasibility decision={self.decision!r}")
        # ladder integrity: a higher rung REQUIRES the lower rungs' evidence.
        rung = FEASIBILITY_DECISIONS.index(self.decision)
        # scorer_effect+ requires exact scorer measurement (projection residual
        # alone is a surrogate, not a scorer effect).
        if rung >= 1 and (self.exact_d_seg_after is None or self.exact_d_pose_after is None):
            raise SchemaError(
                "decision >= scorer_effect requires exact_d_seg_after + exact_d_pose_after "
                "(projection residual alone is a surrogate, not a scorer effect)"
            )
        if rung >= 2 and self.bytes_estimate is None:  # byte_real+ requires bytes
            raise SchemaError("decision >= byte_real requires bytes_estimate")

    @property
    def is_exact_candidate(self) -> bool:
        """Only the top rung is a contest candidate; everything below is a
        surrogate / not-yet-byte-real."""
        return self.decision == "exact_candidate"


def rank_candidates(rows: list[ScorerQuotientCandidateRow]) -> list[ScorerQuotientCandidateRow]:
    """Rank by recomputed ΔS (most negative first). Promotion-eligibility is a
    SEPARATE gate (``pointer_update_eligible``) — advisory rows may rank high
    for prioritization but never promote."""
    return sorted(rows, key=lambda r: r.delta_score_total)


def promotable(rows: list[ScorerQuotientCandidateRow]) -> list[ScorerQuotientCandidateRow]:
    """The rows that may actually move the canonical frontier pointer."""
    return [r for r in rank_candidates(rows) if r.pointer_update_eligible]
