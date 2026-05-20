# SPDX-License-Identifier: MIT
"""UncertaintyAwareCandidateRow — typed wrapper around CandidateRow.

Per SLOT MG-1 scope-discipline: rather than mutate
``tools/cathedral_autopilot_autonomous_loop.py::CandidateRow`` (which sister
slots MG-2/MG-3/MG-4/MG-5 may be editing), this module defines a typed
wrapper that EMBEDS a CandidateRow-shaped object + the SLOT MG-1 new
fields. The cathedral autopilot's auto-discovery loop per Catalog #335
ingests this consumer module and applies the risk-adjusted ranking via
the canonical ``consume_candidate`` Protocol surface.

New fields per SLOT MG-1 spec:
    predicted_delta_uncertainty: float | None  (1-sigma posterior std)
    n_anchors_consumed: int                    (anchors that fed prediction)
    evidence_grade: str                         (canonical Provenance taxonomy)
    last_updated_utc: str                       (ISO timestamp)

The wrapper does NOT modify the upstream CandidateRow; it carries the
SLOT MG-1 fields alongside. Operators / sister consumers that need the
risk-adjusted variant import this module; everything else operates on the
unwrapped CandidateRow as before. Per CLAUDE.md "Subagent coherence-by-
default" anti-fragmentation primitive.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any


# Canonical evidence grades per ProvenanceEvidenceGrade enum
# (tac.provenance.contract.ProvenanceEvidenceGrade values).
# Sister of the canonical Provenance contract per Catalog #323.
CANONICAL_EVIDENCE_GRADES = frozenset(
    {
        "promotable_exact_contest_cuda",
        "promotable_exact_contest_cpu",
        "predicted",
        "empirical_cpu_non_gha",
        "macos_cpu_advisory",
        "mps_proxy",
        "research_only",
        "invalid_byte_identity_artifact",
        # Additional canonical extensions for consumer rows:
        "predicted_with_proxy",  # predicted-from-model + macOS-CPU advisory hint
        "predicted_pure_prior",  # no anchors yet; pure prior
    }
)


def _utc_now_iso() -> str:
    """Canonical UTC timestamp with trailing Z."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class UncertaintyAwareCandidateRow:
    """Typed wrapper carrying SLOT MG-1 uncertainty quantification fields.

    Embeds the upstream CandidateRow (or any candidate-shaped object) and
    adds the 4 SLOT MG-1 fields. Frozen by design per CLAUDE.md "Beauty,
    simplicity, and developer experience" + Catalog #110/#113 HISTORICAL_
    PROVENANCE APPEND-ONLY discipline (immutable; updates require new
    construction).

    Fields:
        candidate: The wrapped candidate-shaped object (typically a
            CandidateRow). Must expose at minimum:
              - ``candidate_id: str``
              - ``predicted_score_delta: float``
              - ``estimated_dispatch_cost_usd: float``
            so the risk-adjusted ranker can read it without depending on
            the full CandidateRow contract.
        predicted_delta_uncertainty: 1-sigma posterior std of the
            predicted_score_delta. None ONLY when no canonical equation
            anchor exists for this candidate's family; in that case the
            ranker treats it as maximally uncertain (equivalent to
            pure-prior NIG output per uncertainty.py).
        n_anchors_consumed: number of EmpiricalAnchor rows that fed the
            uncertainty estimate. 0 means pure-prior; 1+ means posterior-
            updated.
        evidence_grade: one of CANONICAL_EVIDENCE_GRADES. Determines whether
            this candidate is eligible for promotion to a contest score
            claim per Catalog #127 + Catalog #323.
        last_updated_utc: ISO-UTC timestamp of when the uncertainty
            estimate was last refreshed. Sister consumers consume this
            for staleness detection per CLAUDE.md "Required durable state".
    """

    candidate: Any  # CandidateRow-shaped; duck-typed for sister isolation
    predicted_delta_uncertainty: float | None
    n_anchors_consumed: int
    evidence_grade: str
    last_updated_utc: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        # candidate must expose the minimum 3 fields for ranker compatibility.
        for required in ("candidate_id", "predicted_score_delta", "estimated_dispatch_cost_usd"):
            if not hasattr(self.candidate, required):
                raise ValueError(
                    f"candidate missing required attribute '{required}'; "
                    f"got {type(self.candidate).__name__}"
                )

        if self.predicted_delta_uncertainty is not None:
            if not isinstance(self.predicted_delta_uncertainty, (int, float)):
                raise ValueError(
                    f"predicted_delta_uncertainty must be numeric or None, "
                    f"got {type(self.predicted_delta_uncertainty).__name__}"
                )
            if self.predicted_delta_uncertainty != self.predicted_delta_uncertainty:  # NaN
                raise ValueError("predicted_delta_uncertainty must not be NaN")
            if self.predicted_delta_uncertainty < 0.0:
                raise ValueError(
                    f"predicted_delta_uncertainty must be >= 0, "
                    f"got {self.predicted_delta_uncertainty}"
                )

        if not isinstance(self.n_anchors_consumed, int) or self.n_anchors_consumed < 0:
            raise ValueError("n_anchors_consumed must be a non-negative int")

        if not isinstance(self.evidence_grade, str) or not self.evidence_grade:
            raise ValueError("evidence_grade must be a non-empty string")
        if self.evidence_grade not in CANONICAL_EVIDENCE_GRADES:
            raise ValueError(
                f"evidence_grade {self.evidence_grade!r} not in canonical taxonomy; "
                f"valid: {sorted(CANONICAL_EVIDENCE_GRADES)}"
            )

        if not isinstance(self.last_updated_utc, str) or not self.last_updated_utc:
            raise ValueError("last_updated_utc must be a non-empty ISO-UTC string")

    @property
    def candidate_id(self) -> str:
        """Pass-through for ranker key extraction."""
        return self.candidate.candidate_id

    @property
    def predicted_score_delta(self) -> float:
        """Pass-through for ranker base score."""
        return float(self.candidate.predicted_score_delta)

    @property
    def estimated_dispatch_cost_usd(self) -> float:
        """Pass-through for ranker cost-per-eig calculation."""
        return float(self.candidate.estimated_dispatch_cost_usd)
