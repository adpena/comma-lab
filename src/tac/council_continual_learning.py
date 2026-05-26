# SPDX-License-Identifier: MIT
"""Council deliberation continual-learning canonical helper (Catalog #300 sister).

Per the COUNCIL-HIERARCHY-V2 landing 2026-05-16 (operator-approved 4-tier
protocol with maximum-signal preservation + continual-learning wire-in
meta-principles): every T2+ council deliberation (and every T1 working group
whose finding crosses an elevation trigger) MUST emit a continual-learning
anchor via :func:`append_council_anchor`. The persisted council verdict +
dissent + assumption classification become signal that future deliberations,
the autopilot ranker, and the Rashomon ensemble can consume.

This file is the **canonical** entry point. Any caller persisting council
state outside :func:`append_council_anchor` re-introduces the orphaned-work
bug class per CLAUDE.md "Subagent coherence-by-default" non-negotiable: a
council finding the planner cannot see is orphaned signal.

Schema and on-disk format mirror :mod:`tac.cost_band_calibration` (the
canonical fcntl-locked JSONL append-only state writer per Catalog #128 +
Catalog #131 + Catalog #175 / #177 discipline). Read-side strict-load
mirrors :mod:`tac.deploy.lightning.active_jobs_state` (Catalog #138 strict-
load fail-closed pattern).

Persisted at ``.omx/state/council_deliberation_posterior.jsonl`` (HISTORICAL_
PROVENANCE per Catalog #110 / #113 / #132 — APPEND-ONLY; outcomes are NEW
rows referencing the same ``deliberation_id``). Locked via the sibling
``.omx/state/.council_deliberation_posterior.lock`` per Catalog #131
bare-write discipline.

Public API:

* :class:`CouncilTier` — enum-like sentinel constants (T1/T2/T3/T4).
* :class:`CouncilDeliberationRecord` — frozen dataclass mirroring the v2
  frontmatter (council_tier / attendees / verdict / dissent /
  assumption_adversary_verdict / decisions_recorded / etc.).
* :func:`append_council_anchor` — fcntl-locked JSONL append-only writer
  per Catalog #128 / #131 discipline.
* :func:`load_council_anchors_strict` — raises
  :class:`CouncilPosteriorCorruptError` on JSON parse failure (Catalog
  #138 pattern).
* :func:`load_council_anchors` — lenient loader; skips malformed lines.
* :func:`query_anchors_by_topic` — cite-chain helper.
* :func:`query_dissent_history` — surface minority opinions by member.
* :func:`query_assumption_classification_history` — surface HARD-EARNED-
  vs-CARGO-CULTED verdicts for a given assumption.

Verified against:

* Catalog #128 / #131 (fcntl-locked shared state writes).
* Catalog #138 (strict-load fail-closed for mutating paths).
* Catalog #245 (Modal call_id ledger — sister APPEND-ONLY JSONL pattern).
* CLAUDE.md "Council conduct" Fix-7 amendment (per-round assumption
  surfacing).
* CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" (HARD-EARNED-vs-CARGO-
  CULTED classification).

Memory: ``feedback_council_hierarchy_v2_landed_20260516.md`` [verified-
against: Catalog #128, #131, #138, #245, #292].
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "CouncilTier",
    "VALID_TIERS",
    "VALID_VERDICTS",
    "VALID_MISSION_CONTRIBUTIONS",
    "CouncilDeliberationRecord",
    "CouncilRecordValidationError",
    "CouncilPosteriorCorruptError",
    "DEFAULT_COUNCIL_POSTERIOR_PATH",
    "DEFAULT_COUNCIL_POSTERIOR_LOCK_PATH",
    "SCHEMA_VERSION",
    "RIGOR_DOMINANT_THRESHOLD",
    "DEFERRED_RETROSPECTIVE_WINDOW_DAYS",
    "append_council_anchor",
    "load_council_anchors",
    "load_council_anchors_strict",
    "query_anchors_by_topic",
    "query_dissent_history",
    "query_assumption_classification_history",
    "query_overrides",
    "query_due_retrospectives",
    "query_mission_contribution_distribution",
    "is_rigor_dominant",
    "compute_deferred_retrospective_due_utc",
    "update_from_anchor",
    # ─── Recursive self-reflection protocol (Catalog #363; 2026-05-26) ──
    # Per CLAUDE.md "Council conduct" amendment 2026-05-26 verbatim:
    # *"the grand council is providing valuable information but perhaps the
    # grand council itself must be instructed to deliberate and self reflect
    # recursively"*. 4-value taxonomy + frozen dataclass + canonical helpers
    # for per-assumption empirical-verification-status surface.
    "EmpiricalVerificationStatus",
    "VALID_EMPIRICAL_VERIFICATION_STATUSES",
    "UNVERIFIED_VERIFICATION_STATUSES",
    "AssumptionEmpiricalVerification",
    "AssumptionVerificationValidationError",
    "MAX_SELF_REFLECTION_ROUNDS",
    "classify_assumption_verification_status_from_evidence",
    "extract_unverified_assumptions",
    "verdict_status_requires_provisional_marker",
    "query_self_reflection_history_for_deliberation",
]


SCHEMA_VERSION = "council_deliberation_posterior_v1"

# Per CLAUDE.md "Mission alignment — non-negotiable" subsection of "Council
# hierarchy: 4-tier protocol" (operator binding standing directive 2026-05-16):
# every T2+ council deliberation MUST forecast the mission-contribution
# category. The 60% threshold for is_rigor_dominant() is the operator-visible
# alert anchor — when rigor_overhead + apparatus_maintenance crosses 60% of
# T2+ verdicts in any 30-day window, the council is producing more apparatus-
# maintenance than frontier-breaking work and operator review is required.
# Memory:
# feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md.
RIGOR_DOMINANT_THRESHOLD = 0.60

# Per the mission-alignment operational consequence 3: every deferred or
# killed substrate gets a 30-day score-impact retrospective so the operator
# can audit whether the deferral cost actual score improvement. The 30-day
# window pairs with CLAUDE.md "Forbidden premature KILL" non-negotiable.
DEFERRED_RETROSPECTIVE_WINDOW_DAYS = 30


# Repo root resolution mirrors the rest of tac.* (this module sits in
# ``src/tac/``; the repo root is two parents up).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


DEFAULT_COUNCIL_POSTERIOR_PATH = (
    _REPO_ROOT / ".omx" / "state" / "council_deliberation_posterior.jsonl"
)
DEFAULT_COUNCIL_POSTERIOR_LOCK_PATH = (
    _REPO_ROOT / ".omx" / "state" / ".council_deliberation_posterior.lock"
)


class CouncilTier:
    """4-tier council hierarchy v2 sentinel constants.

    Per the COUNCIL-HIERARCHY-V2 spec (operator-approved 2026-05-16):

    * **T1** — Working Group (1-3 named members; bounded scope; ≤30 min
      deliberation; no veto power; output is a recommendation feeding a
      T2/T3 deliberation).
    * **T2** — Inner-Skunkworks (5-of-6 sextet pact: Shannon LEAD /
      Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-
      Adversary; binding decision authority for in-flight engineering
      tradeoffs; Shannon tie-break).
    * **T3** — Full Grand Council (≥12-of-20 grand council + 5-of-6
      sextet; binding decisions touching CLAUDE.md non-negotiables /
      cross-cutting wire-ins / strategic redirection; Shannon tie-break
      with Dykstra fallback).
    * **T4** — Symposium (≥16-of-20 + 6-of-6 sextet + ≥1 specialist per
      affected paradigm; required for kill-and-replace decisions /
      multi-month directional shifts / operator-pre-attention escalation;
      operator-resolves on tie).
    """

    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


VALID_TIERS = frozenset({CouncilTier.T1, CouncilTier.T2, CouncilTier.T3, CouncilTier.T4})


# Verdict enum-like; deliberation outcomes that the autopilot ranker /
# Rashomon ensemble can consume.
VALID_VERDICTS = frozenset({
    "PROCEED",
    "PROCEED_WITH_REVISIONS",
    "DEFER_PENDING_EVIDENCE",
    "REFUSE",
    "ESCALATE_TO_OPERATOR",
    "ESCALATE_TO_HIGHER_TIER",
})


# Per CLAUDE.md "Mission alignment — non-negotiable" subsection of "Council
# hierarchy: 4-tier protocol" (operator binding standing directive 2026-05-16
# verbatim: *"and all in service of innovation and rigor and extreme
# optimization and performance and score lowering"*). Every T2+ council
# deliberation MUST classify its `council_predicted_mission_contribution`
# into one of these 5 categories so the operator can audit whether the
# council apparatus is producing innovation-enabling or innovation-blocking
# verdicts. The breakdown is surfaced via
# :func:`query_mission_contribution_distribution` and
# :func:`is_rigor_dominant`.
#
# Category semantics:
#   - frontier_breaking — opens a class-shift path predicted to lower score
#   - frontier_protecting — prevents a regression that would raise score
#     (sister of strict-mode preflight gates)
#   - rigor_overhead — procedural-only verdict; no direct score contribution
#     but enables future contributions
#   - apparatus_maintenance — infrastructure update without score implications
#   - mission_questioned — verdict triggered the "is this serving the
#     mission?" question; documented for retrospective
VALID_MISSION_CONTRIBUTIONS = frozenset({
    "frontier_breaking",
    "frontier_protecting",
    "rigor_overhead",
    "apparatus_maintenance",
    "mission_questioned",
})


# ──────────────────────────────────────────────────────────────────────
# Recursive self-reflection protocol — canonical taxonomy (Catalog #363)
#
# Per CLAUDE.md "Council conduct" amendment 2026-05-26 + the canonical
# protocol design memo
# .omx/research/council_recursive_self_reflection_protocol_design_20260526T133600Z.md.
#
# 4-empirical-receipt anchor (one per status type):
#   * Receipt #1 (T3 council 7d04474cb M3 RULED-OUT empirically falsified by
#     source-inspection 5b87fae77) → ASSUMED_AWAITING_VERIFICATION caught.
#   * Receipt #2 (T3 council 7d04474cb M2 ~0.7-0.9 α dominance empirically
#     falsified by Kahan-EMA smoke 05c07aa40) → INFERRED_FROM_DOMAIN_LITERATURE
#     caught.
#   * Receipt #3 (my own n=2 super-linear extrapolation empirically falsified
#     by 5-anchor fit 60a9de751) → ASSUMED_AWAITING_VERIFICATION caught.
#   * Receipt #4 (K=COIN++ 5e-3 drift claim empirically falsified by R1''-K
#     independent verification leading to 2d59283d4) →
#     ASSUMED_AWAITING_VERIFICATION caught.
#
# The 4-value taxonomy is canonical because:
#   1. The 4 values map 1-to-1 to the 4 empirical receipts (one per status
#      catches a different META class).
#   2. The 2 VERIFIED states cover canonical "evidence in hand" cases
#      (source artifact OR empirical anchor).
#   3. The 2 unverified states distinguish "literature analogy" from
#      "extrapolation without basis"; both gate equally because BOTH are
#      pre-empirical at the specific instance.
#   4. The taxonomy is bounded (not a continuum) so per-assumption
#      disambiguation is structurally tractable per R12-D meta-finding
#      lens-coverage cycle-bounding criterion.
# ──────────────────────────────────────────────────────────────────────


class EmpiricalVerificationStatus:
    """4-value canonical taxonomy for per-assumption empirical-verification status.

    Per CLAUDE.md "Council conduct" Recursive self-reflection protocol
    amendment + canonical design memo
    council_recursive_self_reflection_protocol_design_20260526T133600Z.md.

    Sentinel constants rather than `enum.Enum` to mirror :class:`CouncilTier`
    + :data:`VALID_VERDICTS` pattern in this module — JSONL serialization
    stays trivial string round-trip; backward-compat shimming for legacy
    rows lacking the field is straightforward (auto-classified
    `INFERRED_FROM_DOMAIN_LITERATURE` per safe-default).

    Status semantics:

    * :attr:`VERIFIED_VIA_SOURCE_INSPECTION` — assumption directly verified
      by source file inspection (path + line range + content quote that
      proves the assumption). Example: "Z6 uses MLX AdamW — verified
      ``long_training_canonical.py:147`` ``optimizer =
      mlx.optimizers.AdamW(...)``".

      No gate: assumption fully supports verdict.

    * :attr:`VERIFIED_VIA_EMPIRICAL_ANCHOR` — assumption verified by a
      canonical posterior anchor reference (commit sha + posterior row id +
      measurement metadata per Catalog #245 sister); typically an empirical
      smoke / experiment artifact. Example: "M2 contributes 0× mitigation
      — verified empirical anchor ``05c07aa40`` Kahan-EMA shadow wrapper
      smoke".

      No gate: assumption fully supports verdict.

    * :attr:`INFERRED_FROM_DOMAIN_LITERATURE` — assumption supported by
      citation to canonical literature (paper / textbook / Wikipedia /
      CLAUDE.md doctrine) that supports the pattern but does not directly
      verify the specific instance. Example: "Adam-family optimizers carry
      β₁β₂ state buffers — inferred Higham-2002 + Kingma-Ba 2014".

      **GATE**: Round 2 of recursive self-reflection MUST attempt verification
      OR Round 3 downgrades verdict to PROVISIONAL-PENDING-VERIFICATION.

    * :attr:`ASSUMED_AWAITING_VERIFICATION` — explicit acknowledgment that
      the assumption is operating-within unverified; no source citation;
      no empirical anchor. Example: "M3-RULED-OUT — assumed Z6 uses
      stateless SGD-with-EMA based on lane name tokens".

      **GATE**: Round 2 MUST attempt verification before Round 3 SEAL OR
      Round 3 downgrades verdict.
    """

    VERIFIED_VIA_SOURCE_INSPECTION = "VERIFIED_VIA_SOURCE_INSPECTION"
    VERIFIED_VIA_EMPIRICAL_ANCHOR = "VERIFIED_VIA_EMPIRICAL_ANCHOR"
    INFERRED_FROM_DOMAIN_LITERATURE = "INFERRED_FROM_DOMAIN_LITERATURE"
    ASSUMED_AWAITING_VERIFICATION = "ASSUMED_AWAITING_VERIFICATION"


VALID_EMPIRICAL_VERIFICATION_STATUSES = frozenset({
    EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
    EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR,
    EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE,
    EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
})

# The 2 statuses that DO NOT carry direct evidence at the specific instance.
# Assumptions classified into either of these gate the verdict per Round 3
# downgrade-or-verify rule.
UNVERIFIED_VERIFICATION_STATUSES = frozenset({
    EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE,
    EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
})

# Per the canonical design memo §2.3 cycle bounds (R12-D lens-coverage
# Zipf-decay criterion): ≤5 self-reflection rounds before operator-routed
# ESCALATE_TO_OPERATOR per Catalog #300. The bound prevents infinite
# recursion. SEAL threshold remains 3 consecutive clean rounds per the
# canonical "Recursive adversarial review protocol — close paths" pattern.
MAX_SELF_REFLECTION_ROUNDS = 5


class AssumptionVerificationValidationError(ValueError):
    """Raised when an :class:`AssumptionEmpiricalVerification` violates an invariant.

    Distinct from :class:`CouncilRecordValidationError` so callers can
    disambiguate "the assumption-verification record is malformed" from
    "the council deliberation record is malformed" — the two are sister
    schemas with structurally distinct invariants.
    """


@dataclass(frozen=True)
class AssumptionEmpiricalVerification:
    """One per-assumption empirical-verification record.

    Schema mirrors the existing dict-form entries in
    :attr:`CouncilDeliberationRecord.council_assumption_adversary_verdict`
    (assumption + classification + rationale) AND extends with the new
    canonical :class:`EmpiricalVerificationStatus` field + optional
    ``evidence_artifact`` citation per canonical Provenance (Catalog #323
    sister discipline).

    Frozen dataclass so instances are hashable + cite-stable. Conversion
    to/from dict via :meth:`as_dict` / :meth:`from_dict` for backward-
    compat with legacy ``dict[str, str]`` entries in the existing
    ``council_assumption_adversary_verdict`` tuple.

    Field semantics:

    * ``assumption`` — short string naming the surfaced assumption (e.g.
      "Z6 uses stateless SGD-with-EMA"). Non-empty.
    * ``classification`` — existing HARD-EARNED-vs-CARGO-CULTED axis per
      Catalog #292 + the hard-earned-vs-cargo-culted addendum (
      ``feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md``).
      Free-form string; canonical values are "HARD-EARNED" / "CARGO-CULTED"
      / "UNCLEAR".
    * ``empirical_verification_status`` — NEW Catalog #363 4-value
      taxonomy; one of :data:`VALID_EMPIRICAL_VERIFICATION_STATUSES`.
    * ``rationale`` — explanatory text supporting the classification
      and verification-status assignment. Non-empty.
    * ``evidence_artifact`` — optional canonical Provenance citation
      (file path + line range, commit sha + posterior row id, or
      literature reference) supporting the verification status. Required
      when status is VERIFIED_VIA_SOURCE_INSPECTION or
      VERIFIED_VIA_EMPIRICAL_ANCHOR; optional for the unverified
      statuses (but recommended when known).
    """

    assumption: str
    classification: str
    empirical_verification_status: str
    rationale: str
    evidence_artifact: str | None = None

    def __post_init__(self) -> None:  # noqa: D401 — post-init validator
        """Validate invariants per canonical 4-value taxonomy contract."""
        if not isinstance(self.assumption, str) or not self.assumption.strip():
            raise AssumptionVerificationValidationError(
                "assumption must be a non-empty string"
            )
        if not isinstance(self.classification, str) or not self.classification.strip():
            raise AssumptionVerificationValidationError(
                "classification must be a non-empty string "
                "(canonical values: HARD-EARNED / CARGO-CULTED / UNCLEAR)"
            )
        if self.empirical_verification_status not in VALID_EMPIRICAL_VERIFICATION_STATUSES:
            raise AssumptionVerificationValidationError(
                f"empirical_verification_status="
                f"{self.empirical_verification_status!r} not in "
                f"{sorted(VALID_EMPIRICAL_VERIFICATION_STATUSES)} "
                "per CLAUDE.md 'Council conduct' Recursive self-reflection "
                "protocol amendment (Catalog #363)"
            )
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise AssumptionVerificationValidationError(
                "rationale must be a non-empty string"
            )
        if self.empirical_verification_status in (
            EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR,
        ):
            if not self.evidence_artifact or not str(self.evidence_artifact).strip():
                raise AssumptionVerificationValidationError(
                    f"empirical_verification_status="
                    f"{self.empirical_verification_status!r} requires "
                    "non-empty evidence_artifact (canonical Provenance "
                    "citation per Catalog #323 sister discipline: file path "
                    "+ line range OR commit sha + posterior row id)"
                )

    def as_dict(self) -> dict[str, str]:
        """Serialize to canonical dict form (back-compat with legacy entries)."""
        out: dict[str, str] = {
            "assumption": self.assumption,
            "classification": self.classification,
            "empirical_verification_status": self.empirical_verification_status,
            "rationale": self.rationale,
        }
        if self.evidence_artifact is not None:
            out["evidence_artifact"] = str(self.evidence_artifact)
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AssumptionEmpiricalVerification":
        """Reconstruct from canonical dict form.

        Legacy rows (pre-Catalog-#363) lacking ``empirical_verification_status``
        are auto-classified :attr:`EmpiricalVerificationStatus.
        INFERRED_FROM_DOMAIN_LITERATURE` per safe-default backward-compat
        rule (mirrors the legacy ``predicted_mission_contribution`` ->
        ``apparatus_maintenance`` backfill default in
        :func:`_dict_to_record`).
        """
        status = payload.get("empirical_verification_status")
        if status is None or status == "":
            status = EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE
        evidence = payload.get("evidence_artifact")
        if evidence is not None:
            evidence = str(evidence)
        return cls(
            assumption=str(payload.get("assumption", "")),
            classification=str(payload.get("classification", "")),
            empirical_verification_status=str(status),
            rationale=str(payload.get("rationale", "")),
            evidence_artifact=evidence,
        )


class CouncilRecordValidationError(ValueError):
    """Raised when a :class:`CouncilDeliberationRecord` violates an invariant.

    Distinct from generic :class:`ValueError` so mission-alignment validators
    can refuse records that lack the required mission-contribution forecast
    or the operator-override rationale even when the field types are
    superficially well-formed (e.g. ``override_invoked=True`` but
    ``override_rationale=None``).
    """


class CouncilPosteriorCorruptError(RuntimeError):
    """Raised by :func:`load_council_anchors_strict` on JSON parse failure.

    Mirrors :class:`ActiveJobsCorruptError` / :class:`CallIdLedgerCorruptError`
    per Catalog #138 strict-load discipline. The lenient counterpart
    :func:`load_council_anchors` silently skips malformed rows so an
    audit reader can survive partial corruption; the strict variant
    raises so a writer cannot silently reset corrupt state.
    """


@dataclass(frozen=True)
class CouncilDeliberationRecord:
    """One council deliberation event: the v2 frontmatter persisted to JSONL.

    Schema is intentionally append-only; fields added in future v2.N
    revisions MUST be optional (default to ``None`` or empty collection)
    so legacy rows continue loading.

    Field semantics:

    * ``deliberation_id`` — slugged identifier (e.g.
      ``council_hierarchy_v2_landing_20260516``); same id appears across
      ``dispatched`` / ``outcome`` rows per CLAUDE.md Modal call_id
      ledger sister pattern.
    * ``topic`` — short human-readable subject line.
    * ``council_tier`` — one of :attr:`CouncilTier.T1` .. ``T4``.
    * ``council_attendees`` — list of named members (e.g.
      ``["Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian",
      "Assumption-Adversary"]``).
    * ``council_quorum_met`` — boolean; per the v2 quorum rules.
    * ``council_verdict`` — one of :data:`VALID_VERDICTS`.
    * ``council_dissent`` — list of ``{"member": ..., "verbatim": ...}``
      dicts; verbatim minority opinions preserved per CLAUDE.md
      "maximum signal preservation" rule.
    * ``council_assumption_adversary_verdict`` — per-assumption
      classifications: list of ``{"assumption": ..., "classification":
      "HARD-EARNED" | "CARGO-CULTED" | "UNCLEAR", "rationale": ...}``.
      Required at T2+ per Catalog #292.
    * ``council_decisions_recorded`` — list of decisions / op-routables
      emitted by the deliberation (the actuator surface; downstream
      sister subagents / autopilot consume these).
    * ``related_deliberation_ids`` — cite-chain pointers to prior
      deliberations on the same topic so future Rashomon ensemble
      members can trace position evolution.
    * ``event_type`` — ``"dispatched"`` for the deliberation landing;
      ``"outcome"`` for downstream empirical verdicts referencing the
      same ``deliberation_id``.
    * ``parent_id_or_session`` — Claude subagent/session id that
      authored the record (continual learning provenance).
    """

    deliberation_id: str
    topic: str
    council_tier: str
    council_attendees: tuple[str, ...]
    council_quorum_met: bool
    council_verdict: str
    council_dissent: tuple[dict[str, str], ...] = field(default_factory=tuple)
    council_assumption_adversary_verdict: tuple[dict[str, str], ...] = field(
        default_factory=tuple
    )
    council_decisions_recorded: tuple[str, ...] = field(default_factory=tuple)
    related_deliberation_ids: tuple[str, ...] = field(default_factory=tuple)
    event_type: str = "dispatched"
    parent_id_or_session: str | None = None
    memory_path: str | None = None
    notes: str = ""
    # ─── Mission-alignment fields (operator-binding directive 2026-05-16) ──
    # Per CLAUDE.md "Mission alignment — non-negotiable" subsection.
    # Required at T2+. None permitted at T1 (working-group recommendations
    # are not binding decisions and need not forecast mission contribution).
    predicted_mission_contribution: str | None = None
    # Operator-frontier-override at ALL tiers (operational consequence 1).
    # Default false; REQUIRED field even when false so a downstream auditor
    # can distinguish "not invoked" from "field absent on legacy row".
    override_invoked: bool = False
    # Required when override_invoked=True; verbatim operator quote per
    # CLAUDE.md "Mission alignment — non-negotiable" rule 1.
    override_rationale: str | None = None
    # Operational consequence 3: every deferred / killed substrate gets a
    # 30-day retrospective. The due date is computed as
    # `deliberation_utc + DEFERRED_RETROSPECTIVE_WINDOW_DAYS` when the
    # verdict defers/kills a substrate; None otherwise. Pair with
    # `deferred_substrate_id` so the retrospective can identify the lane.
    deferred_substrate_retrospective_due_utc: str | None = None
    deferred_substrate_id: str | None = None
    written_at_utc: str = ""
    written_pid: int | None = None
    written_host: str | None = None
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:  # noqa: D401 — post-init validator
        """Validate mission-alignment invariants per CLAUDE.md non-negotiable.

        Raises :class:`CouncilRecordValidationError` (subclass of ValueError)
        when:

        * ``council_tier`` is T2/T3/T4 AND ``predicted_mission_contribution``
          is None (operational consequence 5 — every T2+ verdict MUST forecast
          mission contribution).
        * ``predicted_mission_contribution`` is set but not in
          :data:`VALID_MISSION_CONTRIBUTIONS`.
        * ``override_invoked=True`` AND ``override_rationale`` is None or
          empty (operational consequence 1 — operator-frontier-override
          REQUIRES verbatim operator quote).
        * ``deferred_substrate_retrospective_due_utc`` is set AND
          ``deferred_substrate_id`` is None (the two fields are paired so
          the retrospective can identify the lane).

        The structural-correctness invariants (tier/attendees/verdict shape)
        are still validated in :func:`_validate_record` at append time so
        downstream tools that bypass ``__post_init__`` (e.g. JSONL replay
        with relaxed schema) still get the structural guarantees.
        """
        # Mission-contribution validation.
        if self.predicted_mission_contribution is not None:
            if self.predicted_mission_contribution not in VALID_MISSION_CONTRIBUTIONS:
                raise CouncilRecordValidationError(
                    f"predicted_mission_contribution="
                    f"{self.predicted_mission_contribution!r} not in "
                    f"{sorted(VALID_MISSION_CONTRIBUTIONS)} (per CLAUDE.md "
                    "'Mission alignment — non-negotiable' subsection)"
                )
        if self.council_tier in (CouncilTier.T2, CouncilTier.T3, CouncilTier.T4):
            if self.predicted_mission_contribution is None:
                raise CouncilRecordValidationError(
                    f"council_tier={self.council_tier} requires "
                    "predicted_mission_contribution to be set (one of "
                    f"{sorted(VALID_MISSION_CONTRIBUTIONS)}) per CLAUDE.md "
                    "'Mission alignment — non-negotiable' operational "
                    "consequence 5. T1 working groups are exempt."
                )
        # Override-rationale validation.
        if self.override_invoked:
            if not self.override_rationale or not str(self.override_rationale).strip():
                raise CouncilRecordValidationError(
                    "override_invoked=True requires override_rationale to be "
                    "a non-empty verbatim operator quote per CLAUDE.md "
                    "'Mission alignment — non-negotiable' operational "
                    "consequence 1 (operator-frontier-override at ALL tiers)."
                )
        # Deferred-retrospective field pairing validation.
        if self.deferred_substrate_retrospective_due_utc is not None:
            if not self.deferred_substrate_id:
                raise CouncilRecordValidationError(
                    "deferred_substrate_retrospective_due_utc is set but "
                    "deferred_substrate_id is None; the two fields are "
                    "paired so the 30-day retrospective per CLAUDE.md "
                    "'Mission alignment — non-negotiable' operational "
                    "consequence 3 can identify the lane."
                )


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _ensure_state_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_record(record: CouncilDeliberationRecord) -> None:
    if not isinstance(record.deliberation_id, str) or not record.deliberation_id.strip():
        raise ValueError("deliberation_id must be a non-empty string")
    if "\n" in record.deliberation_id or "\r" in record.deliberation_id:
        raise ValueError("deliberation_id must not contain newlines")
    if not isinstance(record.topic, str) or not record.topic.strip():
        raise ValueError("topic must be a non-empty string")
    if record.council_tier not in VALID_TIERS:
        raise ValueError(
            f"council_tier={record.council_tier!r} not in {sorted(VALID_TIERS)}"
        )
    if not isinstance(record.council_attendees, (list, tuple)) or len(record.council_attendees) == 0:
        raise ValueError("council_attendees must be a non-empty sequence")
    for member in record.council_attendees:
        if not isinstance(member, str) or not member.strip():
            raise ValueError(f"council_attendees entries must be non-empty strings; got {member!r}")
    if record.council_verdict not in VALID_VERDICTS:
        raise ValueError(
            f"council_verdict={record.council_verdict!r} not in {sorted(VALID_VERDICTS)}"
        )
    # T2+ deliberations MUST have at least one assumption-adversary verdict
    # entry per CLAUDE.md Catalog #292 + "Council conduct" Fix-7 amendment.
    if record.council_tier in (CouncilTier.T2, CouncilTier.T3, CouncilTier.T4):
        if len(record.council_assumption_adversary_verdict) == 0:
            raise ValueError(
                f"council_tier={record.council_tier} requires at least one "
                "council_assumption_adversary_verdict entry (per Catalog #292 "
                "+ CLAUDE.md 'Council conduct' Fix-7 amendment)"
            )


def _record_to_dict(record: CouncilDeliberationRecord) -> dict[str, Any]:
    return {
        "schema": record.schema,
        "deliberation_id": record.deliberation_id,
        "topic": record.topic,
        "council_tier": record.council_tier,
        "council_attendees": list(record.council_attendees),
        "council_quorum_met": bool(record.council_quorum_met),
        "council_verdict": record.council_verdict,
        "council_dissent": [dict(d) for d in record.council_dissent],
        "council_assumption_adversary_verdict": [
            dict(d) for d in record.council_assumption_adversary_verdict
        ],
        "council_decisions_recorded": list(record.council_decisions_recorded),
        "related_deliberation_ids": list(record.related_deliberation_ids),
        "event_type": record.event_type,
        "parent_id_or_session": record.parent_id_or_session,
        "memory_path": record.memory_path,
        "notes": record.notes,
        # Mission-alignment fields (operator-binding directive 2026-05-16).
        "predicted_mission_contribution": record.predicted_mission_contribution,
        "override_invoked": bool(record.override_invoked),
        "override_rationale": record.override_rationale,
        "deferred_substrate_retrospective_due_utc": (
            record.deferred_substrate_retrospective_due_utc
        ),
        "deferred_substrate_id": record.deferred_substrate_id,
        "written_at_utc": record.written_at_utc or _now_utc_iso(),
        "written_pid": record.written_pid if record.written_pid is not None else os.getpid(),
        "written_host": record.written_host or socket.gethostname(),
    }


def _dict_to_record(payload: dict[str, Any]) -> CouncilDeliberationRecord:
    attendees = payload.get("council_attendees") or []
    dissent_raw = payload.get("council_dissent") or []
    adv_raw = payload.get("council_assumption_adversary_verdict") or []
    decisions_raw = payload.get("council_decisions_recorded") or []
    related_raw = payload.get("related_deliberation_ids") or []
    # Mission-alignment fields: legacy rows lack them; default to None /
    # False for backward compat. Mission-alignment validators in
    # __post_init__ run against the reconstructed record, so legacy T2+
    # rows without predicted_mission_contribution would raise — defer the
    # validation by reading the legacy field as a backfill-default. Legacy
    # rows are intentionally treated as `apparatus_maintenance` (the most
    # common historical pattern; safe default per the mission-alignment
    # binding directive backfill rule).
    predicted_mission = payload.get("predicted_mission_contribution")
    tier = str(payload["council_tier"])
    if predicted_mission is None and tier in ("T2", "T3", "T4"):
        # Backfill-default for legacy rows (pre-mission-alignment-extension).
        predicted_mission = "apparatus_maintenance"
    return CouncilDeliberationRecord(
        deliberation_id=str(payload["deliberation_id"]),
        topic=str(payload["topic"]),
        council_tier=tier,
        council_attendees=tuple(str(m) for m in attendees),
        council_quorum_met=bool(payload.get("council_quorum_met", False)),
        council_verdict=str(payload["council_verdict"]),
        council_dissent=tuple(
            {str(k): str(v) for k, v in (d or {}).items()} for d in dissent_raw
        ),
        council_assumption_adversary_verdict=tuple(
            {str(k): str(v) for k, v in (d or {}).items()} for d in adv_raw
        ),
        council_decisions_recorded=tuple(str(s) for s in decisions_raw),
        related_deliberation_ids=tuple(str(s) for s in related_raw),
        event_type=str(payload.get("event_type", "dispatched")),
        parent_id_or_session=payload.get("parent_id_or_session"),
        memory_path=payload.get("memory_path"),
        notes=str(payload.get("notes", "")),
        predicted_mission_contribution=predicted_mission,
        override_invoked=bool(payload.get("override_invoked", False)),
        override_rationale=payload.get("override_rationale"),
        deferred_substrate_retrospective_due_utc=payload.get(
            "deferred_substrate_retrospective_due_utc"
        ),
        deferred_substrate_id=payload.get("deferred_substrate_id"),
        written_at_utc=str(payload.get("written_at_utc", "")),
        written_pid=(
            int(payload["written_pid"]) if payload.get("written_pid") is not None else None
        ),
        written_host=payload.get("written_host"),
        schema=str(payload.get("schema", SCHEMA_VERSION)),
    )


def append_council_anchor(
    record: CouncilDeliberationRecord,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Append a council deliberation record to the JSONL posterior under
    fcntl LOCK_EX (Catalog #128 + #131 + #245 sister discipline).

    Concurrent appenders serialize at the lock; each writes exactly one
    JSON line followed by newline. Per Catalog #110 / #113 HISTORICAL_
    PROVENANCE: the JSONL is APPEND-ONLY — outcome rows for a prior
    deliberation reference the same ``deliberation_id`` rather than
    mutating the original row.

    Raises :class:`ValueError` on missing/invalid required fields. T2+
    records without ``council_assumption_adversary_verdict`` are
    refused per Catalog #292 + CLAUDE.md "Council conduct" Fix-7
    amendment.
    """
    _validate_record(record)
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    lock = lock_path or DEFAULT_COUNCIL_POSTERIOR_LOCK_PATH
    _ensure_state_dir(posterior)
    _ensure_state_dir(lock)
    payload = _record_to_dict(record)
    line = json.dumps(payload, sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with posterior.open("a", encoding="utf-8") as pf:
                pf.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def load_council_anchors(
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Lenient loader; skips malformed lines + wrong-schema rows.

    Mirrors :func:`tac.cost_band_calibration.load_anchors` semantics.
    Use the strict variant when a writer is about to MUTATE / REPLACE
    the file (so a partial corruption doesn't silently reset state).
    """
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    if not posterior.exists():
        return []
    out: list[CouncilDeliberationRecord] = []
    for raw in posterior.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") != SCHEMA_VERSION:
            continue
        try:
            out.append(_dict_to_record(payload))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def load_council_anchors_strict(
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Strict loader; raises :class:`CouncilPosteriorCorruptError` on parse
    failure. Mirror of :func:`tac.deploy.lightning.active_jobs_state.
    load_active_jobs_strict` per Catalog #138 fail-closed discipline.

    Writers about to MUTATE the file MUST use the strict variant so
    a partial corruption raises instead of silently dropping rows on
    next replace.
    """
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    if not posterior.exists():
        return []
    out: list[CouncilDeliberationRecord] = []
    for lineno, raw in enumerate(
        posterior.read_text(encoding="utf-8").splitlines(), start=1
    ):
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CouncilPosteriorCorruptError(
                f"line {lineno} of {posterior} failed JSON parse: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise CouncilPosteriorCorruptError(
                f"line {lineno} of {posterior} is not a dict"
            )
        if payload.get("schema") != SCHEMA_VERSION:
            # Unknown schema: skip (forward compat); strict-load refuses
            # only malformed rows, not legitimate schema-version skew.
            continue
        try:
            out.append(_dict_to_record(payload))
        except (KeyError, ValueError, TypeError) as exc:
            raise CouncilPosteriorCorruptError(
                f"line {lineno} of {posterior} failed schema construction: {exc}"
            ) from exc
    return out


def query_anchors_by_topic(
    topic_substring: str,
    *,
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Case-insensitive substring match on the ``topic`` field.

    Cite-chain helper sister of :func:`tac.deploy.modal.call_id_ledger.
    query_by_call_id`. Future Rashomon ensemble members and the
    Assumption-Adversary use this to trace position evolution on a
    given topic across prior deliberations.
    """
    if not topic_substring:
        return []
    needle = topic_substring.lower()
    anchors = load_council_anchors(posterior_path=posterior_path)
    return [a for a in anchors if needle in a.topic.lower()]


def query_dissent_history(
    member_name: str,
    *,
    posterior_path: Path | None = None,
) -> list[tuple[CouncilDeliberationRecord, dict[str, str]]]:
    """Surface every prior minority opinion by ``member_name``.

    Returns list of ``(record, dissent_entry)`` pairs. Maximum-signal
    preservation surface per CLAUDE.md "Council conduct" maximum-signal
    rule: future deliberations can trace which members have consistently
    flagged X as cargo-culted vs hard-earned.
    """
    if not member_name:
        return []
    needle = member_name.lower()
    anchors = load_council_anchors(posterior_path=posterior_path)
    matches: list[tuple[CouncilDeliberationRecord, dict[str, str]]] = []
    for anchor in anchors:
        for dissent in anchor.council_dissent:
            member_field = str(dissent.get("member", "")).lower()
            if needle in member_field:
                matches.append((anchor, dict(dissent)))
    return matches


def query_assumption_classification_history(
    assumption_substring: str,
    *,
    posterior_path: Path | None = None,
) -> list[tuple[CouncilDeliberationRecord, dict[str, str]]]:
    """Surface every prior HARD-EARNED-vs-CARGO-CULTED verdict for a
    given assumption substring.

    Returns list of ``(record, assumption_entry)`` pairs. Sister of
    :func:`query_dissent_history` at the assumption-classification
    surface. Future Assumption-Adversary deliberations consume this
    to trace classification stability (an assumption repeatedly
    classified HARD-EARNED across 5 deliberations is unlikely to be
    cargo-culted; an assumption flipping HARD-EARNED ↔ CARGO-CULTED is
    a red flag for the Assumption-Adversary to surface explicitly).
    """
    if not assumption_substring:
        return []
    needle = assumption_substring.lower()
    anchors = load_council_anchors(posterior_path=posterior_path)
    matches: list[tuple[CouncilDeliberationRecord, dict[str, str]]] = []
    for anchor in anchors:
        for entry in anchor.council_assumption_adversary_verdict:
            assumption_field = str(entry.get("assumption", "")).lower()
            if needle in assumption_field:
                matches.append((anchor, dict(entry)))
    return matches


# ──────────────────────────────────────────────────────────────────────
# Mission-alignment query helpers (per CLAUDE.md "Mission alignment —
# non-negotiable" subsection; operator binding standing directive 2026-05-16).
# ──────────────────────────────────────────────────────────────────────


def query_overrides(
    *,
    since_utc: str | None = None,
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Surface every operator-frontier-override deliberation for audit.

    Per CLAUDE.md "Mission alignment — non-negotiable" operational
    consequence 1: every operator-frontier-override bypasses quorum +
    tie-break + recusal for the specific decision, PRESERVES maximum-
    signal, and TRIGGERS a 30-day score-impact retrospective. This
    helper is the audit surface — operators / sister subagents can
    enumerate every override invocation and trace whether the override
    delivered the predicted mission contribution.

    ``since_utc`` is an ISO-8601 string; records with
    ``written_at_utc < since_utc`` are excluded. None returns all.
    """
    anchors = load_council_anchors(posterior_path=posterior_path)
    cutoff = _parse_utc_iso_or_none(since_utc) if since_utc else None
    out: list[CouncilDeliberationRecord] = []
    for anchor in anchors:
        if not anchor.override_invoked:
            continue
        if cutoff is not None:
            written = _parse_utc_iso_or_none(anchor.written_at_utc)
            if written is None or written < cutoff:
                continue
        out.append(anchor)
    return out


def query_due_retrospectives(
    *,
    as_of_utc: str | None = None,
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Surface every deferred/killed substrate whose 30-day retrospective is due.

    Per CLAUDE.md "Mission alignment — non-negotiable" operational
    consequence 3: every substrate that received a DEFERRED / KILL /
    research_only verdict gets a 30-day score-impact retrospective.
    This helper returns every record where
    ``as_of_utc >= deferred_substrate_retrospective_due_utc``.

    ``as_of_utc`` defaults to the current UTC time. The returned list is
    sorted by retrospective due date (oldest first) so operators see the
    most-overdue retrospectives first.
    """
    anchors = load_council_anchors(posterior_path=posterior_path)
    if as_of_utc is None:
        now = _dt.datetime.now(_dt.UTC)
    else:
        parsed = _parse_utc_iso_or_none(as_of_utc)
        if parsed is None:
            return []
        now = parsed
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.UTC)
    due: list[tuple[_dt.datetime, CouncilDeliberationRecord]] = []
    for anchor in anchors:
        if anchor.deferred_substrate_retrospective_due_utc is None:
            continue
        due_at = _parse_utc_iso_or_none(
            anchor.deferred_substrate_retrospective_due_utc
        )
        if due_at is None:
            continue
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=_dt.UTC)
        if now >= due_at:
            due.append((due_at, anchor))
    due.sort(key=lambda pair: pair[0])
    return [anchor for _, anchor in due]


def query_mission_contribution_distribution(
    *,
    since_utc: str | None = None,
    posterior_path: Path | None = None,
    tier_filter: tuple[str, ...] = (CouncilTier.T2, CouncilTier.T3, CouncilTier.T4),
) -> dict[str, int]:
    """Count anchors per ``predicted_mission_contribution`` category.

    Per CLAUDE.md "Mission alignment — non-negotiable" operational
    consequence 5: the breakdown surfaces whether the council apparatus
    is producing innovation-enabling or innovation-blocking verdicts. The
    operator-visible alert pairs with :func:`is_rigor_dominant`.

    By default counts only T2+ deliberations (T1 working groups are
    exempt from the mission-contribution forecast). ``since_utc`` filters
    by ``written_at_utc``; None returns all-time. Returns a dict with
    every category in :data:`VALID_MISSION_CONTRIBUTIONS` as a key
    (zero-init missing categories).
    """
    anchors = load_council_anchors(posterior_path=posterior_path)
    cutoff = _parse_utc_iso_or_none(since_utc) if since_utc else None
    counts: dict[str, int] = {cat: 0 for cat in VALID_MISSION_CONTRIBUTIONS}
    for anchor in anchors:
        if anchor.council_tier not in tier_filter:
            continue
        if anchor.predicted_mission_contribution is None:
            continue
        if cutoff is not None:
            written = _parse_utc_iso_or_none(anchor.written_at_utc)
            if written is None or written < cutoff:
                continue
        category = anchor.predicted_mission_contribution
        if category in counts:
            counts[category] += 1
    return counts


def is_rigor_dominant(
    *,
    since_utc: str | None = None,
    posterior_path: Path | None = None,
    threshold: float = RIGOR_DOMINANT_THRESHOLD,
) -> bool:
    """Return True when (rigor_overhead + apparatus_maintenance) / total > threshold.

    Per CLAUDE.md "Mission alignment — non-negotiable" operational
    consequence 5: when the breakdown of T2+ deliberations is rigor-
    dominant, the council apparatus is producing more apparatus-
    maintenance than frontier-breaking work — operator-visible alert
    fires.

    Returns False when the total count is zero (no data to evaluate).
    """
    counts = query_mission_contribution_distribution(
        since_utc=since_utc, posterior_path=posterior_path
    )
    total = sum(counts.values())
    if total == 0:
        return False
    overhead = counts.get("rigor_overhead", 0) + counts.get(
        "apparatus_maintenance", 0
    )
    return (overhead / total) > threshold


def compute_deferred_retrospective_due_utc(
    deliberation_utc: str | None = None,
    *,
    window_days: int = DEFERRED_RETROSPECTIVE_WINDOW_DAYS,
) -> str:
    """Compute the 30-day retrospective due-date for a deferred/killed substrate.

    Per CLAUDE.md "Mission alignment — non-negotiable" operational
    consequence 3. Returns the ISO-8601 UTC string for
    ``deliberation_utc + window_days``. If ``deliberation_utc`` is None,
    uses the current UTC time.

    Use this when constructing a :class:`CouncilDeliberationRecord` whose
    verdict defers or kills a substrate, so the
    ``deferred_substrate_retrospective_due_utc`` field is set correctly.
    """
    if deliberation_utc is None:
        base = _dt.datetime.now(_dt.UTC)
    else:
        parsed = _parse_utc_iso_or_none(deliberation_utc)
        if parsed is None:
            raise ValueError(
                f"deliberation_utc={deliberation_utc!r} is not a valid ISO-8601 UTC string"
            )
        base = parsed
    if base.tzinfo is None:
        base = base.replace(tzinfo=_dt.UTC)
    due = base + _dt.timedelta(days=window_days)
    return due.isoformat(timespec="seconds")


def _parse_utc_iso_or_none(s: str | None) -> _dt.datetime | None:
    """Tolerant ISO-8601 parse; returns None on malformed input."""
    if not s:
        return None
    try:
        normalized = s.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None


# Canonical name for the continual-learning posterior update hook
# (Catalog #265 canonical-contract token + Catalog #125 hook 5).
def update_from_anchor(
    record: CouncilDeliberationRecord,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Alias of :func:`append_council_anchor` for the canonical contract.

    Catalog #265 (`check_symposium_impls_canonical_contract`) requires
    each module to expose ``update_from_anchor`` as the continual-
    learning hook surface (Catalog #125 hook 5). This module is not in
    ``src/tac/symposium_impls/`` but mirrors the same canonical contract
    so future Rashomon ensemble / autopilot consumers route through one
    well-known name regardless of which canonical helper they're hooking.
    """
    append_council_anchor(
        record,
        posterior_path=posterior_path,
        lock_path=lock_path,
    )


# ──────────────────────────────────────────────────────────────────────
# Recursive self-reflection protocol — canonical helpers (Catalog #363)
#
# Per CLAUDE.md "Council conduct" amendment 2026-05-26 + canonical design
# memo council_recursive_self_reflection_protocol_design_20260526T133600Z.md.
# ──────────────────────────────────────────────────────────────────────


def classify_assumption_verification_status_from_evidence(
    *,
    source_artifact: str | None = None,
    empirical_anchor: str | None = None,
    literature_citation: str | None = None,
) -> str:
    """Classify an assumption's :class:`EmpiricalVerificationStatus` from
    available evidence.

    Per the canonical 4-value taxonomy + precedence rule (VERIFIED states
    dominate; the strongest evidence wins):

    1. If ``source_artifact`` (file path + line range + content quote)
       is provided → :attr:`EmpiricalVerificationStatus.
       VERIFIED_VIA_SOURCE_INSPECTION`.
    2. Else if ``empirical_anchor`` (commit sha + posterior row id +
       measurement metadata per Catalog #245 sister) is provided →
       :attr:`EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR`.
    3. Else if ``literature_citation`` (paper / textbook / CLAUDE.md
       doctrine pattern) is provided → :attr:`EmpiricalVerificationStatus.
       INFERRED_FROM_DOMAIN_LITERATURE`.
    4. Else → :attr:`EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION`.

    Per the canonical design memo §2.2 cycle bounds: the 2 unverified
    statuses (INFERRED + ASSUMED) gate the verdict (Round 2 must attempt
    verification OR Round 3 downgrades to PROVISIONAL).

    Canonical caller pattern:

    >>> status = classify_assumption_verification_status_from_evidence(
    ...     source_artifact="long_training_canonical.py:147 optimizer = mlx.optimizers.AdamW(...)",
    ... )
    >>> assert status == EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION
    """
    if source_artifact and str(source_artifact).strip():
        return EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION
    if empirical_anchor and str(empirical_anchor).strip():
        return EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR
    if literature_citation and str(literature_citation).strip():
        return EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE
    return EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION


def extract_unverified_assumptions(
    record: CouncilDeliberationRecord,
) -> list[AssumptionEmpiricalVerification]:
    """Extract assumptions from a record whose status is INFERRED or ASSUMED.

    Returns the per-assumption records that gate the verdict per the
    Recursive self-reflection protocol Round 3 downgrade-or-verify rule.

    Legacy entries (pre-Catalog-#363) lacking ``empirical_verification_status``
    are auto-classified :attr:`EmpiricalVerificationStatus.
    INFERRED_FROM_DOMAIN_LITERATURE` per
    :meth:`AssumptionEmpiricalVerification.from_dict` safe-default — they
    return as unverified per the canonical backward-compat rule.

    Caller pattern (Round 2 self-reflection):

    >>> unverified = extract_unverified_assumptions(record)
    >>> if unverified:
    ...     for av in unverified:
    ...         # Round 3 must verify OR downgrade verdict to PROVISIONAL.
    ...         ...
    """
    unverified: list[AssumptionEmpiricalVerification] = []
    for entry in record.council_assumption_adversary_verdict:
        try:
            ae = AssumptionEmpiricalVerification.from_dict(dict(entry))
        except (AssumptionVerificationValidationError, TypeError):
            continue
        if ae.empirical_verification_status in UNVERIFIED_VERIFICATION_STATUSES:
            unverified.append(ae)
    return unverified


def verdict_status_requires_provisional_marker(
    record: CouncilDeliberationRecord,
) -> bool:
    """Return True when the record's verdict requires PROVISIONAL marker.

    Per the canonical design memo §2.1 Round 3 rule: a verdict requires
    PROVISIONAL-PENDING-VERIFICATION marker when material unverified
    assumptions remain. The simplest canonical criterion (matches the
    4-empirical-receipt anchor pattern): the record's
    ``council_assumption_adversary_verdict`` contains AT LEAST ONE
    assumption whose status is INFERRED_FROM_DOMAIN_LITERATURE or
    ASSUMED_AWAITING_VERIFICATION.

    The Round 3 mechanism is operator-routable; the function does NOT
    auto-downgrade — it surfaces the requirement so the operator or a
    successor subagent can choose between (a) empirical verification
    before landing OR (b) verdict-status downgrade to PROVISIONAL OR
    (c) ESCALATE_TO_OPERATOR per Catalog #300.

    T1 working-group recommendations are exempt from this rule (T1
    findings feed downstream T2/T3 deliberations rather than binding
    directly; the downstream deliberation inherits the discipline).
    """
    if record.council_tier == CouncilTier.T1:
        return False
    return len(extract_unverified_assumptions(record)) > 0


def query_self_reflection_history_for_deliberation(
    deliberation_id: str,
    *,
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Return the chain of self-reflection-round anchors for a deliberation.

    Sister of :func:`query_anchors_by_topic` at the per-deliberation
    self-reflection-round surface. Returns every record whose
    ``deliberation_id`` matches ``deliberation_id`` AND whose
    ``event_type`` is ``"council_self_reflection_round_N"`` for some N
    >= 1, ordered chronologically by ``written_at_utc``.

    Per the canonical design memo §2.4 operator-attention budget rule:
    a deliberation that crosses MAX_SELF_REFLECTION_ROUNDS (5) rounds
    without producing 3 consecutive clean rounds triggers
    ESCALATE_TO_OPERATOR per Catalog #300. The audit query is the
    operator-facing surface for this rule.

    Caller pattern (operator audit):

    >>> history = query_self_reflection_history_for_deliberation(
    ...     "t3_grand_council_mlx_pytorch_drift_20260526"
    ... )
    >>> if len(history) >= MAX_SELF_REFLECTION_ROUNDS:
    ...     # Operator should ESCALATE_TO_OPERATOR per Catalog #300.
    ...     ...
    """
    if not deliberation_id:
        return []
    anchors = load_council_anchors(posterior_path=posterior_path)
    matches = [
        a for a in anchors
        if a.deliberation_id == deliberation_id
        and isinstance(a.event_type, str)
        and a.event_type.startswith("council_self_reflection_round_")
    ]
    matches.sort(key=lambda r: r.written_at_utc or "")
    return matches
