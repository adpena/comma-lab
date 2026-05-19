# SPDX-License-Identifier: MIT
"""Canonical subsumption helpers — read existing atom-shaped sources, emit canonical Atoms.

Subsumption is one-way: any of the seven pre-existing atom-shaped surfaces
(JSONL ledgers / dataclass instances / Markdown ledger rows) can be parsed
into a canonical ``Atom`` instance via the helper here. The reverse
direction is documented but not auto-emitted because the legacy surfaces
should NOT be written-to going forward — new atoms should land at the
canonical ledger ``.omx/state/atom_ledger.jsonl`` via ``tac.atom.ledger``.

Citations:
  - Catalog #110 / #113 HISTORICAL_PROVENANCE — legacy rows without the
    full canonical schema are accepted with ``evidence_grade='research_only'``
    backfill and an explicit notes-token marking the migration source.
  - Operator standing directive 2026-05-18: subsumption preserves signal
    by-construction; downstream consumers (cathedral autopilot ranker /
    Pareto solver / continual-learning posterior) can read uniform atoms
    regardless of source legacy.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .atom import Atom
from .builders import (
    build_arbitrary_value_atom,
    build_cargo_cult_atom,
    build_council_deliberation_atom,
    build_dispatch_claim_atom,
    build_meta_lagrangian_atom,
    build_probe_outcome_atom,
)
from .types import AtomValidationError, ResolutionPath


def _coerce_resolution_path(raw: str | ResolutionPath) -> ResolutionPath:
    """Coerce string -> ResolutionPath with helpful error.

    Accepts canonical 6-member set or the legacy alias 'self alien tech'
    (sister audit uses 'self_alien_tech'). Unknown values raise.
    """
    if isinstance(raw, ResolutionPath):
        return raw
    if not isinstance(raw, str):
        raise AtomValidationError(
            f"resolution_path must be str or ResolutionPath (got {type(raw).__name__})"
        )
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    for member in ResolutionPath:
        if member.value == normalized:
            return member
    raise AtomValidationError(
        f"unknown resolution_path {raw!r}; canonical set: "
        f"{[m.value for m in ResolutionPath]}"
    )


def atom_from_meta_lagrangian_row(row: Mapping[str, Any]) -> Atom:
    """Convert a ``tac.meta_lagrangian_allocator`` row dict into canonical Atom.

    Source format (per ``atoms_from_hnerv_decoder_recode_profile``):
        atom_id / family / family_group / byte_delta / expected_*_dist_delta
        / confidence / evidence_grade / source_archive_sha256 / conflict lists
        / hard_pair_support / pair_support / class_support / geometry_priors
        / openpilot_priors.

    Forward-compat: extra fields are stuffed into ``extra_metadata``.
    """
    atom_id = str(row.get("atom_id") or "")
    if not atom_id:
        raise AtomValidationError("meta-Lagrangian row missing atom_id")
    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "atom_id",
            "family",
            "family_group",
            "byte_delta",
            "expected_seg_dist_delta",
            "expected_pose_dist_delta",
            "confidence",
            "evidence_grade",
            "source_archive_sha256",
            "conflicts_with_families",
            "conflicts_with_atoms",
            "expected_score_delta_lower",
            "expected_score_delta_upper",
            "cost_envelope_usd",
        }
    }
    return build_meta_lagrangian_atom(
        atom_id=atom_id,
        family=str(row.get("family", "")),
        family_group=str(row.get("family_group", "")),
        byte_delta=int(row.get("byte_delta", 0)),
        expected_seg_dist_delta=float(row.get("expected_seg_dist_delta", 0.0)),
        expected_pose_dist_delta=float(row.get("expected_pose_dist_delta", 0.0)),
        expected_score_delta_lower=float(row.get("expected_score_delta_lower", 0.0)),
        expected_score_delta_upper=float(row.get("expected_score_delta_upper", 0.0)),
        cost_envelope_usd=float(row.get("cost_envelope_usd", 0.0)),
        confidence=float(row.get("confidence", 0.5)),
        source_archive_sha256=str(row.get("source_archive_sha256", "")),
        conflicts_with_families=tuple(row.get("conflicts_with_families", []) or []),
        conflicts_with_atoms=tuple(row.get("conflicts_with_atoms", []) or []),
        extra_metadata=extras,
    )


def atom_from_cargo_cult_audit_row(row: Mapping[str, Any]) -> Atom:
    """Convert a ``## Cargo-cult audit per assumption`` row into canonical Atom.

    Expected dict format (per Catalog #303 + the hard-earned-vs-cargo-culted
    addendum):
        substrate_id / assumption / classification (HARD-EARNED / CARGO-CULTED
        / UNDECIDED) / rationale / unwind_test_plan / predicted_impact (band).

    Per CLAUDE.md "Council conduct" Fix-7 + Catalog #292 the per-deliberation
    surface emits these as part of the assumption-adversary verdict block.
    """
    substrate_id = str(row.get("substrate_id") or row.get("substrate") or "")
    assumption = str(row.get("assumption") or "")
    classification = str(row.get("classification") or "UNDECIDED")
    atom_id = str(
        row.get("atom_id") or f"cargo_cult:{substrate_id}:{abs(hash(assumption)) & 0xFFFF:04x}"
    )
    predicted_band = row.get("predicted_impact") or [0.0, 0.0]
    if isinstance(predicted_band, (list, tuple)) and len(predicted_band) == 2:
        lo, hi = float(predicted_band[0]), float(predicted_band[1])
    else:
        lo, hi = 0.0, 0.0
    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "atom_id",
            "substrate_id",
            "substrate",
            "assumption",
            "classification",
            "rationale",
            "unwind_test_plan",
            "predicted_impact",
            "cost_envelope_usd",
        }
    }
    return build_cargo_cult_atom(
        atom_id=atom_id,
        substrate_id=substrate_id,
        assumption=assumption,
        classification=classification,
        rationale=str(row.get("rationale", "")),
        unwind_test_plan=str(row.get("unwind_test_plan", "")),
        predicted_impact_lower=lo,
        predicted_impact_upper=hi,
        cost_envelope_usd=float(row.get("cost_envelope_usd", 0.0)),
        extra_metadata=extras,
    )


def atom_from_probe_outcomes_ledger_row(row: Mapping[str, Any]) -> Atom:
    """Convert a ``.omx/state/probe_outcomes.jsonl`` row into canonical Atom.

    Mirrors ``tac.probe_outcomes_ledger.ProbeOutcomeView`` field set with
    canonical 7-verdict taxonomy. Legacy rows with non-canonical verdicts
    are accepted; ``Atom.__post_init__`` raises AtomValidationError on
    unknown verdicts so the construction surface is the canonical filter.
    """
    probe_id = str(row.get("probe_id") or "")
    if not probe_id:
        raise AtomValidationError("probe-outcome row missing probe_id")
    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "probe_id",
            "substrate",
            "recipe_path",
            "probe_kind",
            "verdict",
            "metric_name",
            "metric_value",
            "threshold",
            "threshold_token",
            "evidence_path",
            "next_action",
            "blocker_status",
            "adjudicated_at_utc",
            "expires_at_utc",
        }
    }
    return build_probe_outcome_atom(
        atom_id=f"probe:{probe_id}",
        probe_id=probe_id,
        substrate=str(row.get("substrate", "")),
        verdict=str(row.get("verdict", "")),
        metric_name=str(row.get("metric_name", "")),
        metric_value=row.get("metric_value"),
        threshold=row.get("threshold"),
        threshold_token=str(row.get("threshold_token", "")),
        evidence_path=str(row.get("evidence_path", "")),
        next_action=str(row.get("next_action", "")),
        blocker_status=str(row.get("blocker_status", "blocking")),
        adjudicated_at_utc=str(row.get("adjudicated_at_utc", "")),
        expires_at_utc=str(row.get("expires_at_utc", "")),
        extra_metadata=extras,
    )


def atom_from_council_deliberation_record(record: Mapping[str, Any]) -> Atom:
    """Convert a ``tac.council_continual_learning.CouncilDeliberationRecord`` row.

    Accepts either the canonical dataclass (via ``dataclasses.asdict``) or
    a pre-serialized JSONL row from
    ``.omx/state/council_deliberation_posterior.jsonl``.
    """
    deliberation_id = str(record.get("deliberation_id") or "")
    if not deliberation_id:
        raise AtomValidationError("council-deliberation row missing deliberation_id")
    extras = {
        k: v
        for k, v in record.items()
        if k
        not in {
            "deliberation_id",
            "topic",
            "council_tier",
            "council_verdict",
            "council_attendees",
            "council_quorum_met",
            "memory_path",
        }
    }
    return build_council_deliberation_atom(
        atom_id=f"council:{deliberation_id}",
        deliberation_id=deliberation_id,
        topic=str(record.get("topic", "")),
        council_tier=str(record.get("council_tier", "T2")),
        council_verdict=str(record.get("council_verdict", "PROCEED")),
        council_attendees=tuple(record.get("council_attendees", []) or []),
        council_quorum_met=bool(record.get("council_quorum_met", True)),
        memory_path=str(record.get("memory_path", "")),
        extra_metadata=extras,
    )


def atom_from_dispatch_claim_row(row: Mapping[str, Any]) -> Atom:
    """Convert a ``.omx/state/active_lane_dispatch_claims.md`` row dict into canonical Atom.

    The Markdown ledger row should be pre-parsed by the caller (the canonical
    parser lives in ``tools/claim_lane_dispatch.py``); this helper accepts
    the dict form so the subsumption surface stays uniform.
    """
    lane_id = str(row.get("lane_id") or "")
    if not lane_id:
        raise AtomValidationError("dispatch-claim row missing lane_id")
    instance_or_job_id = str(
        row.get("instance_or_job_id")
        or row.get("instance_id")
        or row.get("job_id")
        or ""
    )
    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "lane_id",
            "provider",
            "gpu",
            "instance_or_job_id",
            "instance_id",
            "job_id",
            "status",
            "opened_at_utc",
            "cost_envelope_usd",
        }
    }
    return build_dispatch_claim_atom(
        atom_id=f"dispatch:{lane_id}:{instance_or_job_id}",
        lane_id=lane_id,
        provider=str(row.get("provider", "")),
        gpu=str(row.get("gpu", "")),
        instance_or_job_id=instance_or_job_id,
        cost_envelope_usd=float(row.get("cost_envelope_usd", 0.0)),
        status=str(row.get("status", "active")),
        opened_at_utc=str(row.get("opened_at_utc", "")),
        extra_metadata=extras,
    )


def atom_from_arbitrariness_audit_row(row: Mapping[str, Any]) -> Atom:
    """Convert one row of ``.omx/state/arbitrariness_extinction_audit_*.jsonl`` into Atom.

    Sister-audit canonical 21-field schema landed 2026-05-18; the
    subsumption here preserves ALL fields (canonical promoted to top-level
    Atom fields + audit-specific ones stuffed into extra_metadata).
    """
    value_id = str(row.get("value_id") or "")
    if not value_id:
        raise AtomValidationError("arbitrariness-audit row missing value_id")
    resolution_path = _coerce_resolution_path(row.get("resolution_path", "experimental"))
    predicted_band = row.get("predicted_ev_delta_s") or [0.0, 0.0]
    if not isinstance(predicted_band, (list, tuple)) or len(predicted_band) != 2:
        raise AtomValidationError(
            f"predicted_ev_delta_s for value_id={value_id!r} malformed: "
            f"{predicted_band!r}"
        )
    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "value_id",
            "file_path",
            "current_value",
            "predicted_replacement",
            "resolution_path",
            "predicted_ev_delta_s",
            "cost_envelope_usd",
            "literature_citation",
            "canonical_helper_repo_link",
            "cheaper_alternative_path",
            "blocking_dependencies",
            "captured_by_subagent",
            "notes",
            "provenance",
        }
    }
    return build_arbitrary_value_atom(
        atom_id=value_id,
        file_path=str(row.get("file_path", "")),
        current_value=row.get("current_value"),
        predicted_replacement=row.get("predicted_replacement"),
        resolution_path=resolution_path,
        predicted_ev_delta_s=tuple(predicted_band),
        cost_envelope_usd=float(row.get("cost_envelope_usd", 0.0)),
        literature_citation=str(row.get("literature_citation", "")),
        canonical_helper_repo_link=str(row.get("canonical_helper_repo_link", "")),
        cheaper_alternative_path=str(row.get("cheaper_alternative_path", "")),
        blocking_dependencies=tuple(row.get("blocking_dependencies", []) or []),
        captured_by_subagent=str(row.get("captured_by_subagent", "")),
        notes=str(row.get("notes", "")),
        provenance=row.get("provenance"),
        extra_metadata=extras,
    )


__all__ = [
    "atom_from_arbitrariness_audit_row",
    "atom_from_cargo_cult_audit_row",
    "atom_from_council_deliberation_record",
    "atom_from_dispatch_claim_row",
    "atom_from_meta_lagrangian_row",
    "atom_from_probe_outcomes_ledger_row",
]
