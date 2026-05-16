# SPDX-License-Identifier: MIT
"""Typed L5 v2 staircase plan and fail-closed custody surface.

This module is deliberately planning-only. It turns the Time-Traveler L5 v2
staircase into machine-readable steps and gates so Cathedral/autopilot can see
the frontier path without treating source-backed theory as score evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from tac.optimization.l5_v2_probe_disambiguator import L5V2_PROBE_TOOL_PATH
from tac.optimization.prediction_band import (
    BandSource,
    BaselineRef,
    EmpiricalAnchorRef,
    PredictionBand,
    SupersessionRef,
    UncertaintyRef,
    prediction_band_to_dict,
    validate_prediction_band,
    verdict_to_dict,
)
from tac.optimization.research_basis import research_basis_ids_for_family

SUBJECT_ID = "time_traveler_l5_autonomy"
LANE_ID = "lane_time_traveler_l5_autonomy_substrate_20260513"
CAMPAIGN_ID = "campaign_time_traveler_l5_v2_staircase_20260516"
PREDICTED_DELTA_BAND = (-0.0500, -0.0200)
PREDICTED_DELTA_AXIS = "mixed"

GateStatus = Literal["required", "satisfied", "blocked"]
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class L5V2Gate:
    """One non-negotiable gate before L5 v2 can dispatch or promote."""

    gate_id: str
    status: GateStatus
    description: str
    blocker: str
    evidence_path: str = ""


@dataclass(frozen=True)
class L5V2GateEvidence:
    """Artifact-backed proof that one L5 v2 gate actually passed."""

    gate_id: str
    artifact_path: str
    artifact_sha256: str
    predicate_id: str
    predicate_passed: bool
    evidence_grade: str = ""


@dataclass(frozen=True)
class L5V2Step:
    """One ordered staircase step for the L5 v2 campaign."""

    step_id: str
    title: str
    objective: str
    deliverable_surface: str
    required_gate_ids: tuple[str, ...]
    research_basis_ids: tuple[str, ...]
    dispatch_allowed: bool = False
    promotion_eligible: bool = False


def l5_v2_research_basis_ids() -> tuple[str, ...]:
    """Return the source stack that anchors L5 v2 planning claims."""

    return tuple(research_basis_ids_for_family("time_traveler_l5_v2"))


def l5_v2_required_gates() -> tuple[L5V2Gate, ...]:
    """Return the fail-closed gate set for the current L5 v2 staircase."""

    return (
        L5V2Gate(
            gate_id="byte_closed_temporal_sideinfo_consumption",
            status="required",
            description=(
                "Every temporal side-info byte must be emitted, parsed, and "
                "proven to change inflate output under byte mutation."
            ),
            blocker="requires_byte_closed_temporal_sideinfo_consumption_proof",
        ),
        L5V2Gate(
            gate_id="c1_z5_tt5l_probe_disambiguator",
            status="required",
            description=(
                "C1, Z5, and TT5L interpretations must be probed under one "
                "callable disambiguator before architecture lock."
            ),
            blocker="requires_c1_z5_tt5l_probe_disambiguator_before_architecture_lock",
        ),
        L5V2Gate(
            gate_id="paired_cpu_cuda_axis_plan",
            status="required",
            description=(
                "CPU and CUDA axes need a paired plan with archive SHA, "
                "runtime tree SHA, inflate device, eval device, and component deltas."
            ),
            blocker="requires_paired_cpu_cuda_axis_plan_before_promotion",
        ),
        L5V2Gate(
            gate_id="exact_anchor_or_diagnostic_pair",
            status="required",
            description=(
                "At least one paired exact or explicitly diagnostic anchor must "
                "exist before predicted bands can influence rank reward."
            ),
            blocker="requires_l5_v2_empirical_anchor",
        ),
    )


def l5_v2_staircase_steps() -> tuple[L5V2Step, ...]:
    """Return the ordered L5 v2 campaign steps."""

    basis = l5_v2_research_basis_ids()
    return (
        L5V2Step(
            step_id="l5v2_00_source_and_alias_custody",
            title="Source and alias custody",
            objective=(
                "Bind L5 v2 paper/theory claims to canonical research basis ids "
                "and legacy aliases before dispatch planning."
            ),
            deliverable_surface="src/tac/optimization/research_basis.py",
            required_gate_ids=(),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_01_sideinfo_consumption_proof",
            title="Temporal side-info consumption proof",
            objective=(
                "Add byte-mutation and section-consumption proofs for the TT5L "
                "temporal side-info stream."
            ),
            deliverable_surface=(
                "src/tac/substrates/time_traveler_l5_autonomy/archive.py + tests"
            ),
            required_gate_ids=("byte_closed_temporal_sideinfo_consumption",),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_02_probe_disambiguator",
            title="C1/Z5/TT5L probe disambiguator",
            objective=(
                "Ship both defensible interpretations and let paired probes "
                "choose architecture direction before full training."
            ),
            deliverable_surface=L5V2_PROBE_TOOL_PATH,
            required_gate_ids=("c1_z5_tt5l_probe_disambiguator",),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_03_paired_axis_anchor",
            title="Paired CPU/CUDA anchor",
            objective=(
                "Measure the exact same byte-closed packet on contest-compliant "
                "CPU and CUDA axes before any promotion or submission discussion."
            ),
            deliverable_surface="experiments/results/time_traveler_l5_v2/",
            required_gate_ids=(
                "paired_cpu_cuda_axis_plan",
                "exact_anchor_or_diagnostic_pair",
            ),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_04_stack_of_stacks_candidate",
            title="Stack-of-stacks composition candidate",
            objective=(
                "Compose the first proved L5 v2 packet with the best byte-closed "
                "orthogonal winner only after component anchors exist."
            ),
            deliverable_surface="src/tac/optimization/substrate_composition_matrix.py",
            required_gate_ids=(
                "byte_closed_temporal_sideinfo_consumption",
                "exact_anchor_or_diagnostic_pair",
            ),
            research_basis_ids=basis,
        ),
    )


def l5_v2_prediction_band_payload() -> dict[str, Any]:
    """Return the L5 v2 prediction band payload for inventory/autopilot rows.

    The payload carries real source provenance but intentionally lacks baseline
    custody and empirical anchor status. Validators therefore keep it visible
    for dispatch planning while blocking rank reward and promotion.
    """

    band = PredictionBand(
        band_id="time_traveler_l5_v2_delta_prior_20260516",
        subject_id=SUBJECT_ID,
        band_kind="delta_score",
        low=PREDICTED_DELTA_BAND[0],
        high=PREDICTED_DELTA_BAND[1],
        axis=PREDICTED_DELTA_AXIS,
        baseline=BaselineRef(
            label="axis_matched_frontier_baseline_pending",
            axis=PREDICTED_DELTA_AXIS,
            score=None,
            archive_sha256="",
            runtime_tree_sha256="",
            artifact_path="",
        ),
        band_source=BandSource(
            local_ledger_paths=(
                "file:.omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
                "file:.omx/research/campaign_lane_c2_z7_mature_predictive_receiver_l5_20260514.md",
            ),
            research_basis_ids=l5_v2_research_basis_ids(),
            claim_scope=(
                "planning prior for L5 v2 staircase only; not score evidence "
                "until byte-closed paired anchors exist"
            ),
        ),
        uncertainty=UncertaintyRef(
            method="source-backed-prior-with-no-l5-v2-anchor",
            confidence_tag="pre_anchor",
            n_empirical_anchors=0,
            notes="Band remains rank-blocked until L5 v2 empirical anchors land.",
        ),
        supersession=SupersessionRef(status="active"),
        empirical_anchor=EmpiricalAnchorRef(status="pending", anchors=()),
        planning_only=True,
        score_claim=False,
    )
    return prediction_band_to_dict(band)


def l5_v2_prediction_band_verdict() -> dict[str, Any]:
    """Return the current validation verdict for the L5 v2 prediction band."""

    payload = l5_v2_prediction_band_payload()
    band = PredictionBand(
        band_id=str(payload["band_id"]),
        subject_id=str(payload["subject_id"]),
        band_kind="delta_score",
        low=float(payload["low"]),
        high=float(payload["high"]),
        axis=str(payload["axis"]),
        baseline=BaselineRef(**payload["baseline"]),
        band_source=BandSource(
            local_ledger_paths=tuple(payload["band_source"]["local_ledger_paths"]),
            research_basis_ids=tuple(payload["band_source"]["research_basis_ids"]),
            claim_scope=str(payload["band_source"]["claim_scope"]),
        ),
        uncertainty=UncertaintyRef(**payload["uncertainty"]),
        supersession=SupersessionRef(**payload["supersession"]),
        empirical_anchor=EmpiricalAnchorRef(**payload["empirical_anchor"]),
        planning_only=bool(payload["planning_only"]),
        score_claim=bool(payload["score_claim"]),
    )
    return verdict_to_dict(
        validate_prediction_band(
            band,
            expected_subject_id=SUBJECT_ID,
            expected_low=PREDICTED_DELTA_BAND[0],
            expected_high=PREDICTED_DELTA_BAND[1],
        )
    )


def _coerce_gate_evidence(
    gate_evidence: (
        Mapping[str, L5V2GateEvidence | Mapping[str, Any]]
        | Iterable[L5V2GateEvidence | Mapping[str, Any]]
        | None
    ),
) -> dict[str, L5V2GateEvidence]:
    if gate_evidence is None:
        return {}

    values: Iterable[L5V2GateEvidence | Mapping[str, Any]]
    if isinstance(gate_evidence, Mapping):
        values = (
            value if isinstance(value, L5V2GateEvidence) else {"gate_id": key, **value}
            for key, value in gate_evidence.items()
        )
    else:
        values = gate_evidence

    coerced: dict[str, L5V2GateEvidence] = {}
    for value in values:
        if isinstance(value, L5V2GateEvidence):
            evidence = value
        else:
            evidence = L5V2GateEvidence(
                gate_id=str(value.get("gate_id", "")),
                artifact_path=str(value.get("artifact_path", "")),
                artifact_sha256=str(value.get("artifact_sha256", "")),
                predicate_id=str(value.get("predicate_id", "")),
                predicate_passed=bool(value.get("predicate_passed", False)),
                evidence_grade=str(value.get("evidence_grade", "")),
            )
        if evidence.gate_id:
            coerced[evidence.gate_id] = evidence
    return coerced


def _is_transient_artifact_path(path: str) -> bool:
    normalized = path.removeprefix("file:").strip()
    return normalized.startswith(("/tmp/", "/private/tmp/", "/var/tmp/"))


def _gate_evidence_blockers(
    gate: L5V2Gate,
    evidence: L5V2GateEvidence | None,
) -> list[str]:
    if evidence is None:
        return [f"l5_v2_gate_evidence_missing:{gate.gate_id}"]

    blockers: list[str] = []
    if not evidence.artifact_path.strip():
        blockers.append(f"l5_v2_gate_artifact_path_missing:{gate.gate_id}")
    elif _is_transient_artifact_path(evidence.artifact_path):
        blockers.append(f"l5_v2_gate_artifact_path_transient:{gate.gate_id}")
    if not _SHA256_HEX_RE.fullmatch(evidence.artifact_sha256.strip()):
        blockers.append(f"l5_v2_gate_artifact_sha256_invalid:{gate.gate_id}")
    if not evidence.predicate_id.strip():
        blockers.append(f"l5_v2_gate_predicate_id_missing:{gate.gate_id}")
    if not evidence.predicate_passed:
        blockers.append(f"l5_v2_gate_predicate_failed:{gate.gate_id}")
    return blockers


def l5_v2_dispatch_readiness(
    satisfied_gate_ids: Mapping[str, bool] | None = None,
    gate_evidence: (
        Mapping[str, L5V2GateEvidence | Mapping[str, Any]]
        | Iterable[L5V2GateEvidence | Mapping[str, Any]]
        | None
    ) = None,
) -> dict[str, Any]:
    """Return a fail-closed dispatch readiness summary for L5 v2.

    Boolean gate claims are preserved as planning notes only. Dispatch
    readiness requires artifact-backed evidence for every gate so prose or
    stale booleans cannot unlock L5 v2 actuation.
    """

    satisfied_gate_ids = satisfied_gate_ids or {}
    evidence_by_gate = _coerce_gate_evidence(gate_evidence)
    gates = []
    blockers = []
    evidence_blockers = []
    for gate in l5_v2_required_gates():
        evidence = evidence_by_gate.get(gate.gate_id)
        gate_evidence_blockers = _gate_evidence_blockers(gate, evidence)
        evidence_valid = not gate_evidence_blockers
        satisfied = bool(satisfied_gate_ids.get(gate.gate_id)) or evidence_valid
        status = "satisfied" if satisfied else gate.status
        gate_payload = {
            **gate.__dict__,
            "status": status,
            "claimed_satisfied": bool(satisfied_gate_ids.get(gate.gate_id)),
            "evidence_valid": evidence_valid,
            "evidence": evidence.__dict__ if evidence is not None else None,
            "evidence_blockers": gate_evidence_blockers,
        }
        gates.append(gate_payload)
        if not satisfied:
            blockers.append(gate.blocker)
        evidence_blockers.extend(gate_evidence_blockers)
    all_gate_claims_satisfied = all(
        gate["claimed_satisfied"] or gate["evidence_valid"] for gate in gates
    )
    all_gate_evidence_valid = all(gate["evidence_valid"] for gate in gates)
    return {
        "subject_id": SUBJECT_ID,
        "lane_id": LANE_ID,
        "campaign_id": CAMPAIGN_ID,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "all_gate_claims_satisfied": all_gate_claims_satisfied,
        "all_gate_evidence_valid": all_gate_evidence_valid,
        "ready_for_dispatch": all_gate_evidence_valid,
        "blockers": blockers + evidence_blockers,
        "gates": gates,
        "steps": [step.__dict__ for step in l5_v2_staircase_steps()],
        "prediction_band_verdict": l5_v2_prediction_band_verdict(),
    }


__all__ = [
    "CAMPAIGN_ID",
    "LANE_ID",
    "PREDICTED_DELTA_AXIS",
    "PREDICTED_DELTA_BAND",
    "SUBJECT_ID",
    "L5V2Gate",
    "L5V2GateEvidence",
    "L5V2Step",
    "l5_v2_dispatch_readiness",
    "l5_v2_prediction_band_payload",
    "l5_v2_prediction_band_verdict",
    "l5_v2_required_gates",
    "l5_v2_research_basis_ids",
    "l5_v2_staircase_steps",
]
