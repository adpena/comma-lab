# SPDX-License-Identifier: MIT
"""Typed L5 v2 staircase plan and fail-closed custody surface.

This module is deliberately planning-only. It turns the Time-Traveler L5 v2
staircase into machine-readable steps and gates so Cathedral/autopilot can see
the frontier path without treating source-backed theory as score evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.exact_eval_custody import (
    CUDA_DEVICE_TOKENS,
    contains_non_negated_device_token,
    is_contest_cpu_device_text,
    validate_exact_eval_evidence,
)
from tac.optimization.l5_v2_probe_disambiguator import (
    L5V2_CANDIDATES,
    L5V2_PROBE_SCHEMA,
    L5V2_PROBE_TOOL_PATH,
    evaluate_l5_v2_probe,
    observation_from_mapping,
)
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
TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH = (
    ".omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json"
)
TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256 = (
    "8bb68ba5e14f0bbb0511812cbb7b7465e58ef639997e300558c04c3cdae98605"
)
TT5L_CONTEST_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH = (
    "experiments/results/time_traveler_l5_v2/"
    "tt5l_contest_sideinfo_consumption_proof.json"
)
TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID = (
    "tt5l_byte_closed_temporal_sideinfo_consumption_v1"
)
TT5L_CONTEST_SIDEINFO_PROOF_TOOL_PATH = (
    "tools/build_tt5l_contest_sideinfo_consumption_proof.py"
)
TT5L_MODAL_A100_DISPATCH_RECIPE_PATH = (
    ".omx/operator_authorize_recipes/"
    "substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml"
)
TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH = (
    "experiments/results/time_traveler_l5_v2/l5_v2_probe_template.json"
)
TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH = "tools/check_substrate_dykstra_feasibility.py"
TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH = (
    ".omx/state/dykstra_feasibility_time_traveler_l5.json"
)
TT5L_DYKSTRA_SUBSTRATE_ID = "time_traveler_l5_5move"
TT5L_DYKSTRA_SCORE_FORMULA = (
    "100*seg_dist+sqrt(10*pose_dist)+25*archive_bytes/37545489"
)
TT5L_DYKSTRA_PROJECTION_KIND = "score_axis_projection_with_declared_constraints"
TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS = frozenset({
    "contest_rate_budget",
    "contest_seg_dist_budget",
    "contest_pose_dist_budget",
    "tt5l_predictive_coding_hierarchy",
    "tt5l_cooperative_receiver",
    "tt5l_ego_motion_foveation",
    "tt5l_differentiable_world_model",
    "tt5l_tikhonov_rate_regularization",
})
PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH = (
    ".omx/research/pr106_packetir_candidate_matrix_20260516_codex.json"
)
PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256 = (
    "cda1219fb880cc0513a5d0706af1b95fe74e7a8a52391588e054dba2c24ad93c"
)
L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH = (
    ".omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.json"
)
L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256 = (
    "7bf0d1bd267a36d469f34b07c26ad5ddecff4603195d1d44462ae77eeef6efc7"
)
L5_V2_PACKETIR_STACK_EVIDENCE_SCHEMA = "l5_v2_packetir_stack_evidence_v1"
L5_V2_PR106_STACK_CELL_CANDIDATES_SCHEMA = (
    "l5_v2_pr106_packetir_stack_cell_candidates_v1"
)
L5_V2_PACKETIR_SECTION_ENTROPY_EVIDENCE_SCHEMA = (
    "l5_v2_packetir_section_entropy_evidence_v1"
)

GateStatus = Literal["required", "satisfied", "blocked"]
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED_EXACT_AXES = ("contest_cpu", "contest_cuda")
_TT5L_CONTEST_FULL_FRAME_PROOF_SCOPES = frozenset({
    "contest_full_frame_consumption_proof",
    "contest_full_frame_sideinfo_consumption_proof",
})
_TT5L_CONTEST_N_PAIRS = 600
_TT5L_CONTEST_TOTAL_FRAMES = 1200
_TT5L_CONTEST_RAW_OUTPUT_FRAME_NBYTES = 874 * 1164 * 3


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
                "file:.omx/research/l5_v2_paper_fidelity_research_basis_wire_in_20260516_codex.md",
                "file:.omx/research/l5_v2_latest_source_basis_wirein_20260516_codex.md",
                "file:.omx/research/l5_v2_latest_neural_video_codec_source_basis_20260516_codex.md",
                "file:.omx/research/l5_v2_source_basis_sidecar_20260516_codex.md",
                "file:.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.md",
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


def _require_literal_json_bool(value: object, *, field_name: str) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    raise ValueError(f"{field_name} must be a literal JSON boolean")


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
        planning_only=_require_literal_json_bool(
            payload["planning_only"],
            field_name="planning_only",
        ),
        score_claim=_require_literal_json_bool(
            payload["score_claim"],
            field_name="score_claim",
        ),
    )
    return verdict_to_dict(
        validate_prediction_band(
            band,
            expected_subject_id=SUBJECT_ID,
            expected_low=PREDICTED_DELTA_BAND[0],
            expected_high=PREDICTED_DELTA_BAND[1],
            artifact_base_dir=_default_repo_root(),
        )
    )


def l5_v2_packetir_stack_evidence_payload(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return PR106 PacketIR rows as axis-labelled stack evidence only.

    PR106/R2 PacketIR rows can inform stack-of-stacks planning, but they are not
    TT5L score evidence. This surface keeps that distinction machine-readable.
    """

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    matrix_path = resolved_repo_root / PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH
    blockers: list[str] = []
    matrix: Mapping[str, Any] = {}
    artifact_sha = ""
    if not matrix_path.is_file():
        blockers.append("l5_v2_packetir_matrix_artifact_missing")
    else:
        artifact_sha = _sha256_file(matrix_path)
        if artifact_sha != PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256:
            blockers.append("l5_v2_packetir_matrix_artifact_sha_mismatch")
        try:
            loaded = json.loads(matrix_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append("l5_v2_packetir_matrix_artifact_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                matrix = loaded
            else:
                blockers.append("l5_v2_packetir_matrix_artifact_not_object")

    if matrix:
        if matrix.get("score_claim") is not False:
            blockers.append("l5_v2_packetir_matrix_score_claim_not_false")
        if matrix.get("promotion_eligible") is not False:
            blockers.append("l5_v2_packetir_matrix_promotion_eligible_not_false")
        if matrix.get("ready_for_exact_eval_dispatch") is not False:
            blockers.append("l5_v2_packetir_matrix_dispatch_ready_not_false")

    paired_candidates: list[dict[str, Any]] = []
    for row in _mapping_items(matrix.get("rows")):
        if row.get("status") != "paired_exact_measured":
            continue
        runtime_consumption = row.get("runtime_consumption")
        if not isinstance(runtime_consumption, Mapping):
            runtime_consumption = {}
        current_runtime = runtime_consumption.get("current_modal_uploaded_runtime")
        if not isinstance(current_runtime, Mapping):
            current_runtime = {}
        runtime_blockers: list[str] = []
        if runtime_consumption.get("valid") is not True:
            runtime_blockers.append("runtime_consumption_not_valid")
        if runtime_consumption.get("score_claim") is not False:
            runtime_blockers.append("runtime_consumption_score_claim_not_false")
        if runtime_consumption.get("promotion_eligible") is not False:
            runtime_blockers.append(
                "runtime_consumption_promotion_eligible_not_false"
            )
        if runtime_consumption.get("ready_for_exact_eval_dispatch") is not False:
            runtime_blockers.append(
                "runtime_consumption_dispatch_ready_not_false"
            )
        if (
            runtime_consumption.get(
                "runtime_content_tree_sha256_matches_current_runtime_dir"
            )
            is not True
        ):
            runtime_blockers.append(
                "runtime_consumption_current_runtime_content_sha_mismatch"
            )
        if runtime_blockers:
            blockers.append(
                "l5_v2_packetir_matrix_paired_row_runtime_blocked:"
                f"{row.get('candidate_id')}:{','.join(runtime_blockers)}"
            )
            continue
        exact_axis_evidence = row.get("exact_axis_evidence")
        if not isinstance(exact_axis_evidence, Mapping):
            blockers.append(
                "l5_v2_packetir_matrix_paired_row_exact_axis_evidence_missing:"
                f"{row.get('candidate_id')}"
            )
            continue
        axis_payload: dict[str, Any] = {}
        axis_blockers: list[str] = []
        for axis in _REQUIRED_EXACT_AXES:
            evidence = exact_axis_evidence.get(axis)
            if not isinstance(evidence, Mapping):
                axis_blockers.append(f"axis_missing:{axis}")
                continue
            if evidence.get("valid") is not True:
                axis_blockers.append(f"axis_invalid:{axis}")
            if evidence.get("score_claim_in_source_artifact") is not False:
                axis_blockers.append(f"source_score_claim_not_false:{axis}")
            if evidence.get("promotion_eligible_in_source_artifact") is not False:
                axis_blockers.append(f"source_promotion_eligible_not_false:{axis}")
            validation_payload = _packetir_axis_evidence_for_exact_validation(
                evidence,
                axis=axis,
            )
            validation = validate_exact_eval_evidence(
                validation_payload,
                expected_axis=axis,
                expected_archive_sha256=str(row.get("archive_sha256") or "") or None,
                expected_runtime_tree_sha256=(
                    current_runtime.get("runtime_tree_sha256") or None
                ),
                require_artifact_path=True,
                require_hardware=True,
                require_auth_eval_command=True,
                require_log_path=True,
                require_devices=True,
                artifact_base_dir=resolved_repo_root,
                annotation_prefix=f"l5_v2_packetir_{row.get('candidate_id')}_{axis}",
            )
            for validation_blocker in validation.blockers:
                axis_blockers.append(f"exact_eval:{axis}:{validation_blocker}")
            axis_payload[axis] = {
                "valid": evidence.get("valid") is True,
                "canonical_score": evidence.get("canonical_score"),
                "archive_sha256": evidence.get("archive_sha256"),
                "archive_size_bytes": evidence.get("archive_size_bytes"),
                "avg_segnet_dist": evidence.get("avg_segnet_dist"),
                "avg_posenet_dist": evidence.get("avg_posenet_dist"),
                "evidence_grade": evidence.get("evidence_grade"),
                "runtime_tree_sha256": evidence.get("runtime_tree_sha256"),
                "runtime_content_tree_sha256": evidence.get(
                    "runtime_content_tree_sha256"
                ),
                "auth_eval_command": evidence.get("auth_eval_command"),
                "hardware": evidence.get("hardware"),
                "inflate_device": evidence.get("inflate_device"),
                "eval_device": evidence.get("eval_device"),
                "log_path": evidence.get("log_path"),
                "artifact_path": evidence.get("artifact_path"),
                "artifact_sha256": evidence.get("sha256"),
                "path": evidence.get("path"),
                "score_claim_in_source_artifact": evidence.get(
                    "score_claim_in_source_artifact"
                ),
                "promotion_eligible_in_source_artifact": evidence.get(
                    "promotion_eligible_in_source_artifact"
                ),
            }
        if axis_blockers:
            blockers.append(
                "l5_v2_packetir_matrix_paired_row_axis_blocked:"
                f"{row.get('candidate_id')}:{','.join(axis_blockers)}"
            )
            continue
        paired_candidates.append(
            {
                "candidate_id": row.get("candidate_id"),
                "format_id": row.get("format_id"),
                "archive_sha256": row.get("archive_sha256"),
                "archive_path": row.get("archive_path"),
                "notes": row.get("notes"),
                "sidecar_kind": (
                    row.get("sidecar_kind")
                    or runtime_consumption.get("sidecar_kind")
                ),
                "source_artifact_warnings": row.get("source_artifact_warnings", []),
                "runtime_consumption": {
                    "path": runtime_consumption.get("path"),
                    "sha256": runtime_consumption.get("sha256"),
                    "runtime_dir": runtime_consumption.get("runtime_dir"),
                    "runtime_source_tree_sha256": runtime_consumption.get(
                        "runtime_source_tree_sha256"
                    ),
                    "runtime_content_tree_sha256": runtime_consumption.get(
                        "runtime_content_tree_sha256"
                    ),
                    "runtime_content_tree_sha256_source": runtime_consumption.get(
                        "runtime_content_tree_sha256_source"
                    ),
                    "runtime_content_tree_sha256_derived_not_direct_manifested": (
                        runtime_consumption.get(
                            "runtime_content_tree_sha256_derived_not_direct_manifested"
                        )
                    ),
                    "runtime_content_tree_sha256_backfill_required": (
                        runtime_consumption.get(
                            "runtime_content_tree_sha256_backfill_required"
                        )
                    ),
                    "current_runtime_content_tree_sha256": current_runtime.get(
                        "runtime_content_tree_sha256"
                    ),
                    "current_runtime_tree_sha256": current_runtime.get(
                        "runtime_tree_sha256"
                    ),
                    "runtime_content_tree_sha256_matches_current_runtime_dir": (
                        runtime_consumption.get(
                            "runtime_content_tree_sha256_matches_current_runtime_dir"
                        )
                    ),
                },
                "axis_evidence": axis_payload,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    if matrix and not paired_candidates:
        blockers.append("l5_v2_packetir_no_runtime_bound_paired_exact_candidates")

    return {
        "schema": L5_V2_PACKETIR_STACK_EVIDENCE_SCHEMA,
        "source_matrix_artifact_path": PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH,
        "source_matrix_artifact_sha256": artifact_sha,
        "source_matrix_expected_sha256": (
            PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256
        ),
        "source_matrix_schema": matrix.get("schema"),
        "source_candidate_count": matrix.get("candidate_count", 0),
        "source_status_counts": matrix.get("status_counts", {}),
        "paired_candidate_count": len(paired_candidates),
        "paired_candidates": paired_candidates,
        "axis_semantics": {
            "contest_cpu": "kept separate from contest_cuda; no conversion",
            "contest_cuda": "kept separate from contest_cpu; no conversion",
        },
        "evidence_semantics": (
            "PR106 PacketIR exact rows are stack-planning evidence only; "
            "they are not TT5L score evidence and do not unlock promotion"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def l5_v2_packetir_section_entropy_evidence_payload(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return charged PR106 PacketIR section-entropy evidence.

    The section matrix is a negative/redirect signal for L5 v2 PacketIR work:
    real PR106 Format0C/0D streams have large unpriced context floors, but the
    charged static-context prototypes are byte-negative after model overhead.
    Keep it visible to the staircase planner without granting score authority.
    """

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    matrix_path = (
        resolved_repo_root / L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH
    )
    blockers: list[str] = []
    matrix: Mapping[str, Any] = {}
    artifact_sha = ""
    if not matrix_path.is_file():
        blockers.append("l5_v2_packetir_section_entropy_matrix_artifact_missing")
    else:
        artifact_sha = _sha256_file(matrix_path)
        if artifact_sha != L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256:
            blockers.append(
                "l5_v2_packetir_section_entropy_matrix_artifact_sha_mismatch"
            )
        try:
            loaded = json.loads(matrix_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append("l5_v2_packetir_section_entropy_matrix_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                matrix = loaded
            else:
                blockers.append("l5_v2_packetir_section_entropy_matrix_not_object")

    if matrix:
        if matrix.get("schema") != "l5_v2_packetir_section_entropy_matrix_v1":
            blockers.append("l5_v2_packetir_section_entropy_matrix_schema_mismatch")
        if matrix.get("score_claim") is not False:
            blockers.append("l5_v2_packetir_section_entropy_score_claim_not_false")
        if matrix.get("promotion_eligible") is not False:
            blockers.append(
                "l5_v2_packetir_section_entropy_promotion_eligible_not_false"
            )
        if matrix.get("ready_for_exact_eval_dispatch") is not False:
            blockers.append(
                "l5_v2_packetir_section_entropy_dispatch_ready_not_false"
            )

    prototype_rows = [
        prototype
        for row in _mapping_items(matrix.get("rows"))
        for prototype in _mapping_items(row.get("prototype_rows"))
    ]
    profiled_candidate_count = matrix.get("profiled_candidate_count", 0)
    prototype_row_count = matrix.get("prototype_row_count", 0)
    rate_positive_count = matrix.get("rate_positive_prototype_row_count", 0)
    adaptive_row_count = matrix.get("adaptive_prototype_row_count", 0)
    adaptive_rate_positive_count = matrix.get(
        "rate_positive_adaptive_prototype_row_count", 0
    )
    derived_prefix_adaptive_row_count = matrix.get(
        "derived_prefix_adaptive_prototype_row_count", 0
    )
    derived_prefix_adaptive_rate_positive_count = matrix.get(
        "rate_positive_derived_prefix_adaptive_prototype_row_count", 0
    )
    if not isinstance(profiled_candidate_count, int) or isinstance(
        profiled_candidate_count, bool
    ):
        blockers.append(
            "l5_v2_packetir_section_entropy_profiled_candidate_count_not_int"
        )
        profiled_candidate_count = 0
    if not isinstance(prototype_row_count, int) or isinstance(
        prototype_row_count, bool
    ):
        blockers.append("l5_v2_packetir_section_entropy_prototype_row_count_not_int")
        prototype_row_count = 0
    if not isinstance(rate_positive_count, int) or isinstance(rate_positive_count, bool):
        blockers.append(
            "l5_v2_packetir_section_entropy_rate_positive_count_not_int"
        )
        rate_positive_count = 0
    if not isinstance(adaptive_row_count, int) or isinstance(adaptive_row_count, bool):
        blockers.append("l5_v2_packetir_section_entropy_adaptive_row_count_not_int")
        adaptive_row_count = 0
    if not isinstance(adaptive_rate_positive_count, int) or isinstance(
        adaptive_rate_positive_count, bool
    ):
        blockers.append(
            "l5_v2_packetir_section_entropy_adaptive_rate_positive_count_not_int"
        )
        adaptive_rate_positive_count = 0
    if not isinstance(derived_prefix_adaptive_row_count, int) or isinstance(
        derived_prefix_adaptive_row_count, bool
    ):
        blockers.append(
            "l5_v2_packetir_section_entropy_derived_prefix_adaptive_count_not_int"
        )
        derived_prefix_adaptive_row_count = 0
    if not isinstance(
        derived_prefix_adaptive_rate_positive_count, int
    ) or isinstance(derived_prefix_adaptive_rate_positive_count, bool):
        blockers.append(
            "l5_v2_packetir_section_entropy_derived_prefix_adaptive_rate_positive_count_not_int"
        )
        derived_prefix_adaptive_rate_positive_count = 0
    if prototype_rows and prototype_row_count != len(prototype_rows):
        blockers.append("l5_v2_packetir_section_entropy_prototype_count_mismatch")
    adaptive_rows = [
        prototype
        for row in _mapping_items(matrix.get("rows"))
        for prototype in _mapping_items(row.get("adaptive_prototype_rows"))
    ]
    derived_prefix_adaptive_rows = [
        prototype
        for row in _mapping_items(matrix.get("rows"))
        for prototype in _mapping_items(
            row.get("derived_prefix_adaptive_prototype_rows")
        )
    ]
    if adaptive_rows and adaptive_row_count != len(adaptive_rows):
        blockers.append("l5_v2_packetir_section_entropy_adaptive_count_mismatch")
    if (
        derived_prefix_adaptive_rows
        and derived_prefix_adaptive_row_count != len(derived_prefix_adaptive_rows)
    ):
        blockers.append(
            "l5_v2_packetir_section_entropy_derived_prefix_adaptive_count_mismatch"
        )

    best_charged_prototype = None
    best_delta = None
    for prototype in prototype_rows:
        delta = _json_float(prototype.get("delta_bytes_vs_source_section"))
        if delta is None:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_charged_prototype = {
                "section_name": prototype.get("section_name"),
                "context_order": prototype.get("context_order"),
                "source_section_bytes": prototype.get("source_section_bytes"),
                "encoded_section_bytes": prototype.get("encoded_section_bytes"),
                "delta_bytes_vs_source_section": delta,
                "context_model_bytes": prototype.get("context_model_bytes"),
                "range_stream_bytes": prototype.get("range_stream_bytes"),
                "blockers": list(prototype.get("blockers", [])),
            }
    if matrix and rate_positive_count == 0:
        blockers.append(
            "l5_v2_packetir_static_context_recode_no_rate_positive_prototypes"
        )
    if matrix and adaptive_row_count and adaptive_rate_positive_count == 0:
        blockers.append(
            "l5_v2_packetir_adaptive_context_recode_no_rate_positive_prototypes"
        )

    return {
        "schema": L5_V2_PACKETIR_SECTION_ENTROPY_EVIDENCE_SCHEMA,
        "source_matrix_artifact_path": (
            L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH
        ),
        "source_matrix_artifact_sha256": artifact_sha,
        "source_matrix_expected_sha256": (
            L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256
        ),
        "source_matrix_schema": matrix.get("schema"),
        "profiled_candidate_count": profiled_candidate_count,
        "prototype_row_count": prototype_row_count,
        "rate_positive_prototype_row_count": rate_positive_count,
        "adaptive_prototype_row_count": adaptive_row_count,
        "rate_positive_adaptive_prototype_row_count": adaptive_rate_positive_count,
        "derived_prefix_adaptive_prototype_row_count": (
            derived_prefix_adaptive_row_count
        ),
        "rate_positive_derived_prefix_adaptive_prototype_row_count": (
            derived_prefix_adaptive_rate_positive_count
        ),
        "best_rate_positive_prototype": matrix.get("best_rate_positive_prototype"),
        "best_adaptive_prototype": matrix.get("best_adaptive_prototype"),
        "best_rate_positive_adaptive_prototype": matrix.get(
            "best_rate_positive_adaptive_prototype"
        ),
        "best_derived_prefix_adaptive_prototype": matrix.get(
            "best_derived_prefix_adaptive_prototype"
        ),
        "best_rate_positive_derived_prefix_adaptive_prototype": matrix.get(
            "best_rate_positive_derived_prefix_adaptive_prototype"
        ),
        "best_charged_prototype": best_charged_prototype,
        "evidence_semantics": (
            "Charged static-context PacketIR section recodes are planning "
            "evidence only; derived-prefix adaptive rows may be byte-positive "
            "only when the section magic is recoverable from the PacketIR grammar, "
            "and still require runtime decoder integration plus full-frame parity."
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def _json_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def _delta_if_both_present(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def l5_v2_pr106_stack_cell_candidates(
    *,
    repo_root: str | Path | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Propose TT5L x PR106 PacketIR stack cells without promotion authority."""

    packetir_evidence = l5_v2_packetir_stack_evidence_payload(repo_root=repo_root)
    candidates: list[dict[str, Any]] = []
    for row in _mapping_items(packetir_evidence.get("paired_candidates")):
        axis_evidence = row.get("axis_evidence")
        if not isinstance(axis_evidence, Mapping):
            continue
        scores: dict[str, float] = {}
        component_distances: dict[str, dict[str, float]] = {}
        archive_sizes: list[int] = []
        for axis in _REQUIRED_EXACT_AXES:
            axis_row = axis_evidence.get(axis)
            if not isinstance(axis_row, Mapping):
                continue
            score = _json_float(axis_row.get("canonical_score"))
            if score is not None:
                scores[axis] = score
            seg = _json_float(axis_row.get("avg_segnet_dist"))
            pose = _json_float(axis_row.get("avg_posenet_dist"))
            component_distances[axis] = {}
            if seg is not None:
                component_distances[axis]["avg_segnet_dist"] = seg
            if pose is not None:
                component_distances[axis]["avg_posenet_dist"] = pose
            archive_size = _non_bool_int(axis_row.get("archive_size_bytes"))
            if archive_size is not None and archive_size >= 0:
                archive_sizes.append(archive_size)
        source_worst_axis_score = max(scores.values()) if scores else None
        cpu_score = scores.get("contest_cpu")
        cuda_score = scores.get("contest_cuda")
        cpu_components = component_distances.get("contest_cpu", {})
        cuda_components = component_distances.get("contest_cuda", {})
        cpu_cuda_component_delta = {
            "canonical_score": _delta_if_both_present(cpu_score, cuda_score),
            "avg_segnet_dist": _delta_if_both_present(
                cpu_components.get("avg_segnet_dist"),
                cuda_components.get("avg_segnet_dist"),
            ),
            "avg_posenet_dist": _delta_if_both_present(
                cpu_components.get("avg_posenet_dist"),
                cuda_components.get("avg_posenet_dist"),
            ),
        }
        runtime_consumption = row.get("runtime_consumption")
        if not isinstance(runtime_consumption, Mapping):
            runtime_consumption = {}
        candidates.append(
            {
                "cell_id": f"{SUBJECT_ID}+{row.get('candidate_id')}",
                "l5_subject_id": SUBJECT_ID,
                "packetir_candidate_id": row.get("candidate_id"),
                "packetir_format_id": row.get("format_id"),
                "packetir_sidecar_kind": row.get("sidecar_kind"),
                "packetir_notes": row.get("notes"),
                "packetir_source_artifact_warnings": row.get(
                    "source_artifact_warnings",
                    [],
                ),
                "source_archive_sha256": row.get("archive_sha256"),
                "source_archive_path": row.get("archive_path"),
                "source_axis_scores": scores,
                "source_axis_component_distances": component_distances,
                "source_cpu_cuda_score_gap": cpu_cuda_component_delta[
                    "canonical_score"
                ],
                "source_cpu_minus_cuda_component_delta": cpu_cuda_component_delta,
                "source_worst_axis_score": source_worst_axis_score,
                "source_max_archive_size_bytes": max(archive_sizes)
                if archive_sizes
                else None,
                "source_runtime_dir": runtime_consumption.get("runtime_dir"),
                "source_runtime_content_tree_sha256": runtime_consumption.get(
                    "runtime_content_tree_sha256"
                ),
                "current_runtime_content_tree_sha256": runtime_consumption.get(
                    "current_runtime_content_tree_sha256"
                ),
                "source_runtime_content_tree_sha256_matches_current_runtime_dir": (
                    runtime_consumption.get(
                        "runtime_content_tree_sha256_matches_current_runtime_dir"
                    )
                ),
                "source_runtime_content_tree_sha256_source": runtime_consumption.get(
                    "runtime_content_tree_sha256_source"
                ),
                "source_runtime_content_tree_sha256_derived_not_direct_manifested": (
                    runtime_consumption.get(
                        "runtime_content_tree_sha256_derived_not_direct_manifested"
                    )
                ),
                "source_runtime_content_tree_sha256_backfill_required": (
                    runtime_consumption.get(
                        "runtime_content_tree_sha256_backfill_required"
                    )
                ),
                "selection_basis": (
                    "paired PR106 PacketIR source rows only; lower max paired "
                    "axis score is labelled as worst-axis score and sorts first; "
                    "this is not a composite score claim"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": [
                    "requires_l5_v2_composite_archive_materialization",
                    "requires_l5_v2_composite_paired_exact_eval",
                    "requires_composite_sideinfo_consumption_proof",
                ],
            }
        )
    candidates.sort(
        key=lambda item: (
            float("inf")
            if item["source_worst_axis_score"] is None
            else float(item["source_worst_axis_score"]),
            float("inf")
            if item["source_max_archive_size_bytes"] is None
            else float(item["source_max_archive_size_bytes"]),
            str(item["packetir_candidate_id"] or ""),
        )
    )
    if top_k is not None:
        candidates = candidates[: max(int(top_k), 0)]
    blockers = list(packetir_evidence.get("blockers", []))
    if not candidates:
        blockers.append("l5_v2_pr106_stack_cell_candidates_missing")
    return {
        "schema": L5_V2_PR106_STACK_CELL_CANDIDATES_SCHEMA,
        "subject_id": SUBJECT_ID,
        "source_packetir_evidence_schema": packetir_evidence.get("schema"),
        "source_packetir_paired_candidate_count": packetir_evidence.get(
            "paired_candidate_count",
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_semantics": (
            "stack-cell proposal surface only; every cell requires a real "
            "composite archive, side-info consumption proof, and paired exact eval"
        ),
        "blockers": blockers,
    }


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
                predicate_passed=_require_literal_json_bool(
                    value.get("predicate_passed", False),
                    field_name="predicate_passed",
                ),
                evidence_grade=str(value.get("evidence_grade", "")),
            )
        if evidence.gate_id:
            coerced[evidence.gate_id] = evidence
    return coerced


def l5_v2_canonical_sideinfo_gate_evidence(
    *,
    repo_root: str | Path | None = None,
) -> L5V2GateEvidence | None:
    """Return the strongest discoverable TT5L side-info proof."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    candidates = (
        (
            TT5L_CONTEST_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH,
            None,
            "contest_full_frame_inflate_consumption_proof",
        ),
        (
            TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH,
            TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256,
            "legacy_local_parser_and_inflate_consumption_proof",
        ),
    )
    for artifact_path, expected_sha, evidence_grade in candidates:
        proof_path = resolved_repo_root / artifact_path
        if not proof_path.is_file():
            continue
        artifact_sha = _sha256_file(proof_path)
        if expected_sha is not None and artifact_sha != expected_sha:
            continue
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(proof, Mapping):
            continue
        if _gate_semantic_blockers(
            "byte_closed_temporal_sideinfo_consumption",
            proof,
            repo_root=resolved_repo_root,
        ):
            continue
        predicate_id = str(
            proof.get("predicate_id") or TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID
        ).strip()
        return L5V2GateEvidence(
            gate_id="byte_closed_temporal_sideinfo_consumption",
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha,
            predicate_id=predicate_id,
            predicate_passed=True,
            evidence_grade=evidence_grade,
        )
    return None


def _gate_payload_by_id(readiness: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    gates = readiness.get("gates", [])
    if not isinstance(gates, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, Mapping):
            continue
        gate_id = str(gate.get("gate_id") or "")
        if gate_id:
            out[gate_id] = gate
    return out


def _gate_evidence_valid(readiness: Mapping[str, Any], gate_id: str) -> bool:
    gate = _gate_payload_by_id(readiness).get(gate_id)
    return bool(gate and gate.get("evidence_valid") is True)


def _tt5l_dykstra_feasibility_status(*, repo_root: Path) -> dict[str, Any]:
    """Return the TT5L cargo-cult-unwind feasibility artifact status.

    This is deliberately separate from score/eval gates. The Dykstra artifact
    is a planning-control requirement that prevents the retired five-move
    additive score band from quietly re-entering TT5L dispatch decisions.
    """

    tool_path = repo_root / TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH
    artifact_path = repo_root / TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not tool_path.is_file():
        blockers.append("tt5l_dykstra_feasibility_tool_missing")
    if not artifact_path.is_file():
        blockers.append("tt5l_dykstra_feasibility_artifact_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("tt5l_dykstra_feasibility_artifact_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
                if not payload:
                    blockers.append("tt5l_dykstra_feasibility_artifact_empty")
            else:
                blockers.append("tt5l_dykstra_feasibility_artifact_not_object")

    substrate_id = str(payload.get("substrate_id") or "")
    if payload and substrate_id not in {TT5L_DYKSTRA_SUBSTRATE_ID, SUBJECT_ID, LANE_ID}:
        blockers.append("tt5l_dykstra_feasibility_substrate_id_mismatch")
    verdict = str(payload.get("verdict") or "")
    if payload and verdict not in {"FEASIBLE", "INFEASIBLE", "INDETERMINATE"}:
        blockers.append("tt5l_dykstra_feasibility_verdict_invalid")
    archive_size_bytes = _non_bool_int(payload.get("archive_size_bytes"))
    if payload and (archive_size_bytes is None or archive_size_bytes <= 0):
        blockers.append("tt5l_dykstra_feasibility_archive_size_bytes_missing")
    for field in ("feasibility_band_lo", "feasibility_band_hi"):
        value = _json_float(payload.get(field))
        if payload and value is None:
            blockers.append(f"tt5l_dykstra_feasibility_{field}_missing")
    score_formula = str(payload.get("score_formula") or "")
    if payload and score_formula != TT5L_DYKSTRA_SCORE_FORMULA:
        blockers.append("tt5l_dykstra_feasibility_score_formula_missing_or_stale")
    contest_seg_multiplier = _json_float(payload.get("contest_seg_multiplier"))
    if payload and contest_seg_multiplier != 100.0:
        blockers.append("tt5l_dykstra_feasibility_seg_multiplier_missing_or_stale")
    projection_kind = str(payload.get("polytope_projection_kind") or "")
    if payload and projection_kind != TT5L_DYKSTRA_PROJECTION_KIND:
        blockers.append("tt5l_dykstra_feasibility_projection_kind_missing_or_stale")
    constraint_ids_payload = payload.get("constraint_set_ids")
    if isinstance(constraint_ids_payload, list):
        constraint_ids = {str(item) for item in constraint_ids_payload}
    else:
        constraint_ids = set()
    if payload and not TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS.issubset(constraint_ids):
        blockers.append("tt5l_dykstra_feasibility_five_move_constraints_missing")
    if payload.get("score_claim") is True:
        blockers.append("tt5l_dykstra_feasibility_score_claim_true")
    if payload.get("promotion_eligible") is True:
        blockers.append("tt5l_dykstra_feasibility_promotion_eligible_true")
    if payload.get("ready_for_exact_eval_dispatch") is True:
        blockers.append("tt5l_dykstra_feasibility_dispatch_ready_true")

    valid = not blockers
    return {
        "schema": "l5_v2_tt5l_dykstra_feasibility_status_v1",
        "tool_path": TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH,
        "tool_exists": tool_path.is_file(),
        "artifact_path": TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": valid,
        "substrate_id": substrate_id or None,
        "verdict": verdict or None,
        "archive_size_bytes": archive_size_bytes,
        "feasibility_band_lo": payload.get("feasibility_band_lo"),
        "feasibility_band_hi": payload.get("feasibility_band_hi"),
        "feasibility_rationale": payload.get("feasibility_rationale"),
        "score_formula": score_formula or None,
        "contest_seg_multiplier": contest_seg_multiplier,
        "polytope_projection_kind": projection_kind or None,
        "constraint_set_ids": sorted(constraint_ids),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def _l5_v2_tt5l_campaign_readiness_from_dispatch_readiness(
    readiness: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Return the non-PR106 TT5L campaign action visible to operators."""

    sideinfo_valid = _gate_evidence_valid(
        readiness,
        "byte_closed_temporal_sideinfo_consumption",
    )
    probe_valid = _gate_evidence_valid(
        readiness,
        "c1_z5_tt5l_probe_disambiguator",
    )
    paired_axis_plan_valid = _gate_evidence_valid(
        readiness,
        "paired_cpu_cuda_axis_plan",
    )
    anchor_pair_valid = _gate_evidence_valid(
        readiness,
        "exact_anchor_or_diagnostic_pair",
    )
    proof_tool_exists = (repo_root / TT5L_CONTEST_SIDEINFO_PROOF_TOOL_PATH).is_file()
    dispatch_recipe_exists = (repo_root / TT5L_MODAL_A100_DISPATCH_RECIPE_PATH).is_file()
    probe_tool_exists = (repo_root / L5V2_PROBE_TOOL_PATH).is_file()
    dykstra_status = _tt5l_dykstra_feasibility_status(repo_root=repo_root)
    dykstra_valid = dykstra_status["artifact_valid"] is True

    blockers = [
        str(blocker)
        for blocker in readiness.get("blockers", [])
        if str(blocker)
    ]
    if not proof_tool_exists:
        blockers.append("tt5l_contest_sideinfo_proof_tool_missing")
    if not dispatch_recipe_exists:
        blockers.append("tt5l_modal_a100_dispatch_recipe_missing")
    if not probe_tool_exists:
        blockers.append("l5_v2_probe_disambiguator_tool_missing")
    blockers.extend(str(blocker) for blocker in dykstra_status["blockers"])

    if not dykstra_valid:
        next_action = {
            "action_id": "run_tt5l_dykstra_feasibility_polytope",
            "phase": "cargo_cult_unwind_feasibility",
            "command_template": (
                f".venv/bin/python {TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH} "
                f"--substrate-id {TT5L_DYKSTRA_SUBSTRATE_ID} "
                "--predicted-band-lo 0.150 --predicted-band-hi 0.170 "
                "--archive-size-bytes <tt5l_target_or_candidate_archive_bytes> "
                f"--output-json {TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH}"
            ),
            "expected_artifacts": [TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH],
            "cargo_cult_unwind_basis": (
                "retired additive five-move TT5L score band must be projected "
                "through the Dykstra feasibility artifact before side-info "
                "proofs, timing smokes, or dispatch planning"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif not sideinfo_valid:
        next_action = {
            "action_id": "materialize_tt5l_contest_full_frame_sideinfo_consumption_proof",
            "phase": "sideinfo_consumption_proof",
            "command_template": (
                f".venv/bin/python {TT5L_CONTEST_SIDEINFO_PROOF_TOOL_PATH} "
                "--baseline-archive <tt5l_baseline_archive_0.bin> "
                "--mutated-archive <tt5l_sideinfo_mutated_archive_0.bin> "
                "--baseline-output-dir <baseline_inflated_raw_dir> "
                "--mutated-output-dir <mutated_inflated_raw_dir> "
                "--file-list <contest_file_list.txt> "
                "--artifact-out "
                "experiments/results/time_traveler_l5_v2/"
                "tt5l_contest_sideinfo_consumption_proof.json "
                "--manifest-out "
                "experiments/results/time_traveler_l5_v2/"
                "tt5l_contest_sideinfo_outputs_manifest.json"
            ),
            "expected_artifacts": [
                "experiments/results/time_traveler_l5_v2/"
                "tt5l_contest_sideinfo_consumption_proof.json",
                "experiments/results/time_traveler_l5_v2/"
                "tt5l_contest_sideinfo_outputs_manifest.json",
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif not probe_valid:
        next_action = {
            "action_id": "emit_c1_z5_tt5l_probe_template",
            "phase": "probe_disambiguator",
            "command_template": (
                f".venv/bin/python {L5V2_PROBE_TOOL_PATH} "
                f"--emit-template --output-json {TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH}"
            ),
            "expected_artifacts": [TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif not paired_axis_plan_valid:
        next_action = {
            "action_id": "prepare_tt5l_paired_cpu_cuda_axis_plan",
            "phase": "paired_axis_plan",
            "recipe_path": TT5L_MODAL_A100_DISPATCH_RECIPE_PATH,
            "claim_lane_before_dispatch": True,
            "lane_id": LANE_ID,
            "command_template": (
                ".venv/bin/python tools/claim_lane_dispatch.py claim "
                f"--lane-id {LANE_ID} --notes tt5l_l5_v2_first_anchor && "
                "<run TT5L CPU/CUDA paired timing smoke from recipe>"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif not anchor_pair_valid:
        next_action = {
            "action_id": "materialize_tt5l_exact_or_diagnostic_anchor_pair",
            "phase": "first_anchor_pair",
            "recipe_path": TT5L_MODAL_A100_DISPATCH_RECIPE_PATH,
            "claim_lane_before_dispatch": True,
            "lane_id": LANE_ID,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    else:
        next_action = {
            "action_id": "build_tt5l_stack_of_stacks_candidate",
            "phase": "stack_of_stacks",
            "command_template": (
                "compose TT5L with the strongest byte-closed orthogonal "
                "winner after component anchors exist"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    tt5l_blockers = list(dict.fromkeys(blockers))
    return {
        "schema": "l5_v2_tt5l_campaign_readiness_v1",
        "subject_id": SUBJECT_ID,
        "lane_id": LANE_ID,
        "campaign_id": CAMPAIGN_ID,
        "priority": "tt5l_first_non_pr106_l5_v2_staircase",
        "non_pr106_staircase_priority": True,
        "packetir_is_optional_stack_evidence": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dykstra_feasibility_artifact_valid": dykstra_valid,
        "dykstra_feasibility_status": dykstra_status,
        "sideinfo_gate_evidence_valid": sideinfo_valid,
        "probe_gate_evidence_valid": probe_valid,
        "paired_axis_plan_evidence_valid": paired_axis_plan_valid,
        "anchor_pair_evidence_valid": anchor_pair_valid,
        "first_anchor_timing_smoke_allowed": dykstra_valid and sideinfo_valid,
        "architecture_lock_allowed": probe_valid and paired_axis_plan_valid,
        "proof_tool_path": TT5L_CONTEST_SIDEINFO_PROOF_TOOL_PATH,
        "proof_tool_exists": proof_tool_exists,
        "probe_tool_path": L5V2_PROBE_TOOL_PATH,
        "probe_tool_exists": probe_tool_exists,
        "dispatch_recipe_path": TT5L_MODAL_A100_DISPATCH_RECIPE_PATH,
        "dispatch_recipe_exists": dispatch_recipe_exists,
        "next_non_pr106_l5_action": next_action,
        "blockers": tt5l_blockers,
    }


def l5_v2_tt5l_campaign_readiness(
    gate_evidence: (
        Mapping[str, L5V2GateEvidence | Mapping[str, Any]]
        | Iterable[L5V2GateEvidence | Mapping[str, Any]]
        | None
    ) = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the TT5L-first L5 v2 campaign status, separate from PR106."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    readiness = l5_v2_dispatch_readiness(
        gate_evidence=gate_evidence,
        repo_root=resolved_repo_root,
    )
    return _l5_v2_tt5l_campaign_readiness_from_dispatch_readiness(
        readiness,
        repo_root=resolved_repo_root,
    )


def _is_transient_artifact_path(path: str) -> bool:
    normalized = path.removeprefix("file:").strip()
    return normalized.startswith(("/tmp/", "/private/tmp/", "/var/tmp/"))


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_artifact_path(path: str, repo_root: Path) -> tuple[Path | None, str | None]:
    normalized = path.removeprefix("file:").strip()
    if not normalized:
        return None, None
    artifact_path = Path(normalized).expanduser()
    resolved = (
        artifact_path.resolve()
        if artifact_path.is_absolute()
        else (repo_root / artifact_path).resolve()
    )
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return resolved, "outside_repo"
    return resolved, None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_json_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _string_items(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Iterable) or isinstance(value, bytes):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _mapping_items(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _axis_row_map(value: object) -> dict[str, Mapping[str, Any]]:
    rows = _mapping_items(value)
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        axis = str(row.get("axis") or "").strip()
        if axis in _REQUIRED_EXACT_AXES:
            out[axis] = row
    return out


def _axis_row_duplicate_blockers(
    value: object,
    *,
    gate_id: str,
    section: str,
) -> list[str]:
    seen: set[str] = set()
    blockers: list[str] = []
    for row in _mapping_items(value):
        axis = str(row.get("axis") or "").strip()
        if axis not in _REQUIRED_EXACT_AXES:
            continue
        if axis in seen:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:{section}:duplicate_axis:{axis}"
            )
        seen.add(axis)
    return blockers


def _non_bool_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> object:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _repo_local_existing_file_blockers(
    *,
    value: object,
    repo_root: Path,
    missing_blocker: str,
    invalid_blocker: str,
) -> list[str]:
    path_text = str(value or "").strip()
    if not path_text:
        return [missing_blocker]
    if _is_transient_artifact_path(path_text):
        return [f"{invalid_blocker}:transient"]
    resolved, path_error = _resolve_artifact_path(path_text, repo_root)
    if path_error == "outside_repo":
        return [f"{invalid_blocker}:outside_repo"]
    if resolved is None or not resolved.is_file():
        return [f"{missing_blocker}:file_missing"]
    return []


def _inflated_outputs_manifest_sha_blockers(
    *,
    manifest_value: object,
    expected_aggregate_sha256: str,
    repo_root: Path,
    missing_blocker: str,
    invalid_blocker: str,
) -> list[str]:
    blockers = _repo_local_existing_file_blockers(
        value=manifest_value,
        repo_root=repo_root,
        missing_blocker=missing_blocker,
        invalid_blocker=invalid_blocker,
    )
    if blockers:
        return blockers
    if not _SHA256_HEX_RE.fullmatch(expected_aggregate_sha256):
        return [invalid_blocker]

    resolved, path_error = _resolve_artifact_path(str(manifest_value or ""), repo_root)
    if path_error == "outside_repo" or resolved is None or not resolved.is_file():
        return [invalid_blocker]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"{invalid_blocker}:malformed_json"]
    if not isinstance(payload, Mapping):
        return [f"{invalid_blocker}:manifest_not_object"]
    manifest_sha = _raw_output_aggregate_sha(payload)
    if manifest_sha != expected_aggregate_sha256:
        return [f"{invalid_blocker}:aggregate_sha256_mismatch"]
    return []


def _inflate_command_has_canonical_signature(value: object) -> bool:
    command = str(value or "").strip()
    if not command:
        return False
    if "inflate.sh" not in command:
        return False
    parts = re.findall(r"[^\s]+", command)
    try:
        inflate_index = next(idx for idx, part in enumerate(parts) if part.endswith("inflate.sh"))
    except StopIteration:
        return False
    positional_after = [
        part for part in parts[inflate_index + 1 :] if part and not part.startswith("-")
    ]
    return len(positional_after) >= 3


def _raw_output_aggregate_sha(row: Mapping[str, Any]) -> str:
    for key in (
        "inflated_raw_output_aggregate_sha256",
        "raw_output_aggregate_sha256",
        "inflated_outputs_aggregate_sha256",
        "aggregate_sha256",
    ):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _packetir_axis_evidence_for_exact_validation(
    evidence: Mapping[str, Any],
    *,
    axis: str,
) -> dict[str, Any]:
    """Normalize PacketIR matrix rows into the shared exact-eval contract."""

    return {
        **dict(evidence),
        "axis": evidence.get("axis") or axis,
        "score": evidence.get("score") or evidence.get("canonical_score"),
        "seg_dist": evidence.get("seg_dist") or evidence.get("avg_segnet_dist"),
        "pose_dist": evidence.get("pose_dist") or evidence.get("avg_posenet_dist"),
        "archive_bytes": (
            evidence.get("archive_bytes")
            or evidence.get("archive_size_bytes")
            or evidence.get("archive_zip_bytes")
        ),
    }


def _matching_sha_pair(
    mapping: Mapping[str, Any],
    first_keys: Iterable[str],
    second_keys: Iterable[str],
) -> tuple[str, str]:
    first = str(_first_present(mapping, first_keys) or "").strip().lower()
    second = str(_first_present(mapping, second_keys) or "").strip().lower()
    return first, second


def _contest_full_frame_sideinfo_blockers(
    artifact_payload: Mapping[str, Any],
    proof: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    blockers: list[str] = []
    proof_scope = str(
        artifact_payload.get("proof_scope") or proof.get("proof_scope") or ""
    ).strip()
    if proof_scope not in _TT5L_CONTEST_FULL_FRAME_PROOF_SCOPES:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "proof_scope_not_contest_full_frame"
        )
    n_pairs_hashed = _non_bool_int(
        _first_present(proof, ("n_pairs_hashed", "pair_count", "n_pairs"))
    )
    if n_pairs_hashed != _TT5L_CONTEST_N_PAIRS:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "n_pairs_hashed"
        )
    total_frames = _non_bool_int(
        _first_present(proof, ("total_frames", "frames_hashed", "n_frames"))
    )
    if total_frames != _TT5L_CONTEST_TOTAL_FRAMES:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "total_frames"
        )
    if proof.get("raw_output_shape_compatible") is not True:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "raw_output_shape_compatible"
        )
    frame_nbytes = _non_bool_int(
        _first_present(
            proof,
            (
                "raw_output_frame_nbytes",
                "frame_nbytes",
                "contest_frame_nbytes",
            ),
        )
    )
    if frame_nbytes != _TT5L_CONTEST_RAW_OUTPUT_FRAME_NBYTES:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "raw_output_frame_nbytes"
        )
    file_list_sha = str(proof.get("file_list_sha256") or "").strip().lower()
    if not _SHA256_HEX_RE.fullmatch(file_list_sha):
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "file_list_sha256"
        )
    source_raw_sha, candidate_raw_sha = _matching_sha_pair(
        proof,
        (
            "source_raw_output_aggregate_sha256",
            "baseline_raw_output_aggregate_sha256",
            "baseline_inflated_raw_output_aggregate_sha256",
        ),
        (
            "candidate_raw_output_aggregate_sha256",
            "mutated_raw_output_aggregate_sha256",
            "mutated_inflated_raw_output_aggregate_sha256",
        ),
    )
    if (
        not _SHA256_HEX_RE.fullmatch(source_raw_sha)
        or not _SHA256_HEX_RE.fullmatch(candidate_raw_sha)
        or source_raw_sha == candidate_raw_sha
    ):
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "source_candidate_raw_aggregate_sha_pair"
        )

    manifest_value = proof.get("inflated_outputs_manifest_path") or proof.get(
        "inflated_outputs_manifest"
    )
    resolved, path_error = _resolve_artifact_path(str(manifest_value or ""), repo_root)
    if path_error is None and resolved is not None and resolved.is_file():
        try:
            manifest = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, Mapping):
            manifest_pairs = _non_bool_int(
                _first_present(manifest, ("n_pairs_hashed", "pair_count", "n_pairs"))
            )
            if manifest_pairs != _TT5L_CONTEST_N_PAIRS:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    "inflated_outputs_manifest_path:n_pairs_hashed"
                )
            manifest_frames = _non_bool_int(
                _first_present(manifest, ("total_frames", "frames_hashed", "n_frames"))
            )
            if manifest_frames != _TT5L_CONTEST_TOTAL_FRAMES:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    "inflated_outputs_manifest_path:total_frames"
                )
            manifest_frame_nbytes = _non_bool_int(
                _first_present(
                    manifest,
                    (
                        "raw_output_frame_nbytes",
                        "frame_nbytes",
                        "contest_frame_nbytes",
                    ),
                )
            )
            if manifest_frame_nbytes != _TT5L_CONTEST_RAW_OUTPUT_FRAME_NBYTES:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    "inflated_outputs_manifest_path:raw_output_frame_nbytes"
                )
            manifest_file_list_sha = str(
                manifest.get("file_list_sha256") or ""
            ).strip().lower()
            if manifest_file_list_sha != file_list_sha:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    "inflated_outputs_manifest_path:file_list_sha256"
                )
            manifest_source_sha, manifest_candidate_sha = _matching_sha_pair(
                manifest,
                (
                    "source_raw_output_aggregate_sha256",
                    "baseline_raw_output_aggregate_sha256",
                    "baseline_inflated_raw_output_aggregate_sha256",
                ),
                (
                    "candidate_raw_output_aggregate_sha256",
                    "mutated_raw_output_aggregate_sha256",
                    "mutated_inflated_raw_output_aggregate_sha256",
                ),
            )
            if (
                manifest_source_sha != source_raw_sha
                or manifest_candidate_sha != candidate_raw_sha
            ):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    "inflated_outputs_manifest_path:source_candidate_raw_aggregate_sha_pair"
                )
    return blockers


def _project_probe_verdict_for_recompute_check(
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "schema",
        "tool",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "architecture_lock_allowed",
        "selected_candidate_id",
        "selected_delta",
        "selected_delta_source",
        "required_candidates",
        "required_exact_axes",
        "evaluated_observations",
        "blockers",
        "evidence_semantics",
    )
    return {field: verdict.get(field) for field in fields if field in verdict}


def _paired_row_identity_blockers(
    *,
    gate_id: str,
    rows: dict[str, Mapping[str, Any]],
    section: str,
    require_anchor_type: bool = False,
    repo_root: Path | None = None,
) -> list[str]:
    blockers: list[str] = []
    missing_axes = [axis for axis in _REQUIRED_EXACT_AXES if axis not in rows]
    if missing_axes:
        blockers.append(
            f"l5_v2_gate_artifact_semantics_missing:{gate_id}:{section}:"
            + ",".join(missing_axes)
        )
        return blockers

    archive_shas = {
        str(row.get("archive_sha256") or "").strip().lower() for row in rows.values()
    }
    runtime_shas_by_axis = {
        axis: str(row.get("runtime_tree_sha256") or "").strip().lower()
        for axis, row in rows.items()
    }
    runtime_content_shas = {
        str(row.get("runtime_content_tree_sha256") or "").strip().lower()
        for row in rows.values()
    }
    archive_sha = next(iter(archive_shas)) if len(archive_shas) == 1 else ""
    if len(archive_shas) != 1 or not _SHA256_HEX_RE.fullmatch(archive_sha):
        blockers.append(
            f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:{section}:archive_sha256"
        )
        archive_sha = ""
    for axis, runtime_sha in runtime_shas_by_axis.items():
        if not _SHA256_HEX_RE.fullmatch(runtime_sha):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:{section}:{axis}:runtime_tree_sha256"
            )
    if (
        len(runtime_content_shas) != 1
        or not _SHA256_HEX_RE.fullmatch(next(iter(runtime_content_shas), ""))
    ):
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            f"{gate_id}:{section}:runtime_content_tree_sha256"
        )

    for axis, row in rows.items():
        inflate_device = str(row.get("inflate_device") or "").lower()
        eval_device = str(row.get("eval_device") or "").lower()
        if axis == "contest_cpu":
            if not is_contest_cpu_device_text(inflate_device):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cpu_inflate_device"
                )
            if not is_contest_cpu_device_text(eval_device):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cpu_eval_device"
                )
        else:
            if not contains_non_negated_device_token(inflate_device, CUDA_DEVICE_TOKENS):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cuda_inflate_device"
                )
            if not contains_non_negated_device_token(eval_device, CUDA_DEVICE_TOKENS):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cuda_eval_device"
                )
        deltas = row.get("component_deltas")
        if not isinstance(deltas, Mapping):
            blockers.append(
                "l5_v2_gate_artifact_semantics_missing:"
                f"{gate_id}:{section}:{axis}:component_deltas"
            )
        else:
            for field in ("seg_dist_delta", "pose_dist_delta", "score_delta"):
                if not _finite_json_number(deltas.get(field)):
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_invalid:"
                        f"{gate_id}:{section}:{axis}:{field}"
                    )
        anchor_type = str(row.get("anchor_type") or "").strip()
        if require_anchor_type and anchor_type not in {"exact", "diagnostic"}:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:{section}:{axis}:anchor_type"
            )
        elif require_anchor_type:
            if row.get("score_claim") is not False:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:{axis}:score_claim"
                )
            evidence_grade = str(row.get("evidence_grade") or "").strip().lower()
            if anchor_type == "exact" and "contest" not in evidence_grade:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:{axis}:evidence_grade"
                )
            if anchor_type == "exact":
                aggregate_sha = _raw_output_aggregate_sha(row)
                if repo_root is None:
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_missing:"
                        f"{gate_id}:{section}:{axis}:artifact_base_dir"
                    )
                else:
                    blockers.extend(
                        _inflated_outputs_manifest_sha_blockers(
                            manifest_value=(
                                row.get("inflated_outputs_manifest_path")
                                or row.get("inflated_outputs_manifest")
                            ),
                            expected_aggregate_sha256=aggregate_sha,
                            repo_root=repo_root,
                            missing_blocker=(
                                "l5_v2_gate_artifact_semantics_missing:"
                                f"{gate_id}:{section}:{axis}:inflated_outputs_manifest_path"
                            ),
                            invalid_blocker=(
                                "l5_v2_gate_artifact_semantics_invalid:"
                                f"{gate_id}:{section}:{axis}:inflated_outputs_manifest_path"
                            ),
                        )
                    )
                if not _SHA256_HEX_RE.fullmatch(aggregate_sha):
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_invalid:"
                        f"{gate_id}:{section}:{axis}:inflated_raw_output_aggregate_sha256"
                    )
                validation = validate_exact_eval_evidence(
                    row,
                    expected_axis=axis,
                    expected_archive_sha256=archive_sha or None,
                    expected_runtime_tree_sha256=(
                        runtime_shas_by_axis.get(axis) or None
                    ),
                    require_artifact_path=True,
                    require_hardware=True,
                    require_auth_eval_command=True,
                    require_log_path=True,
                    require_devices=True,
                    artifact_base_dir=repo_root,
                    annotation_prefix=f"{gate_id}_{section}_{axis}",
                )
                for validation_blocker in validation.blockers:
                    kind = (
                        "missing"
                        if "missing" in validation_blocker
                        else "invalid"
                    )
                    blockers.append(
                        f"l5_v2_gate_artifact_semantics_{kind}:"
                        f"{gate_id}:{section}:{axis}:exact_eval:"
                        f"{validation_blocker}"
                    )
            if anchor_type == "diagnostic":
                if "diagnostic" not in evidence_grade:
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_invalid:"
                        f"{gate_id}:{section}:{axis}:evidence_grade"
                    )
                if not str(row.get("diagnostic_reason") or "").strip():
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_missing:"
                        f"{gate_id}:{section}:{axis}:diagnostic_reason"
                    )
    return blockers


def _gate_semantic_blockers(
    gate_id: str,
    artifact_payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    blockers: list[str] = []
    if gate_id == "byte_closed_temporal_sideinfo_consumption":
        proof = artifact_payload.get("byte_mutation_proof")
        if not isinstance(proof, Mapping):
            return [f"l5_v2_gate_artifact_semantics_missing:{gate_id}:byte_mutation_proof"]
        if proof.get("parser_consumed_bytes") is not True:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:parser_consumed_bytes"
            )
        if proof.get("output_changed") is not True:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:output_changed"
            )
        if str(proof.get("section") or "").strip() not in {
            "temporal_sideinfo",
            "temporal_side_info",
            "tt5l_temporal_sideinfo",
        }:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:byte_mutation_proof:section"
            )
        offsets = proof.get("mutated_byte_offsets")
        valid_offsets = (
            isinstance(offsets, list)
            and bool(offsets)
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in offsets
            )
        )
        if not valid_offsets:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:mutated_byte_offsets"
            )
        section_offset = _non_bool_int(
            _first_present(
                proof,
                (
                    "section_offset",
                    "section_byte_offset",
                    "section_start_offset",
                    "absolute_section_offset",
                ),
            )
        )
        section_nbytes = _non_bool_int(
            _first_present(
                proof,
                (
                    "section_nbytes",
                    "section_bytes",
                    "section_byte_length",
                    "section_length",
                ),
            )
        )
        if (
            section_offset is None
            or section_offset < 0
            or section_nbytes is None
            or section_nbytes <= 0
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:section_byte_range"
            )
        elif valid_offsets:
            section_end = section_offset + section_nbytes
            outside_offsets = [
                int(item)
                for item in offsets
                if not (section_offset <= int(item) < section_end)
            ]
            if outside_offsets:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:byte_mutation_proof:mutated_byte_offsets_outside_section"
                )
        section_sha = str(
            proof.get("section_sha256") or proof.get("parsed_section_sha256") or ""
        ).strip().lower()
        if not _SHA256_HEX_RE.fullmatch(section_sha):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:section_sha256"
            )
        baseline_archive_sha = str(
            proof.get("baseline_archive_sha256") or proof.get("archive_sha256") or ""
        ).strip().lower()
        mutated_archive_sha = str(
            proof.get("mutated_archive_sha256") or ""
        ).strip().lower()
        if (
            not _SHA256_HEX_RE.fullmatch(baseline_archive_sha)
            or not _SHA256_HEX_RE.fullmatch(mutated_archive_sha)
            or baseline_archive_sha == mutated_archive_sha
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:archive_sha_pair"
            )
        runtime_sha = str(proof.get("runtime_tree_sha256") or "").strip().lower()
        if not _SHA256_HEX_RE.fullmatch(runtime_sha):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:runtime_tree_sha256"
            )
        baseline_sha = str(proof.get("baseline_inflate_sha256") or "").strip().lower()
        mutated_sha = str(proof.get("mutated_inflate_sha256") or "").strip().lower()
        if (
            not _SHA256_HEX_RE.fullmatch(baseline_sha)
            or not _SHA256_HEX_RE.fullmatch(mutated_sha)
            or baseline_sha == mutated_sha
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:inflate_sha_pair"
            )
        aggregate_sha = _raw_output_aggregate_sha(proof)
        blockers.extend(
            _inflated_outputs_manifest_sha_blockers(
                manifest_value=(
                    proof.get("inflated_outputs_manifest_path")
                    or proof.get("inflated_outputs_manifest")
                ),
                expected_aggregate_sha256=aggregate_sha,
                repo_root=repo_root,
                missing_blocker=(
                    "l5_v2_gate_artifact_semantics_missing:"
                    f"{gate_id}:byte_mutation_proof:inflated_outputs_manifest_path"
                ),
                invalid_blocker=(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:byte_mutation_proof:inflated_outputs_manifest_path"
                ),
            )
        )
        if not _SHA256_HEX_RE.fullmatch(aggregate_sha):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:inflated_raw_output_aggregate_sha256"
            )
        if not _inflate_command_has_canonical_signature(proof.get("inflate_command")):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:inflate_command"
            )
        blockers.extend(
            _contest_full_frame_sideinfo_blockers(
                artifact_payload,
                proof,
                repo_root=repo_root,
            )
        )
        return blockers

    if gate_id == "c1_z5_tt5l_probe_disambiguator":
        probe = artifact_payload.get("probe_disambiguator")
        if not isinstance(probe, Mapping):
            return [f"l5_v2_gate_artifact_semantics_missing:{gate_id}:probe_disambiguator"]
        if probe.get("schema") != L5V2_PROBE_SCHEMA:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_schema"
            )
        if probe.get("tool_path") != L5V2_PROBE_TOOL_PATH:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_tool_path"
            )
        candidate_ids = set(_string_items(probe.get("candidate_ids")))
        missing_candidates = [
            candidate_id
            for candidate_id in L5V2_CANDIDATES
            if candidate_id not in candidate_ids
        ]
        if missing_candidates:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_missing:{gate_id}:candidates:"
                + ",".join(missing_candidates)
            )
        if probe.get("paired_exact_axes_required") is not True:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:paired_exact_axes_required"
            )
        observation_rows = _mapping_items(
            probe.get("observations") or probe.get("probe_observations")
        )
        if not observation_rows:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_missing:{gate_id}:probe_observations"
            )
        observations = []
        for index, row in enumerate(observation_rows):
            try:
                observations.append(observation_from_mapping(row))
            except (TypeError, ValueError) as exc:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:probe_observation:{index}:{exc}"
                )
        recomputed_verdict: dict[str, Any] | None = None
        recomputed_verdict_sha256 = ""
        if observations and len(observations) == len(observation_rows):
            recomputed_verdict = evaluate_l5_v2_probe(
                observations,
                repo_root=repo_root,
            )
            recomputed_verdict_sha256 = _canonical_json_sha256(recomputed_verdict)
            if recomputed_verdict.get("architecture_lock_allowed") is not True:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:recomputed_architecture_lock_allowed"
                )
            recomputed_blockers = recomputed_verdict.get("blockers")
            if not isinstance(recomputed_blockers, list) or recomputed_blockers:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:recomputed_probe_blockers_nonempty"
                )
        declared_verdict_sha256 = str(
            probe.get("verdict_sha256") or probe.get("probe_verdict_sha256") or ""
        ).strip().lower()
        if not declared_verdict_sha256:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_missing:{gate_id}:probe_verdict_sha256"
            )
        elif not _SHA256_HEX_RE.fullmatch(declared_verdict_sha256):
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_verdict_sha256"
            )
        elif (
            recomputed_verdict_sha256
            and declared_verdict_sha256 != recomputed_verdict_sha256
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:probe_verdict_sha256_mismatch"
            )
        verdict = probe.get("verdict") or probe.get("probe_verdict")
        if not isinstance(verdict, Mapping):
            blockers.append(
                f"l5_v2_gate_artifact_semantics_missing:{gate_id}:probe_verdict"
            )
            return blockers
        if verdict.get("schema") != L5V2_PROBE_SCHEMA:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_verdict_schema"
            )
        if verdict.get("tool") != L5V2_PROBE_TOOL_PATH:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_verdict_tool"
            )
        if verdict.get("architecture_lock_allowed") is not True:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:architecture_lock_allowed"
            )
        if verdict.get("score_claim") is not False:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:score_claim"
            )
        if verdict.get("promotion_eligible") is not False:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:promotion_eligible"
            )
        if verdict.get("ready_for_exact_eval_dispatch") is not False:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:ready_for_exact_eval_dispatch"
            )
        if recomputed_verdict is not None and (
            _canonical_json_sha256(_project_probe_verdict_for_recompute_check(verdict))
            != _canonical_json_sha256(
                _project_probe_verdict_for_recompute_check(recomputed_verdict)
            )
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:probe_verdict_recompute_mismatch"
            )
        verdict_blockers = verdict.get("blockers")
        if not isinstance(verdict_blockers, list):
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_blockers"
            )
        elif verdict_blockers:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_blockers_nonempty"
            )
        verdict_candidates = set(_string_items(verdict.get("required_candidates")))
        missing_verdict_candidates = [
            candidate_id
            for candidate_id in L5V2_CANDIDATES
            if candidate_id not in verdict_candidates
        ]
        if missing_verdict_candidates:
            blockers.append(
                "l5_v2_gate_artifact_semantics_missing:"
                f"{gate_id}:probe_verdict_candidates:"
                + ",".join(missing_verdict_candidates)
            )
        verdict_axes = set(_string_items(verdict.get("required_exact_axes")))
        missing_verdict_axes = [
            axis for axis in _REQUIRED_EXACT_AXES if axis not in verdict_axes
        ]
        if missing_verdict_axes:
            blockers.append(
                "l5_v2_gate_artifact_semantics_missing:"
                f"{gate_id}:probe_verdict_axes:"
                + ",".join(missing_verdict_axes)
            )
        observation_rows = _mapping_items(verdict.get("evaluated_observations"))
        eligible_by_candidate = {
            str(row.get("candidate_id") or ""): row
            for row in observation_rows
            if row.get("eligible_for_architecture_lock") is True
        }
        missing_eligible = [
            candidate_id
            for candidate_id in L5V2_CANDIDATES
            if candidate_id not in eligible_by_candidate
        ]
        if missing_eligible:
            blockers.append(
                "l5_v2_gate_artifact_semantics_missing:"
                f"{gate_id}:eligible_observations:"
                + ",".join(missing_eligible)
            )
        for candidate_id, row in eligible_by_candidate.items():
            row_axes = set(_string_items(row.get("exact_axes")))
            missing_row_axes = [axis for axis in _REQUIRED_EXACT_AXES if axis not in row_axes]
            if missing_row_axes:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_missing:"
                    f"{gate_id}:eligible_observation_axes:{candidate_id}:"
                    + ",".join(missing_row_axes)
                )
        return blockers

    if gate_id == "paired_cpu_cuda_axis_plan":
        rows_obj = artifact_payload.get("paired_axis_plan") or artifact_payload.get(
            "axis_plan"
        )
        rows = _axis_row_map(rows_obj)
        return [
            *_axis_row_duplicate_blockers(
                rows_obj,
                gate_id=gate_id,
                section="paired_axis_plan",
            ),
            *_paired_row_identity_blockers(
                gate_id=gate_id,
                rows=rows,
                section="paired_axis_plan",
            ),
        ]

    if gate_id == "exact_anchor_or_diagnostic_pair":
        rows_obj = artifact_payload.get("anchor_pair") or artifact_payload.get(
            "diagnostic_pair"
        )
        rows = _axis_row_map(rows_obj)
        return [
            *_axis_row_duplicate_blockers(
                rows_obj,
                gate_id=gate_id,
                section="anchor_pair",
            ),
            *_paired_row_identity_blockers(
                gate_id=gate_id,
                rows=rows,
                section="anchor_pair",
                require_anchor_type=True,
                repo_root=repo_root,
            ),
        ]

    return blockers


def _gate_evidence_blockers(
    gate: L5V2Gate,
    evidence: L5V2GateEvidence | None,
    *,
    repo_root: Path,
) -> list[str]:
    if evidence is None:
        return [f"l5_v2_gate_evidence_missing:{gate.gate_id}"]

    blockers: list[str] = []
    resolved_artifact_path: Path | None = None
    path_error: str | None = None
    if not evidence.artifact_path.strip():
        blockers.append(f"l5_v2_gate_artifact_path_missing:{gate.gate_id}")
    elif _is_transient_artifact_path(evidence.artifact_path):
        blockers.append(f"l5_v2_gate_artifact_path_transient:{gate.gate_id}")
    else:
        resolved_artifact_path, path_error = _resolve_artifact_path(
            evidence.artifact_path,
            repo_root,
        )
        if path_error == "outside_repo":
            blockers.append(f"l5_v2_gate_artifact_path_outside_repo:{gate.gate_id}")
        elif resolved_artifact_path is not None and not resolved_artifact_path.is_file():
            blockers.append(f"l5_v2_gate_artifact_file_missing:{gate.gate_id}")
        elif resolved_artifact_path is not None and resolved_artifact_path.is_file():
            try:
                artifact_payload = json.loads(
                    resolved_artifact_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                blockers.append(
                    f"l5_v2_gate_artifact_json_invalid:{gate.gate_id}:{exc.msg}"
                )
            else:
                if not isinstance(artifact_payload, Mapping):
                    blockers.append(f"l5_v2_gate_artifact_json_not_object:{gate.gate_id}")
                else:
                    artifact_gate_id = str(artifact_payload.get("gate_id") or "").strip()
                    if not artifact_gate_id:
                        blockers.append(f"l5_v2_gate_artifact_gate_id_missing:{gate.gate_id}")
                    elif artifact_gate_id != gate.gate_id:
                        blockers.append(
                            "l5_v2_gate_artifact_gate_id_mismatch:"
                            f"{gate.gate_id}:{artifact_gate_id}"
                        )
                    artifact_predicate_id = str(
                        artifact_payload.get("predicate_id") or ""
                    ).strip()
                    if not artifact_predicate_id:
                        blockers.append(
                            f"l5_v2_gate_artifact_predicate_id_missing:{gate.gate_id}"
                        )
                    elif artifact_predicate_id != evidence.predicate_id.strip():
                        blockers.append(
                            "l5_v2_gate_artifact_predicate_id_mismatch:"
                            f"{gate.gate_id}:{artifact_predicate_id}"
                        )
                    if "predicate_passed" in artifact_payload:
                        artifact_passed = artifact_payload["predicate_passed"]
                        artifact_bool_field = "predicate_passed"
                    elif "passed" in artifact_payload:
                        artifact_passed = artifact_payload["passed"]
                        artifact_bool_field = "passed"
                    else:
                        artifact_passed = None
                        artifact_bool_field = ""
                    if artifact_passed is False:
                        blockers.append(
                            f"l5_v2_gate_artifact_predicate_failed:{gate.gate_id}"
                        )
                    elif artifact_passed is not True and artifact_bool_field:
                        blockers.append(
                            "l5_v2_gate_artifact_predicate_non_bool:"
                            f"{gate.gate_id}:{artifact_bool_field}"
                        )
                    elif artifact_passed is not True:
                        blockers.append(
                            f"l5_v2_gate_artifact_predicate_missing:{gate.gate_id}"
                        )
                    blockers.extend(
                        _gate_semantic_blockers(
                            gate.gate_id,
                            artifact_payload,
                            repo_root=repo_root,
                        )
                    )
    if not _SHA256_HEX_RE.fullmatch(evidence.artifact_sha256.strip()):
        blockers.append(f"l5_v2_gate_artifact_sha256_invalid:{gate.gate_id}")
    elif (
        resolved_artifact_path is not None
        and path_error is None
        and resolved_artifact_path.is_file()
        and _sha256_file(resolved_artifact_path)
        != evidence.artifact_sha256.strip().lower()
    ):
        blockers.append(f"l5_v2_gate_artifact_sha256_mismatch:{gate.gate_id}")
    if not evidence.predicate_id.strip():
        blockers.append(f"l5_v2_gate_predicate_id_missing:{gate.gate_id}")
    if evidence.predicate_passed is not True:
        blockers.append(f"l5_v2_gate_predicate_failed:{gate.gate_id}")
    return blockers


def l5_v2_dispatch_readiness(
    satisfied_gate_ids: Mapping[str, bool] | None = None,
    gate_evidence: (
        Mapping[str, L5V2GateEvidence | Mapping[str, Any]]
        | Iterable[L5V2GateEvidence | Mapping[str, Any]]
        | None
    ) = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed dispatch readiness summary for L5 v2.

    Boolean gate claims are preserved as planning notes only. Dispatch
    readiness requires artifact-backed evidence for every gate so prose or
    stale booleans cannot unlock L5 v2 actuation.
    """

    satisfied_gate_ids = satisfied_gate_ids or {}
    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    evidence_by_gate = _coerce_gate_evidence(gate_evidence)
    if "byte_closed_temporal_sideinfo_consumption" not in evidence_by_gate:
        canonical_sideinfo = l5_v2_canonical_sideinfo_gate_evidence(
            repo_root=resolved_repo_root,
        )
        if canonical_sideinfo is not None:
            evidence_by_gate[canonical_sideinfo.gate_id] = canonical_sideinfo
    gates = []
    blockers = []
    evidence_blockers = []
    for gate in l5_v2_required_gates():
        evidence = evidence_by_gate.get(gate.gate_id)
        gate_evidence_blockers = _gate_evidence_blockers(
            gate,
            evidence,
            repo_root=resolved_repo_root,
        )
        gate_claim_blockers: list[str] = []
        evidence_valid = not gate_evidence_blockers
        if gate.gate_id in satisfied_gate_ids:
            raw_claimed_satisfied = satisfied_gate_ids[gate.gate_id]
            if raw_claimed_satisfied is True:
                claimed_satisfied = True
            elif raw_claimed_satisfied is False:
                claimed_satisfied = False
            else:
                claimed_satisfied = False
                gate_claim_blockers.append(f"l5_v2_gate_claim_non_bool:{gate.gate_id}")
        else:
            claimed_satisfied = False
        status = "satisfied" if evidence_valid else gate.status
        gate_payload = {
            **gate.__dict__,
            "status": status,
            "claimed_satisfied": claimed_satisfied,
            "claimed_satisfied_without_artifact": (
                claimed_satisfied and not evidence_valid
            ),
            "evidence_valid": evidence_valid,
            "evidence": evidence.__dict__ if evidence is not None else None,
            "evidence_blockers": gate_evidence_blockers,
            "claim_blockers": gate_claim_blockers,
        }
        gates.append(gate_payload)
        if not evidence_valid:
            blockers.append(gate.blocker)
        evidence_blockers.extend(gate_evidence_blockers)
        evidence_blockers.extend(gate_claim_blockers)
    all_gate_claims_satisfied = all(gate["evidence_valid"] for gate in gates)
    all_gate_evidence_valid = all(gate["evidence_valid"] for gate in gates)
    prediction_band_verdict = l5_v2_prediction_band_verdict()
    prediction_band_rank_ready_raw = prediction_band_verdict.get(
        "valid_for_rank_reward"
    )
    prediction_band_rank_ready = prediction_band_rank_ready_raw is True
    prediction_band_blockers = [
        str(blocker)
        for blocker in prediction_band_verdict.get("blockers", [])
        if str(blocker)
    ]
    score_dispatch_blockers: list[str] = []
    if (
        prediction_band_rank_ready_raw is not True
        and prediction_band_rank_ready_raw is not False
    ):
        score_dispatch_blockers.append(
            "prediction_band_valid_for_rank_reward_non_bool"
        )
    if not prediction_band_rank_ready:
        score_dispatch_blockers.append("prediction_band_not_dispatch_ready")
        score_dispatch_blockers.extend(
            f"prediction_band:{blocker}" for blocker in prediction_band_blockers
        )
    ready_for_score_or_rank_dispatch = (
        all_gate_evidence_valid and prediction_band_rank_ready
    )
    payload = {
        "subject_id": SUBJECT_ID,
        "lane_id": LANE_ID,
        "campaign_id": CAMPAIGN_ID,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "all_gate_claims_satisfied": all_gate_claims_satisfied,
        "all_gate_evidence_valid": all_gate_evidence_valid,
        "ready_for_gate_probe_dispatch": all_gate_evidence_valid,
        "ready_for_score_or_rank_dispatch": ready_for_score_or_rank_dispatch,
        "ready_for_dispatch": ready_for_score_or_rank_dispatch,
        "blockers": blockers + evidence_blockers + score_dispatch_blockers,
        "gates": gates,
        "steps": [step.__dict__ for step in l5_v2_staircase_steps()],
        "prediction_band_verdict": prediction_band_verdict,
        "prediction_band_rank_ready": prediction_band_rank_ready,
        "packetir_stack_evidence": l5_v2_packetir_stack_evidence_payload(
            repo_root=resolved_repo_root,
        ),
        "packetir_section_entropy_evidence": (
            l5_v2_packetir_section_entropy_evidence_payload(
                repo_root=resolved_repo_root,
            )
        ),
        "pr106_stack_cell_candidates": l5_v2_pr106_stack_cell_candidates(
            repo_root=resolved_repo_root,
        ),
    }
    payload["tt5l_campaign_readiness"] = (
        _l5_v2_tt5l_campaign_readiness_from_dispatch_readiness(
            payload,
            repo_root=resolved_repo_root,
        )
    )
    return payload


__all__ = [
    "CAMPAIGN_ID",
    "L5_V2_PACKETIR_SECTION_ENTROPY_EVIDENCE_SCHEMA",
    "L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH",
    "L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256",
    "L5_V2_PACKETIR_STACK_EVIDENCE_SCHEMA",
    "L5_V2_PR106_STACK_CELL_CANDIDATES_SCHEMA",
    "LANE_ID",
    "PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH",
    "PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256",
    "PREDICTED_DELTA_AXIS",
    "PREDICTED_DELTA_BAND",
    "SUBJECT_ID",
    "TT5L_CONTEST_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH",
    "TT5L_CONTEST_SIDEINFO_PROOF_TOOL_PATH",
    "TT5L_MODAL_A100_DISPATCH_RECIPE_PATH",
    "TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH",
    "TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID",
    "TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH",
    "TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256",
    "L5V2Gate",
    "L5V2GateEvidence",
    "L5V2Step",
    "l5_v2_canonical_sideinfo_gate_evidence",
    "l5_v2_dispatch_readiness",
    "l5_v2_packetir_section_entropy_evidence_payload",
    "l5_v2_packetir_stack_evidence_payload",
    "l5_v2_pr106_stack_cell_candidates",
    "l5_v2_prediction_band_payload",
    "l5_v2_prediction_band_verdict",
    "l5_v2_required_gates",
    "l5_v2_research_basis_ids",
    "l5_v2_staircase_steps",
    "l5_v2_tt5l_campaign_readiness",
]
