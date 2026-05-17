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
import subprocess
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.exact_eval_custody import (
    CPU_DEVICE_TOKENS,
    CUDA_DEVICE_TOKENS,
    contains_forbidden_contest_cpu_token,
    contains_non_negated_device_token,
    extract_archive_sha256,
    validate_exact_eval_evidence,
)
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
    L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH,
    L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH,
    L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    L5V2_SIDEINFO_EFFECT_CURVE_TOOL_PATH,
    validate_l5_v2_sideinfo_effect_curve,
)
from tac.optimization.l5_v2_paired_measurement_dispatch_plan import (
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH,
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_REPORT_PATH,
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH,
)
from tac.optimization.l5_v2_probe_disambiguator import (
    L5V2_CANDIDATES,
    L5V2_PROBE_SCHEMA,
    L5V2_PROBE_TOOL_PATH,
    evaluate_l5_v2_probe,
    observation_from_mapping,
)
from tac.optimization.l5_v2_probe_intake import (
    L5V2_PROBE_OBSERVATION_INTAKE_SCHEMA,
    L5V2_PROBE_OBSERVATION_INTAKE_TOOL_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_harvest import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH,
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
TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_PATH = (
    ".omx/research/tt5l_contest_sideinfo_consumption_proof_20260516_codex.json"
)
TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_SHA256 = (
    "d430dd7ccc97da125ca3985a2f70d7cef4c37d39cccad1952698d37d177c9a86"
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
    ".omx/research/l5_v2_probe_template_20260516_codex.json"
)
TT5L_PROBE_GATE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_probe_gate_artifact_20260516_codex.json"
)
TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_probe_observation_intake_20260516_codex.json"
)
TT5L_PROBE_OBSERVATION_INTAKE_REPORT_PATH = (
    ".omx/research/l5_v2_probe_observation_intake_20260516_codex.md"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json"
)
TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_materialized_modal_provider_blocker_20260517_codex.json"
)
TT5L_MATERIALIZED_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_alt_provider_plan_20260517_codex.json"
)
TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json"
)
TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA = (
    "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_v1"
)
TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json"
)
TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA = (
    "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_v1"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID = (
    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANES = {
    "contest_cpu": "lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cpu",
    "contest_cuda": "lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda",
}
TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_paired_exact_anchor_pair_20260516_codex.json"
)
TT5L_PAIRED_EXACT_ANCHOR_PAIR_PREDICATE_ID = (
    "l5_v2_tt5l_paired_exact_anchor_pair_v1"
)
TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_paired_cpu_cuda_axis_plan_from_anchor_20260517_codex.json"
)
TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_PREDICATE_ID = (
    "l5_v2_tt5l_paired_cpu_cuda_axis_plan_from_anchor_v1"
)
TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH = "tools/check_substrate_dykstra_feasibility.py"
TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH = (
    ".omx/state/dykstra_feasibility_time_traveler_l5.json"
)
TT5L_DYKSTRA_FEASIBILITY_SCHEMA = "dykstra_feasibility_verdict_v1"
TT5L_DYKSTRA_FEASIBILITY_PREDICATE_ID = "dykstra_score_axis_feasibility_v1"
TT5L_DYKSTRA_FEASIBILITY_GENERATED_BY_TOOL = (
    "tools/check_substrate_dykstra_feasibility.py"
)
TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH = (
    ".omx/state/tt5l_move_level_feasibility.json"
)
TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA = "tt5l_move_level_feasibility_v1"
TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID = (
    "tt5l_move_level_constraint_feasibility_v1"
)
TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH = (
    "tools/build_tt5l_move_level_feasibility_artifact.py"
)
TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH = (
    "tools/build_tt5l_first_anchor_timing_smoke_artifact.py"
)
TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_first_anchor_timing_smoke_20260517_codex.json"
)
TT5L_FIRST_ANCHOR_TIMING_SMOKE_SCHEMA = "tt5l_first_anchor_timing_smoke_v1"
TT5L_FIRST_ANCHOR_TIMING_SMOKE_PREDICATE_ID = (
    "tt5l_first_anchor_timing_smoke_rate_v1"
)
TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH = L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH
TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json"
)
TT5L_DYKSTRA_SUBSTRATE_ID = "time_traveler_l5_5move"
TT5L_DYKSTRA_SCORE_FORMULA = (
    "100*seg_dist+sqrt(10*pose_dist)+25*archive_bytes/37545489"
)
TT5L_DYKSTRA_PROJECTION_KIND = "score_axis_projection_with_declared_constraints"
TT5L_DYKSTRA_FEASIBILITY_SCOPE = "score_axis_sanity_only"
TT5L_DYKSTRA_VERDICT_AUTHORITY_SCOPE = (
    "score_axis_consistent_only_no_move_level_or_score_authority"
)
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
L5_V2_ARCHITECTURE_LOCK_PACKET_SCHEMA = "l5_v2_architecture_lock_packet_v1"
L5_V2_ARCHITECTURE_LOCK_PACKET_ARTIFACT_PATH = (
    ".omx/research/l5_v2_architecture_lock_packet_20260516_codex.json"
)
L5_V2_ARCHITECTURE_LOCK_PACKET_REPORT_PATH = (
    ".omx/research/l5_v2_architecture_lock_packet_20260516_codex.md"
)
L5_V2_ARCHITECTURE_LOCK_PACKET_TOOL_PATH = (
    "tools/build_l5_v2_architecture_lock_packet.py"
)
L5_V2_ASYMPTOTIC_PURSUIT_CANDIDATES_SCHEMA = (
    "l5_v2_asymptotic_pursuit_candidates_v1"
)
L5_V2_ASYMPTOTIC_NEXT_ACTION_STATUS_SCHEMA = (
    "l5_v2_asymptotic_next_action_status_v1"
)
L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_asymptotic_candidate_surface_20260516_codex.json"
)
L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_REPORT_PATH = (
    ".omx/research/l5_v2_asymptotic_candidate_surface_20260516_codex.md"
)
Z6_IDENTITY_PREDICTOR_DISAMBIGUATOR_TOOL_PATH = (
    "tools/probe_z6_predictive_coding_vs_identity_disambiguator.py"
)
Z6_REAL_VIDEO_EGO_PROXY_SWEEP_TOOL_PATH = (
    "tools/probe_z6_real_video_ego_proxy_sweep.py"
)
Z6_REAL_VIDEO_EGO_PROXY_SWEEP_ARTIFACT_PATH = (
    ".omx/research/l5_v2_z6_real_video_ego_proxy_sweep_20260516_codex.json"
)
Z6_REAL_VIDEO_EGO_PROXY_SWEEP_SCHEMA = "z6_real_video_ego_proxy_sweep_v1"
Z6_REAL_VIDEO_EGO_PROXY_SWEEP_IDENTITY_DOMINATES_VERDICT = (
    "identity_dominates_all_tested_ego_proxies_real_video_smoke"
)
Z6_REAL_VIDEO_EGO_PROXY_SWEEP_FULL_FILM_VERDICT = (
    "full_film_proxy_found_real_video_smoke"
)
Z6_SCORER_BEARING_PAIRED_SMOKE_TOOL_PATH = (
    "tools/probe_z6_scorer_bearing_paired_smoke.py"
)
Z6_SCORER_BEARING_PAIRED_SMOKE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_z6_scorer_bearing_paired_smoke_20260517_codex.json"
)
Z6_SCORER_BEARING_PAIRED_SMOKE_SCHEMA = "z6_scorer_bearing_paired_smoke_v1"
Z6_POST_L1_PROXY_EVIDENCE_STATUS_SCHEMA = "z6_post_l1_proxy_evidence_status_v1"
TISHBY_D4_PROBE_ARTIFACT_PATH = (
    ".omx/research/tishby_ib_pure_d4_probe_20260516_codex.json"
)
TISHBY_VIB_TRACTABILITY_ARTIFACT_PATH = (
    ".omx/research/tishby_ib_pure_variational_ib_tractability_20260516_codex.json"
)
TISHBY_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA = (
    "tishby_post_l1_probe_evidence_status_v1"
)
TISHBY_D4_INDEPENDENT_VERDICT = "INDEPENDENT"
TISHBY_VIB_TRACTABLE_VERDICT = "TRACTABLE"
RUDIN_PROXY_DISAMBIGUATOR_TOOL_PATH = (
    "tools/probe_rudin_floor_substrate_disambiguator.py"
)
RUDIN_PROXY_DISAMBIGUATOR_ARTIFACT_PATH = (
    ".omx/research/rudin_floor_proxy_disambiguator_20260516_codex.json"
)
RUDIN_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA = (
    "rudin_post_l1_probe_evidence_status_v1"
)
RUDIN_MEANINGFUL_INTERPRETABILITY_VERDICT = "MEANINGFUL_INTERPRETABILITY"

GateStatus = Literal["required", "satisfied", "blocked"]
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT_HEX_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_REQUIRED_EXACT_AXES = ("contest_cpu", "contest_cuda")
_TT5L_CONTEST_FULL_FRAME_PROOF_SCOPES = frozenset({
    "contest_full_frame_consumption_proof",
    "contest_full_frame_sideinfo_consumption_proof",
})
_TT5L_CONTEST_N_PAIRS = 600
_TT5L_CONTEST_TOTAL_FRAMES = 1200
_TT5L_CONTEST_RAW_OUTPUT_FRAME_NBYTES = 874 * 1164 * 3
_TT5L_LIGHTNING_PAIRED_AXIS_STATIC_SOURCE_PATHS = (
    "tools/build_l5_v2_architecture_lock_packet.py",
    "tools/build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py",
    "tools/build_tt5l_sideinfo_variant_packets.py",
    "src/tac/exact_eval_custody.py",
    "src/tac/optimization/l5_staircase_v2.py",
    "src/tac/optimization/l5_v2_measurement_schedule.py",
    "src/tac/optimization/l5_v2_sideinfo_effect_curve.py",
    "src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py",
    "src/tac/optimization/tt5l_sideinfo_variant_packets.py",
    "src/tac/deploy/lightning/batch_jobs.py",
    "scripts/launch_lightning_batch_job.py",
    "scripts/adjudicate_contest_auth_eval.py",
    "src/tac/substrates/time_traveler_l5_autonomy",
    "submissions/robust_current",
)


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


@dataclass(frozen=True)
class L5V2AsymptoticPursuitCandidate:
    """One design-only asymptotic candidate for the L5 v2 frontier queue."""

    candidate_id: str
    lane_id: str
    title: str
    horizon_class: str
    primary_axis: str
    local_ledger_path: str
    recommended_next_action_id: str
    recommended_next_action: str
    expected_first_artifacts: tuple[str, ...]
    dependency_blockers: tuple[str, ...]
    cost_band_usd: tuple[float, float]


_L5_V2_ASYMPTOTIC_READY_FOR_L1_BUILD_IDS = frozenset({
    "z6_z7_z8_predictive_coding_world_models",
})


_L5_V2_ASYMPTOTIC_PURSUIT_CANDIDATES: tuple[
    L5V2AsymptoticPursuitCandidate, ...
] = (
    L5V2AsymptoticPursuitCandidate(
        candidate_id="z6_z7_z8_predictive_coding_world_models",
        lane_id=(
            "lane_time_traveler_l5_z6_z7_z8_predictive_coding_world_models_"
            "scoping_design_20260516"
        ),
        title="Z6/Z7/Z8 predictive-coding world-model staircase",
        horizon_class="asymptotic_pursuit",
        primary_axis="scorer_relationship_class_shift_predictive_coding",
        local_ledger_path=(
            ".omx/research/"
            "time_traveler_l5_z6_z7_z8_predictive_coding_world_models_"
            "asymptotic_pursuit_scoping_design_20260516.md"
        ),
        recommended_next_action_id="build_z6_l1_scaffold_first",
        recommended_next_action=(
            "Build Z6 L1 scaffold first: package, trainer, recipe, identity-"
            "predictor disambiguator, and smoke-before-full gate."
        ),
        expected_first_artifacts=(
            "src/tac/substrates/time_traveler_l5_z6/",
            "experiments/train_substrate_time_traveler_l5_z6.py",
            Z6_IDENTITY_PREDICTOR_DISAMBIGUATOR_TOOL_PATH,
            (
                ".omx/operator_authorize_recipes/"
                "substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml"
            ),
        ),
        dependency_blockers=(
            "requires_z6_l1_scaffold_before_paid_dispatch",
            "requires_identity_predictor_disambiguator_result_before_paradigm_claim",
            "requires_paired_cpu_cuda_anchor_before_score_or_rank_authority",
        ),
        cost_band_usd=(1.0, 12.5),
    ),
    L5V2AsymptoticPursuitCandidate(
        candidate_id="rudin_floor_interpretable_ml_substrate",
        lane_id="lane_rudin_floor_interpretable_ml_substrate_scoping_design_20260516",
        title="Rudin floor interpretable-ML compositional decoder",
        horizon_class="asymptotic_pursuit",
        primary_axis="architecture_decode_contract_scorer_relationship_class_shift",
        local_ledger_path=(
            ".omx/research/"
            "rudin_floor_interpretable_ml_substrate_"
            "asymptotic_pursuit_scoping_design_20260516.md"
        ),
        recommended_next_action_id="ratify_and_build_rudin_k8_l1_scaffold",
        recommended_next_action=(
            "Run T3 ratification and build K=8 Rudin L1 scaffold with RDIF "
            "archive grammar, pure-Python inflate, and byte-mutation proof."
        ),
        expected_first_artifacts=(
            RUDIN_PROXY_DISAMBIGUATOR_ARTIFACT_PATH,
            RUDIN_PROXY_DISAMBIGUATOR_TOOL_PATH,
            "src/tac/substrates/rudin_floor_interpretable_ml/",
            "experiments/train_substrate_rudin_floor_interpretable_ml.py",
            (
                ".omx/operator_authorize_recipes/"
                "substrate_rudin_floor_interpretable_ml_modal_t4_dispatch.yaml"
            ),
        ),
        dependency_blockers=(
            "requires_t3_ratification_before_l1_scaffold_dispatch",
            "requires_dykstra_feasibility_intersection_before_paid_smoke",
            "requires_byte_mutation_proof_before_score_or_rank_authority",
        ),
        cost_band_usd=(3.0, 15.0),
    ),
    L5V2AsymptoticPursuitCandidate(
        candidate_id="tishby_ib_pure_substrate",
        lane_id="lane_tishby_ib_pure_substrate_scoping_design_20260516",
        title="Tishby IB-pure primary Lagrangian substrate",
        horizon_class="asymptotic_pursuit",
        primary_axis="training_paradigm_scorer_relationship_class_shift",
        local_ledger_path=(
            ".omx/research/"
            "tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_"
            "20260516.md"
        ),
        recommended_next_action_id=(
            "run_d4_probe_and_build_variational_ib_tractability_tool"
        ),
        recommended_next_action=(
            "Run D4 H(latent|scorer_class) probe and build the canonical "
            "Variational-IB tractability checker before any substrate scaffold."
        ),
        expected_first_artifacts=(
            TISHBY_D4_PROBE_ARTIFACT_PATH,
            TISHBY_VIB_TRACTABILITY_ARTIFACT_PATH,
            "tools/check_variational_ib_tractability.py",
            "src/tac/substrates/tishby_ib_pure/",
        ),
        dependency_blockers=(
            "requires_d4_probe_verdict_before_tishby_scaffold",
            "requires_variational_ib_tractability_before_path_vib_or_mine",
            "requires_paired_smoke_vs_atw_v2_before_asymptotic_claim",
        ),
        cost_band_usd=(3.0, 60.0),
    ),
)


def l5_v2_research_basis_ids() -> tuple[str, ...]:
    """Return the source stack that anchors L5 v2 planning claims."""

    return tuple(research_basis_ids_for_family("time_traveler_l5_v2"))


def l5_v2_asymptotic_pursuit_candidates(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return fail-closed design-only L5 v2 asymptotic candidate rows."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    lane_registry_ids, lane_registry_blockers = _l5_v2_lane_registry_ids(
        repo_root=resolved_repo_root,
    )
    rows: list[dict[str, Any]] = []
    aggregate_blockers: list[str] = list(lane_registry_blockers)
    for candidate in _L5_V2_ASYMPTOTIC_PURSUIT_CANDIDATES:
        z6_post_l1_proxy_evidence: dict[str, Any] | None = None
        post_l1_probe_evidence: dict[str, Any] | None = None
        ledger_path = resolved_repo_root / candidate.local_ledger_path
        ledger_present = ledger_path.is_file()
        ledger_sha256 = _sha256_file(ledger_path) if ledger_present else ""
        lane_registry_registered = candidate.lane_id in lane_registry_ids
        expected_first_artifact_status = []
        for artifact_path in candidate.expected_first_artifacts:
            expected_path = resolved_repo_root / artifact_path
            expected_first_artifact_status.append(
                {
                    "path": artifact_path,
                    "present": expected_path.exists(),
                }
            )
        expected_first_artifacts_all_present = all(
            bool(row["present"]) for row in expected_first_artifact_status
        )
        blockers = list(candidate.dependency_blockers)
        if not ledger_present:
            blockers.append(
                "l5_v2_asymptotic_pursuit_ledger_missing:"
                f"{candidate.candidate_id}"
            )
        if not lane_registry_registered:
            blockers.append(
                "l5_v2_asymptotic_pursuit_lane_registry_missing:"
                f"{candidate.candidate_id}:{candidate.lane_id}"
            )
        l1_scaffold_present = expected_first_artifacts_all_present
        ready_for_l1_build = (
            ledger_present
            and lane_registry_registered
            and candidate.candidate_id in _L5_V2_ASYMPTOTIC_READY_FOR_L1_BUILD_IDS
            and not l1_scaffold_present
        )
        l1_build_blockers = []
        if not ready_for_l1_build:
            if l1_scaffold_present:
                l1_build_blockers.append(
                    "l1_scaffold_present_next_action_completed_or_superseded"
                )
            if not ledger_present:
                l1_build_blockers.append(
                    "requires_l5_v2_asymptotic_pursuit_source_ledger"
                )
            if not lane_registry_registered:
                l1_build_blockers.append(
                    "requires_l5_v2_asymptotic_pursuit_lane_registry_entry"
                )
            if not l1_scaffold_present:
                l1_build_blockers.append(
                    f"requires_pre_l1_gate:{candidate.recommended_next_action_id}"
                )
        effective_next_action_id = candidate.recommended_next_action_id
        effective_next_action = candidate.recommended_next_action
        recommended_next_action_status = "pending"
        if l1_scaffold_present:
            recommended_next_action_status = "completed_or_superseded"
            effective_next_action_id = (
                f"completed_or_superseded:{candidate.recommended_next_action_id}"
            )
            effective_next_action = (
                "The originally recommended L1 action is complete or superseded; "
                "advance using l1_build_blockers/dependency_blockers rather than "
                "rebuilding the same scaffold."
            )
        ready_for_recommended_next_action = (
            ledger_present and lane_registry_registered and not l1_scaffold_present
        )
        ready_for_l1_build_semantics = (
            "l1_scaffold_present_next_action_completed"
            if l1_scaffold_present
            else "ready_to_start_l1_scaffold_work_only_not_scaffold_ready"
        )
        post_l1_recommended_next_action_id = ""
        post_l1_recommended_next_action = ""
        post_l1_recommended_next_action_status = "not_applicable"
        if candidate.candidate_id == "z6_z7_z8_predictive_coding_world_models":
            z6_post_l1_proxy_evidence = _z6_post_l1_proxy_evidence_status(
                repo_root=resolved_repo_root,
            )
            post_l1_probe_evidence = z6_post_l1_proxy_evidence
        elif candidate.candidate_id == "rudin_floor_interpretable_ml_substrate":
            post_l1_probe_evidence = _rudin_post_l1_probe_evidence_status(
                repo_root=resolved_repo_root,
            )
        elif candidate.candidate_id == "tishby_ib_pure_substrate":
            post_l1_probe_evidence = _tishby_post_l1_probe_evidence_status(
                repo_root=resolved_repo_root,
            )
        if post_l1_probe_evidence is not None:
            blockers.extend(post_l1_probe_evidence["blockers"])
            aggregate_blockers.extend(post_l1_probe_evidence["blockers"])
            post_l1_recommended_next_action_id = str(
                post_l1_probe_evidence["recommended_next_action_id"]
            )
            post_l1_recommended_next_action = str(
                post_l1_probe_evidence["recommended_next_action"]
            )
            post_l1_recommended_next_action_status = str(
                post_l1_probe_evidence["recommended_next_action_status"]
            )
        next_prerequisite_status = {
            "status": recommended_next_action_status,
            "action_id": effective_next_action_id,
            "action": effective_next_action,
            "ready_for_recommended_next_action": ready_for_recommended_next_action,
            "ready_for_l1_build": ready_for_l1_build,
            "ready_for_l1_scaffold_dispatch": False,
            "l1_scaffold_present": l1_scaffold_present,
            "blockers": list(l1_build_blockers),
            "dependency_blockers": list(candidate.dependency_blockers),
            "post_l1_recommended_next_action_status": (
                post_l1_recommended_next_action_status
            ),
            "post_l1_recommended_next_action_id": (
                post_l1_recommended_next_action_id
            ),
            "post_l1_recommended_next_action": post_l1_recommended_next_action,
        }
        l5_v2_asymptotic_next_action_status = {
            "schema": L5_V2_ASYMPTOTIC_NEXT_ACTION_STATUS_SCHEMA,
            "candidate_id": candidate.candidate_id,
            "lane_id": candidate.lane_id,
            "local_ledger_path": candidate.local_ledger_path,
            "ledger_present": ledger_present,
            "ledger_sha256": ledger_sha256,
            "lane_registry_registered": lane_registry_registered,
            "canonical_replacement_lane_id": "",
            "canonical_replacement_lane_registered": False,
            "expected_first_artifact_status": expected_first_artifact_status,
            "expected_first_artifacts_all_present": (
                expected_first_artifacts_all_present
            ),
            "next_prerequisite_status": next_prerequisite_status,
            "ready_for_l1_build_semantics": ready_for_l1_build_semantics,
            "post_l1_probe_evidence": post_l1_probe_evidence,
            "post_l1_proxy_evidence": z6_post_l1_proxy_evidence,
        }
        aggregate_blockers.extend(blockers)
        rows.append(
            {
                **candidate.__dict__,
                "cost_band_usd": list(candidate.cost_band_usd),
                "expected_first_artifacts": list(candidate.expected_first_artifacts),
                "expected_first_artifact_status": expected_first_artifact_status,
                "expected_first_artifacts_all_present": (
                    expected_first_artifacts_all_present
                ),
                "dependency_blockers": list(candidate.dependency_blockers),
                "local_ledger_present": ledger_present,
                "local_ledger_sha256": ledger_sha256,
                "lane_registry_registered": lane_registry_registered,
                "blockers": blockers,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "ready_for_paid_dispatch": False,
                "ready_for_recommended_next_action": ready_for_recommended_next_action,
                "recommended_next_action_status": recommended_next_action_status,
                "effective_recommended_next_action_id": effective_next_action_id,
                "effective_recommended_next_action": effective_next_action,
                "ready_for_l1_build": ready_for_l1_build,
                "ready_for_l1_build_semantics": ready_for_l1_build_semantics,
                "l1_scaffold_present": l1_scaffold_present,
                "recommended_next_action_completed_or_superseded": (
                    l1_scaffold_present
                ),
                "ready_for_l1_scaffold_dispatch": False,
                "l1_build_blockers": l1_build_blockers,
                "l5_v2_asymptotic_next_action_status": (
                    l5_v2_asymptotic_next_action_status
                ),
                "post_l1_proxy_evidence": z6_post_l1_proxy_evidence,
                "post_l1_probe_evidence": post_l1_probe_evidence,
                "post_l1_recommended_next_action_status": (
                    post_l1_recommended_next_action_status
                ),
                "post_l1_recommended_next_action_id": (
                    post_l1_recommended_next_action_id
                ),
                "post_l1_recommended_next_action": post_l1_recommended_next_action,
            }
        )
    return {
        "schema": L5_V2_ASYMPTOTIC_PURSUIT_CANDIDATES_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "subject_id": SUBJECT_ID,
        "candidate_count": len(rows),
        "candidate_ids": [str(row["candidate_id"]) for row in rows],
        "candidates": rows,
        "l5_v2_asymptotic_next_action_status": [
            row["l5_v2_asymptotic_next_action_status"] for row in rows
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "blockers": list(dict.fromkeys(aggregate_blockers)),
    }


def asymptotic_candidate_surface_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON text for the L5 v2 asymptotic surface."""

    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


def render_l5_v2_asymptotic_candidate_surface_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing L5 v2 asymptotic surface report."""

    lines = [
        "# L5 v2 asymptotic candidate surface",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- campaign_id: `{payload.get('campaign_id')}`",
        f"- candidate_count: `{payload.get('candidate_count')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- rank_or_kill_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- ready_for_paid_dispatch: `false`",
        "",
        "This is a planning and no-signal-loss surface. It records whether "
        "asymptotic L5-v2 candidates have their first executable artifacts, "
        "but it does not authorize score claims, rank changes, paid dispatch, "
        "or promotion.",
        "",
        "## Candidates",
    ]
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for row in candidates:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {row.get('candidate_id')}",
                    "",
                    f"- title: {row.get('title')}",
                    f"- lane_id: `{row.get('lane_id')}`",
                    f"- local_ledger_present: `{row.get('local_ledger_present')}`",
                    f"- lane_registry_registered: `{row.get('lane_registry_registered')}`",
                    f"- expected_first_artifacts_all_present: `{row.get('expected_first_artifacts_all_present')}`",
                    f"- l1_scaffold_present: `{row.get('l1_scaffold_present')}`",
                    f"- recommended_next_action_status: `{row.get('recommended_next_action_status')}`",
                    f"- effective_recommended_next_action_id: `{row.get('effective_recommended_next_action_id')}`",
                    f"- ready_for_l1_build: `{row.get('ready_for_l1_build')}`",
                    f"- ready_for_l1_scaffold_dispatch: `{row.get('ready_for_l1_scaffold_dispatch')}`",
                    f"- post_l1_recommended_next_action_status: `{row.get('post_l1_recommended_next_action_status')}`",
                    f"- post_l1_recommended_next_action_id: `{row.get('post_l1_recommended_next_action_id')}`",
                    f"- blockers: `{row.get('blockers')}`",
                    f"- l1_build_blockers: `{row.get('l1_build_blockers')}`",
                    "",
                    "Expected first artifacts:",
                ]
            )
            post_l1_evidence = row.get("post_l1_probe_evidence")
            if not isinstance(post_l1_evidence, Mapping):
                post_l1_evidence = row.get("post_l1_proxy_evidence")
            if isinstance(post_l1_evidence, Mapping):
                lines.extend(
                    [
                        "",
                        "Post-L1 evidence:",
                        f"- artifact_present=`{post_l1_evidence.get('artifact_present')}`",
                        f"- artifact_valid=`{post_l1_evidence.get('artifact_valid')}`",
                        f"- verdict=`{post_l1_evidence.get('verdict')}`",
                        f"- allowed_to_spend=`{post_l1_evidence.get('allowed_to_spend')}`",
                        f"- measured_summary=`{post_l1_evidence.get('measured_summary')}`",
                    ]
                )
            artifacts = row.get("expected_first_artifact_status")
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    if not isinstance(artifact, Mapping):
                        continue
                    lines.append(
                        f"- `{artifact.get('path')}` present=`{artifact.get('present')}`"
                    )
    return "\n".join(lines) + "\n"


def _z6_post_l1_proxy_evidence_status(*, repo_root: Path) -> dict[str, Any]:
    """Return fail-closed Z6 post-L1 proxy evidence for dispatch routing."""

    artifact_path = repo_root / Z6_REAL_VIDEO_EGO_PROXY_SWEEP_ARTIFACT_PATH
    scorer_artifact_path = (
        repo_root / Z6_SCORER_BEARING_PAIRED_SMOKE_ARTIFACT_PATH
    )
    blockers: list[str] = []
    artifact_present = artifact_path.is_file()
    artifact_sha256 = ""
    payload: Mapping[str, Any] = {}
    scorer_artifact_present = scorer_artifact_path.is_file()
    scorer_artifact_sha256 = ""
    scorer_payload: Mapping[str, Any] = {}
    if not artifact_present:
        blockers.append("z6_real_video_ego_proxy_sweep_missing")
    else:
        artifact_sha256 = _sha256_file(artifact_path)
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append("z6_real_video_ego_proxy_sweep_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
            else:
                blockers.append("z6_real_video_ego_proxy_sweep_not_object")
    if scorer_artifact_present:
        scorer_artifact_sha256 = _sha256_file(scorer_artifact_path)
        try:
            loaded_scorer = json.loads(
                scorer_artifact_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            blockers.append("z6_scorer_bearing_paired_smoke_json_invalid")
        else:
            if isinstance(loaded_scorer, Mapping):
                scorer_payload = loaded_scorer
            else:
                blockers.append("z6_scorer_bearing_paired_smoke_not_object")

    for field_name in (
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "ready_for_paid_dispatch",
        "paradigm_claim_allowed",
    ):
        if payload.get(field_name) is not False:
            blockers.append(f"z6_real_video_ego_proxy_sweep_{field_name}_not_false")
    if payload and payload.get("schema") != Z6_REAL_VIDEO_EGO_PROXY_SWEEP_SCHEMA:
        blockers.append("z6_real_video_ego_proxy_sweep_schema_mismatch")
    if payload and payload.get("evidence_grade") != "real_video_smoke_proxy_no_scorer":
        blockers.append("z6_real_video_ego_proxy_sweep_evidence_grade_mismatch")
    if payload and not isinstance(payload.get("rows"), list):
        blockers.append("z6_real_video_ego_proxy_sweep_rows_missing")
    for field_name in (
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "ready_for_paid_dispatch",
        "paradigm_claim_allowed",
    ):
        if scorer_payload and scorer_payload.get(field_name) is not False:
            blockers.append(
                f"z6_scorer_bearing_paired_smoke_{field_name}_not_false"
            )
    if (
        scorer_payload
        and scorer_payload.get("schema") != Z6_SCORER_BEARING_PAIRED_SMOKE_SCHEMA
    ):
        blockers.append("z6_scorer_bearing_paired_smoke_schema_mismatch")
    if (
        scorer_payload
        and scorer_payload.get("evidence_grade")
        != "tiny_cpu_scorer_bearing_proxy_no_archive_eval"
    ):
        blockers.append("z6_scorer_bearing_paired_smoke_evidence_grade_mismatch")
    if scorer_payload and not isinstance(scorer_payload.get("rows"), list):
        blockers.append("z6_scorer_bearing_paired_smoke_rows_missing")

    verdict = str(payload.get("verdict") or "")
    identity_dominates = (
        verdict == Z6_REAL_VIDEO_EGO_PROXY_SWEEP_IDENTITY_DOMINATES_VERDICT
    )
    full_film_proxy_found = (
        verdict == Z6_REAL_VIDEO_EGO_PROXY_SWEEP_FULL_FILM_VERDICT
    )
    semantic_ego_proxy_supported = payload.get("semantic_ego_proxy_supported") is True
    posenet_proxy_tested = payload.get("posenet_proxy_tested") is True
    scorer_semantic_supported = (
        scorer_payload.get("semantic_scorer_proxy_supported") is True
    )
    if scorer_payload and not scorer_semantic_supported:
        blockers.append(
            "z6_full_film_paid_dispatch_blocked_scorer_bearing_semantics_not_hard_earned"
        )
        if scorer_payload.get("best_proxy_id") != "posenet_pose":
            blockers.append(
                "z6_full_film_paid_dispatch_blocked_posenet_pose_scorer_proxy_not_best"
            )
    if identity_dominates:
        blockers.append(
            "z6_full_film_paid_dispatch_blocked_identity_dominates_real_video_proxy_sweep"
        )
        recommended_next_action_id = (
            "advance_z6_only_with_posenet_or_scorer_ego_proxy_or_skip_to_z7"
        )
        recommended_next_action = (
            "Do not spend on Z6-v1 full-FiLM. Either build a PoseNet/scorer-"
            "derived ego proxy that beats identity in smoke, redesign the "
            "predictor objective, or advance the L5-v2 staircase to Z7/Rudin/"
            "Tishby rather than retreading this configuration."
        )
        recommended_next_action_status = "blocked_pending_redesign_or_next_candidate"
    elif payload and full_film_proxy_found and not semantic_ego_proxy_supported:
        blockers.append(
            "z6_full_film_paid_dispatch_blocked_ego_proxy_semantics_not_hard_earned"
        )
        if posenet_proxy_tested and payload.get("best_proxy_id") != "posenet_pose":
            blockers.append(
                "z6_full_film_paid_dispatch_blocked_posenet_pose_proxy_not_best"
            )
        recommended_next_action_id = "z6_proxy_capacity_found_require_semantic_ego_probe"
        recommended_next_action = (
            "Full-FiLM beat identity under matched shared initialization, but "
            "the best proxy did not prove ego-motion semantics. Run a "
            "PoseNet/scorer-derived ego proxy probe before paid dispatch."
        )
        recommended_next_action_status = (
            "proxy_capacity_found_requires_semantic_ego_probe"
        )
        if scorer_payload and not scorer_semantic_supported:
            recommended_next_action_id = (
                "z6_scorer_bearing_probe_blocks_full_film_paid_dispatch"
            )
            recommended_next_action = (
                "The tiny scorer-bearing paired probe found random_control, "
                "not PoseNet-derived ego, as the best proxy. Do not paid-"
                "dispatch Z6-v1 full-FiLM; redesign ego-conditioning or "
                "advance Z7/Z8 as new measured configurations."
            )
            recommended_next_action_status = (
                "scorer_bearing_probe_blocks_z6_v1_paid_dispatch"
            )
    elif payload and not blockers:
        recommended_next_action_id = "z6_proxy_sweep_found_full_film_candidate"
        recommended_next_action = (
            "A full-FiLM proxy candidate beat identity in smoke; run the next "
            "paired scorer-bearing probe before any score or paradigm claim."
        )
        recommended_next_action_status = "proxy_candidate_found_requires_scorer_probe"
    else:
        recommended_next_action_id = "run_z6_real_video_ego_proxy_sweep"
        recommended_next_action = (
            "Run the Z6 real-video ego-proxy sweep before any paid full-FiLM "
            "dispatch."
        )
        recommended_next_action_status = "missing_or_invalid_proxy_evidence"

    return {
        "schema": Z6_POST_L1_PROXY_EVIDENCE_STATUS_SCHEMA,
        "tool_path": Z6_REAL_VIDEO_EGO_PROXY_SWEEP_TOOL_PATH,
        "artifact_path": Z6_REAL_VIDEO_EGO_PROXY_SWEEP_ARTIFACT_PATH,
        "artifact_present": artifact_present,
        "artifact_sha256": artifact_sha256,
        "artifact_valid": bool(payload) and not [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "z6_full_film_paid_dispatch_blocked_identity_dominates_real_video_proxy_sweep",
                "z6_full_film_paid_dispatch_blocked_ego_proxy_semantics_not_hard_earned",
                "z6_full_film_paid_dispatch_blocked_posenet_pose_proxy_not_best",
                "z6_full_film_paid_dispatch_blocked_scorer_bearing_semantics_not_hard_earned",
                "z6_full_film_paid_dispatch_blocked_posenet_pose_scorer_proxy_not_best",
            }
        ],
        "verdict": verdict,
        "best_proxy_id": payload.get("best_proxy_id"),
        "posenet_proxy_tested": payload.get("posenet_proxy_tested"),
        "semantic_ego_proxy_supported": payload.get("semantic_ego_proxy_supported"),
        "paired_control_initialization": payload.get("paired_control_initialization"),
        "best_identity_minus_full_loss_proxy": payload.get(
            "best_identity_minus_full_loss_proxy"
        ),
        "scorer_bearing_paired_smoke": {
            "tool_path": Z6_SCORER_BEARING_PAIRED_SMOKE_TOOL_PATH,
            "artifact_path": Z6_SCORER_BEARING_PAIRED_SMOKE_ARTIFACT_PATH,
            "artifact_present": scorer_artifact_present,
            "artifact_sha256": scorer_artifact_sha256,
            "schema": scorer_payload.get("schema"),
            "verdict": scorer_payload.get("verdict"),
            "evidence_grade": scorer_payload.get("evidence_grade"),
            "hardware_axis": scorer_payload.get("hardware_axis"),
            "best_proxy_id": scorer_payload.get("best_proxy_id"),
            "best_identity_minus_full_score_proxy": scorer_payload.get(
                "best_identity_minus_full_score_proxy"
            ),
            "semantic_scorer_proxy_supported": scorer_payload.get(
                "semantic_scorer_proxy_supported"
            ),
            "score_claim": scorer_payload.get("score_claim"),
            "promotion_eligible": scorer_payload.get("promotion_eligible"),
            "ready_for_paid_dispatch": scorer_payload.get("ready_for_paid_dispatch"),
        },
        "identity_dominates_all_tested_real_video_proxies": identity_dominates,
        "full_film_proxy_found": full_film_proxy_found,
        "allowed_to_spend": False,
        "allowed_to_spend_on_z6_full_film": False,
        "measured_summary": (
            f"best_proxy={payload.get('best_proxy_id')} "
            f"identity_minus_full_loss_proxy="
            f"{payload.get('best_identity_minus_full_loss_proxy')} "
            f"posenet_proxy_tested={payload.get('posenet_proxy_tested')} "
            f"semantic_ego_proxy_supported={payload.get('semantic_ego_proxy_supported')} "
            f"scorer_bearing_best_proxy={scorer_payload.get('best_proxy_id')} "
            f"scorer_bearing_semantic_supported="
            f"{scorer_payload.get('semantic_scorer_proxy_supported')}"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "paradigm_claim_allowed": False,
        "recommended_next_action_status": recommended_next_action_status,
        "recommended_next_action_id": recommended_next_action_id,
        "recommended_next_action": recommended_next_action,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _tishby_post_l1_probe_evidence_status(*, repo_root: Path) -> dict[str, Any]:
    """Return fail-closed Tishby post-L1 probe evidence for dispatch routing."""

    d4_path = repo_root / TISHBY_D4_PROBE_ARTIFACT_PATH
    vib_path = repo_root / TISHBY_VIB_TRACTABILITY_ARTIFACT_PATH
    blockers: list[str] = []
    d4_payload = _load_probe_artifact_mapping(
        d4_path,
        blockers=blockers,
        blocker_prefix="tishby_d4_probe",
    )
    vib_payload = _load_probe_artifact_mapping(
        vib_path,
        blockers=blockers,
        blocker_prefix="tishby_vib_tractability",
    )
    for prefix, payload in (
        ("tishby_d4_probe", d4_payload),
        ("tishby_vib_tractability", vib_payload),
    ):
        if payload and payload.get("substrate_id") != "tishby_ib_pure":
            blockers.append(f"{prefix}_substrate_id_mismatch")
        if payload and payload.get("score_claim") is not False:
            blockers.append(f"{prefix}_score_claim_not_false")
        if payload and payload.get("evidence_grade") != "diagnostic_cpu":
            blockers.append(f"{prefix}_evidence_grade_mismatch")

    d4_verdict = str(d4_payload.get("verdict") or "")
    vib_verdict = str(vib_payload.get("verdict") or "")
    d4_independent = d4_verdict == TISHBY_D4_INDEPENDENT_VERDICT
    vib_tractable = vib_verdict == TISHBY_VIB_TRACTABLE_VERDICT
    if d4_payload and not d4_independent:
        blockers.append("tishby_d4_probe_not_independent_unreviewed_path")
    if vib_payload and not vib_tractable:
        blockers.append("tishby_vib_tractability_not_tractable")
    if d4_independent:
        blockers.append(
            "tishby_path_vib_paid_dispatch_blocked_d4_independent_scorer_class_probe"
        )
        recommended_next_action_id = (
            "advance_tishby_only_with_meaningful_conditioning_or_mine_beta_sweep"
        )
        recommended_next_action = (
            "Do not spend on measured Path-VIB side-info until scorer-class "
            "conditioning is meaningful, Path-MINE is implemented, or a "
            "beta-sweep is justified by a new probe. Prefer the next L5-v2 "
            "candidate if no new Tishby conditioning signal exists."
        )
        recommended_next_action_status = "blocked_pending_conditioning_or_mine"
    elif d4_payload and vib_tractable and not blockers:
        recommended_next_action_id = "run_tishby_scorer_bearing_beta_sweep"
        recommended_next_action = (
            "D4 and VIB gates passed; run scorer-bearing beta sweep before "
            "any score or paradigm claim."
        )
        recommended_next_action_status = "probe_candidate_found_requires_beta_sweep"
    else:
        recommended_next_action_id = "run_tishby_d4_and_vib_probe_bundle"
        recommended_next_action = (
            "Run and preserve the Tishby D4 plus VIB tractability probe bundle "
            "before paid dispatch."
        )
        recommended_next_action_status = "missing_or_invalid_probe_evidence"

    d4_mi = d4_payload.get("mutual_information_bits")
    vib_snr = vib_payload.get("gradient_snr_worst_case")
    return {
        "schema": TISHBY_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA,
        "d4_probe_artifact_path": TISHBY_D4_PROBE_ARTIFACT_PATH,
        "d4_probe_artifact_present": d4_path.is_file(),
        "d4_probe_artifact_sha256": _sha256_file(d4_path) if d4_path.is_file() else "",
        "vib_tractability_artifact_path": TISHBY_VIB_TRACTABILITY_ARTIFACT_PATH,
        "vib_tractability_artifact_present": vib_path.is_file(),
        "vib_tractability_artifact_sha256": (
            _sha256_file(vib_path) if vib_path.is_file() else ""
        ),
        "artifact_present": d4_path.is_file() and vib_path.is_file(),
        "artifact_valid": bool(d4_payload)
        and bool(vib_payload)
        and not [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "tishby_path_vib_paid_dispatch_blocked_d4_independent_scorer_class_probe"
            }
        ],
        "verdict": f"d4={d4_verdict};vib={vib_verdict}",
        "d4_verdict": d4_verdict,
        "vib_tractability_verdict": vib_verdict,
        "mutual_information_bits": d4_mi,
        "wyner_ziv_gain_ceiling_fraction": d4_payload.get(
            "wyner_ziv_gain_ceiling_fraction"
        ),
        "gradient_snr_worst_case": vib_snr,
        "d4_independent": d4_independent,
        "vib_tractable": vib_tractable,
        "allowed_to_spend": False,
        "allowed_to_spend_on_tishby_path_vib": False,
        "measured_summary": f"d4_mi_bits={d4_mi};vib_worst_snr={vib_snr}",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "paradigm_claim_allowed": False,
        "recommended_next_action_status": recommended_next_action_status,
        "recommended_next_action_id": recommended_next_action_id,
        "recommended_next_action": recommended_next_action,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _rudin_post_l1_probe_evidence_status(*, repo_root: Path) -> dict[str, Any]:
    """Return fail-closed Rudin post-L1 probe evidence for dispatch routing."""

    artifact_path = repo_root / RUDIN_PROXY_DISAMBIGUATOR_ARTIFACT_PATH
    blockers: list[str] = []
    payload = _load_probe_artifact_mapping(
        artifact_path,
        blockers=blockers,
        blocker_prefix="rudin_proxy_disambiguator",
    )
    if payload and payload.get("substrate_id") != "rudin_floor_interpretable_ml":
        blockers.append("rudin_proxy_disambiguator_substrate_id_mismatch")
    if payload and payload.get("score_claim") is not False:
        blockers.append("rudin_proxy_disambiguator_score_claim_not_false")
    if payload and payload.get("promotion_eligible") is not False:
        blockers.append("rudin_proxy_disambiguator_promotion_eligible_not_false")
    if payload and payload.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("rudin_proxy_disambiguator_exact_dispatch_not_false")
    if payload and payload.get("score_axis") != "design_time_disambiguator_proxy":
        blockers.append("rudin_proxy_disambiguator_score_axis_mismatch")

    verdict = str(payload.get("verdict") or "")
    meaningful = verdict == RUDIN_MEANINGFUL_INTERPRETABILITY_VERDICT
    if meaningful:
        blockers.append(
            "rudin_proxy_positive_requires_t3_ratification_and_scorer_probe"
        )
        recommended_next_action_id = (
            "run_rudin_t3_ratification_scorer_probe_before_paid_dispatch"
        )
        recommended_next_action = (
            "Rudin proxy is meaningful, but it is still a design-time proxy. "
            "Run T3 ratification plus a scorer-bearing probe before paid "
            "dispatch or any score/paradigm claim."
        )
        recommended_next_action_status = "proxy_positive_requires_ratification"
    elif payload and not blockers:
        blockers.append("rudin_proxy_not_meaningful_research_redesign_required")
        recommended_next_action_id = "redesign_rudin_rule_basis_before_dispatch"
        recommended_next_action = (
            "Rudin proxy did not validate the interpretability premise; redesign "
            "the rule basis before paid dispatch."
        )
        recommended_next_action_status = "blocked_pending_redesign"
    else:
        recommended_next_action_id = "run_rudin_proxy_disambiguator"
        recommended_next_action = (
            "Run and preserve the Rudin design-time proxy disambiguator before "
            "paid dispatch."
        )
        recommended_next_action_status = "missing_or_invalid_probe_evidence"

    return {
        "schema": RUDIN_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA,
        "tool_path": RUDIN_PROXY_DISAMBIGUATOR_TOOL_PATH,
        "artifact_path": RUDIN_PROXY_DISAMBIGUATOR_ARTIFACT_PATH,
        "artifact_present": artifact_path.is_file(),
        "artifact_sha256": (
            _sha256_file(artifact_path) if artifact_path.is_file() else ""
        ),
        "artifact_valid": bool(payload)
        and not [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "rudin_proxy_positive_requires_t3_ratification_and_scorer_probe"
            }
        ],
        "verdict": verdict,
        "interpretability_tax_estimate": payload.get("interpretability_tax_estimate"),
        "total_pixels": payload.get("total_pixels"),
        "n_frames": payload.get("n_frames"),
        "fallback_used": payload.get("fallback_used"),
        "meaningful_interpretability_proxy": meaningful,
        "allowed_to_spend": False,
        "allowed_to_spend_on_rudin_floor": False,
        "measured_summary": (
            f"tax={payload.get('interpretability_tax_estimate')};"
            f"pixels={payload.get('total_pixels')};frames={payload.get('n_frames')}"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "paradigm_claim_allowed": False,
        "recommended_next_action_status": recommended_next_action_status,
        "recommended_next_action_id": recommended_next_action_id,
        "recommended_next_action": recommended_next_action,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _load_probe_artifact_mapping(
    path: Path,
    *,
    blockers: list[str],
    blocker_prefix: str,
) -> Mapping[str, Any]:
    """Load a small probe artifact mapping and append fail-closed blockers."""

    if not path.is_file():
        blockers.append(f"{blocker_prefix}_missing")
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        blockers.append(f"{blocker_prefix}_json_invalid")
        return {}
    if not isinstance(loaded, Mapping):
        blockers.append(f"{blocker_prefix}_not_object")
        return {}
    return loaded


def _l5_v2_lane_registry_ids(*, repo_root: Path) -> tuple[set[str], list[str]]:
    """Return lane ids from the canonical registry without raising on drift."""

    registry_path = repo_root / ".omx/state/lane_registry.json"
    if not registry_path.is_file():
        return set(), ["l5_v2_lane_registry_missing"]
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set(), ["l5_v2_lane_registry_json_invalid"]
    if not isinstance(registry, Mapping):
        return set(), ["l5_v2_lane_registry_not_object"]
    lanes = registry.get("lanes")
    if not isinstance(lanes, list):
        return set(), ["l5_v2_lane_registry_lanes_not_list"]
    lane_ids = {
        str(lane.get("id"))
        for lane in lanes
        if isinstance(lane, Mapping) and str(lane.get("id") or "")
    }
    return lane_ids, []


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
            step_id="l5v2_01_dykstra_score_axis_sanity",
            title="Dykstra score-axis sanity",
            objective=(
                "Project the retired additive TT5L score band through the "
                "contest formula with declared design-move constraint ids before "
                "side-info proof or timing work. This is not a move-level "
                "feasibility proof."
            ),
            deliverable_surface=(
                f"{TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH} + "
                f"{TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH}"
            ),
            required_gate_ids=(),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_02_sideinfo_consumption_proof",
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
            step_id="l5v2_03_probe_disambiguator",
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
            step_id="l5v2_04_paired_axis_anchor",
            title="Paired CPU/CUDA anchor",
            objective=(
                "Measure the exact same byte-closed packet on contest-compliant "
                "CPU and CUDA axes with a timing-smoke custody artifact before "
                "any promotion or submission discussion."
            ),
            deliverable_surface=(
                "experiments/results/time_traveler_l5_v2/ + "
                f"{TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH} + "
                f"{TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH}"
            ),
            required_gate_ids=(
                "paired_cpu_cuda_axis_plan",
                "exact_anchor_or_diagnostic_pair",
            ),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_05_stack_of_stacks_candidate",
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


def _l5_v2_prediction_band_diagnostic_anchor_pair_status(
    *,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_path = repo_root / TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        return {
            "schema": "l5_v2_prediction_band_diagnostic_anchor_pair_status_v1",
            "artifact_path": TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH,
            "artifact_exists": False,
            "artifact_valid": False,
            "artifact_sha256": "",
            "classification": "",
            "paired_axes": [],
            "per_axis_scores": {},
            "archive_sha256": "",
            "runtime_content_tree_sha256": "",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "diagnostic_only": True,
            "blockers": [],
        }
    try:
        loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append("l5_v2_prediction_band_diagnostic_anchor_pair_json_invalid")
    else:
        if isinstance(loaded, Mapping):
            payload = loaded
        else:
            blockers.append("l5_v2_prediction_band_diagnostic_anchor_pair_not_object")

    if payload:
        if payload.get("schema") != "l5_v2_tt5l_paired_exact_anchor_pair_v1":
            blockers.append("l5_v2_prediction_band_diagnostic_anchor_pair_schema_mismatch")
        if payload.get("gate_id") != "exact_anchor_or_diagnostic_pair":
            blockers.append("l5_v2_prediction_band_diagnostic_anchor_pair_gate_mismatch")
        if payload.get("predicate_id") != TT5L_PAIRED_EXACT_ANCHOR_PAIR_PREDICATE_ID:
            blockers.append(
                "l5_v2_prediction_band_diagnostic_anchor_pair_predicate_mismatch"
            )
        if payload.get("score_claim") is not False:
            blockers.append("l5_v2_prediction_band_diagnostic_anchor_pair_score_claim")
        if payload.get("promotion_eligible") is not False:
            blockers.append("l5_v2_prediction_band_diagnostic_anchor_pair_promotional")
        if payload.get("rank_or_kill_eligible") is not False:
            blockers.append("l5_v2_prediction_band_diagnostic_anchor_pair_rankable")
        blockers.extend(
            _gate_semantic_blockers(
                "exact_anchor_or_diagnostic_pair",
                payload,
                repo_root=repo_root,
            )
        )

    rows = payload.get("anchor_pair") if payload else []
    axis_rows = [row for row in rows if isinstance(row, Mapping)]
    per_axis_scores = {
        str(row.get("axis") or ""): row.get("score")
        for row in axis_rows
        if str(row.get("axis") or "")
    }
    paired_axes = sorted(per_axis_scores)
    archive_sha256 = str(payload.get("archive_sha256") or "").strip() if payload else ""
    runtime_content_tree_sha256 = (
        str(payload.get("runtime_content_tree_sha256") or "").strip()
        if payload
        else ""
    )
    if payload and not _SHA256_HEX_RE.fullmatch(archive_sha256):
        blockers.append("l5_v2_prediction_band_diagnostic_anchor_archive_sha_invalid")
    if payload and not _SHA256_HEX_RE.fullmatch(runtime_content_tree_sha256):
        blockers.append(
            "l5_v2_prediction_band_diagnostic_anchor_runtime_content_sha_invalid"
        )

    return {
        "schema": "l5_v2_prediction_band_diagnostic_anchor_pair_status_v1",
        "artifact_path": TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": bool(payload) and not blockers,
        "artifact_sha256": _sha256_file(artifact_path) if artifact_path.is_file() else "",
        "classification": str(payload.get("classification") or "") if payload else "",
        "paired_axes": paired_axes,
        "per_axis_scores": per_axis_scores,
        "archive_sha256": archive_sha256,
        "runtime_content_tree_sha256": runtime_content_tree_sha256,
        "score_claim": payload.get("score_claim") is True if payload else False,
        "promotion_eligible": (
            payload.get("promotion_eligible") is True if payload else False
        ),
        "rank_or_kill_eligible": (
            payload.get("rank_or_kill_eligible") is True if payload else False
        ),
        "diagnostic_only": True,
        "blockers": list(dict.fromkeys(blockers)),
    }


def l5_v2_prediction_band_status(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the rank-blocked prediction band plus preserved diagnostic anchors."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    payload = l5_v2_prediction_band_payload()
    verdict = l5_v2_prediction_band_verdict(repo_root=resolved_repo_root)
    diagnostic_anchor_pair_status = (
        _l5_v2_prediction_band_diagnostic_anchor_pair_status(
            repo_root=resolved_repo_root,
        )
    )
    baseline = payload["baseline"]
    empirical_anchor = payload["empirical_anchor"]
    blockers = [
        str(blocker) for blocker in verdict.get("blockers", []) if str(blocker)
    ]
    return {
        "schema": "l5_v2_prediction_band_status_v1",
        "payload": payload,
        "verdict": verdict,
        "rank_reward_allowed": verdict.get("valid_for_rank_reward") is True,
        "dispatch_planning_allowed": (
            verdict.get("valid_for_dispatch_planning") is True
        ),
        "promotion_eligible": verdict.get("valid_for_promotion") is True,
        "baseline_status": {
            "label": baseline["label"],
            "axis": baseline["axis"],
            "score": baseline["score"],
            "archive_sha256": baseline["archive_sha256"],
            "runtime_tree_sha256": baseline["runtime_tree_sha256"],
            "artifact_path": baseline["artifact_path"],
            "complete": "prediction_band_baseline_missing" not in blockers,
            "custody_complete": (
                "prediction_band_baseline_custody_missing" not in blockers
                and "prediction_band_baseline_artifact_missing" not in blockers
            ),
        },
        "empirical_anchor_status": {
            "status": empirical_anchor["status"],
            "anchor_count": len(empirical_anchor["anchors"]),
            "rankable_anchor_missing": (
                "prediction_band_empirical_anchor_missing" in blockers
            ),
        },
        "diagnostic_anchor_pair_status": diagnostic_anchor_pair_status,
        "diagnostic_anchor_preserved_but_not_rankable": (
            diagnostic_anchor_pair_status["artifact_valid"] is True
            and verdict.get("valid_for_rank_reward") is not True
        ),
        "blockers": blockers,
        "annotations": [
            str(annotation)
            for annotation in verdict.get("annotations", [])
            if str(annotation)
        ],
    }


def l5_v2_prediction_band_verdict(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the current validation verdict for the L5 v2 prediction band."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
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
            artifact_base_dir=resolved_repo_root,
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
        def _mapping_values() -> Iterable[L5V2GateEvidence | Mapping[str, Any]]:
            for key, value in gate_evidence.items():
                if isinstance(value, L5V2GateEvidence):
                    yield value
                elif isinstance(value, Mapping):
                    yield {"gate_id": key, **value}
                else:
                    yield {
                        "gate_id": key,
                        "artifact_path": "",
                        "artifact_sha256": "",
                        "predicate_id": "",
                        "predicate_passed": False,
                        "evidence_grade": "__non_object_gate_evidence__",
                    }

        values = _mapping_values()
    else:
        values = gate_evidence

    coerced: dict[str, L5V2GateEvidence] = {}
    for value in values:
        if isinstance(value, L5V2GateEvidence):
            evidence = value
        elif not isinstance(value, Mapping):
            evidence = L5V2GateEvidence(
                gate_id="",
                artifact_path="",
                artifact_sha256="",
                predicate_id="",
                predicate_passed=False,
                evidence_grade="__non_object_gate_evidence__",
            )
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
            TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_PATH,
            TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_SHA256,
            "contest_full_frame_inflate_consumption_proof_committed_custody",
        ),
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


def l5_v2_canonical_probe_gate_evidence(
    *,
    repo_root: str | Path | None = None,
) -> L5V2GateEvidence | None:
    """Return the discoverable L5 v2 probe-disambiguator gate artifact."""

    status = l5_v2_probe_gate_artifact_status(repo_root=repo_root)
    if status["artifact_valid"] is not True:
        return None
    return L5V2GateEvidence(
        gate_id="c1_z5_tt5l_probe_disambiguator",
        artifact_path=str(status["artifact_path"]),
        artifact_sha256=str(status["artifact_sha256"]),
        predicate_id="l5_v2_probe_disambiguator_architecture_lock_v1",
        predicate_passed=True,
        evidence_grade="l5_v2_probe_gate_artifact",
    )


def l5_v2_probe_gate_artifact_status(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return visible status for the probe gate, including invalid blockers."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    artifact_path = TT5L_PROBE_GATE_ARTIFACT_PATH
    proof_path = resolved_repo_root / artifact_path
    if not proof_path.is_file():
        return {
            "schema": "l5_v2_probe_gate_artifact_status_v1",
            "artifact_path": artifact_path,
            "artifact_exists": False,
            "artifact_valid": False,
            "artifact_sha256": "",
            "architecture_lock_allowed": False,
            "selected_candidate_id": None,
            "verdict_blockers": [],
            "candidate_status": [],
            "blockers": [],
        }
    artifact_sha = _sha256_file(proof_path)
    validation_blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    try:
        loaded = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        validation_blockers.append("l5_v2_probe_gate_artifact_json_invalid")
    else:
        if isinstance(loaded, Mapping):
            payload = loaded
        else:
            validation_blockers.append("l5_v2_probe_gate_artifact_not_object")
    semantic_blockers = (
        _gate_semantic_blockers(
            "c1_z5_tt5l_probe_disambiguator",
            payload,
            repo_root=resolved_repo_root,
        )
        if payload
        else []
    )
    probe = payload.get("probe_disambiguator") if payload else None
    verdict = probe.get("verdict") if isinstance(probe, Mapping) else None
    verdict_map = verdict if isinstance(verdict, Mapping) else {}
    candidate_status: list[dict[str, Any]] = []
    for row in verdict_map.get("evaluated_observations") or []:
        if not isinstance(row, Mapping):
            continue
        candidate_status.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "eligible_for_architecture_lock": (
                    row.get("eligible_for_architecture_lock") is True
                ),
                "exact_axes": [
                    str(axis)
                    for axis in row.get("exact_axes", [])
                    if str(axis).strip()
                ]
                if isinstance(row.get("exact_axes"), list)
                else [],
                "blockers": [
                    str(blocker)
                    for blocker in row.get("blockers", [])
                    if str(blocker).strip()
                ]
                if isinstance(row.get("blockers"), list)
                else [],
            }
        )
    return {
        "schema": "l5_v2_probe_gate_artifact_status_v1",
        "artifact_path": artifact_path,
        "artifact_exists": proof_path.is_file(),
        "artifact_valid": bool(payload) and not validation_blockers and not semantic_blockers,
        "artifact_sha256": artifact_sha,
        "architecture_lock_allowed": (
            verdict_map.get("architecture_lock_allowed") is True
        ),
        "selected_candidate_id": verdict_map.get("selected_candidate_id"),
        "verdict_blockers": [
            str(blocker)
            for blocker in verdict_map.get("blockers", [])
            if str(blocker).strip()
        ]
        if isinstance(verdict_map.get("blockers"), list)
        else [],
        "candidate_status": candidate_status,
        "blockers": list(dict.fromkeys(validation_blockers + semantic_blockers)),
    }


def l5_v2_canonical_paired_axis_plan_gate_evidence(
    *,
    repo_root: str | Path | None = None,
) -> L5V2GateEvidence | None:
    """Return the discoverable non-promotional paired CPU/CUDA plan artifact."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    artifact_path = TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_ARTIFACT_PATH
    proof_path = resolved_repo_root / artifact_path
    if not proof_path.is_file():
        return None
    artifact_sha = _sha256_file(proof_path)
    try:
        payload = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if _gate_semantic_blockers(
        "paired_cpu_cuda_axis_plan",
        payload,
        repo_root=resolved_repo_root,
    ):
        return None
    predicate_id = str(
        payload.get("predicate_id") or TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_PREDICATE_ID
    ).strip()
    return L5V2GateEvidence(
        gate_id="paired_cpu_cuda_axis_plan",
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha,
        predicate_id=predicate_id,
        predicate_passed=True,
        evidence_grade="tt5l_paired_cpu_cuda_axis_plan_from_anchor_artifact",
    )


def l5_v2_canonical_anchor_pair_gate_evidence(
    *,
    repo_root: str | Path | None = None,
) -> L5V2GateEvidence | None:
    """Return the discoverable paired CPU/CUDA TT5L anchor artifact."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    artifact_path = TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH
    proof_path = resolved_repo_root / artifact_path
    if not proof_path.is_file():
        return None
    artifact_sha = _sha256_file(proof_path)
    try:
        payload = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if _gate_semantic_blockers(
        "exact_anchor_or_diagnostic_pair",
        payload,
        repo_root=resolved_repo_root,
    ):
        return None
    predicate_id = str(
        payload.get("predicate_id") or TT5L_PAIRED_EXACT_ANCHOR_PAIR_PREDICATE_ID
    ).strip()
    return L5V2GateEvidence(
        gate_id="exact_anchor_or_diagnostic_pair",
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha,
        predicate_id=predicate_id,
        predicate_passed=True,
        evidence_grade="tt5l_paired_cpu_cuda_anchor_pair_artifact",
    )


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
    schema = str(payload.get("schema") or "")
    if payload and schema != TT5L_DYKSTRA_FEASIBILITY_SCHEMA:
        blockers.append("tt5l_dykstra_feasibility_schema_missing_or_stale")
    predicate_id = str(payload.get("predicate_id") or "")
    if payload and predicate_id != TT5L_DYKSTRA_FEASIBILITY_PREDICATE_ID:
        blockers.append("tt5l_dykstra_feasibility_predicate_id_missing_or_stale")
    generated_by_tool = str(payload.get("generated_by_tool") or "")
    if payload and generated_by_tool != TT5L_DYKSTRA_FEASIBILITY_GENERATED_BY_TOOL:
        blockers.append("tt5l_dykstra_feasibility_generated_by_tool_missing_or_stale")
    generated_at_utc = str(payload.get("generated_at_utc") or "")
    if payload and not generated_at_utc:
        blockers.append("tt5l_dykstra_feasibility_generated_at_utc_missing")
    command_argv = payload.get("command_argv")
    if payload and not (
        isinstance(command_argv, list)
        and command_argv
        and all(isinstance(item, str) and item for item in command_argv)
    ):
        blockers.append("tt5l_dykstra_feasibility_command_argv_missing")
    tool_sha256 = str(payload.get("tool_sha256") or "").lower()
    if payload and not _SHA256_HEX_RE.fullmatch(tool_sha256):
        blockers.append("tt5l_dykstra_feasibility_tool_sha256_invalid")
    elif payload and tool_path.is_file() and _sha256_file(tool_path) != tool_sha256:
        blockers.append("tt5l_dykstra_feasibility_tool_sha256_mismatch")
    if payload and substrate_id not in {TT5L_DYKSTRA_SUBSTRATE_ID, SUBJECT_ID, LANE_ID}:
        blockers.append("tt5l_dykstra_feasibility_substrate_id_mismatch")
    verdict = str(payload.get("verdict") or "")
    if payload and verdict not in {"FEASIBLE", "INFEASIBLE", "INDETERMINATE"}:
        blockers.append("tt5l_dykstra_feasibility_verdict_invalid")
    elif payload and verdict == "INDETERMINATE":
        blockers.append("tt5l_dykstra_feasibility_verdict_indeterminate")
    elif payload and verdict == "INFEASIBLE":
        blockers.append("tt5l_dykstra_feasibility_verdict_infeasible")
    if payload and "predicted_band" in payload:
        blockers.append("tt5l_dykstra_feasibility_active_predicted_band_field_present")
    tested_score_axis_band = payload.get("tested_score_axis_band")
    if payload and not (
        isinstance(tested_score_axis_band, list | tuple)
        and len(tested_score_axis_band) == 2
        and all(_json_float(item) is not None for item in tested_score_axis_band)
    ):
        blockers.append("tt5l_dykstra_feasibility_tested_score_axis_band_missing")
    input_band_role = str(payload.get("input_band_role") or "")
    if payload and input_band_role != "planning_band_not_score_or_rank_authority":
        blockers.append("tt5l_dykstra_feasibility_input_band_role_missing_or_stale")
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
    feasibility_scope = str(payload.get("feasibility_scope") or "")
    if payload and feasibility_scope != TT5L_DYKSTRA_FEASIBILITY_SCOPE:
        blockers.append("tt5l_dykstra_feasibility_scope_missing_or_stale")
    verdict_authority_scope = str(payload.get("verdict_authority_scope") or "")
    if (
        payload
        and verdict_authority_scope != TT5L_DYKSTRA_VERDICT_AUTHORITY_SCOPE
    ):
        blockers.append(
            "tt5l_dykstra_feasibility_verdict_authority_scope_missing_or_stale"
        )
    move_level_constraint_proof = payload.get("move_level_constraint_proof")
    if payload and move_level_constraint_proof is not False:
        blockers.append("tt5l_dykstra_feasibility_move_level_proof_not_false")
    constraint_ids_payload = payload.get("constraint_set_ids")
    if isinstance(constraint_ids_payload, list):
        constraint_ids = {str(item) for item in constraint_ids_payload}
    else:
        constraint_ids = set()
    if payload and not TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS.issubset(constraint_ids):
        blockers.append("tt5l_dykstra_feasibility_five_move_constraints_missing")
    if payload and constraint_ids != TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS:
        blockers.append("tt5l_dykstra_feasibility_constraint_set_ids_not_exact")
    constraint_set_count = _non_bool_int(payload.get("constraint_set_count"))
    if (
        payload
        and constraint_set_count != len(TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS)
    ):
        blockers.append("tt5l_dykstra_feasibility_constraint_set_count_mismatch")
    for field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if payload and payload.get(field) is not False:
            blockers.append(f"tt5l_dykstra_feasibility_{field}_not_false")

    valid = not blockers
    return {
        "schema": "l5_v2_tt5l_dykstra_feasibility_status_v1",
        "tool_path": TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH,
        "tool_exists": tool_path.is_file(),
        "artifact_path": TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": valid,
        "artifact_schema": schema or None,
        "predicate_id": predicate_id or None,
        "generated_by_tool": generated_by_tool or None,
        "generated_at_utc": generated_at_utc or None,
        "command_argv": list(command_argv)
        if isinstance(command_argv, list)
        else None,
        "tool_sha256": tool_sha256 or None,
        "substrate_id": substrate_id or None,
        "verdict": verdict or None,
        "tested_score_axis_band": list(tested_score_axis_band)
        if isinstance(tested_score_axis_band, list | tuple)
        else None,
        "input_band_role": input_band_role or None,
        "archive_size_bytes": archive_size_bytes,
        "feasibility_band_lo": payload.get("feasibility_band_lo"),
        "feasibility_band_hi": payload.get("feasibility_band_hi"),
        "feasibility_rationale": payload.get("feasibility_rationale"),
        "score_formula": score_formula or None,
        "contest_seg_multiplier": contest_seg_multiplier,
        "polytope_projection_kind": projection_kind or None,
        "feasibility_scope": feasibility_scope or None,
        "verdict_authority_scope": verdict_authority_scope or None,
        "move_level_constraint_proof": move_level_constraint_proof,
        "constraint_set_ids": sorted(constraint_ids),
        "constraint_set_count": constraint_set_count,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def _tt5l_move_level_feasibility_status(*, repo_root: Path) -> dict[str, Any]:
    """Return the TT5L move-level feasibility artifact status.

    The Dykstra artifact is only score-axis sanity. This artifact is the
    separate proof that the TT5L move-level constraints have a non-empty,
    residual-bounded intersection before side-info curves or timing smokes
    can advance.
    """

    artifact_path = repo_root / TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        blockers.append("tt5l_move_level_feasibility_artifact_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("tt5l_move_level_feasibility_artifact_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
                if not payload:
                    blockers.append("tt5l_move_level_feasibility_artifact_empty")
            else:
                blockers.append("tt5l_move_level_feasibility_artifact_not_object")

    schema = str(payload.get("schema") or "")
    if payload and schema != TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA:
        blockers.append("tt5l_move_level_feasibility_schema_mismatch")
    subject_id = str(payload.get("subject_id") or payload.get("substrate_id") or "")
    if payload and subject_id not in {TT5L_DYKSTRA_SUBSTRATE_ID, SUBJECT_ID, LANE_ID}:
        blockers.append("tt5l_move_level_feasibility_subject_id_mismatch")
    predicate_id = str(payload.get("predicate_id") or "")
    if payload and predicate_id != TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID:
        blockers.append("tt5l_move_level_feasibility_predicate_id_mismatch")
    if payload and payload.get("predicate_passed") is not True:
        blockers.append("tt5l_move_level_feasibility_predicate_not_passed")
    if payload and payload.get("move_level_constraint_proof") is not True:
        blockers.append("tt5l_move_level_feasibility_proof_not_true")
    generated_by_tool = str(payload.get("generated_by_tool") or "").strip()
    if payload and generated_by_tool != TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH:
        blockers.append("tt5l_move_level_feasibility_generated_by_tool_mismatch")
    tool_sha256 = str(payload.get("tool_sha256") or "").strip().lower()
    resolved_tool: Path | None = None
    if payload and not _SHA256_HEX_RE.fullmatch(tool_sha256):
        blockers.append("tt5l_move_level_feasibility_tool_sha256_invalid")
    elif payload:
        resolved_tool, tool_path_error = _resolve_artifact_path(
            TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH,
            repo_root,
        )
        if tool_path_error is not None:
            blockers.append(
                f"tt5l_move_level_feasibility_tool_path_{tool_path_error}"
            )
            resolved_tool = None
        elif resolved_tool is None or not resolved_tool.is_file():
            blockers.append("tt5l_move_level_feasibility_tool_missing")
            resolved_tool = None
        elif _sha256_file(resolved_tool) != tool_sha256:
            blockers.append("tt5l_move_level_feasibility_tool_sha256_mismatch")
    command_argv = payload.get("command_argv")
    command_argv_valid = (
        isinstance(command_argv, list)
        and bool(command_argv)
        and all(isinstance(item, str) and bool(item.strip()) for item in command_argv)
    )
    if payload and not command_argv_valid:
        blockers.append("tt5l_move_level_feasibility_command_argv_missing")

    residual_max = _json_float(payload.get("residual_max"))
    residual_tolerance = _json_float(payload.get("residual_tolerance"))
    if payload and residual_max is None:
        blockers.append("tt5l_move_level_feasibility_residual_max_missing")
    if payload and residual_tolerance is None:
        blockers.append("tt5l_move_level_feasibility_residual_tolerance_missing")
    if (
        payload
        and residual_max is not None
        and residual_tolerance is not None
        and residual_max > residual_tolerance
    ):
        blockers.append("tt5l_move_level_feasibility_residual_exceeds_tolerance")

    constraint_ids_payload = payload.get("constraint_set_ids")
    if isinstance(constraint_ids_payload, list):
        constraint_ids = {str(item) for item in constraint_ids_payload}
    else:
        constraint_ids = set()
    if payload and constraint_ids != TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS:
        blockers.append("tt5l_move_level_feasibility_constraint_set_ids_not_exact")
    constraint_set_count = _non_bool_int(payload.get("constraint_set_count"))
    if (
        payload
        and constraint_set_count != len(TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS)
    ):
        blockers.append("tt5l_move_level_feasibility_constraint_set_count_mismatch")

    proof_artifact_path = str(payload.get("proof_artifact_path") or "").strip()
    resolved_proof_artifact: Path | None = None
    proof_payload: Mapping[str, Any] = {}
    if not proof_artifact_path:
        if payload:
            blockers.append("tt5l_move_level_feasibility_proof_artifact_path_missing")
    elif _is_transient_artifact_path(proof_artifact_path):
        blockers.append("tt5l_move_level_feasibility_proof_artifact_path_transient")
    else:
        resolved_proof_artifact, proof_path_error = _resolve_artifact_path(
            proof_artifact_path,
            repo_root,
        )
        if proof_path_error is not None:
            blockers.append(
                f"tt5l_move_level_feasibility_proof_artifact_path_{proof_path_error}"
            )
            resolved_proof_artifact = None
        elif resolved_proof_artifact is None or not resolved_proof_artifact.is_file():
            blockers.append("tt5l_move_level_feasibility_proof_artifact_missing")
            resolved_proof_artifact = None
        else:
            try:
                loaded_proof = json.loads(
                    resolved_proof_artifact.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                blockers.append("tt5l_move_level_feasibility_proof_artifact_json_invalid")
            else:
                if isinstance(loaded_proof, Mapping) and loaded_proof:
                    proof_payload = loaded_proof
                else:
                    blockers.append(
                        "tt5l_move_level_feasibility_proof_artifact_not_object"
                    )

    proof_artifact_sha256 = str(
        payload.get("proof_artifact_sha256") or ""
    ).strip().lower()
    if payload and not _SHA256_HEX_RE.fullmatch(proof_artifact_sha256):
        blockers.append("tt5l_move_level_feasibility_proof_artifact_sha256_invalid")
    elif (
        resolved_proof_artifact is not None
        and _sha256_file(resolved_proof_artifact) != proof_artifact_sha256
    ):
        blockers.append("tt5l_move_level_feasibility_proof_artifact_sha256_mismatch")

    score_axis_artifact_path = str(
        payload.get("score_axis_sanity_artifact_path") or ""
    ).strip()
    resolved_score_axis_artifact: Path | None = None
    if not score_axis_artifact_path:
        if payload:
            blockers.append(
                "tt5l_move_level_feasibility_score_axis_sanity_artifact_path_missing"
            )
    elif _is_transient_artifact_path(score_axis_artifact_path):
        blockers.append(
            "tt5l_move_level_feasibility_score_axis_sanity_artifact_path_transient"
        )
    else:
        resolved_score_axis_artifact, score_axis_path_error = _resolve_artifact_path(
            score_axis_artifact_path,
            repo_root,
        )
        if score_axis_path_error is not None:
            blockers.append(
                "tt5l_move_level_feasibility_score_axis_sanity_artifact_path_"
                f"{score_axis_path_error}"
            )
            resolved_score_axis_artifact = None
        elif (
            resolved_score_axis_artifact is None
            or not resolved_score_axis_artifact.is_file()
        ):
            blockers.append(
                "tt5l_move_level_feasibility_score_axis_sanity_artifact_missing"
            )
            resolved_score_axis_artifact = None
    score_axis_artifact_sha256 = str(
        payload.get("score_axis_sanity_artifact_sha256") or ""
    ).strip().lower()
    if payload and not _SHA256_HEX_RE.fullmatch(score_axis_artifact_sha256):
        blockers.append(
            "tt5l_move_level_feasibility_score_axis_sanity_artifact_sha256_invalid"
        )
    elif (
        resolved_score_axis_artifact is not None
        and _sha256_file(resolved_score_axis_artifact) != score_axis_artifact_sha256
    ):
        blockers.append(
            "tt5l_move_level_feasibility_score_axis_sanity_artifact_sha256_mismatch"
        )

    if proof_payload:
        proof_generated_by_tool = str(proof_payload.get("generated_by_tool") or "")
        proof_tool_sha256 = str(proof_payload.get("tool_sha256") or "").strip().lower()
        if not proof_generated_by_tool:
            blockers.append("tt5l_move_level_feasibility_proof_generated_by_tool_missing")
        if not _SHA256_HEX_RE.fullmatch(proof_tool_sha256):
            blockers.append("tt5l_move_level_feasibility_proof_tool_sha256_invalid")
        else:
            proof_tool_path, proof_tool_path_error = _resolve_artifact_path(
                proof_generated_by_tool,
                repo_root,
            )
            if proof_tool_path_error is not None:
                blockers.append(
                    "tt5l_move_level_feasibility_proof_tool_path_"
                    f"{proof_tool_path_error}"
                )
            elif proof_tool_path is None or not proof_tool_path.is_file():
                blockers.append("tt5l_move_level_feasibility_proof_tool_missing")
            elif _sha256_file(proof_tool_path) != proof_tool_sha256:
                blockers.append(
                    "tt5l_move_level_feasibility_proof_tool_sha256_mismatch"
                )
        if proof_payload.get("score_axis_sanity_artifact_sha256") != (
            score_axis_artifact_sha256 or None
        ):
            blockers.append(
                "tt5l_move_level_feasibility_proof_score_axis_sha256_mismatch"
            )
        proof_records_payload = proof_payload.get("mechanism_records")
        if isinstance(proof_records_payload, list):
            proof_records = [
                row for row in proof_records_payload if isinstance(row, Mapping)
            ]
        else:
            proof_records = []
        record_ids = {
            str(record.get("constraint_id") or "")
            for record in proof_records
            if record.get("passed") is True and isinstance(record.get("details"), Mapping)
        }
        if not TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS.issubset(record_ids):
            blockers.append(
                "tt5l_move_level_feasibility_proof_mechanism_records_incomplete"
            )
        witness_variables = proof_payload.get("witness_variables")
        if not isinstance(witness_variables, Mapping) or not witness_variables:
            blockers.append("tt5l_move_level_feasibility_witness_variables_missing")

    for field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if payload and payload.get(field) is not False:
            blockers.append(f"tt5l_move_level_feasibility_{field}_not_false")

    valid = not blockers
    return {
        "schema": "l5_v2_tt5l_move_level_feasibility_status_v1",
        "artifact_path": TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": valid,
        "subject_id": subject_id or None,
        "predicate_id": predicate_id or None,
        "predicate_passed": payload.get("predicate_passed"),
        "move_level_constraint_proof": payload.get("move_level_constraint_proof"),
        "generated_by_tool": generated_by_tool or None,
        "tool_sha256": tool_sha256 or None,
        "command_argv": list(command_argv) if command_argv_valid else None,
        "residual_max": residual_max,
        "residual_tolerance": residual_tolerance,
        "constraint_set_ids": sorted(constraint_ids),
        "constraint_set_count": constraint_set_count,
        "proof_artifact_path": proof_artifact_path or None,
        "proof_artifact_sha256": proof_artifact_sha256 or None,
        "score_axis_sanity_artifact_path": score_axis_artifact_path or None,
        "score_axis_sanity_artifact_sha256": score_axis_artifact_sha256 or None,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def tt5l_move_level_feasibility_status(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return public TT5L move-level feasibility proof custody status."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    return _tt5l_move_level_feasibility_status(repo_root=resolved_repo_root)


def _tt5l_first_anchor_timing_smoke_status(*, repo_root: Path) -> dict[str, Any]:
    """Return TT5L first-anchor timing-smoke artifact custody status."""

    artifact_path = repo_root / TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        blockers.append("tt5l_first_anchor_timing_smoke_artifact_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("tt5l_first_anchor_timing_smoke_artifact_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
                if not payload:
                    blockers.append("tt5l_first_anchor_timing_smoke_artifact_empty")
            else:
                blockers.append("tt5l_first_anchor_timing_smoke_artifact_not_object")

    schema = str(payload.get("schema") or "")
    if payload and schema != TT5L_FIRST_ANCHOR_TIMING_SMOKE_SCHEMA:
        blockers.append("tt5l_first_anchor_timing_smoke_schema_mismatch")
    subject_id = str(payload.get("lane_id") or payload.get("subject_id") or "")
    if payload and subject_id not in {LANE_ID, SUBJECT_ID}:
        blockers.append("tt5l_first_anchor_timing_smoke_subject_id_mismatch")
    predicate_id = str(payload.get("predicate_id") or "")
    if payload and predicate_id != TT5L_FIRST_ANCHOR_TIMING_SMOKE_PREDICATE_ID:
        blockers.append("tt5l_first_anchor_timing_smoke_predicate_id_mismatch")
    if payload and payload.get("predicate_passed") is not True:
        blockers.append("tt5l_first_anchor_timing_smoke_predicate_not_passed")

    axes_payload = payload.get("required_axes", payload.get("paired_axes"))
    paired_axes = (
        tuple(str(axis) for axis in axes_payload)
        if isinstance(axes_payload, list)
        else ()
    )
    if payload and (
        set(paired_axes) != set(_REQUIRED_EXACT_AXES)
        or len(paired_axes) != len(_REQUIRED_EXACT_AXES)
    ):
        blockers.append("tt5l_first_anchor_timing_smoke_required_axes_not_paired")

    provider = str(payload.get("provider") or "").strip()
    if payload and provider.lower() in {"", "tbd", "unknown", "none"}:
        blockers.append("tt5l_first_anchor_timing_smoke_provider_missing")
    hardware = str(payload.get("gpu") or payload.get("hardware") or "").strip()
    if payload and hardware.lower() in {"", "tbd", "unknown", "none"}:
        blockers.append("tt5l_first_anchor_timing_smoke_hardware_missing")
    provider_call_id = str(
        payload.get("provider_call_id") or payload.get("call_id") or ""
    ).strip()
    if payload and not provider_call_id:
        blockers.append("tt5l_first_anchor_timing_smoke_provider_call_id_missing")

    command_argv = payload.get("command_argv")
    command_argv_valid = (
        isinstance(command_argv, list)
        and bool(command_argv)
        and all(isinstance(item, str) and bool(item.strip()) for item in command_argv)
    )
    if payload and not command_argv_valid:
        blockers.append("tt5l_first_anchor_timing_smoke_command_argv_missing")

    elapsed_seconds = _json_float(payload.get("elapsed_seconds"))
    seconds_per_epoch = _json_float(payload.get("seconds_per_epoch"))
    seconds_per_candidate = _json_float(payload.get("seconds_per_candidate"))
    if payload and (elapsed_seconds is None or elapsed_seconds <= 0):
        blockers.append("tt5l_first_anchor_timing_smoke_elapsed_seconds_missing")
    if payload and (
        (seconds_per_epoch is None or seconds_per_epoch <= 0)
        and (seconds_per_candidate is None or seconds_per_candidate <= 0)
    ):
        blockers.append("tt5l_first_anchor_timing_smoke_rate_metric_missing")

    axis_timings_payload = payload.get("axis_timings")
    axis_timings = (
        axis_timings_payload if isinstance(axis_timings_payload, Mapping) else {}
    )
    if payload and axis_timings:
        missing_axis_timings = [
            axis for axis in _REQUIRED_EXACT_AXES if axis not in axis_timings
        ]
        if missing_axis_timings:
            blockers.append(
                "tt5l_first_anchor_timing_smoke_axis_timings_missing:"
                + ",".join(missing_axis_timings)
            )
        for axis, raw_axis_timing in axis_timings.items():
            axis_name = str(axis)
            if axis_name not in _REQUIRED_EXACT_AXES:
                blockers.append(
                    "tt5l_first_anchor_timing_smoke_axis_timing_unexpected_axis:"
                    + axis_name
                )
                continue
            if not isinstance(raw_axis_timing, Mapping):
                blockers.append(
                    "tt5l_first_anchor_timing_smoke_axis_timing_not_object:"
                    + axis_name
                )
                continue
            axis_elapsed = _json_float(
                raw_axis_timing.get("contest_auth_eval_elapsed_seconds")
            )
            if axis_elapsed is None or axis_elapsed <= 0:
                blockers.append(
                    "tt5l_first_anchor_timing_smoke_axis_elapsed_missing:"
                    + axis_name
                )
            axis_result_path = str(
                raw_axis_timing.get("contest_auth_eval_artifact_path") or ""
            ).strip()
            resolved_axis_result: Path | None = None
            if not axis_result_path:
                blockers.append(
                    "tt5l_first_anchor_timing_smoke_axis_result_path_missing:"
                    + axis_name
                )
            elif _is_transient_artifact_path(axis_result_path):
                blockers.append(
                    "tt5l_first_anchor_timing_smoke_axis_result_path_transient:"
                    + axis_name
                )
            else:
                resolved_axis_result, axis_result_error = _resolve_artifact_path(
                    axis_result_path,
                    repo_root,
                )
                if axis_result_error is not None:
                    blockers.append(
                        "tt5l_first_anchor_timing_smoke_axis_result_path_"
                        f"{axis_result_error}:{axis_name}"
                    )
                    resolved_axis_result = None
                elif resolved_axis_result is None or not resolved_axis_result.is_file():
                    blockers.append(
                        "tt5l_first_anchor_timing_smoke_axis_result_missing:"
                        + axis_name
                    )
                    resolved_axis_result = None
            axis_result_sha = str(
                raw_axis_timing.get("contest_auth_eval_artifact_sha256") or ""
            ).strip().lower()
            if not _SHA256_HEX_RE.fullmatch(axis_result_sha):
                blockers.append(
                    "tt5l_first_anchor_timing_smoke_axis_result_sha_invalid:"
                    + axis_name
                )
            elif (
                resolved_axis_result is not None
                and _sha256_file(resolved_axis_result) != axis_result_sha
            ):
                blockers.append(
                    "tt5l_first_anchor_timing_smoke_axis_result_sha_mismatch:"
                    + axis_name
                )

    result_artifact_path = str(payload.get("result_artifact_path") or "").strip()
    resolved_result_artifact: Path | None = None
    if not result_artifact_path:
        if payload:
            blockers.append("tt5l_first_anchor_timing_smoke_result_artifact_path_missing")
    elif _is_transient_artifact_path(result_artifact_path):
        blockers.append("tt5l_first_anchor_timing_smoke_result_artifact_path_transient")
    else:
        resolved_result_artifact, result_path_error = _resolve_artifact_path(
            result_artifact_path,
            repo_root,
        )
        if result_path_error is not None:
            blockers.append(
                f"tt5l_first_anchor_timing_smoke_result_artifact_path_{result_path_error}"
            )
            resolved_result_artifact = None
        elif resolved_result_artifact is None or not resolved_result_artifact.is_file():
            blockers.append("tt5l_first_anchor_timing_smoke_result_artifact_missing")
            resolved_result_artifact = None

    result_artifact_sha256 = str(
        payload.get("result_artifact_sha256") or ""
    ).strip().lower()
    if payload and not _SHA256_HEX_RE.fullmatch(result_artifact_sha256):
        blockers.append("tt5l_first_anchor_timing_smoke_result_artifact_sha256_invalid")
    elif (
        resolved_result_artifact is not None
        and _sha256_file(resolved_result_artifact) != result_artifact_sha256
    ):
        blockers.append("tt5l_first_anchor_timing_smoke_result_artifact_sha256_mismatch")

    for field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if payload and payload.get(field) is not False:
            blockers.append(f"tt5l_first_anchor_timing_smoke_{field}_not_false")

    valid = not blockers
    return {
        "schema": "l5_v2_tt5l_first_anchor_timing_smoke_status_v1",
        "artifact_path": TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": valid,
        "subject_id": subject_id or None,
        "predicate_id": predicate_id or None,
        "predicate_passed": payload.get("predicate_passed"),
        "required_axes": list(paired_axes),
        "provider": provider or None,
        "hardware": hardware or None,
        "provider_call_id": provider_call_id or None,
        "command_argv": list(command_argv) if command_argv_valid else None,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_epoch": seconds_per_epoch,
        "seconds_per_candidate": seconds_per_candidate,
        "axis_timings": dict(axis_timings) if axis_timings else {},
        "result_artifact_path": result_artifact_path or None,
        "result_artifact_sha256": result_artifact_sha256 or None,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def tt5l_first_anchor_timing_smoke_status(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return public TT5L timing-smoke custody status for tools and tests."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    return _tt5l_first_anchor_timing_smoke_status(repo_root=resolved_repo_root)


def _tt5l_sideinfo_effect_curve_status(*, repo_root: Path) -> dict[str, Any]:
    artifact_path = repo_root / TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH
    blockers: list[str] = []
    loaded: object | None = None
    payload: Mapping[str, Any] = {}

    if not artifact_path.is_file():
        blockers.append("tt5l_sideinfo_effect_curve_artifact_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append("tt5l_sideinfo_effect_curve_artifact_json_invalid")
        if loaded is not None and not isinstance(loaded, Mapping):
            blockers.append("tt5l_sideinfo_effect_curve_artifact_not_object")
        if isinstance(loaded, Mapping):
            payload = loaded
            blockers.extend(
                validate_l5_v2_sideinfo_effect_curve(loaded, repo_root=repo_root)
            )

    blockers = list(dict.fromkeys(blockers))
    observed_cells: list[dict[str, Any]] = []
    raw_observed_cells = payload.get("observed_cells")
    if isinstance(raw_observed_cells, list):
        for raw_cell in raw_observed_cells:
            if not isinstance(raw_cell, Mapping):
                continue
            liveness = raw_cell.get("sideinfo_liveness")
            if not isinstance(liveness, Mapping):
                liveness = {}
            observed_cells.append(
                {
                    "axis": str(raw_cell.get("axis") or ""),
                    "variant": str(raw_cell.get("variant") or ""),
                    "score": raw_cell.get("score"),
                    "archive_sha256": str(raw_cell.get("archive_sha256") or ""),
                    "archive_bytes": raw_cell.get("archive_bytes"),
                    "runtime_content_tree_sha256": str(
                        raw_cell.get("runtime_content_tree_sha256") or ""
                    ),
                    "artifact_path": str(raw_cell.get("artifact_path") or ""),
                    "sideinfo_checked": liveness.get("checked"),
                    "sideinfo_nonzero_fraction": liveness.get("nonzero_fraction"),
                    "sideinfo_nonzero_values": liveness.get("nonzero_values"),
                    "sideinfo_total_values": liveness.get("total_values"),
                    "sideinfo_nonzero_pair_count": liveness.get(
                        "nonzero_pair_count"
                    ),
                    "sideinfo_total_pairs": liveness.get("total_pairs"),
                }
            )
    missing_cells: list[str] = []
    missing_prefix = "tt5l_sideinfo_effect_curve_cells_missing:"
    for blocker in blockers:
        if blocker.startswith(missing_prefix):
            missing_cells = [
                cell for cell in blocker.removeprefix(missing_prefix).split(",") if cell
            ]
            break
    effect_blockers = payload.get("effect_blockers")
    contract_blockers = payload.get("contract_blockers")
    axis_effects = payload.get("axis_effects")
    return {
        "schema": "l5_v2_tt5l_sideinfo_effect_curve_status_v1",
        "artifact_path": TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_present": artifact_path.is_file(),
        "artifact_valid": not blockers,
        "measurement_id": str(payload.get("measurement_id") or ""),
        "predicate_id": str(payload.get("predicate_id") or ""),
        "predicate_passed": payload.get("predicate_passed"),
        "required_axes": list(payload.get("required_axes") or []),
        "required_variants": list(payload.get("required_variants") or []),
        "observed_cell_count": len(observed_cells),
        "observed_cells": observed_cells,
        "missing_cells": missing_cells,
        "axis_effects": axis_effects if isinstance(axis_effects, Mapping) else {},
        "effect_blockers": (
            list(effect_blockers) if isinstance(effect_blockers, list) else []
        ),
        "contract_blockers": (
            list(contract_blockers) if isinstance(contract_blockers, list) else []
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def tt5l_sideinfo_effect_curve_status(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return public TT5L side-info effect-curve custody status."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    return _tt5l_sideinfo_effect_curve_status(repo_root=resolved_repo_root)


def _tt5l_sideinfo_effect_curve_harvest_cells_status(
    *,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_path = repo_root / TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        blockers.append("tt5l_sideinfo_effect_curve_harvest_cells_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("tt5l_sideinfo_effect_curve_harvest_cells_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
            else:
                blockers.append("tt5l_sideinfo_effect_curve_harvest_cells_not_object")
    if payload and payload.get("schema") != L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA:
        blockers.append("tt5l_sideinfo_effect_curve_harvest_cells_schema_mismatch")
    for field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
    ):
        if payload and payload.get(field) is not False:
            blockers.append(f"tt5l_sideinfo_effect_curve_harvest_cells_{field}_not_false")
    cells = payload.get("cells") if isinstance(payload.get("cells"), list) else []
    expected_cells = {
        f"{axis}:{variant}"
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    }
    observed_cells: list[dict[str, Any]] = []
    observed_keys: set[str] = set()
    for raw_cell in cells:
        if not isinstance(raw_cell, Mapping):
            blockers.append("tt5l_sideinfo_effect_curve_harvest_cells_cell_not_object")
            continue
        axis = str(raw_cell.get("axis") or "")
        variant = str(raw_cell.get("variant") or "")
        key = f"{axis}:{variant}"
        observed_keys.add(key)
        liveness = raw_cell.get("sideinfo_liveness")
        if not isinstance(liveness, Mapping):
            liveness = {}
        cell_blockers = raw_cell.get("blockers")
        observed_cells.append(
            {
                "axis": axis,
                "variant": variant,
                "archive_sha256": str(raw_cell.get("archive_sha256") or ""),
                "pair_group_id": str(raw_cell.get("pair_group_id") or ""),
                "run_id": str(raw_cell.get("run_id") or ""),
                "sideinfo_checked": liveness.get("checked"),
                "sideinfo_nonzero_fraction": liveness.get("nonzero_fraction"),
                "sideinfo_nonzero_values": liveness.get("nonzero_values"),
                "sideinfo_total_values": liveness.get("total_values"),
                "blockers": list(cell_blockers) if isinstance(cell_blockers, list) else [],
            }
        )
        for field in ("axis", "variant", "archive_sha256", "pair_group_id", "run_id"):
            if not str(raw_cell.get(field) or "").strip():
                blockers.append(
                    f"tt5l_sideinfo_effect_curve_harvest_cells_cell_{field}_missing:"
                    f"{axis or '<missing>'}:{variant or '<missing>'}"
                )
        if liveness.get("checked") is not True:
            blockers.append(
                "tt5l_sideinfo_effect_curve_harvest_cells_sideinfo_unchecked:"
                f"{axis or '<missing>'}:{variant or '<missing>'}"
            )
    missing_cells = sorted(expected_cells - observed_keys)
    extra_cells = sorted(observed_keys - expected_cells)
    if missing_cells:
        blockers.append(
            "tt5l_sideinfo_effect_curve_harvest_cells_missing_cells:"
            + ",".join(missing_cells)
        )
    if extra_cells:
        blockers.append(
            "tt5l_sideinfo_effect_curve_harvest_cells_extra_cells:"
            + ",".join(extra_cells)
        )
    artifact_structurally_valid = artifact_path.is_file() and not blockers
    return {
        "schema": "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_status_v1",
        "artifact_path": TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_present": artifact_path.is_file(),
        "artifact_structurally_valid": artifact_structurally_valid,
        "artifact_valid": artifact_structurally_valid,
        "tool_path": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH,
        "source_plan": str(payload.get("source_plan") or ""),
        "source_variant_manifest": str(payload.get("source_variant_manifest") or ""),
        "cell_count": len(observed_cells),
        "expected_cell_count": len(expected_cells),
        "harvested_exact_eval_artifact_count": payload.get(
            "harvested_exact_eval_artifact_count"
        ),
        "missing_exact_eval_artifact_count": payload.get(
            "missing_exact_eval_artifact_count"
        ),
        "observed_cells": observed_cells,
        "blockers": list(dict.fromkeys(blockers)),
        "cell_blockers": (
            list(payload.get("blockers")) if isinstance(payload.get("blockers"), list) else []
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def tt5l_sideinfo_effect_curve_harvest_cells_status(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return public TT5L side-info harvest-cell custody status."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    return _tt5l_sideinfo_effect_curve_harvest_cells_status(
        repo_root=resolved_repo_root
    )


def _tt5l_sideinfo_effect_curve_dispatch_plan_status(
    *,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_path = repo_root / TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
            else:
                blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_not_object")

    if payload and payload.get("schema") != TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA:
        blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_schema_mismatch")
    for field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "ready_for_provider_dispatch",
        "dispatch_attempted",
    ):
        if payload and payload.get(field) is not False:
            blockers.append(f"tt5l_sideinfo_effect_curve_dispatch_plan_{field}_not_false")
    required_axes = list(payload.get("required_axes") or [])
    required_variants = list(payload.get("required_variants") or [])
    if payload and required_axes != list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES):
        blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_required_axes_mismatch")
    if payload and required_variants != list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS):
        blockers.append(
            "tt5l_sideinfo_effect_curve_dispatch_plan_required_variants_mismatch"
        )

    raw_work_units = payload.get("work_units")
    work_units: list[dict[str, Any]] = []
    if payload and not isinstance(raw_work_units, list):
        blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_work_units_missing")
        raw_work_units = []
    if isinstance(raw_work_units, list):
        seen_variants: set[str] = set()
        for raw_unit in raw_work_units:
            if not isinstance(raw_unit, Mapping):
                blockers.append(
                    "tt5l_sideinfo_effect_curve_dispatch_plan_work_unit_not_object"
                )
                continue
            variant = str(raw_unit.get("variant") or "").strip()
            if not variant:
                blockers.append(
                    "tt5l_sideinfo_effect_curve_dispatch_plan_work_unit_variant_missing"
                )
            elif variant in seen_variants:
                blockers.append(
                    "tt5l_sideinfo_effect_curve_dispatch_plan_duplicate_variant:"
                    + variant
                )
            else:
                seen_variants.add(variant)
            archive = raw_unit.get("archive")
            if not isinstance(archive, Mapping):
                archive = {}
            dispatch_blockers = raw_unit.get("dispatch_blockers")
            score_claim_blockers = raw_unit.get("score_claim_blockers")
            work_units.append(
                {
                    "variant": variant,
                    "work_unit_id": str(raw_unit.get("work_unit_id") or ""),
                    "lane_id": str(raw_unit.get("lane_id") or ""),
                    "pair_group_id": str(raw_unit.get("pair_group_id") or ""),
                    "archive_sha256": str(archive.get("sha256") or ""),
                    "archive_bytes": archive.get("bytes"),
                    "ready_for_operator_dispatch": raw_unit.get(
                        "ready_for_operator_dispatch"
                    )
                    is True,
                    "ready_for_provider_dispatch": raw_unit.get(
                        "ready_for_provider_dispatch"
                    )
                    is True,
                    "dispatch_blockers": (
                        list(dispatch_blockers)
                        if isinstance(dispatch_blockers, list)
                        else []
                    ),
                    "score_claim_blockers": (
                        list(score_claim_blockers)
                        if isinstance(score_claim_blockers, list)
                        else []
                    ),
                }
            )
        missing_variants = [
            variant
            for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
            if variant not in seen_variants
        ]
        if missing_variants:
            blockers.append(
                "tt5l_sideinfo_effect_curve_dispatch_plan_missing_variants:"
                + ",".join(missing_variants)
            )
    ready_count = sum(
        1 for unit in work_units if unit["ready_for_operator_dispatch"] is True
    )
    declared_ready_count = payload.get("ready_work_unit_count")
    declared_work_unit_count = payload.get("work_unit_count")
    if payload and declared_work_unit_count != len(work_units):
        blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_work_unit_count_mismatch")
    if payload and declared_ready_count != ready_count:
        blockers.append("tt5l_sideinfo_effect_curve_dispatch_plan_ready_count_mismatch")

    blockers = list(dict.fromkeys(blockers))
    return {
        "schema": "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_status_v1",
        "artifact_path": TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": bool(payload) and not blockers,
        "plan_id": str(payload.get("plan_id") or ""),
        "measurement_id": str(payload.get("measurement_id") or ""),
        "required_axes": required_axes,
        "required_variants": required_variants,
        "work_unit_count": len(work_units),
        "ready_work_unit_count": ready_count,
        "work_units": work_units,
        "ready_for_operator_dispatch": payload.get("ready_for_operator_dispatch") is True,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def _tt5l_probe_observation_intake_status(*, repo_root: Path) -> dict[str, Any]:
    artifact_path = repo_root / TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    evaluated_observation_count = 0
    candidate_ids: list[str] = []

    if not artifact_path.is_file():
        blockers.append("l5_v2_probe_observation_intake_artifact_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("l5_v2_probe_observation_intake_artifact_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
                if not payload:
                    blockers.append("l5_v2_probe_observation_intake_artifact_empty")
            else:
                blockers.append("l5_v2_probe_observation_intake_artifact_not_object")

    if payload and payload.get("schema") != L5V2_PROBE_OBSERVATION_INTAKE_SCHEMA:
        blockers.append("l5_v2_probe_observation_intake_schema_mismatch")
    for field in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        if payload and payload.get(field) is not False:
            blockers.append(f"l5_v2_probe_observation_intake_{field}_not_false")

    verdict = payload.get("verdict")
    if payload and not isinstance(verdict, Mapping):
        blockers.append("l5_v2_probe_observation_intake_verdict_missing_or_not_object")
    elif isinstance(verdict, Mapping):
        rows = verdict.get("evaluated_observations")
        if not isinstance(rows, list):
            blockers.append(
                "l5_v2_probe_observation_intake_evaluated_observations_missing"
            )
        else:
            evaluated_observation_count = len(rows)
            seen: set[str] = set()
            for row in rows:
                if not isinstance(row, Mapping):
                    blockers.append(
                        "l5_v2_probe_observation_intake_observation_not_object"
                    )
                    continue
                candidate_id = str(row.get("candidate_id") or "").strip()
                if candidate_id:
                    candidate_ids.append(candidate_id)
                    seen.add(candidate_id)
            missing = [candidate for candidate in L5V2_CANDIDATES if candidate not in seen]
            if missing:
                blockers.append(
                    "l5_v2_probe_observation_intake_candidate_coverage_incomplete:"
                    + ",".join(missing)
                )

    return {
        "schema": "l5_v2_probe_observation_intake_status_v1",
        "artifact_path": TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid_for_measurement_planning": not blockers,
        "intake_schema": str(payload.get("schema") or "") if payload else None,
        "evaluated_observation_count": evaluated_observation_count,
        "candidate_ids": candidate_ids,
        "architecture_lock_allowed": (
            payload.get("architecture_lock_allowed") is True if payload else False
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _inspect_tt5l_archive_sideinfo(archive_zip_path: Path) -> dict[str, Any]:
    """Return side-info liveness stats for a TT5L contest archive.zip.

    The L5-v2 paired dispatch gate is allowed to spend on low-rate TT5L
    archives, but not on archives whose temporal side channel is structurally
    dead. A zero side-info stream reproduces the 2026-05-16 TT5L 25ep collapse:
    excellent rate term, but no learned per-pair correction signal.
    """

    stats: dict[str, Any] = {
        "checked": True,
        "valid": False,
        "archive_member": "",
        "num_pairs": 0,
        "per_pair_bytes": 0,
        "total_values": 0,
        "nonzero_values": 0,
        "nonzero_fraction": 0.0,
        "min": None,
        "max": None,
        "error": "",
    }
    try:
        from tac.substrates.time_traveler_l5_autonomy.archive import (
            parse_archive,
            side_info_liveness_stats,
        )

        with zipfile.ZipFile(archive_zip_path, "r") as zf:
            names = set(zf.namelist())
            member = "0.bin" if "0.bin" in names else "x" if "x" in names else ""
            if not member:
                stats["error"] = "missing_tt5l_member_0_bin_or_x"
                return stats
            archive_bytes = zf.read(member)
        parsed = parse_archive(archive_bytes)
        side_info = parsed.per_pair_side_info
        liveness = side_info_liveness_stats(side_info)
        stats.update(
            {
                **liveness,
                "valid": True,
                "archive_member": member,
                "num_pairs": int(parsed.num_pairs),
                "per_pair_bytes": int(parsed.per_pair_bytes),
                "error": "",
            }
        )
    except Exception as exc:  # pragma: no cover - exact error surfaced in stats.
        stats["error"] = f"{type(exc).__name__}:{exc}"
    return stats


def _tt5l_materialized_paired_work_unit_status(*, repo_root: Path) -> dict[str, Any]:
    artifact_path = repo_root / TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    sideinfo_stats: dict[str, Any] = {
        "checked": False,
        "valid": False,
        "archive_member": "",
        "num_pairs": 0,
        "per_pair_bytes": 0,
        "total_values": 0,
        "nonzero_values": 0,
        "nonzero_fraction": 0.0,
        "min": None,
        "max": None,
        "error": "",
    }

    if not artifact_path.is_file():
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_missing")
    else:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("l5_v2_tt5l_materialized_paired_work_unit_json_invalid")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
            else:
                blockers.append("l5_v2_tt5l_materialized_paired_work_unit_not_object")

    if payload and payload.get("schema") != "modal_paired_auth_eval_dispatch_plan_v2":
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_schema_mismatch")
    if payload and payload.get("score_claim") is not False:
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_score_claim_not_false")
    if payload and payload.get("promotion_eligible") is not False:
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_promotion_not_false")
    if payload and payload.get("pair_group_id") != (
        TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID
    ):
        blockers.append(
            "l5_v2_tt5l_materialized_paired_work_unit_pair_group_id_mismatch"
        )

    archive = payload.get("archive") if payload else None
    archive_path = ""
    archive_sha256 = ""
    archive_bytes = 0
    if payload and not isinstance(archive, Mapping):
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_archive_missing")
    elif isinstance(archive, Mapping):
        archive_path = str(archive.get("path") or "").strip()
        archive_sha256 = str(archive.get("sha256") or "").strip()
        try:
            archive_bytes = int(archive.get("bytes") or 0)
        except (TypeError, ValueError):
            archive_bytes = 0
        if not archive_path:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_archive_path_missing"
            )
        if archive_path.startswith("/"):
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_archive_path_absolute"
            )
        if not _SHA256_HEX_RE.fullmatch(archive_sha256):
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_archive_sha_invalid"
            )
        if archive_bytes <= 0:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_archive_bytes_invalid"
            )
        if archive_path:
            archive_file = repo_root / archive_path
            if not archive_file.is_file():
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_archive_file_missing"
                )
            else:
                observed_archive_bytes = archive_file.stat().st_size
                observed_archive_sha256 = hashlib.sha256(
                    archive_file.read_bytes()
                ).hexdigest()
                if observed_archive_bytes != archive_bytes:
                    blockers.append(
                        "l5_v2_tt5l_materialized_paired_work_unit_archive_bytes_mismatch"
                    )
                if observed_archive_sha256 != archive_sha256:
                    blockers.append(
                        "l5_v2_tt5l_materialized_paired_work_unit_archive_sha_mismatch"
                    )
                sideinfo_stats = _inspect_tt5l_archive_sideinfo(archive_file)
                if sideinfo_stats["valid"] is not True:
                    blockers.append(
                        "l5_v2_tt5l_materialized_paired_work_unit_tt5l_sideinfo_invalid"
                    )
                elif int(sideinfo_stats["num_pairs"]) != _TT5L_CONTEST_N_PAIRS:
                    blockers.append(
                        "l5_v2_tt5l_materialized_paired_work_unit_tt5l_num_pairs_not_full_contest"
                    )
                elif int(sideinfo_stats["total_values"]) <= 0:
                    blockers.append(
                        "l5_v2_tt5l_materialized_paired_work_unit_tt5l_sideinfo_empty"
                    )
                elif int(sideinfo_stats["nonzero_values"]) <= 0:
                    blockers.append(
                        "l5_v2_tt5l_materialized_paired_work_unit_tt5l_sideinfo_all_zero"
                    )

    runtime = payload.get("runtime") if payload else None
    submission_dir = ""
    runtime_tree_by_axis: Mapping[str, Any] = {}
    runtime_content_by_axis: Mapping[str, Any] = {}
    if payload and not isinstance(runtime, Mapping):
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_runtime_missing")
    elif isinstance(runtime, Mapping):
        submission_dir = str(runtime.get("submission_dir") or "").strip()
        if not submission_dir:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_submission_dir_missing"
            )
        if submission_dir.startswith("/"):
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_submission_dir_absolute"
            )
        if submission_dir:
            runtime_dir = repo_root / submission_dir
            if not runtime_dir.is_dir():
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_submission_dir_missing_on_disk"
                )
            elif not (runtime_dir / "inflate.sh").is_file():
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_inflate_sh_missing"
                )
        trees = runtime.get("expected_runtime_tree_sha256_by_axis")
        contents = runtime.get("expected_runtime_content_tree_sha256_by_axis")
        if isinstance(trees, Mapping):
            runtime_tree_by_axis = trees
        else:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_runtime_tree_by_axis_missing"
            )
        if isinstance(contents, Mapping):
            runtime_content_by_axis = contents
        else:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_runtime_content_by_axis_missing"
            )
        for axis in ("contest_cpu", "contest_cuda"):
            if not _SHA256_HEX_RE.fullmatch(str(runtime_tree_by_axis.get(axis) or "")):
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_runtime_tree_missing:"
                    + axis
                )
            if not _SHA256_HEX_RE.fullmatch(
                str(runtime_content_by_axis.get(axis) or "")
            ):
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_runtime_content_missing:"
                    + axis
                )
        runtime_content_values = [
            str(runtime_content_by_axis.get(axis) or "")
            for axis in ("contest_cpu", "contest_cuda")
        ]
        if all(_SHA256_HEX_RE.fullmatch(value) for value in runtime_content_values) and (
            len(set(runtime_content_values)) != 1
        ):
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_runtime_content_axis_mismatch"
            )

    required_axes = payload.get("required_axes") if payload else None
    if isinstance(required_axes, list):
        missing = [axis for axis in ("contest_cpu", "contest_cuda") if axis not in required_axes]
        if missing:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_required_axes_missing:"
                + ",".join(missing)
            )
    elif payload:
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_required_axes_missing")

    commands = payload.get("commands") if payload else None
    if not isinstance(commands, Mapping):
        commands = {}
        if payload:
            blockers.append("l5_v2_tt5l_materialized_paired_work_unit_commands_missing")
    lanes = payload.get("lanes") if payload else None
    if not isinstance(lanes, Mapping):
        lanes = {}
        if payload:
            blockers.append("l5_v2_tt5l_materialized_paired_work_unit_lanes_missing")
    for axis, expected_lane in TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANES.items():
        if str(lanes.get(axis) or "") != expected_lane:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_lane_mismatch:" + axis
            )

    run_id = str(payload.get("run_id") or "").strip() if payload else ""
    output_root = str(payload.get("output_root") or "").strip() if payload else ""
    if payload and not run_id:
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_run_id_missing")
    if payload and not output_root:
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_output_root_missing")
    outputs = payload.get("outputs") if payload else None
    output_by_axis: dict[str, str] = {}
    if not isinstance(outputs, Mapping):
        if payload:
            blockers.append("l5_v2_tt5l_materialized_paired_work_unit_outputs_missing")
    else:
        for axis in ("contest_cpu", "contest_cuda"):
            output_dir = str(outputs.get(axis) or "").strip()
            output_by_axis[axis] = output_dir
            if not output_dir:
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_output_dir_missing:"
                    + axis
                )
                continue
            if output_dir.startswith("/"):
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_output_dir_absolute:"
                    + axis
                )
                continue
            output_result = repo_root / output_dir / "contest_auth_eval.json"
            if output_result.is_file():
                try:
                    output_payload = json.loads(output_result.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    blockers.append(
                        "l5_v2_tt5l_materialized_paired_work_unit_output_collision_unreadable:"
                        + axis
                    )
                else:
                    if not isinstance(output_payload, Mapping):
                        blockers.append(
                            "l5_v2_tt5l_materialized_paired_work_unit_output_collision_not_object:"
                            + axis
                        )
                    else:
                        existing_archive_sha = extract_archive_sha256(output_payload)
                        if existing_archive_sha and existing_archive_sha != archive_sha256:
                            blockers.append(
                                "l5_v2_tt5l_materialized_paired_work_unit_output_collision_archive_mismatch:"
                                + axis
                            )

    def _flag_value(command: list[object], flag: str) -> str:
        try:
            index = command.index(flag)
        except ValueError:
            return ""
        next_index = index + 1
        if next_index >= len(command):
            return ""
        return str(command[next_index])

    for axis in ("contest_cpu", "contest_cuda"):
        command = commands.get(axis)
        if not isinstance(command, list) or not command:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_axis_command_missing:" + axis
            )
            continue
        expected_script = (
            "experiments/modal_auth_eval_cpu.py"
            if axis == "contest_cpu"
            else "experiments/modal_auth_eval.py"
        )
        expected_lane = TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANES[axis]
        expected_runtime_sha = str(runtime_tree_by_axis.get(axis) or "")
        if ".venv/bin/modal" not in command or "run" not in command:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_axis_command_not_modal_run:"
                + axis
            )
        if expected_script not in command:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_axis_command_wrong_wrapper:"
                + axis
            )
        if "--detach" not in command or "--provider-detach-ack" not in command:
            blockers.append(
                "l5_v2_tt5l_materialized_paired_work_unit_axis_command_detach_missing:"
                + axis
            )
        expected_values = {
            "--archive": archive_path,
            "--expected-archive-sha256": archive_sha256,
            "--submission-dir": submission_dir,
            "--pair-group-id": TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID,
            "--lane-id": expected_lane,
            "--expected-runtime-tree-sha256": expected_runtime_sha,
        }
        if run_id:
            expected_values["--instance-job-id"] = f"{run_id}_{'cpu' if axis == 'contest_cpu' else 'cuda'}"
        if output_by_axis.get(axis):
            expected_values["--output-dir"] = output_by_axis[axis]
        for flag, expected_value in expected_values.items():
            if _flag_value(command, flag) != expected_value:
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_axis_command_flag_mismatch:"
                    + axis
                    + ":"
                    + flag.lstrip("-").replace("-", "_")
                )

    axes_skipped = payload.get("axes_skipped_due_to_existing_anchor") if payload else None
    existing_anchors = payload.get("existing_anchors_reused") if payload else None
    if isinstance(axes_skipped, Mapping):
        for axis in ("contest_cpu", "contest_cuda"):
            if axes_skipped.get(axis) is not False:
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_axis_skip_unexpected:"
                    + axis
                )
    elif payload:
        blockers.append("l5_v2_tt5l_materialized_paired_work_unit_axis_skip_missing")
    if isinstance(existing_anchors, Mapping):
        for axis in ("contest_cpu", "contest_cuda"):
            if existing_anchors.get(axis) is not None:
                blockers.append(
                    "l5_v2_tt5l_materialized_paired_work_unit_anchor_reuse_unexpected:"
                    + axis
                )

    operator_plan_parts = [
        ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
        f"--archive {archive_path} "
        f"--submission-dir {submission_dir} "
        "--inflate-sh inflate.sh "
        "--label l5_v2_time_traveler_l5_autonomy "
        f"--expected-archive-sha256 {archive_sha256} "
        f"--run-id {run_id or '<missing_run_id>'} "
        "--pair-group-id pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda "
        "--lane-id-base lane_l5_v2_measure_tt5l_autonomy_paired_exact "
        f"--output-root {output_root or '<missing_output_root>'} "
        "--modal-bin .venv/bin/modal "
        "--gpu T4 "
        "--claim-agent codex:l5_v2_paired_measurement_dispatch "
        "--claim-notes "
        "l5_v2_paired_measurement:pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda "
        "--expected-runtime-tree-sha256 auto"
    ]
    if payload and payload.get("skip_axis_if_promotable_anchor_exists") is True:
        operator_plan_parts.append(" --skip-axis-if-promotable-anchor-exists")
    operator_plan_command = "".join(operator_plan_parts)
    provider_blocker_status = _tt5l_materialized_modal_provider_blocker_status(
        repo_root=repo_root,
        archive_sha256=archive_sha256,
        pair_group_id=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID,
    )
    alternate_provider_plan_status = _tt5l_materialized_lightning_alt_provider_plan_status(
        repo_root=repo_root,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        pair_group_id=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID,
        runtime_path=submission_dir,
    )

    suppress_execute_template = provider_blocker_status["active"] is True or (
        provider_blocker_status["artifact_exists"] is True
        and provider_blocker_status["artifact_valid"] is not True
    )
    status: dict[str, Any] = {
        "schema": "l5_v2_tt5l_materialized_paired_work_unit_status_v1",
        "artifact_path": TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": not blockers,
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "submission_dir": submission_dir,
        "runtime_tree_sha256_by_axis": {
            axis: str(runtime_tree_by_axis.get(axis) or "")
            for axis in ("contest_cpu", "contest_cuda")
        },
        "runtime_content_tree_sha256_by_axis": {
            axis: str(runtime_content_by_axis.get(axis) or "")
            for axis in ("contest_cpu", "contest_cuda")
        },
        "tt5l_sideinfo_stats": sideinfo_stats,
        "operator_plan_command_template": operator_plan_command,
        "provider_blocker_status": provider_blocker_status,
        "alternate_provider_plan_status": alternate_provider_plan_status,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "blockers": list(dict.fromkeys(blockers)),
    }
    if not suppress_execute_template:
        status["operator_execute_command_template_after_review"] = (
            operator_plan_command + " --execute"
        )
    return status


def _tt5l_materialized_lightning_alt_provider_plan_status(
    *,
    repo_root: Path,
    archive_path: str,
    archive_sha256: str,
    pair_group_id: str,
    runtime_path: str,
) -> dict[str, Any]:
    artifact_path = (
        repo_root / TT5L_MATERIALIZED_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH
    )
    validation_blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        return {
            "schema": "l5_v2_tt5l_materialized_lightning_alt_provider_plan_status_v1",
            "artifact_path": TT5L_MATERIALIZED_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH,
            "artifact_exists": False,
            "artifact_valid": False,
            "provider": "",
            "execution_ready": False,
            "local_supply_chain_ok": False,
            "exact_eval_dry_run_ok": False,
            "source_manifest_probe_current": False,
            "command_sha256": "",
            "job_name": "",
            "execution_blockers": [],
            "blockers": [],
        }
    try:
        loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        validation_blockers.append("l5_v2_tt5l_lightning_alt_provider_json_invalid")
    else:
        if isinstance(loaded, Mapping):
            payload = loaded
        else:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_alt_provider_not_object"
            )
    if payload and payload.get("schema") != (
        "l5_v2_tt5l_lightning_alternate_provider_plan_v1"
    ):
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_schema_mismatch"
        )
    provider = str(payload.get("provider") or "").strip() if payload else ""
    payload_archive_sha = str(payload.get("archive_sha256") or "").strip() if payload else ""
    payload_pair_group_id = str(payload.get("pair_group_id") or "").strip() if payload else ""
    if payload and provider != "lightning":
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_provider_not_lightning"
        )
    if payload and payload_archive_sha != archive_sha256:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_archive_sha_mismatch"
        )
    if payload and payload_pair_group_id != pair_group_id:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_pair_group_mismatch"
        )
    if payload and payload.get("score_claim") is not False:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_score_claim_not_false"
        )
    if payload and payload.get("promotion_eligible") is not False:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_promotion_not_false"
        )
    if payload and payload.get("ready_for_exact_eval_dispatch") is not False:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_exact_dispatch_not_false"
        )
    source_manifest_probe = payload.get("source_manifest_probe") if payload else None
    source_manifest_probe_current = False
    if payload and not isinstance(source_manifest_probe, Mapping):
        validation_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_source_manifest_probe_missing"
        )
    elif isinstance(source_manifest_probe, Mapping):
        source_manifest_command = str(source_manifest_probe.get("command") or "")
        source_manifest_status_basis = str(
            source_manifest_probe.get("status_basis") or ""
        )
        if not str(source_manifest_probe.get("observed_at_utc") or "").strip():
            validation_blockers.append(
                "l5_v2_tt5l_lightning_alt_provider_source_manifest_probe_observed_at_missing"
            )
        if f"--artifact {archive_path}" not in source_manifest_command:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_alt_provider_source_manifest_archive_mismatch"
            )
        if f"--artifact {runtime_path}" not in source_manifest_command:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_alt_provider_source_manifest_runtime_mismatch"
            )
        if (
            "rerun_after_archive_refresh_against_current_archive_and_runtime"
            not in source_manifest_status_basis
        ):
            validation_blockers.append(
                "l5_v2_tt5l_lightning_alt_provider_source_manifest_probe_stale"
            )
        source_manifest_probe_current = not any(
            blocker.startswith(
                "l5_v2_tt5l_lightning_alt_provider_source_manifest_"
            )
            for blocker in validation_blockers
        )

    doctor = payload.get("local_lightning_doctor") if payload else None
    local_supply_chain_ok = (
        isinstance(doctor, Mapping)
        and doctor.get("status") == "OK"
        and doctor.get("local_supply_chain_ok") is True
    )
    dry_run = payload.get("exact_eval_dry_run") if payload else None
    command_sha256 = ""
    job_name = ""
    exact_eval_dry_run_ok = False
    if isinstance(dry_run, Mapping):
        command_sha256 = str(dry_run.get("command_sha256") or "").strip()
        job_name = str(dry_run.get("job_name") or "").strip()
        exact_eval_dry_run_ok = (
            dry_run.get("status") == "DRY_RUN"
            and _SHA256_HEX_RE.fullmatch(command_sha256) is not None
            and bool(job_name)
        )
    execution_blockers = [
        str(blocker)
        for blocker in (payload.get("execution_blockers") if payload else []) or []
        if str(blocker)
    ]
    readiness_blockers: list[str] = []
    if payload and not local_supply_chain_ok:
        readiness_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_local_supply_chain_not_ok"
        )
    if payload and not exact_eval_dry_run_ok:
        readiness_blockers.append(
            "l5_v2_tt5l_lightning_alt_provider_exact_eval_dry_run_missing"
        )
    readiness_blockers.extend(
        "l5_v2_tt5l_lightning_alt_provider_blocked:" + blocker
        for blocker in execution_blockers
    )
    artifact_valid = bool(payload) and not validation_blockers
    execution_ready = artifact_valid and not readiness_blockers
    return {
        "schema": "l5_v2_tt5l_materialized_lightning_alt_provider_plan_status_v1",
        "artifact_path": TT5L_MATERIALIZED_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": artifact_valid,
        "provider": provider,
        "execution_ready": execution_ready,
        "local_supply_chain_ok": local_supply_chain_ok,
        "exact_eval_dry_run_ok": exact_eval_dry_run_ok,
        "source_manifest_probe_current": source_manifest_probe_current,
        "command_sha256": command_sha256,
        "job_name": job_name,
        "execution_blockers": list(dict.fromkeys(execution_blockers)),
        "blockers": list(
            dict.fromkeys(validation_blockers + readiness_blockers)
        ),
    }


def _tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_status(
    *,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_path = (
        repo_root / TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
    )
    validation_blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        return {
            "schema": (
                "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_status_v1"
            ),
            "artifact_path": (
                TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
            ),
            "artifact_exists": False,
            "artifact_valid": False,
            "provider": "lightning",
            "cell_count": 0,
            "expected_cell_count": (
                len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
                * len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
            ),
            "covered_axes": [],
            "covered_variants": [],
            "all_cells_dry_run_structurally_valid": False,
            "all_cells_dry_run_ready": False,
            "execution_ready": False,
            "source_commit": "",
            "current_head_commit": "",
            "source_commit_matches_head": False,
            "source_commit_is_ancestor": False,
            "source_relevant_paths": [],
            "source_relevant_diff_paths": [],
            "source_relevant_paths_match": False,
            "source_custody_valid": False,
            "source_custody_current_for_execution": False,
            "blockers": [],
        }
    try:
        loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        validation_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_json_invalid"
        )
    else:
        if isinstance(loaded, Mapping):
            payload = loaded
        else:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_not_object"
            )

    if payload and payload.get("schema") != (
        TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA
    ):
        validation_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_schema_mismatch"
        )
    false_fields = (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "ready_for_provider_dispatch",
        "dispatch_attempted",
    )
    for field in false_fields:
        if payload and payload.get(field) is not False:
            validation_blockers.append(
                f"l5_v2_tt5l_lightning_paired_axis_plan_{field}_not_false"
            )
    if payload and payload.get("all_cells_dry_run_ready") is not True:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_cells_not_dry_run_ready"
        )

    cells = _mapping_items(payload.get("cells") if payload else None)
    expected_cell_count = (
        len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
        * len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
    )
    if payload and len(cells) != expected_cell_count:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_cell_count_mismatch"
        )
    cell_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_keys: list[str] = []
    for cell in cells:
        variant = str(cell.get("variant") or "").strip()
        axis = str(cell.get("axis") or "").strip()
        key = (variant, axis)
        if key in cell_by_key:
            duplicate_keys.append(f"{variant}:{axis}")
        elif variant and axis:
            cell_by_key[key] = cell
    if duplicate_keys:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_duplicate_cells:"
            + ",".join(sorted(duplicate_keys))
        )
    missing_cells = [
        f"{variant}:{axis}"
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        if (variant, axis) not in cell_by_key
    ]
    if missing_cells:
        validation_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_missing_cells:"
            + ",".join(missing_cells)
        )

    for (variant, axis), cell in cell_by_key.items():
        expected_device = "cpu" if axis == "contest_cpu" else "cuda"
        expected_role = f"exact_{expected_device}_eval"
        pair_group_id = str(cell.get("pair_group_id") or "").strip()
        run_id = str(cell.get("run_id") or "").strip()
        if not pair_group_id:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_pair_group_id_missing:"
                f"{variant}:{axis}"
            )
        if not run_id:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_run_id_missing:"
                f"{variant}:{axis}"
            )
        if cell.get("role") != expected_role:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_role_mismatch:"
                f"{variant}:{axis}"
            )
        if cell.get("required_device") != expected_device:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_device_mismatch:"
                f"{variant}:{axis}"
            )
        if cell.get("ready_for_operator_dispatch") is not True:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_cell_not_ready:"
                f"{variant}:{axis}"
            )
        if cell.get("ready_for_provider_dispatch") is not False:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_cell_provider_ready:"
                f"{variant}:{axis}"
            )
        for sha_field in (
            "archive_sha256",
            "command_sha256",
            "state_sha256",
            "dry_run_stdout_sha256",
            "dry_run_stderr_sha256",
        ):
            sha_value = str(cell.get(sha_field) or "").strip().lower()
            if not _SHA256_HEX_RE.fullmatch(sha_value):
                validation_blockers.append(
                    "l5_v2_tt5l_lightning_paired_axis_plan_cell_sha_invalid:"
                    f"{variant}:{axis}:{sha_field}"
                )
        invariants = cell.get("invariants")
        if not isinstance(invariants, Mapping) or not invariants:
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_invariants_missing:"
                f"{variant}:{axis}"
            )
        elif any(value is not True for value in invariants.values()):
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_invariants_failed:"
                f"{variant}:{axis}"
            )
        spec = cell.get("spec")
        if not isinstance(spec, Mapping):
            validation_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_spec_missing:"
                f"{variant}:{axis}"
            )
        else:
            if spec.get("role") != expected_role:
                validation_blockers.append(
                    "l5_v2_tt5l_lightning_paired_axis_plan_spec_role_mismatch:"
                    f"{variant}:{axis}"
                )
            queue_metadata = spec.get("queue_metadata")
            if not isinstance(queue_metadata, Mapping):
                validation_blockers.append(
                    "l5_v2_tt5l_lightning_paired_axis_plan_spec_queue_metadata_missing:"
                    f"{variant}:{axis}"
                )
            else:
                if queue_metadata.get("pair_group_id") != pair_group_id:
                    validation_blockers.append(
                        "l5_v2_tt5l_lightning_paired_axis_plan_"
                        f"spec_pair_group_id_mismatch:{variant}:{axis}"
                    )
                if queue_metadata.get("run_id") != run_id:
                    validation_blockers.append(
                        "l5_v2_tt5l_lightning_paired_axis_plan_"
                        f"spec_run_id_mismatch:{variant}:{axis}"
                    )
            adjudication = spec.get("adjudication")
            if (
                not isinstance(adjudication, Mapping)
                or adjudication.get("required_device") != expected_device
            ):
                validation_blockers.append(
                    "l5_v2_tt5l_lightning_paired_axis_plan_spec_device_mismatch:"
                    f"{variant}:{axis}"
                )

    covered_axes = sorted({axis for _, axis in cell_by_key})
    covered_variants = sorted({variant for variant, _ in cell_by_key})
    top_blockers = [
        str(blocker)
        for blocker in (payload.get("blockers") if payload else []) or []
        if str(blocker)
    ]
    source_commit = str(payload.get("source_commit") or "").strip() if payload else ""
    current_head_commit = _git_head_commit(repo_root) if payload else ""
    source_commit_matches_head = bool(
        source_commit
        and current_head_commit
        and source_commit == current_head_commit
    )
    source_relevant_paths = [
        path
        for path in dict.fromkeys(
            [
                *_TT5L_LIGHTNING_PAIRED_AXIS_STATIC_SOURCE_PATHS,
                str(payload.get("source_variant_manifest") or "").strip(),
                str(payload.get("source_dispatch_plan") or "").strip(),
            ]
        )
        if path
    ]
    source_commit_is_ancestor = source_commit_matches_head or (
        _git_is_ancestor(repo_root, source_commit, current_head_commit)
        if payload and _GIT_COMMIT_HEX_RE.fullmatch(source_commit)
        else False
    )
    source_relevant_diff_paths = (
        list(
            _git_diff_name_only(
                repo_root,
                source_commit,
                current_head_commit,
                source_relevant_paths,
            )
        )
        if payload and source_commit_is_ancestor
        else []
    )
    source_relevant_paths_match = (
        bool(payload)
        and bool(current_head_commit)
        and source_commit_is_ancestor
        and not source_relevant_diff_paths
    )
    source_custody_current_for_execution = (
        bool(payload)
        and not validation_blockers
        and source_relevant_paths_match
    )
    dry_run_structurally_valid = (
        bool(payload)
        and not validation_blockers
        and payload.get("all_cells_dry_run_ready") is True
    )
    dry_run_ready = dry_run_structurally_valid and source_custody_current_for_execution
    execution_blockers = [
        "l5_v2_tt5l_lightning_paired_axis_plan_dry_run_only_no_provider_job_launched",
        *(
            "l5_v2_tt5l_lightning_paired_axis_plan_blocked:" + blocker
            for blocker in top_blockers
        ),
    ]
    if payload and not _GIT_COMMIT_HEX_RE.fullmatch(source_commit):
        execution_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_source_commit_missing_or_invalid"
        )
    elif payload and not current_head_commit:
        execution_blockers.append(
            "l5_v2_tt5l_lightning_paired_axis_plan_current_head_unknown"
        )
    elif payload and source_commit != current_head_commit:
        if not source_commit_is_ancestor:
            execution_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_source_commit_not_head_ancestor"
            )
        elif source_relevant_diff_paths:
            execution_blockers.append(
                "l5_v2_tt5l_lightning_paired_axis_plan_source_relevant_paths_changed"
            )
    return {
        "schema": (
            "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_status_v1"
        ),
        "artifact_path": (
            TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
        ),
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": bool(payload) and not validation_blockers,
        "provider": "lightning",
        "cell_count": len(cells),
        "expected_cell_count": expected_cell_count,
        "covered_axes": covered_axes,
        "covered_variants": covered_variants,
        "all_cells_dry_run_structurally_valid": dry_run_structurally_valid,
        "all_cells_dry_run_ready": dry_run_ready,
        "execution_ready": False,
        "source_commit": source_commit,
        "current_head_commit": current_head_commit,
        "source_commit_matches_head": source_commit_matches_head,
        "source_commit_is_ancestor": source_commit_is_ancestor,
        "source_relevant_paths": source_relevant_paths,
        "source_relevant_diff_paths": source_relevant_diff_paths,
        "source_relevant_paths_match": source_relevant_paths_match,
        "source_custody_valid": source_custody_current_for_execution,
        "source_custody_current_for_execution": source_custody_current_for_execution,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "execution_blockers": list(dict.fromkeys(execution_blockers)),
        "blockers": list(dict.fromkeys(validation_blockers + execution_blockers)),
    }


def _tt5l_materialized_modal_provider_blocker_status(
    *,
    repo_root: Path,
    archive_sha256: str,
    pair_group_id: str,
) -> dict[str, Any]:
    artifact_path = repo_root / TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not artifact_path.is_file():
        return {
            "schema": "l5_v2_tt5l_materialized_modal_provider_blocker_status_v1",
            "artifact_path": TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH,
            "artifact_exists": False,
            "artifact_valid": False,
            "active": False,
            "provider": "",
            "failure_class": "",
            "resolved": False,
            "blocker": "",
            "blockers": [],
        }
    try:
        loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append("l5_v2_tt5l_modal_provider_blocker_json_invalid")
    else:
        if isinstance(loaded, Mapping):
            payload = loaded
        else:
            blockers.append("l5_v2_tt5l_modal_provider_blocker_not_object")
    if payload and payload.get("schema") != "l5_v2_tt5l_materialized_provider_blocker_v1":
        blockers.append("l5_v2_tt5l_modal_provider_blocker_schema_mismatch")
    provider = str(payload.get("provider") or "").strip() if payload else ""
    failure_class = str(payload.get("failure_class") or "").strip() if payload else ""
    payload_archive_sha = str(payload.get("archive_sha256") or "").strip() if payload else ""
    payload_pair_group_id = str(payload.get("pair_group_id") or "").strip() if payload else ""
    resolved = payload.get("resolved") is True if payload else False
    if payload and provider != "modal":
        blockers.append("l5_v2_tt5l_modal_provider_blocker_provider_not_modal")
    if payload and payload_archive_sha != archive_sha256:
        blockers.append("l5_v2_tt5l_modal_provider_blocker_archive_sha_mismatch")
    if payload and payload_pair_group_id != pair_group_id:
        blockers.append("l5_v2_tt5l_modal_provider_blocker_pair_group_mismatch")
    if payload and payload.get("score_claim") is not False:
        blockers.append("l5_v2_tt5l_modal_provider_blocker_score_claim_not_false")
    if payload and payload.get("promotion_eligible") is not False:
        blockers.append("l5_v2_tt5l_modal_provider_blocker_promotion_not_false")
    active = (
        bool(payload)
        and not blockers
        and not resolved
        and failure_class == "modal_workspace_billing_cycle_spend_limit_reached"
    )
    blocker = (
        "l5_v2_tt5l_modal_provider_blocker_active:"
        "modal_workspace_billing_cycle_spend_limit_reached"
        if active
        else ""
    )
    return {
        "schema": "l5_v2_tt5l_materialized_modal_provider_blocker_status_v1",
        "artifact_path": TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH,
        "artifact_exists": artifact_path.is_file(),
        "artifact_valid": bool(payload) and not blockers,
        "active": active,
        "provider": provider,
        "failure_class": failure_class,
        "resolved": resolved,
        "blocker": blocker,
        "blockers": list(dict.fromkeys(blockers + ([blocker] if blocker else []))),
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
    probe_template_exists = (
        repo_root / TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    ).is_file()
    dykstra_status = _tt5l_dykstra_feasibility_status(repo_root=repo_root)
    dykstra_valid = dykstra_status["artifact_valid"] is True
    move_level_status = _tt5l_move_level_feasibility_status(repo_root=repo_root)
    move_level_valid = move_level_status["artifact_valid"] is True
    timing_smoke_status = _tt5l_first_anchor_timing_smoke_status(repo_root=repo_root)
    timing_smoke_valid = timing_smoke_status["artifact_valid"] is True
    probe_gate_artifact_status = l5_v2_probe_gate_artifact_status(repo_root=repo_root)
    probe_intake_status = _tt5l_probe_observation_intake_status(repo_root=repo_root)
    materialized_work_unit_status = _tt5l_materialized_paired_work_unit_status(
        repo_root=repo_root
    )
    lightning_paired_axis_plan_status = (
        _tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_status(
            repo_root=repo_root
        )
    )
    sideinfo_effect_curve_dispatch_plan_status = (
        _tt5l_sideinfo_effect_curve_dispatch_plan_status(repo_root=repo_root)
    )
    sideinfo_effect_curve_harvest_cells_status = (
        _tt5l_sideinfo_effect_curve_harvest_cells_status(repo_root=repo_root)
    )
    sideinfo_effect_curve_status = _tt5l_sideinfo_effect_curve_status(
        repo_root=repo_root
    )
    sideinfo_effect_curve_valid = (
        sideinfo_effect_curve_status["artifact_valid"] is True
    )
    sideinfo_effect_curve_allowed = dykstra_valid and move_level_valid and sideinfo_valid
    first_anchor_timing_smoke_allowed = (
        sideinfo_effect_curve_allowed
        and sideinfo_effect_curve_valid
        and probe_valid
        and paired_axis_plan_valid
    )
    architecture_lock_allowed = (
        sideinfo_effect_curve_allowed
        and sideinfo_effect_curve_valid
        and probe_valid
        and paired_axis_plan_valid
        and timing_smoke_valid
        and anchor_pair_valid
    )

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
    if (
        probe_gate_artifact_status["artifact_exists"] is True
        and probe_gate_artifact_status["artifact_valid"] is not True
    ):
        blockers.extend(str(blocker) for blocker in probe_gate_artifact_status["blockers"])
    blockers.extend(str(blocker) for blocker in dykstra_status["blockers"])
    if dykstra_valid:
        blockers.extend(str(blocker) for blocker in move_level_status["blockers"])
    if sideinfo_effect_curve_allowed and probe_valid and paired_axis_plan_valid:
        blockers.extend(
            str(blocker) for blocker in sideinfo_effect_curve_status["blockers"]
        )
        if sideinfo_effect_curve_valid:
            blockers.extend(str(blocker) for blocker in timing_smoke_status["blockers"])
    provider_blocker_status = materialized_work_unit_status["provider_blocker_status"]
    provider_blocker_invalid = (
        provider_blocker_status["artifact_exists"] is True
        and provider_blocker_status["artifact_valid"] is not True
    )
    if provider_blocker_invalid:
        blockers.extend(str(blocker) for blocker in provider_blocker_status["blockers"])
    if provider_blocker_status["active"] is True:
        blockers.extend(str(blocker) for blocker in provider_blocker_status["blockers"])
        alternate_plan = materialized_work_unit_status[
            "alternate_provider_plan_status"
        ]
        if alternate_plan["artifact_valid"] is True:
            blockers.extend(str(blocker) for blocker in alternate_plan["blockers"])
    if (
        lightning_paired_axis_plan_status["artifact_exists"] is True
        and lightning_paired_axis_plan_status["artifact_valid"] is not True
    ):
        blockers.extend(
            str(blocker)
            for blocker in lightning_paired_axis_plan_status["blockers"]
        )
    if (
        sideinfo_effect_curve_dispatch_plan_status["artifact_exists"] is True
        and sideinfo_effect_curve_dispatch_plan_status["artifact_valid"] is not True
    ):
        blockers.extend(
            str(blocker)
            for blocker in sideinfo_effect_curve_dispatch_plan_status["blockers"]
        )
    if (
        sideinfo_effect_curve_harvest_cells_status["artifact_exists"] is True
        and sideinfo_effect_curve_harvest_cells_status["artifact_valid"] is not True
    ):
        blockers.extend(
            str(blocker)
            for blocker in sideinfo_effect_curve_harvest_cells_status["blockers"]
        )

    if not dykstra_valid:
        next_action = {
            "action_id": "run_tt5l_dykstra_score_axis_sanity",
            "phase": "cargo_cult_unwind_feasibility",
            "command_template": (
                f".venv/bin/python {TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH} "
                f"--substrate-id {TT5L_DYKSTRA_SUBSTRATE_ID} "
                "--predicted-band-lo <score_axis_lower_bound> "
                "--predicted-band-hi <score_axis_upper_bound> "
                "--archive-size-bytes <tt5l_target_or_candidate_archive_bytes> "
                "--tt5l-five-move-polytope "
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
    elif not move_level_valid:
        next_action = {
            "action_id": "materialize_tt5l_move_level_feasibility_proof",
            "phase": "cargo_cult_unwind_move_level_feasibility",
            "tool_path": TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH,
            "command_template": (
                f".venv/bin/python {TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH} "
                "--proof-artifact "
                "experiments/results/time_traveler_l5_v2/"
                "tt5l_move_level_solver_proof.json "
                "--proof-command-argv-json '<json-array-from-solver-run>' "
                f"--output-json {TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH}"
            ),
            "artifact_path": TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH,
            "predicate_id": TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
            "required_constraint_set_ids": sorted(TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS),
            "expected_artifacts": [TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH],
            "score_axis_sanity_artifact_path": TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
            "score_axis_sanity_is_not_move_level_proof": True,
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
                "--baseline-inflate-provenance <baseline_inflate_provenance.json> "
                "--mutated-inflate-provenance <mutated_inflate_provenance.json> "
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
    elif not probe_valid and not probe_template_exists:
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
    elif not probe_valid and not probe_intake_status[
        "artifact_valid_for_measurement_planning"
    ]:
        measurement_schedule_command = (
            f".venv/bin/python {L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH} "
            f"--probe-intake-json {TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH} "
            f"--output-json {L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH} "
            f"--output-md {L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH}"
        )
        next_action = {
            "action_id": "populate_and_evaluate_c1_z5_tt5l_probe_observations",
            "phase": "probe_disambiguator",
            "probe_status": "observation_intake_required",
            "input_template": TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH,
            "command_template": (
                f".venv/bin/python {L5V2_PROBE_OBSERVATION_INTAKE_TOOL_PATH} "
                f"--output-json {TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH} "
                f"--output-md {TT5L_PROBE_OBSERVATION_INTAKE_REPORT_PATH} "
                f"--probe-gate-out {TT5L_PROBE_GATE_ARTIFACT_PATH}"
            ),
            "expected_artifacts": [
                TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH,
                TT5L_PROBE_OBSERVATION_INTAKE_REPORT_PATH,
                TT5L_PROBE_GATE_ARTIFACT_PATH,
            ],
            "measurement_schedule_tool_path": L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH,
            "measurement_schedule_command_template": measurement_schedule_command,
            "measurement_schedule_expected_artifacts": [
                L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
                L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH,
            ],
            "measurement_schedule_semantics": (
                "planning-only first-match lattice; routes the next paired "
                "C1/Z5/TT5L measurements without score, rank, promotion, or "
                "dispatch authority"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif (
        not probe_valid
        and materialized_work_unit_status["artifact_valid"] is True
        and provider_blocker_invalid
    ):
        next_action = {
            "action_id": "refresh_or_retire_l5_v2_tt5l_modal_provider_blocker",
            "phase": "probe_disambiguator_paired_measurements",
            "probe_status": "tt5l_work_unit_provider_blocker_artifact_stale",
            "materialized_work_unit_status": materialized_work_unit_status,
            "provider_blocker_status": provider_blocker_status,
            "execution_order": [
                "inspect_provider_blocker_archive_and_pair_group_against_current_work_unit",
                "regenerate_provider_blocker_for_current_archive_or_mark_historical",
                "rerun_l5_v2_architecture_lock_packet_before_any_execute_command",
            ],
            "score_lowering_unblocker": (
                "TT5L has a byte-closed work unit, but a stale or invalid "
                "provider-blocker artifact is present. Refusing to surface an "
                "execute command until provider state is refreshed or retired."
            ),
            "ready_for_operator_dispatch": False,
            "ready_for_provider_dispatch": False,
            "ready_for_alternate_provider_planning": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        }
    elif (
        not probe_valid
        and materialized_work_unit_status["artifact_valid"] is True
        and materialized_work_unit_status["provider_blocker_status"]["active"] is True
    ):
        next_action = {
            "action_id": (
                "resolve_l5_v2_tt5l_modal_provider_blocker_or_dispatch_alternate_provider"
            ),
            "phase": "probe_disambiguator_paired_measurements",
            "probe_status": "tt5l_work_unit_materialized_provider_blocked",
            "materialized_work_unit_status": materialized_work_unit_status,
            "provider_blocker_status": materialized_work_unit_status[
                "provider_blocker_status"
            ],
            "alternate_provider_plan_status": materialized_work_unit_status[
                "alternate_provider_plan_status"
            ],
            "operator_plan_command_template": materialized_work_unit_status[
                "operator_plan_command_template"
            ],
            "modal_execute_command_suppressed_until_provider_blocker_resolved": True,
            "execution_order": [
                "resolve_modal_workspace_billing_cycle_spend_limit_or_select_alternate_provider",
                "execute_canonical_paired_auth_eval_against_same_archive_and_runtime",
                "harvest_both_axes_through_provider_specific_recovery",
                "convert_returned_results_into_l5_v2_probe_observations",
            ],
            "score_lowering_unblocker": (
                "TT5L has a byte-closed archive/runtime work unit, but the "
                "Modal provider path is currently blocked before job spawn by "
                "workspace billing limit. This is provider capacity, not a "
                "method result."
            ),
            "ready_for_operator_dispatch": False,
            "ready_for_provider_dispatch": False,
            "ready_for_alternate_provider_planning": True,
            "ready_for_alternate_provider_dispatch": materialized_work_unit_status[
                "alternate_provider_plan_status"
            ]["execution_ready"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": True,
        }
    elif not probe_valid and materialized_work_unit_status["artifact_valid"] is True:
        next_action = {
            "action_id": "review_and_execute_l5_v2_tt5l_materialized_paired_measurement",
            "phase": "probe_disambiguator_paired_measurements",
            "probe_status": "tt5l_work_unit_materialized_fail_closed",
            "materialized_work_unit_status": materialized_work_unit_status,
            "operator_plan_command_template": materialized_work_unit_status[
                "operator_plan_command_template"
            ],
            "operator_execute_command_template_after_review": materialized_work_unit_status[
                "operator_execute_command_template_after_review"
            ],
            "execution_order": [
                "review_materialized_tt5l_archive_runtime_custody",
                "execute_canonical_paired_modal_auth_eval_command",
                "harvest_both_axes_through_recover_modal_auth_eval",
                "convert_returned_results_into_l5_v2_probe_observations",
            ],
            "score_lowering_unblocker": (
                "TT5L has a byte-closed archive/runtime work unit; the next "
                "material L5 v2 evidence is paired CPU/CUDA execution under "
                "one runtime contract"
            ),
            "ready_for_operator_dispatch": True,
            "ready_for_provider_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        }
    elif not probe_valid:
        measurement_schedule_command = (
            f".venv/bin/python {L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH} "
            f"--probe-intake-json {TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH} "
            f"--output-json {L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH} "
            f"--output-md {L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH}"
        )
        paired_dispatch_plan_command = (
            f".venv/bin/python {L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH} "
            f"--schedule-json {L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH} "
            f"--output-json {L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH} "
            f"--output-md {L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_REPORT_PATH}"
        )
        next_action = {
            "action_id": "materialize_l5_v2_paired_probe_measurements",
            "phase": "probe_disambiguator_paired_measurements",
            "probe_status": "observation_intake_present_fail_closed",
            "probe_observation_intake_status": probe_intake_status,
            "input_template": TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH,
            "probe_observation_intake_artifact_path": (
                TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
            ),
            "measurement_schedule_tool_path": L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH,
            "measurement_schedule_command_template": measurement_schedule_command,
            "measurement_schedule_expected_artifacts": [
                L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
                L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH,
            ],
            "paired_measurement_dispatch_plan_tool_path": (
                L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH
            ),
            "paired_measurement_dispatch_plan_command_template": (
                paired_dispatch_plan_command
            ),
            "paired_measurement_dispatch_plan_expected_artifacts": [
                L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH,
                L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_REPORT_PATH,
            ],
            "execution_order": [
                "build_l5_v2_lattice_measurement_schedule",
                "build_l5_v2_paired_measurement_dispatch_plan",
                "fill_each_work_unit_archive_runtime_sha_and_operator_execute_flag",
            ],
            "score_lowering_unblocker": (
                "paired exact C1/Z5/TT5L observations are the next material "
                "L5 v2 staircase evidence; re-running intake without new source "
                "artifacts is a retread"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif not paired_axis_plan_valid:
        pair_group_id = f"{LANE_ID}_paired_cpu_cuda_first_anchor"
        next_action = {
            "action_id": "prepare_tt5l_paired_cpu_cuda_axis_plan",
            "phase": "paired_axis_plan",
            "recipe_path": TT5L_MODAL_A100_DISPATCH_RECIPE_PATH,
            "claim_lane_before_dispatch": True,
            "terminal_claim_required": True,
            "paired_dispatch_tool": "tools/dispatch_modal_paired_auth_eval.py",
            "claim_lifecycle_owner": (
                "tools/dispatch_modal_paired_auth_eval.py plus per-axis Modal "
                "auth-eval wrappers"
            ),
            "preclaim_forbidden": True,
            "standalone_active_claim_command": None,
            "lane_id": LANE_ID,
            "pair_group_id": pair_group_id,
            "required_axes": list(_REQUIRED_EXACT_AXES),
            "per_axis_job_id_fields": {
                "contest_cpu": "contest_cpu_job_id",
                "contest_cuda": "contest_cuda_job_id",
            },
            "harvest_command_template": (
                ".venv/bin/python tools/recover_modal_auth_eval.py "
                f"--lane-id {LANE_ID} --pair-group-id {pair_group_id} "
                "--axis <contest_cpu|contest_cuda> --call-id <modal_call_id>"
            ),
            "terminal_claim_success_template": (
                ".venv/bin/python tools/claim_lane_dispatch.py claim --force "
                f"--lane-id {LANE_ID} --status completed_paired_axis_plan "
                f"--notes {pair_group_id}:<cpu_job_id>:<cuda_job_id>"
            ),
            "terminal_claim_failure_template": (
                ".venv/bin/python tools/claim_lane_dispatch.py claim --force "
                f"--lane-id {LANE_ID} --status failed_paired_axis_plan "
                f"--notes {pair_group_id}:<failure_class>:<job_id>"
            ),
            "command_template": (
                ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
                "--archive <byte_closed_archive.zip> "
                "--submission-dir <submission_runtime_dir> "
                "--expected-archive-sha256 <archive_sha256> "
                f"--lane-id-base {LANE_ID} "
                f"--pair-group-id {pair_group_id} "
                "--label l5_v2_tt5l_first_anchor "
                "--run-id <utc_run_id> "
                "--output-root experiments/results/l5_v2_paired_measurements "
                "--gpu A100 "
                "--claim-agent codex:l5_v2_paired_axis_plan "
                "--claim-notes tt5l_l5_v2_first_anchor "
                "--expected-runtime-tree-sha256 auto "
                "--json-out <paired_dispatch_plan.json> "
                "[--execute only after operator approval]"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif not sideinfo_effect_curve_valid:
        next_action = {
            "action_id": "measure_tt5l_sideinfo_effect_curve",
            "phase": "sideinfo_causal_effect_curve",
            "artifact_path": TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
            "harvest_cells_artifact_path": (
                TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH
            ),
            "harvest_cells_tool_path": (
                L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH
            ),
            "measurement_schedule_tool_path": L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH,
            "command_template": (
                "run paired CPU/CUDA TT5L side-info variants "
                "zero, random_lsb, shuffled, trained, and ablated into the "
                "Lightning plan local_artifact_dir paths; then "
                f".venv/bin/python {L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH} "
                f"--lightning-plan-json {TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH} "
                f"--output-json {TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH} "
                "--repo-root . "
                "&& "
                f".venv/bin/python {L5V2_SIDEINFO_EFFECT_CURVE_TOOL_PATH} "
                f"--cell-json {TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH} "
                f"--output-json {TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH} "
                "--repo-root . "
                "&& "
                f".venv/bin/python {L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH} "
                f"--probe-intake-json {TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH} "
                f"--sideinfo-effect-curve-json {TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH} "
                f"--output-json {L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH} "
                f"--output-md {L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH}"
            ),
            "expected_artifacts": [
                TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH,
                TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
                L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
                L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH,
            ],
            "sideinfo_effect_curve_tool_path": L5V2_SIDEINFO_EFFECT_CURVE_TOOL_PATH,
            "required_axes": list(_REQUIRED_EXACT_AXES),
            "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
            "architecture_lock_blocker": (
                "requires_paired_cpu_cuda_sideinfo_effect_curve_before_"
                "architecture_lock"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    elif not timing_smoke_valid:
        next_action = {
            "action_id": "materialize_tt5l_first_anchor_timing_smoke_artifact",
            "phase": "first_anchor_timing_smoke_custody",
            "tool_path": TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH,
            "command_template": (
                f".venv/bin/python {TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH} "
                "--result-artifact <timing_smoke_result_json> "
                "--provider <provider> --hardware <hardware> "
                "--provider-call-id <provider_call_id> "
                "--elapsed-seconds <seconds> "
                "--seconds-per-epoch <seconds_per_epoch> "
                "--command-argv-json '<exact argv json array>'"
            ),
            "artifact_path": TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH,
            "predicate_id": TT5L_FIRST_ANCHOR_TIMING_SMOKE_PREDICATE_ID,
            "required_axes": list(_REQUIRED_EXACT_AXES),
            "required_fields": [
                "schema",
                "lane_id",
                "predicate_id",
                "predicate_passed",
                "required_axes",
                "provider",
                "hardware",
                "provider_call_id",
                "command_argv",
                "elapsed_seconds",
                "seconds_per_epoch_or_seconds_per_candidate",
                "result_artifact_path",
                "result_artifact_sha256",
                "score_claim",
                "promotion_eligible",
                "ready_for_exact_eval_dispatch",
            ],
            "expected_artifacts": [TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH],
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
            "terminal_claim_required": True,
            "lane_id": LANE_ID,
            "pair_group_id": f"{LANE_ID}_exact_or_diagnostic_anchor_pair",
            "required_axes": list(_REQUIRED_EXACT_AXES),
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
        "dykstra_score_axis_sanity_valid": dykstra_valid,
        "dykstra_feasibility_status": dykstra_status,
        "move_level_feasibility_artifact_valid": move_level_valid,
        "move_level_feasibility_status": move_level_status,
        "sideinfo_gate_evidence_valid": sideinfo_valid,
        "probe_gate_evidence_valid": probe_valid,
        "paired_axis_plan_evidence_valid": paired_axis_plan_valid,
        "anchor_pair_evidence_valid": anchor_pair_valid,
        "sideinfo_effect_curve_allowed": sideinfo_effect_curve_allowed,
        "sideinfo_effect_curve_artifact_valid": sideinfo_effect_curve_valid,
        "sideinfo_effect_curve_status": sideinfo_effect_curve_status,
        "sideinfo_effect_curve_harvest_cells_status": (
            sideinfo_effect_curve_harvest_cells_status
        ),
        "first_anchor_timing_smoke_artifact_valid": timing_smoke_valid,
        "first_anchor_timing_smoke_status": timing_smoke_status,
        "first_anchor_timing_smoke_allowed": first_anchor_timing_smoke_allowed,
        "architecture_lock_allowed": architecture_lock_allowed,
        "proof_tool_path": TT5L_CONTEST_SIDEINFO_PROOF_TOOL_PATH,
        "proof_tool_exists": proof_tool_exists,
        "probe_tool_path": L5V2_PROBE_TOOL_PATH,
        "probe_tool_exists": probe_tool_exists,
        "probe_gate_artifact_status": probe_gate_artifact_status,
        "probe_observation_intake_status": probe_intake_status,
        "materialized_tt5l_paired_work_unit_status": materialized_work_unit_status,
        "sideinfo_effect_curve_lightning_paired_axis_plan_status": (
            lightning_paired_axis_plan_status
        ),
        "sideinfo_effect_curve_dispatch_plan_status": (
            sideinfo_effect_curve_dispatch_plan_status
        ),
        "measurement_schedule_tool_path": L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH,
        "measurement_schedule_artifact_path": L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
        "measurement_schedule_report_path": L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH,
        "measurement_schedule_score_claim": False,
        "measurement_schedule_promotion_eligible": False,
        "measurement_schedule_ready_for_exact_eval_dispatch": False,
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


def _git_head_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    try:
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _git_diff_name_only(
    repo_root: Path,
    base_commit: str,
    head_commit: str,
    paths: Iterable[str],
) -> tuple[str, ...]:
    clean_paths = tuple(path for path in paths if path)
    if not base_commit or not head_commit or not clean_paths:
        return ()
    try:
        proc = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                f"{base_commit}..{head_commit}",
                "--",
                *clean_paths,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if proc.returncode != 0:
        return ()
    return tuple(line.strip() for line in proc.stdout.splitlines() if line.strip())


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
    header_delta_allowed = False
    allowed_header_delta = proof.get("allowed_header_delta")
    if isinstance(allowed_header_delta, Mapping):
        header_delta_allowed = allowed_header_delta.get("allowed") is True
    if (
        proof.get("non_target_sections_identical") is not True
        and header_delta_allowed is not True
    ):
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "non_target_sections_identical"
        )
    if proof.get("non_target_payload_sections_identical") is not True:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "non_target_payload_sections_identical"
        )
    if proof.get("inflate_provenance_valid") is not True:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "inflate_provenance_valid"
        )
    provenance_blockers = proof.get("inflate_provenance_blockers")
    if not isinstance(provenance_blockers, list) or provenance_blockers:
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "inflate_provenance_blockers"
        )
    section_hashes = proof.get("section_hashes")
    if not isinstance(section_hashes, Mapping):
        blockers.append(
            "l5_v2_gate_artifact_semantics_missing:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "section_hashes"
        )
    else:
        header_section = section_hashes.get("tt5l_header")
        if not isinstance(header_section, Mapping):
            blockers.append(
                "l5_v2_gate_artifact_semantics_missing:"
                "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                "section_hashes:tt5l_header"
            )
        elif header_section.get("identical") is not True:
            header_allowed = header_section.get("allowed_delta")
            if not isinstance(header_allowed, Mapping) or header_allowed.get("allowed") is not True:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    "section_hashes:tt5l_header"
                )
        for section_name in ("world_model_blob", "ac_state_blob", "meta_blob"):
            section = section_hashes.get(section_name)
            if not isinstance(section, Mapping):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_missing:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    f"section_hashes:{section_name}"
                )
                continue
            baseline_sha = str(section.get("baseline_sha256") or "").strip().lower()
            mutated_sha = str(section.get("mutated_sha256") or "").strip().lower()
            if (
                not _SHA256_HEX_RE.fullmatch(baseline_sha)
                or not _SHA256_HEX_RE.fullmatch(mutated_sha)
                or baseline_sha != mutated_sha
                or section.get("identical") is not True
                or section.get("target_section") is True
            ):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                    f"section_hashes:{section_name}"
                )
        target_section = section_hashes.get("per_pair_side_info_blob")
        if not isinstance(target_section, Mapping):
            blockers.append(
                "l5_v2_gate_artifact_semantics_missing:"
                "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                "section_hashes:per_pair_side_info_blob"
            )
        elif (
            target_section.get("target_section") is not True
            or target_section.get("identical") is True
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
                "section_hashes:per_pair_side_info_blob"
            )
    provenance = proof.get("inflate_provenance")
    if not isinstance(provenance, Mapping):
        blockers.append(
            "l5_v2_gate_artifact_semantics_missing:"
            "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
            "inflate_provenance"
        )
    else:
        blockers.extend(
            _inflate_provenance_log_blockers(provenance, repo_root=repo_root)
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


def _inflate_provenance_log_blockers(
    provenance: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    blockers: list[str] = []
    prefix = (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "inflate_provenance"
    )
    missing_prefix = (
        "l5_v2_gate_artifact_semantics_missing:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "inflate_provenance"
    )
    for label in ("baseline", "mutated"):
        entry = provenance.get(label)
        if not isinstance(entry, Mapping):
            blockers.append(f"{missing_prefix}:{label}")
            continue
        if entry.get("schema") != "tt5l_inflate_provenance_v1":
            blockers.append(f"{prefix}:{label}:schema")
        log_path_value = str(entry.get("log_path") or "")
        resolved_log, path_error = _resolve_artifact_path(log_path_value, repo_root)
        if not log_path_value:
            blockers.append(f"{missing_prefix}:{label}:log_path")
            resolved_log = None
        elif path_error is not None:
            blockers.append(f"{prefix}:{label}:log_path_{path_error}")
            resolved_log = None
        elif resolved_log is None or not resolved_log.is_file():
            blockers.append(f"{prefix}:{label}:log_path_missing")
            resolved_log = None
        elif resolved_log.stat().st_size <= 0:
            blockers.append(f"{prefix}:{label}:log_empty")

        log_sha = str(entry.get("log_sha256") or "").strip().lower()
        if not _SHA256_HEX_RE.fullmatch(log_sha):
            blockers.append(f"{missing_prefix}:{label}:log_sha256")
        elif resolved_log is not None and _sha256_file(resolved_log) != log_sha:
            blockers.append(f"{prefix}:{label}:log_sha256")

        log_bytes = _non_bool_int(entry.get("log_bytes"))
        if log_bytes is None or log_bytes <= 0:
            blockers.append(f"{missing_prefix}:{label}:log_bytes")
        elif resolved_log is not None and resolved_log.stat().st_size != log_bytes:
            blockers.append(f"{prefix}:{label}:log_bytes")
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
            if not contains_non_negated_device_token(
                inflate_device,
                CPU_DEVICE_TOKENS,
            ) or contains_forbidden_contest_cpu_token(inflate_device):
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cpu_inflate_device"
                )
            if not contains_non_negated_device_token(
                eval_device,
                CPU_DEVICE_TOKENS,
            ) or contains_forbidden_contest_cpu_token(eval_device):
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
    if evidence.evidence_grade == "__non_object_gate_evidence__":
        blockers.append(f"l5_v2_gate_evidence_non_object:{gate.gate_id}")
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
    if "c1_z5_tt5l_probe_disambiguator" not in evidence_by_gate:
        canonical_probe = l5_v2_canonical_probe_gate_evidence(
            repo_root=resolved_repo_root,
        )
        if canonical_probe is not None:
            evidence_by_gate[canonical_probe.gate_id] = canonical_probe
    if "paired_cpu_cuda_axis_plan" not in evidence_by_gate:
        canonical_paired_axis_plan = l5_v2_canonical_paired_axis_plan_gate_evidence(
            repo_root=resolved_repo_root,
        )
        if canonical_paired_axis_plan is not None:
            evidence_by_gate[canonical_paired_axis_plan.gate_id] = (
                canonical_paired_axis_plan
            )
    if "exact_anchor_or_diagnostic_pair" not in evidence_by_gate:
        canonical_anchor_pair = l5_v2_canonical_anchor_pair_gate_evidence(
            repo_root=resolved_repo_root,
        )
        if canonical_anchor_pair is not None:
            evidence_by_gate[canonical_anchor_pair.gate_id] = canonical_anchor_pair
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
    prediction_band_status = l5_v2_prediction_band_status(
        repo_root=resolved_repo_root,
    )
    prediction_band_verdict = prediction_band_status["verdict"]
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
        "asymptotic_pursuit_candidates": l5_v2_asymptotic_pursuit_candidates(
            repo_root=resolved_repo_root,
        ),
        "prediction_band_verdict": prediction_band_verdict,
        "prediction_band_status": prediction_band_status,
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
    tt5l_campaign = payload["tt5l_campaign_readiness"]
    tt5l_cargo_cult_preconditions_valid = (
        tt5l_campaign["dykstra_feasibility_artifact_valid"] is True
        and tt5l_campaign["move_level_feasibility_artifact_valid"] is True
    )
    payload["tt5l_cargo_cult_preconditions_valid"] = (
        tt5l_cargo_cult_preconditions_valid
    )
    if not tt5l_cargo_cult_preconditions_valid:
        payload["ready_for_gate_probe_dispatch"] = False
        payload["ready_for_score_or_rank_dispatch"] = False
        payload["ready_for_dispatch"] = False
        if all_gate_evidence_valid:
            payload["blockers"].append(
                "tt5l_cargo_cult_preconditions_not_gate_probe_ready"
            )
            payload["blockers"].extend(
                f"tt5l_campaign:{blocker}"
                for blocker in tt5l_campaign.get("blockers", [])
                if str(blocker)
            )
    return payload


def l5_v2_architecture_lock_packet(
    gate_evidence: (
        Mapping[str, L5V2GateEvidence | Mapping[str, Any]]
        | Iterable[L5V2GateEvidence | Mapping[str, Any]]
        | None
    ) = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the canonical TT5L L5-v2 architecture lock/no-lock packet.

    The packet and intermediate readiness boolean intentionally share the same
    authority threshold: architecture lock requires all gate evidence plus
    side-info effect curve, first-anchor timing-smoke custody, and
    exact/diagnostic anchor custody.
    """

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    readiness = l5_v2_dispatch_readiness(
        gate_evidence=gate_evidence,
        repo_root=resolved_repo_root,
    )
    tt5l = readiness["tt5l_campaign_readiness"]
    required_checks = {
        "all_gate_evidence_valid": readiness.get("all_gate_evidence_valid") is True,
        "dykstra_score_axis_sanity_valid": (
            tt5l.get("dykstra_score_axis_sanity_valid") is True
        ),
        "move_level_feasibility_artifact_valid": (
            tt5l.get("move_level_feasibility_artifact_valid") is True
        ),
        "sideinfo_gate_evidence_valid": (
            tt5l.get("sideinfo_gate_evidence_valid") is True
        ),
        "probe_gate_evidence_valid": tt5l.get("probe_gate_evidence_valid") is True,
        "paired_axis_plan_evidence_valid": (
            tt5l.get("paired_axis_plan_evidence_valid") is True
        ),
        "sideinfo_effect_curve_artifact_valid": (
            tt5l.get("sideinfo_effect_curve_artifact_valid") is True
        ),
        "first_anchor_timing_smoke_artifact_valid": (
            tt5l.get("first_anchor_timing_smoke_artifact_valid") is True
        ),
        "anchor_pair_evidence_valid": tt5l.get("anchor_pair_evidence_valid") is True,
    }
    blocker_by_check = {
        "all_gate_evidence_valid": "requires_all_l5_v2_gate_evidence_valid",
        "dykstra_score_axis_sanity_valid": "requires_tt5l_dykstra_score_axis_sanity",
        "move_level_feasibility_artifact_valid": (
            "requires_tt5l_move_level_feasibility_artifact"
        ),
        "sideinfo_gate_evidence_valid": "requires_tt5l_sideinfo_gate_evidence",
        "probe_gate_evidence_valid": "requires_c1_z5_tt5l_probe_gate_evidence",
        "paired_axis_plan_evidence_valid": "requires_paired_cpu_cuda_axis_plan",
        "sideinfo_effect_curve_artifact_valid": (
            "requires_paired_cpu_cuda_sideinfo_effect_curve"
        ),
        "first_anchor_timing_smoke_artifact_valid": (
            "requires_tt5l_first_anchor_timing_smoke_artifact"
        ),
        "anchor_pair_evidence_valid": "requires_exact_or_diagnostic_anchor_pair",
    }
    architecture_lock_blockers = [
        blocker_by_check[check_id]
        for check_id, passed in required_checks.items()
        if passed is not True
    ]
    architecture_lock_allowed = not architecture_lock_blockers
    return {
        "schema": L5_V2_ARCHITECTURE_LOCK_PACKET_SCHEMA,
        "subject_id": SUBJECT_ID,
        "lane_id": LANE_ID,
        "campaign_id": CAMPAIGN_ID,
        "artifact_path": L5_V2_ARCHITECTURE_LOCK_PACKET_ARTIFACT_PATH,
        "report_path": L5_V2_ARCHITECTURE_LOCK_PACKET_REPORT_PATH,
        "tool_path": L5_V2_ARCHITECTURE_LOCK_PACKET_TOOL_PATH,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "architecture_lock_allowed": architecture_lock_allowed,
        "readiness_architecture_lock_allowed": (
            tt5l.get("architecture_lock_allowed") is True
        ),
        "required_checks": required_checks,
        "architecture_lock_blockers": architecture_lock_blockers,
        "next_non_pr106_l5_action": tt5l.get("next_non_pr106_l5_action", {}),
        "prediction_band_status": readiness.get("prediction_band_status", {}),
        "tt5l_campaign_readiness": tt5l,
        "readiness_blockers": list(readiness.get("blockers", [])),
        "authority_semantics": (
            "lock/no-lock planning packet only; no score, rank, promotion, "
            "or exact-dispatch authority"
        ),
    }


def render_l5_v2_architecture_lock_packet_markdown(
    packet: Mapping[str, Any],
) -> str:
    """Render the architecture lock/no-lock packet for .omx review."""

    checks = packet.get("required_checks")
    next_action = packet.get("next_non_pr106_l5_action")
    tt5l = packet.get("tt5l_campaign_readiness")
    if not isinstance(checks, Mapping):
        checks = {}
    if not isinstance(next_action, Mapping):
        next_action = {}
    if not isinstance(tt5l, Mapping):
        tt5l = {}
    prediction_band_status = packet.get("prediction_band_status")
    if not isinstance(prediction_band_status, Mapping):
        prediction_band_status = {}
    prediction_band_verdict = prediction_band_status.get("verdict")
    if not isinstance(prediction_band_verdict, Mapping):
        prediction_band_verdict = {}
    diagnostic_anchor_pair_status = prediction_band_status.get(
        "diagnostic_anchor_pair_status"
    )
    if not isinstance(diagnostic_anchor_pair_status, Mapping):
        diagnostic_anchor_pair_status = {}
    probe_gate_artifact_status = tt5l.get("probe_gate_artifact_status")
    if not isinstance(probe_gate_artifact_status, Mapping):
        probe_gate_artifact_status = {}
    first_anchor_timing_smoke_status = tt5l.get("first_anchor_timing_smoke_status")
    if not isinstance(first_anchor_timing_smoke_status, Mapping):
        first_anchor_timing_smoke_status = {}
    paired_axis_plan_status = tt5l.get(
        "sideinfo_effect_curve_lightning_paired_axis_plan_status"
    )
    if not isinstance(paired_axis_plan_status, Mapping):
        paired_axis_plan_status = {}
    sideinfo_effect_curve_status = tt5l.get("sideinfo_effect_curve_status")
    if not isinstance(sideinfo_effect_curve_status, Mapping):
        sideinfo_effect_curve_status = {}
    sideinfo_effect_curve_harvest_cells_status = tt5l.get(
        "sideinfo_effect_curve_harvest_cells_status"
    )
    if not isinstance(sideinfo_effect_curve_harvest_cells_status, Mapping):
        sideinfo_effect_curve_harvest_cells_status = {}
    sideinfo_effect_curve_dispatch_plan_status = tt5l.get(
        "sideinfo_effect_curve_dispatch_plan_status"
    )
    if not isinstance(sideinfo_effect_curve_dispatch_plan_status, Mapping):
        sideinfo_effect_curve_dispatch_plan_status = {}
    materialized_work_unit_status = next_action.get("materialized_work_unit_status")
    if not isinstance(materialized_work_unit_status, Mapping):
        materialized_work_unit_status = {}
    provider_blocker_status = next_action.get("provider_blocker_status")
    if not isinstance(provider_blocker_status, Mapping):
        provider_blocker_status = materialized_work_unit_status.get(
            "provider_blocker_status"
        )
    if not isinstance(provider_blocker_status, Mapping):
        provider_blocker_status = {}
    alternate_provider_plan_status = next_action.get("alternate_provider_plan_status")
    if not isinstance(alternate_provider_plan_status, Mapping):
        alternate_provider_plan_status = materialized_work_unit_status.get(
            "alternate_provider_plan_status"
        )
    if not isinstance(alternate_provider_plan_status, Mapping):
        alternate_provider_plan_status = {}
    lines = [
        "# L5 v2 architecture lock packet",
        "",
        f"- schema: `{packet.get('schema')}`",
        f"- subject_id: `{packet.get('subject_id')}`",
        f"- lane_id: `{packet.get('lane_id')}`",
        f"- architecture_lock_allowed: `{packet.get('architecture_lock_allowed')}`",
        (
            "- readiness_architecture_lock_allowed: "
            f"`{packet.get('readiness_architecture_lock_allowed')}`"
        ),
        f"- next_action: `{next_action.get('action_id', 'missing')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "",
        "## Required Checks",
    ]
    for check_id, passed in checks.items():
        lines.append(f"- `{check_id}`: `{passed}`")
    if prediction_band_status:
        lines.extend(
            [
                "",
                "## Prediction Band",
                "",
                (
                    "- rank_reward_allowed: "
                    f"`{prediction_band_status.get('rank_reward_allowed')}`"
                ),
                (
                    "- dispatch_planning_allowed: "
                    f"`{prediction_band_status.get('dispatch_planning_allowed')}`"
                ),
                (
                    "- verdict_blockers: "
                    f"`{prediction_band_verdict.get('blockers', [])}`"
                ),
                (
                    "- diagnostic_anchor_pair_exists: "
                    f"`{diagnostic_anchor_pair_status.get('artifact_exists')}`"
                ),
                (
                    "- diagnostic_anchor_pair_valid: "
                    f"`{diagnostic_anchor_pair_status.get('artifact_valid')}`"
                ),
                (
                    "- diagnostic_anchor_classification: "
                    f"`{diagnostic_anchor_pair_status.get('classification', '')}`"
                ),
                (
                    "- diagnostic_anchor_axes: "
                    f"`{diagnostic_anchor_pair_status.get('paired_axes', [])}`"
                ),
                (
                    "- diagnostic_anchor_scores: "
                    f"`{diagnostic_anchor_pair_status.get('per_axis_scores', {})}`"
                ),
                (
                    "- diagnostic_anchor_preserved_but_not_rankable: "
                    f"`{prediction_band_status.get('diagnostic_anchor_preserved_but_not_rankable')}`"
                ),
            ]
        )
    if paired_axis_plan_status:
        lines.extend(
            [
                "",
                "## Lightning Paired-Axis Dry-Run Plan",
                "",
                (
                    "- artifact_path: "
                    f"`{paired_axis_plan_status.get('artifact_path')}`"
                ),
                (
                    "- artifact_valid: "
                    f"`{paired_axis_plan_status.get('artifact_valid')}`"
                ),
                (
                    "- source_commit: "
                    f"`{paired_axis_plan_status.get('source_commit')}`"
                ),
                (
                    "- source_relevant_paths_match: "
                    f"`{paired_axis_plan_status.get('source_relevant_paths_match')}`"
                ),
                (
                    "- source_relevant_diff_paths: "
                    f"`{paired_axis_plan_status.get('source_relevant_diff_paths', [])}`"
                ),
                (
                    "- source_custody_current_for_execution: "
                    f"`{paired_axis_plan_status.get('source_custody_current_for_execution')}`"
                ),
                (
                    "- cells: "
                    f"`{paired_axis_plan_status.get('cell_count')}`/"
                    f"`{paired_axis_plan_status.get('expected_cell_count')}`"
                ),
                (
                    "- axes: "
                    f"`{paired_axis_plan_status.get('covered_axes')}`"
                ),
                (
                    "- all_cells_dry_run_ready: "
                    f"`{paired_axis_plan_status.get('all_cells_dry_run_ready')}`"
                ),
                (
                    "- all_cells_dry_run_structurally_valid: "
                    f"`{paired_axis_plan_status.get('all_cells_dry_run_structurally_valid')}`"
                ),
                (
                    "- execution_ready: "
                    f"`{paired_axis_plan_status.get('execution_ready')}`"
                ),
                (
                    "- execution_blockers: "
                    f"`{paired_axis_plan_status.get('execution_blockers', [])}`"
                ),
                "- score_claim: `false`",
                "- promotion_eligible: `false`",
            ]
        )
    if sideinfo_effect_curve_harvest_cells_status:
        lines.extend(
            [
                "",
                "## Sideinfo Harvest Cells",
                "",
                (
                    "- artifact_path: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('artifact_path')}`"
                ),
                (
                    "- artifact_valid: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('artifact_valid')}`"
                ),
                (
                    "- tool_path: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('tool_path')}`"
                ),
                (
                    "- cells: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('cell_count')}`/"
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('expected_cell_count')}`"
                ),
                (
                    "- harvested_exact_eval_artifact_count: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('harvested_exact_eval_artifact_count')}`"
                ),
                (
                    "- missing_exact_eval_artifact_count: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('missing_exact_eval_artifact_count')}`"
                ),
                (
                    "- source_plan: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('source_plan')}`"
                ),
                (
                    "- cell_blockers: "
                    f"`{sideinfo_effect_curve_harvest_cells_status.get('cell_blockers', [])[:10]}`"
                ),
                "- score_claim: `false`",
                "- promotion_eligible: `false`",
            ]
        )
    if sideinfo_effect_curve_status:
        observed_cells = sideinfo_effect_curve_status.get("observed_cells")
        if not isinstance(observed_cells, list):
            observed_cells = []
        lines.extend(
            [
                "",
                "## Sideinfo Effect Curve",
                "",
                (
                    "- artifact_path: "
                    f"`{sideinfo_effect_curve_status.get('artifact_path')}`"
                ),
                (
                    "- artifact_valid: "
                    f"`{sideinfo_effect_curve_status.get('artifact_valid')}`"
                ),
                (
                    "- measurement_id: "
                    f"`{sideinfo_effect_curve_status.get('measurement_id', '')}`"
                ),
                (
                    "- predicate_passed: "
                    f"`{sideinfo_effect_curve_status.get('predicate_passed')}`"
                ),
                (
                    "- observed_cell_count: "
                    f"`{sideinfo_effect_curve_status.get('observed_cell_count')}`"
                ),
                (
                    "- missing_cells: "
                    f"`{sideinfo_effect_curve_status.get('missing_cells', [])}`"
                ),
                (
                    "- effect_blockers: "
                    f"`{sideinfo_effect_curve_status.get('effect_blockers', [])}`"
                ),
                (
                    "- axis_effects: "
                    f"`{sideinfo_effect_curve_status.get('axis_effects', {})}`"
                ),
            ]
        )
        for cell in observed_cells[:3]:
            if not isinstance(cell, Mapping):
                continue
            axis = str(cell.get("axis") or "")
            variant = str(cell.get("variant") or "")
            lines.append(
                f"- observed_cell `{axis}/{variant}`: "
                f"score=`{cell.get('score')}`, "
                f"sideinfo_nonzero_fraction="
                f"`{cell.get('sideinfo_nonzero_fraction')}`, "
                f"sideinfo_nonzero_values="
                f"`{cell.get('sideinfo_nonzero_values')}`/"
                f"`{cell.get('sideinfo_total_values')}`, "
                f"archive_sha256=`{cell.get('archive_sha256')}`, "
                f"runtime_content_tree_sha256="
                f"`{cell.get('runtime_content_tree_sha256')}`"
            )
        lines.extend(
            [
                "- score_claim: `false`",
                "- promotion_eligible: `false`",
            ]
        )
    if sideinfo_effect_curve_dispatch_plan_status:
        work_units = sideinfo_effect_curve_dispatch_plan_status.get("work_units")
        if not isinstance(work_units, list):
            work_units = []
        lines.extend(
            [
                "",
                "## TT5L Sideinfo Dispatch Plan",
                "",
                (
                    "- artifact_path: "
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('artifact_path')}`"
                ),
                (
                    "- artifact_valid: "
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('artifact_valid')}`"
                ),
                (
                    "- plan_id: "
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('plan_id', '')}`"
                ),
                (
                    "- work_units: "
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('ready_work_unit_count')}`/"
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('work_unit_count')}`"
                ),
                (
                    "- required_variants: "
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('required_variants', [])}`"
                ),
                (
                    "- ready_for_operator_dispatch: "
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('ready_for_operator_dispatch')}`"
                ),
                "- ready_for_provider_dispatch: `false`",
                "- dispatch_attempted: `false`",
                (
                    "- blockers: "
                    f"`{sideinfo_effect_curve_dispatch_plan_status.get('blockers', [])}`"
                ),
            ]
        )
        for unit in work_units:
            if not isinstance(unit, Mapping):
                continue
            lines.append(
                f"- work_unit `{unit.get('variant')}`: "
                f"ready=`{unit.get('ready_for_operator_dispatch')}`, "
                f"archive_sha256=`{unit.get('archive_sha256')}`, "
                f"archive_bytes=`{unit.get('archive_bytes')}`, "
                f"pair_group_id=`{unit.get('pair_group_id')}`"
            )
        lines.extend(
            [
                "- score_claim: `false`",
                "- promotion_eligible: `false`",
            ]
        )
    if probe_gate_artifact_status:
        lines.extend(
            [
                "",
                "## Probe Gate Artifact",
                "",
                (
                    "- artifact_path: "
                    f"`{probe_gate_artifact_status.get('artifact_path')}`"
                ),
                (
                    "- artifact_exists: "
                    f"`{probe_gate_artifact_status.get('artifact_exists')}`"
                ),
                (
                    "- artifact_valid: "
                    f"`{probe_gate_artifact_status.get('artifact_valid')}`"
                ),
                (
                    "- architecture_lock_allowed: "
                    f"`{probe_gate_artifact_status.get('architecture_lock_allowed')}`"
                ),
                (
                    "- selected_candidate_id: "
                    f"`{probe_gate_artifact_status.get('selected_candidate_id')}`"
                ),
                (
                    "- verdict_blockers: "
                    f"`{probe_gate_artifact_status.get('verdict_blockers', [])}`"
                ),
            ]
        )
        for candidate in probe_gate_artifact_status.get("candidate_status", []) or []:
            if not isinstance(candidate, Mapping):
                continue
            lines.append(
                "- candidate "
                f"`{candidate.get('candidate_id')}`: "
                f"eligible=`{candidate.get('eligible_for_architecture_lock')}`, "
                f"axes=`{candidate.get('exact_axes', [])}`, "
                f"blockers=`{candidate.get('blockers', [])}`"
            )
    if first_anchor_timing_smoke_status:
        lines.extend(
            [
                "",
                "## First Anchor Timing Smoke",
                "",
                (
                    "- artifact_path: "
                    f"`{first_anchor_timing_smoke_status.get('artifact_path')}`"
                ),
                (
                    "- artifact_valid: "
                    f"`{first_anchor_timing_smoke_status.get('artifact_valid')}`"
                ),
                (
                    "- provider: "
                    f"`{first_anchor_timing_smoke_status.get('provider')}`"
                ),
                (
                    "- hardware: "
                    f"`{first_anchor_timing_smoke_status.get('hardware')}`"
                ),
                (
                    "- elapsed_seconds: "
                    f"`{first_anchor_timing_smoke_status.get('elapsed_seconds')}`"
                ),
                (
                    "- seconds_per_candidate: "
                    f"`{first_anchor_timing_smoke_status.get('seconds_per_candidate')}`"
                ),
                (
                    "- axis_timings: "
                    f"`{first_anchor_timing_smoke_status.get('axis_timings', {})}`"
                ),
                "- score_claim: `false`",
                "- promotion_eligible: `false`",
            ]
        )
    if materialized_work_unit_status or provider_blocker_status or alternate_provider_plan_status:
        lines.extend(
            [
                "",
                "## Materialized TT5L Provider Routing",
                "",
                (
                    "- work_unit_artifact_valid: "
                    f"`{materialized_work_unit_status.get('artifact_valid')}`"
                ),
                (
                    "- archive_sha256: "
                    f"`{materialized_work_unit_status.get('archive_sha256', '')}`"
                ),
                (
                    "- provider_blocker_active: "
                    f"`{provider_blocker_status.get('active')}`"
                ),
                (
                    "- provider_blocker_failure_class: "
                    f"`{provider_blocker_status.get('failure_class', '')}`"
                ),
                (
                    "- modal_execute_suppressed_until_blocker_resolved: "
                    f"`{next_action.get('modal_execute_command_suppressed_until_provider_blocker_resolved', False)}`"
                ),
                (
                    "- alternate_provider: "
                    f"`{alternate_provider_plan_status.get('provider', '')}`"
                ),
                (
                    "- alternate_artifact_valid: "
                    f"`{alternate_provider_plan_status.get('artifact_valid')}`"
                ),
                (
                    "- lightning_source_manifest_probe_current: "
                    f"`{alternate_provider_plan_status.get('source_manifest_probe_current')}`"
                ),
                (
                    "- lightning_execution_ready: "
                    f"`{alternate_provider_plan_status.get('execution_ready')}`"
                ),
                (
                    "- lightning_execution_blockers: "
                    f"`{alternate_provider_plan_status.get('execution_blockers', [])}`"
                ),
            ]
        )
    blockers = packet.get("architecture_lock_blockers")
    lines.extend(["", "## Blockers"])
    if isinstance(blockers, list) and blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Authority",
            "",
            str(packet.get("authority_semantics") or ""),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CAMPAIGN_ID",
    "L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH",
    "L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH",
    "L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH",
    "L5_V2_ARCHITECTURE_LOCK_PACKET_ARTIFACT_PATH",
    "L5_V2_ARCHITECTURE_LOCK_PACKET_REPORT_PATH",
    "L5_V2_ARCHITECTURE_LOCK_PACKET_SCHEMA",
    "L5_V2_ARCHITECTURE_LOCK_PACKET_TOOL_PATH",
    "L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_ARTIFACT_PATH",
    "L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_REPORT_PATH",
    "L5_V2_ASYMPTOTIC_NEXT_ACTION_STATUS_SCHEMA",
    "L5_V2_ASYMPTOTIC_PURSUIT_CANDIDATES_SCHEMA",
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
    "RUDIN_MEANINGFUL_INTERPRETABILITY_VERDICT",
    "RUDIN_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA",
    "RUDIN_PROXY_DISAMBIGUATOR_ARTIFACT_PATH",
    "RUDIN_PROXY_DISAMBIGUATOR_TOOL_PATH",
    "SUBJECT_ID",
    "TISHBY_D4_INDEPENDENT_VERDICT",
    "TISHBY_D4_PROBE_ARTIFACT_PATH",
    "TISHBY_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA",
    "TISHBY_VIB_TRACTABILITY_ARTIFACT_PATH",
    "TISHBY_VIB_TRACTABLE_VERDICT",
    "TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_PATH",
    "TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_SHA256",
    "TT5L_CONTEST_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH",
    "TT5L_CONTEST_SIDEINFO_PROOF_TOOL_PATH",
    "TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH",
    "TT5L_DYKSTRA_FEASIBILITY_GENERATED_BY_TOOL",
    "TT5L_DYKSTRA_FEASIBILITY_PREDICATE_ID",
    "TT5L_DYKSTRA_FEASIBILITY_SCHEMA",
    "TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH",
    "TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS",
    "TT5L_DYKSTRA_SCORE_FORMULA",
    "TT5L_DYKSTRA_VERDICT_AUTHORITY_SCOPE",
    "TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH",
    "TT5L_FIRST_ANCHOR_TIMING_SMOKE_PREDICATE_ID",
    "TT5L_FIRST_ANCHOR_TIMING_SMOKE_SCHEMA",
    "TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH",
    "TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH",
    "TT5L_MODAL_A100_DISPATCH_RECIPE_PATH",
    "TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH",
    "TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID",
    "TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA",
    "TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH",
    "TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_ARTIFACT_PATH",
    "TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_PREDICATE_ID",
    "TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH",
    "TT5L_PAIRED_EXACT_ANCHOR_PAIR_PREDICATE_ID",
    "TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH",
    "TT5L_PROBE_GATE_ARTIFACT_PATH",
    "TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH",
    "TT5L_PROBE_OBSERVATION_INTAKE_REPORT_PATH",
    "TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID",
    "TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH",
    "TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256",
    "TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH",
    "TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH",
    "Z6_POST_L1_PROXY_EVIDENCE_STATUS_SCHEMA",
    "Z6_REAL_VIDEO_EGO_PROXY_SWEEP_ARTIFACT_PATH",
    "Z6_REAL_VIDEO_EGO_PROXY_SWEEP_FULL_FILM_VERDICT",
    "Z6_REAL_VIDEO_EGO_PROXY_SWEEP_IDENTITY_DOMINATES_VERDICT",
    "Z6_REAL_VIDEO_EGO_PROXY_SWEEP_SCHEMA",
    "Z6_REAL_VIDEO_EGO_PROXY_SWEEP_TOOL_PATH",
    "L5V2AsymptoticPursuitCandidate",
    "L5V2Gate",
    "L5V2GateEvidence",
    "L5V2Step",
    "asymptotic_candidate_surface_json",
    "l5_v2_architecture_lock_packet",
    "l5_v2_asymptotic_pursuit_candidates",
    "l5_v2_canonical_anchor_pair_gate_evidence",
    "l5_v2_canonical_paired_axis_plan_gate_evidence",
    "l5_v2_canonical_probe_gate_evidence",
    "l5_v2_canonical_sideinfo_gate_evidence",
    "l5_v2_dispatch_readiness",
    "l5_v2_packetir_section_entropy_evidence_payload",
    "l5_v2_packetir_stack_evidence_payload",
    "l5_v2_pr106_stack_cell_candidates",
    "l5_v2_prediction_band_payload",
    "l5_v2_prediction_band_status",
    "l5_v2_prediction_band_verdict",
    "l5_v2_probe_gate_artifact_status",
    "l5_v2_required_gates",
    "l5_v2_research_basis_ids",
    "l5_v2_staircase_steps",
    "l5_v2_tt5l_campaign_readiness",
    "render_l5_v2_architecture_lock_packet_markdown",
    "render_l5_v2_asymptotic_candidate_surface_markdown",
    "tt5l_first_anchor_timing_smoke_status",
    "tt5l_move_level_feasibility_status",
    "tt5l_sideinfo_effect_curve_harvest_cells_status",
    "tt5l_sideinfo_effect_curve_status",
]
