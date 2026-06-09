# SPDX-License-Identifier: MIT
"""Queue-owned SNeRV LF/HF learned replacement planning.

This module consumes measured LF payload byte reports plus the current SNeRV LF
reroute/campaign handoff surfaces.  It deliberately stays false-authority: the
rows are local prototype work orders or explicit blockers, never score claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac.analysis.snerv_lf_hf_runtime_binding import (
    SCHEMA as SNERV_LF_HF_RUNTIME_BINDING_PROOF_SCHEMA,
)
from tac.analysis.snerv_lf_hf_runtime_binding import (
    bounded_training_blocker_for_solution_family,
    proof_cli_flag_for_solution_family,
    runtime_binding_blocker_for_solution_family,
)
from tac.analysis.snerv_official_tub_lf_hf_replacement_authority_gate import (
    summarize_snerv_official_tub_lf_hf_replacement_authority_gates,
)
from tac.analysis.snerv_source_forward_proof import (
    SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA,
    validate_snerv_source_forward_proof_action_effect,
)
from tac.optimization.evaluator_action_waterfill import CandidateActionEvaluation
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_PROOF_STATUS_SCHEMA,
    OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_HASH_FIELDS,
    OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_NUMERIC_FIELDS,
    OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_REQUIRED_PROOF_FIELDS,
    OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_TENSOR_HASH_GROUP_FIELDS,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SnervFrameCode,
    decode_frame,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    quantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    encode_lf_quant_payload_v2_with_report,
)

SCHEMA = "snerv_lf_hf_replacement_queue.v1"
ROW_SCHEMA = "snerv_lf_hf_replacement_candidate_row.v1"
DEFAULT_LANE_ID = "lane_snerv_lf_hf_replacement_queue_20260605"
DEFAULT_QUEUE_ID = "snerv_lf_hf_replacement_queue.v1"
AXIS_TAG = "[planning/control:false-authority]"
PLANNER_ROW_LAUNCH_CONTRACT_SCHEMA = (
    "nerv_long_training_queue_launch_authority_contract.v1"
)
BOUNDED_TRAINING_BINDING_CONTRACT_SCHEMA = (
    "snerv_lf_hf_bounded_training_binding_contract.v1"
)
UNBLOCK_LAUNCH_CONTRACT_SCHEMA = "snerv_lf_hf_queue_unblock_launch_contract.v1"
SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_OCCUPIED_CLASS_FRACTION = "0.400001"
SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_TARGET_CLASS_COVERAGE_FRACTION = "0.8"
SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_TARGET_CLASS_MIN_RATIO = "0.2"
SNERV_BOUNDED_SMOKE_MAX_POST_SEGNET_TARGET_CLASS_RATIO_DROP = "0.05"
SNERV_BOUNDED_SMOKE_SEGNET_RARE_CLASS_LOGIT_WEIGHT = "4"
SNERV_BOUNDED_SMOKE_SEGNET_TARGET_MASS_FLOOR_WEIGHT = "0.5"
SNERV_BOUNDED_SMOKE_SEGNET_TARGET_MIN_RATIO_FLOOR_WEIGHT = "0.5"
SNERV_BOUNDED_SMOKE_SEGNET_ESCAPE_WARMUP_EPOCHS = "32"
SNERV_BOUNDED_SMOKE_SEGNET_ESCAPE_CLASS_MULTIPLIER = "16"
SNERV_BOUNDED_SMOKE_POSENET_YUV6_GEOMETRY_TETHER_WEIGHT = "0.5"
SNERV_BOUNDED_SMOKE_MAX_POST_SEGNET_DISTRIBUTION_MAE = "0.31"
SNERV_BOUNDED_SMOKE_MAX_POST_POSENET_YUV6_DISTRIBUTION_MAE = "0.22"
SNERV_BOUNDED_SMOKE_MAX_POST_POSENET_YUV6_CONTRAST_RATIO = "3.75"
SNERV_SCORER_LOOP_QAT_MIN_RENDERER_PAIR_COUNT = 16
DEFAULT_MIN_FREE_BYTES = 1_000_000_000
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

QUEUE_FALSE_AUTHORITY = {
    **FALSE_AUTHORITY,
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "production_hardened_claim": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "local_mlx_long_training_allowed": False,
    "dispatch_allowed": False,
    "exact_or_full_video_cuda_allowed": False,
}

_SOURCE_FORWARD_BLOCKERS = (
    "snerv_official_mfu_hfr_tub_export_not_bound",
    "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
    "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
    "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
    "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
)
_SOURCE_FORWARD_FRAME_REPLAY_CLOSED_BLOCKERS = (
    "snerv_official_mfu_hfr_tub_export_not_bound",
    "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
    "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
)
_TUB_SOURCE_FIXTURE_CLOSED_BLOCKER_ALIASES = {
    "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity": (
        "snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity",
    ),
}
_SOURCE_FORWARD_QUEUE_FAMILIES = (
    "official_tub_lf_hf_decoder_replacement",
)
_RENDERER_BLOCKERS = (
    "snerv_renderer_nondegenerate_smoke_missing",
    "snerv_renderer_nondegenerate_smoke_failed",
    "snerv_renderer_nondegenerate_smoke_min16_pairs_missing",
    "snerv_renderer_nondegenerate_export_value_domain_not_passed",
    "snerv_renderer_nondegenerate_receiver_reconstruction_not_verified",
    "snerv_scorer_input_distribution_guard_missing",
)
_SCORER_DOMAIN_CLOSED_BLOCKERS = (
    "snerv_scorer_input_distribution_guard_missing",
)
_SCORER_DOMAIN_REQUIRED_METRICS = (
    "snerv_posenet_yuv6_pair_distill",
    "snerv_segnet_last_frame_distill",
)
_SCORER_LOOP_QAT_BOOLEAN_FLAGS = ("--snerv-scorer-loop-qat",)
_SCORER_LOOP_QAT_VALUE_FLAGS = (
    "--snerv-scorer-loop-max-trials",
    "--snerv-scorer-loop-search-mode",
    "--snerv-scorer-loop-step-map-bins",
    "--snerv-scorer-loop-qat-bits",
    "--snerv-scorer-loop-lf-payload-codec",
    "--snerv-scorer-loop-perturb-scale",
    "--snerv-scorer-loop-byte-pressure-multiplier",
    "--snerv-scorer-loop-section-value-pressure-multiplier",
    "--snerv-scorer-loop-max-archive-byte-growth",
    "--snerv-scorer-loop-byte-growth-admission-mode",
    "--snerv-scorer-loop-pose-slack",
    "--snerv-scorer-loop-seg-slack",
    "--snerv-scorer-loop-pair-stride",
    "--snerv-scorer-loop-start-pair",
    "--snerv-scorer-loop-pair-guard-min-score-improved-fraction",
    "--snerv-scorer-loop-pair-guard-max-pose-worsened-fraction",
    "--snerv-scorer-loop-component-guard-mode",
)
_TERMINAL_RENDERER_FEEDBACK_BLOCKERS = (
    "snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed",
    "snerv_renderer_nondegenerate_telemetry_contract_missing_or_failed",
)
_RENDERER_NONDEGENERATE_UNBLOCK_ALLOWED_BLOCKERS = (
    *_RENDERER_BLOCKERS,
    *_TERMINAL_RENDERER_FEEDBACK_BLOCKERS,
    "snerv_scorer_domain_tether_missing_telemetry",
    "snerv_scorer_domain_tether_lambda_inactive_telemetry",
)
_SKIP_HIGH_BLOCKERS = (
    "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
    "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
    "snerv_renderer_nondegenerate_target_value_domain_not_passed",
)
_VALUE_DOMAIN_CLOSED_BLOCKERS = _SKIP_HIGH_BLOCKERS
_LF_CONDITIONED_HF_FAMILY = "lf_conditioned_hf_residual_generator"
_HF_RESIDUAL_PAYLOAD_CLOSED_BLOCKERS = (
    "snerv_hf_residual_generator_receiver_payload_not_implemented",
)
_LF_CONDITIONED_HF_POST_PAYLOAD_BLOCKER = (
    "snerv_lf_conditioned_hf_bounded_training_binding_missing"
)
_JOINT_CODEBOOK_CLOSED_BLOCKERS = (
    "snerv_joint_lf_hf_factorized_codebook_not_implemented",
    "snerv_joint_lf_hf_codebook_numpy_receiver_missing",
    "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing",
)
_JOINT_CODEBOOK_POST_PAYLOAD_BLOCKER = (
    "snerv_joint_lf_hf_bounded_training_binding_missing"
)
_TEMPORAL_LF_PREDICTOR_CLOSED_BLOCKERS = (
    "snerv_temporal_lf_predictor_gate_not_implemented",
    "snerv_temporal_lf_predictor_correction_stream_not_byte_charged",
)
_LF_SUPER_RESOLUTION_CLOSED_BLOCKERS = (
    "snerv_lf_super_resolution_receiver_payload_not_implemented",
    "snerv_lf_downsampled_anchor_component_deltas_missing",
)
_SPECTRAL_BAND_ALLOCATOR_CLOSED_BLOCKERS = (
    "snerv_score_tethered_lf_hf_band_allocator_not_implemented",
    "snerv_mfu_hfr_section_native_byte_telemetry_missing",
)
_LF_LATENT_HYPERPRIOR_CLOSED_BLOCKERS = (
    "snerv_lf_latent_hyperprior_not_implemented",
    "snerv_lf_latent_hyperprior_numpy_decoder_missing",
    "snerv_lf_latent_hyperprior_receiver_replay_missing",
)
_RUNTIME_BINDING_EVIDENCE_KEY_BY_FAMILY = {
    _LF_CONDITIONED_HF_FAMILY: "hf_residual_payload_evidence",
    "joint_lf_hf_factorized_codebook": "joint_codebook_evidence",
    "temporal_lf_predictor_gate": "temporal_lf_predictor_evidence",
    "lf_super_resolution_from_tiny_anchor": "lf_super_resolution_evidence",
    "score_tethered_spectral_band_allocator": "spectral_band_allocator_evidence",
    "entropy_modeled_lf_latent_hyperprior": "lf_latent_hyperprior_evidence",
}
_OFFICIAL_TUB_LF_HF_FAMILY = "official_tub_lf_hf_decoder_replacement"


class SnervLfHfReplacementQueueError(ValueError):
    """Raised when the LF/HF replacement queue cannot be built."""


def build_snerv_lf_hf_replacement_queue(
    *,
    lf_payload_reports: Sequence[Mapping[str, Any]] = (),
    reroute_queues: Sequence[Mapping[str, Any]] = (),
    campaign_plans: Sequence[Mapping[str, Any]] = (),
    source_forward_artifacts: Sequence[Mapping[str, Any]] = (),
    official_replacement_authority_gates: Sequence[Mapping[str, Any]] = (),
    candidate_feedback_rows: Sequence[Mapping[str, Any]] = (),
    value_domain_xray_reports: Sequence[Mapping[str, Any]] = (),
    hf_residual_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    joint_codebook_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    temporal_lf_predictor_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    lf_super_resolution_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    spectral_band_allocator_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    lf_latent_hyperprior_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    lf_hf_runtime_binding_proofs: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path,
    lane_id: str = DEFAULT_LANE_ID,
    queue_id: str = DEFAULT_QUEUE_ID,
    queue_artifact_path: str | Path | None = None,
    generated_utc: str | None = None,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    allow_local_output: bool = False,
) -> dict[str, Any]:
    """Build a queue for learned SNeRV LF/HF replacement candidates.

    The builder is intentionally conservative.  Older LF byte reports remain
    useful as acquisition signal, but if the freshest SNAR2-era queue has no LF
    over-ceiling rows, the output records that as a blocker against re-enabling
    long training from LF dominance alone.
    """

    if not str(lane_id).strip():
        raise SnervLfHfReplacementQueueError("lane_id must be non-empty")
    if not str(queue_id).strip():
        raise SnervLfHfReplacementQueueError("queue_id must be non-empty")
    generated = generated_utc or datetime.now(UTC).isoformat()
    root = Path(output_root)
    storage_preflight = _storage_preflight(
        root,
        min_free_bytes=int(min_free_bytes),
        allow_local_output=bool(allow_local_output),
    )
    evidence_rows = [_lf_evidence_row(report, idx) for idx, report in enumerate(lf_payload_reports)]
    evidence_rows = [row for row in evidence_rows if row is not None]
    campaign_rows = _snerv_campaign_rows(campaign_plans)
    reroute_state = _reroute_state(reroute_queues)
    source_forward_state = _source_forward_state(source_forward_artifacts)
    official_replacement_authority_state = (
        summarize_snerv_official_tub_lf_hf_replacement_authority_gates(
            official_replacement_authority_gates
        )
    )
    scorer_domain_state = _scorer_domain_state(candidate_feedback_rows)
    value_domain_state = _value_domain_state(value_domain_xray_reports)
    hf_residual_payload_state = _hf_residual_payload_state(
        hf_residual_receiver_payload_proofs
    )
    joint_codebook_state = _joint_codebook_state(
        joint_codebook_receiver_payload_proofs
    )
    temporal_lf_predictor_state = _temporal_lf_predictor_state(
        temporal_lf_predictor_receiver_payload_proofs
    )
    lf_super_resolution_state = _lf_super_resolution_state(
        lf_super_resolution_receiver_payload_proofs
    )
    spectral_band_allocator_state = _spectral_band_allocator_state(
        spectral_band_allocator_receiver_payload_proofs
    )
    lf_latent_hyperprior_state = _lf_latent_hyperprior_state(
        lf_latent_hyperprior_receiver_payload_proofs
    )
    runtime_binding_state = _runtime_binding_state(lf_hf_runtime_binding_proofs)
    current_state = _current_state(
        campaign_rows=campaign_rows,
        reroute_state=reroute_state,
        evidence_rows=evidence_rows,
        source_forward_state=source_forward_state,
        official_replacement_authority_state=official_replacement_authority_state,
        scorer_domain_state=scorer_domain_state,
        value_domain_state=value_domain_state,
        hf_residual_payload_state=hf_residual_payload_state,
        joint_codebook_state=joint_codebook_state,
        temporal_lf_predictor_state=temporal_lf_predictor_state,
        lf_super_resolution_state=lf_super_resolution_state,
        spectral_band_allocator_state=spectral_band_allocator_state,
        lf_latent_hyperprior_state=lf_latent_hyperprior_state,
        runtime_binding_state=runtime_binding_state,
    )
    selected_evidence = _selected_lf_evidence(evidence_rows)
    input_source_paths = {
        "lf_payload_reports": _source_paths(evidence_rows),
        "reroute_queues": _source_paths(reroute_queues),
        "campaign_plans": _source_paths(campaign_plans),
        "source_forward_artifacts": _source_paths(source_forward_artifacts),
        "official_replacement_authority_gates": _source_paths(
            official_replacement_authority_gates
        ),
        "candidate_feedback_rows": _source_paths(candidate_feedback_rows),
        "value_domain_xray_reports": _source_paths(value_domain_xray_reports),
        "hf_residual_receiver_payload_proofs": _source_paths(
            hf_residual_receiver_payload_proofs
        ),
        "joint_codebook_receiver_payload_proofs": _source_paths(
            joint_codebook_receiver_payload_proofs
        ),
        "temporal_lf_predictor_receiver_payload_proofs": _source_paths(
            temporal_lf_predictor_receiver_payload_proofs
        ),
        "lf_super_resolution_receiver_payload_proofs": _source_paths(
            lf_super_resolution_receiver_payload_proofs
        ),
        "spectral_band_allocator_receiver_payload_proofs": _source_paths(
            spectral_band_allocator_receiver_payload_proofs
        ),
        "lf_latent_hyperprior_receiver_payload_proofs": _source_paths(
            lf_latent_hyperprior_receiver_payload_proofs
        ),
        "lf_hf_runtime_binding_proofs": _source_paths(
            lf_hf_runtime_binding_proofs
        ),
    }
    rebuild_command = _queue_rebuild_command(
        output_root=root,
        input_source_paths=input_source_paths,
    )
    official_gate_ready = (
        official_replacement_authority_state.get(
            "official_tub_lf_hf_decoder_replacement_ready"
        )
        is True
    )
    official_gate_queue_blockers = [
        str(blocker)
        for blocker in official_replacement_authority_state.get("queue_blockers", ())
        if str(blocker)
    ]
    official_next_unblock_command = (
        []
        if official_gate_ready and not official_gate_queue_blockers
        else _dedupe_command(
            _nested(
                official_replacement_authority_state,
                ("next_unblock_command_argv",),
            )
            or ()
        )
    )
    rows = _candidate_rows(
        campaign_rows=campaign_rows,
        selected_evidence=selected_evidence,
        current_state=current_state,
        output_root=root,
        queue_artifact_path=queue_artifact_path,
    )
    if not rows:
        rows = [
            _global_blocker_row(
                output_root=root,
                blocker="snerv_lf_hf_replacement_no_snerv_campaign_rows",
                selected_evidence=selected_evidence,
            )
        ]
    next_unblock_command = official_next_unblock_command or _first_unblock_command(
        rows
    )
    blocked_rows = [row for row in rows if row["blocked"]]
    executable_rows = [row for row in rows if row["command_argv"] and not row["blocked"]]
    roadmap_dag_nodes = _roadmap_dag_nodes(
        current_state=current_state,
        selected_evidence=selected_evidence,
        queue_rows=rows,
    )
    blockers = _dedupe(
        [
            "snerv_lf_hf_replacement_queue_false_authority",
            *current_state.get("blockers", ()),
            *[blocker for row in rows for blocker in row.get("blockers", ())],
        ]
    )
    return {
        "schema": SCHEMA,
        "queue_id": str(queue_id),
        "lane_id": str(lane_id),
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "queue_kind": "planner_queue_not_training_queue",
        "allowed_use": (
            "local bounded LF/HF replacement prototype selection and blocker "
            "routing before any long training or exact eval"
        ),
        "forbidden_use": (
            "score claim, promotion, rank/kill decision, exact eval dispatch, "
            "or long-training re-enable without row blockers clearing"
        ),
        "storage_preflight": storage_preflight,
        "current_state": current_state,
        "source_forward_evidence": source_forward_state,
        "official_replacement_authority_evidence": official_replacement_authority_state,
        "scorer_domain_evidence": scorer_domain_state,
        "value_domain_evidence": value_domain_state,
        "hf_residual_payload_evidence": hf_residual_payload_state,
        "joint_codebook_evidence": joint_codebook_state,
        "temporal_lf_predictor_evidence": temporal_lf_predictor_state,
        "lf_super_resolution_evidence": lf_super_resolution_state,
        "spectral_band_allocator_evidence": spectral_band_allocator_state,
        "lf_latent_hyperprior_evidence": lf_latent_hyperprior_state,
        "runtime_binding_evidence": runtime_binding_state,
        "lf_payload_evidence_rows": evidence_rows,
        "lf_payload_evidence_row_count": len(evidence_rows),
        "selected_lf_payload_evidence": selected_evidence,
        "queue_rows": rows,
        "queue_row_count": len(rows),
        "blocked_queue_row_count": len(blocked_rows),
        "local_executable_command_row_count": len(executable_rows),
        "roadmap_dag_nodes": roadmap_dag_nodes,
        "roadmap_dag_node_count": len(roadmap_dag_nodes),
        "blocked_roadmap_dag_node_count": sum(
            1 for row in roadmap_dag_nodes if row["blocked"]
        ),
        "input_source_paths": input_source_paths,
        "runnable_rebuild_command_argv": rebuild_command,
        "next_unblock_command_argv": next_unblock_command,
        "learned_replacement_candidate_row_count": sum(
            1 for row in rows if row["candidate_class"] == "learned_lf_hf_replacement"
        ),
        "blocking_queue_row_ids": [row["queue_row_id"] for row in blocked_rows],
        "runnable_queue_row_ids": [row["queue_row_id"] for row in executable_rows],
        "blockers": blockers,
        **QUEUE_FALSE_AUTHORITY,
    }


def summarize_snerv_lf_hf_source_forward_evidence(
    source_forward_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize official SNeRV source-forward evidence for planner consumers."""

    return _source_forward_state(source_forward_artifacts)


def _source_forward_replay_proof(
    selected: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for raw in (
        selected.get("source_forward_replay_proof"),
        replay.get("source_forward_replay_proof"),
    ):
        if isinstance(raw, Mapping):
            return dict(raw)
    return None


def _first_source_forward_mapping(
    keys: Sequence[str],
    *sources: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        for nested_key in (
            "pr95_distortion_axis_trace",
            "pair_local_distortion_servo",
            "nerv_pair_local_distortion_servo",
        ):
            nested = source.get(nested_key)
            if not isinstance(nested, Mapping):
                continue
            for key in keys:
                value = nested.get(key)
                if isinstance(value, Mapping):
                    return dict(value)
    return None


def _source_forward_axis_trace_measurements(
    *sources: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in (
            "pr95_distortion_axis_trace_measurements",
            "distortion_axis_trace_measurements",
            "axis_trace_measurements",
            "axis_trace_rows",
        ):
            value = source.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                rows.extend(dict(item) for item in value if isinstance(item, Mapping))
        nested = source.get("pr95_distortion_axis_trace")
        if isinstance(nested, Mapping):
            value = nested.get("measurements") or nested.get("rows")
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                rows.extend(dict(item) for item in value if isinstance(item, Mapping))
    return rows


def _source_forward_replay_proof_status(
    proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    proof_present = isinstance(proof, Mapping)
    missing = list(OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_REQUIRED_PROOF_FIELDS)
    invalid: list[str] = []
    action_effect_status: dict[str, Any] | None = None
    if proof_present and proof.get("schema") == SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA:
        action_effect_status = validate_snerv_source_forward_proof_action_effect(proof)
        missing = []
        invalid = list(action_effect_status["blockers"])
    elif proof_present:
        missing = [
            field
            for field in OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_REQUIRED_PROOF_FIELDS
            if field not in proof
        ]
        for field in OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_HASH_FIELDS:
            if field in proof and not _looks_like_sha256(proof.get(field)):
                invalid.append(field)
        for field in OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_NUMERIC_FIELDS:
            if field in proof and not _zero_float(proof.get(field)):
                invalid.append(field)
        for field in OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_TENSOR_HASH_GROUP_FIELDS:
            if field not in proof:
                continue
            group = proof.get(field)
            if (
                not isinstance(group, Mapping)
                or not group
                or any(
                    not str(name) or not _looks_like_sha256(value)
                    for name, value in group.items()
                )
            ):
                invalid.append(field)
        invalid.append("source_forward_action_effect_proof_missing")
    complete = bool(
        proof_present
        and not missing
        and not invalid
        and action_effect_status is not None
        and action_effect_status["passed"] is True
    )
    return {
        "schema": DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_PROOF_STATUS_SCHEMA,
        "source_forward_replay_proof_present": proof_present,
        "source_forward_replay_required_fields": list(
            OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_REQUIRED_PROOF_FIELDS
        ),
        "source_forward_replay_required_fields_missing": missing,
        "source_forward_replay_invalid_fields": _dedupe(invalid),
        "source_forward_replay_action_effect_schema": (
            SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA
        ),
        "source_forward_replay_action_effect_valid": bool(
            action_effect_status is not None
            and action_effect_status["passed"] is True
        ),
        "source_forward_replay_action_effect_blockers": (
            []
            if action_effect_status is None
            else list(action_effect_status["blockers"])
        ),
        "source_forward_proof_action_effect": (
            dict(proof)
            if (
                proof_present
                and proof.get("schema") == SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA
            )
            else None
        ),
        "source_forward_replay_numerical_proof_complete": complete,
        "source_forward_replay_proof_status": (
            "complete_numerical_source_forward_proof_present"
            if complete
            else (
                "metadata_only_or_incomplete_source_forward_proof"
                if proof_present
                else "missing_source_forward_proof"
            )
        ),
    }


def _looks_like_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _zero_float(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed == 0.0


# ---------------------------------------------------------------------------
# C3 — LF/HF carrier byte-pressure ↔ receiver-RGB-collapse curve
#
# The SNeRV hard blocker (per AGENTS.md / CLAUDE.md) is "LF/HF representation
# collapse under real byte pressure".  The existing lf_payload_codec_sweep is a
# RATE-ONLY surface (bytes, no visual metric, no scorer replay).  This C3 surface
# adds the missing DISTORTION axis: it drives the REAL LF/HF carrier (encode_frame_lf
# wavelet pyramid -> quantize_lf at a sweep of quantization granularities ->
# entropy-code -> dequantize_lf -> decode_frame inverse-DWT + HF restorer) and
# measures the receiver-RGB collapse (frame float linf, uint8 linf, and the
# argmax-disagreement RATE that the SegNet term is defined over) at each byte
# operating point.  No real SegNet/PoseNet is run here, so the row is
# research_only=true with a concrete pending blocker: the real scorer terms at
# the real operating point must be measured on contest hardware (C4).
# ---------------------------------------------------------------------------

LF_HF_BYTE_PRESSURE_CURVE_SCHEMA = "snerv_lf_hf_byte_pressure_curve.v1"
LF_HF_BYTE_PRESSURE_POINT_SCHEMA = "snerv_lf_hf_byte_pressure_point.v1"

# Canonical byte-pressure sweep: quantization granularity (n_levels) is the LF
# carrier's rate knob — fewer levels => coarser LF => fewer entropy-coded bytes.
DEFAULT_LF_HF_BYTE_PRESSURE_N_LEVELS: tuple[int, ...] = (256, 128, 64, 32, 16, 8, 4, 2)

# The byte operating point where the receiver RGB "collapses": argmax-disagreement
# fraction (the SegNet-term functional) crossing this threshold means a majority
# of last-frame pixels flip class vs the finest-quant baseline.  Non-authority
# threshold; the real SegNet d_seg at the operating point is the C4 question.
LF_HF_COLLAPSE_ARGMAX_DISAGREE_THRESHOLD = 0.5

LF_HF_BYTE_PRESSURE_SCORER_PENDING_BLOCKER = (
    "snerv_lf_hf_byte_pressure_real_scorer_terms_pending_contest_hardware"
)


def _lf_hf_byte_pressure_reference_frame(
    height: int, width: int, *, hf_amplitude: float = 0.0
) -> np.ndarray:
    """Deterministic gray test frame: smooth LF gradient + optional HF texture.

    The LF gradient exercises the LF carrier (the rate knob).  ``hf_amplitude``
    adds high-frequency texture; EMPIRICALLY the single-frame least-squares HF
    restorer (``fit_hf_decoder_least_squares``) DIVERGES on any nonzero HF content
    (the receiver frame blows far outside [0, 255] even at the finest quant), so
    the default is smooth-only (``hf_amplitude=0``) which keeps the finest-quant
    baseline faithful and isolates the LF-coarsening collapse.  Passing nonzero
    ``hf_amplitude`` is how the caller probes the HF-restorer-instability regime;
    the curve flags it via ``baseline_faithful=False``.  Values are in [0, 255].
    """

    yy, xx = np.meshgrid(
        np.arange(int(height)), np.arange(int(width)), indexing="ij"
    )
    lf = 128.0 + 60.0 * np.sin(xx / 4.0) + 40.0 * np.cos(yy / 3.0)
    frame = lf + float(hf_amplitude) * np.sin(xx / 1.3) * np.cos(yy / 1.1)
    return np.clip(frame, 0.0, 255.0).astype(np.float64)


def _lf_hf_argmax_disagree_fraction(
    base_uint8: np.ndarray, candidate_uint8: np.ndarray
) -> float:
    """Per-pixel disagreement RATE — the functional the SegNet d_seg term uses.

    For a single gray plane this is the fraction of pixels whose rounded uint8
    value differs; it is the receiver-RGB-collapse proxy, NON-AUTHORITY (the real
    SegNet argmaxes 5-class logits, not gray pixels)."""

    a = np.asarray(base_uint8)
    b = np.asarray(candidate_uint8)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.mean((a != b).astype(np.float64)))


def build_snerv_lf_hf_byte_pressure_curve(
    *,
    frame_height: int = 32,
    frame_width: int = 48,
    levels: int = 3,
    wavelet: str = "db2",
    hf_amplitude: float = 0.0,
    n_levels_sweep: Sequence[int] = DEFAULT_LF_HF_BYTE_PRESSURE_N_LEVELS,
    scorer_fn: Any | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Measure the LF/HF carrier receiver-RGB collapse under byte pressure.

    Drives the REAL SNeRV LF/HF carrier and reports, per byte operating point:

    * ``lf_payload_bytes`` — entropy-coded LF carrier bytes (the rate axis);
    * ``receiver_frame_float_linf`` / ``receiver_rgb_uint8_linf`` — receiver-RGB
      distortion vs the finest-quantization baseline (the distortion axis);
    * ``argmax_disagree_fraction`` — the per-pixel disagreement RATE the SegNet
      term is defined over (the collapse metric);
    * ``reference_scorer_delta`` — a NON-AUTHORITY scorer-shaped d_seg/d_pose pair.

    When a real ``scorer_fn`` ``(base_rgb_uint8, candidate_rgb_uint8) ->
    {"d_seg": float, "d_pose": float}`` is supplied (contest hardware), each point
    also emits a base-bound ``CandidateActionEvaluation``: a byte-pressure step
    REMOVES bytes (``delta_bytes < 0``) so any score reduction is unconditionally
    rent-paying — the curve doubles as the LF-carrier rate-distortion frontier the
    waterfilling law selects an operating point from.

    Without a real scorer this row is ``research_only=true`` and carries
    ``LF_HF_BYTE_PRESSURE_SCORER_PENDING_BLOCKER`` (the real SegNet/PoseNet terms
    at the operating point are the C4 question).  NO score / promotion authority.
    """

    if generated_utc is None:
        generated_utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep = tuple(int(v) for v in n_levels_sweep)
    base = {
        "schema": LF_HF_BYTE_PRESSURE_CURVE_SCHEMA,
        "family": "snerv",
        "component_id": "lf_hf_carrier",
        "generated_utc": generated_utc,
        "axis_tag": AXIS_TAG,
        "authority": "lf_hf_byte_pressure_distortion_curve_no_score_claim",
        "real_scorer_supplied": scorer_fn is not None,
        "research_only": scorer_fn is None,
        "frame_shape": [int(frame_height), int(frame_width)],
        "levels": int(levels),
        "wavelet": str(wavelet),
        "hf_amplitude": float(hf_amplitude),
        "n_levels_sweep": list(sweep),
        "collapse_argmax_disagree_threshold": LF_HF_COLLAPSE_ARGMAX_DISAGREE_THRESHOLD,
        "points": [],
        "candidate_action_evaluations": [],
        "blockers": [],
        **QUEUE_FALSE_AUTHORITY,
    }

    if not sweep:
        return {**base, "blockers": ["snerv_lf_hf_byte_pressure_empty_sweep"]}
    if any(v < 2 for v in sweep):
        return {**base, "blockers": ["snerv_lf_hf_byte_pressure_n_levels_below_2"]}

    try:
        frame = _lf_hf_byte_pressure_reference_frame(
            frame_height, frame_width, hf_amplitude=float(hf_amplitude)
        )
        pyramid = encode_frame_lf(frame, levels=int(levels), wavelet=str(wavelet))
        decoder = fit_hf_decoder_least_squares([pyramid], int(levels))
    except Exception as exc:  # fail-closed: a broken carrier proves nothing.
        return {
            **base,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": [
                "snerv_lf_hf_byte_pressure_carrier_failed:" + type(exc).__name__
            ],
        }

    def _reconstruct(n_levels: int) -> tuple[int, np.ndarray]:
        quant, scale, zero = quantize_lf(pyramid.lf, n_levels=int(n_levels))
        payload, _report = encode_lf_quant_payload_v2_with_report([quant])
        code = SnervFrameCode(
            lf_quant=quant,
            lf_scale=scale,
            lf_zero=zero,
            lf_shape=tuple(int(v) for v in quant.shape),
            levels=int(levels),
            wavelet=str(wavelet),
            orig_hw=(int(frame_height), int(frame_width)),
            per_element_steps=None,
        )
        frame_recon = np.asarray(decode_frame(code, decoder), dtype=np.float64)
        return len(bytes(payload)), frame_recon

    def _rgb_uint8(gray: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(np.asarray(gray, dtype=np.float64)), 0, 255).astype(
            np.uint8
        )

    # Finest quantization is the baseline the collapse is measured against.
    finest_levels = max(sweep)
    finest_bytes, base_frame = _reconstruct(finest_levels)
    base_rgb_uint8 = _rgb_uint8(base_frame)
    # HF-restorer instability detector: the least-squares HF decoder diverges on
    # HF content even at the finest quant (the receiver frame leaves [0, 255]).
    # When the baseline is itself unfaithful, the byte-pressure curve below is
    # measuring divergence, not graceful LF coarsening — flag it honestly.
    base_min = float(np.min(base_frame))
    base_max = float(np.max(base_frame))
    baseline_faithful = bool(base_min >= -1.0 and base_max <= 256.0)

    points: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    blockers: list[str] = []
    collapse_n_levels: int | None = None
    collapse_bytes: int | None = None
    if not baseline_faithful:
        blockers.append(
            "snerv_lf_hf_byte_pressure_hf_restorer_diverges_finest_baseline_unfaithful"
        )

    for n_levels in sweep:
        payload_bytes, frame_recon = _reconstruct(n_levels)
        cand_rgb_uint8 = _rgb_uint8(frame_recon)
        frame_float_linf = float(np.max(np.abs(base_frame - frame_recon)))
        rgb_uint8_linf = int(
            np.max(np.abs(base_rgb_uint8.astype(np.int64) - cand_rgb_uint8.astype(np.int64)))
        )
        argmax_disagree = _lf_hf_argmax_disagree_fraction(base_rgb_uint8, cand_rgb_uint8)
        collapsed = bool(argmax_disagree >= LF_HF_COLLAPSE_ARGMAX_DISAGREE_THRESHOLD)
        if collapsed and collapse_n_levels is None:
            collapse_n_levels = int(n_levels)
            collapse_bytes = int(payload_bytes)

        reference_scorer = _lf_hf_byte_pressure_reference_scorer(
            base_rgb_uint8, cand_rgb_uint8
        )

        candidate_action_evaluation: dict[str, Any] | None = None
        scorer_delta: dict[str, Any] | None = None
        if scorer_fn is not None:
            point_blockers: list[str] = []
            base_metrics = _coerce_lf_hf_scorer_metrics(
                scorer_fn(base_rgb_uint8, base_rgb_uint8),
                n_levels=n_levels,
                blockers=point_blockers,
            )
            cand_metrics = _coerce_lf_hf_scorer_metrics(
                scorer_fn(base_rgb_uint8, cand_rgb_uint8),
                n_levels=n_levels,
                blockers=point_blockers,
            )
            blockers.extend(point_blockers)
            if base_metrics is not None and cand_metrics is not None:
                evaluation = CandidateActionEvaluation(
                    action_id=f"snerv_lf_hf_byte_pressure:n_levels_{int(n_levels)}",
                    action_kind="snerv_lf_hf_carrier_quantization_byte_pressure",
                    base_archive_sha256=_lf_hf_payload_sha(finest_bytes, finest_levels),
                    with_action_archive_sha256=_lf_hf_payload_sha(
                        payload_bytes, n_levels
                    ),
                    d_seg_base=float(base_metrics["d_seg"]),
                    d_pose_base=float(base_metrics["d_pose"]),
                    bytes_base=int(finest_bytes),
                    d_seg_with_action=float(cand_metrics["d_seg"]),
                    d_pose_with_action=float(cand_metrics["d_pose"]),
                    bytes_with_action=int(payload_bytes),
                    scorer_effect_survived=bool(rgb_uint8_linf > 0),
                    evidence_grade="advisory",
                )
                candidate_action_evaluation = evaluation.to_row()
                scorer_delta = {
                    "delta_score_total": evaluation.delta_score_total,
                    "delta_score_nonrate": evaluation.delta_score_nonrate,
                    "delta_bytes": evaluation.delta_bytes,
                    "value_per_byte": evaluation.value_per_byte,
                    "pays_rent": evaluation.pays_rent,
                }
                evaluations.append(candidate_action_evaluation)
        else:
            blockers.append(LF_HF_BYTE_PRESSURE_SCORER_PENDING_BLOCKER)

        points.append(
            {
                "schema": LF_HF_BYTE_PRESSURE_POINT_SCHEMA,
                "n_levels": int(n_levels),
                "lf_payload_bytes": int(payload_bytes),
                "delta_bytes_vs_finest": int(payload_bytes) - int(finest_bytes),
                "receiver_frame_float_linf": frame_float_linf,
                "receiver_rgb_uint8_linf": rgb_uint8_linf,
                "argmax_disagree_fraction": argmax_disagree,
                "receiver_rgb_collapsed": collapsed,
                "is_finest_baseline": bool(int(n_levels) == int(finest_levels)),
                "reference_scorer_delta": reference_scorer,
                "reference_scorer_authority": False,
                "scorer_delta": scorer_delta,
                "candidate_action_evaluation": candidate_action_evaluation,
                **QUEUE_FALSE_AUTHORITY,
            }
        )

    return {
        **base,
        "carrier_executed": True,
        "carrier_scope": "snerv_real_lf_hf_wavelet_carrier_encode_quantize_decode_cpu_portable",
        "finest_n_levels": int(finest_levels),
        "finest_lf_payload_bytes": int(finest_bytes),
        "finest_baseline_frame_min": base_min,
        "finest_baseline_frame_max": base_max,
        "finest_baseline_faithful": baseline_faithful,
        "hf_restorer_diverges": not baseline_faithful,
        "base_rgb_uint8_sha256": hashlib.sha256(
            np.ascontiguousarray(base_rgb_uint8).tobytes()
        ).hexdigest(),
        "collapse_onset_n_levels": collapse_n_levels,
        "collapse_onset_lf_payload_bytes": collapse_bytes,
        "receiver_rgb_collapses_under_byte_pressure": bool(
            collapse_n_levels is not None
        ),
        "points": points,
        "candidate_action_evaluations": evaluations,
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _lf_hf_payload_sha(payload_bytes: int, n_levels: int) -> str:
    return hashlib.sha256(
        f"snerv_lf_hf_byte_pressure:{int(payload_bytes)}:{int(n_levels)}".encode()
    ).hexdigest()


def _lf_hf_byte_pressure_reference_scorer(
    base_rgb_uint8: np.ndarray, candidate_rgb_uint8: np.ndarray
) -> dict[str, float]:
    """NON-AUTHORITY scorer-shaped d_seg/d_pose for the byte-pressure point.

    d_seg mirrors the argmax-disagreement RATE; d_pose mirrors a first-6-dim MSE
    on a fixed linear pose proxy.  Magnitudes carry no score authority."""

    base = np.asarray(base_rgb_uint8, dtype=np.float64) / 255.0
    cand = np.asarray(candidate_rgb_uint8, dtype=np.float64) / 255.0
    if base.shape != cand.shape or base.size == 0:
        return {"d_seg": 0.0, "d_pose": 0.0}
    d_seg = _lf_hf_argmax_disagree_fraction(base_rgb_uint8, candidate_rgb_uint8)
    n_features = int(base.size)
    rng = np.random.default_rng(20260609)
    proj = rng.standard_normal((n_features, 6)) / np.sqrt(float(n_features))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        base_pose = np.ascontiguousarray(base.reshape(1, -1)) @ proj
        cand_pose = np.ascontiguousarray(cand.reshape(1, -1)) @ proj
    d_pose = float(np.mean((base_pose - cand_pose) ** 2))
    return {"d_seg": float(d_seg), "d_pose": d_pose}


def _coerce_lf_hf_scorer_metrics(
    raw: Any, *, n_levels: int, blockers: list[str]
) -> dict[str, float] | None:
    if not isinstance(raw, Mapping):
        blockers.append(f"snerv_lf_hf_byte_pressure_scorer_output_invalid:{int(n_levels)}")
        return None
    out: dict[str, float] = {}
    for field in ("d_seg", "d_pose"):
        value = raw.get(field)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            blockers.append(
                f"snerv_lf_hf_byte_pressure_scorer_metric_invalid:{int(n_levels)}:{field}"
            )
            return None
        if not math.isfinite(parsed) or parsed < 0.0:
            blockers.append(
                f"snerv_lf_hf_byte_pressure_scorer_metric_invalid:{int(n_levels)}:{field}"
            )
            return None
        out[field] = parsed
    return out


def render_snerv_lf_hf_replacement_queue_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact handoff for operator review."""

    current = report.get("current_state", {}) if isinstance(report, Mapping) else {}
    selected = report.get("selected_lf_payload_evidence") if isinstance(report, Mapping) else None
    selected = selected if isinstance(selected, Mapping) else {}
    lines = [
        "# SNeRV LF/HF Replacement Queue",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- lane: `{report.get('lane_id')}`",
        f"- axis: `{report.get('axis_tag')}`",
        f"- queue rows: `{report.get('queue_row_count')}`",
        f"- runnable local rows: `{report.get('local_executable_command_row_count')}`",
        f"- current reroute rows: `{current.get('freshest_reroute_queue_row_count')}`",
        f"- current SNAR2 no-LF-overrun: `{current.get('freshest_queue_has_no_lf_over_ceiling_rows')}`",
        f"- LF dominance launch signal active: `{current.get('lf_dominance_launch_signal_active')}`",
        "- receiver payload frame replay proven: "
        f"`{_nested(current, ('source_forward_evidence', 'receiver_payload_frame_replay_proven'))}`",
        "- official replacement authority ready: "
        f"`{_nested(current, ('official_replacement_authority_evidence', 'official_tub_lf_hf_decoder_replacement_ready'))}`",
        "- scorer domain tether proof passed: "
        f"`{_nested(current, ('scorer_domain_evidence', 'scorer_domain_tether_proof_passed'))}`",
        "- value-domain noncollapse proof passed: "
        f"`{_nested(current, ('value_domain_evidence', 'value_domain_noncollapse_proof_passed'))}`",
        "- receiver runtime binding families: "
        f"`{', '.join(_nested(current, ('runtime_binding_evidence', 'runtime_bound_solution_families')) or [])}`",
        f"- selected LF evidence bytes: `{selected.get('lf_payload_bytes')}`",
        "",
        "## Roadmap DAG",
    ]
    for node in report.get("roadmap_dag_nodes", []) if isinstance(report, Mapping) else []:
        if not isinstance(node, Mapping):
            continue
        lines.extend(
            [
                "",
                f"### `{node.get('node_id')}`",
                f"- blocked: `{node.get('blocked')}`",
                f"- depends on: `{', '.join(node.get('depends_on') or [])}`",
                "- blockers:",
            ]
        )
        lines.extend(f"  - `{blocker}`" for blocker in node.get("blockers") or ())
    lines.extend(
        [
            "",
            "## Candidate Rows",
        ]
    )
    for row in report.get("queue_rows", []) if isinstance(report, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                "",
                f"### `{row.get('queue_row_id')}`",
                f"- family: `{row.get('solution_family')}`",
                f"- action: `{row.get('planner_action')}`",
                f"- blocked: `{row.get('blocked')}`",
                f"- command: `{_shell_join(row.get('command_argv') or [])}`",
                "- blockers:",
            ]
        )
        blockers = [str(v) for v in row.get("blockers") or ()]
        lines.extend(f"  - `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def _roadmap_dag_nodes(
    *,
    current_state: Mapping[str, Any],
    selected_evidence: Mapping[str, Any] | None,
    queue_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source = current_state.get("source_forward_evidence")
    source = source if isinstance(source, Mapping) else {}
    scorer = current_state.get("scorer_domain_evidence")
    scorer = scorer if isinstance(scorer, Mapping) else {}
    value = current_state.get("value_domain_evidence")
    value = value if isinstance(value, Mapping) else {}
    official_rows = [
        row
        for row in queue_rows
        if row.get("solution_family") == "official_tub_lf_hf_decoder_replacement"
    ]
    lf_conditioned_rows = [
        row
        for row in queue_rows
        if row.get("solution_family") == _LF_CONDITIONED_HF_FAMILY
    ]
    implementation_rows = [
        row
        for row in queue_rows
        if row.get("solution_family")
        in {
            "joint_lf_hf_factorized_codebook",
            "temporal_lf_predictor_gate",
            "lf_super_resolution_from_tiny_anchor",
            "score_tethered_spectral_band_allocator",
            "entropy_modeled_lf_latent_hyperprior",
        }
    ]
    return [
        _roadmap_node(
            "measured_lf_payload_reports",
            [],
            selected_evidence is not None,
            [] if selected_evidence is not None else ["snerv_lf_hf_measured_lf_payload_report_missing"],
        ),
        _roadmap_node(
            "current_snar2_lf_overrun_handoff",
            ["measured_lf_payload_reports"],
            current_state.get("lf_dominance_launch_signal_active") is True,
            []
            if current_state.get("lf_dominance_launch_signal_active") is True
            else list(current_state.get("demoted_blockers") or ()),
        ),
        _roadmap_node(
            "official_checkpoint_export_binding",
            ["current_snar2_lf_overrun_handoff"],
            source.get("official_checkpoint_export_bound") is True,
            []
            if source.get("official_checkpoint_export_bound") is True
            else ["snerv_official_mfu_hfr_tub_export_not_bound"],
        ),
        _roadmap_node(
            "receiver_output2_frame_replay",
            ["official_checkpoint_export_binding"],
            bool(
                source.get("receiver_payload_frame_replay_proven") is True
                and source.get("receiver_frame_decode_consumes_output2") is True
            ),
            [
                blocker
                for blocker in (
                    "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
                    "snerv_official_tub_output2_receiver_frame_decode_not_bound",
                )
                if blocker in set(source.get("queue_blockers") or ())
            ],
        ),
        _roadmap_node(
            "scorer_domain_guard",
            ["receiver_output2_frame_replay"],
            scorer.get("scorer_input_distribution_guard_proof_passed") is True,
            scorer.get("queue_blockers") or (),
        ),
        _roadmap_node(
            "official_tub_lf_hf_decoder_replacement",
            ["scorer_domain_guard"],
            bool(official_rows) and all(row.get("blocked") is False for row in official_rows),
            _family_blockers(official_rows),
        ),
        _roadmap_node(
            "lf_conditioned_hf_residual_generator",
            ["official_tub_lf_hf_decoder_replacement"],
            bool(lf_conditioned_rows)
            and all(row.get("blocked") is False for row in lf_conditioned_rows)
            and value.get("value_domain_noncollapse_proof_passed") is True,
            _family_blockers(lf_conditioned_rows),
        ),
        _roadmap_node(
            "remaining_lf_hf_family_implementations",
            ["lf_conditioned_hf_residual_generator"],
            bool(implementation_rows)
            and all(row.get("blocked") is False for row in implementation_rows),
            _family_blockers(implementation_rows),
        ),
    ]


def _roadmap_node(
    node_id: str,
    depends_on: Sequence[str],
    ready: bool,
    blockers: Sequence[Any],
) -> dict[str, Any]:
    clean_blockers = _dedupe(blockers)
    if not ready and not clean_blockers:
        clean_blockers = [f"{node_id}_not_proven"]
    return {
        "schema": "snerv_lf_hf_replacement_roadmap_dag_node.v1",
        "node_id": node_id,
        "depends_on": list(depends_on),
        "blocked": bool(clean_blockers),
        "status": "ready_no_authority" if not clean_blockers else "blocked_until_prerequisites",
        "blockers": clean_blockers,
        **QUEUE_FALSE_AUTHORITY,
    }


def _family_blockers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["snerv_lf_hf_replacement_family_rows_missing"]
    return _dedupe([blocker for row in rows for blocker in row.get("blockers") or ()])


def _candidate_rows(
    *,
    campaign_rows: Sequence[Mapping[str, Any]],
    selected_evidence: Mapping[str, Any] | None,
    current_state: Mapping[str, Any],
    output_root: Path,
    queue_artifact_path: str | Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = list(campaign_rows)
    if not source_rows:
        return rows
    for row in source_rows[:4]:
        rows.extend(
            [
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family=_OFFICIAL_TUB_LF_HF_FAMILY,
                    planner_action="run_bounded_source_faithful_lf_hf_decoder_smoke",
                    learning_objective=(
                        "learn receiver-visible official MFU/HFR/TUB decoder "
                        "that reconstructs LF and generates HF under scorer "
                        "tether, replacing stored full LF grids only after "
                        "source-forward replay passes"
                    ),
                    static_blockers=(),
                    campaign_blocker_prefixes=_SOURCE_FORWARD_BLOCKERS + _RENDERER_BLOCKERS,
                    command_kind="bounded_snerv_training_smoke",
                    priority=10,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family=_LF_CONDITIONED_HF_FAMILY,
                    planner_action="probe_non_scalar_hf_generation_without_skip_high_collapse",
                    learning_objective=(
                        "keep a small LF carrier and learn HF residuals only "
                        "when SegNet/PoseNet price them; reject scalar/channel "
                        "mean skip-high collapse before replay"
                    ),
                    static_blockers=(
                        "snerv_hf_residual_generator_receiver_payload_not_implemented",
                    ),
                    campaign_blocker_prefixes=_SKIP_HIGH_BLOCKERS + _RENDERER_BLOCKERS,
                    command_kind="lf_conditioned_hf_residual_payload_proof",
                    priority=20,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family="joint_lf_hf_factorized_codebook",
                    planner_action="build_score_tethered_joint_lf_hf_codebook_export",
                    learning_objective=(
                        "learn a byte-charged LF/HF factorized codebook with "
                        "NumPy receiver decode and section telemetry before "
                        "any full run"
                    ),
                    static_blockers=(
                        "snerv_joint_lf_hf_factorized_codebook_not_implemented",
                        "snerv_joint_lf_hf_codebook_numpy_receiver_missing",
                        "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="joint_lf_hf_codebook_payload_proof",
                    priority=30,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family="temporal_lf_predictor_gate",
                    planner_action="learn_temporal_lf_delta_predictor_with_receiver_gate",
                    learning_objective=(
                        "predict LF planes from temporal context and store only "
                        "a byte-charged correction stream when the official "
                        "source-forward path proves the predictor is consumed"
                    ),
                    static_blockers=(
                        "snerv_temporal_lf_predictor_gate_not_implemented",
                        "snerv_temporal_lf_predictor_correction_stream_not_byte_charged",
                    ),
                    campaign_blocker_prefixes=(),
                    command_kind="temporal_lf_predictor_payload_proof",
                    priority=40,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family="lf_super_resolution_from_tiny_anchor",
                    planner_action="store_tiny_lf_anchor_then_learn_receiver_super_resolution",
                    learning_objective=(
                        "store a deliberately tiny LF anchor and learn a NumPy "
                        "receiver super-resolution decoder whose HF errors are "
                        "priced by SegNet/PoseNet component telemetry"
                    ),
                    static_blockers=(
                        "snerv_lf_super_resolution_receiver_payload_not_implemented",
                        "snerv_lf_downsampled_anchor_component_deltas_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="lf_super_resolution_tiny_anchor_payload_proof",
                    priority=50,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family="score_tethered_spectral_band_allocator",
                    planner_action="learn_mfu_hfr_lf_hf_band_budget_from_scorer_telemetry",
                    learning_objective=(
                        "learn the LF/HF band split under scorer telemetry so "
                        "MFU/HFR controls actuate section bytes instead of only "
                        "nominal model-size tokens"
                    ),
                    static_blockers=(
                        "snerv_score_tethered_lf_hf_band_allocator_not_implemented",
                        "snerv_mfu_hfr_section_native_byte_telemetry_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="spectral_band_allocator_payload_proof",
                    priority=60,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family="entropy_modeled_lf_latent_hyperprior",
                    planner_action="replace_i64_lzma_lf_planes_with_learned_entropy_model",
                    learning_objective=(
                        "replace generic int64+LZMA LF storage with a learned "
                        "latent entropy model and deterministic NumPy decode, "
                        "then require receiver proof and component replay"
                    ),
                    static_blockers=(
                        "snerv_lf_latent_hyperprior_not_implemented",
                        "snerv_lf_latent_hyperprior_numpy_decoder_missing",
                        "snerv_lf_latent_hyperprior_receiver_replay_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="lf_latent_hyperprior_payload_proof",
                    priority=70,
                ),
            ]
        )
    rows.sort(
        key=lambda row: (
            int(row.get("priority") or 999),
            str(row.get("source_campaign_row_id") or ""),
            str(row.get("solution_family") or ""),
        )
    )
    return rows


def _candidate_row(
    *,
    campaign_row: Mapping[str, Any],
    selected_evidence: Mapping[str, Any] | None,
    current_state: Mapping[str, Any],
    output_root: Path,
    queue_artifact_path: str | Path | None,
    solution_family: str,
    planner_action: str,
    learning_objective: str,
    static_blockers: Sequence[str],
    campaign_blocker_prefixes: Sequence[str],
    command_kind: str,
    priority: int,
) -> dict[str, Any]:
    source_row_id = str(campaign_row.get("row_id") or campaign_row.get("candidate_id") or "snerv")
    candidate_id = str(campaign_row.get("candidate_id") or "candidate")
    token = _stable_safe_token(f"{source_row_id}_{solution_family}")
    queue_row_id = f"snerv_lf_hf_replace_{token}"
    evidence_blockers: list[str] = []
    if selected_evidence is None:
        evidence_blockers.append("snerv_lf_hf_measured_lf_payload_report_missing")
    elif _positive_int(selected_evidence.get("lf_payload_bytes")) is None:
        evidence_blockers.append("snerv_lf_hf_selected_lf_payload_bytes_missing")
    current_blockers = [
        str(blocker)
        for blocker in current_state.get("blockers", ())
        if str(blocker)
    ]
    campaign_blockers = _campaign_blockers(campaign_row, campaign_blocker_prefixes)
    source_forward_closed = set(
        _nested(current_state, ("source_forward_evidence", "closed_campaign_blockers"))
        or ()
    )
    scorer_domain_closed = set(
        _nested(current_state, ("scorer_domain_evidence", "closed_campaign_blockers"))
        or ()
    )
    value_domain_closed = set(
        _nested(current_state, ("value_domain_evidence", "closed_campaign_blockers"))
        or ()
    )
    hf_residual_payload_closed = set(
        _nested(
            current_state,
            ("hf_residual_payload_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    joint_codebook_closed = set(
        _nested(
            current_state,
            ("joint_codebook_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    temporal_lf_predictor_closed = set(
        _nested(
            current_state,
            ("temporal_lf_predictor_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    lf_super_resolution_closed = set(
        _nested(
            current_state,
            ("lf_super_resolution_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    spectral_band_allocator_closed = set(
        _nested(
            current_state,
            ("spectral_band_allocator_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    lf_latent_hyperprior_closed = set(
        _nested(
            current_state,
            ("lf_latent_hyperprior_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    runtime_binding_closed = set(
        _nested(
            current_state,
            ("runtime_binding_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    official_replacement_closed = set(
        _nested(
            current_state,
            ("official_replacement_authority_evidence", "closed_campaign_blockers"),
        )
        or ()
    )
    inherited_queue_authority_blockers = _inherited_queue_authority_blockers(
        current_state
    )
    source_forward_extra_blockers: list[str] = []
    if solution_family in _SOURCE_FORWARD_QUEUE_FAMILIES:
        source_forward_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(current_state, ("source_forward_evidence", "queue_blockers"))
                or ()
            )
            if blocker
        ]
    official_replacement_extra_blockers: list[str] = []
    if solution_family == _OFFICIAL_TUB_LF_HF_FAMILY:
        official_replacement_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(
                    current_state,
                    ("official_replacement_authority_evidence", "queue_blockers"),
                )
                or ()
            )
            if blocker
        ]
    official_gate_source_forward_contradiction_blockers = (
        _official_gate_source_forward_contradiction_blockers(current_state)
        if solution_family == _OFFICIAL_TUB_LF_HF_FAMILY
        else []
    )
    terminal_renderer_extra_blockers: list[str] = []
    if command_kind == "bounded_snerv_training_smoke":
        terminal_renderer_extra_blockers = _candidate_terminal_renderer_blockers(
            current_state,
            candidate_id=candidate_id,
        )
    value_domain_extra_blockers: list[str] = []
    if solution_family == _LF_CONDITIONED_HF_FAMILY:
        value_domain_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(current_state, ("value_domain_evidence", "queue_blockers"))
                or ()
            )
            if blocker
        ]
    hf_residual_payload_extra_blockers: list[str] = []
    if solution_family == _LF_CONDITIONED_HF_FAMILY:
        hf_residual_payload_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(
                    current_state,
                    ("hf_residual_payload_evidence", "queue_blockers"),
                )
                or ()
            )
            if blocker
        ]
    effective_static_blockers = [
        str(blocker)
        for blocker in static_blockers
        if str(blocker) not in hf_residual_payload_closed
        and str(blocker) not in joint_codebook_closed
        and str(blocker) not in temporal_lf_predictor_closed
        and str(blocker) not in lf_super_resolution_closed
        and str(blocker) not in spectral_band_allocator_closed
        and str(blocker) not in lf_latent_hyperprior_closed
        and str(blocker) not in runtime_binding_closed
    ]
    joint_codebook_extra_blockers: list[str] = []
    if solution_family == "joint_lf_hf_factorized_codebook":
        joint_codebook_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(current_state, ("joint_codebook_evidence", "queue_blockers"))
                or ()
            )
            if blocker
        ]
    temporal_lf_predictor_extra_blockers: list[str] = []
    if solution_family == "temporal_lf_predictor_gate":
        temporal_lf_predictor_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(
                    current_state,
                    ("temporal_lf_predictor_evidence", "queue_blockers"),
                )
                or ()
            )
            if blocker
        ]
    lf_super_resolution_extra_blockers: list[str] = []
    if solution_family == "lf_super_resolution_from_tiny_anchor":
        lf_super_resolution_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(
                    current_state,
                    ("lf_super_resolution_evidence", "queue_blockers"),
                )
                or ()
            )
            if blocker
        ]
    spectral_band_allocator_extra_blockers: list[str] = []
    if solution_family == "score_tethered_spectral_band_allocator":
        spectral_band_allocator_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(
                    current_state,
                    ("spectral_band_allocator_evidence", "queue_blockers"),
                )
                or ()
            )
            if blocker
        ]
    lf_latent_hyperprior_extra_blockers: list[str] = []
    if solution_family == "entropy_modeled_lf_latent_hyperprior":
        lf_latent_hyperprior_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(
                    current_state,
                    ("lf_latent_hyperprior_evidence", "queue_blockers"),
                )
                or ()
            )
            if blocker
        ]
    campaign_blockers = [
        blocker
        for blocker in campaign_blockers
        if blocker not in source_forward_closed
        and blocker not in scorer_domain_closed
        and blocker not in value_domain_closed
        and blocker not in hf_residual_payload_closed
        and blocker not in joint_codebook_closed
        and blocker not in temporal_lf_predictor_closed
        and blocker not in lf_super_resolution_closed
        and blocker not in spectral_band_allocator_closed
        and blocker not in lf_latent_hyperprior_closed
        and blocker not in runtime_binding_closed
        and blocker not in official_replacement_closed
    ]
    blockers = _dedupe(
        [
            *effective_static_blockers,
            *evidence_blockers,
            *current_blockers,
            *campaign_blockers,
            *source_forward_extra_blockers,
            *official_replacement_extra_blockers,
            *official_gate_source_forward_contradiction_blockers,
            *terminal_renderer_extra_blockers,
            *value_domain_extra_blockers,
            *hf_residual_payload_extra_blockers,
            *joint_codebook_extra_blockers,
            *temporal_lf_predictor_extra_blockers,
            *lf_super_resolution_extra_blockers,
            *spectral_band_allocator_extra_blockers,
            *lf_latent_hyperprior_extra_blockers,
        ]
    )
    if (
        solution_family == _OFFICIAL_TUB_LF_HF_FAMILY
        and _nested(
            current_state,
            ("official_replacement_authority_evidence", "artifact_count"),
        )
    ):
        source_forward_gate_blockers = set(source_forward_extra_blockers)
        blockers = _dedupe(
            [
                *[
                    blocker
                    for blocker in blockers
                    if blocker in source_forward_gate_blockers
                    or (
                        not blocker.startswith("snerv_official_")
                        and not blocker.startswith("official_")
                    )
                ],
                *official_replacement_extra_blockers,
            ]
        )
    command: list[str] = []
    unblock_command: list[str] = []
    effective_command_kind = command_kind
    if (
        command_kind == "bounded_snerv_training_smoke"
        and not blockers
    ):
        command = _bounded_snerv_smoke_command(
            campaign_row,
            current_state=current_state,
            queue_row_id=queue_row_id,
            output_root=output_root,
            queue_artifact_path=queue_artifact_path,
            solution_family=solution_family,
        )
        if not command:
            blockers = _dedupe([*blockers, "snerv_lf_hf_base_snerv_command_missing"])
    if command_kind == "lf_conditioned_hf_residual_payload_proof":
        hf_residual_proof_blockers = set(_HF_RESIDUAL_PAYLOAD_CLOSED_BLOCKERS)
        non_hf_residual_blockers = [
            blocker for blocker in blockers if blocker not in hf_residual_proof_blockers
        ]
        if non_hf_residual_blockers:
            blockers = _dedupe(non_hf_residual_blockers)
        elif blockers:
            unblock_command = _lf_conditioned_hf_residual_payload_proof_command(
                current_state,
                queue_row_id=queue_row_id,
                output_root=output_root,
            )
            if not unblock_command:
                blockers = _dedupe(
                    [
                        *blockers,
                        "snerv_lf_conditioned_hf_residual_bounded_command_missing",
                    ]
                )
        elif not blockers:
            runtime_blocker = runtime_binding_blocker_for_solution_family(
                solution_family
            )
            post_runtime_blocker = bounded_training_blocker_for_solution_family(
                solution_family
            )
            if runtime_blocker in runtime_binding_closed:
                command = _bounded_snerv_smoke_command(
                    campaign_row,
                    current_state=current_state,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family=solution_family,
                )
                effective_command_kind = "bounded_snerv_training_smoke"
                if not command:
                    blockers = _dedupe(
                        [
                            *blockers,
                            "snerv_lf_conditioned_hf_bounded_command_missing",
                        ]
                    )
            elif runtime_blocker:
                blockers = _dedupe([*blockers, runtime_blocker])
                unblock_command = _runtime_binding_proof_command(
                    current_state,
                    solution_family=solution_family,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                )
            else:
                blockers = _dedupe([*blockers, post_runtime_blocker])
    if command_kind == "joint_lf_hf_codebook_payload_proof":
        joint_codebook_proof_blockers = set(_JOINT_CODEBOOK_CLOSED_BLOCKERS)
        non_joint_codebook_blockers = [
            blocker
            for blocker in blockers
            if blocker not in joint_codebook_proof_blockers
        ]
        if non_joint_codebook_blockers:
            blockers = _dedupe(non_joint_codebook_blockers)
        elif blockers:
            unblock_command = _joint_lf_hf_codebook_payload_proof_command(
                current_state,
                queue_row_id=queue_row_id,
                output_root=output_root,
            )
            if not unblock_command:
                blockers = _dedupe(
                    [
                        *blockers,
                        "snerv_joint_lf_hf_codebook_bounded_command_missing",
                    ]
                )
        elif not blockers:
            runtime_blocker = runtime_binding_blocker_for_solution_family(
                solution_family
            )
            post_runtime_blocker = bounded_training_blocker_for_solution_family(
                solution_family
            )
            if runtime_blocker in runtime_binding_closed:
                command = _bounded_snerv_smoke_command(
                    campaign_row,
                    current_state=current_state,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                    queue_artifact_path=queue_artifact_path,
                    solution_family=solution_family,
                )
                effective_command_kind = "bounded_snerv_training_smoke"
                if not command:
                    blockers = _dedupe(
                        [*blockers, "snerv_joint_lf_hf_codebook_bounded_command_missing"]
                    )
            elif runtime_blocker:
                blockers = _dedupe([*blockers, runtime_blocker])
                unblock_command = _runtime_binding_proof_command(
                    current_state,
                    solution_family=solution_family,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                )
            else:
                blockers = _dedupe([*blockers, post_runtime_blocker])
    if command_kind == "lf_super_resolution_tiny_anchor_payload_proof":
        lf_super_resolution_proof_blockers = set(_LF_SUPER_RESOLUTION_CLOSED_BLOCKERS)
        non_lf_super_resolution_blockers = [
            blocker
            for blocker in blockers
            if blocker not in lf_super_resolution_proof_blockers
        ]
        if non_lf_super_resolution_blockers:
            blockers = _dedupe(non_lf_super_resolution_blockers)
        elif blockers:
            unblock_command = _lf_super_resolution_tiny_anchor_payload_proof_command(
                current_state,
                selected_evidence=selected_evidence,
                queue_row_id=queue_row_id,
                output_root=output_root,
            )
            if not unblock_command:
                blockers = _dedupe(
                    [*blockers, "snerv_lf_super_resolution_bounded_command_missing"]
                )
        elif not blockers:
            runtime_blocker = runtime_binding_blocker_for_solution_family(
                solution_family
            )
            post_runtime_blocker = bounded_training_blocker_for_solution_family(
                solution_family
            )
            if runtime_blocker in runtime_binding_closed:
                blockers = _dedupe([*blockers, post_runtime_blocker])
            else:
                blockers = _dedupe([*blockers, runtime_blocker])
                unblock_command = _runtime_binding_proof_command(
                    current_state,
                    solution_family=solution_family,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                )
    if command_kind == "temporal_lf_predictor_payload_proof":
        temporal_proof_blockers = set(_TEMPORAL_LF_PREDICTOR_CLOSED_BLOCKERS)
        non_temporal_blockers = [
            blocker for blocker in blockers if blocker not in temporal_proof_blockers
        ]
        if non_temporal_blockers:
            blockers = _dedupe(non_temporal_blockers)
        elif blockers:
            unblock_command = _temporal_lf_predictor_payload_proof_command(
                current_state,
                selected_evidence=selected_evidence,
                queue_row_id=queue_row_id,
                output_root=output_root,
            )
            if not unblock_command:
                blockers = _dedupe(
                    [
                        *blockers,
                        "snerv_temporal_lf_predictor_bounded_command_missing",
                    ]
                )
        elif not blockers:
            runtime_blocker = runtime_binding_blocker_for_solution_family(
                solution_family
            )
            post_runtime_blocker = bounded_training_blocker_for_solution_family(
                solution_family
            )
            if runtime_blocker in runtime_binding_closed:
                blockers = _dedupe([*blockers, post_runtime_blocker])
            else:
                blockers = _dedupe([*blockers, runtime_blocker])
                unblock_command = _runtime_binding_proof_command(
                    current_state,
                    solution_family=solution_family,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                )
    if command_kind == "spectral_band_allocator_payload_proof":
        spectral_proof_blockers = set(_SPECTRAL_BAND_ALLOCATOR_CLOSED_BLOCKERS)
        non_spectral_blockers = [
            blocker for blocker in blockers if blocker not in spectral_proof_blockers
        ]
        if non_spectral_blockers:
            blockers = _dedupe(non_spectral_blockers)
        elif blockers:
            unblock_command = _spectral_band_allocator_payload_proof_command(
                current_state,
                selected_evidence=selected_evidence,
                queue_row_id=queue_row_id,
                output_root=output_root,
            )
            if not unblock_command:
                blockers = _dedupe(
                    [*blockers, "snerv_spectral_band_allocator_bounded_command_missing"]
                )
        elif not blockers:
            runtime_blocker = runtime_binding_blocker_for_solution_family(
                solution_family
            )
            post_runtime_blocker = bounded_training_blocker_for_solution_family(
                solution_family
            )
            if runtime_blocker in runtime_binding_closed:
                blockers = _dedupe([*blockers, post_runtime_blocker])
            else:
                blockers = _dedupe([*blockers, runtime_blocker])
                unblock_command = _runtime_binding_proof_command(
                    current_state,
                    solution_family=solution_family,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                )
    if command_kind == "lf_latent_hyperprior_payload_proof":
        hyperprior_proof_blockers = set(_LF_LATENT_HYPERPRIOR_CLOSED_BLOCKERS)
        non_hyperprior_blockers = [
            blocker
            for blocker in blockers
            if blocker not in hyperprior_proof_blockers
        ]
        if non_hyperprior_blockers:
            blockers = _dedupe(non_hyperprior_blockers)
        elif blockers:
            unblock_command = _lf_latent_hyperprior_payload_proof_command(
                current_state,
                selected_evidence=selected_evidence,
                queue_row_id=queue_row_id,
                output_root=output_root,
            )
            if not unblock_command:
                blockers = _dedupe(
                    [*blockers, "snerv_lf_latent_hyperprior_bounded_command_missing"]
                )
        elif not blockers:
            runtime_blocker = runtime_binding_blocker_for_solution_family(
                solution_family
            )
            post_runtime_blocker = bounded_training_blocker_for_solution_family(
                solution_family
            )
            if runtime_blocker in runtime_binding_closed:
                blockers = _dedupe([*blockers, post_runtime_blocker])
            else:
                blockers = _dedupe([*blockers, runtime_blocker])
                unblock_command = _runtime_binding_proof_command(
                    current_state,
                    solution_family=solution_family,
                    queue_row_id=queue_row_id,
                    output_root=output_root,
                )
    renderer_unblock_blockers = _renderer_nondegenerate_unblock_blockers(blockers)
    runtime_blocker = runtime_binding_blocker_for_solution_family(solution_family)
    renderer_unblock_command_allowed = command_kind == "bounded_snerv_training_smoke" or (
        bool(runtime_blocker) and runtime_blocker in runtime_binding_closed
    )
    if (
        renderer_unblock_command_allowed
        and not command
        and blockers
        and not renderer_unblock_blockers
    ):
        unblock_command = _bounded_snerv_smoke_command(
            campaign_row,
            current_state=current_state,
            queue_row_id=queue_row_id,
            output_root=output_root,
            queue_artifact_path=queue_artifact_path,
            solution_family=solution_family,
        )
    if not blockers and not command:
        blockers = _dedupe(
            [*blockers, f"snerv_lf_hf_{solution_family}_runnable_command_missing"]
        )
    status = (
        "local_bounded_smoke_ready_no_authority"
        if command and not blockers
        else "blocked_until_prerequisite_evidence"
    )
    launch_contract = _launch_authority_contract(
        status=status,
        blockers=blockers,
        command=command,
    )
    bounded_training_binding_contract = _bounded_training_binding_contract(
        solution_family=solution_family,
        command_kind=effective_command_kind,
        command=command,
    )
    unblock_launch_contract = _unblock_launch_authority_contract(
        unblock_kind="snerv_renderer_nondegenerate_smoke",
        blockers=renderer_unblock_blockers,
        command=unblock_command,
    )
    return {
        "schema": ROW_SCHEMA,
        "queue_row_id": queue_row_id,
        "row_id": queue_row_id,
        "lane_id": DEFAULT_LANE_ID,
        "family": "snerv",
        "execute_family": "snerv",
        "source_campaign_row_id": source_row_id,
        "candidate_id": candidate_id,
        "candidate_class": "learned_lf_hf_replacement",
        "solution_family": solution_family,
        "planner_action": planner_action,
        "learning_objective": learning_objective,
        "priority": int(priority),
        "status": status,
        "blocked": bool(blockers),
        "blockers": blockers,
        "inherited_queue_authority_blockers": inherited_queue_authority_blockers,
        "launch_authority_contract": launch_contract,
        "bounded_training_binding_contract": bounded_training_binding_contract,
        "unblock_launch_authority_contract": unblock_launch_contract,
        "selected_lf_payload_evidence": selected_evidence,
        "measured_lf_payload_bytes": (
            None if selected_evidence is None else selected_evidence.get("lf_payload_bytes")
        ),
        "measured_raw_lf_bytes": (
            None if selected_evidence is None else selected_evidence.get("raw_lf_bytes")
        ),
        "hard_byte_ceiling": _positive_int(campaign_row.get("hard_byte_ceiling")),
        "source_campaign_status": {
            "implementation_status": campaign_row.get("implementation_status"),
            "local_mlx_launch_command_ready": bool(campaign_row.get("local_mlx_launch_command_ready")),
            "hard_byte_ceiling_satisfied_for_long_training": campaign_row.get("hard_byte_ceiling_satisfied_for_long_training"),
            "candidate_nominal_under_ceiling": campaign_row.get("candidate_nominal_under_ceiling"),
            "campaign_blockers": list(campaign_row.get("blockers") or ()),
        },
        "snar2_current_state": {
            "freshest_queue_has_no_lf_over_ceiling_rows": current_state.get(
                "freshest_queue_has_no_lf_over_ceiling_rows"
            ),
            "freshest_reroute_queue_row_count": current_state.get(
                "freshest_reroute_queue_row_count"
            ),
            "snar_header_minimization_report_count": current_state.get(
                "snar_header_minimization_report_count"
            ),
            "lf_dominance_launch_signal_active": current_state.get(
                "lf_dominance_launch_signal_active"
            ),
            "lf_dominance_signal_demoted": current_state.get(
                "lf_dominance_signal_demoted"
            ),
            "demoted_blockers": list(current_state.get("demoted_blockers") or ()),
        },
        "source_forward_evidence": current_state.get("source_forward_evidence"),
        "official_replacement_authority_evidence": current_state.get(
            "official_replacement_authority_evidence"
        ),
        "official_gate_selected_path": _nested(
            current_state,
            ("official_replacement_authority_evidence", "source_path"),
        ),
        "official_gate_selected_sha256": _nested(
            current_state,
            ("official_replacement_authority_evidence", "source_sha256"),
        ),
        "official_gate_selected_generated_utc": _nested(
            current_state,
            ("official_replacement_authority_evidence", "selected_artifact_generated_utc"),
        ),
        "official_gate_selection_policy": _nested(
            current_state,
            ("official_replacement_authority_evidence", "selection_policy"),
        ),
        "official_gate_source_forward_contradiction_blockers": (
            official_gate_source_forward_contradiction_blockers
        ),
        "closed_source_forward_blockers": _dedupe(
            [
                *(
                    _nested(
                        current_state,
                        ("source_forward_evidence", "closed_campaign_blockers"),
                    )
                    or ()
                ),
                *(
                    _nested(
                        current_state,
                        (
                            "official_replacement_authority_evidence",
                            "closed_campaign_blockers",
                        ),
                    )
                    or ()
                ),
            ]
        ),
        "source_forward_authority_residual_blockers": _nested(
            current_state,
            (
                "official_replacement_authority_evidence",
                "source_forward_authority_residual_blockers",
            ),
        )
        or [],
        "scorer_domain_evidence": current_state.get("scorer_domain_evidence"),
        "value_domain_evidence": current_state.get("value_domain_evidence"),
        "hf_residual_payload_evidence": current_state.get(
            "hf_residual_payload_evidence"
        ),
        "joint_codebook_evidence": current_state.get("joint_codebook_evidence"),
        "temporal_lf_predictor_evidence": current_state.get(
            "temporal_lf_predictor_evidence"
        ),
        "lf_super_resolution_evidence": current_state.get(
            "lf_super_resolution_evidence"
        ),
        "spectral_band_allocator_evidence": current_state.get(
            "spectral_band_allocator_evidence"
        ),
        "lf_latent_hyperprior_evidence": current_state.get(
            "lf_latent_hyperprior_evidence"
        ),
        "runtime_binding_evidence": current_state.get("runtime_binding_evidence"),
        "target_consumers": [
            "nerv_long_training_campaign_plan",
            "snerv_lf_over_ceiling_reroute_queue",
            "nerv_rate_allocator_queue",
            "cathedral_autopilot",
        ],
        "command_argv": command,
        "unblock_command_argv": unblock_command,
        "output_root": output_root.as_posix(),
        **QUEUE_FALSE_AUTHORITY,
    }


def _inherited_queue_authority_blockers(
    current_state: Mapping[str, Any],
) -> list[str]:
    """Expose queue-level authority blockers on every row without changing semantics."""

    return _dedupe(
        [
            *(
                _nested(current_state, ("source_forward_evidence", "queue_blockers"))
                or ()
            ),
            *(
                _nested(
                    current_state,
                    ("official_replacement_authority_evidence", "queue_blockers"),
                )
                or ()
            ),
        ]
    )


def _official_gate_source_forward_contradiction_blockers(
    current_state: Mapping[str, Any],
) -> list[str]:
    official_ready = (
        _nested(
            current_state,
            (
                "official_replacement_authority_evidence",
                "official_tub_lf_hf_decoder_replacement_ready",
            ),
        )
        is True
    )
    source_blockers = [
        str(blocker)
        for blocker in (
            _nested(current_state, ("source_forward_evidence", "queue_blockers"))
            or ()
        )
        if str(blocker)
    ]
    if not official_ready or not source_blockers:
        return []
    return _dedupe(
        [
            "snerv_ready_official_tub_lf_hf_gate_contradicts_source_forward_queue_blockers",
            *source_blockers,
        ]
    )


def _launch_authority_contract(
    *,
    status: str,
    blockers: Sequence[str],
    command: Sequence[str],
) -> dict[str, Any]:
    runnable = bool(command) and not blockers and (
        status == "local_bounded_smoke_ready_no_authority"
    )
    return {
        "schema": PLANNER_ROW_LAUNCH_CONTRACT_SCHEMA,
        "queue_status_is_local_mlx_plan": True,
        "queue_status_is_runnable_plan": runnable,
        "queue_launch_step_count": 1 if runnable else 0,
        "queue_steps_retained_as_post_unblock_handoff": not runnable,
        "queue_launch_blockers": []
        if runnable
        else _dedupe(
            [
                *[str(blocker) for blocker in blockers if blocker],
                "snerv_lf_hf_replacement_queue_row_not_runnable",
            ]
        ),
        "queue_status_is_receiver_proof": False,
        "queue_status_is_cpu_replay_proof": False,
        "queue_status_is_exact_eval_authority": False,
        "source_queue_schema": SCHEMA,
        "source_row_schema": ROW_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _renderer_nondegenerate_unblock_blockers(blockers: Sequence[str]) -> list[str]:
    allowed = set(_RENDERER_NONDEGENERATE_UNBLOCK_ALLOWED_BLOCKERS)
    return _dedupe(
        str(blocker) for blocker in blockers if str(blocker) and str(blocker) not in allowed
    )


def _unblock_launch_authority_contract(
    *,
    unblock_kind: str,
    blockers: Sequence[str],
    command: Sequence[str],
) -> dict[str, Any]:
    runnable = bool(command) and not blockers
    return {
        "schema": UNBLOCK_LAUNCH_CONTRACT_SCHEMA,
        "queue_unblock_status_is_local_mlx_plan": True,
        "queue_unblock_status_is_runnable_plan": runnable,
        "queue_unblock_kind": unblock_kind,
        "queue_unblock_step_count": 1 if runnable else 0,
        "queue_unblock_blockers": []
        if runnable
        else _dedupe(
            [
                *[str(blocker) for blocker in blockers if blocker],
                "snerv_lf_hf_queue_unblock_command_not_runnable",
            ]
        ),
        "source_queue_schema": SCHEMA,
        "source_row_schema": ROW_SCHEMA,
        **QUEUE_FALSE_AUTHORITY,
    }


def _bounded_training_binding_contract(
    *,
    solution_family: str,
    command_kind: str,
    command: Sequence[str],
) -> dict[str, Any]:
    boundable_families = {
        _OFFICIAL_TUB_LF_HF_FAMILY,
        _LF_CONDITIONED_HF_FAMILY,
        "joint_lf_hf_factorized_codebook",
    }
    bound = (
        solution_family in boundable_families
        and command_kind == "bounded_snerv_training_smoke"
        and bool(command)
    )
    family_blocker = bounded_training_blocker_for_solution_family(solution_family)
    blockers: list[str] = []
    if not bound:
        if family_blocker:
            blockers.append(family_blocker)
        elif solution_family == _OFFICIAL_TUB_LF_HF_FAMILY:
            blockers.append("snerv_official_tub_lf_hf_bounded_training_smoke_not_bound")
        else:
            blockers.append("snerv_lf_hf_bounded_training_binding_missing")
    runner_actuator = None
    if bound:
        runner_actuator = {
            "kind": command_kind,
            "runner": "tools/run_compact_renderer_mlx_spine_runner.py",
            "consumes_queue_artifact": True,
            "consumes_solution_family": solution_family,
            "command_prefix": list(command[:4]),
        }
    return {
        "schema": BOUNDED_TRAINING_BINDING_CONTRACT_SCHEMA,
        "solution_family": solution_family,
        "runner_actuator_required": True,
        "runner_actuator_bound": bound,
        "runner_actuator": runner_actuator,
        "family_bounded_training_blocker": family_blocker,
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _candidate_terminal_renderer_blockers(
    current_state: Mapping[str, Any],
    *,
    candidate_id: str,
) -> list[str]:
    evidence = _nested(current_state, ("scorer_domain_evidence", "terminal_renderer_feedback_blockers_by_candidate"))
    evidence = evidence if isinstance(evidence, Mapping) else {}
    blockers: list[str] = []
    for key in (candidate_id, ""):
        raw = evidence.get(str(key))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            blockers.extend(str(blocker) for blocker in raw if blocker)
    return _dedupe(blockers)


def _bounded_snerv_smoke_command(
    campaign_row: Mapping[str, Any],
    *,
    current_state: Mapping[str, Any],
    queue_row_id: str,
    output_root: Path,
    queue_artifact_path: str | Path | None,
    solution_family: str,
) -> list[str]:
    command = [str(part) for part in campaign_row.get("command_argv") or ()]
    if not command:
        return []
    replacements = {
        "--planner-row-id": queue_row_id,
        "--modelsize-candidate-id": str(
            campaign_row.get("candidate_id") or "auto"
        ),
        "--num-pairs": "16",
        "--epochs": "128",
        "--snerv-score-aware-long-training-epochs": "128",
        "--snerv-score-aware-long-training-batch-pairs": "2",
        "--mlx-prefilter-scorer-batch-pairs": "1",
        "--mlx-prefilter-progress-every": "4",
        "--snerv-native-mlx-receiver-proof-timeout": "600",
        "--snerv-lf-hf-solution-family": str(solution_family),
        "--output-dir": (output_root / queue_row_id / "bounded_smoke").as_posix(),
        "--planner-row-queue-artifact": (
            Path(queue_artifact_path).as_posix()
            if queue_artifact_path is not None
            else (output_root / "snerv_lf_hf_replacement_queue.json").as_posix()
        ),
        "--scorer-space-step-guard-min-post-segnet-occupied-class-fraction": (
            SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_OCCUPIED_CLASS_FRACTION
        ),
        "--scorer-space-step-guard-min-post-segnet-target-class-coverage-fraction": (
            SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_TARGET_CLASS_COVERAGE_FRACTION
        ),
        "--scorer-space-step-guard-min-post-segnet-target-class-min-ratio": (
            SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_TARGET_CLASS_MIN_RATIO
        ),
        "--scorer-space-step-guard-max-post-segnet-target-class-ratio-drop": (
            SNERV_BOUNDED_SMOKE_MAX_POST_SEGNET_TARGET_CLASS_RATIO_DROP
        ),
        "--segnet-direct-live-rare-class-logit-weight": (
            SNERV_BOUNDED_SMOKE_SEGNET_RARE_CLASS_LOGIT_WEIGHT
        ),
        "--segnet-direct-live-target-mass-floor-weight": (
            SNERV_BOUNDED_SMOKE_SEGNET_TARGET_MASS_FLOOR_WEIGHT
        ),
        "--segnet-direct-live-target-min-ratio-floor-weight": (
            SNERV_BOUNDED_SMOKE_SEGNET_TARGET_MIN_RATIO_FLOOR_WEIGHT
        ),
        "--segnet-direct-live-escape-warmup-epochs": (
            SNERV_BOUNDED_SMOKE_SEGNET_ESCAPE_WARMUP_EPOCHS
        ),
        "--segnet-direct-live-escape-class-multiplier": (
            SNERV_BOUNDED_SMOKE_SEGNET_ESCAPE_CLASS_MULTIPLIER
        ),
        "--posenet-yuv6-geometry-tether-weight": (
            SNERV_BOUNDED_SMOKE_POSENET_YUV6_GEOMETRY_TETHER_WEIGHT
        ),
        "--scorer-space-step-guard-max-post-segnet-distribution-mae": (
            SNERV_BOUNDED_SMOKE_MAX_POST_SEGNET_DISTRIBUTION_MAE
        ),
        "--scorer-space-step-guard-max-post-posenet-yuv6-distribution-mae": (
            SNERV_BOUNDED_SMOKE_MAX_POST_POSENET_YUV6_DISTRIBUTION_MAE
        ),
        "--scorer-space-step-guard-max-post-posenet-yuv6-contrast-ratio": (
            SNERV_BOUNDED_SMOKE_MAX_POST_POSENET_YUV6_CONTRAST_RATIO
        ),
    }
    scorer_loop_qat_ready = (
        _nested(
            current_state,
            ("scorer_domain_evidence", "scorer_loop_qat_nondegenerate_evidence", "passed"),
        )
        is True
    )
    if scorer_loop_qat_ready:
        replacements.update(
            {
                "--snerv-scorer-loop-max-trials": "1",
                "--snerv-scorer-loop-pair-guard-min-score-improved-fraction": "0",
                "--snerv-scorer-loop-pair-guard-max-pose-worsened-fraction": "1",
                "--snerv-scorer-loop-max-archive-byte-growth": "0",
                "--snerv-scorer-loop-byte-growth-admission-mode": "hard_cap",
            }
        )
    state_dict_path = _bounded_snerv_official_state_dict_path(
        current_state,
        solution_family=solution_family,
    )
    if state_dict_path is not None:
        replacements["--snerv-official-trained-checkpoint-state-dict-path"] = (
            state_dict_path
        )
    boolean_flags = _SCORER_LOOP_QAT_BOOLEAN_FLAGS if scorer_loop_qat_ready else ()
    stripped_value_flags = set(_SCORER_LOOP_QAT_VALUE_FLAGS) if not scorer_loop_qat_ready else set()
    stripped_boolean_flags = (
        set(_SCORER_LOOP_QAT_BOOLEAN_FLAGS) if not scorer_loop_qat_ready else set()
    )
    out: list[str] = []
    idx = 0
    while idx < len(command):
        token = command[idx]
        if token in stripped_boolean_flags:
            idx += 1
            continue
        if token in stripped_value_flags:
            idx += 2 if idx + 1 < len(command) else 1
            continue
        out.append(token)
        if token in replacements and idx + 1 < len(command):
            out.append(replacements[token])
            idx += 2
            continue
        idx += 1
    present = set(command)
    for flag, value in replacements.items():
        if flag not in present:
            out.extend([flag, value])
    for flag in boolean_flags:
        if flag not in present:
            out.append(flag)
    existing_modelsize_feedback_paths = {
        out[idx + 1]
        for idx, token in enumerate(out[:-1])
        if token == "--modelsize-byte-cap-feedback-json"
    }
    for path in _bounded_snerv_extra_modelsize_feedback_paths(current_state):
        if path not in existing_modelsize_feedback_paths:
            out.extend(["--modelsize-byte-cap-feedback-json", path])
            existing_modelsize_feedback_paths.add(path)
    archive_skip_flag = "--skip-snerv-native-mlx-archive-export"
    if archive_skip_flag not in set(out):
        out.append(archive_skip_flag)
    return out


def _lf_conditioned_hf_residual_payload_proof_command(
    current_state: Mapping[str, Any],
    *,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    packet_path = _evidence_packet_path(
        current_state,
        ("value_domain_evidence", "hf_residual_payload_evidence"),
    )
    if not packet_path:
        return []
    output_dir = output_root / queue_row_id / "lf_conditioned_hf_residual_payload_proof"
    return [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_conditioned_hf_residual_payload_proof.py",
        "--packet",
        packet_path,
        "--pair-indices",
        _evidence_pair_indices_csv(
            current_state,
            ("value_domain_evidence", "hf_residual_payload_evidence"),
        ),
        "--output-json",
        (output_dir / "snerv_lf_conditioned_hf_residual_receiver_proof.json").as_posix(),
        "--output-payload",
        (output_dir / "snerv_lf_conditioned_hf_residual.slhr").as_posix(),
    ]


def _joint_lf_hf_codebook_payload_proof_command(
    current_state: Mapping[str, Any],
    *,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    packet_path = _evidence_packet_path(
        current_state,
        ("joint_codebook_evidence", "value_domain_evidence"),
    )
    if not packet_path:
        return []
    output_dir = output_root / queue_row_id / "joint_lf_hf_codebook_payload_proof"
    return [
        "uv",
        "run",
        "python",
        "tools/build_snerv_joint_lf_hf_codebook_payload_proof.py",
        "--packet",
        packet_path,
        "--pair-indices",
        _evidence_pair_indices_csv(
            current_state,
            ("joint_codebook_evidence", "value_domain_evidence"),
        ),
        "--output-json",
        (
            output_dir
            / "snerv_joint_lf_hf_factorized_codebook_receiver_proof.json"
        ).as_posix(),
        "--output-payload",
        (output_dir / "snerv_joint_lf_hf_factorized_codebook.sjlc").as_posix(),
    ]


def _lf_super_resolution_tiny_anchor_payload_proof_command(
    current_state: Mapping[str, Any],
    *,
    selected_evidence: Mapping[str, Any] | None = None,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    packet_path = _evidence_packet_path(
        current_state,
        ("lf_super_resolution_evidence", "value_domain_evidence"),
    )
    if not packet_path and selected_evidence is not None:
        packet_path = str(
            selected_evidence.get("packet_path")
            or selected_evidence.get("candidate_packet_path")
            or ""
        ).strip()
    if not packet_path:
        return []
    output_dir = output_root / queue_row_id / "lf_super_resolution_tiny_anchor_proof"
    return [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_super_resolution_tiny_anchor_payload_proof.py",
        "--packet",
        packet_path,
        "--pair-indices",
        _evidence_pair_indices_csv(
            current_state,
            ("lf_super_resolution_evidence", "value_domain_evidence"),
        ),
        "--output-json",
        (
            output_dir
            / "snerv_lf_super_resolution_tiny_anchor_receiver_proof.json"
        ).as_posix(),
        "--output-payload",
        (output_dir / "snerv_lf_super_resolution_tiny_anchor.slsr").as_posix(),
    ]


def _temporal_lf_predictor_payload_proof_command(
    current_state: Mapping[str, Any],
    *,
    selected_evidence: Mapping[str, Any] | None,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    packet_path = _evidence_packet_path(
        current_state,
        (
            "value_domain_evidence",
            "temporal_lf_predictor_evidence",
            "hf_residual_payload_evidence",
            "joint_codebook_evidence",
        ),
    )
    if not packet_path and selected_evidence is not None:
        packet_path = str(
            selected_evidence.get("packet_path")
            or selected_evidence.get("candidate_packet_path")
            or ""
        ).strip()
    if not packet_path:
        return []
    output_dir = output_root / queue_row_id / "temporal_lf_predictor_payload_proof"
    return [
        "uv",
        "run",
        "python",
        "tools/build_snerv_temporal_lf_predictor_payload_proof.py",
        "--packet",
        packet_path,
        "--pair-indices",
        _evidence_pair_indices_csv(
            current_state,
            (
                "value_domain_evidence",
                "temporal_lf_predictor_evidence",
                "hf_residual_payload_evidence",
                "joint_codebook_evidence",
            ),
        ),
        "--output-json",
        (output_dir / "snerv_temporal_lf_predictor_receiver_proof.json").as_posix(),
        "--output-payload",
        (output_dir / "snerv_temporal_lf_predictor.stlp").as_posix(),
    ]


def _spectral_band_allocator_payload_proof_command(
    current_state: Mapping[str, Any],
    *,
    selected_evidence: Mapping[str, Any] | None,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    packet_path = _evidence_packet_path(
        current_state,
        (
            "value_domain_evidence",
            "spectral_band_allocator_evidence",
            "hf_residual_payload_evidence",
            "joint_codebook_evidence",
        ),
    )
    if not packet_path and selected_evidence is not None:
        packet_path = str(
            selected_evidence.get("packet_path")
            or selected_evidence.get("candidate_packet_path")
            or ""
        ).strip()
    if not packet_path:
        return []
    output_dir = output_root / queue_row_id / "spectral_band_allocator_payload_proof"
    return [
        "uv",
        "run",
        "python",
        "tools/build_snerv_spectral_band_allocator_payload_proof.py",
        "--packet",
        packet_path,
        "--pair-indices",
        _evidence_pair_indices_csv(
            current_state,
            (
                "value_domain_evidence",
                "spectral_band_allocator_evidence",
                "hf_residual_payload_evidence",
                "joint_codebook_evidence",
            ),
        ),
        "--output-json",
        (
            output_dir
            / "snerv_score_tethered_spectral_band_allocator_receiver_proof.json"
        ).as_posix(),
        "--output-payload",
        (output_dir / "snerv_score_tethered_spectral_band_allocator.ssba").as_posix(),
    ]


def _lf_latent_hyperprior_payload_proof_command(
    current_state: Mapping[str, Any],
    *,
    selected_evidence: Mapping[str, Any] | None,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    packet_path = _evidence_packet_path(
        current_state,
        (
            "value_domain_evidence",
            "lf_latent_hyperprior_evidence",
            "temporal_lf_predictor_evidence",
            "lf_super_resolution_evidence",
        ),
    )
    if not packet_path and selected_evidence is not None:
        packet_path = str(
            selected_evidence.get("packet_path")
            or selected_evidence.get("candidate_packet_path")
            or ""
        ).strip()
    if not packet_path:
        return []
    output_dir = output_root / queue_row_id / "lf_latent_hyperprior_payload_proof"
    return [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_latent_hyperprior_payload_proof.py",
        "--packet",
        packet_path,
        "--pair-indices",
        _evidence_pair_indices_csv(
            current_state,
            (
                "value_domain_evidence",
                "lf_latent_hyperprior_evidence",
                "temporal_lf_predictor_evidence",
                "lf_super_resolution_evidence",
            ),
        ),
        "--output-json",
        (output_dir / "snerv_lf_latent_hyperprior_receiver_proof.json").as_posix(),
        "--output-payload",
        (output_dir / "snerv_lf_latent_hyperprior.slhp").as_posix(),
    ]


def _runtime_binding_proof_command(
    current_state: Mapping[str, Any],
    *,
    solution_family: str,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    evidence_key = _RUNTIME_BINDING_EVIDENCE_KEY_BY_FAMILY.get(solution_family)
    flag = proof_cli_flag_for_solution_family(solution_family)
    if not evidence_key or not flag:
        return []
    proof_path = str(
        _nested(current_state, (evidence_key, "source_path")) or ""
    ).strip()
    payload_path = str(
        _nested(current_state, (evidence_key, "payload_path")) or ""
    ).strip()
    payload_sha256 = str(
        _nested(current_state, (evidence_key, "payload_sha256")) or ""
    ).strip()
    if not proof_path or not payload_path or not payload_sha256:
        return []
    output_dir = output_root / queue_row_id / "runtime_binding_proof"
    return [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_hf_runtime_binding_proof.py",
        flag,
        proof_path,
        "--output-json",
        (output_dir / "snerv_lf_hf_runtime_binding_proof.json").as_posix(),
    ]


def _evidence_packet_path(
    current_state: Mapping[str, Any],
    evidence_keys: Sequence[str],
) -> str:
    for evidence_key in evidence_keys:
        packet_path = _nested(current_state, (evidence_key, "packet_path"))
        packet_path = str(packet_path or "").strip()
        if packet_path:
            return packet_path
    return ""


def _evidence_pair_indices_csv(
    current_state: Mapping[str, Any],
    evidence_keys: Sequence[str],
) -> str:
    raw: Any = None
    for evidence_key in evidence_keys:
        raw = _nested(current_state, (evidence_key, "pair_indices"))
        if raw:
            break
    values: list[int] = []
    if isinstance(raw, str):
        pieces: Sequence[Any] = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        pieces = raw
    else:
        pieces = ()
    for piece in pieces:
        if piece is None or str(piece).strip() == "":
            continue
        try:
            values.append(int(piece))
        except (TypeError, ValueError):
            continue
    if not values:
        values = [0, 1]
    return ",".join(str(value) for value in values)


def _first_unblock_command(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    for row in rows:
        command = _dedupe_command(row.get("unblock_command_argv") or ())
        if command:
            return command
    return []


def _global_blocker_row(
    *,
    output_root: Path,
    blocker: str,
    selected_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": ROW_SCHEMA,
        "queue_row_id": "snerv_lf_hf_replace_global_blocker",
        "lane_id": DEFAULT_LANE_ID,
        "source_campaign_row_id": None,
        "candidate_id": None,
        "candidate_class": "global_blocker",
        "solution_family": "lf_hf_replacement_queue_bootstrap",
        "planner_action": "attach_current_snerv_campaign_plan_before_candidate_emission",
        "learning_objective": "blocked before any prototype can be selected",
        "priority": 999,
        "status": "blocked_until_prerequisite_evidence",
        "blocked": True,
        "blockers": [str(blocker)],
        "selected_lf_payload_evidence": selected_evidence,
        "target_consumers": ["nerv_long_training_campaign_plan"],
        "command_argv": [],
        "unblock_command_argv": [],
        "output_root": output_root.as_posix(),
        **QUEUE_FALSE_AUTHORITY,
    }


def _lf_evidence_row(report: Mapping[str, Any], source_index: int) -> dict[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    schema = str(report.get("schema") or "")
    base = {
        "schema": "snerv_lf_hf_payload_evidence_ref.v1",
        "source_index": int(source_index),
        "source_schema": schema,
        "source_path": report.get("_source_path") or report.get("report_path"),
        "source_sha256": report.get("_source_sha256"),
        "authority": report.get("authority"),
        "axis_tag": report.get("axis_tag"),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if schema == "snerv_lf_payload_codec_sweep.v1":
        selected = report.get("selected_rate_only_row")
        selected = selected if isinstance(selected, Mapping) else {}
        return {
            **base,
            "evidence_kind": "lf_payload_codec_sweep",
            "plane_count": _positive_int(report.get("plane_count")),
            "plane_shapes": report.get("plane_shapes"),
            "raw_lf_bytes": _positive_int(report.get("raw_i64_bytes")),
            "lf_payload_bytes": _positive_int(selected.get("payload_bytes")),
            "selected_mode": selected.get("mode"),
            "baseline_payload_bytes": _positive_int(report.get("baseline_payload_bytes")),
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    if schema == "snerv_lf_payload_archive_recode.v1":
        lf_payload = report.get("lf_payload")
        lf_payload = lf_payload if isinstance(lf_payload, Mapping) else {}
        return {
            **base,
            "evidence_kind": "receiver_packet_lf_recode",
            "plane_count": _positive_int(report.get("lf_plane_count")),
            "raw_lf_bytes": _positive_int(_nested(lf_payload, ("source_header", "raw_bytes"))),
            "lf_payload_bytes": _positive_int(lf_payload.get("source_bytes")),
            "candidate_lf_payload_bytes": _positive_int(lf_payload.get("candidate_bytes")),
            "candidate_packet_bytes": _positive_int(_nested(report, ("candidate_packet", "bytes"))),
            "candidate_packet_path": _nested(report, ("candidate_packet", "path")),
            "selected_mode": report.get("mode"),
            "receiver_contract_satisfied": report.get("receiver_contract_satisfied") is True,
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    if schema == "snerv_lf_payload_recode_admission_plan.v1":
        selected = report.get("selected_row")
        selected = selected if isinstance(selected, Mapping) else {}
        return {
            **base,
            "evidence_kind": "lf_recode_admission_plan",
            "lf_payload_bytes": _positive_int(selected.get("lf_source_bytes")),
            "candidate_lf_payload_bytes": _positive_int(selected.get("lf_candidate_bytes")),
            "candidate_packet_bytes": _positive_int(selected.get("candidate_packet_bytes")),
            "candidate_packet_path": selected.get("candidate_packet_path"),
            "selected_mode": selected.get("mode"),
            "post_recode_over_waterline_bytes": _positive_int(
                selected.get("post_recode_over_waterline_bytes")
            ),
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    if schema == "snerv_checkpoint_archive_export.v1":
        codec_report = report.get("lf_payload_codec_selection_report")
        codec_report = codec_report if isinstance(codec_report, Mapping) else {}
        packet_section_bytes = report.get("packet_section_bytes")
        packet_section_bytes = (
            packet_section_bytes if isinstance(packet_section_bytes, Mapping) else {}
        )
        return {
            **base,
            "evidence_kind": "checkpoint_export_lf_payload_section",
            "packet_path": report.get("packet_path"),
            "packet_bytes": _positive_int(report.get("packet_bytes")),
            "packet_sha256": report.get("packet_sha256"),
            "raw_lf_bytes": _positive_int(
                codec_report.get("canonical_int64_raw_bytes")
                or codec_report.get("raw_bytes")
            ),
            "lf_payload_bytes": _positive_int(
                report.get("lf_payload_section_bytes")
                or packet_section_bytes.get("lf_payload")
                or codec_report.get("section_bytes")
                or codec_report.get("payload_bytes")
            ),
            "selected_mode": report.get("lf_payload_codec_selected")
            or report.get("lf_payload_codec"),
            "receiver_contract_satisfied": (
                report.get("receiver_contract_satisfied") is True
            ),
            "lf_payload_report_status": report.get("lf_payload_report_status"),
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    if schema == "snerv_official_dummy_lf_payload_codec_sweep.v1":
        selected = report.get("selected_rate_only_row")
        selected = selected if isinstance(selected, Mapping) else {}
        return {
            **base,
            "evidence_kind": "official_dummy_lf_receiver_section",
            "plane_count": _positive_int(report.get("lf_plane_count")),
            "raw_lf_bytes": _positive_int(report.get("raw_i64_bytes")),
            "lf_payload_bytes": _positive_int(selected.get("receiver_section_total_bytes")),
            "selected_mode": selected.get("mode"),
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    return None


def _selected_lf_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if _positive_int(row.get("lf_payload_bytes")) is not None]
    if not valid:
        return None
    selected = max(
        valid,
        key=lambda row: (
            int(row.get("lf_payload_bytes") or 0),
            int(row.get("raw_lf_bytes") or 0),
            str(row.get("source_path") or ""),
        ),
    )
    return dict(selected)


def _source_forward_state(
    source_forward_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in source_forward_artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("schema") == "snerv_official_mfu_hfr_tub_forward_parity.v1"
    ]
    if not artifacts:
        return {
            "schema": "snerv_lf_hf_source_forward_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_frame_replay_proven": False,
            "receiver_runtime_decode_proven": False,
            "frame_producing_official_payload_replay_proven": False,
            "receiver_frame_decode_consumes_output2": False,
            "full_tub_source_forward_parity_proven": False,
            "closed_campaign_blockers": [],
            "queue_blockers": ["snerv_lf_hf_source_forward_artifact_missing"],
            "blockers": ["snerv_lf_hf_source_forward_artifact_missing"],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        artifacts,
        key=lambda artifact: (
            str(artifact.get("generated_utc") or ""),
            str(artifact.get("_source_path") or ""),
        ),
    )
    replay = selected.get("receiver_payload_frame_replay")
    replay = replay if isinstance(replay, Mapping) else {}
    source_forward_proof = _source_forward_replay_proof(selected, replay)
    source_forward_proof_status = _source_forward_replay_proof_status(
        source_forward_proof
    )
    numerical_source_forward_proof_complete = bool(
        source_forward_proof_status[
            "source_forward_replay_numerical_proof_complete"
        ]
    )
    receiver_runtime_decode = replay.get("receiver_runtime_decode_proven") is True
    frame_payload_replay = (
        replay.get("frame_producing_official_payload_replay_proven") is True
    )
    frame_replay_proven = receiver_runtime_decode and frame_payload_replay
    receiver_consumes_output2 = replay.get("receiver_frame_decode_consumes_output2") is True
    export_binding = selected.get("official_checkpoint_export_binding_evidence")
    export_binding = export_binding if isinstance(export_binding, Mapping) else {}
    trained_mapping = selected.get("official_trained_checkpoint_mapping_manifest")
    trained_mapping = trained_mapping if isinstance(trained_mapping, Mapping) else {}
    servo_receipt = _first_source_forward_mapping(
        (
            "pair_local_distortion_servo_receipt",
            "nerv_pair_local_distortion_servo_receipt",
            "servo_receipt",
        ),
        selected,
        replay,
        export_binding,
        trained_mapping,
        source_forward_proof,
    )
    axis_trace_measurements = _source_forward_axis_trace_measurements(
        selected,
        replay,
        export_binding,
        trained_mapping,
        source_forward_proof,
    )
    trained_checkpoint_loaded = (
        selected.get("official_trained_checkpoint_loaded") is True
        or trained_mapping.get("official_trained_checkpoint_loaded") is True
    )
    state_dict_mapping_verified = (
        selected.get("official_trained_checkpoint_state_dict_mapping_verified") is True
        or trained_mapping.get("official_trained_checkpoint_state_dict_mapping_verified")
        is True
    )
    hfr_weight_mapping_proven = (
        selected.get("official_hfr_trained_checkpoint_weight_mapping_proven")
        is True
        or trained_mapping.get(
            "official_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
    )
    mfu_weight_mapping_proven = (
        selected.get("official_mfu_trained_checkpoint_weight_mapping_proven")
        is True
        or trained_mapping.get(
            "official_mfu_trained_checkpoint_weight_mapping_proven"
        )
        is True
    )
    combined_mfu_hfr_weight_mapping_proven = (
        selected.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven")
        is True
        or trained_mapping.get(
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
    )
    mfu_hfr_weight_mapping_proven = (
        combined_mfu_hfr_weight_mapping_proven
        or (hfr_weight_mapping_proven and mfu_weight_mapping_proven)
    )
    mfu_activation_payload_bound = (
        selected.get("official_mfu_receiver_activation_payload_bound") is True
        or trained_mapping.get("official_mfu_receiver_activation_payload_bound")
        is True
    )
    tub_temporal_encoder_weight_mapping_proven = (
        selected.get("official_tub_temporal_encoder_weight_mapping_proven") is True
        or trained_mapping.get("official_tub_temporal_encoder_weight_mapping_proven")
        is True
    )
    tub_output2_decoder_weight_mapping_proven = (
        selected.get("official_tub_output2_decoder_weight_mapping_proven") is True
        or trained_mapping.get("official_tub_output2_decoder_weight_mapping_proven")
        is True
    )
    native_receiver_state_mapping_proven = (
        selected.get("official_native_receiver_state_mapping_proven") is True
        or trained_mapping.get("official_native_receiver_state_mapping_proven")
        is True
    )
    state_dict_artifact = _source_forward_state_dict_value_artifact(selected)
    state_dict_path = str(state_dict_artifact.get("path") or "").strip()
    state_dict_file_present = state_dict_artifact.get("file_present") is True
    state_dict_sha256_matches_report = state_dict_artifact.get(
        "sha256_matches_report"
    )
    state_dict_bytes_match_report = state_dict_artifact.get("bytes_match_report")
    state_dict_npz_opened = state_dict_artifact.get("npz_opened") is True
    state_dict_member_names_match_report = state_dict_artifact.get(
        "member_names_match_report"
    )
    state_dict_value_artifact_ready = bool(
        state_dict_path
        and state_dict_file_present
        and state_dict_npz_opened
        and state_dict_mapping_verified
        and tub_temporal_encoder_weight_mapping_proven
        and tub_output2_decoder_weight_mapping_proven
        and state_dict_bytes_match_report is not False
        and state_dict_sha256_matches_report is not False
        and state_dict_member_names_match_report is not False
    )
    payload_bytes = _positive_int(replay.get("payload_bytes"))
    payload_sha256 = str(replay.get("payload_sha256") or "").strip()
    receiver_bound_export = bool(
        frame_replay_proven
        and receiver_consumes_output2
        and payload_bytes is not None
        and len(payload_sha256) == 64
    )
    export_bound = bool(
        selected.get("official_export_bound") is True
        or export_binding.get("official_export_bound") is True
    )
    full_tub_parity = selected.get("full_tub_source_forward_parity_proven") is True
    raw_source_blockers = {str(blocker) for blocker in selected.get("blockers") or ()}
    source_authority_conflicting_blockers = {
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
        "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
    }
    source_authority_blocked_by_raw_evidence = bool(
        raw_source_blockers.intersection(source_authority_conflicting_blockers)
    )
    source_authority = bool(
        full_tub_parity
        and numerical_source_forward_proof_complete
        and not source_authority_blocked_by_raw_evidence
        and state_dict_value_artifact_ready
        and (
            replay.get("source_forward_replay_authority") is True
            or selected.get("source_forward_replay_authority") is True
        )
    )
    tub_source_fixture_proven = (
        selected.get("official_tub_source_fixture_forward_parity_proven") is True
    )
    tub_source_fixture_closed = _tub_source_fixture_closed_blockers(selected)
    closed = []
    if export_bound:
        closed.append("snerv_official_mfu_hfr_tub_export_not_bound")
    if frame_replay_proven:
        closed.extend(
            [
                "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
                "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
            ]
        )
    if trained_checkpoint_loaded:
        closed.append("snerv_official_trained_checkpoint_state_dict_not_loaded")
    if state_dict_mapping_verified:
        closed.append("snerv_official_trained_checkpoint_state_dict_mapping_missing")
    if hfr_weight_mapping_proven:
        closed.append("snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete")
    if mfu_weight_mapping_proven:
        closed.append("snerv_official_trained_checkpoint_mfu_weight_mapping_incomplete")
        closed.append(
            "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping"
        )
    if mfu_hfr_weight_mapping_proven:
        closed.append("snerv_official_mfu_hfr_tub_weight_mapping_missing")
    if tub_temporal_encoder_weight_mapping_proven:
        closed.extend(
            [
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
            ]
        )
    if tub_output2_decoder_weight_mapping_proven:
        closed.append("snerv_official_tub_portable_output2_decoder_weight_mapping_missing")
    if tub_source_fixture_proven:
        closed.extend(tub_source_fixture_closed)
    if source_authority:
        closed.extend(
            [
                "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
            ]
        )
    closed.extend(str(blocker) for blocker in trained_mapping.get("closed_campaign_blockers") or ())
    closed = _dedupe(closed)
    trained_closed = set(closed)
    export_binding = _without_closed_source_forward_blockers_in_mapping(
        export_binding,
        closed=trained_closed,
    )
    trained_mapping = _without_closed_source_forward_blockers_in_mapping(
        trained_mapping,
        closed=trained_closed,
    )
    trained_mapping_blockers = [
        str(blocker)
        for blocker in (
            *(trained_mapping.get("blockers") or ()),
            *(selected.get("blockers") or ()),
        )
        if str(blocker)
        and str(blocker) not in trained_closed
        and (
            str(blocker).startswith("snerv_official_trained_checkpoint_")
            or str(blocker).startswith("snerv_official_tub_")
            or str(blocker).startswith("snerv_official_mfu_")
            or str(blocker) == "snerv_official_mfu_hfr_tub_weight_mapping_missing"
            or str(blocker) == "snerv_official_mfu_hfr_tub_source_forward_replay_missing"
        )
    ]
    queue_blockers: list[str] = []
    if not frame_replay_proven:
        queue_blockers.extend(
            [
                "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
                "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
            ]
        )
    if not export_bound:
        queue_blockers.append("snerv_official_mfu_hfr_tub_export_not_bound")
    if not receiver_consumes_output2:
        queue_blockers.append("snerv_official_tub_output2_receiver_frame_decode_not_bound")
    if not source_authority:
        queue_blockers.extend(
            [
                "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
            ]
        )
    if not numerical_source_forward_proof_complete:
        queue_blockers.append(
            "snerv_official_mfu_hfr_tub_numerical_source_forward_proof_missing"
        )
    if source_authority_blocked_by_raw_evidence:
        queue_blockers.extend(
            sorted(raw_source_blockers.intersection(source_authority_conflicting_blockers))
        )
    if (
        full_tub_parity
        and state_dict_mapping_verified
        and tub_temporal_encoder_weight_mapping_proven
        and not state_dict_value_artifact_ready
    ):
        queue_blockers.append(
            "snerv_official_trained_checkpoint_state_dict_value_artifact_missing"
        )
    if state_dict_file_present and not state_dict_npz_opened:
        queue_blockers.append(
            "snerv_official_trained_checkpoint_state_dict_value_artifact_npz_invalid"
        )
    if state_dict_member_names_match_report is False:
        queue_blockers.append(
            "snerv_official_trained_checkpoint_state_dict_value_artifact_member_mismatch"
        )
    if state_dict_sha256_matches_report is False:
        queue_blockers.append(
            "snerv_official_trained_checkpoint_state_dict_value_artifact_sha256_mismatch"
        )
    if state_dict_bytes_match_report is False:
        queue_blockers.append(
            "snerv_official_trained_checkpoint_state_dict_value_artifact_bytes_mismatch"
        )
    queue_blockers.extend(trained_mapping_blockers)
    return {
        "schema": "snerv_lf_hf_source_forward_evidence.v1",
        "artifact_count": len(artifacts),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path"),
        "source_sha256": selected.get("_source_sha256"),
        "receiver_payload_frame_replay_proven": frame_replay_proven,
        "receiver_runtime_decode_proven": receiver_runtime_decode,
        "frame_producing_official_payload_replay_proven": frame_payload_replay,
        "receiver_bound_export_proven": receiver_bound_export,
        "official_checkpoint_export_bound": export_bound,
        "official_checkpoint_export_binding_evidence": dict(export_binding) or None,
        "official_trained_checkpoint_loaded": trained_checkpoint_loaded,
        "official_trained_checkpoint_state_dict_mapping_verified": (
            state_dict_mapping_verified
        ),
        "official_trained_checkpoint_state_dict_value_artifact_ready": (
            state_dict_value_artifact_ready
        ),
        "official_trained_checkpoint_state_dict_path": (
            state_dict_path or None
        ),
        "official_trained_checkpoint_state_dict_slice_path": (
            state_dict_path or None
        ),
        "official_trained_checkpoint_state_dict_slice_present": bool(
            state_dict_artifact.get("present")
        ),
        "official_trained_checkpoint_state_dict_slice_file_present": (
            state_dict_file_present
        ),
        "official_trained_checkpoint_state_dict_slice_bytes": (
            state_dict_artifact.get("bytes")
        ),
        "official_trained_checkpoint_state_dict_slice_sha256": (
            state_dict_artifact.get("sha256")
        ),
        "official_trained_checkpoint_state_dict_slice_sha256_matches_report": (
            state_dict_sha256_matches_report
        ),
        "official_trained_checkpoint_state_dict_slice_bytes_match_report": (
            state_dict_bytes_match_report
        ),
        "official_trained_checkpoint_state_dict_slice_npz_opened": (
            state_dict_npz_opened
        ),
        "official_trained_checkpoint_state_dict_slice_member_names_match_report": (
            state_dict_member_names_match_report
        ),
        "official_trained_checkpoint_state_dict_slice_member_count": (
            state_dict_artifact.get("member_count")
        ),
        "official_trained_checkpoint_state_dict_slice_member_names": list(
            state_dict_artifact.get("member_names") or []
        ),
        "official_trained_checkpoint_state_dict_slice_runner_arg": (
            "--snerv-official-trained-checkpoint-state-dict-path"
            if state_dict_path
            else None
        ),
        "official_hfr_trained_checkpoint_weight_mapping_proven": (
            hfr_weight_mapping_proven
        ),
        "official_mfu_trained_checkpoint_weight_mapping_proven": (
            mfu_weight_mapping_proven
        ),
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": (
            mfu_hfr_weight_mapping_proven
        ),
        "official_tub_temporal_encoder_weight_mapping_proven": (
            tub_temporal_encoder_weight_mapping_proven
        ),
        "official_tub_output2_decoder_weight_mapping_proven": (
            tub_output2_decoder_weight_mapping_proven
        ),
        "official_mfu_receiver_activation_payload_bound": mfu_activation_payload_bound,
        "official_tub_receiver_activation_payload_bound": (
            selected.get("official_tub_receiver_activation_payload_bound") is True
            or trained_mapping.get("official_tub_receiver_activation_payload_bound")
            is True
        ),
        "official_native_receiver_state_mapping_proven": (
            native_receiver_state_mapping_proven
        ),
        "official_trained_checkpoint_mapping_manifest": dict(trained_mapping) or None,
        "receiver_frame_decode_consumes_output2": receiver_consumes_output2,
        "official_tub_source_fixture_forward_parity_proven": tub_source_fixture_proven,
        "tub_source_fixture_closed_blockers": tub_source_fixture_closed,
        "full_tub_source_forward_parity_proven": full_tub_parity,
        "source_forward_replay_authority": source_authority,
        "source_forward_replay_proof": (
            dict(source_forward_proof) if isinstance(source_forward_proof, Mapping) else None
        ),
        "source_forward_replay_proof_status": source_forward_proof_status,
        "source_forward_replay_numerical_proof_complete": (
            numerical_source_forward_proof_complete
        ),
        "pair_local_distortion_servo_receipt": servo_receipt,
        "pr95_distortion_axis_trace_measurements": axis_trace_measurements,
        "decoded_frames_shape": replay.get("decoded_frames_shape"),
        "decoded_frames_sha256": replay.get("decoded_frames_sha256"),
        "payload_bytes": payload_bytes,
        "payload_sha256": replay.get("payload_sha256"),
        "closed_campaign_blockers": closed,
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*(selected.get("blockers") or ()), *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _without_closed_source_forward_blockers_in_mapping(
    value: Mapping[str, Any],
    *,
    closed: set[str],
) -> dict[str, Any]:
    def scrub(raw: Any, key: str | None = None) -> Any:
        if isinstance(raw, Mapping):
            return {str(k): scrub(v, str(k)) for k, v in raw.items()}
        if isinstance(raw, list):
            if key == "blockers" or (key or "").endswith("_blockers"):
                return [
                    str(item)
                    for item in raw
                    if str(item) and str(item).removeprefix("source_parity:") not in closed
                ]
            return [scrub(item) for item in raw]
        return raw

    scrubbed = scrub(value)
    assert isinstance(scrubbed, dict)
    return scrubbed


def _bounded_snerv_official_state_dict_path(
    current_state: Mapping[str, Any],
    *,
    solution_family: str,
) -> str | None:
    if solution_family != "official_tub_lf_hf_decoder_replacement":
        return None
    evidence = current_state.get("source_forward_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if evidence.get("source_forward_replay_authority") is not True:
        return None
    if evidence.get("official_trained_checkpoint_state_dict_value_artifact_ready") is not True:
        return None
    path = _existing_file_text(
        evidence.get("official_trained_checkpoint_state_dict_path")
        or evidence.get("official_trained_checkpoint_state_dict_slice_path")
    )
    if path is None:
        return None
    return path.as_posix()


def _bounded_snerv_extra_modelsize_feedback_paths(
    current_state: Mapping[str, Any],
) -> list[str]:
    paths: list[str] = []
    for section_name in ("scorer_domain_evidence",):
        section = current_state.get(section_name)
        section = section if isinstance(section, Mapping) else {}
        path = _existing_file_text(section.get("source_path"))
        if path is not None:
            paths.append(path.as_posix())
    return _dedupe(paths)


def _source_forward_state_dict_value_artifact(
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    sources: list[Mapping[str, Any]] = [selected]
    for key in (
        "official_tub_source_forward_replay",
        "official_checkpoint_export_binding_evidence",
        "official_checkpoint_export_binding",
    ):
        section = selected.get(key)
        if isinstance(section, Mapping):
            sources.append(section)
    source_root = _path_parent_or_none(selected.get("_source_path"))
    for source in sources:
        artifact = source.get("official_trained_checkpoint_state_dict_artifact")
        if isinstance(artifact, Mapping):
            candidate = _state_dict_candidate_from_mapping(
                artifact,
                source_root=source_root,
            )
            if candidate["present"]:
                return candidate
        candidate = _state_dict_candidate_from_mapping(
            source,
            source_root=source_root,
        )
        if candidate["present"]:
            return candidate
    return {
        "present": False,
        "path": None,
        "file_present": False,
        "bytes": None,
        "sha256": None,
        "sha256_matches_report": None,
        "bytes_match_report": None,
        "npz_opened": False,
        "member_names_match_report": None,
        "member_count": None,
        "member_names": [],
    }


def _state_dict_candidate_from_mapping(
    source: Mapping[str, Any],
    *,
    source_root: Path | None,
) -> dict[str, Any]:
    raw_path = (
        source.get("path")
        or source.get("official_trained_checkpoint_state_dict_path")
        or source.get("official_trained_checkpoint_state_dict_slice_path")
        or source.get("snerv_official_trained_checkpoint_state_dict_path")
        or source.get("snerv_official_trained_checkpoint_state_dict_slice_path")
    )
    if not raw_path:
        return {
            "present": False,
            "path": None,
            "file_present": False,
            "bytes": None,
            "sha256": None,
            "sha256_matches_report": None,
            "bytes_match_report": None,
            "npz_opened": False,
            "member_names_match_report": None,
            "member_count": None,
            "member_names": [],
        }
    path = _resolve_maybe_relative_path(raw_path, source_root=source_root)
    file_present = path.is_file()
    reported_sha = str(
        source.get("sha256")
        or source.get("official_trained_checkpoint_state_dict_slice_sha256")
        or source.get("snerv_official_trained_checkpoint_state_dict_slice_sha256")
        or ""
    ).strip()
    actual_sha = None
    if file_present:
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_matches_report = (
        None
        if not actual_sha or len(reported_sha) != 64
        else actual_sha == reported_sha
    )
    names = (
        source.get("member_names")
        or source.get("official_trained_checkpoint_state_dict_slice_member_names")
        or source.get("snerv_official_trained_checkpoint_state_dict_slice_member_names")
        or []
    )
    member_names = (
        [str(name) for name in names]
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray))
        else []
    )
    reported_bytes = (
        _positive_int(source.get("bytes"))
        or _positive_int(source.get("official_trained_checkpoint_state_dict_slice_bytes"))
        or _positive_int(source.get("snerv_official_trained_checkpoint_state_dict_slice_bytes"))
    )
    actual_bytes = int(path.stat().st_size) if file_present else None
    bytes_match_report = (
        None if actual_bytes is None or reported_bytes is None else actual_bytes == reported_bytes
    )
    bytes_value = reported_bytes
    if bytes_value is None and file_present:
        bytes_value = actual_bytes
    member_count = (
        _positive_int(source.get("member_count"))
        or _positive_int(
            source.get("official_trained_checkpoint_state_dict_slice_member_count")
        )
        or _positive_int(
            source.get("snerv_official_trained_checkpoint_state_dict_slice_member_count")
        )
    )
    if member_count is None and member_names:
        member_count = len(member_names)
    actual_member_names: list[str] = []
    npz_opened = False
    if file_present:
        try:
            with zipfile.ZipFile(path, "r") as zf:
                actual_member_names = sorted(str(name) for name in zf.namelist())
            npz_opened = True
        except (OSError, zipfile.BadZipFile):
            actual_member_names = []
            npz_opened = False
    member_names_match_report = None
    if npz_opened:
        member_names_match_report = (
            actual_member_names == sorted(member_names) if member_names else True
        )
        member_names = actual_member_names
        member_count = len(actual_member_names)
    return {
        "present": True,
        "path": path.as_posix(),
        "file_present": file_present,
        "bytes": bytes_value,
        "sha256": actual_sha or (reported_sha if len(reported_sha) == 64 else None),
        "sha256_matches_report": sha_matches_report,
        "bytes_match_report": bytes_match_report,
        "npz_opened": npz_opened,
        "member_names_match_report": member_names_match_report,
        "member_count": member_count,
        "member_names": member_names,
    }


def _existing_file_text(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_file():
        return path.resolve(strict=False)
    return None


def _resolve_maybe_relative_path(value: Any, *, source_root: Path | None) -> Path:
    raw = Path(str(value)).expanduser()
    path = raw if raw.is_absolute() else (source_root / raw if source_root else raw)
    return path.resolve(strict=False)


def _path_parent_or_none(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value)).expanduser().resolve(strict=False).parent


def _tub_source_fixture_closed_blockers(selected: Mapping[str, Any]) -> list[str]:
    closed: list[str] = [
        str(blocker)
        for blocker in selected.get("tub_source_fixture_closed_blockers") or ()
        if blocker
    ]
    nested = selected.get("official_tub_source_forward_replay")
    nested = nested if isinstance(nested, Mapping) else {}
    if (
        nested.get("schema") == "snerv_official_tub_source_forward_replay.v1"
        and nested.get("official_tub_temporal_encoder_output2_source_fixture_replay_passed")
        is True
    ):
        closed.extend(str(blocker) for blocker in nested.get("closed_blockers") or ())
        for key in (
            "temporal_path",
            "portable_output2_fusion",
            "frame_reconstruction_equivalence",
        ):
            section = nested.get(key)
            if isinstance(section, Mapping):
                closed.extend(str(blocker) for blocker in section.get("closed_blockers") or ())
    for blocker in tuple(closed):
        closed.extend(_TUB_SOURCE_FIXTURE_CLOSED_BLOCKER_ALIASES.get(blocker, ()))
    return _dedupe(closed)


def _scorer_domain_state(
    candidate_feedback_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in candidate_feedback_rows
        if isinstance(row, Mapping) and row.get("schema") == "nerv_candidate_feedback_row.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_lf_hf_scorer_domain_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_created_utc": None,
            "source_path": None,
            "source_sha256": None,
            "scorer_domain_tether_proof_passed": False,
            "required_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
            "metric_health": {},
            "missing_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
            "lambda_inactive_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
            "closed_campaign_blockers": [],
            "queue_blockers": ["snerv_scorer_input_distribution_guard_missing"],
            "blockers": ["snerv_lf_hf_scorer_domain_candidate_feedback_missing"],
            **QUEUE_FALSE_AUTHORITY,
        }
    def _guard_rank(row: Mapping[str, Any]) -> tuple[int, int, int, int, str, str]:
        health = row.get("snerv_scorer_domain_tether_health")
        health = health if isinstance(health, Mapping) else {}
        metric_health = health.get("metric_health")
        metric_health = metric_health if isinstance(metric_health, Mapping) else {}
        metrics_present = all(
            isinstance(metric_health.get(metric), Mapping)
            and metric_health[metric].get("metric_observed") is True
            and metric_health[metric].get("lambda_active_observed") is True
            for metric in _SCORER_DOMAIN_REQUIRED_METRICS
        )
        guard = row.get("snerv_scorer_input_distribution_guard_proof")
        guard = guard if isinstance(guard, Mapping) else {}
        qat_nondegenerate = _snerv_scorer_loop_qat_nondegenerate_evidence(row)
        return (
            1 if row.get("snerv_scorer_input_distribution_guard_proof_passed") is True else 0,
            1 if row.get("snerv_scorer_domain_tether_passed") is True else 0,
            1 if health.get("passed") is True and metrics_present else 0,
            1 if qat_nondegenerate.get("passed") is True else 0,
            str(row.get("created_utc") or row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("source_report_path") or ""),
        )

    selected = max(rows, key=_guard_rank)
    terminal_renderer_blockers_by_candidate: dict[str, list[str]] = {}
    for row in rows:
        terminal_blockers = _dedupe(
            [
                str(blocker)
                for blocker in (row.get("blockers") or ())
                if str(blocker) in _TERMINAL_RENDERER_FEEDBACK_BLOCKERS
            ]
        )
        if not terminal_blockers:
            continue
        key = str(row.get("candidate_id") or "")
        terminal_renderer_blockers_by_candidate[key] = _dedupe(
            [
                *(terminal_renderer_blockers_by_candidate.get(key) or ()),
                *terminal_blockers,
            ]
        )
    health = selected.get("snerv_scorer_domain_tether_health")
    health = health if isinstance(health, Mapping) else {}
    metric_health = health.get("metric_health")
    metric_health = metric_health if isinstance(metric_health, Mapping) else {}
    missing_metrics: list[str] = []
    lambda_inactive_metrics: list[str] = []
    for metric in _SCORER_DOMAIN_REQUIRED_METRICS:
        metric_row = metric_health.get(metric)
        metric_row = metric_row if isinstance(metric_row, Mapping) else {}
        if metric_row.get("metric_observed") is not True:
            missing_metrics.append(metric)
        if metric_row.get("lambda_active_observed") is not True:
            lambda_inactive_metrics.append(metric)
    explicit_blockers = _dedupe(
        [
            *(selected.get("snerv_scorer_domain_tether_blockers") or ()),
            *(health.get("blockers") or ()),
        ]
    )
    tether_proof_passed = bool(
        selected.get("snerv_scorer_domain_tether_passed") is True
        and health.get("passed") is True
        and not missing_metrics
        and not lambda_inactive_metrics
        and not explicit_blockers
    )
    guard_proof = selected.get("snerv_scorer_input_distribution_guard_proof")
    guard_proof = guard_proof if isinstance(guard_proof, Mapping) else {}
    guard_proof_passed = bool(
        selected.get("snerv_scorer_input_distribution_guard_proof_passed") is True
        and guard_proof.get("passed") is True
    )
    qat_nondegenerate_evidence = _snerv_scorer_loop_qat_nondegenerate_evidence(
        selected
    )
    blockers: list[str] = []
    if not guard_proof_passed:
        blockers.append("snerv_scorer_input_distribution_guard_missing")
    if not tether_proof_passed and missing_metrics:
        blockers.append("snerv_scorer_domain_tether_missing_telemetry")
    if not tether_proof_passed and lambda_inactive_metrics:
        blockers.append("snerv_scorer_domain_tether_lambda_inactive_telemetry")
    blockers.extend(explicit_blockers)
    blockers.extend(
        str(blocker)
        for blocker in guard_proof.get("blockers") or ()
        if str(blocker)
        and str(blocker)
        != "snerv_scorer_input_distribution_guard_not_required_by_feedback"
    )
    return {
        "schema": "snerv_lf_hf_scorer_domain_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_created_utc": selected.get("created_utc"),
        "source_path": (
            selected.get("_source_path")
            or selected.get("_candidate_feedback_source_path")
            or selected.get("source_report_path")
        ),
        "source_sha256": (
            selected.get("_source_sha256")
            or selected.get("_candidate_feedback_source_sha256")
            or selected.get("source_report_sha256")
        ),
        "candidate_id": selected.get("candidate_id"),
        "family": selected.get("family"),
        "scorer_domain_tether_proof_passed": tether_proof_passed,
        "scorer_input_distribution_guard_proof_passed": guard_proof_passed,
        "scorer_input_distribution_guard_proof": dict(guard_proof) or None,
        "scorer_loop_qat_nondegenerate_evidence": qat_nondegenerate_evidence,
        "scorer_loop_qat_ready": qat_nondegenerate_evidence["passed"],
        "terminal_renderer_feedback_blockers_by_candidate": (
            terminal_renderer_blockers_by_candidate
        ),
        "required_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
        "metric_health": {str(k): v for k, v in metric_health.items()},
        "missing_metrics": missing_metrics,
        "lambda_inactive_metrics": lambda_inactive_metrics,
        "closed_campaign_blockers": (
            list(_SCORER_DOMAIN_CLOSED_BLOCKERS) if guard_proof_passed else []
        ),
        "queue_blockers": (
            [] if guard_proof_passed else ["snerv_scorer_input_distribution_guard_missing"]
        ),
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _snerv_scorer_loop_qat_nondegenerate_evidence(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    proof = row.get("snerv_renderer_nondegenerate_proof")
    proof = proof if isinstance(proof, Mapping) else {}
    explicit_blockers = _dedupe(
        [
            *(row.get("snerv_renderer_nondegenerate_blockers") or ()),
            *(proof.get("blockers") or ()),
        ]
    )
    measured_num_pairs = _positive_int(
        proof.get("measured_num_pairs")
        or proof.get("pair_count")
        or proof.get("num_pairs")
        or row.get("measured_num_pairs")
        or row.get("candidate_num_pairs")
    )
    occupied_fraction = _first_finite_float(
        proof,
        row,
        keys=(
            "segnet_direct_live_max_candidate_occupied_class_fraction",
            "candidate_occupied_class_fraction",
            "train_direct_live_max_candidate_occupied_class_fraction",
            "receiver_candidate_occupied_class_fraction",
            "post_export_receiver_segnet_candidate_occupied_class_fraction",
        ),
    )
    target_class_coverage_fraction = _first_finite_float(
        proof,
        row,
        keys=(
            "segnet_direct_live_max_candidate_target_class_coverage_fraction",
            "candidate_target_class_coverage_fraction",
            "train_direct_live_max_candidate_target_class_coverage_fraction",
            "receiver_candidate_target_class_coverage_fraction",
            "post_export_receiver_segnet_candidate_target_class_coverage_fraction",
        ),
    )
    min_occupied_fraction = float(
        SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_OCCUPIED_CLASS_FRACTION
    )
    min_target_class_coverage_fraction = float(
        SNERV_BOUNDED_SMOKE_MIN_POST_SEGNET_TARGET_CLASS_COVERAGE_FRACTION
    )
    blockers: list[str] = []
    if not proof:
        blockers.append("snerv_renderer_nondegenerate_measured_qat_evidence_missing")
    if row.get("snerv_renderer_nondegenerate_proof_passed") is not True:
        blockers.append("snerv_renderer_nondegenerate_proof_not_passed_for_qat")
    if proof and proof.get("passed") is not True:
        blockers.append("snerv_renderer_nondegenerate_proof_failed_for_qat")
    if (
        measured_num_pairs is None
        or measured_num_pairs < SNERV_SCORER_LOOP_QAT_MIN_RENDERER_PAIR_COUNT
    ):
        blockers.append("snerv_renderer_nondegenerate_qat_min16_pairs_missing")
    if occupied_fraction is None:
        blockers.append(
            "snerv_renderer_nondegenerate_qat_occupied_class_fraction_missing"
        )
    elif occupied_fraction < min_occupied_fraction:
        blockers.append("snerv_renderer_nondegenerate_qat_candidate_argmax_collapsed")
    if target_class_coverage_fraction is None:
        blockers.append(
            "snerv_renderer_nondegenerate_qat_target_class_coverage_missing"
        )
    elif target_class_coverage_fraction < min_target_class_coverage_fraction:
        blockers.append(
            "snerv_renderer_nondegenerate_qat_target_class_coverage_collapsed"
        )
    blockers = _dedupe([*blockers, *explicit_blockers])
    return {
        "schema": "snerv_scorer_loop_qat_nondegenerate_evidence.v1",
        "required": True,
        "proof_attached": bool(proof),
        "proof_passed": bool(
            row.get("snerv_renderer_nondegenerate_proof_passed") is True
            and proof.get("passed") is True
        ),
        "measured_num_pairs": measured_num_pairs,
        "min_required_pair_count": SNERV_SCORER_LOOP_QAT_MIN_RENDERER_PAIR_COUNT,
        "segnet_candidate_occupied_class_fraction": occupied_fraction,
        "min_segnet_candidate_occupied_class_fraction": min_occupied_fraction,
        "segnet_candidate_target_class_coverage_fraction": (
            target_class_coverage_fraction
        ),
        "min_segnet_candidate_target_class_coverage_fraction": (
            min_target_class_coverage_fraction
        ),
        "passed": not blockers,
        "blockers": blockers,
        **QUEUE_FALSE_AUTHORITY,
    }


def _value_domain_state(
    value_domain_xray_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in value_domain_xray_reports
        if isinstance(row, Mapping)
        and row.get("schema") == "snerv_receiver_value_domain_xray.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_lf_hf_value_domain_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_decode_sample_proven": False,
            "value_domain_noncollapse_proof_passed": False,
            "closed_campaign_blockers": [],
            "queue_blockers": ["snerv_lf_conditioned_hf_value_domain_xray_missing"],
            "blockers": ["snerv_lf_conditioned_hf_value_domain_xray_missing"],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    decode_sample = selected.get("receiver_payload_decode_sample_proven") is True
    noncollapse = selected.get("value_domain_noncollapse_proof_passed") is True
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker) != "snerv_receiver_value_domain_xray_false_authority"
    ]
    closed = [
        blocker
        for blocker in selected.get("closed_campaign_blockers") or ()
        if blocker in _VALUE_DOMAIN_CLOSED_BLOCKERS
    ]
    queue_blockers: list[str] = []
    if not decode_sample:
        queue_blockers.append(
            "snerv_lf_conditioned_hf_receiver_value_domain_sample_decode_missing"
        )
    if not noncollapse:
        queue_blockers.append(
            "snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing"
        )
    queue_blockers.extend(selected_blockers)
    return {
        "schema": "snerv_lf_hf_value_domain_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "packet_path": selected.get("packet_path"),
        "packet_bytes": _positive_int(selected.get("packet_bytes")),
        "packet_sha256": selected.get("packet_sha256"),
        "pair_indices": selected.get("pair_indices"),
        "sample_shape_b2chw": selected.get("sample_shape_b2chw"),
        "value_domain_sample_status": selected.get("value_domain_sample_status"),
        "receiver_payload_decode_sample_proven": decode_sample,
        "value_domain_noncollapse_proof_passed": noncollapse,
        "verdict": selected.get("verdict"),
        "official_skip_high_value_domain": selected.get(
            "official_skip_high_value_domain"
        ),
        "official_scalar_skip_high_value_domain_scan": selected.get(
            "official_scalar_skip_high_value_domain_scan"
        ),
        "recommended_next_actions": [
            str(action) for action in selected.get("recommended_next_actions") or ()
        ],
        "closed_campaign_blockers": _dedupe(closed) if noncollapse else [],
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*selected_blockers, *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _hf_residual_payload_state(
    hf_residual_receiver_payload_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in hf_residual_receiver_payload_proofs
        if isinstance(row, Mapping)
        and row.get("schema")
        == "snerv_lf_conditioned_hf_residual_receiver_proof.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_lf_conditioned_hf_residual_payload_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_implemented": False,
            "receiver_decode_proven": False,
            "section_native_byte_telemetry_present": False,
            "closed_campaign_blockers": [],
            "queue_blockers": [
                "snerv_hf_residual_generator_receiver_payload_not_implemented"
            ],
            "blockers": [
                "snerv_hf_residual_generator_receiver_payload_not_implemented"
            ],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    implemented = selected.get("receiver_payload_implemented") is True
    decoded = selected.get("receiver_decode_proven") is True
    telemetry = selected.get("section_native_byte_telemetry_present") is True
    proof_passed = implemented and decoded and telemetry
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker)
        != "snerv_lf_conditioned_hf_residual_payload_false_authority"
    ]
    closed = [
        blocker
        for blocker in selected.get("closed_campaign_blockers") or ()
        if blocker in _HF_RESIDUAL_PAYLOAD_CLOSED_BLOCKERS
    ]
    queue_blockers: list[str] = []
    if not proof_passed:
        queue_blockers.append(
            "snerv_hf_residual_generator_receiver_payload_not_implemented"
        )
    queue_blockers.extend(selected_blockers)
    return {
        "schema": "snerv_lf_conditioned_hf_residual_payload_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "packet_path": selected.get("packet_path"),
        "source_packet_sha256": selected.get("source_packet_sha256"),
        "payload_path": selected.get("payload_path"),
        "payload_bytes": _positive_int(selected.get("payload_bytes")),
        "payload_sha256": selected.get("payload_sha256"),
        "lf_anchor_bytes": _positive_int(selected.get("lf_anchor_bytes")),
        "hf_residual_bytes": _positive_int(selected.get("hf_residual_bytes")),
        "compressed_payload_bytes": _positive_int(
            selected.get("compressed_payload_bytes")
        ),
        "pair_indices": selected.get("pair_indices"),
        "sample_shape_b2chw": selected.get("sample_shape_b2chw"),
        "receiver_payload_implemented": implemented,
        "receiver_decode_proven": decoded,
        "section_native_byte_telemetry_present": telemetry,
        "closed_campaign_blockers": _dedupe(closed) if proof_passed else [],
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*selected_blockers, *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _joint_codebook_state(
    joint_codebook_receiver_payload_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in joint_codebook_receiver_payload_proofs
        if isinstance(row, Mapping)
        and row.get("schema")
        == "snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_joint_lf_hf_factorized_codebook_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_implemented": False,
            "receiver_decode_proven": False,
            "numpy_receiver_decode": False,
            "section_native_byte_telemetry_present": False,
            "closed_campaign_blockers": [],
            "queue_blockers": list(_JOINT_CODEBOOK_CLOSED_BLOCKERS),
            "blockers": list(_JOINT_CODEBOOK_CLOSED_BLOCKERS),
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    implemented = selected.get("receiver_payload_implemented") is True
    decoded = selected.get("receiver_decode_proven") is True
    numpy_decode = selected.get("numpy_receiver_decode") is True
    telemetry = selected.get("section_native_byte_telemetry_present") is True
    proof_passed = implemented and decoded and numpy_decode and telemetry
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker) != "snerv_joint_lf_hf_factorized_codebook_false_authority"
    ]
    closed = [
        blocker
        for blocker in selected.get("closed_campaign_blockers") or ()
        if blocker in _JOINT_CODEBOOK_CLOSED_BLOCKERS
    ]
    queue_blockers: list[str] = []
    if not proof_passed:
        queue_blockers.extend(_JOINT_CODEBOOK_CLOSED_BLOCKERS)
    queue_blockers.extend(selected_blockers)
    return {
        "schema": "snerv_joint_lf_hf_factorized_codebook_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "packet_path": selected.get("packet_path"),
        "source_packet_sha256": selected.get("source_packet_sha256"),
        "payload_path": selected.get("payload_path"),
        "payload_bytes": _positive_int(selected.get("payload_bytes")),
        "payload_sha256": selected.get("payload_sha256"),
        "codebook_raw_bytes": _positive_int(selected.get("codebook_raw_bytes")),
        "index_raw_bytes": _positive_int(selected.get("index_raw_bytes")),
        "compressed_payload_bytes": _positive_int(
            selected.get("compressed_payload_bytes")
        ),
        "codebook_entry_count": _positive_int(selected.get("codebook_entry_count")),
        "block_count": _positive_int(selected.get("block_count")),
        "pair_indices": selected.get("pair_indices"),
        "sample_shape_b2chw": selected.get("sample_shape_b2chw"),
        "receiver_payload_implemented": implemented,
        "receiver_decode_proven": decoded,
        "numpy_receiver_decode": numpy_decode,
        "section_native_byte_telemetry_present": telemetry,
        "closed_campaign_blockers": _dedupe(closed) if proof_passed else [],
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*selected_blockers, *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _temporal_lf_predictor_state(
    temporal_lf_predictor_receiver_payload_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in temporal_lf_predictor_receiver_payload_proofs
        if isinstance(row, Mapping)
        and row.get("schema") == "snerv_temporal_lf_predictor_receiver_proof.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_temporal_lf_predictor_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_implemented": False,
            "receiver_decode_proven": False,
            "numpy_receiver_decode": False,
            "correction_stream_byte_charged": False,
            "section_native_byte_telemetry_present": False,
            "closed_campaign_blockers": [],
            "queue_blockers": list(_TEMPORAL_LF_PREDICTOR_CLOSED_BLOCKERS),
            "blockers": list(_TEMPORAL_LF_PREDICTOR_CLOSED_BLOCKERS),
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    implemented = selected.get("receiver_payload_implemented") is True
    decoded = selected.get("receiver_decode_proven") is True
    numpy_decode = selected.get("numpy_receiver_decode") is True
    correction_charged = selected.get("correction_stream_byte_charged") is True
    telemetry = selected.get("section_native_byte_telemetry_present") is True
    proof_passed = (
        implemented and decoded and numpy_decode and correction_charged and telemetry
    )
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker) != "snerv_temporal_lf_predictor_payload_false_authority"
    ]
    closed = [
        blocker
        for blocker in selected.get("closed_campaign_blockers") or ()
        if blocker in _TEMPORAL_LF_PREDICTOR_CLOSED_BLOCKERS
    ]
    queue_blockers: list[str] = []
    if not proof_passed:
        queue_blockers.extend(_TEMPORAL_LF_PREDICTOR_CLOSED_BLOCKERS)
    queue_blockers.extend(selected_blockers)
    return {
        "schema": "snerv_temporal_lf_predictor_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "packet_path": selected.get("packet_path"),
        "source_packet_sha256": selected.get("source_packet_sha256"),
        "payload_path": selected.get("payload_path"),
        "payload_bytes": _positive_int(selected.get("payload_bytes")),
        "payload_sha256": selected.get("payload_sha256"),
        "first_lf_anchor_bytes": _positive_int(
            selected.get("first_lf_anchor_bytes")
        ),
        "correction_stream_raw_bytes": _positive_int(
            selected.get("correction_stream_raw_bytes")
        ),
        "compressed_payload_bytes": _positive_int(
            selected.get("compressed_payload_bytes")
        ),
        "pair_indices": selected.get("pair_indices"),
        "sample_shape_b2chw": selected.get("sample_shape_b2chw"),
        "lf_shape_b2chw": selected.get("lf_shape_b2chw"),
        "receiver_payload_implemented": implemented,
        "receiver_decode_proven": decoded,
        "numpy_receiver_decode": numpy_decode,
        "correction_stream_byte_charged": correction_charged,
        "section_native_byte_telemetry_present": telemetry,
        "closed_campaign_blockers": _dedupe(closed) if proof_passed else [],
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*selected_blockers, *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _lf_super_resolution_state(
    lf_super_resolution_receiver_payload_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in lf_super_resolution_receiver_payload_proofs
        if isinstance(row, Mapping)
        and row.get("schema")
        == "snerv_lf_super_resolution_tiny_anchor_receiver_proof.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_lf_super_resolution_tiny_anchor_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_implemented": False,
            "receiver_decode_proven": False,
            "numpy_receiver_decode": False,
            "tiny_anchor_component_deltas_present": False,
            "section_native_byte_telemetry_present": False,
            "closed_campaign_blockers": [],
            "queue_blockers": list(_LF_SUPER_RESOLUTION_CLOSED_BLOCKERS),
            "blockers": list(_LF_SUPER_RESOLUTION_CLOSED_BLOCKERS),
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    implemented = selected.get("receiver_payload_implemented") is True
    decoded = selected.get("receiver_decode_proven") is True
    numpy_decode = selected.get("numpy_receiver_decode") is True
    component_deltas = selected.get("tiny_anchor_component_deltas_present") is True
    telemetry = selected.get("section_native_byte_telemetry_present") is True
    proof_passed = implemented and decoded and numpy_decode and component_deltas and telemetry
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker)
        != "snerv_lf_super_resolution_tiny_anchor_payload_false_authority"
    ]
    closed = [
        blocker
        for blocker in selected.get("closed_campaign_blockers") or ()
        if blocker in _LF_SUPER_RESOLUTION_CLOSED_BLOCKERS
    ]
    queue_blockers: list[str] = []
    if not proof_passed:
        queue_blockers.extend(_LF_SUPER_RESOLUTION_CLOSED_BLOCKERS)
    queue_blockers.extend(selected_blockers)
    return {
        "schema": "snerv_lf_super_resolution_tiny_anchor_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "packet_path": selected.get("packet_path"),
        "source_packet_sha256": selected.get("source_packet_sha256"),
        "payload_path": selected.get("payload_path"),
        "payload_bytes": _positive_int(selected.get("payload_bytes")),
        "payload_sha256": selected.get("payload_sha256"),
        "anchor_raw_bytes": _positive_int(selected.get("anchor_raw_bytes")),
        "compressed_payload_bytes": _positive_int(
            selected.get("compressed_payload_bytes")
        ),
        "pair_indices": selected.get("pair_indices"),
        "sample_shape_b2chw": selected.get("sample_shape_b2chw"),
        "anchor_shape_b2chw": selected.get("anchor_shape_b2chw"),
        "receiver_component_delta_stats": selected.get(
            "receiver_component_delta_stats"
        ),
        "component_delta_scope": selected.get("component_delta_scope"),
        "receiver_payload_implemented": implemented,
        "receiver_decode_proven": decoded,
        "numpy_receiver_decode": numpy_decode,
        "tiny_anchor_component_deltas_present": component_deltas,
        "section_native_byte_telemetry_present": telemetry,
        "closed_campaign_blockers": _dedupe(closed) if proof_passed else [],
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*selected_blockers, *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _spectral_band_allocator_state(
    spectral_band_allocator_receiver_payload_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in spectral_band_allocator_receiver_payload_proofs
        if isinstance(row, Mapping)
        and row.get("schema")
        == "snerv_score_tethered_spectral_band_allocator_receiver_proof.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_score_tethered_spectral_band_allocator_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_implemented": False,
            "receiver_decode_proven": False,
            "numpy_receiver_decode": False,
            "score_tethered_allocation_implemented": False,
            "section_native_byte_telemetry_present": False,
            "closed_campaign_blockers": [],
            "queue_blockers": list(_SPECTRAL_BAND_ALLOCATOR_CLOSED_BLOCKERS),
            "blockers": list(_SPECTRAL_BAND_ALLOCATOR_CLOSED_BLOCKERS),
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    implemented = selected.get("receiver_payload_implemented") is True
    decoded = selected.get("receiver_decode_proven") is True
    numpy_decode = selected.get("numpy_receiver_decode") is True
    allocation = selected.get("score_tethered_allocation_implemented") is True
    telemetry = selected.get("section_native_byte_telemetry_present") is True
    proof_passed = implemented and decoded and numpy_decode and allocation and telemetry
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker)
        != "snerv_score_tethered_spectral_band_allocator_false_authority"
    ]
    closed = [
        blocker
        for blocker in selected.get("closed_campaign_blockers") or ()
        if blocker in _SPECTRAL_BAND_ALLOCATOR_CLOSED_BLOCKERS
    ]
    queue_blockers: list[str] = []
    if not proof_passed:
        queue_blockers.extend(_SPECTRAL_BAND_ALLOCATOR_CLOSED_BLOCKERS)
    queue_blockers.extend(selected_blockers)
    return {
        "schema": "snerv_score_tethered_spectral_band_allocator_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "packet_path": selected.get("packet_path"),
        "source_packet_sha256": selected.get("source_packet_sha256"),
        "payload_path": selected.get("payload_path"),
        "payload_bytes": _positive_int(selected.get("payload_bytes")),
        "payload_sha256": selected.get("payload_sha256"),
        "allocation_table_raw_bytes": _positive_int(
            selected.get("allocation_table_raw_bytes")
        ),
        "allocation_band_count": _positive_int(
            selected.get("allocation_band_count")
        ),
        "allocation_budget_units": _positive_int(
            selected.get("allocation_budget_units")
        ),
        "pair_indices": selected.get("pair_indices"),
        "sample_shape_b2chw": selected.get("sample_shape_b2chw"),
        "receiver_payload_implemented": implemented,
        "receiver_decode_proven": decoded,
        "numpy_receiver_decode": numpy_decode,
        "score_tethered_allocation_implemented": allocation,
        "section_native_byte_telemetry_present": telemetry,
        "human_readable_payload_labels": selected.get(
            "human_readable_payload_labels"
        ),
        "closed_campaign_blockers": _dedupe(closed) if proof_passed else [],
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*selected_blockers, *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _lf_latent_hyperprior_state(
    lf_latent_hyperprior_receiver_payload_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in lf_latent_hyperprior_receiver_payload_proofs
        if isinstance(row, Mapping)
        and row.get("schema") == "snerv_lf_latent_hyperprior_receiver_proof.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_lf_latent_hyperprior_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_implemented": False,
            "receiver_decode_proven": False,
            "numpy_receiver_decode": False,
            "entropy_model_implemented": False,
            "hyperprior_scale_present": False,
            "receiver_replay_proven": False,
            "section_native_byte_telemetry_present": False,
            "closed_campaign_blockers": [],
            "queue_blockers": list(_LF_LATENT_HYPERPRIOR_CLOSED_BLOCKERS),
            "blockers": list(_LF_LATENT_HYPERPRIOR_CLOSED_BLOCKERS),
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    implemented = selected.get("receiver_payload_implemented") is True
    decoded = selected.get("receiver_decode_proven") is True
    numpy_decode = selected.get("numpy_receiver_decode") is True
    entropy_model = selected.get("entropy_model_implemented") is True
    hyperprior_scale = selected.get("hyperprior_scale_present") is True
    replay = selected.get("receiver_replay_proven") is True
    telemetry = selected.get("section_native_byte_telemetry_present") is True
    proof_passed = (
        implemented
        and decoded
        and numpy_decode
        and entropy_model
        and hyperprior_scale
        and replay
        and telemetry
    )
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker) != "snerv_lf_latent_hyperprior_payload_false_authority"
    ]
    closed = [
        blocker
        for blocker in selected.get("closed_campaign_blockers") or ()
        if blocker in _LF_LATENT_HYPERPRIOR_CLOSED_BLOCKERS
    ]
    queue_blockers: list[str] = []
    if not proof_passed:
        queue_blockers.extend(_LF_LATENT_HYPERPRIOR_CLOSED_BLOCKERS)
    queue_blockers.extend(selected_blockers)
    return {
        "schema": "snerv_lf_latent_hyperprior_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "packet_path": selected.get("packet_path"),
        "source_packet_sha256": selected.get("source_packet_sha256"),
        "payload_path": selected.get("payload_path"),
        "payload_bytes": _positive_int(selected.get("payload_bytes")),
        "payload_sha256": selected.get("payload_sha256"),
        "mean_raw_bytes": _positive_int(selected.get("mean_raw_bytes")),
        "scale_raw_bytes": _positive_int(selected.get("scale_raw_bytes")),
        "latent_symbol_raw_bytes": _positive_int(
            selected.get("latent_symbol_raw_bytes")
        ),
        "compressed_payload_bytes": _positive_int(
            selected.get("compressed_payload_bytes")
        ),
        "estimated_entropy_bits": selected.get("estimated_entropy_bits"),
        "pair_indices": selected.get("pair_indices"),
        "sample_shape_b2chw": selected.get("sample_shape_b2chw"),
        "lf_shape_b2chw": selected.get("lf_shape_b2chw"),
        "receiver_payload_implemented": implemented,
        "receiver_decode_proven": decoded,
        "numpy_receiver_decode": numpy_decode,
        "entropy_model_implemented": entropy_model,
        "hyperprior_scale_present": hyperprior_scale,
        "receiver_replay_proven": replay,
        "section_native_byte_telemetry_present": telemetry,
        "human_readable_payload_labels": selected.get(
            "human_readable_payload_labels"
        ),
        "closed_campaign_blockers": _dedupe(closed) if proof_passed else [],
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*selected_blockers, *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _runtime_binding_state(
    lf_hf_runtime_binding_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in lf_hf_runtime_binding_proofs
        if isinstance(row, Mapping)
        and row.get("schema") == SNERV_LF_HF_RUNTIME_BINDING_PROOF_SCHEMA
    ]
    if not rows:
        return {
            "schema": "snerv_lf_hf_runtime_binding_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "runtime_bound_solution_families": [],
            "closed_campaign_blockers": [],
            "queue_blockers": [],
            "blockers": [],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("report_path") or ""),
        ),
    )
    closed = [
        str(blocker)
        for blocker in selected.get("closed_campaign_blockers") or ()
        if str(blocker)
        and str(blocker)
        in {
            blocker
            for blocker in (
                runtime_binding_blocker_for_solution_family(
                    _LF_CONDITIONED_HF_FAMILY
                ),
                runtime_binding_blocker_for_solution_family(
                    "joint_lf_hf_factorized_codebook"
                ),
                runtime_binding_blocker_for_solution_family(
                    "temporal_lf_predictor_gate"
                ),
                runtime_binding_blocker_for_solution_family(
                    "lf_super_resolution_from_tiny_anchor"
                ),
                runtime_binding_blocker_for_solution_family(
                    "score_tethered_spectral_band_allocator"
                ),
                runtime_binding_blocker_for_solution_family(
                    "entropy_modeled_lf_latent_hyperprior"
                ),
            )
            if blocker
        }
    ]
    selected_blockers = [
        str(blocker)
        for blocker in selected.get("blockers") or ()
        if str(blocker)
        and str(blocker)
        != "snerv_lf_hf_runtime_binding_payload_proofs_missing"
    ]
    return {
        "schema": "snerv_lf_hf_runtime_binding_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "runtime_binding_row_count": _nonnegative_int(
            selected.get("runtime_binding_row_count")
        ),
        "runtime_bound_solution_families": [
            str(value) for value in selected.get("runtime_bound_solution_families") or ()
        ],
        "closed_campaign_blockers": _dedupe(closed),
        "queue_blockers": _dedupe(selected_blockers),
        "blockers": _dedupe(selected_blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _snerv_campaign_rows(campaign_plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in campaign_plans:
        for row in plan.get("campaign_rows") or ():
            if isinstance(row, Mapping) and row.get("family") == "snerv":
                rows.append(dict(row))
    rows.sort(
        key=lambda row: (
            0 if row.get("local_mlx_launch_command_ready") is True else 1,
            int(row.get("priority") or 999),
            str(row.get("candidate_id") or ""),
        )
    )
    return rows


def _reroute_state(reroute_queues: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queues = [q for q in reroute_queues if isinstance(q, Mapping)]
    if not queues:
        return {
            "reroute_queue_count": 0,
            "freshest_reroute_queue_row_count": None,
            "freshest_queue_has_no_lf_over_ceiling_rows": None,
            "snar_header_minimization_report_count": 0,
            "all_reroute_queue_row_count": 0,
        }
    freshest = queues[-1]
    return {
        "reroute_queue_count": len(queues),
        "freshest_schema": freshest.get("schema"),
        "freshest_generated_utc": freshest.get("generated_utc"),
        "freshest_reroute_queue_row_count": _nonnegative_int(
            freshest.get("queue_row_count")
        ),
        "freshest_queue_has_no_lf_over_ceiling_rows": (
            _nonnegative_int(freshest.get("queue_row_count")) == 0
        ),
        "snar_header_minimization_report_count": _nonnegative_int(
            freshest.get("snar_header_minimization_report_count")
        )
        or 0,
        "all_reroute_queue_row_count": sum(
            int(q.get("queue_row_count") or 0) for q in queues
        ),
    }


def _current_state(
    *,
    campaign_rows: Sequence[Mapping[str, Any]],
    reroute_state: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    source_forward_state: Mapping[str, Any],
    official_replacement_authority_state: Mapping[str, Any],
    scorer_domain_state: Mapping[str, Any],
    value_domain_state: Mapping[str, Any],
    hf_residual_payload_state: Mapping[str, Any],
    joint_codebook_state: Mapping[str, Any],
    temporal_lf_predictor_state: Mapping[str, Any],
    lf_super_resolution_state: Mapping[str, Any],
    spectral_band_allocator_state: Mapping[str, Any],
    lf_latent_hyperprior_state: Mapping[str, Any],
    runtime_binding_state: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    demoted_blockers: list[str] = []
    if not evidence_rows:
        blockers.append("snerv_lf_hf_measured_lf_payload_report_missing")
    if not campaign_rows:
        blockers.append("snerv_lf_hf_current_campaign_plan_missing")
    if reroute_state.get("freshest_queue_has_no_lf_over_ceiling_rows") is True:
        demoted_blockers.append(
            "snerv_lf_hf_current_snar2_queue_has_no_lf_over_ceiling_rows"
        )
    if reroute_state.get("reroute_queue_count") == 0:
        blockers.append("snerv_lf_hf_reroute_queue_missing")
    ready_rows = [
        row for row in campaign_rows if row.get("local_mlx_launch_command_ready") is True
    ]
    return {
        **dict(reroute_state),
        "snerv_campaign_row_count": len(campaign_rows),
        "snerv_local_mlx_launch_command_ready_row_count": len(ready_rows),
        "lf_payload_evidence_row_count": len(evidence_rows),
        "lf_dominance_launch_signal_active": (
            reroute_state.get("freshest_queue_has_no_lf_over_ceiling_rows") is False
        ),
        "lf_dominance_signal_demoted": bool(demoted_blockers),
        "demoted_blockers": _dedupe(demoted_blockers),
        "source_forward_evidence": dict(source_forward_state),
        "official_replacement_authority_evidence": dict(
            official_replacement_authority_state
        ),
        "scorer_domain_evidence": dict(scorer_domain_state),
        "value_domain_evidence": dict(value_domain_state),
        "hf_residual_payload_evidence": dict(hf_residual_payload_state),
        "joint_codebook_evidence": dict(joint_codebook_state),
        "temporal_lf_predictor_evidence": dict(temporal_lf_predictor_state),
        "lf_super_resolution_evidence": dict(lf_super_resolution_state),
        "spectral_band_allocator_evidence": dict(spectral_band_allocator_state),
        "lf_latent_hyperprior_evidence": dict(lf_latent_hyperprior_state),
        "runtime_binding_evidence": dict(runtime_binding_state),
        "blockers": _dedupe(blockers),
    }


def _campaign_blockers(
    campaign_row: Mapping[str, Any],
    prefixes: Sequence[str],
) -> list[str]:
    source = [str(blocker) for blocker in campaign_row.get("blockers") or () if blocker]
    out: list[str] = []
    for blocker in source:
        if blocker in prefixes or any(blocker.startswith(prefix) for prefix in prefixes):
            out.append(blocker)
    return _dedupe(out)


def _source_paths(items: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        path = (
            item.get("_source_path")
            or item.get("_candidate_feedback_source_path")
            or item.get("source_path")
            or item.get("report_path")
        )
        if path:
            paths.append(str(path))
    return _dedupe(paths)


def _queue_rebuild_command(
    *,
    output_root: Path,
    input_source_paths: Mapping[str, Sequence[str]],
) -> list[str]:
    argv = ["uv", "run", "python", "tools/build_snerv_lf_hf_replacement_queue.py"]
    flag_by_key = {
        "lf_payload_reports": "--lf-payload-report",
        "reroute_queues": "--reroute-queue",
        "campaign_plans": "--campaign-plan",
        "source_forward_artifacts": "--source-forward-artifact",
        "official_replacement_authority_gates": "--official-replacement-authority-gate",
        "candidate_feedback_rows": "--candidate-feedback-row",
        "value_domain_xray_reports": "--value-domain-xray",
        "hf_residual_receiver_payload_proofs": (
            "--hf-residual-receiver-payload-proof"
        ),
        "joint_codebook_receiver_payload_proofs": (
            "--joint-codebook-receiver-payload-proof"
        ),
        "temporal_lf_predictor_receiver_payload_proofs": (
            "--temporal-lf-predictor-receiver-payload-proof"
        ),
        "lf_super_resolution_receiver_payload_proofs": (
            "--lf-super-resolution-receiver-payload-proof"
        ),
        "spectral_band_allocator_receiver_payload_proofs": (
            "--spectral-band-allocator-receiver-payload-proof"
        ),
        "lf_latent_hyperprior_receiver_payload_proofs": (
            "--lf-latent-hyperprior-receiver-payload-proof"
        ),
        "lf_hf_runtime_binding_proofs": "--lf-hf-runtime-binding-proof",
    }
    for key, flag in flag_by_key.items():
        for path in input_source_paths.get(key) or ():
            argv.extend([flag, str(path)])
    argv.extend(
        [
            "--output-root",
            output_root.as_posix(),
            "--output-json",
            (output_root / "snerv_lf_hf_replacement_queue.json").as_posix(),
            "--output-md",
            (output_root / "snerv_lf_hf_replacement_queue.md").as_posix(),
        ]
    )
    return argv


def _dedupe_command(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values if str(value)]


def _storage_preflight(
    output_root: Path,
    *,
    min_free_bytes: int,
    allow_local_output: bool,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve(strict=False)
    on_ssd = any(_is_relative_to(root, ssd_root) for ssd_root in SSD_ROOTS)
    free = shutil.disk_usage(_nearest_existing_parent(root)).free
    blockers: list[str] = []
    if not on_ssd and not allow_local_output:
        blockers.append("snerv_lf_hf_replacement_output_root_not_on_ssd_tier")
    if free < int(min_free_bytes):
        blockers.append("snerv_lf_hf_replacement_output_root_free_space_below_floor")
    if blockers:
        raise SnervLfHfReplacementQueueError(
            f"{root}: storage preflight blocked: {', '.join(blockers)}"
        )
    return {
        "schema": "snerv_lf_hf_replacement_storage_preflight.v1",
        "output_root": root.as_posix(),
        "ssd_tier": _ssd_tier(root),
        "free_bytes_before": int(free),
        "min_free_bytes": int(min_free_bytes),
        "allow_local_output": bool(allow_local_output),
        "blockers": [],
    }


def _ssd_tier(path: Path) -> str:
    for root in SSD_ROOTS:
        if _is_relative_to(path, root):
            return root.as_posix()
    return "local_or_unknown"


def _nearest_existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists() and cursor.parent != cursor:
        cursor = cursor.parent
    return cursor


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _nested(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _first_finite_float(
    *mappings: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        for key in keys:
            value = mapping.get(key)
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(out):
                return out
    return None


def _stable_safe_token(text: str, *, max_len: int = 120) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(text).lower())
    clean = "_".join(part for part in clean.split("_") if part)
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]
    base = clean[: max(1, max_len - 13)].strip("_")
    return f"{base}_{digest}" if base else digest


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _shell_join(argv: Sequence[Any]) -> str:
    return " ".join(str(part) for part in argv)


def attach_source_identity(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Return ``payload`` with source path and SHA-256 metadata attached."""

    source_path = Path(path)
    data = source_path.read_bytes()
    return {
        **dict(payload),
        "_source_path": source_path.as_posix(),
        "_source_sha256": hashlib.sha256(data).hexdigest(),
    }


def load_json_with_source_identity(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and attach path/SHA metadata for custody."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SnervLfHfReplacementQueueError(f"{source_path}: JSON payload must be an object")
    return attach_source_identity(payload, source_path)
