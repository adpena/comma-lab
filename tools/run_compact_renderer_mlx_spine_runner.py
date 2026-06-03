#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run MLX-first compact renderer spine acquisition.

The PR95/HNeRV PyTorch continuation lane is useful as a control, but the
production path for new compact bases is MLX/Metal first with portable NumPy
artifacts. This tool turns MLX long-training reports into the shared HPRC
representation spine, acquisition report, and bounded-runner plan. It never
grants score authority; exact CPU/CUDA still owns promotion.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.local_submission_replay import (  # noqa: E402
    run_local_submission_replay,
    stage_local_replay_submission,
)
from tac.adaptation.hard_pair_indices import (  # noqa: E402
    HardPairIndicesError,
    load_pair_indices_file,
    merge_pair_indices,
    normalize_pair_indices,
    parse_pair_indices_csv,
    validate_pair_indices_in_range,
)
from tac.analysis.nerv_candidate_curriculum import (  # noqa: E402
    build_hinerv_candidate_curriculum_plan,
    build_snerv_candidate_curriculum_plan,
    strip_candidate_curriculum_authority_fields,
)
from tac.analysis.nerv_candidate_feedback import (  # noqa: E402
    write_nerv_candidate_feedback_files,
)
from tac.analysis.nerv_decoder_weight_waterfill import (  # noqa: E402
    DEFAULT_ACTION_BITS as NERV_DECODER_WEIGHT_WATERFILL_ACTION_BITS,
)
from tac.analysis.nerv_decoder_weight_waterfill import (  # noqa: E402
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
)
from tac.analysis.nerv_long_training_campaign_plan import (  # noqa: E402
    build_nerv_long_training_campaign_plan,
)
from tac.analysis.nerv_modelsize_budget import (  # noqa: E402
    DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
    DEFAULT_SNERV_MODELSIZE_DEC_STRDS,
    DEFAULT_SNERV_MODELSIZE_ENC_STRDS,
    HINERV_COMPACT_FINE_INJECTION_BLOCK_INDEX,
    HINERV_COMPACT_MID_INJECTION_BLOCK_INDEX,
    SNERV_MODELSIZE_CONTROL_PROFILES,
    analyze_hinerv_modelsize_candidate,
    analyze_snerv_modelsize_candidate,
    build_hinerv_modelsize_budget_report,
    build_snerv_modelsize_budget_report,
    enumerate_hinerv_modelsize_candidates,
    enumerate_snerv_modelsize_candidates,
    modelsize_control_precedence_contract,
    official_nerv_oss_flag_audit,
    snerv_model_size_adapter_from_id_token,
    snerv_modelsize_control_profile,
    snerv_temporal_mode_from_id_token,
    tag_hinerv_target_modelsize_candidate,
)
from tac.analysis.nerv_receiver_closed_modelsize_ladder import (  # noqa: E402
    build_nerv_receiver_closed_modelsize_ladder,
)
from tac.analysis.nerv_source_parity_contract import (  # noqa: E402
    build_nerv_source_parity_contract,
)
from tac.analysis.nerv_stack_synergy_audit import (  # noqa: E402
    build_nerv_stack_synergy_audit,
)
from tac.analysis.snerv_binary_profile import (  # noqa: E402
    SnervBinaryProfileError,
    write_snerv_binary_profile,
)
from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    PR95_MLX_SOURCE_VIDEO_RGB_YUV6_BLOCKERS,
)
from tac.local_acceleration.pr95_hnerv_mlx_contract import (  # noqa: E402
    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
)
from tac.substrates._shared.mlx_score_aware.adapter import (  # noqa: E402
    DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
    MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS,
    SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
)
from tac.substrates._shared.mlx_score_aware.carrier_training_plan import (  # noqa: E402
    build_score_aware_carrier_training_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.hprc.mlx_prefilter_coverage import (  # noqa: E402
    summarize_mlx_prefilter_coverage,
)
from tac.substrates.hprc.representation_spine import (  # noqa: E402
    build_pr95_hnerv_spine_from_archive,
    write_representation_spine_projection,
)
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT  # noqa: E402
from tac.substrates.hprc.spine_acquisition import (  # noqa: E402
    DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    build_spine_acquisition_report,
)
from tac.substrates.hprc.spine_bounded_runner import (  # noqa: E402
    build_spine_bounded_runner_plan,
    write_spine_bounded_runner_plan,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    SnervArchiveError,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (  # noqa: E402
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_adapter_contract import (  # noqa: E402
    build_snerv_mlx_native_adapter_contract,
    build_snerv_mlx_native_file_backed_evidence,
    build_snerv_mlx_native_training_export_guard,
)
from tac.training.long_training_canonical import (  # noqa: E402
    DEFAULT_CHECKPOINT_INTERVAL_EPOCHS,
    LongTrainingStopRequested,
)
from tools.emit_compact_renderer_spine_adapter import (  # noqa: E402
    emit_compact_renderer_spine_adapter,
)

COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA = "compact_renderer_mlx_spine_runner.v1"
ACTIVE_CAMPAIGN_LOCK_SCHEMA = "compact_renderer_active_campaign_lock.v1"
ACTIVE_FAMILY_PROCESS_REFUSAL_SCHEMA = "compact_renderer_active_family_process_refusal.v1"
DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS = DEFAULT_CHECKPOINT_INTERVAL_EPOCHS
RAW_NERV_MODELSIZE_BUDGET_SCHEMAS = frozenset(
    {"nerv_modelsize_budget.v1", "snerv_modelsize_budget.v1"}
)
DEFAULT_SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
DEFAULT_PR95_SOURCE_ARCHIVE_ZIP = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view"
    / "public_pr95_intake_20260505_auto/archive.zip"
)
DEFAULT_PR95_RECEIVER_RUNTIME_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view"
    / "public_pr95_intake_20260505_auto/source/submissions/hnerv_muon"
)
DEFAULT_UPSTREAM_DIR = Path(os.environ.get("TAC_UPSTREAM_DIR", REPO_ROOT / "upstream"))
CANONICAL_UPSTREAM_FALLBACK_DIR = Path.home() / "Projects" / "pact" / "upstream"
HI_NERV_MODELSIZE_DEFAULT_SEGNET_DISTILLATION_WEIGHT = 1.0
HI_NERV_MODELSIZE_DEFAULT_POSE_DISTILLATION_WEIGHT = 1.0
DEFAULT_SOURCE_VIDEO_PATH = Path("upstream/videos/0.mkv")
TARGET_FAMILIES = (
    "pr95_hnerv",
    "hi_nerv",
    "snerv",
    "rnerv",
    "sr_nerv",
    "boostnerv",
    "pvq_nerv",
    "rt_vq_nerv",
    "pact_nerv_selector_v4",
    "pact_nerv_vq",
)
EXECUTABLE_FAMILIES = (
    "pr95_hnerv",
    "pact_nerv_selector_v4",
    "pact_nerv_vq",
    "hi_nerv",
    "snerv",
)
PLANNER_GATED_FAMILIES: tuple[str, ...] = ()
CLI_EXECUTE_FAMILIES = (*EXECUTABLE_FAMILIES, *PLANNER_GATED_FAMILIES)
PLANNER_ROW_REQUIRED_FAMILIES = ("hi_nerv", "snerv")
PLANNER_ROW_QUEUE_RUNNABLE_STATUSES = ("queued", "runnable")
PLANNER_ROW_QUEUE_ARTIFACT_SCHEMAS = (
    "nerv_long_training_campaign_plan.v1",
    "experiment_queue.v1",
    COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
)
PLANNER_ROW_LAUNCH_CONTRACT_SCHEMA = (
    "nerv_long_training_queue_launch_authority_contract.v1"
)
PLANNER_ROW_TIMING_SMOKE_MAX_PAIRS = 32
PLANNER_ROW_TIMING_SMOKE_MAX_EPOCHS = 3
HI_NERV_OPTIMIZER_POLICIES = (
    "auto",
    "pr95_curriculum",
    "native_optimizer",
)
DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_GRAD_CLIP_MAX_NORM = 1.0
DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_WEIGHT_DECAY = 1.0e-4
DEFAULT_PACT_CODER_QAT_QUANT_RESIDUAL_WEIGHT = 1.0e-3
DEFAULT_PACT_CODER_QAT_MAGNITUDE_WEIGHT = 1.0e-4
DEFAULT_PACT_CODER_QAT_DELTA_WEIGHT = 2.0e-4
DEFAULT_PACT_CODER_QAT_C1A_ENTROPY_WEIGHT = 1.0e-4
DEFAULT_PACT_CODER_QAT_C1A_SIGMA = 0.2
DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE = 512
COMPACT_FAMILY_STARTUP_MARKER_FILENAME = (
    "compact_renderer_mlx_spine_runner_startup.json"
)
COMPACT_FAMILY_BACKENDS: dict[str, dict[str, Any]] = {
    "pr95_hnerv": {
        "canonical_family": "pr95_hnerv",
        "backend_status": "executable_mlx_archive_export_control_arm",
        "trainer_kind": "canonical_mlx_score_aware_harness_public_pr95_seeded_control_arm",
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family pr95_hnerv",
        "archive_exporter": "tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip",
        "receiver_proof": "pr95_public_inflate_sh_required_before_exact_gate",
        "next_action": "run_pr95_hnerv_scoreaware_full_pair_continuation_then_receiver_proof",
        "execution_scope": (
            "MLX advisory archive-export control arm; not a PR95-faithful "
            "reproduction or score authority until source-faithful curriculum, "
            "receiver proof, full-frame parity, and exact CPU/CUDA gates close"
        ),
    },
    "pact_nerv_vq": {
        "canonical_family": "pact_nerv_vq",
        "backend_status": "executable_mlx_backend_available",
        "trainer_kind": "canonical_mlx_score_aware_harness",
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family pact_nerv_vq",
        "archive_exporter": (
            "tac.substrates.pact_nerv_vq.archive_candidate."
            "export_pact_nerv_vq_mlx_archive"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_archive_exporter",
        "next_action": "train_mlx_export_archive_spine_receiver_proof_then_full_video_replay",
        "execution_scope": "MLX advisory train/export/archive candidate lane",
    },
    "pvq_nerv": {
        "canonical_family": "pact_nerv_vq",
        "backend_status": "executable_via_pact_nerv_vq_adapter",
        "trainer_kind": "canonical_mlx_score_aware_harness",
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family pact_nerv_vq",
        "archive_exporter": (
            "tac.substrates.pact_nerv_vq.archive_candidate."
            "export_pact_nerv_vq_mlx_archive"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_archive_exporter",
        "next_action": "route_to_pact_nerv_vq_until_pvq_specific_adapter_diverges",
        "execution_scope": "MLX advisory adapter route through pact_nerv_vq",
    },
    "pact_nerv_selector_v4": {
        "canonical_family": "pact_nerv",
        "backend_status": "executable_mlx_backend_available",
        "trainer_kind": "selector_v4_mlx_score_aware_harness",
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family pact_nerv_selector_v4",
        "archive_exporter": (
            "tac.substrates.pact_nerv_selector_v4.archive_candidate."
            "export_pact_nerv_selector_v4_mlx_archive"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_archive_exporter",
        "section_value_profiler": (
            "tools/profile_pact_nerv_selector_v4_mlx_section_value.py"
        ),
        "next_action": (
            "train_mlx_export_psv4_archive_spine_receiver_proof_then_"
            "full_video_section_value_profile"
        ),
        "execution_scope": (
            "MLX advisory train/export/archive candidate lane; PSV4 selector "
            "primitive is charged at archive encode time and remains "
            "false-authority until full replay plus exact CPU/CUDA gates close"
        ),
    },
    "hi_nerv": {
        "canonical_family": "hi_nerv",
        "backend_status": (
            "mlx_archive_export_adapter_available_"
            "distortion_fit_actuator_pending"
        ),
        "trainer_kind": "mlx_renderer_export_smoke_available_scoreaware_trainer_pending",
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv",
        "archive_exporter": (
            "tac.substrates.hi_nerv.archive_candidate.export_hi_nerv_mlx_archive"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_archive_exporter",
        "next_action": (
            "scale_score_faithful_hinerv_mlx_training_with_full_video_"
            "prefilter_then_cpu_replay_gate"
        ),
        "execution_scope": (
            "primary compact carrier candidate; MLX renderer/export/receiver "
            "adapter is executable, but promotion is blocked until score-aware "
            "full-video training, MLX prefilter, local CPU replay, and exact "
            "CPU/CUDA gates exist"
        ),
        "rate_axis_evidence": (
            "advisory HiNeRV HIV1 packets show super-small-rate-by-design is "
            "structural; distortion remains the unsolved score-aware fit axis"
        ),
        "score_aware_training_evidence": {
            "schema": "compact_carrier_advisory_evidence.v1",
            "archive_bytes": 40_491,
            "archive_pair_count": 2,
            "projected_archive_bytes_600pair": 36_000,
            "d_seg": 0.508,
            "advisory_score": 92.84,
            "g3_adjoint_exact": True,
            "latent_jvp_norm_max": 1.0e-4,
            "linf_delta_vs_l2": 0.31,
            "modelsize_knob_present": True,
            "evidence_axis": "[macOS-MLX research-signal]",
            "authority": "advisory_planning_only",
        },
        "distortion_fit_blocker": (
            "local per-pixel-MSE HiNeRV smokes remain far from scorer-faithful "
            "SegNet/PoseNet distortion, so cheap bytes alone cannot promote"
        ),
        "stack_role": "primary_carrier",
        "carrier_priority": 10,
        "architecture_priors": [
            "hierarchical_multi_scale_latent_pyramid",
            "protect_low_frequency_structure_before_scorer_priced_residuals",
            "section_value_pricing_required_for_decoder_latent_scale_splits",
        ],
        "allowed_enhancers": [
            "sr_nerv_lowres_encode_superresolve_resolution_deadzone",
            "rnerv_component_search_and_recurrence",
            "ffnerv_flow_pose_channel",
            "boostnerv_temporal_affine_bolt_on",
            "p18_p19_scorer_priced_residual_tokens",
        ],
    },
    "snerv": {
        "canonical_family": "snerv",
        "backend_status": (
            "executable_cpu_advisory_plus_mlx_native_export_adapter_available"
        ),
        "trainer_kind": (
            "mlx_native_target_hydration_receiver_export_available_"
            "scoreaware_long_training_missing"
        ),
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family snerv",
        "archive_exporter": (
            "tac.substrates.snerv_inverse_steg_carrier.archive_candidate."
            "export_snerv_archive_bound_candidate_package"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_snerv_packet",
        "next_action": (
            "bind_learned_mlx_scoreaware_decoder_training_to_snerv_native_export_"
            "under_same_packet_spine_with_charged_wavelet_features_no_hidden_sidecars"
        ),
        "execution_scope": (
            "primary compact carrier candidate; SNeRV wavelet/frequency split "
            "must be charged as decoder/latent/selector/codebook bytes and "
            "validated by receiver proof before promotion. Current executable "
            "path is CPU advisory plus archive-bound runtime package, with an "
            "optional MLX-native target-hydration/export/receiver-proof "
            "attachment. Learned MLX score-aware training remains the next "
            "blocker."
        ),
        "stack_role": "primary_carrier",
        "carrier_priority": 9,
        "architecture_priors": [
            "spectra_preserving_wavelet_low_frequency_encoding",
            "implicit_high_frequency_restoration_only_when_score_priced",
            "aligns_with_z8_wavelet_findings_without_bulk_float_coeff_storage",
        ],
        "allowed_enhancers": [
            "sr_nerv_lowres_encode_superresolve_resolution_deadzone",
            "rnerv_component_search_and_recurrence",
            "ffnerv_flow_pose_channel",
            "boostnerv_temporal_affine_bolt_on",
            "p18_p19_scorer_priced_residual_tokens",
        ],
    },
    "rnerv": {
        "canonical_family": "rnerv",
        "backend_status": "migration_required",
        "trainer_kind": "missing_mlx_compact_base_trainer",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_adapter_implemented",
        "next_action": "implement_rnerv_mlx_renderer_exporter_under_spine_contract",
        "execution_scope": (
            "enhancer/search prior over the winning compact carrier; not a "
            "separate top-priority carrier unless it reduces charged "
            "decoder+latent entropy under the same packet spine"
        ),
        "stack_role": "enhancer_or_search_prior",
        "carrier_priority": 6,
        "architecture_priors": [
            "component_search_recipe_for_nerv_family",
            "recurrent_or_hypernetwork_latent_generator_only_if_bytes_are_charged",
        ],
        "allowed_enhancers": [
            "hi_nerv_carrier",
            "snerv_carrier",
            "p18_p19_scorer_priced_residual_tokens",
        ],
    },
    "sr_nerv": {
        "canonical_family": "sr_nerv",
        "backend_status": "migration_required",
        "trainer_kind": "missing_mlx_compact_base_trainer",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_adapter_implemented",
        "next_action": (
            "run_lowres_to_contest_resolution_scorer_mirror_check_then_"
            "implement_charged_sr_upsampler_mlx_adapter"
        ),
        "execution_scope": (
            "high-priority resolution-axis enhancer/design knob for the "
            "winning carrier; low-res encode plus charged super-resolve may "
            "exploit the scorer downsample dead-zone, but must pass a "
            "low-res->SR->contest-output->scorer-downsample mirror check"
        ),
        "stack_role": "resolution_axis_enhancer_or_design_knob",
        "carrier_priority": 8,
        "enhancer_priority": 9,
        "architecture_priors": [
            "encode_at_or_below_512x384_internal_resolution",
            "charged_super_resolution_to_1164x874_output",
            "verify_lowres_sr_roundtrip_preserves_posenet_segnet",
            "rank_above_flow_and_boost_because_rate_axis_deadzone_is_structural",
        ],
        "allowed_enhancers": [
            "hi_nerv_carrier",
            "snerv_carrier",
            "pr95_hnerv_control_carrier",
            "rnerv_component_search_and_recurrence",
            "boostnerv_temporal_affine_bolt_on",
            "p18_p19_scorer_priced_residual_tokens",
        ],
    },
    "boostnerv": {
        "canonical_family": "boostnerv",
        "backend_status": "migration_required",
        "trainer_kind": "pytorch_or_l0_scaffold_not_mlx_first_runner_ready",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_mlx_or_portable_runtime_adapter_implemented",
        "next_action": "migrate_boost_residual_to_mlx_or_mark_as_non_primary_sidecar",
        "execution_scope": (
            "bolt-on enhancer for the winning compact carrier; not a standalone "
            "carrier in this runner until it proves charged byte-value"
        ),
        "stack_role": "enhancer_bolt_on",
        "carrier_priority": 4,
        "architecture_priors": [
            "conditional_decoder_boost",
            "temporal_affine_modulation",
            "apply_only_when_section_value_per_byte_is_positive",
        ],
        "allowed_enhancers": ["hi_nerv_carrier", "snerv_carrier", "sr_nerv_carrier"],
    },
    "rt_vq_nerv": {
        "canonical_family": "rt_vq_nerv",
        "backend_status": "migration_required",
        "trainer_kind": "missing_residual_token_vq_mlx_adapter",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_adapter_implemented",
        "next_action": "implement_residual_token_vq_as_charged_section_not_hidden_sidecar",
        "execution_scope": "not executable until residual-token adapter lands",
    },
}

_SCORE_AWARE_STACK_READINESS_FALSE: dict[str, bool] = {
    "real_segnet_teacher_ready": False,
    "real_posenet_teacher_ready": False,
    "eval_roundtrip_ready": False,
    "ema_ready": False,
    "pr95_curriculum_ready": False,
    "adamw_ready": False,
    "muon_ready": False,
    "coder_aware_regularization_ready": False,
    "sigma_noise_qat_ready": False,
    "quant_noise_qat_ready": False,
    "nvrc_learned_quant_ready": False,
    "byte_closed_archive_export_ready": False,
}

PR95_HNERV_SCOREAWARE_ADVISORY_NOT_EXACT_BLOCKER = (
    "pr95_mlx_scoreaware_teacher_distillation_is_advisory_not_exact_contest_loss"
)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pr95_scoreaware_training_metadata(
    artifact_dict: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = artifact_dict.get("substrate_artifact_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    scoreaware = metadata.get("score_aware_training")
    if not isinstance(scoreaware, Mapping):
        return {}
    return dict(scoreaware)


def _decoder_weight_saliency_metadata(
    artifact_dict: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = artifact_dict.get("substrate_artifact_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    saliency = metadata.get("decoder_weight_gradient_saliency")
    if not isinstance(saliency, Mapping):
        return {}
    return dict(saliency)


def _write_decoder_weight_saliency_artifact(
    *,
    artifact_dict: Mapping[str, Any],
    output_dir: Path,
    family: str,
) -> dict[str, Any]:
    """Write train-time decoder-gradient saliency for waterfill planning."""

    saliency = _decoder_weight_saliency_metadata(artifact_dict)
    if not saliency:
        return {
            "schema": "compact_runner_decoder_weight_saliency_artifact.v1",
            "family": str(family),
            "written": False,
            "reason": "decoder_weight_gradient_saliency_missing",
            "authority": "macos_mlx_research_signal_false_authority",
        }
    path = output_dir / "decoder_weight_gradient_saliency.json"
    payload = {
        **saliency,
        "schema": saliency.get("schema")
        or "mlx_decoder_weight_gradient_saliency.v1",
        "artifact_schema": "compact_runner_decoder_weight_saliency_artifact.v1",
        "family": str(family),
        "source": "MlxScoreAwareAdapter.artifact_metadata",
        "source_training_artifact_path": (
            output_dir / "training_artifact.json"
        ).as_posix(),
        "artifact_path": path.as_posix(),
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _write_json(path, payload)
    return {
        "schema": "compact_runner_decoder_weight_saliency_artifact.v1",
        "family": str(family),
        "written": True,
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "row_count": int(payload.get("row_count") or 0),
        "source_schema": payload.get("schema"),
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _pr95_has_joint_real_scorer_binding(
    *,
    artifact_dict: Mapping[str, Any],
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
) -> bool:
    scoreaware = _pr95_scoreaware_training_metadata(artifact_dict)
    seg_weight = _as_float(
        scoreaware.get("segnet_distillation_weight"),
        default=float(segnet_distillation_weight),
    )
    pose_weight = _as_float(
        scoreaware.get("pose_distillation_weight"),
        default=float(pose_distillation_weight),
    )
    return (
        bool(scoreaware.get("has_real_segnet_teacher"))
        and bool(scoreaware.get("has_real_posenet_teacher"))
        and seg_weight > 0.0
        and pose_weight > 0.0
        and not bool(scoreaware.get("allow_mock_scorer_teacher"))
        and not bool(scoreaware.get("allow_segnet_only_research"))
    )


def _pr95_hnerv_control_arm_exact_blockers(
    *,
    artifact_dict: Mapping[str, Any],
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
) -> list[str]:
    has_joint_binding = _pr95_has_joint_real_scorer_binding(
        artifact_dict=artifact_dict,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
    )
    blockers: list[str] = [
        "pr95_hnerv_mlx_archive_export_control_arm_not_pr95_faithful_reproduction"
    ]
    for blocker in PR95_MLX_SOURCE_VIDEO_RGB_YUV6_BLOCKERS:
        if has_joint_binding and blocker == PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER:
            blockers.append(PR95_HNERV_SCOREAWARE_ADVISORY_NOT_EXACT_BLOCKER)
            continue
        if (
            has_joint_binding
            and blocker == PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER
        ):
            continue
        blockers.append(blocker)
    blockers.append("pr95_hnerv_stage8_muon_continuation_not_wired")
    if not has_joint_binding:
        blockers.append(
            "pr95_hnerv_default_scorer_distillation_weights_are_zero_unless_cli_overridden"
        )
    blockers.extend(
        [
            "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
            "requires_exact_cpu_cuda_auth_eval_before_score_claim",
        ]
    )
    return _dedupe(blockers)


class CompactRendererMlxSpineRunnerError(ValueError):
    """Raised when an MLX compact renderer row cannot enter the spine."""


def _local_cpu_replay_enabled_by_default(
    num_pairs: int,
    *,
    mlx_prefilter_local_replay_passed: bool = False,
) -> bool:
    """Return whether local replay should run without an explicit CLI override."""

    return int(num_pairs) >= CONTEST_PAIR_COUNT and bool(
        mlx_prefilter_local_replay_passed
    )


def _run_compact_local_cpu_replay_gate(
    *,
    archive_zip_path: str | Path | None,
    runtime_submission_dir: str | Path,
    output_dir: str | Path,
    upstream_dir: str | Path,
    num_pairs: int,
    requested: bool | None,
    has_full_video_mlx_prefilter: bool = False,
    mlx_prefilter_local_replay_passed: bool = False,
    keep_inflated: bool = False,
    cleanup_failed_scratch: bool = True,
    repo_root: str | Path = REPO_ROOT,
) -> tuple[dict[str, Any] | None, list[Path], list[str]]:
    """Run the reusable local CPU replay gate for full-coverage candidates.

    The gate is deliberately coverage-aware. A partial 1/32/128-pair smoke can
    prove archive/runtime consumption, but it cannot be a local score authority
    because upstream evaluate expects the contest-shaped full-video output.
    Full-video candidates run the gate by default unless the operator opts out.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    pairs = int(num_pairs)
    should_run = (
        bool(requested)
        if requested is not None
        else _local_cpu_replay_enabled_by_default(
            pairs,
            mlx_prefilter_local_replay_passed=mlx_prefilter_local_replay_passed,
        )
    )
    if pairs < CONTEST_PAIR_COUNT:
        return None, [], ["local_cpu_replay_not_run_partial_pair_coverage"]
    if (requested is None or requested is True) and not has_full_video_mlx_prefilter:
        return None, [], ["local_cpu_replay_waiting_for_full_video_mlx_prefilter"]
    if (requested is None or requested is True) and not mlx_prefilter_local_replay_passed:
        return None, [], ["local_cpu_replay_blocked_by_mlx_prefilter_score"]
    if not should_run:
        return None, [], ["local_cpu_replay_not_executed"]
    if archive_zip_path is None:
        return None, [], ["local_cpu_replay_archive_zip_missing"]

    archive = _optional_existing(archive_zip_path, base=root)
    if archive is None:
        return None, [], ["local_cpu_replay_archive_zip_missing_or_unreadable"]
    runtime_dir = _resolve(runtime_submission_dir, base=root)
    if not runtime_dir.is_dir():
        return None, [], ["local_cpu_replay_runtime_submission_dir_missing"]

    replay_dir = _resolve(output_dir, base=root)
    replay_dir.mkdir(parents=True, exist_ok=True)
    staged_submission = stage_local_replay_submission(
        runtime_submission_dir=runtime_dir,
        archive_zip_path=archive,
        output_dir=replay_dir,
        force=True,
    )
    summary = run_local_submission_replay(
        submission_dir=staged_submission,
        source_runtime_submission_dir=runtime_dir,
        archive_zip_path=archive,
        device="cpu",
        upstream_root=_resolve(upstream_dir, base=root),
        keep_inflated=bool(keep_inflated),
        cleanup_failed_scratch=bool(cleanup_failed_scratch),
        certify_failed_scratch_rebuildable=bool(cleanup_failed_scratch),
    )
    summary_dict = json.loads(summary.to_json())
    summary_path = replay_dir / "local_submission_replay_summary.json"
    _write_json(summary_path, summary_dict)
    blockers: list[str] = []
    if not bool(summary_dict.get("evaluation_passed")):
        blockers.append("local_cpu_replay_failed")
        blockers.extend(str(item) for item in summary_dict.get("blockers") or [])
    return summary_dict, [summary_path], _dedupe(blockers)


def _compact_queue_id_token(value: str, *, fallback: str = "carrier") -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return token.strip("._-") or fallback


def _runnable_runtime_submission_dir(
    runtime_submission_dir: str | Path | None,
    *,
    repo_root: Path,
) -> tuple[Path | None, list[str]]:
    """Return a runtime dir only when it can actually feed ``inflate.sh``."""

    if runtime_submission_dir is None:
        return None, []
    runtime_dir = _resolve(runtime_submission_dir, base=repo_root)
    if not runtime_dir.is_dir():
        return None, ["carrier_post_export_runtime_submission_dir_missing"]
    if not (runtime_dir / "inflate.sh").is_file():
        return None, ["carrier_post_export_runtime_submission_dir_missing_inflate_sh"]
    return runtime_dir, []


def _compile_carrier_post_export_materializer_plan(
    *,
    output_dir: str | Path,
    archive_path: str | Path | None,
    archive_sha256: Any = None,
    archive_bytes: Any = None,
    runtime_submission_dir: str | Path | None = None,
    family: str,
    repo_root: str | Path = REPO_ROOT,
    local_cpu_concurrency: int = 2,
) -> dict[str, Any]:
    """Compile reusable final-rate materializer queues for a carrier archive.

    The result is a deterministic handoff only: it writes bootstrap, context,
    backlog, work-queue, and experiment-queue JSON, but never executes the
    queue and never grants local score authority.
    """

    from comma_lab.scheduler.frontier_rate_attack_bootstrap import (
        DEFAULT_EXECUTABLE_TARGET_KINDS,
        FrontierRateAttackBootstrapError,
        archive_record,
        build_frontier_rate_attack_payloads,
    )

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    result_root = out / "carrier_post_export_materializers"
    plan_path = result_root / "post_export_materializer_plan.json"
    family_token = _compact_queue_id_token(str(family), fallback="carrier")
    effective_runtime_dir, runtime_context_blockers = _runnable_runtime_submission_dir(
        runtime_submission_dir,
        repo_root=root,
    )
    blockers: list[str] = []
    archive_bytes_int: int | None = None
    if archive_bytes is not None:
        try:
            archive_bytes_int = int(archive_bytes)
        except (TypeError, ValueError):
            blockers.append("carrier_post_export_archive_bytes_invalid")
    base_result: dict[str, Any] = {
        "schema": "compact_carrier_post_export_materializer_plan.v1",
        "family": str(family),
        "compiled": False,
        "archive_path": str(archive_path) if archive_path is not None else None,
        "archive_sha256": str(archive_sha256) if archive_sha256 else None,
        "archive_bytes": archive_bytes_int,
        "runtime_submission_dir": (
            str(runtime_submission_dir) if runtime_submission_dir is not None else None
        ),
        "effective_runtime_submission_dir": (
            effective_runtime_dir.as_posix()
            if effective_runtime_dir is not None
            else None
        ),
        "runtime_context_blockers": list(runtime_context_blockers),
        "materializer_results_root": result_root.as_posix(),
        "plan_path": plan_path.as_posix(),
        "queue_launch_executed": False,
        "allowed_use": "compile_only_post_export_local_materializer_queue",
        **FALSE_AUTHORITY,
    }
    if blockers:
        result = {
            **base_result,
            "blockers": _dedupe(blockers),
        }
        _write_json(plan_path, result)
        return result
    if archive_path is None:
        result = {
            **base_result,
            "blockers": ["carrier_post_export_archive_missing"],
        }
        _write_json(plan_path, result)
        return result

    try:
        archive = _resolve(archive_path, base=root)
        expected_sha = str(archive_sha256) if archive_sha256 else None
        record = archive_record(
            label=f"{family_token}_post_export",
            archive_path=archive,
            repo_root=root,
            source_kind="compact_carrier_byte_closed_export",
            expected_sha256=expected_sha,
            expected_bytes=archive_bytes_int,
            source_runtime_dir=effective_runtime_dir,
        )
        queue_id = (
            f"carrier_post_export_{family_token}_{str(record['sha256'])[:12]}"
        )
        output_root = result_root / queue_id
        bootstrap_path = output_root / "frontier_rate_attack_bootstrap.json"
        target_coverage_path = output_root / "target_coverage.json"
        contexts_path = output_root / "materializer_contexts.json"
        backlog_path = output_root / "materializer_backlog.json"
        work_queue_path = output_root / "materializer_work_queue.json"
        experiment_queue_path = output_root / "experiment_queue.json"
        experiment_queue_state_path = output_root / "experiment_queue.sqlite"
        payloads = build_frontier_rate_attack_payloads(
            repo_root=root,
            queue_id=queue_id,
            archive_records=[record],
            results_root=result_root,
            target_kinds=DEFAULT_EXECUTABLE_TARGET_KINDS,
            include_optional_target_blockers=True,
            local_cpu_concurrency=int(local_cpu_concurrency),
            lane_id=f"compact_carrier_post_export:{family_token}",
            source_work_queue_path=work_queue_path,
            source_state_path=experiment_queue_state_path,
            include_exact_readiness_followup=True,
            exact_readiness_followup_require_ready=False,
        )
        _write_json(bootstrap_path, payloads["bootstrap"])
        _write_json(target_coverage_path, payloads["target_coverage"])
        _write_json(contexts_path, payloads["contexts"])
        _write_json(backlog_path, payloads["backlog"])
        _write_json(work_queue_path, payloads["work_queue"])
        _write_json(experiment_queue_path, payloads["queue"])
        result = {
            **base_result,
            "compiled": True,
            "archive_record": record,
            "queue_id": queue_id,
            "queue_output_dir": output_root.as_posix(),
            "bootstrap_path": bootstrap_path.as_posix(),
            "target_coverage_path": target_coverage_path.as_posix(),
            "materializer_contexts_path": contexts_path.as_posix(),
            "materializer_backlog_path": backlog_path.as_posix(),
            "materializer_work_queue_path": work_queue_path.as_posix(),
            "experiment_queue_path": experiment_queue_path.as_posix(),
            "experiment_queue_state_path": experiment_queue_state_path.as_posix(),
            "executable_target_count": payloads["bootstrap"].get(
                "executable_target_count"
            ),
            "experiment_count": payloads["bootstrap"].get("experiment_count"),
            "step_count": payloads["bootstrap"].get("step_count"),
            "target_coverage": payloads["target_coverage"],
            "blockers": _dedupe(runtime_context_blockers),
        }
    except (
        FrontierRateAttackBootstrapError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        result = {
            **base_result,
            "compiled": False,
            "blockers": _dedupe(
                [f"carrier_post_export_materializer_plan_failed:{exc}"]
            ),
        }
    _write_json(plan_path, result)
    return result


def _post_export_materializer_handoff_summary(
    output_root: Path,
) -> dict[str, Any]:
    """Summarize queue-produced exact-readiness handoffs without granting authority."""

    rows: list[dict[str, Any]] = []
    for handoff_dir in sorted(output_root.glob("*_exact_eval_handoff")):
        if not handoff_dir.is_dir():
            continue
        target_kind = handoff_dir.name.removesuffix("_exact_eval_handoff")
        source_queue_path = handoff_dir / "source_queue.json"
        harvest_report_path = handoff_dir / "harvest_report.json"
        closure_report_path = (
            handoff_dir / "submission_closure" / "submission_closure_report.json"
        )
        exact_readiness_bridge_path = handoff_dir / "exact_readiness_bridge_report.json"
        dispatch_plan_path = handoff_dir / "dispatch_plan.json"
        dispatch_queue_path = handoff_dir / "dispatch_queue.json"
        row: dict[str, Any] = {
            "schema": "compact_carrier_post_export_materializer_handoff.v1",
            "target_kind": target_kind,
            "handoff_dir": handoff_dir.as_posix(),
            "source_queue_path": (
                source_queue_path.as_posix() if source_queue_path.is_file() else None
            ),
            "harvest_report_path": (
                harvest_report_path.as_posix()
                if harvest_report_path.is_file()
                else None
            ),
            "submission_closure_report_path": (
                closure_report_path.as_posix() if closure_report_path.is_file() else None
            ),
            "exact_readiness_bridge_report_path": (
                exact_readiness_bridge_path.as_posix()
                if exact_readiness_bridge_path.is_file()
                else None
            ),
            "dispatch_plan_path": (
                dispatch_plan_path.as_posix() if dispatch_plan_path.is_file() else None
            ),
            "dispatch_queue_path": (
                dispatch_queue_path.as_posix() if dispatch_queue_path.is_file() else None
            ),
            **FALSE_AUTHORITY,
        }
        for key, path in (
            ("source_queue_schema", source_queue_path),
            ("harvest_report_schema", harvest_report_path),
            ("submission_closure_report_schema", closure_report_path),
            ("exact_readiness_bridge_report_schema", exact_readiness_bridge_path),
            ("dispatch_plan_schema", dispatch_plan_path),
            ("dispatch_queue_schema", dispatch_queue_path),
        ):
            if not path.is_file():
                row[key] = None
                continue
            try:
                payload = _load_json(path)
            except (OSError, CompactRendererMlxSpineRunnerError, json.JSONDecodeError):
                row[key] = "unreadable"
            else:
                row[key] = payload.get("schema")
                if key == "harvest_report_schema":
                    row["harvest_report_blockers"] = payload.get("blockers")
                    row["harvest_report_ready_for_exact_eval_dispatch"] = payload.get(
                        "ready_for_exact_eval_dispatch"
                    )
        rows.append(row)
    return {
        "schema": "compact_carrier_post_export_materializer_handoff_summary.v1",
        "handoff_count": len(rows),
        "rows": rows,
        **FALSE_AUTHORITY,
    }


def _post_export_materializer_sweep_feedback_summary(
    output_root: Path,
) -> dict[str, Any]:
    """Preserve byte-saving post-export atoms as chain-solver signal."""

    rows: list[dict[str, Any]] = []
    for sweep_path in sorted(output_root.glob("**/sweep.json")):
        try:
            payload = _load_json(sweep_path)
        except (OSError, CompactRendererMlxSpineRunnerError, json.JSONDecodeError):
            continue
        if payload.get("schema") != "family_agnostic_materializer_empirical_sweep.v1":
            continue
        target_kind = str(payload.get("target_kind") or sweep_path.parent.name)
        rate_positive_count = int(payload.get("rate_positive_count") or 0)
        total_positive_saved_bytes = int(
            payload.get("total_positive_saved_bytes") or 0
        )
        max_saved_bytes = int(payload.get("max_saved_bytes") or 0)
        planner_feedback = payload.get("planner_feedback")
        recommended_rule = (
            planner_feedback.get("recommended_acquisition_rule")
            if isinstance(planner_feedback, Mapping)
            else None
        )
        byte_saving = rate_positive_count > 0 or total_positive_saved_bytes > 0
        rows.append(
            {
                "schema": "compact_carrier_post_export_sweep_feedback_row.v1",
                "target_kind": target_kind,
                "sweep_path": sweep_path.as_posix(),
                "observation_count": int(payload.get("observation_count") or 0),
                "rate_positive_count": rate_positive_count,
                "rate_nonpositive_count": int(
                    payload.get("rate_nonpositive_count") or 0
                ),
                "max_saved_bytes": max_saved_bytes,
                "total_positive_saved_bytes": total_positive_saved_bytes,
                "recommended_acquisition_rule": recommended_rule,
                "full_stack_chain_disposition": (
                    "retain_byte_saving_atom_for_ordered_chain_solver"
                    if byte_saving
                    else "demote_only_matching_zero_save_archive_class"
                ),
                "byte_saving_atom": byte_saving,
                **FALSE_AUTHORITY,
            }
        )
    byte_saving_rows = [row for row in rows if row["byte_saving_atom"] is True]
    total_positive_saved_bytes = sum(
        int(row["total_positive_saved_bytes"]) for row in byte_saving_rows
    )
    max_saved_bytes = max(
        (int(row["max_saved_bytes"]) for row in byte_saving_rows),
        default=0,
    )
    return {
        "schema": "compact_carrier_post_export_sweep_feedback_summary.v1",
        "sweep_count": len(rows),
        "byte_saving_sweep_count": len(byte_saving_rows),
        "zero_save_sweep_count": len(rows) - len(byte_saving_rows),
        "total_positive_saved_bytes": total_positive_saved_bytes,
        "max_saved_bytes": max_saved_bytes,
        "retain_target_kinds": [
            row["target_kind"] for row in rows if row["byte_saving_atom"] is True
        ],
        "zero_save_target_kinds": [
            row["target_kind"] for row in rows if row["byte_saving_atom"] is False
        ],
        "recommended_global_rule": (
            "retain_and_order_byte_saving_atoms_before_demoting_full_lane"
            if byte_saving_rows
            else "demote_only_matching_zero_save_archive_classes"
        ),
        "full_stack_ordering_note": (
            "A non-exact-ready byte-saving materializer is still reusable chain "
            "signal. Compose it with upstream/downstream atoms and order it in the "
            "full stack before declaring the family exhausted."
        ),
        "rows": rows,
        **FALSE_AUTHORITY,
    }


def _execute_carrier_post_export_materializer_plan(
    *,
    plan: Mapping[str, Any],
    requested: bool,
    max_steps: int = 1,
    max_parallel: int = 0,
    max_experiments: int | None = 1,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run a bounded local wave of a compiled carrier materializer queue."""

    from comma_lab.scheduler.experiment_queue import (
        connect_state,
        initialize_queue_state,
        load_queue_definition,
        queue_summary,
        run_queue_worker,
    )

    root = Path(repo_root).expanduser().resolve(strict=False)
    output_root_raw = plan.get("queue_output_dir") or plan.get("materializer_results_root")
    output_root = (
        _resolve(output_root_raw, base=root)
        if isinstance(output_root_raw, str) and output_root_raw
        else root / ".omx" / "state" / "compact_carrier_post_export_materializers"
    )
    execution_path = output_root / "post_export_materializer_execution.json"
    state_path_raw = plan.get("experiment_queue_state_path")
    state_path = (
        _resolve(state_path_raw, base=root)
        if isinstance(state_path_raw, str) and state_path_raw
        else output_root / "experiment_queue.sqlite"
    )
    base_result: dict[str, Any] = {
        "schema": "compact_carrier_post_export_materializer_execution.v1",
        "requested": bool(requested),
        "executed": False,
        "queue_id": plan.get("queue_id"),
        "queue_path": plan.get("experiment_queue_path"),
        "state_path": state_path.as_posix(),
        "log_root": (output_root / "experiment_queue_logs").as_posix(),
        "execution_path": execution_path.as_posix(),
        "max_steps": int(max_steps),
        "max_parallel": int(max_parallel),
        "max_experiments": max_experiments,
        **FALSE_AUTHORITY,
    }
    if not requested:
        result = {
            **base_result,
            "blockers": [],
            "mode": "compile_only_execution_not_requested",
        }
        _write_json(execution_path, result)
        return result
    if plan.get("compiled") is not True:
        result = {
            **base_result,
            "blockers": ["post_export_materializer_queue_not_compiled"],
        }
        _write_json(execution_path, result)
        return result
    queue_path_raw = plan.get("experiment_queue_path")
    if not isinstance(queue_path_raw, str) or not queue_path_raw:
        result = {
            **base_result,
            "blockers": ["post_export_materializer_experiment_queue_path_missing"],
        }
        _write_json(execution_path, result)
        return result
    if int(max_steps) < 1:
        result = {
            **base_result,
            "blockers": ["post_export_materializer_max_steps_must_be_positive"],
        }
        _write_json(execution_path, result)
        return result
    if max_experiments is not None and int(max_experiments) < 1:
        result = {
            **base_result,
            "blockers": [
                "post_export_materializer_max_experiments_must_be_positive_or_null"
            ],
        }
        _write_json(execution_path, result)
        return result

    try:
        queue_path = _resolve(queue_path_raw, base=root)
        queue = load_queue_definition(queue_path)
        log_root = output_root / "experiment_queue_logs"
        with connect_state(state_path) as conn:
            initialize_queue_state(conn, queue)
            before = queue_summary(conn, queue, repo_root=root)
            worker = run_queue_worker(
                conn,
                queue,
                repo_root=root,
                execute=True,
                max_steps=int(max_steps),
                max_parallel=int(max_parallel),
                idle_sleep_seconds=0.1,
                max_idle_cycles=1,
                poll_interval_seconds=0.1,
                stop_policy="drain",
                allow_cloud=False,
                allow_orphaned_state=False,
                noncanonical_state_rationale=(
                    "archive-specific post-export queue state is scoped under "
                    "the carrier output directory to avoid cross-run state reuse"
                ),
                log_root=log_root,
                max_experiments=max_experiments,
            )
            after = queue_summary(conn, queue, repo_root=root)
        blockers: list[str] = []
        if int(worker.get("failure_count") or 0) > 0:
            blockers.append("post_export_materializer_worker_failures")
        if int(worker.get("steps_started") or 0) < 1:
            blockers.append("post_export_materializer_worker_started_no_steps")
        result = {
            **base_result,
            "executed": True,
            "queue_path": queue_path.as_posix(),
            "state_path": state_path.as_posix(),
            "log_root": log_root.as_posix(),
            "before": before,
            "worker": worker,
            "after": after,
            "handoff_summary": _post_export_materializer_handoff_summary(
                output_root
            ),
            "sweep_feedback_summary": (
                _post_export_materializer_sweep_feedback_summary(output_root)
            ),
            "blockers": _dedupe(blockers),
        }
    except (OSError, RuntimeError, ValueError) as exc:
        result = {
            **base_result,
            "blockers": [f"post_export_materializer_execution_failed:{exc}"],
        }
    _write_json(execution_path, result)
    return result


def _scorer_coupled_rd_metadata() -> dict[str, Any]:
    """Return durable scorer-domain facts for advisory compact-run metadata."""

    return {
        "schema": "contest_scorer_coupled_rd_allocation_facts.v1",
        "score_formula": (
            "100*d_seg + sqrt(10*d_pose) + "
            "25*(archive_zip_bytes/uncompressed_total)"
        ),
        "fixed_marginal_byte_price": "25/uncompressed_total",
        "segnet_domain": {
            "pair_frame": 1,
            "domain": "last_frame_only",
            "num_classes": 5,
            "input_size": [384, 512],
        },
        "posenet_domain": {
            "pair_frames": [0, 1],
            "pose_dims_scored": 6,
            "input_kind": "yuv6_pair",
            "input_size": [384, 512],
        },
        "gradient_note": (
            "Pose Jacobian probes must use differentiable rgb_to_yuv6 "
            "roundtrip; upstream clamp is scorer-forward authority only."
        ),
        "allocation_rule": (
            "spend a charged bit only when measured scorer-value-per-bit "
            "exceeds fixed_marginal_byte_price"
        ),
        "authority": "planning_metadata_only_not_score_authority",
    }


def _resolve_scorer_upstream_dir(
    repo_root: str | Path,
    upstream_dir: str | Path | None,
) -> Path:
    root = Path(repo_root).expanduser().resolve(strict=False)
    repo_default = (root / "upstream").resolve(strict=False)

    def _complete(candidate: Path) -> bool:
        return (
            (candidate / "modules.py").is_file()
            and (candidate / "models" / "posenet.safetensors").is_file()
            and (candidate / "models" / "segnet.safetensors").is_file()
        )

    def _resolve_candidate(path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve(strict=False)

    if upstream_dir is not None:
        candidate = _resolve_candidate(upstream_dir)
        default_candidate = _resolve_candidate(DEFAULT_UPSTREAM_DIR)
        if candidate != default_candidate or _complete(candidate):
            return candidate
    candidates = [repo_default]
    env_upstream = os.environ.get("TAC_UPSTREAM_DIR")
    if env_upstream:
        candidates.insert(0, _resolve_candidate(env_upstream))
    candidates.append(CANONICAL_UPSTREAM_FALLBACK_DIR.expanduser().resolve(strict=False))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _complete(candidate):
            return candidate
    if upstream_dir is None:
        return repo_default
    candidate = Path(upstream_dir).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def _file_sha256_or_none(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _scorer_upstream_metadata(upstream_dir: str | Path) -> dict[str, Any]:
    upstream = Path(upstream_dir).expanduser().resolve(strict=False)
    modules_path = upstream / "modules.py"
    posenet_path = upstream / "models" / "posenet.safetensors"
    segnet_path = upstream / "models" / "segnet.safetensors"
    return {
        "schema": "compact_runner_scorer_upstream_snapshot.v1",
        "upstream_dir": upstream.as_posix(),
        "modules_py_exists": modules_path.is_file(),
        "modules_py_sha256": _file_sha256_or_none(modules_path),
        "posenet_safetensors_exists": posenet_path.is_file(),
        "posenet_safetensors_sha256": _file_sha256_or_none(posenet_path),
        "segnet_safetensors_exists": segnet_path.is_file(),
        "segnet_safetensors_sha256": _file_sha256_or_none(segnet_path),
    }


def _require_scorer_upstream_dir_for_distillation(
    *,
    upstream_dir: str | Path,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
) -> None:
    if segnet_distillation_weight <= 0.0 and pose_distillation_weight <= 0.0:
        return
    upstream = Path(upstream_dir).expanduser().resolve(strict=False)
    required = (
        upstream / "modules.py",
        upstream / "models" / "posenet.safetensors",
        upstream / "models" / "segnet.safetensors",
    )
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise CompactRendererMlxSpineRunnerError(
            "real scorer distillation requires --upstream-dir to point at the "
            "pinned contest upstream snapshot; missing: " + ", ".join(missing)
        )


def _resolve_torch_scorer_device_alias(
    requested_device: str,
    *,
    torch_module: Any | None = None,
) -> str:
    """Resolve planner-level scorer device aliases to real PyTorch devices."""

    requested = str(requested_device or "cpu").strip().lower()
    if requested in {"cpu", "cuda", "mps"}:
        return requested
    if requested == "metal":
        return "mps"
    if requested != "gpu":
        raise CompactRendererMlxSpineRunnerError(
            f"unsupported scorer distillation device: {requested_device!r}"
        )
    torch = torch_module
    if torch is None:
        try:
            import torch as torch  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - import failure is environment.
            raise CompactRendererMlxSpineRunnerError(
                "distillation_device='gpu' requires PyTorch to resolve a concrete "
                "scorer teacher device"
            ) from exc
    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(
        getattr(cuda, "is_available", lambda: False)()
    )
    if cuda_available:
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    mps_available = bool(
        getattr(mps, "is_available", lambda: False)()
    )
    if mps_available:
        return "mps"
    raise CompactRendererMlxSpineRunnerError(
        "distillation_device='gpu' requested, but neither torch.cuda nor "
        "torch.backends.mps is available"
    )


def adapt_pr95_mlx_report_to_spine(
    *,
    pr95_mlx_report_path: str | Path,
    output_dir: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
    upstream_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Adapt the latest exported PR95 MLX checkpoint into the shared spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, upstream_dir)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    report_path = _resolve(pr95_mlx_report_path, base=root)
    pr95_report = _load_json(report_path)
    checkpoint = _select_latest_exported_checkpoint(pr95_report, base=root)
    pt_path = _resolve_existing(checkpoint["pytorch_state_dict_path"], base=root)
    latents_path = _resolve_existing(checkpoint["latents_path"], base=root)
    export_manifest_path = _optional_existing(
        checkpoint.get("pytorch_export_manifest_path"),
        base=root,
    )
    projection_dir = out / "pr95_hnerv_spine"
    adapter_report = emit_compact_renderer_spine_adapter(
        family="pr95_hnerv",
        output_dir=projection_dir,
        decoder_blob=pt_path,
        latents_blob=latents_path,
        trained_weights_provenance=_trained_provenance(
            report_path=report_path,
            checkpoint=checkpoint,
            role="weights",
        ),
        trained_latents_provenance=_trained_provenance(
            report_path=report_path,
            checkpoint=checkpoint,
            role="latents",
        ),
        manifest_extra=_coverage_manifest_extra(pr95_report),
    )
    projection_manifest = Path(adapter_report["projection"]["manifest_path"])
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[projection_manifest],
        hard_byte_ceilings=hard_byte_ceilings,
    )
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    _write_json(acquisition_path, acquisition)
    runner_plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        mlx_profile_paths=mlx_profile_paths,
        hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
        repo_root=root,
        upstream_dir=scorer_upstream,
    )
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    write_spine_bounded_runner_plan(
        output_path=runner_plan_path,
        plan=runner_plan,
        allow_overwrite=True,
    )
    final_report = _base_report(
        output_dir=out,
        mode="adapted_pr95_mlx_report",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final_report.update(
        {
            "pr95_mlx_report_path": report_path.as_posix(),
            "pr95_mlx_report_sha256": _sha256_file(report_path),
            "selected_checkpoint": _checkpoint_summary(checkpoint, base=root),
            "pytorch_export_manifest_path": (
                None if export_manifest_path is None else export_manifest_path.as_posix()
            ),
            "spine_adapter_report_path": adapter_report["report_path"],
            "projection_manifest_paths": [projection_manifest.as_posix()],
            "acquisition_report_path": acquisition_path.as_posix(),
            "bounded_runner_plan_path": runner_plan_path.as_posix(),
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "selected_runner_rows": runner_plan["selected_runner_rows"],
            "blockers": _dedupe(
                [
                    *adapter_report["exact_gate"]["blockers"],
                    *runner_plan["blockers"],
                    "mlx_local_report_is_advisory_not_score_authority",
                ]
            ),
        }
    )
    final_report_path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(final_report_path, final_report)
    return {**final_report, "report_path": final_report_path.as_posix()}


def adapt_pr95_stage8_report_to_spine(
    *,
    pr95_stage8_report_path: str | Path,
    output_dir: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    run_receiver_proof: bool = False,
    receiver_proof_runtime_dir: str | Path = DEFAULT_PR95_RECEIVER_RUNTIME_DIR,
    keep_receiver_proof_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
    upstream_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Adapt a source-faithful PR95 Stage-8 report into the compact spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, upstream_dir)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    report_path = _resolve_existing(pr95_stage8_report_path, base=root)
    stage8_report = _load_json(report_path)
    archive_raw = (
        stage8_report.get("candidate_archive_zip_path")
        or (stage8_report.get("package_report") or {}).get("archive_zip_path")
    )
    if not isinstance(archive_raw, str) or not archive_raw:
        raise CompactRendererMlxSpineRunnerError(
            "pr95_stage8_report_missing_candidate_archive_zip_path"
        )
    archive = _resolve_existing(archive_raw, base=root)
    receiver_proof_report: dict[str, Any] | None = None
    receiver_proof_paths: list[Path] = []
    package_report = (
        stage8_report.get("package_report")
        if isinstance(stage8_report.get("package_report"), dict)
        else {}
    )
    embedded_receiver_proof = (
        package_report.get("archive_bound_candidate_receiver_proof")
        if isinstance(package_report.get("archive_bound_candidate_receiver_proof"), dict)
        else None
    )
    if embedded_receiver_proof is not None:
        receiver_proof_report = embedded_receiver_proof
        proof_path = embedded_receiver_proof.get("proof_path")
        if isinstance(proof_path, str) and proof_path:
            receiver_proof_paths = [_resolve(proof_path, base=root)]
    if run_receiver_proof:
        receiver_proof_report = run_pr95_hnerv_receiver_proof(
            archive_zip=archive,
            runtime_dir=receiver_proof_runtime_dir,
            output_dir=out / "receiver_proof",
            keep_output=keep_receiver_proof_output,
            timeout_seconds=receiver_proof_timeout_seconds,
            repo_root=root,
        )
        proof_path = receiver_proof_report.get("report_path")
        if isinstance(proof_path, str) and proof_path:
            receiver_proof_paths = [Path(proof_path)]

    spine = build_pr95_hnerv_spine_from_archive(
        archive,
        runtime_submission_dir=receiver_proof_runtime_dir,
    )
    projection = write_representation_spine_projection(
        output_dir=out / "pr95_stage8_hnerv_spine",
        spine=spine,
        basename="pr95_stage8_hnerv_representation_spine",
    )
    projection_manifest = Path(projection["manifest_path"])
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[projection_manifest],
        hard_byte_ceilings=hard_byte_ceilings,
    )
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    _write_json(acquisition_path, acquisition)
    runner_plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        mlx_profile_paths=mlx_profile_paths,
        receiver_proof_report_paths=receiver_proof_paths,
        hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
        repo_root=root,
        upstream_dir=scorer_upstream,
    )
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    write_spine_bounded_runner_plan(
        output_path=runner_plan_path,
        plan=runner_plan,
        allow_overwrite=True,
    )

    exact_gate = stage8_report.get("exact_gate")
    stage8_blockers = (
        exact_gate.get("blockers")
        if isinstance(exact_gate, dict) and isinstance(exact_gate.get("blockers"), list)
        else []
    )
    local_training = stage8_report.get("local_training_result")
    raw_result = (
        local_training.get("raw_result")
        if isinstance(local_training, dict)
        and isinstance(local_training.get("raw_result"), dict)
        else {}
    )
    public_stage8_train_called = raw_result.get("public_stage8_train_stage_called")
    blockers: list[Any] = [
        *stage8_blockers,
        *runner_plan.get("blockers", []),
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not mlx_profile_paths:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    receiver_proof_passed = (
        receiver_proof_report is not None
        and (
            receiver_proof_report.get("receiver_proof_valid") is True
            or receiver_proof_report.get("runtime_consumption_proof_passed") is True
            or receiver_proof_report.get("receiver_contract_satisfied") is True
        )
    )
    if receiver_proof_report is None:
        blockers.append("receiver_proof_not_executed")
    elif not receiver_proof_passed:
        blockers.append("receiver_proof_failed")
        blockers.extend(receiver_proof_report.get("blockers") or [])
    else:
        refusal = receiver_proof_report.get("exact_readiness_refusal")
        if isinstance(refusal, dict):
            blockers.extend(refusal.get("blockers") or [])

    final_report = _base_report(
        output_dir=out,
        mode="adapted_pr95_stage8_public_archive_report",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final_report.update(
        {
            "pr95_stage8_report_path": report_path.as_posix(),
            "pr95_stage8_report_sha256": _sha256_file(report_path),
            "stage8_mode": stage8_report.get("mode"),
            "source_archive_zip": stage8_report.get("source_archive_zip"),
            "candidate_archive_zip_path": archive.as_posix(),
            "candidate_archive_zip_bytes": archive.stat().st_size,
            "candidate_archive_zip_sha256": _sha256_file(archive),
            "projection_manifest_paths": [projection_manifest.as_posix()],
            "receiver_proof_report_paths": [
                path.as_posix() for path in receiver_proof_paths
            ],
            "receiver_proof_report": receiver_proof_report,
            "acquisition_report_path": acquisition_path.as_posix(),
            "bounded_runner_plan_path": runner_plan_path.as_posix(),
            "selected_runner_rows": runner_plan["selected_runner_rows"],
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "stage8_source_faithfulness": {
                "schema": "pr95_stage8_source_faithfulness.v1",
                "public_stage8_train_stage_called": public_stage8_train_called is True,
                "source_faithful_training_complete": (
                    public_stage8_train_called is True
                    and "stage8_zero_epoch_source_seed_packaged_no_training"
                    not in stage8_blockers
                    and "stage8_training_not_executed_plan_only"
                    not in stage8_blockers
                ),
                "optimizer_semantics": "public_pr95_stage8_muon_adamw_source_code",
                "score_authority": "none_until_exact_cpu_cuda",
            },
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "blockers": _dedupe(blockers),
        }
    )
    final_report_path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(final_report_path, final_report)
    return {**final_report, "report_path": final_report_path.as_posix()}


def execute_pr95_stage8_source_and_adapt(
    *,
    output_dir: str | Path,
    source_archive_zip: str | Path,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    stage8_epochs: int = 0,
    stage8_eval_every: int = 1,
    stage8_batch_size: int = 1,
    stage8_device: str = "cpu",
    stage8_muon_weight_decay: float = 5e-4,
    stage8_target_cache_path: str | Path | None = None,
    stage8_build_target_cache_if_missing: bool = True,
    run_receiver_proof: bool = False,
    receiver_proof_runtime_dir: str | Path = DEFAULT_PR95_RECEIVER_RUNTIME_DIR,
    keep_receiver_proof_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
    upstream_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the public PR95 Stage-8 source lane, then adapt it to the spine."""

    from tools.run_pr95_stage8_from_public_archive import (
        DEFAULT_CHALLENGE_ROOT,
        DEFAULT_PUBLIC_SUBMISSION_ROOT,
        run_pr95_stage8_from_public_archive,
    )

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    stage8_report = run_pr95_stage8_from_public_archive(
        source_archive_zip=_resolve_existing(source_archive_zip, base=root),
        public_submission_root=DEFAULT_PUBLIC_SUBMISSION_ROOT,
        challenge_root=DEFAULT_CHALLENGE_ROOT,
        source_video_path=_resolve(source_video_path, base=root),
        output_dir=out / "pr95_stage8_source_lane",
        epochs=int(stage8_epochs),
        eval_every=int(stage8_eval_every),
        batch_size=int(stage8_batch_size),
        muon_weight_decay=float(stage8_muon_weight_decay),
        device=stage8_device,
        execute=True,
        target_cache_path=None
        if stage8_target_cache_path is None
        else _resolve(stage8_target_cache_path, base=root),
        build_target_cache_if_missing=bool(stage8_build_target_cache_if_missing),
        overwrite=True,
    )
    return adapt_pr95_stage8_report_to_spine(
        pr95_stage8_report_path=stage8_report["report_path"],
        output_dir=out,
        hard_byte_ceilings=hard_byte_ceilings,
        mlx_profile_paths=mlx_profile_paths,
        hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
        run_receiver_proof=run_receiver_proof,
        receiver_proof_runtime_dir=receiver_proof_runtime_dir,
        keep_receiver_proof_output=keep_receiver_proof_output,
        receiver_proof_timeout_seconds=receiver_proof_timeout_seconds,
        allow_overwrite=True,
        repo_root=root,
        upstream_dir=upstream_dir,
    )


def build_plan_only_report(
    *,
    output_dir: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    repo_root: str | Path = REPO_ROOT,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Write the executable backlog when trained MLX artifacts are absent."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    report = _base_report(
        output_dir=out,
        mode="plan_only",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    report["blockers"] = [
        "trained_mlx_checkpoint_report_or_execute_pr95_mlx_smoke_required",
        "receiver_proof_not_yet_emitted",
        "contest_cpu_cuda_exact_eval_missing",
    ]
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, report)
    return {**report, "report_path": path.as_posix()}


def execute_planner_gated_compact_family(
    *,
    family: str,
    output_dir: str | Path,
    num_pairs: int,
    epochs: int,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Emit a planner-owned refusal for families whose adapters are not real yet.

    Planner-gated families are accepted by ``--execute-family`` only to produce
    a normal runner report carrying the score-aware planner row and exact
    blockers. They are not allowed to fake execution.
    """

    if family not in PLANNER_GATED_FAMILIES:
        raise CompactRendererMlxSpineRunnerError(
            f"planner-gated execution only supports {PLANNER_GATED_FAMILIES}; got {family!r}"
        )
    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    backend = COMPACT_FAMILY_BACKENDS[family]
    planner = _score_aware_carrier_training_plan(family, backend)
    blockers = _dedupe(
        [
            *planner.get("dispatch_blockers", []),
            f"{family}_mlx_native_train_export_archive_adapter_missing",
            f"{family}_byte_closed_archive_export_missing",
            f"{family}_receiver_proof_missing",
            f"{family}_full_video_mlx_prefilter_not_executed",
            "local_cpu_replay_not_executed",
            "contest_cpu_cuda_exact_eval_not_executed",
        ]
    )
    report = _base_report(
        output_dir=out,
        mode=f"{family}_planner_gated_execution_refused",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    report.update(
        {
            "execute_family": family,
            "trainer_launch_allowed": False,
            "launch_refusal_reason": (
                "planner row consumed; native MLX train/export/archive adapter "
                "is missing, so no fake training or manual launch is allowed"
            ),
            "requested_campaign": {
                "schema": "compact_carrier_campaign_request.v1",
                "family": family,
                "num_pairs": int(num_pairs),
                "epochs": int(epochs),
                "hard_byte_ceilings": [int(value) for value in hard_byte_ceilings],
                "mlx_profile_paths": [
                    _resolve(path, base=root).as_posix() for path in mlx_profile_paths
                ],
                "hprc_queue_followup_report_paths": [
                    _resolve(path, base=root).as_posix()
                    for path in hprc_queue_followup_report_paths
                ],
            },
            "score_aware_carrier_training_plan": planner,
            "adapter_contract_required": {
                "schema": "mlx_native_compact_carrier_adapter_contract.v1",
                "required_surfaces": [
                    "MLX renderer module with differentiable pair decode",
                    "real SegNet teacher cache",
                    "real PoseNet teacher cache",
                    "PR95-faithful staged optimizer bridge",
                    "coder-aware regularization and QAT hooks",
                    "byte-closed archive exporter",
                    "numpy-portable inflate runtime",
                    "receiver proof",
                    "full-video MLX prefilter report",
                    "local CPU replay gate",
                ],
                "false_authority_until_all_surfaces_exist": True,
            },
            "next_actions": [
                f"implement_{family}_mlx_native_renderer_bundle",
                f"implement_{family}_archive_exporter_and_numpy_inflate",
                f"run_{family}_2_pair_mlx_training_smoke_with_real_scorer_teachers",
                f"scale_{family}_32_128_600_pair_campaigns_after_smoke",
                "promote_only_byte_closed_local_winners_to_contest_cpu_then_cuda",
            ],
            "blockers": blockers,
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, report)
    return {**report, "report_path": path.as_posix()}


def _resolve_execute_modelsize_candidate(
    *,
    family: str,
    candidate_id: str,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int = CONTEST_PAIR_COUNT,
    target_modelsize_mparams: tuple[float, ...] = (),
    hinerv_target_modelsize_mparams: tuple[float, ...] = (),
    snerv_official_modelsize_mparams: tuple[float, ...] = (),
    snerv_modelsize_control_profile_id: str = DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
    snerv_official_enc_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_ENC_STRDS,
    snerv_official_dec_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_DEC_STRDS,
    snerv_temporal_context: int = 0,
    snerv_temporal_modes: tuple[str, ...] = ("delta",),
) -> dict[str, Any] | None:
    """Resolve an executable NeRV byte-budget candidate for a family launch.

    ``auto`` chooses a byte-plausible candidate from the complete enumeration,
    not just the truncated report rows. ``none/manual/off`` preserves the
    explicit CLI knobs for focused probes. Unknown ids fail before any campaign
    starts so the planner cannot silently drift away from curriculum provenance.
    """

    token = str(candidate_id or "auto").strip()
    if _modelsize_candidate_resolution_disabled(token):
        return None
    shared_targets = tuple(float(value) for value in target_modelsize_mparams)
    hinerv_targets = _dedupe_float_tuple(
        (*shared_targets, *tuple(float(v) for v in hinerv_target_modelsize_mparams))
    )
    snerv_targets = _dedupe_float_tuple(
        (*shared_targets, *tuple(float(v) for v in snerv_official_modelsize_mparams))
    )
    hinerv_auto_official_only = family == "hi_nerv" and token.lower() == "auto"
    if family == "hi_nerv":
        candidates = [
            row.as_dict()
            for row in enumerate_hinerv_modelsize_candidates(
                hard_byte_ceilings=hard_byte_ceilings,
                num_pairs=int(num_pairs),
                use_hierarchical_feature_grid_options=(
                    (True,) if hinerv_auto_official_only else (False, True)
                ),
                use_convnext_blocks_options=(
                    (True,) if hinerv_auto_official_only else (False, True)
                ),
                target_modelsize_mparams=hinerv_targets,
            )
        ]
    elif family == "snerv":
        candidates = [
            row.as_dict()
            for row in enumerate_snerv_modelsize_candidates(
                hard_byte_ceilings=hard_byte_ceilings,
                num_pairs=int(num_pairs),
                official_modelsize_mparams=snerv_targets,
                official_enc_strds=tuple(int(value) for value in snerv_official_enc_strds),
                official_dec_strds=tuple(int(value) for value in snerv_official_dec_strds),
                modelsize_control_profile_id=str(snerv_modelsize_control_profile_id),
                temporal_context=int(snerv_temporal_context),
                temporal_modes=tuple(str(value) for value in snerv_temporal_modes),
            )
        ]
    else:
        raise CompactRendererMlxSpineRunnerError(
            f"modelsize candidate resolution is only supported for hi_nerv/snerv; got {family!r}"
        )
    if not candidates:
        raise CompactRendererMlxSpineRunnerError(
            f"no {family} modelsize candidates were enumerated"
        )
    if token.lower() == "auto":
        if family == "hi_nerv":
            official_candidates = [
                row
                for row in candidates
                if _hi_nerv_modelsize_candidate_has_official_controls(row)
            ]
            official_under = [
                row
                for row in official_candidates
                if bool(row.get("nominal_under_ceiling"))
            ]
            if not official_under:
                raise CompactRendererMlxSpineRunnerError(
                    "hinerv_official_control_candidate_missing_under_ceiling"
                )
            candidates = official_candidates
        if family == "hi_nerv" and hinerv_targets:
            target_candidates = [
                row
                for row in candidates
                if row.get("capacity_source") == "local_hinerv_target_modelsize"
            ]
            if target_candidates:
                candidates = target_candidates
        if family == "snerv" and snerv_targets:
            official_candidates = [
                row
                for row in candidates
                if row.get("official_modelsize_solution") is not None
            ]
            if official_candidates:
                candidates = official_candidates
        under = [row for row in candidates if bool(row.get("nominal_under_ceiling"))]
        if under:
            tightest_ceiling = min(int(row["hard_byte_ceiling"]) for row in under)
            tightest_under = [
                row for row in under if int(row["hard_byte_ceiling"]) == tightest_ceiling
            ]
            if family == "hi_nerv" and hinerv_targets:
                return min(
                    tightest_under,
                    key=lambda row: (
                        float(row.get("modelsize_error_mparams") or 0.0),
                        int(row["nominal_total_payload_bytes"]),
                        -float(row.get("modelsize_mparams") or 0.0),
                        int(row.get("total_trainable_params", 0)),
                    ),
                )
            return max(
                tightest_under,
                key=lambda row: (
                    int(row["nominal_total_payload_bytes"]),
                    int(row.get("capacity_source") == "local_hinerv_target_modelsize"),
                    int(row.get("official_modelsize_solution") is not None),
                    float(row.get("modelsize_mparams") or 0.0),
                    int(row.get("total_trainable_params", 0)),
                ),
            )
        return min(
            candidates,
            key=lambda row: (
                abs(int(row.get("byte_headroom") or 0)),
                float(row.get("modelsize_error_mparams") or 0.0),
                -int(row.get("capacity_source") == "local_hinerv_target_modelsize"),
                -int(row.get("official_modelsize_solution") is not None),
                -float(row.get("modelsize_mparams") or 0.0),
                int(row["hard_byte_ceiling"]),
            ),
        )
    for row in candidates:
        if row["candidate_id"] == token:
            return row
    rebuilt = _modelsize_candidate_from_self_describing_id(
        family=family,
        candidate_id=token,
    )
    if rebuilt is not None:
        return rebuilt
    raise CompactRendererMlxSpineRunnerError(
        f"unknown {family} --modelsize-candidate-id {token!r}; "
        "rerun plan mode and select one of the emitted candidate_id values"
    )


def _modelsize_candidate_resolution_disabled(candidate_id: Any) -> bool:
    """Return whether the launch requested manual modelsize knobs only."""

    return str(candidate_id or "auto").strip().lower() in {
        "none",
        "manual",
        "off",
        "false",
        "0",
    }


def _snerv_official_modelsize_candidate_resolution_blockers(
    args: argparse.Namespace,
) -> list[str]:
    """Block source-named SNeRV capacity flags when candidates are disabled."""

    if getattr(args, "execute_family", None) != "snerv":
        return []
    if not _modelsize_candidate_resolution_disabled(
        getattr(args, "modelsize_candidate_id", "auto")
    ):
        return []
    flag_attrs = (
        ("--target-modelsize-mparams", "target_modelsize_mparams"),
        ("--snerv-official-modelsize-mparams", "snerv_official_modelsize_mparams"),
        ("--snerv-official-enc-strds", "snerv_official_enc_strds"),
        ("--snerv-official-dec-strds", "snerv_official_dec_strds"),
    )
    blockers = [
        f"snerv_official_modelsize_control_requires_candidate_resolution:{flag}"
        for flag, attr in flag_attrs
        if getattr(args, attr, None) is not None
    ]
    profile = str(
        getattr(
            args,
            "snerv_modelsize_control_profile",
            DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
        )
    )
    if profile != DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID:
        blockers.append(
            "snerv_official_modelsize_control_requires_candidate_resolution:"
            "--snerv-modelsize-control-profile"
        )
    return blockers


def _dedupe_float_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        normalized = float(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _hi_nerv_modelsize_candidate_has_official_controls(row: Mapping[str, Any]) -> bool:
    """Return whether a candidate enables the official HiNeRV control spine."""

    return bool(row.get("use_hierarchical_feature_grid")) and bool(
        row.get("use_convnext_blocks")
    )


_HINERV_MODEL_SIZE_ID_RE = re.compile(
    r"^hinerv_np(?P<num_pairs>\d+)_ld(?P<latent_dim>\d+)_"
    r"ed(?P<embed_dim>\d+)_dc(?P<decoder_channel>\d+)"
    r"(?:_mi(?P<mid_injection_block_index>\d+)fi(?P<fine_injection_block_index>\d+))?"
    r"(?P<control_suffix>(?:_hfg)?(?:_cnx)?)_"
    r"(?:(?:lg(?P<local_grid_levels>\d+)c(?P<local_grid_channels>\d+)_"
    r"cx(?P<convnext_mlp_ratio>\d+)k(?P<convnext_kernel_size>\d+)_))?"
    r"(?P<decoder_codec>.+)_ceil(?P<hard_byte_ceiling>\d+)"
    r"(?:_tgtmp(?P<target_modelsize>\d+(?:p\d+)?))?$"
)
_SNERV_MODEL_SIZE_ID_RE = re.compile(
    r"^snerv_np(?P<num_pairs>\d+)_(?P<wavelet>[A-Za-z0-9]+)_"
    r"lv(?P<levels>\d+)_"
    r"lfb(?P<bits_per_coeff>\d+(?:p\d+)?)_"
    r"stepb(?P<step_map_bits_per_coeff>\d+(?:p\d+)?)_"
    r"fc(?P<fc_dim>\d+)e(?P<emb_size>\d+)_"
    r"p(?P<patch_radius>\d+)_"
    r"mfu(?P<mfu_scales>\d+(?:-\d+)*)_"
    r"hfr(?P<hfr_gain>\d+(?:p\d+)?)_"
    r"t(?P<temporal_context>\d+)(?:_tm(?P<temporal_mode_token>[A-Za-z0-9]+))?_"
    r"ad(?P<adapter_token>[A-Za-z0-9]+)"
    r"(?:_oms(?P<official_modelsize>\d+(?:p\d+)?))?_"
    r"(?P<decoder_payload_codec>.+)_ceil(?P<hard_byte_ceiling>\d+)$"
)
_SNERV_PARTIAL_MODEL_SIZE_ID_RE = re.compile(
    r"^snerv_np(?P<num_pairs>\d+)_(?P<wavelet>[A-Za-z0-9]+)_"
    r"lv(?P<levels>\d+)_"
    r"lfb(?P<bits_per_coeff>\d+(?:p\d+)?)_"
    r"stepb(?P<step_map_bits_per_coeff>\d+(?:p\d+)?)_"
    r"fc(?P<fc_dim>\d+)e(?P<emb_size>\d+)_"
    r"(?!(?:p\d+_mfu))"
    r"(?P<decoder_payload_codec>.+)_ceil(?P<hard_byte_ceiling>\d+)$"
)
_SNERV_LEGACY_MODEL_SIZE_ID_RE = re.compile(
    r"^snerv_np(?P<num_pairs>\d+)_lv(?P<levels>\d+)_"
    r"lfb(?P<bits_per_coeff>\d+(?:p\d+)?)_"
    r"stepb(?P<step_map_bits_per_coeff>\d+(?:p\d+)?)_"
    r"(?P<decoder_payload_codec>.+)_ceil(?P<hard_byte_ceiling>\d+)$"
)


def _float_token(value: str) -> float:
    return float(value.replace("p", "."))


def _modelsize_candidate_from_self_describing_id(
    *,
    family: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    """Rebuild a queue-owned candidate from its stable self-describing id.

    The long-campaign queue stores the candidate id as custody. Execution must
    not depend on rerunning the same planner ceiling list: otherwise a valid
    queue can become unlaunchable after harmless planner-default changes. The
    id grammar is generated by ``tac.analysis.nerv_modelsize_budget``; this
    helper parses that grammar, reuses the canonical analyzer, then verifies the
    round-trip id before returning an executable row.
    """

    token = str(candidate_id).strip()
    if family == "hi_nerv":
        match = _HINERV_MODEL_SIZE_ID_RE.match(token)
        if match is None:
            return None
        control_suffix = match.group("control_suffix") or ""
        row_obj = analyze_hinerv_modelsize_candidate(
            hard_byte_ceiling=int(match.group("hard_byte_ceiling")),
            num_pairs=int(match.group("num_pairs")),
            latent_dim=int(match.group("latent_dim")),
            embed_dim=int(match.group("embed_dim")),
            decoder_channel=int(match.group("decoder_channel")),
            decoder_codec=match.group("decoder_codec"),
            use_hierarchical_feature_grid="_hfg" in control_suffix,
            use_convnext_blocks="_cnx" in control_suffix,
            local_grid_levels=int(match.group("local_grid_levels") or 2),
            local_grid_channels=int(match.group("local_grid_channels") or 4),
            convnext_mlp_ratio=int(match.group("convnext_mlp_ratio") or 2),
            convnext_kernel_size=int(match.group("convnext_kernel_size") or 7),
            mid_injection_block_index=int(
                match.group("mid_injection_block_index")
                or HINERV_COMPACT_MID_INJECTION_BLOCK_INDEX
            ),
            fine_injection_block_index=int(
                match.group("fine_injection_block_index")
                or HINERV_COMPACT_FINE_INJECTION_BLOCK_INDEX
            ),
        )
        target_modelsize = match.group("target_modelsize")
        if target_modelsize is not None:
            row_obj = tag_hinerv_target_modelsize_candidate(
                row_obj,
                target_modelsize_mparams=_float_token(target_modelsize),
            )
        row = row_obj.as_dict()
        if row["candidate_id"] != token:
            row["canonical_candidate_id"] = row["candidate_id"]
            row["candidate_id"] = token
            row["legacy_candidate_id"] = True
            row["blockers"] = [
                *list(row.get("blockers") or []),
                "legacy_hinerv_modelsize_candidate_id_missing_graph_controls",
            ]
        else:
            row["legacy_candidate_id"] = False
    elif family == "snerv":
        match = _SNERV_MODEL_SIZE_ID_RE.match(token)
        partial_match = _SNERV_PARTIAL_MODEL_SIZE_ID_RE.match(token)
        legacy_match = _SNERV_LEGACY_MODEL_SIZE_ID_RE.match(token)
        if match is None and partial_match is None and legacy_match is None:
            return None
        matched = match or partial_match or legacy_match
        if matched is None:  # pragma: no cover - defensive for type checkers
            return None
        groups = matched.groupdict()
        row = analyze_snerv_modelsize_candidate(
            hard_byte_ceiling=int(matched.group("hard_byte_ceiling")),
            num_pairs=int(matched.group("num_pairs")),
            wavelet=(
                matched.group("wavelet")
                if "wavelet" in matched.groupdict()
                else "db2"
            ),
            levels=int(matched.group("levels")),
            bits_per_coeff=_float_token(matched.group("bits_per_coeff")),
            step_map_bits_per_coeff=_float_token(
                matched.group("step_map_bits_per_coeff")
            ),
            decoder_payload_codec=matched.group("decoder_payload_codec"),
            fc_dim=(
                int(matched.group("fc_dim"))
                if "fc_dim" in matched.groupdict()
                else 9
            ),
            emb_size=(
                int(matched.group("emb_size"))
                if "emb_size" in groups
                else 0
            ),
            patch_radius=(
                int(matched.group("patch_radius"))
                if "patch_radius" in groups
                else 1
            ),
            mfu_scales=(
                tuple(int(v) for v in matched.group("mfu_scales").split("-"))
                if "mfu_scales" in groups
                else (1, 2, 4)
            ),
            hfr_gain=(
                _float_token(matched.group("hfr_gain"))
                if "hfr_gain" in groups
                else 0.0
            ),
            temporal_context=(
                int(matched.group("temporal_context"))
                if "temporal_context" in groups
                else 0
            ),
            temporal_mode=(
                snerv_temporal_mode_from_id_token(groups["temporal_mode_token"])
                if groups.get("temporal_mode_token") is not None
                else "delta"
            ),
            snerv_model_size_adapter=(
                snerv_model_size_adapter_from_id_token(
                    matched.group("adapter_token")
                )
                if "adapter_token" in groups
                else "snerv_fc_dim_emb_size_adapter_v1"
            ),
            official_modelsize_mparams=(
                _float_token(groups["official_modelsize"])
                if groups.get("official_modelsize") is not None
                else None
            ),
        ).as_dict()
        if partial_match is not None or legacy_match is not None:
            row["candidate_id"] = token
            row["legacy_candidate_id"] = True
        elif row["candidate_id"] != token:
            return None
    else:
        return None
    if row["candidate_id"] != token:
        raise CompactRendererMlxSpineRunnerError(
            f"{family} --modelsize-candidate-id {token!r} did not round-trip "
            f"through the canonical modelsize analyzer; rebuilt {row['candidate_id']!r}"
        )
    return row


def _modelsize_control_contract(
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return canonical modelsize-control semantics without inventing authority."""

    if not candidate:
        return None
    contract = candidate.get("modelsize_control_contract")
    if not isinstance(contract, Mapping):
        return None
    out = dict(contract)
    out.setdefault(
        "control_precedence",
        modelsize_control_precedence_contract(candidate),
    )
    return out


def _write_snerv_binary_profile_attachment(
    *,
    archive_path: str | Path | None,
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Write a SNeRV binary profile, fail-closed if unavailable."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    if archive_path is None or str(archive_path).strip() == "":
        return {
            "schema": "compact_runner_snerv_binary_profile_attachment.v1",
            "profile_written": False,
            "blockers": ["snerv_binary_profile_archive_path_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    profile_path = out / "snerv_binary_profile.json"
    try:
        resolved_archive = _resolve(archive_path, base=Path(repo_root))
        profile = write_snerv_binary_profile(
            input_path=resolved_archive,
            output_path=profile_path,
        )
    except (
        OSError,
        SnervArchiveError,
        SnervBinaryProfileError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        return {
            "schema": "compact_runner_snerv_binary_profile_attachment.v1",
            "profile_written": False,
            "profile_path": profile_path.as_posix(),
            "blockers": [f"snerv_binary_profile_failed:{exc}"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    lf_profile = dict(profile.get("lf_quant_profile") or {})
    return {
        "schema": "compact_runner_snerv_binary_profile_attachment.v1",
        "profile_written": True,
        "profile_path": profile_path.as_posix(),
        "verdict": profile.get("verdict"),
        "charged_archive_bytes": profile.get("charged_archive_bytes"),
        "snar1_packet_bytes": profile.get("snar1_packet_bytes"),
        "lf_payload_bytes": lf_profile.get("section_bytes"),
        "lf_payload_fraction_of_packet": lf_profile.get("section_fraction_of_packet"),
        "lf_payload_bytes_per_coeff": lf_profile.get("bytes_per_coeff"),
        "blockers": list(profile.get("blockers") or []),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _run_snerv_scorer_loop_qat_attachment(
    *,
    requested: bool,
    output_dir: str | Path,
    num_pairs: int,
    levels: int,
    wavelet: str,
    target_bits_per_coeff: float,
    source_video_path: str | Path,
    upstream_dir: str | Path,
    distillation_device: str,
    step_map_bins: int,
    qat_bits: int,
    decoder_payload_codec: str,
    lf_payload_codec: str,
    snerv_spectra_preserving_adapter: bool,
    snerv_model_size_adapter: str,
    snerv_fc_dim: int,
    snerv_emb_size: int,
    snerv_patch_radius: int,
    snerv_mfu_scales: tuple[int, ...],
    snerv_hfr_gain: float,
    snerv_temporal_context: int,
    snerv_temporal_mode: str,
    max_trials: int,
    search_mode: str,
    perturb_scale: float,
    byte_pressure_multiplier: float,
    section_value_pressure_multiplier: float,
    max_archive_byte_growth: int | None,
    pose_slack: float,
    seg_slack: float,
    pair_stride: int,
    start_pair: int,
    prioritized_pair_indices: tuple[int, ...],
    pair_guard_min_score_improved_fraction: float,
    pair_guard_max_pose_worsened_fraction: float,
    component_guard_mode: str,
    seed: int,
) -> dict[str, Any]:
    """Run SNeRV receiver-priced scorer-loop QAT and persist its result.

    This is executable score-aware evidence for the SNeRV carrier, but it is not
    native MLX training and it never becomes promotion authority by itself.
    """

    attachment_dir = Path(output_dir).expanduser().resolve(strict=False)
    result_path = attachment_dir / "snerv_scorer_loop_qat_result.json"
    progress_path = attachment_dir / "snerv_scorer_loop_qat_progress.jsonl"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    if not requested:
        payload = {
            "schema": "compact_runner_snerv_scorer_loop_qat_attachment.v1",
            "executed": False,
            "requested": False,
            "component_guard_mode": str(component_guard_mode),
            "decoder_payload_codec": str(decoder_payload_codec),
            "lf_payload_codec": str(lf_payload_codec),
            "blockers": ["snerv_scorer_loop_qat_not_requested"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        _write_json(result_path, payload)
        return {**payload, "result_path": result_path.as_posix()}

    try:
        from tac.substrates.snerv_inverse_steg_carrier import (
            scorer_loop_decoder_qat as qat_mod,
        )

        result = qat_mod.run_snerv_scorer_loop_decoder_qat(
            n_pairs=int(num_pairs),
            levels=int(levels),
            wavelet=str(wavelet),
            target_bits_per_coeff=float(target_bits_per_coeff),
            pair_stride=int(pair_stride),
            start_pair=int(start_pair),
            pair_indices=(
                tuple(int(value) for value in prioritized_pair_indices)
                if prioritized_pair_indices
                else None
            ),
            upstream_dir=Path(upstream_dir).as_posix(),
            video_path=Path(source_video_path).as_posix(),
            device=str(distillation_device),
            step_map_bins=int(step_map_bins),
            snerv_spectra_preserving_adapter=bool(snerv_spectra_preserving_adapter),
            snerv_model_size_adapter=str(snerv_model_size_adapter),
            snerv_fc_dim=int(snerv_fc_dim),
            snerv_emb_size=int(snerv_emb_size),
            snerv_patch_radius=int(snerv_patch_radius),
            snerv_mfu_scales=tuple(int(v) for v in snerv_mfu_scales),
            snerv_hfr_gain=float(snerv_hfr_gain),
            snerv_temporal_context=int(snerv_temporal_context),
            snerv_temporal_mode=str(snerv_temporal_mode),
            decoder_payload_codec=str(decoder_payload_codec),
            lf_payload_codec=str(lf_payload_codec),
            qat_bits=int(qat_bits),
            max_trials=int(max_trials),
            search_mode=str(search_mode),
            perturb_scale=float(perturb_scale),
            byte_pressure_multiplier=float(byte_pressure_multiplier),
            section_value_pressure_multiplier=float(section_value_pressure_multiplier),
            max_archive_byte_growth=(
                None
                if max_archive_byte_growth is None
                else int(max_archive_byte_growth)
            ),
            pose_slack=float(pose_slack),
            seg_slack=float(seg_slack),
            pair_guard_min_score_improved_fraction=float(
                pair_guard_min_score_improved_fraction
            ),
            pair_guard_max_pose_worsened_fraction=float(
                pair_guard_max_pose_worsened_fraction
            ),
            component_guard_mode=str(component_guard_mode),
            seed=int(seed),
            progress_callback=_snerv_scorer_loop_progress_callback(progress_path),
        )
        result_payload = (
            result.as_jsonable() if hasattr(result, "as_jsonable") else dict(result)
        )
        blockers = list(result_payload.get("blockers") or [])
        if int(num_pairs) < CONTEST_PAIR_COUNT:
            blockers.append("snerv_scorer_loop_qat_partial_pair_coverage")
        payload = {
            "schema": "compact_runner_snerv_scorer_loop_qat_attachment.v1",
            "executed": True,
            "requested": True,
            "axis_tag": "[macOS-CPU advisory]",
            "n_pairs": int(num_pairs),
            "source_pair_indices": [
                int(value)
                for value in result_payload.get("source_pair_indices") or []
            ],
            "prioritized_pair_training": {
                "schema": "compact_snerv_prioritized_pair_training.v1",
                "enabled": bool(prioritized_pair_indices),
                "pair_indices": [int(value) for value in prioritized_pair_indices],
                "pair_count": len(prioritized_pair_indices),
                "consumed_by_cpu_advisory_qat": bool(prioritized_pair_indices),
                "sampling_scope": "snerv_cpu_advisory_and_scorer_loop_qat_pair_subset",
                **FALSE_AUTHORITY,
            },
            "levels": int(levels),
            "wavelet": str(wavelet),
            "target_bits_per_coeff": float(target_bits_per_coeff),
            "snerv_model_size_adapter": str(snerv_model_size_adapter),
            "snerv_spectra_preserving_adapter": bool(snerv_spectra_preserving_adapter),
            "snerv_fc_dim": int(snerv_fc_dim),
            "snerv_emb_size": int(snerv_emb_size),
            "snerv_patch_radius": int(snerv_patch_radius),
            "snerv_mfu_scales": [int(v) for v in snerv_mfu_scales],
            "snerv_hfr_gain": float(snerv_hfr_gain),
            "snerv_temporal_context": int(snerv_temporal_context),
            "snerv_temporal_mode": str(snerv_temporal_mode),
            "qat_bits": int(qat_bits),
            "lf_payload_codec": str(
                result_payload.get("lf_payload_codec") or lf_payload_codec
            ),
            "max_trials": int(max_trials),
            "search_mode": str(search_mode),
            "component_guard_mode": str(
                result_payload.get("component_guard_mode") or component_guard_mode
            ),
            "pair_robust_admission": result_payload.get(
                "pair_robust_admission"
            ),
            "section_value_pressure_multiplier": float(
                section_value_pressure_multiplier
            ),
            "result": result_payload,
            "accepted_improvement": bool(
                result_payload.get("accepted_improvement")
            ),
            "receiver_contract_satisfied": bool(
                result_payload.get("receiver_contract_satisfied")
            ),
            "ready_for_pose_guard_gate": bool(
                result_payload.get("ready_for_pose_guard_gate")
            ),
            "improvement_score_delta": result_payload.get("improvement_score_delta"),
            "improvement_d_pose_delta": result_payload.get("improvement_d_pose_delta"),
            "improvement_d_seg_delta": result_payload.get("improvement_d_seg_delta"),
            "scorer_loop_evaluations": result_payload.get(
                "scorer_loop_evaluations"
            ),
            "progress_jsonl_path": progress_path.as_posix(),
            "progress_jsonl_sha256": (
                _sha256_file(progress_path) if progress_path.is_file() else None
            ),
            "blockers": _dedupe(blockers),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    except Exception as exc:  # pragma: no cover - covered through runner contract
        payload = {
            "schema": "compact_runner_snerv_scorer_loop_qat_attachment.v1",
            "executed": False,
            "requested": True,
            "failure": repr(exc),
            "component_guard_mode": str(component_guard_mode),
            "decoder_payload_codec": str(decoder_payload_codec),
            "lf_payload_codec": str(lf_payload_codec),
            "progress_jsonl_path": progress_path.as_posix(),
            "blockers": ["snerv_scorer_loop_qat_failed"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    _write_json(result_path, payload)
    return {
        **payload,
        "result_path": result_path.as_posix(),
        "result_sha256": _sha256_file(result_path),
    }


def _run_snerv_native_mlx_export_attachment(
    *,
    requested: bool,
    output_dir: str | Path,
    num_pairs: int,
    source_video_path: str | Path,
    scorer_upstream_dir: str | Path,
    modelsize_candidate: Mapping[str, Any] | None,
    prioritized_pair_indices: tuple[int, ...],
    repo_root: str | Path,
    allow_overwrite: bool,
    retain_receiver_output: bool,
    receiver_proof_timeout_seconds: int,
    run_scorer_loop_qat: bool,
    scorer_loop_qat_max_trials: int,
    scorer_loop_qat_search_mode: str,
    scorer_loop_qat_qat_bits: int,
    scorer_loop_qat_decoder_payload_codec: str,
    scorer_loop_qat_lf_payload_codec: str,
    scorer_loop_qat_component_guard_mode: str,
    scorer_loop_qat_device: str,
    recon_pixel_weight_path: str | Path | None,
    recon_pixel_weight_manifest_path: str | Path | None,
    recon_pixel_weight_normalize: str,
    native_mlx_decoder_train_steps: int,
    native_mlx_decoder_train_lr: float,
    native_mlx_decoder_train_ridge: float,
    native_mlx_decoder_train_optimizer: str,
) -> dict[str, Any]:
    """Run the native MLX SNeRV train/export/archive bridge.

    This attachment is the first real MLX-owned SNeRV export path. It remains
    false-authority until scorer-aware long training and exact auth replay
    clear, but a receiver-proofed archive here is stronger than a surface-only
    adapter contract and should feed the curriculum gate.
    """

    attachment_dir = Path(output_dir).expanduser().resolve(strict=False)
    result_path = attachment_dir / "snerv_mlx_native_export_attachment.json"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    if not requested:
        payload = {
            "schema": "compact_runner_snerv_mlx_native_export_attachment.v1",
            "executed": False,
            "requested": False,
            "blockers": ["snerv_mlx_native_export_not_requested"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        _write_json(result_path, payload)
        return {**payload, "result_path": result_path.as_posix()}

    try:
        from tac.substrates.snerv_inverse_steg_carrier import (
            mlx_native_train_export as native_mod,
        )

        artifact = native_mod.train_export_snerv_mlx_native(
            output_dir=attachment_dir / "native_train_export",
            num_pairs=int(num_pairs),
            source_video_path=Path(source_video_path),
            modelsize_candidate=modelsize_candidate,
            scorer_upstream_dir=Path(scorer_upstream_dir),
            repo_root=Path(repo_root),
            run_archive_export=True,
            pair_indices=(
                tuple(int(value) for value in prioritized_pair_indices)
                if prioritized_pair_indices
                else None
            ),
            retain_receiver_output=bool(retain_receiver_output),
            receiver_proof_timeout_seconds=int(receiver_proof_timeout_seconds),
            run_scorer_loop_qat=bool(run_scorer_loop_qat),
            scorer_loop_qat_max_trials=int(scorer_loop_qat_max_trials),
            scorer_loop_qat_search_mode=str(scorer_loop_qat_search_mode),
            scorer_loop_qat_qat_bits=int(scorer_loop_qat_qat_bits),
            scorer_loop_qat_decoder_payload_codec=str(
                scorer_loop_qat_decoder_payload_codec
            ),
            scorer_loop_qat_lf_payload_codec=str(scorer_loop_qat_lf_payload_codec),
            scorer_loop_qat_component_guard_mode=str(
                scorer_loop_qat_component_guard_mode
            ),
            scorer_loop_qat_device=str(scorer_loop_qat_device),
            recon_pixel_weight_path=recon_pixel_weight_path,
            recon_pixel_weight_manifest_path=recon_pixel_weight_manifest_path,
            recon_pixel_weight_normalize=str(recon_pixel_weight_normalize),
            native_mlx_decoder_train_steps=int(native_mlx_decoder_train_steps),
            native_mlx_decoder_train_lr=float(native_mlx_decoder_train_lr),
            native_mlx_decoder_train_ridge=float(native_mlx_decoder_train_ridge),
            native_mlx_decoder_train_optimizer=str(native_mlx_decoder_train_optimizer),
            allow_overwrite=bool(allow_overwrite),
        )
        native_training_export_guard = build_snerv_mlx_native_training_export_guard(
            artifact
        )
        blockers = [
            *list(artifact.get("blockers") or []),
            *list(native_training_export_guard.get("blockers") or []),
        ]
        if int(num_pairs) < CONTEST_PAIR_COUNT:
            blockers.append("snerv_mlx_native_export_partial_pair_coverage")
        if artifact.get("receiver_proof_passed") is not True:
            blockers.append("snerv_mlx_native_receiver_proof_missing_or_failed")
        native_scorer_loop = dict(artifact.get("scorer_loop_qat") or {})
        native_mlx_full600_export_proof_ready = int(num_pairs) >= CONTEST_PAIR_COUNT
        native_mlx_full600_campaign_ready = bool(
            native_mlx_full600_export_proof_ready
            and native_training_export_guard.get("export_guard_passed") is True
            and artifact.get("score_aware_long_training_executed") is True
            and artifact.get("native_mlx_training_executed") is True
            and artifact.get("receiver_proof_passed") is True
        )
        if native_mlx_full600_export_proof_ready and not native_mlx_full600_campaign_ready:
            blockers.extend(
                [
                    "snerv_mlx_native_export_closed_form_not_training",
                    "snerv_mlx_native_full600_not_campaign_ready_without_learned_training",
                ]
            )
        payload = {
            "schema": "compact_runner_snerv_mlx_native_export_attachment.v1",
            "executed": True,
            "requested": True,
            "axis_tag": "[macOS-MLX research-signal]",
            "num_pairs": int(num_pairs),
            "source_pair_indices": [
                int(value) for value in artifact.get("source_pair_indices") or []
            ],
            "prioritized_pair_training": {
                "schema": "compact_snerv_native_mlx_prioritized_pair_training.v1",
                "enabled": bool(prioritized_pair_indices),
                "pair_indices": [int(value) for value in prioritized_pair_indices],
                "pair_count": len(prioritized_pair_indices),
                "consumed_by_native_mlx_train_export": bool(prioritized_pair_indices),
                "sampling_scope": "snerv_native_mlx_target_hydration_pair_subset",
                **FALSE_AUTHORITY,
            },
            "artifact_schema": artifact.get("schema"),
            "native_mlx_training_executed": bool(
                artifact.get("native_mlx_training_executed")
            ),
            "native_mlx_training_kind": artifact.get("native_mlx_training_kind"),
            "native_mlx_hf_decoder_training": artifact.get(
                "native_mlx_hf_decoder_training"
            ),
            "native_mlx_training_export_guard": native_training_export_guard,
            "artifact_report_path": artifact.get("report_path"),
            "packet_path": artifact.get("packet_path"),
            "packet_bytes": artifact.get("packet_bytes"),
            "packet_sha256": artifact.get("packet_sha256"),
            "archive_path": artifact.get("archive_path"),
            "archive_bytes": artifact.get("archive_bytes"),
            "archive_sha256": artifact.get("archive_sha256"),
            "receiver_proof_path": artifact.get("receiver_proof_path"),
            "receiver_proof_passed": bool(artifact.get("receiver_proof_passed")),
            "receiver_contract_satisfied": bool(
                artifact.get("receiver_contract_satisfied")
            ),
            "scorer_loop_qat_attached": bool(native_scorer_loop.get("executed")),
            "scorer_loop_qat_receiver_contract_satisfied": bool(
                native_scorer_loop.get("receiver_contract_satisfied")
            ),
            "scorer_loop_qat_ready_for_pose_guard_gate": bool(
                native_scorer_loop.get("ready_for_pose_guard_gate")
            ),
            "scorer_loop_qat_accepted_improvement": bool(
                native_scorer_loop.get("accepted_improvement")
            ),
            "scorer_loop_qat_best_materialized": bool(
                native_scorer_loop.get("emitted_packet_uses_scorer_loop_best_decoder")
            ),
            "score_aware_hf_decoder_fit_executed": bool(
                artifact.get("score_aware_hf_decoder_fit_executed")
            ),
            "score_aware_long_training_executed": bool(
                artifact.get("score_aware_long_training_executed")
            ),
            "native_mlx_train_export_attached": True,
            "native_mlx_full600_export_proof_ready": (
                native_mlx_full600_export_proof_ready
            ),
            "native_mlx_full600_campaign_ready": native_mlx_full600_campaign_ready,
            "recon_pixel_weight_manifest_path": (
                Path(recon_pixel_weight_manifest_path).as_posix()
                if recon_pixel_weight_manifest_path is not None
                else None
            ),
            "recon_pixel_weight": artifact.get("recon_pixel_weight"),
            "blockers": _dedupe(blockers),
            "artifact": artifact,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    except Exception as exc:  # pragma: no cover - exercised through runner tests
        payload = {
            "schema": "compact_runner_snerv_mlx_native_export_attachment.v1",
            "executed": False,
            "requested": True,
            "failure": repr(exc),
            "blockers": ["snerv_mlx_native_export_failed"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    _write_json(result_path, payload)
    return {
        **payload,
        "result_path": result_path.as_posix(),
        "result_sha256": _sha256_file(result_path),
    }


def _pr95_long_campaign_prelaunch_blockers(
    candidate_curriculum_plan: Mapping[str, Any],
    *,
    epochs: int,
) -> list[str]:
    """Return blockers that forbid long carrier campaigns before launch."""

    if int(epochs) < 8:
        return []
    gate = candidate_curriculum_plan.get("long_campaign_prelaunch_gate")
    if not isinstance(gate, Mapping):
        return ["pr95_long_campaign_prelaunch_gate_missing"]
    if gate.get("launch_allowed") is True:
        return []
    return _dedupe(
        [
            "pr95_long_campaign_prelaunch_gate_failed",
            *list(gate.get("blockers") or []),
        ]
    )


def _bind_hi_nerv_modelsize_launch_pressure(
    *,
    modelsize_candidate: Mapping[str, Any] | None,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    allow_unscored_research_smoke: bool,
) -> dict[str, Any]:
    """Return effective scorer weights for a modelsize-conditioned launch."""

    candidate_selected = bool(modelsize_candidate)
    effective_seg = float(segnet_distillation_weight)
    effective_pose = float(pose_distillation_weight)
    mutations: list[dict[str, Any]] = []
    if candidate_selected and not bool(allow_unscored_research_smoke):
        if effective_seg == 0.0:
            effective_seg = HI_NERV_MODELSIZE_DEFAULT_SEGNET_DISTILLATION_WEIGHT
            mutations.append(
                {
                    "field": "segnet_distillation_weight",
                    "before": float(segnet_distillation_weight),
                    "after": effective_seg,
                    "reason": (
                        "modelsize-conditioned HiNeRV campaigns require a real "
                        "SegNet teacher; zero remains available only for "
                        "explicit unscored research smokes"
                    ),
                }
            )
        if effective_pose == 0.0:
            effective_pose = HI_NERV_MODELSIZE_DEFAULT_POSE_DISTILLATION_WEIGHT
            mutations.append(
                {
                    "field": "pose_distillation_weight",
                    "before": float(pose_distillation_weight),
                    "after": effective_pose,
                    "reason": (
                        "modelsize-conditioned HiNeRV campaigns require a real "
                        "PoseNet teacher; SegNet-only pressure is not a "
                        "frontier-targeting contest objective"
                    ),
                }
            )
    source = "caller_supplied"
    if mutations:
        source = "modelsize_candidate_minimum_joint_scorer_pressure"
    elif candidate_selected and not bool(allow_unscored_research_smoke):
        source = "caller_supplied_modelsize_frontier_pressure"
    return {
        "schema": "compact_hi_nerv_modelsize_launch_pressure.v1",
        "candidate_conditioned": candidate_selected,
        "allow_unscored_research_smoke": bool(allow_unscored_research_smoke),
        "segnet_distillation_weight": effective_seg,
        "pose_distillation_weight": effective_pose,
        "source": source,
        "mutations": mutations,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _hi_nerv_launch_control_precedence_report(
    *,
    modelsize_candidate: Mapping[str, Any] | None,
    source_faithfulness: Mapping[str, Any],
    modelsize_launch_pressure: Mapping[str, Any],
    decoder_weight_waterfill_plan_metadata: Mapping[str, Any],
    optimizer_policy: Mapping[str, Any],
) -> dict[str, Any]:
    modelsize_precedence = (
        modelsize_control_precedence_contract(modelsize_candidate)
        if modelsize_candidate
        else None
    )
    child_layers = [
        {
            "layer_id": "official_hinerv_source_base",
            "specificity": 30,
            "active": bool(source_faithfulness.get("official_hinerv_control")),
            "role": "required_source_faithfulness_guardrail",
        },
        {
            "layer_id": "pact_modelsize_candidate",
            "specificity": 70,
            "active": bool(modelsize_candidate),
            "role": "receiver_visible_capacity_and_byte_budget_child_rule",
        },
        {
            "layer_id": "pact_joint_scorer_pressure",
            "specificity": 80,
            "active": bool(modelsize_launch_pressure.get("candidate_conditioned")),
            "role": "contest_scorer_teacher_child_rule",
        },
        {
            "layer_id": "pact_optimizer_qat_policy",
            "specificity": 85,
            "active": bool(optimizer_policy),
            "role": "contest_optimizer_and_rate_pressure_child_rule",
        },
        {
            "layer_id": "pact_decoder_weight_waterfill",
            "specificity": 90,
            "active": bool(decoder_weight_waterfill_plan_metadata.get("attached")),
            "role": "decoder_weight_saliency_allocator_child_rule",
        },
        {
            "layer_id": "promotion_and_exact_eval_gates",
            "specificity": 100,
            "active": True,
            "role": "fail_closed_authority_guardrail",
        },
    ]
    active_layers = [row for row in child_layers if row["active"]]
    highest = max(active_layers, key=lambda row: int(row["specificity"]))
    return {
        "schema": "hi_nerv_launch_control_precedence.v1",
        "cascade_model": "css_like_specificity_with_fail_closed_authority_gates",
        "more_finely_grained_child_rules_take_priority": True,
        "child_rules_override_parent_defaults": True,
        "parent_rules_remain_required_guardrails": True,
        "official_controls_are_required_base_not_final_optimizer": True,
        "pact_controls_take_priority_inside_source_faithful_subset": True,
        "conflict_resolution_high_to_low": [
            "promotion_and_exact_eval_gates",
            "pact_decoder_weight_waterfill",
            "pact_optimizer_qat_policy",
            "pact_joint_scorer_pressure",
            "pact_target_modelsize_or_modelsize_candidate",
            "pact_hard_byte_ceiling",
            "official_hinerv_source_base",
            "manual_cli_defaults",
        ],
        "highest_specificity_active_layer": highest["layer_id"],
        "layers_low_to_high_specificity": sorted(
            child_layers,
            key=lambda row: int(row["specificity"]),
        ),
        "modelsize_control_precedence": modelsize_precedence,
        "source_base_blockers": list(
            source_faithfulness.get("official_hinerv_blockers") or []
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _resolve_hi_nerv_optimizer_policy(
    *,
    requested_policy: str,
    epochs: int,
    optimizer_kind: str,
) -> dict[str, Any]:
    """Resolve whether HiNeRV uses PR95 curriculum or native MLX optimizer.

    Long runs used to infer PR95 curriculum solely from ``epochs >= 8``. That
    made planner rows labelled ``lion``/``adafactor``/``rmsprop`` silently run
    the PR95 Muon+AdamW curriculum instead. This helper makes that authority
    explicit and machine-checkable.
    """

    policy = str(requested_policy or "auto").strip().lower()
    if policy not in HI_NERV_OPTIMIZER_POLICIES:
        raise CompactRendererMlxSpineRunnerError(
            "hi_nerv_optimizer_policy must be one of "
            f"{HI_NERV_OPTIMIZER_POLICIES}; got {requested_policy!r}"
        )
    optimizer = str(
        optimizer_kind or DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    ).strip().lower()
    if policy == "auto":
        # Preserve the PR95-faithful control row for the historical AdamW
        # curriculum.  The Pact default is already Muon+AdamW and must be
        # consumed by the shared score-aware adapter, not swallowed by the
        # source-PR95 8-stage policy.
        resolved = (
            "pr95_curriculum"
            if int(epochs) >= 8 and optimizer == "adamw"
            else "native_optimizer"
        )
    else:
        resolved = policy
    if resolved == "pr95_curriculum" and int(epochs) < 8:
        raise CompactRendererMlxSpineRunnerError(
            "hi_nerv PR95 curriculum requires epochs >= 8; use "
            "--hi-nerv-optimizer-policy native_optimizer for short native "
            "optimizer probes"
        )
    if resolved == "pr95_curriculum" and optimizer != "adamw":
        raise CompactRendererMlxSpineRunnerError(
            "hi_nerv PR95 curriculum owns the optimizer schedule "
            "(Muon+AdamW); non-adamw --optimizer-kind would be ignored. Use "
            "--hi-nerv-optimizer-policy native_optimizer to run "
            f"{optimizer!r} as a real native MLX optimizer."
        )
    pr95_enabled = resolved == "pr95_curriculum"
    return {
        "schema": "compact_hi_nerv_optimizer_policy.v1",
        "requested_policy": policy,
        "resolved_policy": resolved,
        "optimizer_kind": optimizer,
        "pr95_faithful_curriculum_enabled": pr95_enabled,
        "native_optimizer_active": resolved == "native_optimizer",
        "optimizer_kind_consumed_by_native_mlx": resolved == "native_optimizer",
        "optimizer_kind_consumed_by_pr95_curriculum": pr95_enabled,
        "effective_optimizer_label": (
            "pr95_8stage_muon_adamw" if pr95_enabled else optimizer
        ),
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _resolve_mlx_score_aware_optimizer_controls(
    *,
    optimizer_kind: str,
    requested_weight_decay: float | None,
    grad_clip_max_norm: float | None,
    warmup_epochs: int,
    warmup_steps_per_epoch: int,
    cosine_decay_enabled: bool,
    cosine_decay_total_epochs: int | None,
    cosine_decay_min_lr_ratio: float,
    run_epochs: int,
) -> dict[str, Any]:
    """Resolve concrete optimizer controls before the MLX adapter is built."""

    kind = str(
        optimizer_kind or DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    ).strip().lower()
    if kind not in SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS:
        raise CompactRendererMlxSpineRunnerError(
            "optimizer_kind must be one of "
            f"{SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS}; got {optimizer_kind!r}"
        )
    weight_decay_supported = kind in MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS
    if requested_weight_decay is not None and not weight_decay_supported:
        raise CompactRendererMlxSpineRunnerError(
            "--optimizer-weight-decay is only supported for "
            f"{MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS}; got {kind!r}"
        )
    effective_weight_decay = (
        (
            DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_WEIGHT_DECAY
            if requested_weight_decay is None
            else float(requested_weight_decay)
        )
        if weight_decay_supported
        else None
    )
    if grad_clip_max_norm is not None and float(grad_clip_max_norm) <= 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "--optimizer-grad-clip-max-norm must be > 0 when provided"
        )
    resolved_warmup_epochs = int(warmup_epochs)
    resolved_warmup_steps_per_epoch = int(warmup_steps_per_epoch)
    if resolved_warmup_epochs < 0:
        raise CompactRendererMlxSpineRunnerError(
            "--optimizer-warmup-epochs must be >= 0"
        )
    if resolved_warmup_steps_per_epoch <= 0:
        raise CompactRendererMlxSpineRunnerError(
            "--optimizer-warmup-steps-per-epoch must be > 0"
        )
    resolved_cosine_total_epochs = (
        int(cosine_decay_total_epochs)
        if cosine_decay_total_epochs is not None
        else None
    )
    cosine_total_defaulted = False
    if cosine_decay_enabled:
        if resolved_warmup_epochs <= 0:
            raise CompactRendererMlxSpineRunnerError(
                "--optimizer-cosine-decay-enabled requires "
                "--optimizer-warmup-epochs > 0"
            )
        if resolved_cosine_total_epochs is None:
            resolved_cosine_total_epochs = max(
                int(run_epochs), resolved_warmup_epochs + 1
            )
            cosine_total_defaulted = True
        if resolved_cosine_total_epochs <= resolved_warmup_epochs:
            raise CompactRendererMlxSpineRunnerError(
                "--optimizer-cosine-decay-total-epochs must be > "
                "--optimizer-warmup-epochs"
            )
    if float(cosine_decay_min_lr_ratio) < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "--optimizer-cosine-decay-min-lr-ratio must be >= 0"
        )
    return {
        "schema": "compact_mlx_score_aware_optimizer_controls.v1",
        "optimizer_kind": kind,
        "grad_clip_max_norm": (
            float(grad_clip_max_norm) if grad_clip_max_norm is not None else None
        ),
        "weight_decay_requested": (
            None if requested_weight_decay is None else float(requested_weight_decay)
        ),
        "weight_decay_effective": effective_weight_decay,
        "weight_decay_defaulted": (
            requested_weight_decay is None and weight_decay_supported
        ),
        "weight_decay_supported_by_optimizer": weight_decay_supported,
        "warmup_epochs": resolved_warmup_epochs,
        "warmup_steps_per_epoch": resolved_warmup_steps_per_epoch,
        "cosine_decay_enabled": bool(cosine_decay_enabled),
        "cosine_decay_total_epochs": resolved_cosine_total_epochs,
        "cosine_decay_total_epochs_defaulted_to_run_epochs": cosine_total_defaulted,
        "cosine_decay_min_lr_ratio": float(cosine_decay_min_lr_ratio),
        "borrowed_pr95_partition_rule": kind == "pact_muon_adamw",
        "original_pact_default_optimizer": kind == "pact_muon_adamw",
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _resolve_pact_compact_optimizer_policy(
    *,
    family: str,
    optimizer_controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare native-optimizer routing for Pact compact runner families."""

    controls = dict(optimizer_controls or {})
    optimizer = str(
        controls.get("optimizer_kind") or DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    ).strip().lower()
    return {
        "schema": "compact_pact_native_optimizer_policy.v1",
        "family": str(family),
        "requested_policy": "native_optimizer",
        "resolved_policy": "native_optimizer",
        "optimizer_kind": optimizer,
        "pr95_faithful_curriculum_enabled": False,
        "native_optimizer_active": True,
        "optimizer_kind_consumed_by_native_mlx": True,
        "optimizer_kind_consumed_by_pr95_curriculum": False,
        "effective_optimizer_label": optimizer,
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _build_pact_coder_qat_config_and_metadata(
    *,
    coder_aware_qat: bool,
    coder_qat_quant_bits: int,
    coder_qat_quant_residual_weight: float,
    coder_qat_magnitude_weight: float,
    coder_qat_delta_weight: float,
    coder_qat_c1a_entropy_weight: float,
    coder_qat_c1a_sigma: float,
    coder_qat_c1a_sample_size: int,
) -> tuple[Any, dict[str, Any]]:
    from tac.substrates._shared.mlx_score_aware import (
        CoderAwareQATConfig,
        coder_qat_metadata,
    )

    cfg = CoderAwareQATConfig(
        enabled=bool(coder_aware_qat),
        quant_bits=int(coder_qat_quant_bits),
        quant_residual_weight=float(coder_qat_quant_residual_weight),
        magnitude_weight=float(coder_qat_magnitude_weight),
        delta_weight=float(coder_qat_delta_weight),
        c1a_entropy_weight=float(coder_qat_c1a_entropy_weight),
        c1a_sigma=float(coder_qat_c1a_sigma),
        c1a_sample_size=int(coder_qat_c1a_sample_size),
    ).validated()
    return cfg, coder_qat_metadata(cfg)


def execute_snerv_inverse_steg_advisory_and_adapt(
    *,
    output_dir: str | Path,
    num_pairs: int,
    epochs: int,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    run_local_cpu_replay: bool | None = None,
    keep_local_replay_inflated: bool = False,
    cleanup_failed_local_replay_scratch: bool = True,
    run_post_export_materializers: bool = False,
    post_export_materializer_max_steps: int = 1,
    post_export_materializer_max_parallel: int = 0,
    post_export_materializer_max_experiments: int | None = 1,
    distillation_device: str = "cpu",
    modelsize_candidate: Mapping[str, Any] | None = None,
    prioritized_pair_indices: tuple[int, ...] = (),
    step_map_coder_mode: str | None = None,
    snerv_spectra_preserving_adapter: bool = False,
    snerv_model_size_adapter_override: str | None = None,
    snerv_fc_dim_override: int | None = None,
    snerv_emb_size_override: int | None = None,
    snerv_patch_radius_override: int | None = None,
    snerv_mfu_scales: tuple[int, ...] = (1, 2, 4),
    snerv_hfr_gain: float = 0.0,
    snerv_temporal_context_override: int | None = None,
    snerv_temporal_mode_override: str | None = None,
    recon_pixel_weight_path: str | Path | None = None,
    auto_joint_recon_pixel_weight: bool = False,
    recon_pixel_weight_normalize: str = "mean",
    run_native_mlx_export: bool = False,
    snerv_native_mlx_receiver_proof_timeout_seconds: int = 1800,
    snerv_native_mlx_decoder_train_steps: int = 0,
    snerv_native_mlx_decoder_train_lr: float = 1.0e-5,
    snerv_native_mlx_decoder_train_ridge: float = 1.0e-6,
    snerv_native_mlx_decoder_train_optimizer: str = "pact_guarded_adamw",
    run_scorer_loop_qat: bool = False,
    snerv_scorer_loop_max_trials: int = 2,
    snerv_scorer_loop_search_mode: str = "nes_pair_robust",
    snerv_scorer_loop_step_map_bins: int = 16,
    snerv_scorer_loop_qat_bits: int = 8,
    snerv_scorer_loop_lf_payload_codec: str = "portfolio_auto",
    snerv_scorer_loop_perturb_scale: float = 0.02,
    snerv_scorer_loop_byte_pressure_multiplier: float = 1.0,
    snerv_scorer_loop_section_value_pressure_multiplier: float = 1.0,
    snerv_scorer_loop_max_archive_byte_growth: int | None = None,
    snerv_scorer_loop_pose_slack: float = 0.0,
    snerv_scorer_loop_seg_slack: float = 0.0,
    snerv_scorer_loop_pair_stride: int = 1,
    snerv_scorer_loop_start_pair: int = 0,
    snerv_scorer_loop_pair_guard_min_score_improved_fraction: float = 0.0,
    snerv_scorer_loop_pair_guard_max_pose_worsened_fraction: float = 1.0,
    snerv_scorer_loop_component_guard_mode: str = "score_primary",
    random_seed: int = 0,
    upstream_dir: str | Path = DEFAULT_UPSTREAM_DIR,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run SNeRV's archive-bound advisory lane through the shared runner.

    This is not the missing MLX-native SNeRV trainer. It promotes the existing
    SNeRV packet/runtime/proof stack into the same queue-owned surface as
    HiNeRV while preserving explicit blockers until native MLX training exists.
    """

    from tac.substrates.snerv_inverse_steg_carrier.advisory import run_snerv_advisory
    from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (
        export_snerv_archive_bound_candidate_package,
    )
    from tac.substrates.snerv_inverse_steg_carrier.trained_ladder_bridge import (
        build_snerv_trained_ladder_row_from_advisory,
    )

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, upstream_dir)
    resolved_source_video = _resolve_source_video_path(source_video_path, base=root)
    prioritized_pair_indices = _normalize_nonnegative_int_sequence(
        prioritized_pair_indices
    )
    if (
        _has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        and not allow_overwrite
    ):
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    effective_recon_pixel_weight_path = recon_pixel_weight_path
    effective_recon_pixel_weight_manifest_path: str | Path | None = None
    recon_pixel_weight_auto_discovery: dict[str, Any] | None = None
    if auto_joint_recon_pixel_weight:
        if recon_pixel_weight_path is not None:
            raise CompactRendererMlxSpineRunnerError(
                "pass either --recon-pixel-weight-path or "
                "--auto-joint-recon-pixel-weight, not both"
            )
        (
            effective_recon_pixel_weight_path,
            recon_pixel_weight_auto_discovery,
        ) = _discover_joint_recon_pixel_weight_path(
            repo_root=root,
            num_pairs=int(num_pairs),
        )
        if recon_pixel_weight_auto_discovery is not None:
            selected_manifest_path = recon_pixel_weight_auto_discovery.get(
                "selected_manifest_path"
            )
            if selected_manifest_path:
                effective_recon_pixel_weight_manifest_path = str(
                    selected_manifest_path
                )

    planner = _score_aware_carrier_training_plan(
        "snerv",
        COMPACT_FAMILY_BACKENDS["snerv"],
    )
    candidate = dict(modelsize_candidate or {})
    levels = int(candidate.get("levels", 3))
    wavelet = str(candidate.get("wavelet", "haar"))
    target_bits_per_coeff = float(candidate.get("bits_per_coeff", 2.5))
    step_map_waterfill_bits_per_coeff = float(
        candidate.get("step_map_bits_per_coeff", 4.0)
    )
    decoder_payload_codec = str(
        candidate.get("decoder_payload_codec", "float32_lzma")
    )
    snerv_fc_dim = int(
        candidate.get(
            "fc_dim",
            candidate.get(
                "snerv_fc_dim",
                9 if snerv_fc_dim_override is None else int(snerv_fc_dim_override),
            ),
        )
    )
    snerv_emb_size = int(
        candidate.get(
            "emb_size",
            candidate.get(
                "snerv_emb_size",
                0 if snerv_emb_size_override is None else int(snerv_emb_size_override),
            ),
        )
    )
    snerv_patch_radius = int(
        candidate.get(
            "patch_radius",
            candidate.get(
                "snerv_patch_radius",
                (
                    1
                    if snerv_patch_radius_override is None
                    else int(snerv_patch_radius_override)
                ),
            ),
        )
    )
    snerv_temporal_context = int(
        candidate.get(
            "temporal_context",
            candidate.get(
                "snerv_temporal_context",
                (
                    0
                    if snerv_temporal_context_override is None
                    else int(snerv_temporal_context_override)
                ),
            ),
        )
    )
    snerv_temporal_mode = str(
        candidate.get(
            "temporal_mode",
            candidate.get(
                "snerv_temporal_mode",
                (
                    "delta"
                    if snerv_temporal_mode_override is None
                    else str(snerv_temporal_mode_override)
                ),
            ),
        )
    )
    raw_mfu_scales = candidate.get("mfu_scales", candidate.get("snerv_mfu_scales"))
    if raw_mfu_scales is None:
        resolved_snerv_mfu_scales = tuple(int(v) for v in snerv_mfu_scales)
    elif isinstance(raw_mfu_scales, str):
        resolved_snerv_mfu_scales = tuple(
            int(v.strip()) for v in raw_mfu_scales.split(",") if v.strip()
        )
    else:
        resolved_snerv_mfu_scales = tuple(int(v) for v in raw_mfu_scales)
    if not resolved_snerv_mfu_scales:
        raise CompactRendererMlxSpineRunnerError(
            "SNeRV modelsize candidate mfu_scales resolved to an empty tuple"
        )
    resolved_snerv_hfr_gain = float(
        candidate.get("hfr_gain", candidate.get("snerv_hfr_gain", snerv_hfr_gain))
    )
    candidate_adapter = str(candidate.get("snerv_model_size_adapter") or "")
    explicit_adapter = str(snerv_model_size_adapter_override or "")
    if candidate_adapter:
        if (
            snerv_spectra_preserving_adapter
            and candidate_adapter != SNERV_SPECTRA_PRESERVING_ADAPTER
        ):
            raise CompactRendererMlxSpineRunnerError(
                "SNeRV modelsize candidate adapter conflicts with "
                "--snerv-spectra-preserving-adapter: "
                f"{candidate_adapter!r} is not {SNERV_SPECTRA_PRESERVING_ADAPTER!r}"
            )
        if explicit_adapter and explicit_adapter != candidate_adapter:
            raise CompactRendererMlxSpineRunnerError(
                "SNeRV modelsize candidate adapter conflicts with "
                f"--snerv-model-size-adapter {explicit_adapter!r}; planner "
                f"candidate requires {candidate_adapter!r}"
            )
        snerv_model_size_adapter = candidate_adapter
    else:
        snerv_model_size_adapter = (
            explicit_adapter
            or (
                SNERV_SPECTRA_PRESERVING_ADAPTER
                if snerv_spectra_preserving_adapter
                else "snerv_fc_dim_emb_size_adapter_v1"
            )
        )
    resolved_step_map_coder_mode = (
        step_map_coder_mode
        if step_map_coder_mode is not None
        else ("waterfill" if candidate else "uniform")
    )
    prelaunch_curriculum_plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate or None,
        requested_epochs=int(epochs),
        num_pairs=int(num_pairs),
        step_map_coder_mode=str(resolved_step_map_coder_mode),
    )
    prelaunch_blockers = _pr95_long_campaign_prelaunch_blockers(
        prelaunch_curriculum_plan,
        epochs=int(epochs),
    )
    local_proof_bypasses_pr95_prelaunch = bool(run_native_mlx_export)
    if prelaunch_blockers and not local_proof_bypasses_pr95_prelaunch:
        refusal = _base_report(
            output_dir=out,
            mode="snerv_pr95_binding_prelaunch_refused",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        refusal.update(
            {
                "execute_family": "snerv",
                "num_pairs": int(num_pairs),
                "epochs_requested": int(epochs),
                "training_executed": False,
                "launch_refusal_reason": (
                    "8+ epoch SNeRV campaigns require the PR95-grade "
                    "prelaunch stack before advisory or training execution"
                ),
                "modelsize_candidate_selection": {
                    "schema": "compact_execute_modelsize_candidate_selection.v1",
                    "family": "snerv",
                    "selection_mode": (
                        "planner_candidate" if candidate else "manual_cli_knobs"
                    ),
                    "candidate": candidate or None,
                    "modelsize_control_contract": _modelsize_control_contract(
                        candidate
                    ),
                    "candidate_curriculum_plan": prelaunch_curriculum_plan,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "candidate_curriculum_plan": prelaunch_curriculum_plan,
                "blockers": prelaunch_blockers,
            }
        )
        refusal["candidate_feedback"] = write_nerv_candidate_feedback_files(
            runner_report=refusal,
            output_dir=out,
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, refusal)
        return {**refusal, "report_path": path.as_posix()}
    local_proof_prelaunch_blockers = list(prelaunch_blockers)
    advisory = run_snerv_advisory(
        n_pairs=int(num_pairs),
        pair_indices=(
            tuple(int(value) for value in prioritized_pair_indices)
            if prioritized_pair_indices
            else None
        ),
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        video_path=resolved_source_video.as_posix(),
        upstream_dir=scorer_upstream.as_posix(),
        device=distillation_device,
        step_map_coder_mode=str(resolved_step_map_coder_mode),
        step_map_waterfill_bits_per_coeff=step_map_waterfill_bits_per_coeff,
        decoder_payload_codec=decoder_payload_codec,
        snerv_model_size_adapter=snerv_model_size_adapter,
        snerv_fc_dim=snerv_fc_dim,
        snerv_emb_size=snerv_emb_size,
        snerv_patch_radius=snerv_patch_radius,
        snerv_mfu_scales=resolved_snerv_mfu_scales,
        snerv_hfr_gain=resolved_snerv_hfr_gain,
        snerv_temporal_context=snerv_temporal_context,
        snerv_temporal_mode=snerv_temporal_mode,
    )
    packet_path = out / "snerv_inverse_steg_advisory.snar"
    packet_path.write_bytes(advisory.receiver_archive_packet)
    advisory_payload = advisory.as_jsonable()
    advisory_payload.setdefault("schema", "snerv_inverse_steg_advisory.v1")
    advisory_source_pair_indices = _snerv_advisory_source_pair_indices(
        advisory,
        advisory_payload,
        requested_num_pairs=int(num_pairs),
    )
    advisory_payload["source_pair_indices"] = [
        int(value) for value in advisory_source_pair_indices
    ]
    advisory_payload["receiver_archive_packet_path"] = packet_path.as_posix()
    advisory_archive_bytes_total = int(
        getattr(
            advisory,
            "archive_bytes_total",
            advisory_payload.get(
                "archive_bytes_total",
                len(advisory.receiver_archive_packet),
            ),
        )
    )
    advisory_path = out / "snerv_inverse_steg_advisory.json"
    _write_json(advisory_path, advisory_payload)

    package_dir = out / "snerv_archive_bound_package"
    package = export_snerv_archive_bound_candidate_package(
        packet=advisory.receiver_archive_packet,
        output_dir=package_dir,
        repo_root=root,
        retain_receiver_output=bool(keep_local_replay_inflated),
        receiver_proof_timeout_seconds=1800,
        mlx_triage_argv=[
            "tools/run_compact_renderer_mlx_spine_runner.py",
            "--execute-family",
            "snerv",
        ],
    )
    package_path = package_dir / "archive_bound_candidate_adapter_package.json"
    candidate_rows = list(
        package.get("archive_bound_candidate_adapter_package", {}).get(
            "candidate_rows",
            [],
        )
        or []
    )
    row = candidate_rows[0] if candidate_rows else {}
    archive_path = row.get("candidate_archive_path")
    receiver_proof = dict(package.get("receiver_proof") or {})
    receiver_proof_path = receiver_proof.get("proof_path")
    receiver_proof_attached = bool(receiver_proof_path) and (
        receiver_proof.get("receiver_contract_satisfied") is True
        or receiver_proof.get("runtime_consumption_proof_ready") is True
        or receiver_proof.get("runtime_consumption_proof_passed") is True
    )
    trained_ladder_row_payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=advisory,
        archive_path=_resolve(archive_path, base=root) if archive_path else packet_path,
        archive_path_kind="contest_archive_zip" if archive_path else "receiver_snar_packet",
        receiver_proof=receiver_proof if archive_path else None,
        target_bits_per_coeff=target_bits_per_coeff,
        repo_root=root,
    )
    trained_ladder_row_path = out / "snerv_trained_ladder_row_payload.json"
    _write_json(trained_ladder_row_path, trained_ladder_row_payload)
    advisory_payload["trained_ladder_row_payload"] = trained_ladder_row_payload
    advisory_payload["trained_ladder_row_payload_path"] = (
        trained_ladder_row_path.as_posix()
    )
    _write_json(advisory_path, advisory_payload)

    mlx_prefilter_coverage = summarize_mlx_prefilter_coverage(
        mlx_profile_paths,
        root=root,
    )
    has_full_video_mlx_prefilter = bool(
        mlx_prefilter_coverage["has_full_video_mlx_prefilter"]
    )
    mlx_prefilter_local_replay_passed = bool(
        mlx_prefilter_coverage["local_replay_mlx_prefilter_passed"]
    )
    local_cpu_replay_summary: dict[str, Any] | None = None
    local_cpu_replay_paths: list[Path] = []
    local_cpu_replay_blockers: list[str] = []
    if archive_path:
        (
            local_cpu_replay_summary,
            local_cpu_replay_paths,
            local_cpu_replay_blockers,
        ) = _run_compact_local_cpu_replay_gate(
            archive_zip_path=_resolve(archive_path, base=root),
            runtime_submission_dir=package_dir / "submission",
            output_dir=out / "local_cpu_replay",
            upstream_dir=scorer_upstream,
            num_pairs=int(num_pairs),
            requested=run_local_cpu_replay,
            has_full_video_mlx_prefilter=has_full_video_mlx_prefilter,
            mlx_prefilter_local_replay_passed=mlx_prefilter_local_replay_passed,
            keep_inflated=keep_local_replay_inflated,
            cleanup_failed_scratch=cleanup_failed_local_replay_scratch,
            repo_root=root,
        )
    post_export_materializer_plan = _compile_carrier_post_export_materializer_plan(
        output_dir=out,
        archive_path=archive_path,
        archive_sha256=row.get("candidate_archive_sha256"),
        archive_bytes=row.get("candidate_archive_bytes"),
        family="snerv",
        runtime_submission_dir=package_dir / "submission",
        repo_root=root,
    )
    post_export_materializer_execution = (
        _execute_carrier_post_export_materializer_plan(
            plan=post_export_materializer_plan,
            requested=run_post_export_materializers,
            max_steps=post_export_materializer_max_steps,
            max_parallel=post_export_materializer_max_parallel,
            max_experiments=post_export_materializer_max_experiments,
            repo_root=root,
        )
    )
    snerv_binary_profile = _write_snerv_binary_profile_attachment(
        archive_path=archive_path,
        output_dir=out,
        repo_root=root,
    )
    snerv_scorer_loop_qat = _run_snerv_scorer_loop_qat_attachment(
        requested=bool(run_scorer_loop_qat),
        output_dir=out / "snerv_scorer_loop_qat",
        num_pairs=int(num_pairs),
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        source_video_path=resolved_source_video,
        upstream_dir=scorer_upstream,
        distillation_device=distillation_device,
        step_map_bins=int(snerv_scorer_loop_step_map_bins),
        qat_bits=int(snerv_scorer_loop_qat_bits),
        decoder_payload_codec=decoder_payload_codec,
        lf_payload_codec=str(snerv_scorer_loop_lf_payload_codec),
        snerv_spectra_preserving_adapter=bool(snerv_spectra_preserving_adapter),
        snerv_model_size_adapter=snerv_model_size_adapter,
        snerv_fc_dim=snerv_fc_dim,
        snerv_emb_size=snerv_emb_size,
        snerv_patch_radius=snerv_patch_radius,
        snerv_mfu_scales=resolved_snerv_mfu_scales,
        snerv_hfr_gain=resolved_snerv_hfr_gain,
        snerv_temporal_context=snerv_temporal_context,
        snerv_temporal_mode=snerv_temporal_mode,
        max_trials=int(snerv_scorer_loop_max_trials),
        search_mode=str(snerv_scorer_loop_search_mode),
        perturb_scale=float(snerv_scorer_loop_perturb_scale),
        byte_pressure_multiplier=float(
            snerv_scorer_loop_byte_pressure_multiplier
        ),
        section_value_pressure_multiplier=float(
            snerv_scorer_loop_section_value_pressure_multiplier
        ),
        max_archive_byte_growth=snerv_scorer_loop_max_archive_byte_growth,
        pose_slack=float(snerv_scorer_loop_pose_slack),
        seg_slack=float(snerv_scorer_loop_seg_slack),
        pair_stride=int(snerv_scorer_loop_pair_stride),
        start_pair=int(snerv_scorer_loop_start_pair),
        prioritized_pair_indices=tuple(int(value) for value in prioritized_pair_indices),
        pair_guard_min_score_improved_fraction=float(
            snerv_scorer_loop_pair_guard_min_score_improved_fraction
        ),
        pair_guard_max_pose_worsened_fraction=float(
            snerv_scorer_loop_pair_guard_max_pose_worsened_fraction
        ),
        component_guard_mode=str(snerv_scorer_loop_component_guard_mode),
        seed=int(random_seed),
    )
    resolved_snerv_modelsize_candidate = {
        **candidate,
        "fc_dim": int(snerv_fc_dim),
        "emb_size": int(snerv_emb_size),
        "patch_radius": int(snerv_patch_radius),
        "mfu_scales": [int(v) for v in resolved_snerv_mfu_scales],
        "hfr_gain": float(resolved_snerv_hfr_gain),
        "temporal_context": int(snerv_temporal_context),
        "temporal_mode": str(snerv_temporal_mode),
        "snerv_model_size_adapter": str(snerv_model_size_adapter),
        "decoder_payload_codec": str(decoder_payload_codec),
        "levels": int(levels),
        "wavelet": str(wavelet),
        "bits_per_coeff": float(target_bits_per_coeff),
        "step_map_bits_per_coeff": float(step_map_waterfill_bits_per_coeff),
        "snerv_native_mlx_decoder_train_steps": int(
            snerv_native_mlx_decoder_train_steps
        ),
        "snerv_native_mlx_decoder_train_lr": float(
            snerv_native_mlx_decoder_train_lr
        ),
        "snerv_native_mlx_decoder_train_ridge": float(
            snerv_native_mlx_decoder_train_ridge
        ),
        "snerv_native_mlx_decoder_train_optimizer": str(
            snerv_native_mlx_decoder_train_optimizer
        ),
    }
    snerv_mlx_native_export = _run_snerv_native_mlx_export_attachment(
        requested=bool(run_native_mlx_export),
        output_dir=out / "snerv_mlx_native_export",
        num_pairs=int(num_pairs),
        source_video_path=resolved_source_video,
        scorer_upstream_dir=scorer_upstream,
        modelsize_candidate=resolved_snerv_modelsize_candidate,
        prioritized_pair_indices=prioritized_pair_indices,
        repo_root=root,
        allow_overwrite=bool(allow_overwrite),
        retain_receiver_output=bool(keep_local_replay_inflated),
        receiver_proof_timeout_seconds=int(
            snerv_native_mlx_receiver_proof_timeout_seconds
        ),
        run_scorer_loop_qat=bool(run_scorer_loop_qat),
        scorer_loop_qat_max_trials=int(snerv_scorer_loop_max_trials),
        scorer_loop_qat_search_mode=str(snerv_scorer_loop_search_mode),
        scorer_loop_qat_qat_bits=int(snerv_scorer_loop_qat_bits),
        scorer_loop_qat_decoder_payload_codec=decoder_payload_codec,
        scorer_loop_qat_lf_payload_codec=str(snerv_scorer_loop_lf_payload_codec),
        scorer_loop_qat_component_guard_mode=str(
            snerv_scorer_loop_component_guard_mode
        ),
        scorer_loop_qat_device=str(distillation_device),
        recon_pixel_weight_path=effective_recon_pixel_weight_path,
        recon_pixel_weight_manifest_path=effective_recon_pixel_weight_manifest_path,
        recon_pixel_weight_normalize=str(recon_pixel_weight_normalize),
        native_mlx_decoder_train_steps=int(snerv_native_mlx_decoder_train_steps),
        native_mlx_decoder_train_lr=float(snerv_native_mlx_decoder_train_lr),
        native_mlx_decoder_train_ridge=float(snerv_native_mlx_decoder_train_ridge),
        native_mlx_decoder_train_optimizer=str(snerv_native_mlx_decoder_train_optimizer),
    )
    snerv_mlx_native_file_backed_evidence = (
        build_snerv_mlx_native_file_backed_evidence(
            snerv_mlx_native_export,
            required_num_pairs=CONTEST_PAIR_COUNT,
        )
    )
    snerv_mlx_native_adapter_contract_after_export = (
        build_snerv_mlx_native_adapter_contract(
            extra_evidence={
                "file_backed_export_artifact": snerv_mlx_native_export,
                "required_num_pairs": CONTEST_PAIR_COUNT,
            }
        )
    )
    snerv_mlx_native_export_verified = bool(
        snerv_mlx_native_export.get("executed")
        and snerv_mlx_native_export.get("receiver_proof_passed") is True
        and snerv_mlx_native_export.get("receiver_contract_satisfied") is True
        and snerv_mlx_native_file_backed_evidence.get(
            "required_pair_file_backed_export_proof_passed"
        )
    )
    candidate_curriculum_plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate or None,
        requested_epochs=int(epochs),
        num_pairs=int(num_pairs),
        step_map_coder_mode=str(resolved_step_map_coder_mode),
        measured_packet_bytes=advisory_archive_bytes_total,
        measured_archive_bytes=(
            int(row["candidate_archive_bytes"])
            if row.get("candidate_archive_bytes") is not None
            else None
        ),
        scorer_loop_qat_attached=bool(snerv_scorer_loop_qat.get("executed")),
        scorer_loop_qat_receiver_contract_satisfied=bool(
            snerv_scorer_loop_qat.get("receiver_contract_satisfied")
        ),
        scorer_loop_qat_ready_for_pose_guard_gate=bool(
            snerv_scorer_loop_qat.get("ready_for_pose_guard_gate")
        ),
        scorer_loop_qat_accepted_improvement=bool(
            snerv_scorer_loop_qat.get("accepted_improvement")
        ),
        receiver_proof_attached=receiver_proof_attached,
        full_video_local_prefilter_attached=has_full_video_mlx_prefilter,
        local_cpu_replay_gate_attached=local_cpu_replay_summary is not None,
        native_mlx_train_export_attached=bool(
            snerv_mlx_native_export.get("executed")
        ),
        native_mlx_long_training_bound=bool(
            snerv_mlx_native_export.get("score_aware_long_training_executed")
        ),
        native_mlx_receiver_proof_passed=snerv_mlx_native_export_verified,
        native_mlx_full600_campaign_ready=bool(
            snerv_mlx_native_export.get("native_mlx_full600_campaign_ready")
        ),
        native_mlx_scorer_loop_qat_attached=bool(
            snerv_mlx_native_export.get("scorer_loop_qat_attached")
        ),
        native_mlx_scorer_loop_qat_receiver_contract_satisfied=bool(
            snerv_mlx_native_export.get(
                "scorer_loop_qat_receiver_contract_satisfied"
            )
        ),
        native_mlx_scorer_loop_qat_ready_for_pose_guard_gate=bool(
            snerv_mlx_native_export.get("scorer_loop_qat_ready_for_pose_guard_gate")
        ),
        native_mlx_scorer_loop_qat_accepted_improvement=bool(
            snerv_mlx_native_export.get("scorer_loop_qat_accepted_improvement")
        ),
        native_mlx_scorer_loop_qat_best_materialized=bool(
            snerv_mlx_native_export.get("scorer_loop_qat_best_materialized")
        ),
        native_mlx_artifact_evidence=snerv_mlx_native_export,
    )

    blockers = _dedupe(
        [
            "contest_cpu_cuda_exact_eval_not_executed",
            *(
                []
                if snerv_mlx_native_export_verified
                else list(
                    snerv_mlx_native_adapter_contract_after_export.get("blockers")
                    or []
                )
            ),
            *local_proof_prelaunch_blockers,
            (
                "snerv_mlx_native_longer_staged_training_not_executed"
                if snerv_scorer_loop_qat.get("executed")
                else "snerv_longer_staged_score_aware_training_not_executed"
            ),
            *list(candidate_curriculum_plan.get("blockers") or []),
            *list(snerv_mlx_native_export.get("blockers") or []),
            *local_cpu_replay_blockers,
            *list(post_export_materializer_plan.get("blockers") or []),
            *list(post_export_materializer_execution.get("blockers") or []),
            *list(snerv_scorer_loop_qat.get("blockers") or []),
            *(
                []
                if snerv_binary_profile.get("profile_written")
                else list(snerv_binary_profile.get("blockers") or [])
            ),
            *list(row.get("blockers") or []),
            *list(mlx_prefilter_coverage.get("blockers") or []),
        ]
    )
    final = _base_report(
        output_dir=out,
        mode="executed_snerv_archive_bound_advisory_and_exported",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "execute_family": "snerv",
            "num_pairs": int(num_pairs),
            "epochs_requested": int(epochs),
            "coverage_valid_for_base_comparison": int(num_pairs) >= CONTEST_PAIR_COUNT,
            "training_executed": False,
            "adapter_smoke_only": False,
            "modelsize_candidate_selection": {
                "schema": "compact_execute_modelsize_candidate_selection.v1",
                "family": "snerv",
                "selection_mode": "planner_candidate" if candidate else "manual_cli_knobs",
                "candidate": candidate or None,
                "modelsize_control_contract": _modelsize_control_contract(candidate),
                "num_pairs_for_budget": CONTEST_PAIR_COUNT,
                "launch_levels": levels,
                "launch_bits_per_coeff": target_bits_per_coeff,
                "launch_step_map_coder_mode": str(resolved_step_map_coder_mode),
                "launch_step_map_waterfill_bits_per_coeff": (
                    step_map_waterfill_bits_per_coeff
                ),
                "launch_decoder_payload_codec": decoder_payload_codec,
                "candidate_curriculum_plan": candidate_curriculum_plan,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "snerv_recon_pixel_weight": {
                "schema": "compact_snerv_recon_pixel_weight_consumption.v1",
                "requested": effective_recon_pixel_weight_path is not None,
                "enabled": bool(
                    (
                        snerv_mlx_native_export.get("recon_pixel_weight") or {}
                    ).get("selected_packet_consumed")
                ),
                "path": (
                    Path(effective_recon_pixel_weight_path).as_posix()
                    if effective_recon_pixel_weight_path is not None
                    else None
                ),
                "manifest_path": (
                    Path(effective_recon_pixel_weight_manifest_path).as_posix()
                    if effective_recon_pixel_weight_manifest_path is not None
                    else None
                ),
                "normalize": str(recon_pixel_weight_normalize),
                "auto_discovery": recon_pixel_weight_auto_discovery,
                "native_export_consumed": bool(
                    (
                        snerv_mlx_native_export.get("recon_pixel_weight") or {}
                    ).get("selected_packet_consumed")
                ),
                "native_export_packet_source": snerv_mlx_native_export.get(
                    "packet_source"
                ),
                "native_mlx_export_consumption": snerv_mlx_native_export.get(
                    "recon_pixel_weight"
                ),
                "primary_archive_consumed": False,
                "primary_archive_source": (
                    "snerv_advisory_archive_packet_not_native_mlx_export"
                ),
                "primary_archive_path": archive_path,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "archive_path": archive_path,
            "archive_bytes": row.get("candidate_archive_bytes"),
            "archive_sha256": row.get("candidate_archive_sha256"),
            "advisory_report_path": advisory_path.as_posix(),
            "receiver_archive_packet_path": packet_path.as_posix(),
            "trained_ladder_row_payload_path": trained_ladder_row_path.as_posix(),
            "trained_ladder_row_payload": trained_ladder_row_payload,
            "runtime_package_dir": package_dir.as_posix(),
            "runtime_package_path": (
                package_path.as_posix() if package_path.is_file() else None
            ),
            "receiver_proof_report_paths": (
                [_resolve(receiver_proof_path, base=root).as_posix()]
                if receiver_proof_path
                else []
            ),
            "archive_bound_candidate_rows": candidate_rows,
            "candidate_curriculum_plan": candidate_curriculum_plan,
            "score_aware_carrier_training_plan": planner,
            "snerv_mlx_native_adapter_contract": (
                snerv_mlx_native_adapter_contract_after_export
            ),
            "score_aware_training": {
                "schema": "compact_snerv_archive_bound_advisory.v1",
                "status": (
                    "executed_cpu_advisory_plus_receiver_priced_scorer_loop_qat_"
                    "mlx_native_training_missing"
                    if snerv_scorer_loop_qat.get("executed")
                    else "executed_cpu_advisory_mlx_native_training_missing"
                ),
                "axis_tag": "[macOS-CPU advisory]",
                "levels": int(advisory.levels),
                "wavelet": advisory.wavelet,
                "target_bits_per_coeff": target_bits_per_coeff,
                "step_map_coder_mode": str(resolved_step_map_coder_mode),
                "step_map_waterfill_bits_per_coeff": (
                    step_map_waterfill_bits_per_coeff
                ),
                "decoder_payload_codec": decoder_payload_codec,
                "source_pair_indices": [
                    int(value) for value in advisory_source_pair_indices
                ],
                "prioritized_pair_training": {
                    "schema": "compact_snerv_prioritized_pair_training.v1",
                    "enabled": bool(prioritized_pair_indices),
                    "pair_indices": [
                        int(value) for value in prioritized_pair_indices
                    ],
                    "pair_count": len(prioritized_pair_indices),
                    "consumed_by_cpu_advisory": bool(prioritized_pair_indices),
                    "consumed_by_scorer_loop_qat": bool(
                        prioritized_pair_indices
                        and snerv_scorer_loop_qat.get("executed")
                    ),
                    "consumed_by_mlx_native_export": bool(
                        prioritized_pair_indices
                        and (
                            (
                                snerv_mlx_native_export.get(
                                    "prioritized_pair_training"
                                )
                                or {}
                            ).get("consumed_by_native_mlx_train_export")
                            is True
                        )
                    ),
                    "mlx_native_export_blocker": (
                        None
                        if (
                            not prioritized_pair_indices
                            or (
                                (
                                    snerv_mlx_native_export.get(
                                        "prioritized_pair_training"
                                    )
                                    or {}
                                ).get("consumed_by_native_mlx_train_export")
                                is True
                            )
                        )
                        else "snerv_mlx_native_prioritized_pair_hydration_not_consumed"
                    ),
                    "sampling_scope": (
                        "snerv_cpu_advisory_and_scorer_loop_qat_pair_subset"
                    ),
                    **FALSE_AUTHORITY,
                },
                "scorer_loop_component_guard_mode": str(
                    snerv_scorer_loop_component_guard_mode
                ),
                "score_linf": float(advisory.score_linf),
                "score_l2": float(advisory.score_l2),
                "d_seg_mean_linf": float(advisory.d_seg_mean_linf),
                "d_pose_mean_linf": float(advisory.d_pose_mean_linf),
                "archive_bytes_total": advisory_archive_bytes_total,
                "beats_frontier_rate": bool(advisory.beats_frontier_rate),
                "receiver_archive_replay_verified": bool(
                    advisory.receiver_archive_replay_verified
                ),
                "scorer_loop_qat": snerv_scorer_loop_qat,
                "mlx_native_export": snerv_mlx_native_export,
                "mlx_native_file_backed_export_evidence": (
                    snerv_mlx_native_file_backed_evidence
                ),
                "mlx_native_train_export_attached": bool(
                    snerv_mlx_native_export.get("executed")
                ),
                "mlx_native_hf_decoder_fit_executed": bool(
                    snerv_mlx_native_export.get(
                        "score_aware_hf_decoder_fit_executed"
                    )
                ),
                "mlx_native_receiver_proof_passed": snerv_mlx_native_export_verified,
                "mlx_native_training_required_next": (
                    not bool(
                        snerv_mlx_native_export.get(
                            "score_aware_long_training_executed"
                        )
                    )
                ),
                "authority": "macos_cpu_advisory_false_authority",
            },
            "reusable_optimization_followups": {
                "schema": "compact_carrier_reusable_optimization_hooks.v1",
                "applies_after_byte_closed_export": True,
                "required_hooks": [
                    "final_rate_attack_and_repair_materializers",
                    "bit_mask_and_step_map_entropy_recode",
                    "p18_p19_scorer_priced_waterfill",
                    "p11_selector_context_codec",
                    "p15_rebrotli_repack_order_search",
                    "receiver_proof_after_each_semantic_mutation",
                ],
                "no_ad_hoc_leaf_rule": (
                    "rate/materializer work must enter as queue-owned reusable "
                    "stages for all byte-closed carriers, not as private SNeRV "
                    "or HiNeRV one-offs"
                ),
                "post_export_materializer_plan_path": (
                    post_export_materializer_plan.get("plan_path")
                ),
                "post_export_experiment_queue_path": (
                    post_export_materializer_plan.get("experiment_queue_path")
                ),
                "post_export_execution_path": (
                    post_export_materializer_execution.get("execution_path")
                ),
                "authority": "planner_hook_false_authority_until_executed",
            },
            "post_export_materializer_plan": post_export_materializer_plan,
            "post_export_materializer_execution": post_export_materializer_execution,
            "snerv_binary_profile": snerv_binary_profile,
            "snerv_scorer_loop_qat": snerv_scorer_loop_qat,
            "snerv_mlx_native_export": snerv_mlx_native_export,
            "snerv_mlx_native_file_backed_export_evidence": (
                snerv_mlx_native_file_backed_evidence
            ),
            "local_cpu_replay_summary_paths": [
                path.as_posix() for path in local_cpu_replay_paths
            ],
            "local_cpu_replay_summary": local_cpu_replay_summary,
            "local_cpu_replay_gate": {
                "schema": "compact_runner_local_cpu_replay_gate.v1",
                "requested": run_local_cpu_replay,
                "default_enabled_for_full_coverage": (
                    _local_cpu_replay_enabled_by_default(
                        int(num_pairs),
                        mlx_prefilter_local_replay_passed=(
                            mlx_prefilter_local_replay_passed
                        ),
                    )
                ),
                "has_full_video_mlx_prefilter": has_full_video_mlx_prefilter,
                "local_replay_mlx_prefilter_passed": (
                    mlx_prefilter_local_replay_passed
                ),
                "coverage_valid_for_replay": int(num_pairs) >= CONTEST_PAIR_COUNT,
                "executed": local_cpu_replay_summary is not None,
                "axis_tag": "[macOS-CPU advisory]",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "mlx_prefilter_coverage": mlx_prefilter_coverage,
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "blockers": blockers,
        }
    )
    final["candidate_feedback"] = write_nerv_candidate_feedback_files(
        runner_report=final,
        output_dir=out,
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def adapt_snerv_advisory_report_to_spine(
    *,
    snerv_advisory_report_path: str | Path,
    output_dir: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    run_local_cpu_replay: bool | None = None,
    keep_local_replay_inflated: bool = False,
    cleanup_failed_local_replay_scratch: bool = True,
    run_post_export_materializers: bool = False,
    post_export_materializer_max_steps: int = 1,
    post_export_materializer_max_parallel: int = 0,
    post_export_materializer_max_experiments: int | None = 1,
    upstream_dir: str | Path = DEFAULT_UPSTREAM_DIR,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Ingest an existing SNeRV advisory/package without rerunning SNeRV."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    report_path = _resolve(snerv_advisory_report_path, base=root)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    advisory_payload = _load_json(report_path)
    package = advisory_payload.get("runtime_package")
    if isinstance(package, Mapping):
        package_payload = dict(package)
    elif isinstance(advisory_payload.get("archive_bound_candidate_adapter_package"), Mapping):
        package_payload = advisory_payload
    else:
        package_payload = {}
    candidate_rows = list(
        (
            package_payload.get("archive_bound_candidate_adapter_package", {})
            if isinstance(
                package_payload.get("archive_bound_candidate_adapter_package"),
                Mapping,
            )
            else {}
        ).get("candidate_rows", [])
        or []
    )
    row = dict(candidate_rows[0]) if candidate_rows else {}
    package_dir_raw = advisory_payload.get("runtime_package_dir")
    package_dir = (
        _resolve(str(package_dir_raw), base=root)
        if isinstance(package_dir_raw, str) and package_dir_raw.strip()
        else report_path.parent
    )
    archive_path = row.get("candidate_archive_path") or (package_dir / "archive.zip")
    receiver_proof = (
        dict(package_payload.get("receiver_proof"))
        if isinstance(package_payload.get("receiver_proof"), Mapping)
        else {}
    )
    receiver_proof_path = receiver_proof.get("proof_path")
    trained_ladder_row_payload = advisory_payload.get("trained_ladder_row_payload")
    trained_ladder_row_payload_path = advisory_payload.get(
        "trained_ladder_row_payload_path"
    )
    scorer_upstream = _resolve_scorer_upstream_dir(root, upstream_dir)
    mlx_prefilter_coverage = summarize_mlx_prefilter_coverage(
        mlx_profile_paths,
        root=root,
    )
    has_full_video_mlx_prefilter = bool(
        mlx_prefilter_coverage["has_full_video_mlx_prefilter"]
    )
    mlx_prefilter_local_replay_passed = bool(
        mlx_prefilter_coverage["local_replay_mlx_prefilter_passed"]
    )
    local_cpu_replay_summary: dict[str, Any] | None = None
    local_cpu_replay_paths: list[Path] = []
    local_cpu_replay_blockers: list[str] = []
    resolved_archive_path = _optional_existing(archive_path, base=root)
    runtime_submission_dir = package_dir / "submission"
    num_pairs_raw = advisory_payload.get("n_pairs")
    if not num_pairs_raw:
        runtime_manifest = row.get("runtime_adapter_manifest")
        if not isinstance(runtime_manifest, Mapping):
            contract = row.get("archive_bound_candidate_contract")
            runtime_manifest = (
                contract.get("runtime_adapter_manifest")
                if isinstance(contract, Mapping)
                else {}
            )
        if isinstance(runtime_manifest, Mapping):
            num_pairs_raw = runtime_manifest.get("n_pairs")
    num_pairs = int(num_pairs_raw or 0)
    if resolved_archive_path is not None:
        (
            local_cpu_replay_summary,
            local_cpu_replay_paths,
            local_cpu_replay_blockers,
        ) = _run_compact_local_cpu_replay_gate(
            archive_zip_path=resolved_archive_path,
            runtime_submission_dir=runtime_submission_dir,
            output_dir=out / "local_cpu_replay",
            upstream_dir=scorer_upstream,
            num_pairs=num_pairs,
            requested=run_local_cpu_replay,
            has_full_video_mlx_prefilter=has_full_video_mlx_prefilter,
            mlx_prefilter_local_replay_passed=mlx_prefilter_local_replay_passed,
            keep_inflated=keep_local_replay_inflated,
            cleanup_failed_scratch=cleanup_failed_local_replay_scratch,
            repo_root=root,
        )
    post_export_materializer_plan = _compile_carrier_post_export_materializer_plan(
        output_dir=out,
        archive_path=archive_path,
        archive_sha256=row.get("candidate_archive_sha256"),
        archive_bytes=row.get("candidate_archive_bytes"),
        family="snerv",
        runtime_submission_dir=runtime_submission_dir,
        repo_root=root,
    )
    post_export_materializer_execution = (
        _execute_carrier_post_export_materializer_plan(
            plan=post_export_materializer_plan,
            requested=run_post_export_materializers,
            max_steps=post_export_materializer_max_steps,
            max_parallel=post_export_materializer_max_parallel,
            max_experiments=post_export_materializer_max_experiments,
            repo_root=root,
        )
    )
    snerv_binary_profile = _write_snerv_binary_profile_attachment(
        archive_path=archive_path,
        output_dir=out,
        repo_root=root,
    )
    source_parity_contract = build_nerv_source_parity_contract(
        repo_root=root,
        families=("snerv",),
    )
    source_parity_blockers = [
        f"source_parity:{blocker}"
        for blocker in source_parity_contract.get("blockers") or ()
    ]
    source_parity_nonblocking_gaps = [
        f"source_parity:{gap}"
        for gap in source_parity_contract.get("nonblocking_gaps") or ()
    ]
    blockers = _dedupe(
        [
            "contest_cpu_cuda_exact_eval_not_executed",
            "snerv_mlx_native_adapter_surfaces_present_but_unproven",
            "snerv_longer_staged_score_aware_training_not_executed",
            *source_parity_blockers,
            *(
                []
                if receiver_proof.get("runtime_consumption_proof_passed") is True
                else ["snerv_runtime_package_receiver_proof_missing_or_failed"]
            ),
            *(
                []
                if num_pairs >= CONTEST_PAIR_COUNT
                else ["snerv_packet_not_full_600_pairs"]
            ),
            *list(row.get("blockers") or []),
            *local_cpu_replay_blockers,
            *list(post_export_materializer_plan.get("blockers") or []),
            *list(post_export_materializer_execution.get("blockers") or []),
            *(
                []
                if snerv_binary_profile.get("profile_written")
                else list(snerv_binary_profile.get("blockers") or [])
            ),
        ]
    )
    final = _base_report(
        output_dir=out,
        mode="adapted_snerv_advisory_report_to_spine",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "execute_family": "snerv",
            "source_snerv_advisory_report_path": report_path.as_posix(),
            "source_snerv_advisory_report_sha256": _sha256_file(report_path),
            "runtime_package_dir": package_dir.as_posix(),
            "archive_path": str(archive_path),
            "archive_bytes": row.get("candidate_archive_bytes"),
            "archive_sha256": row.get("candidate_archive_sha256"),
            "receiver_proof_report_paths": (
                [str(receiver_proof_path)] if receiver_proof_path else []
            ),
            "trained_ladder_row_payload_path": trained_ladder_row_payload_path,
            "trained_ladder_row_payload": (
                trained_ladder_row_payload
                if isinstance(trained_ladder_row_payload, Mapping)
                else None
            ),
            "post_export_materializer_plan": post_export_materializer_plan,
            "post_export_materializer_execution": post_export_materializer_execution,
            "snerv_binary_profile": snerv_binary_profile,
            "source_parity_contract": source_parity_contract,
            "source_parity_required_for_long_training_ready": bool(
                source_parity_contract.get("required_for_long_training_ready")
            ),
            "source_parity_blockers": source_parity_blockers,
            "source_parity_nonblocking_gaps": source_parity_nonblocking_gaps,
            "legacy_advisory_ingest_contract": {
                "schema": "snerv_legacy_advisory_ingest_contract.v1",
                "source_parity_consumed": True,
                "legacy_advisory_is_not_long_training_authority": True,
                "legacy_advisory_is_not_score_authority": True,
                "exact_cpu_cuda_required_for_promotion": True,
                **FALSE_AUTHORITY,
            },
            "local_cpu_replay_summary_paths": [
                path.as_posix() for path in local_cpu_replay_paths
            ],
            "local_cpu_replay_summary": local_cpu_replay_summary,
            "local_cpu_replay_gate": {
                "schema": "compact_runner_local_cpu_replay_gate.v1",
                "requested": run_local_cpu_replay,
                "default_enabled_for_full_coverage": (
                    _local_cpu_replay_enabled_by_default(
                        num_pairs,
                        mlx_prefilter_local_replay_passed=(
                            mlx_prefilter_local_replay_passed
                        ),
                    )
                ),
                "has_full_video_mlx_prefilter": has_full_video_mlx_prefilter,
                "local_replay_mlx_prefilter_passed": (
                    mlx_prefilter_local_replay_passed
                ),
                "coverage_valid_for_replay": num_pairs >= CONTEST_PAIR_COUNT,
                "executed": local_cpu_replay_summary is not None,
                "axis_tag": "[macOS-CPU advisory]",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "mlx_prefilter_coverage": mlx_prefilter_coverage,
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "blockers": blockers,
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def execute_pr95_mlx_smoke_and_adapt(
    *,
    output_dir: str | Path,
    max_frames: int,
    smoke_epochs_per_stage: int,
    training_loss_surface: str,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    latent_dim: int | None = None,
    base_channels: int | None = None,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run the existing MLX smoke path, then adapt its checkpoint output."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    pr95_report_path = out / "pr95_mlx_long_training_smoke_report.json"
    checkpoint_root = out / "pr95_mlx_checkpoints"
    telemetry_path = out / "pr95_mlx_telemetry.jsonl"
    command = [
        sys.executable,
        str(root / "tools/run_pr95_mlx_long_training.py"),
        "--output-report",
        pr95_report_path.as_posix(),
        "--checkpoint-root",
        checkpoint_root.as_posix(),
        "--telemetry-path",
        telemetry_path.as_posix(),
        "--source-video-path",
        Path(source_video_path).as_posix(),
        "--max-frames",
        str(int(max_frames)),
        "--smoke-mode",
        "--execute-smoke",
        "--smoke-epochs-per-stage",
        str(int(smoke_epochs_per_stage)),
        "--checkpoint-every-epochs",
        str(int(smoke_epochs_per_stage)),
        "--training-loss-surface",
        training_loss_surface,
        "--random-seed",
        str(int(random_seed)),
        "--operator-run-label",
        "compact_renderer_mlx_spine_runner",
    ]
    if latent_dim is not None:
        command.extend(["--latent-dim", str(int(latent_dim))])
    if base_channels is not None:
        command.extend(["--base-channels", str(int(base_channels))])
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        blocker_report = _base_report(
            output_dir=out,
            mode="pr95_mlx_smoke_failed",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        blocker_report.update(
            {
                "pr95_mlx_smoke_command": command,
                "pr95_mlx_smoke_returncode": completed.returncode,
                "blockers": ["pr95_mlx_smoke_command_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}
    adapted = adapt_pr95_mlx_report_to_spine(
        pr95_mlx_report_path=pr95_report_path,
        output_dir=out / "spine_from_pr95_mlx_smoke",
        hard_byte_ceilings=hard_byte_ceilings,
        mlx_profile_paths=mlx_profile_paths,
        hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
        allow_overwrite=allow_overwrite,
        repo_root=root,
    )
    final = _base_report(
        output_dir=out,
        mode="executed_pr95_mlx_smoke_and_adapted",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "pr95_mlx_smoke_command": command,
            "pr95_mlx_smoke_report_path": pr95_report_path.as_posix(),
            "pr95_mlx_smoke_report_sha256": _sha256_file(pr95_report_path),
            "adapted_report_path": adapted["report_path"],
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "adapted_blockers": adapted["blockers"],
            "blockers": adapted["blockers"],
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def _run_pr95_hnerv_mlx_scoreaware_smoke(
    *,
    output_dir: Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    source_archive_zip: str | Path,
    latent_dim: int,
    base_channels: int,
    ema_decay: float,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    pose_distillation_loss: str,
    pose_distillation_huber_delta: float,
    segnet_distillation_objective: str,
    distillation_temperature: float,
    segnet_tau_boundary: float,
    segnet_hinge_margin: float,
    distillation_device: str,
    requested_distillation_device: str | None,
    allow_segnet_only_research: bool,
    checkpoint_interval_epochs: int,
    checkpoint_dir: str | Path | None,
    resume_from_checkpoint: str | Path | None,
    random_seed: int,
    scorer_upstream_dir: Path,
    repo_root: Path,
) -> Any:
    import mlx.core as mx
    import numpy as np

    from tac.local_acceleration.pr95_hnerv_mlx import (
        HNeRVSyntheticTrainingBundleMLX,
        load_pytorch_state_dict_into_mlx,
        parse_pr95_public_archive_zip,
        pytorch_state_dict_from_mlx,
        write_pr95_public_archive_zip,
    )
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    pairs = int(num_pairs)
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    requested_distillation_device = str(
        requested_distillation_device or distillation_device
    )
    resolved_distillation_device = _resolve_torch_scorer_device_alias(
        str(distillation_device)
    )
    if pairs < 1:
        raise CompactRendererMlxSpineRunnerError("num_pairs must be >= 1")
    if segnet_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "segnet_distillation_weight must be >= 0"
        )
    if pose_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_weight must be >= 0"
        )
    if str(pose_distillation_loss) not in {"mse", "huber"}:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_loss must be one of ['mse', 'huber']"
        )
    if float(pose_distillation_huber_delta) <= 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_huber_delta must be > 0"
        )
    if (
        segnet_distillation_weight > 0.0
        and pose_distillation_weight <= 0.0
        and not allow_segnet_only_research
    ):
        raise CompactRendererMlxSpineRunnerError(
            "SegNet-bound PR95 training must also bind PoseNet. Pass "
            "--pose-distillation-weight > 0, or explicitly pass "
            "--allow-segnet-only-research for a false-authority SegNet-axis probe."
        )
    _require_scorer_upstream_dir_for_distillation(
        upstream_dir=scorer_upstream_dir,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
    )
    packet = parse_pr95_public_archive_zip(Path(source_archive_zip))
    if pairs > int(packet.latents.shape[0]):
        raise CompactRendererMlxSpineRunnerError(
            f"requested {pairs} pairs but source archive has "
            f"{int(packet.latents.shape[0])} latent rows"
        )
    if int(latent_dim) != int(packet.latents.shape[1]):
        raise CompactRendererMlxSpineRunnerError(
            f"latent_dim={latent_dim} does not match source archive "
            f"latent_dim={int(packet.latents.shape[1])}"
        )
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        source_video_path,
        num_pairs=pairs,
        output_height=384,
        output_width=512,
    )
    model = HNeRVSyntheticTrainingBundleMLX(
        latent_count=pairs,
        latent_dim=int(latent_dim),
        base_channels=int(base_channels),
        seed=int(random_seed),
        output_layout="n2chw",
    )
    load_pytorch_state_dict_into_mlx(model.decoder, packet.state_dict)
    model.latents = mx.array(np.asarray(packet.latents[:pairs], dtype=np.float32))

    def _export_archive(model_obj: Any, archive_output_dir: Path) -> tuple[Path, str, int]:
        export = write_pr95_public_archive_zip(
            pytorch_state_dict_from_mlx(model_obj.decoder),
            np.asarray(model_obj.latents).astype(np.float32, copy=False),
            meta={
                "n_pairs": pairs,
                "latent_dim": int(latent_dim),
                "base_channels": int(base_channels),
                "eval_size": [384, 512],
                "training_fidelity": (
                    "pr95_public_archive_seeded_scoreaware_mlx_advisory"
                ),
                "source_archive_sha256": packet.archive_zip_sha256,
            },
            output_zip_path=archive_output_dir / "pr95_public_archive.zip",
        )
        return (
            Path(export["archive_zip_path"]),
            str(export["archive_zip_sha256"]),
            int(export["archive_zip_bytes"]),
        )

    artifact_metadata = {
        "schema": "compact_pr95_hnerv_scoreaware_mlx_runner_metadata.v1",
        "family": "pr95_hnerv",
        "num_pairs": pairs,
        "full_video_pairs_required_for_promotion": 600,
        "source_archive_zip": str(Path(source_archive_zip)),
        "source_archive_sha256": packet.archive_zip_sha256,
        "archive_exporter": (
            "tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip"
        ),
        "score_aware_training": {
            "schema": "compact_pr95_hnerv_scoreaware_training.v1",
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "pose_distillation_loss": str(pose_distillation_loss),
            "pose_distillation_huber_delta": float(pose_distillation_huber_delta),
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": resolved_distillation_device,
            "requested_distillation_device": requested_distillation_device,
            "distillation_device_resolution": {
                "schema": "compact_runner_torch_scorer_device_resolution.v1",
                "requested": requested_distillation_device,
                "resolved": resolved_distillation_device,
                "scope": "real_pytorch_segnet_posenet_teacher_cache",
            },
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "checkpoint_interval_epochs": checkpoint_interval,
            "checkpoint_dir": (
                Path(checkpoint_dir).as_posix() if checkpoint_dir is not None else None
            ),
            "resume_from_checkpoint": (
                Path(resume_from_checkpoint).as_posix()
                if resume_from_checkpoint is not None
                else None
            ),
            "checkpoint_policy": "periodic_canonical_long_training_checkpoint",
            "scorer_upstream_snapshot": _scorer_upstream_metadata(
                scorer_upstream_dir
            ),
            "pr95_public_archive_seeded": True,
            "stage8_muon_continuation_optimizer_wired": False,
            "stage8_muon_continuation_blocker": (
                "shared_mlx_scoreaware_harness_lacks_stage8_start_epoch_offset"
            ),
        },
        "score_authority": "false_macos_mlx_research_signal",
    }
    bundle_kwargs: dict[str, Any] = {
        "model": model,
        "target_rgb_0": target_rgb_0,
        "target_rgb_1": target_rgb_1,
        "num_pairs": pairs,
        "forward_convention": "call_b2chw_255",
        "export_archive_fn": _export_archive,
        "substrate_artifact_metadata": artifact_metadata,
    }
    teacher_probe_bundle = RendererBundle(**bundle_kwargs)
    scorer_teacher = None
    learnable_student_head = None
    pose_scorer_teacher = None
    learnable_pose_student_head = None
    if segnet_distillation_weight > 0.0:
        scorer_teacher = build_mlx_segnet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=resolved_distillation_device,
        )
        learnable_student_head = build_learnable_student_head(
            num_classes=int(scorer_teacher.num_classes),
            seed=int(random_seed),
        )
    if pose_distillation_weight > 0.0:
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=resolved_distillation_device,
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=int(pose_scorer_teacher.pose_dims),
            seed=int(random_seed) + 1,
        )
    bundle = RendererBundle(
        **bundle_kwargs,
        distillation_weight=float(segnet_distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        distillation_temperature=float(distillation_temperature),
        segnet_distillation_objective=segnet_distillation_objective,
        segnet_tau_boundary=float(segnet_tau_boundary),
        segnet_hinge_margin=float(segnet_hinge_margin),
        distillation_num_classes=(
            int(scorer_teacher.num_classes) if scorer_teacher is not None else 5
        ),
        pose_distillation_weight=float(pose_distillation_weight),
        pose_distillation_loss=str(pose_distillation_loss),
        pose_distillation_huber_delta=float(pose_distillation_huber_delta),
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=int(pose_scorer_teacher.pose_dims)
        if pose_scorer_teacher is not None
        else 6,
        allow_segnet_only_research=bool(allow_segnet_only_research),
    )
    return run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="compact_runner_pr95_hnerv_mlx",
        lane_id="lane_compact_renderer_mlx_spine_runner_pr95_hnerv_20260601",
        output_dir=output_dir,
        epochs=int(epochs),
        batch_pair_indices_per_step=max(1, int(batch_pair_indices_per_step)),
        learning_rate=float(learning_rate),
        ema_decay=float(ema_decay),
        seed=int(random_seed),
        checkpoint_interval_epochs=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        telemetry_flush_interval_epochs=1,
        pr95_faithful_curriculum_enabled=False,
        notes=(
            "Compact PR95/HNeRV MLX spine runner seeded from the public PR95 "
            "archive, trained on real contest-video targets, exported as a "
            "PR95-compatible byte-closed archive, and false-authority MLX only."
        ),
    )


def execute_pr95_hnerv_mlx_scoreaware_and_adapt(
    *,
    output_dir: str | Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    source_archive_zip: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    latent_dim: int = 28,
    base_channels: int = 36,
    ema_decay: float = 0.9,
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    pose_distillation_loss: str = "mse",
    pose_distillation_huber_delta: float = 1.0,
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    requested_distillation_device: str | None = None,
    allow_segnet_only_research: bool = False,
    checkpoint_interval_epochs: int = DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS,
    checkpoint_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    scorer_upstream_dir: str | Path | None = None,
    run_receiver_proof: bool = False,
    receiver_proof_runtime_dir: str | Path = DEFAULT_PR95_RECEIVER_RUNTIME_DIR,
    keep_receiver_proof_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    run_post_export_materializers: bool = False,
    post_export_materializer_max_steps: int = 1,
    post_export_materializer_max_parallel: int = 0,
    post_export_materializer_max_experiments: int | None = 1,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a public-PR95-seeded HNeRV candidate through the spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    scorer_upstream = _resolve_scorer_upstream_dir(root, scorer_upstream_dir)
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    resolved_checkpoint_dir = _resolve_optional_compact_family_path(
        checkpoint_dir,
        base=root,
    )
    resolved_resume_from_checkpoint = _resolve_optional_compact_family_path(
        resume_from_checkpoint,
        base=root,
    )
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    try:
        artifact = _run_pr95_hnerv_mlx_scoreaware_smoke(
            output_dir=out / "pr95_hnerv_mlx_training",
            num_pairs=num_pairs,
            epochs=epochs,
            batch_pair_indices_per_step=batch_pair_indices_per_step,
            learning_rate=learning_rate,
            source_video_path=source_video_path,
            source_archive_zip=source_archive_zip,
            latent_dim=latent_dim,
            base_channels=base_channels,
            ema_decay=ema_decay,
            segnet_distillation_weight=segnet_distillation_weight,
            pose_distillation_weight=pose_distillation_weight,
            pose_distillation_loss=pose_distillation_loss,
            pose_distillation_huber_delta=pose_distillation_huber_delta,
            segnet_distillation_objective=segnet_distillation_objective,
            distillation_temperature=distillation_temperature,
            segnet_tau_boundary=segnet_tau_boundary,
            segnet_hinge_margin=segnet_hinge_margin,
            distillation_device=distillation_device,
            requested_distillation_device=requested_distillation_device,
            allow_segnet_only_research=allow_segnet_only_research,
            checkpoint_interval_epochs=checkpoint_interval_epochs,
            checkpoint_dir=resolved_checkpoint_dir,
            resume_from_checkpoint=resolved_resume_from_checkpoint,
            random_seed=random_seed,
            scorer_upstream_dir=scorer_upstream,
            repo_root=root,
        )
    except Exception as exc:
        blocker_report = _base_report(
            output_dir=out,
            mode="pr95_hnerv_mlx_scoreaware_failed",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        blocker_report.update(
            {
                "execute_family": "pr95_hnerv",
                "failure": repr(exc),
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "blockers": ["pr95_hnerv_mlx_scoreaware_or_export_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}

    artifact_dict = artifact.as_dict() if hasattr(artifact, "as_dict") else dict(artifact)
    archive_path = artifact_dict.get("archive_path")
    archive_file = _optional_existing(archive_path, base=root)
    spine_projection_error: str | None = None
    projection_paths: list[Path] = []
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    selected_runner_rows: list[dict[str, Any]] = []
    runner_plan_blockers: list[Any] = []
    receiver_proof_report: dict[str, Any] | None = None
    receiver_proof_paths: list[Path] = []
    if archive_file is not None:
        if run_receiver_proof:
            try:
                receiver_proof_report = run_pr95_hnerv_receiver_proof(
                    archive_zip=archive_file,
                    runtime_dir=receiver_proof_runtime_dir,
                    output_dir=out / "receiver_proof",
                    keep_output=keep_receiver_proof_output,
                    timeout_seconds=receiver_proof_timeout_seconds,
                    repo_root=root,
                )
                report_path = receiver_proof_report.get(
                    "report_path",
                    receiver_proof_report.get("proof_path"),
                )
                if isinstance(report_path, str) and report_path:
                    receiver_proof_paths = [Path(report_path)]
            except Exception as exc:
                receiver_proof_report = {
                    "schema": "pr95_hnerv_receiver_proof.v1",
                    "receiver_proof_valid": False,
                    "failure": repr(exc),
                    "blockers": ["pr95_receiver_proof_execution_failed"],
                    **FALSE_AUTHORITY,
                }
        try:
            spine = build_pr95_hnerv_spine_from_archive(
                archive_file,
                runtime_submission_dir=receiver_proof_runtime_dir,
            )
            projection = write_representation_spine_projection(
                output_dir=out / "pr95_hnerv_spine",
                spine=spine,
                basename="pr95_hnerv_representation_spine",
            )
            projection_manifest = Path(projection["manifest_path"])
            projection_paths = [projection_manifest]
            acquisition = build_spine_acquisition_report(
                projection_manifest_paths=projection_paths,
                hard_byte_ceilings=hard_byte_ceilings,
            )
            _write_json(acquisition_path, acquisition)
            runner_plan = build_spine_bounded_runner_plan(
                acquisition_report_path=acquisition_path,
                mlx_profile_paths=mlx_profile_paths,
                receiver_proof_report_paths=receiver_proof_paths,
                hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
                repo_root=root,
                upstream_dir=scorer_upstream,
            )
            write_spine_bounded_runner_plan(
                output_path=runner_plan_path,
                plan=runner_plan,
                allow_overwrite=True,
            )
            selected_runner_rows = list(runner_plan.get("selected_runner_rows") or [])
            runner_plan_blockers = list(runner_plan.get("blockers") or [])
        except Exception as exc:
            spine_projection_error = repr(exc)
    blockers: list[Any] = [
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not mlx_profile_paths:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    if receiver_proof_report is None:
        blockers.append("receiver_proof_not_executed")
    elif not receiver_proof_report.get("receiver_proof_valid"):
        blockers.append("receiver_proof_failed")
        blockers.extend(receiver_proof_report.get("blockers") or [])
    else:
        proof_refusal = receiver_proof_report.get("exact_readiness_refusal")
        if isinstance(proof_refusal, dict):
            blockers.extend(proof_refusal.get("blockers") or [])
    blockers.extend(
        _pr95_hnerv_control_arm_exact_blockers(
            artifact_dict=artifact_dict,
            segnet_distillation_weight=segnet_distillation_weight,
            pose_distillation_weight=pose_distillation_weight,
        )
    )
    if receiver_proof_report is None or not receiver_proof_report.get(
        "runtime_consumption_proof_passed"
    ):
        blockers.extend(
            [
                "runtime_consumption_proof_missing",
                "receiver_proof_missing",
            ]
        )
    if int(num_pairs) < 600:
        blockers.append("partial_pair_coverage_not_promotion_comparable")
    if archive_path is None:
        blockers.append("pr95_hnerv_archive_export_missing")
    if archive_file is None:
        blockers.append("pr95_hnerv_archive_export_missing_or_unreadable")
    if spine_projection_error is not None:
        blockers.append("pr95_hnerv_spine_projection_failed")
    blockers.extend(runner_plan_blockers)
    post_export_materializer_plan = _compile_carrier_post_export_materializer_plan(
        output_dir=out,
        archive_path=archive_path,
        archive_sha256=artifact_dict.get("archive_sha256"),
        archive_bytes=artifact_dict.get("archive_bytes"),
        family="pr95_hnerv",
        runtime_submission_dir=receiver_proof_runtime_dir,
        repo_root=root,
    )
    blockers.extend(post_export_materializer_plan.get("blockers") or [])
    post_export_materializer_execution = _execute_carrier_post_export_materializer_plan(
        plan=post_export_materializer_plan,
        requested=run_post_export_materializers,
        max_steps=post_export_materializer_max_steps,
        max_parallel=post_export_materializer_max_parallel,
        max_experiments=post_export_materializer_max_experiments,
        repo_root=root,
    )
    blockers.extend(post_export_materializer_execution.get("blockers") or [])

    final = _base_report(
        output_dir=out,
        mode="executed_pr95_hnerv_mlx_scoreaware_and_exported",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "execute_family": "pr95_hnerv",
            "num_pairs": int(num_pairs),
            "coverage_valid_for_base_comparison": int(num_pairs) >= 600,
            "training_artifact": artifact_dict,
            "archive_path": archive_path,
            "archive_bytes": artifact_dict.get("archive_bytes"),
            "archive_sha256": artifact_dict.get("archive_sha256"),
            "source_archive_zip": str(Path(source_archive_zip)),
            "projection_manifest_paths": [path.as_posix() for path in projection_paths],
            "receiver_proof_report_paths": [
                path.as_posix() for path in receiver_proof_paths
            ],
            "receiver_proof_report": receiver_proof_report,
            "spine_projection_error": spine_projection_error,
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "acquisition_report_path": (
                acquisition_path.as_posix() if acquisition_path.is_file() else None
            ),
            "bounded_runner_plan_path": (
                runner_plan_path.as_posix() if runner_plan_path.is_file() else None
            ),
            "selected_runner_rows": selected_runner_rows,
            "post_export_materializer_plan": post_export_materializer_plan,
            "post_export_materializer_execution": post_export_materializer_execution,
            "ema_decay": float(ema_decay),
            "score_aware_training": {
                "schema": "compact_pr95_hnerv_scoreaware_training.v1",
                "segnet_distillation_weight": float(segnet_distillation_weight),
                "pose_distillation_weight": float(pose_distillation_weight),
                "segnet_distillation_objective": segnet_distillation_objective,
                "distillation_temperature": float(distillation_temperature),
                "segnet_tau_boundary": float(segnet_tau_boundary),
                "segnet_hinge_margin": float(segnet_hinge_margin),
                "distillation_device": distillation_device,
                "allow_segnet_only_research": bool(allow_segnet_only_research),
                "checkpoint_interval_epochs": checkpoint_interval,
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "stage8_muon_continuation_optimizer_wired": False,
                "stage8_muon_continuation_blocker": (
                    "shared_mlx_scoreaware_harness_lacks_stage8_start_epoch_offset"
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "control_arm_scope": {
                "schema": "pr95_hnerv_mlx_control_arm_scope.v1",
                "archive_export_executable": archive_file is not None,
                "source_faithful_pr95_reproduction": False,
                "score_authority": False,
                "runtime_consumption_proven": (
                    receiver_proof_report is not None
                    and receiver_proof_report.get("runtime_consumption_proof_passed")
                    is True
                ),
                "full_frame_inflate_parity_proven": False,
                "exact_cpu_cuda_authority": False,
            },
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "blockers": _dedupe(blockers),
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def run_pr95_hnerv_receiver_proof(
    *,
    archive_zip: str | Path,
    runtime_dir: str | Path,
    output_dir: str | Path,
    keep_output: bool,
    timeout_seconds: int,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Prove the PR95 runtime consumes the candidate archive bytes.

    The raw RGB output can be several GB for full 600-pair archives, so the
    default is certify-and-delete: hash the deterministic output, record the
    archive/runtime custody that rebuilds it, then remove the rebuildable raw
    file unless the operator explicitly asks to retain it.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    archive_path = _resolve_existing(archive_zip, base=root)
    runtime = _resolve(runtime_dir, base=root)
    inflate_sh = runtime / "inflate.sh"
    if not inflate_sh.is_file():
        raise CompactRendererMlxSpineRunnerError(
            f"PR95 receiver runtime missing inflate.sh: {inflate_sh}"
        )
    out = Path(output_dir).expanduser().resolve(strict=False)
    data_dir = out / "data"
    raw_dir = out / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = out / "pr95_hnerv_receiver_proof.json"

    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if "0.bin" not in names:
            raise CompactRendererMlxSpineRunnerError(
                f"PR95 receiver proof expects 0.bin; archive members={names!r}"
            )
        member_bytes = zf.read("0.bin")
    member_path = data_dir / "0.bin"
    member_path.write_bytes(member_bytes)
    file_list_path = out / "file_list.txt"
    file_list_path.write_text("0.mkv\n", encoding="utf-8")

    packet = None
    expected_raw_bytes: int | None = None
    try:
        from tac.local_acceleration.pr95_hnerv_mlx import parse_pr95_public_archive_zip

        packet = parse_pr95_public_archive_zip(archive_path)
        expected_raw_bytes = int(packet.meta["n_pairs"]) * 2 * 874 * 1164 * 3
    except Exception:
        packet = None

    runtime_files = [
        runtime / "inflate.sh",
        runtime / "inflate.py",
        runtime / "src/model.py",
        runtime / "src/codec.py",
    ]
    command = [
        "bash",
        inflate_sh.as_posix(),
        data_dir.as_posix(),
        raw_dir.as_posix(),
        file_list_path.as_posix(),
    ]
    env = dict(os.environ)
    venv_bin = root / ".venv/bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin.as_posix()}:{env.get('PATH', '')}"
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    raw_path = raw_dir / "0.raw"
    raw_exists = raw_path.is_file()
    raw_bytes = raw_path.stat().st_size if raw_exists else 0
    raw_sha256 = _sha256_file(raw_path) if raw_exists else None
    receiver_proof_valid = (
        completed.returncode == 0
        and raw_exists
        and raw_bytes > 0
        and (expected_raw_bytes is None or raw_bytes == expected_raw_bytes)
    )
    cleanup: dict[str, Any] = {
        "schema": "receiver_proof_output_cleanup.v1",
        "keep_output_requested": bool(keep_output),
        "raw_output_path": raw_path.as_posix(),
        "staged_member_path": member_path.as_posix(),
        "file_list_path": file_list_path.as_posix(),
        "raw_output_rebuildable_from_archive_and_runtime": receiver_proof_valid,
        "raw_output_retained": bool(raw_exists),
        "deleted_rebuildable_raw_output": False,
        "deleted_rebuildable_work_files": False,
    }
    if receiver_proof_valid and raw_exists and not keep_output:
        raw_path.unlink()
        for scratch in (member_path, file_list_path):
            if scratch.exists():
                scratch.unlink()
        for scratch_dir in (raw_dir, data_dir):
            if scratch_dir.exists():
                try:
                    scratch_dir.rmdir()
                except OSError:
                    pass
        cleanup["raw_output_retained"] = False
        cleanup["deleted_rebuildable_raw_output"] = True
        cleanup["deleted_rebuildable_work_files"] = True

    blockers: list[str] = []
    if completed.returncode != 0:
        blockers.append("pr95_receiver_inflate_sh_failed")
    if not raw_exists:
        blockers.append("pr95_receiver_raw_output_missing")
    if expected_raw_bytes is not None and raw_bytes != expected_raw_bytes:
        blockers.append("pr95_receiver_raw_output_byte_count_mismatch")

    report = {
        "schema": "pr95_hnerv_receiver_proof.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "proof_path": report_path.as_posix(),
        "archive_path": archive_path.as_posix(),
        "archive_zip_path": archive_path.as_posix(),
        "archive_zip_bytes": archive_path.stat().st_size,
        "archive_zip_sha256": _sha256_file(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "archive_member": {
            "name": "0.bin",
            "bytes": len(member_bytes),
            "sha256": hashlib.sha256(member_bytes).hexdigest(),
        },
        "runtime_dir": runtime.as_posix(),
        "runtime_files": [_file_record(path) for path in runtime_files],
        "command": command,
        "cwd": root.as_posix(),
        "env_overrides": {
            "PATH_prefix": venv_bin.as_posix() if venv_bin.is_dir() else None,
        },
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_raw": {
            "path": raw_path.as_posix(),
            "bytes": raw_bytes,
            "sha256": raw_sha256,
            "expected_bytes": expected_raw_bytes,
            "retained": cleanup["raw_output_retained"],
        },
        "parsed_archive_meta": None if packet is None else dict(packet.meta),
        "receiver_proof_valid": receiver_proof_valid,
        "runtime_consumption_proof_passed": receiver_proof_valid,
        "receiver_contract_satisfied": receiver_proof_valid,
        "receiver_output_kind": "contest_raw_rgb_interleaved",
        "receiver_output_bytes": raw_bytes,
        "full_frame_inflate_parity": False,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "runtime_consumption_smoke_is_not_score_authority",
                "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        "blockers": _dedupe(blockers),
        "cleanup": cleanup,
        **FALSE_AUTHORITY,
    }
    _write_json(report_path, report)
    return {**report, "report_path": report_path.as_posix()}


def execute_pact_nerv_vq_mlx_smoke_and_adapt(
    *,
    output_dir: str | Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    latent_dim: int = 8,
    embed_dim: int = 8,
    codebook_size: int = 16,
    decoder_channel: int = 8,
    decoder_codec: str = "int8_mixed",
    ema_decay: float = 0.9,
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    pose_distillation_loss: str = "mse",
    pose_distillation_huber_delta: float = 1.0,
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    requested_distillation_device: str | None = None,
    allow_segnet_only_research: bool = False,
    scorer_upstream_dir: str | Path | None = None,
    coder_aware_qat: bool = False,
    coder_qat_quant_bits: int = 8,
    coder_qat_quant_residual_weight: float = (
        DEFAULT_PACT_CODER_QAT_QUANT_RESIDUAL_WEIGHT
    ),
    coder_qat_magnitude_weight: float = DEFAULT_PACT_CODER_QAT_MAGNITUDE_WEIGHT,
    coder_qat_delta_weight: float = DEFAULT_PACT_CODER_QAT_DELTA_WEIGHT,
    coder_qat_c1a_entropy_weight: float = DEFAULT_PACT_CODER_QAT_C1A_ENTROPY_WEIGHT,
    coder_qat_c1a_sigma: float = DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
    coder_qat_c1a_sample_size: int = DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
    optimizer_kind: str = DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
    optimizer_grad_clip_max_norm: float | None = (
        DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_GRAD_CLIP_MAX_NORM
    ),
    optimizer_weight_decay: float | None = None,
    optimizer_warmup_epochs: int = 0,
    optimizer_warmup_steps_per_epoch: int = 1,
    optimizer_cosine_decay_enabled: bool = False,
    optimizer_cosine_decay_total_epochs: int | None = None,
    optimizer_cosine_decay_min_lr_ratio: float = 1e-2,
    checkpoint_interval_epochs: int = DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS,
    checkpoint_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    run_post_export_materializers: bool = False,
    post_export_materializer_max_steps: int = 1,
    post_export_materializer_max_parallel: int = 0,
    post_export_materializer_max_experiments: int | None = 1,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a tiny real-video PACT-NeRV-VQ candidate through the spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    resolved_checkpoint_dir = _resolve_optional_compact_family_path(
        checkpoint_dir,
        base=root,
    )
    resolved_resume_from_checkpoint = _resolve_optional_compact_family_path(
        resume_from_checkpoint,
        base=root,
    )
    scorer_upstream = _resolve_scorer_upstream_dir(root, scorer_upstream_dir)
    optimizer_controls = _resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind=optimizer_kind,
        requested_weight_decay=optimizer_weight_decay,
        grad_clip_max_norm=optimizer_grad_clip_max_norm,
        warmup_epochs=optimizer_warmup_epochs,
        warmup_steps_per_epoch=optimizer_warmup_steps_per_epoch,
        cosine_decay_enabled=optimizer_cosine_decay_enabled,
        cosine_decay_total_epochs=optimizer_cosine_decay_total_epochs,
        cosine_decay_min_lr_ratio=optimizer_cosine_decay_min_lr_ratio,
        run_epochs=epochs,
    )
    optimizer_policy = _resolve_pact_compact_optimizer_policy(
        family="pact_nerv_vq",
        optimizer_controls=optimizer_controls,
    )
    _, coder_qat_metadata_row = _build_pact_coder_qat_config_and_metadata(
        coder_aware_qat=coder_aware_qat,
        coder_qat_quant_bits=coder_qat_quant_bits,
        coder_qat_quant_residual_weight=coder_qat_quant_residual_weight,
        coder_qat_magnitude_weight=coder_qat_magnitude_weight,
        coder_qat_delta_weight=coder_qat_delta_weight,
        coder_qat_c1a_entropy_weight=coder_qat_c1a_entropy_weight,
        coder_qat_c1a_sigma=coder_qat_c1a_sigma,
        coder_qat_c1a_sample_size=coder_qat_c1a_sample_size,
    )
    out = Path(output_dir).expanduser().resolve(strict=False)
    resolved_source_video = _resolve_source_video_path(source_video_path, base=root)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    try:
        artifact = _run_pact_nerv_vq_mlx_smoke(
            output_dir=out / "pact_nerv_vq_mlx_training",
            num_pairs=num_pairs,
            epochs=epochs,
            batch_pair_indices_per_step=batch_pair_indices_per_step,
            learning_rate=learning_rate,
            source_video_path=resolved_source_video,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            codebook_size=codebook_size,
            decoder_channel=decoder_channel,
            decoder_codec=decoder_codec,
            ema_decay=ema_decay,
            segnet_distillation_weight=segnet_distillation_weight,
            pose_distillation_weight=pose_distillation_weight,
            pose_distillation_loss=pose_distillation_loss,
            pose_distillation_huber_delta=pose_distillation_huber_delta,
            segnet_distillation_objective=segnet_distillation_objective,
            distillation_temperature=distillation_temperature,
            segnet_tau_boundary=segnet_tau_boundary,
            segnet_hinge_margin=segnet_hinge_margin,
            distillation_device=distillation_device,
            allow_segnet_only_research=allow_segnet_only_research,
            coder_aware_qat=coder_aware_qat,
            coder_qat_quant_bits=coder_qat_quant_bits,
            coder_qat_quant_residual_weight=coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=coder_qat_magnitude_weight,
            coder_qat_delta_weight=coder_qat_delta_weight,
            coder_qat_c1a_entropy_weight=coder_qat_c1a_entropy_weight,
            coder_qat_c1a_sigma=coder_qat_c1a_sigma,
            coder_qat_c1a_sample_size=coder_qat_c1a_sample_size,
            optimizer_kind=optimizer_kind,
            optimizer_policy=optimizer_policy,
            optimizer_controls=optimizer_controls,
            checkpoint_interval_epochs=checkpoint_interval,
            checkpoint_dir=resolved_checkpoint_dir,
            resume_from_checkpoint=resolved_resume_from_checkpoint,
            random_seed=random_seed,
            scorer_upstream_dir=scorer_upstream,
            repo_root=root,
        )
    except Exception as exc:
        blocker_report = _base_report(
            output_dir=out,
            mode="pact_nerv_vq_mlx_smoke_failed",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        blocker_report.update(
            {
                "execute_family": "pact_nerv_vq",
                "failure": repr(exc),
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "blockers": ["pact_nerv_vq_mlx_smoke_or_export_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}

    artifact_dict = artifact.as_dict() if hasattr(artifact, "as_dict") else dict(artifact)
    archive_path = artifact_dict.get("archive_path")
    training_dir = out / "pact_nerv_vq_mlx_training"
    spine_manifest = training_dir / "hprc_representation_spine_pact_nerv_vq_manifest.json"
    receiver_proof_path = (
        training_dir / "receiver_proof" / "pact_nerv_vq_mlx_receiver_proof.json"
    )
    projection_paths = [spine_manifest] if spine_manifest.is_file() else []
    receiver_proof_paths = [receiver_proof_path] if receiver_proof_path.is_file() else []
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    selected_runner_rows: list[dict[str, Any]] = []
    blockers: list[Any] = [
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not mlx_profile_paths:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    if projection_paths:
        acquisition = build_spine_acquisition_report(
            projection_manifest_paths=projection_paths,
            hard_byte_ceilings=hard_byte_ceilings,
        )
        _write_json(acquisition_path, acquisition)
        runner_plan = build_spine_bounded_runner_plan(
            acquisition_report_path=acquisition_path,
            mlx_profile_paths=mlx_profile_paths,
            receiver_proof_report_paths=receiver_proof_paths,
            hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
            repo_root=root,
            upstream_dir=scorer_upstream,
        )
        write_spine_bounded_runner_plan(
            output_path=runner_plan_path,
            plan=runner_plan,
            allow_overwrite=True,
        )
        selected_runner_rows = list(runner_plan.get("selected_runner_rows") or [])
        blockers.extend(runner_plan.get("blockers") or [])
    else:
        blockers.append("pact_nerv_vq_spine_projection_manifest_missing")
    if archive_path is None:
        blockers.append("pact_nerv_vq_archive_export_missing")
    post_export_materializer_plan = _compile_carrier_post_export_materializer_plan(
        output_dir=out,
        archive_path=archive_path,
        archive_sha256=artifact_dict.get("archive_sha256"),
        archive_bytes=artifact_dict.get("archive_bytes"),
        family="pact_nerv_vq",
        runtime_submission_dir=training_dir / "submission",
        repo_root=root,
    )
    blockers.extend(post_export_materializer_plan.get("blockers") or [])
    post_export_materializer_execution = _execute_carrier_post_export_materializer_plan(
        plan=post_export_materializer_plan,
        requested=run_post_export_materializers,
        max_steps=post_export_materializer_max_steps,
        max_parallel=post_export_materializer_max_parallel,
        max_experiments=post_export_materializer_max_experiments,
        repo_root=root,
    )
    blockers.extend(post_export_materializer_execution.get("blockers") or [])

    final = _base_report(
        output_dir=out,
        mode="executed_pact_nerv_vq_mlx_smoke_and_exported",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "execute_family": "pact_nerv_vq",
            "num_pairs": int(num_pairs),
            "coverage_valid_for_base_comparison": int(num_pairs) >= 600,
            "training_artifact": artifact_dict,
            "archive_path": archive_path,
            "archive_bytes": artifact_dict.get("archive_bytes"),
            "archive_sha256": artifact_dict.get("archive_sha256"),
            "ema_decay": float(ema_decay),
            "score_aware_training": {
                "schema": "compact_pact_nerv_vq_score_aware_training.v1",
                "segnet_distillation_weight": float(segnet_distillation_weight),
                "pose_distillation_weight": float(pose_distillation_weight),
                "pose_distillation_loss": str(pose_distillation_loss),
                "pose_distillation_huber_delta": float(
                    pose_distillation_huber_delta
                ),
                "segnet_distillation_objective": segnet_distillation_objective,
                "distillation_temperature": float(distillation_temperature),
                "segnet_tau_boundary": float(segnet_tau_boundary),
                "segnet_hinge_margin": float(segnet_hinge_margin),
                "distillation_device": distillation_device,
                "allow_segnet_only_research": bool(allow_segnet_only_research),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "optimizer_policy": strip_candidate_curriculum_authority_fields(
                    optimizer_policy
                ),
                "pr95_faithful_curriculum_enabled": bool(
                    optimizer_policy.get("pr95_faithful_curriculum_enabled")
                ),
                "native_optimizer_active": bool(
                    optimizer_policy.get("native_optimizer_active")
                ),
                "optimizer_kind": str(
                    optimizer_policy.get("optimizer_kind") or optimizer_kind
                ),
                "optimizer_controls": strip_candidate_curriculum_authority_fields(
                    optimizer_controls
                ),
                "effective_weight_decay": optimizer_controls.get(
                    "weight_decay_effective"
                ),
                "checkpoint_interval_epochs": checkpoint_interval,
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "coder_aware_qat": coder_qat_metadata_row,
                "decoder_codec": str(decoder_codec),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "projection_manifest_paths": [path.as_posix() for path in projection_paths],
            "receiver_proof_report_paths": [
                path.as_posix() for path in receiver_proof_paths
            ],
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "acquisition_report_path": (
                acquisition_path.as_posix() if acquisition_path.is_file() else None
            ),
            "bounded_runner_plan_path": (
                runner_plan_path.as_posix() if runner_plan_path.is_file() else None
            ),
            "selected_runner_rows": selected_runner_rows,
            "post_export_materializer_plan": post_export_materializer_plan,
            "post_export_materializer_execution": post_export_materializer_execution,
            "blockers": _dedupe(blockers),
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def execute_hi_nerv_mlx_scoreaware_and_adapt(
    *,
    output_dir: str | Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    latent_dim: int = 8,
    embed_dim: int = 8,
    decoder_channel: int = 8,
    decoder_codec: str = "portfolio_auto",
    hi_nerv_latent_codec: str = "int16_raw",
    modelsize_candidate: Mapping[str, Any] | None = None,
    allow_unscored_research_smoke: bool = False,
    modelsize_budget_json_paths: tuple[str | Path, ...] = (),
    receiver_closed_ladder_json_paths: tuple[str | Path, ...] = (),
    ema_decay: float = 0.9,
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    pose_distillation_loss: str = "mse",
    pose_distillation_huber_delta: float = 1.0,
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    requested_distillation_device: str | None = None,
    allow_segnet_only_research: bool = False,
    coder_aware_qat: bool = False,
    coder_qat_quant_bits: int = 8,
    coder_qat_quant_residual_weight: float = (
        DEFAULT_PACT_CODER_QAT_QUANT_RESIDUAL_WEIGHT
    ),
    coder_qat_magnitude_weight: float = DEFAULT_PACT_CODER_QAT_MAGNITUDE_WEIGHT,
    coder_qat_delta_weight: float = DEFAULT_PACT_CODER_QAT_DELTA_WEIGHT,
    coder_qat_c1a_entropy_weight: float = DEFAULT_PACT_CODER_QAT_C1A_ENTROPY_WEIGHT,
    coder_qat_c1a_sigma: float = DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
    coder_qat_c1a_sample_size: int = DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
    decoder_weight_waterfill_plan_json: str | Path | None = None,
    recon_pixel_weight_path: str | Path | None = None,
    auto_joint_recon_pixel_weight: bool = False,
    auto_segnet_boundary_recon_weight: bool = False,
    recon_pixel_weight_tau: float = 1.0,
    recon_pixel_weight_normalize: str = "mean",
    mlx_prefilter_scorer_device: str | None = None,
    mlx_prefilter_scorer_batch_pairs: int = 1,
    mlx_prefilter_progress_every: int = 50,
    telemetry_flush_interval_epochs: int = 1,
    checkpoint_interval_epochs: int = DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS,
    checkpoint_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    optimizer_kind: str = DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
    hi_nerv_optimizer_policy: str = "auto",
    optimizer_grad_clip_max_norm: float | None = (
        DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_GRAD_CLIP_MAX_NORM
    ),
    optimizer_weight_decay: float | None = None,
    optimizer_warmup_epochs: int = 0,
    optimizer_warmup_steps_per_epoch: int = 1,
    optimizer_cosine_decay_enabled: bool = False,
    optimizer_cosine_decay_total_epochs: int | None = None,
    optimizer_cosine_decay_min_lr_ratio: float = 1e-2,
    prioritized_pair_indices: tuple[int, ...] = (),
    random_seed: int = 0,
    run_local_cpu_replay: bool | None = None,
    keep_local_replay_inflated: bool = False,
    cleanup_failed_local_replay_scratch: bool = True,
    run_post_export_materializers: bool = False,
    post_export_materializer_max_steps: int = 1,
    post_export_materializer_max_parallel: int = 0,
    post_export_materializer_max_experiments: int | None = 1,
    upstream_dir: str | Path = DEFAULT_UPSTREAM_DIR,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a HiNeRV MLX candidate through the real receiver bundle."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    prioritized_pair_indices = _normalize_nonnegative_int_sequence(
        prioritized_pair_indices
    )
    try:
        prioritized_pair_indices = validate_pair_indices_in_range(
            prioritized_pair_indices,
            num_pairs=int(num_pairs),
            field="prioritized_pair_indices",
        )
    except HardPairIndicesError as exc:
        raise CompactRendererMlxSpineRunnerError(str(exc)) from exc
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    resolved_checkpoint_dir = _resolve_optional_compact_family_path(
        checkpoint_dir,
        base=root,
    )
    resolved_resume_from_checkpoint = _resolve_optional_compact_family_path(
        resume_from_checkpoint,
        base=root,
    )
    out = Path(output_dir).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, upstream_dir)
    optimizer_policy = _resolve_hi_nerv_optimizer_policy(
        requested_policy=hi_nerv_optimizer_policy,
        epochs=int(epochs),
        optimizer_kind=str(optimizer_kind),
    )
    optimizer_controls = _resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind=str(optimizer_policy.get("optimizer_kind") or optimizer_kind),
        requested_weight_decay=optimizer_weight_decay,
        grad_clip_max_norm=optimizer_grad_clip_max_norm,
        warmup_epochs=int(optimizer_warmup_epochs),
        warmup_steps_per_epoch=int(optimizer_warmup_steps_per_epoch),
        cosine_decay_enabled=bool(optimizer_cosine_decay_enabled),
        cosine_decay_total_epochs=optimizer_cosine_decay_total_epochs,
        cosine_decay_min_lr_ratio=float(optimizer_cosine_decay_min_lr_ratio),
        run_epochs=int(epochs),
    )
    resolved_source_video = _resolve_source_video_path(source_video_path, base=root)
    effective_requested_distillation_device = str(
        requested_distillation_device or distillation_device
    )
    if (
        _has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        and not allow_overwrite
    ):
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    decoder_weight_waterfill_plan: dict[str, Any] | None = None
    decoder_weight_waterfill_plan_metadata: dict[str, Any] = {
        "schema": "compact_hi_nerv_decoder_weight_waterfill_plan_attachment.v1",
        "attached": False,
        "active": False,
        "validated": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if decoder_weight_waterfill_plan_json is not None:
        plan_path = Path(decoder_weight_waterfill_plan_json).expanduser()
        if not plan_path.is_absolute():
            plan_path = root / plan_path
        plan_path = plan_path.resolve(strict=False)
        decoder_weight_waterfill_plan_metadata.update(
            {
                "path": plan_path.as_posix(),
                "blockers": [],
            }
        )
        if not plan_path.is_file():
            decoder_weight_waterfill_plan_metadata["blockers"] = [
                "decoder_weight_waterfill_plan_json_missing"
            ]
        else:
            decoder_weight_waterfill_plan = _load_json(plan_path)
            decoder_weight_waterfill_plan_metadata.update(
                {
                    "sha256": _sha256_file(plan_path),
                    "source_schema": decoder_weight_waterfill_plan.get("schema"),
                    "family": decoder_weight_waterfill_plan.get("family"),
                    "candidate_id": decoder_weight_waterfill_plan.get("candidate_id"),
                    "group_count": decoder_weight_waterfill_plan.get("group_count"),
                    "row_count": len(
                        decoder_weight_waterfill_plan.get("rows") or []
                    ),
                    "source_blockers": list(
                        decoder_weight_waterfill_plan.get("blockers") or []
                    ),
                }
            )
            if (
                decoder_weight_waterfill_plan.get("schema")
                != NERV_DECODER_WEIGHT_WATERFILL_SCHEMA
            ):
                decoder_weight_waterfill_plan_metadata["blockers"] = [
                    "decoder_weight_waterfill_plan_schema_mismatch"
                ]
    modelsize_budget_rows, modelsize_budget_sources = (
        _load_compact_modelsize_budget_rows(
            (*modelsize_budget_json_paths, *receiver_closed_ladder_json_paths),
            base=root,
        )
    )
    score_aware_evidence = (
        {"modelsize_budget_rows": modelsize_budget_rows}
        if modelsize_budget_rows
        else {}
    )
    score_aware_training_plan = _score_aware_carrier_training_plan(
        "hi_nerv",
        _backend_with_score_aware_evidence("hi_nerv", score_aware_evidence),
    )
    modelsize_budget_evidence = {
        "schema": "compact_hi_nerv_modelsize_budget_evidence.v1",
        "source_count": len(modelsize_budget_sources),
        "row_count": len(modelsize_budget_rows),
        "sources": modelsize_budget_sources,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    candidate = dict(modelsize_candidate or {})
    launch_pressure_binding = _bind_hi_nerv_modelsize_launch_pressure(
        modelsize_candidate=candidate or None,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
        allow_unscored_research_smoke=allow_unscored_research_smoke,
    )
    effective_segnet_distillation_weight = float(
        launch_pressure_binding["segnet_distillation_weight"]
    )
    effective_pose_distillation_weight = float(
        launch_pressure_binding["pose_distillation_weight"]
    )
    candidate_supplied = bool(candidate)
    launch_latent_dim = int(candidate.get("latent_dim", latent_dim))
    launch_embed_dim = int(candidate.get("embed_dim", embed_dim))
    launch_decoder_channel = int(candidate.get("decoder_channel", decoder_channel))
    launch_decoder_codec = str(candidate.get("decoder_codec", decoder_codec))
    launch_use_hierarchical_feature_grid = bool(
        candidate.get("use_hierarchical_feature_grid", not candidate_supplied)
    )
    launch_use_convnext_blocks = bool(
        candidate.get("use_convnext_blocks", not candidate_supplied)
    )
    launch_local_grid_levels = int(candidate.get("local_grid_levels", 2))
    launch_local_grid_channels = int(candidate.get("local_grid_channels", 4))
    launch_convnext_mlp_ratio = int(candidate.get("convnext_mlp_ratio", 2))
    launch_convnext_kernel_size = int(candidate.get("convnext_kernel_size", 7))
    launch_mid_injection_block_index = int(
        candidate.get(
            "mid_injection_block_index",
            HINERV_COMPACT_MID_INJECTION_BLOCK_INDEX,
        )
    )
    launch_fine_injection_block_index = int(
        candidate.get(
            "fine_injection_block_index",
            HINERV_COMPACT_FINE_INJECTION_BLOCK_INDEX,
        )
    )
    launch_source_faithfulness = _hi_nerv_launch_source_faithfulness_report(
        use_hierarchical_feature_grid=launch_use_hierarchical_feature_grid,
        use_convnext_blocks=launch_use_convnext_blocks,
        local_grid_levels=launch_local_grid_levels,
        local_grid_channels=launch_local_grid_channels,
        convnext_mlp_ratio=launch_convnext_mlp_ratio,
        convnext_kernel_size=launch_convnext_kernel_size,
        decoder_codec=launch_decoder_codec,
        hi_nerv_latent_codec=str(
            candidate.get("hi_nerv_latent_codec", hi_nerv_latent_codec)
        ),
    )
    (
        decoder_weight_waterfill_plan,
        decoder_weight_waterfill_plan_metadata,
    ) = _validate_hi_nerv_decoder_weight_waterfill_plan_attachment(
        plan=decoder_weight_waterfill_plan,
        metadata=decoder_weight_waterfill_plan_metadata,
        candidate=candidate or None,
        num_pairs=int(num_pairs),
        latent_dim=launch_latent_dim,
        embed_dim=launch_embed_dim,
        decoder_channel=launch_decoder_channel,
        use_hierarchical_feature_grid=launch_use_hierarchical_feature_grid,
        use_convnext_blocks=launch_use_convnext_blocks,
        local_grid_levels=launch_local_grid_levels,
        local_grid_channels=launch_local_grid_channels,
        convnext_mlp_ratio=launch_convnext_mlp_ratio,
        convnext_kernel_size=launch_convnext_kernel_size,
        mid_injection_block_index=launch_mid_injection_block_index,
        fine_injection_block_index=launch_fine_injection_block_index,
    )
    launch_control_precedence = _hi_nerv_launch_control_precedence_report(
        modelsize_candidate=candidate or None,
        source_faithfulness=launch_source_faithfulness,
        modelsize_launch_pressure=launch_pressure_binding,
        decoder_weight_waterfill_plan_metadata=decoder_weight_waterfill_plan_metadata,
        optimizer_policy=optimizer_policy,
    )
    waterfill_validation_blockers = list(
        decoder_weight_waterfill_plan_metadata.get("blockers") or []
    )
    if decoder_weight_waterfill_plan_json is not None and waterfill_validation_blockers:
        refusal = _base_report(
            output_dir=out,
            mode="hi_nerv_decoder_weight_waterfill_plan_launch_refused",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        refusal.update(
            {
                "execute_family": "hi_nerv",
                "num_pairs": int(num_pairs),
                "epochs_requested": int(epochs),
                "training_executed": False,
                "trainer_launch_allowed": False,
                "launch_refusal_reason": (
                    "HiNeRV decoder-weight waterfill plans must match the "
                    "resolved modelsize candidate and launch decoder state "
                    "before they can be attached to training."
                ),
                "modelsize_candidate_selection": {
                    "schema": "compact_execute_modelsize_candidate_selection.v1",
                    "family": "hi_nerv",
                    "selection_mode": (
                        "planner_candidate" if candidate else "manual_cli_knobs"
                    ),
                    "candidate": candidate or None,
                    "modelsize_control_contract": _modelsize_control_contract(
                        candidate
                    ),
                    "num_pairs_for_budget": CONTEST_PAIR_COUNT,
                    "launch_latent_dim": launch_latent_dim,
                    "launch_embed_dim": launch_embed_dim,
                    "launch_decoder_channel": launch_decoder_channel,
                    "launch_decoder_codec": launch_decoder_codec,
                    "launch_use_hierarchical_feature_grid": (
                        launch_use_hierarchical_feature_grid
                    ),
                    "launch_use_convnext_blocks": launch_use_convnext_blocks,
                    "launch_mid_injection_block_index": (
                        launch_mid_injection_block_index
                    ),
                    "launch_fine_injection_block_index": (
                        launch_fine_injection_block_index
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_aware_training": {
                    "schema": "compact_hi_nerv_score_aware_training.v1",
                    "status": "refused_before_mlx_training",
                    "decoder_weight_waterfill_plan": (
                        decoder_weight_waterfill_plan_metadata
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "hi_nerv_modelsize_launch_pressure": launch_pressure_binding,
                "hi_nerv_source_faithfulness": launch_source_faithfulness,
                "hi_nerv_control_precedence": launch_control_precedence,
                "hi_nerv_optimizer_policy": optimizer_policy,
                "hi_nerv_optimizer_controls": optimizer_controls,
                "score_aware_carrier_training_plan": score_aware_training_plan,
                "modelsize_budget_evidence": modelsize_budget_evidence,
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "blockers": _dedupe(
                    [
                        *waterfill_validation_blockers,
                        "hi_nerv_decoder_weight_waterfill_plan_not_attached",
                        "hi_nerv_training_not_launched",
                        "contest_cpu_cuda_exact_eval_not_executed",
                    ]
                ),
            }
        )
        refusal["candidate_feedback"] = write_nerv_candidate_feedback_files(
            runner_report=refusal,
            output_dir=out,
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, refusal)
        return {**refusal, "report_path": path.as_posix()}
    config_gate = _validate_hi_nerv_frontier_training_config(
        segnet_distillation_weight=effective_segnet_distillation_weight,
        pose_distillation_weight=effective_pose_distillation_weight,
        allow_segnet_only_research=allow_segnet_only_research,
        allow_unscored_research_smoke=allow_unscored_research_smoke,
        score_aware_training_plan=score_aware_training_plan,
    )
    if not config_gate["launch_allowed"]:
        refusal = _base_report(
            output_dir=out,
            mode="hi_nerv_mlx_scoreaware_launch_refused",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        refusal.update(
            {
                "execute_family": "hi_nerv",
                "num_pairs": int(num_pairs),
                "epochs_requested": int(epochs),
                "training_executed": False,
                "launch_refusal_reason": (
                    "HiNeRV frontier runs require real SegNet and PoseNet "
                    "score-aware loss weights unless explicitly marked as an "
                    "unscored research smoke"
                ),
                "score_aware_training_config_gate": config_gate,
                "hi_nerv_modelsize_launch_pressure": launch_pressure_binding,
                "hi_nerv_source_faithfulness": launch_source_faithfulness,
                "hi_nerv_control_precedence": launch_control_precedence,
                "score_aware_training": {
                    "schema": "compact_hi_nerv_score_aware_training.v1",
                    "status": "refused_before_mlx_training",
                    "decoder_weight_waterfill_plan": (
                        decoder_weight_waterfill_plan_metadata
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_aware_carrier_training_plan": score_aware_training_plan,
                "modelsize_budget_evidence": modelsize_budget_evidence,
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "blockers": _dedupe(
                    [
                        *config_gate.get("blockers", []),
                        *score_aware_training_plan.get("blockers", []),
                        "hi_nerv_training_not_launched",
                        "contest_cpu_cuda_exact_eval_not_executed",
                    ]
                ),
            }
        )
        refusal["candidate_feedback"] = write_nerv_candidate_feedback_files(
            runner_report=refusal,
            output_dir=out,
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, refusal)
        return {**refusal, "report_path": path.as_posix()}
    if (
        not bool(launch_source_faithfulness["official_hinerv_control"])
        and not bool(allow_unscored_research_smoke)
    ):
        refusal = _base_report(
            output_dir=out,
            mode="hi_nerv_official_control_launch_refused",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        refusal.update(
            {
                "execute_family": "hi_nerv",
                "num_pairs": int(num_pairs),
                "epochs_requested": int(epochs),
                "training_executed": False,
                "trainer_launch_allowed": False,
                "launch_refusal_reason": (
                    "HiNeRV top-priority launches require the official "
                    "hierarchical feature grid and ConvNeXt controls. Use "
                    "allow_unscored_research_smoke only for explicit "
                    "false-authority local probes."
                ),
                "modelsize_candidate_selection": {
                    "schema": "compact_execute_modelsize_candidate_selection.v1",
                    "family": "hi_nerv",
                    "selection_mode": (
                        "planner_candidate" if candidate else "manual_cli_knobs"
                    ),
                    "candidate": candidate or None,
                    "modelsize_control_contract": _modelsize_control_contract(
                        candidate
                    ),
                    "num_pairs_for_budget": CONTEST_PAIR_COUNT,
                    "launch_latent_dim": launch_latent_dim,
                    "launch_embed_dim": launch_embed_dim,
                    "launch_decoder_channel": launch_decoder_channel,
                    "launch_decoder_codec": launch_decoder_codec,
                    "launch_use_hierarchical_feature_grid": (
                        launch_use_hierarchical_feature_grid
                    ),
                    "launch_use_convnext_blocks": launch_use_convnext_blocks,
                    "launch_mid_injection_block_index": (
                        launch_mid_injection_block_index
                    ),
                    "launch_fine_injection_block_index": (
                        launch_fine_injection_block_index
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_aware_training_config_gate": config_gate,
                "hi_nerv_modelsize_launch_pressure": launch_pressure_binding,
                "hi_nerv_source_faithfulness": launch_source_faithfulness,
                "hi_nerv_control_precedence": launch_control_precedence,
                "score_aware_training": {
                    "schema": "compact_hi_nerv_score_aware_training.v1",
                    "status": "refused_before_mlx_training",
                    "decoder_weight_waterfill_plan": (
                        decoder_weight_waterfill_plan_metadata
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_aware_carrier_training_plan": score_aware_training_plan,
                "modelsize_budget_evidence": modelsize_budget_evidence,
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "blockers": _dedupe(
                    [
                        "hinerv_official_control_required_for_top_priority_launch",
                        *launch_source_faithfulness.get(
                            "official_hinerv_blockers",
                            [],
                        ),
                        "hi_nerv_training_not_launched",
                        "contest_cpu_cuda_exact_eval_not_executed",
                    ]
                ),
            }
        )
        refusal["candidate_feedback"] = write_nerv_candidate_feedback_files(
            runner_report=refusal,
            output_dir=out,
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, refusal)
        return {**refusal, "report_path": path.as_posix()}
    effective_recon_pixel_weight_path = recon_pixel_weight_path
    recon_pixel_weight_auto_discovery: dict[str, Any] | None = None
    enabled_recon_weight_modes = sum(
        int(bool(value))
        for value in (
            recon_pixel_weight_path is not None,
            auto_joint_recon_pixel_weight,
            auto_segnet_boundary_recon_weight,
        )
    )
    if enabled_recon_weight_modes > 1:
        raise CompactRendererMlxSpineRunnerError(
            "choose exactly one recon weight source: --recon-pixel-weight-path, "
            "--auto-joint-recon-pixel-weight, or "
            "--auto-segnet-boundary-recon-weight"
        )
    if auto_joint_recon_pixel_weight:
        (
            effective_recon_pixel_weight_path,
            recon_pixel_weight_auto_discovery,
        ) = _discover_joint_recon_pixel_weight_path(
            repo_root=root,
            num_pairs=int(num_pairs),
        )
    launch_curriculum_plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate or None,
        requested_epochs=int(epochs),
        num_pairs=int(num_pairs),
        segnet_distillation_weight=effective_segnet_distillation_weight,
        pose_distillation_weight=effective_pose_distillation_weight,
        coder_aware_qat=bool(coder_aware_qat),
        coder_qat_quant_bits=int(coder_qat_quant_bits),
        recon_pixel_weight_attached=bool(
            effective_recon_pixel_weight_path is not None
            or auto_segnet_boundary_recon_weight
        ),
        eval_roundtrip_ste_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
    )
    effective_coder_aware_qat = bool(
        launch_curriculum_plan["coder_pressure"]["enabled"]
    )
    effective_coder_qat_quant_bits = int(
        launch_curriculum_plan["coder_pressure"]["quant_bits"]
    )
    prelaunch_blockers = _pr95_long_campaign_prelaunch_blockers(
        launch_curriculum_plan,
        epochs=int(epochs),
    )
    if prelaunch_blockers:
        refusal = _base_report(
            output_dir=out,
            mode="hi_nerv_pr95_binding_prelaunch_refused",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        refusal.update(
            {
                "execute_family": "hi_nerv",
                "num_pairs": int(num_pairs),
                "epochs_requested": int(epochs),
                "training_executed": False,
                "launch_refusal_reason": (
                    "8+ epoch HiNeRV campaigns require the PR95-grade "
                    "prelaunch stack before MLX training execution"
                ),
                "modelsize_candidate_selection": {
                    "schema": "compact_execute_modelsize_candidate_selection.v1",
                    "family": "hi_nerv",
                    "selection_mode": (
                        "planner_candidate" if candidate else "manual_cli_knobs"
                    ),
                    "candidate": candidate or None,
                    "modelsize_control_contract": _modelsize_control_contract(
                        candidate
                    ),
                    "num_pairs_for_budget": CONTEST_PAIR_COUNT,
                    "launch_latent_dim": launch_latent_dim,
                    "launch_embed_dim": launch_embed_dim,
                    "launch_decoder_channel": launch_decoder_channel,
                    "launch_decoder_codec": launch_decoder_codec,
                    "launch_use_hierarchical_feature_grid": (
                        launch_use_hierarchical_feature_grid
                    ),
                    "launch_use_convnext_blocks": launch_use_convnext_blocks,
                    "launch_mid_injection_block_index": (
                        launch_mid_injection_block_index
                    ),
                    "launch_fine_injection_block_index": (
                        launch_fine_injection_block_index
                    ),
                    "candidate_curriculum_plan": launch_curriculum_plan,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "candidate_curriculum_plan": launch_curriculum_plan,
                "score_aware_training_config_gate": config_gate,
                "hi_nerv_modelsize_launch_pressure": launch_pressure_binding,
                "hi_nerv_source_faithfulness": launch_source_faithfulness,
                "hi_nerv_control_precedence": launch_control_precedence,
                "hi_nerv_optimizer_policy": optimizer_policy,
                "hi_nerv_optimizer_controls": optimizer_controls,
                "score_aware_carrier_training_plan": score_aware_training_plan,
                "modelsize_budget_evidence": modelsize_budget_evidence,
                "blockers": prelaunch_blockers,
            }
        )
        refusal["candidate_feedback"] = write_nerv_candidate_feedback_files(
            runner_report=refusal,
            output_dir=out,
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, refusal)
        return {**refusal, "report_path": path.as_posix()}
    try:
        artifact = _run_hi_nerv_mlx_scoreaware_smoke(
            output_dir=out / "hi_nerv_mlx_training",
            num_pairs=num_pairs,
            epochs=epochs,
            batch_pair_indices_per_step=batch_pair_indices_per_step,
            learning_rate=learning_rate,
            source_video_path=resolved_source_video,
            latent_dim=launch_latent_dim,
            embed_dim=launch_embed_dim,
            decoder_channel=launch_decoder_channel,
            use_hierarchical_feature_grid=launch_use_hierarchical_feature_grid,
            use_convnext_blocks=launch_use_convnext_blocks,
            local_grid_levels=launch_local_grid_levels,
            local_grid_channels=launch_local_grid_channels,
            convnext_mlp_ratio=launch_convnext_mlp_ratio,
            convnext_kernel_size=launch_convnext_kernel_size,
            mid_injection_block_index=launch_mid_injection_block_index,
            fine_injection_block_index=launch_fine_injection_block_index,
            decoder_codec=launch_decoder_codec,
            hi_nerv_latent_codec=str(hi_nerv_latent_codec),
            ema_decay=ema_decay,
            segnet_distillation_weight=effective_segnet_distillation_weight,
            pose_distillation_weight=effective_pose_distillation_weight,
            pose_distillation_loss=str(pose_distillation_loss),
            pose_distillation_huber_delta=float(pose_distillation_huber_delta),
            segnet_distillation_objective=segnet_distillation_objective,
            distillation_temperature=distillation_temperature,
            segnet_tau_boundary=segnet_tau_boundary,
            segnet_hinge_margin=segnet_hinge_margin,
            distillation_device=distillation_device,
            requested_distillation_device=effective_requested_distillation_device,
            allow_segnet_only_research=allow_segnet_only_research,
            coder_aware_qat=effective_coder_aware_qat,
            coder_qat_quant_bits=effective_coder_qat_quant_bits,
            coder_qat_quant_residual_weight=coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=coder_qat_magnitude_weight,
            coder_qat_delta_weight=coder_qat_delta_weight,
            coder_qat_c1a_entropy_weight=coder_qat_c1a_entropy_weight,
            coder_qat_c1a_sigma=coder_qat_c1a_sigma,
            coder_qat_c1a_sample_size=coder_qat_c1a_sample_size,
            recon_pixel_weight_path=effective_recon_pixel_weight_path,
            decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
            recon_pixel_weight_auto_discovery=recon_pixel_weight_auto_discovery,
            auto_segnet_boundary_recon_weight=auto_segnet_boundary_recon_weight,
            recon_pixel_weight_tau=recon_pixel_weight_tau,
            recon_pixel_weight_normalize=recon_pixel_weight_normalize,
            mlx_prefilter_scorer_device=mlx_prefilter_scorer_device,
            mlx_prefilter_scorer_batch_pairs=mlx_prefilter_scorer_batch_pairs,
            mlx_prefilter_progress_every=mlx_prefilter_progress_every,
            telemetry_flush_interval_epochs=telemetry_flush_interval_epochs,
            checkpoint_interval_epochs=checkpoint_interval,
            checkpoint_dir=resolved_checkpoint_dir,
            resume_from_checkpoint=resolved_resume_from_checkpoint,
            optimizer_kind=str(optimizer_kind),
            hi_nerv_optimizer_policy=optimizer_policy,
            optimizer_controls=optimizer_controls,
            prioritized_pair_indices=prioritized_pair_indices,
            random_seed=random_seed,
            scorer_upstream_dir=scorer_upstream,
            repo_root=root,
            candidate_curriculum_plan=launch_curriculum_plan,
        )
    except Exception as exc:
        blocker_report = _base_report(
            output_dir=out,
            mode="hi_nerv_mlx_scoreaware_failed",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        blocker_report.update(
            {
                "execute_family": "hi_nerv",
                "training_executed": False,
                "trainer_launch_allowed": True,
                "requested_distillation_device": effective_requested_distillation_device,
                "distillation_device": str(distillation_device),
                "failure": repr(exc),
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "modelsize_candidate_selection": {
                    "schema": "compact_execute_modelsize_candidate_selection.v1",
                    "family": "hi_nerv",
                    "selection_mode": (
                        "planner_candidate" if candidate else "manual_cli_knobs"
                    ),
                    "candidate": candidate or None,
                    "modelsize_control_contract": _modelsize_control_contract(
                        candidate
                    ),
                    "num_pairs_for_budget": CONTEST_PAIR_COUNT,
                    "launch_latent_dim": launch_latent_dim,
                    "launch_embed_dim": launch_embed_dim,
                    "launch_decoder_channel": launch_decoder_channel,
                    "launch_decoder_codec": launch_decoder_codec,
                    "launch_use_hierarchical_feature_grid": (
                        launch_use_hierarchical_feature_grid
                    ),
                    "launch_use_convnext_blocks": launch_use_convnext_blocks,
                    "launch_mid_injection_block_index": (
                        launch_mid_injection_block_index
                    ),
                    "launch_fine_injection_block_index": (
                        launch_fine_injection_block_index
                    ),
                    "candidate_curriculum_plan": launch_curriculum_plan,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "score_aware_training_config_gate": config_gate,
                "hi_nerv_modelsize_launch_pressure": launch_pressure_binding,
                "hi_nerv_source_faithfulness": launch_source_faithfulness,
                "hi_nerv_control_precedence": launch_control_precedence,
                "hi_nerv_optimizer_policy": optimizer_policy,
                "hi_nerv_optimizer_controls": optimizer_controls,
                "checkpoint_interval_epochs": checkpoint_interval,
                "score_aware_carrier_training_plan": score_aware_training_plan,
                "modelsize_budget_evidence": modelsize_budget_evidence,
                "blockers": ["hi_nerv_mlx_scoreaware_or_export_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}

    artifact_dict = artifact.as_dict() if hasattr(artifact, "as_dict") else dict(artifact)
    archive_path = artifact_dict.get("archive_path")
    training_dir = out / "hi_nerv_mlx_training"
    decoder_weight_saliency_artifact = _write_decoder_weight_saliency_artifact(
        artifact_dict=artifact_dict,
        output_dir=training_dir,
        family="hi_nerv",
    )
    substrate_metadata = artifact_dict.get("substrate_artifact_metadata")
    if isinstance(substrate_metadata, dict):
        scoreaware_metadata = substrate_metadata.get("score_aware_training")
        if isinstance(scoreaware_metadata, dict):
            scoreaware_metadata["decoder_weight_gradient_saliency_artifact"] = (
                decoder_weight_saliency_artifact
            )
    archive_file_path = Path(archive_path) if archive_path else None
    archive_artifact_dir = (
        archive_file_path.parent
        if archive_file_path is not None and archive_file_path.parent.is_dir()
        else training_dir
    )
    auto_mlx_prefilter_profile_path = training_dir / "local_mlx_prefilter_profile.json"
    effective_mlx_profile_paths: tuple[str | Path, ...] = tuple(mlx_profile_paths)
    if auto_mlx_prefilter_profile_path.is_file():
        effective_mlx_profile_paths = (
            *effective_mlx_profile_paths,
            auto_mlx_prefilter_profile_path,
        )
    spine_manifest = (
        archive_artifact_dir / "hprc_representation_spine_hi_nerv_manifest.json"
    )
    receiver_proof_path = (
        archive_artifact_dir / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
    )
    projection_paths = [spine_manifest] if spine_manifest.is_file() else []
    receiver_proof_paths = [receiver_proof_path] if receiver_proof_path.is_file() else []
    mlx_prefilter_coverage = summarize_mlx_prefilter_coverage(
        effective_mlx_profile_paths,
        root=root,
    )
    has_full_video_mlx_prefilter = bool(
        mlx_prefilter_coverage["has_full_video_mlx_prefilter"]
    )
    mlx_prefilter_local_replay_passed = bool(
        mlx_prefilter_coverage["local_replay_mlx_prefilter_passed"]
    )
    local_cpu_replay_summary: dict[str, Any] | None = None
    local_cpu_replay_paths: list[Path] = []
    local_cpu_replay_blockers: list[str] = []
    if archive_path:
        (
            local_cpu_replay_summary,
            local_cpu_replay_paths,
            local_cpu_replay_blockers,
        ) = _run_compact_local_cpu_replay_gate(
            archive_zip_path=archive_file_path or Path(archive_path),
            runtime_submission_dir=archive_artifact_dir / "submission",
            output_dir=out / "local_cpu_replay",
            upstream_dir=scorer_upstream,
            num_pairs=int(num_pairs),
            requested=run_local_cpu_replay,
            has_full_video_mlx_prefilter=has_full_video_mlx_prefilter,
            mlx_prefilter_local_replay_passed=mlx_prefilter_local_replay_passed,
            keep_inflated=keep_local_replay_inflated,
            cleanup_failed_scratch=cleanup_failed_local_replay_scratch,
            repo_root=root,
        )
    trained_archive_byte_oracle = _write_hi_nerv_trained_archive_byte_oracle(
        output_dir=out,
        artifact_dict=artifact_dict,
        modelsize_candidate=candidate or None,
        num_pairs=int(num_pairs),
        receiver_proof_path=receiver_proof_path,
        local_cpu_replay_summary=local_cpu_replay_summary,
        mlx_prefilter_coverage=mlx_prefilter_coverage,
        repo_root=root,
    )
    candidate_curriculum_plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate or None,
        requested_epochs=int(epochs),
        num_pairs=int(num_pairs),
        segnet_distillation_weight=effective_segnet_distillation_weight,
        pose_distillation_weight=effective_pose_distillation_weight,
        coder_aware_qat=bool(coder_aware_qat),
        coder_qat_quant_bits=int(coder_qat_quant_bits),
        recon_pixel_weight_attached=bool(
            effective_recon_pixel_weight_path is not None
            or auto_segnet_boundary_recon_weight
        ),
        eval_roundtrip_ste_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        receiver_proof_attached=bool(receiver_proof_paths),
        full_video_local_prefilter_attached=has_full_video_mlx_prefilter,
        local_cpu_replay_gate_attached=local_cpu_replay_summary is not None,
        measured_archive_bytes=(
            int(trained_archive_byte_oracle["measured_archive_bytes"])
            if trained_archive_byte_oracle.get("measured_archive_bytes") is not None
            else None
        ),
    )
    if isinstance(candidate_curriculum_plan.get("byte_oracle_logging"), dict):
        candidate_curriculum_plan["byte_oracle_logging"].update(
            {
                "byte_feedback_source": "hi_nerv_trained_archive_byte_oracle",
                "trained_archive_byte_oracle_path": trained_archive_byte_oracle[
                    "path"
                ],
                "trained_archive_byte_oracle_sha256": trained_archive_byte_oracle[
                    "sha256"
                ],
                "receiver_closed_modelsize_ladder_path": (
                    trained_archive_byte_oracle[
                        "receiver_closed_modelsize_ladder_path"
                    ]
                ),
                "receiver_closed_modelsize_ladder_sha256": (
                    trained_archive_byte_oracle[
                        "receiver_closed_modelsize_ladder_sha256"
                    ]
                ),
                "receiver_proof_passed": trained_archive_byte_oracle["row"][
                    "receiver_proof_passed"
                ],
                "local_cpu_replay_executed": trained_archive_byte_oracle["row"][
                    "local_cpu_replay_executed"
                ],
                "local_cpu_replay_axis_tag": trained_archive_byte_oracle["row"][
                    "local_cpu_replay_axis_tag"
                ],
                "byte_oracle_feedback_ready": trained_archive_byte_oracle[
                    "feedback_ready"
                ],
                "byte_oracle_blockers": list(
                    trained_archive_byte_oracle.get("blockers") or []
                ),
            }
        )
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    selected_runner_rows: list[dict[str, Any]] = []
    blockers: list[Any] = [
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if artifact_dict.get("early_stopped") is True:
        early_reason = str(artifact_dict.get("early_stop_reason") or "unknown")
        blockers.append(f"hi_nerv_training_early_stopped:{early_reason}")
        if "pose_instability" in early_reason:
            blockers.append("hi_nerv_pose_instability_guard_triggered")
    blockers.extend(config_gate.get("blockers") or [])
    blockers.extend(score_aware_training_plan.get("blockers") or [])
    blockers.extend(candidate_curriculum_plan.get("blockers") or [])
    blockers.extend(trained_archive_byte_oracle.get("blockers") or [])
    blockers.extend(local_cpu_replay_blockers)
    if not decoder_weight_saliency_artifact.get("written"):
        blockers.append(
            decoder_weight_saliency_artifact.get("reason")
            or "hi_nerv_decoder_weight_saliency_artifact_missing"
        )
    elif int(decoder_weight_saliency_artifact.get("row_count") or 0) <= 0:
        blockers.append("hi_nerv_decoder_weight_saliency_no_decoder_rows")
    pr95_curriculum_enabled = bool(
        optimizer_policy.get("pr95_faithful_curriculum_enabled")
    )
    if (
        effective_segnet_distillation_weight <= 0.0
        or effective_pose_distillation_weight <= 0.0
    ):
        blockers.append("hi_nerv_real_segnet_posenet_teachers_not_both_attached")
    blockers.extend(mlx_prefilter_coverage.get("blockers") or [])
    if projection_paths:
        acquisition = build_spine_acquisition_report(
            projection_manifest_paths=projection_paths,
            hard_byte_ceilings=hard_byte_ceilings,
        )
        _write_json(acquisition_path, acquisition)
        runner_plan = build_spine_bounded_runner_plan(
            acquisition_report_path=acquisition_path,
            mlx_profile_paths=effective_mlx_profile_paths,
            receiver_proof_report_paths=receiver_proof_paths,
            hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
            repo_root=root,
            upstream_dir=scorer_upstream,
        )
        write_spine_bounded_runner_plan(
            output_path=runner_plan_path,
            plan=runner_plan,
            allow_overwrite=True,
        )
        selected_runner_rows = list(runner_plan.get("selected_runner_rows") or [])
        blockers.extend(runner_plan.get("blockers") or [])
    else:
        blockers.append("hi_nerv_spine_projection_manifest_missing")
    if not archive_path:
        blockers.append("hi_nerv_archive_export_missing")
    post_export_materializer_plan = _compile_carrier_post_export_materializer_plan(
        output_dir=out,
        archive_path=archive_file_path,
        archive_sha256=artifact_dict.get("archive_sha256"),
        archive_bytes=artifact_dict.get("archive_bytes"),
        family="hi_nerv",
        runtime_submission_dir=archive_artifact_dir / "submission",
        repo_root=root,
    )
    blockers.extend(post_export_materializer_plan.get("blockers") or [])
    post_export_materializer_execution = _execute_carrier_post_export_materializer_plan(
        plan=post_export_materializer_plan,
        requested=run_post_export_materializers,
        max_steps=post_export_materializer_max_steps,
        max_parallel=post_export_materializer_max_parallel,
        max_experiments=post_export_materializer_max_experiments,
        repo_root=root,
    )
    blockers.extend(post_export_materializer_execution.get("blockers") or [])

    final = _base_report(
        output_dir=out,
        mode="executed_hi_nerv_mlx_scoreaware_and_exported",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "execute_family": "hi_nerv",
            "num_pairs": int(num_pairs),
            "coverage_valid_for_base_comparison": int(num_pairs) >= 600,
            "training_executed": True,
            "adapter_smoke_only": False,
            "modelsize_candidate_selection": {
                "schema": "compact_execute_modelsize_candidate_selection.v1",
                "family": "hi_nerv",
                "selection_mode": "planner_candidate" if candidate else "manual_cli_knobs",
                "candidate": candidate or None,
                "modelsize_control_contract": _modelsize_control_contract(candidate),
                "num_pairs_for_budget": CONTEST_PAIR_COUNT,
                "launch_latent_dim": launch_latent_dim,
                "launch_embed_dim": launch_embed_dim,
                "launch_decoder_channel": launch_decoder_channel,
                "launch_decoder_codec": launch_decoder_codec,
                "launch_use_hierarchical_feature_grid": (
                    launch_use_hierarchical_feature_grid
                ),
                "launch_use_convnext_blocks": launch_use_convnext_blocks,
                "launch_mid_injection_block_index": launch_mid_injection_block_index,
                "launch_fine_injection_block_index": (
                    launch_fine_injection_block_index
                ),
                "candidate_curriculum_plan": candidate_curriculum_plan,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "archive_path": archive_path,
            "archive_bytes": artifact_dict.get("archive_bytes"),
            "archive_sha256": artifact_dict.get("archive_sha256"),
            "archive_selection_manifest_path": artifact_dict.get(
                "archive_selection_manifest_path"
            ),
            "scorer_upstream_snapshot": _scorer_upstream_metadata(
                scorer_upstream
            ),
            "training_artifact": artifact_dict,
            "trained_archive_byte_oracle": trained_archive_byte_oracle,
            "receiver_closed_modelsize_ladder_path": (
                trained_archive_byte_oracle[
                    "receiver_closed_modelsize_ladder_path"
                ]
            ),
            "candidate_curriculum_plan": candidate_curriculum_plan,
            "score_aware_training_config_gate": config_gate,
            "hi_nerv_modelsize_launch_pressure": launch_pressure_binding,
            "hi_nerv_source_faithfulness": launch_source_faithfulness,
            "hi_nerv_control_precedence": launch_control_precedence,
            "hi_nerv_optimizer_policy": optimizer_policy,
            "hi_nerv_optimizer_controls": optimizer_controls,
            "score_aware_carrier_training_plan": score_aware_training_plan,
            "modelsize_budget_evidence": modelsize_budget_evidence,
            "score_aware_training": {
                "schema": "compact_hi_nerv_score_aware_training.v1",
                "status": "executed_mlx_local_false_authority",
                "requested_segnet_distillation_weight": float(
                    segnet_distillation_weight
                ),
                "requested_pose_distillation_weight": float(
                    pose_distillation_weight
                ),
                "segnet_distillation_weight": (
                    effective_segnet_distillation_weight
                ),
                "pose_distillation_weight": effective_pose_distillation_weight,
                "pose_distillation_loss": str(pose_distillation_loss),
                "pose_distillation_huber_delta": float(
                    pose_distillation_huber_delta
                ),
                "modelsize_launch_pressure": launch_pressure_binding,
                "segnet_distillation_objective": segnet_distillation_objective,
                "distillation_temperature": float(distillation_temperature),
                "segnet_tau_boundary": float(segnet_tau_boundary),
                "segnet_hinge_margin": float(segnet_hinge_margin),
                "distillation_device": distillation_device,
                "allow_segnet_only_research": bool(allow_segnet_only_research),
                "allow_unscored_research_smoke": bool(allow_unscored_research_smoke),
                "config_gate": config_gate,
                "decoder_codec": str(launch_decoder_codec),
                "requested_optimizer_kind": str(optimizer_kind),
                "optimizer_kind": str(
                    optimizer_policy.get("optimizer_kind") or optimizer_kind
                ),
                "optimizer_policy": optimizer_policy,
                "optimizer_controls": optimizer_controls,
                "pr95_faithful_curriculum_enabled": pr95_curriculum_enabled,
                "native_optimizer_active": bool(
                    optimizer_policy.get("native_optimizer_active")
                ),
                "effective_weight_decay": (
                    optimizer_controls.get("weight_decay_effective")
                ),
                "checkpoint_interval_epochs": checkpoint_interval,
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "prioritized_pair_training": {
                    "schema": "compact_hi_nerv_prioritized_pair_training.v1",
                    "enabled": bool(prioritized_pair_indices),
                    "pair_indices": [int(value) for value in prioritized_pair_indices],
                    "pair_count": len(prioritized_pair_indices),
                    "sampling_scope": "training_batch_emphasis_only",
                    "pair_index_domain": (
                        "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
                    ),
                    "arbitrary_source_pair_hydration": False,
                    "target_hydration_pair_indices_consumed": False,
                    "requires_num_pairs_covering_pair_ids": bool(
                        prioritized_pair_indices
                    ),
                    "authority": "macos_mlx_research_signal_false_authority",
                    **FALSE_AUTHORITY,
                },
                "coder_aware_qat": _coder_qat_report_metadata(
                    artifact_dict=artifact_dict,
                    enabled=effective_coder_aware_qat,
                    quant_bits=effective_coder_qat_quant_bits,
                    quant_residual_weight=coder_qat_quant_residual_weight,
                    magnitude_weight=coder_qat_magnitude_weight,
                    delta_weight=coder_qat_delta_weight,
                    c1a_entropy_weight=coder_qat_c1a_entropy_weight,
                    c1a_sigma=coder_qat_c1a_sigma,
                    c1a_sample_size=coder_qat_c1a_sample_size,
                ),
                "decoder_weight_waterfill_plan": decoder_weight_waterfill_plan_metadata,
                "eval_roundtrip_ste": _eval_roundtrip_ste_report_metadata(
                    artifact_dict
                ),
                "pose_student_input_preprocess": (
                    _pose_student_input_preprocess_report_metadata(artifact_dict)
                ),
                "recon_pixel_weight": _recon_pixel_weight_report_metadata(
                    artifact_dict
                ),
                "local_mlx_prefilter": {
                    "schema": "compact_hi_nerv_local_mlx_prefilter_config.v1",
                    "scorer_device": (
                        mlx_prefilter_scorer_device or distillation_device
                    ),
                    "scorer_batch_pairs": int(mlx_prefilter_scorer_batch_pairs),
                    "progress_every": int(mlx_prefilter_progress_every),
                    "singleton_required_for_local_cpu_replay_unlock": True,
                    "gpu_profiles_are_prefilter_only": (
                        str(mlx_prefilter_scorer_device or distillation_device)
                        != "cpu"
                    ),
                    "batched_profiles_are_prefilter_only": (
                        int(mlx_prefilter_scorer_batch_pairs) != 1
                    ),
                    "authority": "macos_mlx_research_signal_false_authority",
                },
                "pose_instability_monitor": (
                    _pose_instability_monitor_report_metadata(artifact_dict)
                ),
                "decoder_weight_gradient_saliency_artifact": (
                    decoder_weight_saliency_artifact
                ),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "projection_manifest_paths": [path.as_posix() for path in projection_paths],
            "receiver_proof_report_paths": [
                path.as_posix() for path in receiver_proof_paths
            ],
            "local_cpu_replay_summary_paths": [
                path.as_posix() for path in local_cpu_replay_paths
            ],
            "local_cpu_replay_summary": local_cpu_replay_summary,
            "hi_nerv_trained_archive_byte_oracle": trained_archive_byte_oracle,
            "post_export_materializer_plan": post_export_materializer_plan,
            "post_export_materializer_execution": post_export_materializer_execution,
            "local_cpu_replay_gate": {
                "schema": "compact_runner_local_cpu_replay_gate.v1",
                "requested": run_local_cpu_replay,
                "default_enabled_for_full_coverage": (
                    _local_cpu_replay_enabled_by_default(
                        int(num_pairs),
                        mlx_prefilter_local_replay_passed=(
                            mlx_prefilter_local_replay_passed
                        ),
                    )
                ),
                "has_full_video_mlx_prefilter": has_full_video_mlx_prefilter,
                "local_replay_mlx_prefilter_passed": (
                    mlx_prefilter_local_replay_passed
                ),
                "coverage_valid_for_replay": int(num_pairs) >= CONTEST_PAIR_COUNT,
                "executed": local_cpu_replay_summary is not None,
                "axis_tag": "[macOS-CPU advisory]",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "mlx_prefilter_coverage": mlx_prefilter_coverage,
            "auto_mlx_prefilter_profile_path": (
                auto_mlx_prefilter_profile_path.as_posix()
                if auto_mlx_prefilter_profile_path.is_file()
                else None
            ),
            "auto_mlx_prefilter_progress_path": (
                (training_dir / "local_mlx_prefilter_progress.jsonl").as_posix()
                if (training_dir / "local_mlx_prefilter_progress.jsonl").is_file()
                else None
            ),
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix()
                for path in effective_mlx_profile_paths
            ],
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "acquisition_report_path": (
                acquisition_path.as_posix() if acquisition_path.is_file() else None
            ),
            "bounded_runner_plan_path": (
                runner_plan_path.as_posix() if runner_plan_path.is_file() else None
            ),
            "selected_runner_rows": selected_runner_rows,
            "blockers": _dedupe(blockers),
        }
    )
    final["candidate_feedback"] = write_nerv_candidate_feedback_files(
        runner_report=final,
        output_dir=out,
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def execute_pact_nerv_selector_v4_mlx_smoke_and_adapt(
    *,
    output_dir: str | Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    latent_dim: int = 8,
    embed_dim: int = 8,
    selector_palette_size: int = 16,
    decoder_channel: int = 8,
    decoder_codec: str = "int8_mixed",
    ema_decay: float = 0.9,
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    pose_distillation_loss: str = "mse",
    pose_distillation_huber_delta: float = 1.0,
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    requested_distillation_device: str | None = None,
    allow_segnet_only_research: bool = False,
    scorer_upstream_dir: str | Path | None = None,
    coder_aware_qat: bool = False,
    coder_qat_quant_bits: int = 8,
    coder_qat_quant_residual_weight: float = (
        DEFAULT_PACT_CODER_QAT_QUANT_RESIDUAL_WEIGHT
    ),
    coder_qat_magnitude_weight: float = DEFAULT_PACT_CODER_QAT_MAGNITUDE_WEIGHT,
    coder_qat_delta_weight: float = DEFAULT_PACT_CODER_QAT_DELTA_WEIGHT,
    coder_qat_c1a_entropy_weight: float = DEFAULT_PACT_CODER_QAT_C1A_ENTROPY_WEIGHT,
    coder_qat_c1a_sigma: float = DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
    coder_qat_c1a_sample_size: int = DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
    optimizer_kind: str = DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
    optimizer_grad_clip_max_norm: float | None = (
        DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_GRAD_CLIP_MAX_NORM
    ),
    optimizer_weight_decay: float | None = None,
    optimizer_warmup_epochs: int = 0,
    optimizer_warmup_steps_per_epoch: int = 1,
    optimizer_cosine_decay_enabled: bool = False,
    optimizer_cosine_decay_total_epochs: int | None = None,
    optimizer_cosine_decay_min_lr_ratio: float = 1e-2,
    checkpoint_interval_epochs: int = DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS,
    checkpoint_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    run_post_export_materializers: bool = False,
    post_export_materializer_max_steps: int = 1,
    post_export_materializer_max_parallel: int = 0,
    post_export_materializer_max_experiments: int | None = 1,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a tiny real-video PACT-NeRV-SELECTOR-V4 candidate."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    resolved_checkpoint_dir = _resolve_optional_compact_family_path(
        checkpoint_dir,
        base=root,
    )
    resolved_resume_from_checkpoint = _resolve_optional_compact_family_path(
        resume_from_checkpoint,
        base=root,
    )
    scorer_upstream = _resolve_scorer_upstream_dir(root, scorer_upstream_dir)
    optimizer_controls = _resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind=optimizer_kind,
        requested_weight_decay=optimizer_weight_decay,
        grad_clip_max_norm=optimizer_grad_clip_max_norm,
        warmup_epochs=optimizer_warmup_epochs,
        warmup_steps_per_epoch=optimizer_warmup_steps_per_epoch,
        cosine_decay_enabled=optimizer_cosine_decay_enabled,
        cosine_decay_total_epochs=optimizer_cosine_decay_total_epochs,
        cosine_decay_min_lr_ratio=optimizer_cosine_decay_min_lr_ratio,
        run_epochs=epochs,
    )
    optimizer_policy = _resolve_pact_compact_optimizer_policy(
        family="pact_nerv_selector_v4",
        optimizer_controls=optimizer_controls,
    )
    _, coder_qat_metadata_row = _build_pact_coder_qat_config_and_metadata(
        coder_aware_qat=coder_aware_qat,
        coder_qat_quant_bits=coder_qat_quant_bits,
        coder_qat_quant_residual_weight=coder_qat_quant_residual_weight,
        coder_qat_magnitude_weight=coder_qat_magnitude_weight,
        coder_qat_delta_weight=coder_qat_delta_weight,
        coder_qat_c1a_entropy_weight=coder_qat_c1a_entropy_weight,
        coder_qat_c1a_sigma=coder_qat_c1a_sigma,
        coder_qat_c1a_sample_size=coder_qat_c1a_sample_size,
    )
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    try:
        artifact = _run_pact_nerv_selector_v4_mlx_smoke(
            output_dir=out / "pact_nerv_selector_v4_mlx_training",
            num_pairs=num_pairs,
            epochs=epochs,
            batch_pair_indices_per_step=batch_pair_indices_per_step,
            learning_rate=learning_rate,
            source_video_path=source_video_path,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            selector_palette_size=selector_palette_size,
            decoder_channel=decoder_channel,
            decoder_codec=decoder_codec,
            ema_decay=ema_decay,
            segnet_distillation_weight=segnet_distillation_weight,
            pose_distillation_weight=pose_distillation_weight,
            pose_distillation_loss=pose_distillation_loss,
            pose_distillation_huber_delta=pose_distillation_huber_delta,
            segnet_distillation_objective=segnet_distillation_objective,
            distillation_temperature=distillation_temperature,
            segnet_tau_boundary=segnet_tau_boundary,
            segnet_hinge_margin=segnet_hinge_margin,
            distillation_device=distillation_device,
            allow_segnet_only_research=allow_segnet_only_research,
            coder_aware_qat=coder_aware_qat,
            coder_qat_quant_bits=coder_qat_quant_bits,
            coder_qat_quant_residual_weight=coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=coder_qat_magnitude_weight,
            coder_qat_delta_weight=coder_qat_delta_weight,
            coder_qat_c1a_entropy_weight=coder_qat_c1a_entropy_weight,
            coder_qat_c1a_sigma=coder_qat_c1a_sigma,
            coder_qat_c1a_sample_size=coder_qat_c1a_sample_size,
            optimizer_kind=optimizer_kind,
            optimizer_policy=optimizer_policy,
            optimizer_controls=optimizer_controls,
            checkpoint_interval_epochs=checkpoint_interval,
            checkpoint_dir=resolved_checkpoint_dir,
            resume_from_checkpoint=resolved_resume_from_checkpoint,
            random_seed=random_seed,
            scorer_upstream_dir=scorer_upstream,
            repo_root=root,
        )
    except Exception as exc:
        blocker_report = _base_report(
            output_dir=out,
            mode="pact_nerv_selector_v4_mlx_smoke_failed",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        blocker_report.update(
            {
                "execute_family": "pact_nerv_selector_v4",
                "failure": repr(exc),
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "blockers": ["pact_nerv_selector_v4_mlx_smoke_or_export_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}

    artifact_dict = artifact.as_dict() if hasattr(artifact, "as_dict") else dict(artifact)
    archive_path = artifact_dict.get("archive_path")
    selector_v4_render_quality = (
        artifact_dict.get("substrate_artifact_metadata", {}) or {}
    ).get("selector_v4_render_quality")
    training_dir = out / "pact_nerv_selector_v4_mlx_training"
    spine_manifest = (
        training_dir / "hprc_representation_spine_pact_nerv_selector_v4_manifest.json"
    )
    receiver_proof_path = (
        training_dir
        / "receiver_proof"
        / "pact_nerv_selector_v4_mlx_receiver_proof.json"
    )
    projection_paths = [spine_manifest] if spine_manifest.is_file() else []
    receiver_proof_paths = [receiver_proof_path] if receiver_proof_path.is_file() else []
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    selected_runner_rows: list[dict[str, Any]] = []
    blockers: list[Any] = [
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not mlx_profile_paths:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    if int(num_pairs) < 600:
        blockers.append("partial_pair_coverage_not_promotion_comparable")
    if projection_paths:
        acquisition = build_spine_acquisition_report(
            projection_manifest_paths=projection_paths,
            hard_byte_ceilings=hard_byte_ceilings,
        )
        _write_json(acquisition_path, acquisition)
        runner_plan = build_spine_bounded_runner_plan(
            acquisition_report_path=acquisition_path,
            mlx_profile_paths=mlx_profile_paths,
            receiver_proof_report_paths=receiver_proof_paths,
            hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
            repo_root=root,
            upstream_dir=scorer_upstream,
        )
        write_spine_bounded_runner_plan(
            output_path=runner_plan_path,
            plan=runner_plan,
            allow_overwrite=True,
        )
        selected_runner_rows = list(runner_plan.get("selected_runner_rows") or [])
        blockers.extend(runner_plan.get("blockers") or [])
    else:
        blockers.append("pact_nerv_selector_v4_spine_projection_manifest_missing")
    if archive_path is None:
        blockers.append("pact_nerv_selector_v4_archive_export_missing")
    if isinstance(selector_v4_render_quality, Mapping):
        blockers.extend(selector_v4_render_quality.get("blockers") or [])
        if selector_v4_render_quality.get("export_blocked_recommended"):
            blockers.append("pact_nerv_selector_v4_render_quality_gate_failed")
    post_export_materializer_plan = _compile_carrier_post_export_materializer_plan(
        output_dir=out,
        archive_path=archive_path,
        archive_sha256=artifact_dict.get("archive_sha256"),
        archive_bytes=artifact_dict.get("archive_bytes"),
        family="pact_nerv_selector_v4",
        runtime_submission_dir=training_dir / "submission",
        repo_root=root,
    )
    blockers.extend(post_export_materializer_plan.get("blockers") or [])
    post_export_materializer_execution = _execute_carrier_post_export_materializer_plan(
        plan=post_export_materializer_plan,
        requested=run_post_export_materializers,
        max_steps=post_export_materializer_max_steps,
        max_parallel=post_export_materializer_max_parallel,
        max_experiments=post_export_materializer_max_experiments,
        repo_root=root,
    )
    blockers.extend(post_export_materializer_execution.get("blockers") or [])

    final = _base_report(
        output_dir=out,
        mode="executed_pact_nerv_selector_v4_mlx_smoke_and_exported",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "execute_family": "pact_nerv_selector_v4",
            "num_pairs": int(num_pairs),
            "coverage_valid_for_base_comparison": int(num_pairs) >= 600,
            "training_artifact": artifact_dict,
            "archive_path": archive_path,
            "archive_bytes": artifact_dict.get("archive_bytes"),
            "archive_sha256": artifact_dict.get("archive_sha256"),
            "selector_v4_render_quality": selector_v4_render_quality,
            "ema_decay": float(ema_decay),
            "score_aware_training": {
                "schema": "compact_pact_nerv_selector_v4_score_aware_training.v1",
                "segnet_distillation_weight": float(segnet_distillation_weight),
                "pose_distillation_weight": float(pose_distillation_weight),
                "pose_distillation_loss": str(pose_distillation_loss),
                "pose_distillation_huber_delta": float(
                    pose_distillation_huber_delta
                ),
                "segnet_distillation_objective": segnet_distillation_objective,
                "distillation_temperature": float(distillation_temperature),
                "segnet_tau_boundary": float(segnet_tau_boundary),
                "segnet_hinge_margin": float(segnet_hinge_margin),
                "distillation_device": distillation_device,
                "allow_segnet_only_research": bool(allow_segnet_only_research),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "scorer_coupled_rd": _scorer_coupled_rd_metadata(),
                "optimizer_policy": strip_candidate_curriculum_authority_fields(
                    optimizer_policy
                ),
                "pr95_faithful_curriculum_enabled": bool(
                    optimizer_policy.get("pr95_faithful_curriculum_enabled")
                ),
                "native_optimizer_active": bool(
                    optimizer_policy.get("native_optimizer_active")
                ),
                "optimizer_kind": str(
                    optimizer_policy.get("optimizer_kind") or optimizer_kind
                ),
                "optimizer_controls": strip_candidate_curriculum_authority_fields(
                    optimizer_controls
                ),
                "effective_weight_decay": optimizer_controls.get(
                    "weight_decay_effective"
                ),
                "checkpoint_interval_epochs": checkpoint_interval,
                "checkpoint_dir": (
                    resolved_checkpoint_dir.as_posix()
                    if resolved_checkpoint_dir is not None
                    else None
                ),
                "resume_from_checkpoint": (
                    resolved_resume_from_checkpoint.as_posix()
                    if resolved_resume_from_checkpoint is not None
                    else None
                ),
                "coder_aware_qat": coder_qat_metadata_row,
                "decoder_codec": str(decoder_codec),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "selector_v4_archive_surface": {
                "schema": "pact_nerv_selector_v4_archive_surface.v1",
                "selector_codec": "run_length_varint_selector",
                "selector_palette_size": int(selector_palette_size),
                "archive_exporter": (
                    "tac.substrates.pact_nerv_selector_v4.archive_candidate."
                    "export_pact_nerv_selector_v4_mlx_archive"
                ),
                "primitive_timing": "archive_encode_time_not_training_forward_pass",
            },
            "projection_manifest_paths": [path.as_posix() for path in projection_paths],
            "receiver_proof_report_paths": [
                path.as_posix() for path in receiver_proof_paths
            ],
            "mlx_profile_paths": [
                _resolve(path, base=root).as_posix() for path in mlx_profile_paths
            ],
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "acquisition_report_path": (
                acquisition_path.as_posix() if acquisition_path.is_file() else None
            ),
            "bounded_runner_plan_path": (
                runner_plan_path.as_posix() if runner_plan_path.is_file() else None
            ),
            "selected_runner_rows": selected_runner_rows,
            "post_export_materializer_plan": post_export_materializer_plan,
            "post_export_materializer_execution": post_export_materializer_execution,
            "blockers": _dedupe(blockers),
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def _base_report(
    *,
    output_dir: Path,
    mode: str,
    hard_byte_ceilings: tuple[int, ...],
    repo_root: Path,
) -> dict[str, Any]:
    hinerv_modelsize_budget = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=CONTEST_PAIR_COUNT,
        per_ceiling_limit=6,
    )
    snerv_modelsize_budget = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=CONTEST_PAIR_COUNT,
        per_ceiling_limit=6,
    )
    return {
        "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "repo_root": repo_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "hard_byte_ceilings": [int(value) for value in hard_byte_ceilings],
        "execution_contract": {
            "schema": "compact_renderer_mlx_first_execution_contract.v1",
            "primary_accelerator": "MLX/Metal",
            "portable_reference": "NumPy-compatible serialized weights/latents/tokens",
            "pytorch_role": "control, calibration, and optional export bridge only",
            "score_authority": "contest CPU/CUDA exact eval only",
            "promotion_surface": "archive.zip bytes plus receiver proof plus exact gate",
        },
        "nerv_oss_flag_audit": official_nerv_oss_flag_audit(),
        "hinerv_modelsize_budget": hinerv_modelsize_budget,
        "snerv_modelsize_budget": snerv_modelsize_budget,
        "nerv_long_training_campaign_plan": build_nerv_long_training_campaign_plan(
            hinerv_modelsize_budget=hinerv_modelsize_budget,
            snerv_modelsize_budget=snerv_modelsize_budget,
            planner_row_queue_artifact_path=(
                output_dir / "compact_renderer_mlx_spine_runner_report.json"
            ),
        ),
        "nerv_stack_synergy_audit": build_nerv_stack_synergy_audit(
            repo_root=repo_root,
            hard_byte_ceilings=hard_byte_ceilings,
            num_pairs=CONTEST_PAIR_COUNT,
            memo_limit_per_stack=24,
            marker_limit_per_stack=40,
        ),
        "target_family_rows": _target_family_rows(),
        "compact_base_campaign_rows": _compact_base_campaign_rows(
            hard_byte_ceilings=hard_byte_ceilings
        ),
        "cleanup_contract": {
            "schema": "ssd_first_cleanup_contract.v1",
            "artifact_tier": "ssd_preferred",
            "large_artifacts_under_output_dir": True,
            "delete_policy": "success_scratch_only; evidence manifests retained",
        },
        **FALSE_AUTHORITY,
    }


def _substrate_score_aware_training_from_artifact(
    artifact_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Return substrate-supplied score-training metadata from an artifact.

    The shared MLX harness owns the canonical ``score_aware_training`` key for
    its own objective summary. If a substrate also supplies score-training
    metadata, the adapter preserves it under
    ``substrate_supplied_score_aware_training`` to avoid duplicate authority
    readers. Runner reports need the substrate slot first so knobs such as
    coder-aware QAT do not disappear after a real harness run.
    """

    metadata = artifact_dict.get("substrate_artifact_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    for key in ("substrate_supplied_score_aware_training", "score_aware_training"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _coder_qat_report_metadata(
    *,
    artifact_dict: Mapping[str, Any],
    enabled: bool,
    quant_bits: int,
    quant_residual_weight: float,
    magnitude_weight: float,
    delta_weight: float,
    c1a_entropy_weight: float = 0.0,
    c1a_sigma: float = DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
    c1a_sample_size: int = DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Return machine-readable coder-QAT metadata for runner reports."""

    score_training = _substrate_score_aware_training_from_artifact(artifact_dict)
    artifact_qat = score_training.get("coder_aware_qat")
    if isinstance(artifact_qat, Mapping):
        return dict(artifact_qat)
    return {
        "enabled": bool(enabled),
        "quant_bits": int(quant_bits),
        "quant_residual_weight": float(quant_residual_weight),
        "magnitude_weight": float(magnitude_weight),
        "delta_weight": float(delta_weight),
        "c1a_entropy_weight": float(c1a_entropy_weight),
        "c1a_sigma": float(c1a_sigma),
        "c1a_sample_size": int(c1a_sample_size),
        "c1a_source": (
            "PR95 cat_entropy_v2 soft categorical entropy adapted to selected "
            "decoder weights"
        ),
        "authority": "false_macos_mlx_research_signal",
    }


def _decoder_waterfill_candidate_aliases(value: Any) -> tuple[str, ...]:
    """Return candidate id aliases accepted by the campaign planner."""

    text = str(value or "").strip()
    if not text:
        return ()
    aliases = [text]
    has_double_colon = "::" in text
    if has_double_colon:
        parts = [part.strip() for part in text.split("::") if part.strip()]
        if len(parts) >= 3 and parts[0] in {"hi_nerv", "hinerv", "snerv"}:
            aliases.append(parts[1])
        elif parts:
            aliases.append(parts[-1])
    for marker in (
        ":hi_nerv_decoder_weight_waterfill:",
        ":hinerv_decoder_weight_waterfill:",
        ":snerv_decoder_weight_waterfill:",
    ):
        if marker in text:
            aliases.append(text.split(marker, 1)[0])
    if ":" in text and not has_double_colon:
        aliases.append(text.rsplit(":", 1)[-1])
        aliases.append(text.split(":", 1)[0])
    return tuple(str(alias) for alias in _dedupe([alias for alias in aliases if alias]))


def _candidate_waterfill_match_keys(candidate: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not candidate:
        return ()
    keys: list[str] = []
    for field in ("candidate_id", "row_id", "planner_row_id", "_modelsize_row_id"):
        keys.extend(_decoder_waterfill_candidate_aliases(candidate.get(field)))
    return tuple(str(key) for key in _dedupe(keys))


def _shape_tuple_or_none(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CompactRendererMlxSpineRunnerError("shape must be a sequence")
    out: list[int] = []
    for dim in value:
        try:
            parsed = int(dim)
        except (TypeError, ValueError) as exc:
            raise CompactRendererMlxSpineRunnerError(
                f"shape dim is not an integer: {dim!r}"
            ) from exc
        if parsed < 0:
            raise CompactRendererMlxSpineRunnerError(
                f"shape dim must be non-negative: {parsed}"
            )
        out.append(parsed)
    return tuple(out)


def _row_declared_shape(row: Mapping[str, Any]) -> tuple[int, ...] | None:
    for field in ("shape", "tensor_shape", "expected_shape", "state_shape"):
        if field in row:
            return _shape_tuple_or_none(row.get(field))
    return None


def _row_declared_numel(row: Mapping[str, Any]) -> int | None:
    for field in ("numel", "tensor_numel", "group_numel", "expected_numel"):
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise CompactRendererMlxSpineRunnerError(
                f"numel field {field} is not an integer: {value!r}"
            ) from exc
        if parsed < 0:
            raise CompactRendererMlxSpineRunnerError(
                f"numel field {field} must be non-negative: {parsed}"
            )
        return parsed
    return None


def _hi_nerv_expected_decoder_state_shapes(
    *,
    num_pairs: int,
    latent_dim: int,
    embed_dim: int,
    decoder_channel: int,
    use_hierarchical_feature_grid: bool,
    use_convnext_blocks: bool,
    local_grid_levels: int,
    local_grid_channels: int,
    convnext_mlp_ratio: int,
    convnext_kernel_size: int,
    mid_injection_block_index: int,
    fine_injection_block_index: int,
) -> dict[str, tuple[int, ...]]:
    from tac.substrates.hi_nerv.architecture import (
        HinervConfig,
        expected_decoder_state_shapes,
    )

    cfg = HinervConfig(
        latent_dim_coarse=max(1, int(latent_dim) // 2),
        latent_dim_mid=max(1, int(latent_dim)),
        latent_dim_fine=max(1, int(latent_dim) * 2),
        embed_dim=max(1, int(embed_dim)),
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=tuple([max(1, int(decoder_channel))] * 7),
        sin_frequency=30.0,
        num_upsample_blocks=7,
        mid_injection_block_index=int(mid_injection_block_index),
        fine_injection_block_index=int(fine_injection_block_index),
        num_pairs=int(num_pairs),
        output_height=384,
        output_width=512,
        use_hierarchical_feature_grid=bool(use_hierarchical_feature_grid),
        use_convnext_blocks=bool(use_convnext_blocks),
        local_grid_levels=int(local_grid_levels),
        local_grid_channels=int(local_grid_channels),
        convnext_mlp_ratio=int(convnext_mlp_ratio),
        convnext_kernel_size=int(convnext_kernel_size),
    )
    return expected_decoder_state_shapes(cfg)


def _validate_hi_nerv_decoder_weight_waterfill_plan_attachment(
    *,
    plan: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    num_pairs: int,
    latent_dim: int,
    embed_dim: int,
    decoder_channel: int,
    use_hierarchical_feature_grid: bool,
    use_convnext_blocks: bool,
    local_grid_levels: int,
    local_grid_channels: int,
    convnext_mlp_ratio: int,
    convnext_kernel_size: int,
    mid_injection_block_index: int,
    fine_injection_block_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate a waterfill plan against the exact HiNeRV launch state."""

    out = {
        **dict(metadata),
        "attached": False,
        "active": False,
        "validated": False,
        "validation_schema": "compact_hi_nerv_decoder_weight_waterfill_validation.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    blockers: list[str] = list(out.get("blockers") or [])
    if plan is None:
        out["blockers"] = _dedupe(blockers)
        return None, out
    if plan.get("schema") != NERV_DECODER_WEIGHT_WATERFILL_SCHEMA:
        blockers.append("decoder_weight_waterfill_plan_schema_mismatch")
    family = plan.get("family")
    if family not in (None, "", "hi_nerv"):
        blockers.append(f"decoder_weight_waterfill_family_mismatch:{family}")
    rows = plan.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        blockers.append("decoder_weight_waterfill_rows_not_list")
        rows = []
    elif not rows:
        blockers.append("decoder_weight_waterfill_rows_empty")

    candidate_keys = _candidate_waterfill_match_keys(candidate)
    plan_candidate_id = str(plan.get("candidate_id") or "").strip()
    plan_candidate_keys = _decoder_waterfill_candidate_aliases(plan_candidate_id)
    out["candidate_match"] = {
        "schema": "compact_hi_nerv_decoder_weight_waterfill_candidate_match.v1",
        "plan_candidate_id": plan_candidate_id or None,
        "plan_candidate_keys": list(plan_candidate_keys),
        "launch_candidate_keys": list(candidate_keys),
        "matched": bool(set(plan_candidate_keys) & set(candidate_keys)),
    }
    if not candidate_keys:
        blockers.append("decoder_weight_waterfill_launch_candidate_missing")
    if not plan_candidate_keys:
        blockers.append("decoder_weight_waterfill_plan_candidate_id_missing")
    elif not set(plan_candidate_keys) & set(candidate_keys):
        blockers.append(
            f"decoder_weight_waterfill_candidate_id_mismatch:{plan_candidate_id}"
        )

    expected_shapes = _hi_nerv_expected_decoder_state_shapes(
        num_pairs=num_pairs,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        decoder_channel=decoder_channel,
        use_hierarchical_feature_grid=use_hierarchical_feature_grid,
        use_convnext_blocks=use_convnext_blocks,
        local_grid_levels=local_grid_levels,
        local_grid_channels=local_grid_channels,
        convnext_mlp_ratio=convnext_mlp_ratio,
        convnext_kernel_size=convnext_kernel_size,
        mid_injection_block_index=mid_injection_block_index,
        fine_injection_block_index=fine_injection_block_index,
    )
    validated_rows: list[dict[str, Any]] = []
    for idx, row_obj in enumerate(rows):
        if not isinstance(row_obj, Mapping):
            blockers.append(f"decoder_weight_waterfill_row_not_mapping:{idx}")
            continue
        group_name = row_obj.get("group_name")
        if not isinstance(group_name, str) or not group_name:
            blockers.append(f"decoder_weight_waterfill_row_missing_group_name:{idx}")
            continue
        expected_shape = expected_shapes.get(group_name)
        if expected_shape is None:
            blockers.append(f"decoder_weight_waterfill_group_missing:{group_name}")
            continue
        try:
            declared_shape = _row_declared_shape(row_obj)
            declared_numel = _row_declared_numel(row_obj)
        except CompactRendererMlxSpineRunnerError as exc:
            blockers.append(
                f"decoder_weight_waterfill_row_metadata_invalid:{group_name}:{exc}"
            )
            continue
        expected_numel = int(math.prod(expected_shape))
        row_validation = {
            "group_name": group_name,
            "expected_shape": [int(dim) for dim in expected_shape],
            "expected_numel": expected_numel,
            "shape_checked": declared_shape is not None,
            "numel_checked": declared_numel is not None,
        }
        if declared_shape is not None and declared_shape != expected_shape:
            blockers.append(f"decoder_weight_waterfill_shape_mismatch:{group_name}")
            row_validation["declared_shape"] = [int(dim) for dim in declared_shape]
        if declared_numel is not None and declared_numel != expected_numel:
            blockers.append(f"decoder_weight_waterfill_numel_mismatch:{group_name}")
            row_validation["declared_numel"] = int(declared_numel)
        validated_rows.append(row_validation)

    try:
        bits_by_name = _decoder_weight_waterfill_fake_quant_bits_by_name(plan)
    except CompactRendererMlxSpineRunnerError as exc:
        blockers.append(f"decoder_weight_waterfill_fake_quant_bits_invalid:{exc}")
        bits_by_name = {}
    out.update(
        {
            "source_schema": plan.get("schema"),
            "family": plan.get("family"),
            "candidate_id": plan.get("candidate_id"),
            "group_count": plan.get("group_count"),
            "row_count": len(rows),
            "launch_state_group_count": len(expected_shapes),
            "validated_row_count": len(validated_rows),
            "validated_rows": validated_rows,
            "per_tensor_fake_quant_group_count": len(bits_by_name),
            "per_tensor_fake_quant_bits_by_name": dict(sorted(bits_by_name.items())),
            "blockers": _dedupe(blockers),
        }
    )
    if not out["blockers"]:
        out["attached"] = True
        out["active"] = True
        out["validated"] = True
        return dict(plan), out
    return None, out


def _decoder_weight_waterfill_fake_quant_bits_by_name(
    decoder_weight_waterfill_plan: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Extract train-time per-tensor fake-quant bits from a waterfill plan."""

    if decoder_weight_waterfill_plan is None:
        return {}
    if decoder_weight_waterfill_plan.get("schema") != NERV_DECODER_WEIGHT_WATERFILL_SCHEMA:
        raise CompactRendererMlxSpineRunnerError(
            "decoder_weight_waterfill_plan must have schema "
            f"{NERV_DECODER_WEIGHT_WATERFILL_SCHEMA}"
        )
    bits_by_name: dict[str, int] = {}
    for idx, row in enumerate(decoder_weight_waterfill_plan.get("rows") or []):
        if not isinstance(row, Mapping):
            raise CompactRendererMlxSpineRunnerError(
                f"decoder_weight_waterfill_plan row {idx} is not a mapping"
            )
        group_name = row.get("group_name")
        if not isinstance(group_name, str) or not group_name:
            raise CompactRendererMlxSpineRunnerError(
                f"decoder_weight_waterfill_plan row {idx} missing group_name"
            )
        try:
            selected_bits = int(row["selected_bits"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompactRendererMlxSpineRunnerError(
                f"decoder_weight_waterfill_plan row {idx} missing selected_bits"
            ) from exc
        if selected_bits not in set(NERV_DECODER_WEIGHT_WATERFILL_ACTION_BITS):
            raise CompactRendererMlxSpineRunnerError(
                "decoder_weight_waterfill_plan selected_bits must be one of "
                f"{list(NERV_DECODER_WEIGHT_WATERFILL_ACTION_BITS)}; "
                f"got {selected_bits} for {group_name}"
            )
        bits_by_name[group_name] = selected_bits
    return bits_by_name


def _hi_nerv_eval_roundtrip_ste_metadata() -> dict[str, Any]:
    """Return the canonical HiNeRV PR95 eval-roundtrip training metadata."""

    return {
        "schema": "mlx_score_aware_eval_roundtrip_ste.v1",
        "enabled": True,
        "surface": "pr95_bicubic_camera_bilinear_scorer_uint8_ste",
        "camera_hw": [874, 1164],
        "applied_before": [
            "reconstruction_loss",
            "segnet_student_head_loss",
            "posenet_student_head_loss",
        ],
        "authority": "macos_mlx_research_signal_false_authority",
    }


def _hi_nerv_pose_preprocess_metadata() -> dict[str, Any]:
    return {
        "schema": "mlx_score_aware_pose_student_input_preprocess.v1",
        "mode": "pr95_yuv6",
        "differentiable": True,
        "source": (
            "tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx"
        ),
        "consumed_by": "learnable_pose_student_head",
        "authority": "macos_mlx_research_signal_false_authority",
    }


def _eval_roundtrip_ste_report_metadata(
    artifact_dict: Mapping[str, Any],
) -> dict[str, Any]:
    score_training = _substrate_score_aware_training_from_artifact(artifact_dict)
    value = score_training.get("eval_roundtrip_ste")
    if isinstance(value, Mapping):
        return dict(value)
    harness_training = (
        artifact_dict.get("substrate_artifact_metadata", {})
        if isinstance(artifact_dict.get("substrate_artifact_metadata"), Mapping)
        else {}
    )
    value = harness_training.get("score_aware_training", {}).get(
        "eval_roundtrip_ste"
    ) if isinstance(harness_training.get("score_aware_training"), Mapping) else None
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "schema": "mlx_score_aware_eval_roundtrip_ste.v1",
        "enabled": False,
        "authority": "macos_mlx_research_signal_false_authority",
    }


def _pose_student_input_preprocess_report_metadata(
    artifact_dict: Mapping[str, Any],
) -> dict[str, Any]:
    score_training = _substrate_score_aware_training_from_artifact(artifact_dict)
    value = score_training.get("pose_student_input_preprocess")
    if isinstance(value, Mapping):
        return dict(value)
    harness_training = (
        artifact_dict.get("substrate_artifact_metadata", {})
        if isinstance(artifact_dict.get("substrate_artifact_metadata"), Mapping)
        else {}
    )
    value = harness_training.get("score_aware_training", {}).get(
        "pose_student_input_preprocess"
    ) if isinstance(harness_training.get("score_aware_training"), Mapping) else None
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "schema": "mlx_score_aware_pose_student_input_preprocess.v1",
        "mode": "rgb",
        "differentiable": True,
        "authority": "macos_mlx_research_signal_false_authority",
    }


def _pose_instability_monitor_report_metadata(
    artifact_dict: Mapping[str, Any],
) -> dict[str, Any]:
    score_training = _substrate_score_aware_training_from_artifact(artifact_dict)
    value = score_training.get("pose_instability_monitor")
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "schema": "compact_pose_instability_epoch_monitor.v1",
        "enabled": False,
        "reason": "monitor_metadata_missing_from_training_artifact",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _weight_stats(arr: Any) -> dict[str, Any]:
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    return {
        "shape": [int(v) for v in a.shape],
        "dtype": "float32",
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "nonzero_fraction": float(np.count_nonzero(a) / max(int(a.size), 1)),
    }


def _validate_recon_pixel_weight_array(
    arr: Any,
    *,
    expected_hw: tuple[int, int] = (384, 512),
    expected_pairs: int | None = None,
) -> Any:
    import numpy as np

    weight = np.asarray(arr, dtype=np.float32)
    if weight.ndim == 2:
        h, w = weight.shape
        channels = 1
    elif weight.ndim == 3:
        h, w, channels = weight.shape
    elif weight.ndim == 4:
        leading, h, w, channels = weight.shape
        if int(leading) != 1 and (
            expected_pairs is None or int(leading) != int(expected_pairs)
        ):
            raise CompactRendererMlxSpineRunnerError(
                "recon pixel weight with 4 dims must have leading dimension "
                f"1 or expected pair count {expected_pairs}; got {int(leading)}"
            )
    elif weight.ndim == 5:
        pairs, frames, h, w, channels = weight.shape
        if int(frames) != 2:
            raise CompactRendererMlxSpineRunnerError(
                "recon pixel weight with 5 dims must have frame dimension 2"
            )
        if expected_pairs is not None and int(pairs) != int(expected_pairs):
            raise CompactRendererMlxSpineRunnerError(
                "recon pixel weight pair count must match num_pairs "
                f"({int(pairs)} vs {int(expected_pairs)})"
            )
    else:
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight must be shaped (H,W), (H,W,1/3), "
            "(1|N,H,W,1/3), or (N,2,H,W,1/3)"
        )
    if (int(h), int(w)) != tuple(int(v) for v in expected_hw):
        raise CompactRendererMlxSpineRunnerError(
            f"recon pixel weight spatial shape {(int(h), int(w))} != {expected_hw}"
        )
    if int(channels) not in (1, 3):
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight channel count must be 1 or 3"
        )
    if not bool(np.all(np.isfinite(weight))):
        raise CompactRendererMlxSpineRunnerError("recon pixel weight must be finite")
    if float(np.min(weight)) < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight must be non-negative"
        )
    if float(np.mean(weight)) <= 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight must have positive total mass"
        )
    return weight


def _load_recon_pixel_weight(
    path: str | Path,
    *,
    base: Path,
    expected_hw: tuple[int, int] = (384, 512),
    expected_pairs: int | None = None,
    normalize: str = "mean",
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if normalize not in ("mean", "none"):
        raise CompactRendererMlxSpineRunnerError(
            "recon_pixel_weight_normalize must be 'mean' or 'none'"
        )
    resolved = _resolve_existing(path, base=base)
    key: str | None = None
    if resolved.suffix == ".npz":
        with np.load(resolved) as data:
            keys = sorted(str(item) for item in data.files)
            if not keys:
                raise CompactRendererMlxSpineRunnerError(
                    f"recon pixel weight npz is empty: {resolved}"
                )
            key = "weight" if "weight" in data.files else keys[0]
            weight = np.asarray(data[key], dtype=np.float32)
    else:
        weight = np.asarray(np.load(resolved), dtype=np.float32)
    source_sha256 = _sha256_file(resolved)
    weight = _validate_recon_pixel_weight_array(
        weight,
        expected_hw=expected_hw,
        expected_pairs=expected_pairs,
    )
    producer_manifest = _recon_pixel_weight_producer_manifest(
        resolved,
        expected_weight_sha256=source_sha256,
    )
    metadata = {
        "schema": "compact_recon_pixel_weight.v1",
        "enabled": True,
        "source_kind": "file",
        "path": resolved.as_posix(),
        "sha256": source_sha256,
        "npz_key": key,
        "normalize": normalize,
        "scorer_terms": {
            "p18_segnet": "caller_supplied",
            "p19_posenet": "caller_supplied",
        },
        "stats": _weight_stats(weight),
        "producer_manifest": producer_manifest,
        "authority": "false_macos_mlx_research_signal",
    }
    if expected_pairs is not None:
        metadata["expected_pairs"] = int(expected_pairs)
    return weight, metadata


def _discover_joint_recon_pixel_weight_path(
    *,
    repo_root: str | Path,
    num_pairs: int,
) -> tuple[Path, dict[str, Any]]:
    """Find the latest verified joint P18/P19 recon-weight artifact.

    Discovery is intentionally strict. It only returns artifacts generated by
    the finite-gradient joint-scorer surface producer, for the exact requested
    pair count, with the manifest's SHA matching the NPZ bytes. Anything else is
    ignored here and will remain a manual ``--recon-pixel-weight-path`` choice.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    search_roots = [
        root / "experiments" / "results",
        *(ssd / "experiments" / "results" for ssd in DEFAULT_SSD_ROOTS),
    ]
    candidates: list[tuple[str, Path, Path, str]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for search_root in search_roots:
        resolved_root = search_root.expanduser().resolve(strict=False)
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        if not resolved_root.is_dir():
            continue
        for manifest_path in sorted(
            resolved_root.rglob("joint_p18_p19_recon_pixel_weight_manifest.json")
        ):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("manifest_not_object")
                config = manifest.get("config")
                metadata = manifest.get("metadata")
                if not isinstance(config, dict) or not isinstance(metadata, dict):
                    raise ValueError("manifest_missing_config_or_metadata")
                if int(config.get("num_pairs", -1)) != int(num_pairs):
                    raise ValueError("num_pairs_mismatch")
                health = metadata.get("gradient_health")
                if not isinstance(health, dict):
                    raise ValueError("gradient_health_missing")
                blockers = list(metadata.get("blockers") or [])
                if health.get("status") != "pass_finite":
                    raise ValueError("gradient_health_not_pass_finite")
                if not bool(metadata.get("training_consumption_recommended")):
                    raise ValueError("training_consumption_not_recommended")
                if blockers:
                    raise ValueError("manifest_has_blockers")
                weight_path_value = manifest.get("weight_path")
                if weight_path_value is None:
                    raise ValueError("weight_path_missing")
                weight_path = _resolve_existing(
                    weight_path_value,
                    base=manifest_path.parent,
                )
                expected_sha = str(manifest.get("weight_sha256") or "")
                actual_sha = _sha256_file(weight_path)
                if expected_sha and expected_sha != actual_sha:
                    raise ValueError("weight_sha256_mismatch")
                candidates.append(
                    (
                        manifest_path.parent.as_posix(),
                        manifest_path,
                        weight_path,
                        actual_sha,
                    )
                )
            except Exception as exc:
                rejected.append(
                    {
                        "manifest_path": manifest_path.as_posix(),
                        "reason": f"{type(exc).__name__}:{exc!s}",
                    }
                )
    if not candidates:
        raise CompactRendererMlxSpineRunnerError(
            "no verified joint P18/P19 recon_pixel_weight artifact found for "
            f"num_pairs={int(num_pairs)}; build one with "
            "tools/build_joint_recon_pixel_weight_surface.py or pass "
            "--recon-pixel-weight-path explicitly"
        )
    _sort_key, manifest_path, weight_path, weight_sha = sorted(candidates)[-1]
    discovery = {
        "schema": "compact_auto_joint_recon_pixel_weight_discovery.v1",
        "status": "selected_verified_joint_p18_p19_weight",
        "num_pairs": int(num_pairs),
        "search_roots": [
            path.expanduser().resolve(strict=False).as_posix()
            for path in search_roots
        ],
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "selected_manifest_path": manifest_path.as_posix(),
        "selected_manifest_sha256": _sha256_file(manifest_path),
        "selected_weight_path": weight_path.as_posix(),
        "selected_weight_sha256": weight_sha,
        "authority": "false_macos_mlx_research_signal",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if rejected:
        discovery["rejected_manifests"] = rejected[:16]
    return weight_path, discovery


def _recon_pixel_weight_producer_manifest(
    weight_path: Path,
    *,
    expected_weight_sha256: str,
) -> dict[str, Any]:
    """Return fail-closed producer-manifest custody for a recon weight file."""

    manifest_path = weight_path.with_name(
        "joint_p18_p19_recon_pixel_weight_manifest.json"
    )
    if not manifest_path.is_file():
        return {
            "schema": "compact_recon_pixel_weight_producer_manifest.v1",
            "status": "not_found_unverified_manual_or_legacy_weight",
            "path": manifest_path.as_posix(),
            "consumption_certified": False,
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactRendererMlxSpineRunnerError(
            f"recon pixel weight producer manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise CompactRendererMlxSpineRunnerError(
            f"recon pixel weight producer manifest must be an object: {manifest_path}"
        )

    manifest_weight_path = manifest.get("weight_path")
    if manifest_weight_path is not None:
        manifest_resolved = _resolve_existing(manifest_weight_path, base=weight_path.parent)
        if manifest_resolved != weight_path:
            raise CompactRendererMlxSpineRunnerError(
                "recon pixel weight producer manifest points at a different "
                f"weight file: {manifest_resolved} != {weight_path}"
            )
    manifest_weight_sha = manifest.get("weight_sha256")
    if manifest_weight_sha is not None and str(manifest_weight_sha) != expected_weight_sha256:
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight producer manifest SHA does not match loaded "
            f"weight file: {manifest_weight_sha} != {expected_weight_sha256}"
        )

    producer_metadata = manifest.get("metadata")
    if not isinstance(producer_metadata, dict):
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight producer manifest is missing metadata object"
        )
    gradient_health = producer_metadata.get("gradient_health")
    if not isinstance(gradient_health, dict):
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight producer manifest is missing gradient_health; "
            "regenerate the surface with the finite-gradient producer"
        )
    blockers = list(producer_metadata.get("blockers") or [])
    consumption_recommended = bool(
        producer_metadata.get("training_consumption_recommended", False)
    )
    if gradient_health.get("status") != "pass_finite":
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight producer manifest did not pass finite-gradient "
            f"health: {gradient_health.get('status')}"
        )
    if not consumption_recommended or blockers:
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight producer manifest is not recommended for "
            f"training consumption; blockers={blockers}"
        )

    return {
        "schema": "compact_recon_pixel_weight_producer_manifest.v1",
        "status": "verified_finite_gradient_manifest",
        "path": manifest_path.as_posix(),
        "sha256": _sha256_file(manifest_path),
        "producer_schema": manifest.get("schema"),
        "producer_metadata_schema": producer_metadata.get("schema"),
        "weight_path": Path(str(manifest.get("weight_path", weight_path))).as_posix(),
        "weight_sha256": expected_weight_sha256,
        "gradient_health": gradient_health,
        "blockers": blockers,
        "training_consumption_recommended": consumption_recommended,
        "consumption_certified": True,
    }


def _segnet_boundary_recon_pixel_weight(
    scorer_teacher: Any,
    *,
    tau: float,
    normalize: str = "mean",
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if tau <= 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "recon pixel weight tau must be > 0"
        )
    if normalize not in ("mean", "none"):
        raise CompactRendererMlxSpineRunnerError(
            "recon_pixel_weight_normalize must be 'mean' or 'none'"
        )
    logits = np.asarray(scorer_teacher.teacher_logits_thwk, dtype=np.float32)
    if logits.ndim != 4 or int(logits.shape[-1]) < 2:
        raise CompactRendererMlxSpineRunnerError(
            "SegNet boundary recon weight requires teacher logits shaped (T,H,W,K>=2)"
        )
    sorted_logits = np.sort(logits, axis=-1)
    margin = sorted_logits[..., -1] - sorted_logits[..., -2]
    saliency = np.exp(-np.maximum(margin, 0.0) / float(tau)).astype(np.float32)
    weight = np.mean(saliency, axis=0).astype(np.float32)
    weight = _validate_recon_pixel_weight_array(weight, expected_hw=(384, 512))
    metadata = {
        "schema": "compact_recon_pixel_weight.v1",
        "enabled": True,
        "source_kind": "auto_segnet_top2_boundary_margin",
        "tau": float(tau),
        "normalize": normalize,
        "scorer_terms": {
            "p18_segnet": "top2_margin_exp_boundary_saliency_from_real_teacher",
            "p19_posenet": "not_included_use_recon_pixel_weight_path_for_joint_map",
        },
        "stats": _weight_stats(weight),
        "authority": "false_macos_mlx_research_signal",
    }
    return weight, metadata


def _disabled_recon_pixel_weight_metadata() -> dict[str, Any]:
    return {
        "schema": "compact_recon_pixel_weight.v1",
        "enabled": False,
        "authority": "false_macos_mlx_research_signal",
    }


def _recon_pixel_weight_report_metadata(
    artifact_dict: Mapping[str, Any],
) -> dict[str, Any]:
    score_training = _substrate_score_aware_training_from_artifact(artifact_dict)
    value = score_training.get("recon_pixel_weight")
    if isinstance(value, Mapping):
        return dict(value)
    return _disabled_recon_pixel_weight_metadata()


def _target_family_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        backend = COMPACT_FAMILY_BACKENDS[family]
        rows.append(
            {
                "schema": "compact_renderer_mlx_target_family_row.v1",
                "family": family,
                "canonical_family": backend["canonical_family"],
                "status": backend["backend_status"],
                "trainer_kind": backend["trainer_kind"],
                "trainer_entrypoint": backend["trainer_entrypoint"],
                "archive_exporter": backend["archive_exporter"],
                "receiver_proof": backend["receiver_proof"],
                "next_action": backend["next_action"],
                "execution_scope": backend["execution_scope"],
                "required_inputs": [
                    "trained_decoder_weights_or_program",
                    "trained_latents_or_tokens",
                    "archive_charged_runtime_config",
                ],
                "automatic_outputs": [
                    "hprc_representation_spine_projection",
                    "spine_acquisition_row",
                    "bounded_runner_row",
                    "receiver_proof_gate",
                    "mlx_component_neutralization_profile_when_profiler_exists",
                    "exact_axis_blocker_or_dispatch_packet",
                ],
                "section_value_profiler": backend.get("section_value_profiler"),
                "stack_role": backend.get("stack_role", "primary_or_control_candidate"),
                "carrier_priority": backend.get("carrier_priority", 0),
                "enhancer_priority": backend.get("enhancer_priority", 0),
                "architecture_priors": backend.get("architecture_priors", []),
                "allowed_enhancers": backend.get("allowed_enhancers", []),
                "rate_axis_evidence": backend.get("rate_axis_evidence"),
                "distortion_fit_blocker": backend.get("distortion_fit_blocker"),
                "score_aware_carrier_training_plan": (
                    _score_aware_carrier_training_plan(family, backend)
                ),
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _compact_base_campaign_rows(
    *,
    hard_byte_ceilings: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        backend = COMPACT_FAMILY_BACKENDS[family]
        for ceiling in hard_byte_ceilings:
            status = str(backend["backend_status"])
            if status in {
                "executable_mlx_backend_available",
                "executable_mlx_archive_export_control_arm",
                "executable_via_pact_nerv_vq_adapter",
            }:
                route_status = "queued_for_mlx_training_archive_export_receiver_proof"
            elif status == (
                "mlx_archive_export_adapter_available_"
                "distortion_fit_actuator_pending"
            ):
                route_status = (
                    "queued_for_mlx_archive_adapter_smoke_"
                    "scoreaware_training_pending"
                )
            elif status == "checkpoint_adapter_available":
                route_status = "queued_for_checkpoint_import_or_long_continuation"
            elif status in {
                "archive_exporter_available_trainer_actuator_pending",
                "archive_projection_available_trainer_actuator_pending",
                "rate_axis_structural_win_archive_projection_available_"
                "distortion_fit_actuator_pending",
            }:
                route_status = "trainer_actuator_migration_required"
            else:
                route_status = "migration_required_before_runner_execution"
            rows.append(
                {
                    "schema": "compact_renderer_mlx_campaign_row.v1",
                    "family": family,
                    "canonical_family": backend["canonical_family"],
                    "hard_byte_ceiling": int(ceiling),
                    "backend_status": backend["backend_status"],
                    "route_status": route_status,
                    "trainer_kind": backend["trainer_kind"],
                    "trainer_entrypoint": backend["trainer_entrypoint"],
                    "archive_exporter": backend["archive_exporter"],
                    "receiver_proof": backend["receiver_proof"],
                    "section_value_profiler": backend.get("section_value_profiler"),
                    "stack_role": backend.get("stack_role", "primary_or_control_candidate"),
                    "carrier_priority": backend.get("carrier_priority", 0),
                    "enhancer_priority": backend.get("enhancer_priority", 0),
                    "architecture_priors": backend.get("architecture_priors", []),
                    "allowed_enhancers": backend.get("allowed_enhancers", []),
                    "rate_axis_evidence": backend.get("rate_axis_evidence"),
                    "distortion_fit_blocker": backend.get("distortion_fit_blocker"),
                    "modelsize_budget": (
                        build_hinerv_modelsize_budget_report(
                            hard_byte_ceilings=(int(ceiling),),
                            num_pairs=CONTEST_PAIR_COUNT,
                            per_ceiling_limit=4,
                        )
                        if family == "hi_nerv"
                        else (
                            build_snerv_modelsize_budget_report(
                                hard_byte_ceilings=(int(ceiling),),
                                num_pairs=CONTEST_PAIR_COUNT,
                                per_ceiling_limit=4,
                            )
                            if family == "snerv"
                            else None
                        )
                    ),
                    "score_aware_carrier_training_plan": (
                        _score_aware_carrier_training_plan(family, backend)
                    ),
                    "execution_scope": backend["execution_scope"],
                    "byte_policy": (
                        "train/export only charged weights, latents, selectors, "
                        "codebooks, and residual tokens; no hidden sidecars"
                    ),
                    "value_policy": (
                        "full-video MLX scorer replay prices section value; "
                        "residuals admitted only when delta_nonrate + rate_cost < 0"
                    ),
                    "exact_gate_policy": (
                        "receiver-proven byte-closed archive must be plausible "
                        "under the hard ceiling before contest CPU/CUDA spend"
                    ),
                    "next_action": backend["next_action"],
                    **FALSE_AUTHORITY,
                }
            )
    return rows


def _score_aware_carrier_training_plan(
    family: str,
    backend: dict[str, Any],
) -> dict[str, Any]:
    """Return the fail-closed score-aware route consumed by compact queues."""

    evidence = dict(_SCORE_AWARE_STACK_READINESS_FALSE)
    evidence.update(backend.get("score_aware_training_evidence") or {})
    evidence.update(backend.get("score_aware_training_readiness") or {})
    return build_score_aware_carrier_training_plan(
        evidence,
        carrier_id=family,
        baseline_id="pr95_hnerv",
    )


def _backend_with_score_aware_evidence(
    family: str,
    evidence_updates: Mapping[str, Any],
) -> dict[str, Any]:
    backend = dict(COMPACT_FAMILY_BACKENDS.get(family) or {})
    existing = dict(backend.get("score_aware_training_evidence") or {})
    existing.update(dict(evidence_updates))
    backend["score_aware_training_evidence"] = existing
    return backend


def _load_compact_modelsize_budget_rows(
    paths: tuple[str | Path, ...],
    *,
    base: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        resolved = _resolve(path, base=base)
        record: dict[str, Any] = {
            "path": resolved.as_posix(),
            "exists": resolved.is_file(),
            "rows_seen": 0,
            "rows_added": 0,
            "rows_rejected": 0,
            "blockers": [],
        }
        if not resolved.is_file():
            record["blockers"].append("modelsize_budget_json_missing")
            sources.append(record)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["blockers"].append("modelsize_budget_json_unreadable")
            record["error"] = repr(exc)
            sources.append(record)
            continue
        raw_refusal = _raw_modelsize_budget_payload_refusal(payload)
        if raw_refusal is not None:
            selected = raw_refusal["selected_candidates"]
            record.update(
                {
                    "source_schema": raw_refusal["source_schema"],
                    "authority": (
                        "planning_artifact_only_not_receiver_closed_ladder_evidence"
                    ),
                    "rows_seen": len(selected),
                    "rows_added": 0,
                    "rows_rejected": len(selected),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
            record["blockers"].extend(raw_refusal["blockers"])
            record["rejected_rows"] = [
                {
                    "row_index": index,
                    "row_id": row.get("candidate_id") or row.get("row_id"),
                    "blockers": raw_refusal["row_blockers"],
                }
                for index, row in enumerate(selected)
                if isinstance(row, Mapping)
            ]
            sources.append(record)
            continue
        extracted = _modelsize_budget_rows_from_payload(
            payload,
            source_path=resolved,
        )
        record["rows_seen"] = len(extracted)
        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        for index, row in enumerate(extracted):
            row_blockers = _compact_modelsize_budget_row_blockers(row)
            if row_blockers:
                rejected_rows.append(
                    {
                        "row_index": index,
                        "row_id": row.get("row_id"),
                        "blockers": row_blockers,
                    }
                )
                continue
            accepted_rows.append(row)
        rows.extend(accepted_rows)
        record["rows_added"] = len(accepted_rows)
        record["rows_rejected"] = len(rejected_rows)
        if rejected_rows:
            record["rejected_rows"] = rejected_rows
            record["blockers"].append("modelsize_budget_json_rows_rejected")
        if not accepted_rows:
            record["blockers"].append("modelsize_budget_json_rows_missing")
        sources.append(record)
    return rows, sources


def _raw_modelsize_budget_payload_refusal(
    payload: Any,
) -> dict[str, Any] | None:
    """Classify raw model-size planning reports as non-ladder evidence."""

    if not isinstance(payload, Mapping):
        return None
    source_schema = str(payload.get("schema") or "")
    if source_schema not in RAW_NERV_MODELSIZE_BUDGET_SCHEMAS:
        return None
    selected = payload.get("selected_candidates")
    if not isinstance(selected, list):
        selected = []
    selected_by_ceiling = payload.get("selected_candidates_by_ceiling")
    has_selected_surface = bool(selected) or isinstance(selected_by_ceiling, Mapping)
    if not has_selected_surface:
        return None
    row_blockers = [
        "selected_candidates_are_planning_rows_not_receiver_closed_ladder",
        "receiver_closed_byte_proof_missing",
        "measured_receiver_archive_bytes_missing",
    ]
    return {
        "source_schema": source_schema,
        "selected_candidates": selected,
        "row_blockers": row_blockers,
        "blockers": [
            "raw_nerv_modelsize_budget_artifact_not_receiver_closed_ladder",
            "receiver_closed_modelsize_ladder_schema_required",
            "selected_candidates_are_planning_rows_not_receiver_closed_ladder",
            "modelsize_budget_json_rows_missing",
        ],
    }


def _modelsize_budget_rows_from_payload(
    payload: Any,
    *,
    source_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_schema = None
    if isinstance(payload, list):
        source_rows = payload
    elif isinstance(payload, Mapping):
        source_schema = str(payload.get("schema") or "")
        source_rows = payload.get("modelsize_budget_rows")
        if not isinstance(source_rows, list):
            source_rows = _modelsize_budget_rows_from_plan(
                payload.get("modelsize_budget_plan")
            )
        if not source_rows and isinstance(payload.get("rows"), list):
            source_rows = payload["rows"]
    else:
        source_rows = []
    if not isinstance(source_rows, list):
        return rows
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            continue
        normalized = dict(row)
        normalized.setdefault("source_modelsize_budget_json", source_path.as_posix())
        if source_schema:
            normalized.setdefault("source_modelsize_budget_schema", source_schema)
        normalized.setdefault("source_modelsize_budget_row_index", index)
        rows.append(normalized)
    return rows


def _compact_modelsize_budget_row_blockers(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    source_schema = str(
        row.get("source_modelsize_budget_schema") or row.get("schema") or ""
    )
    if source_schema != "nerv_receiver_closed_modelsize_ladder.v1":
        blockers.append("receiver_closed_modelsize_ladder_schema_required")
    archive_bytes = _compact_positive_int_from_keys(
        row,
        (
            "archive_bytes",
            "archive_zip_bytes",
            "archive_bytes_total",
            "measured_archive_bytes",
            "archive_size_bytes",
        ),
    )
    if archive_bytes is None:
        blockers.append("modelsize_budget_row_missing_measured_receiver_archive_bytes")
    if not (
        row.get("receiver_closed") is True
        and (
            row.get("receiver_proof_passed") is True
            or row.get("receiver_archive_replay_verified") is True
        )
    ):
        blockers.append("receiver_closed_byte_proof_missing")
    blockers.extend(_compact_receiver_closed_identity_blockers(row))
    if not _compact_modelsize_row_has_source_bound_capacity(row):
        blockers.append("source_bound_modelsize_or_fc_dim_missing")
    if not _compact_modelsize_row_has_nonrate_signal(row):
        blockers.append("modelsize_budget_row_missing_nonrate_score")
    return blockers


def _write_hi_nerv_trained_archive_byte_oracle(
    *,
    output_dir: Path,
    artifact_dict: Mapping[str, Any],
    modelsize_candidate: Mapping[str, Any] | None,
    num_pairs: int,
    receiver_proof_path: Path | None,
    local_cpu_replay_summary: Mapping[str, Any] | None,
    mlx_prefilter_coverage: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Write planner-facing measured archive bytes for a trained HiNeRV row."""

    candidate = dict(modelsize_candidate or {})
    archive_path_raw = artifact_dict.get("archive_path")
    archive_path = Path(str(archive_path_raw)) if archive_path_raw else None
    archive_bytes = _compact_positive_int_from_keys(
        artifact_dict,
        ("archive_bytes", "archive_zip_bytes", "measured_archive_bytes"),
    )
    archive_sha256 = _compact_first_present_str(
        artifact_dict,
        ("archive_sha256", "archive_zip_sha256", "candidate_archive_sha256"),
    )
    proof_sha256 = (
        _sha256_file(receiver_proof_path)
        if receiver_proof_path is not None and receiver_proof_path.is_file()
        else None
    )
    proof_payload = (
        _load_json(receiver_proof_path)
        if receiver_proof_path is not None and receiver_proof_path.is_file()
        else {}
    )
    proof_passed = bool(
        isinstance(proof_payload, Mapping)
        and (
            proof_payload.get("runtime_consumption_proof_ready") is True
            or proof_payload.get("receiver_archive_replay_verified") is True
            or proof_payload.get("runtime_consumption_proof_passed") is True
            or proof_payload.get("receiver_proof_passed") is True
        )
        and not proof_payload.get("blockers")
    )
    local_replay_axis = (
        str(local_cpu_replay_summary.get("axis_tag") or "")
        if isinstance(local_cpu_replay_summary, Mapping)
        else ""
    )
    local_replay_executed = isinstance(local_cpu_replay_summary, Mapping)
    local_replay_passed = bool(
        local_replay_executed
        and not local_cpu_replay_summary.get("blockers")
        and (
            local_cpu_replay_summary.get("local_receiver_archive_replay_verified")
            is True
            or local_cpu_replay_summary.get("receiver_archive_replay_verified")
            is True
            or local_cpu_replay_summary.get("score_claim_valid") is True
            or local_cpu_replay_summary.get("score_claim") is False
        )
    )
    candidate_id = str(candidate.get("candidate_id") or "manual_cli_hi_nerv")
    modelsize_mparams = _compact_finite_float_from_keys(
        candidate,
        (
            "modelsize_mparams",
            "target_modelsize_mparams",
        ),
    )
    row: dict[str, Any] = {
        "schema": "hi_nerv_trained_archive_byte_oracle_row.v1",
        "row_id": candidate_id,
        "family": "hi_nerv",
        "carrier_id": "hi_nerv",
        "candidate_id": candidate_id,
        "num_pairs": int(num_pairs),
        "sample_count": int(num_pairs),
        "modelsize_mparams": modelsize_mparams,
        "hard_byte_ceiling": _compact_first_present_int(
            candidate,
            ("hard_byte_ceiling",),
        ),
        "nominal_total_payload_bytes": _compact_first_present_int(
            candidate,
            ("nominal_total_payload_bytes",),
        ),
        "archive_path": archive_path.as_posix() if archive_path is not None else None,
        "archive_bytes": archive_bytes,
        "measured_archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "receiver_proof_path": (
            receiver_proof_path.as_posix()
            if receiver_proof_path is not None and receiver_proof_path.is_file()
            else None
        ),
        "receiver_proof_sha256": proof_sha256,
        "receiver_proof_passed": proof_passed,
        "receiver_closed": proof_passed,
        "byte_closed_receiver_proof": proof_passed,
        "receiver_archive_replay_verified": proof_passed,
        "local_cpu_replay_executed": local_replay_executed,
        "local_cpu_replay_passed": local_replay_passed,
        "local_cpu_replay_axis_tag": local_replay_axis or None,
        "full_video_mlx_prefilter_attached": bool(
            mlx_prefilter_coverage.get("has_full_video_mlx_prefilter")
        ),
        "axis_tag": "[planning/control]",
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    blockers: list[str] = []
    if not candidate:
        blockers.append("hi_nerv_modelsize_candidate_not_selected")
    if archive_bytes is None:
        blockers.append("hi_nerv_trained_archive_byte_oracle_archive_bytes_missing")
    if not _compact_is_sha256_hex(archive_sha256):
        blockers.append("hi_nerv_trained_archive_byte_oracle_archive_sha_missing")
    if not proof_passed:
        blockers.append("hi_nerv_receiver_proof_missing_or_not_passed")
    if int(num_pairs) < CONTEST_PAIR_COUNT:
        blockers.append("hi_nerv_trained_archive_byte_oracle_partial_pair_scope")
    if not local_replay_executed:
        blockers.append("hi_nerv_local_cpu_replay_gate_missing")
    if local_replay_axis and not local_replay_axis.lower().startswith("[contest-"):
        blockers.append("hi_nerv_local_cpu_replay_not_contest_auth_axis")
    row["blockers"] = _dedupe(blockers)

    ladder = build_nerv_receiver_closed_modelsize_ladder(
        [row],
        carrier_id="hi_nerv",
        source_artifact_path=(archive_path.as_posix() if archive_path is not None else None),
        repo_root=repo_root,
    )
    ladder_path = output_dir / "hi_nerv_receiver_closed_modelsize_ladder.json"
    _write_json(ladder_path, ladder)
    oracle = {
        "schema": "hi_nerv_trained_archive_byte_oracle.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "row": row,
        "receiver_closed_modelsize_ladder_path": ladder_path.as_posix(),
        "receiver_closed_modelsize_ladder_status": ladder.get("status"),
        "receiver_closed_modelsize_ladder_ready": bool(
            ladder.get("ready_for_carrier_training_plan")
        ),
        "measured_archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "feedback_ready": bool(
            archive_bytes is not None
            and _compact_is_sha256_hex(archive_sha256)
            and proof_passed
            and int(num_pairs) >= CONTEST_PAIR_COUNT
        ),
        "blockers": _dedupe([*blockers, *(ladder.get("blockers") or [])]),
        **FALSE_AUTHORITY,
    }
    oracle_path = output_dir / "hi_nerv_trained_archive_byte_oracle.json"
    _write_json(oracle_path, oracle)
    return {
        **oracle,
        "path": oracle_path.as_posix(),
        "sha256": _sha256_file(oracle_path),
        "receiver_closed_modelsize_ladder_sha256": _sha256_file(ladder_path),
        "receiver_closed_modelsize_ladder": ladder,
    }


def _compact_receiver_closed_identity_blockers(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if _compact_first_present_str(
        row,
        (
            "receiver_proof_path",
            "receiver_proof_report_path",
            "receiver_closed_proof_path",
        ),
    ) is None:
        blockers.append("receiver_proof_path_missing")
    if not _compact_is_sha256_hex(
        _compact_first_present_str(
            row,
            (
                "receiver_proof_sha256",
                "receiver_proof_report_sha256",
                "receiver_closed_proof_sha256",
            ),
        )
    ):
        blockers.append("receiver_proof_sha256_missing_or_invalid")
    if not _compact_is_sha256_hex(
        _compact_first_present_str(
            row,
            (
                "archive_sha256",
                "candidate_archive_sha256",
                "receiver_archive_sha256",
                "source_archive_sha256",
                "archive_zip_sha256",
            ),
        )
    ):
        blockers.append("archive_sha256_missing_or_invalid")
    if _compact_first_present_str(
        row,
        (
            "axis_tag",
            "score_axis_tag",
            "measured_score_axis_tag",
            "receiver_proof_axis_tag",
        ),
    ) is None:
        blockers.append("receiver_proof_axis_tag_missing")
    sample_count = _compact_first_present_int(
        row,
        ("sample_pair_count", "sample_pairs", "n_pairs", "num_pairs", "pair_count"),
    )
    full_video = any(
        bool(row.get(key))
        for key in ("full_video_coverage", "full600_coverage", "full_sample_coverage")
    )
    if sample_count is None and not full_video:
        blockers.append("receiver_proof_full_sample_count_missing")
    elif sample_count is not None and sample_count < CONTEST_PAIR_COUNT and not full_video:
        blockers.append("receiver_proof_full_sample_count_incomplete")
    return blockers


def _compact_first_present_str(
    row: Mapping[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _compact_first_present_int(
    row: Mapping[str, Any],
    keys: tuple[str, ...],
) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _compact_is_sha256_hex(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _compact_modelsize_row_has_source_bound_capacity(row: Mapping[str, Any]) -> bool:
    if _compact_finite_float_from_keys(row, ("modelsize_mparams",)) is not None:
        return True
    if _compact_positive_int_from_keys(row, ("fc_dim",)) is not None:
        return True
    official_controls = row.get("official_controls")
    if isinstance(official_controls, Mapping):
        return (
            _compact_finite_float_from_keys(official_controls, ("--modelsize",))
            is not None
            or _compact_positive_int_from_keys(official_controls, ("fc_dim",)) is not None
        )
    solved = row.get("solved_budget")
    if isinstance(solved, Mapping):
        return _compact_modelsize_row_has_source_bound_capacity(solved)
    return False


def _compact_modelsize_row_has_nonrate_signal(row: Mapping[str, Any]) -> bool:
    if _compact_finite_float_from_keys(
        row,
        ("nonrate_score", "nonrate_score_value"),
    ) is not None:
        return True
    if _compact_finite_float_from_keys(row, ("avg_segnet_dist", "d_seg")) is not None:
        return _compact_finite_float_from_keys(
            row,
            ("avg_posenet_dist", "d_pose"),
        ) is not None
    return False


def _compact_positive_int_from_keys(
    row: Mapping[str, Any],
    keys: tuple[str, ...],
) -> int | None:
    for key in keys:
        try:
            value = int(row.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _compact_finite_float_from_keys(
    row: Mapping[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _metric_mapping_float(
    mapping: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    if mapping is None:
        return None
    try:
        value = float(mapping.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class _PoseInstabilityEpochMonitor:
    """Fail-fast monitor for full-video HiNeRV PoseNet instability."""

    def __init__(
        self,
        *,
        min_epoch: int = 64,
        consecutive_bad_epochs: int = 8,
        pose_loss_threshold: float = 1_000.0,
        pose_axis_threshold: float = 1_000.0,
    ) -> None:
        self.min_epoch = max(0, int(min_epoch))
        self.consecutive_bad_epochs = max(1, int(consecutive_bad_epochs))
        self.pose_loss_threshold = float(pose_loss_threshold)
        self.pose_axis_threshold = float(pose_axis_threshold)
        self.bad_epoch_count = 0
        self.last_reason = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "compact_pose_instability_epoch_monitor.v1",
            "min_epoch": int(self.min_epoch),
            "consecutive_bad_epochs": int(self.consecutive_bad_epochs),
            "pose_loss_threshold": float(self.pose_loss_threshold),
            "pose_axis_threshold": float(self.pose_axis_threshold),
            "bad_epoch_count": int(self.bad_epoch_count),
            "last_reason": self.last_reason,
        }

    def __call__(self, metrics: Any) -> None:
        epoch = int(metrics.epoch)
        if epoch < self.min_epoch:
            return
        loss_components = getattr(metrics, "loss_components", None)
        per_axis = getattr(metrics, "per_axis_decomposition", None)
        pose_loss = _metric_mapping_float(loss_components, "loss_part_pose_distill")
        pose_axis = _metric_mapping_float(per_axis, "pose")
        bad_loss = (
            pose_loss is not None and pose_loss >= self.pose_loss_threshold
        )
        bad_axis = (
            pose_axis is not None and pose_axis >= self.pose_axis_threshold
        )
        if bad_loss or bad_axis:
            self.bad_epoch_count += 1
            self.last_reason = (
                f"hi_nerv_pose_instability_epoch_{epoch}:"
                f"pose_loss={pose_loss}:pose_axis={pose_axis}:"
                f"bad_epochs={self.bad_epoch_count}"
            )
        else:
            self.bad_epoch_count = 0
            self.last_reason = ""
        if self.bad_epoch_count >= self.consecutive_bad_epochs:
            raise LongTrainingStopRequested(
                "hi_nerv_pose_instability_guard:"
                f"epoch={epoch}:"
                f"pose_loss={pose_loss}:"
                f"pose_axis={pose_axis}:"
                f"consecutive_bad_epochs={self.bad_epoch_count}:"
                f"thresholds=loss>={self.pose_loss_threshold},"
                f"axis>={self.pose_axis_threshold}"
            )


def _modelsize_budget_rows_from_plan(plan: Any) -> list[dict[str, Any]]:
    if not isinstance(plan, Mapping):
        return []
    source_rows = plan.get("receiver_closed_points") or plan.get("points") or []
    if not isinstance(source_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for point in source_rows:
        if not isinstance(point, Mapping):
            continue
        row = dict(point.get("source") or {})
        row.setdefault("row_id", point.get("row_id"))
        row.setdefault("archive_bytes", point.get("archive_bytes"))
        row.setdefault("nonrate_score", point.get("nonrate_score"))
        row.setdefault("receiver_closed", point.get("receiver_closed_bytes"))
        row.setdefault("receiver_proof_passed", point.get("receiver_closed_bytes"))
        rows.append(row)
    return rows


def _validate_hi_nerv_frontier_training_config(
    *,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    allow_segnet_only_research: bool,
    allow_unscored_research_smoke: bool,
    score_aware_training_plan: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    segnet_attached = float(segnet_distillation_weight) > 0.0
    posenet_attached = float(pose_distillation_weight) > 0.0
    if not segnet_attached:
        blockers.append("hi_nerv_real_segnet_teacher_missing")
    if not posenet_attached:
        blockers.append("hi_nerv_real_posenet_teacher_missing")
    if allow_segnet_only_research and segnet_attached and not posenet_attached:
        blockers.append("hi_nerv_segnet_only_research_not_frontier_targeting")
    unscored_smoke = bool(allow_unscored_research_smoke)
    launch_allowed = (segnet_attached and posenet_attached) or unscored_smoke
    return {
        "schema": "compact_hi_nerv_score_aware_training_config_gate.v1",
        "launch_allowed": launch_allowed,
        "frontier_targeting": segnet_attached and posenet_attached,
        "allow_segnet_only_research": bool(allow_segnet_only_research),
        "allow_unscored_research_smoke": unscored_smoke,
        "real_segnet_teacher_attached": segnet_attached,
        "real_posenet_teacher_attached": posenet_attached,
        "modelsize_budget_receiver_closed_ready": bool(
            score_aware_training_plan.get("modelsize_budget_receiver_closed_ready")
        ),
        "planner_action": score_aware_training_plan.get("planner_action"),
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _hi_nerv_launch_source_faithfulness_report(
    *,
    use_hierarchical_feature_grid: bool,
    use_convnext_blocks: bool,
    local_grid_levels: int,
    local_grid_channels: int,
    convnext_mlp_ratio: int,
    convnext_kernel_size: int,
    decoder_codec: str,
    hi_nerv_latent_codec: str,
) -> dict[str, Any]:
    cfg = SimpleNamespace(
        use_hierarchical_feature_grid=bool(use_hierarchical_feature_grid),
        use_convnext_blocks=bool(use_convnext_blocks),
        local_grid_levels=int(local_grid_levels),
        local_grid_channels=int(local_grid_channels),
        convnext_mlp_ratio=int(convnext_mlp_ratio),
        convnext_kernel_size=int(convnext_kernel_size),
    )
    return _hi_nerv_source_faithfulness_report(
        cfg=cfg,
        decoder_codec=str(decoder_codec),
    )


def _hi_nerv_source_faithfulness_report(*, cfg: Any, decoder_codec: str) -> dict[str, Any]:
    """Classify whether a HiNeRV launch is official-control faithful or local."""

    hierarchical_grid = bool(getattr(cfg, "use_hierarchical_feature_grid", False))
    convnext_blocks = bool(getattr(cfg, "use_convnext_blocks", False))
    official_hinerv_blockers: list[str] = []
    if not hierarchical_grid:
        official_hinerv_blockers.append(
            "hinerv_official_hierarchical_feature_grid_not_enabled"
        )
    if not convnext_blocks:
        official_hinerv_blockers.append("hinerv_official_convnext_blocks_not_enabled")
    pr95_better_blockers = [
        "hinerv_pr95_pixelshuffle_bilinear_skip_refine_path_missing",
        "hinerv_pr95_pr101_latent_delta_brotli_codec_missing",
    ]
    official_hinerv_control = not official_hinerv_blockers
    source_parity_binding = _hi_nerv_source_parity_binding()
    source_parity_blockers = list(source_parity_binding.get("required_blockers") or [])
    source_parity_attached = bool(
        source_parity_binding.get("contract_attached")
        and not source_parity_blockers
    )
    source_faithful = False
    if official_hinerv_control and pr95_better_blockers and source_parity_attached:
        classification = (
            "official_hinerv_control_candidate_source_parity_bound_"
            "pr95_better_gaps"
        )
    elif official_hinerv_control and pr95_better_blockers:
        classification = (
            "official_hinerv_control_candidate_source_parity_missing_"
            "pr95_better_gaps"
        )
    elif official_hinerv_control:
        classification = (
            "official_hinerv_control_candidate_source_parity_bound"
            if source_parity_attached
            else "official_hinerv_control_candidate_source_parity_missing"
        )
    else:
        classification = "local_hiv1_adaptation_not_official_hinerv"
    return {
        "schema": "hi_nerv_source_faithfulness.v1",
        "classification": classification,
        "source_faithful_official_hinerv": source_faithful,
        "official_hinerv_control": official_hinerv_control,
        "official_source_parity_proof_required": True,
        "official_source_parity_proof_attached": source_parity_attached,
        "source_parity_binding": source_parity_binding,
        "local_hiv1_adaptation": not source_faithful,
        "use_hierarchical_feature_grid": hierarchical_grid,
        "use_convnext_blocks": convnext_blocks,
        "local_grid_levels": int(getattr(cfg, "local_grid_levels", 0) or 0),
        "local_grid_channels": int(getattr(cfg, "local_grid_channels", 0) or 0),
        "convnext_mlp_ratio": int(getattr(cfg, "convnext_mlp_ratio", 0) or 0),
        "convnext_kernel_size": int(getattr(cfg, "convnext_kernel_size", 0) or 0),
        "decoder_codec": str(decoder_codec),
        "official_hinerv_blockers": official_hinerv_blockers,
        "source_parity_blockers": source_parity_blockers,
        "pr95_better_blockers": pr95_better_blockers,
        "blockers": [
            *official_hinerv_blockers,
            *source_parity_blockers,
            *pr95_better_blockers,
        ],
        "authority_note": (
            "This classifies source/architecture custody only. It is not score "
            "authority, and local HiNeRV/MLX rows remain false-authority until "
            "byte-closed archive/runtime plus contest CPU/CUDA replay."
        ),
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _hi_nerv_source_parity_binding() -> dict[str, Any]:
    """Return the long-training source-parity binding for HiNeRV rows."""

    contract = build_nerv_source_parity_contract(
        repo_root=REPO_ROOT,
        families=("hi_nerv",),
    )
    family_rows = [
        row
        for row in contract.get("family_rows") or ()
        if isinstance(row, Mapping) and row.get("family") == "hi_nerv"
    ]
    family_row = family_rows[0] if family_rows else {}
    blockers = [
        f"source_parity:{blocker}"
        for blocker in (family_row.get("blockers") if isinstance(family_row, Mapping) else ())
    ]
    feature_statuses = {
        str(row.get("feature_id")): str(row.get("status"))
        for row in contract.get("feature_rows") or ()
        if isinstance(row, Mapping) and row.get("family") == "hi_nerv"
    }
    return {
        "schema": "hi_nerv_source_parity_binding.v1",
        "contract_schema": contract.get("schema"),
        "contract_authority": contract.get("authority"),
        "contract_attached": contract.get("schema") == "nerv_source_parity_contract.v1",
        "required_for_long_training_ready": bool(
            contract.get("required_for_long_training_ready")
        ),
        "required_blockers": blockers,
        "nonblocking_gaps": [
            f"source_parity:{gap}" for gap in contract.get("nonblocking_gaps") or ()
        ],
        "feature_statuses": feature_statuses,
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _hi_nerv_source_faithfulness_metadata(
    *, cfg: Any, decoder_codec: str
) -> dict[str, Any]:
    """Return source-faithfulness facts safe for nested substrate metadata.

    ``_hi_nerv_source_faithfulness_report`` is also used as a top-level report
    surface and therefore carries canonical false-authority fields. The shared
    MLX harness intentionally rejects those fields inside
    ``substrate_artifact_metadata`` so there is one custody surface. Keep the
    source-control signal, strip only authority/readiness flags at the boundary.
    """

    report = _hi_nerv_source_faithfulness_report(
        cfg=cfg,
        decoder_codec=decoder_codec,
    )
    return strip_candidate_curriculum_authority_fields(report)


def _run_hi_nerv_mlx_scoreaware_smoke(
    *,
    output_dir: Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    latent_dim: int,
    embed_dim: int,
    decoder_channel: int,
    use_hierarchical_feature_grid: bool,
    use_convnext_blocks: bool,
    local_grid_levels: int,
    local_grid_channels: int,
    convnext_mlp_ratio: int,
    convnext_kernel_size: int,
    mid_injection_block_index: int,
    fine_injection_block_index: int,
    decoder_codec: str,
    hi_nerv_latent_codec: str,
    ema_decay: float,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    pose_distillation_loss: str,
    pose_distillation_huber_delta: float,
    segnet_distillation_objective: str,
    distillation_temperature: float,
    segnet_tau_boundary: float,
    segnet_hinge_margin: float,
    distillation_device: str,
    requested_distillation_device: str | None,
    allow_segnet_only_research: bool,
    coder_aware_qat: bool,
    coder_qat_quant_bits: int,
    coder_qat_quant_residual_weight: float,
    coder_qat_magnitude_weight: float,
    coder_qat_delta_weight: float,
    coder_qat_c1a_entropy_weight: float,
    coder_qat_c1a_sigma: float,
    coder_qat_c1a_sample_size: int,
    recon_pixel_weight_path: str | Path | None,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None,
    recon_pixel_weight_auto_discovery: Mapping[str, Any] | None,
    auto_segnet_boundary_recon_weight: bool,
    recon_pixel_weight_tau: float,
    recon_pixel_weight_normalize: str,
    mlx_prefilter_scorer_device: str | None,
    mlx_prefilter_scorer_batch_pairs: int,
    mlx_prefilter_progress_every: int,
    telemetry_flush_interval_epochs: int,
    checkpoint_interval_epochs: int,
    checkpoint_dir: str | Path | None,
    resume_from_checkpoint: str | Path | None,
    optimizer_kind: str,
    hi_nerv_optimizer_policy: Mapping[str, Any],
    optimizer_controls: Mapping[str, Any],
    prioritized_pair_indices: tuple[int, ...],
    random_seed: int,
    scorer_upstream_dir: Path,
    repo_root: Path,
    candidate_curriculum_plan: Mapping[str, Any] | None = None,
) -> Any:
    pairs = int(num_pairs)
    if pairs < 1:
        raise CompactRendererMlxSpineRunnerError("num_pairs must be >= 1")
    try:
        prioritized_pair_indices = validate_pair_indices_in_range(
            prioritized_pair_indices,
            num_pairs=pairs,
            field="prioritized_pair_indices",
        )
    except HardPairIndicesError as exc:
        raise CompactRendererMlxSpineRunnerError(str(exc)) from exc

    from tac.substrates._shared.mlx_score_aware import (
        CoderAwareQATConfig,
        RendererBundle,
        build_decoder_coder_qat_terms,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        coder_qat_loss_weights,
        coder_qat_metadata,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.hi_nerv.architecture import HinervConfig
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    requested_distillation_device = str(
        requested_distillation_device or distillation_device
    )
    resolved_distillation_device = _resolve_torch_scorer_device_alias(
        str(distillation_device)
    )
    effective_prefilter_scorer_device = str(
        mlx_prefilter_scorer_device or requested_distillation_device
    )
    if segnet_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "segnet_distillation_weight must be >= 0"
        )
    if pose_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_weight must be >= 0"
        )
    if str(pose_distillation_loss) not in {"mse", "huber"}:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_loss must be one of ['mse', 'huber']"
        )
    if float(pose_distillation_huber_delta) <= 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_huber_delta must be > 0"
        )
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    if (
        segnet_distillation_weight > 0.0
        and pose_distillation_weight <= 0.0
        and not allow_segnet_only_research
    ):
        raise CompactRendererMlxSpineRunnerError(
            "SegNet-bound HiNeRV training must also bind PoseNet. Pass "
            "--pose-distillation-weight > 0, or explicitly pass "
            "--allow-segnet-only-research for a false-authority SegNet-axis probe."
        )
    _require_scorer_upstream_dir_for_distillation(
        upstream_dir=scorer_upstream_dir,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
    )
    if recon_pixel_weight_path is not None and auto_segnet_boundary_recon_weight:
        raise CompactRendererMlxSpineRunnerError(
            "pass either --recon-pixel-weight-path or "
            "--auto-segnet-boundary-recon-weight, not both"
        )
    cfg = HinervConfig(
        latent_dim_coarse=max(1, int(latent_dim) // 2),
        latent_dim_mid=max(1, int(latent_dim)),
        latent_dim_fine=max(1, int(latent_dim) * 2),
        embed_dim=max(1, int(embed_dim)),
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=tuple([max(1, int(decoder_channel))] * 7),
        sin_frequency=30.0,
        num_upsample_blocks=7,
        mid_injection_block_index=int(mid_injection_block_index),
        fine_injection_block_index=int(fine_injection_block_index),
        num_pairs=pairs,
        output_height=384,
        output_width=512,
        use_hierarchical_feature_grid=bool(use_hierarchical_feature_grid),
        use_convnext_blocks=bool(use_convnext_blocks),
        local_grid_levels=int(local_grid_levels),
        local_grid_channels=int(local_grid_channels),
        convnext_mlp_ratio=int(convnext_mlp_ratio),
        convnext_kernel_size=int(convnext_kernel_size),
    )
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        source_video_path,
        num_pairs=pairs,
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
    )
    model = HinervSubstrateMLX(cfg)
    optimizer_policy = dict(hi_nerv_optimizer_policy or {})
    optimizer_control = dict(optimizer_controls or {})
    pr95_curriculum_enabled = bool(
        optimizer_policy.get("pr95_faithful_curriculum_enabled")
    )
    native_optimizer_active = bool(optimizer_policy.get("native_optimizer_active"))
    effective_optimizer_kind = str(
        optimizer_policy.get("optimizer_kind") or optimizer_kind
    )
    effective_weight_decay = optimizer_control.get("weight_decay_effective")
    coder_qat_cfg = CoderAwareQATConfig(
        enabled=bool(coder_aware_qat),
        quant_bits=int(coder_qat_quant_bits),
        quant_residual_weight=float(coder_qat_quant_residual_weight),
        magnitude_weight=float(coder_qat_magnitude_weight),
        delta_weight=float(coder_qat_delta_weight),
        c1a_entropy_weight=float(coder_qat_c1a_entropy_weight),
        c1a_sigma=float(coder_qat_c1a_sigma),
        c1a_sample_size=int(coder_qat_c1a_sample_size),
    ).validated()
    decoder_waterfill_fake_quant_bits_by_name = (
        _decoder_weight_waterfill_fake_quant_bits_by_name(decoder_weight_waterfill_plan)
    )
    model.configure_decoder_fake_quant_forward(
        enabled=bool(coder_qat_cfg.enabled or decoder_waterfill_fake_quant_bits_by_name),
        quant_bits=int(coder_qat_cfg.quant_bits) if bool(coder_qat_cfg.enabled) else None,
        per_tensor_bits=decoder_waterfill_fake_quant_bits_by_name,
    )
    pose_instability_monitor = _PoseInstabilityEpochMonitor()

    def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
        return build_decoder_coder_qat_terms(model_obj, coder_qat_cfg)

    def _export_archive(model_obj: Any, archive_output_dir: Path) -> tuple[Path, str, int]:
        return export_hi_nerv_mlx_archive(
            model_obj,
            archive_output_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=True,
            retain_receiver_proof_output=False,
            mlx_triage_argv=[
                "tools/run_compact_renderer_mlx_spine_runner.py",
                "--execute-family",
                "hi_nerv",
            ],
            decoder_codec=str(decoder_codec),
            latent_codec=str(hi_nerv_latent_codec),
            decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
        )

    artifact_metadata = {
        "schema": "compact_renderer_hi_nerv_mlx_adapter_smoke_metadata.v1",
        "family": "hi_nerv",
        "num_pairs": pairs,
        "full_video_pairs_required_for_promotion": 600,
        "decoder_codec": str(decoder_codec),
        "hi_nerv_latent_codec": str(hi_nerv_latent_codec),
        "model_num_parameters_at_init": int(model.num_parameters()),
        "source_faithfulness": _hi_nerv_source_faithfulness_metadata(
            cfg=cfg,
            decoder_codec=str(decoder_codec),
        ),
        "config": {
            "latent_dim_coarse": int(cfg.latent_dim_coarse),
            "latent_dim_mid": int(cfg.latent_dim_mid),
            "latent_dim_fine": int(cfg.latent_dim_fine),
            "embed_dim": int(cfg.embed_dim),
            "decoder_channels": [int(value) for value in cfg.decoder_channels],
            "num_upsample_blocks": int(cfg.num_upsample_blocks),
            "mid_injection_block_index": int(cfg.mid_injection_block_index),
            "fine_injection_block_index": int(cfg.fine_injection_block_index),
            "output_height": int(cfg.output_height),
            "output_width": int(cfg.output_width),
            "use_hierarchical_feature_grid": bool(cfg.use_hierarchical_feature_grid),
            "use_convnext_blocks": bool(cfg.use_convnext_blocks),
            "local_grid_levels": int(cfg.local_grid_levels),
            "local_grid_channels": int(cfg.local_grid_channels),
            "convnext_mlp_ratio": int(cfg.convnext_mlp_ratio),
            "convnext_kernel_size": int(cfg.convnext_kernel_size),
        },
        "score_aware_training": {
            "schema": "compact_hi_nerv_score_aware_training.v1",
            "candidate_curriculum_plan": strip_candidate_curriculum_authority_fields(
                candidate_curriculum_plan or {}
            ),
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "pose_distillation_loss": str(pose_distillation_loss),
            "pose_distillation_huber_delta": float(pose_distillation_huber_delta),
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": resolved_distillation_device,
            "requested_distillation_device": requested_distillation_device,
            "distillation_device_resolution": {
                "schema": "compact_runner_torch_scorer_device_resolution.v1",
                "requested": requested_distillation_device,
                "resolved": resolved_distillation_device,
                "scope": "real_pytorch_segnet_posenet_teacher_cache",
            },
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "optimizer_policy": strip_candidate_curriculum_authority_fields(
                optimizer_policy
            ),
            "pr95_faithful_curriculum_enabled": pr95_curriculum_enabled,
            "native_optimizer_active": native_optimizer_active,
            "optimizer_kind": effective_optimizer_kind,
            "optimizer_controls": strip_candidate_curriculum_authority_fields(
                optimizer_control
            ),
            "effective_weight_decay": effective_weight_decay,
            "checkpoint_interval_epochs": checkpoint_interval,
            "checkpoint_dir": (
                Path(checkpoint_dir).as_posix() if checkpoint_dir is not None else None
            ),
            "resume_from_checkpoint": (
                Path(resume_from_checkpoint).as_posix()
                if resume_from_checkpoint is not None
                else None
            ),
            "checkpoint_policy": "periodic_canonical_long_training_checkpoint",
            "prioritized_pair_training": {
                "schema": "compact_hi_nerv_prioritized_pair_training.v1",
                "enabled": bool(prioritized_pair_indices),
                "pair_indices": [int(value) for value in prioritized_pair_indices],
                "pair_count": len(prioritized_pair_indices),
                "sampling_scope": "training_batch_emphasis_only",
                "pair_index_domain": "decoded_prefix_pair_indices_0_to_num_pairs_minus_1",
                "arbitrary_source_pair_hydration": False,
                "target_hydration_pair_indices_consumed": False,
                "requires_num_pairs_covering_pair_ids": bool(
                    prioritized_pair_indices
                ),
                "authority": "macos_mlx_research_signal_false_authority",
                "canonical_authority_surface": (
                    "TrainingArtifact top-level false-authority fields"
                ),
            },
            "coder_aware_qat": coder_qat_metadata(coder_qat_cfg),
            "decoder_fake_quant_forward": {
                "schema": "hi_nerv_decoder_fake_quant_forward_qat.v1",
                "enabled": bool(
                    coder_qat_cfg.enabled or decoder_waterfill_fake_quant_bits_by_name
                ),
                "global_fake_quant_enabled": bool(coder_qat_cfg.enabled),
                "quant_bits": int(coder_qat_cfg.quant_bits),
                "global_quant_bits": (
                    int(coder_qat_cfg.quant_bits) if bool(coder_qat_cfg.enabled) else None
                ),
                "per_tensor_waterfill_enabled": bool(
                    decoder_waterfill_fake_quant_bits_by_name
                ),
                "per_tensor_waterfill_group_count": len(
                    decoder_waterfill_fake_quant_bits_by_name
                ),
                "per_tensor_waterfill_bits_by_name": dict(
                    sorted(decoder_waterfill_fake_quant_bits_by_name.items())
                ),
                "quantizer_geometry": (
                    "symmetric_signed_axis0_fp16_scale_for_matrix_conv_weights_"
                    "per_tensor_fp16_scale_for_biases"
                ),
                "target": (
                    "decoder weights and decoder biases; latents remain priced "
                    "by their archive section"
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "eval_roundtrip_ste": _hi_nerv_eval_roundtrip_ste_metadata(),
            "pose_student_input_preprocess": _hi_nerv_pose_preprocess_metadata(),
            "recon_pixel_weight": _disabled_recon_pixel_weight_metadata(),
            "local_mlx_prefilter": {
                "schema": "compact_hi_nerv_local_mlx_prefilter_config.v1",
                "scorer_device": effective_prefilter_scorer_device,
                "scorer_batch_pairs": int(mlx_prefilter_scorer_batch_pairs),
                "progress_every": int(mlx_prefilter_progress_every),
                "singleton_required_for_local_cpu_replay_unlock": True,
                "gpu_profiles_are_prefilter_only": (
                    effective_prefilter_scorer_device != "cpu"
                ),
                "batched_profiles_are_prefilter_only": (
                    int(mlx_prefilter_scorer_batch_pairs) != 1
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "pose_instability_monitor": pose_instability_monitor.as_dict(),
            "scorer_upstream_snapshot": _scorer_upstream_metadata(
                scorer_upstream_dir
            ),
        },
        "training_executed": True,
        "score_authority": "false_macos_mlx_research_signal",
    }
    bundle_kwargs: dict[str, Any] = {
        "model": model,
        "target_rgb_0": target_rgb_0,
        "target_rgb_1": target_rgb_1,
        "num_pairs": pairs,
        "forward_convention": "call_b2chw_255",
        "extra_loss_terms": _extra_loss_terms,
        "extra_loss_weights": coder_qat_loss_weights(coder_qat_cfg),
        "export_archive_fn": _export_archive,
        "substrate_artifact_metadata": artifact_metadata,
        "eval_roundtrip_ste_enabled": True,
        "eval_roundtrip_camera_hw": (874, 1164),
        "pose_student_input_preprocess": "pr95_yuv6",
    }
    recon_pixel_weight = None
    if recon_pixel_weight_path is not None:
        recon_pixel_weight, recon_metadata = _load_recon_pixel_weight(
            recon_pixel_weight_path,
            base=repo_root,
            expected_pairs=pairs,
            normalize=recon_pixel_weight_normalize,
        )
        if recon_pixel_weight_auto_discovery is not None:
            recon_metadata["source_kind"] = "auto_discovered_joint_p18_p19_file"
            recon_metadata["auto_discovery"] = (
                strip_candidate_curriculum_authority_fields(
                    recon_pixel_weight_auto_discovery
                )
            )
            recon_metadata["scorer_terms"] = {
                "p18_segnet": "auto_discovered_joint_torch_exact_vjp",
                "p19_posenet": "auto_discovered_joint_torch_exact_vjp",
            }
        artifact_metadata["score_aware_training"]["recon_pixel_weight"] = (
            recon_metadata
        )
    teacher_probe_bundle = RendererBundle(**bundle_kwargs)
    scorer_teacher = None
    learnable_student_head = None
    pose_scorer_teacher = None
    learnable_pose_student_head = None
    if segnet_distillation_weight > 0.0:
        scorer_teacher = build_mlx_segnet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=resolved_distillation_device,
        )
        learnable_student_head = build_learnable_student_head(
            num_classes=int(scorer_teacher.num_classes),
            seed=int(random_seed),
        )
    if auto_segnet_boundary_recon_weight:
        if scorer_teacher is None:
            raise CompactRendererMlxSpineRunnerError(
                "--auto-segnet-boundary-recon-weight requires "
                "--segnet-distillation-weight > 0 so real SegNet teacher "
                "logits exist"
            )
        recon_pixel_weight, recon_metadata = _segnet_boundary_recon_pixel_weight(
            scorer_teacher,
            tau=float(recon_pixel_weight_tau),
            normalize=recon_pixel_weight_normalize,
        )
        artifact_metadata["score_aware_training"]["recon_pixel_weight"] = (
            recon_metadata
        )
    if pose_distillation_weight > 0.0:
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=resolved_distillation_device,
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=int(pose_scorer_teacher.pose_dims),
            input_channels=6,
            seed=int(random_seed) + 1,
        )
    bundle = RendererBundle(
        **bundle_kwargs,
        recon_pixel_weight=recon_pixel_weight,
        recon_pixel_weight_normalize=recon_pixel_weight_normalize,
        distillation_weight=float(segnet_distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        distillation_temperature=float(distillation_temperature),
        segnet_distillation_objective=segnet_distillation_objective,
        segnet_tau_boundary=float(segnet_tau_boundary),
        segnet_hinge_margin=float(segnet_hinge_margin),
        distillation_num_classes=(
            int(scorer_teacher.num_classes) if scorer_teacher is not None else 5
        ),
        pose_distillation_weight=float(pose_distillation_weight),
        pose_distillation_loss=str(pose_distillation_loss),
        pose_distillation_huber_delta=float(pose_distillation_huber_delta),
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=int(pose_scorer_teacher.pose_dims)
        if pose_scorer_teacher is not None
        else 6,
        allow_segnet_only_research=bool(allow_segnet_only_research),
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="compact_runner_hi_nerv_mlx",
        lane_id="lane_compact_renderer_mlx_spine_runner_hi_nerv_20260601",
        output_dir=output_dir,
        epochs=int(epochs),
        batch_pair_indices_per_step=max(1, int(batch_pair_indices_per_step)),
        learning_rate=float(learning_rate),
        ema_decay=float(ema_decay),
        seed=int(random_seed),
        checkpoint_interval_epochs=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        telemetry_flush_interval_epochs=max(1, int(telemetry_flush_interval_epochs)),
        pr95_faithful_curriculum_enabled=pr95_curriculum_enabled,
        pr95_curriculum_total_epochs=max(8, int(epochs)),
        ema_archive_selection_enabled=True,
        grad_clip_max_norm=optimizer_control.get("grad_clip_max_norm"),
        weight_decay=effective_weight_decay,
        optimizer_kind=effective_optimizer_kind,
        warmup_epochs=int(optimizer_control.get("warmup_epochs", 0)),
        warmup_steps_per_epoch=max(
            1, int(optimizer_control.get("warmup_steps_per_epoch", 1))
        ),
        cosine_decay_enabled=bool(optimizer_control.get("cosine_decay_enabled")),
        cosine_decay_total_epochs=optimizer_control.get("cosine_decay_total_epochs"),
        cosine_decay_min_lr_ratio=float(
            optimizer_control.get("cosine_decay_min_lr_ratio", 1e-2)
        ),
        prioritized_pair_indices=tuple(int(value) for value in prioritized_pair_indices),
        on_epoch_end=pose_instability_monitor,
        notes=(
            "Compact renderer MLX spine runner HiNeRV training using real "
            "contest video targets, byte-closed archive export, receiver proof, "
            "explicit optimizer-policy routing, and false-authority MLX evidence."
        ),
    )
    artifact_metadata["score_aware_training"]["pose_instability_monitor"] = (
        pose_instability_monitor.as_dict()
    )
    artifact_dict = artifact.as_dict() if hasattr(artifact, "as_dict") else dict(artifact)
    profile_path = output_dir / "local_mlx_prefilter_profile.json"
    progress_path = output_dir / "local_mlx_prefilter_progress.jsonl"
    archive_bytes = artifact_dict.get("archive_bytes")
    archive_sha256 = artifact_dict.get("archive_sha256")
    try:
        if archive_bytes is None or archive_sha256 is None:
            raise CompactRendererMlxSpineRunnerError(
                "archive_bytes/archive_sha256 missing; cannot build MLX prefilter"
            )
        from tac.local_acceleration.mlx_renderer_prefilter_profile import (
            write_mlx_renderer_prefilter_profile,
        )

        write_mlx_renderer_prefilter_profile(
            bundle=bundle,
            output_path=profile_path,
            archive_bytes=int(archive_bytes),
            archive_sha256=str(archive_sha256),
            upstream_dir=scorer_upstream_dir,
            scorer_device=effective_prefilter_scorer_device,
            scorer_batch_pairs=int(mlx_prefilter_scorer_batch_pairs),
            run_id="compact_runner_hi_nerv_mlx_local_prefilter",
            source_video_path=source_video_path,
            progress_jsonl_path=progress_path,
            progress_every=int(mlx_prefilter_progress_every),
        )
    except Exception as exc:
        from tac.local_acceleration.mlx_renderer_prefilter_profile import (
            write_mlx_renderer_prefilter_failure_profile,
        )

        write_mlx_renderer_prefilter_failure_profile(
            output_path=profile_path,
            archive_bytes=int(archive_bytes) if archive_bytes is not None else None,
            archive_sha256=str(archive_sha256) if archive_sha256 is not None else None,
            num_pairs=pairs,
            failure=repr(exc),
            run_id="compact_runner_hi_nerv_mlx_local_prefilter",
        )
    return artifact


def _run_pact_nerv_vq_mlx_smoke(
    *,
    output_dir: Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    latent_dim: int,
    embed_dim: int,
    codebook_size: int,
    decoder_channel: int,
    decoder_codec: str,
    ema_decay: float,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    pose_distillation_loss: str,
    pose_distillation_huber_delta: float,
    segnet_distillation_objective: str,
    distillation_temperature: float,
    segnet_tau_boundary: float,
    segnet_hinge_margin: float,
    distillation_device: str,
    allow_segnet_only_research: bool,
    coder_aware_qat: bool,
    coder_qat_quant_bits: int,
    coder_qat_quant_residual_weight: float,
    coder_qat_magnitude_weight: float,
    coder_qat_delta_weight: float,
    coder_qat_c1a_entropy_weight: float,
    coder_qat_c1a_sigma: float,
    coder_qat_c1a_sample_size: int,
    optimizer_kind: str,
    optimizer_policy: Mapping[str, Any],
    optimizer_controls: Mapping[str, Any],
    checkpoint_interval_epochs: int,
    checkpoint_dir: str | Path | None,
    resume_from_checkpoint: str | Path | None,
    random_seed: int,
    scorer_upstream_dir: Path,
    repo_root: Path,
) -> Any:
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_decoder_coder_qat_terms,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        coder_qat_loss_weights,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )
    from tac.substrates.pact_nerv_vq.architecture import PactNervVqConfig
    from tac.substrates.pact_nerv_vq.archive_candidate import (
        export_pact_nerv_vq_mlx_archive,
    )
    from tac.substrates.pact_nerv_vq.mlx_renderer import PactNervVqSubstrateMLX

    pairs = int(num_pairs)
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    if pairs < 1:
        raise CompactRendererMlxSpineRunnerError("num_pairs must be >= 1")
    if segnet_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "segnet_distillation_weight must be >= 0"
        )
    if pose_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_weight must be >= 0"
        )
    if str(pose_distillation_loss) not in {"mse", "huber"}:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_loss must be one of ['mse', 'huber']"
        )
    if float(pose_distillation_huber_delta) <= 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_huber_delta must be > 0"
        )
    if (
        segnet_distillation_weight > 0.0
        and pose_distillation_weight <= 0.0
        and not allow_segnet_only_research
    ):
        raise CompactRendererMlxSpineRunnerError(
            "SegNet-bound compact training must also bind PoseNet. Pass "
            "--pose-distillation-weight > 0, or explicitly pass "
            "--allow-segnet-only-research for a false-authority SegNet-axis probe."
        )
    _require_scorer_upstream_dir_for_distillation(
        upstream_dir=scorer_upstream_dir,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
    )
    cfg = PactNervVqConfig(
        latent_dim=int(latent_dim),
        embed_dim=int(embed_dim),
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=tuple([int(decoder_channel)] * 7),
        num_upsample_blocks=7,
        codebook_size=int(codebook_size),
        num_pairs=pairs,
        output_height=384,
        output_width=512,
    )
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        source_video_path,
        num_pairs=pairs,
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
    )
    model = PactNervVqSubstrateMLX(cfg)
    optimizer_policy_row = dict(optimizer_policy or {})
    optimizer_control = dict(optimizer_controls or {})
    pr95_curriculum_enabled = bool(
        optimizer_policy_row.get("pr95_faithful_curriculum_enabled")
    )
    effective_optimizer_kind = str(
        optimizer_policy_row.get("optimizer_kind")
        or optimizer_control.get("optimizer_kind")
        or optimizer_kind
    )
    effective_weight_decay = optimizer_control.get("weight_decay_effective")
    coder_qat_cfg, coder_qat_metadata_row = _build_pact_coder_qat_config_and_metadata(
        coder_aware_qat=coder_aware_qat,
        coder_qat_quant_bits=coder_qat_quant_bits,
        coder_qat_quant_residual_weight=coder_qat_quant_residual_weight,
        coder_qat_magnitude_weight=coder_qat_magnitude_weight,
        coder_qat_delta_weight=coder_qat_delta_weight,
        coder_qat_c1a_entropy_weight=coder_qat_c1a_entropy_weight,
        coder_qat_c1a_sigma=coder_qat_c1a_sigma,
        coder_qat_c1a_sample_size=coder_qat_c1a_sample_size,
    )

    def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
        terms = {"vq_commitment": model_obj.last_commitment_loss}
        terms.update(build_decoder_coder_qat_terms(model_obj, coder_qat_cfg))
        return terms

    def _export_archive(model_obj: Any, archive_output_dir: Path) -> tuple[Path, str, int]:
        return export_pact_nerv_vq_mlx_archive(
            model_obj,
            archive_output_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=True,
            retain_receiver_proof_output=False,
            decoder_codec=str(decoder_codec),
            mlx_triage_argv=[
                "tools/run_compact_renderer_mlx_spine_runner.py",
                "--execute-family",
                "pact_nerv_vq",
            ],
        )

    artifact_metadata = {
        "schema": "compact_renderer_pact_nerv_vq_mlx_runner_metadata.v1",
        "family": "pact_nerv_vq",
        "num_pairs": pairs,
        "full_video_pairs_required_for_promotion": 600,
        "archive_exporter": (
            "tac.substrates.pact_nerv_vq.archive_candidate."
            "export_pact_nerv_vq_mlx_archive"
        ),
        "score_aware_training": {
            "schema": "compact_pact_nerv_vq_score_aware_training.v1",
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "pose_distillation_loss": str(pose_distillation_loss),
            "pose_distillation_huber_delta": float(pose_distillation_huber_delta),
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": distillation_device,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "pr95_faithful_curriculum_enabled": pr95_curriculum_enabled,
            "native_optimizer_active": bool(
                optimizer_policy_row.get("native_optimizer_active")
            ),
            "optimizer_policy": strip_candidate_curriculum_authority_fields(
                optimizer_policy_row
            ),
            "optimizer_kind": effective_optimizer_kind,
            "optimizer_controls": strip_candidate_curriculum_authority_fields(
                optimizer_control
            ),
            "effective_weight_decay": effective_weight_decay,
            "checkpoint_interval_epochs": checkpoint_interval,
            "checkpoint_dir": (
                Path(checkpoint_dir).as_posix() if checkpoint_dir is not None else None
            ),
            "resume_from_checkpoint": (
                Path(resume_from_checkpoint).as_posix()
                if resume_from_checkpoint is not None
                else None
            ),
            "checkpoint_policy": "periodic_canonical_long_training_checkpoint",
            "scorer_upstream_snapshot": _scorer_upstream_metadata(
                scorer_upstream_dir
            ),
            "coder_aware_qat": coder_qat_metadata_row,
            "decoder_codec": str(decoder_codec),
        },
        "score_authority": "false_macos_mlx_research_signal",
    }
    extra_loss_weights = {"vq_commitment": float(cfg.commitment_weight)}
    extra_loss_weights.update(coder_qat_loss_weights(coder_qat_cfg))
    bundle_kwargs: dict[str, Any] = {
        "model": model,
        "target_rgb_0": target_rgb_0,
        "target_rgb_1": target_rgb_1,
        "num_pairs": pairs,
        "forward_convention": "call_b2chw_255",
        "extra_loss_terms": _extra_loss_terms,
        "extra_loss_weights": extra_loss_weights,
        "export_archive_fn": _export_archive,
        "substrate_artifact_metadata": artifact_metadata,
    }
    teacher_probe_bundle = RendererBundle(**bundle_kwargs)
    scorer_teacher = None
    learnable_student_head = None
    pose_scorer_teacher = None
    learnable_pose_student_head = None
    if segnet_distillation_weight > 0.0:
        scorer_teacher = build_mlx_segnet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=distillation_device,
        )
        learnable_student_head = build_learnable_student_head(
            num_classes=int(scorer_teacher.num_classes),
            seed=int(random_seed),
        )
    if pose_distillation_weight > 0.0:
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=distillation_device,
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=int(pose_scorer_teacher.pose_dims),
            seed=int(random_seed) + 1,
        )
    bundle = RendererBundle(
        **bundle_kwargs,
        distillation_weight=float(segnet_distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        distillation_temperature=float(distillation_temperature),
        segnet_distillation_objective=segnet_distillation_objective,
        segnet_tau_boundary=float(segnet_tau_boundary),
        segnet_hinge_margin=float(segnet_hinge_margin),
        distillation_num_classes=(
            int(scorer_teacher.num_classes) if scorer_teacher is not None else 5
        ),
        pose_distillation_weight=float(pose_distillation_weight),
        pose_distillation_loss=str(pose_distillation_loss),
        pose_distillation_huber_delta=float(pose_distillation_huber_delta),
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=int(pose_scorer_teacher.pose_dims)
        if pose_scorer_teacher is not None
        else 6,
        allow_segnet_only_research=bool(allow_segnet_only_research),
    )
    return run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="compact_runner_pact_nerv_vq_mlx",
        lane_id="lane_compact_renderer_mlx_spine_runner_pact_nerv_vq_20260601",
        output_dir=output_dir,
        epochs=int(epochs),
        batch_pair_indices_per_step=max(1, int(batch_pair_indices_per_step)),
        learning_rate=float(learning_rate),
        ema_decay=float(ema_decay),
        seed=int(random_seed),
        checkpoint_interval_epochs=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        telemetry_flush_interval_epochs=1,
        pr95_faithful_curriculum_enabled=pr95_curriculum_enabled,
        pr95_curriculum_total_epochs=max(8, int(epochs))
        if pr95_curriculum_enabled
        else None,
        grad_clip_max_norm=optimizer_control.get("grad_clip_max_norm"),
        weight_decay=effective_weight_decay,
        optimizer_kind=effective_optimizer_kind,
        warmup_epochs=int(optimizer_control.get("warmup_epochs", 0)),
        warmup_steps_per_epoch=max(
            1, int(optimizer_control.get("warmup_steps_per_epoch", 1))
        ),
        cosine_decay_enabled=bool(optimizer_control.get("cosine_decay_enabled")),
        cosine_decay_total_epochs=optimizer_control.get("cosine_decay_total_epochs"),
        cosine_decay_min_lr_ratio=float(
            optimizer_control.get("cosine_decay_min_lr_ratio", 1e-2)
        ),
        notes=(
            "Compact renderer MLX spine runner PACT-NeRV-VQ smoke using real "
            "contest video targets, byte-closed archive export, receiver proof, "
            "explicit native optimizer controls, and false-authority MLX "
            "evidence only."
        ),
    )


def _run_pact_nerv_selector_v4_mlx_smoke(
    *,
    output_dir: Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    latent_dim: int,
    embed_dim: int,
    selector_palette_size: int,
    decoder_channel: int,
    decoder_codec: str,
    ema_decay: float,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    pose_distillation_loss: str,
    pose_distillation_huber_delta: float,
    segnet_distillation_objective: str,
    distillation_temperature: float,
    segnet_tau_boundary: float,
    segnet_hinge_margin: float,
    distillation_device: str,
    allow_segnet_only_research: bool,
    coder_aware_qat: bool,
    coder_qat_quant_bits: int,
    coder_qat_quant_residual_weight: float,
    coder_qat_magnitude_weight: float,
    coder_qat_delta_weight: float,
    coder_qat_c1a_entropy_weight: float,
    coder_qat_c1a_sigma: float,
    coder_qat_c1a_sample_size: int,
    optimizer_kind: str,
    optimizer_policy: Mapping[str, Any],
    optimizer_controls: Mapping[str, Any],
    checkpoint_interval_epochs: int,
    checkpoint_dir: str | Path | None,
    resume_from_checkpoint: str | Path | None,
    random_seed: int,
    scorer_upstream_dir: Path,
    repo_root: Path,
) -> Any:
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_decoder_coder_qat_terms,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        coder_qat_loss_weights,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )
    from tac.substrates.pact_nerv_selector_v4.architecture import (
        PactNervSelectorV4Config,
    )
    from tac.substrates.pact_nerv_selector_v4.archive_candidate import (
        export_pact_nerv_selector_v4_mlx_archive,
    )
    from tac.substrates.pact_nerv_selector_v4.mlx_renderer import (
        PactNervSelectorV4SubstrateMLX,
        build_selector_v4_mlx_render_quality_report,
    )

    pairs = int(num_pairs)
    checkpoint_interval = _resolve_checkpoint_interval_epochs(
        checkpoint_interval_epochs,
        epochs=epochs,
    )
    if pairs < 1:
        raise CompactRendererMlxSpineRunnerError("num_pairs must be >= 1")
    if int(selector_palette_size) < 2:
        raise CompactRendererMlxSpineRunnerError(
            "selector_palette_size must be >= 2"
        )
    if segnet_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "segnet_distillation_weight must be >= 0"
        )
    if pose_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_weight must be >= 0"
        )
    if str(pose_distillation_loss) not in {"mse", "huber"}:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_loss must be one of ['mse', 'huber']"
        )
    if float(pose_distillation_huber_delta) <= 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_huber_delta must be > 0"
        )
    if (
        segnet_distillation_weight > 0.0
        and pose_distillation_weight <= 0.0
        and not allow_segnet_only_research
    ):
        raise CompactRendererMlxSpineRunnerError(
            "SegNet-bound selector-v4 compact training must also bind PoseNet. "
            "Pass --pose-distillation-weight > 0, or explicitly pass "
            "--allow-segnet-only-research for a false-authority SegNet-axis probe."
        )
    _require_scorer_upstream_dir_for_distillation(
        upstream_dir=scorer_upstream_dir,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
    )
    cfg = PactNervSelectorV4Config(
        latent_dim=int(latent_dim),
        embed_dim=int(embed_dim),
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=tuple([int(decoder_channel)] * 7),
        num_upsample_blocks=7,
        num_pairs=pairs,
        output_height=384,
        output_width=512,
        selector_palette_size=int(selector_palette_size),
    )
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        source_video_path,
        num_pairs=pairs,
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
    )
    model = PactNervSelectorV4SubstrateMLX(cfg)
    selector_render_quality_holder: dict[str, Any] = {}
    optimizer_policy_row = dict(optimizer_policy or {})
    optimizer_control = dict(optimizer_controls or {})
    pr95_curriculum_enabled = bool(
        optimizer_policy_row.get("pr95_faithful_curriculum_enabled")
    )
    effective_optimizer_kind = str(
        optimizer_policy_row.get("optimizer_kind")
        or optimizer_control.get("optimizer_kind")
        or optimizer_kind
    )
    effective_weight_decay = optimizer_control.get("weight_decay_effective")
    coder_qat_cfg, coder_qat_metadata_row = _build_pact_coder_qat_config_and_metadata(
        coder_aware_qat=coder_aware_qat,
        coder_qat_quant_bits=coder_qat_quant_bits,
        coder_qat_quant_residual_weight=coder_qat_quant_residual_weight,
        coder_qat_magnitude_weight=coder_qat_magnitude_weight,
        coder_qat_delta_weight=coder_qat_delta_weight,
        coder_qat_c1a_entropy_weight=coder_qat_c1a_entropy_weight,
        coder_qat_c1a_sigma=coder_qat_c1a_sigma,
        coder_qat_c1a_sample_size=coder_qat_c1a_sample_size,
    )

    def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
        return build_decoder_coder_qat_terms(model_obj, coder_qat_cfg)

    def _export_archive(model_obj: Any, archive_output_dir: Path) -> tuple[Path, str, int]:
        render_quality = build_selector_v4_mlx_render_quality_report(
            model_obj,
            sample_pair_indices=tuple(range(min(4, pairs))),
        )
        render_quality_path = archive_output_dir / "selector_v4_render_quality_report.json"
        render_quality["report_path"] = render_quality_path.as_posix()
        _write_json(render_quality_path, render_quality)
        selector_render_quality_holder.clear()
        selector_render_quality_holder.update(render_quality)
        if render_quality.get("export_blocked_recommended"):
            blockers = ", ".join(str(b) for b in render_quality.get("blockers", ()))
            raise CompactRendererMlxSpineRunnerError(
                "Selector-v4 render quality gate blocked archive export: "
                f"{blockers}"
            )
        return export_pact_nerv_selector_v4_mlx_archive(
            model_obj,
            archive_output_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=True,
            retain_receiver_proof_output=False,
            decoder_codec=str(decoder_codec),
            mlx_triage_argv=[
                "tools/run_compact_renderer_mlx_spine_runner.py",
                "--execute-family",
                "pact_nerv_selector_v4",
            ],
        )

    artifact_metadata = {
        "schema": "compact_renderer_pact_nerv_selector_v4_mlx_runner_metadata.v1",
        "family": "pact_nerv_selector_v4",
        "num_pairs": pairs,
        "full_video_pairs_required_for_promotion": 600,
        "archive_exporter": (
            "tac.substrates.pact_nerv_selector_v4.archive_candidate."
            "export_pact_nerv_selector_v4_mlx_archive"
        ),
        "selector_codec": "run_length_varint_selector",
        "selector_palette_size": int(selector_palette_size),
        "primitive_timing": "archive_encode_time_not_training_forward_pass",
        "score_aware_training": {
            "schema": "compact_pact_nerv_selector_v4_score_aware_training.v1",
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "pose_distillation_loss": str(pose_distillation_loss),
            "pose_distillation_huber_delta": float(pose_distillation_huber_delta),
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": distillation_device,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "optimizer_policy": strip_candidate_curriculum_authority_fields(
                optimizer_policy_row
            ),
            "pr95_faithful_curriculum_enabled": pr95_curriculum_enabled,
            "native_optimizer_active": bool(
                optimizer_policy_row.get("native_optimizer_active")
            ),
            "optimizer_kind": effective_optimizer_kind,
            "optimizer_controls": strip_candidate_curriculum_authority_fields(
                optimizer_control
            ),
            "effective_weight_decay": effective_weight_decay,
            "checkpoint_interval_epochs": checkpoint_interval,
            "checkpoint_dir": (
                Path(checkpoint_dir).as_posix() if checkpoint_dir is not None else None
            ),
            "resume_from_checkpoint": (
                Path(resume_from_checkpoint).as_posix()
                if resume_from_checkpoint is not None
                else None
            ),
            "checkpoint_policy": "periodic_canonical_long_training_checkpoint",
            "scorer_upstream_snapshot": _scorer_upstream_metadata(
                scorer_upstream_dir
            ),
            "scorer_coupled_rd": _scorer_coupled_rd_metadata(),
            "coder_aware_qat": coder_qat_metadata_row,
            "decoder_codec": str(decoder_codec),
        },
        "score_authority": "false_macos_mlx_research_signal",
        "selector_v4_render_quality": selector_render_quality_holder,
    }
    bundle_kwargs: dict[str, Any] = {
        "model": model,
        "target_rgb_0": target_rgb_0,
        "target_rgb_1": target_rgb_1,
        "num_pairs": pairs,
        "forward_convention": "call_b2chw_255",
        "extra_loss_terms": _extra_loss_terms,
        "extra_loss_weights": coder_qat_loss_weights(coder_qat_cfg),
        "export_archive_fn": _export_archive,
        "substrate_artifact_metadata": artifact_metadata,
    }
    teacher_probe_bundle = RendererBundle(**bundle_kwargs)
    scorer_teacher = None
    learnable_student_head = None
    pose_scorer_teacher = None
    learnable_pose_student_head = None
    if segnet_distillation_weight > 0.0:
        scorer_teacher = build_mlx_segnet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=distillation_device,
        )
        learnable_student_head = build_learnable_student_head(
            num_classes=int(scorer_teacher.num_classes),
            seed=int(random_seed),
        )
    if pose_distillation_weight > 0.0:
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=scorer_upstream_dir,
            device=distillation_device,
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=int(pose_scorer_teacher.pose_dims),
            seed=int(random_seed) + 1,
        )
    bundle = RendererBundle(
        **bundle_kwargs,
        distillation_weight=float(segnet_distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        distillation_temperature=float(distillation_temperature),
        segnet_distillation_objective=segnet_distillation_objective,
        segnet_tau_boundary=float(segnet_tau_boundary),
        segnet_hinge_margin=float(segnet_hinge_margin),
        distillation_num_classes=(
            int(scorer_teacher.num_classes) if scorer_teacher is not None else 5
        ),
        pose_distillation_weight=float(pose_distillation_weight),
        pose_distillation_loss=str(pose_distillation_loss),
        pose_distillation_huber_delta=float(pose_distillation_huber_delta),
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=int(pose_scorer_teacher.pose_dims)
        if pose_scorer_teacher is not None
        else 6,
        allow_segnet_only_research=bool(allow_segnet_only_research),
    )
    return run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="compact_runner_pact_nerv_selector_v4_mlx",
        lane_id="lane_compact_renderer_mlx_spine_runner_selector_v4_20260601",
        output_dir=output_dir,
        epochs=int(epochs),
        batch_pair_indices_per_step=max(1, int(batch_pair_indices_per_step)),
        learning_rate=float(learning_rate),
        ema_decay=float(ema_decay),
        seed=int(random_seed),
        checkpoint_interval_epochs=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        telemetry_flush_interval_epochs=1,
        pr95_faithful_curriculum_enabled=pr95_curriculum_enabled,
        pr95_curriculum_total_epochs=max(8, int(epochs))
        if pr95_curriculum_enabled
        else None,
        grad_clip_max_norm=optimizer_control.get("grad_clip_max_norm"),
        weight_decay=effective_weight_decay,
        optimizer_kind=effective_optimizer_kind,
        warmup_epochs=int(optimizer_control.get("warmup_epochs", 0)),
        warmup_steps_per_epoch=max(
            1, int(optimizer_control.get("warmup_steps_per_epoch", 1))
        ),
        cosine_decay_enabled=bool(optimizer_control.get("cosine_decay_enabled")),
        cosine_decay_total_epochs=optimizer_control.get("cosine_decay_total_epochs"),
        cosine_decay_min_lr_ratio=float(
            optimizer_control.get("cosine_decay_min_lr_ratio", 1e-2)
        ),
        notes=(
            "Compact renderer MLX spine runner PACT-NeRV-SELECTOR-V4 smoke "
            "using real contest video targets, selector-v4 PSV4 archive export, "
            "receiver proof, explicit native optimizer controls, and "
            "false-authority MLX evidence only."
        ),
    )


def _coverage_manifest_extra(report: dict[str, Any]) -> dict[str, Any]:
    frames = _positive_int(report.get("source_video_frame_count"))
    if frames is None:
        frames = _positive_int(report.get("max_frames"))
    num_pairs = None if frames is None else frames // 2
    return {
        "num_frames": frames,
        "num_pairs": num_pairs,
        "coverage_source": "pr95_mlx_long_training_report",
        "coverage_note": (
            "max_frames smoke coverage is not full-video comparable; "
            "bounded runner must scale before base-byte promotion"
        ),
    }


def _select_latest_exported_checkpoint(
    report: dict[str, Any],
    *,
    base: Path,
) -> dict[str, Any]:
    rows = report.get("checkpoint_artifacts")
    if not isinstance(rows, list):
        raise CompactRendererMlxSpineRunnerError("checkpoint_artifacts_missing")
    candidates: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        if raw.get("pytorch_export_succeeded") is not True:
            continue
        if raw.get("trained_latents_exported") is not True:
            continue
        if not raw.get("pytorch_state_dict_path") or not raw.get("latents_path"):
            continue
        _resolve_existing(raw["pytorch_state_dict_path"], base=base)
        _resolve_existing(raw["latents_path"], base=base)
        candidates.append(raw)
    if not candidates:
        raise CompactRendererMlxSpineRunnerError(
            "no_checkpoint_with_exported_weights_and_latents"
        )
    return max(
        candidates,
        key=lambda row: (int(row.get("global_epoch") or 0), int(row.get("stage_index") or 0)),
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_positive_int_csv(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise CompactRendererMlxSpineRunnerError(
            "expected at least one positive integer in comma-separated list"
        )
    out = []
    for part in parts:
        try:
            parsed = int(part)
        except ValueError as exc:
            raise CompactRendererMlxSpineRunnerError(
                f"invalid positive integer {part!r} in comma-separated list"
            ) from exc
        if parsed < 1:
            raise CompactRendererMlxSpineRunnerError(
                f"invalid non-positive integer {part!r} in comma-separated list"
            )
        out.append(parsed)
    return tuple(out)


def _parse_nonnegative_int_csv(value: str) -> tuple[int, ...]:
    try:
        return parse_pair_indices_csv(value, field="prioritized_pair_indices")
    except HardPairIndicesError as exc:
        raise CompactRendererMlxSpineRunnerError(str(exc)) from exc


def _normalize_nonnegative_int_sequence(value: Sequence[Any] | None) -> tuple[int, ...]:
    try:
        return normalize_pair_indices(value, field="prioritized_pair_indices")
    except HardPairIndicesError as exc:
        raise CompactRendererMlxSpineRunnerError(str(exc)) from exc


def _prioritized_pair_indices_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    try:
        return merge_pair_indices(
            parse_pair_indices_csv(
                str(getattr(args, "prioritized_pair_indices", "") or ""),
                field="prioritized_pair_indices",
            ),
            load_pair_indices_file(
                getattr(args, "prioritized_pair_indices_file", None),
                base=getattr(args, "repo_root", REPO_ROOT),
                field="prioritized_pair_indices_file",
            ),
        )
    except HardPairIndicesError as exc:
        raise CompactRendererMlxSpineRunnerError(str(exc)) from exc


def _resolve_checkpoint_interval_epochs(value: Any, *, epochs: int) -> int:
    """Return a positive periodic checkpoint cadence for long compact runs."""

    if isinstance(value, bool):
        raise CompactRendererMlxSpineRunnerError(
            "checkpoint_interval_epochs must be a positive integer, not bool"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CompactRendererMlxSpineRunnerError(
            f"checkpoint_interval_epochs must be a positive integer; got {value!r}"
        ) from exc
    if parsed <= 0:
        raise CompactRendererMlxSpineRunnerError(
            f"checkpoint_interval_epochs must be > 0; got {parsed}"
        )
    if int(epochs) <= 0:
        raise CompactRendererMlxSpineRunnerError(f"epochs must be > 0; got {epochs!r}")
    return parsed


def _checkpoint_summary(row: dict[str, Any], *, base: Path) -> dict[str, Any]:
    summary = dict(row)
    for key in (
        "mlx_checkpoint_path",
        "pytorch_state_dict_path",
        "latents_path",
        "pytorch_export_manifest_path",
    ):
        value = row.get(key)
        if isinstance(value, str) and value:
            path = _resolve(value, base=base)
            summary[f"{key}_resolved"] = path.as_posix()
            if path.is_file():
                summary[f"{key}_bytes"] = path.stat().st_size
                summary[f"{key}_sha256"] = _sha256_file(path)
    return summary


def _trained_provenance(
    *,
    report_path: Path,
    checkpoint: dict[str, Any],
    role: str,
) -> str:
    return (
        f"role={role}; source=tools/run_pr95_mlx_long_training.py; "
        f"report={report_path.as_posix()}; report_sha256={_sha256_file(report_path)}; "
        f"stage={checkpoint.get('stage_index')}; global_epoch={checkpoint.get('global_epoch')}; "
        f"evidence_grade={checkpoint.get('evidence_grade')}; "
        "authority=macos_mlx_research_signal_false_authority"
    )


def _default_output_dir() -> Path:
    for root in DEFAULT_SSD_ROOTS:
        if root.exists():
            return root / "compact_renderer_mlx_spine_runner" / _stamp()
    raise CompactRendererMlxSpineRunnerError(
        "no SSD artifact root found; pass --output-dir explicitly"
    )


def _jsonable_lock_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.expanduser().resolve(strict=False).as_posix()
    if isinstance(value, tuple | list):
        return [_jsonable_lock_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable_lock_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    return value


def _resolve_optional_compact_family_path(
    value: str | Path | None,
    *,
    base: Path,
) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _snerv_advisory_source_pair_indices(
    advisory: Any,
    advisory_payload: Mapping[str, Any],
    *,
    requested_num_pairs: int,
) -> tuple[int, ...]:
    raw = getattr(advisory, "source_pair_indices", None)
    if raw is None:
        raw = advisory_payload.get("source_pair_indices")
    if raw is not None:
        return tuple(int(value) for value in raw)
    raw_count = getattr(
        advisory,
        "n_pairs",
        advisory_payload.get("n_pairs", requested_num_pairs),
    )
    count = max(0, int(raw_count))
    return tuple(range(count))


def _active_campaign_lock_payload(
    args: argparse.Namespace,
    *,
    source_video_path: Path,
    hard_byte_ceilings: tuple[int, ...],
) -> dict[str, Any]:
    """Return a normalized campaign identity, excluding artifact destination."""

    excluded = {"allow_duplicate_campaign", "output_dir", "overwrite", "repo_root"}
    argv_payload = {
        key: _jsonable_lock_value(value)
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }
    recon_weight = getattr(args, "recon_pixel_weight_path", None)
    recon_weight_sha256 = None
    if recon_weight is not None:
        recon_path = Path(recon_weight).expanduser().resolve(strict=False)
        if recon_path.is_file():
            recon_weight_sha256 = _sha256_file(recon_path)
    auto_joint_recon_weight = getattr(args, "auto_joint_recon_pixel_weight", False)
    auto_joint_recon_weight_sha256 = None
    auto_joint_recon_weight_path = None
    auto_joint_recon_weight_error = None
    if auto_joint_recon_weight:
        try:
            discovered, _discovery = _discover_joint_recon_pixel_weight_path(
                repo_root=getattr(args, "repo_root", REPO_ROOT),
                num_pairs=int(getattr(args, "num_pairs", 0)),
            )
            auto_joint_recon_weight_path = discovered.as_posix()
            auto_joint_recon_weight_sha256 = _sha256_file(discovered)
        except Exception as exc:
            auto_joint_recon_weight_error = f"{type(exc).__name__}:{exc!s}"
    return {
        "schema": "compact_renderer_campaign_identity.v1",
        "argv": argv_payload,
        "hard_byte_ceilings": [int(value) for value in hard_byte_ceilings],
        "source_video_path_resolved": source_video_path.as_posix(),
        "recon_pixel_weight_sha256": recon_weight_sha256,
        "auto_joint_recon_pixel_weight_path": auto_joint_recon_weight_path,
        "auto_joint_recon_pixel_weight_sha256": auto_joint_recon_weight_sha256,
        "auto_joint_recon_pixel_weight_error": auto_joint_recon_weight_error,
    }


def _campaign_lock_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _active_process_table_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,etime=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 4:
            continue
        pid_text, ppid_text, elapsed, command = parts
        try:
            pid = int(pid_text)
            ppid = int(ppid_text)
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "elapsed": elapsed,
                "command": command,
            }
        )
    return rows


def _active_family_process_needles(family: str) -> tuple[str, ...]:
    family_token = str(family or "").strip()
    generic = (
        f"--execute-family {family_token}",
        f"--execute-family={family_token}",
    )
    if family_token == "snerv":
        return (
            *generic,
            "tools/run_snerv_inverse_steg_advisory.py",
            "run_snerv_inverse_steg_advisory.py",
        )
    if family_token == "hi_nerv":
        return generic
    return generic


def _active_family_campaign_processes(
    *,
    family: str | None,
    current_pid: int,
    process_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not family:
        return []
    needles = _active_family_process_needles(str(family))
    if not needles:
        return []
    rows = process_rows if process_rows is not None else _active_process_table_rows()
    by_pid: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            by_pid[int(row.get("pid") or -1)] = row
        except (TypeError, ValueError):
            continue
    excluded_pids = {int(current_pid)}
    cursor = int(current_pid)
    while True:
        parent = by_pid.get(cursor, {}).get("ppid")
        try:
            parent_pid = int(parent)
        except (TypeError, ValueError):
            break
        if parent_pid <= 0 or parent_pid in excluded_pids:
            break
        excluded_pids.add(parent_pid)
        cursor = parent_pid
    matches: list[dict[str, Any]] = []
    for row in rows:
        try:
            pid = int(row.get("pid") or -1)
        except (TypeError, ValueError):
            continue
        if pid <= 0 or pid in excluded_pids:
            continue
        if not _pid_is_alive(pid):
            continue
        command = str(row.get("command") or "")
        if not any(needle and needle in command for needle in needles):
            continue
        matches.append(
            {
                "pid": pid,
                "ppid": row.get("ppid"),
                "elapsed": row.get("elapsed"),
                "command": command,
            }
        )
    return matches


def _planner_row_launch_blockers(args: argparse.Namespace) -> list[str]:
    """Return fail-closed blockers for direct launches of top-priority carriers."""

    return list(_planner_row_launch_guard(args).get("blockers") or [])


def _planner_row_launch_guard(args: argparse.Namespace) -> dict[str, Any]:
    family = str(getattr(args, "execute_family", "") or "").strip()
    row_id = str(getattr(args, "planner_row_id", "") or "").strip()
    manual_launch_allowed = bool(
        getattr(args, "allow_manual_compact_family_launch", False)
    )
    guard: dict[str, Any] = {
        "schema": "compact_carrier_planner_row_launch_guard.v1",
        "required_families": list(PLANNER_ROW_REQUIRED_FAMILIES),
        "execute_family": family or None,
        "planner_row_id": row_id or None,
        "allow_manual_compact_family_launch": manual_launch_allowed,
        "queue_artifact_status": None,
        "timing_smoke_waiver_status": None,
        "passed": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if family not in PLANNER_ROW_REQUIRED_FAMILIES:
        return guard
    if not row_id:
        if manual_launch_allowed:
            guard["manual_launch_false_authority"] = True
            return guard
        guard["passed"] = False
        guard["blockers"] = [
            f"{family}_planner_row_id_missing",
            (
                "top_priority_compact_carrier_launch_must_come_from_"
                "nerv_long_training_campaign_plan"
            ),
        ]
        return guard

    queue_status = _planner_row_queue_artifact_status(args, family=family, row_id=row_id)
    guard["queue_artifact_status"] = queue_status
    if queue_status.get("passed") is True:
        return guard

    waiver_status = _planner_row_timing_smoke_waiver_status(
        args,
        family=family,
        row_id=row_id,
    )
    guard["timing_smoke_waiver_status"] = waiver_status
    if waiver_status.get("passed") is True:
        guard["bounded_timing_smoke_waiver_consumed"] = True
        return guard

    guard["passed"] = False
    guard["blockers"] = _dedupe(
        [
            *list(queue_status.get("blockers") or []),
            *list(waiver_status.get("blockers") or []),
        ]
    )
    return guard


def _planner_row_queue_artifact_status(
    args: argparse.Namespace,
    *,
    family: str,
    row_id: str,
) -> dict[str, Any]:
    paths = _planner_row_queue_artifact_paths(args)
    status: dict[str, Any] = {
        "schema": "compact_carrier_planner_row_queue_artifact_status.v1",
        "family": family,
        "planner_row_id": row_id,
        "artifact_paths": [Path(path).as_posix() for path in paths],
        "artifact_records": [],
        "matched_records": [],
        "passed": False,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if not paths:
        status["blockers"] = [
            f"{family}_planner_row_queue_artifact_missing",
            "planner_row_queue_artifact_required_for_planner_row_launch",
        ]
        return status

    repo_root = Path(getattr(args, "repo_root", REPO_ROOT)).expanduser().resolve(
        strict=False
    )
    records: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for raw_path in paths:
        record = _planner_row_queue_artifact_record(
            raw_path,
            family=family,
            row_id=row_id,
            repo_root=repo_root,
            args=args,
        )
        records.append(record)
        matches.extend(
            row for row in record.get("matched_records", []) if isinstance(row, dict)
        )
    runnable_matches = [
        row
        for row in matches
        if row.get("row_status_runnable") is True
        and row.get("launch_contract_runnable") is True
        and not row.get("command_control_blockers")
    ]
    status["artifact_records"] = records
    status["matched_records"] = matches
    if runnable_matches:
        status["passed"] = True
        status["matched_runnable_records"] = runnable_matches
        return status
    blockers: list[str] = []
    for record in records:
        blockers.extend(str(item) for item in record.get("blockers") or [])
    if not matches:
        blockers.append(f"{family}_planner_row_id_not_found_in_queue_artifact")
    else:
        for match in matches:
            blockers.extend(
                str(item) for item in match.get("command_control_blockers") or []
            )
        blockers.append(f"{family}_planner_row_queue_artifact_not_queued_or_runnable")
    status["blockers"] = _dedupe(blockers)
    return status


def _planner_row_queue_artifact_paths(args: argparse.Namespace) -> list[Path]:
    raw = getattr(args, "planner_row_queue_artifact", []) or []
    if isinstance(raw, (str, os.PathLike)):
        raw = [raw]
    return [Path(path).expanduser() for path in raw if str(path)]


def _planner_row_queue_artifact_record(
    path: str | Path,
    *,
    family: str,
    row_id: str,
    repo_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved = _resolve(path, base=repo_root)
    record: dict[str, Any] = {
        "schema": "compact_carrier_planner_row_queue_artifact_record.v1",
        "path": resolved.as_posix(),
        "exists": resolved.is_file(),
        "bytes": None,
        "sha256": None,
        "source_schema": None,
        "matched_records": [],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if not resolved.is_file():
        record["blockers"].append("planner_row_queue_artifact_path_missing")
        return record
    record["bytes"] = int(resolved.stat().st_size)
    record["sha256"] = _sha256_file(resolved)
    try:
        payload = _load_json(resolved)
    except (OSError, json.JSONDecodeError, CompactRendererMlxSpineRunnerError) as exc:
        record["blockers"].append(
            f"planner_row_queue_artifact_unreadable:{type(exc).__name__}"
        )
        return record
    record["source_schema"] = payload.get("schema")
    if payload.get("schema") not in PLANNER_ROW_QUEUE_ARTIFACT_SCHEMAS:
        record["blockers"].append("planner_row_queue_artifact_schema_not_allowed")
        return record
    matched = _planner_row_records_from_payload(
        payload,
        family=family,
        row_id=row_id,
        artifact_path=resolved,
        args=args,
    )
    record["matched_records"] = matched
    if not matched:
        record["blockers"].append("planner_row_queue_artifact_no_matching_row")
    return record


def _planner_row_records_from_payload(
    payload: Mapping[str, Any],
    *,
    family: str,
    row_id: str,
    artifact_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(payload.get("experiment_queue"), Mapping):
        candidates.extend(
            _planner_row_experiment_candidates(
                payload["experiment_queue"],
                context="experiment_queue",
            )
        )
    candidates.extend(_planner_row_experiment_candidates(payload, context="root"))
    rows = payload.get("campaign_rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for index, row in enumerate(rows):
            if isinstance(row, Mapping):
                candidates.append((f"campaign_rows[{index}]", row))
                entry = row.get("experiment_queue_entry")
                if isinstance(entry, Mapping):
                    candidates.append(
                        (f"campaign_rows[{index}].experiment_queue_entry", entry)
                    )

    matches: list[dict[str, Any]] = []
    for context, row in candidates:
        command = _planner_row_command(row)
        row_id_values = _planner_row_identity_values(row, command)
        if row_id not in row_id_values:
            continue
        if family and str(row.get("family") or "") not in {"", family}:
            continue
        status_text = str(row.get("status") or "").strip().lower()
        blocked = bool(row.get("blocked"))
        contract = _planner_row_launch_contract(row)
        contract_schema_valid = (
            contract.get("schema") == PLANNER_ROW_LAUNCH_CONTRACT_SCHEMA
        )
        launch_blockers = [
            str(item) for item in (contract.get("queue_launch_blockers") or [])
        ]
        if not contract_schema_valid:
            launch_blockers.append("planner_row_launch_contract_schema_mismatch")
        row_status_runnable = (
            status_text in PLANNER_ROW_QUEUE_RUNNABLE_STATUSES and not blocked
        )
        command_control_blockers = _planner_row_command_control_blockers(
            args,
            command,
        )
        launch_contract_runnable = (
            contract_schema_valid
            and contract.get("queue_status_is_local_mlx_plan") is True
            and contract.get("queue_status_is_runnable_plan") is True
            and not launch_blockers
            and contract.get("queue_status_is_receiver_proof") is not True
            and contract.get("queue_status_is_cpu_replay_proof") is not True
            and contract.get("queue_status_is_exact_eval_authority") is not True
        )
        matches.append(
            {
                "schema": "compact_carrier_planner_row_queue_match.v1",
                "artifact_path": artifact_path.as_posix(),
                "context": context,
                "row_id": row_id,
                "family": str(row.get("family") or family),
                "status": status_text or None,
                "blocked": blocked,
                "row_status_runnable": row_status_runnable,
                "launch_contract_runnable": launch_contract_runnable,
                "launch_contract_schema": contract.get("schema"),
                "launch_contract_schema_valid": contract_schema_valid,
                "launch_contract_blockers": launch_blockers,
                "command_control_blockers": command_control_blockers,
                "command": command,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return matches


def _planner_row_experiment_candidates(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> list[tuple[str, Mapping[str, Any]]]:
    experiments = payload.get("experiments")
    if not isinstance(experiments, Sequence) or isinstance(experiments, (str, bytes)):
        return []
    out: list[tuple[str, Mapping[str, Any]]] = []
    for index, experiment in enumerate(experiments):
        if isinstance(experiment, Mapping):
            out.append((f"{context}.experiments[{index}]", experiment))
            metadata = experiment.get("metadata")
            if isinstance(metadata, Mapping):
                source_row = metadata.get("source_selected_row")
                if isinstance(source_row, Mapping):
                    merged_source_row = dict(source_row)
                    for key in ("status", "blocked", "steps", "command", "family"):
                        if key not in merged_source_row and key in experiment:
                            merged_source_row[key] = experiment[key]
                    out.append(
                        (
                            f"{context}.experiments[{index}].metadata.source_selected_row",
                            merged_source_row,
                        )
                    )
    return out


def _planner_row_identity_values(
    row: Mapping[str, Any],
    command: Sequence[str],
) -> set[str]:
    values = {
        str(row.get("row_id") or "").strip(),
        str(row.get("planner_row_id") or "").strip(),
        str(row.get("id") or "").strip(),
    }
    if "--planner-row-id" in command:
        try:
            values.add(str(command[command.index("--planner-row-id") + 1]).strip())
        except IndexError:
            pass
    return {value for value in values if value}


def _planner_row_command(row: Mapping[str, Any]) -> list[str]:
    command = row.get("command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        return [str(item) for item in command]
    command_argv = row.get("command_argv")
    if isinstance(command_argv, Sequence) and not isinstance(command_argv, (str, bytes)):
        return [str(item) for item in command_argv]
    steps = row.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            step_command = step.get("command")
            if isinstance(step_command, Sequence) and not isinstance(
                step_command, (str, bytes)
            ):
                return [str(item) for item in step_command]
    return []


def _planner_row_launch_contract(row: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = row.get("launch_authority_contract")
    if isinstance(contract, Mapping):
        return contract
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        source_row = metadata.get("source_selected_row")
        if isinstance(source_row, Mapping):
            contract = source_row.get("launch_authority_contract")
            if isinstance(contract, Mapping):
                return contract
    return {}


def _planner_row_command_control_blockers(
    args: argparse.Namespace,
    command: Sequence[str],
) -> list[str]:
    flag_attrs = (
        ("--execute-family", "execute_family"),
        ("--modelsize-candidate-id", "modelsize_candidate_id"),
        ("--epochs", "epochs"),
        ("--num-pairs", "num_pairs"),
        ("--optimizer-kind", "optimizer_kind"),
        ("--target-modelsize-mparams", "target_modelsize_mparams"),
        ("--snerv-official-modelsize-mparams", "snerv_official_modelsize_mparams"),
        ("--snerv-modelsize-control-profile", "snerv_modelsize_control_profile"),
        ("--snerv-official-enc-strds", "snerv_official_enc_strds"),
        ("--snerv-official-dec-strds", "snerv_official_dec_strds"),
        ("--hi-nerv-optimizer-policy", "hi_nerv_optimizer_policy"),
        ("--decoder-weight-waterfill-plan-json", "decoder_weight_waterfill_plan_json"),
        ("--recon-pixel-weight-path", "recon_pixel_weight_path"),
        ("--snerv-model-size-adapter", "snerv_model_size_adapter"),
        ("--snerv-fc-dim", "snerv_fc_dim"),
        ("--snerv-emb-size", "snerv_emb_size"),
        ("--snerv-patch-radius", "snerv_patch_radius"),
        ("--snerv-mfu-scales", "snerv_mfu_scales"),
        ("--snerv-hfr-gain", "snerv_hfr_gain"),
        ("--snerv-temporal-context", "snerv_temporal_context"),
        ("--snerv-temporal-mode", "snerv_temporal_mode"),
    )
    blockers: list[str] = []
    for flag, attr in flag_attrs:
        expected = _command_flag_value(command, flag)
        if expected is None:
            continue
        actual = _namespace_flag_value(args, attr)
        if actual is None or str(actual) != str(expected):
            blockers.append(f"planner_row_command_mismatch:{flag}")
    return _dedupe(blockers)


def _command_flag_value(command: Sequence[str], flag: str) -> str | None:
    items = [str(item) for item in command]
    if flag not in items:
        return None
    index = items.index(flag)
    if index + 1 >= len(items):
        return ""
    return str(items[index + 1])


def _namespace_flag_value(args: argparse.Namespace, attr: str) -> str | None:
    if not hasattr(args, attr):
        return None
    value = getattr(args, attr)
    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value)


def _planner_row_timing_smoke_waiver_status(
    args: argparse.Namespace,
    *,
    family: str,
    row_id: str,
) -> dict[str, Any]:
    waiver_requested = bool(
        getattr(args, "allow_bounded_planner_row_timing_smoke_waiver", False)
    )
    num_pairs = _optional_int(getattr(args, "num_pairs", None))
    epochs = _optional_int(getattr(args, "epochs", None))
    within_bounds = (
        num_pairs is not None
        and epochs is not None
        and 0 < num_pairs <= PLANNER_ROW_TIMING_SMOKE_MAX_PAIRS
        and 0 < epochs <= PLANNER_ROW_TIMING_SMOKE_MAX_EPOCHS
    )
    blockers: list[str] = []
    if not waiver_requested:
        blockers.append("bounded_planner_row_timing_smoke_waiver_missing")
    elif not within_bounds:
        blockers.append(f"{family}_bounded_timing_smoke_waiver_exceeds_limits")
    return {
        "schema": "compact_carrier_bounded_timing_smoke_waiver_status.v1",
        "family": family,
        "planner_row_id": row_id,
        "waiver_requested": waiver_requested,
        "num_pairs": num_pairs,
        "epochs": epochs,
        "max_num_pairs": PLANNER_ROW_TIMING_SMOKE_MAX_PAIRS,
        "max_epochs": PLANNER_ROW_TIMING_SMOKE_MAX_EPOCHS,
        "passed": waiver_requested and within_bounds,
        "blockers": blockers,
        "authority": "bounded_timing_smoke_false_authority_not_production_launch",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_planner_row_launch_refusal(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    blockers: list[str],
    hard_byte_ceilings: tuple[int, ...],
    repo_root: Path,
) -> dict[str, Any]:
    """Write a normal runner report for planner-custody launch refusals."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report = _base_report(
        output_dir=output_dir,
        mode="compact_carrier_planner_row_launch_refused",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=repo_root,
    )
    family = str(getattr(args, "execute_family", "") or "").strip()
    report.update(
        {
            "execute_family": family,
            "training_executed": False,
            "trainer_launch_allowed": False,
            "launch_refusal_reason": (
                "HiNeRV/SNeRV production launches must carry the planner row "
                "identity emitted by nerv_long_training_campaign_plan; direct "
                "manual launches require --allow-manual-compact-family-launch "
                "and remain false-authority research."
            ),
            "planner_launch_contract": {
                "schema": "compact_carrier_planner_launch_contract.v1",
                "required_families": list(PLANNER_ROW_REQUIRED_FAMILIES),
                "planner_row_id": str(
                    getattr(args, "planner_row_id", "") or ""
                ).strip()
                or None,
                "allow_manual_compact_family_launch": bool(
                    getattr(args, "allow_manual_compact_family_launch", False)
                ),
                "required_source": (
                    "tac.analysis.nerv_long_training_campaign_plan."
                    "build_nerv_long_training_campaign_plan"
                ),
                "launch_guard": _planner_row_launch_guard(args),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "blockers": _dedupe(blockers),
        }
    )
    path = output_dir / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, report)
    return {**report, "report_path": path.as_posix()}


def _write_compact_family_startup_marker(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    source_video_path: Path,
    hard_byte_ceilings: tuple[int, ...],
    modelsize_candidate: Mapping[str, Any] | None,
) -> Path | None:
    """Write launch custody before heavy scorer/teacher/training work starts."""

    family = str(getattr(args, "execute_family", "") or "").strip()
    if family not in PLANNER_ROW_REQUIRED_FAMILIES:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_identity = _active_campaign_lock_payload(
        args,
        source_video_path=source_video_path,
        hard_byte_ceilings=hard_byte_ceilings,
    )
    payload = {
        "schema": "compact_carrier_startup_marker.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "campaign_identity": campaign_identity,
        "auto_joint_recon_pixel_weight_path": campaign_identity.get(
            "auto_joint_recon_pixel_weight_path"
        ),
        "auto_joint_recon_pixel_weight_sha256": campaign_identity.get(
            "auto_joint_recon_pixel_weight_sha256"
        ),
        "auto_joint_recon_pixel_weight_error": campaign_identity.get(
            "auto_joint_recon_pixel_weight_error"
        ),
        "execute_family": family,
        "planner_row_id": str(getattr(args, "planner_row_id", "") or "").strip()
        or None,
        "modelsize_candidate_id": str(
            getattr(args, "modelsize_candidate_id", "") or ""
        ).strip()
        or None,
        "modelsize_candidate": _jsonable_lock_value(dict(modelsize_candidate or {})),
        "output_dir": output_dir.as_posix(),
        "source_video_path": source_video_path.as_posix(),
        "hard_byte_ceilings": list(hard_byte_ceilings),
        "requested_distillation_device": str(
            getattr(
                args,
                "requested_distillation_device",
                getattr(args, "distillation_device", ""),
            )
            or ""
        ),
        "distillation_device": str(getattr(args, "distillation_device", "") or ""),
        "mlx_prefilter_scorer_device": str(
            getattr(args, "mlx_prefilter_scorer_device", "") or ""
        ),
        "mlx_prefilter_scorer_batch_pairs": int(
            getattr(args, "mlx_prefilter_scorer_batch_pairs", 1) or 1
        ),
        "mlx_prefilter_progress_every": int(
            getattr(args, "mlx_prefilter_progress_every", 50) or 50
        ),
        "command_args": _jsonable_lock_value(vars(args)),
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "false_authority_flags": [
            "macos_mlx_research_signal_until_archive_receiver_and_exact_eval",
            "startup_marker_before_trained_export_or_full_video_replay",
        ],
    }
    path = output_dir / COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    _write_json(path, payload)
    return path


def _compact_family_interruption_blockers(family: str) -> list[str]:
    blockers = [
        "compact_renderer_run_interrupted_before_terminal_report",
        "byte_closed_archive_export_missing",
        "receiver_proof_missing",
        "full_video_local_prefilter_missing",
        "local_cpu_replay_gate_missing",
        "paired_contest_cpu_cuda_pass_missing",
    ]
    if family == "hi_nerv":
        blockers.extend(
            [
                "hi_nerv_training_interrupted_before_export",
                "hi_nerv_receiver_proof_missing",
                "hi_nerv_full_video_local_prefilter_missing",
                "hi_nerv_local_cpu_replay_gate_missing",
            ]
        )
    elif family == "snerv":
        blockers.extend(
            [
                "snerv_training_interrupted_before_export",
                "snerv_receiver_proof_missing",
                "snerv_full_video_local_prefilter_missing",
                "snerv_local_cpu_replay_gate_missing",
            ]
        )
    return _dedupe(blockers)


def _compact_family_interruption_evidence_files(output_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    names = {
        COMPACT_FAMILY_STARTUP_MARKER_FILENAME,
        "telemetry.jsonl",
        "training_artifact.json",
        "decoder_weight_gradient_saliency.json",
        "final_checkpoint_emission_failed.json",
    }
    checkpoint_suffixes = (
        ".meta.json",
        ".live.state",
        ".live.state.npsd",
        ".live.state.npz",
        ".ema_shadow.state",
        ".ema_shadow.state.npsd",
        ".ema_shadow.state.npz",
        ".ema_kahan_compensation.pkl",
    )
    if not output_dir.exists():
        return files
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        is_named_evidence = path.name in names
        is_checkpoint_evidence = (
            path.parent.name == "checkpoints"
            and any(path.name.endswith(suffix) for suffix in checkpoint_suffixes)
        )
        if not (is_named_evidence or is_checkpoint_evidence):
            continue
        try:
            files.append(
                {
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        except OSError:
            files.append(
                {
                    "path": path.as_posix(),
                    "error": "stat_or_hash_failed",
                }
            )
    return files


def _write_compact_family_interrupted_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    source_video_path: Path,
    hard_byte_ceilings: tuple[int, ...],
    modelsize_candidate: Mapping[str, Any] | None,
    signum: int | None,
    reason: str,
) -> dict[str, Any]:
    """Persist false-authority custody when a long compact run is interrupted."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "compact_renderer_mlx_spine_runner_report.json"
    if path.is_file():
        existing = _load_json(path)
        return {**existing, "report_path": path.as_posix()}
    family = str(getattr(args, "execute_family", "") or "").strip() or None
    signal_name = None
    if signum is not None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"SIG{signum}"
    telemetry_summary = _compact_family_telemetry_summary(output_dir)
    training_executed = bool(telemetry_summary.get("row_count"))
    campaign_identity = _active_campaign_lock_payload(
        args,
        source_video_path=source_video_path,
        hard_byte_ceilings=hard_byte_ceilings,
    )
    report = {
        "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
        "mode": "interrupted_compact_family_run",
        "created_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "signal": signum,
        "signal_name": signal_name,
        "interruption_reason": reason,
        "execute_family": family,
        "planner_row_id": str(getattr(args, "planner_row_id", "") or "").strip()
        or None,
        "modelsize_candidate_id": str(
            getattr(args, "modelsize_candidate_id", "") or ""
        ).strip()
        or None,
        "modelsize_candidate": _jsonable_lock_value(dict(modelsize_candidate or {})),
        "campaign_identity": campaign_identity,
        "output_dir": output_dir.as_posix(),
        "source_video_path": source_video_path.as_posix(),
        "hard_byte_ceilings": list(hard_byte_ceilings),
        "command_args": _jsonable_lock_value(vars(args)),
        "evidence_files": _compact_family_interruption_evidence_files(output_dir),
        "telemetry_summary": telemetry_summary,
        "training_executed": training_executed,
        "training_started": True,
        "score_authority": "false_macos_mlx_research_signal",
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "false_authority_flags": [
            "interrupted_before_terminal_export_or_receiver_proof",
            "macos_mlx_research_signal_until_archive_receiver_and_exact_eval",
        ],
        "blockers": _compact_family_interruption_blockers(family or ""),
    }
    _write_json(path, report)
    return {**report, "report_path": path.as_posix()}


def _write_compact_family_interrupted_report_from_startup_marker(
    *,
    output_dir: Path,
    reason: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recover fail-closed terminal custody from startup+telemetry artifacts."""

    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    report_path = output_dir / "compact_renderer_mlx_spine_runner_report.json"
    if report_path.is_file() and not overwrite:
        existing = _load_json(report_path)
        return {**existing, "report_path": report_path.as_posix(), "recovered": False}
    startup_path = output_dir / COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    if not startup_path.is_file():
        raise CompactRendererMlxSpineRunnerError(
            f"startup marker missing for interrupted run recovery: {startup_path}"
        )
    startup = _load_json(startup_path)
    family = str(startup.get("execute_family") or "").strip() or None
    telemetry_summary = _compact_family_telemetry_summary(output_dir)
    training_started = bool(telemetry_summary.get("row_count"))
    report = {
        "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
        "mode": "recovered_interrupted_compact_family_run",
        "created_utc": datetime.now(UTC).isoformat(),
        "recovered": True,
        "recovery_reason": str(reason),
        "interruption_reason": str(reason),
        "startup_marker_path": startup_path.as_posix(),
        "startup_marker_sha256": _sha256_file(startup_path),
        "startup_marker": _jsonable_lock_value(startup),
        "pid": startup.get("pid"),
        "execute_family": family,
        "planner_row_id": startup.get("planner_row_id"),
        "modelsize_candidate_id": startup.get("modelsize_candidate_id"),
        "modelsize_candidate": startup.get("modelsize_candidate"),
        "campaign_identity": startup.get("campaign_identity"),
        "output_dir": output_dir.as_posix(),
        "source_video_path": startup.get("source_video_path"),
        "hard_byte_ceilings": list(startup.get("hard_byte_ceilings") or []),
        "command_args": startup.get("command_args"),
        "evidence_files": _compact_family_interruption_evidence_files(output_dir),
        "telemetry_summary": telemetry_summary,
        "training_executed": training_started,
        "training_started": training_started,
        "score_authority": "false_macos_mlx_research_signal",
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "false_authority_flags": [
            "recovered_interrupted_run_before_terminal_export_or_receiver_proof",
            "macos_mlx_research_signal_until_archive_receiver_and_exact_eval",
        ],
        "blockers": _compact_family_interruption_blockers(family or ""),
    }
    _write_json(report_path, report)
    return {**report, "report_path": report_path.as_posix()}


def _compact_family_telemetry_summary(output_dir: Path) -> dict[str, Any]:
    candidates = (
        output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl",
        output_dir / "snerv_mlx_training" / "telemetry.jsonl",
        output_dir / "telemetry.jsonl",
    )
    telemetry_path = next((path for path in candidates if path.is_file()), None)
    if telemetry_path is None:
        return {
            "schema": "compact_family_interrupted_telemetry_summary.v1",
            "present": False,
            "row_count": 0,
        }
    first_row: dict[str, Any] | None = None
    last_row: dict[str, Any] | None = None
    row_count = 0
    with telemetry_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                row = {
                    "line_number": line_number,
                    "parse_error": "invalid_json",
                }
            row_dict = (
                dict(row)
                if isinstance(row, Mapping)
                else {"line_number": line_number, "non_mapping_row": True}
            )
            if first_row is None:
                first_row = row_dict
            last_row = row_dict
            row_count += 1
    loss_components = (
        dict(last_row.get("loss_components") or {}) if isinstance(last_row, Mapping) else {}
    )
    per_axis = (
        dict(last_row.get("per_axis_decomposition") or {})
        if isinstance(last_row, Mapping)
        else {}
    )
    return {
        "schema": "compact_family_interrupted_telemetry_summary.v1",
        "present": True,
        "path": telemetry_path.as_posix(),
        "bytes": telemetry_path.stat().st_size,
        "sha256": _sha256_file(telemetry_path),
        "row_count": row_count,
        "first_epoch": _optional_int(first_row.get("epoch") if first_row else None),
        "last_epoch": _optional_int(last_row.get("epoch") if last_row else None),
        "last_captured_at_utc": (last_row or {}).get("captured_at_utc"),
        "last_learning_rate": _optional_float((last_row or {}).get("learning_rate")),
        "last_loss": _optional_float((last_row or {}).get("loss")),
        "last_per_axis_decomposition": _jsonable_lock_value(per_axis),
        "last_loss_components": {
            "pr95_stage_index": _optional_float(loss_components.get("pr95_stage_index")),
            "pr95_stage_uses_muon": _optional_float(
                loss_components.get("pr95_stage_uses_muon")
            ),
            "loss_part_distill": _optional_float(loss_components.get("loss_part_distill")),
            "loss_part_pose_distill": _optional_float(
                loss_components.get("loss_part_pose_distill")
            ),
            "loss_part_pr95_c1a_entropy": _optional_float(
                loss_components.get("loss_part_pr95_c1a_entropy")
            ),
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_active_family_process_refusal(
    *,
    lock_dir: Path,
    output_dir: Path,
    family: str,
    conflicts: list[dict[str, Any]],
) -> Path:
    path = lock_dir / f"family_process_refusal_{family}_{_stamp()}.json"
    payload = {
        "schema": ACTIVE_FAMILY_PROCESS_REFUSAL_SCHEMA,
        "family": family,
        "created_utc": datetime.now(UTC).isoformat(),
        "output_dir": output_dir.as_posix(),
        "active_processes": conflicts,
        "refusal_reason": "active_same_family_process_detected",
        "override": "pass --allow-duplicate-campaign only when intentional",
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "false_authority_flags": [
            "local_process_coordination_refusal_no_score_authority",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _release_active_campaign_lock(lock_path: Path, pid: int) -> None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if int(payload.get("pid") or -1) == int(pid):
        lock_path.unlink(missing_ok=True)


def _acquire_active_campaign_lock(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    source_video_path: Path,
    hard_byte_ceilings: tuple[int, ...],
) -> Path | None:
    """Acquire an atomic duplicate-campaign lock for expensive runs."""

    if bool(getattr(args, "allow_duplicate_campaign", False)):
        return None
    lock_dir = output_dir.parent / ".active_compact_renderer_campaign_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    family = str(getattr(args, "execute_family", "") or "").strip()
    family_conflicts = _active_family_campaign_processes(
        family=family,
        current_pid=os.getpid(),
    )
    if family_conflicts:
        refusal_path = _write_active_family_process_refusal(
            lock_dir=lock_dir,
            output_dir=output_dir,
            family=family,
            conflicts=family_conflicts,
        )
        raise SystemExit(
            "active same-family compact-renderer campaign refused: "
            f"family={family} refusal={refusal_path} "
            f"active_pids={[row['pid'] for row in family_conflicts]}; pass "
            "--allow-duplicate-campaign only when this is intentional"
        ) from None
    payload = _active_campaign_lock_payload(
        args,
        source_video_path=source_video_path,
        hard_byte_ceilings=hard_byte_ceilings,
    )
    digest = _campaign_lock_digest(payload)
    lock_path = lock_dir / f"{digest}.json"
    manifest = {
        "schema": ACTIVE_CAMPAIGN_LOCK_SCHEMA,
        "digest": digest,
        "pid": os.getpid(),
        "created_utc": datetime.now(UTC).isoformat(),
        "output_dir": output_dir.as_posix(),
        "identity": payload,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"active campaign lock is unreadable: {lock_path}; "
                    "refusing duplicate launch"
                ) from exc
            existing_pid = int(existing.get("pid") or -1)
            if not _pid_is_alive(existing_pid):
                stale_path = lock_path.with_suffix(f".stale_{_stamp()}.json")
                lock_path.replace(stale_path)
                continue
            raise SystemExit(
                "duplicate active compact-renderer campaign refused: "
                f"lock={lock_path} existing_pid={existing_pid} "
                f"existing_output={existing.get('output_dir')}; pass "
                "--allow-duplicate-campaign only when this is intentional"
            ) from None
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
                fh.write("\n")
            break
    atexit.register(_release_active_campaign_lock, lock_path, os.getpid())
    return lock_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from-pr95-mlx-report", type=Path)
    parser.add_argument(
        "--from-pr95-stage8-report",
        type=Path,
        help="Adapt a source-faithful PR95 Stage-8 report into the compact spine.",
    )
    parser.add_argument(
        "--from-snerv-advisory-report",
        type=Path,
        help=(
            "Adapt an existing SNeRV advisory/package report into the compact "
            "spine without rerunning SNeRV."
        ),
    )
    parser.add_argument("--execute-pr95-mlx-smoke", action="store_true")
    parser.add_argument(
        "--execute-pr95-stage8-source",
        action="store_true",
        help="Run the public PR95 Stage-8 source lane, then adapt its archive.",
    )
    parser.add_argument(
        "--execute-family",
        choices=CLI_EXECUTE_FAMILIES,
        help=(
            "Execute a real MLX compact-family training/export row, or emit a "
            "planner-owned refusal for top-priority families whose adapters are "
            "not real yet."
        ),
    )
    parser.add_argument(
        "--pr95-source-archive",
        default=DEFAULT_PR95_SOURCE_ARCHIVE_ZIP,
        type=Path,
        help="Public PR95 archive.zip used to seed --execute-family pr95_hnerv.",
    )
    parser.add_argument(
        "--run-receiver-proof",
        action="store_true",
        help=(
            "For PR95/HNeRV execution, run the public PR95 inflate.sh against "
            "the exported archive and emit a receiver-proof report."
        ),
    )
    parser.add_argument(
        "--pr95-receiver-runtime-dir",
        default=DEFAULT_PR95_RECEIVER_RUNTIME_DIR,
        type=Path,
        help="Runtime directory containing PR95 inflate.sh for receiver proof.",
    )
    parser.add_argument(
        "--keep-receiver-proof-output",
        action="store_true",
        help="Retain raw receiver-proof output instead of certify-and-delete.",
    )
    parser.add_argument(
        "--receiver-proof-timeout-seconds",
        default=1800,
        type=int,
    )
    parser.add_argument(
        "--upstream-dir",
        default=DEFAULT_UPSTREAM_DIR,
        type=Path,
        help=(
            "Pinned contest upstream snapshot for real scorer teachers "
            "(modules.py plus scorer safetensors). Separate from --repo-root "
            "so clean SSD code worktrees can reuse the canonical upstream bytes; "
            "defaults to $TAC_UPSTREAM_DIR when set."
        ),
    )
    parser.add_argument(
        "--source-video-path",
        default=DEFAULT_SOURCE_VIDEO_PATH,
        type=Path,
        help=(
            "Contest video path. The default upstream/videos/0.mkv resolves "
            "through --upstream-dir when the repo-local upstream tree is absent."
        ),
    )
    parser.add_argument("--max-frames", default=4, type=int)
    parser.add_argument("--num-pairs", default=2, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-pairs", default=1, type=int)
    parser.add_argument(
        "--prioritized-pair-indices",
        default="",
        help=(
            "Comma-separated hard-pair/sensitivity pair indices to emphasize "
            "inside MLX training batches. This is false-authority training "
            "sampling only; full-video MLX prefilter and CPU replay gates still "
            "decide promotion."
        ),
    )
    parser.add_argument(
        "--prioritized-pair-indices-file",
        type=Path,
        default=None,
        help=(
            "JSON/list/text artifact containing prioritized_pair_indices, "
            "hard_pair_indices, pair_indices, or sample-generalization "
            "hard-pair coverage to emphasize in MLX batches."
        ),
    )
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument(
        "--optimizer-kind",
        choices=SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
        default=DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
        help=(
            "Optimizer control for score-aware compact training. Default "
            "pact_muon_adamw is Pact's PR95-derived Muon+AdamW partitioned "
            "adapter; other values are direct MLX optimizer baselines. "
            "HiNeRV, PACT-NeRV-VQ, and selector-v4 consume this directly; "
            "SNeRV consumes it once the shared long-training harness owns "
            "the carrier."
        ),
    )
    parser.add_argument(
        "--hi-nerv-optimizer-policy",
        choices=HI_NERV_OPTIMIZER_POLICIES,
        default="auto",
        help=(
            "HiNeRV optimizer authority. auto uses the PR95 8-stage "
            "Muon+AdamW curriculum only for long AdamW control rows. The "
            "default pact_muon_adamw is consumed directly by the shared "
            "score-aware adapter as Pact's partitioned Muon+AdamW optimizer; "
            "native_optimizer forces --optimizer-kind to be consumed directly."
        ),
    )
    parser.add_argument(
        "--optimizer-grad-clip-max-norm",
        default=DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_GRAD_CLIP_MAX_NORM,
        type=float,
        help=(
            "MLX score-aware optimizer gradient clip. Applies to the Pact "
            "Muon+AdamW default and native optimizer controls."
        ),
    )
    parser.add_argument(
        "--optimizer-weight-decay",
        default=None,
        type=float,
        help=(
            "Optional explicit decoupled weight decay for optimizer kinds that "
            "support it. Omitted means the runner applies the Pact default "
            "1e-4 for supported optimizers and None for no-decay controls."
        ),
    )
    parser.add_argument("--optimizer-warmup-epochs", default=0, type=int)
    parser.add_argument("--optimizer-warmup-steps-per-epoch", default=1, type=int)
    parser.add_argument(
        "--optimizer-cosine-decay-enabled",
        action="store_true",
        help=(
            "Enable MLX warmup plus cosine decay for optimizer controls. "
            "Requires --optimizer-warmup-epochs > 0."
        ),
    )
    parser.add_argument(
        "--optimizer-cosine-decay-total-epochs",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--optimizer-cosine-decay-min-lr-ratio",
        default=1e-2,
        type=float,
    )
    parser.add_argument("--compact-latent-dim", default=8, type=int)
    parser.add_argument("--compact-embed-dim", default=8, type=int)
    parser.add_argument("--compact-codebook-size", default=16, type=int)
    parser.add_argument("--compact-selector-palette-size", default=16, type=int)
    parser.add_argument("--compact-decoder-channel", default=8, type=int)
    parser.add_argument(
        "--compact-decoder-codec",
        default="portfolio_auto",
        choices=(
            "int8_mixed",
            "int8_scale_bundled",
            "portfolio_auto",
            "fp16_enveloped",
            "int4_mixed",
            "int4_scale_bundled",
            "int2_mixed",
            "int2_scale_bundled",
        ),
        help=(
            "Charged decoder-state archive codec for compact PACT/selector "
            "exports. Lower bit-depths are promotion-eligible only after "
            "receiver proof and full-video scorer-value replay."
        ),
    )
    parser.add_argument(
        "--hi-nerv-latent-codec",
        default="int16_raw",
        choices=("int16_raw", "int16_brotli_q11"),
        help=(
            "HiNeRV latent-section codec consumed by HIV1 parse/inflate. "
            "int16_brotli_q11 is lossless over the quantized latent int16 "
            "stream and reduces charged archive bytes without changing decoded "
            "latents."
        ),
    )
    parser.add_argument(
        "--modelsize-candidate-id",
        default="auto",
        help=(
            "HiNeRV/SNeRV launch candidate from the planner budget report. "
            "'auto' selects a byte-plausible planner candidate for the contest "
            "pair count; 'none'/'manual'/'off' uses the explicit compact CLI knobs."
        ),
    )
    parser.add_argument(
        "--target-modelsize-mparams",
        action="append",
        type=float,
        help=(
            "Shared operator-facing model-size target, in millions of params. "
            "For HiNeRV this selects the nearest local receiver-visible target "
            "row; for SNeRV this invokes the source-faithful --modelsize/fc_dim "
            "solve. Repeatable; still false-authority until archive bytes and "
            "receiver proof land."
        ),
    )
    parser.add_argument(
        "--hinerv-target-modelsize-mparams",
        action="append",
        type=float,
        help=(
            "For --execute-family hi_nerv and --modelsize-candidate-id auto, "
            "select the nearest local receiver-visible HiNeRV architecture row "
            "to this target parameter budget. Repeatable; false-authority until "
            "trained archive bytes and receiver proof land."
        ),
    )
    parser.add_argument(
        "--planner-row-id",
        default="",
        help=(
            "Queue/planner row identity emitted by "
            "nerv_long_training_campaign_plan for top-priority HiNeRV/SNeRV "
            "launch custody. Required for production launches."
        ),
    )
    parser.add_argument(
        "--planner-row-queue-artifact",
        action="append",
        default=[],
        type=Path,
        help=(
            "JSON artifact proving --planner-row-id is still queued/runnable. "
            "Accepts a nerv_long_training_campaign_plan payload, an "
            "experiment_queue.v1 payload, or an execution-admission payload "
            "with an embedded queue. Repeatable."
        ),
    )
    parser.add_argument(
        "--allow-bounded-planner-row-timing-smoke-waiver",
        action="store_true",
        help=(
            "Permit --planner-row-id without a queue artifact only for bounded "
            f"timing smokes (<= {PLANNER_ROW_TIMING_SMOKE_MAX_PAIRS} pairs and "
            f"<= {PLANNER_ROW_TIMING_SMOKE_MAX_EPOCHS} epochs). Long production "
            "runs still require --planner-row-queue-artifact."
        ),
    )
    parser.add_argument(
        "--allow-manual-compact-family-launch",
        action="store_true",
        help=(
            "Explicitly bypass the HiNeRV/SNeRV planner-row custody gate for "
            "research-only manual launches. Reports remain false-authority and "
            "not promotion-ready."
        ),
    )
    parser.add_argument(
        "--compact-ema-decay",
        default=0.9,
        type=float,
        help=(
            "Weight EMA decay for compact MLX archive export. Short byte-ceiling "
            "sweeps use a faster shadow than the long-run canonical 0.997 so "
            "receiver-proven archives do not export near-initial gray frames."
        ),
    )
    parser.add_argument(
        "--segnet-distillation-weight",
        default=0.0,
        type=float,
        help=(
            "Bind compact training to the real MLX SegNet teacher through the "
            "learnable student-head loss. False-authority MLX research signal only."
        ),
    )
    parser.add_argument(
        "--pose-distillation-weight",
        default=0.0,
        type=float,
        help=(
            "Bind compact training to the real PoseNet teacher through the "
            "learnable pose-head loss. Required for frontier-targeting "
            "score-aware compact runs."
        ),
    )
    parser.add_argument(
        "--pose-distillation-loss",
        choices=("mse", "huber"),
        default="mse",
        help=(
            "Train-time PoseNet-teacher loss. mse preserves the exact legacy "
            "surrogate; huber keeps the same small-error quadratic but bounds "
            "large-error gradients for pose-protected HiNeRV recovery runs."
        ),
    )
    parser.add_argument(
        "--pose-distillation-huber-delta",
        default=1.0,
        type=float,
        help="Positive Huber transition delta used by --pose-distillation-loss huber.",
    )
    parser.add_argument(
        "--segnet-distillation-objective",
        choices=(
            "kl_t2",
            "boundary_tckd",
            "boundary_decision_tckd",
            "boundary_argmax_hinge",
        ),
        default="kl_t2",
    )
    parser.add_argument("--distillation-temperature", default=2.0, type=float)
    parser.add_argument("--segnet-tau-boundary", default=1.0, type=float)
    parser.add_argument("--segnet-hinge-margin", default=1.0, type=float)
    parser.add_argument(
        "--distillation-device",
        default="cpu",
        help="Device used to build real scorer teacher caches; default CPU.",
    )
    parser.add_argument(
        "--allow-segnet-only-research",
        action="store_true",
        help=(
            "Explicitly allow SegNet-only compact training without PoseNet. "
            "This remains false-authority and is never promotion-ready by itself."
        ),
    )
    parser.add_argument(
        "--allow-unscored-research-smoke",
        action="store_true",
        help=(
            "Permit tiny adapter/runtime smokes without real SegNet/PoseNet "
            "loss weights. Default frontier-targeting HiNeRV runs refuse "
            "unscored launches."
        ),
    )
    parser.add_argument(
        "--modelsize-budget-json",
        action="append",
        default=[],
        type=Path,
        help=(
            "Receiver-closed or advisory modelsize budget JSON to feed the "
            "HiNeRV score-aware training planner. Repeatable."
        ),
    )
    parser.add_argument(
        "--receiver-closed-ladder-json",
        action="append",
        default=[],
        type=Path,
        help=(
            "Receiver-closed NeRV modelsize ladder JSON. Rows become the "
            "byte-price authority for HiNeRV long-run size selection."
        ),
    )
    parser.add_argument(
        "--run-local-cpu-replay",
        dest="run_local_cpu_replay",
        action="store_true",
        default=None,
        help=(
            "Force the local macOS CPU replay gate after byte-closed archive "
            "export. Full 600-pair coverage runs this gate by default."
        ),
    )
    parser.add_argument(
        "--skip-local-cpu-replay",
        dest="run_local_cpu_replay",
        action="store_false",
        help=(
            "Skip the local macOS CPU replay gate even for full-coverage "
            "campaigns. The report remains false-authority and blocked."
        ),
    )
    parser.add_argument(
        "--keep-local-replay-inflated",
        action="store_true",
        help=(
            "Retain local replay inflated raw outputs. Default is certify-and-"
            "delete scratch through local_submission_replay cleanup manifests."
        ),
    )
    parser.add_argument(
        "--retain-failed-local-replay-scratch",
        action="store_true",
        help=(
            "Keep failed local replay scratch for debugging. Default deletes "
            "certified rebuildable failed scratch to protect disk."
        ),
    )
    parser.add_argument(
        "--run-post-export-materializers",
        action="store_true",
        help=(
            "After a byte-closed carrier export, run a bounded local wave of "
            "the compiled post-export final-rate materializer queue. Results "
            "remain false-authority and output-scoped until exact gates pass."
        ),
    )
    parser.add_argument(
        "--post-export-materializer-max-steps",
        default=1,
        type=int,
        help="Maximum local post-export materializer queue steps to start.",
    )
    parser.add_argument(
        "--post-export-materializer-max-parallel",
        default=0,
        type=int,
        help=(
            "Maximum local post-export materializer queue parallelism. Zero "
            "uses the queue worker's sequential bounded execution."
        ),
    )
    parser.add_argument(
        "--post-export-materializer-max-experiments",
        default=1,
        type=int,
        help=(
            "Maximum post-export materializer experiment chains to advance in "
            "one bounded run. Default 1 focuses the step budget on one target "
            "family through harvest/readiness. Use 0 for fanout across all "
            "ready materializer families."
        ),
    )
    parser.add_argument(
        "--mlx-prefilter-scorer-batch-pairs",
        default=1,
        type=int,
        help=(
            "Pairs per MLX scorer forward for the automatic local prefilter. "
            "Values >1 can accelerate advisory acquisition, but singleton "
            "batching remains required to unlock local CPU replay gates."
        ),
    )
    parser.add_argument(
        "--mlx-prefilter-scorer-device",
        choices=("cpu", "gpu"),
        default=None,
        help=(
            "Device for the automatic MLX renderer prefilter. Defaults to "
            "--distillation-device for backward compatibility. Use gpu only "
            "for local acquisition speed; CPU replay and exact auth remain the "
            "promotion gates."
        ),
    )
    parser.add_argument(
        "--mlx-prefilter-progress-every",
        default=50,
        type=int,
        help=(
            "Emit MLX prefilter progress JSONL every N chunks. Use 0 to "
            "disable progress telemetry."
        ),
    )
    parser.add_argument(
        "--telemetry-flush-interval-epochs",
        default=1,
        type=int,
        help=(
            "HiNeRV long-training JSONL flush interval. Queue-owned long runs "
            "default to 1 so observers can see progress before the terminal "
            "runner report exists."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval-epochs",
        default=DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS,
        type=int,
        help=(
            "Canonical long-training checkpoint cadence for MLX compact-family "
            "runs. Default protects long PR95-style HiNeRV runs from losing "
            "weights before terminal archive export; reports remain "
            "false-authority until receiver proof and exact replay."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Optional canonical checkpoint directory for MLX compact-family "
            "training. Relative paths resolve from --repo-root; default uses "
            "the canonical output_dir/checkpoints location inside each "
            "family's training output."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help=(
            "Optional canonical checkpoint .meta.json to resume from. The "
            "shared long-training harness validates lane/curriculum metadata "
            "and restores real NumPy-portable MLX state; reports remain "
            "false-authority until receiver proof and exact replay."
        ),
    )
    parser.add_argument(
        "--coder-aware-qat",
        action="store_true",
        help=(
            "Add false-authority decoder-weight QAT/rate pressure during compact "
            "MLX training. Archive bytes and receiver proof remain the only "
            "promotion surface."
        ),
    )
    parser.add_argument("--coder-qat-quant-bits", default=8, type=int)
    parser.add_argument(
        "--coder-qat-quant-residual-weight",
        default=DEFAULT_PACT_CODER_QAT_QUANT_RESIDUAL_WEIGHT,
        type=float,
    )
    parser.add_argument(
        "--coder-qat-magnitude-weight",
        default=DEFAULT_PACT_CODER_QAT_MAGNITUDE_WEIGHT,
        type=float,
    )
    parser.add_argument(
        "--coder-qat-delta-weight",
        default=DEFAULT_PACT_CODER_QAT_DELTA_WEIGHT,
        type=float,
    )
    parser.add_argument(
        "--coder-qat-c1a-entropy-weight",
        default=DEFAULT_PACT_CODER_QAT_C1A_ENTROPY_WEIGHT,
        type=float,
        help=(
            "PR95-style soft categorical entropy pressure on selected decoder "
            "weights during coder-aware QAT. False-authority until archive "
            "bytes and receiver proof land."
        ),
    )
    parser.add_argument(
        "--coder-qat-c1a-sigma",
        default=DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
        type=float,
    )
    parser.add_argument(
        "--coder-qat-c1a-sample-size",
        default=DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
        type=int,
    )
    parser.add_argument(
        "--decoder-weight-waterfill-plan-json",
        type=Path,
        help=(
            "HiNeRV only: attach a nerv_decoder_weight_waterfill.v1 artifact "
            "and apply its selected 0/2/4/6/7/8/16/32 actions to real decoder "
            "tensors before archive packing. False-authority until replay."
        ),
    )
    parser.add_argument(
        "--snerv-spectra-preserving-adapter",
        action="store_true",
        help=(
            "For --execute-family snerv, use the receiver-visible "
            "spectra-preserving MFU/HFR feature adapter instead of the "
            "historical 3x3 LF-patch adapter."
        ),
    )
    parser.add_argument(
        "--snerv-mfu-scales",
        default="1,2,4",
        help="Comma-separated deterministic MFU scales for the SNeRV adapter.",
    )
    parser.add_argument(
        "--snerv-model-size-adapter",
        help=(
            "Manual SNeRV model-size adapter override. Planner candidate rows "
            "take precedence when selected."
        ),
    )
    parser.add_argument(
        "--snerv-official-modelsize-mparams",
        action="append",
        type=float,
        help=(
            "Include source-faithful SNeRV --modelsize values, in millions of "
            "params, when resolving --modelsize-candidate-id auto. Repeatable; "
            "still advisory until trained archive bytes and receiver proof land."
        ),
    )
    parser.add_argument(
        "--snerv-modelsize-control-profile",
        choices=sorted(SNERV_MODELSIZE_CONTROL_PROFILES),
        default=DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
        help=(
            "Named SNeRV stride/control profile for --modelsize solves. Manual "
            "stride flags still override the selected profile and are recorded "
            "as manual_stride_override in the candidate metadata."
        ),
    )
    parser.add_argument(
        "--snerv-official-enc-strds",
        type=_parse_positive_int_csv,
        default=None,
        help=(
            "Comma-separated SNeRV encoder strides for --modelsize solve. "
            "Overrides --snerv-modelsize-control-profile when supplied."
        ),
    )
    parser.add_argument(
        "--snerv-official-dec-strds",
        type=_parse_positive_int_csv,
        default=None,
        help=(
            "Comma-separated SNeRV decoder strides for --modelsize solve. "
            "Overrides --snerv-modelsize-control-profile when supplied."
        ),
    )
    parser.add_argument(
        "--snerv-fc-dim",
        type=int,
        help="Manual SNeRV fc_dim override for no-candidate probes.",
    )
    parser.add_argument(
        "--snerv-emb-size",
        type=int,
        help="Manual SNeRV embedding-size override for no-candidate probes.",
    )
    parser.add_argument(
        "--snerv-patch-radius",
        type=int,
        help="Manual SNeRV LF patch-radius override for no-candidate probes.",
    )
    parser.add_argument(
        "--snerv-hfr-gain",
        default=0.0,
        type=float,
        help="Deterministic HFR residual gain for the SNeRV adapter.",
    )
    parser.add_argument(
        "--snerv-temporal-context",
        type=int,
        help="Manual SNeRV temporal context override for no-candidate probes.",
    )
    parser.add_argument(
        "--snerv-temporal-mode",
        default=None,
        choices=("delta", "official_haar_dwt1d_lowpass"),
        help=(
            "Manual SNeRV temporal basis override for no-candidate probes. "
            "Planner rows win when they specify temporal_mode."
        ),
    )
    parser.add_argument(
        "--skip-snerv-native-mlx-export",
        action="store_true",
        help=(
            "For --execute-family snerv, skip the MLX-native "
            "train/export/archive receiver-proof attachment. The CLI runs it "
            "by default so SNeRV execution is MLX-first."
        ),
    )
    parser.add_argument(
        "--snerv-native-mlx-receiver-proof-timeout",
        default=1800,
        type=int,
        help="Timeout seconds for the SNeRV MLX-native receiver proof.",
    )
    parser.add_argument(
        "--snerv-native-mlx-decoder-train-steps",
        default=0,
        type=int,
        help=(
            "Full-batch MLX gradient steps for the SNeRV HF decoder before "
            "receiver/archive proof. Zero keeps the closed-form decoder fit."
        ),
    )
    parser.add_argument(
        "--snerv-native-mlx-decoder-train-lr",
        default=1.0e-5,
        type=float,
        help="Learning rate for --snerv-native-mlx-decoder-train-steps.",
    )
    parser.add_argument(
        "--snerv-native-mlx-decoder-train-ridge",
        default=1.0e-6,
        type=float,
        help="L2/ridge pressure for SNeRV native MLX HF decoder training.",
    )
    parser.add_argument(
        "--snerv-native-mlx-decoder-train-optimizer",
        default="pact_guarded_adamw",
        choices=(
            "pact_guarded_adamw",
            "full_batch_gradient_descent",
            "adamw",
            "adam",
            "lion",
            "sgd",
        ),
        help=(
            "Optimizer for SNeRV's narrow native MLX HF-decoder vector "
            "refinement only, not the full SNeRV family train-time optimizer. "
            "The full score-aware NeRV harness default is --optimizer-kind "
            "pact_muon_adamw. This default tries MLX AdamW and falls back per "
            "subband when measured loss worsens."
        ),
    )
    parser.add_argument(
        "--snerv-scorer-loop-qat",
        action="store_true",
        help=(
            "For --execute-family snerv, run the receiver-priced local "
            "SegNet/PoseNet scorer-loop decoder/QAT attachment. "
            "--coder-aware-qat also enables this path for SNeRV."
        ),
    )
    parser.add_argument("--snerv-scorer-loop-max-trials", default=2, type=int)
    parser.add_argument(
        "--snerv-scorer-loop-search-mode",
        choices=(
            "random_signed",
            "top_weight_coordinate",
            "learned_random_subspace",
            "nes_pair_robust",
        ),
        default="nes_pair_robust",
    )
    parser.add_argument("--snerv-scorer-loop-step-map-bins", default=16, type=int)
    parser.add_argument(
        "--snerv-scorer-loop-lf-payload-codec",
        default="portfolio_auto",
        help=(
            "SNeRV scorer-loop/QAT LF payload codec for receiver-priced SNAR1 "
            "packets."
        ),
    )
    parser.add_argument("--snerv-scorer-loop-perturb-scale", default=0.02, type=float)
    parser.add_argument(
        "--snerv-scorer-loop-byte-pressure-multiplier",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--snerv-scorer-loop-section-value-pressure-multiplier",
        default=1.0,
        type=float,
        help=(
            "SNeRV scorer-loop QAT: multiply the train-time SNAR1 "
            "optional-section neutralization pressure term."
        ),
    )
    parser.add_argument("--snerv-scorer-loop-max-archive-byte-growth", type=int)
    parser.add_argument("--snerv-scorer-loop-pose-slack", default=0.0, type=float)
    parser.add_argument("--snerv-scorer-loop-seg-slack", default=0.0, type=float)
    parser.add_argument("--snerv-scorer-loop-pair-stride", default=1, type=int)
    parser.add_argument("--snerv-scorer-loop-start-pair", default=0, type=int)
    parser.add_argument(
        "--snerv-scorer-loop-pair-guard-min-score-improved-fraction",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--snerv-scorer-loop-pair-guard-max-pose-worsened-fraction",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--snerv-scorer-loop-component-guard-mode",
        choices=("score_primary", "pose_hard", "pose_seg_hard"),
        default="score_primary",
        help=(
            "SNeRV scorer-loop/QAT acceptance guard. score_primary accepts "
            "true score wins; pose_hard and pose_seg_hard are stricter "
            "receiver-priced probes and must remain explicit in metadata."
        ),
    )
    parser.add_argument(
        "--recon-pixel-weight-path",
        type=Path,
        help=(
            "File-backed P18/P19 recon_pixel_weight map (.npy or .npz). "
            "Must be shaped (384,512), (384,512,1/3), "
            "(1|N,384,512,1/3), or (N,2,384,512,1/3)."
        ),
    )
    parser.add_argument(
        "--auto-joint-recon-pixel-weight",
        action="store_true",
        help=(
            "Auto-discover a verified joint P18/P19 recon_pixel_weight artifact "
            "for the requested pair count from canonical SSD/local experiment "
            "results. Fails closed if no finite-gradient manifest matches."
        ),
    )
    parser.add_argument(
        "--auto-segnet-boundary-recon-weight",
        action="store_true",
        help=(
            "Build a P18-only recon_pixel_weight from real SegNet teacher "
            "top-2 margins. Use file-backed maps for joint P18/P19 weights."
        ),
    )
    parser.add_argument("--recon-pixel-weight-tau", default=1.0, type=float)
    parser.add_argument(
        "--recon-pixel-weight-normalize",
        default="mean",
        choices=("mean", "none"),
    )
    parser.add_argument("--smoke-epochs-per-stage", default=1, type=int)
    parser.add_argument(
        "--stage8-epochs",
        default=0,
        type=int,
        help=(
            "Epochs for --execute-pr95-stage8-source. Default 0 proves archive "
            "custody without launching a long Stage-8 run."
        ),
    )
    parser.add_argument("--stage8-eval-every", default=1, type=int)
    parser.add_argument("--stage8-batch-size", default=1, type=int)
    parser.add_argument("--stage8-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--stage8-muon-weight-decay", default=5e-4, type=float)
    parser.add_argument("--stage8-target-cache-path", type=Path)
    parser.add_argument("--stage8-no-build-target-cache", action="store_true")
    parser.add_argument(
        "--training-loss-surface",
        choices=("rgb_mse", "rgb_yuv6_mse"),
        default="rgb_yuv6_mse",
    )
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--random-seed", default=0, type=int)
    parser.add_argument("--hard-byte-ceiling", action="append", type=int)
    parser.add_argument(
        "--hprc-queue-followup-report",
        action="append",
        default=[],
        type=Path,
        help=(
            "Existing HPRC queue followup report to feed into the spine "
            "bounded-runner posterior. Repeatable."
        ),
    )
    parser.add_argument(
        "--mlx-profile",
        action="append",
        default=[],
        type=Path,
        help=(
            "HPRC MLX component-neutralization profile to attach to the "
            "bounded runner for section value-per-byte routing. Repeatable."
        ),
    )
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument(
        "--allow-duplicate-campaign",
        action="store_true",
        help=(
            "Allow another active campaign with identical normalized args. "
            "Default refuses duplicate expensive MLX/replay runs even when "
            "they target different output dirs."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    prioritized_pair_indices = _prioritized_pair_indices_from_args(args)
    args.requested_distillation_device = str(args.distillation_device)
    args.distillation_device = _resolve_torch_scorer_device_alias(
        args.distillation_device
    )
    modes = [
        args.execute_pr95_mlx_smoke,
        args.execute_pr95_stage8_source,
        args.from_pr95_mlx_report is not None,
        args.from_pr95_stage8_report is not None,
        args.from_snerv_advisory_report is not None,
        args.execute_family is not None,
    ]
    if sum(1 for item in modes if item) > 1:
        raise SystemExit(
            "pass only one of --execute-pr95-mlx-smoke, "
            "--execute-pr95-stage8-source, --from-pr95-mlx-report, "
            "--from-pr95-stage8-report, --from-snerv-advisory-report, "
            "or --execute-family"
        )
    snerv_modelsize_control_blockers = (
        _snerv_official_modelsize_candidate_resolution_blockers(args)
    )
    if snerv_modelsize_control_blockers:
        raise SystemExit(
            "SNeRV official modelsize controls require "
            "--modelsize-candidate-id auto or a concrete candidate id: "
            + ", ".join(snerv_modelsize_control_blockers)
        )
    ceilings = tuple(args.hard_byte_ceiling or DEFAULT_BASE_RENDERER_BYTE_CEILINGS)
    output_dir = args.output_dir or _default_output_dir()
    scorer_upstream_dir = _resolve_scorer_upstream_dir(
        args.repo_root,
        args.upstream_dir,
    )
    source_video_path = _resolve_source_video_path(
        args.source_video_path,
        base=Path(args.repo_root).expanduser().resolve(strict=False),
        upstream_dir=scorer_upstream_dir,
    )
    post_export_materializer_max_experiments = (
        None
        if args.post_export_materializer_max_experiments == 0
        else args.post_export_materializer_max_experiments
    )
    snerv_modelsize_profile = snerv_modelsize_control_profile(
        str(args.snerv_modelsize_control_profile)
    )
    snerv_official_enc_strds = (
        tuple(int(value) for value in args.snerv_official_enc_strds)
        if args.snerv_official_enc_strds is not None
        else tuple(int(value) for value in snerv_modelsize_profile["enc_strds"])
    )
    snerv_official_dec_strds = (
        tuple(int(value) for value in args.snerv_official_dec_strds)
        if args.snerv_official_dec_strds is not None
        else tuple(int(value) for value in snerv_modelsize_profile["dec_strds"])
    )
    modelsize_candidate: dict[str, Any] | None = None
    if args.execute_family in {"hi_nerv", "snerv"}:
        modelsize_candidate = _resolve_execute_modelsize_candidate(
            family=args.execute_family,
            candidate_id=args.modelsize_candidate_id,
            hard_byte_ceilings=ceilings,
            num_pairs=CONTEST_PAIR_COUNT,
            target_modelsize_mparams=tuple(
                float(value) for value in (args.target_modelsize_mparams or ())
            ),
            hinerv_target_modelsize_mparams=tuple(
                float(value) for value in (args.hinerv_target_modelsize_mparams or ())
            ),
            snerv_official_modelsize_mparams=tuple(
                float(value) for value in (args.snerv_official_modelsize_mparams or ())
            ),
            snerv_modelsize_control_profile_id=str(
                args.snerv_modelsize_control_profile
            ),
            snerv_official_enc_strds=snerv_official_enc_strds,
            snerv_official_dec_strds=snerv_official_dec_strds,
            snerv_temporal_context=(
                0 if args.snerv_temporal_context is None else int(args.snerv_temporal_context)
            ),
            snerv_temporal_modes=(
                ("delta",)
                if args.snerv_temporal_mode is None
                else (str(args.snerv_temporal_mode),)
            ),
        )
    planner_launch_blockers = _planner_row_launch_blockers(args)
    if planner_launch_blockers:
        report = _write_planner_row_launch_refusal(
            output_dir=Path(output_dir).expanduser().resolve(strict=False),
            args=args,
            blockers=planner_launch_blockers,
            hard_byte_ceilings=ceilings,
            repo_root=Path(args.repo_root).expanduser().resolve(strict=False),
        )
        print(
            json.dumps(
                {
                    "schema": "compact_renderer_mlx_spine_runner_cli_result.v1",
                    "report_path": report["report_path"],
                    "mode": report.get("mode"),
                    "blockers": report.get("blockers", []),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )
        )
        return 0
    _acquire_active_campaign_lock(
        output_dir=Path(output_dir).expanduser().resolve(strict=False),
        args=args,
        source_video_path=source_video_path,
        hard_byte_ceilings=ceilings,
    )
    _write_compact_family_startup_marker(
        output_dir=Path(output_dir).expanduser().resolve(strict=False),
        args=args,
        source_video_path=source_video_path,
        hard_byte_ceilings=ceilings,
        modelsize_candidate=modelsize_candidate,
    )
    interrupted_report_written = False

    def _handle_compact_family_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted_report_written
        if interrupted_report_written:
            raise SystemExit(128 + int(signum))
        interrupted_report_written = True
        if args.execute_family in PLANNER_ROW_REQUIRED_FAMILIES:
            report = _write_compact_family_interrupted_report(
                output_dir=Path(output_dir).expanduser().resolve(strict=False),
                args=args,
                source_video_path=source_video_path,
                hard_byte_ceilings=ceilings,
                modelsize_candidate=modelsize_candidate,
                signum=signum,
                reason="process_signal",
            )
            print(
                json.dumps(
                    {
                        "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
                        "mode": report["mode"],
                        "report_path": report["report_path"],
                        "blockers": report.get("blockers", []),
                        "score_claim": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        raise SystemExit(128 + int(signum))

    if args.execute_family in PLANNER_ROW_REQUIRED_FAMILIES:
        signal.signal(signal.SIGTERM, _handle_compact_family_signal)
        signal.signal(signal.SIGINT, _handle_compact_family_signal)
    if args.execute_pr95_mlx_smoke:
        report = execute_pr95_mlx_smoke_and_adapt(
            output_dir=output_dir,
            max_frames=args.max_frames,
            smoke_epochs_per_stage=args.smoke_epochs_per_stage,
            training_loss_surface=args.training_loss_surface,
            source_video_path=source_video_path,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            random_seed=args.random_seed,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.from_pr95_mlx_report is not None:
        report = adapt_pr95_mlx_report_to_spine(
            pr95_mlx_report_path=args.from_pr95_mlx_report,
            output_dir=output_dir,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
            upstream_dir=scorer_upstream_dir,
        )
    elif args.from_pr95_stage8_report is not None:
        report = adapt_pr95_stage8_report_to_spine(
            pr95_stage8_report_path=args.from_pr95_stage8_report,
            output_dir=output_dir,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            run_receiver_proof=args.run_receiver_proof,
            receiver_proof_runtime_dir=args.pr95_receiver_runtime_dir,
            keep_receiver_proof_output=args.keep_receiver_proof_output,
            receiver_proof_timeout_seconds=args.receiver_proof_timeout_seconds,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
            upstream_dir=scorer_upstream_dir,
        )
    elif args.from_snerv_advisory_report is not None:
        report = adapt_snerv_advisory_report_to_spine(
            snerv_advisory_report_path=args.from_snerv_advisory_report,
            output_dir=output_dir,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            run_local_cpu_replay=args.run_local_cpu_replay,
            keep_local_replay_inflated=args.keep_local_replay_inflated,
            cleanup_failed_local_replay_scratch=not args.retain_failed_local_replay_scratch,
            run_post_export_materializers=args.run_post_export_materializers,
            post_export_materializer_max_steps=(
                args.post_export_materializer_max_steps
            ),
            post_export_materializer_max_parallel=(
                args.post_export_materializer_max_parallel
            ),
            post_export_materializer_max_experiments=(
                post_export_materializer_max_experiments
            ),
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
            upstream_dir=scorer_upstream_dir,
        )
    elif args.execute_pr95_stage8_source:
        report = execute_pr95_stage8_source_and_adapt(
            output_dir=output_dir,
            source_archive_zip=args.pr95_source_archive,
            source_video_path=source_video_path,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            stage8_epochs=args.stage8_epochs,
            stage8_eval_every=args.stage8_eval_every,
            stage8_batch_size=args.stage8_batch_size,
            stage8_device=args.stage8_device,
            stage8_muon_weight_decay=args.stage8_muon_weight_decay,
            stage8_target_cache_path=args.stage8_target_cache_path,
            stage8_build_target_cache_if_missing=not args.stage8_no_build_target_cache,
            run_receiver_proof=args.run_receiver_proof,
            receiver_proof_runtime_dir=args.pr95_receiver_runtime_dir,
            keep_receiver_proof_output=args.keep_receiver_proof_output,
            receiver_proof_timeout_seconds=args.receiver_proof_timeout_seconds,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
            upstream_dir=scorer_upstream_dir,
        )
    elif args.execute_family == "pr95_hnerv":
        report = execute_pr95_hnerv_mlx_scoreaware_and_adapt(
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            batch_pair_indices_per_step=args.batch_pairs,
            learning_rate=args.learning_rate,
            source_video_path=source_video_path,
            source_archive_zip=args.pr95_source_archive,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.latent_dim or 28,
            base_channels=args.base_channels or 36,
            ema_decay=args.compact_ema_decay,
            segnet_distillation_weight=args.segnet_distillation_weight,
            pose_distillation_weight=args.pose_distillation_weight,
            pose_distillation_loss=args.pose_distillation_loss,
            pose_distillation_huber_delta=args.pose_distillation_huber_delta,
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            requested_distillation_device=getattr(
                args,
                "requested_distillation_device",
                args.distillation_device,
            ),
            allow_segnet_only_research=args.allow_segnet_only_research,
            checkpoint_interval_epochs=args.checkpoint_interval_epochs,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            scorer_upstream_dir=scorer_upstream_dir,
            run_receiver_proof=args.run_receiver_proof,
            receiver_proof_runtime_dir=args.pr95_receiver_runtime_dir,
            keep_receiver_proof_output=args.keep_receiver_proof_output,
            receiver_proof_timeout_seconds=args.receiver_proof_timeout_seconds,
            run_post_export_materializers=args.run_post_export_materializers,
            post_export_materializer_max_steps=(
                args.post_export_materializer_max_steps
            ),
            post_export_materializer_max_parallel=(
                args.post_export_materializer_max_parallel
            ),
            post_export_materializer_max_experiments=post_export_materializer_max_experiments,
            random_seed=args.random_seed,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.execute_family == "pact_nerv_vq":
        report = execute_pact_nerv_vq_mlx_smoke_and_adapt(
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            batch_pair_indices_per_step=args.batch_pairs,
            learning_rate=args.learning_rate,
            source_video_path=source_video_path,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.compact_latent_dim,
            embed_dim=args.compact_embed_dim,
            codebook_size=args.compact_codebook_size,
            decoder_channel=args.compact_decoder_channel,
            decoder_codec=args.compact_decoder_codec,
            ema_decay=args.compact_ema_decay,
            segnet_distillation_weight=args.segnet_distillation_weight,
            pose_distillation_weight=args.pose_distillation_weight,
            pose_distillation_loss=args.pose_distillation_loss,
            pose_distillation_huber_delta=args.pose_distillation_huber_delta,
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            requested_distillation_device=getattr(
                args,
                "requested_distillation_device",
                args.distillation_device,
            ),
            allow_segnet_only_research=args.allow_segnet_only_research,
            scorer_upstream_dir=scorer_upstream_dir,
            coder_aware_qat=args.coder_aware_qat,
            coder_qat_quant_bits=args.coder_qat_quant_bits,
            coder_qat_quant_residual_weight=args.coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=args.coder_qat_magnitude_weight,
            coder_qat_delta_weight=args.coder_qat_delta_weight,
            coder_qat_c1a_entropy_weight=args.coder_qat_c1a_entropy_weight,
            coder_qat_c1a_sigma=args.coder_qat_c1a_sigma,
            coder_qat_c1a_sample_size=args.coder_qat_c1a_sample_size,
            optimizer_kind=args.optimizer_kind,
            optimizer_grad_clip_max_norm=args.optimizer_grad_clip_max_norm,
            optimizer_weight_decay=args.optimizer_weight_decay,
            optimizer_warmup_epochs=args.optimizer_warmup_epochs,
            optimizer_warmup_steps_per_epoch=args.optimizer_warmup_steps_per_epoch,
            optimizer_cosine_decay_enabled=args.optimizer_cosine_decay_enabled,
            optimizer_cosine_decay_total_epochs=(
                args.optimizer_cosine_decay_total_epochs
            ),
            optimizer_cosine_decay_min_lr_ratio=(
                args.optimizer_cosine_decay_min_lr_ratio
            ),
            checkpoint_interval_epochs=args.checkpoint_interval_epochs,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            run_post_export_materializers=args.run_post_export_materializers,
            post_export_materializer_max_steps=(
                args.post_export_materializer_max_steps
            ),
            post_export_materializer_max_parallel=(
                args.post_export_materializer_max_parallel
            ),
            post_export_materializer_max_experiments=post_export_materializer_max_experiments,
            random_seed=args.random_seed,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.execute_family == "pact_nerv_selector_v4":
        report = execute_pact_nerv_selector_v4_mlx_smoke_and_adapt(
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            batch_pair_indices_per_step=args.batch_pairs,
            learning_rate=args.learning_rate,
            source_video_path=source_video_path,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.compact_latent_dim,
            embed_dim=args.compact_embed_dim,
            selector_palette_size=args.compact_selector_palette_size,
            decoder_channel=args.compact_decoder_channel,
            decoder_codec=args.compact_decoder_codec,
            ema_decay=args.compact_ema_decay,
            segnet_distillation_weight=args.segnet_distillation_weight,
            pose_distillation_weight=args.pose_distillation_weight,
            pose_distillation_loss=args.pose_distillation_loss,
            pose_distillation_huber_delta=args.pose_distillation_huber_delta,
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            allow_segnet_only_research=args.allow_segnet_only_research,
            scorer_upstream_dir=scorer_upstream_dir,
            coder_aware_qat=args.coder_aware_qat,
            coder_qat_quant_bits=args.coder_qat_quant_bits,
            coder_qat_quant_residual_weight=args.coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=args.coder_qat_magnitude_weight,
            coder_qat_delta_weight=args.coder_qat_delta_weight,
            coder_qat_c1a_entropy_weight=args.coder_qat_c1a_entropy_weight,
            coder_qat_c1a_sigma=args.coder_qat_c1a_sigma,
            coder_qat_c1a_sample_size=args.coder_qat_c1a_sample_size,
            optimizer_kind=args.optimizer_kind,
            optimizer_grad_clip_max_norm=args.optimizer_grad_clip_max_norm,
            optimizer_weight_decay=args.optimizer_weight_decay,
            optimizer_warmup_epochs=args.optimizer_warmup_epochs,
            optimizer_warmup_steps_per_epoch=args.optimizer_warmup_steps_per_epoch,
            optimizer_cosine_decay_enabled=args.optimizer_cosine_decay_enabled,
            optimizer_cosine_decay_total_epochs=(
                args.optimizer_cosine_decay_total_epochs
            ),
            optimizer_cosine_decay_min_lr_ratio=(
                args.optimizer_cosine_decay_min_lr_ratio
            ),
            checkpoint_interval_epochs=args.checkpoint_interval_epochs,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            run_post_export_materializers=args.run_post_export_materializers,
            post_export_materializer_max_steps=(
                args.post_export_materializer_max_steps
            ),
            post_export_materializer_max_parallel=(
                args.post_export_materializer_max_parallel
            ),
            post_export_materializer_max_experiments=post_export_materializer_max_experiments,
            random_seed=args.random_seed,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.execute_family == "snerv":
        report = execute_snerv_inverse_steg_advisory_and_adapt(
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            source_video_path=source_video_path,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            run_local_cpu_replay=args.run_local_cpu_replay,
            keep_local_replay_inflated=args.keep_local_replay_inflated,
            cleanup_failed_local_replay_scratch=not args.retain_failed_local_replay_scratch,
            run_post_export_materializers=args.run_post_export_materializers,
            post_export_materializer_max_steps=(
                args.post_export_materializer_max_steps
            ),
            post_export_materializer_max_parallel=(
                args.post_export_materializer_max_parallel
            ),
            post_export_materializer_max_experiments=post_export_materializer_max_experiments,
            distillation_device=args.distillation_device,
            modelsize_candidate=modelsize_candidate,
            prioritized_pair_indices=prioritized_pair_indices,
            snerv_spectra_preserving_adapter=args.snerv_spectra_preserving_adapter,
            snerv_model_size_adapter_override=args.snerv_model_size_adapter,
            snerv_fc_dim_override=args.snerv_fc_dim,
            snerv_emb_size_override=args.snerv_emb_size,
            snerv_patch_radius_override=args.snerv_patch_radius,
            snerv_mfu_scales=_parse_positive_int_csv(args.snerv_mfu_scales),
            snerv_hfr_gain=args.snerv_hfr_gain,
            snerv_temporal_context_override=args.snerv_temporal_context,
            snerv_temporal_mode_override=args.snerv_temporal_mode,
            recon_pixel_weight_path=args.recon_pixel_weight_path,
            auto_joint_recon_pixel_weight=args.auto_joint_recon_pixel_weight,
            recon_pixel_weight_normalize=args.recon_pixel_weight_normalize,
            run_native_mlx_export=not args.skip_snerv_native_mlx_export,
            snerv_native_mlx_receiver_proof_timeout_seconds=(
                args.snerv_native_mlx_receiver_proof_timeout
            ),
            snerv_native_mlx_decoder_train_steps=(
                args.snerv_native_mlx_decoder_train_steps
            ),
            snerv_native_mlx_decoder_train_lr=(
                args.snerv_native_mlx_decoder_train_lr
            ),
            snerv_native_mlx_decoder_train_ridge=(
                args.snerv_native_mlx_decoder_train_ridge
            ),
            snerv_native_mlx_decoder_train_optimizer=(
                args.snerv_native_mlx_decoder_train_optimizer
            ),
            run_scorer_loop_qat=bool(
                args.coder_aware_qat or args.snerv_scorer_loop_qat
            ),
            snerv_scorer_loop_max_trials=args.snerv_scorer_loop_max_trials,
            snerv_scorer_loop_search_mode=args.snerv_scorer_loop_search_mode,
            snerv_scorer_loop_step_map_bins=args.snerv_scorer_loop_step_map_bins,
            snerv_scorer_loop_qat_bits=args.coder_qat_quant_bits,
            snerv_scorer_loop_lf_payload_codec=(
                args.snerv_scorer_loop_lf_payload_codec
            ),
            snerv_scorer_loop_perturb_scale=args.snerv_scorer_loop_perturb_scale,
            snerv_scorer_loop_byte_pressure_multiplier=(
                args.snerv_scorer_loop_byte_pressure_multiplier
            ),
            snerv_scorer_loop_section_value_pressure_multiplier=(
                args.snerv_scorer_loop_section_value_pressure_multiplier
            ),
            snerv_scorer_loop_max_archive_byte_growth=(
                args.snerv_scorer_loop_max_archive_byte_growth
            ),
            snerv_scorer_loop_pose_slack=args.snerv_scorer_loop_pose_slack,
            snerv_scorer_loop_seg_slack=args.snerv_scorer_loop_seg_slack,
            snerv_scorer_loop_pair_stride=args.snerv_scorer_loop_pair_stride,
            snerv_scorer_loop_start_pair=args.snerv_scorer_loop_start_pair,
            snerv_scorer_loop_pair_guard_min_score_improved_fraction=(
                args.snerv_scorer_loop_pair_guard_min_score_improved_fraction
            ),
            snerv_scorer_loop_pair_guard_max_pose_worsened_fraction=(
                args.snerv_scorer_loop_pair_guard_max_pose_worsened_fraction
            ),
            snerv_scorer_loop_component_guard_mode=(
                args.snerv_scorer_loop_component_guard_mode
            ),
            random_seed=args.random_seed,
            upstream_dir=scorer_upstream_dir,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.execute_family == "hi_nerv":
        report = execute_hi_nerv_mlx_scoreaware_and_adapt(
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            batch_pair_indices_per_step=args.batch_pairs,
            learning_rate=args.learning_rate,
            source_video_path=source_video_path,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.compact_latent_dim,
            embed_dim=args.compact_embed_dim,
            decoder_channel=args.compact_decoder_channel,
            decoder_codec=args.compact_decoder_codec,
            hi_nerv_latent_codec=args.hi_nerv_latent_codec,
            modelsize_candidate=modelsize_candidate,
            ema_decay=args.compact_ema_decay,
            segnet_distillation_weight=args.segnet_distillation_weight,
            pose_distillation_weight=args.pose_distillation_weight,
            pose_distillation_loss=args.pose_distillation_loss,
            pose_distillation_huber_delta=args.pose_distillation_huber_delta,
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            requested_distillation_device=getattr(
                args,
                "requested_distillation_device",
                args.distillation_device,
            ),
            allow_segnet_only_research=args.allow_segnet_only_research,
            allow_unscored_research_smoke=args.allow_unscored_research_smoke,
            modelsize_budget_json_paths=tuple(args.modelsize_budget_json),
            receiver_closed_ladder_json_paths=tuple(args.receiver_closed_ladder_json),
            coder_aware_qat=args.coder_aware_qat,
            coder_qat_quant_bits=args.coder_qat_quant_bits,
            coder_qat_quant_residual_weight=args.coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=args.coder_qat_magnitude_weight,
            coder_qat_delta_weight=args.coder_qat_delta_weight,
            coder_qat_c1a_entropy_weight=args.coder_qat_c1a_entropy_weight,
            coder_qat_c1a_sigma=args.coder_qat_c1a_sigma,
            coder_qat_c1a_sample_size=args.coder_qat_c1a_sample_size,
            decoder_weight_waterfill_plan_json=(
                args.decoder_weight_waterfill_plan_json
            ),
            recon_pixel_weight_path=args.recon_pixel_weight_path,
            auto_joint_recon_pixel_weight=args.auto_joint_recon_pixel_weight,
            auto_segnet_boundary_recon_weight=(
                args.auto_segnet_boundary_recon_weight
            ),
            recon_pixel_weight_tau=args.recon_pixel_weight_tau,
            recon_pixel_weight_normalize=args.recon_pixel_weight_normalize,
            mlx_prefilter_scorer_batch_pairs=(
                args.mlx_prefilter_scorer_batch_pairs
            ),
            mlx_prefilter_scorer_device=args.mlx_prefilter_scorer_device,
            mlx_prefilter_progress_every=args.mlx_prefilter_progress_every,
            telemetry_flush_interval_epochs=args.telemetry_flush_interval_epochs,
            checkpoint_interval_epochs=args.checkpoint_interval_epochs,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            optimizer_kind=args.optimizer_kind,
            hi_nerv_optimizer_policy=args.hi_nerv_optimizer_policy,
            optimizer_grad_clip_max_norm=args.optimizer_grad_clip_max_norm,
            optimizer_weight_decay=args.optimizer_weight_decay,
            optimizer_warmup_epochs=args.optimizer_warmup_epochs,
            optimizer_warmup_steps_per_epoch=args.optimizer_warmup_steps_per_epoch,
            optimizer_cosine_decay_enabled=args.optimizer_cosine_decay_enabled,
            optimizer_cosine_decay_total_epochs=(
                args.optimizer_cosine_decay_total_epochs
            ),
            optimizer_cosine_decay_min_lr_ratio=(
                args.optimizer_cosine_decay_min_lr_ratio
            ),
            prioritized_pair_indices=prioritized_pair_indices,
            run_local_cpu_replay=args.run_local_cpu_replay,
            keep_local_replay_inflated=args.keep_local_replay_inflated,
            cleanup_failed_local_replay_scratch=not args.retain_failed_local_replay_scratch,
            run_post_export_materializers=args.run_post_export_materializers,
            post_export_materializer_max_steps=(
                args.post_export_materializer_max_steps
            ),
            post_export_materializer_max_parallel=(
                args.post_export_materializer_max_parallel
            ),
            post_export_materializer_max_experiments=post_export_materializer_max_experiments,
            upstream_dir=scorer_upstream_dir,
            random_seed=args.random_seed,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.execute_family in PLANNER_GATED_FAMILIES:
        report = execute_planner_gated_compact_family(
            family=args.execute_family,
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    else:
        report = build_plan_only_report(
            output_dir=output_dir,
            hard_byte_ceilings=ceilings,
            repo_root=args.repo_root,
            allow_overwrite=args.overwrite,
        )
    print(
        json.dumps(
            {
                "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
                "mode": report["mode"],
                "report_path": report["report_path"],
                "blockers": report.get("blockers", []),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _resolve_existing(path: str | Path, *, base: Path) -> Path:
    resolved = _resolve(path, base=base)
    if not resolved.is_file():
        raise CompactRendererMlxSpineRunnerError(f"required file missing: {resolved}")
    return resolved


def _resolve_source_video_path(
    path: str | Path,
    *,
    base: Path,
    upstream_dir: str | Path | None = None,
) -> Path:
    """Resolve bulky source video paths across clean SSD source worktrees."""

    env_override = os.environ.get("PACT_SOURCE_VIDEO_PATH")
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    raw = Path(path).expanduser()
    candidates.append(raw if raw.is_absolute() else (base / raw))
    if (
        upstream_dir is not None
        and not raw.is_absolute()
        and raw.parts
        and raw.parts[0] == "upstream"
    ):
        upstream = Path(upstream_dir).expanduser().resolve(strict=False)
        candidates.append(upstream / Path(*raw.parts[1:]))
    if not raw.is_absolute():
        candidates.append(Path("/Users/adpena/Projects/pact") / raw)
    elif raw.name == "0.mkv" and "upstream" in raw.parts and "videos" in raw.parts:
        candidates.append(Path("/Users/adpena/Projects/pact/upstream/videos/0.mkv"))
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file():
            return resolved
    tried = ", ".join(str(candidate.resolve(strict=False)) for candidate in candidates)
    raise CompactRendererMlxSpineRunnerError(
        f"source video missing; tried: {tried}"
    )


def _optional_existing(path: Any, *, base: Path) -> Path | None:
    if not isinstance(path, (str, os.PathLike)) or not path:
        return None
    resolved = _resolve(path, base=base)
    return resolved if resolved.is_file() else None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompactRendererMlxSpineRunnerError(f"JSON root must be object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snerv_scorer_loop_progress_callback(progress_path: Path):
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def callback(row: Any) -> None:
        row_payload = row.as_jsonable() if hasattr(row, "as_jsonable") else dict(row)
        payload = {
            "schema": "snerv_scorer_loop_decoder_qat_progress.v1",
            "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "axis_tag": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "row": row_payload,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()

    return callback


def _has_disallowed_existing_output_artifacts(
    output_dir: Path,
    *,
    allow_startup_marker_only: bool = False,
) -> bool:
    """Return whether output_dir contains artifacts that should block a launch."""

    if not output_dir.exists():
        return False
    entries = list(output_dir.iterdir())
    if not entries:
        return False
    if allow_startup_marker_only:
        for entry in entries:
            if (
                entry.is_file()
                and entry.name == COMPACT_FAMILY_STARTUP_MARKER_FILENAME
            ):
                continue
            if entry.is_dir() and not any(
                child.is_file() or child.is_symlink()
                for child in entry.rglob("*")
            ):
                continue
            return True
        return False
    return True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.as_posix(), "exists": False}
    return {
        "path": path.as_posix(),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        json.JSONDecodeError,
        CompactRendererMlxSpineRunnerError,
    ) as exc:
        print(f"run_compact_renderer_mlx_spine_runner failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
