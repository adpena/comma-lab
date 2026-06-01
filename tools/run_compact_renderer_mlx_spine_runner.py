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
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
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
from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    PR95_MLX_SOURCE_VIDEO_RGB_YUV6_BLOCKERS,
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
from tools.emit_compact_renderer_spine_adapter import (  # noqa: E402
    emit_compact_renderer_spine_adapter,
)

COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA = "compact_renderer_mlx_spine_runner.v1"
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
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
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
)
PLANNER_GATED_FAMILIES = ("snerv",)
CLI_EXECUTE_FAMILIES = (*EXECUTABLE_FAMILIES, *PLANNER_GATED_FAMILIES)
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
        "backend_status": "migration_required",
        "trainer_kind": "missing_mlx_spectra_preserving_carrier_trainer",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_adapter_implemented",
        "next_action": (
            "implement_snerv_wavelet_lf_hf_mlx_carrier_under_packet_spine_"
            "with_charged_wavelet_features_no_hidden_sidecars"
        ),
        "execution_scope": (
            "primary compact carrier candidate; SNeRV wavelet/frequency split "
            "must be charged as decoder/latent/selector/codebook bytes and "
            "validated by receiver proof before promotion"
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

PR95_HNERV_CONTROL_ARM_EXACT_BLOCKERS: tuple[str, ...] = (
    "pr95_hnerv_mlx_archive_export_control_arm_not_pr95_faithful_reproduction",
    *PR95_MLX_SOURCE_VIDEO_RGB_YUV6_BLOCKERS,
    "pr95_hnerv_stage8_muon_continuation_not_wired",
    "pr95_hnerv_default_scorer_distillation_weights_are_zero_unless_cli_overridden",
    "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
    "requires_exact_cpu_cuda_auth_eval_before_score_claim",
)


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
    if upstream_dir is None:
        return (root / "upstream").resolve(strict=False)
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

    spine = build_pr95_hnerv_spine_from_archive(archive)
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
    segnet_distillation_objective: str,
    distillation_temperature: float,
    segnet_tau_boundary: float,
    segnet_hinge_margin: float,
    distillation_device: str,
    allow_segnet_only_research: bool,
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
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": distillation_device,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
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
        checkpoint_interval_epochs=max(1, int(epochs)),
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
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    allow_segnet_only_research: bool = False,
    scorer_upstream_dir: str | Path | None = None,
    run_receiver_proof: bool = False,
    receiver_proof_runtime_dir: str | Path = DEFAULT_PR95_RECEIVER_RUNTIME_DIR,
    keep_receiver_proof_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a public-PR95-seeded HNeRV candidate through the spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, scorer_upstream_dir)
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
            segnet_distillation_objective=segnet_distillation_objective,
            distillation_temperature=distillation_temperature,
            segnet_tau_boundary=segnet_tau_boundary,
            segnet_hinge_margin=segnet_hinge_margin,
            distillation_device=distillation_device,
            allow_segnet_only_research=allow_segnet_only_research,
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
            spine = build_pr95_hnerv_spine_from_archive(archive_file)
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
    blockers.extend(PR95_HNERV_CONTROL_ARM_EXACT_BLOCKERS)
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
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    allow_segnet_only_research: bool = False,
    scorer_upstream_dir: str | Path | None = None,
    coder_aware_qat: bool = False,
    coder_qat_quant_bits: int = 8,
    coder_qat_quant_residual_weight: float = 1.0e-4,
    coder_qat_magnitude_weight: float = 0.0,
    coder_qat_delta_weight: float = 0.0,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a tiny real-video PACT-NeRV-VQ candidate through the spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, scorer_upstream_dir)
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
                "segnet_distillation_objective": segnet_distillation_objective,
                "distillation_temperature": float(distillation_temperature),
                "segnet_tau_boundary": float(segnet_tau_boundary),
                "segnet_hinge_margin": float(segnet_hinge_margin),
                "distillation_device": distillation_device,
                "allow_segnet_only_research": bool(allow_segnet_only_research),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "coder_aware_qat": {
                    "enabled": bool(coder_aware_qat),
                    "quant_bits": int(coder_qat_quant_bits),
                    "quant_residual_weight": float(coder_qat_quant_residual_weight),
                    "magnitude_weight": float(coder_qat_magnitude_weight),
                    "delta_weight": float(coder_qat_delta_weight),
                    "authority": "false_macos_mlx_research_signal",
                },
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
    ema_decay: float = 0.9,
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    allow_segnet_only_research: bool = False,
    coder_aware_qat: bool = False,
    coder_qat_quant_bits: int = 8,
    coder_qat_quant_residual_weight: float = 1.0e-4,
    coder_qat_magnitude_weight: float = 0.0,
    coder_qat_delta_weight: float = 0.0,
    random_seed: int = 0,
    run_local_cpu_replay: bool | None = None,
    keep_local_replay_inflated: bool = False,
    cleanup_failed_local_replay_scratch: bool = True,
    upstream_dir: str | Path = DEFAULT_UPSTREAM_DIR,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a HiNeRV MLX candidate through the real receiver bundle."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, upstream_dir)
    resolved_source_video = _resolve_source_video_path(source_video_path, base=root)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    try:
        artifact = _run_hi_nerv_mlx_scoreaware_smoke(
            output_dir=out / "hi_nerv_mlx_training",
            num_pairs=num_pairs,
            epochs=epochs,
            batch_pair_indices_per_step=batch_pair_indices_per_step,
            learning_rate=learning_rate,
            source_video_path=resolved_source_video,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            decoder_channel=decoder_channel,
            ema_decay=ema_decay,
            segnet_distillation_weight=segnet_distillation_weight,
            pose_distillation_weight=pose_distillation_weight,
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
            random_seed=random_seed,
            scorer_upstream_dir=scorer_upstream,
            repo_root=root,
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
                "failure": repr(exc),
                "scorer_upstream_snapshot": _scorer_upstream_metadata(
                    scorer_upstream
                ),
                "blockers": ["hi_nerv_mlx_scoreaware_or_export_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}

    artifact_dict = artifact.as_dict() if hasattr(artifact, "as_dict") else dict(artifact)
    archive_path = artifact_dict.get("archive_path")
    training_dir = out / "hi_nerv_mlx_training"
    spine_manifest = training_dir / "hprc_representation_spine_hi_nerv_manifest.json"
    receiver_proof_path = (
        training_dir / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
    )
    projection_paths = [spine_manifest] if spine_manifest.is_file() else []
    receiver_proof_paths = [receiver_proof_path] if receiver_proof_path.is_file() else []
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
            archive_zip_path=archive_path,
            runtime_submission_dir=training_dir / "submission",
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
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    selected_runner_rows: list[dict[str, Any]] = []
    blockers: list[Any] = [
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    blockers.extend(local_cpu_replay_blockers)
    pr95_curriculum_enabled = int(epochs) >= 8
    if not pr95_curriculum_enabled:
        blockers.append("hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs")
    if segnet_distillation_weight <= 0.0 or pose_distillation_weight <= 0.0:
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
        blockers.append("hi_nerv_spine_projection_manifest_missing")
    if not archive_path:
        blockers.append("hi_nerv_archive_export_missing")

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
            "archive_path": archive_path,
            "archive_bytes": artifact_dict.get("archive_bytes"),
            "archive_sha256": artifact_dict.get("archive_sha256"),
            "training_artifact": artifact_dict,
            "score_aware_training": {
                "schema": "compact_hi_nerv_score_aware_training.v1",
                "status": "executed_mlx_local_false_authority",
                "segnet_distillation_weight": float(segnet_distillation_weight),
                "pose_distillation_weight": float(pose_distillation_weight),
                "segnet_distillation_objective": segnet_distillation_objective,
                "distillation_temperature": float(distillation_temperature),
                "segnet_tau_boundary": float(segnet_tau_boundary),
                "segnet_hinge_margin": float(segnet_hinge_margin),
                "distillation_device": distillation_device,
                "allow_segnet_only_research": bool(allow_segnet_only_research),
                "pr95_faithful_curriculum_enabled": pr95_curriculum_enabled,
                "coder_aware_qat": _coder_qat_report_metadata(
                    artifact_dict=artifact_dict,
                    enabled=coder_aware_qat,
                    quant_bits=coder_qat_quant_bits,
                    quant_residual_weight=coder_qat_quant_residual_weight,
                    magnitude_weight=coder_qat_magnitude_weight,
                    delta_weight=coder_qat_delta_weight,
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
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    allow_segnet_only_research: bool = False,
    scorer_upstream_dir: str | Path | None = None,
    coder_aware_qat: bool = False,
    coder_qat_quant_bits: int = 8,
    coder_qat_quant_residual_weight: float = 1.0e-4,
    coder_qat_magnitude_weight: float = 0.0,
    coder_qat_delta_weight: float = 0.0,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a tiny real-video PACT-NeRV-SELECTOR-V4 candidate."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    scorer_upstream = _resolve_scorer_upstream_dir(root, scorer_upstream_dir)
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
            "ema_decay": float(ema_decay),
            "score_aware_training": {
                "schema": "compact_pact_nerv_selector_v4_score_aware_training.v1",
                "segnet_distillation_weight": float(segnet_distillation_weight),
                "pose_distillation_weight": float(pose_distillation_weight),
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
                "coder_aware_qat": {
                    "enabled": bool(coder_aware_qat),
                    "quant_bits": int(coder_qat_quant_bits),
                    "quant_residual_weight": float(coder_qat_quant_residual_weight),
                    "magnitude_weight": float(coder_qat_magnitude_weight),
                    "delta_weight": float(coder_qat_delta_weight),
                    "authority": "false_macos_mlx_research_signal",
                },
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
        "authority": "false_macos_mlx_research_signal",
    }


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
    ema_decay: float,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
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
    random_seed: int,
    scorer_upstream_dir: Path,
    repo_root: Path,
) -> Any:
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

    pairs = int(num_pairs)
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
        mid_injection_block_index=1,
        fine_injection_block_index=4,
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
    model = HinervSubstrateMLX(cfg)
    pr95_curriculum_enabled = int(epochs) >= 8
    coder_qat_cfg = CoderAwareQATConfig(
        enabled=bool(coder_aware_qat),
        quant_bits=int(coder_qat_quant_bits),
        quant_residual_weight=float(coder_qat_quant_residual_weight),
        magnitude_weight=float(coder_qat_magnitude_weight),
        delta_weight=float(coder_qat_delta_weight),
    ).validated()

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
            decoder_codec="int8_mixed",
        )

    artifact_metadata = {
        "schema": "compact_renderer_hi_nerv_mlx_adapter_smoke_metadata.v1",
        "family": "hi_nerv",
        "num_pairs": pairs,
        "full_video_pairs_required_for_promotion": 600,
        "decoder_codec": "int8_mixed",
        "model_num_parameters_at_init": int(model.num_parameters()),
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
        },
        "score_aware_training": {
            "schema": "compact_hi_nerv_score_aware_training.v1",
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": distillation_device,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "pr95_faithful_curriculum_enabled": pr95_curriculum_enabled,
            "coder_aware_qat": coder_qat_metadata(coder_qat_cfg),
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
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=int(pose_scorer_teacher.pose_dims)
        if pose_scorer_teacher is not None
        else 6,
        allow_segnet_only_research=bool(allow_segnet_only_research),
    )
    return run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="compact_runner_hi_nerv_mlx",
        lane_id="lane_compact_renderer_mlx_spine_runner_hi_nerv_20260601",
        output_dir=output_dir,
        epochs=int(epochs),
        batch_pair_indices_per_step=max(1, int(batch_pair_indices_per_step)),
        learning_rate=float(learning_rate),
        ema_decay=float(ema_decay),
        seed=int(random_seed),
        checkpoint_interval_epochs=max(1, int(epochs)),
        pr95_faithful_curriculum_enabled=pr95_curriculum_enabled,
        pr95_curriculum_total_epochs=max(8, int(epochs)),
        grad_clip_max_norm=1.0,
        weight_decay=1e-4,
        optimizer_kind="adamw",
        notes=(
            "Compact renderer MLX spine runner HiNeRV training using real "
            "contest video targets, byte-closed archive export, receiver proof, "
            "PR95-faithful curriculum routing, and false-authority MLX evidence."
        ),
    )


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
    random_seed: int,
    scorer_upstream_dir: Path,
    repo_root: Path,
) -> Any:
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
    coder_qat_cfg = CoderAwareQATConfig(
        enabled=bool(coder_aware_qat),
        quant_bits=int(coder_qat_quant_bits),
        quant_residual_weight=float(coder_qat_quant_residual_weight),
        magnitude_weight=float(coder_qat_magnitude_weight),
        delta_weight=float(coder_qat_delta_weight),
    ).validated()

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
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": distillation_device,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "scorer_upstream_snapshot": _scorer_upstream_metadata(
                scorer_upstream_dir
            ),
            "coder_aware_qat": coder_qat_metadata(coder_qat_cfg),
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
        checkpoint_interval_epochs=max(1, int(epochs)),
        notes=(
            "Compact renderer MLX spine runner PACT-NeRV-VQ smoke using real "
            "contest video targets, byte-closed archive export, receiver proof, "
            "and false-authority MLX evidence only."
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
    random_seed: int,
    scorer_upstream_dir: Path,
    repo_root: Path,
) -> Any:
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
    )

    pairs = int(num_pairs)
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
    coder_qat_cfg = CoderAwareQATConfig(
        enabled=bool(coder_aware_qat),
        quant_bits=int(coder_qat_quant_bits),
        quant_residual_weight=float(coder_qat_quant_residual_weight),
        magnitude_weight=float(coder_qat_magnitude_weight),
        delta_weight=float(coder_qat_delta_weight),
    ).validated()

    def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
        return build_decoder_coder_qat_terms(model_obj, coder_qat_cfg)

    def _export_archive(model_obj: Any, archive_output_dir: Path) -> tuple[Path, str, int]:
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
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": distillation_device,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "scorer_upstream_snapshot": _scorer_upstream_metadata(
                scorer_upstream_dir
            ),
            "scorer_coupled_rd": _scorer_coupled_rd_metadata(),
            "coder_aware_qat": coder_qat_metadata(coder_qat_cfg),
            "decoder_codec": str(decoder_codec),
        },
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
        checkpoint_interval_epochs=max(1, int(epochs)),
        notes=(
            "Compact renderer MLX spine runner PACT-NeRV-SELECTOR-V4 smoke "
            "using real contest video targets, selector-v4 PSV4 archive export, "
            "receiver proof, and false-authority MLX evidence only."
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from-pr95-mlx-report", type=Path)
    parser.add_argument(
        "--from-pr95-stage8-report",
        type=Path,
        help="Adapt a source-faithful PR95 Stage-8 report into the compact spine.",
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
            "so clean SSD code worktrees can reuse the canonical upstream bytes."
        ),
    )
    parser.add_argument("--source-video-path", default=Path("upstream/videos/0.mkv"), type=Path)
    parser.add_argument("--max-frames", default=4, type=int)
    parser.add_argument("--num-pairs", default=2, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-pairs", default=1, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--compact-latent-dim", default=8, type=int)
    parser.add_argument("--compact-embed-dim", default=8, type=int)
    parser.add_argument("--compact-codebook-size", default=16, type=int)
    parser.add_argument("--compact-selector-palette-size", default=16, type=int)
    parser.add_argument("--compact-decoder-channel", default=8, type=int)
    parser.add_argument(
        "--compact-decoder-codec",
        default="int8_mixed",
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
        default=1.0e-4,
        type=float,
    )
    parser.add_argument("--coder-qat-magnitude-weight", default=0.0, type=float)
    parser.add_argument("--coder-qat-delta-weight", default=0.0, type=float)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = [
        args.execute_pr95_mlx_smoke,
        args.execute_pr95_stage8_source,
        args.from_pr95_mlx_report is not None,
        args.from_pr95_stage8_report is not None,
        args.execute_family is not None,
    ]
    if sum(1 for item in modes if item) > 1:
        raise SystemExit(
            "pass only one of --execute-pr95-mlx-smoke, "
            "--execute-pr95-stage8-source, --from-pr95-mlx-report, "
            "--from-pr95-stage8-report, or --execute-family"
        )
    ceilings = tuple(args.hard_byte_ceiling or DEFAULT_BASE_RENDERER_BYTE_CEILINGS)
    output_dir = args.output_dir or _default_output_dir()
    if args.execute_pr95_mlx_smoke:
        report = execute_pr95_mlx_smoke_and_adapt(
            output_dir=output_dir,
            max_frames=args.max_frames,
            smoke_epochs_per_stage=args.smoke_epochs_per_stage,
            training_loss_surface=args.training_loss_surface,
            source_video_path=args.source_video_path,
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
            upstream_dir=args.upstream_dir,
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
            upstream_dir=args.upstream_dir,
        )
    elif args.execute_pr95_stage8_source:
        report = execute_pr95_stage8_source_and_adapt(
            output_dir=output_dir,
            source_archive_zip=args.pr95_source_archive,
            source_video_path=args.source_video_path,
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
            upstream_dir=args.upstream_dir,
        )
    elif args.execute_family == "pr95_hnerv":
        report = execute_pr95_hnerv_mlx_scoreaware_and_adapt(
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            batch_pair_indices_per_step=args.batch_pairs,
            learning_rate=args.learning_rate,
            source_video_path=args.source_video_path,
            source_archive_zip=args.pr95_source_archive,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.latent_dim or 28,
            base_channels=args.base_channels or 36,
            ema_decay=args.compact_ema_decay,
            segnet_distillation_weight=args.segnet_distillation_weight,
            pose_distillation_weight=args.pose_distillation_weight,
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            allow_segnet_only_research=args.allow_segnet_only_research,
            scorer_upstream_dir=args.upstream_dir,
            run_receiver_proof=args.run_receiver_proof,
            receiver_proof_runtime_dir=args.pr95_receiver_runtime_dir,
            keep_receiver_proof_output=args.keep_receiver_proof_output,
            receiver_proof_timeout_seconds=args.receiver_proof_timeout_seconds,
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
            source_video_path=args.source_video_path,
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
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            allow_segnet_only_research=args.allow_segnet_only_research,
            scorer_upstream_dir=args.upstream_dir,
            coder_aware_qat=args.coder_aware_qat,
            coder_qat_quant_bits=args.coder_qat_quant_bits,
            coder_qat_quant_residual_weight=args.coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=args.coder_qat_magnitude_weight,
            coder_qat_delta_weight=args.coder_qat_delta_weight,
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
            source_video_path=args.source_video_path,
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
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            allow_segnet_only_research=args.allow_segnet_only_research,
            scorer_upstream_dir=args.upstream_dir,
            coder_aware_qat=args.coder_aware_qat,
            coder_qat_quant_bits=args.coder_qat_quant_bits,
            coder_qat_quant_residual_weight=args.coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=args.coder_qat_magnitude_weight,
            coder_qat_delta_weight=args.coder_qat_delta_weight,
            random_seed=args.random_seed,
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
            source_video_path=args.source_video_path,
            hard_byte_ceilings=ceilings,
            mlx_profile_paths=tuple(args.mlx_profile),
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.compact_latent_dim,
            embed_dim=args.compact_embed_dim,
            decoder_channel=args.compact_decoder_channel,
            ema_decay=args.compact_ema_decay,
            segnet_distillation_weight=args.segnet_distillation_weight,
            pose_distillation_weight=args.pose_distillation_weight,
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            allow_segnet_only_research=args.allow_segnet_only_research,
            coder_aware_qat=args.coder_aware_qat,
            coder_qat_quant_bits=args.coder_qat_quant_bits,
            coder_qat_quant_residual_weight=args.coder_qat_quant_residual_weight,
            coder_qat_magnitude_weight=args.coder_qat_magnitude_weight,
            coder_qat_delta_weight=args.coder_qat_delta_weight,
            run_local_cpu_replay=args.run_local_cpu_replay,
            keep_local_replay_inflated=args.keep_local_replay_inflated,
            cleanup_failed_local_replay_scratch=not args.retain_failed_local_replay_scratch,
            upstream_dir=args.upstream_dir,
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


def _resolve_source_video_path(path: str | Path, *, base: Path) -> Path:
    """Resolve bulky source video paths across clean SSD source worktrees."""

    env_override = os.environ.get("PACT_SOURCE_VIDEO_PATH")
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    raw = Path(path).expanduser()
    candidates.append(raw if raw.is_absolute() else (base / raw))
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
    if not isinstance(path, str) or not path:
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
