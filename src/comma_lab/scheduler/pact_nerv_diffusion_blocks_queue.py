# SPDX-License-Identifier: MIT
"""Fail-closed Pact-NeRV DiffusionBlocks local MLX queue planning."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import QUEUE_SCHEMA, normalize_queue_definition

PACT_NERV_DIFFUSION_BLOCKS_SCHEDULE_SCHEMA = "pact_nerv_diffusion_blocks_schedule.v1"
PACT_NERV_DIFFUSION_BLOCKS_QUEUE_SCHEMA = "pact_nerv_diffusion_blocks_mlx_queue.v1"
PAPER_BASIS = "arxiv:2506.14202v3"
PACT_NERV_IA3_MLX_SMOKE_SCHEMA = "pact_nerv_ia3_mlx_smoke_manifest_v1_20260528"
PACT_NERV_DIFFUSION_DISTILLED_SMOKE_SCHEMA = "pact_nerv_diffusion_distilled_l0_scaffold_smoke_v1"
PR95_MLX_LONG_TRAINING_PLAN_SCHEMA = "pr95_mlx_long_training_plan.v1"


class PactNervDiffusionBlocksQueueError(ValueError):
    """Raised when a DiffusionBlocks queue contract cannot be built."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def build_pact_nerv_diffusion_blocks_schedule(
    *,
    block_count: int = 3,
    difficulty_mass_source: str = "scorer_region_waterfill_and_master_gradient",
    overlap_fraction: float = 0.0,
) -> dict[str, Any]:
    """Build the mathematical local-training schedule implied by DiffusionBlocks."""

    if block_count < 1:
        raise PactNervDiffusionBlocksQueueError("block_count must be positive")
    if not 0.0 <= overlap_fraction < 0.5:
        raise PactNervDiffusionBlocksQueueError("overlap_fraction must be in [0, 0.5)")
    blocks: list[dict[str, Any]] = []
    for index in range(block_count):
        lo = index / block_count
        hi = (index + 1) / block_count
        blocks.append(
            {
            "block_index": index,
            "difficulty_mass_interval": [lo, hi],
            "overlap_fraction": overlap_fraction,
            "difficulty_coordinate": "normalized_noise_or_mask_mass_u_in_[0,1]",
            "local_training_axis": "[macOS-MLX research-signal]",
            "training_target": "corrupted_feature_or_frame_to_clean_feature_or_frame",
            "export_contract": "deterministic_student_block_only",
            "authority_after_export": "requires_pytorch_or_numpy_forward_parity_then_exact_cpu_cuda_eval",
        }
    )
    schedule = {
        "schema": PACT_NERV_DIFFUSION_BLOCKS_SCHEDULE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "paper_basis": PAPER_BASIS,
        "substrate_family": "pact_nerv_diffusion_blocks",
        "block_count": block_count,
        "difficulty_mass_source": str(difficulty_mass_source),
        "overlap_fraction": overlap_fraction,
        "blocks": blocks,
        "optimization_functional": {
            "schema": "blockwise_nerv_training_action_functional.v1",
            "expression": (
                "argmin_theta Σ_b E_u∈I_b[D_b(x_u,u,c)-x_0]^2 "
                "+ λ_byte*ΔBytes + λ_axis*AxisShiftPenalty + "
                "Σ_(i,j) ψ_ij(stack_interaction)"
            ),
            "difficulty_partition": "equal_mass_intervals_over_response_or_noise_coordinate",
            "interaction_axes": [
                "pixel",
                "bit",
                "byte",
                "frame",
                "pair",
                "region",
                "boundary",
                "batch",
                "full_video",
            ],
            "entropy_positions": ["before_entropy_coder", "at_entropy_coder", "after_entropy_coder"],
            "authority": "local_mlx_research_signal_only_until_export_parity_and_exact_eval",
        },
        "composition_constraints": {
            "schema": "diffusionblocks_chain_composition_constraints.v1",
            "receiver_runtime": "deterministic_student_only",
            "teacher_runtime_shipped": False,
            "inflate_requires_mlx": False,
            "stack_penalty_required": True,
            "negative_result_posterior_update_required": True,
            "byte_tax_audit_required": True,
        },
        "portable_op_contract": [
            "linear",
            "conv2d",
            "depthwise_conv2d",
            "fixed_resize",
            "pixel_shuffle",
            "silu_or_gelu",
            "film_or_adaln_conditioning",
            "add",
            "mul",
            "clamp",
            "round",
        ],
        "proof_obligations": [
            "mlx_training_hash_manifest",
            "numpy_or_pytorch_export_forward_parity",
            "archive_byte_tax_audit",
            "full_frame_inflate_parity_before_promotion",
            "contest_cpu_or_cuda_auth_eval_before_score_claim",
        ],
        "allowed_use": "local_mlx_training_acquisition_and_substrate_shift_probe",
        "forbidden_use": "score_claim_or_rank_or_kill_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(schedule, context="pact_nerv_diffusion_blocks_schedule")
    return schedule


def build_pact_nerv_diffusion_blocks_mlx_queue(
    *,
    repo_root: str | Path,
    queue_id: str,
    output_root: str | Path,
    source_video_path: str | Path = "upstream/videos/0.mkv",
    block_count: int = 3,
    max_pairs: int = 8,
    difficulty_mass_source: str = "scorer_region_waterfill_and_master_gradient",
    overlap_fraction: float = 0.0,
) -> dict[str, Any]:
    """Return an executable, fail-closed queue for local DiffusionBlocks probes."""

    if max_pairs < 1:
        raise PactNervDiffusionBlocksQueueError("max_pairs must be positive")
    root = _resolve(output_root, repo_root)
    schedule_path = root / "diffusion_blocks_schedule.json"
    ia3_smoke_dir = root / "pact_nerv_ia3_mlx_renderer_smoke"
    diffusion_smoke_dir = root / "pact_nerv_diffusion_distilled_smoke"
    pr95_report = root / "pr95_mlx_blockwise_control_plan.json"
    pr95_checkpoints = root / "pr95_mlx_blockwise_checkpoints"
    pr95_telemetry = root / "pr95_mlx_blockwise_telemetry.jsonl"

    schedule_ref = _repo_rel(schedule_path, repo_root)
    ia3_smoke_dir_ref = _repo_rel(ia3_smoke_dir, repo_root)
    ia3_smoke_manifest_ref = _repo_rel(ia3_smoke_dir / "smoke_manifest.json", repo_root)
    diffusion_smoke_dir_ref = _repo_rel(diffusion_smoke_dir, repo_root)
    diffusion_smoke_provenance_ref = _repo_rel(diffusion_smoke_dir / "provenance.json", repo_root)
    pr95_report_ref = _repo_rel(pr95_report, repo_root)
    pr95_checkpoints_ref = _repo_rel(pr95_checkpoints, repo_root)
    pr95_telemetry_ref = _repo_rel(pr95_telemetry, repo_root)
    source_video_ref = _repo_rel(_resolve(source_video_path, repo_root), repo_root)
    output_root_ref = _repo_rel(root, repo_root)

    queue = {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "metadata": {
            "schema": PACT_NERV_DIFFUSION_BLOCKS_QUEUE_SCHEMA,
            "queue_id": queue_id,
            "paper_basis": PAPER_BASIS,
            "axis": "[macOS-MLX research-signal]",
            "source_video_path": source_video_ref,
            "output_root": output_root_ref,
            "schedule_path": schedule_ref,
            "pact_nerv_ia3_mlx_smoke_manifest_path": ia3_smoke_manifest_ref,
            "pact_nerv_diffusion_distilled_smoke_provenance_path": diffusion_smoke_provenance_ref,
            "block_count": block_count,
            "max_pairs": max_pairs,
            "difficulty_mass_source": difficulty_mass_source,
            "overlap_fraction": overlap_fraction,
            "teacher_runtime_shipped": False,
            "inflate_requires_mlx": False,
            "exact_cpu_cuda_required": True,
            "allowed_use": "local_training_probe_and_substrate_shift_acquisition",
            "forbidden_use": "score_claim_or_promotion_or_exact_dispatch",
            **FALSE_AUTHORITY,
        },
        "controls": {
            "mode": "running",
            "local_first": True,
            "max_concurrency": {"local_cpu": 1, "local_mlx": 1},
        },
        "experiments": [
            {
                "id": "pact_nerv_diffusion_blocks_mlx_smoke",
                "priority": 1,
                "status": "queued",
                "tags": [
                    "pact-nerv",
                    "diffusionblocks",
                    "mlx-local",
                    "class-shift-probe",
                    "no-score-authority",
                ],
                "metadata": {
                    "schema": "pact_nerv_diffusion_blocks_mlx_experiment.v1",
                    "paper_basis": PAPER_BASIS,
                    "budget_spend_allowed": False,
                    **FALSE_AUTHORITY,
                },
                "steps": [
                    {
                        "id": "emit_diffusion_blocks_schedule",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_pact_nerv_diffusion_blocks_schedule.py",
                            "--output",
                            schedule_ref,
                            "--block-count",
                            str(block_count),
                            "--difficulty-mass-source",
                            difficulty_mass_source,
                            "--overlap-fraction",
                            str(overlap_fraction),
                            "--overwrite",
                        ],
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 60,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": schedule_ref,
                                "key": "schema",
                                "equals": PACT_NERV_DIFFUSION_BLOCKS_SCHEDULE_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": schedule_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [schedule_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "run_pact_nerv_diffusion_distilled_smoke",
                        "kind": "command",
                        "requires": ["emit_diffusion_blocks_schedule"],
                        "command": [
                            ".venv/bin/python",
                            "experiments/train_substrate_pact_nerv_diffusion_distilled.py",
                            "--smoke",
                            "--device",
                            "cpu",
                            "--epochs",
                            "1",
                            "--batch-size",
                            str(max(2, min(max_pairs, 8))),
                            "--output-dir",
                            diffusion_smoke_dir_ref,
                        ],
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": diffusion_smoke_provenance_ref,
                                "key": "schema",
                                "equals": PACT_NERV_DIFFUSION_DISTILLED_SMOKE_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": diffusion_smoke_provenance_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [diffusion_smoke_dir_ref, diffusion_smoke_provenance_ref],
                            "input_artifact_paths": [schedule_ref],
                            "include_postcondition_paths": True,
                            "recursive": True,
                            "max_recursive_entries": 64,
                        },
                    },
                    {
                        "id": "run_pact_nerv_ia3_mlx_renderer_smoke",
                        "kind": "command",
                        "requires": ["emit_diffusion_blocks_schedule"],
                        "command": [
                            ".venv/bin/python",
                            "experiments/train_substrate_pact_nerv_ia3_mlx_local.py",
                            "--smoke",
                            "--output-dir",
                            ia3_smoke_dir_ref,
                            "--num-pairs",
                            str(max_pairs),
                        ],
                        "resources": {"kind": "local_mlx"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": ia3_smoke_manifest_ref,
                                "key": "schema_version",
                                "equals": PACT_NERV_IA3_MLX_SMOKE_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": ia3_smoke_manifest_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [ia3_smoke_dir_ref, ia3_smoke_manifest_ref],
                            "input_artifact_paths": [schedule_ref],
                            "include_postcondition_paths": True,
                            "recursive": True,
                            "max_recursive_entries": 64,
                        },
                    },
                    {
                        "id": "plan_pr95_mlx_blockwise_control",
                        "kind": "command",
                        "requires": ["emit_diffusion_blocks_schedule"],
                        "command": [
                            ".venv/bin/python",
                            "tools/run_pr95_mlx_long_training.py",
                            "--output-report",
                            pr95_report_ref,
                            "--source-video-path",
                            source_video_ref,
                            "--checkpoint-root",
                            pr95_checkpoints_ref,
                            "--telemetry-path",
                            pr95_telemetry_ref,
                            "--operator-run-label",
                            "diffusionblocks_blockwise_control",
                            "--smoke-mode",
                            "--smoke-epochs-per-stage",
                            "1",
                            "--max-frames",
                            str(max_pairs * 2),
                            "--hash-source-video",
                            "--execute-smoke",
                        ],
                        "resources": {"kind": "local_mlx"},
                        "timeout_seconds": 900,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": pr95_report_ref,
                                "key": "schema",
                                "equals": PR95_MLX_LONG_TRAINING_PLAN_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": pr95_report_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [pr95_report_ref],
                            "input_artifact_paths": [schedule_ref, source_video_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                ],
            }
        ],
    }
    return normalize_queue_definition(queue)


__all__ = [
    "PACT_NERV_DIFFUSION_BLOCKS_QUEUE_SCHEMA",
    "PACT_NERV_DIFFUSION_BLOCKS_SCHEDULE_SCHEMA",
    "PACT_NERV_DIFFUSION_DISTILLED_SMOKE_SCHEMA",
    "PACT_NERV_IA3_MLX_SMOKE_SCHEMA",
    "PactNervDiffusionBlocksQueueError",
    "build_pact_nerv_diffusion_blocks_mlx_queue",
    "build_pact_nerv_diffusion_blocks_schedule",
]
