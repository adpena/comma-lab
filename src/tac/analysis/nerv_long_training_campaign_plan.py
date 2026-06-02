# SPDX-License-Identifier: MIT
"""Queue-safe long-training campaign plan for HiNeRV and SNeRV.

This module turns modelsize candidates into concrete MLX-first campaign rows.
It is not an executor and never grants score authority. Its job is to keep the
top-priority carrier race dynamic: modelsize, optimizer, scorer pressure, QAT,
archive proof, MLX prefilter, local CPU replay, and exact-auth promotion gates
are compiled into the same row so future agents do not hand-launch arbitrary
partial runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.analysis.nerv_candidate_curriculum import (
    build_hinerv_candidate_curriculum_plan,
    build_snerv_candidate_curriculum_plan,
)
from tac.analysis.nerv_candidate_feedback import (
    SCHEMA as NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA,
)
from tac.analysis.nerv_candidate_feedback import (
    build_nerv_candidate_feedback_row,
)
from tac.analysis.nerv_decoder_weight_waterfill import (
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
)
from tac.analysis.nerv_modelsize_budget import (
    decoder_codec_nominal_bits,
    snerv_decoder_codec_nominal_bits,
)
from tac.optimization.recon_pixel_weight_surface import (
    JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA,
)
from tac.substrates._shared.mlx_score_aware.adapter import (
    SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "nerv_long_training_campaign_plan.v1"
ROW_SCHEMA = "nerv_long_training_campaign_row.v1"
EXPERIMENT_QUEUE_SCHEMA = "experiment_queue.v1"
SCORE_LOWERING_GATE_SCHEMA = "nerv_long_training_score_lowering_gate.v1"
DEFAULT_OUTPUT_ROOT = "/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns"
DEFAULT_EPOCHS = 29_650
DEFAULT_BATCH_PAIRS = 8
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_OPTIMIZER_KINDS = (
    "adamw",
    "lion",
    "adafactor",
    "rmsprop",
)


class NervLongTrainingCampaignPlanError(ValueError):
    """Raised when a long-training campaign plan is malformed."""


def build_nerv_long_training_campaign_plan(
    *,
    hinerv_modelsize_budget: Mapping[str, Any],
    snerv_modelsize_budget: Mapping[str, Any],
    optimizer_kinds: Sequence[str] = DEFAULT_OPTIMIZER_KINDS,
    epochs: int = DEFAULT_EPOCHS,
    batch_pairs: int = DEFAULT_BATCH_PAIRS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_candidates_per_family: int = 3,
    joint_recon_weight_manifest_paths: Sequence[str | Path] = (),
    candidate_feedback_sources: Sequence[Mapping[str, Any]] = (),
    decoder_weight_waterfill_sources: Sequence[Mapping[str, Any]] = (),
    snerv_bounded_proof_only: bool = False,
    snerv_bounded_proof_epochs: int = 3,
) -> dict[str, Any]:
    """Build the shared HiNeRV/SNeRV long-training campaign matrix."""

    _require_schema(
        hinerv_modelsize_budget,
        "nerv_modelsize_budget.v1",
        "hinerv_modelsize_budget",
    )
    _require_schema(
        snerv_modelsize_budget,
        "snerv_modelsize_budget.v1",
        "snerv_modelsize_budget",
    )
    optimizers = _optimizer_tuple(optimizer_kinds)
    if int(epochs) < 8:
        raise NervLongTrainingCampaignPlanError("epochs must be >= 8")
    if int(batch_pairs) <= 0:
        raise NervLongTrainingCampaignPlanError("batch_pairs must be positive")
    if float(learning_rate) <= 0.0:
        raise NervLongTrainingCampaignPlanError("learning_rate must be positive")
    joint_recon_weight_artifacts = _load_verified_joint_recon_weight_artifacts(
        joint_recon_weight_manifest_paths
    )
    candidate_feedback_index = _candidate_feedback_index(candidate_feedback_sources)
    decoder_weight_waterfill_index = _decoder_weight_waterfill_index(
        decoder_weight_waterfill_sources
    )

    rows: list[dict[str, Any]] = []
    hi_candidates = _selected_candidates(
        hinerv_modelsize_budget,
        family="hi_nerv",
        limit=max_candidates_per_family,
    )
    snerv_candidates = _selected_candidates(
        snerv_modelsize_budget,
        family="snerv",
        limit=max_candidates_per_family,
    )
    for candidate in hi_candidates:
        for optimizer in optimizers:
            rows.append(
                _hinerv_campaign_row(
                    candidate=candidate,
                    optimizer_kind=optimizer,
                    epochs=int(epochs),
                    batch_pairs=int(batch_pairs),
                    learning_rate=float(learning_rate),
                    output_root=Path(output_root),
                    joint_recon_weight_artifacts=joint_recon_weight_artifacts,
                    candidate_feedback_index=candidate_feedback_index,
                    decoder_weight_waterfill_index=decoder_weight_waterfill_index,
                )
            )
    for candidate in snerv_candidates:
        rows.append(
            _snerv_campaign_row(
                candidate=candidate,
                epochs=int(epochs),
                output_root=Path(output_root),
                candidate_feedback_index=candidate_feedback_index,
                bounded_proof_only=bool(snerv_bounded_proof_only),
                bounded_proof_epochs=int(snerv_bounded_proof_epochs),
            )
        )

    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("priority") or 999),
            str(row.get("family") or ""),
            str(row.get("row_id") or ""),
        ),
    )
    experiment_queue = _experiment_queue(rows)
    return {
        "schema": SCHEMA,
        "baseline_to_beat": "pr95_public_control_arm_plus_frontier_exact_axes",
        "objective": (
            "Make HiNeRV and SNeRV full-stack PR95-or-better carriers: "
            "modelsize-byte constrained, real SegNet/PoseNet scorer-bound, "
            "QAT/coder pressured, MLX-first, NumPy-portable, receiver-proven, "
            "CPU replay gated, and exact-auth only after local wins."
        ),
        "top_priority_families": ["hi_nerv", "snerv"],
        "optimizer_kinds": list(optimizers),
        "epochs": int(epochs),
        "batch_pairs": int(batch_pairs),
        "learning_rate": float(learning_rate),
        "output_root": Path(output_root).as_posix(),
        "joint_recon_weight_artifacts": list(joint_recon_weight_artifacts.values()),
        "joint_recon_weight_artifact_count": len(joint_recon_weight_artifacts),
        "candidate_feedback_source_count": len(candidate_feedback_sources),
        "snerv_bounded_proof_only": bool(snerv_bounded_proof_only),
        "snerv_bounded_proof_epochs": int(snerv_bounded_proof_epochs),
        "candidate_feedback_row_count": sum(
            len(rows_for_key) for rows_for_key in candidate_feedback_index.values()
        ),
        "decoder_weight_waterfill_source_count": len(decoder_weight_waterfill_sources),
        "decoder_weight_waterfill_row_count": sum(
            len(rows_for_key)
            for rows_for_key in decoder_weight_waterfill_index.values()
        ),
        "campaign_rows": rows,
        "campaign_row_count": len(rows),
        "experiment_queue": experiment_queue,
        "experiment_queue_schema": EXPERIMENT_QUEUE_SCHEMA,
        "experiment_queue_id": experiment_queue["queue_id"],
        "experiment_queue_experiment_count": len(experiment_queue["experiments"]),
        "launchable_local_row_count": sum(
            1 for row in rows if row["local_mlx_launch_command_ready"]
        ),
        "blocked_row_count": sum(1 for row in rows if row["blockers"]),
        "family_counts": _family_counts(rows),
        "decoder_weight_waterfill_attached_row_count": sum(
            1
            for row in rows
            if isinstance(row.get("decoder_weight_waterfill_plan"), Mapping)
            and row["decoder_weight_waterfill_plan"].get("attached") is True
        ),
        "promotion_policy": {
            "schema": "nerv_long_training_campaign_promotion_policy.v1",
            "mlx_role": "fast acquisition and prefilter only",
            "local_cpu_role": "full local replay gate for MLX-filtered winners",
            "exact_cpu_role": "first auth axis for true local winners",
            "exact_cuda_role": "second auth axis only after exact CPU clears",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "blockers": _dedupe(
            [
                "campaign_plan_is_not_execution",
                "exact_cpu_cuda_not_launched_by_campaign_plan",
                *[
                    blocker
                    for row in rows
                    for blocker in row.get("blockers", [])
                    if _plan_level_blocker(blocker)
                ],
            ]
        ),
        **FALSE_AUTHORITY,
    }


def render_nerv_long_training_campaign_plan_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing campaign summary."""

    lines = [
        "# NeRV Long-Training Campaign Plan",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Rows: `{report.get('campaign_row_count')}`",
        f"Launchable local rows: `{report.get('launchable_local_row_count')}`",
        f"Blocked rows: `{report.get('blocked_row_count')}`",
        f"Score claim: `{report.get('score_claim')}`",
        f"Ready for exact dispatch: `{report.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Rows",
        "",
    ]
    for row in report.get("campaign_rows") or ():
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                f"- `{row.get('row_id')}`",
                f"  family: `{row.get('family')}`",
                f"  launchable_mlx: `{row.get('local_mlx_launch_command_ready')}`",
                f"  optimizer: `{row.get('optimizer_kind')}`",
                f"  blockers: `{len(row.get('blockers') or [])}`",
            ]
        )
    lines.extend(["", "## Blockers", ""])
    blockers = list(report.get("blockers") or ())
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _hinerv_campaign_row(
    *,
    candidate: Mapping[str, Any],
    optimizer_kind: str,
    epochs: int,
    batch_pairs: int,
    learning_rate: float,
    output_root: Path,
    joint_recon_weight_artifacts: Mapping[int, Mapping[str, Any]] | None = None,
    candidate_feedback_index: (
        Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None
    ) = None,
    decoder_weight_waterfill_index: (
        Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None
    ) = None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "hinerv_candidate")
    quant_bits = min(8, decoder_codec_nominal_bits(str(candidate.get("decoder_codec"))))
    num_pairs = int(candidate.get("num_pairs") or 600)
    joint_recon_weight = dict(
        (joint_recon_weight_artifacts or {}).get(num_pairs) or {}
    )
    feedback = _candidate_feedback_for(
        candidate=candidate,
        family="hi_nerv",
        index=candidate_feedback_index,
    )
    decoder_weight_waterfill = _decoder_weight_waterfill_for(
        candidate=candidate,
        family="hi_nerv",
        index=decoder_weight_waterfill_index,
    )
    launch_feedback_adjustment = _hinerv_feedback_launch_adjustment(
        feedback=feedback,
        learning_rate=float(learning_rate),
    )
    effective_learning_rate = float(
        launch_feedback_adjustment.get("learning_rate") or learning_rate
    )
    output_dir_basename = _campaign_output_basename(
        row_id=f"hi_nerv::{candidate_id}::{optimizer_kind}",
        launch_feedback_adjustment=launch_feedback_adjustment,
    )
    curriculum = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=int(epochs),
        num_pairs=num_pairs,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=int(quant_bits),
        recon_pixel_weight_attached=True,
        eval_roundtrip_ste_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        receiver_proof_attached=bool(feedback.get("receiver_proof_attached")),
        full_video_local_prefilter_attached=bool(
            feedback.get("full_video_local_prefilter_attached")
        ),
        local_cpu_replay_gate_attached=bool(
            feedback.get("local_cpu_replay_gate_attached")
        ),
        measured_archive_bytes=feedback.get("measured_archive_bytes"),
        measured_num_pairs=feedback.get("measured_num_pairs"),
    )
    row_id = f"hi_nerv::{candidate_id}::{optimizer_kind}"
    command = [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "runtime",
        "--extra",
        "mlx",
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        "hi_nerv",
        "--num-pairs",
        str(num_pairs),
        "--epochs",
        str(int(epochs)),
        "--batch-pairs",
        str(int(batch_pairs)),
        "--learning-rate",
        _float_token(effective_learning_rate),
        "--modelsize-candidate-id",
        candidate_id,
        "--segnet-distillation-weight",
        "1.0",
        "--pose-distillation-weight",
        "1.0",
        "--coder-aware-qat",
        "--coder-qat-quant-bits",
        str(int(quant_bits)),
        "--optimizer-kind",
        str(optimizer_kind),
        "--run-post-export-materializers",
        "--output-dir",
        (output_root / output_dir_basename).as_posix(),
    ]
    if joint_recon_weight:
        command.extend(
            [
                "--recon-pixel-weight-path",
                str(joint_recon_weight["weight_path"]),
            ]
        )
    else:
        command.append("--auto-joint-recon-pixel-weight")
    if decoder_weight_waterfill:
        command.extend(
            [
                "--decoder-weight-waterfill-plan-json",
                str(decoder_weight_waterfill["path"]),
            ]
        )
    blockers = [
        (
            ""
            if joint_recon_weight
            else "requires_verified_joint_p18_p19_recon_pixel_weight_artifact"
        ),
        (
            ""
            if decoder_weight_waterfill
            else "hinerv_decoder_weight_waterfill_plan_missing"
        ),
        "requires_full_video_mlx_prefilter_before_local_cpu_replay_unlock",
        "requires_local_cpu_replay_win_before_exact_cpu_auth",
        *list(curriculum.get("blockers") or []),
    ]
    if (
        feedback.get("pose_instability_detected") is True
        and not launch_feedback_adjustment.get("applied")
    ):
        blockers.append("hinerv_pose_instability_feedback_unapplied")
    if candidate.get("nominal_under_ceiling") is not True:
        blockers.append("hinerv_candidate_nominal_over_byte_ceiling")
    blockers = _dedupe(blockers)
    prelaunch_gate = dict(curriculum.get("long_campaign_prelaunch_gate") or {})
    return _row(
        row_id=row_id,
        family="hi_nerv",
        priority=10 if optimizer_kind in {"adamw", "lion"} else 11,
        candidate=candidate,
        curriculum_plan=curriculum,
        command_argv=command,
        local_mlx_launch_command_ready=bool(prelaunch_gate.get("launch_allowed")),
        implementation_status="shared_mlx_scoreaware_runner_launchable",
        blockers=blockers,
        extra={
            "optimizer_kind": str(optimizer_kind),
            "quant_bits": int(quant_bits),
            "joint_recon_pixel_weight_artifact": joint_recon_weight or None,
            "decoder_weight_waterfill_plan": (
                _decoder_weight_waterfill_row_metadata(decoder_weight_waterfill)
                if decoder_weight_waterfill
                else {
                    "schema": (
                        "nerv_long_training_decoder_weight_waterfill_"
                        "attachment.v1"
                    ),
                    "attached": False,
                    "reason": "no_matching_decoder_weight_waterfill_plan",
                    **FALSE_AUTHORITY,
                }
            ),
            "feedback_launch_adjustment": launch_feedback_adjustment,
            "candidate_feedback": feedback or None,
            "output_dir_basename": output_dir_basename,
            "output_dir_reuse_policy": (
                "fresh_feedback_mutation_path"
                if launch_feedback_adjustment.get("applied")
                else "stable_candidate_optimizer_path"
            ),
        },
    )


def _snerv_campaign_row(
    *,
    candidate: Mapping[str, Any],
    epochs: int,
    output_root: Path,
    candidate_feedback_index: (
        Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None
    ) = None,
    bounded_proof_only: bool = False,
    bounded_proof_epochs: int = 3,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "snerv_candidate")
    feedback = _candidate_feedback_for(
        candidate=candidate,
        family="snerv",
        index=candidate_feedback_index,
    )
    execution_epochs = (
        min(int(epochs), max(1, int(bounded_proof_epochs)))
        if bounded_proof_only
        else int(epochs)
    )
    quant_bits = min(
        8,
        snerv_decoder_codec_nominal_bits(str(candidate.get("decoder_payload_codec"))),
    )
    curriculum = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=int(execution_epochs),
        num_pairs=int(candidate.get("num_pairs") or 600),
        step_map_coder_mode="waterfill",
        native_mlx_train_export_attached=True,
        native_mlx_long_training_bound=not bool(bounded_proof_only),
        native_mlx_receiver_proof_passed=bool(
            feedback.get("native_mlx_receiver_proof_passed")
        ),
        native_mlx_full600_campaign_ready=bool(
            feedback.get("native_mlx_full600_campaign_ready")
        ),
        native_mlx_scorer_loop_qat_attached=True,
        native_mlx_scorer_loop_qat_receiver_contract_satisfied=bool(
            feedback.get("native_mlx_scorer_loop_qat_receiver_contract_satisfied")
        ),
        native_mlx_scorer_loop_qat_ready_for_pose_guard_gate=bool(
            feedback.get("native_mlx_scorer_loop_qat_ready_for_pose_guard_gate")
        ),
        native_mlx_scorer_loop_qat_accepted_improvement=bool(
            feedback.get("native_mlx_scorer_loop_qat_accepted_improvement")
        ),
        native_mlx_scorer_loop_qat_best_materialized=bool(
            feedback.get("native_mlx_scorer_loop_qat_best_materialized")
        ),
        receiver_proof_attached=bool(feedback.get("receiver_proof_attached")),
        full_video_local_prefilter_attached=bool(
            feedback.get("full_video_local_prefilter_attached")
        ),
        local_cpu_replay_gate_attached=bool(
            feedback.get("local_cpu_replay_gate_attached")
        ),
        measured_packet_bytes=feedback.get("measured_payload_bytes"),
        measured_archive_bytes=feedback.get("measured_archive_bytes"),
        measured_num_pairs=feedback.get("measured_num_pairs"),
    )
    row_id = f"snerv::{candidate_id}::native_rate_aware_training"
    command = [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "runtime",
        "--extra",
        "mlx",
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        "snerv",
        "--num-pairs",
        str(int(candidate.get("num_pairs") or 600)),
        "--epochs",
        str(int(execution_epochs)),
        "--modelsize-candidate-id",
        candidate_id,
        "--coder-aware-qat",
        "--coder-qat-quant-bits",
        str(int(quant_bits)),
        "--snerv-scorer-loop-qat",
        "--snerv-scorer-loop-search-mode",
        "learned_random_subspace",
        "--snerv-spectra-preserving-adapter",
        "--output-dir",
        (output_root / _safe_path_token(row_id)).as_posix(),
    ]
    blockers = _dedupe(
        [
            (
                "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
                if bounded_proof_only
                else ""
            ),
            "snerv_native_rate_pressure_in_loop_not_yet_training_authority",
            "snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes"
            if candidate.get("nominal_under_ceiling") is not True
            else "",
            *list(curriculum.get("blockers") or []),
        ]
    )
    return _row(
        row_id=row_id,
        family="snerv",
        priority=12,
        candidate=candidate,
        curriculum_plan=curriculum,
        command_argv=command,
        local_mlx_launch_command_ready=True,
        implementation_status=(
            "bounded_native_export_scorer_loop_stage_ready"
            if bounded_proof_only
            else "native_rate_aware_long_training_queue_ready"
        ),
        blockers=blockers,
        extra={
            "optimizer_kind": None,
            "quant_bits": int(quant_bits),
            "planned_long_training_epochs": int(epochs),
            "execution_epochs": int(execution_epochs),
            "current_command_is_bounded_proof_not_long_training": bool(
                bounded_proof_only
            ),
            "snerv_bounded_proof_epochs": int(bounded_proof_epochs),
            "candidate_feedback": feedback or None,
        },
    )


def _row(
    *,
    row_id: str,
    family: str,
    priority: int,
    candidate: Mapping[str, Any],
    curriculum_plan: Mapping[str, Any],
    command_argv: Sequence[str],
    local_mlx_launch_command_ready: bool,
    implementation_status: str,
    blockers: Sequence[str],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    score_gate = _score_lowering_gate(
        family=family,
        local_mlx_launch_command_ready=local_mlx_launch_command_ready,
        curriculum_plan=curriculum_plan,
        blockers=blockers,
    )
    return {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "family": family,
        "priority": int(priority),
        "candidate_id": candidate.get("candidate_id"),
        "hard_byte_ceiling": int(candidate.get("hard_byte_ceiling") or 0),
        "candidate_nominal_total_payload_bytes": int(
            candidate.get("nominal_total_payload_bytes") or 0
        ),
        "candidate_nominal_under_ceiling": bool(candidate.get("nominal_under_ceiling")),
        "local_mlx_launch_command_ready": bool(local_mlx_launch_command_ready),
        "implementation_status": str(implementation_status),
        "command_argv": list(command_argv),
        "experiment_queue_entry": _experiment_for_row(
            row_id=row_id,
            family=family,
            priority=priority,
            command_argv=command_argv,
            local_mlx_launch_command_ready=local_mlx_launch_command_ready,
            blockers=blockers,
            score_lowering_gate=score_gate,
        ),
        "curriculum_plan": dict(curriculum_plan),
        "score_lowering_gate": score_gate,
        "local_mlx_executable": bool(score_gate["local_mlx_executable"]),
        "cpu_replay_ready": bool(score_gate["cpu_replay_ready"]),
        "exact_gate_ready": bool(score_gate["exact_gate_ready"]),
        "promotion_blockers": list(score_gate["promotion_blockers"]),
        "blockers": _dedupe([str(blocker) for blocker in blockers if blocker]),
        **dict(extra),
        **FALSE_AUTHORITY,
    }


def _experiment_queue(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_QUEUE_SCHEMA,
        "queue_id": "nerv_long_training_campaign_queue.v1",
        "owner": "nerv_long_training_campaign_plan",
        "description": (
            "Queue-owned MLX-first HiNeRV/SNeRV long-training campaign. "
            "Rows are false-authority until receiver proof, local CPU replay, "
            "and exact CPU/CUDA gates pass."
        ),
        "experiments": [
            dict(row["experiment_queue_entry"])
            for row in rows
            if isinstance(row.get("experiment_queue_entry"), Mapping)
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _experiment_for_row(
    *,
    row_id: str,
    family: str,
    priority: int,
    command_argv: Sequence[str],
    local_mlx_launch_command_ready: bool,
    blockers: Sequence[str],
    score_lowering_gate: Mapping[str, Any],
) -> dict[str, Any]:
    output_json = _row_output_report_path(command_argv)
    postconditions = [
        {
            "type": "json_equals",
            "path": output_json,
            "key": "schema",
            "equals": "compact_renderer_mlx_spine_runner.v1",
        },
        {
            "type": "json_equals",
            "path": output_json,
            "key": "score_claim",
            "equals": False,
        },
        {
            "type": "json_equals",
            "path": output_json,
            "key": "ready_for_exact_eval_dispatch",
            "equals": False,
        },
        {
            "type": "json_equals",
            "path": output_json,
            "key": "promotion_eligible",
            "equals": False,
        },
    ]
    if str(family) == "hi_nerv":
        postconditions.extend(
            [
                {
                    "type": "json_equals",
                    "path": output_json,
                    "key": "execute_family",
                    "equals": "hi_nerv",
                },
                {
                    "type": "json_equals",
                    "path": output_json,
                    "key": "training_executed",
                    "equals": True,
                },
            ]
        )
    elif str(family) == "snerv":
        postconditions.append(
            {
                "type": "json_equals",
                "path": output_json,
                "key": "execute_family",
                "equals": "snerv",
            }
        )
        bounded_blocker = (
            "snerv_scoreaware_long_training_not_bound_"
            "bounded_native_export_stage_only"
        )
        if bounded_blocker in blockers:
            postconditions.append(
                {
                    "type": "json_array_contains",
                    "path": output_json,
                    "key": "blockers",
                    "contains": bounded_blocker,
                }
            )
    return {
        "id": _safe_path_token(row_id),
        "family": str(family),
        "priority": int(priority),
        "status": "queued" if bool(local_mlx_launch_command_ready) else "disabled",
        "blocked": not bool(local_mlx_launch_command_ready),
        "blockers": _dedupe([str(blocker) for blocker in blockers if blocker]),
        "score_lowering_gate": dict(score_lowering_gate),
        "cpu_replay_ready": bool(score_lowering_gate["cpu_replay_ready"]),
        "exact_gate_ready": bool(score_lowering_gate["exact_gate_ready"]),
        "steps": [
            {
                "id": "run_mlx_first_campaign_row",
                "command": list(command_argv),
                "resources": {
                    "kind": "local_mlx",
                    "max_parallel_group": "local_mlx_training",
                },
                "postconditions": postconditions,
                "on_postcondition_failure": "failed",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _row_output_report_path(command_argv: Sequence[str]) -> str:
    argv = [str(value) for value in command_argv]
    try:
        out_dir = argv[argv.index("--output-dir") + 1]
    except (ValueError, IndexError):
        out_dir = DEFAULT_OUTPUT_ROOT
    return (Path(out_dir) / "compact_renderer_mlx_spine_runner_report.json").as_posix()


def _score_lowering_gate(
    *,
    family: str,
    local_mlx_launch_command_ready: bool,
    curriculum_plan: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    """Separate launchability from score/promotion authority for campaign rows."""

    binding = (
        curriculum_plan.get("pr95_stack_binding")
        if isinstance(curriculum_plan.get("pr95_stack_binding"), Mapping)
        else {}
    )
    gate = (
        curriculum_plan.get("long_campaign_prelaunch_gate")
        if isinstance(curriculum_plan.get("long_campaign_prelaunch_gate"), Mapping)
        else {}
    )
    missing_rows = [
        dict(row)
        for row in binding.get("rows", [])
        if isinstance(row, Mapping) and row.get("satisfied") is not True
    ]
    missing_requirement_ids = [
        str(row.get("requirement_id"))
        for row in missing_rows
        if row.get("requirement_id")
    ]
    post_run_requirements = [
        str(item) for item in gate.get("post_run_requirements_excluded", []) if item
    ]
    post_run_missing = [
        requirement
        for requirement in missing_requirement_ids
        if requirement in post_run_requirements
    ]
    promotion_blockers = _dedupe(
        [
            *(str(blocker) for blocker in blockers if blocker),
            *(
                f"{family}_{requirement}_missing"
                for requirement in post_run_missing
            ),
        ]
    )
    prelaunch_blockers = [
        str(blocker) for blocker in gate.get("blockers", []) if blocker
    ]
    cpu_replay_ready = (
        bool(local_mlx_launch_command_ready)
        and "receiver_proof" not in post_run_missing
        and "full_video_local_prefilter" not in post_run_missing
        and "local_cpu_replay_gate" not in post_run_missing
        and not prelaunch_blockers
    )
    exact_gate_ready = (
        cpu_replay_ready
        and "exact_auth_gate_plan" not in post_run_missing
        and not promotion_blockers
    )
    return {
        "schema": SCORE_LOWERING_GATE_SCHEMA,
        "family": str(family),
        "local_mlx_executable": bool(local_mlx_launch_command_ready),
        "prelaunch_allowed": bool(gate.get("launch_allowed")),
        "prelaunch_blockers": _dedupe(prelaunch_blockers),
        "post_run_requirements": post_run_requirements,
        "missing_requirement_ids": _dedupe(missing_requirement_ids),
        "post_run_missing_requirement_ids": _dedupe(post_run_missing),
        "receiver_proof_required": "receiver_proof" in post_run_missing,
        "full_video_prefilter_required": "full_video_local_prefilter" in post_run_missing,
        "local_cpu_replay_required": "local_cpu_replay_gate" in post_run_missing,
        "exact_auth_gate_required": "exact_auth_gate_plan" in post_run_missing,
        "cpu_replay_ready": bool(cpu_replay_ready),
        "exact_gate_ready": bool(exact_gate_ready),
        "promotion_blockers": promotion_blockers,
        **FALSE_AUTHORITY,
    }


def _require_schema(payload: Mapping[str, Any], schema: str, name: str) -> None:
    if not isinstance(payload, Mapping):
        raise NervLongTrainingCampaignPlanError(f"{name} must be a mapping")
    if payload.get("schema") != schema:
        raise NervLongTrainingCampaignPlanError(
            f"{name} schema must be {schema}; got {payload.get('schema')}"
        )


def _selected_candidates(
    payload: Mapping[str, Any],
    *,
    family: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in payload.get("selected_candidates", [])
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    return rows[: max(1, int(limit))]


def _optimizer_tuple(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    supported = set(SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS)
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text not in supported:
            raise NervLongTrainingCampaignPlanError(
                f"unsupported optimizer kind: {value!r}"
            )
        if text not in out:
            out.append(text)
    if not out:
        raise NervLongTrainingCampaignPlanError("at least one optimizer is required")
    return tuple(out)


def _candidate_feedback_index(
    sources: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in sources:
        row = _normalize_candidate_feedback_source(source)
        family = _feedback_family(row)
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not family or not candidate_id:
            continue
        index.setdefault((family, candidate_id), []).append(row)
    return {
        key: sorted(
            rows,
            key=lambda row: (
                bool(row.get("scope_matches_candidate")),
                int(row.get("measured_num_pairs") or 0),
                bool(row.get("receiver_proof_attached")),
                bool(row.get("full_video_local_prefilter_attached")),
                bool(row.get("local_cpu_replay_gate_attached")),
            ),
            reverse=True,
        )
        for key, rows in index.items()
    }


def _normalize_candidate_feedback_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("schema") == NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA:
        row = dict(source)
    elif source.get("schema") == "compact_renderer_mlx_spine_runner.v1":
        row = build_nerv_candidate_feedback_row(
            runner_report=source,
            source_report_path=source.get("_candidate_feedback_source_path"),
        )
    else:
        row = dict(source)
    return _augment_feedback_row(row, source)


def _augment_feedback_row(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    native = source.get("snerv_mlx_native_export")
    if isinstance(native, Mapping):
        out.setdefault(
            "native_mlx_receiver_proof_passed",
            bool(
                native.get("receiver_proof_passed")
                and native.get("receiver_contract_satisfied")
            ),
        )
        out.setdefault(
            "native_mlx_full600_campaign_ready",
            bool(native.get("native_mlx_full600_campaign_ready")),
        )
        out.setdefault(
            "native_mlx_scorer_loop_qat_receiver_contract_satisfied",
            bool(native.get("scorer_loop_qat_receiver_contract_satisfied")),
        )
        out.setdefault(
            "native_mlx_scorer_loop_qat_ready_for_pose_guard_gate",
            bool(native.get("scorer_loop_qat_ready_for_pose_guard_gate")),
        )
        out.setdefault(
            "native_mlx_scorer_loop_qat_accepted_improvement",
            bool(native.get("scorer_loop_qat_accepted_improvement")),
        )
        out.setdefault(
            "native_mlx_scorer_loop_qat_best_materialized",
            bool(native.get("scorer_loop_qat_best_materialized")),
        )
    if "receiver_proof_attached" not in out:
        receiver_paths = out.get("receiver_proof_report_paths")
        out["receiver_proof_attached"] = bool(
            out.get("scope_matches_candidate")
            and (
                out.get("native_mlx_receiver_proof_passed")
                or (isinstance(receiver_paths, list) and len(receiver_paths) > 0)
            )
        )
    if "full_video_local_prefilter_attached" not in out:
        out["full_video_local_prefilter_attached"] = bool(
            out.get("mlx_prefilter_has_full_video")
        )
    if "local_cpu_replay_gate_attached" not in out:
        out["local_cpu_replay_gate_attached"] = bool(
            out.get("local_cpu_replay_gate_executed")
            or out.get("local_cpu_replay_summary_present")
        )
    return out


def _candidate_feedback_for(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return {}
    rows = list((index or {}).get((_family_key(family), candidate_id)) or [])
    return dict(rows[0]) if rows else {}


def _decoder_weight_waterfill_index(
    sources: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in sources:
        row = _normalize_decoder_weight_waterfill_source(source)
        family = _family_key(str(row.get("family") or ""))
        if not family:
            continue
        for candidate_key in _candidate_index_keys(row):
            index.setdefault((family, candidate_key), []).append(row)
    return {
        key: sorted(
            rows,
            key=lambda row: (
                int(row.get("full_video_coverage") is True),
                int(row.get("receiver_proof_ready") is True),
                int(row.get("group_count") or 0),
            ),
            reverse=True,
        )
        for key, rows in index.items()
    }


def _normalize_decoder_weight_waterfill_source(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if source.get("schema") != NERV_DECODER_WEIGHT_WATERFILL_SCHEMA:
        raise NervLongTrainingCampaignPlanError(
            "decoder_weight_waterfill_sources must have schema "
            f"{NERV_DECODER_WEIGHT_WATERFILL_SCHEMA}; got {source.get('schema')!r}"
        )
    path = (
        source.get("_decoder_weight_waterfill_plan_path")
        or source.get("decoder_weight_waterfill_plan_path")
        or source.get("path")
    )
    if not path:
        raise NervLongTrainingCampaignPlanError(
            "decoder_weight_waterfill_source missing "
            "_decoder_weight_waterfill_plan_path"
        )
    rows = source.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise NervLongTrainingCampaignPlanError(
            "decoder_weight_waterfill_source must contain non-empty rows"
        )
    out = dict(source)
    out["path"] = str(path)
    out["family"] = _family_key(str(source.get("family") or "hi_nerv"))
    out["group_count"] = int(source.get("group_count") or len(rows))
    out["full_video_coverage"] = bool(source.get("full_video_coverage"))
    out["receiver_proof_ready"] = str(
        source.get("receiver_proof_status") or ""
    ).lower() in {
        "runtime_consumption_proof_ready",
        "receiver_proof_valid",
        "runtime_consumption_proof_passed",
        "satisfied",
        "valid",
        "passed",
    }
    return out


def _candidate_index_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw_values = [
        row.get("candidate_id"),
        row.get("_candidate_id"),
        row.get("_candidate_key"),
        row.get("_modelsize_row_id"),
    ]
    keys: list[str] = []
    for raw in raw_values:
        text = str(raw or "").strip()
        if not text:
            continue
        keys.append(text)
        if ":" in text:
            keys.append(text.rsplit(":", 1)[-1])
    return tuple(_dedupe(keys))


def _decoder_weight_waterfill_for(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return {}
    rows = list((index or {}).get((_family_key(family), candidate_id)) or [])
    return dict(rows[0]) if rows else {}


def _decoder_weight_waterfill_row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "nerv_long_training_decoder_weight_waterfill_attachment.v1",
        "attached": True,
        "path": str(row.get("path")),
        "sha256": row.get("_decoder_weight_waterfill_plan_sha256"),
        "source_path": row.get("_decoder_weight_waterfill_source_path"),
        "family": row.get("family"),
        "candidate_id": row.get("candidate_id"),
        "candidate_keys": list(_candidate_index_keys(row)),
        "group_count": int(row.get("group_count") or 0),
        "full_video_coverage": bool(row.get("full_video_coverage")),
        "receiver_proof_ready": bool(row.get("receiver_proof_ready")),
        "blockers": list(row.get("blockers") or []),
        **FALSE_AUTHORITY,
    }


def _hinerv_feedback_launch_adjustment(
    *,
    feedback: Mapping[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    if not feedback:
        return {
            "schema": "hinerv_feedback_launch_adjustment.v1",
            "applied": False,
            "reason": "no_candidate_feedback",
            "learning_rate": float(learning_rate),
            **FALSE_AUTHORITY,
        }
    observed = _float_or_none(feedback.get("observed_learning_rate"))
    recommended = _float_or_none(feedback.get("recommended_learning_rate"))
    pose_instability = bool(feedback.get("pose_instability_detected"))
    applied = bool(
        pose_instability
        and recommended is not None
        and recommended > 0.0
        and recommended < float(learning_rate)
    )
    return {
        "schema": "hinerv_feedback_launch_adjustment.v1",
        "applied": applied,
        "reason": (
            "pose_instability_recommended_lower_learning_rate"
            if applied
            else (
                "pose_instability_feedback_without_lower_lr"
                if pose_instability
                else "feedback_does_not_request_launch_adjustment"
            )
        ),
        "source_feedback_kind": feedback.get("feedback_kind"),
        "source_feedback_scope": feedback.get("feedback_scope"),
        "pose_instability_detected": pose_instability,
        "observed_learning_rate": observed,
        "requested_learning_rate": float(learning_rate),
        "recommended_learning_rate": recommended,
        "learning_rate": float(recommended if applied else learning_rate),
        "launch_mutations": (
            list(feedback.get("recommended_launch_mutations") or []) if applied else []
        ),
        **FALSE_AUTHORITY,
    }


def _feedback_family(row: Mapping[str, Any]) -> str:
    return _family_key(str(row.get("family") or row.get("execute_family") or ""))


def _family_key(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text == "hinerv":
        return "hi_nerv"
    return text


def _load_verified_joint_recon_weight_artifacts(
    manifest_paths: Sequence[str | Path],
) -> dict[int, dict[str, Any]]:
    artifacts: dict[int, dict[str, Any]] = {}
    for path_value in manifest_paths:
        manifest_path = Path(path_value).expanduser().resolve(strict=False)
        if not manifest_path.is_file():
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest not found: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise NervLongTrainingCampaignPlanError(
                f"invalid joint recon weight manifest JSON: {manifest_path}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest must be an object: {manifest_path}"
            )
        if manifest.get("schema") != JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA:
            raise NervLongTrainingCampaignPlanError(
                f"unsupported joint recon weight manifest schema: {manifest_path}"
            )
        config = manifest.get("config")
        metadata = manifest.get("metadata")
        if not isinstance(config, Mapping) or not isinstance(metadata, Mapping):
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest missing config/metadata: {manifest_path}"
            )
        try:
            num_pairs = int(config.get("num_pairs"))
        except (TypeError, ValueError) as exc:
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest missing numeric num_pairs: {manifest_path}"
            ) from exc
        health = metadata.get("gradient_health")
        if not isinstance(health, Mapping) or health.get("status") != "pass_finite":
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest failed gradient health: {manifest_path}"
            )
        if metadata.get("training_consumption_recommended") is not True:
            raise NervLongTrainingCampaignPlanError(
                "joint recon weight manifest is not recommended for training "
                f"consumption: {manifest_path}"
            )
        blockers = [str(item) for item in metadata.get("blockers") or [] if item]
        if blockers:
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest has blockers: {manifest_path}"
            )
        raw_weight_path = manifest.get("weight_path")
        if raw_weight_path is None:
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest missing weight_path: {manifest_path}"
            )
        weight_path = Path(str(raw_weight_path)).expanduser()
        if not weight_path.is_absolute():
            weight_path = manifest_path.parent / weight_path
        weight_path = weight_path.resolve(strict=False)
        if not weight_path.is_file():
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight file not found: {weight_path}"
            )
        actual_sha = _sha256_file(weight_path)
        expected_sha = str(manifest.get("weight_sha256") or "")
        if expected_sha and actual_sha != expected_sha:
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight sha mismatch: {weight_path}"
            )
        artifact = {
            "schema": "nerv_long_training_joint_recon_weight_artifact.v1",
            "num_pairs": int(num_pairs),
            "manifest_path": manifest_path.as_posix(),
            "weight_path": weight_path.as_posix(),
            "weight_sha256": actual_sha,
            "gradient_health": dict(health),
            "training_consumption_recommended": True,
            **FALSE_AUTHORITY,
        }
        artifacts[int(num_pairs)] = artifact
    return dict(sorted(artifacts.items()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("family") or "unknown")
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _plan_level_blocker(blocker: str) -> bool:
    return str(blocker) in {
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only",
        "snerv_native_rate_pressure_in_loop_not_yet_training_authority",
        "snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes",
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _safe_path_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    token = "_".join(part for part in token.split("_") if part)
    return token.strip("._-")[:180] or "campaign"


def _campaign_output_basename(
    *,
    row_id: str,
    launch_feedback_adjustment: Mapping[str, Any],
) -> str:
    base = _safe_path_token(row_id)
    if not launch_feedback_adjustment.get("applied"):
        return base
    suffix_parts = ["feedback"]
    if launch_feedback_adjustment.get("pose_instability_detected"):
        suffix_parts.append("pose_instability")
    learning_rate = _float_or_none(launch_feedback_adjustment.get("learning_rate"))
    if learning_rate is not None:
        suffix_parts.append(f"lr{_safe_path_token(_float_token(learning_rate))}")
    mutations = [
        _safe_path_token(str(mutation))[:48]
        for mutation in (launch_feedback_adjustment.get("launch_mutations") or [])
        if str(mutation).strip()
    ]
    suffix_parts.extend(mutations[:2])
    suffix = "_".join(part for part in suffix_parts if part)
    return _safe_path_token(f"{base}__{suffix}")


def _float_token(value: float) -> str:
    return f"{float(value):.12g}"


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    import math

    return number if math.isfinite(number) else None


__all__ = [
    "DEFAULT_OPTIMIZER_KINDS",
    "SCHEMA",
    "SCORE_LOWERING_GATE_SCHEMA",
    "NervLongTrainingCampaignPlanError",
    "build_nerv_long_training_campaign_plan",
    "render_nerv_long_training_campaign_plan_markdown",
]
