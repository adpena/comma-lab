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
    recommend_segnet_distillation_weight_for_stagnation,
)
from tac.analysis.nerv_decoder_weight_waterfill import (
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
)
from tac.analysis.nerv_modelsize_budget import (
    NervModelSizeBudgetError,
    decoder_codec_nominal_bits,
    snerv_decoder_codec_nominal_bits,
    snerv_modelsize_candidate_id_from_controls,
)
from tac.analysis.nerv_source_parity_contract import build_nerv_source_parity_contract
from tac.optimization.recon_pixel_weight_surface import (
    JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA,
)
from tac.substrates._shared.mlx_score_aware.adapter import (
    SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)

SCHEMA = "nerv_long_training_campaign_plan.v1"
ROW_SCHEMA = "nerv_long_training_campaign_row.v1"
EXPERIMENT_QUEUE_SCHEMA = "experiment_queue.v1"
DEFAULT_EXPERIMENT_QUEUE_ID = "nerv_long_training_campaign_queue.v1"
SCORE_LOWERING_GATE_SCHEMA = "nerv_long_training_score_lowering_gate.v1"
DEFAULT_OUTPUT_ROOT = "/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns"
DEFAULT_EPOCHS = 29_650
DEFAULT_BATCH_PAIRS = 8
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_HINERV_TELEMETRY_FLUSH_INTERVAL_EPOCHS = 1
DEFAULT_CODER_QAT_QUANT_RESIDUAL_WEIGHT = 1.0e-3
DEFAULT_CODER_QAT_MAGNITUDE_WEIGHT = 1.0e-4
DEFAULT_CODER_QAT_DELTA_WEIGHT = 2.0e-4
DEFAULT_CODER_QAT_C1A_ENTROPY_WEIGHT = 1.0e-4
DEFAULT_CODER_QAT_C1A_SIGMA = 0.2
DEFAULT_CODER_QAT_C1A_SAMPLE_SIZE = 512
# Do not classify the first 9e-5 pose-spike as repeated low-LR failure: its
# telemetry explicitly requested a 2.7e-5 recovery run. The Huber path is real,
# but it is reserved for repeated instability at or below that recovered regime;
# otherwise the planner skips a measured LR recovery and confounds two causes.
HINERV_POSE_INSTABILITY_LOW_LR_FLOOR = 3.0e-5
HINERV_POSE_INSTABILITY_POLICY_LOGIC = (
    "pose instability above low_learning_rate_floor applies the measured lower "
    "learning-rate recommendation; repeated instability at or below the floor "
    "switches to pose_distillation_loss=huber while preserving raw MSE telemetry; "
    "segnet stagnation raises segnet_distillation_weight for the next run without "
    "granting archive, replay, or score authority"
)
HINERV_POSE_PROTECTED_LOSS = "huber"
HINERV_POSE_PROTECTED_HUBER_DELTA = 1.0
_AUTHORITY_TRUE_KEYS: tuple[str, ...] = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "production_hardened_claim",
    "ready_for_exact_eval_dispatch",
)
HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION = "switch_to_hinerv_official_feature_grid_convnext_controls"
DEFAULT_OPTIMIZER_KINDS = (
    "pact_muon_adamw",
    "adamw",
    "muon",
    "lion",
    "adamax",
    "rmsprop",
    "adafactor",
    "adam",
    "adagrad",
    "adadelta",
    "sgd",
)
FIRST_PASS_OPTIMIZER_KINDS = frozenset(("pact_muon_adamw", "adamw", "muon", "lion", "adamax"))
OPTIMIZER_CONTROL_SCHEMA = "nerv_optimizer_control_surface.v1"
HINERV_OPTIMIZER_POLICY_SCHEMA = "nerv_hinerv_optimizer_policy.v1"


class NervLongTrainingCampaignPlanError(ValueError):
    """Raised when a long-training campaign plan is malformed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    snerv_official_source_audit: Mapping[str, Any] | None = None,
    snerv_bounded_proof_only: bool = False,
    snerv_bounded_proof_epochs: int = 3,
    experiment_queue_id: str = DEFAULT_EXPERIMENT_QUEUE_ID,
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
    queue_id = str(experiment_queue_id or "").strip()
    if not queue_id:
        raise NervLongTrainingCampaignPlanError("experiment_queue_id must be non-empty")
    joint_recon_weight_artifacts = _load_verified_joint_recon_weight_artifacts(joint_recon_weight_manifest_paths)
    candidate_feedback_index = _candidate_feedback_index(candidate_feedback_sources)
    decoder_weight_waterfill_index = _decoder_weight_waterfill_index(decoder_weight_waterfill_sources)
    source_parity_contract = build_nerv_source_parity_contract(
        repo_root=_repo_root(),
        families=("hi_nerv", "snerv"),
        snerv_official_source_audit=snerv_official_source_audit,
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
                    source_parity_contract=source_parity_contract,
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
                source_parity_contract=source_parity_contract,
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
    experiment_queue = _experiment_queue(rows, queue_id=queue_id)
    decoder_weight_waterfill_unattached_sources = _decoder_weight_waterfill_unattached_sources(
        index=decoder_weight_waterfill_index,
        campaign_rows=rows,
    )
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
        "optimizer_control_policy": {
            "schema": OPTIMIZER_CONTROL_SCHEMA,
            "backend": "mixed_mlx_optimizers_and_pact_pr95_partition_adapter",
            "native_mlx_on_apple_silicon": True,
            "apple_specific_algorithm_claim": False,
            "applies_to": [
                "hi_nerv_shared_mlx_scoreaware_runner_rows",
                "future_snerv_learned_scoreaware_decoder_rows_after_binding",
            ],
            "does_not_apply_to": ["snerv_current_closed_form_native_export_and_scorer_loop_qat_rows"],
            "optimizer_kinds": list(optimizers),
            "native_mlx_optimizer_kinds": [kind for kind in optimizers if kind != "pact_muon_adamw"],
            "pact_partitioned_optimizer_kinds": [kind for kind in optimizers if kind == "pact_muon_adamw"],
            "first_pass_optimizer_kinds": sorted(FIRST_PASS_OPTIMIZER_KINDS),
            "default_optimizer_kind": "pact_muon_adamw",
            "default_optimizer_backend": "tac.local_acceleration.pr95_hnerv_mlx",
            "borrowed_from_pr95": ("Muon-vs-AdamW parameter partition and Newton-Schulz Muon update helper"),
            "original_pact_contest_adaptation": (
                "default score-aware NeRV optimizer control with false-authority "
                "MLX telemetry, byte/coder pressure, and per-run optimizer "
                "control metadata"
            ),
            "notes": (
                "pact_muon_adamw is Pact's PR95-derived partitioned default; "
                "Muon, Lion, AdamW, Adamax, and the other native controls are "
                "direct MLX optimizer baselines on Apple silicon, not "
                "Apple-invented optimizer algorithms."
            ),
        },
        "epochs": int(epochs),
        "batch_pairs": int(batch_pairs),
        "learning_rate": float(learning_rate),
        "output_root": Path(output_root).as_posix(),
        "joint_recon_weight_artifacts": list(joint_recon_weight_artifacts.values()),
        "joint_recon_weight_artifact_count": len(joint_recon_weight_artifacts),
        "candidate_feedback_source_count": len(candidate_feedback_sources),
        "snerv_bounded_proof_only": bool(snerv_bounded_proof_only),
        "snerv_bounded_proof_epochs": int(snerv_bounded_proof_epochs),
        "candidate_feedback_row_count": _unique_index_row_count(candidate_feedback_index),
        "decoder_weight_waterfill_source_count": len(decoder_weight_waterfill_sources),
        "decoder_weight_waterfill_row_count": _unique_index_row_count(decoder_weight_waterfill_index),
        "decoder_weight_waterfill_unattached_source_count": len(decoder_weight_waterfill_unattached_sources),
        "decoder_weight_waterfill_unattached_sources": (decoder_weight_waterfill_unattached_sources),
        "source_parity_contract": source_parity_contract,
        "snerv_official_source_audit_attached": isinstance(snerv_official_source_audit, Mapping),
        "source_parity_required_for_long_training_ready": bool(
            source_parity_contract.get("required_for_long_training_ready")
        ),
        "source_parity_blockers": list(source_parity_contract.get("blockers") or ()),
        "source_parity_nonblocking_gaps": list(source_parity_contract.get("nonblocking_gaps") or ()),
        "campaign_rows": rows,
        "campaign_row_count": len(rows),
        "experiment_queue": experiment_queue,
        "experiment_queue_schema": EXPERIMENT_QUEUE_SCHEMA,
        "experiment_queue_id": experiment_queue["queue_id"],
        "experiment_queue_experiment_count": len(experiment_queue["experiments"]),
        "launchable_local_row_count": sum(
            1
            for row in rows
            if row["experiment_queue_entry"].get("blocked") is not True
            and row["experiment_queue_entry"].get("status") in {"queued", "ready"}
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
                *[blocker for row in rows for blocker in row.get("blockers", []) if _plan_level_blocker(blocker)],
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
    candidate_feedback_index: (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None) = None,
    decoder_weight_waterfill_index: (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None) = None,
    source_parity_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "hinerv_candidate")
    quant_bits = min(8, decoder_codec_nominal_bits(str(candidate.get("decoder_codec"))))
    num_pairs = int(candidate.get("num_pairs") or 600)
    joint_recon_weight = dict((joint_recon_weight_artifacts or {}).get(num_pairs) or {})
    feedback = _candidate_feedback_for(
        candidate=candidate,
        family="hi_nerv",
        index=candidate_feedback_index,
    )
    feedback_evidence_blockers = _candidate_feedback_evidence_blockers(feedback)
    decoder_weight_waterfill = _decoder_weight_waterfill_for(
        candidate=candidate,
        family="hi_nerv",
        index=decoder_weight_waterfill_index,
    )
    launch_feedback_adjustment = _hinerv_feedback_launch_adjustment(
        feedback=feedback,
        learning_rate=float(learning_rate),
    )
    source_faithfulness_controls = _hinerv_source_faithfulness_controls(
        candidate=candidate,
        feedback=feedback,
    )
    source_parity = _source_parity_family_report(
        family="hi_nerv",
        source_parity_contract=source_parity_contract,
    )
    effective_learning_rate = float(launch_feedback_adjustment.get("learning_rate") or learning_rate)
    effective_segnet_distillation_weight = float(launch_feedback_adjustment.get("segnet_distillation_weight") or 1.0)
    output_dir_basename = _campaign_output_basename(
        row_id=f"hi_nerv::{candidate_id}::{optimizer_kind}",
        launch_feedback_adjustment=launch_feedback_adjustment,
    )
    curriculum = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=int(epochs),
        num_pairs=num_pairs,
        segnet_distillation_weight=effective_segnet_distillation_weight,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=int(quant_bits),
        recon_pixel_weight_attached=True,
        eval_roundtrip_ste_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        receiver_proof_attached=bool(feedback.get("receiver_proof_attached")),
        full_video_local_prefilter_attached=bool(feedback.get("full_video_local_prefilter_attached")),
        local_cpu_replay_gate_attached=bool(feedback.get("local_cpu_replay_gate_attached")),
        measured_archive_bytes=feedback.get("measured_archive_bytes"),
        measured_num_pairs=feedback.get("measured_num_pairs"),
    )
    row_id = f"hi_nerv::{candidate_id}::{optimizer_kind}"
    optimizer_policy = _hinerv_optimizer_policy_for_kind(optimizer_kind)
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
        "--planner-row-id",
        row_id,
        "--num-pairs",
        str(num_pairs),
        "--epochs",
        str(int(epochs)),
        "--batch-pairs",
        str(int(batch_pairs)),
        "--learning-rate",
        _float_token(effective_learning_rate),
        "--distillation-device",
        "gpu",
        "--modelsize-candidate-id",
        candidate_id,
        "--segnet-distillation-weight",
        _float_token(effective_segnet_distillation_weight),
        "--pose-distillation-weight",
        "1.0",
        "--coder-aware-qat",
        "--coder-qat-quant-bits",
        str(int(quant_bits)),
        *_coder_qat_command_args(quant_bits=int(quant_bits)),
        "--optimizer-kind",
        str(optimizer_kind),
        "--hi-nerv-optimizer-policy",
        optimizer_policy,
        "--mlx-prefilter-scorer-device",
        "gpu",
        "--mlx-prefilter-scorer-batch-pairs",
        str(int(batch_pairs)),
        "--mlx-prefilter-progress-every",
        "10",
        "--telemetry-flush-interval-epochs",
        str(DEFAULT_HINERV_TELEMETRY_FLUSH_INTERVAL_EPOCHS),
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
    decoder_weight_waterfill_runner_admitted = (
        _decoder_weight_waterfill_runner_admitted(decoder_weight_waterfill) if decoder_weight_waterfill else False
    )
    if decoder_weight_waterfill_runner_admitted:
        command.extend(
            [
                "--decoder-weight-waterfill-plan-json",
                str(decoder_weight_waterfill["path"]),
            ]
        )
    if launch_feedback_adjustment.get("pose_protected_pathway_applied") is True:
        command.extend(
            [
                "--pose-distillation-loss",
                str(launch_feedback_adjustment["pose_distillation_loss"]),
                "--pose-distillation-huber-delta",
                _float_token(float(launch_feedback_adjustment["pose_distillation_huber_delta"])),
            ]
        )
    candidate_authority_blockers = list(candidate.get("_candidate_authority_blockers") or [])
    blockers = [
        ("" if joint_recon_weight else "requires_verified_joint_p18_p19_recon_pixel_weight_artifact"),
        ("" if decoder_weight_waterfill else "hinerv_decoder_weight_waterfill_plan_missing"),
        (
            ""
            if not decoder_weight_waterfill or decoder_weight_waterfill_runner_admitted
            else "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted"
        ),
        "requires_full_video_mlx_prefilter_before_local_cpu_replay_unlock",
        "requires_local_cpu_replay_win_before_exact_cpu_auth",
        *candidate_authority_blockers,
        *list(source_parity["required_blockers"]),
        *list(curriculum.get("blockers") or []),
        *feedback_evidence_blockers,
    ]
    if feedback.get("pose_instability_detected") is True and not (
        launch_feedback_adjustment.get("applied") or launch_feedback_adjustment.get("pose_protected_pathway_applied")
    ):
        blockers.append("hinerv_pose_instability_feedback_unapplied")
    if feedback.get("seg_stagnation_detected") is True and not launch_feedback_adjustment.get("segnet_weight_applied"):
        blockers.append("hinerv_segnet_stagnation_feedback_unapplied")
    if launch_feedback_adjustment.get(
        "repeated_low_lr_pose_instability"
    ) is True and not launch_feedback_adjustment.get("pose_protected_pathway_applied"):
        blockers.append("hinerv_repeated_low_lr_pose_instability_requires_pose_protected_pathway")
    if candidate.get("nominal_under_ceiling") is not True:
        blockers.append("hinerv_candidate_nominal_over_byte_ceiling")
    blockers = _dedupe(blockers)
    prelaunch_gate = dict(curriculum.get("long_campaign_prelaunch_gate") or {})
    launch_ready = bool(
        prelaunch_gate.get("launch_allowed")
        and joint_recon_weight
        and decoder_weight_waterfill_runner_admitted
        and not candidate_authority_blockers
        and not source_parity["required_blockers"]
    )
    if candidate_authority_blockers:
        implementation_status = "selected_candidate_authority_flags_block_launch"
    elif source_parity["required_blockers"]:
        implementation_status = "source_parity_required_gap_blocks_launch"
    elif decoder_weight_waterfill and not decoder_weight_waterfill_runner_admitted:
        implementation_status = "decoder_weight_waterfill_plan_advisory_only_blocks_launch"
    elif not decoder_weight_waterfill:
        implementation_status = "decoder_weight_waterfill_plan_required_for_launch"
    elif launch_ready:
        implementation_status = "shared_mlx_scoreaware_runner_launchable"
    else:
        implementation_status = "shared_mlx_scoreaware_runner_waiting_for_verified_joint_recon_weight"
    return _row(
        row_id=row_id,
        family="hi_nerv",
        priority=_optimizer_priority(optimizer_kind),
        candidate=candidate,
        curriculum_plan=curriculum,
        command_argv=command,
        local_mlx_launch_command_ready=launch_ready,
        implementation_status=implementation_status,
        blockers=blockers,
        extra={
            "optimizer_kind": str(optimizer_kind),
            "optimizer_control": _optimizer_control(optimizer_kind),
            "optimizer_policy": _hinerv_optimizer_policy_control(
                optimizer_kind=optimizer_kind,
                optimizer_policy=optimizer_policy,
            ),
            "quant_bits": int(quant_bits),
            "coder_qat_control": _coder_qat_control(quant_bits=int(quant_bits)),
            "joint_recon_pixel_weight_artifact": joint_recon_weight or None,
            "decoder_weight_waterfill_plan": (
                _decoder_weight_waterfill_row_metadata(decoder_weight_waterfill)
                if decoder_weight_waterfill
                else {
                    "schema": ("nerv_long_training_decoder_weight_waterfill_attachment.v1"),
                    "attached": False,
                    "reason": "no_matching_decoder_weight_waterfill_plan",
                    **FALSE_AUTHORITY,
                }
            ),
            "feedback_launch_adjustment": launch_feedback_adjustment,
            "candidate_feedback": feedback or None,
            "candidate_feedback_evidence_blockers": feedback_evidence_blockers,
            "source_faithfulness_controls": source_faithfulness_controls,
            "source_parity": source_parity,
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
    candidate_feedback_index: (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None) = None,
    bounded_proof_only: bool = False,
    bounded_proof_epochs: int = 3,
    source_parity_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "snerv_candidate")
    source_control_blockers = _snerv_source_bound_control_blockers(candidate)
    source_parity = _source_parity_family_report(
        family="snerv",
        source_parity_contract=source_parity_contract,
    )
    feedback = _candidate_feedback_for(
        candidate=candidate,
        family="snerv",
        index=candidate_feedback_index,
    )
    feedback_evidence_blockers = _candidate_feedback_evidence_blockers(feedback)
    execution_epochs = min(int(epochs), max(1, int(bounded_proof_epochs))) if bounded_proof_only else int(epochs)
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
        native_mlx_receiver_proof_passed=bool(feedback.get("native_mlx_receiver_proof_passed")),
        native_mlx_full600_campaign_ready=bool(feedback.get("native_mlx_full600_campaign_ready")),
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
        native_mlx_scorer_loop_qat_best_materialized=bool(feedback.get("native_mlx_scorer_loop_qat_best_materialized")),
        native_mlx_artifact_evidence=_snerv_native_artifact_evidence_from_feedback(feedback),
        receiver_proof_attached=bool(feedback.get("receiver_proof_attached")),
        full_video_local_prefilter_attached=bool(feedback.get("full_video_local_prefilter_attached")),
        local_cpu_replay_gate_attached=bool(feedback.get("local_cpu_replay_gate_attached")),
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
        "--planner-row-id",
        row_id,
        "--num-pairs",
        str(int(candidate.get("num_pairs") or 600)),
        "--epochs",
        str(int(execution_epochs)),
        "--modelsize-candidate-id",
        candidate_id,
        "--distillation-device",
        "gpu",
        "--coder-aware-qat",
        "--coder-qat-quant-bits",
        str(int(quant_bits)),
        *_coder_qat_command_args(quant_bits=int(quant_bits)),
        "--snerv-scorer-loop-qat",
        "--snerv-scorer-loop-search-mode",
        "learned_random_subspace",
        "--snerv-model-size-adapter",
        str(candidate.get("snerv_model_size_adapter") or ""),
        "--snerv-fc-dim",
        str(int(candidate.get("fc_dim") or 0)),
        "--snerv-emb-size",
        str(int(candidate.get("emb_size") or 0)),
        "--snerv-patch-radius",
        str(int(candidate.get("patch_radius") or 0)),
        "--snerv-mfu-scales",
        _int_csv(candidate.get("mfu_scales") or (1, 2, 4)),
        "--snerv-hfr-gain",
        _float_token(float(candidate.get("hfr_gain") or 0.0)),
        "--snerv-temporal-context",
        str(int(candidate.get("temporal_context") or 0)),
        "--snerv-temporal-mode",
        str(candidate.get("temporal_mode") or "delta"),
        "--output-dir",
        (output_root / _safe_path_token(row_id)).as_posix(),
    ]
    if str(candidate.get("snerv_model_size_adapter") or "") == (SNERV_SPECTRA_PRESERVING_ADAPTER):
        insert_at = command.index("--snerv-model-size-adapter")
        command.insert(insert_at, "--snerv-spectra-preserving-adapter")
    rate_plausible_for_long_training = _snerv_rate_plausible_for_long_training(candidate)
    blockers = _dedupe(
        [
            ("snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only" if bounded_proof_only else ""),
            "snerv_native_rate_pressure_in_loop_not_yet_training_authority",
            (
                "snerv_nominal_payload_far_over_ceiling_refuse_long_training"
                if not bounded_proof_only and not rate_plausible_for_long_training
                else ""
            ),
            "snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes"
            if candidate.get("nominal_under_ceiling") is not True
            else "",
            "snerv_optimizer_control_requires_learned_scoreaware_training_loop",
            *list(candidate.get("_candidate_authority_blockers") or []),
            *source_control_blockers,
            *list(source_parity["required_blockers"]),
            *list(curriculum.get("blockers") or []),
            *feedback_evidence_blockers,
        ]
    )
    source_controls_ready = (
        not source_control_blockers
        and not candidate.get("_candidate_authority_blockers")
        and not source_parity["required_blockers"]
    )
    launch_ready = bool(
        source_controls_ready and (True if bounded_proof_only else bool(rate_plausible_for_long_training))
    )
    return _row(
        row_id=row_id,
        family="snerv",
        priority=12,
        candidate=candidate,
        curriculum_plan=curriculum,
        command_argv=command,
        local_mlx_launch_command_ready=launch_ready,
        implementation_status=(
            "source_bound_capacity_controls_incomplete"
            if not source_controls_ready
            else (
                "bounded_native_export_scorer_loop_stage_ready"
                if bounded_proof_only
                else (
                    "native_rate_aware_long_training_queue_ready"
                    if rate_plausible_for_long_training
                    else "native_rate_aware_long_training_rate_blocked"
                )
            )
        ),
        blockers=blockers,
        extra={
            "optimizer_kind": None,
            "optimizer_control": _snerv_optimizer_control_blocker(),
            "quant_bits": int(quant_bits),
            "coder_qat_control": _coder_qat_control(quant_bits=int(quant_bits)),
            "planned_long_training_epochs": int(epochs),
            "execution_epochs": int(execution_epochs),
            "current_command_is_bounded_proof_not_long_training": bool(bounded_proof_only),
            "snerv_bounded_proof_epochs": int(bounded_proof_epochs),
            "source_bound_capacity_controls": _snerv_source_bound_controls(candidate),
            "source_bound_capacity_control_blockers": source_control_blockers,
            "source_parity": source_parity,
            "candidate_feedback": feedback or None,
            "candidate_feedback_evidence_blockers": feedback_evidence_blockers,
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
        "candidate": dict(candidate),
        "hard_byte_ceiling": int(candidate.get("hard_byte_ceiling") or 0),
        "candidate_nominal_total_payload_bytes": int(candidate.get("nominal_total_payload_bytes") or 0),
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
            row_metadata=_experiment_row_metadata(extra),
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


def _experiment_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    queue_id: str,
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_QUEUE_SCHEMA,
        "queue_id": queue_id,
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
    row_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = _row_output_dir(command_argv)
    output_json = (output_dir / "compact_renderer_mlx_spine_runner_report.json").as_posix()
    telemetry_artifacts = _row_observable_artifacts(
        family=str(family),
        output_dir=output_dir,
    )
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
        bounded_blocker = "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        if bounded_blocker in blockers:
            postconditions.append(
                {
                    "type": "json_array_contains",
                    "path": output_json,
                    "key": "blockers",
                    "contains": bounded_blocker,
                }
            )
    launch_blockers = _experiment_launch_blockers(blockers)
    runnable = bool(local_mlx_launch_command_ready) and not launch_blockers
    metadata = dict(row_metadata or {})
    source_parity = metadata.get("source_parity")
    source_controls = metadata.get("source_bound_capacity_controls")
    source_control_blockers = metadata.get("source_bound_capacity_control_blockers")
    current_command_is_bounded_proof = bool(
        metadata.get("current_command_is_bounded_proof_not_long_training")
    )
    return {
        "id": _safe_path_token(row_id),
        "family": str(family),
        "priority": int(priority),
        "status": "queued" if runnable else "disabled",
        "blocked": not runnable,
        "launch_authority_contract": {
            "schema": "nerv_long_training_queue_launch_authority_contract.v1",
            "queue_status_is_local_mlx_plan": bool(local_mlx_launch_command_ready),
            "queue_status_is_runnable_plan": runnable,
            "queue_launch_blockers": list(launch_blockers),
            "queue_status_is_receiver_proof": False,
            "queue_status_is_cpu_replay_proof": False,
            "queue_status_is_exact_eval_authority": False,
            "source_parity_contract_consumed": isinstance(source_parity, Mapping),
            "source_bound_capacity_controls_consumed": isinstance(
                source_controls,
                Mapping,
            ),
            "source_bound_capacity_control_blockers": list(
                source_control_blockers or ()
            ),
            "current_command_is_bounded_proof_not_long_training": (
                current_command_is_bounded_proof
            ),
            "receiver_proof_required": bool(score_lowering_gate.get("receiver_proof_required")),
            "cpu_replay_ready": bool(score_lowering_gate["cpu_replay_ready"]),
            "exact_gate_ready": bool(score_lowering_gate["exact_gate_ready"]),
            "source_parity": source_parity if isinstance(source_parity, Mapping) else None,
            "source_bound_capacity_controls": (
                source_controls if isinstance(source_controls, Mapping) else None
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "blockers": _dedupe([str(blocker) for blocker in blockers if blocker]),
        "metadata": {
            "schema": "nerv_long_training_campaign_experiment_metadata.v1",
            **metadata,
            **FALSE_AUTHORITY,
        },
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
                "telemetry": {
                    "artifact_paths": telemetry_artifacts,
                    "include_postcondition_paths": True,
                },
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


def _experiment_launch_blockers(blockers: Sequence[str]) -> list[str]:
    """Return blockers that should prevent a row from being runnable."""

    exact_names = {
        "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted",
        "hinerv_decoder_weight_waterfill_plan_missing",
        "requires_verified_joint_p18_p19_recon_pixel_weight_artifact",
        "snerv_native_rate_pressure_in_loop_not_yet_training_authority",
        "snerv_optimizer_control_requires_learned_scoreaware_training_loop",
        "snerv_candidate_id_source_bound_controls_mismatch",
        "snerv_candidate_id_source_bound_controls_unparseable",
        "snerv_nominal_payload_far_over_ceiling_refuse_long_training",
    }
    prefixes = ("snerv_source_bound_control_missing:", "source_parity:")
    return _dedupe(
        [
            str(blocker)
            for blocker in blockers
            if str(blocker)
            and (str(blocker) in exact_names or any(str(blocker).startswith(prefix) for prefix in prefixes))
        ]
    )


def _candidate_feedback_evidence_blockers(
    feedback: Mapping[str, Any],
) -> list[str]:
    """Carry candidate-feedback evidence debt without making it launch-blocking."""

    if not feedback:
        return []
    blockers = [
        str(blocker)
        for blocker in feedback.get("sample_generalization_blockers") or []
        if blocker
    ]
    gate = feedback.get("sample_generalization_gate")
    if isinstance(gate, Mapping):
        blockers.extend(str(blocker) for blocker in gate.get("blockers") or [])
    return _dedupe(blockers)


def _source_parity_family_report(
    *,
    family: str,
    source_parity_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return row-local source-parity debt without granting score authority."""

    if not isinstance(source_parity_contract, Mapping):
        return {
            "schema": "nerv_row_source_parity_binding.v1",
            "contract_schema": None,
            "family": str(family),
            "long_training_ready": False,
            "required_blockers": ["source_parity:source_parity_contract_missing"],
            "nonblocking_gaps": [],
            "feature_status_rows": [],
            "control_status_rows": [],
            **FALSE_AUTHORITY,
        }
    feature_rows = [
        row
        for row in source_parity_contract.get("feature_rows") or ()
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    control_rows = [
        row
        for row in source_parity_contract.get("control_rows") or ()
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    source_audits = [
        row
        for row in source_parity_contract.get("source_audits") or ()
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    required_blockers = _dedupe(
        [
            f"source_parity:{blocker}"
            for row in (*feature_rows, *control_rows)
            if row.get("required_for_long_training") is True
            for blocker in row.get("blockers") or ()
            if str(blocker)
        ]
    )
    nonblocking_gaps = _dedupe(
        [
            f"source_parity:{blocker}"
            for row in (*feature_rows, *control_rows)
            if row.get("required_for_long_training") is not True
            for blocker in row.get("blockers") or ()
            if str(blocker)
        ]
    )
    family_summary = {}
    for row in source_parity_contract.get("family_rows") or ():
        if isinstance(row, Mapping) and row.get("family") == family:
            family_summary = dict(row)
            break
    return {
        "schema": "nerv_row_source_parity_binding.v1",
        "contract_schema": source_parity_contract.get("schema"),
        "contract_authority": source_parity_contract.get("authority"),
        "family": str(family),
        "long_training_ready": bool(family_summary.get("long_training_ready", not required_blockers)),
        "required_blockers": list(required_blockers),
        "nonblocking_gaps": list(nonblocking_gaps),
        "source_audit_rows": [dict(row) for row in source_audits],
        "feature_status_rows": [
            {
                "feature_id": row.get("feature_id"),
                "status": row.get("status"),
                "required_for_long_training": bool(row.get("required_for_long_training")),
                "source_audit_rows": [
                    dict(audit) for audit in row.get("source_audit_rows") or () if isinstance(audit, Mapping)
                ],
                "blockers": list(row.get("blockers") or ()),
            }
            for row in feature_rows
        ],
        "control_status_rows": [
            {
                "control_id": row.get("control_id"),
                "status": row.get("status"),
                "required_for_long_training": bool(row.get("required_for_long_training")),
                "blockers": list(row.get("blockers") or ()),
            }
            for row in control_rows
        ],
        **FALSE_AUTHORITY,
    }


def _row_output_report_path(command_argv: Sequence[str]) -> str:
    return (_row_output_dir(command_argv) / "compact_renderer_mlx_spine_runner_report.json").as_posix()


def _row_output_dir(command_argv: Sequence[str]) -> Path:
    argv = [str(value) for value in command_argv]
    try:
        out_dir = argv[argv.index("--output-dir") + 1]
    except (ValueError, IndexError):
        out_dir = DEFAULT_OUTPUT_ROOT
    return Path(out_dir)


def _row_observable_artifacts(*, family: str, output_dir: Path) -> list[str]:
    artifacts = [(output_dir / "compact_renderer_mlx_spine_runner_startup.json").as_posix()]
    if str(family) == "hi_nerv":
        artifacts.extend(
            [
                (output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl").as_posix(),
                (output_dir / "hi_nerv_mlx_training" / "local_mlx_prefilter_progress.jsonl").as_posix(),
            ]
        )
    return artifacts


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
        dict(row) for row in binding.get("rows", []) if isinstance(row, Mapping) and row.get("satisfied") is not True
    ]
    missing_requirement_ids = [str(row.get("requirement_id")) for row in missing_rows if row.get("requirement_id")]
    post_run_requirements = [str(item) for item in gate.get("post_run_requirements_excluded", []) if item]
    post_run_missing = [requirement for requirement in missing_requirement_ids if requirement in post_run_requirements]
    promotion_blockers = _dedupe(
        [
            *(str(blocker) for blocker in blockers if blocker),
            *(f"{family}_{requirement}_missing" for requirement in post_run_missing),
        ]
    )
    launch_blockers = _experiment_launch_blockers(blockers)
    prelaunch_blockers = _dedupe(
        [
            *(str(blocker) for blocker in gate.get("blockers", []) if blocker),
            *launch_blockers,
        ]
    )
    local_proof_launch_allowed = bool(local_mlx_launch_command_ready) and not launch_blockers
    cpu_replay_ready = (
        bool(local_proof_launch_allowed)
        and "receiver_proof" not in post_run_missing
        and "full_video_local_prefilter" not in post_run_missing
        and "local_cpu_replay_gate" not in post_run_missing
    )
    exact_gate_ready = cpu_replay_ready and "exact_auth_gate_plan" not in post_run_missing and not promotion_blockers
    return {
        "schema": SCORE_LOWERING_GATE_SCHEMA,
        "family": str(family),
        "command_materialized": bool(local_mlx_launch_command_ready),
        "local_mlx_executable": bool(local_proof_launch_allowed),
        "prelaunch_allowed": bool(local_proof_launch_allowed),
        "promotion_prelaunch_allowed": bool(gate.get("launch_allowed")),
        "prelaunch_blockers": prelaunch_blockers,
        "launch_blockers": launch_blockers,
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
        raise NervLongTrainingCampaignPlanError(f"{name} schema must be {schema}; got {payload.get('schema')}")


def _selected_candidates(
    payload: Mapping[str, Any],
    *,
    family: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("selected_candidates", []):
        if not isinstance(row, Mapping) or row.get("family") != family:
            continue
        authority_blockers = _candidate_authority_blockers(row)
        candidate = _scrub_candidate_authority_flags(row)
        if authority_blockers:
            candidate["_candidate_authority_blockers"] = authority_blockers
        rows.append(candidate)
    if family == "snerv":
        rows.sort(key=_snerv_long_training_candidate_sort_key)
    elif family == "hi_nerv":
        rows.sort(key=_hinerv_long_training_candidate_sort_key)
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        deduped.setdefault(candidate_id, row)
    rows = list(deduped.values())
    return rows[: max(1, int(limit))]


def _candidate_authority_blockers(candidate: Mapping[str, Any]) -> list[str]:
    return _dedupe(
        [
            f"selected_candidate_authority_flag_true:{path}"
            for path in _iter_truthy_authority_paths(candidate)
        ]
    )


def _scrub_candidate_authority_flags(value: Any) -> Any:
    if isinstance(value, Mapping):
        scrubbed: dict[str, Any] = {}
        for key, raw in value.items():
            text_key = str(key)
            if text_key in _AUTHORITY_TRUE_KEYS:
                scrubbed[text_key] = False
            else:
                scrubbed[text_key] = _scrub_candidate_authority_flags(raw)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_candidate_authority_flags(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_candidate_authority_flags(item) for item in value]
    return value


def _iter_truthy_authority_paths(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, raw in value.items():
            text_key = str(key)
            path = f"{prefix}.{text_key}" if prefix else text_key
            if text_key in _AUTHORITY_TRUE_KEYS and _truthy_authority_value(raw):
                paths.append(path)
            paths.extend(_iter_truthy_authority_paths(raw, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, raw in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_iter_truthy_authority_paths(raw, prefix=path))
    return paths


def _truthy_authority_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _snerv_rate_plausible_for_long_training(candidate: Mapping[str, Any]) -> bool:
    if candidate.get("nominal_under_ceiling") is True:
        return True
    ceiling = int(candidate.get("hard_byte_ceiling") or 0)
    byte_headroom = _candidate_byte_headroom(candidate)
    if ceiling <= 0:
        return False
    # Permit near-miss long-training rows because real SNAR1 bytes can move
    # after QAT/coding; refuse rows whose nominal payload is orders over budget.
    return abs(byte_headroom) <= max(int(ceiling), 65_536)


def _snerv_long_training_candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    under = row.get("nominal_under_ceiling") is True
    ceiling = int(row.get("hard_byte_ceiling") or 0)
    headroom = _candidate_byte_headroom(row)
    total = int(row.get("nominal_total_payload_bytes") or 0)
    return (
        0 if under else 1,
        headroom if under else abs(headroom),
        ceiling,
        total,
        str(row.get("candidate_id") or ""),
    )


def _hinerv_long_training_candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    under = row.get("nominal_under_ceiling") is True
    ceiling = int(row.get("hard_byte_ceiling") or 0)
    headroom = _candidate_byte_headroom(row)
    total = int(row.get("nominal_total_payload_bytes") or 0)
    return (
        0 if under else 1,
        -_hinerv_official_control_score(row),
        headroom if under else abs(headroom),
        ceiling,
        total,
        str(row.get("candidate_id") or ""),
    )


def _candidate_byte_headroom(row: Mapping[str, Any]) -> int:
    if row.get("byte_headroom") is not None:
        return int(row.get("byte_headroom") or 0)
    ceiling = int(row.get("hard_byte_ceiling") or 0)
    total = int(row.get("nominal_total_payload_bytes") or 0)
    return int(ceiling - total)


def _snerv_source_bound_control_blockers(candidate: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    candidate_id = str(candidate.get("candidate_id") or "")
    for key in (
        "num_pairs",
        "wavelet",
        "levels",
        "bits_per_coeff",
        "step_map_bits_per_coeff",
        "decoder_payload_codec",
        "snerv_model_size_adapter",
        "fc_dim",
        "emb_size",
        "patch_radius",
        "mfu_scales",
        "hfr_gain",
        "temporal_context",
        "temporal_mode",
        "hard_byte_ceiling",
    ):
        if key not in candidate:
            blockers.append(f"snerv_source_bound_control_missing:{key}")
    if not blockers:
        try:
            expected_candidate_id = _snerv_expected_candidate_id_from_controls(candidate)
        except (KeyError, NervModelSizeBudgetError, TypeError, ValueError):
            blockers.append("snerv_candidate_id_source_bound_controls_unparseable")
        else:
            if candidate_id != expected_candidate_id:
                blockers.append("snerv_candidate_id_source_bound_controls_mismatch")
    return blockers


def _snerv_source_bound_controls(candidate: Mapping[str, Any]) -> dict[str, Any]:
    expected_candidate_id = None
    candidate_id_matches_source_controls = False
    try:
        expected_candidate_id = _snerv_expected_candidate_id_from_controls(candidate)
        candidate_id_matches_source_controls = str(candidate.get("candidate_id") or "") == expected_candidate_id
    except (KeyError, NervModelSizeBudgetError, TypeError, ValueError):
        pass
    return {
        "schema": "snerv_source_bound_capacity_controls.v1",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "expected_candidate_id": expected_candidate_id,
        "candidate_id_matches_source_controls": candidate_id_matches_source_controls,
        "wavelet": candidate.get("wavelet"),
        "levels": candidate.get("levels"),
        "bits_per_coeff": candidate.get("bits_per_coeff"),
        "step_map_bits_per_coeff": candidate.get("step_map_bits_per_coeff"),
        "decoder_payload_codec": candidate.get("decoder_payload_codec"),
        "snerv_model_size_adapter": candidate.get("snerv_model_size_adapter"),
        "fc_dim": candidate.get("fc_dim"),
        "emb_size": candidate.get("emb_size"),
        "patch_radius": candidate.get("patch_radius"),
        "mfu_scales": list(candidate.get("mfu_scales") or ()),
        "hfr_gain": candidate.get("hfr_gain"),
        "temporal_context": candidate.get("temporal_context"),
        "temporal_mode": candidate.get("temporal_mode"),
        "decoder_feature_count": candidate.get("decoder_feature_count"),
        **FALSE_AUTHORITY,
    }


def _snerv_expected_candidate_id_from_controls(candidate: Mapping[str, Any]) -> str:
    return snerv_modelsize_candidate_id_from_controls(
        num_pairs=int(candidate["num_pairs"]),
        wavelet=str(candidate["wavelet"]),
        levels=int(candidate["levels"]),
        bits_per_coeff=float(candidate["bits_per_coeff"]),
        step_map_bits_per_coeff=float(candidate["step_map_bits_per_coeff"]),
        fc_dim=int(candidate["fc_dim"]),
        emb_size=int(candidate["emb_size"]),
        patch_radius=int(candidate["patch_radius"]),
        mfu_scales=tuple(int(value) for value in candidate["mfu_scales"]),
        hfr_gain=float(candidate["hfr_gain"]),
        temporal_context=int(candidate["temporal_context"]),
        temporal_mode=str(candidate["temporal_mode"]),
        snerv_model_size_adapter=str(candidate["snerv_model_size_adapter"]),
        decoder_payload_codec=str(candidate["decoder_payload_codec"]),
        hard_byte_ceiling=int(candidate["hard_byte_ceiling"]),
        official_modelsize_mparams=(
            float(candidate["modelsize_mparams"])
            if candidate.get("capacity_source") == "official_snerv_modelsize"
            and candidate.get("modelsize_mparams") is not None
            else None
        ),
    )


def _snerv_optimizer_control_blocker() -> dict[str, Any]:
    return {
        "schema": OPTIMIZER_CONTROL_SCHEMA,
        "optimizer_kind": None,
        "backend": ("mlx_target_hydration_numpy_closed_form_decoder_fit_plus_scorer_loop_qat"),
        "native_mlx_on_apple_silicon": True,
        "apple_specific_algorithm_claim": False,
        "first_pass_priority": False,
        "borrowed_from_pr95": False,
        "original_pact_contest_adaptation": False,
        "pact_muon_adamw_default_inherited": False,
        "not_applicable_reason": (
            "Current SNeRV rows materialize source-bound closed-form/native "
            "packets plus optional scorer-loop QAT; they do not yet expose a "
            "learned optimizer-controlled decoder-weight training loop."
        ),
        "blocked_until": (
            "snerv_learned_nonlinear_or_shared_mlx_scoreaware_decoder_training_loop_bound_to_receiver_grammar"
        ),
        "required_next_implementation": [
            "source_faithful_snerv_mfu_hfr_tub_forward_parity",
            "receiver_visible_mixed_precision_or_decoder_delta_grammar",
            "pose_guarded_scorer_loop_decoder_weight_qat",
            "full600_byte_closed_receiver_archive_replay",
        ],
        **FALSE_AUTHORITY,
    }


def _int_csv(values: Any) -> str:
    if isinstance(values, str):
        return values
    if not isinstance(values, Sequence):
        return str(int(values))
    return ",".join(str(int(value)) for value in values)


def _optimizer_tuple(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    supported = set(SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS)
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text not in supported:
            raise NervLongTrainingCampaignPlanError(f"unsupported optimizer kind: {value!r}")
        if text not in out:
            out.append(text)
    if not out:
        raise NervLongTrainingCampaignPlanError("at least one optimizer is required")
    return tuple(out)


def _optimizer_priority(optimizer_kind: str) -> int:
    kind = str(optimizer_kind).strip().lower()
    if kind == "pact_muon_adamw":
        return 9
    return 10 if kind in FIRST_PASS_OPTIMIZER_KINDS else 11


def _optimizer_control(optimizer_kind: str) -> dict[str, Any]:
    kind = str(optimizer_kind).strip().lower()
    if kind not in SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS:
        raise NervLongTrainingCampaignPlanError(f"unsupported optimizer kind: {optimizer_kind!r}")
    is_pact_default = kind == "pact_muon_adamw"
    return {
        "schema": OPTIMIZER_CONTROL_SCHEMA,
        "optimizer_kind": kind,
        "backend": ("tac.local_acceleration.pr95_hnerv_mlx" if is_pact_default else "mlx.optimizers"),
        "native_mlx_on_apple_silicon": True,
        "native_mlx_optimizer_object": not is_pact_default,
        "pact_partitioned_muon_adamw": is_pact_default,
        "apple_specific_algorithm_claim": False,
        "first_pass_priority": kind in FIRST_PASS_OPTIMIZER_KINDS,
        "borrowed_from_pr95": is_pact_default,
        "original_pact_contest_adaptation": is_pact_default,
        "provenance_note": (
            "Pact default borrows PR95's Muon-vs-AdamW partition rule and "
            "the existing MLX Newton-Schulz step, then applies it to the "
            "score-aware NeRV train loop with false-authority telemetry."
            if is_pact_default
            else "Direct native MLX optimizer control row."
        ),
        "default_hinerv_optimizer_policy": _hinerv_optimizer_policy_for_kind(kind),
        "pr95_curriculum_optimizer_swallow_guard": (kind != "adamw"),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _hinerv_optimizer_policy_for_kind(optimizer_kind: str) -> str:
    """Return the runner policy that makes this row's optimizer semantics real."""

    kind = str(optimizer_kind).strip().lower()
    return "pr95_curriculum" if kind == "adamw" else "native_optimizer"


def _hinerv_optimizer_policy_control(
    *,
    optimizer_kind: str,
    optimizer_policy: str,
) -> dict[str, Any]:
    kind = str(optimizer_kind).strip().lower()
    policy = str(optimizer_policy).strip().lower()
    return {
        "schema": HINERV_OPTIMIZER_POLICY_SCHEMA,
        "optimizer_kind": kind,
        "requested_policy": policy,
        "pr95_faithful_curriculum_expected": policy == "pr95_curriculum",
        "native_mlx_optimizer_expected": policy == "native_optimizer",
        "effective_optimizer_label": ("pr95_8stage_muon_adamw" if policy == "pr95_curriculum" else kind),
        "why": (
            "adamw owns the PR95-faithful 8-stage Muon+AdamW control row; "
            "non-adamw rows must run as native MLX optimizers so optimizer "
            "diversity is measured rather than swallowed by the curriculum"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _candidate_feedback_index(
    sources: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in sources:
        row = _normalize_candidate_feedback_source(source)
        family = _feedback_family(row)
        if not family:
            continue
        for candidate_key in _candidate_index_keys(row):
            index.setdefault((family, candidate_key), []).append(row)
    return {
        key: sorted(
            rows,
            key=_candidate_feedback_sort_key,
            reverse=True,
        )
        for key, rows in index.items()
    }


def _candidate_feedback_sort_key(
    row: Mapping[str, Any],
) -> tuple[bool, int, bool, int, bool, bool, bool]:
    telemetry = row.get("training_telemetry")
    last_epoch = int(telemetry.get("last_epoch") or 0) if isinstance(telemetry, Mapping) else 0
    return (
        bool(row.get("scope_matches_candidate")),
        int(row.get("measured_num_pairs") or 0),
        row.get("training_stopped") is not True,
        last_epoch,
        bool(row.get("receiver_proof_attached")),
        bool(row.get("full_video_local_prefilter_attached")),
        bool(row.get("local_cpu_replay_gate_attached")),
    )


def _normalize_candidate_feedback_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("schema") == NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA:
        row = _sanitize_direct_candidate_feedback_row(source)
    elif source.get("schema") == "hinerv_training_telemetry_feedback.v1":
        row = _normalize_hinerv_training_telemetry_feedback(source)
    elif source.get("schema") == "compact_renderer_mlx_spine_runner.v1":
        row = build_nerv_candidate_feedback_row(
            runner_report=source,
            source_report_path=source.get("_candidate_feedback_source_path"),
        )
    else:
        row = dict(source)
    return _augment_feedback_row(row, source)


def _sanitize_direct_candidate_feedback_row(source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    if row.get("feedback_kind") == "training_telemetry":
        return row
    blockers = [str(blocker) for blocker in row.get("direct_feedback_blockers") or []]
    _guard_direct_feedback_bool(
        row,
        bool_key="receiver_proof_attached",
        path_keys=("receiver_proof_path", "receiver_proof_report_path"),
        sha_keys=("receiver_proof_sha256",),
        blocker="direct_feedback_receiver_proof_file_missing",
        blockers=blockers,
    )
    _guard_direct_feedback_bool(
        row,
        bool_key="full_video_local_prefilter_attached",
        path_keys=("full_video_local_prefilter_path", "mlx_prefilter_path"),
        sha_keys=("full_video_local_prefilter_sha256", "mlx_prefilter_sha256"),
        blocker="direct_feedback_full_video_prefilter_file_missing",
        blockers=blockers,
    )
    _guard_direct_feedback_bool(
        row,
        bool_key="local_cpu_replay_gate_attached",
        path_keys=("local_cpu_replay_summary_path", "local_cpu_replay_gate_path"),
        sha_keys=("local_cpu_replay_summary_sha256", "local_cpu_replay_gate_sha256"),
        blocker="direct_feedback_local_cpu_replay_file_missing",
        blockers=blockers,
    )
    _guard_direct_feedback_bool(
        row,
        bool_key="native_mlx_receiver_proof_passed",
        path_keys=(
            "snerv_mlx_native_export_receiver_proof_path",
            "receiver_proof_path",
        ),
        sha_keys=("snerv_mlx_native_export_receiver_proof_sha256",),
        blocker="direct_feedback_native_receiver_proof_file_missing",
        blockers=blockers,
    )
    evidence = row.get("snerv_mlx_native_file_backed_export_evidence")
    file_backed_ready = bool(
        isinstance(evidence, Mapping) and evidence.get("required_pair_file_backed_export_proof_passed") is True
    )
    if row.get("native_mlx_full600_campaign_ready") is True and not file_backed_ready:
        row["native_mlx_full600_campaign_ready"] = False
        blockers.append("direct_feedback_native_full600_file_backed_evidence_missing")
    row["direct_feedback_blockers"] = _dedupe(blockers)
    return row


def _guard_direct_feedback_bool(
    row: dict[str, Any],
    *,
    bool_key: str,
    path_keys: Sequence[str],
    sha_keys: Sequence[str],
    blocker: str,
    blockers: list[str],
) -> None:
    if row.get(bool_key) is not True:
        return
    if _direct_feedback_file_evidence_valid(row, path_keys=path_keys, sha_keys=sha_keys):
        return
    row[bool_key] = False
    blockers.append(blocker)


def _direct_feedback_file_evidence_valid(
    row: Mapping[str, Any],
    *,
    path_keys: Sequence[str],
    sha_keys: Sequence[str],
) -> bool:
    for path_key in path_keys:
        raw = row.get(path_key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser().resolve(strict=False)
        if not path.is_file():
            continue
        expected_sha = next(
            (str(row.get(key)) for key in sha_keys if row.get(key)),
            "",
        )
        return not expected_sha or _sha256_file(path) == expected_sha
    return False


def _normalize_hinerv_training_telemetry_feedback(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert compact foreground/harvest telemetry into planner feedback.

    Foreground proof rows are not runner reports and intentionally carry no
    archive/replay authority. They are still high-value launch-pressure signal
    when they come from a full600 candidate and show a real optimization
    dynamic, such as PoseNet recovering while SegNet remains binding. Convert
    only that steering signal into the same false-authority feedback shape the
    campaign planner already consumes.
    """

    candidate_id = str(source.get("candidate_id") or "").strip()
    last_epoch = int(source.get("last_epoch") or 0)
    seg_still_binding = bool(source.get("segnet_still_binding") is True)
    pose_recovered = bool(source.get("pose_recovered_from_initial_spike") is True)
    pose_instability = bool(source.get("pose_instability_detected") is True)
    recommended_lr = _float_or_none(source.get("recommended_learning_rate"))
    observed_seg_weight = _float_or_none(source.get("observed_segnet_distillation_weight"))
    recommended_seg_weight = _float_or_none(source.get("recommended_segnet_distillation_weight"))
    if seg_still_binding and recommended_seg_weight is None:
        recommended_seg_weight = recommend_segnet_distillation_weight_for_stagnation(observed_seg_weight)
    recommended_mutations = list(source.get("recommended_next_mutations") or [])
    if seg_still_binding and (
        "increase_segnet_distillation_weight_from_stagnation_telemetry" not in recommended_mutations
    ):
        recommended_mutations.append("increase_segnet_distillation_weight_from_stagnation_telemetry")
    launch_control_feedback_ready = bool(
        candidate_id
        and last_epoch > 0
        and (
            (pose_instability and recommended_lr is not None and recommended_lr > 0.0)
            or (seg_still_binding and recommended_seg_weight is not None and recommended_seg_weight > 1.0)
        )
    )
    return {
        "schema": NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA,
        "telemetry_feedback_schema": str(source.get("schema")),
        "feedback_kind": "training_telemetry",
        "feedback_scope": "full600_training_telemetry",
        "feedback_ready": False,
        "launch_control_feedback_ready": launch_control_feedback_ready,
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "candidate_num_pairs": 600,
        "measured_num_pairs": 600,
        "scope_matches_candidate": True,
        "training_stopped": False,
        "source_report_path": source.get("telemetry_path"),
        "observed_learning_rate": source.get("learning_rate"),
        "pose_instability_detected": pose_instability,
        "recommended_learning_rate": recommended_lr,
        "pose_recovered_from_initial_spike": pose_recovered,
        "seg_stagnation_detected": seg_still_binding,
        "segnet_still_binding": seg_still_binding,
        "observed_segnet_distillation_weight": observed_seg_weight,
        "recommended_segnet_distillation_weight": recommended_seg_weight,
        "recommended_segnet_distillation_weight_multiplier": (2.0 if seg_still_binding else None),
        "recommended_launch_mutations": recommended_mutations,
        "training_telemetry": {
            "schema": str(source.get("schema")),
            "last_epoch": last_epoch,
            "row_count": int(source.get("row_count") or 0),
            "first_pose_axis": source.get("first_pose_axis"),
            "last_pose_axis": source.get("last_pose_axis"),
            "first_seg_axis": source.get("first_seg_axis"),
            "last_seg_axis": source.get("last_seg_axis"),
            "last_recon_aux": source.get("last_recon_aux"),
            "last_loss": source.get("last_loss"),
            "observed_segnet_distillation_weight": observed_seg_weight,
            "pose_recovered_from_initial_spike": pose_recovered,
            "segnet_still_binding": seg_still_binding,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _augment_feedback_row(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    native = source.get("snerv_mlx_native_export")
    native_file_evidence = source.get("snerv_mlx_native_file_backed_export_evidence")
    if isinstance(native_file_evidence, Mapping):
        out.setdefault(
            "snerv_mlx_native_file_backed_export_evidence",
            dict(native_file_evidence),
        )
        out.setdefault(
            "native_mlx_file_backed_export_proof_passed",
            bool(native_file_evidence.get("required_pair_file_backed_export_proof_passed")),
        )
    if isinstance(native, Mapping):
        out.setdefault(
            "native_mlx_receiver_proof_passed",
            bool(native.get("receiver_proof_passed") and native.get("receiver_contract_satisfied")),
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
        out.setdefault("snerv_mlx_native_export_executed", native.get("executed"))
        out.setdefault(
            "snerv_mlx_native_export_artifact_report_path",
            native.get("artifact_report_path") or native.get("report_path"),
        )
        out.setdefault("snerv_mlx_native_export_packet_path", native.get("packet_path"))
        out.setdefault("snerv_mlx_native_export_packet_sha256", native.get("packet_sha256"))
        out.setdefault("snerv_mlx_native_export_archive_path", native.get("archive_path"))
        out.setdefault("snerv_mlx_native_export_archive_sha256", native.get("archive_sha256"))
        out.setdefault(
            "snerv_mlx_native_export_receiver_proof_path",
            native.get("receiver_proof_path"),
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
        out["full_video_local_prefilter_attached"] = bool(out.get("mlx_prefilter_has_full_video"))
    if "local_cpu_replay_gate_attached" not in out:
        out["local_cpu_replay_gate_attached"] = bool(
            out.get("local_cpu_replay_gate_executed") or out.get("local_cpu_replay_summary_present")
        )
    return out


def _snerv_native_artifact_evidence_from_feedback(
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    embedded = feedback.get("snerv_mlx_native_file_backed_export_evidence")
    artifact = {
        "num_pairs": feedback.get("candidate_num_pairs")
        if feedback.get("scope_matches_candidate")
        else feedback.get("measured_num_pairs"),
        "executed": feedback.get("snerv_mlx_native_export_executed"),
        "artifact_report_path": feedback.get("snerv_mlx_native_export_artifact_report_path"),
        "packet_path": feedback.get("snerv_mlx_native_export_packet_path"),
        "packet_sha256": feedback.get("snerv_mlx_native_export_packet_sha256"),
        "archive_path": feedback.get("snerv_mlx_native_export_archive_path"),
        "archive_sha256": feedback.get("snerv_mlx_native_export_archive_sha256"),
        "receiver_proof_path": feedback.get("snerv_mlx_native_export_receiver_proof_path"),
        "receiver_proof_passed": feedback.get("snerv_mlx_native_export_receiver_proof_passed")
        or feedback.get("native_mlx_receiver_proof_passed"),
        "receiver_contract_satisfied": feedback.get("snerv_mlx_native_export_receiver_contract_satisfied"),
        "scorer_loop_qat": {
            "executed": feedback.get("native_mlx_scorer_loop_qat_attached"),
            "receiver_contract_satisfied": feedback.get("native_mlx_scorer_loop_qat_receiver_contract_satisfied"),
            "ready_for_pose_guard_gate": feedback.get("native_mlx_scorer_loop_qat_ready_for_pose_guard_gate"),
            "accepted_improvement": feedback.get("native_mlx_scorer_loop_qat_accepted_improvement"),
            "emitted_packet_uses_scorer_loop_best_decoder": feedback.get(
                "native_mlx_scorer_loop_qat_best_materialized"
            ),
        },
    }
    compact_artifact = {key: value for key, value in artifact.items() if value is not None}
    if isinstance(embedded, Mapping):
        merged = dict(embedded)
        merged.update(compact_artifact)
        return merged
    return compact_artifact


def _candidate_feedback_for(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return {}
    family_key = _family_key(family)
    rows = list((index or {}).get((family_key, candidate_id)) or [])
    if rows:
        out = dict(rows[0])
        out.setdefault("candidate_id_match", True)
        out.setdefault("feedback_match_scope", "candidate")
        return out
    fallback_rows = _family_level_candidate_feedback_rows(
        candidate=candidate,
        family=family_key,
        index=index,
    )
    if not fallback_rows:
        return {}
    return _sanitize_family_level_candidate_feedback(
        row=fallback_rows[0],
        target_candidate=candidate,
    )


def _family_level_candidate_feedback_rows(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> list[dict[str, Any]]:
    family_key = _family_key(family)
    rows: list[dict[str, Any]] = []
    for (row_family, _row_candidate_id), candidate_rows in (index or {}).items():
        if row_family != family_key:
            continue
        for row in candidate_rows:
            if _family_level_candidate_feedback_applicable(
                candidate=candidate,
                family=family_key,
                row=row,
            ):
                rows.append(dict(row))
    return sorted(rows, key=_candidate_feedback_sort_key, reverse=True)


def _family_level_candidate_feedback_applicable(
    *,
    candidate: Mapping[str, Any],
    family: str,
    row: Mapping[str, Any],
) -> bool:
    # Only reuse optimizer-stability telemetry across sibling HiNeRV candidates.
    # Archive, receiver, and replay evidence remain candidate-specific.
    if _family_key(family) != "hi_nerv":
        return False
    target_candidate_id = str(candidate.get("candidate_id") or "").strip()
    source_candidate_id = str(row.get("candidate_id") or "").strip()
    if not target_candidate_id or not source_candidate_id:
        return False
    if source_candidate_id == target_candidate_id:
        return False
    if str(row.get("feedback_kind") or "").strip() != "training_telemetry":
        return False
    if str(row.get("feedback_scope") or "").strip() != "full600_training_telemetry":
        return False
    pose_feedback = bool(row.get("pose_instability_detected") is True)
    seg_feedback = bool(row.get("seg_stagnation_detected") is True)
    recommended = _float_or_none(row.get("recommended_learning_rate"))
    recommended_seg_weight = _float_or_none(row.get("recommended_segnet_distillation_weight"))
    if not (
        (pose_feedback and recommended is not None and recommended > 0.0)
        or (seg_feedback and recommended_seg_weight is not None and recommended_seg_weight > 1.0)
    ):
        return False
    target_num_pairs = int(candidate.get("num_pairs") or 0)
    measured_num_pairs = int(row.get("measured_num_pairs") or 0)
    return target_num_pairs > 0 and measured_num_pairs == target_num_pairs


def _sanitize_family_level_candidate_feedback(
    *,
    row: Mapping[str, Any],
    target_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    source_candidate_id = str(row.get("candidate_id") or "").strip()
    target_candidate_id = str(target_candidate.get("candidate_id") or "").strip()
    source_official_score = _hinerv_feedback_official_control_score(row)
    target_official_score = _hinerv_official_control_score(target_candidate)
    source_official_superseded = bool(target_official_score > source_official_score)
    out["source_candidate_id"] = source_candidate_id
    out["target_candidate_id"] = str(target_candidate_id)
    out["candidate_id_match"] = False
    out["feedback_match_scope"] = "family_training_telemetry"
    out["family_scope_matches_target"] = True
    out["scope_matches_candidate"] = False
    out["receiver_proof_attached"] = False
    out["full_video_local_prefilter_attached"] = False
    out["local_cpu_replay_gate_attached"] = False
    out["measured_archive_bytes"] = None
    out["measured_payload_bytes"] = None
    out["source_official_control_score"] = int(source_official_score)
    out["target_official_control_score"] = int(target_official_score)
    out["source_official_control_superseded"] = source_official_superseded
    out["feedback_reuse_policy"] = "optimizer_stability_only_no_archive_receiver_or_replay_authority"
    if source_official_superseded:
        mutations = [str(item) for item in (out.get("recommended_launch_mutations") or []) if str(item).strip()]
        if HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION not in mutations:
            mutations.append(HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION)
        out["recommended_launch_mutations"] = mutations
    return out


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
            "decoder_weight_waterfill_source missing _decoder_weight_waterfill_plan_path"
        )
    rows = source.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise NervLongTrainingCampaignPlanError("decoder_weight_waterfill_source must contain non-empty rows")
    out = dict(source)
    out["path"] = str(path)
    out["family"] = _family_key(str(source.get("family") or "hi_nerv"))
    out["group_count"] = int(source.get("group_count") or len(rows))
    out["full_video_coverage"] = bool(source.get("full_video_coverage"))
    out["receiver_proof_ready"] = str(source.get("receiver_proof_status") or "").lower() in {
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
        row.get("row_id"),
        row.get("planner_row_id"),
    ]
    keys: list[str] = []
    for raw in raw_values:
        text = str(raw or "").strip()
        if not text:
            continue
        keys.extend(_candidate_id_aliases(text))
    return tuple(_dedupe(keys))


def _unique_index_row_count(
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> int:
    seen: set[int] = set()
    for rows in index.values():
        for row in rows:
            seen.add(id(row))
    return len(seen)


def _candidate_id_aliases(value: str) -> tuple[str, ...]:
    """Return source-row aliases without dropping the modelsize candidate id."""

    text = str(value or "").strip()
    if not text:
        return ()
    aliases = [text]
    has_double_colon = "::" in text
    if has_double_colon:
        parts = [part.strip() for part in text.split("::") if part.strip()]
        if len(parts) >= 3 and _family_key(parts[0]) in {"hi_nerv", "snerv"}:
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
    return tuple(_dedupe(alias for alias in aliases if alias))


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
    runner_admission = _decoder_weight_waterfill_runner_admission(row)
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
        "runner_admission": runner_admission,
        "runner_admitted": bool(runner_admission["admitted"]),
        "blockers": list(row.get("blockers") or []),
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_runner_admitted(row: Mapping[str, Any]) -> bool:
    return bool(_decoder_weight_waterfill_runner_admission(row)["admitted"])


def _decoder_weight_waterfill_runner_admission(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = {str(blocker) for blocker in row.get("blockers") or ()}
    refusal_reasons: list[str] = []
    if row.get("full_video_coverage") is not True:
        refusal_reasons.append("decoder_weight_waterfill_full_video_coverage_missing")
    if row.get("receiver_proof_ready") is not True:
        refusal_reasons.append("decoder_weight_waterfill_receiver_proof_not_ready")
    for blocker in (
        "decoder_weight_saliency_missing_for_some_groups",
        "full_video_coverage_missing",
        "receiver_proof_not_satisfied",
    ):
        if blocker in blockers:
            refusal_reasons.append(blocker)
    return {
        "schema": "nerv_decoder_weight_waterfill_runner_admission.v1",
        "admitted": not refusal_reasons,
        "mode": (
            "runner_training_pressure_and_export_mutation" if not refusal_reasons else "advisory_learning_signal_only"
        ),
        "refusal_reasons": _dedupe(refusal_reasons),
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_unattached_sources(
    *,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
    campaign_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return source-backed waterfill rows that no campaign row consumed.

    Decoder-weight waterfill is candidate-shape specific. Dropping unmatched
    sources silently loses useful signal and invites false attachments later.
    The planner records them as non-authoritative diagnostics instead.
    """

    attached_paths = {
        str(plan.get("path") or "")
        for row in campaign_rows
        if isinstance((plan := row.get("decoder_weight_waterfill_plan")), Mapping) and plan.get("attached") is True
    }
    target_candidates_by_family: dict[str, list[str]] = {}
    for row in campaign_rows:
        family = _family_key(str(row.get("family") or ""))
        candidate_id = str(row.get("candidate_id") or "").strip()
        if family and candidate_id:
            target_candidates_by_family.setdefault(family, []).append(candidate_id)

    by_path: dict[str, dict[str, Any]] = {}
    for (family, _candidate_key), source_rows in (index or {}).items():
        for source in source_rows:
            path = str(source.get("path") or "")
            if not path or path in attached_paths or path in by_path:
                continue
            family_key = _family_key(str(source.get("family") or family))
            by_path[path] = {
                "schema": ("nerv_long_training_unattached_decoder_weight_waterfill_source.v1"),
                "attached": False,
                "reason": "no_matching_campaign_candidate_id",
                "path": path,
                "sha256": source.get("_decoder_weight_waterfill_plan_sha256"),
                "source_path": source.get("_decoder_weight_waterfill_source_path"),
                "family": family_key,
                "source_candidate_id": source.get("candidate_id"),
                "candidate_keys": list(_candidate_index_keys(source)),
                "target_candidate_ids": sorted(_dedupe(target_candidates_by_family.get(family_key, []))),
                "group_count": int(source.get("group_count") or 0),
                "full_video_coverage": bool(source.get("full_video_coverage")),
                "receiver_proof_ready": bool(source.get("receiver_proof_ready")),
                "blockers": list(source.get("blockers") or []),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
    return sorted(by_path.values(), key=lambda item: str(item.get("path") or ""))


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
            "policy_logic": HINERV_POSE_INSTABILITY_POLICY_LOGIC,
            "learning_rate": float(learning_rate),
            "segnet_distillation_weight": 1.0,
            **FALSE_AUTHORITY,
        }
    launch_control_ready = _feedback_launch_control_ready(feedback)
    if not launch_control_ready:
        return {
            "schema": "hinerv_feedback_launch_adjustment.v1",
            "applied": False,
            "reason": "feedback_not_launch_control_ready",
            "policy_logic": HINERV_POSE_INSTABILITY_POLICY_LOGIC,
            "source_feedback_kind": feedback.get("feedback_kind"),
            "source_feedback_scope": feedback.get("feedback_scope"),
            "feedback_ready": feedback.get("feedback_ready"),
            "launch_control_feedback_ready": feedback.get("launch_control_feedback_ready"),
            "learning_rate": float(learning_rate),
            "segnet_distillation_weight": 1.0,
            **FALSE_AUTHORITY,
        }
    observed = _float_or_none(feedback.get("observed_learning_rate"))
    recommended = _float_or_none(feedback.get("recommended_learning_rate"))
    pose_instability = bool(feedback.get("pose_instability_detected"))
    lr_floor = HINERV_POSE_INSTABILITY_LOW_LR_FLOOR
    repeated_low_lr_instability = bool(pose_instability and observed is not None and observed <= lr_floor)
    lower_learning_rate_applied = bool(
        pose_instability
        and not repeated_low_lr_instability
        and recommended is not None
        and recommended > 0.0
        and recommended < float(learning_rate)
    )
    pose_protected_pathway_applied = bool(repeated_low_lr_instability)
    launch_mutations: list[str] = []
    if lower_learning_rate_applied:
        launch_mutations.extend(list(feedback.get("recommended_launch_mutations") or []))
    if pose_protected_pathway_applied:
        launch_mutations.append("enable_pose_distillation_huber_from_repeated_low_lr_instability")
    seg_stagnation = bool(feedback.get("seg_stagnation_detected"))
    recommended_seg_weight = _float_or_none(feedback.get("recommended_segnet_distillation_weight"))
    segnet_weight_applied = bool(seg_stagnation and recommended_seg_weight is not None and recommended_seg_weight > 1.0)
    if segnet_weight_applied:
        launch_mutations.extend(
            mutation
            for mutation in (feedback.get("recommended_launch_mutations") or [])
            if mutation not in launch_mutations
        )
    official_control_superseded = bool(feedback.get("source_official_control_superseded"))
    if official_control_superseded and (HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION not in launch_mutations):
        launch_mutations.append(HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION)
    applied = bool(
        lower_learning_rate_applied
        or pose_protected_pathway_applied
        or segnet_weight_applied
        or official_control_superseded
    )
    return {
        "schema": "hinerv_feedback_launch_adjustment.v1",
        "applied": applied,
        "lower_learning_rate_applied": lower_learning_rate_applied,
        "pose_protected_pathway_applied": pose_protected_pathway_applied,
        "segnet_weight_applied": segnet_weight_applied,
        "official_control_superseded": official_control_superseded,
        "policy_logic": HINERV_POSE_INSTABILITY_POLICY_LOGIC,
        "reason": (
            "pose_instability_recommended_lower_learning_rate"
            if lower_learning_rate_applied
            else (
                "repeated_pose_instability_at_low_lr_pose_protected_pathway"
                if pose_protected_pathway_applied
                else (
                    "segnet_stagnation_recommended_higher_segnet_weight"
                    if segnet_weight_applied
                    else (
                        "official_hinerv_controls_supersede_source_feedback_run"
                        if official_control_superseded
                        else (
                            "pose_instability_feedback_without_lower_lr"
                            if pose_instability
                            else "feedback_does_not_request_launch_adjustment"
                        )
                    )
                )
            )
        ),
        "source_feedback_kind": feedback.get("feedback_kind"),
        "source_feedback_scope": feedback.get("feedback_scope"),
        "feedback_ready": feedback.get("feedback_ready"),
        "launch_control_feedback_ready": launch_control_ready,
        "pose_instability_detected": pose_instability,
        "seg_stagnation_detected": seg_stagnation,
        "observed_learning_rate": observed,
        "low_learning_rate_floor": lr_floor,
        "repeated_low_lr_pose_instability": repeated_low_lr_instability,
        "requested_learning_rate": float(learning_rate),
        "recommended_learning_rate": recommended,
        "learning_rate": float(recommended if lower_learning_rate_applied else learning_rate),
        "recommended_segnet_distillation_weight": recommended_seg_weight,
        "segnet_distillation_weight": float(recommended_seg_weight if segnet_weight_applied else 1.0),
        "pose_distillation_loss": (HINERV_POSE_PROTECTED_LOSS if pose_protected_pathway_applied else "mse"),
        "pose_distillation_huber_delta": (
            HINERV_POSE_PROTECTED_HUBER_DELTA if pose_protected_pathway_applied else None
        ),
        "launch_mutations": launch_mutations,
        **FALSE_AUTHORITY,
    }


def _feedback_launch_control_ready(feedback: Mapping[str, Any]) -> bool:
    if feedback.get("launch_control_feedback_ready") is True:
        return True
    if (
        str(feedback.get("feedback_kind") or "") == "training_telemetry"
        and str(feedback.get("feedback_scope") or "") == "full600_training_telemetry"
    ):
        recommended_lr = _float_or_none(feedback.get("recommended_learning_rate"))
        recommended_seg_weight = _float_or_none(feedback.get("recommended_segnet_distillation_weight"))
        pose_ready = (
            feedback.get("pose_instability_detected") is True and recommended_lr is not None and recommended_lr > 0.0
        )
        seg_ready = (
            feedback.get("seg_stagnation_detected") is True
            and recommended_seg_weight is not None
            and recommended_seg_weight > 1.0
        )
        return bool(pose_ready or seg_ready)
    if feedback.get("feedback_ready") is False:
        return False
    return bool(feedback)


def _hinerv_source_faithfulness_controls(
    *,
    candidate: Mapping[str, Any],
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    target_score = _hinerv_official_control_score(candidate)
    source_score = _hinerv_feedback_official_control_score(feedback) if feedback else target_score
    return {
        "schema": "hinerv_source_faithfulness_controls.v1",
        "target_candidate_id": str(candidate.get("candidate_id") or ""),
        "target_uses_hierarchical_feature_grid": bool(candidate.get("use_hierarchical_feature_grid")),
        "target_uses_convnext_blocks": bool(candidate.get("use_convnext_blocks")),
        "target_official_control_score": int(target_score),
        "source_feedback_candidate_id": str(feedback.get("source_candidate_id") or feedback.get("candidate_id") or ""),
        "source_official_control_score": int(source_score),
        "source_official_control_superseded": bool(
            feedback.get("source_official_control_superseded") or source_score < target_score
        ),
        **FALSE_AUTHORITY,
    }


def _hinerv_official_control_score(row: Mapping[str, Any]) -> int:
    return int(bool(row.get("use_hierarchical_feature_grid"))) + int(bool(row.get("use_convnext_blocks")))


def _hinerv_feedback_official_control_score(row: Mapping[str, Any]) -> int:
    nested = row.get("candidate")
    if isinstance(nested, Mapping):
        nested_score = _hinerv_official_control_score(nested)
        if nested_score:
            return nested_score
    explicit_score = _hinerv_official_control_score(row)
    if explicit_score:
        return explicit_score
    candidate_id = str(row.get("source_candidate_id") or row.get("candidate_id") or "").lower()
    return int("_hfg" in candidate_id) + int("_cnx" in candidate_id)


def _experiment_row_metadata(extra: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "coder_qat_control",
        "decoder_weight_waterfill_plan",
        "feedback_launch_adjustment",
        "optimizer_control",
        "optimizer_policy",
        "source_faithfulness_controls",
        "source_bound_capacity_controls",
        "source_bound_capacity_control_blockers",
        "source_parity",
        "current_command_is_bounded_proof_not_long_training",
        "snerv_bounded_proof_epochs",
        "output_dir_reuse_policy",
    )
    return {
        key: dict(extra[key]) if isinstance(extra.get(key), Mapping) else extra[key] for key in keys if key in extra
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
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise NervLongTrainingCampaignPlanError(
                f"invalid joint recon weight manifest JSON: {manifest_path}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest must be an object: {manifest_path}")
        if manifest.get("schema") != JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA:
            raise NervLongTrainingCampaignPlanError(f"unsupported joint recon weight manifest schema: {manifest_path}")
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
                f"joint recon weight manifest is not recommended for training consumption: {manifest_path}"
            )
        blockers = [str(item) for item in metadata.get("blockers") or [] if item]
        if blockers:
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest has blockers: {manifest_path}")
        raw_weight_path = manifest.get("weight_path")
        if raw_weight_path is None:
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest missing weight_path: {manifest_path}")
        weight_path = Path(str(raw_weight_path)).expanduser()
        if not weight_path.is_absolute():
            weight_path = manifest_path.parent / weight_path
        weight_path = weight_path.resolve(strict=False)
        if not weight_path.is_file():
            raise NervLongTrainingCampaignPlanError(f"joint recon weight file not found: {weight_path}")
        actual_sha = _sha256_file(weight_path)
        expected_sha = str(manifest.get("weight_sha256") or "")
        if expected_sha and actual_sha != expected_sha:
            raise NervLongTrainingCampaignPlanError(f"joint recon weight sha mismatch: {weight_path}")
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
    if launch_feedback_adjustment.get("seg_stagnation_detected"):
        suffix_parts.append("seg_stagnation")
    learning_rate = _float_or_none(launch_feedback_adjustment.get("learning_rate"))
    if learning_rate is not None:
        suffix_parts.append(f"lr{_safe_path_token(_float_token(learning_rate))}")
    segnet_weight = _float_or_none(launch_feedback_adjustment.get("segnet_distillation_weight"))
    if launch_feedback_adjustment.get("segnet_weight_applied") and segnet_weight is not None:
        suffix_parts.append(f"segw{_safe_path_token(_float_token(segnet_weight))}")
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


def _coder_qat_control(*, quant_bits: int) -> dict[str, Any]:
    return {
        "schema": "nerv_long_training_coder_qat_control.v1",
        "quant_bits": int(quant_bits),
        "quant_residual_weight": float(DEFAULT_CODER_QAT_QUANT_RESIDUAL_WEIGHT),
        "magnitude_weight": float(DEFAULT_CODER_QAT_MAGNITUDE_WEIGHT),
        "delta_weight": float(DEFAULT_CODER_QAT_DELTA_WEIGHT),
        "c1a_entropy_weight": float(DEFAULT_CODER_QAT_C1A_ENTROPY_WEIGHT),
        "c1a_sigma": float(DEFAULT_CODER_QAT_C1A_SIGMA),
        "c1a_sample_size": int(DEFAULT_CODER_QAT_C1A_SAMPLE_SIZE),
        "c1a_source": ("PR95 cat_entropy_v2 soft categorical entropy adapted to selected decoder weights"),
        **FALSE_AUTHORITY,
    }


def _coder_qat_command_args(*, quant_bits: int) -> list[str]:
    control = _coder_qat_control(quant_bits=int(quant_bits))
    return [
        "--coder-qat-quant-residual-weight",
        _float_token(float(control["quant_residual_weight"])),
        "--coder-qat-magnitude-weight",
        _float_token(float(control["magnitude_weight"])),
        "--coder-qat-delta-weight",
        _float_token(float(control["delta_weight"])),
        "--coder-qat-c1a-entropy-weight",
        _float_token(float(control["c1a_entropy_weight"])),
        "--coder-qat-c1a-sigma",
        _float_token(float(control["c1a_sigma"])),
        "--coder-qat-c1a-sample-size",
        str(int(control["c1a_sample_size"])),
    ]


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
