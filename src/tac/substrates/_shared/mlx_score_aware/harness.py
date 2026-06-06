# SPDX-License-Identifier: MIT
"""The substrate-AGNOSTIC ``_full_main`` orchestrator (separation of concerns).

This module owns ONLY the orchestration: verify MLX (device gate) -> verify
inflate portability (optional) -> wrap the bundle in the Style-B adapter ->
build the canonical ``LongTrainingConfig`` -> route through canonical
``run_long_training``. Every step delegates to a focused sub-module; the
orchestrator composes them.

Non-promotable by construction per CLAUDE.md "MLX portable-local-substrate
authority" + Catalog #127/#192/#317/#341: every artifact is tagged
``[macOS-MLX research-signal]`` with ``score_claim=False``,
``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``. The
canonical L2 harness auto-stamps these markers on the ``TrainingArtifact``.

Dispatch gating (Catalog #325): this orchestrator runs on the M5 Max via MLX at
$0; it NEVER triggers a paid GPU dispatch. The device gate fails closed on a
non-MLX host (no silent CPU/CUDA fallback per Catalog #1 + #317).

[verified-against: tac.training.long_training_canonical.run_long_training canonical L2 harness]
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
from tac.substrates._shared.mlx_score_aware.device_gate import (
    require_mlx_for_harness,
)
from tac.substrates._shared.mlx_score_aware.portability import (
    assert_numpy_portable_inflate,
)

if TYPE_CHECKING:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

# Canonical contest constants (sister of pact_nerv_full_main).
CONTEST_NORMALIZER: float = 37_545_489.0
# MLX false-authority canonical marker (sister of pr95_hnerv_mlx FALSE_AUTHORITY).
MLX_EVIDENCE_GRADE: str = "[macOS-MLX research-signal]"


def run_mlx_score_aware_full_main(
    *,
    bundle: RendererBundle,
    substrate_id: str,
    lane_id: str,
    output_dir: Any,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float = 1e-3,
    seed: int = 0,
    ema_decay: float | None = None,
    checkpoint_interval_epochs: int = 10,
    checkpoint_retention_keep_last_n: int | None = None,
    checkpoint_retention_keep_best_n: int = 1,
    checkpoint_retention_keep_every_n_epochs: int | None = None,
    checkpoint_retention_cold_store_roots: tuple[Any, ...] = (),
    telemetry_flush_interval_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    curriculum_stages: Any | None = None,
    checkpoint_dir: Any | None = None,
    resume_from_checkpoint: Any | None = None,
    inflate_py_path: Any | None = None,
    notes: str = "",
    on_epoch_end: Callable[[Any], None] | None = None,
    pr95_faithful_curriculum_enabled: bool = False,
    pr95_curriculum_total_epochs: int | None = None,
    pr95_muon_policy: str = "faithful_stage8_only",
    pr95_stage_source_weight_amplification_enabled: bool = False,
    pr95_force_weighted_extra_qat_when_stage_inactive: bool = False,
    # Wave N+11 Z7-Mamba-2 stabilizer recipe (forwarded to adapter; see
    # MlxScoreAwareAdapter.__init__ for canonical contract docstring).
    # All None/0/"adamw"/False defaults preserve byte-stable legacy behavior.
    grad_clip_max_norm: float | None = None,
    warmup_epochs: int = 0,
    warmup_steps_per_epoch: int = 1,
    weight_decay: float | None = None,
    optimizer_kind: str = "pact_muon_adamw",
    cosine_decay_enabled: bool = False,
    cosine_decay_total_epochs: int | None = None,
    cosine_decay_min_lr_ratio: float = 1e-2,
    train_time_dual_ascent_config: Mapping[str, Any] | None = None,
    prioritized_pair_indices: tuple[int, ...] = (),
    pair_sampling_weights: Mapping[int, float] | None = None,
    pair_sampling_default_weight: float = 1.0,
    gradient_multiplier_by_name: Mapping[str, float] | None = None,
    bias_gradient_multiplier: float | None = None,
    output_head_bias_gradient_multiplier: float = 1.0,
    scorer_space_step_guard_enabled: bool = False,
    scorer_space_step_guard_min_pre_segnet_occupied_class_fraction: float = 0.4,
    scorer_space_step_guard_min_post_segnet_occupied_class_fraction: float = 0.4,
    scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction: float
    | None = None,
    scorer_space_step_guard_min_post_segnet_target_class_min_ratio: float
    | None = None,
    scorer_space_step_guard_max_post_segnet_target_class_ratio_drop: float
    | None = None,
    scorer_space_step_guard_max_post_segnet_contrast_ratio: float | None = None,
    scorer_space_step_guard_max_post_segnet_distribution_mae: float | None = None,
    scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae: float
    | None = None,
    scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio: float
    | None = None,
    scorer_space_step_guard_max_post_segnet_argmax_disagreement: float | None = None,
    scorer_space_step_guard_max_post_pose_score_term: float | None = None,
    scorer_space_step_guard_max_post_pose_direct_live_score_term: float | None = None,
    scorer_space_step_guard_max_pose_score_term_relative_worsening: float
    | None = None,
    scorer_space_step_guard_max_pose_score_term_absolute_worsening: float
    | None = None,
    scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening: float
    | None = None,
    scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening: float
    | None = None,
    scorer_space_step_guard_max_direct_nonrate_score_worsening: float | None = None,
    scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening: float
    | None = None,
    scorer_space_step_guard_backtracking_steps: int = 0,
    scorer_space_step_guard_backtracking_shrink: float = 0.5,
    scorer_support_ladder_enabled: bool = False,
    scorer_support_ladder_target_coverage_floor: float | None = None,
    scorer_support_ladder_target_min_ratio_floor: float | None = None,
    scorer_support_ladder_patience_steps: int = 1,
    scorer_support_ladder_growth_factor: float = 2.0,
    scorer_support_ladder_max_multiplier: float = 16.0,
    scorer_support_ladder_base_loss_max_when_active: float = 0.25,
    scorer_support_ladder_activation_weights: Mapping[str, Any] | None = None,
    ema_archive_selection_enabled: bool = False,
    checkpoint_selection_metric_key: str = "total",
    checkpoint_selection_metric_mode: str = "min",
    checkpoint_selection_metric_required: bool = False,
    checkpoint_selection_tie_break_metric_key: str = "",
    checkpoint_selection_tie_break_metric_mode: str = "min",
    checkpoint_selection_tie_break_metric_required: bool = False,
) -> Any:
    """Run the canonical MLX-first score-aware ``_full_main`` body.

    This is the substrate-AGNOSTIC ``_full_main`` the MLX-first substrate
    trainers route through. It:

    1. Verifies MLX availability (fail-closed; no CPU/CUDA leak per Catalog
       #1 + #317 + #325).
    2. (Optional) Verifies the substrate's ``inflate.py`` is numpy-portable
       (8th directive; HNeRV parity L4) when ``inflate_py_path`` is supplied.
    3. Wraps the substrate ``RendererBundle`` in :class:`MlxScoreAwareAdapter`.
    4. Builds a canonical ``LongTrainingConfig`` (single full-stage curriculum
       by default; the substrate may pass a multi-stage curriculum).
    5. Routes through canonical ``run_long_training`` (EMA / OOM-safe /
       telemetry / checkpoint / Provenance / posterior anchor / archive
       export).

    Args:
        bundle: the substrate RendererBundle (UNIQUE axis).
        substrate_id: canonical substrate id.
        lane_id: canonical lane id per CLAUDE.md "Lane maturity registry".
        output_dir: canonical output dir (MUST NOT be ``/tmp`` per the
            transient-evidence trap; ``run_long_training`` validates this).
        epochs: total epoch budget.
        batch_pair_indices_per_step: training batch size.
        learning_rate / seed / checkpoint_interval_epochs: training hparams.
        checkpoint_retention_keep_last_n / keep_best_n /
            keep_every_n_epochs / cold_store_roots: optional hot-retention
            policy for long checkpoint-heavy MLX campaigns. The canonical
            trainer keeps last/best/milestone checkpoints hot and moves older
            periodic checkpoints to cold store with JSONL provenance.
        telemetry_flush_interval_epochs: optional per-run override for canonical
            telemetry JSONL flush cadence. Use 1 for long carrier campaigns so
            epoch rows are durable while the process is still running.
        ema_decay: optional EMA decay override (default = canonical 0.997).
        early_stopping_patience: optional override (default = epochs + 1, i.e.
            disabled; MLX-local runs are cheap so we run the full budget).
        curriculum_stages: optional ``tuple[CurriculumStage, ...]``; default is
            a single full-budget stage.
        checkpoint_dir: optional canonical checkpoint directory. Long compact
            carrier campaigns pass an SSD-backed directory so periodic
            checkpoints are recoverable before final archive export.
        resume_from_checkpoint: optional canonical checkpoint metadata JSON to
            resume from. The canonical trainer validates substrate, lane, and
            curriculum hash and the MLX adapter performs a real ``.npsd`` state
            restore, so resume cannot degrade into metadata-only continuation.
        inflate_py_path: optional path to the substrate ``inflate.py`` to
            verify numpy-portability before training (8th directive).
        notes: substantive rationale (Catalog #287 placeholder rejected by the
            config).
        on_epoch_end: optional per-epoch callback.
        pr95_faithful_curriculum_enabled: opt-in to PR95-faithful 8-stage
            Muon+AdamW canonical curriculum per CLAUDE.md "HNeRV /
            leaderboard-implementation parity discipline" L14 + L15 +
            the optimizer stack research memo (commit 118ddb1a4) Option A
            MINIMUM-VIABLE recommendation + the m9-v3 canonical helper
            (commit c91481212). Default False preserves the legacy
            default-on AdamW behavior (backward compat per CLAUDE.md
            "Beauty, simplicity, and developer experience"). When True,
            the adapter routes per-stage optimizer state through the
            canonical ``apply_pr95_mlx_optimizer_step`` via the canonical
            ``PR95FaithfulCurriculumFactory``; the canonical
            ``run_long_training`` epoch loop notifies the adapter of the
            current global epoch via the new ``notify_global_epoch``
            wiring point so each stage actually advances per CLAUDE.md
            "NO FAKE IMPLEMENTATIONS" non-negotiable.
        pr95_curriculum_total_epochs: total epoch budget for the PR95
            curriculum; defaults to the canonical 29,650 per L14.
            Used only when ``pr95_faithful_curriculum_enabled=True``;
            ignored otherwise. Smaller budgets (e.g. 100 for an MLX
            smoke) scale per ``PR95FaithfulCurriculumFactory`` canonical
            proportional-ratio rule.
        pr95_muon_policy: Muon activation policy for PR95-curriculum runs.
            ``faithful_stage8_only`` is source-faithful PR95; ``every_stage``
            is an explicit contest-specific optimizer control that keeps the
            PR95 stage loss/QAT curriculum but turns on the same real Muon
            branch for all stages.
        pr95_stage_source_weight_amplification_enabled: opt in to PR95's
            original source-scale SegNet:PoseNet multiplier (100:1). Default
            False keeps generic compact renderer launches on literal operator
            scorer weights so fit-first byte-cap runs do not inherit an
            implicit 100x SegNet amplification.
        pr95_force_weighted_extra_qat_when_stage_inactive: default-off
            bounded-proof switch for non-PR95 archive-section byte controls.
            When true, weighted ``RendererBundle.extra_loss_terms`` are
            evaluated even before PR95's native ``qat_active`` stages.
        prioritized_pair_indices: optional hard-pair/sensitivity pair indices
            sampled before random fill by the shared MLX adapter. This is local
            training emphasis and telemetry only; it does not create full-video
            replay or score authority.
        pair_sampling_weights: optional XRay/scorer-error pair weights consumed
            by the shared MLX adapter sampler. This makes post-export pair
            anatomy usable during subsequent training runs without turning local
            scorer telemetry into promotion authority.
        pair_sampling_default_weight: baseline sampling mass for pairs not
            present in ``pair_sampling_weights``.
        train_time_dual_ascent_config: optional projected dual-ascent controller
            over score-aware loss-part metrics. This is the closed-loop
            train-time byte/scorer pressure surface shared by HiNeRV and SNeRV;
            it remains MLX-local false-authority until archive/runtime and exact
            CPU/CUDA replay evidence exists.
        gradient_multiplier_by_name / bias_gradient_multiplier /
            output_head_bias_gradient_multiplier:
            exact-name optimizer multipliers applied inside the shared adapter
            after finite-gradient validation and before clipping/update. These
            are scorer-aware train-time waterfilling/ablation controls, not
            metadata-only knobs.
        scorer_space_step_guard_enabled / min_pre / min_post / min_target_coverage /
            max_contrast:
            optional scorer-domain trust-region guard. When enabled, renderer
            optimizer steps that collapse real SegNet direct-live class
            occupancy after a noncollapsed pre-update state are rejected by
            restoring renderer parameters and emitting fail-closed telemetry. The
            target-class coverage floor protects the upstream-evaluate SegNet
            last-frame class support separately from generic occupied-class
            count. The backtracking controls optionally accept a smaller
            interpolated fraction of the proposed step before falling back to
            restore.
        ema_archive_selection_enabled: when True, the canonical trainer exports
            both live and EMA final archives, evaluates their local score-aware
            proxy plus charged archive bytes, writes
            ``ema_archive_selection/ema_archive_selection.json``, and returns
            the selected archive. This is advisory MLX-local selection, not
            exact CPU/CUDA authority.
        checkpoint_selection_metric_key / checkpoint_selection_metric_mode:
            scorer-facing best-checkpoint selector threaded into
            ``LongTrainingConfig``. Compact carriers use this to keep archive
            export tied to direct scorer movement instead of aggregate guard or
            coder losses.
        checkpoint_selection_metric_required: fail closed when the named
            checkpoint metric is absent or malformed. Direct-live scorer runs
            use this to prevent fallback-total exports from collapsed renderers.

    Returns:
        the canonical ``TrainingArtifact`` from ``run_long_training``.

    Raises:
        MlxScoreAwareHarnessError: MLX unavailable OR inflate not portable.
    """
    require_mlx_for_harness()
    output_dir = Path(output_dir)

    from tac.training.long_training_canonical import (
        CANONICAL_EMA_DECAY,
        DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS,
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    if inflate_py_path is not None:
        assert_numpy_portable_inflate(inflate_py_path)

    if curriculum_stages is None:
        curriculum_stages = (
            CurriculumStage(
                name=f"{substrate_id}_mlx_score_aware_full",
                start_epoch=0,
                end_epoch=epochs,
                notes=(
                    "MLX-first score-aware full-budget stage via canonical "
                    "mlx_score_aware harness; reconstruction + "
                    "optional Hinton-KL T=2.0 scorer surrogate."
                ),
            ),
        )

    # Construct the canonical MLX score-aware adapter. When
    # ``pr95_faithful_curriculum_enabled=True``, the adapter routes per-stage
    # optimizer state through the canonical
    # ``apply_pr95_mlx_optimizer_step`` via the canonical
    # ``PR95FaithfulCurriculumFactory`` (commit c91481212 m9-v3 helper) and
    # the canonical ``run_long_training`` epoch loop notifies the adapter of
    # the current global epoch via ``notify_global_epoch`` per CLAUDE.md
    # "HNeRV / leaderboard-implementation parity discipline" L14 + L15.
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id=substrate_id,
        pr95_faithful_curriculum_enabled=pr95_faithful_curriculum_enabled,
        pr95_curriculum_total_epochs=pr95_curriculum_total_epochs,
        pr95_muon_policy=pr95_muon_policy,
        pr95_stage_source_weight_amplification_enabled=(
            pr95_stage_source_weight_amplification_enabled
        ),
        pr95_force_weighted_extra_qat_when_stage_inactive=bool(
            pr95_force_weighted_extra_qat_when_stage_inactive
        ),
        # Wave N+11 stabilizer kwargs (forwarded; defaults are
        # legacy-preserving so sister substrates remain byte-stable).
        grad_clip_max_norm=grad_clip_max_norm,
        warmup_epochs=warmup_epochs,
        warmup_steps_per_epoch=warmup_steps_per_epoch,
        weight_decay=weight_decay,
        optimizer_kind=optimizer_kind,
        cosine_decay_enabled=cosine_decay_enabled,
        cosine_decay_total_epochs=cosine_decay_total_epochs,
        cosine_decay_min_lr_ratio=cosine_decay_min_lr_ratio,
        train_time_dual_ascent_config=train_time_dual_ascent_config,
        prioritized_pair_indices=prioritized_pair_indices,
        pair_sampling_weights=pair_sampling_weights,
        pair_sampling_default_weight=pair_sampling_default_weight,
        gradient_multiplier_by_name=gradient_multiplier_by_name,
        bias_gradient_multiplier=bias_gradient_multiplier,
        output_head_bias_gradient_multiplier=output_head_bias_gradient_multiplier,
        scorer_space_step_guard_enabled=scorer_space_step_guard_enabled,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=(
            scorer_space_step_guard_min_pre_segnet_occupied_class_fraction
        ),
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=(
            scorer_space_step_guard_min_post_segnet_occupied_class_fraction
        ),
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=(
            scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
        ),
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio=(
            scorer_space_step_guard_min_post_segnet_target_class_min_ratio
        ),
        scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=(
            scorer_space_step_guard_max_post_segnet_target_class_ratio_drop
        ),
        scorer_space_step_guard_max_post_segnet_contrast_ratio=(
            scorer_space_step_guard_max_post_segnet_contrast_ratio
        ),
        scorer_space_step_guard_max_post_segnet_distribution_mae=(
            scorer_space_step_guard_max_post_segnet_distribution_mae
        ),
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=(
            scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae
        ),
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=(
            scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio
        ),
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=(
            scorer_space_step_guard_max_post_segnet_argmax_disagreement
        ),
        scorer_space_step_guard_max_post_pose_score_term=(
            scorer_space_step_guard_max_post_pose_score_term
        ),
        scorer_space_step_guard_max_post_pose_direct_live_score_term=(
            scorer_space_step_guard_max_post_pose_direct_live_score_term
        ),
        scorer_space_step_guard_max_pose_score_term_relative_worsening=(
            scorer_space_step_guard_max_pose_score_term_relative_worsening
        ),
        scorer_space_step_guard_max_pose_score_term_absolute_worsening=(
            scorer_space_step_guard_max_pose_score_term_absolute_worsening
        ),
        scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening=(
            scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening
        ),
        scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening=(
            scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening
        ),
        scorer_space_step_guard_max_direct_nonrate_score_worsening=(
            scorer_space_step_guard_max_direct_nonrate_score_worsening
        ),
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=(
            scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening
        ),
        scorer_space_step_guard_backtracking_steps=(
            scorer_space_step_guard_backtracking_steps
        ),
        scorer_space_step_guard_backtracking_shrink=(
            scorer_space_step_guard_backtracking_shrink
        ),
        scorer_support_ladder_enabled=scorer_support_ladder_enabled,
        scorer_support_ladder_target_coverage_floor=(
            scorer_support_ladder_target_coverage_floor
        ),
        scorer_support_ladder_target_min_ratio_floor=(
            scorer_support_ladder_target_min_ratio_floor
        ),
        scorer_support_ladder_patience_steps=scorer_support_ladder_patience_steps,
        scorer_support_ladder_growth_factor=scorer_support_ladder_growth_factor,
        scorer_support_ladder_max_multiplier=scorer_support_ladder_max_multiplier,
        scorer_support_ladder_base_loss_max_when_active=(
            scorer_support_ladder_base_loss_max_when_active
        ),
        scorer_support_ladder_activation_weights=(
            scorer_support_ladder_activation_weights
        ),
    )

    config = LongTrainingConfig(
        substrate_id=substrate_id,
        lane_id=lane_id,
        epochs=epochs,
        batch_pair_indices_per_step=batch_pair_indices_per_step,
        curriculum_stages=curriculum_stages,
        ema_decay=CANONICAL_EMA_DECAY if ema_decay is None else float(ema_decay),
        checkpoint_interval_epochs=checkpoint_interval_epochs,
        checkpoint_retention_keep_last_n=checkpoint_retention_keep_last_n,
        checkpoint_retention_keep_best_n=int(checkpoint_retention_keep_best_n),
        checkpoint_retention_keep_every_n_epochs=(
            checkpoint_retention_keep_every_n_epochs
        ),
        checkpoint_retention_cold_store_roots=tuple(
            Path(root) for root in checkpoint_retention_cold_store_roots
        ),
        telemetry_flush_interval_epochs=(
            telemetry_flush_interval_epochs
            if telemetry_flush_interval_epochs is not None
            else DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS
        ),
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir is not None else None,
        resume_from_checkpoint=(
            Path(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        ),
        early_stopping_patience=(
            epochs + 1
            if early_stopping_patience is None
            else early_stopping_patience
        ),
        learning_rate=learning_rate,
        seed=seed,
        output_dir=output_dir,
        device="mlx",
        evidence_grade=MLX_EVIDENCE_GRADE,
        ema_archive_selection_enabled=bool(ema_archive_selection_enabled),
        checkpoint_selection_metric_key=str(checkpoint_selection_metric_key),
        checkpoint_selection_metric_mode=str(checkpoint_selection_metric_mode),
        checkpoint_selection_metric_required=bool(
            checkpoint_selection_metric_required
        ),
        checkpoint_selection_tie_break_metric_key=str(
            checkpoint_selection_tie_break_metric_key
        ),
        checkpoint_selection_tie_break_metric_mode=str(
            checkpoint_selection_tie_break_metric_mode
        ),
        checkpoint_selection_tie_break_metric_required=bool(
            checkpoint_selection_tie_break_metric_required
        ),
        notes=(
            notes
            or (
                f"{substrate_id} MLX-first score-aware L2 via canonical "
                "mlx_score_aware harness; non-promotable "
                "[macOS-MLX research-signal] per Catalog #192/#317/#341."
            )
        ),
    )

    return run_long_training(adapter, config, on_epoch_end=on_epoch_end)


__all__ = [
    "CONTEST_NORMALIZER",
    "MLX_EVIDENCE_GRADE",
    "run_mlx_score_aware_full_main",
]
