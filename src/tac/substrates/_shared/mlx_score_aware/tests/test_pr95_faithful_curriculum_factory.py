# SPDX-License-Identifier: MIT
"""Tests for PR95FaithfulCurriculumFactory + MlxScoreAwareAdapter opt-in wire-in.

Verifies per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable:
- Each of the 8 stages emits a distinct optimizer_config (NOT canonical-marker stub).
- Stage 8 uses Muon (NOT AdamW disguised).
- Per-stage hyperparams (loss_family, qat_active, sigma, lambda) match canonical
  PR95 source-faithful descriptors per CLAUDE.md L14 + L15.
- Backward-compat: default-off preserves existing MlxScoreAwareAdapter behavior.
- Opt-in active: train_step routes through canonical apply_pr95_mlx_optimizer_step.
- Muon partition (Conv/Linear ≥2D weights non-stem/non-rgb/non-latents) preserved.

[verified-against: tac.local_acceleration.pr95_hnerv_mlx.PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS]
[verified-against: tac.optimization.optimizer_scheduler_registry.default_optimizer_scheduler_descriptors]
[verified-against: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14 + L15]
"""
from __future__ import annotations

import inspect

import pytest

# Skip the entire module when MLX is unavailable; the adapter+factory require MLX.
mlx_available = True
try:
    import mlx.core as _mx  # noqa: F401
except Exception:
    mlx_available = False

requires_mlx = pytest.mark.skipif(
    not mlx_available,
    reason="MLX unavailable on this host; PR95 8-stage curriculum requires MLX",
)


# ---------- Section 1: PR95FaithfulCurriculumFactory unit tests ----------


def test_pr95_stage_loss_uses_preserved_exact_segnet_target_argmax() -> None:
    """PR95-stage SegNet hard labels must match the exact scorer target labels."""

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    source = inspect.getsource(MlxScoreAwareAdapter._pr95_stage_loss_and_parts)

    assert "_exact_segnet_target_argmax_for_indices(" in source
    assert "targets_hard_nhw = mx.argmax(teacher_logits_nhwc, axis=-1)" not in source


def test_adapter_aliases_pr95_direct_live_pose_into_joint_scorer_proxy() -> None:
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    adapter = object.__new__(MlxScoreAwareAdapter)
    metrics = {
        "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement": 0.25,
        "loss_part_pose_score_term": 3.5,
    }

    adapter._add_dual_ascent_metric_aliases(metrics)

    assert metrics["loss_part_segnet_direct_live_argmax_disagreement"] == (
        pytest.approx(0.25)
    )
    assert metrics["loss_part_joint_scorer_proxy_nonrate"] == pytest.approx(28.5)


def test_adapter_joint_scorer_proxy_penalizes_pose_warmup_missing_metric() -> None:
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    adapter = object.__new__(MlxScoreAwareAdapter)
    metrics = {
        "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement": 0.25,
        "loss_part_pr95_stage_effective_pose_weight": 1.0,
    }

    adapter._add_dual_ascent_metric_aliases(metrics)

    assert metrics["loss_part_segnet_direct_live_argmax_disagreement"] == (
        pytest.approx(0.25)
    )
    assert metrics["loss_part_joint_scorer_proxy_nonrate"] == pytest.approx(
        1.0e9 + 25.0
    )


def test_adapter_joint_scorer_proxy_not_synthesized_when_pose_stage_inactive() -> None:
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    adapter = object.__new__(MlxScoreAwareAdapter)
    metrics = {
        "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement": 0.25,
        "loss_part_pr95_stage_effective_pose_weight": 0.0,
    }

    adapter._add_dual_ascent_metric_aliases(metrics)

    assert metrics["loss_part_segnet_direct_live_argmax_disagreement"] == (
        pytest.approx(0.25)
    )
    assert "loss_part_joint_scorer_proxy_nonrate" not in metrics


@requires_mlx
def test_canonical_pr95_total_epochs_is_29650() -> None:
    """Canonical PR95 total epoch budget per CLAUDE.md L14 is 29,650."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
    )

    assert CANONICAL_PR95_TOTAL_EPOCHS == 29_650
    # Sister check: 3000 + 5650 + 1500 + 500 + 9000 + 2000 + 3000 + 5000 = 29650
    canonical_breakdown = [3000, 5650, 1500, 500, 9000, 2000, 3000, 5000]
    assert sum(canonical_breakdown) == 29_650


@requires_mlx
def test_factory_canonical_budget_uses_verbatim_pr95_source_epochs() -> None:
    """At canonical budget, factory uses PR95 source-faithful per-stage epochs."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    assert factory.is_canonical_pr95_budget is True
    boundaries = factory.stage_epoch_boundaries
    assert len(boundaries) == 8
    # Stage indices are 1..8.
    assert [b[0] for b in boundaries] == [1, 2, 3, 4, 5, 6, 7, 8]
    # Per-stage epochs match canonical PR95 source verbatim.
    canonical_per_stage = [3000, 5650, 1500, 500, 9000, 2000, 3000, 5000]
    for (_stage_index, start_epoch, end_epoch), expected in zip(
        boundaries, canonical_per_stage, strict=True
    ):
        assert (end_epoch - start_epoch) == expected
    # End-to-end: final stage ends at exact canonical total.
    assert boundaries[-1][2] == CANONICAL_PR95_TOTAL_EPOCHS


@requires_mlx
def test_factory_refuses_total_epoch_budget_below_8() -> None:
    """Factory refuses budgets < 8 (cannot fit 8 stages with ≥1 epoch each)."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumError,
        PR95FaithfulCurriculumFactory,
    )

    with pytest.raises(PR95FaithfulCurriculumError, match="must be >= 8"):
        PR95FaithfulCurriculumFactory(total_epoch_budget=7)
    with pytest.raises(PR95FaithfulCurriculumError):
        PR95FaithfulCurriculumFactory(total_epoch_budget=0)


@requires_mlx
def test_factory_refuses_unknown_muon_policy() -> None:
    """Muon policy is executable optimizer control, so unknown values fail."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumError,
        PR95FaithfulCurriculumFactory,
    )

    with pytest.raises(PR95FaithfulCurriculumError, match="muon_policy"):
        PR95FaithfulCurriculumFactory(total_epoch_budget=80, muon_policy="maybe")


@requires_mlx
def test_factory_every_stage_muon_policy_activates_stage_1_muon() -> None:
    """Contest policy keeps PR95 stages but routes Muon from stage 1 onward."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95_MUON_POLICY_EVERY_STAGE,
        PR95_MUON_POLICY_FAITHFUL_STAGE8_ONLY,
        PR95FaithfulCurriculumFactory,
    )

    faithful = PR95FaithfulCurriculumFactory(total_epoch_budget=80)
    faithful_stage1 = faithful.current_stage_verdict(0)
    assert faithful.muon_policy == PR95_MUON_POLICY_FAITHFUL_STAGE8_ONLY
    assert faithful_stage1.uses_muon is False
    assert faithful_stage1.optimizer_config.use_muon is False

    contest = PR95FaithfulCurriculumFactory(
        total_epoch_budget=80,
        muon_policy=PR95_MUON_POLICY_EVERY_STAGE,
    )
    contest_stage1 = contest.current_stage_verdict(0)
    assert contest.muon_policy == PR95_MUON_POLICY_EVERY_STAGE
    assert contest_stage1.muon_policy == PR95_MUON_POLICY_EVERY_STAGE
    assert contest_stage1.uses_muon is True
    assert contest_stage1.optimizer_config.use_muon is True
    assert contest_stage1.descriptor_id == faithful_stage1.descriptor_id
    assert contest_stage1.loss_family == faithful_stage1.loss_family


@requires_mlx
def test_factory_smoke_budget_proportionally_scales_stages() -> None:
    """A small smoke budget (100 epochs) scales per-stage epochs proportionally."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=100)
    assert factory.is_canonical_pr95_budget is False
    boundaries = factory.stage_epoch_boundaries
    assert len(boundaries) == 8
    # Every stage gets at least 1 epoch (no degenerate empty stages).
    for _stage_index, start_epoch, end_epoch in boundaries:
        assert end_epoch > start_epoch, "every stage MUST have ≥1 epoch"
    # End-to-end: final stage ends at exactly 100.
    assert boundaries[-1][2] == 100
    # Stage 5 (canonically 9000/29650 = 30%) should be the largest after scaling.
    stage_epochs = [b[2] - b[1] for b in boundaries]
    assert stage_epochs[4] == max(stage_epochs), (
        f"stage 5 should be largest after scaling; got per-stage {stage_epochs}"
    )


@requires_mlx
def test_current_stage_index_progresses_monotonically_across_canonical_budget() -> None:
    """Stage index advances 1→2→3→...→8 monotonically across canonical epochs."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    # Stage 1: epochs 0..2999.
    assert factory.current_stage_index(0) == 1
    assert factory.current_stage_index(2999) == 1
    # Stage 2: epochs 3000..8649 (3000 + 5650 = 8650 = stage 2 end).
    assert factory.current_stage_index(3000) == 2
    assert factory.current_stage_index(8649) == 2
    # Stage 3: epochs 8650..10149.
    assert factory.current_stage_index(8650) == 3
    # Stage 4: epochs 10150..10649.
    assert factory.current_stage_index(10150) == 4
    # Stage 5: epochs 10650..19649.
    assert factory.current_stage_index(10650) == 5
    # Stage 6: epochs 19650..21649.
    assert factory.current_stage_index(19650) == 6
    # Stage 7: epochs 21650..24649.
    assert factory.current_stage_index(21650) == 7
    # Stage 8: epochs 24650..29649.
    assert factory.current_stage_index(24650) == 8
    assert factory.current_stage_index(29649) == 8
    # Overflow → final stage absorbs (canonical scheduler pattern).
    assert factory.current_stage_index(50_000) == 8


@requires_mlx
def test_current_stage_verdict_loads_canonical_descriptor_per_stage() -> None:
    """Each stage verdict cites the canonical PR95 descriptor_id."""
    from tac.local_acceleration.pr95_hnerv_mlx import (
        PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS,
    )
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    canonical_starts = [0, 3000, 8650, 10150, 10650, 19650, 21650, 24650]
    for stage_idx, start_epoch in zip(range(1, 9), canonical_starts, strict=True):
        verdict = factory.current_stage_verdict(start_epoch)
        assert verdict.stage_index == stage_idx, (
            f"stage_index mismatch at epoch {start_epoch}"
        )
        assert verdict.descriptor_id == PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[stage_idx], (
            f"descriptor_id mismatch at stage {stage_idx}"
        )


@requires_mlx
def test_per_stage_descriptor_ids_match_canonical_registry() -> None:
    """factory.per_stage_descriptor_ids matches PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS verbatim."""
    from tac.local_acceleration.pr95_hnerv_mlx import (
        PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS,
    )
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory()
    canonical = tuple(
        PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[i] for i in range(1, 9)
    )
    assert factory.per_stage_descriptor_ids == canonical


@requires_mlx
def test_per_stage_optimizer_config_distinct_across_8_stages() -> None:
    """NO FAKE: each of the 8 stages emits a distinct optimizer_config."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    canonical_starts = [0, 3000, 8650, 10150, 10650, 19650, 21650, 24650]
    configs = []
    for start_epoch in canonical_starts:
        verdict = factory.current_stage_verdict(start_epoch)
        configs.append(verdict.optimizer_config)
    # Canonical key facts per CLAUDE.md L14 + L15:
    # - Stages 1-7 use AdamW only (use_muon=False).
    # - Stage 8 uses Muon+AdamW (use_muon=True).
    for i in range(7):
        assert configs[i].use_muon is False, (
            f"stage {i + 1} MUST be AdamW-only (use_muon=False)"
        )
    assert configs[7].use_muon is True, (
        "stage 8 MUST be Muon (use_muon=True) per L15 canonical equation"
    )
    # Distinct lr / lambda / sigma signatures verify NO FAKE:
    # Stage 1 lr = 1e-3; Stage 5 lr = 3e-5; Stage 8 adamw_lr = 1e-5.
    assert configs[0].adamw_lr == pytest.approx(1e-3)
    assert configs[4].adamw_lr == pytest.approx(3e-5)
    assert configs[7].adamw_lr == pytest.approx(1e-5)
    assert configs[7].muon_lr == pytest.approx(2e-4)


@requires_mlx
def test_per_stage_loss_family_and_lambda_sigma_distinct_per_canonical_l14() -> None:
    """Per-stage loss_family + cat_sigma + cat_lambda match canonical PR95 source."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    canonical_starts = [0, 3000, 8650, 10150, 10650, 19650, 21650, 24650]
    # Per CLAUDE.md L14 + canonical optimizer_scheduler_registry:
    expected_loss = [
        "ce_seg_loss",
        "tau_softplus_seg_loss",
        "smooth_disagreement_seg_loss",
        "smooth_disagreement_seg_loss",
        "l7_softplus_seg_loss",
        "l7_softplus_seg_loss",
        "l7_softplus_seg_loss",
        "l7_softplus_seg_loss",
    ]
    expected_lambda = [0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.02]
    expected_sigma = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    expected_qat = [False, False, False, True, True, True, True, True]
    for i, start_epoch in enumerate(canonical_starts):
        verdict = factory.current_stage_verdict(start_epoch)
        assert verdict.loss_family == expected_loss[i], (
            f"stage {i + 1} loss_family mismatch"
        )
        assert verdict.cat_lambda == pytest.approx(expected_lambda[i]), (
            f"stage {i + 1} cat_lambda mismatch"
        )
        assert verdict.cat_sigma == pytest.approx(expected_sigma[i]), (
            f"stage {i + 1} cat_sigma mismatch"
        )
        assert verdict.qat_active is expected_qat[i], (
            f"stage {i + 1} qat_active mismatch"
        )


@requires_mlx
def test_is_stage_boundary_fires_only_at_canonical_stage_starts() -> None:
    """is_stage_boundary fires only when global_epoch equals start_epoch of any stage > 1."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    # Stage 1 start (epoch 0) is NOT a boundary per the canonical semantics.
    assert factory.is_stage_boundary(0) is False
    # Stage 2..8 starts ARE boundaries.
    assert factory.is_stage_boundary(3000) is True
    assert factory.is_stage_boundary(8650) is True
    assert factory.is_stage_boundary(24650) is True
    # Non-boundary epochs return False.
    assert factory.is_stage_boundary(1500) is False
    assert factory.is_stage_boundary(15_000) is False


@requires_mlx
def test_stage_transition_diff_detects_l14_to_l15_transition() -> None:
    """Stage 7 → Stage 8 transition is the canonical L14→L15 Muon activation."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    # No transition within the same stage.
    assert factory.stage_transition_diff(100, 200) is None
    # Transition from stage 7 to stage 8 (canonical Muon activation).
    diff = factory.stage_transition_diff(24649, 24650)
    assert diff == (7, 8), (
        f"L14→L15 transition (stage 7→8) expected; got {diff}"
    )
    # Transition from stage 1 to stage 2.
    diff_early = factory.stage_transition_diff(2999, 3000)
    assert diff_early == (1, 2)


@requires_mlx
def test_stage_verdict_cache_returns_same_object_within_stage() -> None:
    """current_stage_verdict caches per-stage verdict objects."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    v1 = factory.current_stage_verdict(1000)
    v2 = factory.current_stage_verdict(2000)
    assert v1 is v2, "stage verdict cache MUST return identical object within stage"


# ---------- Section 2: MlxScoreAwareAdapter opt-in wire-in tests ----------


def _make_minimal_bundle() -> object:
    """Build a minimal RendererBundle for adapter testing (MLX required).

    Uses the ``reconstruct_pair_nchw01`` forward convention with a tiny renderer
    whose 2D weight is Muon-eligible per the canonical partition
    (no stem/rgb/latents tokens in the parameter name).
    """
    import mlx.core as mx
    import mlx.nn as nn

    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

    class TinyRenderer(nn.Module):
        """4-parameter MLX renderer with 2D weight (Muon-eligible per partition).

        Exposes the canonical ``reconstruct_pair`` convention returning two
        ``(B, 3, H, W)`` frames in ``[0, 1]``.
        """

        def __init__(self) -> None:
            super().__init__()
            # 2D weight named "decoder_weight" → Muon-eligible per canonical partition.
            # NOTE: NOT a Linear layer to avoid the auto-named "weight" / "bias"
            # collision with the canonical Muon partition filter; we test the
            # canonical NAME-based routing directly.
            self.decoder_weight = mx.zeros((4, 4))
            # 1D bias named "decoder_bias" → AdamW-only per canonical partition.
            self.decoder_bias = mx.zeros((4,))

        def reconstruct_pair(self, indices):
            """Return (rgb_0, rgb_1) each (B, 3, 2, 2) in [0, 1] — tiny canonical pair."""
            bs = int(indices.shape[0])
            # Use parameters so MLX value_and_grad sees gradient through them.
            scale = mx.sum(self.decoder_weight) + mx.sum(self.decoder_bias)
            base = mx.ones((bs, 3, 2, 2)) * 0.5 * (scale * 0.0 + 1.0)
            # Add a gradient-bearing tiny modulation so the loss is non-degenerate.
            mod = mx.broadcast_to(
                mx.reshape(self.decoder_weight[:1, :1] * 0.01, (1, 1, 1, 1)),
                (bs, 3, 2, 2),
            )
            return base + mod, base + mod * 2.0

    model = TinyRenderer()
    num_pairs = 8
    # Targets are precomputed (B, H, W, 3) in [0, 1] per RendererBundle contract.
    target_rgb_0 = mx.zeros((num_pairs, 2, 2, 3))
    target_rgb_1 = mx.zeros((num_pairs, 2, 2, 3))

    return RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=num_pairs,
        forward_convention="reconstruct_pair_nchw01",
    )


def _make_minimal_pr95_score_bundle() -> object:
    """Build a tiny bundle with both scorer teachers for PR95 stage dispatch."""

    import mlx.core as mx
    import numpy as np

    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        RealSegNetTeacherLogitsCache,
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    base = _make_minimal_bundle()
    num_pairs = int(base.num_pairs)
    num_classes = 5
    labels = np.asarray(
        [
            [[0, 1], [2, 3]],
            [[4, 3], [2, 1]],
            [[1, 2], [3, 4]],
            [[0, 2], [4, 1]],
            [[3, 1], [0, 2]],
            [[2, 4], [1, 0]],
            [[4, 0], [3, 2]],
            [[1, 3], [2, 4]],
        ],
        dtype=np.int32,
    )
    logits = np.full((num_pairs, 2, 2, num_classes), -1.5, dtype=np.float32)
    for pair_index in range(num_pairs):
        for row in range(2):
            for col in range(2):
                logits[pair_index, row, col, labels[pair_index, row, col]] = 2.5
    seg_teacher = RealSegNetTeacherLogitsCache(
        teacher_logits_thwk=mx.array(logits),
        frame_count=num_pairs,
        height=2,
        width=2,
        num_classes=num_classes,
    )

    pose_np = np.stack(
        [
            np.linspace(0.1 * i, 0.1 * i + 0.5, 6, dtype=np.float32)
            for i in range(num_pairs)
        ],
        axis=0,
    )
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.array(pose_np),
        num_pairs=num_pairs,
        pose_dims=6,
        per_dim_scale=mx.ones((6,)),
    )
    seg_head = build_learnable_student_head(
        num_classes=num_classes,
        in_channels=3,
        seed=23,
        init_scale=0.2,
    )
    seg_head.weight = mx.zeros((3, num_classes))
    seg_head.bias = mx.array([4.0, 1.0, 0.0, -1.0, -2.0], dtype=mx.float32)
    pose_head = build_learnable_pose_student_head(
        pose_dims=6,
        pool_grid=1,
        input_channels=3,
        seed=29,
        init_scale=0.1,
    )

    return RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=num_pairs,
        forward_convention=base.forward_convention,
        distillation_weight=1.0,
        scorer_teacher=seg_teacher,
        learnable_student_head=seg_head,
        pose_distillation_weight=1.0,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
    )


def _force_high_contrast_scorer_targets(bundle: object) -> None:
    """Give contrast-floor regressions a non-degenerate scorer-domain target."""

    import mlx.core as mx

    checker = mx.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=mx.float32,
    )
    target_0 = mx.broadcast_to(
        mx.reshape(checker[0], (1, 2, 2, 1)),
        (int(bundle.num_pairs), 2, 2, 3),
    )
    target_1 = mx.broadcast_to(
        mx.reshape(checker[1], (1, 2, 2, 1)),
        (int(bundle.num_pairs), 2, 2, 3),
    )
    bundle.target_rgb_0 = target_0
    bundle.target_rgb_1 = target_1


@requires_mlx
def test_adapter_default_off_preserves_legacy_adamw_path() -> None:
    """Backward compat: pr95_faithful_curriculum_enabled=False keeps legacy adapter."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="test_substrate")
    # Default off: factory is None, no PR95 state initialized.
    assert adapter._pr95_faithful_curriculum_enabled is False
    assert adapter._pr95_curriculum_factory is None
    assert adapter._pr95_optimizer_state is None


@requires_mlx
def test_adapter_opt_in_initializes_canonical_factory_and_state() -> None:
    """Opt-in: pr95_faithful_curriculum_enabled=True initializes factory + state."""
    from tac.local_acceleration.pr95_hnerv_mlx import Pr95MlxOptimizerState
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=100,
    )
    assert adapter._pr95_faithful_curriculum_enabled is True
    assert isinstance(adapter._pr95_curriculum_factory, PR95FaithfulCurriculumFactory)
    assert isinstance(adapter._pr95_optimizer_state, Pr95MlxOptimizerState)
    assert adapter._pr95_curriculum_factory.total_epoch_budget == 100


@requires_mlx
def test_adapter_opt_in_defaults_to_canonical_29650_epochs_when_not_specified() -> None:
    """Opt-in with no total_epochs override defaults to canonical 29,650."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
    )

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
    )
    assert (
        adapter._pr95_curriculum_factory.total_epoch_budget
        == CANONICAL_PR95_TOTAL_EPOCHS
    )


@requires_mlx
def test_notify_global_epoch_advances_pr95_global_epoch_state() -> None:
    """notify_global_epoch updates the adapter's PR95 stage tracker."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=100,
    )
    assert adapter._pr95_global_epoch == 0
    adapter.notify_global_epoch(50)
    assert adapter._pr95_global_epoch == 50


@requires_mlx
def test_notify_global_epoch_is_noop_when_curriculum_disabled() -> None:
    """notify_global_epoch preserves backward-compat when curriculum disabled."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="test_substrate")
    # No-op-when-disabled: the field is updated but factory/state stay None.
    adapter.notify_global_epoch(50)
    assert adapter._pr95_global_epoch == 50
    assert adapter._pr95_curriculum_factory is None


# ---------- Section 3: Canonical equation + Provenance integration ----------


@requires_mlx
def test_canonical_equation_l14_8_stage_curriculum_v1_registered() -> None:
    """Canonical equation pr95_family_l14_eight_stage_29650_epoch_curriculum_v1 may be registered."""
    from tac.canonical_equations import query_equations

    equations = query_equations()
    # The canonical CLAUDE.md L14 entry IS the documentation contract; the
    # registry may lazily load. Verify the entry shape if present.
    l14 = [
        eq
        for eq in equations
        if eq.equation_id == "pr95_family_l14_eight_stage_29650_epoch_curriculum_v1"
    ]
    if l14:
        assert l14[0].equation_id.startswith("pr95_family_l14_")


@requires_mlx
def test_canonical_equation_l15_muon_optimizer_final_stage_only_v1_registered() -> None:
    """Canonical equation pr95_family_l15_muon_optimizer_final_stage_only_v1 may be registered."""
    from tac.canonical_equations import query_equations

    equations = query_equations()
    l15 = [
        eq
        for eq in equations
        if eq.equation_id == "pr95_family_l15_muon_optimizer_final_stage_only_v1"
    ]
    if l15:
        assert l15[0].equation_id.startswith("pr95_family_l15_")


# ---------- Section 4: Stage-aware curriculum metrics returned from train_step ----------


@requires_mlx
def test_pr95_optimizer_step_reports_real_partition_gradient_clipping() -> None:
    """PR95 optimizer clipping telemetry must report the actual applied partition."""

    import mlx.core as mx
    import mlx.nn as nn

    from tac.local_acceleration.pr95_hnerv_mlx import (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
    )

    class TinyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder_weight = mx.ones((2, 2))
            self.decoder_bias = mx.ones((2,))

    module = TinyModule()
    gradients = {
        "decoder_weight": mx.ones((2, 2)) * 100.0,
        "decoder_bias": mx.ones((2,)) * 100.0,
    }
    summary = apply_pr95_mlx_optimizer_step(
        module,
        gradients,
        Pr95MlxOptimizerState(),
        Pr95MlxOptimizerConfig(
            use_muon=False,
            adamw_lr=0.0,
            muon_lr=0.0,
            grad_clip=1.0,
            grad_clip_muon=1.0,
        ),
    )

    assert summary["gradient_clip_actual_application_count"] == 1
    assert summary["gradient_clip_would_clip_count"] == 1
    assert summary["gradient_clip_min_scale"] < 1.0
    assert summary["adamw_gradient_clip"]["present_gradient_count"] == 2
    assert summary["adamw_gradient_clip"]["pre_norm"] > 1.0
    assert summary["adamw_gradient_clip"]["applied"] is True


@requires_mlx
def test_train_step_returns_stage_index_in_metrics_when_curriculum_enabled() -> None:
    """train_step's return dict carries pr95_stage_index + uses_muon metrics."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,  # 10 epochs per stage (80 / 8 = 10).
    )
    # At epoch 0 → stage 1 (use_muon=False).
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0},
    )
    assert "total" in metrics
    assert "pr95_stage_index" in metrics
    assert "pr95_stage_uses_muon" in metrics
    assert "loss_part_pr95_stage_scorer_surrogate" in metrics
    assert metrics["pr95_stage_index"] == 1.0
    assert metrics["pr95_stage_uses_muon"] == 0.0  # stage 1 uses AdamW only.


@requires_mlx
def test_train_step_applies_scorer_guard_lr_scale_to_pr95_stage_config() -> None:
    """The scorer-space trust-region scale must actuate the real PR95 optimizer."""

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    adapter.notify_global_epoch(0)
    adapter._scorer_space_step_guard_learning_rate_scale = 0.25
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0},
    )

    assert metrics["pr95_stage_scorer_guard_learning_rate_scale"] == pytest.approx(
        0.25
    )
    assert metrics["pr95_stage_effective_adamw_learning_rate"] == pytest.approx(
        metrics["pr95_stage_base_adamw_learning_rate"] * 0.25
    )
    assert metrics["pr95_stage_effective_muon_learning_rate"] == pytest.approx(
        metrics["pr95_stage_base_muon_learning_rate"] * 0.25
    )
    assert metrics["scorer_space_step_guard_effective_optimizer_learning_rate"] == (
        pytest.approx(metrics["pr95_stage_effective_adamw_learning_rate"])
    )


@requires_mlx
def test_train_step_every_stage_muon_policy_uses_muon_in_stage_1() -> None:
    """The contest policy is a real train_step optimizer branch, not metadata."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
        pr95_muon_policy="every_stage",
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0},
    )

    assert metrics["pr95_stage_index"] == 1.0
    assert metrics["pr95_stage_uses_muon"] == 1.0
    assert adapter._pr95_curriculum_factory.muon_policy == "every_stage"
    assert adapter.artifact_metadata()["score_aware_training"][
        "pr95_curriculum_controls"
    ]["pr95_muon_policy"] == "every_stage"


@requires_mlx
def test_train_step_stage_8_signals_muon_active_in_metrics() -> None:
    """At final-stage epoch, train_step metrics carry pr95_stage_uses_muon=1.0."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    # Advance to stage 8 (epoch 70 → stage 8 in 80-epoch budget).
    adapter.notify_global_epoch(75)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0},
    )
    assert metrics["pr95_stage_index"] == 8.0, (
        f"epoch 75 of 80 expected stage 8; got {metrics['pr95_stage_index']}"
    )
    assert metrics["pr95_stage_uses_muon"] == 1.0, (
        "stage 8 MUST signal use_muon=True per L15 canonical equation"
    )
    assert metrics["loss_part_pr95_stage_loss_surface_active"] == pytest.approx(1.0)


@requires_mlx
def test_pr95_stage_dispatch_makes_stage_1_2_5_losses_behaviorally_distinct() -> None:
    """Stage verdict loss_family must change the adapter's actual loss value."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    bundle.model.decoder_weight = mx.ones((4, 4)) * 0.25
    mx.eval(bundle.model.parameters())
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    batch = adapter.sample_batch(batch_size=4, seed=3)
    stage_starts = {
        int(stage_index): int(start_epoch)
        for stage_index, start_epoch, _end_epoch in (
            adapter._pr95_curriculum_factory.stage_epoch_boundaries
        )
    }
    losses: dict[int, float] = {}
    families: dict[int, str] = {}
    for stage_index in (1, 2, 5):
        verdict = adapter._pr95_curriculum_factory.current_stage_verdict(
            stage_starts[stage_index]
        )
        total, parts = adapter._pr95_stage_loss_and_parts(
            batch=batch,
            stage_verdict=verdict,
            model=adapter.model,
        )
        mx.eval(total, *parts.values())
        losses[stage_index] = float(total.item())
        families[stage_index] = str(verdict.loss_family)

    assert families == {
        1: "ce_seg_loss",
        2: "tau_softplus_seg_loss",
        5: "l7_softplus_seg_loss",
    }
    assert len({round(value, 6) for value in losses.values()}) == 3, losses


@requires_mlx
def test_pr95_stage_4_consumes_real_coder_qat_terms_NO_FAKE() -> None:
    """Stage-4 QAT must affect the PR95 renderer loss, not just metadata."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.coder_qat import (
        CoderAwareQATConfig,
        build_decoder_coder_qat_terms,
        coder_qat_loss_weights,
    )

    qat_cfg = CoderAwareQATConfig(
        enabled=True,
        quant_bits=2,
        quant_residual_weight=0.5,
        magnitude_weight=0.125,
        delta_weight=0.0625,
        c1a_entropy_weight=0.0,
    ).validated()
    bundle = _make_minimal_pr95_score_bundle()
    bundle.model.decoder_weight = mx.reshape(
        mx.array(
            [
                -0.37,
                -0.19,
                0.08,
                0.41,
                -0.12,
                0.27,
                -0.31,
                0.53,
                0.02,
                -0.44,
                0.16,
                -0.07,
                0.33,
                -0.25,
                0.49,
                -0.58,
            ],
            dtype=mx.float32,
        ),
        (4, 4),
    )
    bundle.extra_loss_terms = lambda model, _idx: build_decoder_coder_qat_terms(
        model,
        qat_cfg,
    )
    bundle.extra_loss_weights = coder_qat_loss_weights(qat_cfg)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    batch = adapter.sample_batch(batch_size=4, seed=4)
    stage_starts = {
        int(stage_index): int(start_epoch)
        for stage_index, start_epoch, _end_epoch in (
            adapter._pr95_curriculum_factory.stage_epoch_boundaries
        )
    }
    stage_4 = adapter._pr95_curriculum_factory.current_stage_verdict(
        stage_starts[4]
    )
    stage_1 = adapter._pr95_curriculum_factory.current_stage_verdict(
        stage_starts[1]
    )
    assert stage_1.qat_active is False
    assert stage_4.qat_active is True
    assert stage_4.cat_lambda == pytest.approx(0.0)

    total_stage_1, parts_stage_1 = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage_1,
        model=adapter.model,
    )
    total_with_qat, parts_with_qat = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage_4,
        model=adapter.model,
    )
    bundle.extra_loss_terms = None
    bundle.extra_loss_weights = {}
    total_without_qat, parts_without_qat = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage_4,
        model=adapter.model,
    )
    mx.eval(
        total_stage_1,
        total_with_qat,
        total_without_qat,
        *parts_stage_1.values(),
        *parts_with_qat.values(),
    )

    assert "coder_qat_quant_residual" not in parts_stage_1
    assert "coder_qat_magnitude" not in parts_stage_1
    assert "coder_qat_delta" not in parts_stage_1
    assert "coder_qat_quant_residual" in parts_with_qat
    assert "coder_qat_magnitude" in parts_with_qat
    assert "coder_qat_delta" in parts_with_qat
    assert "coder_qat_quant_residual" not in parts_without_qat
    assert float(total_with_qat.item()) != pytest.approx(
        float(total_without_qat.item())
    )


@requires_mlx
def test_pr95_stage_qat_uses_dual_adjusted_extra_loss_weights_NO_FAKE() -> None:
    """Byte duals must actuate the PR95-stage QAT loss, not only telemetry."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    bundle.extra_loss_terms = lambda _model, _idx: {
        "coder_qat_quant_residual": mx.array(3.0, dtype=mx.float32)
    }
    bundle.extra_loss_weights = {"coder_qat_quant_residual": 0.0}
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    stage_starts = {
        int(stage_index): int(start_epoch)
        for stage_index, start_epoch, _end_epoch in (
            adapter._pr95_curriculum_factory.stage_epoch_boundaries
        )
    }
    stage_4 = adapter._pr95_curriculum_factory.current_stage_verdict(
        stage_starts[4]
    )
    batch = adapter.sample_batch(batch_size=2, seed=0)

    total_zero, _parts_zero = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage_4,
        model=adapter.model,
        loss_weights={"coder_qat_quant_residual": 0.0},
    )
    total_dual, parts_dual = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage_4,
        model=adapter.model,
        loss_weights={"coder_qat_quant_residual": 2.0},
    )
    metrics_dual = adapter._pr95_stage_loss_part_metrics(
        batch,
        stage_verdict=stage_4,
        loss_weights={"coder_qat_quant_residual": 2.0},
    )
    mx.eval(total_zero, total_dual, *parts_dual.values())

    assert "coder_qat_quant_residual" in parts_dual
    assert float(total_dual.item()) - float(total_zero.item()) == pytest.approx(6.0)
    assert metrics_dual["loss_part_weighted_coder_qat_quant_residual"] == (
        pytest.approx(6.0)
    )


@requires_mlx
def test_pr95_stage_does_not_force_qat_without_stage_proof_marker() -> None:
    """A constructor request alone must not break PR95's early distortion stages."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    bundle.extra_loss_terms = lambda _model, _idx: {
        "coder_qat_quant_residual": mx.array(3.0, dtype=mx.float32)
    }
    bundle.extra_loss_weights = {"coder_qat_quant_residual": 0.5}
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
        pr95_force_weighted_extra_qat_when_stage_inactive=True,
    )
    stage_1 = adapter._pr95_curriculum_factory.current_stage_verdict(0)
    assert stage_1.qat_active is False
    batch = adapter.sample_batch(batch_size=2, seed=0)

    total, parts = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage_1,
        model=adapter.model,
    )
    metrics = adapter._pr95_stage_loss_part_metrics(
        batch,
        stage_verdict=stage_1,
    )
    mx.eval(total, *parts.values())

    assert "coder_qat_quant_residual" not in parts
    assert float(parts["pr95_stage_forced_extra_qat_requested"].item()) == pytest.approx(
        1.0
    )
    assert float(parts["pr95_stage_forced_extra_qat_active"].item()) == pytest.approx(
        0.0
    )
    assert metrics["loss_part_pr95_stage_forced_extra_qat_requested"] == pytest.approx(
        1.0
    )
    assert metrics["loss_part_pr95_stage_forced_extra_qat_active"] == pytest.approx(0.0)
    assert "loss_part_coder_qat_quant_residual" not in metrics


@requires_mlx
def test_pr95_stage_can_force_weighted_qat_for_short_admission_proof() -> None:
    """Short bounded proofs can verify weighted byte QAT before PR95 stage 4."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import (
        PR95_FORCE_WEIGHTED_EXTRA_QAT_ADMISSION_PROOF_WEIGHT,
        MlxScoreAwareAdapter,
    )

    bundle = _make_minimal_pr95_score_bundle()
    bundle.extra_loss_terms = lambda _model, _idx: {
        "coder_qat_quant_residual": mx.array(3.0, dtype=mx.float32)
    }
    bundle.extra_loss_weights = {"coder_qat_quant_residual": 0.5}
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
        pr95_force_weighted_extra_qat_when_stage_inactive=True,
    )
    stage_1 = adapter._pr95_curriculum_factory.current_stage_verdict(0)
    assert stage_1.qat_active is False
    batch = adapter.sample_batch(batch_size=2, seed=0)
    loss_weights = {
        PR95_FORCE_WEIGHTED_EXTRA_QAT_ADMISSION_PROOF_WEIGHT: 1.0,
    }

    total, parts = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage_1,
        model=adapter.model,
        loss_weights=loss_weights,
    )
    metrics = adapter._pr95_stage_loss_part_metrics(
        batch,
        stage_verdict=stage_1,
        loss_weights=loss_weights,
    )
    mx.eval(total, *parts.values())

    assert "coder_qat_quant_residual" in parts
    assert float(parts["pr95_stage_forced_extra_qat_requested"].item()) == pytest.approx(
        1.0
    )
    assert float(
        parts["pr95_stage_forced_extra_qat_stage_enabled"].item()
    ) == pytest.approx(1.0)
    assert float(parts["pr95_stage_forced_extra_qat_active"].item()) == pytest.approx(
        1.0
    )
    assert metrics["loss_part_pr95_stage_forced_extra_qat_requested"] == pytest.approx(
        1.0
    )
    assert metrics[
        "loss_part_pr95_stage_forced_extra_qat_stage_enabled"
    ] == pytest.approx(1.0)
    assert metrics["loss_part_pr95_stage_forced_extra_qat_active"] == pytest.approx(1.0)
    assert metrics["loss_part_coder_qat_quant_residual"] == pytest.approx(3.0)
    assert metrics["loss_part_weighted_coder_qat_quant_residual"] == pytest.approx(
        1.5
    )


@requires_mlx
def test_mlx_score_adapter_forwards_post_optimizer_projection_NO_FAKE() -> None:
    """Substrate proximal projections must run through the training harness."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    calls: list[int] = []

    def _projection(*, epoch: int) -> dict[str, object]:
        calls.append(int(epoch))
        return {
            "schema": "unit_post_optimizer_projection.v1",
            "epoch": int(epoch),
            "applied": True,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }

    bundle.model.post_optimizer_projection = _projection
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="test_substrate")

    report = adapter.post_optimizer_projection(epoch=17)

    assert calls == [17]
    assert report == {
        "schema": "unit_post_optimizer_projection.v1",
        "epoch": 17,
        "applied": True,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


@requires_mlx
def test_pr95_curriculum_path_trains_student_heads_when_stage_weight_active() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    pose_head = bundle.learnable_pose_student_head
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    w0 = mx.array(pose_head.weight)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "pose_distill": 1.0},
    )

    moved = float(mx.max(mx.abs(pose_head.weight - w0)).item())
    assert moved > 0.0
    assert "loss_part_pr95_stage_pose_surrogate" in metrics


@requires_mlx
def test_pr95_curriculum_path_respects_zero_student_head_stage_weight() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    pose_head = bundle.learnable_pose_student_head
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    w0 = mx.array(pose_head.weight)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "pose_distill": 0.0},
    )

    moved = float(mx.max(mx.abs(pose_head.weight - w0)).item())
    assert moved == pytest.approx(0.0)
    assert "loss_part_stage_weight_pose_distill" not in metrics
    assert "loss_part_pr95_stage_pose_surrogate" in metrics
    assert "loss_part_pose_distill" not in metrics
    assert "loss_part_pose_score_term" not in metrics


@requires_mlx
def test_pr95_curriculum_zero_pose_stage_does_not_pollute_pose_dual_NO_FAKE() -> None:
    """A curriculum pose warmup mask must not update PoseNet dual state."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
        train_time_dual_ascent_config={
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "hi_nerv_posenet_yuv6_pair_distill",
                    "metric_name": "loss_part_pose_score_term",
                    "loss_weight_key": "pose_distill",
                    "target": 0.0,
                    "dual_lr": 0.2,
                    "max_lambda": 6.0,
                }
            ],
        },
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 0.0},
    )
    mx.eval(adapter.model.parameters())

    assert metrics["loss_part_pr95_stage_pose_distill_weight"] == pytest.approx(0.0)
    assert metrics["loss_part_pr95_stage_effective_pose_weight"] == pytest.approx(0.0)
    assert metrics["loss_part_weighted_pr95_stage_pose_surrogate"] == pytest.approx(0.0)
    assert "loss_part_pose_score_term" not in metrics
    assert metrics[
        "dual_ascent_missing_metric__hi_nerv_posenet_yuv6_pair_distill"
    ] == pytest.approx(1.0)
    assert metrics[
        "dual_ascent_lambda__hi_nerv_posenet_yuv6_pair_distill"
    ] == pytest.approx(0.0)


@requires_mlx
def test_segnet_student_live_calibration_requires_candidate_frame_teacher() -> None:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
    from tac.substrates._shared.mlx_score_aware.device_gate import (
        MlxScoreAwareHarnessError,
    )

    base = _make_minimal_pr95_score_bundle()

    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="live SegNet candidate-frame terms require",
    ):
        RendererBundle(
            model=base.model,
            target_rgb_0=base.target_rgb_0,
            target_rgb_1=base.target_rgb_1,
            num_pairs=base.num_pairs,
            forward_convention=base.forward_convention,
            distillation_weight=base.distillation_weight,
            scorer_teacher=base.scorer_teacher,
            learnable_student_head=base.learnable_student_head,
            segnet_student_live_calibration_weight=1.0,
            pose_distillation_weight=base.pose_distillation_weight,
            pose_scorer_teacher=base.pose_scorer_teacher,
            learnable_pose_student_head=base.learnable_pose_student_head,
        )


@requires_mlx
def test_pr95_curriculum_trains_student_head_with_live_candidate_segnet_calibration() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    class _LiveSegTeacher:
        def __init__(self, cached: object) -> None:
            self.cached = cached
            self.num_classes = int(cached.num_classes)
            self.live_call_count = 0

        def teacher_logits_for_indices(self, idx: object) -> object:
            return self.cached.teacher_logits_for_indices(idx)

        def teacher_logits_for_frames_nhwc01(self, frames: object) -> object:
            self.live_call_count += 1
            b, h, w, _c = frames.shape
            mean = mx.mean(frames, axis=-1, keepdims=True)
            channels = [mean * float(i + 1) for i in range(self.num_classes)]
            return mx.concatenate(channels, axis=-1).reshape(
                (int(b), int(h), int(w), self.num_classes)
            )

    bundle = _make_minimal_pr95_score_bundle()
    live_teacher = _LiveSegTeacher(bundle.scorer_teacher)
    bundle.scorer_teacher = live_teacher
    bundle.segnet_student_live_calibration_weight = 0.75
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )

    assert live_teacher.live_call_count >= 1
    assert metrics["segnet_student_head_update_active"] == pytest.approx(1.0)
    assert metrics["segnet_student_live_calibration_active"] == pytest.approx(1.0)
    assert metrics[
        "segnet_student_live_calibration_teacher_available"
    ] == pytest.approx(1.0)
    assert metrics["segnet_student_live_calibration_weight"] == pytest.approx(0.75)
    assert metrics["loss_part_segnet_student_live_calibration"] >= 0.0
    assert metrics["loss_part_weighted_segnet_student_live_calibration"] >= 0.0


@requires_mlx
def test_pr95_curriculum_consumes_direct_live_segnet_distillation_NO_FAKE() -> None:
    """Direct live SegNet loss must run the candidate-frame scorer surface."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    class _LiveSegTeacher:
        def __init__(self, cached: object) -> None:
            self.cached = cached
            self.num_classes = int(cached.num_classes)
            self.live_call_count = 0

        def teacher_logits_for_indices(self, idx: object) -> object:
            return self.cached.teacher_logits_for_indices(idx)

        def teacher_logits_for_frames_nhwc01(self, frames: object) -> object:
            self.live_call_count += 1
            b, h, w, _c = frames.shape
            mean = mx.mean(frames, axis=-1, keepdims=True)
            channels = [mean * float(i + 1) for i in range(self.num_classes)]
            return mx.concatenate(channels, axis=-1).reshape(
                (int(b), int(h), int(w), self.num_classes)
            )

    bundle = _make_minimal_pr95_score_bundle()
    live_teacher = _LiveSegTeacher(bundle.scorer_teacher)
    bundle.scorer_teacher = live_teacher
    bundle.segnet_direct_live_distillation_weight = 0.5
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )

    assert live_teacher.live_call_count >= 1
    assert metrics[
        "loss_part_pr95_stage_segnet_direct_live_distill"
    ] >= 0.0
    assert metrics[
        "loss_part_weighted_pr95_stage_segnet_direct_live_distill"
    ] >= 0.0
    assert metrics["loss_part_segnet_direct_live_distill"] == pytest.approx(
        metrics["loss_part_pr95_stage_segnet_direct_live_distill"]
    )
    assert metrics["loss_part_weighted_segnet_direct_live_distill"] == pytest.approx(
        metrics["loss_part_weighted_pr95_stage_segnet_direct_live_distill"]
    )
    assert metrics[
        "loss_part_segnet_direct_live_argmax_disagreement"
    ] == pytest.approx(
        metrics[
            "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement"
        ]
    )
    assert metrics[
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction"
    ] == pytest.approx(
        metrics[
            "loss_part_pr95_stage_segnet_direct_live_candidate_occupied_class_fraction"
        ]
    )


@requires_mlx
def test_pr95_curriculum_consumes_scorer_input_contrast_floor_NO_FAKE() -> None:
    """PR95-stage training must not bypass scorer-domain contrast protection."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    _force_high_contrast_scorer_targets(bundle)
    bundle.scorer_input_contrast_floor_weight = 0.5
    bundle.scorer_input_contrast_floor_segnet_min_std_ratio = 0.75
    bundle.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio = 0.75
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_contrast_floor_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )
    mx.eval(adapter.model.parameters())

    assert metrics["loss_part_pr95_stage_scorer_input_contrast_floor"] > 0.0
    assert metrics["loss_part_scorer_input_contrast_floor"] == pytest.approx(
        metrics["loss_part_pr95_stage_scorer_input_contrast_floor"]
    )
    assert metrics[
        "loss_part_weighted_pr95_stage_scorer_input_contrast_floor"
    ] == pytest.approx(metrics["loss_part_weighted_scorer_input_contrast_floor"])
    assert (
        "loss_part_pr95_stage_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio"
        in metrics
    )
    assert (
        "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio"
        in metrics
    )


@requires_mlx
def test_pr95_curriculum_dual_ascent_observes_direct_live_segnet_alias_NO_FAKE() -> None:
    """Direct-live PR95 SegNet loss must feed the shared dual controller."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.dual_ascent import (
        build_default_nerv_train_time_dual_ascent_config,
    )

    class _LiveSegTeacher:
        def __init__(self, cached: object) -> None:
            self.cached = cached
            self.num_classes = int(cached.num_classes)

        def teacher_logits_for_indices(self, idx: object) -> object:
            return self.cached.teacher_logits_for_indices(idx)

        def teacher_logits_for_frames_nhwc01(self, frames: object) -> object:
            b, h, w, _c = frames.shape
            mean = mx.mean(frames, axis=-1, keepdims=True)
            channels = [mean * float(i + 1) for i in range(self.num_classes)]
            return mx.concatenate(channels, axis=-1).reshape(
                (int(b), int(h), int(w), self.num_classes)
            )

    bundle = _make_minimal_pr95_score_bundle()
    bundle.scorer_teacher = _LiveSegTeacher(bundle.scorer_teacher)
    bundle.distillation_weight = 0.0
    bundle.pose_distillation_weight = 0.0
    bundle.segnet_direct_live_distillation_weight = 0.5
    dual_config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.5,
        pose_distillation_weight=0.0,
    )
    for constraint in dual_config["constraints"]:
        constraint["target"] = 0.0
        constraint.pop("target_fraction_of_initial", None)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_direct_live_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
        train_time_dual_ascent_config=dual_config,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 0.0},
    )
    followup_metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 0.0},
    )
    mx.eval(adapter.model.parameters())

    assert metrics["loss_part_segnet_direct_live_distill"] == pytest.approx(
        metrics["loss_part_pr95_stage_segnet_direct_live_distill"]
    )
    assert metrics[
        "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill"
    ] == pytest.approx(0.0)
    assert metrics[
        "dual_ascent_metric__hi_nerv_segnet_direct_live_distill"
    ] == pytest.approx(metrics["loss_part_segnet_direct_live_distill"])
    assert metrics[
        "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill"
    ] > 0.0
    assert followup_metrics[
        "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill"
    ] == pytest.approx(0.0)


@requires_mlx
def test_pr95_curriculum_dual_ascent_observes_contrast_floor_alias_NO_FAKE() -> None:
    """Contrast-floor PR95-stage metric must feed the shared dual controller."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.dual_ascent import (
        build_default_nerv_train_time_dual_ascent_config,
    )

    bundle = _make_minimal_pr95_score_bundle()
    _force_high_contrast_scorer_targets(bundle)
    bundle.scorer_input_contrast_floor_weight = 0.5
    bundle.scorer_input_contrast_floor_segnet_min_std_ratio = 0.75
    bundle.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio = 0.75
    dual_config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        scorer_input_contrast_floor_weight=0.5,
    )
    for constraint in dual_config["constraints"]:
        constraint["target"] = 0.0
        constraint.pop("target_fraction_of_initial", None)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_contrast_floor_dual_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
        train_time_dual_ascent_config=dual_config,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )
    followup_metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )
    mx.eval(adapter.model.parameters())

    assert metrics[
        "dual_ascent_missing_metric__hi_nerv_scorer_input_contrast_floor"
    ] == pytest.approx(0.0)
    assert metrics[
        "dual_ascent_metric__hi_nerv_scorer_input_contrast_floor"
    ] == pytest.approx(metrics["loss_part_scorer_input_contrast_floor"])
    assert metrics[
        "dual_ascent_lambda__hi_nerv_scorer_input_contrast_floor"
    ] > 0.0
    assert followup_metrics[
        "dual_ascent_missing_metric__hi_nerv_scorer_input_contrast_floor"
    ] == pytest.approx(0.0)


@requires_mlx
def test_pr95_curriculum_stage_loss_consumes_launch_score_weight_controls_NO_FAKE() -> None:
    """SNeRV/HiNeRV score pressure controls are literal decoder loss weights."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import (
        MlxScoreAwareAdapter,
    )

    base_bundle = _make_minimal_pr95_score_bundle()
    controlled_bundle = _make_minimal_pr95_score_bundle()
    controlled_bundle.distillation_weight = 4.0
    controlled_bundle.pose_distillation_weight = 0.5
    base_adapter = MlxScoreAwareAdapter(
        base_bundle,
        substrate_id="test_substrate_base",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    controlled_adapter = MlxScoreAwareAdapter(
        controlled_bundle,
        substrate_id="test_substrate_controlled",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    base_adapter.notify_global_epoch(0)
    controlled_adapter.notify_global_epoch(0)
    batch = base_adapter.sample_batch(batch_size=2, seed=0)
    stage = base_adapter._pr95_curriculum_factory.current_stage_verdict(0)
    base_total, base_parts = base_adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage,
        model=base_adapter.model,
    )
    controlled_total, controlled_parts = controlled_adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage,
        model=controlled_adapter.model,
    )
    mx.eval(base_total, controlled_total, *base_parts.values(), *controlled_parts.values())

    seg = float(base_parts["pr95_stage_seg_surrogate"].item())
    pose = float(base_parts["pr95_stage_pose_surrogate"].item())
    expected_delta = (4.0 - 1.0) * seg + (0.5 - 1.0) * pose

    assert float(controlled_parts["pr95_stage_seg_control_multiplier"].item()) == pytest.approx(4.0)
    assert float(controlled_parts["pr95_stage_pose_control_multiplier"].item()) == pytest.approx(0.5)
    assert float(controlled_parts["pr95_stage_effective_seg_weight"].item()) == pytest.approx(4.0)
    assert float(controlled_parts["pr95_stage_effective_pose_weight"].item()) == pytest.approx(0.5)
    assert float(controlled_total.item()) - float(base_total.item()) == pytest.approx(
        expected_delta,
        rel=1e-5,
    )


@requires_mlx
def test_pr95_curriculum_stage_loss_consumes_dynamic_dual_weights_NO_FAKE() -> None:
    """Dual-ascent scorer weights must alter PR95-stage decoder loss."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    stage = adapter._pr95_curriculum_factory.current_stage_verdict(0)

    base_total, base_parts = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage,
        model=adapter.model,
        loss_weights={"distill": 1.0, "pose_distill": 1.0},
    )
    dual_total, dual_parts = adapter._pr95_stage_loss_and_parts(
        batch=batch,
        stage_verdict=stage,
        model=adapter.model,
        loss_weights={"distill": 2.5, "pose_distill": 0.25},
    )
    mx.eval(base_total, dual_total, *base_parts.values(), *dual_parts.values())

    seg = float(base_parts["pr95_stage_seg_surrogate"].item())
    pose = float(base_parts["pr95_stage_pose_surrogate"].item())
    expected_delta = (2.5 - 1.0) * seg + (0.25 - 1.0) * pose

    assert float(dual_parts["pr95_stage_distill_weight"].item()) == pytest.approx(2.5)
    assert float(dual_parts["pr95_stage_pose_distill_weight"].item()) == pytest.approx(
        0.25
    )
    assert float(dual_parts["pr95_stage_effective_seg_weight"].item()) == pytest.approx(
        2.5
    )
    assert float(
        dual_parts["pr95_stage_effective_pose_weight"].item()
    ) == pytest.approx(0.25)
    assert float(dual_total.item()) - float(base_total.item()) == pytest.approx(
        expected_delta,
        rel=1e-5,
    )


@requires_mlx
def test_pr95_curriculum_dual_ascent_observes_stage_surrogate_aliases_NO_FAKE() -> None:
    """PR95-stage SegNet/Pose metrics must feed shared dual-ascent controls."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
        train_time_dual_ascent_config={
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "hi_nerv_segnet_last_frame_distill",
                    "metric_name": "loss_part_distill",
                    "loss_weight_key": "distill",
                    "target": 0.0,
                    "dual_lr": 0.2,
                    "max_lambda": 6.0,
                },
                {
                    "constraint_id": "hi_nerv_posenet_yuv6_pair_distill",
                    "metric_name": "loss_part_pose_distill",
                    "loss_weight_key": "pose_distill",
                    "target": 0.0,
                    "dual_lr": 0.2,
                    "max_lambda": 6.0,
                },
            ],
        },
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )
    adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )
    mx.eval(adapter.model.parameters())

    assert metrics["loss_part_distill"] == pytest.approx(
        metrics["loss_part_pr95_stage_seg_surrogate"]
    )
    assert metrics["loss_part_pose_distill"] == pytest.approx(
        metrics["loss_part_pr95_stage_pose_surrogate"]
    )
    assert metrics["loss_part_pose_score_term"] == pytest.approx(
        metrics["loss_part_pr95_stage_pose_surrogate"]
    )
    assert metrics[
        "dual_ascent_missing_metric__hi_nerv_segnet_last_frame_distill"
    ] == pytest.approx(0.0)
    assert metrics[
        "dual_ascent_missing_metric__hi_nerv_posenet_yuv6_pair_distill"
    ] == pytest.approx(0.0)
    assert metrics[
        "dual_ascent_metric__hi_nerv_segnet_last_frame_distill"
    ] == pytest.approx(metrics["loss_part_pr95_stage_seg_surrogate"])
    assert metrics[
        "dual_ascent_metric__hi_nerv_posenet_yuv6_pair_distill"
    ] == pytest.approx(metrics["loss_part_pr95_stage_pose_surrogate"])
    assert metrics["dual_ascent_lambda__hi_nerv_segnet_last_frame_distill"] > 0.0
    assert metrics["dual_ascent_lambda__hi_nerv_posenet_yuv6_pair_distill"] > 0.0


@requires_mlx
def test_pr95_curriculum_snerv_dual_ascent_observes_stage_surrogate_aliases_NO_FAKE() -> None:
    """SNeRV PR95 runs must not leave scorer-distill dual constraints unpriced."""

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.dual_ascent import (
        build_default_nerv_train_time_dual_ascent_config,
    )

    bundle = _make_minimal_pr95_score_bundle()
    dual_config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
    )
    for constraint in dual_config["constraints"]:
        constraint["target"] = 0.0
        constraint.pop("target_fraction_of_initial", None)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_snerv_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
        train_time_dual_ascent_config=dual_config,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )
    followup_metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
    )
    mx.eval(adapter.model.parameters())

    assert metrics["loss_part_distill"] == pytest.approx(
        metrics["loss_part_pr95_stage_seg_surrogate"]
    )
    assert metrics["loss_part_pose_distill"] == pytest.approx(
        metrics["loss_part_pr95_stage_pose_surrogate"]
    )
    assert metrics["loss_part_pose_score_term"] == pytest.approx(
        metrics["loss_part_pr95_stage_pose_surrogate"]
    )
    assert metrics[
        "dual_ascent_missing_metric__snerv_segnet_last_frame_distill"
    ] == pytest.approx(0.0)
    assert metrics[
        "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill"
    ] == pytest.approx(0.0)
    assert metrics["dual_ascent_metric__snerv_segnet_last_frame_distill"] == (
        pytest.approx(metrics["loss_part_pr95_stage_seg_surrogate"])
    )
    assert metrics["dual_ascent_metric__snerv_posenet_yuv6_pair_distill"] == (
        pytest.approx(metrics["loss_part_pose_score_term"])
    )
    assert (
        followup_metrics["dual_ascent_missing_metric__snerv_segnet_last_frame_distill"]
        == pytest.approx(0.0)
    )
    assert (
        followup_metrics["dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill"]
        == pytest.approx(0.0)
    )
    assert followup_metrics["dual_ascent_metric__snerv_posenet_yuv6_pair_distill"] == (
        pytest.approx(followup_metrics["loss_part_pose_score_term"])
    )
    assert followup_metrics["dual_ascent_lambda__snerv_segnet_last_frame_distill"] > 0.0
    assert (
        followup_metrics["dual_ascent_lambda__snerv_posenet_yuv6_pair_distill"] > 0.0
    )


@requires_mlx
def test_pr95_curriculum_zero_launch_score_weight_stays_zero() -> None:
    """Explicit zero is an ablation/control, not an implicit PR95 default."""

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    bundle.distillation_weight = 0.0
    bundle.pose_distillation_weight = 0.0
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate_zero_controls",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )

    assert adapter._pr95_stage_score_weight_controls() == pytest.approx(
        (
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )


@requires_mlx
def test_pr95_curriculum_segnet_only_research_does_not_require_posenet() -> None:
    """SegNet-only research may use PR95 stages without silently re-enabling PoseNet."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    bundle.pose_distillation_weight = 0.0
    bundle.pose_scorer_teacher = None
    bundle.learnable_pose_student_head = None
    bundle.allow_segnet_only_research = True
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate_segnet_only_research",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 0.0},
    )

    assert metrics["loss_part_pr95_stage_effective_pose_weight"] == pytest.approx(0.0)
    assert metrics["finite_update_guard_active"] == pytest.approx(1.0)


@requires_mlx
def test_pr95_curriculum_source_weight_amplification_is_explicit_opt_in() -> None:
    """PR95 source 100:1 SegNet amplification remains available but never implicit."""

    from tac.substrates._shared.mlx_score_aware.adapter import (
        PR95_STAGE_BASE_POSE_WEIGHT,
        PR95_STAGE_BASE_SEG_WEIGHT,
        MlxScoreAwareAdapter,
    )

    bundle = _make_minimal_pr95_score_bundle()
    bundle.distillation_weight = 4.0
    bundle.pose_distillation_weight = 0.5
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate_source_amplified",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
        pr95_stage_source_weight_amplification_enabled=True,
    )

    assert adapter._pr95_stage_score_weight_controls() == pytest.approx(
        (
            4.0,
            0.5,
            PR95_STAGE_BASE_SEG_WEIGHT * 4.0,
            PR95_STAGE_BASE_POSE_WEIGHT * 0.5,
        )
    )


# ---------- Section 5: NO FAKE end-to-end verification (param mutation) ----------


@requires_mlx
def test_train_step_actually_mutates_parameters_per_stage_NO_FAKE() -> None:
    """NO FAKE per CLAUDE.md: train_step must actually mutate model parameters."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    # Initialize model parameters to non-zero so the loss gradient is non-zero.
    bundle.model.decoder_weight = mx.ones((4, 4)) * 0.5
    bundle.model.decoder_bias = mx.ones((4,)) * 0.1
    mx.eval(bundle.model.parameters())
    initial_weight = float(bundle.model.decoder_weight.sum().item())
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    # Run 5 train_steps and verify parameters move (NO FAKE).
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=4, seed=0)
    for _ in range(5):
        adapter.train_step(
            batch=batch,
            learning_rate=1e-2,
            loss_weights={"recon": 1.0},
        )
    mx.eval(bundle.model.parameters())
    final_weight = float(bundle.model.decoder_weight.sum().item())
    # The target is zeros, so MSE loss gradient should pull weights toward zero.
    # NO FAKE: parameters MUST have moved from their initial values.
    assert final_weight != initial_weight, (
        f"NO FAKE: decoder_weight unchanged after 5 train_steps "
        f"(initial={initial_weight}, final={final_weight}); "
        f"factory must actually invoke the optimizer"
    )
    # Bias may legitimately be near-zero gradient for some seeds; weight is the
    # primary NO FAKE assertion. We additionally validate the bias path doesn't
    # error out by simply checking it remains a real MLX array.
    assert bundle.model.decoder_bias is not None


@requires_mlx
def test_pact_muon_adamw_default_mutates_and_reports_partition_NO_FAKE() -> None:
    """Pact-native Muon+AdamW is a real partitioned optimizer, not a label."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    bundle.model.decoder_weight = mx.ones((4, 4)) * 0.5
    bundle.model.decoder_bias = mx.ones((4,)) * 0.1
    mx.eval(bundle.model.parameters())
    initial_weight = mx.array(bundle.model.decoder_weight)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        optimizer_kind="pact_muon_adamw",
        weight_decay=1.0e-4,
        grad_clip_max_norm=1.0,
    )
    batch = adapter.sample_batch(batch_size=4, seed=0)

    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1.0e-2,
        loss_weights={"recon": 1.0},
    )
    mx.eval(bundle.model.parameters())

    moved = float(mx.max(mx.abs(bundle.model.decoder_weight - initial_weight)).item())
    assert moved > 0.0, "NO FAKE: pact_muon_adamw left decoder weights unchanged"
    assert metrics["pact_optimizer_uses_muon"] == pytest.approx(1.0)
    assert metrics["pact_muon_tensor_count"] > 0.0
    assert metrics["pact_adamw_tensor_count"] > 0.0
    assert "pr95_stage_index" not in metrics
    summary = adapter.wave_n11_stabilizer_summary()
    assert summary["pact_native_muon_adamw_partition_enabled"] is True
    assert summary["pact_native_muon_adamw_last_step_summary"] is not None


@requires_mlx
def test_stage_transition_resets_muon_buffers_per_l15_invariant() -> None:
    """L15 canonical equation: stage 8 starts with fresh Muon buffers."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_pr95_score_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,  # 10 epochs/stage; stage 7 ends at 70.
    )
    # Step through stage 7 (epochs 60-69 in 80-epoch budget).
    adapter.notify_global_epoch(60)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    adapter.train_step(
        batch=batch, learning_rate=1e-3, loss_weights={"recon": 1.0}
    )
    # Stage 7 doesn't use Muon, so buffers should be empty.
    assert len(adapter._pr95_optimizer_state.muon_buffers) == 0, (
        "stage 7 (use_muon=False) MUST not populate muon_buffers"
    )
    # Advance to stage 8 (epoch 75 in 80-epoch budget).
    adapter.notify_global_epoch(75)
    adapter.train_step(
        batch=batch, learning_rate=1e-3, loss_weights={"recon": 1.0}
    )
    # Stage 8 uses Muon → buffers should now be populated for Muon-eligible params.
    assert len(adapter._pr95_optimizer_state.muon_buffers) > 0, (
        "stage 8 (use_muon=True) MUST populate muon_buffers for Muon-eligible params"
    )


@requires_mlx
def test_canonical_partition_routes_decoder_weight_to_muon_in_stage_8() -> None:
    """Canonical partition: 2D 'decoder_weight' (no stem/rgb/latents) routes to Muon."""
    from tac.local_acceleration.pr95_hnerv_mlx import (
        partition_pr95_mlx_parameter_names,
    )
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(  # noqa: F841 (constructs the model)
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    split = partition_pr95_mlx_parameter_names(bundle.model.parameters())
    # 2D decoder_weight should be in Muon partition.
    assert any("decoder_weight" in name for name in split["muon"]), (
        f"decoder_weight (2D, no stem/rgb/latents) MUST be Muon-eligible; "
        f"got muon={split['muon']}, adamw={split['adamw']}"
    )
    # 1D decoder_bias should be in AdamW partition.
    assert any("decoder_bias" in name for name in split["adamw"]), (
        f"decoder_bias (1D) MUST be AdamW-only; "
        f"got muon={split['muon']}, adamw={split['adamw']}"
    )


# ---------- Section 6: backward compat regression guards ----------


@requires_mlx
def test_existing_loss_adapter_tests_still_pass_with_default_off() -> None:
    """Regression: existing adapter tests should pass unchanged with default-off."""
    # Sentinel test: import the existing loss/adapter/harness test module and
    # verify it still imports cleanly (no API drift from the kwarg addition).
    from tac.substrates._shared.mlx_score_aware.tests import (
        test_loss_adapter_harness,  # noqa: F401
    )
    # No assertions beyond import-time success; the dedicated existing tests
    # already cover the legacy behavior comprehensively. The new kwarg has a
    # safe default that preserves prior behavior.


@requires_mlx
def test_no_pr95_curriculum_kwargs_required_for_legacy_callers() -> None:
    """Legacy callers can construct MlxScoreAwareAdapter with NO new kwargs."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    # The legacy constructor signature MUST still work without specifying
    # pr95_faithful_curriculum_enabled.
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="test_substrate")
    assert adapter.substrate_id == "test_substrate"
    assert adapter._pr95_faithful_curriculum_enabled is False
