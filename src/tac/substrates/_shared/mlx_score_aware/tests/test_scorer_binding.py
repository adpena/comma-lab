# SPDX-License-Identifier: MIT
"""Gradient-reachability tests for the real-scorer-bound distillation path.

These are the structural self-protection for the C6 IBPS / DreamerV3 RSSM
scorer-blindness bug class (per CLAUDE.md "HNeRV / leaderboard-implementation
parity discipline" L1 + Catalog #164). The harness ``score_aware_loss`` MUST,
when ``scorer_teacher`` is wired, route the renderer-param gradient THROUGH the
scorer surrogate (a learnable student head distilled from the REAL SegNet
teacher) — NOT through a fixed cosine of pixel means.

The DECISIVE test (``test_real_scorer_grad_is_distinct_from_recon`): the
real-scorer-bound distill gradient must point in a MATERIALLY DIFFERENT
direction than the pure-reconstruction gradient (cos < 0.9). The mock-teacher
(scorer-blind) distill gradient, by contrast, is ~parallel to recon (cos > 0.97)
because it carries no SegNet class-boundary information.

MLX-bound tests skip cleanly on non-Apple-Silicon CI.
"""
from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware import (
    MlxScoreAwareHarnessError,
    RendererBundle,
    build_mlx_segnet_pair_teacher,
    decode_frames_nhwc01,
    score_aware_loss,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")


def _real_video_targets(num_pairs: int):
    """Decode real contest video targets at canonical SegNet size (384, 512)."""
    from pathlib import Path

    from tac.substrates._shared.mlx_score_aware import decode_mlx_targets

    repo = Path(__file__).resolve().parents[6]
    video = repo / "upstream" / "videos" / "0.mkv"
    if not video.is_file():
        pytest.skip(f"contest video not staged at {video}")
    return decode_mlx_targets(
        video, num_pairs=num_pairs, output_height=384, output_width=512
    )


def _dreamer_model(num_pairs: int):
    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(
        num_pairs=num_pairs,
        num_groups=2,
        num_categories=4,
        decoder_latent_dim=8,
        base_channels=4,
        eval_size=(384, 512),
    )
    return DreamerV3RSSMSubstrateMLX(cfg)


def _grad_vec(model, loss_closure):
    """Flatten the renderer-param gradient of ``loss_closure`` into a 1-D np vec."""
    import mlx.nn as mlx_nn
    import numpy as np
    from mlx.utils import tree_flatten

    _loss, grads = mlx_nn.value_and_grad(model, loss_closure)(model)
    vecs = [
        np.array(v).astype(np.float32).reshape(-1)
        for _k, v in tree_flatten(grads)
        if hasattr(v, "shape")
    ]
    return np.concatenate(vecs)


def _cos(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# fail-closed bundle invariant (the structural C6 IBPS extinction)
# --------------------------------------------------------------------------- #


@mlx_only
def test_bundle_fails_closed_on_scorer_blind_distill() -> None:
    """distill > 0 with NO scorer_teacher and NOT allow_mock => REJECTED."""
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))
    with pytest.raises(MlxScoreAwareHarnessError, match="SCORER-BLIND"):
        RendererBundle(
            model=model,
            target_rgb_0=t,
            target_rgb_1=t,
            num_pairs=2,
            forward_convention="call_b2chw_255",
            distillation_weight=0.5,
        )


@mlx_only
def test_bundle_allows_mock_when_opted_in() -> None:
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))
    bundle = RendererBundle(
        model=model,
        target_rgb_0=t,
        target_rgb_1=t,
        num_pairs=2,
        forward_convention="call_b2chw_255",
        distillation_weight=0.5,
        allow_mock_scorer_teacher=True,
    )
    assert bundle.allow_mock_scorer_teacher is True


@mlx_only
def test_bundle_requires_head_when_scorer_teacher_set() -> None:
    """scorer_teacher set but no learnable_student_head => REJECTED."""
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))

    class _StubTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            return mx.zeros((idx.shape[0], 384, 512, 5))

    with pytest.raises(MlxScoreAwareHarnessError, match="learnable_student_head"):
        RendererBundle(
            model=model,
            target_rgb_0=t,
            target_rgb_1=t,
            num_pairs=2,
            forward_convention="call_b2chw_255",
            distillation_weight=0.5,
            scorer_teacher=_StubTeacher(),
        )


# --------------------------------------------------------------------------- #
# DECISIVE gradient-reachability: real scorer pulls grad OFF the recon direction
# --------------------------------------------------------------------------- #


def _distill_only_grad(model, bundle, idx):
    """Isolate the renderer-param gradient of JUST the distill term.

    Combined-loss cos vs recon is a weak metric (the distill grad is small
    relative to recon at init, so the SUM barely rotates). The decisive
    gradient-reachability metric is the DISTILL-ONLY gradient: it must flow
    through the scorer surrogate, be finite + nonzero, and (for the real
    scorer) point in a direction NOT explained by reconstruction.
    """
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        hinton_distilled_kl_t2_loss,
    )

    def _closure(_m):
        rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
        seg_rgb = rgb_1 if bundle.segnet_teacher_frame_index == 1 else rgb_0
        target_seg_rgb = (
            bundle.target_rgb_1[idx]
            if bundle.segnet_teacher_frame_index == 1
            else bundle.target_rgb_0[idx]
        )
        if bundle.scorer_teacher is not None:
            student = bundle.learnable_student_head(seg_rgb)
            teacher = mx.stop_gradient(
                bundle.scorer_teacher.teacher_logits_for_indices(idx)
            )
        else:
            from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
                MockTeacherLogitsProvider,
            )

            mock = MockTeacherLogitsProvider(num_classes=bundle.distillation_num_classes)
            student = mock.teacher_logits(seg_rgb)
            teacher = mx.stop_gradient(mock.teacher_logits(target_seg_rgb))
        return hinton_distilled_kl_t2_loss(
            student_logits=student,
            teacher_logits=teacher,
            temperature=bundle.distillation_temperature,
        )

    return _grad_vec(model, _closure)


@mlx_only
def test_real_scorer_distill_grad_is_reachable_finite_nonzero_and_scorer_bound() -> None:
    """The fix proof (Catalog #164): the real-scorer distill gradient

    (1) is gradient-REACHABLE — flows through the student head + decoded frame
        into the renderer params (nonzero);
    (2) is FINITE (the learnable-head surrogate avoids the full-SegNet-backprop
        NaN that bites the renderer's PixelShuffle/bilinear second-order graph);
    (3) is SCORER-BOUND — its direction is NOT explained by reconstruction
        (cos vs recon < 0.5), i.e. it pulls the renderer toward the real
        SegNet's class boundaries, not toward pixel-MSE redundant directions.
    """
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_student_head,
    )

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    idx = mx.arange(num_pairs, dtype=mx.int32)

    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    teacher = build_mlx_segnet_pair_teacher(recon_bundle, device="cpu")
    head = build_learnable_student_head(num_classes=teacher.num_classes, seed=0)
    real_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.5,
        scorer_teacher=teacher, learnable_student_head=head,
    )

    g_distill = _distill_only_grad(model, real_bundle, idx)
    g_recon = _grad_vec(model, lambda _m: score_aware_loss(recon_bundle, idx)[0])

    # (1) reachable + nonzero
    assert float(np.linalg.norm(g_distill)) > 0.0, (
        "real-scorer distill gradient is zero => NOT gradient-reachable through "
        "the renderer (the scorer-blind failure mode)."
    )
    # (2) finite (no full-SegNet-backprop NaN)
    assert np.all(np.isfinite(g_distill)), (
        "real-scorer distill gradient must be finite; full-SegNet backprop "
        "NaNs in MLX, the learnable-head surrogate avoids that."
    )
    # (3) scorer-bound: direction NOT explained by reconstruction
    c = _cos(g_distill, g_recon)
    assert c == c, "cos must not be NaN"
    assert c < 0.5, (
        f"real-scorer distill-only grad cos vs recon = {c:.4f}; expected < 0.5 "
        "(scorer-BOUND: a direction reconstruction does not already provide). "
        "cos ~ 1.0 would mean the distill term is recon-redundant / scorer-blind "
        "(the C6 IBPS / DreamerV3 bug)."
    )


@mlx_only
def test_real_scorer_distill_grad_differs_from_mock() -> None:
    """The real scorer teacher carries information the pixel-cosine mock cannot.

    cos(real-distill-grad, mock-distill-grad) < 0.95 => the real SegNet teacher
    pulls the renderer in a materially different direction than the scorer-blind
    mock. If they were identical the real teacher would add no scorer signal.
    """
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_student_head,
    )

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    idx = mx.arange(num_pairs, dtype=mx.int32)

    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    teacher = build_mlx_segnet_pair_teacher(recon_bundle, device="cpu")
    head = build_learnable_student_head(num_classes=teacher.num_classes, seed=0)
    real_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.5,
        scorer_teacher=teacher, learnable_student_head=head,
    )
    mock_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.5,
        allow_mock_scorer_teacher=True,
    )

    g_real = _distill_only_grad(model, real_bundle, idx)
    g_mock = _distill_only_grad(model, mock_bundle, idx)
    assert np.all(np.isfinite(g_real)) and np.all(np.isfinite(g_mock))
    c = _cos(g_real, g_mock)
    assert c < 0.95, (
        f"cos(real-distill, mock-distill) = {c:.4f}; expected < 0.95 (the real "
        "SegNet teacher carries class-boundary information the fixed pixel-cosine "
        "mock cannot)."
    )


# --------------------------------------------------------------------------- #
# joint head training: the adapter trains the student head alongside the renderer
# --------------------------------------------------------------------------- #


@mlx_only
def test_adapter_trains_head_jointly_and_reduces_loss() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware import MlxScoreAwareAdapter
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_student_head,
    )

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    teacher = build_mlx_segnet_pair_teacher(recon_bundle, device="cpu")
    head = build_learnable_student_head(num_classes=teacher.num_classes, seed=0)
    bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.5,
        scorer_teacher=teacher, learnable_student_head=head,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.arange(num_pairs, dtype=mx.int32)
    w0 = mx.array(head.weight)  # snapshot before
    losses = [
        adapter.train_step(batch, learning_rate=1e-2, loss_weights={})["total"]
        for _ in range(10)
    ]
    # Renderer descends the score-aware loss.
    assert losses[-1] < losses[0]
    # The student head's params actually moved (joint sibling-optimizer step).
    moved = float(mx.max(mx.abs(head.weight - w0)).item())
    assert moved > 0.0, "student head params must train jointly (sibling step)"
