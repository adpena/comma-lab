# SPDX-License-Identifier: MIT
"""Gradient-reachability tests for the real-PoseNet-bound distillation path.

The POSE axis sister of ``test_scorer_binding.py``. PoseNet is the DOMINANT
contest scorer component at the frontier (per CLAUDE.md "SegNet vs PoseNet
importance — operating-point dependent": below pose_avg ~ 2.5e-4 the pose
marginal exceeds SegNet's; the SegNet-only verification run drifted pose +10.6
precisely because no pose teacher was wired). These are the structural
self-protection for the pose half of the C6 IBPS / DreamerV3 RSSM
scorer-blindness bug class (Catalog #164): the harness ``score_aware_loss``
MUST, when ``pose_scorer_teacher`` is wired, route the renderer-param gradient
THROUGH the learnable pose head distilled (MSE) from the REAL PoseNet teacher.

The DECISIVE test
(``test_real_pose_distill_grad_is_reachable_finite_nonzero_and_scorer_bound``):
the real-pose-bound distill gradient must be (1) reachable+nonzero, (2) finite
(the learnable-head surrogate avoids the full-PoseNet second-order NaN), and
(3) scorer-bound — a direction reconstruction does NOT already provide
(cos vs recon < 0.8).

MLX-bound tests skip cleanly on non-Apple-Silicon CI.
"""
from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware import (
    MlxScoreAwareHarnessError,
    RendererBundle,
    build_mlx_posenet_pair_teacher,
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
# fail-closed bundle invariants (the structural pose-axis extinction)
# --------------------------------------------------------------------------- #


class _StubSegTeacher:
    num_classes = 5

    def teacher_logits_for_indices(self, idx):
        import mlx.core as mx

        return mx.zeros((idx.shape[0], 384, 512, 5))


class _StubPoseTeacher:
    pose_dims = 6

    def teacher_pose_for_indices(self, idx):
        import mlx.core as mx

        return mx.zeros((idx.shape[0], 6))


@mlx_only
def test_bundle_fails_closed_on_pose_distill_without_teacher() -> None:
    """pose_distillation_weight > 0 with NO pose_scorer_teacher => REJECTED.

    Unlike SegNet there is NO scorer-blind pose mock (pose is a continuous
    ego-motion vector, not a class distribution), so a pose distill term
    without a real teacher is unconditionally refused.
    """
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))
    with pytest.raises(MlxScoreAwareHarnessError, match="pose_scorer_teacher"):
        RendererBundle(
            model=model,
            target_rgb_0=t,
            target_rgb_1=t,
            num_pairs=2,
            forward_convention="call_b2chw_255",
            pose_distillation_weight=0.5,
        )


@mlx_only
def test_bundle_requires_pose_head_when_pose_teacher_set() -> None:
    """pose_scorer_teacher set but no learnable_pose_student_head => REJECTED."""
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))
    with pytest.raises(
        MlxScoreAwareHarnessError, match="learnable_pose_student_head"
    ):
        RendererBundle(
            model=model,
            target_rgb_0=t,
            target_rgb_1=t,
            num_pairs=2,
            forward_convention="call_b2chw_255",
            pose_distillation_weight=0.5,
            pose_scorer_teacher=_StubPoseTeacher(),
        )


@mlx_only
def test_bundle_fails_closed_on_segnet_only_without_optin() -> None:
    """SegNet bound but NOT pose bound, no opt-in => REJECTED (frontier rule).

    PoseNet is dominant at the frontier; a SegNet-only binding leaves the
    dominant axis drifting (the SegNet-only verification run drifted pose
    +10.6). The bundle refuses it unless allow_segnet_only_research=True.
    """
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))

    class _Head:  # any non-None placeholder; invariant checks presence only
        pass

    with pytest.raises(MlxScoreAwareHarnessError, match="PoseNet is DOMINANT"):
        RendererBundle(
            model=model,
            target_rgb_0=t,
            target_rgb_1=t,
            num_pairs=2,
            forward_convention="call_b2chw_255",
            distillation_weight=0.5,
            scorer_teacher=_StubSegTeacher(),
            learnable_student_head=_Head(),
        )


@mlx_only
def test_bundle_allows_segnet_only_research_optin() -> None:
    """SegNet-only binding WITH explicit opt-in is allowed (research path)."""
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))

    class _Head:
        pass

    bundle = RendererBundle(
        model=model,
        target_rgb_0=t,
        target_rgb_1=t,
        num_pairs=2,
        forward_convention="call_b2chw_255",
        distillation_weight=0.5,
        scorer_teacher=_StubSegTeacher(),
        learnable_student_head=_Head(),
        allow_segnet_only_research=True,
    )
    assert bundle.allow_segnet_only_research is True


@mlx_only
def test_bundle_allows_both_scorers_bound() -> None:
    """Binding BOTH SegNet and PoseNet teachers is the canonical frontier path."""
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))

    class _Head:
        pass

    bundle = RendererBundle(
        model=model,
        target_rgb_0=t,
        target_rgb_1=t,
        num_pairs=2,
        forward_convention="call_b2chw_255",
        distillation_weight=0.5,
        scorer_teacher=_StubSegTeacher(),
        learnable_student_head=_Head(),
        pose_distillation_weight=0.5,
        pose_scorer_teacher=_StubPoseTeacher(),
        learnable_pose_student_head=_Head(),
    )
    assert bundle.pose_distillation_weight == 0.5
    assert bundle.allow_segnet_only_research is False


@mlx_only
def test_pose_only_binding_is_allowed() -> None:
    """Binding ONLY the PoseNet teacher (no SegNet) is allowed.

    The frontier both-scorer invariant only fires when SegNet is bound without
    pose; binding pose alone is a legitimate pose-axis focus and is NOT refused.
    """
    import mlx.core as mx

    model = _dreamer_model(2)
    t = mx.zeros((2, 384, 512, 3))

    class _Head:
        pass

    bundle = RendererBundle(
        model=model,
        target_rgb_0=t,
        target_rgb_1=t,
        num_pairs=2,
        forward_convention="call_b2chw_255",
        pose_distillation_weight=0.5,
        pose_scorer_teacher=_StubPoseTeacher(),
        learnable_pose_student_head=_Head(),
    )
    assert bundle.pose_distillation_weight == 0.5


# --------------------------------------------------------------------------- #
# real PoseNet teacher cache + pose student head construction
# --------------------------------------------------------------------------- #


@mlx_only
def test_build_mlx_posenet_pair_teacher_shapes_and_provider_contract() -> None:
    """The real PoseNet teacher cache has (num_pairs, pose_dims) + the contract."""
    import mlx.core as mx
    import numpy as np

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    teacher = build_mlx_posenet_pair_teacher(recon_bundle, device="cpu")
    assert teacher.num_pairs == num_pairs
    assert teacher.pose_dims == 6  # default bundle.pose_dims
    assert tuple(teacher.per_dim_scale.shape) == (6,)
    # Provider contract: teacher_pose_for_indices(idx) -> (B, pose_dims).
    idx = mx.array([0, 2], dtype=mx.int32)
    pose = teacher.teacher_pose_for_indices(idx)
    mx.eval(pose)
    assert tuple(pose.shape) == (2, 6)
    pose_np = np.array(pose)
    assert np.all(np.isfinite(pose_np))
    # The real PoseNet pose is NOT all-zero (it is a real ego-motion estimate).
    assert float(np.abs(pose_np).max()) > 0.0


@mlx_only
def test_build_mlx_posenet_teacher_rejects_wrong_size_targets() -> None:
    """Targets not at contest (384, 512) are refused for shape alignment."""
    import mlx.core as mx

    num_pairs = 2
    model = _dreamer_model(num_pairs)
    bad = mx.zeros((num_pairs, 256, 256, 3))
    recon_bundle = RendererBundle(
        model=model, target_rgb_0=bad, target_rgb_1=bad, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    with pytest.raises(MlxScoreAwareHarnessError, match=r"\(384, 512\)"):
        build_mlx_posenet_pair_teacher(recon_bundle, device="cpu")


# --------------------------------------------------------------------------- #
# DECISIVE gradient-reachability: real pose teacher pulls grad OFF recon
# --------------------------------------------------------------------------- #


def _pose_distill_only_grad(model, bundle, idx):
    """Isolate the renderer-param gradient of JUST the pose-distill term.

    The pose head consumes the decoded frame PAIR; the teacher pose is
    gradient-blocked. The gradient must flow pose-MSE -> pose_head(decoded) ->
    renderer.
    """
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        pose_distillation_mse_loss,
    )

    def _closure(_m):
        rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
        student = bundle.learnable_pose_student_head(rgb_0, rgb_1)
        teacher = mx.stop_gradient(
            bundle.pose_scorer_teacher.teacher_pose_for_indices(idx)
        )
        return pose_distillation_mse_loss(
            student_pose=student,
            teacher_pose=teacher,
            per_dim_scale=getattr(bundle.pose_scorer_teacher, "per_dim_scale", None),
        )

    return _grad_vec(model, _closure)


@mlx_only
def test_real_pose_distill_grad_is_reachable_finite_nonzero_and_scorer_bound() -> None:
    """The pose fix proof (Catalog #164, POSE axis): the real-pose distill grad

    (1) is gradient-REACHABLE — flows through the pose head + decoded pair into
        the renderer params (nonzero);
    (2) is FINITE (the learnable-head surrogate avoids the full-PoseNet-backprop
        NaN that bites the renderer's PixelShuffle/bilinear second-order graph);
    (3) is SCORER-BOUND — its direction is NOT explained by reconstruction
        (cos vs recon < 0.8), i.e. it pulls the renderer toward the real
        PoseNet's ego-motion estimate, not toward pixel-MSE-redundant directions.
    """
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_pose_student_head,
    )

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    idx = mx.arange(num_pairs, dtype=mx.int32)

    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    teacher = build_mlx_posenet_pair_teacher(recon_bundle, device="cpu")
    pose_head = build_learnable_pose_student_head(
        pose_dims=teacher.pose_dims, seed=0
    )
    pose_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        pose_distillation_weight=0.5,
        pose_scorer_teacher=teacher, learnable_pose_student_head=pose_head,
    )

    g_pose = _pose_distill_only_grad(model, pose_bundle, idx)
    g_recon = _grad_vec(model, lambda _m: score_aware_loss(recon_bundle, idx)[0])

    # (1) reachable + nonzero
    assert float(np.linalg.norm(g_pose)) > 0.0, (
        "real-pose distill gradient is zero => NOT gradient-reachable through "
        "the renderer (the scorer-blind failure mode)."
    )
    # (2) finite (no full-PoseNet-backprop NaN)
    assert np.all(np.isfinite(g_pose)), (
        "real-pose distill gradient must be finite; full-PoseNet backprop NaNs "
        "in MLX's second-order autograd, the learnable-head surrogate avoids it."
    )
    # (3) scorer-bound: direction NOT explained by reconstruction
    c = _cos(g_pose, g_recon)
    assert c == c, "cos must not be NaN"
    assert c < 0.8, (
        f"real-pose distill-only grad cos vs recon = {c:.4f}; expected < 0.8 "
        "(scorer-BOUND: a direction reconstruction does not already provide). "
        "cos ~ 1.0 would mean the pose term is recon-redundant / scorer-blind."
    )


@mlx_only
def test_score_aware_loss_emits_pose_distill_part_when_pose_bound() -> None:
    """When pose binding is active, score_aware_loss exposes ``pose_distill``."""
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_pose_student_head,
    )

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    idx = mx.arange(num_pairs, dtype=mx.int32)
    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    teacher = build_mlx_posenet_pair_teacher(recon_bundle, device="cpu")
    pose_head = build_learnable_pose_student_head(
        pose_dims=teacher.pose_dims, seed=0
    )
    bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        pose_distillation_weight=0.5,
        pose_scorer_teacher=teacher, learnable_pose_student_head=pose_head,
    )
    _total, parts = score_aware_loss(bundle, idx)
    assert "pose_distill" in parts
    assert "recon" in parts
    mx.eval(parts["pose_distill"])
    assert float(parts["pose_distill"].item()) >= 0.0


# --------------------------------------------------------------------------- #
# joint pose-head training: the adapter trains the pose head alongside renderer
# --------------------------------------------------------------------------- #


@mlx_only
def test_adapter_trains_pose_head_jointly_and_reduces_loss() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware import MlxScoreAwareAdapter
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_pose_student_head,
    )

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    teacher = build_mlx_posenet_pair_teacher(recon_bundle, device="cpu")
    pose_head = build_learnable_pose_student_head(
        pose_dims=teacher.pose_dims, seed=0
    )
    bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        pose_distillation_weight=0.5,
        pose_scorer_teacher=teacher, learnable_pose_student_head=pose_head,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.arange(num_pairs, dtype=mx.int32)
    w0 = mx.array(pose_head.weight)  # snapshot before
    losses = [
        adapter.train_step(batch, learning_rate=1e-3, loss_weights={})["total"]
        for _ in range(10)
    ]
    # Renderer descends the score-aware (pose-bound) loss.
    assert losses[-1] < losses[0]
    # The pose head's params actually moved (joint sibling-optimizer step).
    moved = float(mx.max(mx.abs(pose_head.weight - w0)).item())
    assert moved > 0.0, "pose head params must train jointly (sibling step)"


@mlx_only
def test_adapter_trains_both_heads_jointly() -> None:
    """Both the SegNet head AND the pose head train jointly when both bound."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware import (
        MlxScoreAwareAdapter,
        build_mlx_segnet_pair_teacher,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    num_pairs = 4
    model = _dreamer_model(num_pairs)
    t0, t1 = _real_video_targets(num_pairs)
    recon_bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255", distillation_weight=0.0,
    )
    seg_teacher = build_mlx_segnet_pair_teacher(recon_bundle, device="cpu")
    seg_head = build_learnable_student_head(
        num_classes=seg_teacher.num_classes, seed=0
    )
    pose_teacher = build_mlx_posenet_pair_teacher(recon_bundle, device="cpu")
    pose_head = build_learnable_pose_student_head(
        pose_dims=pose_teacher.pose_dims, seed=0
    )
    bundle = RendererBundle(
        model=model, target_rgb_0=t0, target_rgb_1=t1, num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        distillation_weight=0.5,
        scorer_teacher=seg_teacher, learnable_student_head=seg_head,
        pose_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher, learnable_pose_student_head=pose_head,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.arange(num_pairs, dtype=mx.int32)
    seg_w0 = mx.array(seg_head.weight)
    pose_w0 = mx.array(pose_head.weight)
    losses = [
        adapter.train_step(batch, learning_rate=1e-3, loss_weights={})["total"]
        for _ in range(10)
    ]
    assert losses[-1] < losses[0]
    assert float(mx.max(mx.abs(seg_head.weight - seg_w0)).item()) > 0.0
    assert float(mx.max(mx.abs(pose_head.weight - pose_w0)).item()) > 0.0
