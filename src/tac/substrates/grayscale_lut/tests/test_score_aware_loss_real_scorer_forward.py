"""Real-scorer regression for grayscale_lut score-aware loss (WAVE-A-1 sister).

Defends against re-introduction of the WWW4 5D-vs-4D shape mismatch.
See ``src/tac/substrates/_shared/score_aware_loss_real_scorer_test_kit.py``.
"""

from __future__ import annotations

from tac.substrates._shared.score_aware_loss_real_scorer_test_kit import (
    assert_loss_runs_on_pyav_decoded_pair,
    assert_loss_runs_on_real_segnet,
)


def _build_loss(seg_scorer, pose_scorer):
    from tac.substrates.grayscale_lut.score_aware_loss import (
        GrayscaleLutScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    return GrayscaleLutScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=ScoreAwareLossWeights(),
    )


def _invoke_loss(loss_fn, ctx):
    import torch

    # grayscale_param shape: (N, 1, H/D, W/D); D is the downsample factor.
    # Shape isn't load-bearing for the scorer-path regression — we use a
    # small leaf tensor with gradient so the TV-regularizer has somewhere
    # to flow. Use ctx rgb_0 shape to scale.
    h = max(ctx["rgb_0"].shape[-2] // 4, 8)
    w = max(ctx["rgb_0"].shape[-1] // 4, 8)
    grayscale_param = torch.rand(1, 1, h, w, requires_grad=True)
    return loss_fn(
        ctx["rgb_0"],
        ctx["rgb_1"],
        ctx["gt_0"],
        ctx["gt_1"],
        ctx["bytes_proxy"],
        grayscale_param,
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )


def test_grayscale_lut_loss_runs_on_real_segnet_smp_unet_random_init():
    assert_loss_runs_on_real_segnet(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )


def test_grayscale_lut_loss_runs_on_real_pyav_decoded_frame_pair():
    assert_loss_runs_on_pyav_decoded_pair(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )
