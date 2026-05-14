# SPDX-License-Identifier: MIT
"""Real-scorer regression for balle_renderer score-aware loss (WAVE-A-1 sister).

Defends against re-introduction of the WWW4 5D-vs-4D shape mismatch.
See ``src/tac/substrates/_shared/score_aware_loss_real_scorer_test_kit.py``.
"""

from __future__ import annotations

from tac.substrates._shared.score_aware_loss_real_scorer_test_kit import (
    assert_loss_runs_on_pyav_decoded_pair,
    assert_loss_runs_on_real_segnet,
)


def _build_loss(seg_scorer, pose_scorer):
    from tac.substrates.balle_renderer.score_aware_loss import (
        BalleRendererScoreAwareLoss,
        BalleScoreAwareLossWeights,
    )

    return BalleRendererScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=BalleScoreAwareLossWeights(),
    )


def _invoke_loss(loss_fn, ctx):
    import torch

    rate_components = {
        "hyper_rate": torch.tensor(1.0, requires_grad=True),
        "main_rate": torch.tensor(10.0, requires_grad=True),
        "total_rate": torch.tensor(11.0, requires_grad=True),
    }
    return loss_fn(
        ctx["rgb_0"],
        ctx["rgb_1"],
        ctx["gt_0"],
        ctx["gt_1"],
        ctx["bytes_proxy"],
        rate_components,
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )


def test_balle_renderer_loss_runs_on_real_segnet_smp_unet_random_init():
    assert_loss_runs_on_real_segnet(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )


def test_balle_renderer_loss_runs_on_real_pyav_decoded_frame_pair():
    assert_loss_runs_on_pyav_decoded_pair(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )
