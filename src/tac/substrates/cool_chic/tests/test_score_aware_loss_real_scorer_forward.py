"""Real-scorer regression for cool_chic score-aware loss (WAVE-A-1 sister).

Defends against re-introduction of the WWW4 5D-vs-4D shape mismatch.
See ``src/tac/substrates/_shared/score_aware_loss_real_scorer_test_kit.py``.
"""

from __future__ import annotations

import pytest

from tac.substrates._shared.score_aware_loss_real_scorer_test_kit import (
    assert_loss_runs_on_pyav_decoded_pair,
    assert_loss_runs_on_real_segnet,
)


def _build_loss(seg_scorer, pose_scorer):
    from tac.substrates.cool_chic.score_aware_loss import (
        CoolChicScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    return CoolChicScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=ScoreAwareLossWeights(),
    )


def _invoke_loss(loss_fn, ctx):
    import torch

    # ar_log_prob = sum log p(z_t | z_{t-1}); leaf tensor for the
    # scorer-shape regression. AR rate term is downstream of the
    # scorer path the WWW4 bug hit.
    ar_log_prob = torch.tensor(-100.0, requires_grad=True)
    return loss_fn(
        ctx["rgb_0"],
        ctx["rgb_1"],
        ctx["gt_0"],
        ctx["gt_1"],
        ctx["bytes_proxy"],
        ar_log_prob,
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )


def test_cool_chic_loss_runs_on_real_segnet_smp_unet_random_init():
    assert_loss_runs_on_real_segnet(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )


def test_cool_chic_loss_runs_on_real_pyav_decoded_frame_pair():
    assert_loss_runs_on_pyav_decoded_pair(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )
