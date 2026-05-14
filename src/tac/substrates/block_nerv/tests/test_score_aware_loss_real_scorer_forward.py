# SPDX-License-Identifier: MIT
"""Real-scorer regression for block_nerv score-aware loss (WAVE-A-1 sister).

Defends against re-introduction of the WWW4 5D-vs-4D shape mismatch that
crashed the sane_hnerv first-anchor dispatch (Modal A100, 2026-05-12,
fc-01KREXK209TRX7ED5ZRVXHY1VT). FIX-H Part 1 fixed the sane_hnerv
``score_aware_loss.py``; this sister test confirms block_nerv's loss runs
end-to-end against the REAL ``upstream.modules.SegNet`` (``smp.Unet``
``tu-efficientnet_b2`` stem) shape contract.

See ``src/tac/substrates/_shared/score_aware_loss_real_scorer_test_kit.py``
for the canonical assertion helpers and literature anchors.
"""

from __future__ import annotations

from tac.substrates._shared.score_aware_loss_real_scorer_test_kit import (
    assert_loss_runs_on_pyav_decoded_pair,
    assert_loss_runs_on_real_segnet,
)


def _build_loss(seg_scorer, pose_scorer):
    from tac.substrates.block_nerv.score_aware_loss import (
        BlockNervScoreAwareLoss,
        BlockScoreAwareLossWeights,
    )

    return BlockNervScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=BlockScoreAwareLossWeights(),
    )


def _invoke_loss(loss_fn, ctx):
    return loss_fn(
        ctx["rgb_0"],
        ctx["rgb_1"],
        ctx["gt_0"],
        ctx["gt_1"],
        ctx["bytes_proxy"],
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )


def test_block_nerv_loss_runs_on_real_segnet_smp_unet_random_init():
    """Real ``smp.Unet`` SegNet forward consumes loss output without shape error."""
    assert_loss_runs_on_real_segnet(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )


def test_block_nerv_loss_runs_on_real_pyav_decoded_frame_pair():
    """End-to-end on a real ``upstream/videos/0.mkv`` frame pair."""
    assert_loss_runs_on_pyav_decoded_pair(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )
