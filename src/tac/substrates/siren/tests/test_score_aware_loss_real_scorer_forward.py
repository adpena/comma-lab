# SPDX-License-Identifier: MIT
"""Real-scorer regression for siren score-aware loss (WAVE-A-1 sister).

Defends against re-introduction of the WWW4 5D-vs-4D shape mismatch.
See ``src/tac/substrates/_shared/score_aware_loss_real_scorer_test_kit.py``.
"""

from __future__ import annotations

import torch

from tac.substrates._shared.score_aware_loss_real_scorer_test_kit import (
    assert_loss_runs_on_pyav_decoded_pair,
    assert_loss_runs_on_real_segnet,
)
from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
)


def _build_loss(seg_scorer, pose_scorer):
    from tac.substrates.siren.score_aware_loss import (
        ScoreAwareLossWeights,
        SirenScoreAwareLoss,
    )

    return SirenScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=ScoreAwareLossWeights(),
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


def test_siren_loss_runs_on_real_segnet_smp_unet_random_init():
    assert_loss_runs_on_real_segnet(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )


def test_siren_loss_runs_on_real_pyav_decoded_frame_pair():
    assert_loss_runs_on_pyav_decoded_pair(
        loss_factory=_build_loss,
        invoke_loss=_invoke_loss,
    )


def test_siren_loss_formula_uses_shared_scorer_components(monkeypatch):
    import tac.differentiable_eval_roundtrip as eval_roundtrip
    import tac.substrates.siren.score_aware_loss as siren_loss

    calls: dict[str, object] = {"roundtrip_count": 0}

    def fake_roundtrip(x: torch.Tensor) -> torch.Tensor:
        calls["roundtrip_count"] = int(calls["roundtrip_count"]) + 1
        return x

    def fake_score_pair_components(**kwargs):
        calls["score_pair_components_kwargs"] = kwargs
        return torch.tensor(0.25), torch.tensor(0.04)

    monkeypatch.setattr(
        eval_roundtrip,
        "apply_eval_roundtrip_during_training",
        fake_roundtrip,
    )
    monkeypatch.setattr(
        siren_loss,
        "score_pair_components",
        fake_score_pair_components,
    )

    seg = object()
    pose = object()
    loss_fn = siren_loss.SirenScoreAwareLoss(
        seg_scorer=seg,
        pose_scorer=pose,
        weights=siren_loss.ScoreAwareLossWeights(),
    )
    rgb_0 = torch.zeros(1, 3, 4, 4)
    rgb_1 = torch.ones(1, 3, 4, 4)
    gt_0 = torch.full((1, 3, 4, 4), 2.0)
    gt_1 = torch.full((1, 3, 4, 4), 3.0)
    bytes_proxy = torch.tensor(37_545_489.0)

    loss, parts = loss_fn(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        bytes_proxy,
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )

    expected = (
        torch.tensor(CONTEST_RATE_WEIGHT)
        + torch.tensor(CONTEST_SEG_WEIGHT * 0.25)
        + torch.tensor(CONTEST_POSE_SQRT_WEIGHT) * torch.sqrt(torch.tensor(0.04))
    )
    assert torch.allclose(loss, expected)
    assert calls["roundtrip_count"] == 2
    kwargs = calls["score_pair_components_kwargs"]
    assert kwargs["seg_scorer"] is seg
    assert kwargs["pose_scorer"] is pose
    assert kwargs["rgb_0_rt"] is rgb_0
    assert kwargs["rgb_1_rt"] is rgb_1
    assert kwargs["gt_rgb_0"] is gt_0
    assert kwargs["gt_rgb_1"] is gt_1
    assert set(parts) == {"rate_term", "seg_term", "pose_term", "loss_total"}
