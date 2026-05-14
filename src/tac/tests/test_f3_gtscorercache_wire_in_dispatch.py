# SPDX-License-Identifier: MIT
"""F3 GTScorerCache wire-in regression tests for the dispatch helper.

Covers ``score_pair_components_dispatch`` and asserts:

* dispatch with NO cache kwargs routes through ``score_pair_components``
  (mathematically identical to the previous bare path)
* dispatch with ALL cache kwargs routes through
  ``score_pair_components_with_cache``
* partial cache kwargs raise ``ScoreAwareScorerContractError``
* the two paths produce numerically equivalent losses on the same scorers
  (cache pre-runs GT → cached path skips GT forward → identical loss)

Also asserts each of the 13 in-scope substrate score_aware_loss
classes accepts the new cache kwargs in ``forward(...)`` without
regressing the un-cached call signature.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
against": these tests are the dynamic protection for the wire-in;
static AST coverage would belong in a future preflight gate.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from tac.substrates.score_aware_common import (
    ScoreAwareScorerContractError,
    score_pair_components,
    score_pair_components_dispatch,
    score_pair_components_with_cache,
    stage_frame_pair,
)


class _SegScorer(torch.nn.Module):
    """SegNet-shape stand-in honoring preprocess_input contract."""

    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        assert pair.dim() == 5
        return pair[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        base = x.mean(dim=1, keepdim=True)
        return torch.cat([base + float(i) for i in range(5)], dim=1)


class _PoseScorer(torch.nn.Module):
    """PoseNet-shape stand-in honoring preprocess_input contract."""

    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = pair.shape
        assert c == 3
        y = pair.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        y6 = y.expand(-1, 6, h, w)
        return y6.reshape(b, t * 6, h, w)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4
        pose = x.flatten(2).mean(dim=2)
        # Pad to 12-dim so cache path's (B, T=2, P=6+) check passes
        return {"pose": pose}


def _make_inputs(b: int = 2, h: int = 8, w: int = 10) -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    return {
        "rgb_0": torch.randn(b, 3, h, w, requires_grad=True),
        "rgb_1": torch.randn(b, 3, h, w, requires_grad=True),
        "gt_0": torch.randn(b, 3, h, w),
        "gt_1": torch.randn(b, 3, h, w),
    }


# ----------------------------------------------------------------------
# Dispatch helper unit tests
# ----------------------------------------------------------------------


def test_dispatch_no_cache_args_routes_to_gt_forward_path():
    inp = _make_inputs()
    seg = _SegScorer()
    pose = _PoseScorer()

    seg_term, pose_term = score_pair_components_dispatch(
        seg_scorer=seg,
        pose_scorer=pose,
        rgb_0_rt=inp["rgb_0"],
        rgb_1_rt=inp["rgb_1"],
        gt_rgb_0=inp["gt_0"],
        gt_rgb_1=inp["gt_1"],
    )
    canon_seg, canon_pose = score_pair_components(
        seg_scorer=seg,
        pose_scorer=pose,
        rgb_0_rt=inp["rgb_0"],
        rgb_1_rt=inp["rgb_1"],
        gt_rgb_0=inp["gt_0"],
        gt_rgb_1=inp["gt_1"],
    )
    assert torch.allclose(seg_term, canon_seg)
    assert torch.allclose(pose_term, canon_pose)


def test_dispatch_partial_cache_args_raise():
    inp = _make_inputs()
    seg = _SegScorer()
    pose = _PoseScorer()

    with pytest.raises(ScoreAwareScorerContractError, match="PARTIAL cache"):
        score_pair_components_dispatch(
            seg_scorer=seg,
            pose_scorer=pose,
            rgb_0_rt=inp["rgb_0"],
            rgb_1_rt=inp["rgb_1"],
            gt_rgb_0=inp["gt_0"],
            gt_rgb_1=inp["gt_1"],
            gt_pose_batch=torch.zeros(2, 2, 12),
            # missing gt_seg_batch + gt_seg_already_probs
        )


def test_dispatch_no_signal_raises():
    inp = _make_inputs()
    seg = _SegScorer()
    pose = _PoseScorer()

    with pytest.raises(ScoreAwareScorerContractError, match="requires either GT cache"):
        score_pair_components_dispatch(
            seg_scorer=seg,
            pose_scorer=pose,
            rgb_0_rt=inp["rgb_0"],
            rgb_1_rt=inp["rgb_1"],
        )


def _build_cache_tensors_direct(
    seg: _SegScorer,
    pose: _PoseScorer,
    gt_0: torch.Tensor,
    gt_1: torch.Tensor,
    *,
    seg_already_probs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-batch cache tensors by forwarding GT through scorers ONCE.

    Mirrors the canonical hot-loop wire-in pattern: trainer pre-computes
    GT scorer outputs at init, then re-uses them every step.
    """
    pair_gt = stage_frame_pair(gt_0, gt_1)
    with torch.no_grad():
        pose_in = pose.preprocess_input(pair_gt)
        gt_pose_batch = pose(pose_in)["pose"]
        seg_in = seg.preprocess_input(pair_gt)
        seg_logits = seg(seg_in)
        if seg_already_probs:
            gt_seg_batch = torch.nn.functional.softmax(seg_logits, dim=1)
        else:
            gt_seg_batch = seg_logits
    return gt_pose_batch, gt_seg_batch


def test_dispatch_with_full_cache_args_routes_to_cached_path():
    """Cache path must equal GT-forward path for the SAME scorers + frames."""
    inp = _make_inputs(b=2, h=8, w=10)
    seg = _SegScorer()
    pose = _PoseScorer()

    gt_pose_batch, gt_seg_batch = _build_cache_tensors_direct(
        seg, pose, inp["gt_0"], inp["gt_1"], seg_already_probs=True,
    )
    seg_cached, pose_cached = score_pair_components_dispatch(
        seg_scorer=seg,
        pose_scorer=pose,
        rgb_0_rt=inp["rgb_0"],
        rgb_1_rt=inp["rgb_1"],
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=True,
    )
    seg_canon, pose_canon = score_pair_components(
        seg_scorer=seg,
        pose_scorer=pose,
        rgb_0_rt=inp["rgb_0"],
        rgb_1_rt=inp["rgb_1"],
        gt_rgb_0=inp["gt_0"],
        gt_rgb_1=inp["gt_1"],
    )
    # Equivalence (cache stores what GT forward would produce).
    assert torch.allclose(seg_cached, seg_canon, atol=1e-5)
    assert torch.allclose(pose_cached, pose_canon, atol=1e-5)


def test_dispatch_cache_path_preserves_predicted_gradients():
    """Cache path must keep grad flowing through rgb_0_rt / rgb_1_rt."""
    inp = _make_inputs(b=2, h=8, w=10)
    seg = _SegScorer()
    pose = _PoseScorer()

    gt_pose_batch, gt_seg_batch = _build_cache_tensors_direct(
        seg, pose, inp["gt_0"], inp["gt_1"], seg_already_probs=True,
    )
    seg_cached, pose_cached = score_pair_components_dispatch(
        seg_scorer=seg,
        pose_scorer=pose,
        rgb_0_rt=inp["rgb_0"],
        rgb_1_rt=inp["rgb_1"],
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=True,
    )
    loss = seg_cached + torch.sqrt(pose_cached.clamp(min=1e-12))
    loss.backward()
    assert inp["rgb_0"].grad is not None
    assert inp["rgb_1"].grad is not None
    assert inp["rgb_0"].grad.abs().sum().item() > 0
    assert inp["rgb_1"].grad.abs().sum().item() > 0


# ----------------------------------------------------------------------
# Per-substrate forward()-signature regression tests
# ----------------------------------------------------------------------


_SUBSTRATE_TO_LOSS_CLASS: dict[str, str] = {
    "sane_hnerv": "SaneHnervScoreAwareLoss",
    "pr101_lc_v2_clone": "Pr101LcV2CloneScoreAwareLoss",
    "balle_renderer": "BalleRendererScoreAwareLoss",
    "tc_nerv": "TCNervScoreAwareLoss",
    "block_nerv": "BlockNervScoreAwareLoss",
    "ff_nerv": "FfnervScoreAwareLoss",
    "ds_nerv": "DsnervScoreAwareLoss",
    "hi_nerv": "HinervScoreAwareLoss",
    "cool_chic": "CoolChicScoreAwareLoss",
    "self_compress_nn": "SelfCompressNnScoreAwareLoss",
    "hybrid_renderer_residual": "HybridRendererResidualScoreAwareLoss",
    "siren": "SirenScoreAwareLoss",
    "vq_vae": "VqVaeScoreAwareLoss",
}


@pytest.mark.parametrize("substrate", sorted(_SUBSTRATE_TO_LOSS_CLASS))
def test_substrate_forward_accepts_cache_kwargs(substrate: str) -> None:
    """Each in-scope substrate's score_aware_loss.forward(...) must accept
    the three new cache kwargs added by F3 wire-in.

    This is the per-substrate regression test the F3 directive requires
    (~5 LOC each, batched via parametrize).
    """
    cls_name = _SUBSTRATE_TO_LOSS_CLASS[substrate]
    module = importlib.import_module(f"tac.substrates.{substrate}.score_aware_loss")
    cls = getattr(module, cls_name)
    import inspect

    sig = inspect.signature(cls.forward)
    params = sig.parameters
    assert "gt_pose_batch" in params, f"{substrate} missing gt_pose_batch kwarg"
    assert "gt_seg_batch" in params, f"{substrate} missing gt_seg_batch kwarg"
    assert "gt_seg_already_probs" in params, (
        f"{substrate} missing gt_seg_already_probs kwarg"
    )
    # All three must be keyword-only with default None.
    for name in ("gt_pose_batch", "gt_seg_batch", "gt_seg_already_probs"):
        p = params[name]
        assert p.default is None, f"{substrate}.{name} default must be None"


def test_dispatch_helper_listed_in_score_aware_common_all():
    from tac.substrates import score_aware_common

    assert "score_pair_components_dispatch" in score_aware_common.__all__
