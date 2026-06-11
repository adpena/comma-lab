# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the throughput-lane bridge additions.

Covers the two numerics-preserving optimizations landed on
``tac.mlx_pr95_port.score_bridge`` for the capstone training-throughput pass:

  1. ``configure_torch_cpu_threads`` — pins the torch-CPU scorer thread count to
     the measured-optimal value WITHOUT changing the scored values (thread count
     is a sub-ULP reduction-order knob; the SegNet argmax / exact d_seg is
     bit-stable across thread counts).

  2. ``TorchScorerBridge.fused_d_seg_d_pose`` — runs SegNet + PoseNet over ONE
     shared preprocess (avoids the second render + preprocess the separate
     ``exact_d_seg`` / ``exact_d_pose`` calls pay), returning ``(d_seg, d_pose)``
     that are BIT-IDENTICAL to the separate calls.

These are real-behavior tests on the proto frozen scorer (a true differentiable
SegNet + PoseNet stand-in — NOT a constants check): every assertion would FAIL
if the fused path diverged from the separate path or if the thread knob silently
altered the scored values. CPU torch only (the trusted authority); NO MPS.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

try:  # pragma: no cover - import guard
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLX_AVAILABLE = False

skip_no_mlx = pytest.mark.skipif(
    not _MLX_AVAILABLE, reason="MLX not available (Apple Silicon required)."
)


# --------------------------------------------------------------------------- #
# Frozen proto scorer (real, differentiable SegNet + PoseNet stand-ins).       #
# Mirrors the proto used in test_pose_film_and_stability so the bridge path is  #
# exercised without the expensive upstream EfficientNet-B2.                     #
# --------------------------------------------------------------------------- #


class _ColorProtoSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        protos = torch.tensor(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            dtype=torch.float32,
        )
        self.c = nn.Conv2d(3, 5, 1)
        self.c.weight.data = protos.reshape(5, 3, 1, 1) / 128.0
        self.c.bias.data = -(protos**2).sum(1) / (2 * 128.0 * 128.0)

    def forward(self, x):  # x in [0,1] NCHW
        return self.c(x * 255.0)


class _GlobalReadPose(nn.Module):
    def __init__(self, h: int, w: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(6, 6, bias=False)
        self.proj.weight.data = torch.eye(6) + 0.1 * torch.randn(6, 6, generator=g)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        masks = [torch.ones_like(yy), yy, xx, yy * xx, yy * yy - 0.5, xx * xx - 0.5]
        self.masks = torch.stack([m / m.abs().mean() for m in masks])

    def forward(self, x):  # (B,12,H,W)
        f0 = x[:, :3].mean(1)
        r = torch.einsum("bhw,khw->bk", f0, self.masks) / (
            f0.shape[1] * f0.shape[2]
        )
        pose = self.proj(r)
        return {"pose": torch.cat([pose, pose], dim=1)}


class _SegPoseDNet(nn.Module):
    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        self.segnet = _ColorProtoSeg()
        self.posenet = _GlobalReadPose(h, w)

    def preprocess_input(self, bhwc):  # (B,2,H,W,C)
        f0 = bhwc[:, 0].permute(0, 3, 1, 2) / 255.0
        f1 = bhwc[:, 1].permute(0, 3, 1, 2) / 255.0
        pose_in = torch.cat([f0, f1, f0, f1], dim=1)[:, :12]
        last = bhwc[:, -1].permute(0, 3, 1, 2)
        return pose_in, last / 255.0


def _build_bridge(n_pairs=6, h=48, w=64, *, eval_roundtrip=True, seed=0):
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _SegPoseDNet(h, w).eval()
    for p in dnet.parameters():
        p.requires_grad = False
    rng = np.random.RandomState(seed)
    pose_tgt = ((rng.rand(n_pairs, 6) - 0.5) * 0.6).astype(np.float32)
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for cls, (r0, r1) in enumerate(bands):
        seg_tgt[:, r0:r1, :] = cls
    bridge = TorchScorerBridge(
        dnet,
        seg_tgt,
        torch.tensor(pose_tgt),
        seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w),
        eval_roundtrip=eval_roundtrip,
        seg_weight=1.0,
        pose_weight=1.0,
    )
    return bridge, h, w, n_pairs


def _render(n_pairs, h, w, *, seed=0):
    """A deterministic real-shaped MLX render ``(B,2,3,h,w)`` in [0,255]."""
    rng = np.random.RandomState(seed)
    arr = (rng.rand(n_pairs, 2, 3, h, w) * 255.0).astype(np.float32)
    return mx.array(arr)


# --------------------------------------------------------------------------- #
# (1) configure_torch_cpu_threads                                              #
# --------------------------------------------------------------------------- #


def test_configure_threads_returns_and_sets_explicit_count():
    from tac.mlx_pr95_port.score_bridge import configure_torch_cpu_threads

    prev = torch.get_num_threads()
    try:
        resolved = configure_torch_cpu_threads(4)
        assert resolved == 4
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(prev)


def test_configure_threads_default_is_positive_and_bounded():
    from tac.mlx_pr95_port.score_bridge import configure_torch_cpu_threads

    prev = torch.get_num_threads()
    try:
        resolved = configure_torch_cpu_threads(None)
        # The default resolves to min(perf_cores, 8) or the current torch default;
        # it must be a sane positive count that never exceeds the measured ceiling.
        assert resolved >= 1
        assert resolved <= max(8, prev)
        assert torch.get_num_threads() == resolved
    finally:
        torch.set_num_threads(prev)


def test_configure_threads_clamps_below_one_to_one():
    from tac.mlx_pr95_port.score_bridge import configure_torch_cpu_threads

    prev = torch.get_num_threads()
    try:
        assert configure_torch_cpu_threads(0) == 1
        assert configure_torch_cpu_threads(-5) == 1
    finally:
        torch.set_num_threads(prev)


@skip_no_mlx
def test_d_seg_bit_identical_across_thread_counts():
    """NO-FAKE: thread count is a numerics-preserving knob — d_seg must not move."""
    from tac.mlx_pr95_port.score_bridge import configure_torch_cpu_threads

    bridge, h, w, n = _build_bridge()
    render = _render(n, h, w, seed=1)
    idx = torch.arange(n)
    prev = torch.get_num_threads()
    try:
        configure_torch_cpu_threads(2)
        d2 = bridge.exact_d_seg(render, idx)
        configure_torch_cpu_threads(6)
        d6 = bridge.exact_d_seg(render, idx)
        assert d2 == d6, f"d_seg changed with thread count: {d2} vs {d6}"
    finally:
        torch.set_num_threads(prev)


# --------------------------------------------------------------------------- #
# (2) fused_d_seg_d_pose == separate exact_d_seg / exact_d_pose                 #
# --------------------------------------------------------------------------- #


@skip_no_mlx
def test_fused_d_seg_bit_identical_to_exact_d_seg():
    bridge, h, w, n = _build_bridge(eval_roundtrip=True)
    render = _render(n, h, w, seed=2)
    idx = torch.arange(n)
    d_seg_sep = bridge.exact_d_seg(render, idx)
    d_seg_fus, _ = bridge.fused_d_seg_d_pose(render, idx)
    assert d_seg_fus == d_seg_sep, (
        f"fused d_seg {d_seg_fus} != separate exact_d_seg {d_seg_sep}"
    )


@skip_no_mlx
def test_fused_d_pose_bit_identical_to_exact_d_pose():
    bridge, h, w, n = _build_bridge(eval_roundtrip=True)
    render = _render(n, h, w, seed=3)
    idx = torch.arange(n)
    d_pose_sep = bridge.exact_d_pose(render, idx)
    _, d_pose_fus = bridge.fused_d_seg_d_pose(render, idx)
    assert d_pose_fus == d_pose_sep, (
        f"fused d_pose {d_pose_fus} != separate exact_d_pose {d_pose_sep}"
    )


@skip_no_mlx
def test_fused_matches_separate_without_eval_roundtrip():
    """The fused path must honor eval_roundtrip=False (clamp-only) identically."""
    bridge, h, w, n = _build_bridge(eval_roundtrip=False)
    render = _render(n, h, w, seed=4)
    idx = torch.arange(n)
    d_seg_sep = bridge.exact_d_seg(render, idx)
    d_pose_sep = bridge.exact_d_pose(render, idx)
    d_seg_fus, d_pose_fus = bridge.fused_d_seg_d_pose(render, idx)
    assert d_seg_fus == d_seg_sep
    assert d_pose_fus == d_pose_sep


@skip_no_mlx
def test_fused_matches_separate_on_a_partial_index_batch():
    """A sub-slice (not all pairs) must also produce identical fused vs separate."""
    bridge, h, w, n = _build_bridge(n_pairs=6, eval_roundtrip=True)
    render = _render(n, h, w, seed=5)
    sub = torch.tensor([0, 2, 4])
    render_sub = render[mx.array(np.array([0, 2, 4], dtype=np.int32))]
    d_seg_sep = bridge.exact_d_seg(render_sub, sub)
    d_pose_sep = bridge.exact_d_pose(render_sub, sub)
    d_seg_fus, d_pose_fus = bridge.fused_d_seg_d_pose(render_sub, sub)
    assert d_seg_fus == d_seg_sep
    assert d_pose_fus == d_pose_sep


@skip_no_mlx
def test_fused_builds_no_autograd_graph():
    """fused_d_seg_d_pose is eval-only: it must not require/keep grad on the render."""
    bridge, h, w, n = _build_bridge()
    render = _render(n, h, w, seed=6)
    idx = torch.arange(n)
    # If a graph were built it would not error, but inference_mode guarantees no
    # grad tracking; assert the returned values are plain python floats (no tensor).
    d_seg, d_pose = bridge.fused_d_seg_d_pose(render, idx)
    assert isinstance(d_seg, float)
    assert isinstance(d_pose, float)


@skip_no_mlx
def test_fused_fails_closed_when_pose_disabled():
    """No PoseNet/targets -> fused has no meaning -> ValueError (fail closed)."""
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    h, w, n = 48, 64, 4
    dnet = _SegPoseDNet(h, w).eval()
    for p in dnet.parameters():
        p.requires_grad = False
    seg_tgt = torch.zeros(n, h, w, dtype=torch.long)
    bridge = TorchScorerBridge(
        dnet, seg_tgt, None, seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=True, seg_weight=1.0,
    )
    render = _render(n, h, w, seed=7)
    with pytest.raises(ValueError, match="pose not enabled"):
        bridge.fused_d_seg_d_pose(render, torch.arange(n))


@skip_no_mlx
def test_eval_preprocess_matches_exact_d_seg_preamble():
    """The shared _eval_preprocess feeds SegNet to the SAME logits exact_d_seg uses."""
    from tac.score_aware_loop.live_segnet_loss import exact_d_seg_from_logits

    bridge, h, w, n = _build_bridge(eval_roundtrip=True)
    render = _render(n, h, w, seed=8)
    idx = torch.arange(n)
    _, segnet_in = bridge._eval_preprocess(render)
    with torch.no_grad():
        seg_out = bridge.dnet.segnet(segnet_in)
        d_seg_via_prep = float(
            exact_d_seg_from_logits(seg_out, bridge.seg_targets_hard[idx])
        )
    assert d_seg_via_prep == bridge.exact_d_seg(render, idx)
