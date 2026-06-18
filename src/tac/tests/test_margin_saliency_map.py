# SPDX-License-Identifier: MIT
"""NO-FAKE tests for tac.margin_saliency_map.

These verify ACTUAL behavior (real autograd gradient flow through a tiny stub
SegNet, real argmax/margin math, real concentration arithmetic), NOT constants.
If the producer were replaced by ``return zeros``, the gradient-flow +
concentration tests would FAIL.
"""

from __future__ import annotations

import pytest
import torch

from tac.margin_saliency_map import (
    MarginSaliency,
    MarginSaliencyError,
    _gini,
    _topk_margin,
    compute_margin_saliency_map,
    saliency_boundary_concentration,
)


class _StubSegNet:
    """A minimal differentiable SegNet stub: slice last frame, bilinear to a
    small decision grid, then a fixed 1x1 conv to 5 classes. The conv weights
    make some pixels boundary (small margin) and some interior, with a REAL
    dependence on the input so autograd produces a non-trivial saliency."""

    def __init__(self, grid_hw=(8, 12), in_ch=3, classes=5):
        self.grid_hw = grid_hw
        torch.manual_seed(0)
        self.w = torch.randn(classes, in_ch, 1, 1)

    def preprocess_input(self, x):
        x = x[:, -1, ...]  # (B, C, H, W) last frame
        return torch.nn.functional.interpolate(x, size=self.grid_hw, mode="bilinear")

    def __call__(self, seg_in):
        return torch.nn.functional.conv2d(seg_in, self.w)  # (B, 5, H, W)


def _frame(h=16, w=24):
    torch.manual_seed(1)
    return torch.rand(3, h, w) * 255.0


def test_topk_margin_is_nonnegative_and_correct():
    logits = torch.tensor([[[[3.0]], [[1.0]], [[2.0]], [[0.0]], [[-1.0]]]])  # (1,5,1,1)
    m = _topk_margin(logits)
    assert m.shape == (1, 1, 1)
    assert float(m[0, 0, 0]) == pytest.approx(1.0)  # top1=3, top2=2 -> margin 1


def test_topk_margin_rejects_bad_shape():
    with pytest.raises(MarginSaliencyError):
        _topk_margin(torch.zeros(5, 8, 8))  # not (B, C, H, W)


def test_compute_returns_real_grids_on_decision_grid():
    seg = _StubSegNet(grid_hw=(8, 12))
    out = compute_margin_saliency_map(seg, _frame())
    assert isinstance(out, MarginSaliency)
    assert out.grid_hw == (8, 12)
    assert out.saliency.shape == (8, 12)
    assert out.margin.shape == (8, 12)
    # margin is top1-top2 >= 0 always
    assert float(out.margin.min()) >= 0.0


def test_saliency_is_real_gradient_not_zero():
    """The headline NO-FAKE guard: the saliency must be a genuine non-zero
    autograd gradient. If the producer returned zeros/constants this FAILS."""
    seg = _StubSegNet()
    out = compute_margin_saliency_map(seg, _frame())
    assert float(out.saliency.abs().sum()) > 0.0
    # not a constant map (a real gradient varies spatially)
    assert float(out.saliency.std()) > 0.0


def test_saliency_responds_to_input_change():
    """Different input frames produce different saliency (real dependence)."""
    seg = _StubSegNet()
    a = compute_margin_saliency_map(seg, _frame(16, 24)).saliency
    torch.manual_seed(99)
    b = compute_margin_saliency_map(seg, torch.rand(3, 16, 24) * 255.0).saliency
    assert not torch.allclose(a, b)


def test_flip_mask_changes_the_saliency():
    """Targeting a flip-pixel subset gives a different map than the global one."""
    seg = _StubSegNet(grid_hw=(8, 12))
    glob = compute_margin_saliency_map(seg, _frame())
    mask = torch.zeros(8, 12, dtype=torch.bool)
    mask[0:2, 0:2] = True
    targeted = compute_margin_saliency_map(seg, _frame(), flip_pixel_mask=mask)
    assert not torch.allclose(glob.saliency, targeted.saliency)


def test_flip_mask_shape_mismatch_raises():
    seg = _StubSegNet(grid_hw=(8, 12))
    with pytest.raises(MarginSaliencyError):
        compute_margin_saliency_map(
            seg, _frame(), flip_pixel_mask=torch.zeros(4, 4, dtype=torch.bool)
        )


def test_bad_frame_shape_raises():
    seg = _StubSegNet()
    with pytest.raises(MarginSaliencyError):
        compute_margin_saliency_map(seg, torch.zeros(16, 24))  # not (3, H, W)


def test_concentration_arithmetic_real():
    # construct a saliency where boundary (low-margin) pixels carry all mass
    margin = torch.tensor([[0.1, 0.1], [5.0, 5.0]])  # 2 boundary, 2 interior
    sal = torch.tensor([[10.0, 10.0], [0.0, 0.0]])  # all mass at boundary
    c = saliency_boundary_concentration(sal, margin, boundary_margin=0.5)
    assert c["frac_saliency_mass_in_boundary"] == pytest.approx(1.0)
    assert c["mean_saliency_interior"] == pytest.approx(0.0)
    assert c["boundary_over_interior_ratio"] == float("inf")
    assert c["frac_pixels_boundary"] == pytest.approx(0.5)


def test_concentration_uniform_case():
    margin = torch.tensor([[0.1, 0.1], [5.0, 5.0]])
    sal = torch.ones(2, 2)
    c = saliency_boundary_concentration(sal, margin, boundary_margin=0.5)
    assert c["boundary_over_interior_ratio"] == pytest.approx(1.0)
    assert c["frac_saliency_mass_in_boundary"] == pytest.approx(0.5)


def test_gini_extremes():
    assert _gini(torch.ones(100)) == pytest.approx(0.0, abs=1e-6)  # uniform
    spike = torch.zeros(100)
    spike[0] = 1.0
    assert _gini(spike) > 0.9  # concentrated
    assert _gini(torch.zeros(10)) == 0.0  # degenerate


def test_gini_empty():
    assert _gini(torch.tensor([])) == 0.0
