# SPDX-License-Identifier: MIT
"""Lane HM tests: analytical road-plane homography motion."""
from __future__ import annotations

import pytest
import torch

from tac.contrib.homography_motion import HomographyMotionModule
from tac.renderer import build_renderer


def _soft_masks(batch: int = 2, h: int = 24, w: int = 32) -> torch.Tensor:
    logits = torch.randn(batch, 5, h, w, requires_grad=True)
    return logits.softmax(dim=1)


def test_forward_shape_and_finite_values() -> None:
    m = HomographyMotionModule(output_channels=6)
    mask_t = torch.zeros(2, 24, 32, dtype=torch.long)
    mask_t1 = mask_t.clone()
    out = m(mask_t, mask_t1)
    assert out.shape == (2, 6, 24, 32)
    assert torch.isfinite(out).all()


def test_gradient_flow_with_soft_masks_and_scale() -> None:
    m = HomographyMotionModule(output_channels=2, learn_velocity_scale=True)
    mask_t = _soft_masks()
    mask_t1 = torch.roll(mask_t, shifts=1, dims=-1)
    out = m(mask_t, mask_t1)
    loss = out.square().mean()
    loss.backward()
    assert mask_t.grad is not None or m.velocity_scale.grad is not None
    assert m.velocity_scale.grad is not None


def test_edge_cases_zero_and_max_masks_are_finite() -> None:
    m = HomographyMotionModule(output_channels=2, max_flow_px=12.0)
    zeros = torch.zeros(1, 8, 8, dtype=torch.long)
    maxed = torch.full((1, 8, 8), 4, dtype=torch.long)
    assert torch.isfinite(m(zeros, zeros)).all()
    assert torch.isfinite(m(maxed, maxed)).all()


def test_determinism_same_seed_same_output() -> None:
    torch.manual_seed(7)
    mask_t = torch.randint(0, 5, (1, 16, 16))
    mask_t1 = torch.roll(mask_t, shifts=1, dims=-2)
    a = HomographyMotionModule(output_channels=2)(mask_t, mask_t1)
    torch.manual_seed(7)
    b = HomographyMotionModule(output_channels=2)(mask_t, mask_t1)
    assert torch.equal(a, b)


def test_cuda_only_enforcement_raises_on_cpu() -> None:
    m = HomographyMotionModule(require_cuda=True)
    with pytest.raises(RuntimeError, match="CUDA"):
        m(torch.zeros(1, 4, 4, dtype=torch.long), torch.zeros(1, 4, 4, dtype=torch.long))


def test_renderer_dispatch_accepts_homography_motion() -> None:
    model = build_renderer(
        base_ch=8,
        mid_ch=12,
        motion_hidden=4,
        motion_type="homography_analytical",
        blend_mode="scalar",
    )
    assert isinstance(model.motion, HomographyMotionModule)

