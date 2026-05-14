# SPDX-License-Identifier: MIT
"""Lane CG tests: calibrated viewing-ray positional encoding."""
from __future__ import annotations

import pytest
import torch

from tac.contrib.calibrated_positional_encoding import CalibratedPositionalEncoding
from tac.renderer import MaskRenderer


def test_forward_shape_and_unit_rays() -> None:
    enc = CalibratedPositionalEncoding()
    rays = enc(batch_size=2, height=24, width=32, device=torch.device("cpu"))
    assert rays.shape == (2, 3, 24, 32)
    norms = rays.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_gradient_flow_through_calibration_scale() -> None:
    enc = CalibratedPositionalEncoding(learnable_scale=True)
    rays = enc(batch_size=1, height=12, width=16, device=torch.device("cpu"))
    loss = rays[:, 0].square().mean()
    loss.backward()
    assert enc.ray_scale.grad is not None


def test_edge_cases_invalid_dimensions_raise() -> None:
    enc = CalibratedPositionalEncoding()
    with pytest.raises(ValueError, match="positive"):
        enc(batch_size=1, height=0, width=16, device=torch.device("cpu"))


def test_determinism_same_seed_same_rays() -> None:
    torch.manual_seed(99)
    a = CalibratedPositionalEncoding()(1, 8, 8, torch.device("cpu"))
    torch.manual_seed(99)
    b = CalibratedPositionalEncoding()(1, 8, 8, torch.device("cpu"))
    assert torch.equal(a, b)


def test_cuda_only_enforcement_raises_on_cpu() -> None:
    enc = CalibratedPositionalEncoding(require_cuda=True)
    with pytest.raises(RuntimeError, match="CUDA"):
        enc(batch_size=1, height=8, width=8, device=torch.device("cpu"))


def test_mask_renderer_accepts_calibrated_encoding() -> None:
    renderer = MaskRenderer(
        base_ch=8,
        mid_ch=12,
        use_calibrated_positional_encoding=True,
    )
    masks = torch.zeros(1, 16, 16, dtype=torch.long)
    out = renderer(masks)
    assert out.shape == (1, 3, 16, 16)

