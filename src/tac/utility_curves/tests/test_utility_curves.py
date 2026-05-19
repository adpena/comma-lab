# SPDX-License-Identifier: MIT
"""Tests for tac.utility_curves utility callables.

Each utility module receives ≥4 tests per prompt contract:

  1. basic  — invocation works on canonical input
  2. zero   — degenerate / zero input handled
  3. extreme — edge case (very small / very large / shape boundary)
  4. integration — wires correctly into ``make_action_from_track_callables``
"""
from __future__ import annotations

import pytest
import torch

from tac.unified_action import (
    DualVariables,
    make_action_from_track_callables,
)
from tac.utility_curves import (
    per_byte_master_gradient_utility,
    per_pixel_inverse_variance_utility,
    per_tensor_rate_distortion_utility,
)


# ── per_tensor_rate_distortion ─────────────────────────────────────────────


def test_rd_basic_invocation():
    theta = torch.tensor([0.1, 0.2, 0.5, 1.0], requires_grad=True)
    out = per_tensor_rate_distortion_utility(theta)
    assert out.ndim == 0
    assert torch.isfinite(out)
    # Non-zero variance + spread inputs → some channels are below σ² → positive rate
    assert float(out) >= 0.0


def test_rd_zero_numel_returns_zero():
    theta = torch.zeros(0)
    out = per_tensor_rate_distortion_utility(theta)
    assert out.ndim == 0
    assert float(out) == 0.0


def test_rd_rejects_non_1d_input():
    theta = torch.randn(4, 5)
    with pytest.raises(ValueError, match="1-D"):
        per_tensor_rate_distortion_utility(theta)


def test_rd_extreme_uniform_input_zero_variance():
    # All-equal input → variance=0 → eps + 0 → all channels clamped to 0
    theta = torch.ones(8)
    out = per_tensor_rate_distortion_utility(theta)
    assert torch.isfinite(out)


def test_rd_rejects_non_finite_dual_weight():
    @dataclass_dummy(lambda_rate=float("nan"))
    class _BadDuals:
        pass

    bad = _BadDuals()
    theta = torch.tensor([0.1, 0.2, 0.5, 1.0])
    with pytest.raises(ValueError, match="finite"):
        per_tensor_rate_distortion_utility(theta, duals=bad)


def test_rd_integration_with_action_factory():
    theta = torch.tensor([0.1, 0.3, 0.7, 1.5], requires_grad=True)
    action = make_action_from_track_callables(
        rate=per_tensor_rate_distortion_utility,
        duals=DualVariables(lambda_rate=2.0),
    )
    S = action.S_total(theta)
    assert S.ndim == 0
    assert S.grad_fn is not None  # autograd hooked
    assert torch.isfinite(S)


def test_rd_autograd_backwards_finite():
    theta = torch.tensor([0.1, 0.3, 0.7, 1.5], requires_grad=True)
    out = per_tensor_rate_distortion_utility(theta)
    out.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()


# ── per_pixel_inverse_variance ──────────────────────────────────────────────


def test_uniward_basic_invocation_2d():
    img = torch.randn(16, 16, requires_grad=True)
    out = per_pixel_inverse_variance_utility(img)
    assert out.ndim == 0
    assert torch.isfinite(out)
    assert float(out) > 0.0  # positive utility values


def test_uniward_basic_invocation_4d_batch():
    img = torch.randn(2, 3, 16, 16, requires_grad=True)
    out = per_pixel_inverse_variance_utility(img)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_uniward_smooth_input_higher_utility_than_textured():
    """Smooth image → high inverse variance → high utility (cost).
    Textured image → low inverse variance → low utility (cheap)."""
    smooth = torch.zeros(16, 16)
    textured = torch.randn(16, 16) * 5.0
    u_smooth = per_pixel_inverse_variance_utility(smooth)
    u_textured = per_pixel_inverse_variance_utility(textured)
    assert float(u_smooth) > float(u_textured)


def test_uniward_rejects_image_smaller_than_kernel():
    img = torch.randn(3, 3)
    with pytest.raises(ValueError, match="smaller than kernel_size"):
        per_pixel_inverse_variance_utility(img, kernel_size=5)


def test_uniward_rejects_even_kernel_size():
    img = torch.randn(16, 16)
    with pytest.raises(ValueError, match="kernel_size"):
        per_pixel_inverse_variance_utility(img, kernel_size=4)


def test_uniward_rejects_3d_input():
    img = torch.randn(3, 16, 16)
    with pytest.raises(ValueError, match="2-D"):
        per_pixel_inverse_variance_utility(img)


def test_uniward_zero_numel_returns_zero():
    img = torch.zeros(0, 0)
    out = per_pixel_inverse_variance_utility(img)
    # Empty tensor short-circuits to zero scalar before kernel-size check.
    assert out.ndim == 0
    assert float(out) == 0.0


def test_uniward_integration_with_action_factory():
    img = torch.randn(16, 16, requires_grad=True)
    action = make_action_from_track_callables(
        seg=per_pixel_inverse_variance_utility,
        duals=DualVariables(lambda_seg=3.0),
    )
    S = action.S_total(img)
    assert S.ndim == 0
    assert S.grad_fn is not None
    assert torch.isfinite(S)


# ── per_byte_master_gradient ────────────────────────────────────────────────


def test_master_grad_basic_invocation_2d():
    archive_bytes = torch.zeros(20)
    grad = torch.randn(20, 3, requires_grad=True)
    out = per_byte_master_gradient_utility(archive_bytes, grad)
    assert out.ndim == 0
    assert float(out) > 0.0


def test_master_grad_basic_invocation_3d():
    archive_bytes = torch.zeros(20)
    grad = torch.randn(20, 5, 3, requires_grad=True)
    out = per_byte_master_gradient_utility(archive_bytes, grad)
    assert out.ndim == 0
    assert float(out) > 0.0


def test_master_grad_zero_input_returns_zero():
    archive_bytes = torch.zeros(20)
    grad = torch.zeros(20, 3)
    out = per_byte_master_gradient_utility(archive_bytes, grad)
    assert float(out) == 0.0


def test_master_grad_rejects_n_bytes_mismatch():
    archive_bytes = torch.zeros(20)
    grad = torch.randn(15, 3)  # mismatched n_bytes
    with pytest.raises(ValueError, match="share n_bytes"):
        per_byte_master_gradient_utility(archive_bytes, grad)


def test_master_grad_rejects_non_1d_archive_bytes():
    archive_bytes = torch.zeros(5, 4)
    grad = torch.randn(5, 3)
    with pytest.raises(ValueError, match="1-D"):
        per_byte_master_gradient_utility(archive_bytes, grad)


def test_master_grad_rejects_1d_or_4d_master_gradient():
    archive_bytes = torch.zeros(10)
    grad_1d = torch.randn(10)
    grad_4d = torch.randn(10, 2, 2, 2)
    with pytest.raises(ValueError, match="2-D"):
        per_byte_master_gradient_utility(archive_bytes, grad_1d)
    with pytest.raises(ValueError, match="2-D"):
        per_byte_master_gradient_utility(archive_bytes, grad_4d)


def test_master_grad_extreme_large_gradient_values_finite():
    archive_bytes = torch.zeros(10)
    grad = torch.full((10, 3), 1e6)
    out = per_byte_master_gradient_utility(archive_bytes, grad)
    assert torch.isfinite(out)


def test_master_grad_integration_with_action_factory():
    archive_bytes = torch.zeros(20)
    grad = torch.randn(20, 3, requires_grad=True)
    # Wrap as a closure so the Action factory's expected signature
    # `Callable[[theta], Tensor]` is satisfied; the closure captures
    # `master_gradient` as `theta` and supplies a synthetic archive_bytes.
    def rate_callable(theta: torch.Tensor) -> torch.Tensor:
        # `theta` here is the master_gradient tensor.
        return per_byte_master_gradient_utility(archive_bytes, theta)

    action = make_action_from_track_callables(
        rate=rate_callable,
        duals=DualVariables(lambda_rate=1.5),
    )
    S = action.S_total(grad)
    assert S.ndim == 0
    assert S.grad_fn is not None
    assert torch.isfinite(S)


# ── tiny helper for the non-finite-dual test (avoid importing the dataclass) ──


def dataclass_dummy(**kwargs):
    """Tiny decorator that attaches kwargs as class attributes."""

    def deco(cls):
        for k, v in kwargs.items():
            setattr(cls, k, v)
        return cls

    return deco


# ── __all__ regression ──────────────────────────────────────────────────


def test_utility_curves_subpackage_exports():
    import tac.utility_curves as uc

    assert set(uc.__all__) == {
        "per_byte_master_gradient_utility",
        "per_pixel_inverse_variance_utility",
        "per_tensor_rate_distortion_utility",
    }
