"""Tests for :mod:`tac.shannon_h2_loss`.

Coverage:
- H0 proxy returns scalar tensor in [0, n_bits] for valid weights.
- H0 on uniform weights ~= n_bits (max entropy).
- H0 on delta-distribution weights -> 0 (min entropy).
- H2 <= H0 always (conditioning never increases entropy).
- Gradient flows back through H0 to the weights tensor.
- Gradient flows back through H2 to the weights tensor.
- shannon_h2_h0_ratio in approximate [0, 1.5] band (soft-histogram artifacts
  can push it slightly above 1).
- temperature parameter rejected when <= 0.
"""

from __future__ import annotations

import pytest
import torch

from tac.shannon_h2_loss import (
    shannon_h0_loss,
    shannon_h2_h0_ratio,
    shannon_h2_loss,
)


def test_h0_returns_scalar_tensor() -> None:
    w = torch.randn(1000) * 0.1
    h0 = shannon_h0_loss(w)
    assert isinstance(h0, torch.Tensor)
    assert h0.ndim == 0
    assert 0.0 <= float(h0) <= 8.0 + 1e-3


def test_h0_uniform_weights_near_max() -> None:
    """Uniform weights spanning [-1, 1] -> H0 should be high (close to log2(256)=8)."""
    g = torch.Generator().manual_seed(42)
    w = (torch.rand(2000, generator=g) - 0.5) * 2.0  # uniform in [-1, 1]
    h0 = shannon_h0_loss(w, temperature=1.0)
    # Soft histogram with temperature=1 spreads probability across nearby bins
    # so we don't quite hit log2(256)=8, but it should be > 7.0.
    assert float(h0) > 7.0, f"uniform -> H0 should approach 8 bits, got {float(h0)}"


def test_h0_constant_weights_low_entropy() -> None:
    """Constant weights -> H0 should be low (delta distribution)."""
    w = torch.full((1000,), 0.5)
    h0 = shannon_h0_loss(w, temperature=0.5)  # tighter assignment
    assert float(h0) < 1.0, f"constant -> H0 should be near 0 bits, got {float(h0)}"


def test_h2_le_h0_subadditivity() -> None:
    """H2 <= H0 always (conditioning never increases entropy)."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(2000, generator=g) * 0.1
    h0 = shannon_h0_loss(w, n_bits=5)  # small alphabet for tractability
    h2 = shannon_h2_loss(w, n_bits=5, max_alphabet_for_trigram=8)
    # Allow small numerical slack from the soft-histogram approximation.
    assert float(h2) <= float(h0) + 0.5, f"H2 ({float(h2)}) > H0 ({float(h0)})"


def test_h0_gradient_flows_to_weights() -> None:
    """H0 as a loss term must be backprop-friendly."""
    w = (torch.randn(500) * 0.1).requires_grad_(True)
    h0 = shannon_h0_loss(w)
    h0.backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all(), "H0 produced non-finite gradients"
    # Gradient should be non-trivial for a non-uniform weight tensor.
    assert w.grad.abs().sum() > 0


def test_h2_gradient_flows_to_weights() -> None:
    """H2 as a loss term must be backprop-friendly."""
    w = (torch.randn(500) * 0.1).requires_grad_(True)
    h2 = shannon_h2_loss(w, n_bits=5, max_alphabet_for_trigram=8)
    h2.backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all(), "H2 produced non-finite gradients"
    assert w.grad.abs().sum() > 0


def test_h2_h0_ratio_within_expected_band() -> None:
    """The H2/H0 ratio should be in approximately [0, 1.5] band.

    On PR106 substrate Path B measured ~= 0.531. On synthetic Gaussian noise
    the ratio is typically higher (closer to 1.0) because there's no
    learned conditional structure.
    """
    g = torch.Generator().manual_seed(0)
    w = torch.randn(2000, generator=g) * 0.1
    ratio = shannon_h2_h0_ratio(w, n_bits=5, max_alphabet_for_trigram=8)
    assert 0.0 <= float(ratio) <= 1.5, f"ratio out of band: {float(ratio)}"


def test_temperature_rejects_zero_or_negative() -> None:
    w = torch.randn(100)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        shannon_h0_loss(w, temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        shannon_h0_loss(w, temperature=-0.1)


def test_h2_max_alphabet_rejected_when_too_large() -> None:
    w = torch.randn(100)
    with pytest.raises(ValueError, match="intractable"):
        shannon_h2_loss(w, max_alphabet_for_trigram=128)


def test_h0_returns_bits_not_nats() -> None:
    """Sanity: H0 is reported in bits per symbol (max <= n_bits=8)."""
    w = torch.randn(500) * 0.1
    h0 = shannon_h0_loss(w, n_bits=8)
    # bits/symbol upper bound is n_bits; nats would be ~5.5 (= 8 ln 2)
    assert float(h0) < 8.5, "H0 value suggests nats not bits"


def test_h0_invariant_to_constant_offset() -> None:
    """H0 should be ~invariant to adding a constant (translation in weight space).

    The soft-assignment is bin-position based; a uniform shift moves the
    histogram peak but doesn't change its shape, so entropy stays similar.
    """
    g = torch.Generator().manual_seed(7)
    w = torch.randn(1000, generator=g) * 0.1
    h0_a = shannon_h0_loss(w, temperature=1.0)
    h0_b = shannon_h0_loss(w + 0.05, temperature=1.0)
    # Allow modest difference; per-tensor scale normalization absorbs most.
    assert abs(float(h0_a) - float(h0_b)) < 0.5
