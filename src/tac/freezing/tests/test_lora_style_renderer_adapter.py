# SPDX-License-Identifier: MIT
"""Tests for ``tac.freezing.lora_style_renderer_adapter``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.freezing.lora_style_renderer_adapter import (
    LoRAAdapterReport,
    LoRARendererAdapter,
)


def test_base_is_frozen_after_construction():
    """Base linear's parameters become non-trainable."""
    base = nn.Linear(16, 8)
    adapter = LoRARendererAdapter(base, rank=2, alpha=4.0)
    for p in adapter.base.parameters():
        assert p.requires_grad is False


def test_adapter_parameters_are_trainable():
    """``lora_A`` and ``lora_B`` start trainable."""
    base = nn.Linear(16, 8)
    adapter = LoRARendererAdapter(base, rank=2, alpha=4.0)
    assert adapter.a.requires_grad is True
    assert adapter.b.requires_grad is True


def test_adapter_initial_contribution_is_zero():
    """At construction, the adapter's delta is exactly zero (B is zero-init)."""
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    adapter = LoRARendererAdapter(base, rank=4, alpha=1.0)
    x = torch.randn(3, 16)
    expected = base(x)
    actual = adapter(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_forward_shape_correct():
    """Forward output shape matches the base linear's output shape."""
    base = nn.Linear(16, 8)
    adapter = LoRARendererAdapter(base, rank=2, alpha=4.0)
    x = torch.randn(5, 16)
    y = adapter(x)
    assert y.shape == (5, 8)


def test_gradient_flows_to_adapter_only():
    """Backprop yields gradient on ``a, b`` but not on base parameters."""
    base = nn.Linear(16, 8)
    adapter = LoRARendererAdapter(base, rank=2, alpha=4.0)
    # Nudge B off zero so the adapter has a non-zero contribution.
    with torch.no_grad():
        adapter.b.fill_(0.1)
    x = torch.randn(3, 16)
    y = adapter(x).sum()
    y.backward()
    assert adapter.a.grad is not None
    assert adapter.b.grad is not None
    for p in adapter.base.parameters():
        assert p.grad is None


def test_identity_initialization_still_has_live_b_gradient():
    """LoRA starts as an identity delta but is not a dead adapter."""
    torch.manual_seed(2)
    base = nn.Linear(16, 8)
    adapter = LoRARendererAdapter(base, rank=2, alpha=4.0)
    x = torch.randn(3, 16)

    y = adapter(x).sum()
    y.backward()

    assert adapter.b.grad is not None
    assert torch.count_nonzero(adapter.b.grad).item() > 0
    # A is expected to receive zero gradient on step 0 because B starts at zero.
    assert adapter.a.grad is not None
    assert torch.count_nonzero(adapter.a.grad).item() == 0


def test_report_byte_cost_correct():
    """Report's parameter counts + byte cost match the structural formula."""
    base = nn.Linear(64, 32)
    adapter = LoRARendererAdapter(base, rank=2, alpha=1.0)
    report = adapter.report()
    assert isinstance(report, LoRAAdapterReport)
    assert report.in_features == 64
    assert report.out_features == 32
    assert report.rank == 2
    # Adapter params = rank * (in + out) = 2 * (64 + 32) = 192
    assert report.adapter_parameters == 192
    # Base params = 64 * 32 + 32 (bias) = 2080
    assert report.base_parameters == 2080


def test_rank_zero_rejected():
    """Rank zero (or negative) is rejected with ``ValueError``."""
    base = nn.Linear(16, 8)
    with pytest.raises(ValueError):
        LoRARendererAdapter(base, rank=0, alpha=1.0)


def test_rank_above_min_dimension_rejected():
    """Rank exceeding ``min(in_features, out_features)`` is rejected."""
    base = nn.Linear(16, 8)
    with pytest.raises(ValueError):
        LoRARendererAdapter(base, rank=16, alpha=1.0)


def test_alpha_scales_adapter_contribution():
    """Adapter contribution scales linearly with alpha."""
    torch.manual_seed(1)
    base = nn.Linear(8, 8)
    adapter1 = LoRARendererAdapter(base, rank=2, alpha=1.0)
    base2 = nn.Linear(8, 8)
    # Copy base weights so the two adapters share the same base.
    base2.load_state_dict(base.state_dict())
    adapter2 = LoRARendererAdapter(base2, rank=2, alpha=4.0)
    # Set identical A, B for the two adapters.
    with torch.no_grad():
        adapter2.a.copy_(adapter1.a)
        adapter2.b.copy_(adapter1.b)
        adapter1.b.fill_(0.5)
        adapter2.b.copy_(adapter1.b)
    x = torch.randn(2, 8)
    y1 = adapter1(x)
    y2 = adapter2(x)
    # y2 - base(x) = 4 * (y1 - base(x))  since alpha=4 vs alpha=1, same rank.
    delta1 = y1 - base(x)
    delta2 = y2 - base2(x)
    assert torch.allclose(delta2, 4.0 * delta1, atol=1e-5)


def test_adapter_compression_ratio_meaningful():
    """For realistic dimensions, adapter << base parameter count."""
    base = nn.Linear(512, 256)
    adapter = LoRARendererAdapter(base, rank=4, alpha=8.0)
    report = adapter.report()
    # Adapter = 4 * (512 + 256) = 3072
    # Base    = 512 * 256 + 256 = 131_328
    # Ratio   ≈ 42×
    assert report.adapter_parameters < report.base_parameters / 30
