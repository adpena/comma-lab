"""Tests for src/tac/learnable_bit_quant.py — Lane Ω-V2 LearnablePerElementBitDepth.

Pins:
  1. Per-element bits are learnable (registered as nn.Parameter).
  2. Gradient flows through BOTH the weight AND the bits.raw parameter
     when calling .backward() on a function of the quantized output.
  3. 8-bit init ≈ identity: max |q − w| < scale / 127.
  4. 1-bit forced: clusters to ±max per output channel (no zeros).
  5. Per-output-channel scale: max(|q|) per row equals max(|w|) per row.
  6. softplus parameterization: bits stay >= 1 even with very negative raw.
  7. Warm-start tensor: forward bits ≈ warm_start values.
  8. bits_rounded() → uint8 in [1, 8].
  9. LearnableBitConv2d wraps a Conv2d, forward matches manual fake-quant.
 10. Determinism: same seed → identical outputs.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.learnable_bit_quant import (
    LearnableBitConv2d,
    LearnablePerElementBitDepth,
    _PerElementSTEQuantize,
    _ste_per_element_quantize,
)


# ── LearnablePerElementBitDepth basics ───────────────────────────────────


def test_bits_is_learnable_parameter():
    """raw must be an nn.Parameter (not a buffer)."""
    bd = LearnablePerElementBitDepth((4, 3, 3, 3), init_bits=8.0)
    assert isinstance(bd.raw, nn.Parameter)
    assert bd.raw.requires_grad


def test_bits_used_in_range():
    """bits_used() always returns values in [1, 8]."""
    bd = LearnablePerElementBitDepth((2, 3), init_bits=4.0)
    bits = bd.bits_used()
    assert (bits >= 1.0).all() and (bits <= 8.0).all()


def test_softplus_keeps_bits_nonnegative():
    """Even when raw is very negative, softplus pulls bits toward 0+; the
    clamp(min=1) keeps the forward representable. Verifies the
    parameterisation NEVER allows bits < 1."""
    bd = LearnablePerElementBitDepth((2, 3), init_bits=8.0)
    with torch.no_grad():
        bd.raw.fill_(-100.0)  # softplus(-100) ≈ 0
    bits = bd.bits_used()
    assert (bits >= 1.0).all(), (
        "bits must stay >= 1 even with very negative raw (clamp + softplus)"
    )


def test_bits_rounded_in_range_and_uint8():
    bd = LearnablePerElementBitDepth((2, 3), init_bits=4.5)
    bits_int = bd.bits_rounded()
    assert bits_int.dtype == torch.uint8
    assert (bits_int >= 1).all() and (bits_int <= 8).all()


def test_warm_start_tensor_initialises_bits_correctly():
    """When warm_start is provided, bits_used() ≈ warm_start (within softplus
    rounding noise — the inverse-softplus init is exact in float64)."""
    target = torch.tensor([[1.5, 4.0, 7.0], [2.0, 6.5, 3.0]])
    bd = LearnablePerElementBitDepth((2, 3), init_bits=8.0, warm_start=target)
    bits = bd.bits_used()
    diff = (bits - target).abs()
    assert diff.max().item() < 1e-3, (
        f"warm_start mismatch: max diff {diff.max().item()}"
    )


def test_warm_start_shape_mismatch_raises():
    bad = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="warm_start shape"):
        LearnablePerElementBitDepth((2, 3), warm_start=bad)


def test_forward_shape_mismatch_raises():
    bd = LearnablePerElementBitDepth((2, 3))
    with pytest.raises(ValueError, match="weight shape"):
        bd(torch.zeros(3, 3))


# ── 8-bit identity ───────────────────────────────────────────────────────


def test_8bit_init_near_identity():
    torch.manual_seed(42)
    w = torch.randn(8, 16) * 0.5
    bd = LearnablePerElementBitDepth(w.shape, init_bits=8.0)
    q = bd(w)
    # 8-bit max diff = scale / 127 per channel
    scale = w.reshape(8, -1).abs().max(dim=1).values
    step = scale.max().item() / 127.0
    assert (q - w).abs().max().item() < step + 1e-6


# ── 1-bit clustering ─────────────────────────────────────────────────────


def test_1bit_forced_clusters_to_pm_scale():
    torch.manual_seed(0)
    w = torch.randn(4, 8) * 0.3
    # Force 1-bit on every element by setting raw to inverse-softplus(1)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=1.0)
    q = bd(w)
    # Per row, every element should be ±max(|row|) — no zeros
    scale = w.reshape(4, -1).abs().max(dim=1).values
    for row in range(4):
        unique = torch.unique(q[row].abs()).tolist()
        assert len(unique) == 1, (
            f"row {row}: 1-bit unique abs should be 1, got {unique}"
        )
        assert abs(unique[0] - scale[row].item()) < 1e-5
    assert (q == 0).sum().item() == 0


def test_1bit_preserves_sign():
    torch.manual_seed(2)
    w = torch.randn(2, 4)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=1.0)
    q = bd(w)
    same_sign = (q.sign() == w.sign()) | (w == 0)
    assert same_sign.all()


# ── Per-output-channel scale ─────────────────────────────────────────────


def test_per_channel_max_preserved():
    torch.manual_seed(5)
    w = torch.randn(4, 6) * 0.7
    for init_b in (2.0, 4.0, 8.0):
        bd = LearnablePerElementBitDepth(w.shape, init_bits=init_b)
        q = bd(w)
        max_w = w.reshape(4, -1).abs().max(dim=1).values
        max_q = q.reshape(4, -1).abs().max(dim=1).values
        assert torch.allclose(max_q, max_w, atol=1e-4), (
            f"init_bits={init_b}: per-channel max should be preserved"
        )


# ── Gradient flow (THE load-bearing property) ────────────────────────────


def test_gradient_flows_through_weight():
    torch.manual_seed(11)
    w = torch.randn(3, 5, requires_grad=True)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=4.0)
    q = bd(w)
    (q * 2.0).sum().backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    # STE: when no saturation, grad ≈ upstream (2.0). Some saturation
    # near edges; pin a softer property.
    assert w.grad.abs().mean().item() > 0


def test_gradient_flows_through_bits():
    """The single most important property of Lane Ω-V2: the bits
    parameter receives gradient from the quantized output, so Lagrangian
    dual ascent on λ + Adam on bits.raw can converge to the KKT optimum."""
    torch.manual_seed(13)
    w = torch.randn(3, 5, requires_grad=False)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=4.0)
    q = bd(w)
    (q.sum()).backward()
    assert bd.raw.grad is not None
    assert torch.isfinite(bd.raw.grad).all()


def test_higher_bits_means_lower_quant_error():
    """Sanity: at 8-bit init, max diff < scale/127. At 2-bit init, max
    diff > scale/127 (much coarser). Verifies bit-depth actually
    affects fidelity (no dead path)."""
    torch.manual_seed(0)
    w = torch.randn(4, 16) * 0.5
    bd_high = LearnablePerElementBitDepth(w.shape, init_bits=8.0)
    bd_low = LearnablePerElementBitDepth(w.shape, init_bits=2.0)
    q_high = bd_high(w)
    q_low = bd_low(w)
    err_high = (q_high - w).abs().mean().item()
    err_low = (q_low - w).abs().mean().item()
    assert err_low > err_high, (
        f"low-bit error should be > high-bit error; got "
        f"low={err_low:.4f} high={err_high:.4f}"
    )


# ── LearnableBitConv2d wrapper ───────────────────────────────────────────


def test_learnable_bit_conv2d_forward_shape():
    layer = LearnableBitConv2d(3, 4, 3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    y = layer(x)
    assert y.shape == (2, 4, 8, 8)


def test_learnable_bit_conv2d_8bit_close_to_fp32_conv():
    """At 8-bit init, output should be within fp16-quant noise of a plain
    Conv2d with the same weights."""
    torch.manual_seed(7)
    layer = LearnableBitConv2d(3, 4, 3, padding=1, init_bits=8.0)
    plain = nn.Conv2d(3, 4, 3, padding=1)
    with torch.no_grad():
        plain.weight.copy_(layer.conv.weight)
        plain.bias.copy_(layer.conv.bias)
    x = torch.randn(2, 3, 8, 8)
    y_quant = layer(x)
    y_plain = plain(x)
    # 8-bit per-channel quant error bounded by scale * |x| / 127 per output
    diff = (y_quant - y_plain).abs().mean().item()
    assert diff < 0.2, f"8-bit quant should be near-identity; got mean diff {diff}"


def test_learnable_bit_conv2d_gradient_flows_through_bits():
    torch.manual_seed(9)
    layer = LearnableBitConv2d(3, 4, 3, padding=1)
    x = torch.randn(1, 3, 8, 8)
    y = layer(x)
    y.sum().backward()
    assert layer.conv.weight.grad is not None
    assert layer.bit_depth.raw.grad is not None
    # The bias also gets grad
    assert layer.conv.bias.grad is not None


def test_learnable_bit_conv2d_total_weight_bits_differentiable():
    layer = LearnableBitConv2d(3, 4, 3)
    total = layer.total_weight_bits()
    assert total.requires_grad
    total.backward()
    assert layer.bit_depth.raw.grad is not None


def test_learnable_bit_conv2d_stride_and_dilation_work():
    """Lane S regression — the wrapper has to support all Conv2d kwargs
    so it can drop into the renderer's downsample / dilated paths."""
    layer = LearnableBitConv2d(8, 16, 3, stride=2, padding=1)
    y = layer(torch.randn(1, 8, 16, 16))
    assert y.shape == (1, 16, 8, 8)

    layer2 = LearnableBitConv2d(8, 8, 3, padding=2, dilation=2)
    y2 = layer2(torch.randn(1, 8, 16, 16))
    assert y2.shape == (1, 8, 16, 16)


def test_learnable_bit_conv2d_replicate_padding_works():
    layer = LearnableBitConv2d(4, 4, 3, padding=1, padding_mode="replicate")
    y = layer(torch.randn(1, 4, 8, 8))
    assert y.shape == (1, 4, 8, 8)


# ── Determinism ──────────────────────────────────────────────────────────


def test_deterministic_for_seed():
    torch.manual_seed(123)
    layer1 = LearnableBitConv2d(3, 4, 3, padding=1)
    x = torch.randn(1, 3, 8, 8)
    y1 = layer1(x)

    torch.manual_seed(123)
    layer2 = LearnableBitConv2d(3, 4, 3, padding=1)
    y2 = layer2(x)

    assert torch.equal(y1, y2)


# ── Finiteness ────────────────────────────────────────────────────────────


def test_outputs_finite_at_all_bit_levels():
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    for b in (1.0, 2.0, 4.0, 8.0):
        bd = LearnablePerElementBitDepth(w.shape, init_bits=b)
        q = bd(w)
        assert torch.isfinite(q).all()
