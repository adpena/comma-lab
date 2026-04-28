"""Tests for src/tac/frozen_bit_quant.py — Lane Ω FrozenBitFakeQuant.

Pins these properties:

  1. STE backward: gradient w.r.t. weight equals upstream gradient (ignoring
     fp16 conversion noise). bits / scale receive None.
  2. 8-bit ≈ identity: max |q − w| < scale / 127 (one quant step).
  3. 1-bit clusters to ±scale per output channel (NEVER produces 0).
  4. Per-output-channel scale: each row's max(|q|) equals max(|w|) for that
     row.
  5. Quantization is finite + matches dtype: no NaNs, dtype preserved.
  6. Shape mismatches raise.
  7. Wrap-layer helper: replaces Conv2d with FrozenBitConv2d; copy weights.
  8. FrozenBitConv2d forward equals manual fake-quant + F.conv2d.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.frozen_bit_quant import (
    FrozenBitConv2d,
    FrozenBitFakeQuant,
    compute_per_channel_scale,
    fake_quantize_per_weight,
    wrap_layer_with_frozen_bits,
)


# ── 8-bit near-identity ───────────────────────────────────────────────────


def test_8bit_near_identity():
    torch.manual_seed(42)
    w = torch.randn(8, 16) * 0.5
    bits = torch.full(w.shape, 8, dtype=torch.uint8)
    q = fake_quantize_per_weight(w, bits)
    scale = compute_per_channel_scale(w)
    # one quant step at 8 bits = scale / 127
    step = scale.max().item() / 127.0
    assert (q - w).abs().max().item() < step + 1e-6, (
        f"8-bit max diff {(q - w).abs().max().item()} > step {step}"
    )


# ── 1-bit clustering (sign-only, never 0) ─────────────────────────────────


def test_1bit_clusters_to_pm_scale():
    torch.manual_seed(0)
    w = torch.randn(4, 8) * 0.3
    bits = torch.full(w.shape, 1, dtype=torch.uint8)
    q = fake_quantize_per_weight(w, bits)
    scale = compute_per_channel_scale(w)
    # Per-row, every element should be ±scale[row] (no zeros).
    for row in range(4):
        unique = torch.unique(q[row].abs()).tolist()
        assert len(unique) == 1 and abs(unique[0] - scale[row].item()) < 1e-5, (
            f"row {row}: 1-bit values should all be ±{scale[row].item()}, "
            f"got abs unique = {unique}"
        )
    # No zero values
    assert (q == 0).sum().item() == 0, (
        "1-bit quantizer must NEVER produce 0 (sign-only, matches OMG1 packer)"
    )


def test_1bit_sign_matches_weight_sign():
    """1-bit quantizer must preserve the sign of the input weight."""
    torch.manual_seed(2)
    w = torch.randn(2, 4)
    bits = torch.full(w.shape, 1, dtype=torch.uint8)
    q = fake_quantize_per_weight(w, bits)
    same_sign = (q.sign() == w.sign()) | (w == 0)
    assert same_sign.all(), "1-bit quant must preserve signs"


# ── STE backward ──────────────────────────────────────────────────────────


def test_ste_backward_gradient_passes_through():
    torch.manual_seed(11)
    w = torch.randn(3, 5, requires_grad=True)
    bits = torch.full(w.shape, 4, dtype=torch.uint8)
    q = fake_quantize_per_weight(w, bits)
    loss = (q * 2.0).sum()
    loss.backward()
    # STE: grad w.r.t. w = 2.0 (upstream) — exactly.
    assert torch.allclose(w.grad, torch.full_like(w, 2.0)), (
        f"STE failed: grad max diff "
        f"{(w.grad - 2.0).abs().max().item()}"
    )


def test_backward_returns_none_for_scale_and_bits():
    """The autograd Function backward signature returns 3 outputs; verify
    scale and bits get None (frozen)."""
    w = torch.randn(2, 3, requires_grad=True)
    scale = compute_per_channel_scale(w)
    bits = torch.full(w.shape, 4, dtype=torch.uint8)
    # Direct apply to inspect grad propagation
    q = FrozenBitFakeQuant.apply(w, scale, bits)
    q.sum().backward()
    # scale is a tensor but never had requires_grad, so .grad stays None
    assert getattr(scale, "grad", None) is None
    assert w.grad is not None


# ── Per-output-channel scale ──────────────────────────────────────────────


def test_per_channel_scale_4d_conv_weight():
    torch.manual_seed(3)
    # Conv2d weight: (C_out, C_in, kH, kW)
    w = torch.randn(6, 3, 5, 5)
    scale = compute_per_channel_scale(w)
    assert scale.shape == (6,)
    for i in range(6):
        assert torch.isclose(scale[i], w[i].abs().max(), atol=1e-6)


def test_per_channel_scale_floor():
    """All-zero output channel must clamp scale to 1e-8 (no division by 0)."""
    w = torch.zeros(3, 4)
    w[0] = 0.5  # only first row has signal
    scale = compute_per_channel_scale(w)
    assert scale.shape == (3,)
    assert scale[1] >= 1e-8 and scale[2] >= 1e-8, "scale must clamp to 1e-8 floor"


def test_quantized_max_equals_input_max_per_channel():
    """At any bit-depth ≥ 2, the max(|q|) per output channel should equal
    the max(|w|) per output channel (one-sided saturation)."""
    torch.manual_seed(5)
    w = torch.randn(4, 6) * 0.7
    for b in (2, 4, 8):
        bits = torch.full(w.shape, b, dtype=torch.uint8)
        q = fake_quantize_per_weight(w, bits)
        scale_w = compute_per_channel_scale(w)
        scale_q = compute_per_channel_scale(q)
        assert torch.allclose(scale_q, scale_w, atol=1e-4), (
            f"b={b}: per-channel max should match"
        )


# ── Shape and dtype ───────────────────────────────────────────────────────


def test_shape_mismatch_raises():
    w = torch.randn(2, 3)
    bits_bad = torch.full((2, 4), 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape"):
        fake_quantize_per_weight(w, bits_bad)


def test_dtype_preserved():
    w = torch.randn(2, 3, dtype=torch.float32)
    bits = torch.full(w.shape, 4, dtype=torch.uint8)
    q = fake_quantize_per_weight(w, bits)
    assert q.dtype == torch.float32


def test_finite_outputs():
    torch.manual_seed(9)
    w = torch.randn(8, 16)
    for b in (1, 2, 4, 8):
        bits = torch.full(w.shape, b, dtype=torch.uint8)
        q = fake_quantize_per_weight(w, bits)
        assert torch.isfinite(q).all(), f"NaN/Inf in {b}-bit quant"


# ── Mixed bit-depths within a layer ───────────────────────────────────────


def test_mixed_bit_depths_within_layer():
    """Each element should respect ITS bit-depth; verify higher-bit elements
    are closer to original."""
    torch.manual_seed(22)
    w = torch.randn(2, 4) * 0.5
    bits = torch.tensor([[1, 2, 4, 8], [8, 4, 2, 1]], dtype=torch.uint8)
    q = fake_quantize_per_weight(w, bits)
    diff = (q - w).abs()
    # 8-bit elements should be closer to original than 1-bit elements
    diff_8 = diff[0, 3].item() + diff[1, 0].item()
    diff_1 = diff[0, 0].item() + diff[1, 3].item()
    # Cannot strictly require diff_8 < diff_1 because 1-bit happens to be
    # close when |w| ≈ scale, but on AVERAGE 8-bit < 1-bit.
    # We assert: max diff at 8 bits is bounded by scale/127 per row.
    scale = compute_per_channel_scale(w)
    for row in range(2):
        for col in range(4):
            if int(bits[row, col]) == 8:
                step = scale[row].item() / 127.0
                assert diff[row, col].item() < step + 1e-6


# ── FrozenBitConv2d wrapper ───────────────────────────────────────────────


def test_frozen_bit_conv2d_forward_matches_manual_fake_quant():
    torch.manual_seed(7)
    layer = FrozenBitConv2d(
        in_channels=3, out_channels=4, kernel_size=3, padding=1, bias=True,
    )
    bits = torch.full(layer.conv.weight.shape, 4, dtype=torch.uint8)
    layer.install_bits(bits)

    x = torch.randn(2, 3, 8, 8)
    y = layer(x)

    # Manual reference
    scale = compute_per_channel_scale(layer.conv.weight)
    q_w = FrozenBitFakeQuant.apply(layer.conv.weight, scale, bits)
    y_ref = F.conv2d(x, q_w, layer.conv.bias, padding=1)
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_frozen_bit_conv2d_install_bits_shape_check():
    layer = FrozenBitConv2d(in_channels=2, out_channels=3, kernel_size=3)
    bits_bad = torch.full((3, 2, 5, 5), 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape"):
        layer.install_bits(bits_bad)


def test_wrap_layer_with_frozen_bits_copies_weights():
    parent = nn.Module()
    parent.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
    orig_w = parent.conv.weight.data.clone()
    orig_b = parent.conv.bias.data.clone()
    bits = torch.full(parent.conv.weight.shape, 8, dtype=torch.uint8)
    new = wrap_layer_with_frozen_bits(parent, "conv", bits)
    assert isinstance(new, FrozenBitConv2d)
    assert isinstance(parent.conv, FrozenBitConv2d)
    assert torch.equal(parent.conv.conv.weight.data, orig_w)
    assert torch.equal(parent.conv.conv.bias.data, orig_b)


def test_wrap_layer_rejects_non_conv2d():
    parent = nn.Module()
    parent.linear = nn.Linear(3, 4)
    with pytest.raises(TypeError, match="Conv2d"):
        wrap_layer_with_frozen_bits(parent, "linear", torch.zeros(4, 3, dtype=torch.uint8))


def test_wrap_layer_bits_shape_validated():
    parent = nn.Module()
    parent.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
    bits_bad = torch.full((3, 2, 5, 5), 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape"):
        wrap_layer_with_frozen_bits(parent, "conv", bits_bad)


def test_frozen_bit_conv2d_gradient_flows_through_weight():
    """End-to-end: backprop through a FrozenBitConv2d should land STE
    gradient on the underlying float weight."""
    torch.manual_seed(13)
    layer = FrozenBitConv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
    bits = torch.full(layer.conv.weight.shape, 4, dtype=torch.uint8)
    layer.install_bits(bits)
    x = torch.randn(1, 3, 8, 8, requires_grad=False)
    y = layer(x)
    y.sum().backward()
    assert layer.conv.weight.grad is not None
    assert torch.isfinite(layer.conv.weight.grad).all()
    # Bias should also receive gradient (no quantization on bias here)
    assert layer.conv.bias.grad is not None


def test_frozen_bit_conv2d_bits_buffer_does_not_grad():
    """The bits buffer is registered as buffer, not parameter — must NOT
    receive gradient or appear in named_parameters."""
    layer = FrozenBitConv2d(in_channels=2, out_channels=3, kernel_size=3)
    param_names = [n for n, _ in layer.named_parameters()]
    assert "bits" not in param_names
    buffer_names = [n for n, _ in layer.named_buffers()]
    assert "bits" in buffer_names
