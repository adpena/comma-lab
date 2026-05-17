# SPDX-License-Identifier: MIT
"""Tests for tac.quantization_wave.fp4_quantization_wave.

The FP4 (E2M1) round-trip is the canonical Quantizr-pattern primitive.
Tests cover:

* the 15-level grid is correct (8 positive + 7 negative)
* the encode/decode round-trip is byte-stable
* the STE preserves gradient flow except at saturation
* the byte-mutation no-op detector catches a single-byte mutation
  (Catalog #272 distinguishing-feature contract)
"""

from __future__ import annotations

import pytest
import torch

from tac.quantization_wave.fp4_quantization_wave import (
    DEFAULT_FP4_LEVELS,
    FP4_NEG_LEVELS,
    FakeQuantFP4,
    QuantizrFP4Quantizer,
    QUANTIZR_FP4_LEVELS_E2M1,
    byte_mutation_smoke_fp4,
    decode_fp4_per_channel,
    encode_fp4_per_channel,
    fake_quant_fp4,
)


def test_e2m1_grid_has_8_positive_levels():
    """Quantizr's E2M1 canonical positive levels {0, 0.5, 1, 1.5, 2, 3, 4, 6}."""
    assert QUANTIZR_FP4_LEVELS_E2M1 == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def test_default_grid_has_15_signed_levels():
    """8 positive + 7 negative (excluding redundant -0.0) = 15."""
    assert len(DEFAULT_FP4_LEVELS) == 15
    assert min(DEFAULT_FP4_LEVELS) == -6.0
    assert max(DEFAULT_FP4_LEVELS) == 6.0
    assert 0.0 in DEFAULT_FP4_LEVELS


def test_neg_levels_count_is_7():
    """Negative-only set excludes zero."""
    assert len(FP4_NEG_LEVELS) == 7
    assert max(FP4_NEG_LEVELS) < 0
    assert min(FP4_NEG_LEVELS) == -6.0


def test_fake_quant_fp4_2d_per_channel():
    torch.manual_seed(0)
    w = torch.randn(36, 28)
    q = fake_quant_fp4(w)
    assert q.shape == w.shape
    # Output should be at the per-channel scaled FP4 grid
    err = (q - w).abs().max().item()
    assert err < 1.0, f"FP4 error {err} too large for randn(36,28)"


def test_fake_quant_fp4_1d_per_tensor():
    """1D tensor (bias) uses per-tensor scale."""
    torch.manual_seed(0)
    bias = torch.randn(36)
    q = fake_quant_fp4(bias)
    assert q.shape == bias.shape


def test_encode_decode_roundtrip_byte_stable():
    """encode -> decode -> encode produces identical wire bytes."""
    torch.manual_seed(1)
    w = torch.randn(36, 28)
    e1 = encode_fp4_per_channel(w)
    d1 = decode_fp4_per_channel(e1)
    e2 = encode_fp4_per_channel(d1)
    assert torch.equal(e1.codewords, e2.codewords)
    assert torch.allclose(e1.scales, e2.scales, atol=1e-3, rtol=1e-3)


def test_encode_decode_preserves_shape():
    """Round-trip preserves the original tensor shape."""
    for shape in [(36, 28), (64, 16, 3, 3), (16, 64)]:
        torch.manual_seed(int(sum(shape)))
        w = torch.randn(*shape)
        e = encode_fp4_per_channel(w)
        d = decode_fp4_per_channel(e)
        assert d.shape == w.shape


def test_encode_wire_size_for_a1_stem():
    """A1's HNeRVDecoder stem is Linear(28, 36*6*8=1728). Encoded size."""
    w = torch.randn(1728, 28)  # stem.weight shape
    e = encode_fp4_per_channel(w)
    n_elements = 1728 * 28  # = 48,384
    expected_codeword_bytes = (n_elements + 1) // 2  # = 24,192
    expected_scale_bytes = 1728 * 2  # fp16 = 3,456
    assert e.codewords.numel() == expected_codeword_bytes
    assert e.scales.numel() == 1728


def test_encode_requires_2d_or_higher():
    """1D tensors must use the int8 path; FP4 per-channel needs >=2D."""
    bias = torch.randn(36)
    with pytest.raises(ValueError, match=">=2D"):
        encode_fp4_per_channel(bias)


def test_byte_mutation_smoke_detects_single_byte_flip():
    """Catalog #272 distinguishing-feature contract: mutating ONE byte
    of the FP4 codewords blob MUST change the decoded tensor."""
    torch.manual_seed(2)
    w = torch.randn(36, 28)
    differs, original, mutated = byte_mutation_smoke_fp4(w, mutate_byte_index=0)
    assert differs, "byte-mutation smoke failed to detect single-byte flip"
    # The mutation should affect at least one position
    assert not torch.equal(original, mutated)


def test_byte_mutation_smoke_changes_only_a_few_elements():
    """A single byte = 2 nibbles = 2 FP4 elements; the mutation should
    affect only those 2 (or fewer if they map to the same level after
    the XOR)."""
    torch.manual_seed(3)
    w = torch.randn(36, 28)
    differs, original, mutated = byte_mutation_smoke_fp4(w, mutate_byte_index=10)
    diff_mask = original != mutated
    n_changed = diff_mask.sum().item()
    # XOR-1 on a 4-bit index can change the FP4 level for AT MOST 2
    # elements (the low and high nibble of the mutated byte). Some
    # XORs may produce no level change if both pre+post nibbles point
    # at the same level (e.g. small-magnitude weights near zero).
    assert n_changed <= 2, f"mutation changed {n_changed} elements, expected <= 2"


def test_quantizr_fp4_quantizer_wraps_model_correctly():
    """QuantizrFP4Quantizer wraps a small model and runs forward with
    FP4-quantized weights via STE."""
    model = torch.nn.Sequential(
        torch.nn.Linear(28, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 8),
    )
    qz = QuantizrFP4Quantizer(model)
    x = torch.randn(4, 28)
    out = qz(x)
    assert out.shape == (4, 8)
    # Quantized layer list should include both Linear modules
    layers = qz.named_quantized_layers()
    assert len(layers) == 2


def test_ste_gradient_flows_through_unsaturated_elements():
    """FakeQuantFP4 STE: gradient is 1 for unsaturated, 0 for saturated."""
    torch.manual_seed(5)
    w = torch.randn(8, 4, requires_grad=True)
    q = fake_quant_fp4(w)
    loss = q.sum()
    loss.backward()
    # Most elements should have gradient ~= 1 (unsaturated through STE)
    assert w.grad is not None
    # At least 50% of elements should have nonzero grad
    assert (w.grad != 0).float().mean().item() >= 0.5
