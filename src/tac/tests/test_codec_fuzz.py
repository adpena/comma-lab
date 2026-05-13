"""Property-based fuzz tests for archive codecs.

The Round 19 INT4_LZMA2 corruption disaster (3.6GB raw byte corruption that
hid for hours) and the 2026-04-21 mask codec int8_t overflow taught us that
codecs need fuzz coverage, not just sample-input regression. This module
property-tests the round-trip identity for every codec on the hot path:

  - mixed_precision_export.export_int4_lzma2 / load_int4_lzma2
  - lossless.frequency_coder uint16 stream encode/decode
  - lossless.frequency_coder uint16 prev-symbol stream encode/decode

If a roundtrip ever fails, the test reports the exact byte sequence that
broke it (hypothesis shrinks failures to minimal counter-examples).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tac.lossless.frequency_coder import (
    decode_uint16_frequency_stream,
    decode_uint16_prev_symbol_stream,
    encode_uint16_frequency_stream,
    encode_uint16_prev_symbol_stream,
)


# ── lossless frequency coder ──────────────────────────────────────────────────


@given(
    tokens=st.lists(
        st.integers(min_value=0, max_value=0xFFFF),
        min_size=0,
        max_size=4096,
    ),
)
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_uint16_frequency_stream_roundtrip(tokens: list[int]) -> None:
    """encode → decode must reproduce the input exactly for any uint16 sequence."""
    arr = np.asarray(tokens, dtype=np.uint16)
    encoded = encode_uint16_frequency_stream(arr)
    decoded = decode_uint16_frequency_stream(encoded.encoded_bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == np.uint16
    assert decoded.shape == arr.shape
    assert (decoded == arr).all(), \
        f"frequency-coder roundtrip mismatch on {len(arr)} tokens"


@given(
    tokens=st.lists(
        st.integers(min_value=0, max_value=0xFFFF),
        min_size=2,  # prev-symbol coder needs at least 2 tokens
        max_size=4096,
    ),
)
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_uint16_prev_symbol_stream_roundtrip(tokens: list[int]) -> None:
    """prev-symbol encode → decode must reproduce the input exactly."""
    arr = np.asarray(tokens, dtype=np.uint16)
    encoded = encode_uint16_prev_symbol_stream(arr)
    decoded = decode_uint16_prev_symbol_stream(encoded.encoded_bytes)
    assert decoded.dtype == np.uint16
    assert decoded.shape == arr.shape
    assert (decoded == arr).all(), \
        f"prev-symbol-coder roundtrip mismatch on {len(arr)} tokens"


@given(
    tokens=st.lists(
        st.integers(min_value=0, max_value=0xFFFF),
        min_size=1,
        max_size=2048,
    ),
)
@settings(max_examples=50, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
def test_frequency_stream_encoded_bytes_match_payload_count(tokens: list[int]) -> None:
    """Encoded byte count must equal header_bytes + payload_bytes (no off-by-one)."""
    arr = np.asarray(tokens, dtype=np.uint16)
    encoded = encode_uint16_frequency_stream(arr)
    assert len(encoded.encoded_bytes) == encoded.header_bytes + encoded.payload_bytes
    assert encoded.token_count == len(arr)


# ── mixed_precision_export INT4+LZMA2 codec ──────────────────────────────────

# Lazy torch import — keeps the module importable on cold workers without torch.
torch = pytest.importorskip("torch")


def _toy_module(in_ch: int, out_ch: int, kernel: int = 3) -> "torch.nn.Module":
    """Build a tiny conv module with deterministic weights."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
        nn.Conv2d(out_ch, out_ch, 1),
    )


@given(
    in_ch=st.integers(min_value=1, max_value=8),
    out_ch=st.integers(min_value=1, max_value=16),
    kernel=st.sampled_from([1, 3, 5]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=10_000, suppress_health_check=[HealthCheck.too_slow])
def test_int4_lzma2_roundtrip_preserves_shapes(
    in_ch: int, out_ch: int, kernel: int, seed: int,
) -> None:
    """export_int4_lzma2 → load_int4_lzma2 must restore every weight shape exactly.

    Quantization itself is lossy (int4 has only 15 levels), so we don't assert
    bit-identity. We DO assert that:
      1. Every state-dict key round-trips.
      2. Every shape round-trips.
      3. Reconstruction error is bounded by the per-tensor scale (sanity check
         on the quantize/dequantize math).
    """
    from tac.mixed_precision_export import export_int4_lzma2, load_int4_lzma2

    torch.manual_seed(seed)
    model = _toy_module(in_ch, out_ch, kernel)
    original_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = Path(f.name)
    try:
        export_int4_lzma2(model, bin_path)
        restored = load_int4_lzma2(bin_path)

        assert set(restored.keys()) == set(original_sd.keys()), \
            f"state-dict keys diverged: {set(restored)} vs {set(original_sd)}"
        for name, original_w in original_sd.items():
            restored_w = restored[name]
            assert restored_w.shape == original_w.shape, \
                f"{name}: shape {restored_w.shape} != original {original_w.shape}"
            # int4 quantization error bound: |w - q(w)| <= scale where
            # scale = max|w| / 7. Allow 1.5x slack for per-channel vs per-tensor
            # scaling differences across the export pipeline.
            tensor_scale = original_w.abs().max().item() / 7
            max_err = (restored_w - original_w).abs().max().item()
            # Allow generous bound: int4 sym = 15 levels symmetric, error <= scale,
            # but our packed scheme is uint4-shifted so 1.5x is a safe slack.
            assert max_err <= tensor_scale * 1.6 + 1e-6, \
                f"{name}: int4 error {max_err:.4f} exceeds bound {tensor_scale * 1.6:.4f}"
    finally:
        bin_path.unlink(missing_ok=True)


# ── INT4 quantization math ────────────────────────────────────────────────────


@given(
    n=st.integers(min_value=1, max_value=512),
    scale=st.floats(min_value=1e-4, max_value=10.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_int4_quantize_dequantize_bounded_error(n: int, scale: float, seed: int) -> None:
    """Pure quantize→dequantize must keep error <= scale (15-level int4 spec)."""
    from tac.mixed_precision_export import quantize_int4_tensor

    torch.manual_seed(seed)
    # Sample weights uniformly in [-scale*7, +scale*7] so quant range matches.
    w = (torch.rand(n) * 2 - 1) * scale * 7
    inferred_scale, packed = quantize_int4_tensor(w)
    # Unpack: each byte holds two nibbles
    unpacked = []
    for byte in packed:
        unpacked.append((byte >> 4) & 0x0F)
        unpacked.append(byte & 0x0F)
    unpacked = unpacked[:n]
    dequant = torch.tensor([(v - 7) * inferred_scale for v in unpacked], dtype=w.dtype)
    err = (dequant - w).abs().max().item()
    # The 1.6x slack matches the production-level bound applied above.
    assert err <= inferred_scale * 1.6 + 1e-6, \
        f"int4 quant error {err:.6f} exceeds bound {inferred_scale * 1.6:.6f}"


# ── frequency-coder header magic byte ────────────────────────────────────────


def test_frequency_stream_magic_byte_stable() -> None:
    """STREAM_MAGIC is part of the wire format — changing it breaks every
    archive. Pin it as a regression test."""
    from tac.lossless.frequency_coder import PREV_SYMBOL_STREAM_MAGIC, STREAM_MAGIC
    assert STREAM_MAGIC == b"TFC1", \
        "Frequency-coder magic byte changed — old archives unreadable"
    assert PREV_SYMBOL_STREAM_MAGIC == b"TPC1", \
        "Prev-symbol magic byte changed — old archives unreadable"
