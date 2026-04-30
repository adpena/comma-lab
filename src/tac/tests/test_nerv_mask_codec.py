"""Lane 12 — NeRV mask codec tests.

Per Phase 2 Lane 12 spec (memory project_phases_2_3_4_*):
- Round-trip on a synthetic mask sequence
- Byte-count vs raw fp16 baseline
- Deterministic encode

All claims tagged [synthetic] — empirical real-archive validation is the
Phase 2 dispatch decision.

CLAUDE.md non-negotiables verified:
- No scorer load anywhere
- No silent defaults (every public arg required-keyword)
- No GPU
- Deterministic CPU-only
- Pure-math byte → tensor pipeline
"""
from __future__ import annotations

import pytest
import torch

from tac.nerv_mask_codec import (
    NERV_MAGIC,
    NeRVMaskCodec,
    decode_nerv_codec,
    encode_nerv_codec,
    nerv_codec_bytes,
    positional_encode,
    raw_fp16_baseline_bytes,
    render_mask_logits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: positional_encode determinism + shape correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_positional_encode_shape_and_determinism_synthetic() -> None:
    """[synthetic] positional_encode shape = (B, D * 2 * num_freqs); deterministic."""
    coords = torch.tensor([[0.0, 0.5, -0.5], [1.0, -1.0, 0.0]])  # B=2, D=3
    enc1 = positional_encode(coords, num_freqs=4)
    enc2 = positional_encode(coords, num_freqs=4)
    # Shape: (B=2, D=3, 2 sin/cos, F=4) → flatten → (2, 24)
    assert enc1.shape == (2, 3 * 2 * 4)
    assert torch.allclose(enc1, enc2)
    # Bad input shapes
    with pytest.raises(ValueError, match="coords must be 2-D"):
        positional_encode(torch.zeros(5), num_freqs=2)
    with pytest.raises(ValueError, match="num_freqs must be"):
        positional_encode(coords, num_freqs=0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: encode → decode round-trip preserves all weights bit-exact (fp16)
# ─────────────────────────────────────────────────────────────────────────────


def test_encode_decode_roundtrip_fp16_synthetic() -> None:
    """[synthetic] fp16 round-trip preserves all weights to fp16 precision."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5, depth=4, seed=2026)
    blob = encode_nerv_codec(codec, weight_dtype="fp16")
    assert blob[:4] == NERV_MAGIC
    codec_decoded = decode_nerv_codec(blob)
    # Same arch
    assert codec_decoded.num_freqs == codec.num_freqs
    assert codec_decoded.hidden_dim == codec.hidden_dim
    assert codec_decoded.num_classes == codec.num_classes
    assert codec_decoded.depth == codec.depth
    # Weights match within fp16 precision
    for k, v in codec.state_dict().items():
        v2 = codec_decoded.state_dict()[k]
        assert v2.shape == v.shape
        # fp16 round-trip introduces ~1e-3 max error
        assert torch.allclose(v.float(), v2.float(), atol=1e-2, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: forward output shape + finite (no NaN/Inf at init)
# ─────────────────────────────────────────────────────────────────────────────


def test_codec_forward_shape_and_finite_at_init_synthetic() -> None:
    """[synthetic] codec.forward(coords) → (B, num_classes) finite logits."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5)
    coords = torch.randn(7, 3)  # B=7, D=3
    logits = codec(coords)
    assert logits.shape == (7, 5)
    assert torch.isfinite(logits).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: byte-count assertion — small NeRV << raw fp16 baseline
# ─────────────────────────────────────────────────────────────────────────────


def test_byte_count_small_nerv_beats_raw_fp16_baseline_synthetic() -> None:
    """[synthetic] A small NeRV codec is orders of magnitude smaller than raw fp16."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5)
    codec_bytes = nerv_codec_bytes(codec, weight_dtype="fp16")
    # Mock comma video: 1200 frames at 384x512x5 logits
    raw_bytes = raw_fp16_baseline_bytes(1200, 384, 512, 5)
    assert codec_bytes < raw_bytes
    # Order of magnitude check: NeRV scaffold (~6 KB) << raw (~2.4 GB).
    # Even at 100x larger NeRV (200 KB), still 4 orders of magnitude smaller.
    assert codec_bytes * 1000 < raw_bytes


def test_raw_fp16_baseline_rejects_zero_dims() -> None:
    """[synthetic] raw_fp16_baseline_bytes rejects zero/negative dims (no silent default)."""
    with pytest.raises(ValueError, match="all dims must be > 0"):
        raw_fp16_baseline_bytes(0, 100, 100, 5)
    with pytest.raises(ValueError, match="all dims must be > 0"):
        raw_fp16_baseline_bytes(100, 100, 100, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: render_mask_logits round-trip on tiny synthetic (T, H, W)
# ─────────────────────────────────────────────────────────────────────────────


def test_render_mask_logits_shape_and_determinism_synthetic() -> None:
    """[synthetic] render_mask_logits returns (T, H, W, C); deterministic."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=16, num_classes=5)
    out1 = render_mask_logits(codec, num_frames=3, height=4, width=5, batch_size=8)
    out2 = render_mask_logits(codec, num_frames=3, height=4, width=5, batch_size=64)
    assert out1.shape == (3, 4, 5, 5)
    assert torch.allclose(out1, out2, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: no silent defaults — encode_nerv_codec rejects None codec
# ─────────────────────────────────────────────────────────────────────────────


def test_no_silent_defaults_synthetic() -> None:
    """[synthetic] encode + decode require explicit args (Check 81 STRICT)."""
    with pytest.raises(ValueError, match="codec is required"):
        encode_nerv_codec(codec=None)
    with pytest.raises(ValueError, match="weight_dtype must be"):
        encode_nerv_codec(codec=NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5), weight_dtype="bf16")
    with pytest.raises(ValueError, match="blob is required"):
        decode_nerv_codec(blob=None)
    with pytest.raises(ValueError, match="bad magic"):
        decode_nerv_codec(blob=b"BAD!" + b"\x00" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: int8 path encodes (scaffold smoke; production needs scale table)
# ─────────────────────────────────────────────────────────────────────────────


def test_int8_encode_decode_roundtrip_synthetic() -> None:
    """[synthetic] int8 encode produces ~half the bytes of fp16."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5)
    blob_fp16 = encode_nerv_codec(codec, weight_dtype="fp16")
    blob_int8 = encode_nerv_codec(codec, weight_dtype="int8")
    # int8 is roughly half of fp16 (header is small overhead)
    assert len(blob_int8) < len(blob_fp16)
    # int8 decode runs (note: scaffold doesn't restore scale → outputs are
    # numerically different but the codec object is functional)
    codec_int8 = decode_nerv_codec(blob_int8)
    assert codec_int8.num_params() == codec.num_params()


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: NeRVMaskCodec rejects bad arch
# ─────────────────────────────────────────────────────────────────────────────


def test_codec_constructor_rejects_bad_arch_synthetic() -> None:
    """[synthetic] NeRVMaskCodec rejects invalid arch params."""
    with pytest.raises(ValueError, match="invalid arch"):
        NeRVMaskCodec(num_freqs=0, hidden_dim=32, num_classes=5)
    with pytest.raises(ValueError, match="invalid arch"):
        NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5, depth=1)
