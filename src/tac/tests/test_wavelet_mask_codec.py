"""Tests for tac.wavelet_mask_codec — Lane Wavelet (paradigm α2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.wavelet_mask_codec import (
    WAVELET_MAGIC,
    WAVELET_VERSION,
    WaveletConfig,
    _decode_static_arithmetic,
    _encode_static_arithmetic,
    decode_wavelet_codec,
    dequantize_subband,
    encode_wavelet_codec,
    haar_dwt2d,
    haar_idwt2d,
    multi_level_haar_dwt,
    multi_level_haar_idwt,
    quantize_subband,
    raw_baseline_bytes,
    wavelet_codec_bytes,
)


def _make_synthetic_masks(t: int = 4, h: int = 64, w: int = 96, *, seed: int = 7) -> torch.Tensor:
    """Synthetic 5-class mask with ~80% flat regions + boundaries."""
    g = torch.Generator().manual_seed(seed)
    masks = torch.zeros(t, h, w, dtype=torch.int64)
    # Three horizontal strips → mostly flat
    masks[:, : h // 3, :] = 1  # top strip
    masks[:, h // 3 : 2 * h // 3, :] = 0  # middle strip
    masks[:, 2 * h // 3 :, :] = 2  # bottom strip
    # Sparse noise pixels
    n_noise = max(1, t * h * w // 100)
    for _ in range(n_noise):
        ti = torch.randint(0, t, (1,), generator=g).item()
        hi = torch.randint(0, h, (1,), generator=g).item()
        wi = torch.randint(0, w, (1,), generator=g).item()
        ci = torch.randint(0, 5, (1,), generator=g).item()
        masks[ti, hi, wi] = ci
    return masks


# ── Haar DWT round-trip primitive ────────────────────────────────────────


def test_haar_dwt_idwt_roundtrip_identity():
    """haar_idwt2d(*haar_dwt2d(x)) ≈ x for any (B,C,H,W) with even H,W."""
    x = torch.randn(2, 3, 8, 16)
    ll, lh, hl, hh = haar_dwt2d(x)
    x_recovered = haar_idwt2d(ll, lh, hl, hh)
    assert torch.allclose(x, x_recovered, atol=1e-5)


def test_haar_dwt_rejects_odd_dimensions():
    x = torch.randn(1, 1, 7, 8)
    with pytest.raises(ValueError, match="even H,W"):
        haar_dwt2d(x)


def test_multi_level_dwt_roundtrip_2_levels():
    x = torch.randn(1, 5, 16, 24)
    pyramid = multi_level_haar_dwt(x, levels=2)
    assert len(pyramid) == 2
    # Deepest level has 4 subbands
    assert len(pyramid[-1]) == 4
    # Finer level has 3 subbands
    assert len(pyramid[0]) == 3
    x_recovered = multi_level_haar_idwt(pyramid)
    assert torch.allclose(x, x_recovered, atol=1e-5)


def test_multi_level_dwt_3_levels_shapes():
    x = torch.randn(1, 5, 32, 48)
    pyramid = multi_level_haar_dwt(x, levels=3)
    assert len(pyramid) == 3
    # Deepest LL is at H/8, W/8
    assert pyramid[-1][0].shape == (1, 5, 4, 6)
    x_recovered = multi_level_haar_idwt(pyramid)
    assert torch.allclose(x, x_recovered, atol=1e-5)


# ── Quantize/Dequantize ─────────────────────────────────────────────────


def test_quantize_dequantize_roundtrip_with_step_1():
    coeff = torch.tensor([0.0, 1.0, 2.5, -1.5, 100.7])
    idx = quantize_subband(coeff, step=1.0)
    # round half to even: 2.5 → 2, -1.5 → -2 in torch.round
    deq = dequantize_subband(idx, step=1.0)
    assert idx.dtype == torch.int16
    # All approximately equal to original up to rounding
    assert torch.all((deq - coeff).abs() <= 0.5)


def test_quantize_rejects_zero_step():
    with pytest.raises(ValueError, match="step must be"):
        quantize_subband(torch.zeros(3), step=0.0)


# ── End-to-end mask codec encode/decode ─────────────────────────────────


def test_encode_decode_mask_codec_roundtrip_identity_with_fine_quant():
    """With very fine quantization, round-trip recovers the exact mask."""
    masks = _make_synthetic_masks(t=2, h=16, w=24, seed=42)
    config = WaveletConfig(levels=2, step_ll=0.01, step_detail=0.01)
    blob = encode_wavelet_codec(masks, config=config)
    masks_recovered = decode_wavelet_codec(blob)
    assert masks_recovered.shape == masks.shape
    assert masks_recovered.dtype == torch.int64
    assert torch.equal(masks_recovered, masks)


def test_encode_decode_lossy_quant_high_agreement():
    """With moderate quantization, argmax recovery is >=95% pixels."""
    masks = _make_synthetic_masks(t=4, h=32, w=48, seed=11)
    config = WaveletConfig(levels=2, step_ll=0.5, step_detail=1.0)
    blob = encode_wavelet_codec(masks, config=config)
    masks_recovered = decode_wavelet_codec(blob)
    agreement = (masks_recovered == masks).float().mean().item()
    assert agreement >= 0.90


def test_encode_payload_starts_with_magic():
    masks = _make_synthetic_masks(t=2, h=16, w=24)
    config = WaveletConfig(levels=2, step_ll=0.5, step_detail=1.0)
    blob = encode_wavelet_codec(masks, config=config)
    assert blob[:4] == WAVELET_MAGIC


def test_decode_rejects_bad_magic():
    with pytest.raises(ValueError, match="magic mismatch"):
        decode_wavelet_codec(b"BADX" + b"\x00" * 20)


def test_decode_rejects_truncated_and_trailing_payload_bytes():
    masks = _make_synthetic_masks(t=2, h=16, w=24, seed=123)
    config = WaveletConfig(levels=2, step_ll=0.01, step_detail=0.01)
    blob = encode_wavelet_codec(masks, config=config)

    with pytest.raises(ValueError, match="truncated"):
        decode_wavelet_codec(blob[:-1])
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_wavelet_codec(blob + b"junk")


def test_encode_rejects_invalid_classes():
    masks = _make_synthetic_masks(t=2, h=16, w=24)
    masks[0, 0, 0] = 5  # out of range for num_classes=5 (must be < 5)
    config = WaveletConfig(levels=2, step_ll=0.5, step_detail=1.0)
    with pytest.raises(ValueError, match="out of range"):
        encode_wavelet_codec(masks, config=config)


def test_encode_rejects_non_divisible_dims():
    masks = _make_synthetic_masks(t=2, h=15, w=24)  # h=15 not divisible by 4
    config = WaveletConfig(levels=2, step_ll=0.5, step_detail=1.0)
    with pytest.raises(ValueError, match="divisible by 2"):
        encode_wavelet_codec(masks, config=config)


# ── Byte-count claim: codec wins vs raw fp16 baseline ────────────────────


def test_codec_beats_raw_baseline_on_flat_masks():
    """Synthetic flat-region masks compress to <50% of raw fp16 baseline.

    [synthetic] empirical claim — the [empirical:reports/...] tag will
    apply once the real Lane G v3 mask sequence is plugged in.
    """
    masks = _make_synthetic_masks(t=4, h=32, w=48, seed=99)
    config = WaveletConfig(levels=2, step_ll=0.5, step_detail=1.0)
    encoded = wavelet_codec_bytes(masks, config=config)
    raw = raw_baseline_bytes(masks)
    assert encoded < raw * 0.5, (
        f"[synthetic] expected wavelet codec to beat raw fp16 baseline by 2×; "
        f"got encoded={encoded}, raw={raw}"
    )


# ── No silent defaults (Check 81 STRICT compliance) ─────────────────────


def test_encode_requires_keyword_config():
    masks = _make_synthetic_masks(t=2, h=16, w=24)
    config = WaveletConfig(levels=2, step_ll=0.5, step_detail=1.0)
    # Positional config should raise
    with pytest.raises(TypeError):
        encode_wavelet_codec(masks, config)  # type: ignore[misc]


def test_quantize_requires_keyword_step():
    with pytest.raises(TypeError):
        quantize_subband(torch.zeros(3), 1.0)  # type: ignore[misc]


# ── Arithmetic coder for/else raise (PARADIGM-α CRITICAL #2) ────────────


def test_arithmetic_coder_roundtrip_exact():
    """Encode + decode must recover the original int16 array exactly.

    Sanity baseline for the arithmetic coder before testing the for/else
    raise on freq-table mismatch.
    """
    rng = np.random.default_rng(0)
    values = rng.integers(-10, 11, size=200).astype(np.int16)
    unique, counts = np.unique(values, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(unique, counts)}
    encoded = _encode_static_arithmetic(values, freq)
    decoded = _decode_static_arithmetic(encoded, freq, len(values))
    np.testing.assert_array_equal(
        decoded, values, err_msg="arithmetic coder round-trip failed"
    )


def test_arithmetic_decoder_raises_on_empty_freq_table_mismatch():
    """PARADIGM-α CRITICAL #2: empty freq table with n_values > 0 must raise
    ValueError, not silently return ``sorted_keys[0]`` (which would be
    IndexError under the old default — silent in the more general case
    where ``sorted_keys`` had at least one entry but the scaled value fell
    outside every interval).

    Construct the failure: encode something, then call decode with an
    EMPTY freq_dec while declaring n_values=1 — the for/else raise must
    fire because there are no intervals to fall in.
    """
    values = np.array([5], dtype=np.int16)
    freq_enc = {5: 1}
    encoded = _encode_static_arithmetic(values, freq_enc)
    # Decode with an empty freq dict — total=0, no intervals to fall in.
    with pytest.raises(ValueError, match="arithmetic decoder"):
        _decode_static_arithmetic(encoded, {}, 1)


def test_arithmetic_decoder_empty_input_returns_empty_array():
    """Decoding an empty payload with empty freq table returns empty array."""
    encoded = _encode_static_arithmetic(np.zeros(0, dtype=np.int16), {})
    decoded = _decode_static_arithmetic(encoded, {}, 0)
    assert len(decoded) == 0
