# SPDX-License-Identifier: MIT
"""Tests for ``tac.fec6_haar_residual`` codec (Ext 5 of fec6 stacking wave).

Design memo: ``.omx/research/fec6_plus_haar_residual_design_20260517.md``
Lane: ``lane_fec6_stacking_wave_5_grammar_extensions_20260517``

Test surface:
- Haar forward + inverse round-trip (orthonormal: f → forward → inverse → f' with f' == f)
- Per-band quantization + dequantization round-trip within int8 tolerance
- encode → decode round-trip preserves all bands within fp16/int8 tolerance
- Byte-deterministic encode (same input → same bytes)
- Wrap / unwrap symmetry
- Negative cases (malformed payload, odd dimensions)

Per CLAUDE.md Catalog #158 byte-determinism + Catalog #287 evidence-tag.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

from tac.fec6_haar_residual import (
    WAVELET_MAGIC,
    HaarResidualDecodeError,
    HaarResidualEncodeError,
    decode_haar_residual_payload,
    dequantize_per_band,
    encode_haar_residual_payload,
    haar_forward_1level,
    haar_inverse_1level,
    quantize_per_band,
    unwrap_fec6_archive_with_haar,
    wrap_fec6_archive_with_haar,
)
from tools.build_fec6_plus_haar_residual_packet import build_packet


def test_haar_forward_inverse_roundtrip_exact_on_4x4():
    """Orthonormal Haar is exact-roundtrip on float32 inputs."""
    rng = np.random.default_rng(seed=42)
    frame = rng.standard_normal((8, 8)).astype(np.float32) * 10.0
    ll, lh, hl, hh = haar_forward_1level(frame)
    assert ll.shape == lh.shape == hl.shape == hh.shape == (4, 4)
    reconstructed = haar_inverse_1level(ll, lh, hl, hh)
    assert np.allclose(reconstructed, frame, atol=1e-5)


def test_haar_forward_rejects_odd_dimensions():
    frame = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="must both be even"):
        haar_forward_1level(frame)
    frame2 = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must both be even"):
        haar_forward_1level(frame2)


def test_haar_forward_rejects_non_2d():
    frame = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2-D"):
        haar_forward_1level(frame)


def test_haar_inverse_rejects_mismatched_bands():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="band shapes must match"):
        haar_inverse_1level(a, b, a, a)


def test_haar_preserves_l2_energy_orthonormality():
    """Orthonormal transform preserves L2 energy: ||f||^2 == ||LL||^2 + ||LH||^2 + ||HL||^2 + ||HH||^2."""
    rng = np.random.default_rng(seed=1)
    frame = rng.standard_normal((16, 16)).astype(np.float32)
    e_f = float((frame ** 2).sum())
    ll, lh, hl, hh = haar_forward_1level(frame)
    e_bands = float((ll**2).sum() + (lh**2).sum() + (hl**2).sum() + (hh**2).sum())
    assert np.isclose(e_f, e_bands, rtol=1e-5)


def test_quantize_dequantize_roundtrip_within_int8_tolerance():
    rng = np.random.default_rng(seed=2)
    band = rng.standard_normal((4, 4, 4)).astype(np.float32) * 5.0
    quant, scale = quantize_per_band(band)
    assert quant.dtype == np.int8
    deq = dequantize_per_band(quant, scale)
    # Tolerance: int8 + fp16 scale; max error is roughly half a scale step (~scale/2)
    assert np.allclose(deq, band, atol=2 * scale)


def test_quantize_zero_band_handled():
    band = np.zeros((2, 4, 4), dtype=np.float32)
    quant, scale = quantize_per_band(band)
    assert quant.shape == band.shape
    assert np.all(quant == 0)
    assert scale > 0  # non-zero scale to allow downstream divide


def test_quantize_rejects_empty_band():
    with pytest.raises(HaarResidualEncodeError, match="empty band"):
        quantize_per_band(np.array([], dtype=np.float32))


def test_encode_decode_roundtrip_preserves_bands_within_quantization():
    rng = np.random.default_rng(seed=3)
    n_frames, residual_h, residual_w = 4, 16, 24
    residuals = rng.standard_normal((n_frames, residual_h, residual_w)).astype(np.float32) * 2.0
    payload = encode_haar_residual_payload(
        residuals=residuals, frame_h=64, frame_w=96
    )
    decoded = decode_haar_residual_payload(payload)
    assert decoded["n_frames"] == n_frames
    assert decoded["h_ll"] == residual_h // 2
    assert decoded["w_ll"] == residual_w // 2
    assert decoded["frame_h"] == 64
    assert decoded["frame_w"] == 96

    # Verify per-band dequantization + inverse-Haar reconstructs each frame
    # within tolerance of the original residual.
    for i in range(n_frames):
        ll = dequantize_per_band(decoded["ll_quant"][i], decoded["scale_ll"])
        lh = dequantize_per_band(decoded["lh_quant"][i], decoded["scale_lh"])
        hl = dequantize_per_band(decoded["hl_quant"][i], decoded["scale_hl"])
        hh = dequantize_per_band(decoded["hh_quant"][i], decoded["scale_hh"])
        reconstructed = haar_inverse_1level(ll, lh, hl, hh)
        # Tolerance: ~quantization step per band; tighter on small bands
        max_scale = max(decoded["scale_ll"], decoded["scale_lh"], decoded["scale_hl"], decoded["scale_hh"])
        assert np.allclose(reconstructed, residuals[i], atol=4 * max_scale)


def test_encode_byte_determinism():
    rng = np.random.default_rng(seed=4)
    residuals = rng.standard_normal((2, 8, 8)).astype(np.float32)
    a = encode_haar_residual_payload(residuals=residuals, frame_h=32, frame_w=32)
    b = encode_haar_residual_payload(residuals=residuals, frame_h=32, frame_w=32)
    assert a == b


def test_encode_rejects_3d_input_with_odd_residual_dims():
    residuals = np.zeros((2, 7, 8), dtype=np.float32)
    with pytest.raises(HaarResidualEncodeError, match="must both be even"):
        encode_haar_residual_payload(residuals=residuals, frame_h=32, frame_w=32)


def test_encode_rejects_2d_input():
    residuals = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(HaarResidualEncodeError, match="must be 3-D"):
        encode_haar_residual_payload(residuals=residuals, frame_h=32, frame_w=32)


def test_decode_rejects_short_payload():
    with pytest.raises(HaarResidualDecodeError, match="too short"):
        decode_haar_residual_payload(b"\x00" * 10)


def test_decode_size_mismatch_via_truncated_payload():
    # Construct a header claiming n=2, h_ll=4, w_ll=4 then a too-short body
    import struct as _struct
    header = _struct.pack("<HHHHH", 2, 4, 4, 32, 32)
    scales = np.zeros(4, dtype=np.float16).tobytes()
    body = b"\x00" * 50  # too short (expected 4 * 2*4*4 = 128 bytes)
    bad = header + scales + body
    with pytest.raises(HaarResidualDecodeError, match="payload size mismatch"):
        decode_haar_residual_payload(bad)


def test_wrap_unwrap_roundtrip():
    rng = np.random.default_rng(seed=5)
    residuals = rng.standard_normal((3, 8, 8)).astype(np.float32)
    payload = encode_haar_residual_payload(residuals=residuals, frame_h=32, frame_w=32)
    base = b"<fake fec6 base bytes>" * 50
    wrapped = wrap_fec6_archive_with_haar(fec6_archive_bytes=base, haar_payload=payload)
    assert wrapped.startswith(base)
    assert wrapped.endswith(payload)
    base_back, payload_back = unwrap_fec6_archive_with_haar(wrapped)
    assert base_back == base
    assert payload_back == payload


def test_unwrap_returns_none_when_no_slot():
    base = b"a fec6 archive with no haar slot" * 10
    base_back, payload_back = unwrap_fec6_archive_with_haar(base)
    assert base_back == base
    assert payload_back is None


def test_wavelet_magic_pinned():
    assert WAVELET_MAGIC == b"FE6W"
    assert len(WAVELET_MAGIC) == 4


def test_unwrap_with_wavelet_magic_substring_in_base_handled_correctly():
    rng = np.random.default_rng(seed=6)
    residuals = rng.standard_normal((2, 8, 8)).astype(np.float32)
    payload = encode_haar_residual_payload(residuals=residuals, frame_h=32, frame_w=32)
    base = b"prefix" + WAVELET_MAGIC + b"random" + WAVELET_MAGIC + b"middle"
    wrapped = wrap_fec6_archive_with_haar(fec6_archive_bytes=base, haar_payload=payload)
    base_back, payload_back = unwrap_fec6_archive_with_haar(wrapped)
    assert base_back == base
    assert payload_back == payload


def test_build_packet_manifest_fails_closed_while_runtime_scaffold_only(tmp_path):
    """The Phase 1 builder must not emit dispatch authority before inflate wiring."""
    fec6_archive = tmp_path / "fec6_source.zip"
    with zipfile.ZipFile(fec6_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"FP11 fake fec6 inner member")

    residuals_npz = tmp_path / "residuals.npz"
    residuals = np.zeros((2, 8, 8), dtype=np.float32)
    residuals[0, 0, 0] = 1.0
    residuals[1, 3, 4] = -0.5
    np.savez(residuals_npz, residuals=residuals)

    output_dir = tmp_path / "packet"
    manifest = build_packet(
        fec6_archive=fec6_archive,
        residuals_npz=residuals_npz,
        output_dir=output_dir,
        frame_h=32,
        frame_w=32,
    )

    assert (output_dir / "archive.zip").is_file()
    assert (output_dir / "inflate.py").is_file()
    assert "NotImplementedError" in (output_dir / "inflate.py").read_text(
        encoding="utf-8"
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_provider_dispatch"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["runtime_scaffold_only"] is True
    assert manifest["runtime_consumption_proof"] is False
    assert manifest["byte_consumption_proof"] is False
    assert manifest["payload_compression"] == "none_raw_int8_phase1_scaffold"
    assert "haar_payload_not_entropy_compressed" in manifest["dispatch_blockers"]
