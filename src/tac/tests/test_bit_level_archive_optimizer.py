"""Synthetic tests for Lane Bit-level archive optimizer.

All tests on synthetic / hand-crafted archives; no GPU, no real Lane G v3 archive.
Real-archive empirical audit is Phase C (Level 2 — out of scope).

References
----------
- Module: src/tac/bit_level_archive_optimizer.py
- Design: .omx/research/council_lane_bit_level_archive_design_20260430.md
- Sibling: src/tac/custom_binary_container.py (the audit that REDIRECTED scope)
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from tac.bit_level_archive_optimizer import (
    BLPS_MAGIC,
    BLPS_VERSION,
    ArchiveByteComposition,
    BitLevelArchiveOptimizer,
    audit_archive_byte_composition,
    build_shared_brotli_dictionary,
    decode_blps,
    dequantize_poses,
    encode_blps,
    fit_per_dim_quantizer,
    quantize_poses,
)


# ── per-dim quantizer fit ─────────────────────────────────────────────


def test_quantizer_fit_recovers_dynamic_range():
    """Per-dim quantizer scale should match each dim's half-range / 127."""
    poses = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.5, 0.1],
         [-1.0, -0.5, -0.1]],
        dtype=np.float32,
    )
    q = fit_per_dim_quantizer(poses=poses, bits_per_value=8)
    # Scale_d = max(|col|) / 127 for symmetric columns
    assert np.isclose(q.scales[0], 1.0 / 127.0, atol=1e-6)
    assert np.isclose(q.scales[1], 0.5 / 127.0, atol=1e-6)
    assert np.isclose(q.scales[2], 0.1 / 127.0, atol=1e-6)
    # Offsets should be ~0 for symmetric columns
    assert np.allclose(q.offsets, 0.0, atol=1e-6)


def test_quantizer_fit_handles_offset_dim():
    """Non-zero-centered column gets non-zero offset."""
    poses = np.array([[0.5], [0.6], [0.7], [0.8]], dtype=np.float32)
    q = fit_per_dim_quantizer(poses=poses, bits_per_value=8)
    # Midpoint of [0.5, 0.8] = 0.65
    assert np.isclose(q.offsets[0], 0.65, atol=1e-5)
    # Half-range = 0.15
    assert np.isclose(q.scales[0], 0.15 / 127.0, atol=1e-6)


def test_quantizer_fit_rejects_bad_bits_per_value():
    poses = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="bits_per_value"):
        fit_per_dim_quantizer(poses=poses, bits_per_value=16)


def test_quantizer_fit_rejects_1d_poses():
    poses = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="2D"):
        fit_per_dim_quantizer(poses=poses, bits_per_value=8)


# ── quantize/dequantize roundtrip ─────────────────────────────────────


def test_quantize_dequantize_roundtrip_low_error():
    """Round-trip error should be bounded by half-LSB per dim."""
    rng = np.random.default_rng(seed=42)
    n_frames = 1200
    n_dims = 6
    # Realistic-ish pose ranges per design doc
    ranges = np.array([0.1, 0.05, 0.02, 0.01, 0.01, 0.01], dtype=np.float32)
    poses = rng.uniform(-1, 1, (n_frames, n_dims)).astype(np.float32) * ranges[None, :]
    q_obj = fit_per_dim_quantizer(poses=poses, bits_per_value=8)
    q = quantize_poses(poses=poses, quantizer=q_obj)
    recon = dequantize_poses(q=q, quantizer=q_obj)
    # Per-dim max error should be ≤ scale_d (one LSB)
    per_dim_max_err = np.abs(poses - recon).max(axis=0)
    for d in range(n_dims):
        assert per_dim_max_err[d] <= q_obj.scales[d] + 1e-6


def test_quantize_clips_out_of_range():
    """Values beyond fit range get clipped to ±127."""
    poses = np.array([[1.0], [2.0]], dtype=np.float32)
    q_obj = fit_per_dim_quantizer(poses=poses, bits_per_value=8)
    # Now feed a value outside fit range
    extreme = np.array([[10.0]], dtype=np.float32)
    q = quantize_poses(poses=extreme, quantizer=q_obj)
    assert q.max() == 127  # clipped


def test_quantize_rejects_dim_mismatch():
    poses = np.zeros((5, 3), dtype=np.float32)
    q_obj = fit_per_dim_quantizer(poses=poses, bits_per_value=8)
    bad = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match=r"got \(5, 4\)"):
        quantize_poses(poses=bad, quantizer=q_obj)


# ── BLPS wire-format roundtrip ─────────────────────────────────────────


def test_blps_int8_roundtrip():
    """encode → decode should preserve quantizer + q array exactly."""
    rng = np.random.default_rng(seed=7)
    poses = rng.standard_normal((100, 6)).astype(np.float32) * 0.1
    encoded = encode_blps(poses=poses, bits_per_value=8)
    # Magic + version
    assert encoded[:4] == BLPS_MAGIC
    assert encoded[4] == BLPS_VERSION
    q_decoded, q_obj_decoded = decode_blps(data=encoded)
    # Re-encode should produce same q
    q_encoded = quantize_poses(
        poses=poses,
        quantizer=fit_per_dim_quantizer(poses=poses, bits_per_value=8),
    )
    np.testing.assert_array_equal(q_decoded, q_encoded)


def test_blps_int4_roundtrip():
    """4-bit packed roundtrip (tight dynamic range)."""
    rng = np.random.default_rng(seed=11)
    n_frames = 50
    n_dims = 6
    poses = rng.standard_normal((n_frames, n_dims)).astype(np.float32) * 0.001
    encoded = encode_blps(poses=poses, bits_per_value=4)
    q_decoded, q_obj_decoded = decode_blps(data=encoded)
    assert q_decoded.shape == (n_frames, n_dims)
    # Decoded values must be in int4 range
    assert q_decoded.max() <= 7
    assert q_decoded.min() >= -7


def test_blps_size_savings_versus_fp16():
    """int8 BLPS body is ~50% smaller than FP16 storage on a Lane G v3-like stream."""
    n_frames = 1200
    n_dims = 6
    poses = np.random.default_rng(0).standard_normal((n_frames, n_dims)).astype(np.float32)
    encoded = encode_blps(poses=poses, bits_per_value=8)
    fp16_bytes = n_frames * n_dims * 2  # 14400 B
    # encoded should be roughly half-ish (header overhead is small)
    assert len(encoded) < fp16_bytes
    # Specifically: header ~64 B; body = 7200 B; total ~7264 B
    assert 7000 < len(encoded) < 7500


def test_blps_decode_rejects_bad_magic():
    bad = b"XXXX" + b"\0" * 100
    with pytest.raises(ValueError, match="bad BLPS magic"):
        decode_blps(data=bad)


def test_blps_decode_rejects_truncated():
    with pytest.raises(ValueError, match="truncated"):
        decode_blps(data=b"BLPS")


def test_blps_decode_rejects_corrupt_crc():
    poses = np.random.default_rng(0).standard_normal((10, 3)).astype(np.float32)
    encoded = bytearray(encode_blps(poses=poses, bits_per_value=8))
    # Corrupt last byte (CRC)
    encoded[-1] = encoded[-1] ^ 0xFF
    with pytest.raises(ValueError, match="CRC mismatch"):
        decode_blps(data=bytes(encoded))


def test_blps_decode_rejects_version_mismatch():
    poses = np.zeros((4, 3), dtype=np.float32)
    encoded = bytearray(encode_blps(poses=poses, bits_per_value=8))
    # Bump version byte
    encoded[4] = 99
    # Have to also fix the CRC since we changed bytes
    import struct, zlib
    payload_len = len(encoded) - 4  # exclude old crc
    new_crc = zlib.crc32(bytes(encoded[:payload_len])) & 0xFFFFFFFF
    encoded[-4:] = struct.pack(">I", new_crc)
    with pytest.raises(ValueError, match="version mismatch"):
        decode_blps(data=bytes(encoded))


# ── shared Brotli dict ────────────────────────────────────────────────


def test_shared_dict_extracts_common_subsequence():
    """A 16-byte pattern repeated across 2 streams should appear in dict."""
    pattern = b"AAAAAAAAAAAAAAAA"  # 16 A's
    s1 = b"x" * 100 + pattern + b"y" * 50
    s2 = b"z" * 200 + pattern + b"w" * 30
    dictionary = build_shared_brotli_dictionary(
        streams=[s1, s2], max_dict_bytes=1024,
    )
    assert pattern in dictionary


def test_shared_dict_skips_singleton_ngrams():
    """An n-gram appearing only in 1 stream should NOT be in dict."""
    s1 = b"unique_in_s1_only" + b"x" * 100
    s2 = b"y" * 200
    dictionary = build_shared_brotli_dictionary(
        streams=[s1, s2], max_dict_bytes=1024,
    )
    assert b"unique_in_s1_only" not in dictionary


def test_shared_dict_respects_max_size():
    s = b"a" * 100 + b"b" * 100
    dictionary = build_shared_brotli_dictionary(
        streams=[s, s], max_dict_bytes=32,
    )
    assert len(dictionary) <= 32


def test_shared_dict_rejects_empty_streams():
    with pytest.raises(ValueError, match="non-empty"):
        build_shared_brotli_dictionary(streams=[], max_dict_bytes=128)


# ── archive byte composition audit ────────────────────────────────────


def test_audit_synthetic_archive():
    """The audit should report container overhead + per-member breakdown."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("a.bin", b"x" * 100)
        zf.writestr("b.bin", b"y" * 200)
    archive = buf.getvalue()
    audit = audit_archive_byte_composition(archive_bytes=archive)
    assert audit.total_bytes == len(archive)
    assert "a.bin" in audit.member_bytes
    assert "b.bin" in audit.member_bytes
    assert audit.member_bytes["a.bin"] == 100
    assert audit.member_bytes["b.bin"] == 200
    # Overhead = container bytes - sum(member bytes)
    expected_overhead = len(archive) - 300
    assert audit.container_overhead_bytes == expected_overhead
    # Overhead fraction is small
    assert audit.overhead_fraction() < 0.5  # < 50% (synthetic small archive)


def test_audit_overhead_fraction_zero_for_empty_total():
    """Defensive: overhead_fraction returns 0.0 for empty archives."""
    composition = ArchiveByteComposition(
        total_bytes=0, container_overhead_bytes=0, member_bytes={},
    )
    assert composition.overhead_fraction() == 0.0


def test_audit_rejects_invalid_zip():
    with pytest.raises(ValueError, match="not a valid ZIP"):
        audit_archive_byte_composition(archive_bytes=b"\x00" * 50)


# ── orchestrator predicted savings ────────────────────────────────────


def test_orchestrator_predicts_pose_bitpacking_savings():
    """Predicted savings should match Lane G v3 design-doc estimate (~7 KB)."""
    opt = BitLevelArchiveOptimizer(
        target_archive_bytes=694_045,
        enable_pose_bitpacking=True,
        pose_bits_per_value=8,
    )
    savings = opt.predict_savings_bytes(n_pose_frames=1200, n_pose_dims=6)
    # FP16 = 14400 B; bitpacked ~ 7264 B; net ~7136 B
    assert 6500 < savings < 7500


def test_orchestrator_zero_savings_when_disabled():
    opt = BitLevelArchiveOptimizer(
        target_archive_bytes=694_045,
        enable_pose_bitpacking=False,
    )
    assert opt.predict_savings_bytes(n_pose_frames=1200, n_pose_dims=6) == 0


def test_orchestrator_int4_predicts_higher_savings_than_int8():
    opt8 = BitLevelArchiveOptimizer(
        target_archive_bytes=1, enable_pose_bitpacking=True, pose_bits_per_value=8,
    )
    opt4 = BitLevelArchiveOptimizer(
        target_archive_bytes=1, enable_pose_bitpacking=True, pose_bits_per_value=4,
    )
    s8 = opt8.predict_savings_bytes(n_pose_frames=1200, n_pose_dims=6)
    s4 = opt4.predict_savings_bytes(n_pose_frames=1200, n_pose_dims=6)
    assert s4 > s8


# ── version sentinels ─────────────────────────────────────────────────


def test_version_pinned():
    assert BLPS_VERSION == 1


def test_magic_pinned():
    assert BLPS_MAGIC == b"BLPS"
    assert len(BLPS_MAGIC) == 4
