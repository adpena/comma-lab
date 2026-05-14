# SPDX-License-Identifier: MIT
"""Tests for Lane PFP16 — pose float-16 cast codec.

Covers:
  - basic encode/decode roundtrip exactness within fp16 precision
  - byte count == N * pose_dim * 2 (no header/wrapper)
  - Lane G v3 baseline poses subset roundtrip < 0.06 max-abs error
  - shape / dtype / NaN / inf rejection paths
  - encode_pose_file_pfp16 stats correctness
  - inflate-side compatibility — decoded buffer matches what
    `tac.submission_archive.load_optimized_poses` returns from a raw fp16
    binary (Branch B path)

These tests are pure-CPU, deterministic, no GPU required.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.pfp16_codec import (
    PFP16_FORMAT_SENTINEL,
    PFP16_MAX_ROUNDTRIP_ERROR_TOL,
    decode_pfp16,
    encode_pfp16,
    encode_pose_file_pfp16,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LANE_G_V3_POSES = (
    REPO_ROOT / "experiments" / "results" / "lane_g_v3_landed"
    / "iter_0" / "optimized_poses.pt"
)


# ───────────────────────────────────────────────────────────────────────────
# Basic encode/decode roundtrip
# ───────────────────────────────────────────────────────────────────────────


def test_encode_decode_roundtrip_basic():
    """Round-trip a small synthetic pose tensor; bytes count + shape match."""
    torch.manual_seed(0)
    poses = torch.randn(10, 6) * 5.0  # mild dynamic range
    raw = encode_pfp16(poses)
    assert isinstance(raw, bytes)
    # Byte count == N * pose_dim * 2 with no header/wrapper.
    assert len(raw) == 10 * 6 * 2
    decoded = decode_pfp16(raw, pose_dim=6)
    assert decoded.shape == (10, 6)
    assert decoded.dtype == torch.float32
    # fp16 precision floor at this dynamic range is ~0.005.
    err = (poses - decoded).abs().max().item()
    assert err < 0.05, f"unexpected roundtrip error {err}"


def test_byte_count_invariant_at_lane_g_v3_dimensions():
    """Lane G v3 ships 600 pairs × 6 dims → exactly 7200 bytes."""
    poses = torch.zeros(600, 6)
    raw = encode_pfp16(poses)
    assert len(raw) == 600 * 6 * 2 == 7200


def test_decode_pose_dim_independence():
    """Same bytes decoded with different pose_dim produces correct shapes."""
    poses = torch.randn(20, 4)
    raw = encode_pfp16(poses)
    # Decode with the right pose_dim:
    out4 = decode_pfp16(raw, pose_dim=4)
    assert out4.shape == (20, 4)
    # Decode with pose_dim=2 reshapes the same byte stream to (40, 2):
    out2 = decode_pfp16(raw, pose_dim=2)
    assert out2.shape == (40, 2)


# ───────────────────────────────────────────────────────────────────────────
# Lane G v3 baseline real-data roundtrip
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not LANE_G_V3_POSES.exists(),
    reason="Lane G v3 baseline poses not present (test-only environment).",
)
def test_lane_g_v3_baseline_roundtrip_within_tol():
    """The Lane G v3 baseline poses MUST round-trip below the encoder tol."""
    from tac.submission_archive import load_optimized_poses
    poses = load_optimized_poses(str(LANE_G_V3_POSES), pose_dim=6)
    assert poses.shape == (600, 6)
    raw = encode_pfp16(poses)
    assert len(raw) == 7200
    decoded = decode_pfp16(raw, pose_dim=6)
    err = (poses - decoded).abs().max().item()
    # Empirically measured 2026-04-30: max-abs error 0.0156 on the actual
    # Lane G v3 baseline. The encoder's tol (PFP16_MAX_ROUNDTRIP_ERROR_TOL =
    # 0.06) provides 4x cushion. If THIS test fails, either Lane G v3
    # poses changed shape/range or fp16 precision invariants are broken.
    assert err < PFP16_MAX_ROUNDTRIP_ERROR_TOL, (
        f"Lane G v3 baseline roundtrip error {err:.6f} exceeds "
        f"tol {PFP16_MAX_ROUNDTRIP_ERROR_TOL}"
    )
    # The empirical floor is ~0.0156; any drift well above this is a red
    # flag that bears investigation even if still under the tol.
    assert err < 0.025, (
        f"Lane G v3 roundtrip error {err:.6f} is much larger than "
        f"the empirical floor 0.0156 — investigate."
    )


@pytest.mark.skipif(
    not LANE_G_V3_POSES.exists(),
    reason="Lane G v3 baseline poses not present (test-only environment).",
)
def test_lane_g_v3_savings_vs_fp32_pickle():
    """Lane PFP16 must be smaller than the fp32 pickle baseline."""
    fp32_pickle_size = LANE_G_V3_POSES.stat().st_size
    from tac.submission_archive import load_optimized_poses
    poses = load_optimized_poses(str(LANE_G_V3_POSES), pose_dim=6)
    raw = encode_pfp16(poses)
    fp16_raw_size = len(raw)
    # Lane G v3 fp32 pickle is 15,620B; fp16 raw is 7,200B → ~46% size.
    assert fp16_raw_size < fp32_pickle_size, (
        f"Lane PFP16 ({fp16_raw_size}B) must be smaller than fp32 pickle "
        f"({fp32_pickle_size}B)"
    )
    savings_bytes = fp32_pickle_size - fp16_raw_size
    # Empirical savings on Lane G v3 = 8420B. Allow ±200B slack for
    # zip overhead variance across PyTorch versions.
    assert savings_bytes >= 8000, (
        f"Lane PFP16 savings {savings_bytes}B less than expected ~8420B"
    )


# ───────────────────────────────────────────────────────────────────────────
# Inflate-path compatibility (Branch B of submission_archive)
# ───────────────────────────────────────────────────────────────────────────


def test_decoded_matches_submission_archive_loader(tmp_path):
    """Lane PFP16 raw bytes loaded via submission_archive.load_optimized_poses
    must produce the SAME tensor as a direct decode_pfp16 call. This is
    the inflate-path compatibility guarantee — we ship the same bytes the
    inflate path content-detects and reshapes.
    """
    torch.manual_seed(42)
    poses = torch.randn(30, 6) * 3.0
    raw = encode_pfp16(poses)
    bin_path = tmp_path / "optimized_poses.bin"
    bin_path.write_bytes(raw)

    # Branch B of load_optimized_poses (raw fp16 buffer detection).
    from tac.submission_archive import load_optimized_poses
    via_archive = load_optimized_poses(str(bin_path), pose_dim=6)
    via_codec = decode_pfp16(raw, pose_dim=6)

    assert via_archive.shape == via_codec.shape == (30, 6)
    assert torch.equal(via_archive, via_codec), (
        "submission_archive Branch B and decode_pfp16 must produce "
        "identical tensors from the same raw fp16 bytes — otherwise "
        "the build-side and inflate-side disagree."
    )


# ───────────────────────────────────────────────────────────────────────────
# Validation paths
# ───────────────────────────────────────────────────────────────────────────


def test_encode_rejects_non_tensor():
    with pytest.raises(TypeError, match="must be torch.Tensor"):
        encode_pfp16([[1.0, 2.0]])  # type: ignore[arg-type]


def test_encode_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        encode_pfp16(torch.randn(10))  # 1D
    with pytest.raises(ValueError, match="must be 2-D"):
        encode_pfp16(torch.randn(2, 3, 4))  # 3D


def test_encode_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        encode_pfp16(torch.zeros(0, 6))


def test_encode_rejects_nan_inf():
    bad = torch.zeros(3, 6)
    bad[1, 2] = float("nan")
    with pytest.raises(ValueError, match="NaN or inf"):
        encode_pfp16(bad)
    bad2 = torch.zeros(3, 6)
    bad2[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or inf"):
        encode_pfp16(bad2)


def test_encode_rejects_outside_fp16_range():
    """Values >> fp16 max (~6.5e4) trigger roundtrip tol guard."""
    huge = torch.full((3, 6), 1.0e6)
    with pytest.raises(RuntimeError, match="roundtrip max-abs error"):
        encode_pfp16(huge)


def test_decode_rejects_empty_buffer():
    with pytest.raises(ValueError, match="empty"):
        decode_pfp16(b"", pose_dim=6)


def test_decode_rejects_bad_length():
    # 13 bytes is not a multiple of 6*2=12.
    with pytest.raises(ValueError, match="not a multiple"):
        decode_pfp16(b"\x00" * 13, pose_dim=6)


def test_decode_rejects_zero_pose_dim():
    with pytest.raises(ValueError, match="pose_dim must be positive"):
        decode_pfp16(b"\x00" * 12, pose_dim=0)


def test_decode_rejects_non_bytes():
    with pytest.raises(TypeError, match="must be bytes-like"):
        decode_pfp16("not bytes", pose_dim=6)  # type: ignore[arg-type]


# ───────────────────────────────────────────────────────────────────────────
# encode_pose_file_pfp16 stats
# ───────────────────────────────────────────────────────────────────────────


def test_encode_pose_file_pfp16_stats(tmp_path):
    """File-level encoder produces the expected stats dict."""
    poses = torch.randn(50, 6) * 2.0
    src = tmp_path / "src_optimized_poses.pt"
    torch.save(poses, str(src))
    dst = tmp_path / "dst_optimized_poses.bin"
    stats = encode_pose_file_pfp16(src, dst, pose_dim=6)
    assert stats["n_pairs"] == 50
    assert stats["pose_dim"] == 6
    assert stats["output_bytes"] == 50 * 6 * 2
    assert stats["output_bytes"] < stats["input_bytes"], (
        "fp16 raw must be smaller than fp32 pickle"
    )
    assert stats["savings_bytes"] == stats["input_bytes"] - stats["output_bytes"]
    assert 0.0 < stats["savings_pct"] < 100.0
    assert stats["max_roundtrip_error"] < PFP16_MAX_ROUNDTRIP_ERROR_TOL
    assert stats["mean_roundtrip_error"] <= stats["max_roundtrip_error"]
    assert stats["format_sentinel"] == PFP16_FORMAT_SENTINEL


def test_encode_pose_file_pfp16_writes_canonical_raw_bytes(tmp_path):
    """File-level encoder MUST write raw fp16 bytes (no pickle wrapper).
    A subsequent torch.frombuffer roundtrip must succeed without invoking
    torch.load — this is the inflate-path Branch B contract.
    """
    poses = torch.randn(20, 6)
    src = tmp_path / "src.pt"
    torch.save(poses, str(src))
    dst = tmp_path / "dst.bin"
    encode_pose_file_pfp16(src, dst, pose_dim=6)
    raw = dst.read_bytes()
    # Canonical fp16 raw bytes do NOT start with pickle / zip magic.
    pickle_magics = (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05",
                     b"PK\x03\x04")
    for m in pickle_magics:
        assert not raw.startswith(m), (
            f"Lane PFP16 wire bytes must not start with pickle magic {m!r}"
        )
    decoded = torch.frombuffer(bytearray(raw), dtype=torch.float16) \
        .reshape(-1, 6).float()
    err = (poses.float() - decoded).abs().max().item()
    assert err < PFP16_MAX_ROUNDTRIP_ERROR_TOL


# ───────────────────────────────────────────────────────────────────────────
# Determinism — same input → same bytes (bit-identical)
# ───────────────────────────────────────────────────────────────────────────


def test_encode_is_bit_deterministic():
    torch.manual_seed(7)
    poses = torch.randn(100, 6) * 1.5
    raw1 = encode_pfp16(poses)
    raw2 = encode_pfp16(poses)
    assert raw1 == raw2, "encode_pfp16 must be bit-deterministic"


def test_encode_handles_dtype_variants():
    """Input poses can be fp32, fp64, or fp16 — output is identical fp16
    bytes when the value content is the same.
    """
    base = torch.randn(8, 6) * 4.0
    raw_f32 = encode_pfp16(base.float())
    raw_f64 = encode_pfp16(base.double())
    raw_f16 = encode_pfp16(base.half().float())  # cast back to f32 first
    # f32 → f16 cast and f64 → f16 cast yield same bytes (the rounding
    # rule is the same).
    assert raw_f32 == raw_f64
    # The f16 → f32 → f16 path SHOULD be identical to direct f32 → f16
    # because the values are already representable.
    assert raw_f32 == raw_f16
