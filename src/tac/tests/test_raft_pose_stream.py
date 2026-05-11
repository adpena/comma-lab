"""Tests for ``tac.raft_pose_stream`` — RAFT optical flow pose stream sidecar.

Per CLAUDE.md HNeRV parity discipline lessons + no-op detector + byte budget +
MPS-forbidden rule. The actual RAFT model is NOT invoked in these tests
(``compute_raft_pose_delta`` is a CUDA-only compress-time path); we test the
wire-format primitives that the inflate runtime consumes.
"""
from __future__ import annotations

import pytest
import torch

from tac.raft_pose_stream import (
    MAX_ENCODED_BYTES,
    NO_OP_VARIANCE_THRESHOLD,
    POSE_DIM,
    PR106_RESIDUAL_MAGIC,
    RAFT_POSE_STREAM_FORMAT_ID,
    RaftPoseStream,
    compute_raft_pose_stream_bytes,
    decode_raft_pose_stream,
    encode_raft_pose_stream,
    is_no_op,
)


def _make_smooth_stream(n_pairs: int = 100, seed: int = 42) -> RaftPoseStream:
    """A smooth-trajectory driving-video-like pose stream."""
    torch.manual_seed(seed)
    deltas = torch.randn(n_pairs, POSE_DIM) * 0.001
    poses = torch.cumsum(deltas, dim=0) + 0.05 * torch.randn(POSE_DIM)
    return RaftPoseStream(poses=poses.float())


def _make_constant_stream(n_pairs: int = 50) -> RaftPoseStream:
    poses = torch.zeros(n_pairs, POSE_DIM)
    return RaftPoseStream(poses=poses)


def test_validate_rejects_bad_shape():
    bad = RaftPoseStream(poses=torch.zeros(10, 5))  # wrong dim
    with pytest.raises(ValueError, match=r"shape \(n_pairs, 6\)"):
        bad.validate()


def test_validate_rejects_empty_stream():
    bad = RaftPoseStream(poses=torch.zeros(0, POSE_DIM))
    with pytest.raises(ValueError, match="at least 1 row"):
        bad.validate()


def test_validate_rejects_nan_poses():
    poses = torch.zeros(5, POSE_DIM)
    poses[2, 3] = float("nan")
    bad = RaftPoseStream(poses=poses)
    with pytest.raises(ValueError, match="non-finite"):
        bad.validate()


def test_encode_decode_round_trip_smooth_trajectory():
    """Smooth trajectory should round-trip within int8 quant error."""
    stream = _make_smooth_stream(n_pairs=100)
    blob = encode_raft_pose_stream(stream)
    decoded = decode_raft_pose_stream(blob)
    assert decoded.n_pairs == stream.n_pairs
    # Frame 0 is fp16-precision exact.
    assert (decoded.poses[0] - stream.poses[0]).abs().max().item() < 1e-2
    # Cumulative quant error bounded by (n_pairs-1) * (per-axis-scale/127).
    err = (decoded.poses - stream.poses).abs().max().item()
    assert err < 0.1, f"cumulative quant error too large: {err}"


def test_encode_decode_round_trip_single_pair():
    stream = RaftPoseStream(poses=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))
    blob = encode_raft_pose_stream(stream)
    decoded = decode_raft_pose_stream(blob)
    assert decoded.n_pairs == 1
    assert (decoded.poses[0] - stream.poses[0]).abs().max().item() < 1e-2


def test_encode_blob_starts_with_correct_magic_and_format_id():
    stream = _make_smooth_stream(n_pairs=10)
    blob = encode_raft_pose_stream(stream)
    assert blob[0] == PR106_RESIDUAL_MAGIC
    assert blob[1] == RAFT_POSE_STREAM_FORMAT_ID


def test_encode_enforces_budget():
    """Massive stream exceeds ≤ 4 KB budget and must raise."""
    big = RaftPoseStream(poses=torch.randn(50000, POSE_DIM))  # very large
    with pytest.raises(ValueError, match=r"> budget|exceeds"):
        encode_raft_pose_stream(big, enforce_budget=True)


def test_encode_budget_disabled_returns_oversized_blob():
    big = RaftPoseStream(poses=torch.randn(20000, POSE_DIM))
    blob = encode_raft_pose_stream(big, enforce_budget=False)
    assert len(blob) > 0
    decoded = decode_raft_pose_stream(blob)
    assert decoded.n_pairs == 20000


def test_decode_rejects_bad_magic():
    stream = _make_smooth_stream(n_pairs=10)
    blob = bytearray(encode_raft_pose_stream(stream))
    blob[0] = 0x00
    with pytest.raises(ValueError, match="magic mismatch"):
        decode_raft_pose_stream(bytes(blob))


def test_decode_rejects_bad_format_id():
    stream = _make_smooth_stream(n_pairs=10)
    blob = bytearray(encode_raft_pose_stream(stream))
    blob[1] = 0x99
    with pytest.raises(ValueError, match="format_id mismatch"):
        decode_raft_pose_stream(bytes(blob))


def test_decode_rejects_truncated_blob():
    stream = _make_smooth_stream(n_pairs=10)
    blob = encode_raft_pose_stream(stream)
    with pytest.raises(ValueError):
        decode_raft_pose_stream(blob[:5])


def test_decode_rejects_trailing_bytes():
    stream = _make_smooth_stream(n_pairs=10)
    blob = encode_raft_pose_stream(stream) + b"\xde\xad"
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_raft_pose_stream(blob)


def test_decode_rejects_wrong_pose_dim_field():
    stream = _make_smooth_stream(n_pairs=10)
    blob = bytearray(encode_raft_pose_stream(stream))
    # pose_dim byte is at offset 2 + 4 = 6.
    blob[6] = 0x07
    with pytest.raises(ValueError, match="pose_dim"):
        decode_raft_pose_stream(bytes(blob))


def test_is_no_op_detects_constant_stream():
    stream = _make_constant_stream()
    assert is_no_op(stream)


def test_is_no_op_returns_false_for_smooth_trajectory():
    stream = _make_smooth_stream(n_pairs=100)
    assert not is_no_op(stream)


def test_no_op_variance_threshold_constant_is_small():
    assert NO_OP_VARIANCE_THRESHOLD > 0.0
    assert NO_OP_VARIANCE_THRESHOLD < 1e-3


def test_compute_bytes_matches_encode_length():
    stream = _make_smooth_stream(n_pairs=50)
    bytes_size = compute_raft_pose_stream_bytes(stream)
    blob = encode_raft_pose_stream(stream, enforce_budget=False)
    assert bytes_size == len(blob)


def test_byte_budget_under_4kb_for_600_pair_typical():
    """600-pair smooth trajectory (production-typical) fits in 4 KB."""
    stream = _make_smooth_stream(n_pairs=600)
    blob = encode_raft_pose_stream(stream)
    assert len(blob) <= MAX_ENCODED_BYTES, f"expected ≤4096 B, got {len(blob)}"


def test_pose_dim_is_six():
    assert POSE_DIM == 6


def test_quantization_per_axis_handles_dimensions_of_different_scales():
    """Per-axis quantization ensures small-scale axes don't lose precision
    when other axes have large magnitudes."""
    poses = torch.zeros(20, POSE_DIM)
    # Axis 0: large; axes 1-5: tiny
    poses[:, 0] = torch.linspace(0.0, 10.0, 20)
    poses[:, 1] = torch.linspace(0.0, 0.001, 20)
    stream = RaftPoseStream(poses=poses)
    blob = encode_raft_pose_stream(stream)
    decoded = decode_raft_pose_stream(blob)
    # Per-axis scale → both axes round-trip with their own quant grid.
    axis0_err = (decoded.poses[:, 0] - poses[:, 0]).abs().max().item()
    axis1_err = (decoded.poses[:, 1] - poses[:, 1]).abs().max().item()
    # axis 0 quant grid ~ 10/127 ~ 0.078
    assert axis0_err < 1.0
    # axis 1 quant grid ~ 0.001/127 ~ 7.87e-6
    assert axis1_err < 0.01
