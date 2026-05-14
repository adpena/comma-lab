# SPDX-License-Identifier: MIT
"""Lane PD pose-delta codec roundtrip + sentinel-detection tests."""
from __future__ import annotations

import os
import tempfile

import pytest
import torch

from tac.pose_delta_codec import (
    POSE_DELTA_FORMAT_SENTINEL_V1,
    decode_pose_deltas,
    encode_pose_deltas,
    is_pose_delta_dict,
)
from tac.submission_archive import load_optimized_poses


def _make_smooth_trajectory(n_pairs: int = 600, pose_dim: int = 6) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.cumsum(torch.randn(n_pairs, pose_dim) * 0.001, dim=0) + torch.randn(pose_dim) * 0.01


def test_roundtrip_smooth_trajectory():
    poses = _make_smooth_trajectory()
    encoded = encode_pose_deltas(poses)
    decoded = decode_pose_deltas(encoded, pose_dim=6)
    err = (poses - decoded).abs().max().item()
    assert decoded.shape == poses.shape
    # Smooth driving trajectory + per-channel int8 quantisation should yield
    # < 1e-3 max-abs reconstruction error.
    assert err < 1e-3, f"max-abs error {err:.6e} too large"


def test_sentinel_detection():
    encoded = encode_pose_deltas(_make_smooth_trajectory())
    assert is_pose_delta_dict(encoded)
    assert encoded["format"] == POSE_DELTA_FORMAT_SENTINEL_V1


def test_load_optimized_poses_sentinel_branch():
    """Verify the canonical loader detects the pose-delta sentinel and
    returns a materialised float32 tensor — the inflate-side contract.
    """
    poses = _make_smooth_trajectory()
    encoded = encode_pose_deltas(poses)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(encoded, path)
        recovered = load_optimized_poses(path, pose_dim=6, expected_n_pairs=600)
        assert recovered.shape == poses.shape
        assert recovered.dtype == torch.float32
        assert (poses - recovered).abs().max().item() < 1e-3
    finally:
        os.unlink(path)


def test_short_trajectory_minimum_two_pairs():
    poses = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]])
    encoded = encode_pose_deltas(poses)
    decoded = decode_pose_deltas(encoded, pose_dim=6)
    assert decoded.shape == poses.shape
    assert (poses - decoded).abs().max().item() < 1e-3


def test_single_pair_rejected():
    poses = torch.zeros(1, 6)
    with pytest.raises(ValueError, match="at least 2"):
        encode_pose_deltas(poses)


def test_wrong_pose_dim_rejected_on_decode():
    poses = _make_smooth_trajectory()
    encoded = encode_pose_deltas(poses)
    with pytest.raises(ValueError, match="declared pose_dim"):
        decode_pose_deltas(encoded, pose_dim=12)


def test_unsupported_delta_bits_rejected():
    poses = _make_smooth_trajectory()
    with pytest.raises(NotImplementedError):
        encode_pose_deltas(poses, delta_bits=4)


def test_savings_vs_fp16_baseline():
    """Lane PD claims ~30-50% savings on smooth trajectories. Verify the
    encoded torch.save() blob is meaningfully smaller than a vanilla fp16
    save of the same tensor (the rate-attack motivation).
    """
    import io

    poses = _make_smooth_trajectory(n_pairs=600)
    encoded = encode_pose_deltas(poses)
    buf = io.BytesIO()
    torch.save(encoded, buf)
    encoded_bytes = len(buf.getvalue())

    buf2 = io.BytesIO()
    torch.save(poses.to(torch.float16), buf2)
    fp16_bytes = len(buf2.getvalue())

    # Smooth driving trajectory should give >25% savings over raw fp16.
    savings = 1 - encoded_bytes / fp16_bytes
    assert savings > 0.20, (
        f"expected >20% savings vs fp16; got {savings*100:.1f}% "
        f"(encoded={encoded_bytes}B fp16={fp16_bytes}B)"
    )
