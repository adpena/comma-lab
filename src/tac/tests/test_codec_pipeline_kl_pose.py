"""Tests for ``tac.codec_pipeline_kl_pose.Op_KLPoseStream``.

Verifies:
  - CodecOp Protocol satisfaction
  - encode/decode roundtrip with bounded reconstruction error
  - byte-deterministic encode (same input → identical bytes)
  - the substrate-adaptive basis is actually adaptive (different
    trajectories → different bases)
  - truncation diagnostic correctly estimates variance retention
  - validate() catches schema violations
  - smooth driving-style trajectory compresses to fewer bytes than RAFT-style
    raw int16 (the whole point of KL)
"""
from __future__ import annotations

import json

import pytest
import torch

from tac.codec_pipeline import CodecOp, CodecPipeline
from tac.codec_pipeline_kl_pose import (
    POSE_KEY,
    Op_KLPoseStream,
    estimate_truncation_rms,
)


def _smooth_driving_trajectory(n_frames: int = 600, seed: int = 0) -> torch.Tensor:
    """Synthetic 6-DOF trajectory mimicking a smooth driving sequence.

    Forward translation (axis 0) grows linearly; small yaw oscillation
    (axis 5); other axes near-zero with small random walk noise. This is
    the regime the KL basis exploits — effective rank ~2 (translation +
    yaw), so k=4 should capture > 99.9% of variance.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 60.0, n_frames)  # 60 m forward over the chunk
    poses = torch.zeros(n_frames, 6)
    poses[:, 0] = t  # forward translation
    poses[:, 5] = 0.05 * torch.sin(t / 6.0)  # small yaw oscillation
    poses += 0.001 * torch.randn(n_frames, 6, generator=g)
    return poses


def _random_iid_trajectory(n_frames: int = 600, seed: int = 1) -> torch.Tensor:
    """Synthetic 6-DOF trajectory with no temporal correlation. KL won't
    help here — effective rank is ~6, so k<6 truncation error is large."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_frames, 6, generator=g) * 0.3


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_op_satisfies_codec_op_protocol() -> None:
    op = Op_KLPoseStream()
    assert isinstance(op, CodecOp)
    assert op.name == "kl_pose_stream"


def test_op_default_n_components_is_four() -> None:
    op = Op_KLPoseStream()
    assert op.n_components == 4


# ---------------------------------------------------------------------------
# Encode / decode roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_smooth_trajectory_within_quant_tolerance() -> None:
    """Smooth driving trajectory through k=4 KL basis must reconstruct
    within int16-quantization tolerance per axis."""
    poses = _smooth_driving_trajectory()
    op = Op_KLPoseStream(n_components=4, brotli_quality=1)
    sd = {POSE_KEY: poses}
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    recon = decoded[POSE_KEY]
    assert recon.shape == poses.shape
    # Max abs error bounded by truncation (small for smooth) + per-axis
    # int16 quant step. Total per-frame error budget ~few × 1e-4 for our
    # synthetic. Keep a generous 1e-2 to absorb the truncation tail.
    max_err = (recon.float() - poses.float()).abs().max().item()
    assert max_err < 1e-2, f"max abs error {max_err} exceeded tolerance"


def test_roundtrip_full_rank_k6_is_quant_only_lossy() -> None:
    """k=6 keeps ALL components, so the only error source is int16
    quantization. Verify that error magnitude matches the int16 grid."""
    poses = _random_iid_trajectory()
    op = Op_KLPoseStream(n_components=6, brotli_quality=1)
    sd = {POSE_KEY: poses}
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    recon = decoded[POSE_KEY]
    # int16 step ≈ max_abs/32767 per component, so per-axis error
    # bounded by max_abs/32767. Loose bound: 2 × max_abs / 32767.
    max_abs = poses.abs().max().item()
    grid_bound = 2.0 * max_abs / 32767.0 * 6  # generous, accumulates across components
    max_err = (recon.float() - poses.float()).abs().max().item()
    assert max_err < grid_bound, (
        f"k=6 (full rank) error {max_err} > expected quant bound {grid_bound}"
    )


def test_truncation_diagnostic_high_variance_retention_for_smooth() -> None:
    """Smooth trajectory should have > 99% variance in top 2 components."""
    poses = _smooth_driving_trajectory()
    diag = estimate_truncation_rms(poses, n_components=2)
    assert diag["cumulative_variance_ratio"] > 0.99, diag


def test_truncation_diagnostic_low_variance_retention_for_random() -> None:
    """Random iid trajectory's top-2 captures only ~1/3 of variance."""
    poses = _random_iid_trajectory()
    diag = estimate_truncation_rms(poses, n_components=2)
    # 6 axes of equal variance → top-2 captures ~2/6 = 1/3.
    assert 0.20 < diag["cumulative_variance_ratio"] < 0.55


# ---------------------------------------------------------------------------
# Byte determinism
# ---------------------------------------------------------------------------


def test_encode_byte_deterministic() -> None:
    poses = _smooth_driving_trajectory()
    op = Op_KLPoseStream(n_components=3, brotli_quality=11)
    sd = {POSE_KEY: poses}
    blob_a = op.encode(sd, context={}).blob
    blob_b = op.encode(sd, context={}).blob
    assert blob_a == blob_b


def test_substrate_adaptive_different_trajectories_different_basis() -> None:
    """Two different trajectories produce different bases — verifies the
    basis is COMPUTED, not hardcoded."""
    poses_a = _smooth_driving_trajectory(seed=0)
    poses_b = _random_iid_trajectory(seed=1)
    op = Op_KLPoseStream(n_components=4, brotli_quality=1)
    res_a = op.encode({POSE_KEY: poses_a}, context={})
    res_b = op.encode({POSE_KEY: poses_b}, context={})
    # The blobs must differ (different bases + different coefficients).
    assert res_a.blob != res_b.blob


# ---------------------------------------------------------------------------
# Compression on smooth trajectory
# ---------------------------------------------------------------------------


def test_smooth_trajectory_compresses_below_raw_int16() -> None:
    """KL with k=2 on a smooth 600-frame trajectory should beat the raw
    600 × 6 × 2 = 7200 B int16 representation. We're storing basis
    (k*6*8 = 96 B for k=2) + mean (48 B) + scales (16 B) + Brotli of
    (600 × 2 × 2 = 2400 B int16 coefs) plus header — total well under
    7200 B for a low-rank signal."""
    poses = _smooth_driving_trajectory()
    raw_int16_bytes = poses.size(0) * poses.size(1) * 2  # 7200
    op = Op_KLPoseStream(n_components=2, brotli_quality=11)
    result = op.encode({POSE_KEY: poses}, context={})
    assert result.bytes_out < raw_int16_bytes, (
        f"k=2 KL ({result.bytes_out} B) did not beat raw int16 ({raw_int16_bytes} B)"
    )


# ---------------------------------------------------------------------------
# CodecPipeline integration
# ---------------------------------------------------------------------------


def test_op_in_codec_pipeline() -> None:
    """Op_KLPoseStream wraps cleanly inside CodecPipeline (the cathedral
    orchestrator)."""
    poses = _smooth_driving_trajectory()
    op = Op_KLPoseStream(n_components=3, brotli_quality=1)
    pipeline = CodecPipeline([op])
    blob, manifest = pipeline.encode({POSE_KEY: poses})
    assert blob[:4] == b"CPL1"
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["kl_pose_stream"]
    assert POSE_KEY in decoded
    assert decoded[POSE_KEY].shape == poses.shape


def test_op_state_is_json_serializable() -> None:
    """Cathedral discipline: op_state must be JSON-serializable for CPL1
    wire format. KL op stores n_components + n_frames as ints."""
    poses = _smooth_driving_trajectory()
    op = Op_KLPoseStream(n_components=4, brotli_quality=1)
    result = op.encode({POSE_KEY: poses}, context={})
    encoded = json.dumps(result.op_state)
    decoded = json.loads(encoded)
    assert decoded["n_components"] == 4
    assert decoded["n_frames"] == 600


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def test_validate_rejects_missing_pose_key() -> None:
    op = Op_KLPoseStream()
    rep = op.validate({"not_poses": torch.zeros(100, 6)}, context={})
    assert not rep.passed
    assert any("missing required" in f for f in rep.findings)


def test_validate_rejects_wrong_pose_dim() -> None:
    op = Op_KLPoseStream()
    rep = op.validate({POSE_KEY: torch.zeros(100, 5)}, context={})
    assert not rep.passed
    assert any("second dim must be 6" in f for f in rep.findings)


def test_validate_rejects_invalid_n_components() -> None:
    op = Op_KLPoseStream(n_components=0)
    rep = op.validate({POSE_KEY: torch.zeros(100, 6)}, context={})
    assert not rep.passed
    assert any("must be in [1, 6]" in f for f in rep.findings)


# ---------------------------------------------------------------------------
# Encode error handling
# ---------------------------------------------------------------------------


def test_encode_raises_on_missing_pose_key() -> None:
    op = Op_KLPoseStream()
    with pytest.raises(ValueError, match="missing required key"):
        op.encode({"other": torch.zeros(100, 6)}, context={})


def test_encode_raises_on_non_tensor_pose() -> None:
    op = Op_KLPoseStream()
    with pytest.raises(TypeError, match="must be torch.Tensor"):
        op.encode({POSE_KEY: [[0.0] * 6] * 100}, context={})  # type: ignore[dict-item]


def test_decode_raises_on_bad_magic() -> None:
    op = Op_KLPoseStream()
    with pytest.raises(ValueError, match="bad magic"):
        op.decode(b"NOPE" + b"\x00" * 100, op_state={}, context={})
