# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.codec_pipeline_raft_pose`.

Coverage:
- Op_RAFTPoseStream satisfies the CodecOp Protocol
- encode/decode roundtrip preserves int16 grid (bit-faithful within scale)
- byte-determinism: encoding the same poses twice produces the same blob
- validate rejects bad shape, missing key, NaN/Inf, wrong type, zero frames
- validate warns (still passes) on extreme magnitude
- encode size scales sensibly with brotli quality + N_frames
- composes with :class:`tac.codec_pipeline.CodecPipeline`

Strict-scorer-rule: pure CPU; no scorer load anywhere.
"""
from __future__ import annotations

import pytest
import torch

from tac.codec_pipeline import CodecOp, CodecPipeline
from tac.codec_pipeline_raft_pose import (
    POSE_KEY,
    Op_RAFTPoseStream,
)


def _synthetic_poses(n_frames: int = 600, seed: int = 0) -> torch.Tensor:
    """Smooth synthetic 6DOF poses (cumulative random walk) - mimics RAFT
    output on a driving video chunk."""
    g = torch.Generator().manual_seed(seed)
    deltas = torch.randn(n_frames, 6, generator=g) * 0.01
    return torch.cumsum(deltas, dim=0).to(torch.float32)


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------

def test_op_raft_pose_stream_satisfies_codec_op_protocol() -> None:
    op = Op_RAFTPoseStream()
    assert isinstance(op, CodecOp)
    # Op_state schema: name + transforms_state_dict default
    assert op.name == "raft_pose_stream"
    assert op.transforms_state_dict is False


# ---------------------------------------------------------------------------
# Encode/decode roundtrip (bit-faithful within int16 grid)
# ---------------------------------------------------------------------------

def test_roundtrip_preserves_int16_grid() -> None:
    poses = _synthetic_poses(n_frames=600)
    op = Op_RAFTPoseStream()
    sd = {POSE_KEY: poses}
    res = op.encode(sd, context={})
    assert res.bytes_out > 0
    decoded = op.decode(res.blob, op_state=res.op_state, context={})
    rec = decoded[POSE_KEY]
    assert rec.shape == poses.shape
    # Reconstruction error bounded by per-axis scale / 32767. The encoder uses
    # max_abs / 32767 as scale, so per-element error is at most one quantum.
    scales = res.op_state["per_axis_scales"]
    err = (rec - poses).abs()
    for axis in range(6):
        # One full quantum tolerance + small numerical slack.
        tol = float(scales[axis]) * 1.5 + 1e-9
        assert err[:, axis].max().item() <= tol, (
            f"axis {axis} max-err {err[:, axis].max().item():.4e} > tol {tol:.4e}"
        )


def test_byte_determinism_same_input_same_blob() -> None:
    poses = _synthetic_poses(n_frames=600, seed=42)
    op = Op_RAFTPoseStream()
    sd = {POSE_KEY: poses}
    blob_a = op.encode(sd, context={}).blob
    blob_b = op.encode(sd, context={}).blob
    assert blob_a == blob_b
    # Different seed -> different blob.
    poses_other = _synthetic_poses(n_frames=600, seed=43)
    blob_c = op.encode({POSE_KEY: poses_other}, context={}).blob
    assert blob_c != blob_a


def test_explicit_per_axis_scales_override() -> None:
    poses = _synthetic_poses(n_frames=300)
    explicit = [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]
    op = Op_RAFTPoseStream(per_axis_scales=explicit)
    sd = {POSE_KEY: poses}
    res = op.encode(sd, context={})
    assert res.op_state["per_axis_scales"] == explicit
    decoded = op.decode(res.blob, op_state=res.op_state, context={})
    # Reconstruction should still be close (depending on whether scale
    # is large enough); for 1e-3/1e-4 explicit values the int16 range is
    # ~33k * scale = 32-3.3 which more than covers cumulative ~6 magnitude.
    assert decoded[POSE_KEY].shape == poses.shape


# ---------------------------------------------------------------------------
# Validate gates
# ---------------------------------------------------------------------------

def test_validate_passes_on_good_input() -> None:
    poses = _synthetic_poses(n_frames=600)
    op = Op_RAFTPoseStream()
    rep = op.validate({POSE_KEY: poses}, context={})
    assert rep.passed is True
    assert rep.findings == []


def test_validate_rejects_missing_key() -> None:
    op = Op_RAFTPoseStream()
    rep = op.validate({}, context={})
    assert rep.passed is False
    assert any("missing required key" in f for f in rep.findings)


def test_validate_rejects_bad_shape() -> None:
    op = Op_RAFTPoseStream()
    # 1D tensor.
    rep1 = op.validate({POSE_KEY: torch.randn(600)}, context={})
    assert rep1.passed is False
    # Wrong second dim.
    rep2 = op.validate({POSE_KEY: torch.randn(600, 3)}, context={})
    assert rep2.passed is False
    # Zero frames.
    rep3 = op.validate({POSE_KEY: torch.randn(0, 6)}, context={})
    assert rep3.passed is False


def test_validate_rejects_nan_or_inf() -> None:
    op = Op_RAFTPoseStream()
    bad = torch.randn(600, 6)
    bad[5, 2] = float("nan")
    rep_nan = op.validate({POSE_KEY: bad}, context={})
    assert rep_nan.passed is False
    assert any("NaN" in f for f in rep_nan.findings)
    bad2 = torch.randn(600, 6)
    bad2[10, 0] = float("inf")
    rep_inf = op.validate({POSE_KEY: bad2}, context={})
    assert rep_inf.passed is False
    assert any("Inf" in f for f in rep_inf.findings)


def test_validate_rejects_non_tensor_value() -> None:
    op = Op_RAFTPoseStream()
    rep = op.validate({POSE_KEY: [[0.0] * 6] * 600}, context={})  # type: ignore[dict-item]
    assert rep.passed is False
    assert any("torch.Tensor" in f for f in rep.findings)


def test_validate_warns_on_extreme_magnitude() -> None:
    op = Op_RAFTPoseStream(magnitude_warn_threshold=1.0)
    # Magnitude above threshold -> warns but still passes.
    poses = torch.full((100, 6), 5.0)
    rep = op.validate({POSE_KEY: poses}, context={})
    assert rep.passed is True  # warning, not failure
    assert any("warn-threshold" in f for f in rep.findings)


# ---------------------------------------------------------------------------
# Encode size sensitivity
# ---------------------------------------------------------------------------

def test_encode_size_scales_with_n_frames() -> None:
    op = Op_RAFTPoseStream()
    small = _synthetic_poses(n_frames=100)
    large = _synthetic_poses(n_frames=600, seed=1)
    s_small = op.encode({POSE_KEY: small}, context={}).bytes_out
    s_large = op.encode({POSE_KEY: large}, context={}).bytes_out
    assert s_large > s_small  # 6x more frames -> bigger blob (compressed)


def test_encode_size_quality_monotonic_for_smooth_data() -> None:
    """Higher Brotli quality should not produce a STRICTLY larger output for
    smooth data; q=11 is the standard target. Check q=11 <= q=1 within a
    small slack."""
    poses = _synthetic_poses(n_frames=600)
    op_low = Op_RAFTPoseStream(brotli_quality=1)
    op_hi = Op_RAFTPoseStream(brotli_quality=11)
    s_low = op_low.encode({POSE_KEY: poses}, context={}).bytes_out
    s_hi = op_hi.encode({POSE_KEY: poses}, context={}).bytes_out
    # q=11 should not be larger than q=1 (give 5% slack for boundary cases).
    assert s_hi <= int(s_low * 1.05)


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------

def test_op_composes_with_codec_pipeline() -> None:
    """Op_RAFTPoseStream as a singleton pipeline encodes/decodes through CPL1
    container."""
    poses = _synthetic_poses(n_frames=600)
    sd = {POSE_KEY: poses}
    pipeline = CodecPipeline([Op_RAFTPoseStream()])
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    assert manifest.final_bytes == len(blob)
    assert manifest.final_blob_sha256
    # Decode round-trip via the pipeline.
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["raft_pose_stream"]
    rec = decoded[POSE_KEY]
    assert rec.shape == poses.shape
    # Reconstruction within scale tolerance.
    op_state = manifest.op_results[0].op_state
    scales = op_state["per_axis_scales"]
    for axis in range(6):
        tol = float(scales[axis]) * 1.5 + 1e-9
        err = (rec[:, axis] - poses[:, axis]).abs().max().item()
        assert err <= tol


def test_op_state_schema_keys_match_contract() -> None:
    """The op_state dict must contain exactly per_axis_scales + n_frames per
    the documented schema."""
    poses = _synthetic_poses(n_frames=300)
    op = Op_RAFTPoseStream()
    res = op.encode({POSE_KEY: poses}, context={})
    assert set(res.op_state.keys()) == {"per_axis_scales", "n_frames"}
    assert res.op_state["n_frames"] == 300
    assert isinstance(res.op_state["per_axis_scales"], list)
    assert len(res.op_state["per_axis_scales"]) == 6


def test_decode_rejects_bad_magic() -> None:
    op = Op_RAFTPoseStream()
    with pytest.raises(ValueError, match="bad magic"):
        op.decode(
            b"BAD!1234567890",
            op_state={"per_axis_scales": [1.0] * 6, "n_frames": 1},
            context={},
        )


def test_decode_rejects_n_frames_mismatch() -> None:
    poses = _synthetic_poses(n_frames=300)
    op = Op_RAFTPoseStream()
    res = op.encode({POSE_KEY: poses}, context={})
    bad_state = dict(res.op_state)
    bad_state["n_frames"] = 999
    with pytest.raises(ValueError, match="n_frames"):
        op.decode(res.blob, op_state=bad_state, context={})
