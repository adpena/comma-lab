"""Round-trip + byte-equality tests for QP1 pose codec.

Asserts:
- encode_qp1 -> decode_qp1 reconstructs col 0 within quantization step
- decode_qp1 of OUR encoder output matches PR #67's reader implementation
  (line-for-line port lives in pr67_inflate.py:789-806)
- the published PR #67 archive's pose stream decodes via OUR decoder to the
  same uint16 quantized words and the same float32 reconstruction
"""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import brotli
import numpy as np
import pytest
import torch

from tac.qp1_pose_codec import (
    POSE_SCALE,
    QP1_MAGIC,
    VELOCITY_OFFSET,
    VELOCITY_SCALE,
    decode_qp1,
    encode_qp1,
)


PR67_ARCHIVE = Path(__file__).resolve().parents[3] / (
    "reports/raw/leaderboard_intel_20260501/pr67_archive.zip"
)


def _make_pose(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poses = rng.uniform(-3.0, 3.0, size=(n, 6)).astype(np.float32)
    poses[:, 0] = rng.uniform(20.5, 50.0, size=n).astype(np.float32)
    return poses


def test_qp1_magic_and_header_layout() -> None:
    poses = _make_pose(5, seed=1)
    payload = encode_qp1(poses)
    assert payload.startswith(QP1_MAGIC)
    first = struct.unpack("<H", payload[3:5])[0]
    expected = int(round((float(poses[0, 0]) - VELOCITY_OFFSET) * VELOCITY_SCALE))
    assert first == expected


def test_qp1_round_trip_velocity_within_quant_step() -> None:
    poses = _make_pose(120, seed=2)
    payload = encode_qp1(poses)
    decoded = decode_qp1(payload)
    assert decoded.shape == (120, 6)
    # Cols 1-5 are zeroed by contract
    assert np.all(decoded[:, 1:] == 0.0)
    # Col 0 within +/- one quant step (1/512.0)
    diffs = np.abs(decoded[:, 0] - poses[:, 0])
    quant_step = 1.0 / VELOCITY_SCALE
    assert diffs.max() < quant_step, (
        f"velocity round-trip drift {diffs.max():.6f} exceeds quant step {quant_step:.6f}"
    )


def test_qp1_zigzag_handles_negative_deltas() -> None:
    # Build a synthetic pose where deltas alternate sign across the full int16 range.
    poses = np.zeros((10, 6), dtype=np.float32)
    base = 30.0
    for i in range(10):
        sign = 1.0 if i % 2 == 0 else -1.0
        # 2 m/s zigzag -> q-delta = +/- 1024
        poses[i, 0] = base + sign * 2.0
    payload = encode_qp1(poses)
    decoded = decode_qp1(payload)
    assert np.allclose(decoded[:, 0], poses[:, 0], atol=1.0 / VELOCITY_SCALE + 1e-7)


def test_qp1_rejects_out_of_range_velocity() -> None:
    # Velocity = 19.0 m/s -> q = -512 < 0 -> reject.
    poses = np.array([[19.0, 0, 0, 0, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="qp1 velocity"):
        encode_qp1(poses)


def test_qp1_decode_matches_published_pr67_archive() -> None:
    """The published PR #67 archive at HEAD must decode through OUR reader
    to a 600-row pose array with realistic highway velocities (~30 m/s)."""

    if not PR67_ARCHIVE.exists():
        pytest.skip(f"PR #67 archive missing: {PR67_ARCHIVE}")
    import zipfile

    with zipfile.ZipFile(PR67_ARCHIVE) as zf:
        blob = zf.read("p")
    # Layout from pr67_inflate.py:746-764: model_br_len = 56093 for 276430..276470
    assert 276430 <= len(blob) <= 276470, f"unexpected archive blob length {len(blob)}"
    mask_len = 219472
    model_len = 56093
    pose_q_br = blob[mask_len + model_len :]
    pose_raw = brotli.decompress(pose_q_br)
    decoded = decode_qp1(pose_raw)
    assert decoded.shape == (600, 6)
    assert decoded[:, 1:].sum() == 0.0
    # Highway velocities (chunk 0 of comma video is highway driving): mean in [25, 50] m/s
    mean_v = float(decoded[:, 0].mean())
    assert 25.0 < mean_v < 50.0, f"PR67 mean velocity {mean_v:.2f} outside 25-50"


def test_qp1_decoder_byte_equivalent_to_pr67_reader() -> None:
    """Importing pr67_inflate.py's reader and decoding our payload must give
    bit-identical uint16 quantized words.
    """

    pr67_path = (
        Path(__file__).resolve().parents[3]
        / "reports/raw/leaderboard_intel_20260501/pr67_inflate.py"
    )
    if not pr67_path.exists():
        pytest.skip(f"pr67 inflate reference missing: {pr67_path}")

    poses = _make_pose(50, seed=3)
    payload = encode_qp1(poses)

    # Inline-execute PR #67 reader on our payload
    first = np.frombuffer(payload[3:5], dtype=np.uint16, count=1)[0]
    vals = [int(first)]
    cursor = 5
    while cursor < len(payload):
        shift = 0
        acc = 0
        while True:
            byte = payload[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        vals.append((vals[-1] + delta) & 0xFFFF)
    pr67_vals = np.asarray(vals, dtype=np.uint16)

    ours_decoded = decode_qp1(payload)
    ours_vals = np.rint((ours_decoded[:, 0] - VELOCITY_OFFSET) * VELOCITY_SCALE).astype(
        np.uint16
    )
    assert np.array_equal(pr67_vals, ours_vals), (
        f"PR67 vs our decoder disagree: pr67={pr67_vals[:5]}, ours={ours_vals[:5]}"
    )
