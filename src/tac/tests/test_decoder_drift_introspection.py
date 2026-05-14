# SPDX-License-Identifier: MIT
"""Tests for ``tac.diagnostics.decoder_drift_introspection``."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.diagnostics.decoder_drift_introspection import (
    DecoderDriftIntrospector,
    FrameByteFingerprint,
    lipschitz_pose_drift_prediction,
    quantify_drift,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_VIDEO = REPO_ROOT / "upstream" / "videos" / "0.mkv"


def _synthetic_rgb_frame(seed: int = 1234, H: int = 64, W: int = 96) -> torch.Tensor:
    """Build a synthetic uint8 RGB frame deterministically (avoids pulling
    in the heavyweight 35MB upstream video for unit tests)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, generator=g)


# ---------- FrameByteFingerprint --------------------------------------------


def test_frame_byte_fingerprint_basic_shape_dtype():
    frame = _synthetic_rgb_frame()
    fp = FrameByteFingerprint.from_tensor(0, frame)
    assert fp.frame_index == 0
    assert fp.height == 64
    assert fp.width == 96
    assert fp.pixel_count == 64 * 96
    assert len(fp.per_channel_mean) == 3
    assert len(fp.per_channel_std) == 3
    assert len(fp.rgb_sha256) == 32
    assert fp.advisory_tag == "[diagnostic-not-score]"


def test_frame_byte_fingerprint_rejects_wrong_dtype():
    frame_f32 = torch.zeros(8, 8, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="uint8"):
        FrameByteFingerprint.from_tensor(0, frame_f32)


def test_frame_byte_fingerprint_rejects_wrong_shape():
    bad = torch.zeros(8, 8, 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="H, W, 3"):
        FrameByteFingerprint.from_tensor(0, bad)


def test_frame_byte_fingerprint_deterministic():
    frame = _synthetic_rgb_frame(seed=42)
    fp_a = FrameByteFingerprint.from_tensor(7, frame)
    fp_b = FrameByteFingerprint.from_tensor(7, frame.clone())
    assert fp_a.rgb_sha256 == fp_b.rgb_sha256
    assert fp_a.per_channel_mean == fp_b.per_channel_mean


# ---------- quantify_drift --------------------------------------------------


def test_quantify_drift_zero_when_identical():
    frames = _synthetic_rgb_frame().unsqueeze(0)  # (1, H, W, 3)
    rep = quantify_drift(frames, frames.clone())
    assert rep.l2_mean == pytest.approx(0.0)
    assert rep.max_abs_global == 0
    assert rep.l1_per_frame == [0.0]
    assert rep.mean_abs_per_channel == (0.0, 0.0, 0.0)
    assert rep.histogram_diff_signed[0] == frames[0].numel()


def test_quantify_drift_synthetic_perturbation_recovers_known_drift():
    """If we add a known perturbation, drift report should reflect it.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", the
    fixture's expected values are derived (not measured) and the test asserts
    against the closed-form derivation, not an external claim.
    """
    av = _synthetic_rgb_frame(seed=7).unsqueeze(0).repeat(3, 1, 1, 1)  # (3, H, W, 3)

    # Add deterministic +1 LSB on green channel only, leaving R/B identical.
    dali = av.clone()
    g = (dali[..., 1].to(torch.int16) + 1).clamp(0, 255).to(torch.uint8)
    dali[..., 1] = g

    rep = quantify_drift(av, dali)
    # Each pixel differs by exactly 1 LSB on G; R and B unchanged.
    n_pixels_per_frame = av.shape[1] * av.shape[2]
    assert rep.mean_abs_per_channel[0] == pytest.approx(0.0)
    # G mean abs = 1.0 (every pixel +1, except saturated 255s — but with
    # uniform random fixture, ~1/256 saturate; tolerance loose)
    assert 0.99 < rep.mean_abs_per_channel[1] <= 1.0
    assert rep.mean_abs_per_channel[2] == pytest.approx(0.0)
    # Max abs across all frames is 1.
    assert rep.max_abs_global == 1
    # L2 per frame: sqrt(N * 1^2) = sqrt(n_pixels_per_frame); the channel
    # contributes only on G so L2 over (H,W,3) of signed diff = sqrt(N) on G plane.
    expected_l2 = math.sqrt(n_pixels_per_frame * 1.0)
    # Allow slack for clamp at 255 (fewer non-saturated pixels contribute 1).
    assert abs(rep.l2_per_frame[0] - expected_l2) < 5.0


def test_quantify_drift_shape_mismatch_raises():
    av = _synthetic_rgb_frame().unsqueeze(0)
    dali = _synthetic_rgb_frame(H=32, W=48).unsqueeze(0)
    with pytest.raises(ValueError, match="shape mismatch"):
        quantify_drift(av, dali)


def test_quantify_drift_dtype_mismatch_raises():
    av = _synthetic_rgb_frame().unsqueeze(0)
    dali = av.float()
    with pytest.raises(ValueError, match="uint8"):
        quantify_drift(av, dali)


def test_quantify_drift_signed_histogram_centered_when_unbiased():
    """If we add zero-mean noise, the signed histogram should be ~symmetric."""
    av = _synthetic_rgb_frame(seed=99, H=128, W=128)
    av_batch = av.unsqueeze(0).repeat(2, 1, 1, 1)  # (2, H, W, 3)
    g = torch.Generator()
    g.manual_seed(0)
    noise = torch.randint(-2, 3, av_batch.shape, generator=g, dtype=torch.int16)
    dali = (av_batch.to(torch.int16) + noise).clamp(0, 255).to(torch.uint8)
    rep = quantify_drift(av_batch, dali, histogram_range=(-3, 3))
    # Symmetry: count(+1) ~ count(-1), count(+2) ~ count(-2)
    assert abs(rep.histogram_diff_signed[1] - rep.histogram_diff_signed[-1]) < 5000
    assert abs(rep.histogram_diff_signed[2] - rep.histogram_diff_signed[-2]) < 5000


# ---------- lipschitz_pose_drift_prediction ---------------------------------


def test_lipschitz_prediction_basic_math():
    """Verify the closed-form math: input_l2 = sqrt(N * d^2 / 3) where
    d = drift_lsb / std."""
    pixel_count = 1000
    drift_lsb = 1.5
    std = 63.75
    L = 1e-4
    pred = lipschitz_pose_drift_prediction(
        per_pixel_rgb_drift_lsb=drift_lsb,
        pixel_count=pixel_count,
        posenet_input_normalize_std=std,
        lipschitz_estimate=L,
    )
    d = drift_lsb / std
    var = d**2 / 3.0
    expected_l2 = math.sqrt(pixel_count * var)
    expected_pose = L * expected_l2
    assert pred["drift_normalized_per_pixel"] == pytest.approx(d)
    assert pred["input_l2_normalized"] == pytest.approx(expected_l2)
    assert pred["predicted_pose_component_drift"] == pytest.approx(expected_pose)


def test_lipschitz_prediction_pr106_pixel_count_dominant():
    """With realistic PR106-frontier numbers (1.5 LSB drift on ~3M-pixel
    frame, L=1e-4 conservative Lipschitz), the predicted pose drift should
    be in the same order of magnitude as the observed 1.4e-4 gap."""
    H, W = 874, 1164
    pixel_count = H * W * 3
    pred = lipschitz_pose_drift_prediction(
        per_pixel_rgb_drift_lsb=1.5,
        pixel_count=pixel_count,
        lipschitz_estimate=1e-4,
    )
    # We expect "decoder-dominant" verdict (pred > 0.7 * 1.4e-4 = 9.8e-5)
    assert pred["predicted_pose_component_drift"] > 9.8e-5
    assert pred["verdict"] in ("decoder-dominant", "decoder-mixed")


def test_lipschitz_prediction_subdominant_with_tiny_lipschitz():
    """At the smallest Lipschitz floor (L=1e-7), even worst-case decoder
    drift should NOT explain the observed gap → 'decoder-subdominant'."""
    H, W = 874, 1164
    pixel_count = H * W * 3
    pred = lipschitz_pose_drift_prediction(
        per_pixel_rgb_drift_lsb=1.5,
        pixel_count=pixel_count,
        lipschitz_estimate=1e-7,
    )
    assert pred["verdict"] == "decoder-subdominant"


def test_lipschitz_prediction_zero_drift_zero_pose():
    pred = lipschitz_pose_drift_prediction(
        per_pixel_rgb_drift_lsb=0.0,
        pixel_count=1_000_000,
        lipschitz_estimate=1e-4,
    )
    assert pred["predicted_pose_component_drift"] == 0.0
    assert pred["verdict"] == "decoder-subdominant"


# ---------- DecoderDriftIntrospector ----------------------------------------


def test_introspector_dali_design_returns_serializable_recipe():
    intro = DecoderDriftIntrospector()
    design = intro.decode_dali_design(REPO_ROOT / "fake.mkv", frame_indices=[0, 1])
    # Must be JSON-serializable for cross-host dispatch
    import json

    s = json.dumps(design)
    assert "DaliVideoDataset" in s
    assert "BT.601 limited range" in s
    assert design["requires"]["cuda_capable"] is True
    assert design["frame_indices"] == [0, 1]


def test_introspector_fingerprint_returns_one_per_frame():
    intro = DecoderDriftIntrospector()
    batch = torch.stack([_synthetic_rgb_frame(seed=i) for i in range(5)])
    fps = intro.fingerprint(batch)
    assert len(fps) == 5
    assert [fp.frame_index for fp in fps] == [0, 1, 2, 3, 4]


def test_introspector_ingest_dali_dump_npz_roundtrip(tmp_path):
    intro = DecoderDriftIntrospector()
    batch = torch.stack([_synthetic_rgb_frame(seed=i) for i in range(3)])
    npz_path = tmp_path / "dali.npz"
    np.savez_compressed(npz_path, frames=batch.numpy())
    loaded = intro.ingest_dali_dump(npz_path)
    assert torch.equal(loaded, batch)


def test_introspector_ingest_dali_dump_rejects_float_dtype(tmp_path):
    intro = DecoderDriftIntrospector()
    npz_path = tmp_path / "bad.npz"
    np.savez_compressed(npz_path, frames=np.zeros((2, 8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        intro.ingest_dali_dump(npz_path)


# ---------- AV decode determinism (skipped if upstream video missing) -------


@pytest.mark.skipif(
    not TEST_VIDEO.exists(), reason="upstream/videos/0.mkv not present (private/CI-only)"
)
def test_av_decode_is_deterministic():
    """Run AVVideoDataset's exact decode path twice on 0.mkv; outputs must
    be bit-identical (no nondeterminism from libav under the same input).

    This is the core determinism guarantee for the decoder-drift hypothesis:
    if AV decode is bit-stable, then any DALI-vs-AV drift we measure is
    purely from the codec backend, not from undefined behavior in either
    decoder.
    """
    intro = DecoderDriftIntrospector()
    a = intro.decode_av(TEST_VIDEO, max_frames=2)
    b = intro.decode_av(TEST_VIDEO, max_frames=2)
    assert torch.equal(a, b), (
        "AV decode is NOT deterministic on this host; decoder-drift "
        "investigation must control for this before attributing drift to "
        "DALI-vs-AV."
    )
