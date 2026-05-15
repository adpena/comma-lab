# SPDX-License-Identifier: MIT
"""Tests for D1 L2 INTEGRATION overlay + margin-map shrink helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tac.substrates.d1_segnet_margin_polytope.margin_map import (
    MARGIN_MAP_DEFAULT_RESOLUTION,
    MARGIN_MAP_SHRUNK_RESOLUTION,
    upsample_margin_map_for_overlay,
)
from tac.substrates.d1_segnet_margin_polytope.overlay import (
    D1_OVERLAY_AMPLITUDE_SCALES,
    D1_OVERLAY_SIGN_POLICIES,
    _build_camera_resolution_overlay,
    _upsample_int8_levels_to_camera,
    apply_l2_overlay_for_video_list,
    apply_polytope_overlay_inplace,
    attenuate_overlay_levels,
    channel_policy_weights,
    normalize_overlay_amplitude_scale,
    overlay_sign_for_pair,
    pack_pair_sign_mask,
    unpack_pair_sign_mask,
    validate_polytope_margin_contract,
)
from tac.substrates.d1_segnet_margin_polytope.polytope_encoder import (
    encode_polytope_payload,
)

# ---------------------------------------------------------------------------
# Margin map shrink helpers
# ---------------------------------------------------------------------------


def test_shrunk_resolution_constant_is_16x_smaller():
    """96x128 = 12,288 pixels vs 384x512 = 196,608 pixels = 16x ratio."""
    full = MARGIN_MAP_DEFAULT_RESOLUTION
    shrunk = MARGIN_MAP_SHRUNK_RESOLUTION
    full_pixels = full[0] * full[1]
    shrunk_pixels = shrunk[0] * shrunk[1]
    assert full_pixels == 16 * shrunk_pixels


def test_upsample_margin_map_passthrough_when_already_target():
    """Pass-through when input shape == target_resolution."""
    margin = torch.ones((384, 512), dtype=torch.float32) * 2.5
    out = upsample_margin_map_for_overlay(margin)
    assert out.shape == (384, 512)
    assert torch.allclose(out, margin)


def test_upsample_margin_map_shrunk_to_full_bicubic():
    margin = torch.full((96, 128), 1.0, dtype=torch.float32)
    out = upsample_margin_map_for_overlay(
        margin, target_resolution=MARGIN_MAP_DEFAULT_RESOLUTION
    )
    assert out.shape == (384, 512)
    # Constant input should produce ~constant upsample (bicubic preserves
    # constants exactly within float roundoff).
    assert torch.allclose(out, torch.full((384, 512), 1.0), atol=1e-5)


def test_upsample_margin_map_clamps_negative_ringing():
    """Bicubic can ring slightly negative near zero-margin transitions; the
    helper must clamp_min(0.0) to preserve the polytope-interior invariant.
    """
    margin = torch.zeros((96, 128), dtype=torch.float32)
    margin[40:50, 60:70] = 5.0  # interior peak surrounded by boundary
    out = upsample_margin_map_for_overlay(margin)
    assert (out >= 0).all()


def test_upsample_margin_map_rejects_non_2d():
    margin = torch.ones((2, 96, 128), dtype=torch.float32)
    with pytest.raises(ValueError, match="2D margin map"):
        upsample_margin_map_for_overlay(margin)


def test_upsample_margin_map_rejects_bad_mode():
    margin = torch.ones((96, 128), dtype=torch.float32)
    with pytest.raises(ValueError, match="upsample_mode"):
        upsample_margin_map_for_overlay(margin, upsample_mode="trilinear")


def test_upsample_margin_map_supports_bilinear():
    margin = torch.full((96, 128), 1.0, dtype=torch.float32)
    out = upsample_margin_map_for_overlay(margin, upsample_mode="bilinear")
    assert out.shape == (384, 512)
    assert torch.allclose(out, torch.full((384, 512), 1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Overlay int8 upsample
# ---------------------------------------------------------------------------


def test_int8_levels_upsample_to_camera_shape():
    levels = np.zeros((96, 128), dtype=np.int8)
    levels[40, 60] = 2
    out = _upsample_int8_levels_to_camera(levels)
    assert out.shape == (874, 1164)
    assert out.dtype == np.int8


def test_int8_levels_preserves_lattice_range():
    rng = np.random.RandomState(7)
    levels = rng.randint(-2, 3, size=(96, 128)).astype(np.int8)
    out = _upsample_int8_levels_to_camera(levels)
    assert out.min() >= -2
    assert out.max() <= 2


def test_int8_levels_passthrough_when_already_camera():
    levels = np.zeros((874, 1164), dtype=np.int8)
    out = _upsample_int8_levels_to_camera(levels)
    assert out.shape == levels.shape
    assert out.dtype == np.int8
    assert (out == levels).all()


def test_int8_levels_rejects_non_int8():
    levels = np.zeros((96, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="int8"):
        _upsample_int8_levels_to_camera(levels)


def test_int8_levels_rejects_non_2d():
    levels = np.zeros((2, 96, 128), dtype=np.int8)
    with pytest.raises(ValueError, match="2D"):
        _upsample_int8_levels_to_camera(levels)


def test_build_camera_overlay_from_flat_size_mismatch():
    flat = np.zeros(100, dtype=np.int8)
    with pytest.raises(ValueError, match="!= expected"):
        _build_camera_resolution_overlay(
            noise_levels_flat=flat,
            encoder_grid_h=96,
            encoder_grid_w=128,
        )


def test_build_camera_overlay_zero_levels_zero_overlay():
    flat = np.zeros(96 * 128, dtype=np.int8)
    out = _build_camera_resolution_overlay(
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
    )
    assert out.shape == (874, 1164)
    assert (out == 0).all()


# ---------------------------------------------------------------------------
# apply_polytope_overlay_inplace
# ---------------------------------------------------------------------------


def _write_synthetic_raw(path: Path, n_pairs: int = 2) -> int:
    """Helper: write n_pairs * 2 frames at camera resolution as uint8."""
    camera_h, camera_w = 874, 1164
    frame_bytes = camera_h * camera_w * 3
    # Use 128 (gray) so we have positive and negative delta headroom.
    payload = np.full(n_pairs * 2 * frame_bytes, 128, dtype=np.uint8)
    path.write_bytes(payload.tobytes())
    return n_pairs


def test_apply_polytope_overlay_inplace_zero_overlay_is_noop(tmp_path):
    raw = tmp_path / "0.raw"
    n_pairs = 2
    _write_synthetic_raw(raw, n_pairs=n_pairs)
    original = raw.read_bytes()
    flat_zero = np.zeros(96 * 128, dtype=np.int8)
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat_zero,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
    )
    assert diag["pairs_modified"] == 0
    assert diag["bytes_changed"] == 0
    assert diag["nonzero_overlay_pixels"] == 0
    assert raw.read_bytes() == original


def test_apply_polytope_overlay_inplace_nonzero_modifies_frame_1(tmp_path):
    raw = tmp_path / "0.raw"
    n_pairs = 2
    _write_synthetic_raw(raw, n_pairs=n_pairs)
    camera_h, camera_w = 874, 1164
    frame_bytes = camera_h * camera_w * 3
    # Build noise levels with a single +1 in interior — overlay should
    # broadcast across all 3 channels and propagate to camera resolution.
    flat = np.zeros(96 * 128, dtype=np.int8)
    flat[40 * 128 + 60] = 2  # interior pixel
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
    )
    assert diag["nonzero_overlay_pixels"] > 0
    assert diag["pairs_modified"] == n_pairs  # both pairs modified
    assert diag["bytes_changed"] > 0
    assert diag["frame_bytes"] == frame_bytes
    # Verify frame_0 unchanged, frame_1 modified.
    new_data = raw.read_bytes()
    for pair_idx in range(n_pairs):
        f0_start = pair_idx * 2 * frame_bytes
        f0_end = f0_start + frame_bytes
        f1_start = f0_end
        f1_end = f1_start + frame_bytes
        # frame_0 should be entirely 128 (untouched).
        assert all(new_data[f0_start:f0_end][i] == 128 for i in range(0, frame_bytes, 1000))
        # frame_1 should have at least one pixel != 128.
        frame_1_arr = np.frombuffer(new_data[f1_start:f1_end], dtype=np.uint8)
        assert (frame_1_arr != 128).any()


def test_apply_polytope_overlay_channel_policy_green_only(tmp_path):
    raw = tmp_path / "0.raw"
    n_pairs = 1
    _write_synthetic_raw(raw, n_pairs=n_pairs)
    frame_bytes = 874 * 1164 * 3
    flat = np.ones(96 * 128, dtype=np.int8)
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
        channel_policy="green",
    )
    assert diag["channel_policy"] == "green"
    frame_1 = np.frombuffer(raw.read_bytes()[frame_bytes:], dtype=np.uint8).reshape(-1, 3)
    assert (frame_1[:, 0] == 128).all()
    assert (frame_1[:, 1] != 128).any()
    assert (frame_1[:, 2] == 128).all()


def test_apply_polytope_overlay_pair_mask_skips_disabled_pairs(tmp_path):
    raw = tmp_path / "0.raw"
    n_pairs = 3
    _write_synthetic_raw(raw, n_pairs=n_pairs)
    frame_bytes = 874 * 1164 * 3
    flat = np.ones(96 * 128, dtype=np.int8)
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
        channel_policy="green",
        sign_policy="pair_mask",
        pair_sign_mask=[1, 0, -1],
    )
    assert diag["pairs_modified"] == 2
    payload = raw.read_bytes()
    pair0_frame1 = np.frombuffer(payload[frame_bytes : 2 * frame_bytes], dtype=np.uint8)
    pair1_frame1 = np.frombuffer(
        payload[3 * frame_bytes : 4 * frame_bytes], dtype=np.uint8
    )
    pair2_frame1 = np.frombuffer(
        payload[5 * frame_bytes : 6 * frame_bytes], dtype=np.uint8
    )
    assert (pair0_frame1 != 128).any()
    assert (pair1_frame1 == 128).all()
    assert (pair2_frame1 != 128).any()


def test_channel_policy_weights_rejects_unknown():
    with pytest.raises(ValueError, match="overlay_channel_policy"):
        channel_policy_weights("cyan")


def test_overlay_policy_constants_cover_noop_half_and_sign_schedules():
    assert D1_OVERLAY_AMPLITUDE_SCALES == (0.0, 0.5, 1.0)
    assert "payload" in D1_OVERLAY_SIGN_POLICIES
    assert "negate_payload" in D1_OVERLAY_SIGN_POLICIES
    assert "alternating_pairs" in D1_OVERLAY_SIGN_POLICIES
    assert "pair_mask" in D1_OVERLAY_SIGN_POLICIES


def test_pair_sign_mask_pack_unpack_roundtrip():
    signs = (1, 0, -1, 1, -1, 0, 0, 1, 1)
    encoded = pack_pair_sign_mask(signs)
    assert unpack_pair_sign_mask(encoded, n_pairs=len(signs)) == signs


def test_pair_sign_mask_base85_is_smaller_than_base64_for_contest_mask():
    signs = tuple(1 if idx % 5 == 0 else -1 if idx % 7 == 0 else 0 for idx in range(600))
    encoded = pack_pair_sign_mask(signs)
    assert len(encoded) == 188
    assert len(encoded) < 200
    assert len(encoded) < 300
    assert unpack_pair_sign_mask(encoded, n_pairs=len(signs)) == signs


def test_normalize_overlay_amplitude_scale_rejects_amplification():
    assert normalize_overlay_amplitude_scale(0.5) == 0.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        normalize_overlay_amplitude_scale(1.25)
    with pytest.raises(ValueError, match="finite"):
        normalize_overlay_amplitude_scale(float("nan"))


def test_attenuate_overlay_levels_preserves_lattice():
    levels = np.array([[-2, -1, 0, 1, 2]], dtype=np.int8)
    half = attenuate_overlay_levels(levels, amplitude_scale=0.5)
    np.testing.assert_array_equal(
        half, np.array([[-1, -1, 0, 1, 1]], dtype=np.int8)
    )
    zero = attenuate_overlay_levels(levels, amplitude_scale=0.0)
    assert (zero == 0).all()


def test_overlay_sign_policy_schedules_are_deterministic():
    assert overlay_sign_for_pair("payload", 0) == 1
    assert overlay_sign_for_pair("negate_payload", 0) == -1
    assert overlay_sign_for_pair("alternating_pairs", 0) == 1
    assert overlay_sign_for_pair("alternating_pairs", 1) == -1
    assert overlay_sign_for_pair("pair_mask", 0, [1, 0, -1]) == 1
    assert overlay_sign_for_pair("pair_mask", 1, [1, 0, -1]) == 0
    assert overlay_sign_for_pair("pair_mask", 2, [1, 0, -1]) == -1
    with pytest.raises(ValueError, match="pair_sign_mask"):
        overlay_sign_for_pair("pair_mask", 0)
    with pytest.raises(ValueError, match="overlay_sign_policy"):
        overlay_sign_for_pair("coinflip", 0)


def test_apply_polytope_overlay_amplitude_zero_is_exact_noop(tmp_path):
    raw = tmp_path / "0.raw"
    n_pairs = 1
    _write_synthetic_raw(raw, n_pairs=n_pairs)
    original = raw.read_bytes()
    flat = np.ones(96 * 128, dtype=np.int8)
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
        amplitude_scale=0.0,
    )
    assert diag["pairs_modified"] == 0
    assert diag["bytes_changed"] == 0
    assert diag["overlay_amplitude_scale"] == 0.0
    assert raw.read_bytes() == original


def test_apply_polytope_overlay_negate_payload_flips_frame_1_delta(tmp_path):
    raw = tmp_path / "0.raw"
    n_pairs = 1
    _write_synthetic_raw(raw, n_pairs=n_pairs)
    frame_bytes = 874 * 1164 * 3
    flat = np.ones(96 * 128, dtype=np.int8)
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
        channel_policy="red",
        sign_policy="negate_payload",
    )
    assert diag["pairs_modified"] == 1
    assert diag["overlay_sign_policy"] == "negate_payload"
    data = raw.read_bytes()
    frame_0 = np.frombuffer(data[:frame_bytes], dtype=np.uint8).reshape(-1, 3)
    frame_1 = np.frombuffer(data[frame_bytes:], dtype=np.uint8).reshape(-1, 3)
    assert (frame_0 == 128).all()
    assert (frame_1[:, 0] == 127).all()
    assert (frame_1[:, 1] == 128).all()
    assert (frame_1[:, 2] == 128).all()


def test_apply_polytope_overlay_alternating_pairs_changes_frame_1_sign(tmp_path):
    raw = tmp_path / "0.raw"
    n_pairs = 2
    _write_synthetic_raw(raw, n_pairs=n_pairs)
    frame_bytes = 874 * 1164 * 3
    flat = np.ones(96 * 128, dtype=np.int8)
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
        channel_policy="green",
        sign_policy="alternating_pairs",
    )
    assert diag["pairs_modified"] == 2
    data = raw.read_bytes()
    pair0_frame1 = np.frombuffer(
        data[frame_bytes:2 * frame_bytes], dtype=np.uint8
    ).reshape(-1, 3)
    pair1_frame0_start = 2 * frame_bytes
    pair1_frame1_start = pair1_frame0_start + frame_bytes
    pair1_frame0 = np.frombuffer(
        data[pair1_frame0_start:pair1_frame1_start], dtype=np.uint8
    ).reshape(-1, 3)
    pair1_frame1 = np.frombuffer(
        data[pair1_frame1_start:pair1_frame1_start + frame_bytes],
        dtype=np.uint8,
    ).reshape(-1, 3)
    assert (pair0_frame1[:, 1] == 129).all()
    assert (pair1_frame0 == 128).all()
    assert (pair1_frame1[:, 1] == 127).all()


def test_apply_polytope_overlay_inplace_size_mismatch_raises(tmp_path):
    raw = tmp_path / "0.raw"
    # Write a too-small .raw.
    raw.write_bytes(b"\x00" * 100)
    flat = np.zeros(96 * 128, dtype=np.int8)
    with pytest.raises(ValueError, match="size"):
        apply_polytope_overlay_inplace(
            raw,
            noise_levels_flat=flat,
            encoder_grid_h=96,
            encoder_grid_w=128,
            n_pairs=2,
        )


def test_apply_polytope_overlay_inplace_missing_raw_raises(tmp_path):
    flat = np.zeros(96 * 128, dtype=np.int8)
    with pytest.raises(FileNotFoundError):
        apply_polytope_overlay_inplace(
            tmp_path / "nope.raw",
            noise_levels_flat=flat,
            encoder_grid_h=96,
            encoder_grid_w=128,
        )


def test_apply_polytope_overlay_clamps_to_uint8_range(tmp_path):
    """Pixels near uint8 boundary [0, 255] must clamp correctly."""
    raw = tmp_path / "0.raw"
    n_pairs = 1
    camera_h, camera_w = 874, 1164
    frame_bytes = camera_h * camera_w * 3
    # Frame 0 all 0, frame 1 all 254 — so +2 noise would overflow to 256.
    f0 = np.zeros(frame_bytes, dtype=np.uint8)
    f1 = np.full(frame_bytes, 254, dtype=np.uint8)
    raw.write_bytes(f0.tobytes() + f1.tobytes())
    # Allocate +2 in all interior pixels.
    flat = np.full(96 * 128, 2, dtype=np.int8)
    diag = apply_polytope_overlay_inplace(
        raw,
        noise_levels_flat=flat,
        encoder_grid_h=96,
        encoder_grid_w=128,
        n_pairs=n_pairs,
    )
    assert diag["pairs_modified"] == 1
    new_data = raw.read_bytes()
    f1_new = np.frombuffer(new_data[frame_bytes:], dtype=np.uint8)
    # Every pixel should be 255 (254 + 2 clamped to 255).
    assert (f1_new == 255).all()


# ---------------------------------------------------------------------------
# apply_l2_overlay_for_video_list
# ---------------------------------------------------------------------------


def test_apply_l2_overlay_walks_video_list(tmp_path):
    """The orchestrator helper walks the video list and processes each .raw."""
    # Make 2 synthetic videos.
    for name in ("0", "1"):
        _write_synthetic_raw(tmp_path / f"{name}.raw", n_pairs=2)
    # Build a real polytope payload.
    rng = np.random.RandomState(13)
    margin = torch.from_numpy(
        rng.rand(96, 128).astype(np.float32) * 2.0 + 0.1
    )
    payload = encode_polytope_payload(
        margin, jacobian_lipschitz=1.0, budget_bits=2000
    )
    diag = apply_l2_overlay_for_video_list(
        output_dir=tmp_path,
        video_names=["0.mkv", "1.mkv"],
        polytope_payload=payload,
        encoder_grid_h=96,
        encoder_grid_w=128,
    )
    assert diag["videos_processed"] == 2


def test_apply_l2_overlay_missing_raw_raises(tmp_path):
    rng = np.random.RandomState(13)
    margin = torch.from_numpy(rng.rand(96, 128).astype(np.float32) + 1.0)
    payload = encode_polytope_payload(margin, jacobian_lipschitz=1.0, budget_bits=2000)
    with pytest.raises(FileNotFoundError, match="cannot locate"):
        apply_l2_overlay_for_video_list(
            output_dir=tmp_path,
            video_names=["nonexistent.mkv"],
            polytope_payload=payload,
            encoder_grid_h=96,
            encoder_grid_w=128,
        )


def test_apply_l2_overlay_falls_back_to_sole_raw(tmp_path):
    """If basename-derived path missing but there's exactly one .raw, use it."""
    _write_synthetic_raw(tmp_path / "actual_name.raw", n_pairs=1)
    rng = np.random.RandomState(13)
    margin = torch.from_numpy(rng.rand(96, 128).astype(np.float32) + 1.0)
    payload = encode_polytope_payload(margin, jacobian_lipschitz=1.0, budget_bits=2000)
    diag = apply_l2_overlay_for_video_list(
        output_dir=tmp_path,
        video_names=["something_else.mkv"],
        polytope_payload=payload,
        encoder_grid_h=96,
        encoder_grid_w=128,
    )
    assert diag["videos_processed"] == 1


# ---------------------------------------------------------------------------
# Catalog #105 / #139 no-op detector visibility
# ---------------------------------------------------------------------------


def test_overlay_produces_observable_bytes_changed(tmp_path):
    """Catalog #105 requires that sidecar bytes are STRUCTURALLY CONSUMED by
    inflate. The overlay diagnostic must surface bytes_changed > 0 when the
    margin map has interior pixels and budget_bits > 0.
    """
    _write_synthetic_raw(tmp_path / "0.raw", n_pairs=1)
    rng = np.random.RandomState(31)
    margin = torch.from_numpy(rng.rand(96, 128).astype(np.float32) * 5 + 1.0)
    payload = encode_polytope_payload(margin, jacobian_lipschitz=1.0, budget_bits=4000)
    diag = apply_l2_overlay_for_video_list(
        output_dir=tmp_path,
        video_names=["0.mkv"],
        polytope_payload=payload,
        encoder_grid_h=96,
        encoder_grid_w=128,
    )
    # With interior margins and a 4000-bit budget the encoder allocates
    # nonzero noise; the overlay must apply some bytes.
    assert diag["total_pairs_modified"] >= 1
    assert diag["total_bytes_changed"] > 0


def test_overlay_contract_rejects_lipschitz_mismatch():
    margin_i8 = np.ones((2, 2), dtype=np.int8)
    noise = np.ones(4, dtype=np.int8)
    with pytest.raises(ValueError, match="jacobian_lipschitz mismatch"):
        validate_polytope_margin_contract(
            noise_levels_flat=noise,
            margin_map_int8=margin_i8,
            margin_map_scale=1.0,
            archive_jacobian_lipschitz=10.0,
            payload_jacobian_lipschitz=20.0,
        )


def test_overlay_contract_rejects_boundary_noise():
    margin_i8 = np.array([[0, 2], [0, 3]], dtype=np.int8)
    noise = np.array([1, 0, 0, -1], dtype=np.int8)
    with pytest.raises(ValueError, match="boundary violation"):
        validate_polytope_margin_contract(
            noise_levels_flat=noise,
            margin_map_int8=margin_i8,
            margin_map_scale=1.0,
            archive_jacobian_lipschitz=10.0,
            payload_jacobian_lipschitz=10.0,
        )


def test_overlay_contract_rejects_integer_safe_budget_violation():
    margin_i8 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    noise = np.array([2, 1, 1, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="safe-budget violation"):
        validate_polytope_margin_contract(
            noise_levels_flat=noise,
            margin_map_int8=margin_i8,
            margin_map_scale=1.0,
            archive_jacobian_lipschitz=2.0,
            payload_jacobian_lipschitz=2.0,
        )
