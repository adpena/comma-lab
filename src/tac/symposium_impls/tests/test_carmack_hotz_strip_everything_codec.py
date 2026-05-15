# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.carmack_hotz_strip_everything_codec`."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.symposium_impls.carmack_hotz_strip_everything_codec import (
    DEFAULT_ITERATIVE_REFINE_PASSES,
    DEFAULT_POSE_BITS_PER_FRAME,
    DEFAULT_SEGNET_BITS_PER_PIXEL,
    DEFAULT_T4_DECODE_BUDGET_SECONDS,
    CarmackHotzCodecConfig,
    CarmackHotzPayload,
    IterativeRefineResult,
    compute_compression_ratio,
    compute_minimum_archive_payload,
    embodied_prior_initial_texture,
    iterative_refine_decode,
    update_from_anchor,
)


# ----- canonical defaults ---------------------------------------------------------------------


def test_default_segnet_bits_per_pixel_is_four() -> None:
    """log2(5) ≈ 2.32; rounded up to 4 bits per pixel."""
    assert DEFAULT_SEGNET_BITS_PER_PIXEL == 4


def test_default_pose_bits_per_frame_is_48() -> None:
    """6-DOF * 8-bit Lloyd-Max quantization."""
    assert DEFAULT_POSE_BITS_PER_FRAME == 48


def test_default_decode_budget_is_28_minutes() -> None:
    assert DEFAULT_T4_DECODE_BUDGET_SECONDS == 28.0 * 60.0


def test_default_refine_passes_is_5() -> None:
    assert DEFAULT_ITERATIVE_REFINE_PASSES == 5


# ----- config validation ----------------------------------------------------------------------


def test_config_default_values_well_formed() -> None:
    cfg = CarmackHotzCodecConfig(n_frames=600, mask_height=96, mask_width=128)
    assert cfg.n_frames == 600


def test_config_invalid_n_frames_raises() -> None:
    with pytest.raises(ValueError):
        CarmackHotzCodecConfig(n_frames=0, mask_height=96, mask_width=128)


def test_config_invalid_mask_dims_raise() -> None:
    with pytest.raises(ValueError):
        CarmackHotzCodecConfig(n_frames=1, mask_height=0, mask_width=128)
    with pytest.raises(ValueError):
        CarmackHotzCodecConfig(n_frames=1, mask_height=96, mask_width=-1)


def test_config_invalid_bits_raise() -> None:
    with pytest.raises(ValueError):
        CarmackHotzCodecConfig(
            n_frames=1, mask_height=96, mask_width=128, segnet_bits_per_pixel=0
        )
    with pytest.raises(ValueError):
        CarmackHotzCodecConfig(
            n_frames=1, mask_height=96, mask_width=128, pose_bits_per_frame=0
        )


def test_config_invalid_passes_raises() -> None:
    with pytest.raises(ValueError):
        CarmackHotzCodecConfig(
            n_frames=1, mask_height=96, mask_width=128, iterative_refine_passes=0
        )


def test_config_invalid_decode_budget_raises() -> None:
    with pytest.raises(ValueError):
        CarmackHotzCodecConfig(
            n_frames=1, mask_height=96, mask_width=128, decode_budget_seconds=0.0
        )


# ----- minimum payload tests -----------------------------------------------------------------


def test_minimum_payload_basic_calculation() -> None:
    """For 96x128 mask at 4 bits/pixel + 48 bits/pose:

    mask_bits_per_frame = 96 * 128 * 4 = 49152 bits → 6144 bytes
    pose_bytes_per_frame = ceil(48/8) = 6 bytes
    total per frame = 6144 + 6 = 6150 bytes.
    """
    cfg = CarmackHotzCodecConfig(n_frames=1, mask_height=96, mask_width=128)
    payload = compute_minimum_archive_payload(config=cfg)
    assert payload.mask_bytes_per_frame == 6144  # 96*128*4 bits / 8 = 6144 bytes
    assert payload.pose_bytes_per_frame == 6
    assert payload.total_archive_bytes == 6150


def test_minimum_payload_scales_with_n_frames() -> None:
    cfg = CarmackHotzCodecConfig(n_frames=600, mask_height=96, mask_width=128)
    payload = compute_minimum_archive_payload(config=cfg)
    assert payload.total_archive_bytes == 600 * 6150


# ----- embodied prior tests ------------------------------------------------------------------


def test_embodied_prior_returns_rgb_texture() -> None:
    masks = np.array([[0, 1], [1, 0]], dtype=np.int32)
    texture = embodied_prior_initial_texture(masks=masks)
    assert texture.shape == (2, 2, 3)


def test_embodied_prior_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        embodied_prior_initial_texture(masks=np.zeros((2, 2, 3), dtype=np.int32))


def test_embodied_prior_invalid_dtype_raises() -> None:
    with pytest.raises(ValueError):
        embodied_prior_initial_texture(masks=np.zeros((2, 2), dtype=np.float32))


def test_embodied_prior_upsampling() -> None:
    masks = np.array([[0, 1], [1, 0]], dtype=np.int32)
    texture = embodied_prior_initial_texture(masks=masks, height=4, width=4)
    assert texture.shape == (4, 4, 3)


def test_embodied_prior_deterministic_across_calls() -> None:
    masks = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
    a = embodied_prior_initial_texture(masks=masks, seed=42)
    b = embodied_prior_initial_texture(masks=masks, seed=42)
    assert np.allclose(a, b)


def test_embodied_prior_different_seeds_diverge() -> None:
    masks = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
    a = embodied_prior_initial_texture(masks=masks, seed=1)
    b = embodied_prior_initial_texture(masks=masks, seed=2)
    # Different seeds → different class colors for at least one class
    assert not np.allclose(a, b)


# ----- iterative refinement tests ------------------------------------------------------------


def test_iterative_refine_converges_to_target() -> None:
    """Per Banach: the contraction (1-α)x + αy converges to y for 0<α<1."""
    target = np.full((4, 4, 3), 100.0)
    initial = np.zeros_like(target)
    result = iterative_refine_decode(
        initial_texture=initial,
        receiver_target=target,
        n_passes=20,
        step_size=0.5,
    )
    assert result.final_residual_norm < 1.0


def test_iterative_refine_already_at_target_converges_first_pass() -> None:
    target = np.full((4, 4), 5.0)
    result = iterative_refine_decode(
        initial_texture=target.copy(),
        receiver_target=target,
        convergence_tolerance=1e-3,
    )
    assert result.converged
    assert result.n_passes == 1


def test_iterative_refine_step_size_one_lands_on_target_in_one_pass() -> None:
    """Step size 1.0 → texture^1 = target exactly."""
    target = np.array([1.0, 2.0, 3.0])
    initial = np.zeros(3)
    result = iterative_refine_decode(
        initial_texture=initial,
        receiver_target=target,
        n_passes=2,
        step_size=1.0,
        convergence_tolerance=1e-9,
    )
    assert np.allclose(result.final_texture, target)


def test_iterative_refine_invalid_step_size_raises() -> None:
    with pytest.raises(ValueError):
        iterative_refine_decode(
            initial_texture=np.zeros(3),
            receiver_target=np.zeros(3),
            step_size=0.0,
        )
    with pytest.raises(ValueError):
        iterative_refine_decode(
            initial_texture=np.zeros(3),
            receiver_target=np.zeros(3),
            step_size=2.5,
        )


def test_iterative_refine_invalid_passes_raises() -> None:
    with pytest.raises(ValueError):
        iterative_refine_decode(
            initial_texture=np.zeros(3),
            receiver_target=np.zeros(3),
            n_passes=0,
        )


def test_iterative_refine_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        iterative_refine_decode(
            initial_texture=np.zeros((2, 2)),
            receiver_target=np.zeros((3, 3)),
        )


# ----- compression ratio tests ---------------------------------------------------------------


def test_compression_ratio_above_one_when_smaller() -> None:
    payload = CarmackHotzPayload(
        n_frames=100,
        mask_bytes_per_frame=768,
        pose_bytes_per_frame=6,
        total_archive_bytes=77400,
        notes="",
    )
    ratio = compute_compression_ratio(payload=payload, baseline_archive_bytes=300_000)
    # 300K / 77.4K ≈ 3.876
    assert ratio == pytest.approx(300_000 / 77_400, abs=1e-9)


def test_compression_ratio_invalid_baseline_raises() -> None:
    payload = CarmackHotzPayload(
        n_frames=1, mask_bytes_per_frame=1, pose_bytes_per_frame=1, total_archive_bytes=2, notes=""
    )
    with pytest.raises(ValueError):
        compute_compression_ratio(payload=payload, baseline_archive_bytes=0)


def test_compression_ratio_zero_payload_returns_inf() -> None:
    payload = CarmackHotzPayload(
        n_frames=0, mask_bytes_per_frame=0, pose_bytes_per_frame=0, total_archive_bytes=0, notes=""
    )
    assert compute_compression_ratio(payload=payload, baseline_archive_bytes=1000) == float("inf")


# ----- continual learning hook ----------------------------------------------------------------


def test_update_from_anchor_no_n_frames_returns_none() -> None:
    assert update_from_anchor({}) is None


def test_update_from_anchor_invalid_n_frames_returns_none() -> None:
    assert update_from_anchor({"n_frames": "100"}) is None
    assert update_from_anchor({"n_frames": 0}) is None


def test_update_from_anchor_returns_payload() -> None:
    payload = update_from_anchor({"n_frames": 600})
    assert payload is not None
    assert payload.n_frames == 600
