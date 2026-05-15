# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.uniward_die_distortion_informed_embedding_map`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.symposium_impls.uniward_die_distortion_informed_embedding_map import (
    DEFAULT_UNIWARD_EPSILON,
    UniwardCompositeWeights,
    UniwardDIECostMap,
    compose_bit_allocation_map,
    compute_die_blind_region,
    compute_uniward_cost_map,
    horizontal_wavelet_band,
    load_cached_uniward_die_map,
    save_uniward_die_map,
    update_from_anchor,
    vertical_wavelet_band,
)


# ----- composite weights validation -------------------------------------------------------------


def test_uniform_weights_must_sum_to_one() -> None:
    UniwardCompositeWeights(alpha=0.5, beta=0.3, gamma=0.2)


def test_weights_summing_to_other_value_raises() -> None:
    with pytest.raises(ValueError):
        UniwardCompositeWeights(alpha=0.5, beta=0.5, gamma=0.5)


def test_negative_weight_raises() -> None:
    with pytest.raises(ValueError):
        UniwardCompositeWeights(alpha=-0.1, beta=0.5, gamma=0.6)


# ----- wavelet band tests ----------------------------------------------------------------------


def test_horizontal_band_constant_image_is_zero() -> None:
    img = np.full((4, 4), 5.0, dtype=np.float64)
    band = horizontal_wavelet_band(img)
    assert np.allclose(band, 0.0)
    assert band.shape == img.shape


def test_horizontal_band_strong_edge_detected() -> None:
    img = np.zeros((4, 4), dtype=np.float64)
    img[:, 2:] = 1.0  # vertical edge at column 2
    band = horizontal_wavelet_band(img)
    # Column 1 holds the diff between column 1 and column 2 (= 1.0)
    assert band[0, 1] == pytest.approx(1.0)


def test_vertical_band_strong_edge_detected() -> None:
    img = np.zeros((4, 4), dtype=np.float64)
    img[2:, :] = 1.0
    band = vertical_wavelet_band(img)
    assert band[1, 0] == pytest.approx(1.0)


# ----- UNIWARD cost map tests ------------------------------------------------------------------


def test_uniward_cost_constant_image_is_max_cost() -> None:
    """Flat (zero detail) → cost ≈ 1/epsilon (high cost; preserve at high precision)."""
    img = np.full((8, 8), 10.0, dtype=np.float64)
    cost = compute_uniward_cost_map(img)
    expected = 1.0 / DEFAULT_UNIWARD_EPSILON
    assert np.allclose(cost, expected)


def test_uniward_cost_textured_image_has_lower_max_cost() -> None:
    """Textured (high detail) → some pixels have low cost (< 1/epsilon)."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((16, 16)) * 10
    cost = compute_uniward_cost_map(img)
    flat_cost = 1.0 / DEFAULT_UNIWARD_EPSILON
    # Most pixels should have detail-band response > 0; cost < flat
    assert (cost < flat_cost).sum() > cost.size // 2


def test_uniward_cost_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        compute_uniward_cost_map(np.zeros((4, 4, 3)))


def test_uniward_cost_invalid_epsilon_raises() -> None:
    with pytest.raises(ValueError):
        compute_uniward_cost_map(np.zeros((4, 4)), epsilon=0.0)


def test_uniward_cost_empty_image() -> None:
    cost = compute_uniward_cost_map(np.zeros((0, 0)))
    assert cost.shape == (0, 0)


# ----- DIE blind region tests ------------------------------------------------------------------


def test_die_blind_region_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        compute_die_blind_region(0, 4)
    with pytest.raises(ValueError):
        compute_die_blind_region(4, -1)
    with pytest.raises(ValueError):
        compute_die_blind_region(4, 4, downsample_factor=0)


def test_die_blind_region_default_factor_2_has_canonical_pattern() -> None:
    """For 2x2 stride: bottom-right (1,1) survives → blind=0; rest are 0.5 (edge)."""
    blind = compute_die_blind_region(4, 4, downsample_factor=2)
    assert blind[1, 1] == 0.0
    assert blind[3, 3] == 0.0
    assert blind[1, 0] == 0.5  # edge cell
    assert blind[0, 1] == 0.5  # edge cell


def test_die_blind_region_factor_3_has_center_blindness() -> None:
    """For 3x3 stride: center (1,1) is the off-corner non-edge → 0.75 blind."""
    blind = compute_die_blind_region(3, 3, downsample_factor=3)
    assert blind[2, 2] == 0.0  # survivor
    assert blind[1, 1] == 0.75  # center blind


def test_die_blind_region_in_zero_one_range() -> None:
    blind = compute_die_blind_region(8, 8)
    assert blind.min() >= 0.0
    assert blind.max() <= 1.0


# ----- composite bit allocation ----------------------------------------------------------------


def test_composite_bit_allocation_uniform_weights() -> None:
    cost = compute_uniward_cost_map(np.random.default_rng(1).standard_normal((8, 8)))
    composite = compose_bit_allocation_map(cost)
    assert composite.shape == cost.shape
    assert composite.min() >= 0.0


def test_composite_with_die_blind_zeroes_blind_pixels() -> None:
    cost = compute_uniward_cost_map(np.full((4, 4), 5.0))
    blind = compute_die_blind_region(4, 4)
    composite = compose_bit_allocation_map(cost, die_blind=blind)
    survivors = composite[1::2, 1::2]
    # Survivor pixels can be non-zero; blind pixels are scaled down
    assert (composite <= cost.max() + 1e-9).all()


def test_composite_invalid_attention_shape_raises() -> None:
    cost = np.zeros((4, 4))
    with pytest.raises(ValueError):
        compose_bit_allocation_map(cost, attention=np.zeros((3, 3)))


def test_composite_invalid_difficulty_shape_raises() -> None:
    cost = np.zeros((4, 4))
    with pytest.raises(ValueError):
        compose_bit_allocation_map(cost, difficulty=np.zeros((5, 5)))


def test_composite_invalid_die_shape_raises() -> None:
    cost = np.zeros((4, 4))
    with pytest.raises(ValueError):
        compose_bit_allocation_map(cost, die_blind=np.zeros((3, 3)))


def test_composite_2d_cost_required() -> None:
    with pytest.raises(ValueError):
        compose_bit_allocation_map(np.zeros((4, 4, 3)))


# ----- save / load round trip -----------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    img = rng.standard_normal((8, 8))
    cost = compute_uniward_cost_map(img)
    blind = compute_die_blind_region(8, 8)
    composite = compose_bit_allocation_map(cost, die_blind=blind)
    state_path = tmp_path / "uniward.npz"
    save_uniward_die_map(
        uniward_cost_map=cost,
        die_blind_region=blind,
        composite_bit_allocation=composite,
        state_path=state_path,
    )
    loaded = load_cached_uniward_die_map(state_path=state_path)
    assert loaded is not None
    cost_loaded, blind_loaded, composite_loaded, bundle = loaded
    assert np.allclose(cost_loaded, cost)
    assert np.allclose(blind_loaded, blind)
    assert np.allclose(composite_loaded, composite)
    assert bundle.evidence_grade == "research-only-cost-map"
    assert bundle.score_claim is False


def test_load_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_cached_uniward_die_map(state_path=tmp_path / "absent.npz") is None


# ----- continual-learning hook -----------------------------------------------------------------


def test_update_from_anchor_with_no_image_returns_none(tmp_path: Path) -> None:
    state_path = tmp_path / "uniward.npz"
    assert update_from_anchor({}, state_path=state_path) is None


def test_update_from_anchor_grayscale_image(tmp_path: Path) -> None:
    state_path = tmp_path / "uniward.npz"
    img = np.random.default_rng(7).standard_normal((16, 16))
    bundle = update_from_anchor({}, image=img, state_path=state_path)
    assert bundle is not None
    assert bundle.height == 16
    assert state_path.is_file()


def test_update_from_anchor_rgb_image_reduces_to_luma(tmp_path: Path) -> None:
    state_path = tmp_path / "uniward.npz"
    img = np.random.default_rng(7).standard_normal((16, 16, 3))
    bundle = update_from_anchor({}, image=img, state_path=state_path)
    assert bundle is not None
    assert bundle.height == 16
    assert bundle.width == 16
