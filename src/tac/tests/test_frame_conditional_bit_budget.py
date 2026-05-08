from __future__ import annotations

import numpy as np
import pytest

from tac.codec.frame_conditional_bit_budget import (
    ComplexityComponents,
    allocate_per_frame_bits,
)


def test_complexity_components_multiplies_three_axes() -> None:
    components = ComplexityComponents(
        edge_density=np.array([1.0, 2.0, 3.0]),
        pixel_variance=np.array([10.0, 20.0, 30.0]),
        frame_difference=np.array([0.5, 0.25, 0.125]),
    )

    np.testing.assert_allclose(components.complexity, np.array([5.0, 10.0, 11.25]))


def test_allocate_per_frame_bits_eta_zero_is_uniform() -> None:
    out = allocate_per_frame_bits(
        np.array([1.0, 10.0, 100.0]),
        total_bit_budget=300.0,
        eta=0.0,
    )

    np.testing.assert_allclose(out, np.array([100.0, 100.0, 100.0]))


def test_allocate_per_frame_bits_preserves_sum_and_respects_floor_cap() -> None:
    out = allocate_per_frame_bits(
        np.array([1.0, 2.0, 100.0, 200.0]),
        total_bit_budget=400.0,
        eta=1.0,
        floor=0.5,
        cap=1.5,
    )

    assert float(out.sum()) == pytest.approx(400.0, abs=1e-6)
    assert float(out.min()) >= 50.0 - 1e-6
    assert float(out.max()) <= 150.0 + 1e-6
    assert out[-1] >= out[-2] >= out[1] >= out[0]


def test_allocate_per_frame_bits_zero_complexity_falls_back_to_uniform() -> None:
    out = allocate_per_frame_bits(
        np.zeros(4, dtype=np.float64),
        total_bit_budget=80.0,
        eta=2.0,
    )

    np.testing.assert_allclose(out, np.array([20.0, 20.0, 20.0, 20.0]))


def test_allocate_per_frame_bits_single_frame_gets_total() -> None:
    out = allocate_per_frame_bits([42.0], total_bit_budget=123.0)

    np.testing.assert_allclose(out, np.array([123.0]))


def test_allocate_per_frame_bits_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        allocate_per_frame_bits([], 10.0)
    with pytest.raises(ValueError, match="1-D"):
        allocate_per_frame_bits(np.zeros((2, 2)), 10.0)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_per_frame_bits([1.0, -1.0], 10.0)
    with pytest.raises(ValueError, match="total_bit_budget"):
        allocate_per_frame_bits([1.0], -1.0)
    with pytest.raises(ValueError, match="floor"):
        allocate_per_frame_bits([1.0, 2.0], 10.0, floor=1.2)
    with pytest.raises(ValueError, match="cap"):
        allocate_per_frame_bits([1.0, 2.0], 10.0, cap=0.9)
