# SPDX-License-Identifier: MIT
"""Behavioral tests for the full shared-resize kernel compiler."""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.resize_full_kernel import (
    UINT8_REACHABILITY_SEMANTICS,
    FullResizeKernel,
    FullResizeKernelError,
    _select_coder_admitted_name,
)

CAMERA_H, CAMERA_W = 8, 10
SCORER_H, SCORER_W = 3, 4


@pytest.fixture(scope="module")
def kernel() -> FullResizeKernel:
    return FullResizeKernel.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )


def test_axis_primitive_atoms_are_exact_integer_null_vectors(kernel):
    for axis in (kernel.height, kernel.width):
        for support, atom in zip(
            axis.supports, axis.primitive_null_atoms(), strict=True
        ):
            assert sum(
                numerator * value
                for numerator, value in zip(support.numerators, atom, strict=True)
            ) == 0


def test_axis_basis_count_is_exact_nullity(kernel):
    assert kernel.height.local_atom_count + len(kernel.height.unowned_indices) == (
        CAMERA_H - SCORER_H
    )


def test_integer_axis_coefficients_promote_before_primitive_atom_multiply(kernel):
    coefficients = np.zeros(kernel.height.nullity, dtype=np.int8)
    coefficients[0] = 1
    value = kernel.height.synthesize_kernel(coefficients)
    support = kernel.height.supports[0]
    atom = kernel.height.primitive_null_atoms()[0]
    assert value.dtype == np.int64
    assert tuple(value[list(support.indices)]) == atom
    assert kernel.width.local_atom_count + len(kernel.width.unowned_indices) == (
        CAMERA_W - SCORER_W
    )


def test_full_projector_maps_to_kernel(kernel):
    rng = np.random.default_rng(1)
    projected = kernel.project_kernel(rng.normal(size=(CAMERA_H, CAMERA_W)))
    assert np.max(np.abs(kernel.operator.apply(projected))) < 1e-12


def test_projector_decomposes_input(kernel):
    rng = np.random.default_rng(2)
    value = rng.normal(size=(CAMERA_H, CAMERA_W, 3))
    assert np.allclose(
        kernel.project_range(value) + kernel.project_kernel(value), value, atol=1e-12
    )


def test_kernel_projector_is_idempotent(kernel):
    rng = np.random.default_rng(3)
    once = kernel.project_kernel(rng.normal(size=(CAMERA_H, CAMERA_W)))
    twice = kernel.project_kernel(once)
    assert np.max(np.abs(twice - once)) < 1e-12


def test_range_and_kernel_are_orthogonal(kernel):
    rng = np.random.default_rng(4)
    value = rng.normal(size=(CAMERA_H, CAMERA_W))
    assert abs(float(np.vdot(kernel.project_range(value), kernel.project_kernel(value)))) < 1e-11


def test_fp32_projector_is_explicit_and_near_null(kernel):
    rng = np.random.default_rng(5)
    projected = kernel.project_kernel(
        rng.normal(size=(CAMERA_H, CAMERA_W)).astype(np.float32), dtype=np.float32
    )
    assert projected.dtype == np.float32
    assert np.max(np.abs(kernel.operator.apply(projected))) < 2e-5


def test_parameterization_synthesizes_null_image(kernel):
    rng = np.random.default_rng(6)
    left = rng.normal(size=(kernel.height.nullity, CAMERA_W, 2))
    right = rng.normal(size=(SCORER_H, kernel.width.nullity, 2))
    value = kernel.synthesize(left, right)
    assert value.shape == (CAMERA_H, CAMERA_W, 2)
    assert np.max(np.abs(kernel.operator.apply(value))) < 1e-12


def test_integer_parameterization_promotes_row_space_without_truncation(kernel):
    left = np.zeros((kernel.height.nullity, CAMERA_W), dtype=np.int64)
    right = np.zeros((SCORER_H, kernel.width.nullity), dtype=np.int64)
    left[0, 0] = 1
    right[0, 0] = 1
    value = kernel.synthesize(left, right)
    assert value.dtype == np.float64
    assert np.max(np.abs(kernel.operator.apply(value))) < 1e-12


def test_parameterization_is_nonredundant_on_small_geometry(kernel):
    n_left = kernel.height.nullity * CAMERA_W
    n_right = SCORER_H * kernel.width.nullity
    columns: list[np.ndarray] = []
    for flat_index in range(n_left + n_right):
        left = np.zeros((kernel.height.nullity, CAMERA_W))
        right = np.zeros((SCORER_H, kernel.width.nullity))
        if flat_index < n_left:
            left.reshape(-1)[flat_index] = 1.0
        else:
            right.reshape(-1)[flat_index - n_left] = 1.0
        columns.append(kernel.synthesize(left, right).reshape(-1))
    basis = np.stack(columns, axis=1)
    assert np.linalg.matrix_rank(basis) == CAMERA_H * CAMERA_W - SCORER_H * SCORER_W


def test_contest_coverage_closes_exact_real_linear_ceiling():
    coverage = FullResizeKernel.build().coverage().to_dict()
    assert coverage["domain_dimension_per_channel"] == 1_017_336
    assert coverage["resize_rank_per_channel"] == 196_608
    assert coverage["full_nullity_per_channel"] == 820_728
    assert coverage["old_zero_weight_nullity_per_channel"] == 230_904
    assert coverage["added_full_kernel_dimensions_per_channel"] == 589_824
    assert coverage["left_tensor_dimensions"] == 570_360
    assert coverage["right_tensor_dimensions"] == 250_368
    assert coverage["identity_check"] is True


def test_contest_per_axis_nullity_decomposition():
    coverage = FullResizeKernel.build().coverage().to_dict()
    assert coverage["height_axis_nullity"] == 490
    assert coverage["width_axis_nullity"] == 652
    assert coverage["height_axis_nullity_percent"] == pytest.approx(56.06407322654462)
    assert coverage["width_axis_nullity_percent"] == pytest.approx(56.013745704467354)


def test_uint8_center_reaches_every_canonical_basis_direction(kernel):
    frame = np.full((CAMERA_H, CAMERA_W, 3), 128, dtype=np.uint8)
    result = kernel.uint8_reachability(frame)
    assert result.to_dict()["semantics"] == UINT8_REACHABILITY_SEMANTICS
    assert result.feasible_basis_directions_lower_bound == result.full_basis_directions
    assert result.active_cell_channel_rank_histogram == (0, 0, 0, SCORER_H * SCORER_W * 3)


def test_uint8_reachability_is_honestly_labeled_lower_bound(kernel):
    frame = np.zeros((CAMERA_H, CAMERA_W, 1), dtype=np.uint8)
    payload = kernel.uint8_reachability(frame).to_dict()
    assert payload["is_lower_bound_on_full_bounded_lattice_intersection"] is True
    assert 0.0 <= payload["feasible_basis_fraction_lower_bound"] <= 1.0
    assert payload["zero_weight_coordinate_directions"] == 32


def test_full_kernel_fill_is_exact_and_never_worse_than_old_mask(kernel):
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    original_numerators, original_denominator = kernel.operator.apply_numerators(frame)
    result = kernel.compile_min_description_preimage(
        frame, preferences=("constant",), max_nodes_per_block=128
    )
    realized_numerators, realized_denominator = kernel.operator.apply_numerators(result.frame)
    assert realized_denominator == original_denominator
    assert np.array_equal(realized_numerators, original_numerators)
    assert result.selected_bytes["brotli"] <= result.old_mask_bytes["brotli"]
    assert result.selected_bytes["lzma"] <= result.old_mask_bytes["lzma"]
    assert all(candidate.exact_numerator_equal for candidate in result.candidates)


def test_coder_admission_rejects_cross_coder_regression_and_keeps_ties():
    sizes = {
        "old_zero_weight_mask": {"brotli": 100, "lzma": 100},
        "brotli_only_improvement": {"brotli": 90, "lzma": 101},
        "lzma_only_improvement": {"brotli": 101, "lzma": 90},
        "exact_tie": {"brotli": 100, "lzma": 100},
    }
    assert (
        _select_coder_admitted_name(sizes, baseline_name="old_zero_weight_mask")
        == "old_zero_weight_mask"
    )


def test_full_kernel_fill_does_not_mutate_input(kernel):
    frame = np.arange(CAMERA_H * CAMERA_W * 3, dtype=np.uint8).reshape(
        CAMERA_H, CAMERA_W, 3
    )
    before = frame.copy()
    kernel.compile_min_description_preimage(
        frame, preferences=("horizontal",), max_nodes_per_block=64
    )
    assert np.array_equal(frame, before)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((CAMERA_H, CAMERA_W), dtype=np.uint8),
        np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.float32),
        np.zeros((CAMERA_H - 1, CAMERA_W, 3), dtype=np.uint8),
    ],
)
def test_uint8_surfaces_fail_closed_on_bad_frames(kernel, bad):
    with pytest.raises(FullResizeKernelError):
        kernel.uint8_reachability(bad)


def test_parameterization_rejects_wrong_shapes(kernel):
    with pytest.raises(FullResizeKernelError):
        kernel.synthesize(
            np.zeros((kernel.height.nullity - 1, CAMERA_W)),
            np.zeros((SCORER_H, kernel.width.nullity)),
        )
