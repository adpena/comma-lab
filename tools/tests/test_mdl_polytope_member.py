from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.mdl_polytope_member import (
    CANDIDATE_ORDER,
    MdlMemberError,
    MdlPolytopeMemberSolver,
    fit_proxy,
    lawref_manifest,
    modular_uint8_residual,
    reconstruct_modular_uint8,
    residual_zlib9_bytes,
    saturated_integer_kernel_basis,
    zlib9_bytes,
)
from tac.optimization.resize_full_kernel import FullResizeKernel


@pytest.fixture(scope="module")
def solver() -> MdlPolytopeMemberSolver:
    kernel = FullResizeKernel.build(camera_h=8, camera_w=12, scorer_h=4, scorer_w=6)
    return MdlPolytopeMemberSolver(kernel, tile_scorer_hw=(2, 3))


def _canonical(solver: MdlPolytopeMemberSolver, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = rng.integers(
        0,
        256,
        size=(solver.kernel.scorer_h, solver.kernel.scorer_w, 3),
        dtype=np.uint8,
    )
    return solver.kernel.operator.realize_factor2_uint8(target)


def test_canonicalize_rejects_noninteger_plane(solver: MdlPolytopeMemberSolver) -> None:
    frame = _canonical(solver)
    frame[0, 0, 0] ^= np.uint8(1)
    with pytest.raises(MdlMemberError, match="integer scorer plane"):
        solver.canonicalize(frame)


def test_all_projected_candidates_preserve_exact_integer_numerators(
    solver: MdlPolytopeMemberSolver,
) -> None:
    canonical = _canonical(solver)
    temporal = _canonical(solver, seed=5678)
    expected, denominator = solver.kernel.operator.apply_numerators(canonical)
    candidates = solver.generate_candidates(canonical, temporal=temporal)
    assert tuple(candidates) == CANDIDATE_ORDER
    for candidate in candidates.values():
        actual, actual_denominator = solver.kernel.operator.apply_numerators(candidate)
        assert actual_denominator == denominator
        assert np.array_equal(actual, expected)
        assert candidate.dtype == np.uint8


def test_proxy_calibration_and_level_ordered_solve_are_coder_safe(
    solver: MdlPolytopeMemberSolver,
) -> None:
    canonical = _canonical(solver)
    temporal = _canonical(solver, seed=5678)
    candidates = solver.generate_candidates(canonical, temporal=temporal)
    features, actual, metadata = solver.calibration_rows(
        candidates, temporal=temporal, max_rows=48
    )
    calibration = fit_proxy(features, actual)
    assert calibration.row_count == 48
    assert len(metadata) == 48
    assert np.isfinite(calibration.pearson_r)
    labels = np.zeros((solver.kernel.scorer_h, solver.kernel.scorer_w), dtype=np.uint8)
    labels[:, labels.shape[1] // 2 :] = 1
    result = solver.solve(
        canonical,
        temporal=temporal,
        labels=labels,
        calibration=calibration,
    )
    assert [level.name for level in result.levels] == [
        "canonical",
        "chart",
        "object_class_stratum",
        "pixel_tile_residual",
    ]
    sizes = [level.coder_bytes for level in result.levels]
    assert sizes == sorted(sizes, reverse=True)
    assert result.selected_bytes <= zlib9_bytes(canonical)
    assert result.exact_numerators_equal


def test_stratum_classifier(solver: MdlPolytopeMemberSolver) -> None:
    assert solver.tile_stratum(np.zeros((3, 3), dtype=np.uint8)) == "cell"
    edge = np.zeros((3, 3), dtype=np.uint8)
    edge[:, 2] = 1
    assert solver.tile_stratum(edge) == "edge"
    saddle = np.asarray([[0, 1], [2, 2]], dtype=np.uint8)
    assert solver.tile_stratum(saddle) == "saddle"


def test_lawref_manifest_covers_all_numeric_defaults() -> None:
    rows = lawref_manifest()
    assert {row["name"] for row in rows} == {
        "resize_geometry_and_integer_kernel",
        "coder_level",
        "tile_scorer_hw",
        "calibration_rows",
        "ridge",
        "admission_threshold_delta_s_per_byte",
    }
    assert all(row["law_id"].endswith("_v1") for row in rows)


def test_saturated_integer_kernel_basis_is_exact_and_primitive() -> None:
    coefficients = np.asarray((103416, 181256, 182280, 319480), dtype=np.int64)
    basis = saturated_integer_kernel_basis(coefficients)
    assert basis.shape == (3, 4)
    assert np.array_equal(basis @ coefficients, np.zeros(3, dtype=np.int64))
    gram = basis.astype(object) @ basis.astype(object).T
    determinant = (
        gram[0, 0] * (gram[1, 1] * gram[2, 2] - gram[1, 2] * gram[2, 1])
        - gram[0, 1] * (gram[1, 0] * gram[2, 2] - gram[1, 2] * gram[2, 0])
        + gram[0, 2] * (gram[1, 0] * gram[2, 1] - gram[1, 1] * gram[2, 0])
    )
    primitive = coefficients // np.gcd.reduce(coefficients)
    assert determinant == sum(int(value) ** 2 for value in primitive)


def test_basis_summary_and_local_facet_dimensions_are_exact(
    solver: MdlPolytopeMemberSolver,
) -> None:
    summary = solver.basis_norm_summary()
    assert summary["count"] == 3 * solver.kernel.scorer_h * solver.kernel.scorer_w
    assert 0 < summary["norm_min"] <= summary["norm_p50"]
    assert summary["norm_p50"] <= summary["norm_p95"] <= summary["norm_max"]
    interior = np.full(
        (solver.kernel.camera_h, solver.kernel.camera_w, 3),
        128,
        dtype=np.uint8,
    )
    dimensions = solver.local_facet_dimensions(interior)
    assert dimensions.shape == (solver.kernel.scorer_h, solver.kernel.scorer_w, 3)
    assert np.all(dimensions == 3)
    lookup = solver._facet_dimension_lookup_cache
    assert lookup is not None
    second = solver.local_facet_dimensions(interior)
    assert solver._facet_dimension_lookup_cache is lookup
    np.testing.assert_array_equal(second, dimensions)


def test_modular_residual_is_bijective_and_origin_solve_is_coder_safe(
    solver: MdlPolytopeMemberSolver,
) -> None:
    canonical = _canonical(solver)
    origin = _canonical(solver, seed=5678)
    residual = modular_uint8_residual(canonical, origin)
    assert np.array_equal(reconstruct_modular_uint8(origin, residual), canonical)
    result = solver.solve_against_origin(canonical, origin=origin)
    assert result.selected_residual_bytes <= residual_zlib9_bytes(canonical, origin)
    assert result.exact_numerators_equal
    expected, denominator = solver.kernel.operator.apply_numerators(canonical)
    actual, actual_denominator = solver.kernel.operator.apply_numerators(result.selected)
    assert actual_denominator == denominator
    assert np.array_equal(actual, expected)
