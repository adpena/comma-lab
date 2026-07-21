from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.mdl_polytope_member import (
    CANDIDATE_ORDER,
    MdlMemberError,
    MdlPolytopeMemberSolver,
    fit_proxy,
    lawref_manifest,
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
