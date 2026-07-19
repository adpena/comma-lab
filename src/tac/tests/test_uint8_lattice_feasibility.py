from __future__ import annotations

import importlib.util
import struct
import sys
import zlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import (
    MAX_UINT8_FRAME_BYTES,
    AxisSupport,
    BlockSolveStatus,
    CandidateProvenance,
    DisjointResizeOperator,
    HardOracleEvaluation,
    IntegerMove,
    RepairStatus,
    Uint8LatticeError,
    parse_uint8_frame,
    realize_factor2_uint8_scorer_plane,
    repair_with_hard_oracle,
    serialize_uint8_frame,
    solve_bounded_integer_block,
    verify_factor2_uint8_scorer_plane,
)


def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)


_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_CANDIDATES = (
    _WORKTREE_ROOT / "upstream",
    Path("/Users/adpena/Projects/pact/upstream"),
)
_FROZEN_UPSTREAM = next(
    (
        path
        for path in _UPSTREAM_CANDIDATES
        if (path / "modules.py").is_file()
        and (path / "models/segnet.safetensors").is_file()
    ),
    None,
)


def _measurement_tool() -> ModuleType:
    name = "_uint8_lattice_measurement_tool_for_tests"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    path = _WORKTREE_ROOT / "tools/measure_uint8_lattice_feasibility.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_disjoint_operator_ab_identity_and_bounded_continuous_preimage() -> None:
    operator = _operator()
    target = np.linspace(0.0, 255.0, 3 * 4 * 2).reshape(3, 4, 2)
    real = operator.minimum_norm_real_preimage(target)
    np.testing.assert_allclose(operator.apply(real), target, atol=2e-12)
    bounded = operator.bounded_continuous_preimage(target, reference=real)
    assert np.min(bounded) >= 0.0
    assert np.max(bounded) <= 255.0
    np.testing.assert_allclose(operator.apply(bounded), target, atol=2e-9)


def test_factor2_constant_support_realization_is_integer_exact_and_canonical() -> None:
    operator = _operator()
    target = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3) * 7
    frame = realize_factor2_uint8_scorer_plane(operator, target)
    proof = verify_factor2_uint8_scorer_plane(operator, frame, target)
    numerators, denominator = operator.apply_numerators(frame)
    np.testing.assert_array_equal(numerators, target.astype(np.int64) * denominator)
    assert proof.numerator_exact is True
    assert proof.certified_exact is True
    assert proof.numerator_equal_values == target.size
    assert proof.canonical_equal_values == frame.size
    assert proof.owned_camera_values + proof.unowned_camera_values == frame.size


def test_factor2_realization_zeros_every_unowned_camera_coordinate() -> None:
    operator = _operator()
    target = np.full((3, 4, 1), 173, dtype=np.uint8)
    frame = operator.realize_factor2_uint8(target)
    owned = np.zeros((operator.camera_h, operator.camera_w), dtype=bool)
    for row_support in operator.row_supports:
        for col_support in operator.col_supports:
            owned[np.ix_(row_support.indices, col_support.indices)] = True
    assert np.all(frame[owned] == 173)
    assert np.all(frame[~owned] == 0)


@pytest.mark.parametrize(
    "target",
    [
        np.zeros((3, 4, 1), dtype=np.int16),
        np.zeros((3, 4, 0), dtype=np.uint8),
        np.zeros((4, 4, 1), dtype=np.uint8),
    ],
)
def test_factor2_realization_refuses_dtype_channel_and_geometry_drift(
    target: np.ndarray,
) -> None:
    with pytest.raises(Uint8LatticeError, match="factor-2 scorer plane"):
        _operator().realize_factor2_uint8(target)


def test_minimum_norm_real_preimage_refuses_finite_input_that_overflows_output() -> None:
    target = np.full((3, 4, 1), np.finfo(np.float64).max, dtype=np.float64)
    with pytest.raises(Uint8LatticeError, match="preimage produced non-finite"):
        _operator().minimum_norm_real_preimage(target)


def test_operator_refuses_overlapping_resize_supports() -> None:
    with pytest.raises(Uint8LatticeError, match="overlapping"):
        DisjointResizeOperator.build(camera_h=4, camera_w=4, scorer_h=3, scorer_w=3)


def test_operator_direct_constructor_refuses_uncertified_float_supports() -> None:
    certified = _operator()
    first = certified.row_supports[0]
    malformed = AxisSupport(
        indices=first.indices,
        numerators=(float(first.numerators[0]), *first.numerators[1:]),
        denominator=first.denominator,
        weights=first.weights,
    )
    with pytest.raises(Uint8LatticeError, match="derived half-pixel integer geometry"):
        DisjointResizeOperator(
            camera_h=certified.camera_h,
            camera_w=certified.camera_w,
            scorer_h=certified.scorer_h,
            scorer_w=certified.scorer_w,
            row_supports=(malformed, *certified.row_supports[1:]),
            col_supports=certified.col_supports,
        )


@pytest.mark.parametrize(
    "bad_weight",
    [True, complex(1.0, 0.0), "1.0", float("nan"), float("inf")],
)
def test_operator_direct_constructor_refuses_untyped_or_nonfinite_weights(
    bad_weight: object,
) -> None:
    certified = _operator()
    first = certified.row_supports[0]
    malformed = AxisSupport(
        indices=first.indices,
        numerators=first.numerators,
        denominator=first.denominator,
        weights=(bad_weight, *first.weights[1:]),  # type: ignore[arg-type]
    )
    with pytest.raises(Uint8LatticeError, match="derived half-pixel integer geometry"):
        DisjointResizeOperator(
            camera_h=certified.camera_h,
            camera_w=certified.camera_w,
            scorer_h=certified.scorer_h,
            scorer_w=certified.scorer_w,
            row_supports=(malformed, *certified.row_supports[1:]),
            col_supports=certified.col_supports,
        )


@pytest.mark.parametrize(
    "frame",
    [
        np.full((8, 10, 1), np.iinfo(np.int64).max, dtype=np.int64),
        np.full((8, 10, 1), np.iinfo(np.uint64).max, dtype=np.uint64),
        np.full((8, 10, 1), -1, dtype=np.int64),
        np.full((8, 10, 1), 256, dtype=np.int16),
        np.zeros((8, 10, 1), dtype=bool),
        np.zeros((8, 10, 1), dtype=object),
        np.zeros((8, 10, 1), dtype=np.float64),
    ],
)
def test_apply_numerators_rejects_non_uint8_lattice_or_unsafe_values(
    frame: np.ndarray,
) -> None:
    with pytest.raises(Uint8LatticeError):
        _operator().apply_numerators(frame)


def test_apply_numerators_accepts_bounded_non_uint8_integer_arrays_exactly() -> None:
    operator = _operator()
    bounded = (np.arange(8 * 10 * 2).reshape(8, 10, 2) % 256).astype(np.int16)
    expected, expected_denominator = operator.apply_numerators(
        bounded.astype(np.uint8)
    )
    actual, actual_denominator = operator.apply_numerators(bounded)
    assert actual.dtype == np.int64
    assert actual_denominator == expected_denominator
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_float_apply_rejects_nonfinite_camera_values(bad_value: float) -> None:
    frame = np.zeros((8, 10, 1), dtype=np.float64)
    frame[0, 0, 0] = bad_value
    with pytest.raises(Uint8LatticeError, match="must be finite"):
        _operator().apply(frame)


@pytest.mark.parametrize(
    "dtype",
    [np.bool_, np.complex128, np.str_, object],
)
def test_float_apply_rejects_non_real_or_coercive_dtypes(dtype: object) -> None:
    frame = np.zeros((8, 10, 1), dtype=dtype)  # type: ignore[arg-type]
    with pytest.raises(Uint8LatticeError, match="real numeric non-boolean"):
        _operator().apply(frame)


def test_apply_and_exact_apply_reject_empty_channels() -> None:
    empty = np.empty((8, 10, 0), dtype=np.uint8)
    with pytest.raises(Uint8LatticeError, match="nonempty channels"):
        _operator().apply(empty)
    with pytest.raises(Uint8LatticeError, match="nonempty channels"):
        _operator().apply_numerators(empty)
    empty_spatial = np.empty((0, 10, 1), dtype=np.uint8)
    with pytest.raises(Uint8LatticeError, match="shape"):
        _operator().apply(empty_spatial)


@pytest.mark.parametrize(
    "dtype",
    [np.bool_, np.complex128, np.str_, object],
)
def test_target_and_reference_reject_non_real_or_coercive_dtypes(
    dtype: object,
) -> None:
    operator = _operator()
    target = np.zeros((3, 4, 1), dtype=dtype)  # type: ignore[arg-type]
    with pytest.raises(Uint8LatticeError, match="target must have a real numeric"):
        operator.minimum_norm_real_preimage(target)

    valid_target = np.zeros((3, 4, 1), dtype=np.float64)
    reference = np.zeros((8, 10, 1), dtype=dtype)  # type: ignore[arg-type]
    with pytest.raises(Uint8LatticeError, match="reference must have a real numeric"):
        operator.bounded_continuous_preimage(valid_target, reference=reference)


def test_exact_diophantine_solver_finds_distant_non_corner_solution() -> None:
    # Target has the distant exact witness (200, 17, 31, 99).  The preferred
    # point is deliberately unrelated, so adjacent-corner rounding cannot pass.
    coefficients = (21, 15, 7, 5)
    denominator = sum(coefficients)
    target = (21 * 200 + 15 * 17 + 7 * 31 + 5 * 99) / denominator
    preferred = np.array([10.2, 10.2, 10.2, 10.2])
    result = solve_bounded_integer_block(
        coefficients,
        denominator,
        target,
        target_integer=round(target * denominator),
        preferred=preferred,
        max_nodes=100_000,
    )
    assert result.status is BlockSolveStatus.FEASIBLE_EXACT
    assert result.candidate_provenance is CandidateProvenance.EXACT_FEASIBLE_POINT
    assert sum(c * x for c, x in zip(coefficients, result.values, strict=True)) == round(target * denominator)
    assert result.projection_residual < 1e-12
    fallback = solve_bounded_integer_block(coefficients, denominator, target, preferred=preferred, max_nodes=100_000)
    assert fallback.status is BlockSolveStatus.HEURISTIC_CANDIDATE
    assert fallback.candidate_provenance is CandidateProvenance.ADJACENT_CORNER_FALLBACK
    assert fallback.projection_residual > 0.0


def test_block_solver_distinguishes_proof_budget_and_heuristic() -> None:
    impossible = solve_bounded_integer_block((2, 4), 6, 0.5, target_integer=3, max_nodes=100)
    assert impossible.status is BlockSolveStatus.INFEASIBLE_EXHAUSTIVE
    assert impossible.candidate_provenance is CandidateProvenance.ADJACENT_CORNER_FALLBACK
    budget = solve_bounded_integer_block(
        (21, 15, 7, 5),
        48,
        77.0,
        target_integer=77 * 48,
        preferred=(0, 0, 0, 0),
        max_nodes=1,
    )
    assert budget.status is BlockSolveStatus.NOT_FOUND_BUDGET
    assert budget.candidate_provenance is CandidateProvenance.ADJACENT_CORNER_FALLBACK
    nonrational = solve_bounded_integer_block((1, 1), 2, 0.123456, max_nodes=100)
    assert nonrational.status is BlockSolveStatus.HEURISTIC_CANDIDATE
    assert nonrational.candidate_provenance is CandidateProvenance.ADJACENT_CORNER_FALLBACK
    assert nonrational.target_integer is None


@pytest.mark.parametrize("coefficient_count", [5, 30])
def test_block_solver_refuses_more_than_factor2_four_taps_before_fallback(
    coefficient_count: int,
) -> None:
    with pytest.raises(Uint8LatticeError, match="at most four"):
        solve_bounded_integer_block(
            (1,) * coefficient_count,
            coefficient_count,
            0.0,
            target_integer=0,
            max_nodes=1,
        )


@pytest.mark.parametrize(
    (
        "coefficients",
        "common_denominator",
        "target",
        "target_integer",
        "max_nodes",
        "tolerance",
    ),
    [
        ((1.9, 1), 2, 0.5, 1, 10, 0.0),
        ((2.0, 1), 3, 1.0, 3, 10, 0.0),
        ((True, 1), 2, 0.5, 1, 10, 0.0),
        ((1, 1), 2.5, 0.5, 1, 10, 0.0),
        ((1, 1), True, 0.5, 1, 10, 0.0),
        ((1, 1), 2, float("nan"), 1, 10, 0.0),
        ((1, 1), 2, True, 1, 10, 0.0),
        ((1, 1), 2, 0.5, 1.5, 10, 0.0),
        ((1, 1), 2, 0.5, True, 10, 0.0),
        ((1, 1), 2, 0.5, 1, True, 0.0),
        ((1, 1), 2, 0.5, 1, 10, float("nan")),
        ((1, 1), 2, 0.5, 1, 10, -1e-9),
        ((1, 1), 2, 0.5, 1, 10, True),
    ],
)
def test_exact_block_solver_refuses_coercive_or_nonfinite_certificate_inputs(
    coefficients: tuple[object, ...],
    common_denominator: object,
    target: object,
    target_integer: object,
    max_nodes: object,
    tolerance: object,
) -> None:
    with pytest.raises(Uint8LatticeError):
        solve_bounded_integer_block(
            coefficients,  # type: ignore[arg-type]
            common_denominator,  # type: ignore[arg-type]
            target,  # type: ignore[arg-type]
            target_integer=target_integer,  # type: ignore[arg-type]
            max_nodes=max_nodes,  # type: ignore[arg-type]
            target_verification_tolerance=tolerance,  # type: ignore[arg-type]
        )


def test_exact_block_solver_accepts_numpy_integer_scalar_custody() -> None:
    result = solve_bounded_integer_block(
        (np.int64(1), np.int32(1)),
        np.int64(2),
        np.float64(0.5),
        target_integer=np.int32(1),
        max_nodes=np.int64(10),
        target_verification_tolerance=np.float64(0.0),
    )
    assert result.status is BlockSolveStatus.FEASIBLE_EXACT
    assert sum(result.values) == 1
    assert result.common_denominator == 2
    assert result.target_integer == 1


def test_exact_block_solver_caller_tolerance_cannot_authorize_target_mismatch() -> None:
    with pytest.raises(Uint8LatticeError, match="fixed machine-derived bound"):
        solve_bounded_integer_block(
            (1, 1),
            2,
            1.5,
            target_integer=1,
            target_verification_tolerance=1.0,
        )


def test_exact_block_solver_dbl_max_observation_cannot_expand_match_bound() -> None:
    maximum = np.finfo(np.float64).max
    with pytest.raises(Uint8LatticeError, match="fixed machine-derived bound"):
        solve_bounded_integer_block(
            (1, 1),
            2,
            maximum,
            target_integer=1,
            target_verification_tolerance=maximum,
        )


@pytest.mark.parametrize(
    "preferred",
    [
        np.array([True, False]),
        np.array([1 + 0j, 2 + 0j]),
        np.array(["1", "2"]),
        np.array([1.0, 2.0], dtype=object),
    ],
)
def test_block_solver_rejects_coercive_preferred_dtype(
    preferred: np.ndarray,
) -> None:
    with pytest.raises(Uint8LatticeError, match="preferred must have a real numeric"):
        solve_bounded_integer_block(
            (1, 1),
            2,
            0.5,
            target_integer=1,
            preferred=preferred,
        )


@pytest.mark.parametrize("preferred", [np.array([-1.0, 0.0]), np.array([0.0, 256.0])])
def test_block_solver_rejects_preferred_outside_uint8_box(
    preferred: np.ndarray,
) -> None:
    with pytest.raises(Uint8LatticeError, match="uint8 box"):
        solve_bounded_integer_block(
            (1, 1),
            2,
            0.5,
            target_integer=1,
            preferred=preferred,
        )


def test_frame_exact_scalar_surfaces_reject_bools_and_nonfinite_tolerance() -> None:
    with pytest.raises(Uint8LatticeError, match="camera_h must be an integer"):
        DisjointResizeOperator.build(
            camera_h=True, camera_w=10, scorer_h=3, scorer_w=4
        )
    operator = _operator()
    target = np.zeros((3, 4, 1), dtype=np.float64)
    numerators = np.zeros((3, 4, 1), dtype=np.int64)
    with pytest.raises(Uint8LatticeError, match="max_nodes_per_block"):
        operator.solve_uint8(
            target, target_numerators=numerators, max_nodes_per_block=True
        )
    with pytest.raises(Uint8LatticeError, match="target_verification_tolerance"):
        operator.solve_uint8(
            target,
            target_numerators=numerators,
            target_verification_tolerance=float("nan"),
        )


def test_frame_cannot_certify_mismatched_target_via_widened_tolerance() -> None:
    operator = _operator()
    numerators = np.zeros((3, 4, 1), dtype=np.int64)
    mismatched_target = np.zeros((3, 4, 1), dtype=np.float64)
    mismatched_target[0, 0, 0] = 1.0
    with pytest.raises(Uint8LatticeError, match="fixed machine-derived bound"):
        operator.solve_uint8(
            mismatched_target,
            target_numerators=numerators,
            target_verification_tolerance=1.0,
        )


def test_frame_solver_rejects_empty_target_channels_without_raw_reshape_error() -> None:
    operator = _operator()
    empty_target = np.empty((3, 4, 0), dtype=np.float64)
    empty_numerators = np.empty((3, 4, 0), dtype=np.int64)
    with pytest.raises(Uint8LatticeError, match="target shape/value"):
        operator.solve_uint8(
            empty_target,
            target_numerators=empty_numerators,
        )


def test_fixed_rational_float_bound_accepts_real_operator_rounding_only() -> None:
    operator = _operator()
    source = (
        np.arange(8 * 10 * 3, dtype=np.uint16).reshape(8, 10, 3) % 256
    ).astype(np.uint8)
    numerators, denominator = operator.apply_numerators(source)
    target_from_float_operator = operator.apply(source)
    exact_target = numerators / denominator
    assert np.max(np.abs(target_from_float_operator - exact_target)) > 0.0
    result = operator.solve_uint8(
        target_from_float_operator,
        target_numerators=numerators,
        max_nodes_per_block=50_000,
    )
    assert result.certified_exact


def test_frame_solver_is_deterministic_uint8_and_beats_clip_round() -> None:
    operator = _operator()
    # A saturated valid target makes the minimum-norm lift concentrate mass in
    # the largest-weight coordinate beyond 255; clipping then breaks A x = y.
    source = np.full((8, 10, 1), 255, dtype=np.uint8)
    target_numerators, denominator = operator.apply_numerators(source)
    target = target_numerators / denominator
    np.testing.assert_allclose(operator.apply(source), target, atol=2e-12)
    real = operator.minimum_norm_real_preimage(target)
    baseline = np.clip(np.rint(real), 0, 255).astype(np.uint8)
    baseline_error = float(np.max(np.abs(operator.apply(baseline) - target)))
    first = operator.solve_uint8(
        target,
        target_numerators=target_numerators,
        reference=real,
        max_nodes_per_block=50_000,
    )
    second = operator.solve_uint8(
        target,
        target_numerators=target_numerators,
        reference=real,
        max_nodes_per_block=50_000,
    )
    assert first.frame.dtype == np.uint8
    assert first.frame.shape == source.shape
    assert np.array_equal(first.frame, second.frame)
    assert first.diagnostics.exact_blocks == target.size
    assert first.diagnostics.exact_candidate_blocks == target.size
    assert first.diagnostics.heuristic_blocks == 0
    assert first.certified_exact
    assert first.aggregate_status is BlockSolveStatus.FEASIBLE_EXACT
    assert first.diagnostics.max_projection_discrepancy < 1e-10
    assert first.diagnostics.max_projection_discrepancy < baseline_error
    assert first.diagnostics.out_of_gamut_before_bounded_solve > 0
    with pytest.raises(ValueError, match="read-only"):
        first.frame[0, 0, 0] = 0
    with pytest.raises(ValueError):
        first.frame.setflags(write=True)


def test_solve_reference_is_only_a_target_derived_integrity_assertion() -> None:
    operator = _operator()
    source = np.full((8, 10, 1), 255, dtype=np.uint8)
    numerators, denominator = operator.apply_numerators(source)
    target = numerators / denominator
    target_derived = operator.minimum_norm_real_preimage(target)
    accepted = operator.solve_uint8(
        target,
        target_numerators=numerators,
        reference=target_derived,
        max_nodes_per_block=50_000,
    )
    assert accepted.certified_exact
    with pytest.raises(Uint8LatticeError, match="source-dependent preferences are forbidden"):
        operator.solve_uint8(
            target,
            target_numerators=numerators,
            reference=source,
            max_nodes_per_block=50_000,
        )


def test_float_target_alone_is_uncertified_and_integer_custody_fails_closed() -> None:
    operator = _operator()
    source = np.full((8, 10, 1), 127, dtype=np.uint8)
    numerators, denominator = operator.apply_numerators(source)
    target = numerators / denominator
    heuristic = operator.solve_uint8(target, max_nodes_per_block=100)
    assert heuristic.diagnostics.heuristic_blocks == target.size
    assert heuristic.diagnostics.exact_candidate_blocks == 0
    assert not heuristic.certified_exact
    assert heuristic.aggregate_status is BlockSolveStatus.HEURISTIC_CANDIDATE
    corrupted = numerators.copy()
    corrupted[0, 0, 0] += 1
    with pytest.raises(Uint8LatticeError, match="disagrees"):
        operator.solve_uint8(target, target_numerators=corrupted)


def test_frame_diagnostics_keep_proof_and_candidate_provenance_orthogonal() -> None:
    operator = DisjointResizeOperator.build(
        camera_h=4, camera_w=4, scorer_h=2, scorer_w=2
    )
    # Every block coefficient is divisible by four, so numerator one is an
    # exhaustively certified affine impossibility.  The returned all-zero
    # camera block is still only an adjacent-corner fallback candidate.
    numerators = np.ones((2, 2, 1), dtype=np.int64)
    target = numerators / 16.0
    result = operator.solve_uint8(
        target,
        target_numerators=numerators,
        max_nodes_per_block=10_000,
    )
    assert result.aggregate_status is BlockSolveStatus.INFEASIBLE_EXHAUSTIVE
    assert not result.certified_exact
    assert result.diagnostics.exact_blocks == 0
    assert result.diagnostics.exact_candidate_blocks == 0
    assert result.diagnostics.proven_affine_infeasible_blocks == target.size
    assert result.diagnostics.heuristic_blocks == target.size


def test_hard_oracle_positive_canary_moves_then_passes_monotonically() -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def oracle(candidate: np.ndarray) -> HardOracleEvaluation:
        value = int(candidate[0, 0, 0])
        return HardOracleEvaluation(
            satisfied=np.array([value >= 2]),
            margins=np.array([value - 2.0]),
            proposals=() if value >= 2 else (IntegerMove(0, 0, 0, 1),),
        )

    result = repair_with_hard_oracle(frame, operator, oracle, max_iterations=4)
    assert result.status is RepairStatus.FEASIBLE
    assert result.frame[0, 0, 0] == 2
    assert all(step.key_after < step.key_before for step in result.iterations)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": True},
        {"max_iterations": 1.5},
        {"max_proposals_per_iteration": True},
        {"max_proposals_per_iteration": 1.5},
        {"max_projection_drift": True},
        {"max_projection_drift": float("nan")},
        {"max_projection_drift": float("inf")},
        {"max_projection_drift": -1e-9},
    ],
)
def test_hard_repair_rejects_coercive_or_nonfinite_numeric_controls(
    kwargs: dict[str, object],
) -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def impossible(_: np.ndarray) -> HardOracleEvaluation:
        return HardOracleEvaluation(
            satisfied=np.array([False]), margins=np.array([-1.0])
        )

    with pytest.raises(Uint8LatticeError):
        repair_with_hard_oracle(frame, operator, impossible, **kwargs)  # type: ignore[arg-type]


def test_hard_repair_accepts_numpy_numeric_control_scalars() -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def impossible(_: np.ndarray) -> HardOracleEvaluation:
        return HardOracleEvaluation(
            satisfied=np.array([False]), margins=np.array([-1.0])
        )

    result = repair_with_hard_oracle(
        frame,
        operator,
        impossible,
        max_iterations=np.int64(1),
        max_proposals_per_iteration=np.int32(2),
        max_projection_drift=np.float64(0.0),
    )
    assert result.status is RepairStatus.STALLED_UNKNOWN


def test_hard_repair_result_owns_immutable_frame_copy() -> None:
    operator = _operator()
    original = np.zeros((8, 10, 1), dtype=np.uint8)

    def already_feasible(_: np.ndarray) -> HardOracleEvaluation:
        return HardOracleEvaluation(
            satisfied=np.array([True]), margins=np.array([1.0])
        )

    result = repair_with_hard_oracle(original, operator, already_feasible)
    original[0, 0, 0] = 255
    assert result.frame[0, 0, 0] == 0
    with pytest.raises(ValueError, match="read-only"):
        result.frame[0, 0, 0] = 1
    with pytest.raises(ValueError):
        result.frame.setflags(write=True)


def test_integer_move_refuses_bool_and_fractional_coordinates_or_delta() -> None:
    with pytest.raises(Uint8LatticeError):
        IntegerMove(True, 0, 0, 1)
    with pytest.raises(Uint8LatticeError):
        IntegerMove(0, 0.5, 0, 1)
    with pytest.raises(Uint8LatticeError):
        IntegerMove(0, 0, 0, 1.0)


@pytest.mark.parametrize(
    ("satisfied", "margin"),
    [(True, -1e-9), (False, 1e-9)],
)
def test_hard_oracle_rejects_inconsistent_satisfaction_and_margin(
    satisfied: bool, margin: float
) -> None:
    with pytest.raises(Uint8LatticeError, match="signed margins disagree"):
        HardOracleEvaluation(
            satisfied=np.array([satisfied]),
            margins=np.array([margin]),
        )


def test_hard_oracle_rejects_finite_margins_whose_aggregate_debt_overflows() -> None:
    maximum = np.finfo(np.float64).max
    with pytest.raises(Uint8LatticeError, match="aggregate margin debt"):
        HardOracleEvaluation(
            satisfied=np.array([False, False]),
            margins=np.array([-maximum, -maximum]),
        )


def test_hard_oracle_refuses_coercive_array_and_proposal_types() -> None:
    with pytest.raises(Uint8LatticeError, match="actual booleans"):
        HardOracleEvaluation(
            satisfied=np.array([1], dtype=np.int64),
            margins=np.array([1.0]),
        )
    with pytest.raises(Uint8LatticeError, match="real numeric"):
        HardOracleEvaluation(
            satisfied=np.array([False]),
            margins=np.array(["-1.0"]),
        )
    with pytest.raises(Uint8LatticeError, match="tuple of IntegerMove"):
        HardOracleEvaluation(
            satisfied=np.array([False]),
            margins=np.array([-1.0]),
            proposals=[IntegerMove(0, 0, 0, 1)],  # type: ignore[arg-type]
        )


def test_hard_oracle_allows_zero_margin_tie_with_either_hard_tie_outcome() -> None:
    for satisfied in (False, True):
        evaluation = HardOracleEvaluation(
            satisfied=np.array([satisfied]),
            margins=np.array([0.0]),
        )
        assert bool(evaluation.satisfied[0]) is satisfied


def test_hard_oracle_authority_arrays_are_owned_and_irreversibly_read_only() -> None:
    satisfied = np.array([False, True])
    margins = np.array([-1.0, 1.0])
    evaluation = HardOracleEvaluation(satisfied=satisfied, margins=margins)
    satisfied[:] = True
    margins[:] = 99.0
    assert np.array_equal(evaluation.satisfied, np.array([False, True]))
    assert np.array_equal(evaluation.margins, np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        evaluation.satisfied.setflags(write=True)
    with pytest.raises(ValueError):
        evaluation.margins.setflags(write=True)


def test_hard_oracle_impossible_canary_stalls_without_soft_fake() -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def impossible(_: np.ndarray) -> HardOracleEvaluation:
        return HardOracleEvaluation(satisfied=np.array([False]), margins=np.array([-1.0]), proposals=())

    result = repair_with_hard_oracle(frame, operator, impossible)
    assert result.status is RepairStatus.STALLED_UNKNOWN
    assert not result.evaluation.satisfied[0]
    assert result.changed_lattice_coordinates == 0


def test_hard_oracle_two_state_cycle_terminates() -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def cyclic(candidate: np.ndarray) -> HardOracleEvaluation:
        value = int(candidate[0, 0, 0])
        # The first +1 is admitted by a smaller debt.  Its sole proposal points
        # back to the already-seen starting frame, which must terminate.
        return HardOracleEvaluation(
            satisfied=np.array([False]),
            margins=np.array([-2.0 if value == 0 else -1.0]),
            proposals=(IntegerMove(0, 0, 0, 1 if value == 0 else -1),),
        )

    result = repair_with_hard_oracle(frame, operator, cyclic, max_iterations=4)
    assert result.status is RepairStatus.CYCLE_DETECTED_UNKNOWN
    assert result.frame[0, 0, 0] == 1


def test_hard_oracle_skips_seen_proposal_before_trying_unseen_improvement() -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def oracle(candidate: np.ndarray) -> HardOracleEvaluation:
        first = int(candidate[0, 0, 0])
        second = int(candidate[0, 1, 0])
        debt = 2 - first - second
        # Sorted order visits the already-seen -1 state before the unseen
        # improving coordinate.  Cycle detection must not return early.
        proposals = (
            (IntegerMove(0, 0, 0, 1),)
            if first == 0
            else (
                IntegerMove(0, 0, 0, -1),
                IntegerMove(0, 1, 0, 1),
            )
        )
        return HardOracleEvaluation(
            satisfied=np.array([debt <= 0]),
            margins=np.array([-float(debt)]),
            proposals=proposals,
        )

    result = repair_with_hard_oracle(
        frame,
        operator,
        oracle,
        max_iterations=3,
        max_proposals_per_iteration=1,
    )
    assert result.status is RepairStatus.FEASIBLE
    assert result.frame[0, 0, 0] == 1
    assert result.frame[0, 1, 0] == 1


def test_hard_repair_refuses_changed_obligation_shape_before_feasible_admission() -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def shrinking_oracle(candidate: np.ndarray) -> HardOracleEvaluation:
        if int(candidate[0, 0, 0]) == 0:
            return HardOracleEvaluation(
                satisfied=np.array([False, False]),
                margins=np.array([-2.0, -1.0]),
                proposals=(IntegerMove(0, 0, 0, 1),),
            )
        return HardOracleEvaluation(
            satisfied=np.array([True]),
            margins=np.array([1.0]),
        )

    with pytest.raises(Uint8LatticeError, match="obligation shape changed"):
        repair_with_hard_oracle(
            frame, operator, shrinking_oracle, max_iterations=2
        )


def test_hard_repair_refuses_non_evaluation_oracle_return() -> None:
    operator = _operator()
    frame = np.zeros((8, 10, 1), dtype=np.uint8)

    def malformed(_: np.ndarray) -> object:
        return {"satisfied": True}

    with pytest.raises(Uint8LatticeError, match="must return"):
        repair_with_hard_oracle(
            frame,
            operator,
            malformed,  # type: ignore[arg-type]
        )


def test_payload_parse_back_is_byte_exact_and_rejects_trailing_data() -> None:
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    payload = serialize_uint8_frame(frame)
    decoded = parse_uint8_frame(payload)
    assert np.array_equal(decoded, frame)
    assert decoded.dtype == np.uint8
    with pytest.raises(Uint8LatticeError, match="trailing"):
        parse_uint8_frame(payload + b"x")


def test_serializer_refuses_frame_larger_than_default_parser_contract() -> None:
    oversized = np.zeros((MAX_UINT8_FRAME_BYTES + 1, 1, 1), dtype=np.uint8)
    with pytest.raises(Uint8LatticeError, match="default parser byte contract"):
        serialize_uint8_frame(oversized)


def test_payload_parser_rejects_oversized_header_and_zlib_bomb() -> None:
    frame = np.zeros((1, 1, 1), dtype=np.uint8)
    payload = bytearray(serialize_uint8_frame(frame))
    struct.pack_into(">III", payload, 5, 1000, 1000, 1000)
    with pytest.raises(Uint8LatticeError, match="byte cap"):
        parse_uint8_frame(bytes(payload), max_frame_bytes=1024)

    header_size = struct.calcsize(">5sIII32s")
    bomb = serialize_uint8_frame(frame)[:header_size] + zlib.compress(b"x" * 10_000)
    with pytest.raises(Uint8LatticeError, match="exceeds declared"):
        parse_uint8_frame(bomb, max_frame_bytes=1024)


@pytest.mark.parametrize("cap", [True, 1024.5, float("nan"), float("inf")])
def test_payload_parser_refuses_coercive_or_nonfinite_byte_caps(cap: object) -> None:
    payload = serialize_uint8_frame(np.zeros((1, 1, 1), dtype=np.uint8))
    with pytest.raises(Uint8LatticeError, match="max_frame_bytes"):
        parse_uint8_frame(payload, max_frame_bytes=cap)  # type: ignore[arg-type]


def test_resume_plan_discards_stored_scientific_fields() -> None:
    tool = _measurement_tool()
    state = {
        "schema": tool.STATE_SCHEMA,
        "config_sha256": "cfg",
        "pair_rows": [
            {
                "pair_id": 90,
                "arms": {"clip_round_minimum_norm": {"mismatched_pixels": 0}},
                "lattice_diagnostics": {"nodes_visited": -1},
            }
        ],
    }
    assert tool._resume_revalidation_count(
        state, config_sha256="cfg", pair_ids=[90, 175]
    ) == 1
    state["pair_rows"][0]["arms"]["clip_round_minimum_norm"][
        "mismatched_pixels"
    ] = 999_999
    assert tool._resume_revalidation_count(
        state, config_sha256="cfg", pair_ids=[90, 175]
    ) == 1


def test_scorer_module_path_custody_refuses_preloaded_wrong_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _measurement_tool()
    expected_root = tmp_path / "expected"
    wrong_root = tmp_path / "wrong"
    expected_root.mkdir()
    wrong_root.mkdir()
    expected = expected_root / "modules.py"
    wrong = wrong_root / "modules.py"
    expected.write_text("# expected\n")
    wrong.write_text("# wrong\n")
    fake = ModuleType("modules")
    fake.__file__ = str(wrong)
    monkeypatch.setitem(sys.modules, "modules", fake)
    with pytest.raises(tool.MeasurementError, match="wrong source"):
        tool._require_module_path("modules", expected, allow_absent=False)

    monkeypatch.delitem(sys.modules, "modules")
    monkeypatch.setattr(sys, "path", [str(wrong_root), str(expected_root), "sentinel"])
    tool._prepend_exact_import_root(expected_root)
    assert Path(sys.path[0]).resolve() == expected_root.resolve()
    assert sum(
        Path(entry).resolve() == expected_root.resolve()
        for entry in sys.path
        if entry != "sentinel"
    ) == 1


def test_storage_preflight_checks_stage_filesystem_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _measurement_tool()
    sidecar_root = tmp_path / "sidecar"
    stage_root = tmp_path / "stages"
    sidecar_root.mkdir()
    stage_root.mkdir()
    stage_target = stage_root / "candidate.stages"
    stage_target.mkdir()

    class Usage:
        total = 1 << 42
        used = 0

        def __init__(self, free: int) -> None:
            self.free = free

    def fake_disk_usage(path: object) -> Usage:
        resolved = Path(path).resolve()
        if resolved in {sidecar_root.resolve(), stage_root.resolve()}:
            return Usage(1 << 40)
        return Usage(1)

    monkeypatch.setattr(tool.shutil, "disk_usage", fake_disk_usage)
    with pytest.raises(tool.MeasurementError, match="storage preflight refused"):
        tool._storage_preflights(
            sidecar_root / "candidate.u8lfs",
            stage_target,
            1,
        )


def test_aggregate_custody_binds_payload_hash_not_only_decoded_frame() -> None:
    tool = _measurement_tool()
    frame = np.zeros((2, 2, 1), dtype=np.uint8)
    frame_hash = tool._sha256_array(frame)
    rows = [
        {
            "pair_id": 90,
            "candidate_stage": {
                "decoded_frame_sha256": frame_hash,
                "payload_sha256": "stage-payload-hash",
            },
        }
    ]
    with pytest.raises(tool.MeasurementError, match="frame/payload/config"):
        tool._validate_aggregate_custody(
            {"config_sha256": "cfg"},
            [(90, frame, "different-payload-hash")],
            rows,
            config_sha256="cfg",
        )
    tool._validate_aggregate_custody(
        {"config_sha256": "cfg"},
        [(90, frame, "stage-payload-hash")],
        rows,
        config_sha256="cfg",
    )


@pytest.mark.skipif(
    _FROZEN_UPSTREAM is None,
    reason="frozen upstream modules.py and SegNet weights are unavailable",
)
def test_decoded_uint8_real_resize_and_frozen_cpu_segnet_hard_cell_positive_control() -> None:
    """Exercise the full hard positive-control chain without a mock scorer."""

    import torch
    from safetensors.torch import load_file

    assert _FROZEN_UPSTREAM is not None
    modules_path = _FROZEN_UPSTREAM / "modules.py"
    upstream_string = str(_FROZEN_UPSTREAM)
    if upstream_string not in sys.path:
        sys.path.insert(0, upstream_string)
    spec = importlib.util.spec_from_file_location(
        "_uint8_lattice_frozen_upstream_modules", modules_path
    )
    assert spec is not None and spec.loader is not None
    frozen_modules = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = frozen_modules
    spec.loader.exec_module(frozen_modules)

    torch.manual_seed(20260718)
    torch.use_deterministic_algorithms(True)
    model = frozen_modules.SegNet().eval().to("cpu")
    model.load_state_dict(
        load_file(
            str(_FROZEN_UPSTREAM / "models/segnet.safetensors"), device="cpu"
        )
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    operator = DisjointResizeOperator.build(
        camera_h=874, camera_w=1164, scorer_h=384, scorer_w=512
    )
    rows = np.arange(874, dtype=np.uint32)[:, None]
    cols = np.arange(1164, dtype=np.uint32)[None, :]
    frame = np.empty((874, 1164, 3), dtype=np.uint8)
    frame[..., 0] = ((3 * rows + 5 * cols + 17) % 256).astype(np.uint8)
    frame[..., 1] = ((7 * rows + 11 * cols + 29) % 256).astype(np.uint8)
    frame[..., 2] = ((13 * rows + 19 * cols + 43) % 256).astype(np.uint8)
    decoded = parse_uint8_frame(serialize_uint8_frame(frame))
    assert np.array_equal(decoded, frame)

    pair_uint8 = torch.from_numpy(np.stack((decoded, decoded), axis=0))[None].permute(
        0, 1, 4, 2, 3
    )
    with torch.inference_mode():
        # Float64 isolates the exact half-pixel A parity from float32 kernel
        # rounding; the actual frozen scorer forward below remains canonical
        # float32 CPU Torch.
        scorer_input_float64 = model.preprocess_input(pair_uint8.double())
        scorer_input = model.preprocess_input(pair_uint8.float())
        target_logits = model(scorer_input)
        target_cells = target_logits.argmax(dim=1)
    real_a = operator.apply(decoded)
    np.testing.assert_allclose(
        scorer_input_float64[0].permute(1, 2, 0).cpu().numpy(),
        real_a,
        atol=3e-11,
        rtol=0.0,
    )

    def frozen_hard_oracle(candidate: np.ndarray) -> HardOracleEvaluation:
        candidate_pair = (
            torch.from_numpy(np.stack((candidate, candidate), axis=0))[None]
            .permute(0, 1, 4, 2, 3)
            .float()
        )
        with torch.inference_mode():
            logits = model(model.preprocess_input(candidate_pair))
            predicted = logits.argmax(dim=1)
            target_values = logits.gather(1, target_cells[:, None]).squeeze(1)
            rivals = logits.clone()
            rivals.scatter_(1, target_cells[:, None], -torch.inf)
            margins = target_values - rivals.amax(dim=1)
        return HardOracleEvaluation(
            satisfied=(predicted == target_cells).cpu().numpy(),
            margins=margins.cpu().numpy(),
        )

    result = repair_with_hard_oracle(
        decoded, operator, frozen_hard_oracle, max_iterations=1
    )
    assert result.status is RepairStatus.FEASIBLE
    assert np.all(result.evaluation.satisfied)
    assert np.min(result.evaluation.margins) >= 0.0
    assert np.array_equal(result.frame, decoded)


def test_torch_resize_parity_when_available() -> None:
    torch = pytest.importorskip("torch")
    operator = _operator()
    rng = np.random.default_rng(520)
    frame = rng.integers(0, 256, size=(8, 10, 3), dtype=np.uint8)
    expected = operator.apply(frame)
    tensor = torch.from_numpy(frame).permute(2, 0, 1)[None].to(torch.float64)
    actual = (
        torch.nn.functional.interpolate(tensor, size=(3, 4), mode="bilinear", align_corners=False)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12)
