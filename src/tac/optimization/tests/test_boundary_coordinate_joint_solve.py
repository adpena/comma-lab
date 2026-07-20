from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from tac.boundary_math.compact_shearlet_frame import CompactShearletConfig
from tac.boundary_math.shared_receiver_admission import RATE_PRICE_PER_BYTE
from tac.boundary_math.windowed_curvelet_frame import WindowedCurveletConfig
from tac.optimization.boundary_coordinate_joint_solve import (
    BoundaryCoordinatePacket,
    BoundaryJointSolveError,
    ERMStatus,
    FrameFamily,
    JointSolveStatus,
    MeasuredCoordinate,
    QPSolveStatus,
    apply_boundary_packet,
    decode_boundary_packet,
    encode_boundary_packet,
    rgb_direction_matrix,
    run_exact_erm_fallback,
    select_measured_boundary_coordinates,
    selected_frame_features,
    solve_corrected_active_set_qp,
    solve_joint_boundary_candidate,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    HardOracleEvaluation,
    RepairStatus,
)


def _packet(family: FrameFamily = FrameFamily.WINDOWED_CURVELET) -> BoundaryCoordinatePacket:
    if family is FrameFamily.WINDOWED_CURVELET:
        config = asdict(
            WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)
        )
    else:
        config = asdict(
            CompactShearletConfig(
                n_scales=1, n_shear=1, two_cones=True, n_trans=1
            )
        )
    coefficients = np.zeros((2, 2, 3), dtype=np.int8)
    coefficients[1, 0, 0] = 3
    return BoundaryCoordinatePacket(
        family=family,
        frame_config=config,
        scorer_height=3,
        scorer_width=4,
        atom_indices=np.asarray([0, 1], dtype=np.uint32),
        coefficients=coefficients,
        scales=np.asarray([0.5, 1.0], dtype=np.float16),
    )


@pytest.mark.parametrize(
    "family", [FrameFamily.WINDOWED_CURVELET, FrameFamily.COMPACT_SHEARLET]
)
def test_packet_roundtrip_regenerates_genuine_localized_frame(family: FrameFamily) -> None:
    packet = _packet(family)
    payload = encode_boundary_packet(packet)
    decoded = decode_boundary_packet(payload)
    assert decoded.family is family
    assert decoded.frame_config == packet.frame_config
    assert np.array_equal(decoded.atom_indices, packet.atom_indices)
    assert np.array_equal(decoded.coefficients, packet.coefficients)
    assert np.array_equal(decoded.scales, packet.scales)
    features = selected_frame_features(decoded)
    assert features.shape == (12, 2)
    assert np.ptp(features[:, 0]) > 0.01


def test_packet_refuses_corruption_trailing_bytes_and_fourier_alias() -> None:
    payload = encode_boundary_packet(_packet())
    corrupted = bytearray(payload)
    corrupted[-5] ^= 1
    with pytest.raises(BoundaryJointSolveError, match="CRC"):
        decode_boundary_packet(bytes(corrupted))
    with pytest.raises(BoundaryJointSolveError, match="trailing"):
        decode_boundary_packet(payload + b"x")
    with pytest.raises(ValueError):
        FrameFamily("fourier")


def test_packet_application_is_pair_causal_and_uint8() -> None:
    packet = _packet()
    baseline = np.full((3, 4, 3), 100, dtype=np.uint8)
    first = apply_boundary_packet(baseline, packet, 0)
    second = apply_boundary_packet(baseline, packet, 1)
    assert np.array_equal(first, baseline)
    assert second.dtype == np.uint8
    assert np.any(second != baseline)


def test_packet_application_saturates_without_integer_wraparound() -> None:
    packet = _packet()
    packet = BoundaryCoordinatePacket(
        family=packet.family,
        frame_config=packet.frame_config,
        scorer_height=packet.scorer_height,
        scorer_width=packet.scorer_width,
        atom_indices=packet.atom_indices,
        coefficients=np.full((2, 2, 3), 127, dtype=np.int8),
        scales=np.full(2, 60_000.0, dtype=np.float16),
    )
    output = apply_boundary_packet(
        np.full((3, 4, 3), 128, dtype=np.uint8), packet, 0
    )
    assert output.dtype == np.uint8
    assert np.all(output == 255)
    negative_packet = BoundaryCoordinatePacket(
        family=packet.family,
        frame_config=packet.frame_config,
        scorer_height=packet.scorer_height,
        scorer_width=packet.scorer_width,
        atom_indices=packet.atom_indices,
        coefficients=np.full((2, 2, 3), -127, dtype=np.int8),
        scales=packet.scales,
    )
    negative = apply_boundary_packet(
        np.full((3, 4, 3), 128, dtype=np.uint8), negative_packet, 0
    )
    assert np.all(negative == 0)


def test_rgb_direction_matrix_preserves_interleaved_channels() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    directions = rgb_direction_matrix(features)
    coefficients = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert np.array_equal(
        directions @ coefficients,
        np.asarray([90.0, 120.0, 150.0, 190.0, 260.0, 330.0]),
    )


def test_corrected_active_set_qp_satisfies_kkt() -> None:
    result = solve_corrected_active_set_qp(
        np.eye(2),
        np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 1.0]),
    )
    assert result.status is QPSolveStatus.SOLVED
    assert np.allclose(result.coefficients, [1.0, 2.0])
    assert result.max_primal_violation == pytest.approx(0.0)
    assert result.min_active_multiplier == pytest.approx(1.0)
    assert result.stationarity_residual <= 1e-12


def test_corrected_active_set_qp_scopes_contradiction_as_cycle_unknown() -> None:
    result = solve_corrected_active_set_qp(
        np.asarray([[1.0], [-1.0]]),
        np.zeros((2, 1)),
        np.ones(2),
        np.ones(1),
        max_iterations=20,
    )
    assert result.status is QPSolveStatus.CYCLE_DETECTED_UNKNOWN
    assert result.max_primal_violation > 0.0


def _operator_and_oracle():
    operator = DisjointResizeOperator.build(
        camera_h=2, camera_w=2, scorer_h=1, scorer_w=1
    )
    calls: list[int] = []

    def oracle(camera: np.ndarray) -> HardOracleEvaluation:
        red = round(float(operator.apply(camera)[0, 0, 0]))
        calls.append(red)
        margin = float(red - 101)
        return HardOracleEvaluation(
            np.asarray([margin >= 0.0], dtype=bool),
            np.asarray([margin], dtype=np.float64),
        )

    return operator, oracle, calls


def test_joint_solver_places_exact_uint8_and_hard_oracle_inside_admission() -> None:
    operator, oracle, calls = _operator_and_oracle()
    baseline = np.asarray([[[100, 100, 100]]], dtype=np.uint8)
    directions = np.asarray([[1.0], [0.0], [0.0]])
    result = solve_joint_boundary_candidate(
        baseline_scorer_plane=baseline,
        direction_matrix=directions,
        operator=operator,
        first_order_jacobian=np.asarray([[0.75]]),
        secant_jacobian=np.asarray([[0.25]]),
        debt=np.asarray([1.0]),
        fisher_diagonal=np.asarray([1.0]),
        hard_oracle=oracle,
    )
    assert result.status is JointSolveStatus.FEASIBLE_HARD_ACCEPT
    assert result.candidate is not None
    assert result.candidate.exact_verification.certified_exact
    assert calls == [100, 101]


def test_joint_solver_does_not_promote_hard_rejection() -> None:
    operator, oracle, _calls = _operator_and_oracle()
    baseline = np.asarray([[[100, 100, 100]]], dtype=np.uint8)
    result = solve_joint_boundary_candidate(
        baseline_scorer_plane=baseline,
        direction_matrix=np.asarray([[0.1], [0.0], [0.0]]),
        operator=operator,
        first_order_jacobian=np.asarray([[1.0]]),
        secant_jacobian=np.asarray([[0.0]]),
        debt=np.asarray([1.0]),
        fisher_diagonal=np.asarray([1.0]),
        hard_oracle=oracle,
    )
    assert result.status is JointSolveStatus.HARD_REJECTED_UNKNOWN
    assert result.candidate is not None
    assert result.candidate.hard_evaluation.key[0] == 1


def test_erm_is_exactly_four_by_sixteen_and_hard_terminal_only() -> None:
    operator, oracle, calls = _operator_and_oracle()
    result = run_exact_erm_fallback(
        unknown_status=RepairStatus.STALLED_UNKNOWN,
        seed_coefficients=np.asarray([0.0]),
        baseline_scorer_plane=np.asarray([[[100, 100, 100]]], dtype=np.uint8),
        direction_matrix=np.asarray([[1.0], [0.0], [0.0]]),
        operator=operator,
        hard_oracle=oracle,
        cheap_energy=lambda coefficient: float((coefficient[0] - 2.0) ** 2),
        seed=7,
    )
    assert result.status is ERMStatus.HARD_ACCEPT
    assert result.cheap_evaluations == 4 * 16
    assert result.hard_terminal_evaluations == 4
    assert len(calls) == 5  # one fresh baseline plus four terminal candidates
    assert result.candidate is not None
    assert result.candidate.hard_evaluation.key[0] == 0


def test_erm_degenerate_spread_means_no_adoption() -> None:
    operator, oracle, calls = _operator_and_oracle()
    result = run_exact_erm_fallback(
        unknown_status=JointSolveStatus.CYCLE_DETECTED_UNKNOWN,
        seed_coefficients=np.asarray([0.0]),
        baseline_scorer_plane=np.asarray([[[100, 100, 100]]], dtype=np.uint8),
        direction_matrix=np.asarray([[1.0], [0.0], [0.0]]),
        operator=operator,
        hard_oracle=oracle,
        cheap_energy=lambda _coefficient: 1.0,
        seed=9,
    )
    assert result.status is ERMStatus.DEGENERATE_ENERGY_SPREAD
    assert result.cheap_evaluations == 64
    assert result.hard_terminal_evaluations == 0
    assert result.candidate is None
    assert len(calls) == 1


def test_erm_refuses_proven_feasible_route() -> None:
    operator, oracle, _calls = _operator_and_oracle()
    with pytest.raises(BoundaryJointSolveError, match="unknown"):
        run_exact_erm_fallback(
            unknown_status=RepairStatus.FEASIBLE,
            seed_coefficients=np.asarray([0.0]),
            baseline_scorer_plane=np.asarray([[[100, 100, 100]]], dtype=np.uint8),
            direction_matrix=np.asarray([[1.0], [0.0], [0.0]]),
            operator=operator,
            hard_oracle=oracle,
            cheap_energy=lambda coefficient: float(coefficient[0] ** 2),
            seed=0,
        )


def test_waterfill_targets_radius_one_then_widens_only_above_rate_price() -> None:
    rows = [
        MeasuredCoordinate("wide_high", 2, RATE_PRICE_PER_BYTE * 4.0, 1),
        MeasuredCoordinate("r1_keep", 1, RATE_PRICE_PER_BYTE * 2.0, 1),
        MeasuredCoordinate("r1_equal_stop", 1, RATE_PRICE_PER_BYTE, 1),
        MeasuredCoordinate("wide_low", 4, RATE_PRICE_PER_BYTE * 0.5, 1),
    ]
    result = select_measured_boundary_coordinates(rows, byte_budget=4)
    assert result.selected_ids == ("r1_keep", "wide_high")
    assert result.first_rejected_id == "r1_equal_stop"
    assert result.spent_bytes == 2


def test_waterfill_charges_all_bytes_and_stops_at_budget() -> None:
    rows = [
        MeasuredCoordinate("a", 1, 0.1, 3),
        MeasuredCoordinate("b", 1, 0.1, 3),
    ]
    result = select_measured_boundary_coordinates(rows, byte_budget=3)
    assert result.selected_ids == ("a",)
    assert result.spent_bytes == 3
    assert result.first_rejected_id == "b"
