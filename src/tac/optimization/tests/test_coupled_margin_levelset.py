# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import numpy as np

from tac.optimization.coupled_margin_levelset import (
    CouplingOperator,
    ReductionStatus,
    babai_nearest_plane,
    certify_quadratic_margin_infeasible_on_trust_ball,
    predicted_margin,
    solve_active_set_kkt,
    validate_coupling_operator_fd,
)


def _operator(matrix: np.ndarray, margin: np.ndarray, required: np.ndarray) -> CouplingOperator:
    m = np.asarray(matrix, dtype=np.float64)
    return CouplingOperator(
        matrix=m,
        margin=np.asarray(margin, dtype=np.float64),
        required_margin=np.asarray(required, dtype=np.float64),
        targeted_count=1,
        row_labels=tuple(f"row_{index}" for index in range(m.shape[0])),
        dof_labels=tuple(f"dof_{index}" for index in range(m.shape[1])),
        activation_pattern_sha256=hashlib.sha256(b"fixture-pattern").hexdigest(),
    )


def test_active_pattern_qp_is_solved_as_closed_kkt_equations() -> None:
    operator = _operator(
        np.asarray([[1.0, 1.0], [1.0, -1.0]]),
        np.asarray([-2.0, 0.0]),
        np.asarray([0.0, 0.0]),
    )
    solved = solve_active_set_kkt(
        operator,
        description_metric=np.diag([1.0, 4.0]),
        damping=1e-8,
        trust_radius=8.0,
    )
    assert solved.diagnostics.converged
    assert solved.diagnostics.status == "KKT_SOLVED"
    assert np.min(predicted_margin(operator, solved.step) - operator.required_margin) >= -1e-7
    assert solved.diagnostics.stationarity_linf < 1e-6
    assert solved.diagnostics.complementarity_linf < 1e-6
    assert operator.reduction_status is ReductionStatus.EXACT_ACTIVE_PATTERN_QP


def test_kkt_fails_closed_when_trust_box_cannot_cross_level_set() -> None:
    operator = _operator(np.asarray([[1.0]]), np.asarray([-3.0]), np.asarray([0.0]))
    solved = solve_active_set_kkt(operator, trust_radius=1.0, max_iterations=24)
    assert solved.diagnostics.converged is False
    assert solved.diagnostics.primal_min_slack < 0.0


def test_coupling_operator_vjp_entries_validate_by_central_fd() -> None:
    matrix = np.asarray([[2.0, -1.0], [0.5, 3.0]])
    margin = np.asarray([-1.0, 2.0])
    operator = _operator(matrix, margin, np.zeros(2))

    def measured(step: np.ndarray) -> np.ndarray:
        return margin + matrix @ step + 1e-6 * np.square(step).sum()

    receipt = validate_coupling_operator_fd(
        operator,
        measured,
        epsilon=1e-3,
        maximum_entries=4,
        absolute_tolerance=1e-8,
        relative_tolerance=1e-8,
    )
    assert receipt.sampled_entries == 4
    assert receipt.passed


def test_babai_projection_carries_quadratic_error_bound() -> None:
    hessian = np.asarray([[4.0, 1.0], [1.0, 2.0]])
    projected = babai_nearest_plane(np.asarray([1.45, -0.55]), hessian)
    assert projected.integer_step.dtype == np.int64
    assert projected.inside_bound
    assert projected.quadratic_error <= projected.covering_bound + 1e-9


def test_order1_sos_bound_can_certify_no_crossing_inside_trust_ball() -> None:
    certificate = certify_quadratic_margin_infeasible_on_trust_ball(
        constant=-2.0,
        gradient=np.asarray([0.1, 0.1]),
        quadratic=-np.eye(2),
        radius=1.0,
        required_margin=0.0,
    )
    assert certificate.certified_infeasible
    assert certificate.upper_bound < certificate.required_margin
    assert certificate.reduction_status is ReductionStatus.RELAXATION_BOUNDED
