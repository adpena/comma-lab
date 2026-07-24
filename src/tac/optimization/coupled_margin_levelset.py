# SPDX-License-Identifier: MIT
"""Coupled argmax level-set equations for compact receiver programs.

The scorer and receiver stay outside this module.  A caller supplies the
measured margin vector and its receiver-through-R Jacobian ``M``.  Inside one
frozen activation pattern the constraints are affine and the minimum
description-norm problem is a convex QP.  This module solves that QP as a
deterministic active-set KKT system, then exposes the integer-lattice Babai
projection and the trust-ball quadratic certificate used at pattern borders.

No function here claims a contest score or a nonlinear global optimum.  A
hard receiver/scorer callback must remeasure every proposed step.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class CoupledMarginSolveError(ValueError):
    """Malformed operator, constraint, or numerical solve."""


class ReductionStatus(StrEnum):
    """Honest mathematical authority of one local reduction."""

    EXACT_ACTIVE_PATTERN_QP = "EXACT_ACTIVE_PATTERN_QP"
    RELAXATION_BOUNDED = "RELAXATION_BOUNDED"
    SEARCHED_PATTERN_SWITCH = "SEARCHED_PATTERN_SWITCH"


@dataclass(frozen=True, slots=True)
class CouplingOperator:
    """One local receiver-realized margin Jacobian.

    Rows are signed margins that must be nonnegative after subtracting their
    required epsilon.  The first ``targeted_count`` rows are target crossings;
    the remainder preserve already-correct protected-cell signs.
    """

    matrix: np.ndarray
    margin: np.ndarray
    required_margin: np.ndarray
    targeted_count: int
    row_labels: tuple[str, ...]
    dof_labels: tuple[str, ...]
    activation_pattern_sha256: str
    reduction_status: ReductionStatus = ReductionStatus.EXACT_ACTIVE_PATTERN_QP

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        margin = np.asarray(self.margin, dtype=np.float64)
        required = np.asarray(self.required_margin, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise CoupledMarginSolveError("coupling matrix must be nonempty and two-dimensional")
        if margin.shape != (matrix.shape[0],) or required.shape != margin.shape:
            raise CoupledMarginSolveError("margin vectors must match coupling rows")
        if not np.isfinite(matrix).all() or not np.isfinite(margin).all() or not np.isfinite(required).all():
            raise CoupledMarginSolveError("coupling operator inputs must be finite")
        if np.any(required < 0.0):
            raise CoupledMarginSolveError("required margins must be nonnegative")
        if not 0 <= int(self.targeted_count) <= matrix.shape[0]:
            raise CoupledMarginSolveError("targeted_count is outside the row range")
        if len(self.row_labels) != matrix.shape[0] or len(self.dof_labels) != matrix.shape[1]:
            raise CoupledMarginSolveError("operator labels must be bijective with rows and columns")
        if len(set(self.dof_labels)) != len(self.dof_labels):
            raise CoupledMarginSolveError("DOF labels must be unique")
        if len(self.activation_pattern_sha256) != 64:
            raise CoupledMarginSolveError("activation pattern SHA-256 must be present")
        object.__setattr__(self, "matrix", _immutable(matrix))
        object.__setattr__(self, "margin", _immutable(margin))
        object.__setattr__(self, "required_margin", _immutable(required))

    @property
    def deficit(self) -> np.ndarray:
        """Right-hand side in ``M step >= required_margin - margin``."""

        return self.required_margin - self.margin


@dataclass(frozen=True, slots=True)
class KKTDiagnostics:
    active_rows: tuple[int, ...]
    iterations: int
    primal_min_slack: float
    stationarity_linf: float
    complementarity_linf: float
    multiplier_min: float
    objective: float
    converged: bool
    status: str


@dataclass(frozen=True, slots=True)
class KKTStep:
    step: np.ndarray
    multipliers: np.ndarray
    hessian: np.ndarray
    diagnostics: KKTDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _immutable(np.asarray(self.step, dtype=np.float64)))
        object.__setattr__(self, "multipliers", _immutable(np.asarray(self.multipliers, dtype=np.float64)))
        object.__setattr__(self, "hessian", _immutable(np.asarray(self.hessian, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class FiniteDifferenceValidation:
    sampled_entries: int
    maximum_absolute_error: float
    maximum_relative_error: float
    median_relative_error: float
    passed: bool
    tolerance_absolute: float
    tolerance_relative: float


@dataclass(frozen=True, slots=True)
class BabaiProjection:
    integer_step: np.ndarray
    quadratic_error: float
    covering_bound: float
    inside_bound: bool
    basis_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "integer_step", _immutable(np.asarray(self.integer_step, dtype=np.int64)))


@dataclass(frozen=True, slots=True)
class QuadraticTrustCertificate:
    upper_bound: float
    required_margin: float
    certified_infeasible: bool
    radius: float
    lambda_max_positive: float
    method: str = "ORDER1_SOS_S_LEMMA_TRUST_BALL_UPPER_BOUND"
    reduction_status: ReductionStatus = ReductionStatus.RELAXATION_BOUNDED


def gauss_newton_hessian(
    matrix: np.ndarray,
    *,
    description_metric: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
    damping: float = 1e-6,
) -> np.ndarray:
    """Return ``G + M.T W M + damping I`` for the local trust-region QP."""

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or not np.isfinite(m).all():
        raise CoupledMarginSolveError("matrix must be finite and two-dimensional")
    n = m.shape[1]
    if description_metric is None:
        g = np.eye(n, dtype=np.float64)
    else:
        g = np.asarray(description_metric, dtype=np.float64)
        if g.shape != (n, n) or not np.isfinite(g).all() or not np.allclose(g, g.T, atol=1e-10):
            raise CoupledMarginSolveError("description metric must be finite symmetric DxD")
    if row_weights is None:
        weighted = m
    else:
        w = np.asarray(row_weights, dtype=np.float64)
        if w.shape != (m.shape[0],) or not np.isfinite(w).all() or np.any(w < 0.0):
            raise CoupledMarginSolveError("row weights must be finite nonnegative per-row values")
        weighted = np.sqrt(w)[:, None] * m
    lam = float(damping)
    if not math.isfinite(lam) or lam < 0.0:
        raise CoupledMarginSolveError("damping must be finite and nonnegative")
    h = 0.5 * (g + g.T) + weighted.T @ weighted + lam * np.eye(n)
    eigen_min = float(np.linalg.eigvalsh(h).min())
    if eigen_min <= 0.0:
        raise CoupledMarginSolveError("damped Gauss-Newton Hessian is not positive definite")
    return h


def solve_active_set_kkt(
    operator: CouplingOperator,
    *,
    description_metric: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
    damping: float = 1e-6,
    trust_radius: float | np.ndarray = 8.0,
    use_gauss_newton: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int | None = None,
) -> KKTStep:
    r"""Solve the local convex QP by deterministic working-set KKT equations.

    ``min 0.5*x.T H x`` subject to ``M x >= deficit`` and the box trust
    region.  Once an active set is fixed, the solution is the closed system

    ``[H -A.T; A 0] [x,lambda] = [0,b]``.

    The outer working-set loop only identifies which measured half-spaces bind;
    it does not search RGB candidates.  Receiver remeasurement remains outside.
    """

    m = np.asarray(operator.matrix, dtype=np.float64)
    b = np.asarray(operator.deficit, dtype=np.float64)
    n = m.shape[1]
    tol = float(tolerance)
    if not math.isfinite(tol) or tol <= 0.0:
        raise CoupledMarginSolveError("tolerance must be finite and positive")
    base_metric = np.eye(n) if description_metric is None else np.asarray(description_metric, dtype=np.float64)
    h = (
        gauss_newton_hessian(
            m,
            description_metric=base_metric,
            row_weights=row_weights,
            damping=damping,
        )
        if use_gauss_newton
        else gauss_newton_hessian(
            np.zeros_like(m),
            description_metric=base_metric,
            damping=damping,
        )
    )
    radius = np.asarray(trust_radius, dtype=np.float64)
    if radius.ndim == 0:
        radius = np.full(n, float(radius), dtype=np.float64)
    if radius.shape != (n,) or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise CoupledMarginSolveError("trust radius must be positive scalar or per-DOF vector")

    # Express the box in the same >= convention as the level-set rows.
    box_a = np.concatenate((np.eye(n), -np.eye(n)), axis=0)
    box_b = np.concatenate((-radius, -radius), axis=0)
    a_all = np.concatenate((m, box_a), axis=0)
    b_all = np.concatenate((b, box_b), axis=0)
    active: list[int] = []
    x = np.zeros(n, dtype=np.float64)
    multipliers_active = np.empty(0, dtype=np.float64)
    cap = int(max_iterations or (4 * (a_all.shape[0] + n) + 16))
    status = "MAX_ITERATIONS"

    iteration = 0
    for _iteration in range(1, cap + 1):
        iteration = _iteration
        slack = a_all @ x - b_all
        inactive = [idx for idx in range(a_all.shape[0]) if idx not in active]
        if inactive:
            worst = min(inactive, key=lambda idx: (slack[idx], idx))
            if slack[worst] < -tol:
                active.append(worst)

        if active:
            aa = a_all[np.asarray(active, dtype=np.int64)]
            bb = b_all[np.asarray(active, dtype=np.int64)]
            kkt = np.block(
                [[h, -aa.T], [aa, np.zeros((len(active), len(active)), dtype=np.float64)]]
            )
            rhs = np.concatenate((np.zeros(n, dtype=np.float64), bb))
            solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=1e-12)
            x = solution[:n]
            multipliers_active = solution[n:]
            negative = [i for i, value in enumerate(multipliers_active) if value < -tol]
            if negative:
                drop_local = min(negative, key=lambda i: (multipliers_active[i], active[i]))
                active.pop(drop_local)
                continue
        else:
            x.fill(0.0)
            multipliers_active = np.empty(0, dtype=np.float64)

        slack = a_all @ x - b_all
        if float(slack.min()) >= -tol and (not active or float(multipliers_active.min()) >= -tol):
            status = "KKT_SOLVED"
            break
    else:
        iteration = cap

    # A last-iteration multiplier drop can leave the cached multiplier vector
    # one element longer than the retained set.  Re-solve that retained system
    # solely for consistent fail-closed diagnostics; it cannot turn an
    # infeasible working-set cycle into a converged result.
    if len(multipliers_active) != len(active):
        if active:
            aa = a_all[np.asarray(active, dtype=np.int64)]
            bb = b_all[np.asarray(active, dtype=np.int64)]
            kkt = np.block(
                [[h, -aa.T], [aa, np.zeros((len(active), len(active)), dtype=np.float64)]]
            )
            solution, *_ = np.linalg.lstsq(
                kkt,
                np.concatenate((np.zeros(n, dtype=np.float64), bb)),
                rcond=1e-12,
            )
            x = solution[:n]
            multipliers_active = solution[n:]
        else:
            x.fill(0.0)
            multipliers_active = np.empty(0, dtype=np.float64)

    full_multipliers = np.zeros(a_all.shape[0], dtype=np.float64)
    if active:
        full_multipliers[np.asarray(active, dtype=np.int64)] = multipliers_active
    stationarity = h @ x - a_all.T @ full_multipliers
    slack = a_all @ x - b_all
    complementarity = full_multipliers * slack
    converged = bool(
        status == "KKT_SOLVED"
        and float(slack.min()) >= -tol
        and float(full_multipliers.min()) >= -tol
        and float(np.max(np.abs(stationarity), initial=0.0)) <= max(1e-7, 10.0 * tol)
    )
    diagnostics = KKTDiagnostics(
        active_rows=tuple(active),
        iterations=iteration,
        primal_min_slack=float(slack.min()),
        stationarity_linf=float(np.max(np.abs(stationarity), initial=0.0)),
        complementarity_linf=float(np.max(np.abs(complementarity), initial=0.0)),
        multiplier_min=float(full_multipliers.min()),
        objective=float(0.5 * x @ h @ x),
        converged=converged,
        status=status if converged else f"{status}_RESIDUAL_NOT_CLEAN",
    )
    return KKTStep(x, full_multipliers[: m.shape[0]], h, diagnostics)


def validate_coupling_operator_fd(
    operator: CouplingOperator,
    margin_function: Callable[[np.ndarray], np.ndarray],
    *,
    epsilon: float = 1e-3,
    maximum_entries: int = 32,
    seed: int = 0,
    absolute_tolerance: float = 2e-3,
    relative_tolerance: float = 5e-2,
) -> FiniteDifferenceValidation:
    """Validate sampled ``M[row,col]`` entries by central finite differences."""

    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise CoupledMarginSolveError("finite-difference epsilon must be positive")
    total = operator.matrix.size
    take = min(int(maximum_entries), total)
    if take <= 0:
        raise CoupledMarginSolveError("maximum_entries must be positive")
    rng = np.random.default_rng(int(seed))
    flat = np.sort(rng.choice(total, size=take, replace=False))
    absolute: list[float] = []
    relative: list[float] = []
    for value in flat:
        row, col = np.unravel_index(int(value), operator.matrix.shape)
        direction = np.zeros(operator.matrix.shape[1], dtype=np.float64)
        direction[col] = eps
        plus = np.asarray(margin_function(direction), dtype=np.float64)
        minus = np.asarray(margin_function(-direction), dtype=np.float64)
        if plus.shape != operator.margin.shape or minus.shape != operator.margin.shape:
            raise CoupledMarginSolveError("finite-difference callback changed margin geometry")
        estimate = float((plus[row] - minus[row]) / (2.0 * eps))
        expected = float(operator.matrix[row, col])
        ae = abs(estimate - expected)
        re = ae / max(abs(estimate), abs(expected), 1e-12)
        absolute.append(ae)
        relative.append(re)
    max_abs = max(absolute)
    max_rel = max(relative)
    passed = all(a <= absolute_tolerance or r <= relative_tolerance for a, r in zip(absolute, relative, strict=True))
    return FiniteDifferenceValidation(
        sampled_entries=take,
        maximum_absolute_error=max_abs,
        maximum_relative_error=max_rel,
        median_relative_error=float(np.median(relative)),
        passed=passed,
        tolerance_absolute=float(absolute_tolerance),
        tolerance_relative=float(relative_tolerance),
    )


def babai_nearest_plane(step: np.ndarray, hessian: np.ndarray) -> BabaiProjection:
    """Project a continuous step to ``Z^D`` in the Hessian metric.

    QR on a square root of ``H`` yields the standard nearest-plane recursion.
    ``0.25 * sum(diag(R)^2)`` is the covering bound of the Babai cell in the
    transformed quadratic norm.
    """

    x = np.asarray(step, dtype=np.float64)
    h = np.asarray(hessian, dtype=np.float64)
    if x.ndim != 1 or h.shape != (x.size, x.size) or not np.isfinite(x).all() or not np.isfinite(h).all():
        raise CoupledMarginSolveError("Babai inputs must be finite D and DxD arrays")
    try:
        square_root = np.linalg.cholesky(0.5 * (h + h.T)).T
    except np.linalg.LinAlgError as exc:
        raise CoupledMarginSolveError("Babai Hessian must be positive definite") from exc
    q, r = np.linalg.qr(square_root)
    transformed = q.T @ (square_root @ x)
    integer = np.zeros(x.size, dtype=np.int64)
    for index in range(x.size - 1, -1, -1):
        residual = transformed[index] - float(r[index, index + 1 :] @ integer[index + 1 :])
        integer[index] = int(np.rint(residual / r[index, index]))
    error = integer.astype(np.float64) - x
    quadratic_error = float(error @ h @ error)
    bound = float(0.25 * np.square(np.diag(r)).sum())
    return BabaiProjection(integer, quadratic_error, bound, quadratic_error <= bound + 1e-9, x.size)


def certify_quadratic_margin_infeasible_on_trust_ball(
    *,
    constant: float,
    gradient: np.ndarray,
    quadratic: np.ndarray,
    radius: float,
    required_margin: float,
) -> QuadraticTrustCertificate:
    r"""Order-1 SOS/S-lemma upper bound for a quadratic margin on ``||x||<=r``.

    For ``m(x)=c+g.T x+x.T Q x``, Cauchy--Schwarz and the positive part of
    ``lambda_max(Q)`` give a certified upper bound.  If that upper bound is
    below the required crossing margin, the trust-region instance is proven
    infeasible.  Otherwise the result is only a bound, never a feasibility
    claim.
    """

    g = np.asarray(gradient, dtype=np.float64)
    q = np.asarray(quadratic, dtype=np.float64)
    r = float(radius)
    req = float(required_margin)
    c = float(constant)
    if g.ndim != 1 or q.shape != (g.size, g.size) or not np.isfinite(g).all() or not np.isfinite(q).all():
        raise CoupledMarginSolveError("quadratic certificate inputs must be finite")
    if not all(math.isfinite(v) for v in (r, req, c)) or r < 0.0:
        raise CoupledMarginSolveError("certificate radius/margins must be finite with radius>=0")
    lambda_positive = max(0.0, float(np.linalg.eigvalsh(0.5 * (q + q.T)).max()))
    upper = c + float(np.linalg.norm(g)) * r + lambda_positive * r * r
    return QuadraticTrustCertificate(
        upper_bound=upper,
        required_margin=req,
        certified_infeasible=upper < req,
        radius=r,
        lambda_max_positive=lambda_positive,
    )


def predicted_margin(operator: CouplingOperator, step: np.ndarray) -> np.ndarray:
    """Affine active-pattern margin prediction."""

    value = np.asarray(step, dtype=np.float64)
    if value.shape != (operator.matrix.shape[1],):
        raise CoupledMarginSolveError("step geometry differs from coupling columns")
    return operator.margin + operator.matrix @ value


def _immutable(value: np.ndarray) -> np.ndarray:
    out = np.array(value, copy=True)
    out.setflags(write=False)
    return out


__all__ = [
    "BabaiProjection",
    "CoupledMarginSolveError",
    "CouplingOperator",
    "FiniteDifferenceValidation",
    "KKTDiagnostics",
    "KKTStep",
    "QuadraticTrustCertificate",
    "ReductionStatus",
    "babai_nearest_plane",
    "certify_quadratic_margin_infeasible_on_trust_ball",
    "gauss_newton_hessian",
    "predicted_margin",
    "solve_active_set_kkt",
    "validate_coupling_operator_fd",
]
