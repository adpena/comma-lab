# SPDX-License-Identifier: MIT
"""Local Bregman/Hessian metric quantities used by the V9 policy.

The ordinary primal Hessian metric and its Fisher-natural cotangent form are
the same quantity when ``delta_eta = H @ delta_theta``.  Raw Euclidean length
in the dual coordinates is a different geometry: it is the squared-Hessian
quadratic form in primal coordinates.  Keeping four named helpers prevents a
no-solve shortcut from silently changing the canonical metric.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector: TypeAlias = NDArray[np.float64]
FloatMatrix: TypeAlias = NDArray[np.float64]

_SYMMETRY_ATOL = 1.0e-12
DELTA_ETA_CONSISTENCY_RTOL = 1.0e-12
DELTA_ETA_CONSISTENCY_ATOL = 1.0e-12


class GeometryValidationError(ValueError):
    """Raised when a local metric input is not finite, symmetric, and SPD."""


Generator = Callable[[FloatVector], float]
Gradient = Callable[[FloatVector], ArrayLike]


def _as_float64(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(f"{name} must be numeric: {exc}") from exc
    if not np.all(np.isfinite(array)):
        raise GeometryValidationError(f"{name} must contain only finite values")
    return array


def _validate_spd_hessian(hessian: ArrayLike) -> FloatMatrix:
    matrix = _as_float64(hessian, name="hessian")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise GeometryValidationError(
            f"hessian must be a non-empty square matrix, got shape={matrix.shape}"
        )
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_SYMMETRY_ATOL):
        raise GeometryValidationError("hessian must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise GeometryValidationError("hessian must be positive definite") from exc
    return matrix


def _validate_vector(value: ArrayLike, *, name: str, dimension: int) -> FloatVector:
    vector = _as_float64(value, name=name)
    if vector.ndim != 1 or vector.shape != (dimension,):
        raise GeometryValidationError(
            f"{name} must have shape ({dimension},), got shape={vector.shape}"
        )
    return vector


def _quadratic(vector: FloatVector, operator_vector: FloatVector) -> float:
    value = float(vector @ operator_vector)
    if not np.isfinite(value):
        raise GeometryValidationError("quadratic form produced a non-finite value")
    return value


def _as_nonempty_vector(value: ArrayLike, *, name: str) -> FloatVector:
    vector = _as_float64(value, name=name)
    if vector.ndim != 1 or vector.size == 0:
        raise GeometryValidationError(
            f"{name} must be a non-empty vector, got shape={vector.shape}"
        )
    return np.asarray(vector, dtype=np.float64)


def _normalized_positive_weights(weights: ArrayLike, *, count: int) -> FloatVector:
    vector = _as_float64(weights, name="weights")
    if vector.shape != (count,):
        raise GeometryValidationError(
            f"weights must have shape ({count},), got shape={vector.shape}"
        )
    if np.any(vector <= 0.0):
        raise GeometryValidationError(
            "weights must be strictly positive for the unique-centroid invariant"
        )
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise GeometryValidationError("weights must have a finite positive sum")
    return np.asarray(vector / total, dtype=np.float64)


def stable_logsumexp(logits: ArrayLike) -> float:
    """Return ``log(sum(exp(logits)))`` with the additive-logit gauge preserved."""

    vector = _as_nonempty_vector(logits, name="logits")
    maximum = float(np.max(vector))
    return float(maximum + np.log(np.exp(vector - maximum).sum()))


def categorical_softmax(logits: ArrayLike) -> FloatVector:
    """Return the categorical expectation coordinate ``eta = grad logsumexp``."""

    vector = _as_nonempty_vector(logits, name="logits")
    shifted = vector - float(np.max(vector))
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(), dtype=np.float64)


def categorical_kl_from_logits(
    numerator_logits: ArrayLike,
    denominator_logits: ArrayLike,
) -> float:
    """Return categorical KL from logits without taking ``log(softmax)``.

    Computing the log probabilities as ``z-logsumexp(z)`` preserves finite
    values even when an extreme but finite logit makes ``exp(z)`` underflow to
    zero in fp64.  This is the numerically stable realization of the dual
    negative-entropy identity used by the closed-form Bregman receipt.
    """

    numerator = _as_nonempty_vector(numerator_logits, name="numerator_logits")
    denominator = _as_nonempty_vector(
        denominator_logits, name="denominator_logits"
    )
    if numerator.shape != denominator.shape:
        raise GeometryValidationError(
            "numerator_logits and denominator_logits must share shape"
        )
    log_numerator = numerator - stable_logsumexp(numerator)
    log_denominator = denominator - stable_logsumexp(denominator)
    probability = categorical_softmax(numerator)
    value = float(probability @ (log_numerator - log_denominator))
    return require_nonnegative_bregman(value)


def categorical_log_partition_hessian(logits: ArrayLike) -> FloatMatrix:
    """Return ``H_F = diag(p) - p p.T = Cov_p[e_Y]`` for ``F=logsumexp``.

    The matrix is positive semidefinite on ambient logits and positive definite
    only on the quotient by the additive-constant gauge.  It is therefore kept
    separate from :func:`_validate_spd_hessian`, whose inverse-metric helpers
    intentionally require a gauge-fixed SPD chart.
    """

    probability = categorical_softmax(logits)
    return np.diag(probability) - np.outer(probability, probability)


def bregman_divergence(
    generator: Generator,
    gradient: Gradient,
    point: ArrayLike,
    reference: ArrayLike,
) -> float:
    """Evaluate ``B_F(point || reference)`` from its defining Taylor remainder."""

    p = _as_nonempty_vector(point, name="point")
    q = _as_nonempty_vector(reference, name="reference")
    if p.shape != q.shape:
        raise GeometryValidationError(
            f"point and reference must share shape, got {p.shape} and {q.shape}"
        )
    try:
        f_p = float(generator(p))
        f_q = float(generator(q))
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise GeometryValidationError(f"generator evaluation failed: {exc}") from exc
    grad_q = _as_float64(gradient(q), name="gradient(reference)")
    if grad_q.shape != q.shape:
        raise GeometryValidationError(
            "gradient(reference) must have the same shape as the points"
        )
    value = f_p - f_q - float(grad_q @ (p - q))
    if not np.isfinite(value):
        raise GeometryValidationError("Bregman divergence produced a non-finite value")
    return float(value)


def require_nonnegative_bregman(value: float, *, atol: float = 1.0e-12) -> float:
    """Fail closed when a claimed convex-generator divergence is materially negative.

    For differentiable ``F``, ``B_F(p||q) >= 0`` for every pair is equivalent
    to the first-order convexity inequality.  A tiny negative floating-point
    residue within ``atol`` is returned as zero; larger negatives are a real
    convexity/domain/implementation defect.
    """

    scalar = float(value)
    tolerance = float(atol)
    if not np.isfinite(scalar):
        raise GeometryValidationError("Bregman divergence must be finite")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise GeometryValidationError("atol must be finite and non-negative")
    if scalar < -tolerance:
        raise GeometryValidationError(
            "Bregman divergence is negative beyond tolerance; the generator is "
            "non-convex on this pair, the arguments left its domain, or the "
            f"implementation is wrong (value={scalar:.17g}, atol={tolerance:.3g})"
        )
    return 0.0 if scalar < 0.0 else scalar


def logsumexp_bregman(point_logits: ArrayLike, reference_logits: ArrayLike) -> float:
    """Return the exact finite Bregman divergence of categorical log-partition.

    ``B_F(z_p||z_q) = KL(softmax(z_q) || softmax(z_p))``.  This global
    potential-difference form is solve-free.  It is locally
    ``0.5 * dz.T @ H_F(z_q) @ dz + O(||dz||^3)``; it is not the finite
    Fisher--Rao geodesic distance.
    """

    value = bregman_divergence(
        stable_logsumexp,
        categorical_softmax,
        point_logits,
        reference_logits,
    )
    return require_nonnegative_bregman(value)


def categorical_negative_entropy_bregman(
    point_probability: ArrayLike,
    reference_probability: ArrayLike,
) -> float:
    """Return ``B_{F*}(p||q)=KL(p||q)`` for ``F*(p)=sum p log p``."""

    p = _as_nonempty_vector(point_probability, name="point_probability")
    q = _as_nonempty_vector(reference_probability, name="reference_probability")
    if p.shape != q.shape:
        raise GeometryValidationError("categorical probabilities must share shape")
    for name, vector in (("point_probability", p), ("reference_probability", q)):
        if np.any(vector <= 0.0):
            raise GeometryValidationError(f"{name} must be strictly positive")
        if not np.isclose(float(vector.sum()), 1.0, rtol=0.0, atol=1.0e-12):
            raise GeometryValidationError(f"{name} must sum to one")
    value = float(np.sum(p * (np.log(p) - np.log(q))))
    return require_nonnegative_bregman(value)


def logsumexp_bregman_closed_form_summary(
    point_logits: ArrayLike,
    reference_logits: ArrayLike,
) -> dict[str, float]:
    """Verify dual-generator reversal and gradient-pairing cancellation.

    The exact identities are

    ``B_F(p||q) = B_F*(grad F(q)||grad F(p))`` and
    ``B_F(p||q)+B_F(q||p) = <grad F(p)-grad F(q), p-q>``.
    Neither identity requires a Hessian solve.  Raw Euclidean dual length is
    deliberately absent; that separate quantity is owned by the squared-
    Hessian guard below.
    """

    p = _as_nonempty_vector(point_logits, name="point_logits")
    q = _as_nonempty_vector(reference_logits, name="reference_logits")
    if p.shape != q.shape:
        raise GeometryValidationError("point_logits and reference_logits must share shape")
    eta_p = categorical_softmax(p)
    eta_q = categorical_softmax(q)
    primal = logsumexp_bregman(p, q)
    reverse = logsumexp_bregman(q, p)
    # Preserve the direct negative-entropy evaluation on the representable
    # simplex interior.  Fall back to originating logits only when an extreme
    # finite logit underflows to the numerical boundary.
    if np.all(eta_p > 0.0) and np.all(eta_q > 0.0):
        dual = categorical_negative_entropy_bregman(eta_q, eta_p)
    else:
        dual = categorical_kl_from_logits(q, p)
    pairing = float((eta_p - eta_q) @ (p - q))
    return {
        "primal_bregman": primal,
        "reversed_dual_bregman": dual,
        "dual_identity_abs_error": abs(primal - dual),
        "symmetrized_bregman": primal + reverse,
        "gradient_pairing": pairing,
        "cancellation_abs_error": abs((primal + reverse) - pairing),
    }


def affine_legendre_logsumexp_summary(
    point: ArrayLike,
    reference: ArrayLike,
    matrix: ArrayLike,
    offset: ArrayLike,
    *,
    scale: float,
    linear_term: ArrayLike,
    constant: float = 0.0,
) -> dict[str, Any]:
    """Evaluate Nielsen's affine-Legendre Bregman covariance law.

    For ``Fbar(theta)=scale*F(A theta+b)+<c,theta>+d`` with ``scale>0``::

        B_Fbar(theta1||theta2) = scale B_F(A theta1+b || A theta2+b)
        grad Fbar = scale A.T grad F(A theta+b) + c
        Hess Fbar = scale A.T Hess F(A theta+b) A.

    Linear and constant gauge terms cancel from the divergence.  This helper
    proves the mathematical identity only; it does not assert that the live
    V9 latent model implements the transform or factors through ``(xi,R)``.
    """

    theta_p = _as_nonempty_vector(point, name="point")
    theta_q = _as_nonempty_vector(reference, name="reference")
    if theta_p.shape != theta_q.shape:
        raise GeometryValidationError("point and reference must share shape")
    transform = _as_float64(matrix, name="matrix")
    if transform.ndim != 2 or transform.shape[1] != theta_p.size:
        raise GeometryValidationError(
            "matrix must be two-dimensional with one column per parameter"
        )
    b = _validate_vector(offset, name="offset", dimension=transform.shape[0])
    c = _validate_vector(linear_term, name="linear_term", dimension=theta_p.size)
    lam = float(scale)
    d = float(constant)
    if not np.isfinite(lam) or lam <= 0.0:
        raise GeometryValidationError("scale must be finite and positive")
    if not np.isfinite(d):
        raise GeometryValidationError("constant must be finite")

    def transformed_generator(theta: FloatVector) -> float:
        return lam * stable_logsumexp(transform @ theta + b) + float(c @ theta) + d

    def transformed_gradient(theta: FloatVector) -> FloatVector:
        return np.asarray(
            lam * transform.T @ categorical_softmax(transform @ theta + b) + c,
            dtype=np.float64,
        )

    transformed_p = transform @ theta_p + b
    transformed_q = transform @ theta_q + b
    gauged = bregman_divergence(
        transformed_generator,
        transformed_gradient,
        theta_p,
        theta_q,
    )
    base_scaled = lam * logsumexp_bregman(transformed_p, transformed_q)
    base_hessian = categorical_log_partition_hessian(transformed_q)
    pulled_back_hessian = lam * transform.T @ base_hessian @ transform
    return {
        "gauged_bregman": require_nonnegative_bregman(gauged),
        "scaled_base_bregman": base_scaled,
        "covariance_abs_error": abs(gauged - base_scaled),
        "transformed_gradient_reference": transformed_gradient(theta_q),
        "pulled_back_hessian_reference": np.asarray(
            pulled_back_hessian, dtype=np.float64
        ),
        "status": "GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED",
    }


def categorical_right_data_centroid(
    sample_logits: ArrayLike,
    weights: ArrayLike,
) -> dict[str, Any]:
    """Return the Bregman centroid when the samples occupy the right argument.

    With the orientation used in this repository,

    ``argmin_c sum_i w_i B_F(c || theta_i)``
    ``= (grad F)^-1(sum_i w_i grad F(theta_i))``.

    For ``F=logsumexp``, ``grad F=softmax`` is not injective on ambient logits.
    The minimizer is unique only on the quotient by additive constants; this
    helper fixes the gauge by returning zero-mean ``log(mean probability)``.
    """

    samples = _as_float64(sample_logits, name="sample_logits")
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 2:
        raise GeometryValidationError(
            "sample_logits must have shape (N,K) with N>=1 and K>=2"
        )
    normalized = _normalized_positive_weights(weights, count=samples.shape[0])
    probabilities = np.stack([categorical_softmax(row) for row in samples], axis=0)
    dual_mean = np.asarray(normalized @ probabilities, dtype=np.float64)
    centroid = np.log(dual_mean)
    centroid -= float(centroid.mean())
    recovered = categorical_softmax(centroid)
    objective = float(
        sum(
            weight * logsumexp_bregman(centroid, sample)
            for weight, sample in zip(normalized, samples, strict=True)
        )
    )
    return {
        "centroid_logits_zero_mean": centroid,
        "dual_probability_mean": dual_mean,
        "recovered_probability": recovered,
        "first_order_residual_linf": float(np.max(np.abs(recovered - dual_mean))),
        "weighted_objective": objective,
        "uniqueness_scope": "UNIQUE_ON_ADDITIVE_LOGIT_GAUGE_QUOTIENT",
    }


def categorical_left_data_centroid(
    sample_logits: ArrayLike,
    weights: ArrayLike,
) -> FloatVector:
    """Return ``argmin_c sum_i w_i B_F(theta_i || c)`` in a fixed logit gauge.

    This opposite orientation is the weighted arithmetic mean of primal logits,
    modulo the additive-logit gauge.  Exposing it prevents the two asymmetric
    Bregman centroid conventions from being silently interchanged.
    """

    samples = _as_float64(sample_logits, name="sample_logits")
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 2:
        raise GeometryValidationError(
            "sample_logits must have shape (N,K) with N>=1 and K>=2"
        )
    normalized = _normalized_positive_weights(weights, count=samples.shape[0])
    centroid = np.asarray(normalized @ samples, dtype=np.float64)
    centroid -= float(centroid.mean())
    return centroid


def positive_unscented_sigma_points(
    mean: ArrayLike,
    covariance: ArrayLike,
    *,
    kappa: float = 1.0,
) -> tuple[FloatMatrix, FloatVector]:
    """Return a positive-weight ``2D+1`` unscented rule matching mean/covariance.

    ``w0=kappa/(D+kappa)`` and each paired point has
    ``wi=1/(2(D+kappa))`` at displacement ``sqrt(D+kappa)*chol(cov)[:,i]``.
    Requiring ``kappa>0`` keeps every weight positive, which is necessary for
    the unique Bregman-centroid invariant.  The rule is exact for input mean
    and covariance, not for an arbitrary nonlinear output map.
    """

    mu = _as_nonempty_vector(mean, name="mean")
    cov = _validate_spd_hessian(covariance)
    if cov.shape != (mu.size, mu.size):
        raise GeometryValidationError(
            f"covariance must have shape ({mu.size},{mu.size})"
        )
    kap = float(kappa)
    if not np.isfinite(kap) or kap <= 0.0:
        raise GeometryValidationError(
            "kappa must be finite and > 0 so all sigma weights are positive"
        )
    dimension = mu.size
    spread = np.sqrt(float(dimension) + kap) * np.linalg.cholesky(cov)
    points = [mu]
    for column in range(dimension):
        delta = spread[:, column]
        points.extend((mu + delta, mu - delta))
    w0 = kap / (float(dimension) + kap)
    wi = 1.0 / (2.0 * (float(dimension) + kap))
    weights = np.asarray([w0] + [wi] * (2 * dimension), dtype=np.float64)
    return np.stack(points, axis=0), weights


def exponential_family_sigma_kl_error(
    theta_p: ArrayLike,
    theta_q: ArrayLike,
    eta_p: ArrayLike,
    eta_hat: ArrayLike,
) -> float:
    """Return the exact EF sigma-quadrature KL error.

    For log-density ratio ``log p_theta_p / p_theta_q`` and a weighted support
    whose sufficient-statistic mean is ``eta_hat``, the quadrature error is
    ``(theta_p-theta_q).T @ (eta_hat-eta_p)``.  It vanishes exactly only when
    the expectation coordinate is matched; an ordinary unscented input rule
    does not imply this after a nonlinear sufficient-statistic map.
    """

    p = _as_nonempty_vector(theta_p, name="theta_p")
    q = _as_nonempty_vector(theta_q, name="theta_q")
    expected = _as_nonempty_vector(eta_p, name="eta_p")
    observed = _as_nonempty_vector(eta_hat, name="eta_hat")
    if not (p.shape == q.shape == expected.shape == observed.shape):
        raise GeometryValidationError("all EF vectors must share shape")
    return float((p - q) @ (observed - expected))


def categorical_bregman_sigma_propagation(
    mean: ArrayLike,
    covariance: ArrayLike,
    transform: Callable[[FloatVector], ArrayLike],
    *,
    kappa: float = 1.0,
) -> dict[str, Any]:
    """Propagate a positive unscented rule and summarize it in Bregman geometry.

    The transformed sigma logits are collapsed with the right-data Bregman
    centroid.  ``exact_bregman_dispersion`` is the exact weighted objective at
    those transformed points.  ``local_hessian_dispersion`` is only the local
    second-order approximation around the centroid and is labeled separately.
    """

    points, weights = positive_unscented_sigma_points(
        mean, covariance, kappa=kappa
    )
    mapped_rows: list[FloatVector] = []
    for index, point in enumerate(points):
        try:
            mapped = _as_nonempty_vector(transform(point), name=f"transform(point[{index}])")
        except (TypeError, ValueError, FloatingPointError) as exc:
            raise GeometryValidationError(
                f"sigma-point transform failed at index {index}: {exc}"
            ) from exc
        if mapped.size < 2:
            raise GeometryValidationError("transformed categorical logits require K>=2")
        if mapped_rows and mapped.shape != mapped_rows[0].shape:
            raise GeometryValidationError("all transformed sigma points must share shape")
        mapped_rows.append(mapped)
    mapped_logits = np.stack(mapped_rows, axis=0)
    centroid = categorical_right_data_centroid(mapped_logits, weights)
    center = np.asarray(centroid["centroid_logits_zero_mean"], dtype=np.float64)
    hessian = categorical_log_partition_hessian(center)
    exact = float(centroid["weighted_objective"])
    local = 0.0
    for weight, mapped in zip(weights, mapped_logits, strict=True):
        displacement = mapped - center
        local += 0.5 * float(weight) * float(displacement @ hessian @ displacement)

    mu = _as_nonempty_vector(mean, name="mean")
    cov = _validate_spd_hessian(covariance)
    reconstructed_mean = np.asarray(weights @ points, dtype=np.float64)
    centered = points - reconstructed_mean
    reconstructed_cov = np.einsum("n,ni,nj->ij", weights, centered, centered)
    return {
        "sigma_points": points,
        "weights": weights,
        "mapped_logits": mapped_logits,
        "centroid_logits_zero_mean": center,
        "dual_probability_mean": centroid["dual_probability_mean"],
        "exact_bregman_dispersion": exact,
        "local_hessian_dispersion": float(local),
        "input_mean_abs_error": float(np.max(np.abs(reconstructed_mean - mu))),
        "input_covariance_abs_error": float(np.max(np.abs(reconstructed_cov - cov))),
        "quadrature_scope": (
            "INPUT_MEAN_COVARIANCE_EXACT; NONLINEAR_OUTPUT_APPROXIMATE; "
            "EF_KL_EXACT_ONLY_IF_SUFFICIENT_STAT_EXPECTATION_MATCHES"
        ),
    }


def deterministic_bregman_application_fixture() -> dict[str, float]:
    """Return a small deterministic fp64 receipt for the registered identities."""

    point = np.asarray([0.8, -0.3, 0.2], dtype=np.float64)
    reference = np.asarray([-0.1, 0.6, -0.4], dtype=np.float64)
    closed = logsumexp_bregman_closed_form_summary(point, reference)
    gauge = affine_legendre_logsumexp_summary(
        point,
        reference,
        np.asarray(
            [[1.0, 0.2, -0.1], [0.0, 1.1, 0.3], [0.2, -0.2, 0.9]],
            dtype=np.float64,
        ),
        np.asarray([0.1, -0.2, 0.05], dtype=np.float64),
        scale=1.7,
        linear_term=np.asarray([0.3, -0.4, 0.2], dtype=np.float64),
        constant=2.5,
    )
    samples = np.asarray(
        [[0.4, -0.2, 0.1], [-0.3, 0.7, -0.1], [0.2, 0.0, -0.5]],
        dtype=np.float64,
    )
    weights = np.asarray([0.2, 0.5, 0.3], dtype=np.float64)
    centroid = categorical_right_data_centroid(samples, weights)
    sigma = categorical_bregman_sigma_propagation(
        np.asarray([0.2, -0.1], dtype=np.float64),
        np.asarray([[0.3, 0.04], [0.04, 0.2]], dtype=np.float64),
        lambda x: np.asarray([x[0], x[1], 0.25 * x[0] * x[1]], dtype=np.float64),
        kappa=1.0,
    )
    return {
        "closed_form_dual_error": float(closed["dual_identity_abs_error"]),
        "closed_form_cancellation_error": float(closed["cancellation_abs_error"]),
        "affine_gauge_covariance_error": float(gauge["covariance_abs_error"]),
        "centroid_first_order_residual": float(centroid["first_order_residual_linf"]),
        "sigma_input_mean_error": float(sigma["input_mean_abs_error"]),
        "sigma_input_covariance_error": float(sigma["input_covariance_abs_error"]),
        "sigma_exact_bregman_dispersion": float(sigma["exact_bregman_dispersion"]),
        "sigma_local_hessian_dispersion": float(sigma["local_hessian_dispersion"]),
        "ef_exact_condition_error": exponential_family_sigma_kl_error(
            np.asarray([0.3, -0.2]),
            np.asarray([-0.1, 0.4]),
            np.asarray([0.6, 0.4]),
            np.asarray([0.6, 0.4]),
        ),
    }


def _validate_dual_coordinate_consistency(
    hessian: FloatMatrix,
    delta_theta: FloatVector,
    delta_eta: FloatVector,
) -> None:
    """Require ``delta_eta = H @ delta_theta`` to fp64 numerical tolerance.

    The identity-bearing aggregate uses a symmetric ``1e-12`` relative and
    absolute tolerance.  Inputs outside that tolerance are different local
    coordinate displacements, so reporting the primal/dual identities for
    them would be a false claim and must fail closed.
    """

    expected_delta_eta = hessian @ delta_theta
    if not np.allclose(
        delta_eta,
        expected_delta_eta,
        rtol=DELTA_ETA_CONSISTENCY_RTOL,
        atol=DELTA_ETA_CONSISTENCY_ATOL,
    ):
        max_abs_residual = float(np.max(np.abs(delta_eta - expected_delta_eta)))
        raise GeometryValidationError(
            "delta_eta must equal hessian @ delta_theta within fp64 tolerance "
            f"(rtol={DELTA_ETA_CONSISTENCY_RTOL:.1e}, "
            f"atol={DELTA_ETA_CONSISTENCY_ATOL:.1e}); "
            f"max_abs_residual={max_abs_residual:.17g}"
        )


def primal_hessian_quadratic(hessian: ArrayLike, delta_theta: ArrayLike) -> float:
    """Return ``delta_theta.T @ H @ delta_theta`` for a validated SPD ``H``."""

    matrix = _validate_spd_hessian(hessian)
    vector = _validate_vector(delta_theta, name="delta_theta", dimension=matrix.shape[0])
    return _quadratic(vector, matrix @ vector)


def solve_fisher_natural_cotangent(
    hessian: ArrayLike, delta_eta: ArrayLike
) -> FloatVector:
    """Solve ``H x = delta_eta`` on the Fisher-natural cotangent path.

    This named, typed solve is the only inverse-Hessian operation in this
    module.  It intentionally calls :func:`numpy.linalg.solve`; an explicit
    inverse and a raw-dual no-solve alias are forbidden.
    """

    matrix = _validate_spd_hessian(hessian)
    cotangent = _validate_vector(delta_eta, name="delta_eta", dimension=matrix.shape[0])
    try:
        solution = np.linalg.solve(matrix, cotangent)
    except np.linalg.LinAlgError as exc:  # Defensive: SPD validation should catch this first.
        raise GeometryValidationError("Fisher-natural cotangent solve failed") from exc
    if not np.all(np.isfinite(solution)):
        raise GeometryValidationError("Fisher-natural cotangent solve returned non-finite values")
    return np.asarray(solution, dtype=np.float64)


def fisher_natural_cotangent_quadratic(
    hessian: ArrayLike, delta_eta: ArrayLike
) -> float:
    """Return ``delta_eta.T @ solve(H, delta_eta)`` via the typed solve."""

    matrix = _validate_spd_hessian(hessian)
    cotangent = _validate_vector(delta_eta, name="delta_eta", dimension=matrix.shape[0])
    solved = solve_fisher_natural_cotangent(matrix, cotangent)
    return _quadratic(cotangent, solved)


def raw_dual_euclidean_quadratic(delta_eta: ArrayLike) -> float:
    """Return raw dual Euclidean length ``delta_eta.T @ delta_eta``."""

    cotangent = _as_float64(delta_eta, name="delta_eta")
    if cotangent.ndim != 1 or cotangent.size == 0:
        raise GeometryValidationError(
            f"delta_eta must be a non-empty vector, got shape={cotangent.shape}"
        )
    return _quadratic(cotangent, cotangent)


def squared_hessian_quadratic(hessian: ArrayLike, delta_theta: ArrayLike) -> float:
    """Return ``delta_theta.T @ H @ H @ delta_theta`` for validated inputs."""

    matrix = _validate_spd_hessian(hessian)
    vector = _validate_vector(delta_theta, name="delta_theta", dimension=matrix.shape[0])
    return _quadratic(vector, matrix @ (matrix @ vector))


def local_hessian_dual_geometry_summary(
    hessian: ArrayLike,
    delta_theta: ArrayLike,
    delta_eta: ArrayLike,
) -> dict[str, float]:
    """Compute four quantities for one consistent primal/dual displacement.

    This aggregate bears the two coordinate identities, so it rejects inputs
    unless ``delta_eta = H @ delta_theta`` within the documented fp64
    tolerance.  The individual quadratic helpers remain available when a
    caller intentionally needs to evaluate unrelated vectors.
    """

    matrix = _validate_spd_hessian(hessian)
    primal = _validate_vector(delta_theta, name="delta_theta", dimension=matrix.shape[0])
    dual = _validate_vector(delta_eta, name="delta_eta", dimension=matrix.shape[0])
    _validate_dual_coordinate_consistency(matrix, primal, dual)
    return {
        "primal_hessian": primal_hessian_quadratic(matrix, primal),
        "fisher_natural_cotangent": fisher_natural_cotangent_quadratic(matrix, dual),
        "raw_dual_euclidean": raw_dual_euclidean_quadratic(dual),
        "squared_hessian": squared_hessian_quadratic(matrix, primal),
    }


__all__ = [
    "DELTA_ETA_CONSISTENCY_ATOL",
    "DELTA_ETA_CONSISTENCY_RTOL",
    "GeometryValidationError",
    "affine_legendre_logsumexp_summary",
    "bregman_divergence",
    "categorical_bregman_sigma_propagation",
    "categorical_kl_from_logits",
    "categorical_left_data_centroid",
    "categorical_log_partition_hessian",
    "categorical_negative_entropy_bregman",
    "categorical_right_data_centroid",
    "categorical_softmax",
    "deterministic_bregman_application_fixture",
    "exponential_family_sigma_kl_error",
    "fisher_natural_cotangent_quadratic",
    "local_hessian_dual_geometry_summary",
    "logsumexp_bregman",
    "logsumexp_bregman_closed_form_summary",
    "positive_unscented_sigma_points",
    "primal_hessian_quadratic",
    "raw_dual_euclidean_quadratic",
    "require_nonnegative_bregman",
    "solve_fisher_natural_cotangent",
    "squared_hessian_quadratic",
    "stable_logsumexp",
]
