# SPDX-License-Identifier: MIT
"""First-block trust regions for amortized frozen-SegNet costates.

The module makes two different statements explicit:

* an anchor margin plus a downstream pairwise-logit Lipschitz bound protects
  the anchor argmax cell; and
* exact-teacher descent additionally needs a suffix-costate Lipschitz bound,
  a renderer-VJP norm bound, and a nonzero projected-gradient floor.

A local Jacobian or a margin/Fisher correlation is not silently promoted to
an upper bound.  Empirical inputs produce only ``PROXY_REUSE`` decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from tac.boundary_math.segnet_validation_certificate import (
    BoundAuthority,
    derive_feature_trust_region,
)

ReuseStatus = Literal["CERTIFIED_REUSE", "PROXY_REUSE", "REFRESH", "BLOCKED"]
InputMetric = Literal["linf", "margin_fisher_rms"]
ProviderMode = Literal["current_prefix_vjp_banked_suffix"]
DirectProviderMode = Literal["direct_full_input_costate"]
DirectInputMetric = Literal["l2"]


def _finite_nonnegative(value: float, *, name: str, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _sha256(value: str | None, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or array.dtype.hasobject or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a non-empty finite numeric array")
    return array


def array_sha256(value: Any) -> str:
    """Content identity including dtype and shape."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def prefix_feature_envelope(
    displacement: float,
    *,
    anchor_jacobian_norm_upper: float,
    jacobian_lipschitz_upper: float,
) -> float:
    """Taylor remainder bound ``J0*r + beta*r^2/2``."""

    radius = _finite_nonnegative(displacement, name="displacement")
    jacobian = _finite_nonnegative(anchor_jacobian_norm_upper, name="anchor_jacobian_norm_upper")
    lipschitz = _finite_nonnegative(jacobian_lipschitz_upper, name="jacobian_lipschitz_upper")
    return jacobian * radius + math.ldexp(lipschitz * radius * radius, -1)


def input_radius_from_feature_radius(
    feature_radius: float,
    *,
    anchor_jacobian_norm_upper: float,
    jacobian_lipschitz_upper: float,
) -> float:
    """Invert the first-block Taylor envelope with a cancellation-safe root."""

    rho = _finite_nonnegative(feature_radius, name="feature_radius", positive=True)
    jacobian = _finite_nonnegative(anchor_jacobian_norm_upper, name="anchor_jacobian_norm_upper")
    lipschitz = _finite_nonnegative(jacobian_lipschitz_upper, name="jacobian_lipschitz_upper")
    if lipschitz == 0.0:
        if jacobian == 0.0:
            return math.inf
        return rho / jacobian
    return (2.0 * rho) / (jacobian + math.sqrt(jacobian * jacobian + 2.0 * lipschitz * rho))


def margin_fisher_proxy(margins: Any, *, mask: Any | None = None) -> float:
    """Top-two Bernoulli Fisher proxy ``exp(-|m|)/(1+exp(-|m|))^2``.

    This is O(pixels), dimensionless, and diagnostic.  Correlation with a true
    Fisher surface does not make it a neighborhood upper bound.
    """

    values = _finite_array(margins, name="margins")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != values.shape or not bool(selected.any()):
            raise ValueError("mask must match margins and select at least one element")
        values = values[selected]
    exponential = np.exp(-np.abs(values))
    fisher = exponential / np.square(1.0 + exponential)
    return float(np.mean(fisher, dtype=np.float64))


def costate_error_envelope(
    displacement: float,
    *,
    anchor_jacobian_norm_upper: float,
    jacobian_lipschitz_upper: float,
    suffix_costate_lipschitz_upper: float,
) -> float:
    """Bound ``||J_f(x)^T(p(h)-p(h0))||`` inside the input ball."""

    radius = _finite_nonnegative(displacement, name="displacement")
    jacobian = _finite_nonnegative(anchor_jacobian_norm_upper, name="anchor_jacobian_norm_upper")
    beta = _finite_nonnegative(jacobian_lipschitz_upper, name="jacobian_lipschitz_upper")
    kappa = _finite_nonnegative(suffix_costate_lipschitz_upper, name="suffix_costate_lipschitz_upper")
    feature_delta = prefix_feature_envelope(
        radius,
        anchor_jacobian_norm_upper=jacobian,
        jacobian_lipschitz_upper=beta,
    )
    return (jacobian + beta * radius) * kappa * feature_delta


def fit_empirical_jacobian_envelope(
    *, input_displacements: Any, feature_displacements: Any
) -> tuple[float, float]:
    """Fit the smallest two-term envelope over supplied prefix-only probes.

    The returned values are a local estimate, never an upper-bound artifact.
    ``J`` is the smallest observed secant slope and ``beta`` is the smallest
    nonnegative quadratic remainder coefficient covering every probe.
    """

    inputs = _finite_array(input_displacements, name="input_displacements").reshape(-1)
    features = _finite_array(feature_displacements, name="feature_displacements").reshape(-1)
    if inputs.shape != features.shape or bool((inputs <= 0.0).any()) or bool((features < 0.0).any()):
        raise ValueError("empirical envelope probes require matched positive inputs and nonnegative features")
    slopes = features / inputs
    jacobian = float(np.min(slopes))
    remainders = np.maximum(features - jacobian * inputs, 0.0)
    beta = float(np.max(2.0 * remainders / np.square(inputs)))
    return jacobian, beta


def _descent_radius(
    *,
    margin_radius: float,
    anchor_jacobian_norm_upper: float,
    jacobian_lipschitz_upper: float,
    suffix_costate_lipschitz_upper: float,
    renderer_vjp_norm_upper: float,
    projected_gradient_floor: float,
) -> float:
    """Largest representable radius whose projected-gradient error is strict."""

    if not math.isfinite(margin_radius) or margin_radius <= 0.0:
        raise ValueError("a finite positive margin radius is required for a descent certificate")
    renderer = _finite_nonnegative(renderer_vjp_norm_upper, name="renderer_vjp_norm_upper")
    floor = _finite_nonnegative(projected_gradient_floor, name="projected_gradient_floor", positive=True)

    def holds(radius: float) -> bool:
        error = costate_error_envelope(
            radius,
            anchor_jacobian_norm_upper=anchor_jacobian_norm_upper,
            jacobian_lipschitz_upper=jacobian_lipschitz_upper,
            suffix_costate_lipschitz_upper=suffix_costate_lipschitz_upper,
        )
        return renderer * error < floor

    if holds(margin_radius):
        return margin_radius
    lower, upper = 0.0, margin_radius
    while True:
        middle = lower + (upper - lower) / 2.0
        if middle in (lower, upper):
            break
        if holds(middle):
            lower = middle
        else:
            upper = middle
    if lower <= 0.0:
        raise ValueError("no positive representable descent radius exists")
    return lower


@dataclass(frozen=True)
class CostateTrustRegion:
    """Content-bound input-space region for one exact teacher anchor."""

    input_radius: float
    input_metric: InputMetric
    provider_mode: ProviderMode
    margin_radius: float
    descent_radius: float
    feature_radius: float
    authority: BoundAuthority
    anchor_frame_sha256: str
    anchor_margin_sha256: str
    bound_artifact_sha256: str | None
    calibration_receipt_sha256: str | None
    protected_pixels: int
    anchor_jacobian_norm_upper: float
    jacobian_lipschitz_upper: float
    suffix_costate_lipschitz_upper: float | None
    renderer_vjp_norm_upper: float | None
    projected_gradient_floor: float | None
    descent_bound_available: bool
    anchor_fisher_proxy: float
    boundary_fisher_proxy_upper: float
    anchor_margins: np.ndarray = field(repr=False, compare=False)
    protected_mask: np.ndarray = field(repr=False, compare=False)
    pairwise_suffix_bounds: np.ndarray = field(repr=False, compare=False)

    @property
    def certificate_authoritative(self) -> bool:
        return self.authority == "rigorous_upper_bound"


@dataclass(frozen=True)
class CostateReuseDecision:
    status: ReuseStatus
    input_displacement: float | None
    input_radius: float
    predicted_feature_displacement_upper: float | None
    predicted_minimum_margin: float | None
    predicted_fisher_proxy_upper: float | None
    label_cell_authoritative: bool
    descent_authoritative: bool
    reason: str

    @property
    def reuses_costate(self) -> bool:
        return self.status in {"CERTIFIED_REUSE", "PROXY_REUSE"}


def derive_costate_trust_region(
    *,
    anchor_margins: Any,
    anchor_correct_mask: Any,
    pairwise_suffix_lipschitz_upper: Any,
    anchor_jacobian_norm_upper: float,
    jacobian_lipschitz_upper: float,
    suffix_costate_lipschitz_upper: float | None = None,
    renderer_vjp_norm_upper: float | None = None,
    projected_gradient_floor: float | None = None,
    input_metric: InputMetric = "linf",
    provider_mode: ProviderMode,
    authority: BoundAuthority,
    anchor_frame_sha256: str,
    bound_artifact_sha256: str | None = None,
    calibration_receipt_sha256: str | None = None,
) -> CostateTrustRegion:
    """Derive the intersection of label-cell and exact-descent safe balls."""

    margins = _finite_array(anchor_margins, name="anchor_margins")
    bounds = _finite_array(pairwise_suffix_lipschitz_upper, name="pairwise_suffix_lipschitz_upper")
    protected = np.asarray(anchor_correct_mask, dtype=bool)
    if margins.shape != bounds.shape or margins.shape != protected.shape:
        raise ValueError("margins, suffix bounds, and protected mask must have identical shapes")
    if input_metric not in {"linf", "margin_fisher_rms"}:
        raise ValueError("input_metric must be linf or margin_fisher_rms")
    if provider_mode != "current_prefix_vjp_banked_suffix":
        raise ValueError("certificate requires the current prefix VJP around a banked suffix costate")
    _sha256(anchor_frame_sha256, name="anchor_frame_sha256")
    margin_sha = array_sha256(margins)
    feature_region = derive_feature_trust_region(
        anchor_margins=margins,
        anchor_correct_mask=protected,
        pairwise_logit_change_bounds=bounds,
        authority=authority,
        anchor_feature_sha256=margin_sha,
        bound_artifact_sha256=bound_artifact_sha256,
        calibration_receipt_sha256=calibration_receipt_sha256,
    )
    jacobian = _finite_nonnegative(anchor_jacobian_norm_upper, name="anchor_jacobian_norm_upper")
    beta = _finite_nonnegative(jacobian_lipschitz_upper, name="jacobian_lipschitz_upper")
    descent_values = (
        suffix_costate_lipschitz_upper,
        renderer_vjp_norm_upper,
        projected_gradient_floor,
    )
    descent_bound_available = all(value is not None for value in descent_values)
    if any(value is not None for value in descent_values) and not descent_bound_available:
        raise ValueError("descent certificate inputs must be supplied together or all omitted")
    if authority == "rigorous_upper_bound" and not descent_bound_available:
        raise ValueError("rigorous costate reuse requires the complete projected-descent bound")
    kappa = (
        _finite_nonnegative(suffix_costate_lipschitz_upper, name="suffix_costate_lipschitz_upper")
        if suffix_costate_lipschitz_upper is not None
        else None
    )
    renderer = (
        _finite_nonnegative(renderer_vjp_norm_upper, name="renderer_vjp_norm_upper")
        if renderer_vjp_norm_upper is not None
        else None
    )
    gradient_floor = (
        _finite_nonnegative(projected_gradient_floor, name="projected_gradient_floor", positive=True)
        if projected_gradient_floor is not None
        else None
    )
    margin_radius = input_radius_from_feature_radius(
        feature_region.feature_radius,
        anchor_jacobian_norm_upper=jacobian,
        jacobian_lipschitz_upper=beta,
    )
    descent_radius = (
        _descent_radius(
            margin_radius=margin_radius,
            anchor_jacobian_norm_upper=jacobian,
            jacobian_lipschitz_upper=beta,
            suffix_costate_lipschitz_upper=kappa,
            renderer_vjp_norm_upper=renderer,
            projected_gradient_floor=gradient_floor,
        )
        if kappa is not None and renderer is not None and gradient_floor is not None
        else math.inf
    )
    radius = min(margin_radius, descent_radius)
    feature_at_boundary = prefix_feature_envelope(
        radius,
        anchor_jacobian_norm_upper=jacobian,
        jacobian_lipschitz_upper=beta,
    )
    predicted_margins = margins - bounds * feature_at_boundary
    margins_copy = margins.copy()
    protected_copy = protected.copy()
    bounds_copy = bounds.copy()
    for value in (margins_copy, protected_copy, bounds_copy):
        value.setflags(write=False)
    return CostateTrustRegion(
        input_radius=radius,
        input_metric=input_metric,
        provider_mode=provider_mode,
        margin_radius=margin_radius,
        descent_radius=descent_radius,
        feature_radius=feature_region.feature_radius,
        authority=authority,
        anchor_frame_sha256=anchor_frame_sha256,
        anchor_margin_sha256=margin_sha,
        bound_artifact_sha256=bound_artifact_sha256,
        calibration_receipt_sha256=calibration_receipt_sha256,
        protected_pixels=feature_region.protected_pixels,
        anchor_jacobian_norm_upper=jacobian,
        jacobian_lipschitz_upper=beta,
        suffix_costate_lipschitz_upper=kappa,
        renderer_vjp_norm_upper=renderer,
        projected_gradient_floor=gradient_floor,
        descent_bound_available=descent_bound_available,
        anchor_fisher_proxy=margin_fisher_proxy(margins, mask=protected),
        boundary_fisher_proxy_upper=margin_fisher_proxy(predicted_margins, mask=protected),
        anchor_margins=margins_copy,
        protected_mask=protected_copy,
        pairwise_suffix_bounds=bounds_copy,
    )


def _pixel_displacement(anchor: np.ndarray, current: np.ndarray, margin_shape: tuple[int, ...]) -> np.ndarray:
    if anchor.shape != current.shape or anchor.size == 0:
        raise ValueError("anchor/current frames must have the same non-empty shape")
    difference = np.abs(current - anchor)
    if difference.shape == margin_shape:
        return difference
    candidates: list[np.ndarray] = []
    if difference.ndim >= 3 and difference.shape[-1] in {1, 3}:
        candidates.append(np.max(difference, axis=-1))
    if difference.ndim >= 3 and difference.shape[-3] in {1, 3}:
        candidates.append(np.max(difference, axis=-3))
    for candidate in candidates:
        if candidate.shape == margin_shape:
            return candidate
    raise ValueError("frame geometry cannot be aligned with the anchor margin field")


def check_costate_trust_region(
    *,
    anchor_frame: Any,
    current_frame: Any,
    region: CostateTrustRegion,
    current_anchor_frame_sha256: str,
    custody_matches: bool = True,
) -> CostateReuseDecision:
    """O(pixels) in-region check with no frozen-SegNet forward."""

    if not custody_matches or current_anchor_frame_sha256 != region.anchor_frame_sha256:
        return CostateReuseDecision(
            "BLOCKED", None, region.input_radius, None, None, None, False, False,
            "anchor/provider custody changed",
        )
    try:
        anchor = _finite_array(anchor_frame, name="anchor_frame")
        current = _finite_array(current_frame, name="current_frame")
        pixel_delta = _pixel_displacement(anchor, current, region.anchor_margins.shape)
    except ValueError as exc:
        return CostateReuseDecision(
            "BLOCKED", None, region.input_radius, None, None, None, False, False, str(exc)
        )
    if region.input_metric == "linf":
        displacement = float(np.max(pixel_delta))
    else:
        anchor_fisher = np.exp(-np.abs(region.anchor_margins))
        anchor_fisher /= np.square(1.0 + anchor_fisher)
        weights = anchor_fisher * region.protected_mask
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            return CostateReuseDecision(
                "BLOCKED", None, region.input_radius, None, None, None, False, False,
                "anchor margin/Fisher weights have zero or nonfinite mass",
            )
        displacement = math.sqrt(float(np.sum(weights * np.square(pixel_delta), dtype=np.float64)) / weight_sum)
    feature_upper = prefix_feature_envelope(
        displacement,
        anchor_jacobian_norm_upper=region.anchor_jacobian_norm_upper,
        jacobian_lipschitz_upper=region.jacobian_lipschitz_upper,
    )
    predicted_margins = region.anchor_margins - region.pairwise_suffix_bounds * feature_upper
    predicted_minimum = float(np.min(predicted_margins[region.protected_mask]))
    fisher = margin_fisher_proxy(predicted_margins, mask=region.protected_mask)
    if not displacement < region.input_radius:
        return CostateReuseDecision(
            "REFRESH", displacement, region.input_radius, feature_upper, predicted_minimum, fisher,
            False, False, "input trust ball exited",
        )
    if predicted_minimum <= 0.0:
        return CostateReuseDecision(
            "REFRESH", displacement, region.input_radius, feature_upper, predicted_minimum, fisher,
            False, False, "predicted protected margin is nonpositive",
        )
    status: ReuseStatus = "CERTIFIED_REUSE" if region.certificate_authoritative else "PROXY_REUSE"
    return CostateReuseDecision(
        status,
        displacement,
        region.input_radius,
        feature_upper,
        predicted_minimum,
        fisher,
        region.certificate_authoritative,
        region.certificate_authoritative and region.descent_bound_available,
        (
            "strict input, margin, and projected-gradient balls hold"
            if region.descent_bound_available
            else "strict empirical input/margin ball holds; exact descent remains a shadow-measured hypothesis"
        ),
    )


def validation_economics(
    *,
    baseline_validation_forwards: int,
    baseline_teacher_calls: int,
    new_anchor_validations: int,
    new_anchors: int,
    shadow_control_forwards: int,
) -> dict[str, float | int]:
    """Derive transparent operational and measurement-control accounting."""

    counts = {
        "baseline_validation_forwards": baseline_validation_forwards,
        "baseline_teacher_calls": baseline_teacher_calls,
        "new_anchor_validations": new_anchor_validations,
        "new_anchors": new_anchors,
        "shadow_control_forwards": shadow_control_forwards,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise ValueError("economics counts must be nonnegative integers")
    if baseline_validation_forwards == 0 or baseline_teacher_calls == 0 or new_anchors == 0:
        raise ValueError("baseline validations, baseline teacher calls, and new anchors must be positive")
    baseline_per_call = baseline_validation_forwards / baseline_teacher_calls
    new_per_anchor = new_anchor_validations / new_anchors
    normalized_reduction = 1.0 - new_per_anchor / baseline_per_call
    return counts | {
        "baseline_validations_per_teacher_call": baseline_per_call,
        "new_operational_validations_per_anchor": new_per_anchor,
        "normalized_validation_reduction_fraction": normalized_reduction,
        "normalized_validation_reduction_factor": (
            math.inf if new_per_anchor == 0.0 else baseline_per_call / new_per_anchor
        ),
        "new_actual_probe_forwards_including_shadow_controls": new_anchor_validations + shadow_control_forwards,
    }


def direct_costate_error_envelope(
    displacement: float,
    *,
    anchor_jacobian_norm_upper: float,
    anchor_jacobian_derivative_norm_upper: float,
    jacobian_derivative_lipschitz_upper: float,
    anchor_adjoint_norm_upper: float,
    adjoint_lipschitz_upper: float,
    correction_error_lipschitz_upper: float,
) -> float:
    """Evaluate the direct-full-costate remainder envelope ``E(r)``.

    ``jacobian_derivative_lipschitz_upper`` is ``Lip(DJ)`` on the complete
    custody-bounded ball.  A point HVP is deliberately not accepted as a
    substitute for that third-derivative bound.
    """

    radius = _finite_nonnegative(displacement, name="displacement")
    b_j = _finite_nonnegative(anchor_jacobian_norm_upper, name="anchor_jacobian_norm_upper")
    b_h = _finite_nonnegative(
        anchor_jacobian_derivative_norm_upper,
        name="anchor_jacobian_derivative_norm_upper",
    )
    l_h = _finite_nonnegative(
        jacobian_derivative_lipschitz_upper,
        name="jacobian_derivative_lipschitz_upper",
    )
    q_a = _finite_nonnegative(anchor_adjoint_norm_upper, name="anchor_adjoint_norm_upper")
    l_q = _finite_nonnegative(adjoint_lipschitz_upper, name="adjoint_lipschitz_upper")
    l_c = _finite_nonnegative(
        correction_error_lipschitz_upper,
        name="correction_error_lipschitz_upper",
    )
    linear = (b_j * l_q + l_c) * radius
    quadratic = (b_h * l_q + 0.5 * l_h * q_a) * radius * radius
    cubic = 0.5 * l_h * l_q * radius * radius * radius
    return linear + quadratic + cubic


def composed_prefix_adjoint_error_envelope(
    displacement: float,
    *,
    anchor_prefix_jacobian_norm_upper: float,
    prefix_jacobian_lipschitz_upper: float,
    suffix_costate_lipschitz_upper: float,
    jacobian_derivative_lipschitz_upper: float,
    anchor_adjoint_norm_upper: float,
    correction_error_lipschitz_upper: float,
) -> float:
    """Task-454 adjoint envelope plus the fixed-adjoint correction remainder.

    This specialization is
    ``(J0+beta*r)*kappa*(J0*r+beta*r^2/2) + M*Qa*r^2/2 + Lc*r``.
    It is exposed separately so custody-bearing task-454 constants compose
    without being silently reinterpreted as the general ``L_q`` constants.
    """

    radius = _finite_nonnegative(displacement, name="displacement")
    adjoint = costate_error_envelope(
        radius,
        anchor_jacobian_norm_upper=anchor_prefix_jacobian_norm_upper,
        jacobian_lipschitz_upper=prefix_jacobian_lipschitz_upper,
        suffix_costate_lipschitz_upper=suffix_costate_lipschitz_upper,
    )
    m = _finite_nonnegative(
        jacobian_derivative_lipschitz_upper,
        name="jacobian_derivative_lipschitz_upper",
    )
    q_a = _finite_nonnegative(anchor_adjoint_norm_upper, name="anchor_adjoint_norm_upper")
    l_c = _finite_nonnegative(
        correction_error_lipschitz_upper,
        name="correction_error_lipschitz_upper",
    )
    return adjoint + 0.5 * m * q_a * radius * radius + l_c * radius


def _strict_direct_error_radius(
    *,
    radius_cap: float,
    costate_error_tolerance: float,
    envelope_kwargs: dict[str, float],
) -> float:
    """Largest representable radius in ``[0, cap]`` satisfying strict ``E(r)<tau``."""

    cap = _finite_nonnegative(radius_cap, name="radius_cap", positive=True)
    tolerance = _finite_nonnegative(
        costate_error_tolerance, name="costate_error_tolerance", positive=True
    )

    def holds(radius: float) -> bool:
        return direct_costate_error_envelope(radius, **envelope_kwargs) < tolerance

    if holds(cap):
        return cap
    lower, upper = 0.0, cap
    while True:
        middle = lower + (upper - lower) / 2.0
        if middle in (lower, upper):
            break
        if holds(middle):
            lower = middle
        else:
            upper = middle
    if lower <= 0.0:
        raise ValueError("no positive representable direct-costate radius exists")
    return lower


@dataclass(frozen=True)
class DirectCostateCertificate:
    """Content-bound self-adjusting certificate for a full input costate."""

    provider_mode: DirectProviderMode
    authority: BoundAuthority
    geometry_authority: BoundAuthority
    input_metric: DirectInputMetric
    input_radius: float
    error_radius: float
    geometry_radius: float
    calibration_radius_cap: float
    costate_error_tolerance: float
    anchor_source_sha256: str
    anchor_costate_sha256: str
    correction_source_sha256: str
    geometry_artifact_sha256: str
    norm_artifact_sha256: str
    bound_artifact_sha256: str | None
    calibration_receipt_sha256: str | None
    anchor_jacobian_norm_upper: float
    anchor_jacobian_derivative_norm_upper: float
    jacobian_derivative_lipschitz_upper: float | None
    anchor_adjoint_norm_upper: float
    adjoint_lipschitz_upper: float
    correction_error_lipschitz_upper: float | None
    renderer_vjp_norm_upper: float
    # Lower bound on the CURRENT corrected reused renderer-gradient norm over
    # the whole ball, not merely the anchor gradient norm.
    projected_gradient_floor: float
    # Bound-artifact-backed C^{2,1} full-SegNet activation cell along [a, x].
    full_segnet_c21_cell_verified: bool
    norm_coercivity_verified: bool
    correction_numerical_bound_verified: bool
    envelope_available: bool
    rigorous_bound_complete: bool

    @property
    def certificate_authoritative(self) -> bool:
        return self.authority == "rigorous_upper_bound" and self.rigorous_bound_complete


@dataclass(frozen=True)
class DirectCostateDecision:
    status: ReuseStatus
    input_displacement: float | None
    input_radius: float
    predicted_costate_error_upper: float | None
    costate_error_tolerance: float
    descent_authoritative: bool
    reason: str
    target_scorer_input_sha256: str | None
    displacement_vector_sha256: str | None
    correction_tensor_sha256: str | None
    corrected_costate_sha256: str | None
    anchor_costate_sha256: str | None
    certificate_sha256: str | None
    correction_derivation_status: Literal["MISSING_INTEGRATED_HVP_DERIVATION_RECEIPT"]
    correction_derivation_receipt_sha256: None

    @property
    def reuses_costate(self) -> bool:
        return (
            self.status in {"CERTIFIED_REUSE", "PROXY_REUSE"}
            and self.correction_derivation_authoritative
        )

    @property
    def correction_derivation_authoritative(self) -> bool:
        """Whether a trusted integrated HVP path produced the correction.

        This landing has no such integrated path.  The property is derived
        from a closed receipt status rather than a caller-provided boolean, so
        an externally supplied tensor cannot self-authorize.
        """

        return False


def derive_direct_costate_certificate(
    *,
    anchor_jacobian_norm_upper: float,
    anchor_jacobian_derivative_norm_upper: float,
    anchor_adjoint_norm_upper: float,
    adjoint_lipschitz_upper: float,
    renderer_vjp_norm_upper: float,
    projected_gradient_floor: float,
    geometry_radius: float,
    calibration_radius_cap: float,
    authority: BoundAuthority,
    anchor_source_sha256: str,
    anchor_costate_sha256: str,
    correction_source_sha256: str,
    geometry_authority: BoundAuthority,
    geometry_artifact_sha256: str,
    norm_artifact_sha256: str,
    jacobian_derivative_lipschitz_upper: float | None = None,
    correction_error_lipschitz_upper: float | None = None,
    bound_artifact_sha256: str | None = None,
    calibration_receipt_sha256: str | None = None,
    provider_mode: DirectProviderMode = "direct_full_input_costate",
    full_segnet_c21_cell_verified: bool = False,
    norm_coercivity_verified: bool = False,
    correction_numerical_bound_verified: bool = False,
    input_metric: DirectInputMetric = "l2",
) -> DirectCostateCertificate:
    """Derive ``r*=min(r_error,r_geometry,r_cap)`` without a radius knob."""

    if provider_mode != "direct_full_input_costate":
        raise ValueError("direct certificate requires direct_full_input_costate provider mode")
    if authority not in {"rigorous_upper_bound", "empirical_local_estimate"}:
        raise ValueError("bound authority must be explicitly labelled")
    if geometry_authority not in {"rigorous_upper_bound", "empirical_local_estimate"}:
        raise ValueError("geometry authority must be explicitly labelled")
    if input_metric != "l2":
        raise ValueError("direct certificate currently supports only the coercive l2 norm")
    anchor_sha = _sha256(anchor_source_sha256, name="anchor_source_sha256")
    anchor_costate_sha = _sha256(anchor_costate_sha256, name="anchor_costate_sha256")
    correction_sha = _sha256(correction_source_sha256, name="correction_source_sha256")
    geometry_sha = _sha256(geometry_artifact_sha256, name="geometry_artifact_sha256")
    norm_sha = _sha256(norm_artifact_sha256, name="norm_artifact_sha256")
    geometry = _finite_nonnegative(geometry_radius, name="geometry_radius", positive=True)
    cap = _finite_nonnegative(
        calibration_radius_cap, name="calibration_radius_cap", positive=True
    )
    renderer = _finite_nonnegative(
        renderer_vjp_norm_upper, name="renderer_vjp_norm_upper", positive=True
    )
    floor = _finite_nonnegative(
        projected_gradient_floor, name="projected_gradient_floor", positive=True
    )
    tolerance = floor / renderer
    common = {
        "anchor_jacobian_norm_upper": _finite_nonnegative(
            anchor_jacobian_norm_upper, name="anchor_jacobian_norm_upper"
        ),
        "anchor_jacobian_derivative_norm_upper": _finite_nonnegative(
            anchor_jacobian_derivative_norm_upper,
            name="anchor_jacobian_derivative_norm_upper",
        ),
        "anchor_adjoint_norm_upper": _finite_nonnegative(
            anchor_adjoint_norm_upper, name="anchor_adjoint_norm_upper"
        ),
        "adjoint_lipschitz_upper": _finite_nonnegative(
            adjoint_lipschitz_upper, name="adjoint_lipschitz_upper"
        ),
    }
    bounds_complete = (
        jacobian_derivative_lipschitz_upper is not None
        and correction_error_lipschitz_upper is not None
    )
    complete = (
        bounds_complete
        and full_segnet_c21_cell_verified
        and norm_coercivity_verified
        and correction_numerical_bound_verified
        and geometry_authority == "rigorous_upper_bound"
    )
    if authority == "rigorous_upper_bound":
        if not complete:
            raise ValueError(
                "rigorous direct reuse requires custody-bearing Lip(DJ)/correction bounds, "
                "rigorous geometry, a verified C2,1 full-SegNet activation cell, norm "
                "coercivity, and a correction numerical-error bound"
            )
        bound_sha = _sha256(bound_artifact_sha256, name="bound_artifact_sha256")
        if calibration_receipt_sha256 is not None:
            raise ValueError("rigorous direct reuse cannot use empirical calibration custody")
        calibration_sha = None
    else:
        calibration_sha = _sha256(
            calibration_receipt_sha256, name="calibration_receipt_sha256"
        )
        if bound_artifact_sha256 is not None:
            raise ValueError("empirical direct reuse cannot claim a rigorous bound artifact")
        bound_sha = None
    # Missing empirical terms explicitly block operational reuse; zero would
    # understate the measured envelope and is therefore not substituted.
    if not bounds_complete:
        error_radius = 0.0
    else:
        envelope = common | {
            "jacobian_derivative_lipschitz_upper": _finite_nonnegative(
                jacobian_derivative_lipschitz_upper,
                name="jacobian_derivative_lipschitz_upper",
            ),
            "correction_error_lipschitz_upper": _finite_nonnegative(
                correction_error_lipschitz_upper,
                name="correction_error_lipschitz_upper",
            ),
        }
        error_radius = _strict_direct_error_radius(
            radius_cap=cap,
            costate_error_tolerance=tolerance,
            envelope_kwargs=envelope,
        )
    return DirectCostateCertificate(
        provider_mode=provider_mode,
        authority=authority,
        geometry_authority=geometry_authority,
        input_metric=input_metric,
        input_radius=min(error_radius, geometry, cap),
        error_radius=error_radius,
        geometry_radius=geometry,
        calibration_radius_cap=cap,
        costate_error_tolerance=tolerance,
        anchor_source_sha256=anchor_sha,
        anchor_costate_sha256=anchor_costate_sha,
        correction_source_sha256=correction_sha,
        geometry_artifact_sha256=geometry_sha,
        norm_artifact_sha256=norm_sha,
        bound_artifact_sha256=bound_sha,
        calibration_receipt_sha256=calibration_sha,
        jacobian_derivative_lipschitz_upper=(
            None
            if jacobian_derivative_lipschitz_upper is None
            else float(jacobian_derivative_lipschitz_upper)
        ),
        correction_error_lipschitz_upper=(
            None
            if correction_error_lipschitz_upper is None
            else float(correction_error_lipschitz_upper)
        ),
        renderer_vjp_norm_upper=renderer,
        projected_gradient_floor=floor,
        full_segnet_c21_cell_verified=bool(full_segnet_c21_cell_verified),
        norm_coercivity_verified=bool(norm_coercivity_verified),
        correction_numerical_bound_verified=bool(correction_numerical_bound_verified),
        envelope_available=bounds_complete,
        rigorous_bound_complete=complete,
        **common,
    )


def direct_certificate_sha256(certificate: DirectCostateCertificate) -> str:
    """Stable identity of every scalar and custody field in a certificate."""

    encoded = json.dumps(
        asdict(certificate), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_direct_costate_certificate(
    *,
    input_displacement: float | None = None,
    displacement_vector: Any | None = None,
    anchor_scorer_input: Any | None = None,
    target_scorer_input: Any | None = None,
    anchor_costate: Any | None = None,
    correction: Any | None = None,
    certificate: DirectCostateCertificate,
    current_anchor_source_sha256: str,
    current_correction_source_sha256: str,
    current_geometry_artifact_sha256: str | None = None,
    current_norm_artifact_sha256: str | None = None,
    input_metric: DirectInputMetric = "l2",
    custody_matches: bool = True,
) -> DirectCostateDecision:
    """Apply strict membership to actual content-bound tensors.

    The legacy scalar-only call remains accepted, but can only return
    ``BLOCKED``.  It cannot authorize reuse because it does not identify the
    target point, direction, correction, or corrected costate bytes.
    """

    certificate_sha = direct_certificate_sha256(certificate)

    def blocked(reason: str, displacement: float | None = None) -> DirectCostateDecision:
        return DirectCostateDecision(
            status="BLOCKED",
            input_displacement=displacement,
            input_radius=certificate.input_radius,
            predicted_costate_error_upper=None,
            costate_error_tolerance=certificate.costate_error_tolerance,
            descent_authoritative=False,
            reason=reason,
            target_scorer_input_sha256=None,
            displacement_vector_sha256=None,
            correction_tensor_sha256=None,
            corrected_costate_sha256=None,
            anchor_costate_sha256=None,
            certificate_sha256=certificate_sha,
            correction_derivation_status="MISSING_INTEGRATED_HVP_DERIVATION_RECEIPT",
            correction_derivation_receipt_sha256=None,
        )

    if (
        not custody_matches
        or current_anchor_source_sha256 != certificate.anchor_source_sha256
        or current_correction_source_sha256 != certificate.correction_source_sha256
        or current_geometry_artifact_sha256 != certificate.geometry_artifact_sha256
        or current_norm_artifact_sha256 != certificate.norm_artifact_sha256
        or input_metric != certificate.input_metric
    ):
        return blocked("anchor/correction/geometry/norm custody changed")
    if any(
        value is None
        for value in (
            displacement_vector,
            anchor_scorer_input,
            target_scorer_input,
            anchor_costate,
            correction,
        )
    ):
        legacy = (
            None
            if input_displacement is None
            else _finite_nonnegative(input_displacement, name="input_displacement")
        )
        return blocked("scalar-only direct-costate membership lacks tensor custody", legacy)
    try:
        raw_direction_sha = array_sha256(displacement_vector)
        raw_anchor_input_sha = array_sha256(anchor_scorer_input)
        raw_target_sha = array_sha256(target_scorer_input)
        raw_anchor_costate_sha = array_sha256(anchor_costate)
        raw_correction_sha = array_sha256(correction)
        direction = _finite_array(displacement_vector, name="displacement_vector")
        anchor_input = _finite_array(anchor_scorer_input, name="anchor_scorer_input")
        target = _finite_array(target_scorer_input, name="target_scorer_input")
        anchor = _finite_array(anchor_costate, name="anchor_costate")
        delta = _finite_array(correction, name="correction")
    except ValueError as exc:
        return blocked(str(exc))
    if anchor.shape != delta.shape:
        return blocked("anchor costate and correction shapes differ")
    if direction.shape != target.shape or anchor_input.shape != target.shape:
        return blocked("anchor/target scorer inputs and displacement vector shapes differ")
    if raw_anchor_input_sha != certificate.anchor_source_sha256:
        return blocked("anchor scorer-input bytes differ from certificate custody")
    actual_direction = target - anchor_input
    if array_sha256(direction) != array_sha256(actual_direction):
        return blocked("displacement vector does not equal target minus anchor")
    displacement = float(np.linalg.norm(direction.reshape(-1), ord=2))
    if input_displacement is not None and float(input_displacement) != displacement:
        return blocked("supplied scalar displacement differs from actual vector norm", displacement)
    anchor_costate_sha = raw_anchor_costate_sha
    if anchor_costate_sha != certificate.anchor_costate_sha256:
        return blocked("anchor costate bytes differ from certificate custody", displacement)
    corrected = apply_direct_costate_correction(anchor_costate=anchor, correction=delta)
    target_sha = raw_target_sha
    direction_sha = raw_direction_sha
    correction_sha = raw_correction_sha
    corrected_sha = array_sha256(corrected)
    if not certificate.envelope_available:
        return blocked("measured Lip(DJ) or correction-error envelope term is absent", displacement)
    error = direct_costate_error_envelope(
        displacement,
        anchor_jacobian_norm_upper=certificate.anchor_jacobian_norm_upper,
        anchor_jacobian_derivative_norm_upper=certificate.anchor_jacobian_derivative_norm_upper,
        jacobian_derivative_lipschitz_upper=certificate.jacobian_derivative_lipschitz_upper,
        anchor_adjoint_norm_upper=certificate.anchor_adjoint_norm_upper,
        adjoint_lipschitz_upper=certificate.adjoint_lipschitz_upper,
        correction_error_lipschitz_upper=certificate.correction_error_lipschitz_upper,
    )
    if not displacement < certificate.input_radius:
        return DirectCostateDecision(
            "REFRESH", displacement, certificate.input_radius, error,
            certificate.costate_error_tolerance, False, "direct trust ball exited",
            target_sha, direction_sha, correction_sha, corrected_sha,
            anchor_costate_sha, certificate_sha,
            "MISSING_INTEGRATED_HVP_DERIVATION_RECEIPT", None,
        )
    status: ReuseStatus = (
        "CERTIFIED_REUSE" if certificate.certificate_authoritative else "PROXY_REUSE"
    )
    return DirectCostateDecision(
        status, displacement, certificate.input_radius, error,
        certificate.costate_error_tolerance, False,
        (
            "strict direct-costate error, geometry, and calibration balls hold geometrically; "
            "integrated HVP derivation receipt is absent, so operational reuse remains blocked"
        ),
        target_sha, direction_sha, correction_sha, corrected_sha,
        anchor_costate_sha, certificate_sha,
        "MISSING_INTEGRATED_HVP_DERIVATION_RECEIPT", None,
    )


def apply_direct_costate_correction(*, anchor_costate: Any, correction: Any) -> np.ndarray:
    """Return ``p(a)+c_tilde_a(d)`` with shape and finiteness checks."""

    anchor = _finite_array(anchor_costate, name="anchor_costate")
    delta = _finite_array(correction, name="correction")
    if anchor.shape != delta.shape:
        raise ValueError("anchor costate and correction must have identical shapes")
    return anchor + delta


def torch_fixed_adjoint_jacobian_hvp(
    *, logits: Any, scorer_input: Any, anchor_adjoint: Any, direction: Any
) -> Any:
    """Compute ``(DJ(a)[d])^T q_a`` while holding ``q_a`` detached.

    ``logits`` must retain the graph from ``scorer_input``.  This helper does
    not compute the full CE Hessian-vector, whose additional term is adjoint
    drift ``J(a)^T Dq(a)[d]``.
    """

    import torch

    q_anchor = anchor_adjoint.detach()
    p_anchor = torch.autograd.grad(
        logits,
        scorer_input,
        grad_outputs=q_anchor,
        create_graph=True,
        retain_graph=True,
    )[0]
    vector = direction.detach()
    if vector.shape != scorer_input.shape:
        raise ValueError("direction and scorer_input must have identical shapes")
    if not p_anchor.requires_grad:
        return torch.zeros_like(scorer_input)
    return torch.autograd.grad(
        torch.sum(p_anchor * vector), scorer_input, retain_graph=True
    )[0]


__all__ = [
    "CostateReuseDecision",
    "CostateTrustRegion",
    "DirectCostateCertificate",
    "DirectCostateDecision",
    "apply_direct_costate_correction",
    "array_sha256",
    "check_costate_trust_region",
    "check_direct_costate_certificate",
    "composed_prefix_adjoint_error_envelope",
    "costate_error_envelope",
    "derive_costate_trust_region",
    "derive_direct_costate_certificate",
    "direct_certificate_sha256",
    "direct_costate_error_envelope",
    "fit_empirical_jacobian_envelope",
    "input_radius_from_feature_radius",
    "margin_fisher_proxy",
    "prefix_feature_envelope",
    "torch_fixed_adjoint_jacobian_hvp",
    "validation_economics",
]
