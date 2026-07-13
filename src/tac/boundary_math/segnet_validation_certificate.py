# SPDX-License-Identifier: MIT
"""Fail-closed SegNet margin trust regions at the YOPO first-block cut.

The rigorous path consumes an externally supplied *upper bound* for the
downstream pairwise-logit map.  A local Jacobian is deliberately not accepted
as that artifact.  The empirical path uses disjoint calibration candidates and
can emit only ``PROXY_ACCEPT``; it is never a certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

BoundAuthority = Literal["rigorous_upper_bound", "empirical_local_estimate"]
DecisionStatus = Literal["ACCEPT", "PROXY_ACCEPT", "REFRESH", "BLOCKED"]


def _sha256(value: str | None, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite_array(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


@dataclass(frozen=True)
class FeatureTrustRegion:
    """One content-bound global feature-space trust region."""

    feature_radius: float
    protected_pixels: int
    authority: BoundAuthority
    anchor_feature_sha256: str
    bound_artifact_sha256: str | None
    calibration_receipt_sha256: str | None
    minimum_anchor_margin: float

    @property
    def certificate_authoritative(self) -> bool:
        return self.authority == "rigorous_upper_bound"


@dataclass(frozen=True)
class ValidationDecision:
    status: DecisionStatus
    feature_displacement_linf: float | None
    feature_radius: float
    certificate_authoritative: bool
    reason: str

    @property
    def reuses_provider(self) -> bool:
        return self.status in {"ACCEPT", "PROXY_ACCEPT"}


def derive_feature_trust_region(
    *,
    anchor_margins: Any,
    anchor_correct_mask: Any,
    pairwise_logit_change_bounds: Any,
    authority: BoundAuthority,
    anchor_feature_sha256: str,
    bound_artifact_sha256: str | None = None,
    calibration_receipt_sha256: str | None = None,
) -> FeatureTrustRegion:
    """Derive ``min(correct_margin / pairwise_change_bound)`` in feature units."""

    margins = _finite_array(anchor_margins, name="anchor_margins")
    bounds = _finite_array(pairwise_logit_change_bounds, name="pairwise_logit_change_bounds")
    correct = np.asarray(anchor_correct_mask)
    if margins.shape != bounds.shape or margins.shape != correct.shape:
        raise ValueError("margins, bounds, and anchor-correct mask must have identical shapes")
    if margins.size == 0 or correct.dtype.hasobject:
        raise ValueError("trust-region arrays must be non-empty and non-object")
    if authority not in {"rigorous_upper_bound", "empirical_local_estimate"}:
        raise ValueError("bound authority must be explicitly labelled")
    correct = correct.astype(bool, copy=False)
    if not bool(correct.any()):
        raise ValueError("anchor has no correctly predicted pixels to protect")
    protected_margins = margins[correct]
    protected_bounds = bounds[correct]
    if bool((protected_margins <= 0.0).any()):
        raise ValueError("every anchor-correct pixel must have a strictly positive margin")
    if bool((protected_bounds < 0.0).any()):
        raise ValueError("pairwise-logit change bounds must be nonnegative")
    _sha256(anchor_feature_sha256, field="anchor_feature_sha256")
    if authority == "rigorous_upper_bound":
        _sha256(bound_artifact_sha256, field="bound_artifact_sha256")
        if calibration_receipt_sha256 is not None:
            raise ValueError("empirical calibration inputs cannot be presented as rigorous")
    else:
        _sha256(calibration_receipt_sha256, field="calibration_receipt_sha256")
        if bound_artifact_sha256 is not None:
            raise ValueError("empirical estimates cannot carry a rigorous-bound artifact field")
    ratios = np.divide(
        protected_margins,
        protected_bounds,
        out=np.full_like(protected_margins, np.inf),
        where=protected_bounds > 0.0,
    )
    radius = float(np.min(ratios))
    if math.isnan(radius) or radius <= 0.0:
        raise ValueError("derived feature radius must be positive")
    return FeatureTrustRegion(
        feature_radius=radius,
        protected_pixels=int(correct.sum()),
        authority=authority,
        anchor_feature_sha256=anchor_feature_sha256,
        bound_artifact_sha256=bound_artifact_sha256,
        calibration_receipt_sha256=calibration_receipt_sha256,
        minimum_anchor_margin=float(np.min(protected_margins)),
    )


def calibrate_empirical_pairwise_bounds(
    *, anchor_pairwise_margins: Any, candidate_pairwise_margins: Any, feature_displacements_linf: Any, eps: float = 1e-12
) -> np.ndarray:
    """Estimate per-pixel local slopes on a pre-registered calibration subset."""

    anchor = _finite_array(anchor_pairwise_margins, name="anchor_pairwise_margins")
    candidates = _finite_array(candidate_pairwise_margins, name="candidate_pairwise_margins")
    displacement = _finite_array(feature_displacements_linf, name="feature_displacements_linf")
    if candidates.ndim != anchor.ndim + 1 or candidates.shape[1:] != anchor.shape:
        raise ValueError("candidate margins must have shape (candidates, *anchor_shape)")
    if displacement.shape != (candidates.shape[0],) or candidates.shape[0] == 0:
        raise ValueError("one feature displacement is required for each calibration candidate")
    if not math.isfinite(eps) or eps <= 0.0 or bool((displacement < 0.0).any()):
        raise ValueError("eps must be positive and displacements nonnegative")
    denominator = np.maximum(displacement, eps).reshape((-1,) + (1,) * anchor.ndim)
    return np.max(np.abs(candidates - anchor[None, ...]) / denominator, axis=0)


def check_feature_trust_region(
    *,
    anchor_feature: Any,
    current_feature: Any,
    region: FeatureTrustRegion,
    current_anchor_feature_sha256: str,
    custody_matches: bool = True,
) -> ValidationDecision:
    """Apply the cheap event gate; equality is an exit because the theorem is strict."""

    if not custody_matches or current_anchor_feature_sha256 != region.anchor_feature_sha256:
        return ValidationDecision("BLOCKED", None, region.feature_radius, False, "anchor/provider custody changed")
    try:
        anchor = _finite_array(anchor_feature, name="anchor_feature")
        current = _finite_array(current_feature, name="current_feature")
    except ValueError as exc:
        return ValidationDecision("BLOCKED", None, region.feature_radius, False, str(exc))
    if anchor.shape != current.shape or anchor.size == 0:
        return ValidationDecision("BLOCKED", None, region.feature_radius, False, "feature shapes differ or are empty")
    displacement = float(np.max(np.abs(current - anchor)))
    if not displacement < region.feature_radius:
        return ValidationDecision("REFRESH", displacement, region.feature_radius, False, "feature ball exited")
    status: DecisionStatus = "ACCEPT" if region.certificate_authoritative else "PROXY_ACCEPT"
    return ValidationDecision(status, displacement, region.feature_radius, region.certificate_authoritative, "strict feature ball holds")


@dataclass
class ProxyConfusionAccumulator:
    """Holdout meter where any exact CE/d_seg/d_pose worsening is unsafe."""

    unsafe_accepts_any: int = 0
    unsafe_accepts_ce: int = 0
    unsafe_accepts_dseg: int = 0
    unsafe_accepts_dpose: int = 0
    safe_rejects: int = 0
    exact_safe_accepts: int = 0
    exact_unsafe_rejects: int = 0

    def update(
        self,
        *,
        proxy_accepts: bool,
        exact_ce_worsens: bool,
        exact_dseg_worsens: bool,
        exact_dpose_worsens: bool,
    ) -> None:
        exact_worsens = exact_ce_worsens or exact_dseg_worsens or exact_dpose_worsens
        if proxy_accepts and exact_worsens:
            self.unsafe_accepts_any += 1
            self.unsafe_accepts_ce += int(exact_ce_worsens)
            self.unsafe_accepts_dseg += int(exact_dseg_worsens)
            self.unsafe_accepts_dpose += int(exact_dpose_worsens)
        elif not proxy_accepts and exact_worsens:
            self.safe_rejects += 1
        elif proxy_accepts:
            self.exact_safe_accepts += 1
        else:
            self.exact_unsafe_rejects += 1

    @property
    def false_negative(self) -> int:
        """Pre-registered d_seg-only false-negative count."""

        return self.unsafe_accepts_dseg

    def to_dict(self) -> dict[str, int]:
        return {
            "unsafe_accepts_any": self.unsafe_accepts_any,
            "unsafe_accepts_ce": self.unsafe_accepts_ce,
            "unsafe_accepts_dseg": self.unsafe_accepts_dseg,
            "unsafe_accepts_dpose": self.unsafe_accepts_dpose,
            "false_negative": self.false_negative,
            "false_negative_dseg": self.false_negative,
            "safe_rejects": self.safe_rejects,
            "exact_safe_accepts": self.exact_safe_accepts,
            "exact_unsafe_rejects": self.exact_unsafe_rejects,
        }


def confusion_meter_canaries() -> dict[str, Any]:
    meter = ProxyConfusionAccumulator()
    meter.update(proxy_accepts=True, exact_ce_worsens=True, exact_dseg_worsens=False, exact_dpose_worsens=True)
    meter.update(proxy_accepts=False, exact_ce_worsens=False, exact_dseg_worsens=False, exact_dpose_worsens=False)
    passed = (
        meter.unsafe_accepts_any == 1
        and meter.unsafe_accepts_ce == 1
        and meter.unsafe_accepts_dseg == 0
        and meter.unsafe_accepts_dpose == 1
        and meter.exact_unsafe_rejects == 1
    )
    return {"status": "PASS" if passed else "FAIL", "positive_and_negative_meter_canaries": meter.to_dict()}


def cadence_speedup(*, cadence: int, t_exact: float, t_approx: float, t_validate_cheap: float, t_fallback: float) -> float:
    """Derived whole-cycle economics from measured component timings."""

    values = (t_exact, t_approx, t_validate_cheap, t_fallback)
    if not isinstance(cadence, int) or isinstance(cadence, bool) or cadence < 1:
        raise ValueError("cadence must be an integer >= 1")
    if any(not math.isfinite(value) or value < 0.0 for value in values) or t_exact <= 0.0:
        raise ValueError("timings must be finite and nonnegative, with t_exact > 0")
    return cadence * t_exact / (t_exact + (cadence - 1) * (t_approx + t_validate_cheap + t_fallback))


__all__ = [
    "FeatureTrustRegion", "ProxyConfusionAccumulator", "ValidationDecision",
    "cadence_speedup", "calibrate_empirical_pairwise_bounds", "check_feature_trust_region",
    "confusion_meter_canaries", "derive_feature_trust_region",
]
