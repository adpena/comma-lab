from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from tac.scorer_surrogate.costate_trust_region import (
    array_sha256,
    check_costate_trust_region,
    costate_error_envelope,
    derive_costate_trust_region,
    fit_empirical_jacobian_envelope,
    input_radius_from_feature_radius,
    margin_fisher_proxy,
    prefix_feature_envelope,
    validation_economics,
)

SHA_A = hashlib.sha256(b"anchor").hexdigest()
SHA_B = hashlib.sha256(b"bound").hexdigest()
SHA_C = hashlib.sha256(b"calibration").hexdigest()


def _region(*, authority: str = "rigorous_upper_bound"):
    kwargs = (
        {"bound_artifact_sha256": SHA_B}
        if authority == "rigorous_upper_bound"
        else {"calibration_receipt_sha256": SHA_C}
    )
    return derive_costate_trust_region(
        anchor_margins=np.array([[[2.0, 3.0], [4.0, 5.0]]]),
        anchor_correct_mask=np.ones((1, 2, 2), dtype=bool),
        pairwise_suffix_lipschitz_upper=np.ones((1, 2, 2)),
        anchor_jacobian_norm_upper=2.0,
        jacobian_lipschitz_upper=1.0,
        suffix_costate_lipschitz_upper=0.25,
        renderer_vjp_norm_upper=1.0,
        projected_gradient_floor=0.5,
        provider_mode="current_prefix_vjp_banked_suffix",
        authority=authority,
        anchor_frame_sha256=SHA_A,
        **kwargs,
    )


def test_stable_quadratic_radius_inverts_prefix_envelope() -> None:
    radius = input_radius_from_feature_radius(
        2.0, anchor_jacobian_norm_upper=3.0, jacobian_lipschitz_upper=4.0
    )
    assert prefix_feature_envelope(
        radius, anchor_jacobian_norm_upper=3.0, jacobian_lipschitz_upper=4.0
    ) == pytest.approx(2.0)
    assert input_radius_from_feature_radius(
        2.0, anchor_jacobian_norm_upper=4.0, jacobian_lipschitz_upper=0.0
    ) == pytest.approx(0.5)


def test_fisher_proxy_is_margin_sensitive_without_overflow() -> None:
    near = margin_fisher_proxy(np.array([0.0]))
    far = margin_fisher_proxy(np.array([1000.0]))
    assert near == pytest.approx(0.25)
    assert far == 0.0


def test_empirical_jacobian_envelope_covers_every_calibration_probe() -> None:
    inputs = np.array([1.0, 2.0, 4.0])
    features = np.array([2.0, 5.0, 14.0])
    jacobian, beta = fit_empirical_jacobian_envelope(
        input_displacements=inputs, feature_displacements=features
    )
    assert jacobian == pytest.approx(2.0)
    for displacement, observed in zip(inputs, features, strict=True):
        assert prefix_feature_envelope(
            float(displacement),
            anchor_jacobian_norm_upper=jacobian,
            jacobian_lipschitz_upper=beta,
        ) >= observed


def test_rigorous_region_accepts_only_strict_interior_and_binds_custody() -> None:
    region = _region()
    anchor = np.zeros((1, 2, 2, 3), dtype=np.float64)
    inside = np.full_like(anchor, np.nextafter(region.input_radius, 0.0))
    decision = check_costate_trust_region(
        anchor_frame=anchor,
        current_frame=inside,
        region=region,
        current_anchor_frame_sha256=SHA_A,
    )
    assert decision.status == "CERTIFIED_REUSE"
    assert decision.label_cell_authoritative
    assert decision.descent_authoritative
    boundary = check_costate_trust_region(
        anchor_frame=anchor,
        current_frame=np.full_like(anchor, region.input_radius),
        region=region,
        current_anchor_frame_sha256=SHA_A,
    )
    assert boundary.status == "REFRESH"
    blocked = check_costate_trust_region(
        anchor_frame=anchor,
        current_frame=anchor,
        region=region,
        current_anchor_frame_sha256=SHA_B,
    )
    assert blocked.status == "BLOCKED"


def test_empirical_region_never_claims_a_certificate() -> None:
    region = _region(authority="empirical_local_estimate")
    frame = np.zeros((1, 2, 2, 3), dtype=np.float32)
    decision = check_costate_trust_region(
        anchor_frame=frame,
        current_frame=frame,
        region=region,
        current_anchor_frame_sha256=SHA_A,
    )
    assert decision.status == "PROXY_REUSE"
    assert not decision.label_cell_authoritative
    assert not decision.descent_authoritative


def test_empirical_margin_only_region_exposes_missing_descent_bound() -> None:
    region = derive_costate_trust_region(
        anchor_margins=np.array([[[2.0]]]),
        anchor_correct_mask=np.array([[[True]]]),
        pairwise_suffix_lipschitz_upper=np.array([[[1.0]]]),
        anchor_jacobian_norm_upper=1.0,
        jacobian_lipschitz_upper=0.0,
        provider_mode="current_prefix_vjp_banked_suffix",
        authority="empirical_local_estimate",
        anchor_frame_sha256=SHA_A,
        calibration_receipt_sha256=SHA_C,
    )
    assert not region.descent_bound_available
    assert math.isinf(region.descent_radius)
    frame = np.zeros((1, 1, 1, 3))
    decision = check_costate_trust_region(
        anchor_frame=frame,
        current_frame=frame,
        region=region,
        current_anchor_frame_sha256=SHA_A,
    )
    assert decision.status == "PROXY_REUSE"
    assert not decision.descent_authoritative
    assert "shadow-measured" in decision.reason


def test_certificate_refuses_direct_full_input_costate_reuse() -> None:
    with pytest.raises(ValueError, match="current prefix VJP"):
        derive_costate_trust_region(
            anchor_margins=np.array([[[2.0]]]),
            anchor_correct_mask=np.array([[[True]]]),
            pairwise_suffix_lipschitz_upper=np.array([[[1.0]]]),
            anchor_jacobian_norm_upper=1.0,
            jacobian_lipschitz_upper=0.0,
            provider_mode="direct_full_input_costate",  # type: ignore[arg-type]
            authority="empirical_local_estimate",
            anchor_frame_sha256=SHA_A,
            calibration_receipt_sha256=SHA_C,
        )


def test_costate_error_bound_controls_a_known_scalar_teacher_descent() -> None:
    # Prefix h=x has J=1 and beta=0.  For CE(logits=[h,0], target=0),
    # |d p / d h| <= 1/4 globally.  The renderer is identity.
    region = derive_costate_trust_region(
        anchor_margins=np.array([[[2.0]]]),
        anchor_correct_mask=np.array([[[True]]]),
        pairwise_suffix_lipschitz_upper=np.array([[[1.0]]]),
        anchor_jacobian_norm_upper=1.0,
        jacobian_lipschitz_upper=0.0,
        suffix_costate_lipschitz_upper=0.25,
        renderer_vjp_norm_upper=1.0,
        projected_gradient_floor=1.0 / (1.0 + math.exp(2.0)),
        provider_mode="current_prefix_vjp_banked_suffix",
        authority="rigorous_upper_bound",
        anchor_frame_sha256=SHA_A,
        bound_artifact_sha256=SHA_B,
    )
    x = 2.0
    banked_costate = -1.0 / (1.0 + math.exp(x))
    step = min(region.input_radius / 4.0, abs(banked_costate) / 2.0)
    candidate = x - step * banked_costate
    def loss(value: float) -> float:
        return math.log1p(math.exp(-value))
    assert abs(candidate - x) < region.input_radius
    assert loss(candidate) < loss(x)
    assert costate_error_envelope(
        abs(candidate - x),
        anchor_jacobian_norm_upper=1.0,
        jacobian_lipschitz_upper=0.0,
        suffix_costate_lipschitz_upper=0.25,
    ) < region.projected_gradient_floor


def test_arrays_are_content_bound_and_region_copies_are_immutable() -> None:
    original = np.array([[[2.0, 3.0], [4.0, 5.0]]])
    before = array_sha256(original)
    region = _region()
    original[...] = -1.0
    assert array_sha256(original) != before
    assert region.anchor_margin_sha256 == array_sha256(region.anchor_margins)
    with pytest.raises(ValueError):
        region.anchor_margins[...] = 0.0


def test_validation_economics_keeps_shadow_controls_visible() -> None:
    row = validation_economics(
        baseline_validation_forwards=402,
        baseline_teacher_calls=48,
        new_anchor_validations=3,
        new_anchors=3,
        shadow_control_forwards=12,
    )
    assert row["baseline_validations_per_teacher_call"] == pytest.approx(8.375)
    assert row["new_operational_validations_per_anchor"] == 1.0
    assert row["normalized_validation_reduction_factor"] == 8.375
    assert row["normalized_validation_reduction_fraction"] == pytest.approx(1.0 - 1.0 / 8.375)
    assert row["new_actual_probe_forwards_including_shadow_controls"] == 15


@pytest.mark.parametrize(
    "kwargs",
    [
        {"suffix_costate_lipschitz_upper": -1.0},
        {"renderer_vjp_norm_upper": -1.0},
        {"projected_gradient_floor": 0.0},
    ],
)
def test_invalid_descent_bounds_fail_closed(kwargs: dict[str, float]) -> None:
    base = {
        "anchor_margins": np.array([[[2.0]]]),
        "anchor_correct_mask": np.array([[[True]]]),
        "pairwise_suffix_lipschitz_upper": np.array([[[1.0]]]),
        "anchor_jacobian_norm_upper": 1.0,
        "jacobian_lipschitz_upper": 0.0,
        "suffix_costate_lipschitz_upper": 0.25,
        "renderer_vjp_norm_upper": 1.0,
        "projected_gradient_floor": 0.1,
        "provider_mode": "current_prefix_vjp_banked_suffix",
        "authority": "rigorous_upper_bound",
        "anchor_frame_sha256": SHA_A,
        "bound_artifact_sha256": SHA_B,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        derive_costate_trust_region(**base)
