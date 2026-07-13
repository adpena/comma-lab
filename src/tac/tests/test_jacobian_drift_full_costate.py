from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from tac.scorer_surrogate.costate_trust_region import (
    apply_direct_costate_correction,
    array_sha256,
    check_direct_costate_certificate,
    composed_prefix_adjoint_error_envelope,
    derive_direct_costate_certificate,
    direct_costate_error_envelope,
    torch_fixed_adjoint_jacobian_hvp,
)

ANCHOR_INPUT = np.zeros(2, dtype=np.float64)
TARGET_INPUT = np.array([0.01, 0.0], dtype=np.float64)
DISPLACEMENT = TARGET_INPUT - ANCHOR_INPUT
ANCHOR_COSTATE = np.array([0.5, -0.25], dtype=np.float64)
CORRECTION_TENSOR = np.array([0.01, -0.01], dtype=np.float64)
ANCHOR = array_sha256(ANCHOR_INPUT)
ANCHOR_COSTATE_SHA = array_sha256(ANCHOR_COSTATE)
CORRECTION = hashlib.sha256(b"correction-source").hexdigest()
BOUND = hashlib.sha256(b"rigorous-bounds").hexdigest()
CALIBRATION = hashlib.sha256(b"empirical-calibration").hexdigest()
GEOMETRY = hashlib.sha256(b"geometry").hexdigest()
NORM = hashlib.sha256(b"norm").hexdigest()


def _certificate(*, authority: str = "rigorous_upper_bound"):
    custody = (
        {"bound_artifact_sha256": BOUND}
        if authority == "rigorous_upper_bound"
        else {"calibration_receipt_sha256": CALIBRATION}
    )
    return derive_direct_costate_certificate(
        anchor_jacobian_norm_upper=2.0,
        anchor_jacobian_derivative_norm_upper=3.0,
        jacobian_derivative_lipschitz_upper=4.0,
        anchor_adjoint_norm_upper=5.0,
        adjoint_lipschitz_upper=0.25,
        correction_error_lipschitz_upper=0.125,
        renderer_vjp_norm_upper=2.0,
        projected_gradient_floor=0.5,
        geometry_radius=1.0,
        calibration_radius_cap=2.0,
        authority=authority,
        anchor_source_sha256=ANCHOR,
        anchor_costate_sha256=ANCHOR_COSTATE_SHA,
        correction_source_sha256=CORRECTION,
        geometry_authority=authority,
        geometry_artifact_sha256=GEOMETRY,
        norm_artifact_sha256=NORM,
        full_segnet_c21_cell_verified=True,
        norm_coercivity_verified=True,
        correction_numerical_bound_verified=True,
        **custody,
    )


def _check(certificate, **overrides):
    kwargs = {
        "displacement_vector": DISPLACEMENT,
        "anchor_scorer_input": ANCHOR_INPUT,
        "target_scorer_input": TARGET_INPUT,
        "anchor_costate": ANCHOR_COSTATE,
        "correction": CORRECTION_TENSOR,
        "certificate": certificate,
        "current_anchor_source_sha256": ANCHOR,
        "current_correction_source_sha256": CORRECTION,
        "current_geometry_artifact_sha256": GEOMETRY,
        "current_norm_artifact_sha256": NORM,
    }
    kwargs.update(overrides)
    return check_direct_costate_certificate(**kwargs)


def test_direct_envelope_is_exact_derived_cubic() -> None:
    radius = 0.5
    observed = direct_costate_error_envelope(
        radius,
        anchor_jacobian_norm_upper=2.0,
        anchor_jacobian_derivative_norm_upper=3.0,
        jacobian_derivative_lipschitz_upper=4.0,
        anchor_adjoint_norm_upper=5.0,
        adjoint_lipschitz_upper=0.25,
        correction_error_lipschitz_upper=0.125,
    )
    expected = (2.0 * 0.25 + 0.125) * radius
    expected += (3.0 * 0.25 + 0.5 * 4.0 * 5.0) * radius**2
    expected += 0.5 * 4.0 * 0.25 * radius**3
    assert observed == pytest.approx(expected)


def test_task454_composed_specialization_is_exposed() -> None:
    radius = 0.2
    observed = composed_prefix_adjoint_error_envelope(
        radius,
        anchor_prefix_jacobian_norm_upper=2.0,
        prefix_jacobian_lipschitz_upper=3.0,
        suffix_costate_lipschitz_upper=0.5,
        jacobian_derivative_lipschitz_upper=4.0,
        anchor_adjoint_norm_upper=5.0,
        correction_error_lipschitz_upper=0.1,
    )
    expected = (2.0 + 3.0 * radius) * 0.5 * (2.0 * radius + 1.5 * radius**2)
    expected += 0.5 * 4.0 * 5.0 * radius**2 + 0.1 * radius
    assert observed == pytest.approx(expected)


def test_self_adjusting_radius_is_strict_and_empirical_never_certifies() -> None:
    rigorous = _certificate()
    interior = _check(rigorous)
    assert interior.status == "CERTIFIED_REUSE"
    assert not interior.descent_authoritative
    assert not interior.reuses_costate
    assert "operational reuse remains blocked" in interior.reason
    boundary_target = np.array([rigorous.input_radius, 0.0])
    boundary = _check(
        rigorous,
        displacement_vector=boundary_target - ANCHOR_INPUT,
        target_scorer_input=boundary_target,
    )
    assert boundary.status == "REFRESH"
    empirical = _certificate(authority="empirical_local_estimate")
    proxy = _check(empirical)
    assert proxy.status == "PROXY_REUSE"
    assert not proxy.descent_authoritative
    assert not proxy.reuses_costate


def test_scalar_only_membership_is_backward_compatible_but_blocked() -> None:
    certificate = _certificate()
    decision = check_direct_costate_certificate(
        input_displacement=0.0,
        certificate=certificate,
        current_anchor_source_sha256=ANCHOR,
        current_correction_source_sha256=CORRECTION,
        current_geometry_artifact_sha256=GEOMETRY,
        current_norm_artifact_sha256=NORM,
    )
    assert decision.status == "BLOCKED"
    assert "scalar-only" in decision.reason


def test_wrong_direction_target_and_anchor_costate_fail_closed() -> None:
    certificate = _certificate()
    assert _check(
        certificate,
        displacement_vector=np.array([0.0, 0.01]),
    ).status == "BLOCKED"
    wrong_anchor_costate = ANCHOR_COSTATE.copy()
    wrong_anchor_costate[0] += 1.0
    assert _check(certificate, anchor_costate=wrong_anchor_costate).status == "BLOCKED"
    wrong_target = TARGET_INPUT.copy()
    wrong_target[0] += 0.01
    assert _check(certificate, target_scorer_input=wrong_target).status == "BLOCKED"


def test_reuse_decision_binds_all_live_tensor_and_certificate_hashes() -> None:
    certificate = _certificate()
    decision = _check(certificate)
    assert decision.status == "CERTIFIED_REUSE"
    assert decision.target_scorer_input_sha256 == array_sha256(TARGET_INPUT)
    assert decision.displacement_vector_sha256 == array_sha256(DISPLACEMENT)
    assert decision.correction_tensor_sha256 == array_sha256(CORRECTION_TENSOR)
    assert decision.anchor_costate_sha256 == ANCHOR_COSTATE_SHA
    assert decision.corrected_costate_sha256 == array_sha256(
        ANCHOR_COSTATE + CORRECTION_TENSOR
    )
    assert not decision.correction_derivation_authoritative
    assert decision.correction_derivation_receipt_sha256 is None
    assert decision.correction_derivation_status == "MISSING_INTEGRATED_HVP_DERIVATION_RECEIPT"


@pytest.mark.parametrize(
    ("coefficients", "expected_radius"),
    [
        (
            {
                "anchor_jacobian_norm_upper": 1.0,
                "anchor_jacobian_derivative_norm_upper": 0.0,
                "jacobian_derivative_lipschitz_upper": 0.0,
                "anchor_adjoint_norm_upper": 0.0,
                "adjoint_lipschitz_upper": 1.0,
                "correction_error_lipschitz_upper": 0.0,
            },
            0.25,
        ),
        (
            {
                "anchor_jacobian_norm_upper": 0.0,
                "anchor_jacobian_derivative_norm_upper": 0.0,
                "jacobian_derivative_lipschitz_upper": 2.0,
                "anchor_adjoint_norm_upper": 0.0,
                "adjoint_lipschitz_upper": 1.0,
                "correction_error_lipschitz_upper": 0.0,
            },
            0.25 ** (1.0 / 3.0),
        ),
    ],
)
def test_strict_radius_handles_linear_and_cubic_degeneracies(
    coefficients: dict[str, float], expected_radius: float
) -> None:
    certificate = derive_direct_costate_certificate(
        **coefficients,
        renderer_vjp_norm_upper=1.0,
        projected_gradient_floor=0.25,
        geometry_radius=1.0,
        calibration_radius_cap=1.0,
        authority="rigorous_upper_bound",
        anchor_source_sha256=ANCHOR,
        anchor_costate_sha256=ANCHOR_COSTATE_SHA,
        correction_source_sha256=CORRECTION,
        geometry_authority="rigorous_upper_bound",
        geometry_artifact_sha256=GEOMETRY,
        norm_artifact_sha256=NORM,
        bound_artifact_sha256=BOUND,
        full_segnet_c21_cell_verified=True,
        norm_coercivity_verified=True,
        correction_numerical_bound_verified=True,
    )
    assert certificate.input_radius == pytest.approx(expected_radius)
    assert direct_costate_error_envelope(certificate.input_radius, **coefficients) < 0.25
    above = np.nextafter(certificate.input_radius, math.inf)
    assert direct_costate_error_envelope(above, **coefficients) >= 0.25


@pytest.mark.parametrize(
    "missing",
    ["jacobian_derivative_lipschitz_upper", "correction_error_lipschitz_upper"],
)
def test_rigorous_missing_bound_fails_closed(missing: str) -> None:
    kwargs = {
        "jacobian_derivative_lipschitz_upper": 1.0,
        "correction_error_lipschitz_upper": 0.0,
    }
    kwargs[missing] = None
    with pytest.raises(ValueError, match="C2,1 full-SegNet"):
        derive_direct_costate_certificate(
            anchor_jacobian_norm_upper=1.0,
            anchor_jacobian_derivative_norm_upper=1.0,
            anchor_adjoint_norm_upper=1.0,
            adjoint_lipschitz_upper=1.0,
            renderer_vjp_norm_upper=1.0,
            projected_gradient_floor=1.0,
            geometry_radius=1.0,
            calibration_radius_cap=1.0,
            authority="rigorous_upper_bound",
            anchor_source_sha256=ANCHOR,
            anchor_costate_sha256=ANCHOR_COSTATE_SHA,
            correction_source_sha256=CORRECTION,
            geometry_authority="rigorous_upper_bound",
            geometry_artifact_sha256=GEOMETRY,
            norm_artifact_sha256=NORM,
            bound_artifact_sha256=BOUND,
            full_segnet_c21_cell_verified=True,
            norm_coercivity_verified=True,
            correction_numerical_bound_verified=True,
            **kwargs,
        )


def test_empirical_missing_lip_dj_is_blocked_and_custody_mismatch_blocks() -> None:
    certificate = derive_direct_costate_certificate(
        anchor_jacobian_norm_upper=1.0,
        anchor_jacobian_derivative_norm_upper=1.0,
        anchor_adjoint_norm_upper=1.0,
        adjoint_lipschitz_upper=1.0,
        renderer_vjp_norm_upper=1.0,
        projected_gradient_floor=1.0,
        geometry_radius=1.0,
        calibration_radius_cap=1.0,
        authority="empirical_local_estimate",
        anchor_source_sha256=ANCHOR,
        anchor_costate_sha256=ANCHOR_COSTATE_SHA,
        correction_source_sha256=CORRECTION,
        geometry_authority="empirical_local_estimate",
        geometry_artifact_sha256=GEOMETRY,
        norm_artifact_sha256=NORM,
        calibration_receipt_sha256=CALIBRATION,
    )
    assert _check(certificate).status == "BLOCKED"
    assert _check(_certificate(), current_anchor_source_sha256=BOUND).status == "BLOCKED"


def test_fixed_adjoint_hvp_excludes_adjoint_drift() -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    logits = torch.stack((x.square(), x.pow(3))).reshape(2)
    q_anchor = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    correction = torch_fixed_adjoint_jacobian_hvp(
        logits=logits,
        scorer_input=x,
        anchor_adjoint=q_anchor,
        direction=direction,
    )
    expected = direction.detach() * (4.0 - 6.0 * x.detach())
    assert torch.allclose(correction, expected)
    assert q_anchor.grad is None
    assert direction.grad is None


def test_full_ce_hvp_minus_fixed_q_hvp_is_independent_adjoint_term() -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.2], dtype=torch.float64)
    logits = torch.stack((x.square(), -x)).reshape(1, 2)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    q = torch.autograd.grad(loss, logits, create_graph=True, retain_graph=True)[0]
    fixed = torch_fixed_adjoint_jacobian_hvp(
        logits=logits, scorer_input=x, anchor_adjoint=q, direction=direction
    )
    full_gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
    full = torch.autograd.grad(torch.sum(full_gradient * direction), x, retain_graph=True)[0]
    dq_direction = torch.autograd.grad(
        q, x, grad_outputs=torch.ones_like(q), retain_graph=True, allow_unused=True
    )[0]
    # Compute J^T Dq[h] directly with a JVP of q, rather than trusting the
    # full-Hessian subtraction as its own validation.
    _, dq_h = torch.autograd.functional.jvp(
        lambda value: torch.softmax(
            torch.stack((value.square(), -value)).reshape(1, 2), dim=1
        )
        - torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        x,
        direction,
    )
    independent = torch.autograd.grad(
        logits, x, grad_outputs=dq_h, retain_graph=True
    )[0]
    assert torch.allclose(full - fixed, independent)
    assert dq_direction is not None


def test_apply_correction_checks_shape() -> None:
    assert np.array_equal(
        apply_direct_costate_correction(anchor_costate=[1.0, 2.0], correction=[0.5, -0.5]),
        np.array([1.5, 1.5]),
    )
    with pytest.raises(ValueError, match="identical shapes"):
        apply_direct_costate_correction(anchor_costate=[1.0], correction=[1.0, 2.0])
