from __future__ import annotations

import hashlib

import numpy as np

from tac.scorer_surrogate.costate_trust_region import (
    CostateReuseDecision,
    array_sha256,
    check_direct_costate_certificate,
    derive_direct_costate_certificate,
)
from tac.witness_dsl.costate_trust_region_policy import (
    CostateTrustRegionPolicy,
    DirectFullCostatePolicy,
)

BOUND = hashlib.sha256(b"bound").hexdigest()
CALIBRATION = hashlib.sha256(b"calibration").hexdigest()
ANCHOR_INPUT = np.zeros(2, dtype=np.float64)
TARGET_INPUT = np.array([0.01, 0.0], dtype=np.float64)
DIRECTION = TARGET_INPUT - ANCHOR_INPUT
ANCHOR_COSTATE = np.array([0.5, -0.25], dtype=np.float64)
CORRECTION_TENSOR = np.array([0.01, -0.01], dtype=np.float64)
ANCHOR_INPUT_SHA = array_sha256(ANCHOR_INPUT)
ANCHOR_COSTATE_SHA = array_sha256(ANCHOR_COSTATE)
CORRECTION_SOURCE = hashlib.sha256(b"correction-source").hexdigest()
GEOMETRY = hashlib.sha256(b"geometry").hexdigest()
NORM = hashlib.sha256(b"norm").hexdigest()


def _decision(status: str, *, authority: bool) -> CostateReuseDecision:
    return CostateReuseDecision(
        status=status,
        input_displacement=0.0,
        input_radius=1.0,
        predicted_feature_displacement_upper=0.0,
        predicted_minimum_margin=1.0,
        predicted_fisher_proxy_upper=0.1,
        label_cell_authoritative=authority,
        descent_authoritative=authority,
        reason="test",
    )


def _direct_certificate(*, cap: float = 1.0, rigorous: bool = False):
    authority = "rigorous_upper_bound" if rigorous else "empirical_local_estimate"
    custody = (
        {"bound_artifact_sha256": BOUND}
        if rigorous
        else {"calibration_receipt_sha256": CALIBRATION}
    )
    return derive_direct_costate_certificate(
        anchor_jacobian_norm_upper=1.0,
        anchor_jacobian_derivative_norm_upper=1.0,
        jacobian_derivative_lipschitz_upper=1.0,
        anchor_adjoint_norm_upper=1.0,
        adjoint_lipschitz_upper=0.1,
        correction_error_lipschitz_upper=0.1,
        renderer_vjp_norm_upper=1.0,
        projected_gradient_floor=1.0,
        geometry_radius=1.0,
        calibration_radius_cap=cap,
        authority=authority,
        anchor_source_sha256=ANCHOR_INPUT_SHA,
        anchor_costate_sha256=ANCHOR_COSTATE_SHA,
        correction_source_sha256=CORRECTION_SOURCE,
        geometry_authority=authority,
        geometry_artifact_sha256=GEOMETRY,
        norm_artifact_sha256=NORM,
        full_segnet_c21_cell_verified=rigorous,
        norm_coercivity_verified=rigorous,
        correction_numerical_bound_verified=rigorous,
        **custody,
    )


def _direct_decision(certificate):
    return check_direct_costate_certificate(
        displacement_vector=DIRECTION,
        anchor_scorer_input=ANCHOR_INPUT,
        target_scorer_input=TARGET_INPUT,
        anchor_costate=ANCHOR_COSTATE,
        correction=CORRECTION_TENSOR,
        certificate=certificate,
        current_anchor_source_sha256=ANCHOR_INPUT_SHA,
        current_correction_source_sha256=CORRECTION_SOURCE,
        current_geometry_artifact_sha256=GEOMETRY,
        current_norm_artifact_sha256=NORM,
    )


def _live_custody(decision):
    return {
        "current_target_scorer_input_sha256": decision.target_scorer_input_sha256,
        "current_displacement_vector_sha256": decision.displacement_vector_sha256,
        "current_correction_tensor_sha256": decision.correction_tensor_sha256,
        "current_corrected_costate_sha256": decision.corrected_costate_sha256,
        "current_anchor_costate_sha256": decision.anchor_costate_sha256,
    }


def test_rigorous_policy_reuses_only_dual_authority_decision() -> None:
    policy = CostateTrustRegionPolicy(mode="rigorous", rigorous_bound_artifact_sha256=BOUND)
    assert policy.select_action(_decision("CERTIFIED_REUSE", authority=True)) == "reuse_costate"
    assert policy.select_action(_decision("REFRESH", authority=False)) == "full_teacher_refresh"


def test_empirical_policy_cannot_launder_authority() -> None:
    policy = CostateTrustRegionPolicy(
        mode="empirical", empirical_calibration_receipt_sha256=CALIBRATION
    )
    assert policy.select_action(_decision("PROXY_REUSE", authority=False)) == "reuse_costate_advisory"
    assert policy.compile_measurement_contract()["live_trainer_argv"] == []


def test_direct_full_costate_policy_is_default_off_and_flag_free() -> None:
    policy = DirectFullCostatePolicy(
        empirical_calibration_receipt_sha256=CALIBRATION
    )
    assert not policy.enabled
    assert policy.select_action(_decision("PROXY_REUSE", authority=False)) == "full_teacher_refresh"
    contract = policy.compile_measurement_contract()
    assert contract["provider_mode"] == "direct_full_input_costate"
    assert contract["live_trainer_argv"] == []
    assert "gamma_theta/B_R" in contract["radius_control_law"]


def test_tensor_consistent_proxy_decision_still_refreshes_without_derivation_receipt() -> None:
    policy = DirectFullCostatePolicy(
        enabled=True,
        mode="empirical",
        empirical_calibration_receipt_sha256=CALIBRATION,
    )
    certificate = _direct_certificate()
    decision = _direct_decision(certificate)
    assert decision.status == "PROXY_REUSE"
    assert not decision.correction_derivation_authoritative
    assert policy.select_action(
        decision, certificate=certificate, **_live_custody(decision)
    ) == "full_teacher_refresh"


def test_tensor_consistent_certified_decision_still_refreshes_without_derivation_receipt() -> None:
    policy = DirectFullCostatePolicy(
        enabled=True,
        mode="rigorous",
        rigorous_bound_artifact_sha256=BOUND,
    )
    certificate = _direct_certificate(rigorous=True)
    decision = _direct_decision(certificate)
    assert decision.status == "CERTIFIED_REUSE"
    assert not decision.correction_derivation_authoritative
    assert policy.select_action(
        decision, certificate=certificate, **_live_custody(decision)
    ) == "full_teacher_refresh"


def test_direct_policy_blocks_stale_direction_correction_target_and_certificate() -> None:
    policy = DirectFullCostatePolicy(
        enabled=True,
        mode="empirical",
        empirical_calibration_receipt_sha256=CALIBRATION,
    )
    certificate = _direct_certificate()
    decision = _direct_decision(certificate)
    live = _live_custody(decision)
    for field in (
        "current_target_scorer_input_sha256",
        "current_displacement_vector_sha256",
        "current_correction_tensor_sha256",
        "current_corrected_costate_sha256",
        "current_anchor_costate_sha256",
    ):
        stale = dict(live)
        stale[field] = BOUND
        assert policy.select_action(
            decision, certificate=certificate, **stale
        ) == "full_teacher_refresh"
    mismatched_certificate = _direct_certificate(cap=0.5)
    assert policy.select_action(
        decision, certificate=mismatched_certificate, **live
    ) == "full_teacher_refresh"
