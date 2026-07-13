from __future__ import annotations

import pytest

from tac.boundary_math.segnet_validation_certificate import ValidationDecision, derive_feature_trust_region
from tac.witness_dsl.segnet_validation_certificate_policy import SegNetValidationCertificatePolicy


def test_rigorous_policy_requires_bound_and_emits_no_trainer_flag() -> None:
    sha = "a" * 64
    policy = SegNetValidationCertificatePolicy(mode="rigorous", rigorous_bound_artifact_sha256=sha)
    region = derive_feature_trust_region(
        anchor_margins=[1.0], anchor_correct_mask=[True], pairwise_logit_change_bounds=[1.0],
        authority="rigorous_upper_bound", anchor_feature_sha256="b" * 64, bound_artifact_sha256=sha,
    )
    policy.validate_region(region)
    decision = ValidationDecision("ACCEPT", 0.1, 1.0, True, "inside")
    assert policy.select_action(decision) == "reuse"
    assert policy.compile_measurement_contract()["live_trainer_argv"] == []
    assert policy.compile_measurement_contract()["research_only"] is True


def test_empirical_policy_is_advisory_and_rejection_refreshes() -> None:
    policy = SegNetValidationCertificatePolicy(mode="empirical", empirical_calibration_receipt_sha256="c" * 64)
    proxy = ValidationDecision("PROXY_ACCEPT", 0.1, 1.0, False, "inside")
    reject = ValidationDecision("REFRESH", 2.0, 1.0, False, "outside")
    assert policy.select_action(proxy) == "reuse_advisory"
    assert policy.select_action(reject) == "full_teacher_and_refresh"
    assert policy.compile_measurement_contract()["accept_authority"] == "advisory_proxy_only"


def test_policy_refuses_missing_or_cross_authority_artifacts() -> None:
    with pytest.raises(ValueError, match="requires only"):
        SegNetValidationCertificatePolicy(mode="rigorous")
    with pytest.raises(ValueError, match="requires only"):
        SegNetValidationCertificatePolicy(mode="empirical", empirical_calibration_receipt_sha256="a" * 64,
                                          rigorous_bound_artifact_sha256="b" * 64)
