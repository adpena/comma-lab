from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.segnet_validation_certificate import (
    ProxyConfusionAccumulator,
    cadence_speedup,
    calibrate_empirical_pairwise_bounds,
    check_feature_trust_region,
    confusion_meter_canaries,
    derive_feature_trust_region,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_rigorous_linear_map_accepts_inside_and_refreshes_at_boundary() -> None:
    region = derive_feature_trust_region(
        anchor_margins=np.array([2.0, 4.0]), anchor_correct_mask=np.array([True, True]),
        pairwise_logit_change_bounds=np.array([2.0, 1.0]), authority="rigorous_upper_bound",
        anchor_feature_sha256=SHA_A, bound_artifact_sha256=SHA_B,
    )
    assert region.feature_radius == 1.0
    inside = check_feature_trust_region(anchor_feature=[0.0], current_feature=[0.999], region=region,
                                        current_anchor_feature_sha256=SHA_A)
    boundary = check_feature_trust_region(anchor_feature=[0.0], current_feature=[1.0], region=region,
                                          current_anchor_feature_sha256=SHA_A)
    assert inside.status == "ACCEPT" and inside.certificate_authoritative
    assert boundary.status == "REFRESH"


def test_empirical_calibration_can_only_proxy_accept_and_custody_blocks() -> None:
    bounds = calibrate_empirical_pairwise_bounds(
        anchor_pairwise_margins=[2.0, 3.0], candidate_pairwise_margins=[[1.5, 2.5], [1.0, 2.0]],
        feature_displacements_linf=[0.5, 1.0],
    )
    np.testing.assert_allclose(bounds, [1.0, 1.0])
    region = derive_feature_trust_region(
        anchor_margins=[2.0, 3.0], anchor_correct_mask=[True, True], pairwise_logit_change_bounds=bounds,
        authority="empirical_local_estimate", anchor_feature_sha256=SHA_A, calibration_receipt_sha256=SHA_B,
    )
    decision = check_feature_trust_region(anchor_feature=[0.0], current_feature=[0.25], region=region,
                                          current_anchor_feature_sha256=SHA_A)
    blocked = check_feature_trust_region(anchor_feature=[0.0], current_feature=[0.25], region=region,
                                         current_anchor_feature_sha256="c" * 64)
    assert decision.status == "PROXY_ACCEPT" and not decision.certificate_authoritative
    assert blocked.status == "BLOCKED"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"anchor_margins": [1.0], "anchor_correct_mask": [False], "pairwise_logit_change_bounds": [1.0]}, "no correctly"),
        ({"anchor_margins": [0.0], "anchor_correct_mask": [True], "pairwise_logit_change_bounds": [1.0]}, "strictly positive"),
        ({"anchor_margins": [1.0], "anchor_correct_mask": [True], "pairwise_logit_change_bounds": [np.nan]}, "finite"),
    ],
)
def test_invalid_geometry_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        derive_feature_trust_region(authority="rigorous_upper_bound", anchor_feature_sha256=SHA_A,
                                    bound_artifact_sha256=SHA_B, **kwargs)


def test_empirical_data_cannot_be_laundered_as_rigorous() -> None:
    with pytest.raises(ValueError, match="cannot be presented as rigorous"):
        derive_feature_trust_region(
            anchor_margins=[1.0], anchor_correct_mask=[True], pairwise_logit_change_bounds=[1.0],
            authority="rigorous_upper_bound", anchor_feature_sha256=SHA_A, bound_artifact_sha256=SHA_B,
            calibration_receipt_sha256="c" * 64,
        )


def test_confusion_meter_canaries_and_economics() -> None:
    assert confusion_meter_canaries()["status"] == "PASS"
    meter = ProxyConfusionAccumulator()
    meter.update(proxy_accepts=True, exact_ce_worsens=False, exact_dseg_worsens=True, exact_dpose_worsens=False)
    assert meter.false_negative == 1
    assert meter.unsafe_accepts_dseg == 1


def test_false_negative_is_dseg_only_not_joint_unsafe_accept() -> None:
    meter = ProxyConfusionAccumulator()
    meter.update(proxy_accepts=True, exact_ce_worsens=True, exact_dseg_worsens=False, exact_dpose_worsens=True)
    receipt = meter.to_dict()
    assert receipt["unsafe_accepts_any"] == 1
    assert receipt["false_negative"] == 0
    assert receipt["false_negative_dseg"] == 0
    assert cadence_speedup(cadence=4, t_exact=10.0, t_approx=1.0, t_validate_cheap=1.0, t_fallback=0.0) == 2.5
