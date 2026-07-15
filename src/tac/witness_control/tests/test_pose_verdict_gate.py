# SPDX-License-Identifier: MIT
"""Guards for the score-ineligible pose-blind compute gate."""
from __future__ import annotations

import pytest

from tac.witness_control.pose_verdict_gate import (
    BANKED_R1_DPOSE_DEFAULT,
    PoseVerdictGateError,
    banked_pose_telemetry,
    canary_drift,
    check_pose_verdict_fallback_is_live_or_refused,
    decide_pose_verdict,
)


def test_gate_off_always_requires_live_posenet() -> None:
    decision = decide_pose_verdict(
        epoch=10,
        pose_engaged_epoch=726,
        verdict_index=3,
        gate_on=False,
        canary_every=8,
    )
    assert decision.compute_live is True
    assert decision.d_pose_source == "live"
    assert decision.is_canary is False


def test_enabled_gate_skips_only_during_pose_blind_phase() -> None:
    blind = decide_pose_verdict(
        epoch=10,
        pose_engaged_epoch=-1,
        verdict_index=3,
        gate_on=True,
        canary_every=8,
    )
    assert blind.compute_live is False
    assert blind.d_pose_source == "missing_progress_only"
    engaged = decide_pose_verdict(
        epoch=726,
        pose_engaged_epoch=726,
        verdict_index=4,
        gate_on=True,
        canary_every=8,
    )
    assert engaged.compute_live is True
    assert engaged.d_pose_source == "live"


def test_preloop_baseline_stays_live_under_gate() -> None:
    decision = decide_pose_verdict(
        epoch=0,
        pose_engaged_epoch=-1,
        verdict_index=0,
        gate_on=True,
        canary_every=8,
        force_live=True,
    )
    assert decision.compute_live is True
    assert decision.reason == "forced_live_current_payload_anchor"


def test_numeric_nonlive_pose_is_refused() -> None:
    with pytest.raises(PoseVerdictGateError, match="numeric non-live d_pose"):
        check_pose_verdict_fallback_is_live_or_refused(
            gate_on=False, configured_nonlive_dpose=0.001610
        )


def test_legacy_waiver_is_read_only_and_cannot_enable_gate() -> None:
    receipt = check_pose_verdict_fallback_is_live_or_refused(
        gate_on=False,
        configured_nonlive_dpose=0.001610,
        legacy_read_only_waiver=True,
    )
    assert receipt["numeric_nonlive_dpose_admitted"] is False
    assert receipt["score_path_d_pose_source"] == "live_posenet"
    assert receipt["reference_authority"].endswith("unselected")
    with pytest.raises(PoseVerdictGateError, match="numeric non-live d_pose"):
        check_pose_verdict_fallback_is_live_or_refused(
            gate_on=True,
            configured_nonlive_dpose=0.001610,
            legacy_read_only_waiver=True,
        )


def test_legacy_import_surface_cannot_reenable_numeric_substitution() -> None:
    with pytest.raises(PoseVerdictGateError, match="live PoseNet is required"):
        banked_pose_telemetry(BANKED_R1_DPOSE_DEFAULT, "historical")
    diagnostic = canary_drift(0.002, BANKED_R1_DPOSE_DEFAULT)
    assert diagnostic["score_claim"] is False
    assert diagnostic["selection_eligible"] is False
