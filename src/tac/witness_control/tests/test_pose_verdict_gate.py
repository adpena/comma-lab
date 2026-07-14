# SPDX-License-Identifier: MIT
"""Tests for the pose-verdict gate (skip PoseNet forward while pose is frozen)."""
from __future__ import annotations

from tac.witness_control.pose_verdict_gate import (
    BANKED_R1_DPOSE_DEFAULT,
    banked_pose_telemetry,
    canary_drift,
    decide_pose_verdict,
)


def _d(**kw):
    base = {"epoch": 10, "pose_engaged_epoch": 726, "verdict_index": 3,
            "gate_on": True, "canary_every": 8}
    base.update(kw)
    return decide_pose_verdict(**base)


def test_gate_off_always_live():
    d = _d(gate_on=False)
    assert d.compute_live and not d.is_canary and d.d_pose_source == "live"


def test_pose_engaged_always_live():
    # epoch past the engage point => pose descending => must be live.
    d = _d(epoch=800, pose_engaged_epoch=726)
    assert d.compute_live and d.d_pose_source == "live"


def test_engaged_epoch_boundary_is_live():
    d = _d(epoch=726, pose_engaged_epoch=726)
    assert d.compute_live


def test_pre_finish_ships_banked():
    d = _d(epoch=10, pose_engaged_epoch=726, verdict_index=3, canary_every=8)
    assert not d.compute_live
    assert d.d_pose_source == "banked_R1_pose_gated"
    assert "pre_finish" in d.reason


def test_canary_fires_on_cadence():
    d = _d(verdict_index=8, canary_every=8)  # 8 % 8 == 0
    assert d.compute_live and d.is_canary and d.d_pose_source == "live"


def test_index_zero_is_always_a_live_anchor():
    d = _d(verdict_index=0, canary_every=8)
    assert d.compute_live and d.is_canary  # first verdict never a bare banked constant


def test_never_engaged_pre_finish_ships_banked():
    d = _d(epoch=100, pose_engaged_epoch=-1, verdict_index=1, canary_every=8)
    assert not d.compute_live and d.d_pose_source == "banked_R1_pose_gated"


def test_canary_every_one_is_always_live():
    # K=1 => every verdict is a canary => full drift visibility, zero savings (safe default rollout).
    for i in range(5):
        assert _d(verdict_index=i, canary_every=1).compute_live


def test_banked_telemetry_is_labelled_nonlive():
    t = banked_pose_telemetry(BANKED_R1_DPOSE_DEFAULT, "pose_frozen_pre_finish")
    assert t["d_pose"] == BANKED_R1_DPOSE_DEFAULT
    assert t["d_pose_live"] is False
    assert t["d_pose_source"] == "banked_R1_pose_gated"
    assert "NON-LIVE" in t["d_pose_axis"] and "NON-PROMOTABLE" in t["d_pose_axis"]


def test_canary_drift_computation():
    r = canary_drift(0.00200, 0.00161)
    assert abs(r["abs_drift"] - 0.00039) < 1e-9
    assert r["d_pose_live"] == 0.00200 and r["d_pose_banked"] == 0.00161
    assert r["rel_drift"] > 0.0
