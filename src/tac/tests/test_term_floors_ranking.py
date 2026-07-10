"""Tests for P8 floor-aware duty-to-measure ranking (design_philosophies_eightfold_20260709).

Covers: the term-floor resolver value-provenance ladder; apply_term_floor semantics (AT_FLOOR /
HEADROOM_CAPPED / ABOVE_FLOOR / FLOOR_KNOWN_CURRENT_UNKNOWN / FLOOR_UNMEASURED); and the
backward-compatibility invariant (no term_current -> rel_sig + ordering UNCHANGED, only metadata added).
"""
from __future__ import annotations

import json

import pytest

from tac.witness_dsl import activation_ledger as al
from tac.witness_dsl import term_floors as tf


# ─────────────────────────── the resolver / FloorSpec ───────────────────────────
def test_resolver_returns_measured_dseg_loose_rate_owed_dpose():
    floors = tf.resolve_term_floors()
    assert floors["d_seg"].label == tf.FLOOR_MEASURED
    assert floors["d_seg"].value == pytest.approx(0.00087)
    assert floors["d_seg"].usable_for_capping is True
    # rate floor is LOOSE by its own source -> surfaced but never caps
    assert floors["rate"].label == tf.FLOOR_LOOSE
    assert floors["rate"].usable_for_capping is False
    # d_pose floor is an owed measurement (None value, OWED_UNMEASURED)
    assert floors["d_pose"].value is None
    assert floors["d_pose"].usable_for_capping is False
    for f in floors.values():
        assert f.provenance  # NO-FAKE: every floor cites its source


def test_floorspec_measured_requires_value():
    with pytest.raises(ValueError):
        tf.FloorSpec(axis="d_seg", value=None, label=tf.FLOOR_MEASURED, provenance="x")
    with pytest.raises(ValueError):
        tf.FloorSpec(axis="d_seg", value=0.001, label=tf.FLOOR_MEASURED, provenance="")  # no provenance


# ─────────────────────────── apply_term_floor semantics ───────────────────────────
_MFLOOR = tf.FloorSpec(axis="d_seg", value=0.001, label=tf.FLOOR_MEASURED, provenance="test")


def test_apply_at_floor_when_current_at_or_below_floor():
    app = tf.apply_term_floor("d_seg", 0.03, 0.001, _MFLOOR)  # current == floor
    assert app.floor_status == tf.FLOOR_STATUS_AT_FLOOR
    assert app.capped_est == 0.0
    app2 = tf.apply_term_floor("d_seg", 0.03, 0.0005, _MFLOOR)  # current below floor
    assert app2.floor_status == tf.FLOOR_STATUS_AT_FLOOR


def test_apply_headroom_cap_when_est_exceeds_distance_to_floor():
    # current 0.002, floor 0.001 -> headroom = 100*(0.002-0.001) = 0.1 S; est 0.3 > 0.1 -> capped to 0.1
    app = tf.apply_term_floor("d_seg", 0.3, 0.002, _MFLOOR)
    assert app.floor_status == tf.FLOOR_STATUS_CAPPED
    assert app.capped_est == pytest.approx(0.1)
    assert app.headroom_s == pytest.approx(0.1)


def test_apply_above_floor_est_within_headroom():
    app = tf.apply_term_floor("d_seg", 0.03, 0.006, _MFLOOR)  # headroom 100*(0.006-0.001)=0.5 > 0.03
    assert app.floor_status == tf.FLOOR_STATUS_ABOVE_FLOOR
    assert app.capped_est == pytest.approx(0.03)


def test_apply_current_unknown_passes_through():
    app = tf.apply_term_floor("d_seg", 0.03, None, _MFLOOR)
    assert app.floor_status == tf.FLOOR_STATUS_CURRENT_UNKNOWN
    assert app.capped_est == 0.03  # unchanged
    assert app.floor is _MFLOOR    # floor still surfaced


def test_apply_unusable_floor_passes_through():
    loose = tf.FloorSpec(axis="rate", value=0.118, label=tf.FLOOR_LOOSE, provenance="loose")
    app = tf.apply_term_floor("rate", 0.05, 0.118, loose)  # LOOSE floor never caps even if current==floor
    assert app.floor_status == tf.FLOOR_STATUS_UNMEASURED
    assert app.capped_est == 0.05


def test_apply_dpose_nonlinear_headroom():
    # S_pose = sqrt(10*d_pose); floor 0.0001, current 0.01 -> headroom = sqrt(0.1)-sqrt(0.001)
    import math
    f = tf.FloorSpec(axis="d_pose", value=0.0001, label=tf.FLOOR_MEASURED, provenance="t")
    app = tf.apply_term_floor("d_pose", 100.0, 0.01, f)  # est huge -> capped to headroom
    exp = math.sqrt(0.1) - math.sqrt(0.001)
    assert app.headroom_s == pytest.approx(exp)
    assert app.capped_est == pytest.approx(exp)


# ─────────────────────────── ranking integration ───────────────────────────
@pytest.fixture()
def sig(tmp_path):
    return tmp_path / "sig.jsonl"


@pytest.fixture()
def led(tmp_path):
    return tmp_path / "led.jsonl"


@pytest.fixture()
def pointer(tmp_path):
    p = tmp_path / "ptr.json"
    p.write_text(json.dumps({"our_local_frontier_contest_cpu": {"score": 0.19110}}))
    return p


def test_ranking_backward_compat_no_term_current_unchanged(sig, led, pointer):
    """The core backward-compat invariant: with NO term_current, rel_sig + order are UNCHANGED;
    rows only GAIN floor metadata (floor_status=FLOOR_KNOWN_CURRENT_UNKNOWN for the measured d_seg floor)."""
    al.record_relative_significance("seg_big", 0.03, label="ESTIMATED", source_anchor="s",
                                    axis="d_seg", path=sig)
    al.record_relative_significance("seg_small", 0.005, label="ESTIMATED", source_anchor="s",
                                    axis="d_seg", path=sig)
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig, pointer_path=pointer)
    by = {r["lever"]: r for r in rows}
    # unchanged rel_sig (raw est / gap-to-goal)
    assert by["seg_big"]["rel_sig"] == pytest.approx(0.03 / (0.19110 - 0.15))
    assert by["seg_big"]["est_floor_capped"] == 0.03
    # measured d_seg floor surfaced, but current unknown -> no cap
    assert by["seg_big"]["floor_status"] == tf.FLOOR_STATUS_CURRENT_UNKNOWN
    assert by["seg_big"]["floor"]["value"] == pytest.approx(0.00087)
    # order unchanged (big before small)
    order = [r["lever"] for r in rows]
    assert order.index("seg_big") < order.index("seg_small")


def test_ranking_at_floor_deranks_lever(sig, led, pointer):
    """A d_seg lever whose target term is AT its floor ranks ~0 (rel_sig 0) vs a rate lever above floor."""
    al.record_relative_significance("seg_atfloor", 0.03, label="ESTIMATED", source_anchor="s",
                                    axis="d_seg", path=sig)
    al.record_relative_significance("rate_ok", 0.001, label="ESTIMATED", source_anchor="s",
                                    axis="rate", path=sig)
    # current d_seg AT the measured floor 0.00087 -> AT_FLOOR
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig, pointer_path=pointer,
                                     term_current={"d_seg": 0.00087})
    by = {r["lever"]: r for r in rows}
    assert by["seg_atfloor"]["floor_status"] == tf.FLOOR_STATUS_AT_FLOOR
    assert by["seg_atfloor"]["rel_sig"] == 0.0
    assert by["seg_atfloor"]["est_floor_capped"] == 0.0
    # the at-floor lever now ranks BELOW the (tiny) rate lever which is above its (non-capping) floor
    order = [r["lever"] for r in rows]
    assert order.index("rate_ok") < order.index("seg_atfloor")


def test_ranking_headroom_cap_changes_relsig(sig, led, pointer):
    """A d_seg lever claiming more ΔS than the distance-to-floor is capped -> rel_sig drops (NO-FAKE:
    a lever cannot buy more d_seg than exists above the measured floor)."""
    al.record_relative_significance("greedy", 0.5, label="ESTIMATED", source_anchor="s",
                                    axis="d_seg", path=sig)
    # current d_seg 0.002 -> headroom to 0.00087 = 100*(0.002-0.00087) = 0.113 S; est 0.5 capped to 0.113
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig, pointer_path=pointer,
                                     term_current={"d_seg": 0.002})
    g = next(r for r in rows if r["lever"] == "greedy")
    assert g["floor_status"] == tf.FLOOR_STATUS_CAPPED
    assert g["est_delta_s"] == 0.5  # raw estimate preserved
    assert g["est_floor_capped"] == pytest.approx(0.113)
    assert g["rel_sig"] == pytest.approx(0.113 / (0.19110 - 0.15))


def test_ranking_floor_aware_false_adds_no_floor_metadata(sig, led, pointer):
    al.record_relative_significance("x", 0.01, label="ESTIMATED", source_anchor="s", axis="d_seg", path=sig)
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig, pointer_path=pointer,
                                     floor_aware=False)
    assert rows[0]["floor_status"] is None
    assert rows[0]["floor"] is None
