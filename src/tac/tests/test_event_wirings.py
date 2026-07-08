"""SENSOR->START EVENT WIRINGS verification harness (operator override 2026-07-08).

Covers the pure decision logic in ``tac.witness_control.event_wirings`` + the S4 R1 role
discriminator surfaces (``typed_config.ScheduleGovernance.role`` +
``tools.schedule_provenance_gate``). The trainer glue is thin (it computes the raw sensor
ingredients and calls these deciders); ALL the falsification-relevant logic — the event-vs-cap
gate, the REV-B positive control, the would_fire telemetry, the annulus plateau detector — is
proven here at $0.

means != ends: gates a MEANS. Only a byte-closed n600 exact row < 0.19110 moves the pointer.
"""
from __future__ import annotations

import json

import pytest

from tac.witness_control import event_wirings as ew
from tac.witness_control.event_wirings import (
    EventBackstopGate,
    annulus_plateau_event,
    ladder_arms_complete,
    lane_nucleus_event,
    lane_would_fire_row,
    muon_meat_event,
)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (A) EventBackstopGate — the event/backstop-cap primitive.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _muon_off(cap):
    return EventBackstopGate("muon", "--muon-start-epoch", "--muon-start-event", sensor=None, cap=cap)


def _muon_on(cap):
    return EventBackstopGate("muon", "--muon-start-epoch", "--muon-start-event",
                             sensor=ew.SENSOR_POWERLAW_MEAT, cap=cap)


def test_off_mode_reduces_to_fixed_cap_no_telemetry():
    """Event mode OFF (sensor None) == the incumbent ``ep >= cap`` gate EXACTLY, NO telemetry
    (the binding byte-identity contract)."""
    g = _muon_off(726)
    for ep in range(1, 726):
        s = g.update(ep)
        assert s.start_reached is False and s.just_fired is False and s.telemetry is None
    s = g.update(726)
    assert s.start_reached and s.just_fired and s.fired_by == "cap" and s.telemetry is None


def test_off_mode_cap_none_never_fires():
    """Event mode OFF + cap None (e.g. --muon-start-epoch unset) => the gate never fires (matches
    the incumbent ``args.muon_start_epoch is not None`` guard)."""
    g = _muon_off(None)
    for ep in (1, 100, 5000):
        assert g.update(ep).start_reached is False
    g0 = _muon_off(0)  # cap <= 0 is disabled/always-on-elsewhere, never a schedule transition
    assert g0.update(9999).start_reached is False


def test_off_mode_latches_after_cap():
    g = _muon_off(300)
    assert g.update(300).just_fired
    later = g.update(301)
    assert later.start_reached and later.just_fired is False and later.telemetry is None


def test_event_fires_and_latches():
    g = _muon_on(726)
    assert g.update(300, event_fired=False).start_reached is False
    s = g.update(400, event_fired=True)
    assert s.just_fired and s.fired_by == "event"
    assert s.telemetry is not None and s.telemetry["stage"] == "start_event_fired"
    assert s.telemetry["epoch"] == 400 and s.telemetry["sensor"] == ew.SENSOR_POWERLAW_MEAT
    # latched: a later cap-hit does NOT re-fire.
    later = g.update(726, event_fired=False)
    assert later.start_reached and later.just_fired is False and later.telemetry is None


def test_event_beats_cap_when_both_available():
    """If the sensor fires at exactly the cap epoch, it is credited to the EVENT, not the cap."""
    g = _muon_on(400)
    s = g.update(400, event_fired=True)
    assert s.fired_by == "event" and s.telemetry["stage"] == "start_event_fired"


def test_cap_backstop_fires_loud_when_event_never_fires():
    """The backstop cap fires ONLY if the sensor did not by the cap epoch, emitting the LOUD
    cap_fired_before_event row (falsification-relevant per S5)."""
    g = _muon_on(726)
    for ep in range(1, 726):
        assert g.update(ep, event_fired=False).just_fired is False
    s = g.update(726, event_fired=False)
    assert s.just_fired and s.fired_by == "cap"
    assert s.telemetry is not None and s.telemetry["stage"] == "cap_fired_before_event"
    assert s.telemetry["epoch"] == 726 and s.telemetry["cap"] == 726
    assert "FAIL-SAFE BACKSTOP FIRED" in s.telemetry["note"]


def test_event_mode_cap_none_only_event_can_fire():
    g = _muon_on(None)
    for ep in (100, 726, 5000):
        assert g.update(ep, event_fired=False).start_reached is False
    assert g.update(6000, event_fired=True).fired_by == "event"


def test_gate_telemetry_is_json_serializable():
    for maker, evf in ((_muon_on(10), True), (_muon_on(10), False)):
        maker.update(10, event_fired=evf)  # forces event or cap fire at ep 10
    g = _muon_on(5)
    s = g.update(5, event_fired=False)  # cap fire
    json.dumps(s.telemetry)  # must not raise
    g2 = _muon_on(999)
    s2 = g2.update(3, event_fired=True)  # event fire
    json.dumps(s2.telemetry)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (B) muon <- powerlaw_meat + S2 REV-B nucleation-complete positive control.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_muon_meat_too_few_points_not_fired():
    """Fail-safe: too few points => powerlaw_meat NOT exhausted => not fired (never declare
    exhaustion on a bad measurement)."""
    ev = muon_meat_event([(1, 0.01), (2, 0.009)], nucleation_complete=True)
    assert ev["fired"] is False and ev["meat_exhausted"] is False


def test_muon_meat_exhausted_and_nucleated_fires():
    traj = [(i, 0.001) for i in range(1, 40)]  # long flat tail => exhausted
    ev = muon_meat_event(traj, nucleation_complete=True)
    assert ev["meat_exhausted"] is True and ev["fired"] is True


def test_muon_rev_b_positive_control_holds_on_incomplete_nucleation():
    """S2 REV-B: even when the tau-descent meat is exhausted, an incomplete nucleation HOLDS the
    Muon entry (an island-birth transient must not be read as first-order exhaustion)."""
    traj = [(i, 0.001) for i in range(1, 40)]
    ev = muon_meat_event(traj, nucleation_complete=False)
    assert ev["meat_exhausted"] is True and ev["nucleation_complete"] is False
    assert ev["fired"] is False and "REV-B" in ev["reason"]


def test_ladder_arms_complete_positive_control():
    assert ladder_arms_complete(5, []) is True          # LADDER off => vacuously complete
    assert ladder_arms_complete(300, [260, 320]) is False  # movable arm (320) not done
    assert ladder_arms_complete(320, [260, 320]) is True   # both arms past their window
    assert ladder_arms_complete(319, [260, 320]) is False


def test_muon_gate_wired_end_to_end_with_rev_b():
    """The gate + REV-B compose: exhausted-but-not-nucleated HOLDS; nucleated fires; if neither by
    the cap, the LOUD backstop fires."""
    traj = [(i, 0.001) for i in range(1, 40)]
    g = _muon_on(726)
    # ep 200: exhausted but arms not done (window 320) => HELD
    ev = muon_meat_event(traj, nucleation_complete=ladder_arms_complete(200, [260, 320]))
    assert g.update(200, event_fired=ev["fired"]).start_reached is False
    # ep 320: arms done => fires on event
    ev2 = muon_meat_event(traj, nucleation_complete=ladder_arms_complete(320, [260, 320]))
    assert g.update(320, event_fired=ev2["fired"]).fired_by == "event"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (C) lane-band <- lane-class critical nucleus + S3 would_fire telemetry.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_lane_nucleus_not_born_not_fired():
    ev = lane_nucleus_event(0.0, 0.1, within_flip_thresh=0.5)
    assert ev["born"] is False and ev["fired"] is False


def test_lane_nucleus_born_but_not_formed_not_fired():
    ev = lane_nucleus_event(0.004, 0.9, within_flip_thresh=0.5)
    assert ev["born"] is True and ev["formed"] is False and ev["fired"] is False


def test_lane_nucleus_born_and_formed_fires():
    ev = lane_nucleus_event(0.005, 0.3, within_flip_thresh=0.5)
    assert ev["born"] and ev["formed"] and ev["fired"] is True


def test_lane_would_fire_row_emits_regardless_of_event_mode():
    """S3: the would_fire row carries the sensor verdict whether or not event mode is on (so
    calibration data accrues even under cap operation)."""
    ev = lane_nucleus_event(0.005, 0.3, within_flip_thresh=0.5)
    for mode in (True, False):
        row = lane_would_fire_row(400, ev, event_mode=mode)
        assert row["stage"] == "lane_band_would_fire" and row["epoch"] == 400
        assert row["event_mode"] is mode and row["would_fire"] is True
        json.dumps(row)  # JSON-safe telemetry


def test_lane_band_gate_fires_on_nucleus_event():
    g = EventBackstopGate("lane_band", "--lane-band-start-epoch", "--lane-band-start-event",
                          sensor=ew.SENSOR_LANE_NUCLEUS, cap=500)
    assert g.update(300, event_fired=lane_nucleus_event(0.0, 0.1, within_flip_thresh=0.5)["fired"]
                    ).start_reached is False
    s = g.update(420, event_fired=lane_nucleus_event(0.005, 0.2, within_flip_thresh=0.5)["fired"])
    assert s.just_fired and s.fired_by == "event"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (D) seg-chroma <- annulus_frac plateau detector.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _series(vals, step=50, start=200):
    return [(start + step * i, v) for i, v in enumerate(vals)]


def test_annulus_plateau_flat_and_dwelled_fires():
    # 4-point window spanning 150 epochs (>= min_epochs 150), perfectly flat => plateau.
    s = annulus_plateau_event(_series([0.05, 0.05, 0.05, 0.05]))
    assert s["fired"] is True and s["dwell_ok"] is True and s["plateau_ok"] is True


def test_annulus_plateau_descending_not_fired():
    s = annulus_plateau_event(_series([0.08, 0.065, 0.05, 0.035]))
    assert s["fired"] is False and s["plateau_ok"] is False


def test_annulus_plateau_dwell_too_short_not_fired():
    # flat but the 4-point window spans only 30 epochs < min_epochs 150.
    s = annulus_plateau_event(_series([0.05, 0.05, 0.05, 0.05], step=10))
    assert s["fired"] is False and s["dwell_ok"] is False


def test_annulus_plateau_too_few_points_fail_safe():
    assert annulus_plateau_event([(1, 0.05)])["fired"] is False
    assert annulus_plateau_event([])["fired"] is False


def test_annulus_plateau_detector_params_are_tagged_constants():
    """req-T: the detector params are named TAGGED module constants, not bare literals."""
    assert ew.ANNULUS_PLATEAU_REL_EPS == 1e-4
    assert ew.ANNULUS_PLATEAU_DWELL_WINDOWS == 4
    assert ew.ANNULUS_PLATEAU_MIN_EPOCHS == 150


def test_chroma_gate_fires_on_plateau_event():
    g = EventBackstopGate("seg_chroma_boundary", "--seg-chroma-boundary-start-epoch",
                          "--seg-chroma-boundary-start-event", sensor=ew.SENSOR_ANNULUS_PLATEAU, cap=450)
    desc = annulus_plateau_event(_series([0.08, 0.065, 0.05, 0.035]))
    assert g.update(350, event_fired=desc["fired"]).start_reached is False
    flat = annulus_plateau_event(_series([0.05, 0.05, 0.05, 0.05]))
    assert g.update(360, event_fired=flat["fired"]).fired_by == "event"


def test_recognised_start_event_sensors():
    assert ew.RECOGNISED_START_EVENT_SENSORS == frozenset(
        {ew.SENSOR_POWERLAW_MEAT, ew.SENSOR_LANE_NUCLEUS, ew.SENSOR_ANNULUS_PLATEAU})


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (E) S4 R1 role discriminator — typed_config + schedule_provenance_gate.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_typed_config_role_defaults_from_class():
    from tac.witness_dsl.typed_config import GovernanceRole, ScheduleGovernance
    cap = ScheduleGovernance(**{"class": "cap", "sensor": "--muon-start-event", "rationale": "x" * 12})
    ev = ScheduleGovernance(**{"class": "event", "sensor": "--muon-start-event", "rationale": "x" * 12})
    assert cap.effective_role is GovernanceRole.BACKSTOPS
    assert ev.effective_role is GovernanceRole.FIRES


def test_typed_config_role_mismatch_rejected():
    from tac.witness_dsl.typed_config import ScheduleGovernance
    with pytest.raises(ValueError, match="role=fires requires class=event"):
        ScheduleGovernance(**{"class": "cap", "role": "fires",
                              "sensor": "--muon-start-event", "rationale": "x" * 12})
    with pytest.raises(ValueError, match="role=backstops requires class=cap"):
        ScheduleGovernance(**{"class": "event", "role": "backstops",
                              "sensor": "--muon-start-event", "rationale": "x" * 12})


def test_gate_event_start_flags_parses_trainer_argparse():
    from pathlib import Path

    import tools.schedule_provenance_gate as gate
    from tac.witness_dsl.curriculum_dsl import TRAINER_REL
    ereg = gate.event_start_flags(Path(TRAINER_REL).read_text())
    assert {"--muon-start-event", "--lane-band-start-event",
            "--seg-chroma-boundary-start-event"} <= ereg


def test_gate_validate_governance_entry_role_agreement():
    import tools.schedule_provenance_gate as gate
    emitted = {"--muon-start-event", "--muon-start-epoch"}
    ok, _ = gate.validate_governance_entry(
        "--muon-start-epoch",
        {"class": "cap", "role": "backstops", "sensor": "--muon-start-event",
         "rationale": "req-B backstop for the powerlaw_meat event"}, emitted)
    assert ok
    bad, detail = gate.validate_governance_entry(
        "--muon-start-epoch",
        {"class": "cap", "role": "fires", "sensor": "--muon-start-event",
         "rationale": "req-B backstop for the powerlaw_meat event"}, emitted)
    assert bad is False and "role=backstops" in detail


def test_gate_classify_launch_event_registry_surfaces_event():
    import tools.schedule_provenance_gate as gate
    pairs = [("--muon-start-event", "powerlaw_meat"), ("--muon-start-epoch", "726")]
    gov = {
        "--muon-start-event": {"class": "event", "role": "fires", "sensor": "--muon-start-event",
                               "rationale": "fires Muon on powerlaw_meat exhaustion"},
        "--muon-start-epoch": {"class": "cap", "role": "backstops", "sensor": "--muon-start-event",
                               "rationale": "req-B backstop for the powerlaw_meat event"},
    }
    verdicts = gate.classify_launch(
        pairs, registry=frozenset({"--muon-start-epoch"}), manifest_keys=set(), governance=gov,
        event_registry=frozenset({"--muon-start-event"}))
    by = {v.flag: v.cls for v in verdicts}
    assert by["--muon-start-event"] == gate.CLASS_EVENT
    assert by["--muon-start-epoch"] == gate.CLASS_CAP
