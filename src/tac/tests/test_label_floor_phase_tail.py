# SPDX-License-Identifier: MIT
"""Tests for the label-floor SENSE detector + the phase-tail schedule/DECIDE fold
(#247/#302/#303 phase-reframe, 2026-07-10 flicker memo). All advisory, $0, no run.
"""
from __future__ import annotations

from types import SimpleNamespace

from tac.canonical_equations.curriculum_derivation_laws_20260705 import (
    LABEL_FLOOR_DSEG,
    build_label_floor_to_phase_tail_handoff_v1,
    phase_tail_ready,
)
from tac.witness_control.label_floor_detector import (
    LABEL_FLOOR_REACHED,
    NO_FLOOR,
    ORACLE_LABEL_FLOOR_DSEG,
    _is_phase_tail_lever,
    label_floor_reached,
)


def _verdicts(stage: str, series: list[tuple[int, float]]) -> list[dict]:
    return [{"seg_form": stage, "epoch": e, "d_seg": d} for e, d in series]


# ---- SENSE detector -------------------------------------------------------------

def test_fires_at_flat_floor_in_label_stage():
    # d_seg sitting flat at ~0.0052 in an l7 (label-smooth) stage => phase regime
    v = _verdicts("l7", [(300, 0.00530), (320, 0.00528), (340, 0.00529), (360, 0.00527)])
    sig = label_floor_reached(v)
    assert sig.fired()
    assert sig.classification == LABEL_FLOOR_REACHED
    assert sig.phase_regime == LABEL_FLOOR_REACHED
    assert sig.within_floor_band and sig.flat and sig.stage_is_label_smooth


def test_no_fire_while_still_descending():
    # steep descent toward the floor is NOT the floor yet (not flat)
    v = _verdicts("l7", [(100, 0.020), (120, 0.014), (140, 0.009), (160, 0.0055)])
    sig = label_floor_reached(v)
    assert not sig.fired()
    assert sig.classification == NO_FLOOR
    assert not sig.flat


def test_no_fire_above_band():
    # flat but well above the floor band (e.g. mod32cap ~0.0047 is below; here 0.012 above)
    v = _verdicts("tau", [(200, 0.0120), (220, 0.0121), (240, 0.0119), (260, 0.0120)])
    sig = label_floor_reached(v)
    assert not sig.fired()
    assert not sig.within_floor_band


def test_no_fire_in_phase_stage():
    # already in the phase regime => the label floor is behind us
    v = _verdicts("phase", [(500, 0.0052), (520, 0.0051), (540, 0.0052), (560, 0.0051)])
    sig = label_floor_reached(v)
    assert not sig.fired()
    assert not sig.stage_is_label_smooth


def test_unidentifiable_too_few_rows():
    v = _verdicts("l7", [(300, 0.0053)])
    sig = label_floor_reached(v)
    assert not sig.fired()
    assert sig.classification == "LABEL_FLOOR_UNIDENTIFIABLE"


def test_oracle_floor_matches_equation():
    assert ORACLE_LABEL_FLOOR_DSEG == LABEL_FLOOR_DSEG == 0.005318


def test_phase_lever_matcher():
    assert _is_phase_tail_lever("phase_advection_consistency")
    assert _is_phase_tail_lever("--seg-spike-downweight")
    assert _is_phase_tail_lever("p0_force_subpix")
    assert not _is_phase_tail_lever("curriculum_nucleus_guard")
    assert not _is_phase_tail_lever("eikonal_viscosity")


def test_confound_alarm_row_is_advisory_never_halt():
    v = _verdicts("l7", [(300, 0.00530), (320, 0.00528), (340, 0.00529)])
    row = label_floor_reached(v).to_confound_alarm_row()
    assert row["stage"] == "regime_signal"
    assert row["alarm"] == "label_floor_reached"
    assert "advisory" in row["axis"].lower()


# ---- equation law (Law 5) -------------------------------------------------------

def test_phase_tail_ready_callable():
    # at floor, flat, label stage
    assert phase_tail_ready(rel_slope=1e-5, d_seg_latest=0.0053, stage_is_label_smooth=True)
    # descending (not flat)
    assert not phase_tail_ready(rel_slope=1e-2, d_seg_latest=0.0053, stage_is_label_smooth=True)
    # above band
    assert not phase_tail_ready(rel_slope=1e-5, d_seg_latest=0.02, stage_is_label_smooth=True)
    # phase stage (not label-smooth)
    assert not phase_tail_ready(rel_slope=1e-5, d_seg_latest=0.0053, stage_is_label_smooth=False)


def test_law5_builds_and_cites_flicker_floor():
    eq = build_label_floor_to_phase_tail_handoff_v1()
    assert eq.equation_id == "label_floor_to_phase_tail_handoff_v1"
    assert "tac.witness_control.label_floor_detector" in eq.canonical_consumers
    # the derived floor value is in the anchor's empirical output
    a = eq.empirical_anchors[0]
    assert "0.005318" in str(a.empirical_output)


# ---- schedule surface -----------------------------------------------------------

def test_describe_phase_tail_inactive_when_flags_absent():
    from tac.witness_dsl.schedule_readback import describe_phase_tail
    ns = SimpleNamespace(seg_phase_advect_weight=0.0, seg_spike_downweight=0.0)
    spec = describe_phase_tail(ns)
    assert spec["active"] is False
    assert spec["law"] == "label_floor_to_phase_tail_handoff_v1"
    assert spec["level"].startswith("finest-persistence")


def test_describe_phase_tail_active_when_declared():
    from tac.witness_dsl.schedule_readback import describe_phase_tail
    ns = SimpleNamespace(seg_phase_advect_weight=0.5, seg_spike_downweight=0.0)
    spec = describe_phase_tail(ns)
    assert spec["active"] is True


def test_phase_tail_stage_entry_gated_on_declaration():
    from tac.witness_dsl.schedule_readback import phase_tail_stage_entry
    # not declared => None (build owed, never fabricated)
    ns_off = SimpleNamespace(seg_phase_advect_weight=0.0, seg_spike_downweight=0.0)
    assert phase_tail_stage_entry(ns_off, l7_start=600) is None
    # declared => an event-gated terminal stage capped at l7
    ns_on = SimpleNamespace(seg_phase_advect_weight=0.5, seg_spike_downweight=0.25)
    st = phase_tail_stage_entry(ns_on, l7_start=600)
    assert st is not None and st.name == "phase" and st.mode == "event" and st.cap == 600


def test_phase_stage_has_terminal_precedence():
    from tac.witness_dsl.schedule_readback import _ORDER_IDX, STAGE_ORDER
    assert _ORDER_IDX["phase"] > _ORDER_IDX["Muon"]
    assert "phase" not in STAGE_ORDER  # iteration/transition logic stays 4-stage


# ---- DECIDE (controller two-move recommendation) --------------------------------

def test_decide_emits_two_move_phase_handoff():
    from pathlib import Path

    from tac.witness_control.shadow_controller import RunInputs, _recommendations
    inputs = RunInputs(run_dir=Path("."), verdicts=[], stage_rows={}, flags={})
    classification = {
        "classification": "plateau", "stage": "l7",
        "phase_regime": LABEL_FLOOR_REACHED,
        "label_floor": {"d_seg_latest": 0.0053, "oracle_floor": 0.005318, "stage": "l7"},
    }
    recs, refused = _recommendations(inputs, [], classification)
    actions = {r["action"] for r in recs}
    # both moves present, opposite-sign branches on the same locus
    assert "FIRE_T2_SPIKE_DOWNWEIGHT_ALEATORIC" in actions
    assert "ENGAGE_T1_PHASE_ADVECTION_PLUS_360_FORCES" in actions
    # the naive plateau early-stop is SUPPRESSED at the floor (would abandon sub-floor path)
    assert "ADVANCE_STAGE_OR_EARLY_STOP" not in actions
    assert "PHASE_HANDOFF_NOT_EARLY_STOP" in actions
    # advisory-only: predicted ΔS is 0.0 (no fabricated drop for a never-fired lever)
    for r in recs:
        assert r["predicted_dS"] == 0.0


def test_decide_no_phase_move_when_not_at_floor():
    from pathlib import Path

    from tac.witness_control.shadow_controller import RunInputs, _recommendations
    inputs = RunInputs(run_dir=Path("."), verdicts=[], stage_rows={}, flags={})
    classification = {"classification": "converging", "stage": "l7", "phase_regime": None}
    recs, _ = _recommendations(inputs, [], classification)
    actions = {r["action"] for r in recs}
    assert "FIRE_T2_SPIKE_DOWNWEIGHT_ALEATORIC" not in actions
    assert "ENGAGE_T1_PHASE_ADVECTION_PLUS_360_FORCES" not in actions


def test_duty_to_measure_boosts_phase_levers_when_active():
    from tac.witness_control.shadow_controller import _duty_to_measure
    # phase-active path appends the #336 producer row (regardless of ledger contents)
    rows = _duty_to_measure(phase_active=True)
    assert any("sensitivity_bit_alloc_phase_carrier" in str(r.get("lever", "")) for r in rows)
