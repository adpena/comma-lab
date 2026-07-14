"""Tests for the CURRICULUM POSITION + POSE-DESCENT READINESS truth models
(tools/dashboard_curriculum_panel.py).

Covers the operator-mandated truth-rendering fix: the curriculum is DERIVED from
the schedule read-back + parsed flags (no hardcoded PR95 epoch skeleton), event
states are honest (pending when no row), the provenance line is present, and the
R1 pose artifact is READ as an unselected advisory reference (never hardcoded or
substituted into current vehicle claims).
"""
from __future__ import annotations

import ast
import inspect
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dashboard_curriculum_panel as cp  # noqa: E402


def _readback_dict(*, tau_cap=300, muon_start=726, epochs=3000):
    """A synthetic ScheduleReadback.to_dict() — event tau + fixed Muon, like the
    live v7.5.2 config. Values are the TEST's, so a build that echoes them proves
    the model is readback-driven (not hardcoded)."""
    return {
        "ok": True, "source": "launch.sh", "epochs": epochs, "eval_every": 25,
        "event_triggered": True,
        "stages": [
            {"name": "CE", "start": 0, "mode": "fixed", "status": "scheduled"},
            {"name": "tau", "start": None, "mode": "event", "status": "pending",
             "cap": tau_cap, "fired_epoch": None,
             "trigger": "event-gated: loss-plateau OR hard-ceiling cap"},
            {"name": "Muon", "start": muon_start, "mode": "fixed", "status": "scheduled"},
        ],
    }


def _flags():
    return {
        "muon-start-event": "powerlaw_meat", "muon-start-epoch": "726",
        "lane-render-band": True, "lane-band-start-event": "lane_nucleus",
        "lane-band-start-epoch": "500",
        "seg-chroma-boundary-weight": "0.1",
        "seg-chroma-boundary-start-event": "annulus_plateau",
        "seg-chroma-boundary-start-epoch": "450",
        "seg-temporal-screw-weight": "0.1",
        "seg-temporal-screw-start-event": "annulus_plateau",
        "seg-temporal-screw-start-epoch": "450",
        "birth-completion-event": True, "ladder-island-homotopy": True,
        "tail-cycles-max": "2", "tail-start-epoch": "0",
        "polyak-finisher-arm": True, "polyak-finisher-start-epoch": "2546",
        "pose-finish-start-epoch": "726", "pose-finish-engage-on": "sigma_min_plateau",
        "tau-anneal-shape": "geometric", "softmax-temp-start": "1.0",
        "softmax-temp-end": "0.31", "tau-softplus-tau": "0.3",
    }


# ── 1) readback-driven rendering: stage caps ECHO the read-back, not a constant ──
def test_curriculum_model_is_readback_driven_not_hardcoded():
    m1 = cp.build_curriculum_panel_model(_readback_dict(tau_cap=300, muon_start=726), _flags(), {})
    m2 = cp.build_curriculum_panel_model(_readback_dict(tau_cap=411, muon_start=812), _flags(), {})
    tau1 = next(s for s in m1["stages"] if s["name"] == "tau")
    tau2 = next(s for s in m2["stages"] if s["name"] == "tau")
    assert tau1["cap"] == 300 and tau2["cap"] == 411, "tau cap must track the read-back"
    mu1 = next(s for s in m1["stages"] if s["name"] == "Muon")
    mu2 = next(s for s in m2["stages"] if s["name"] == "Muon")
    assert mu1["cap"] == 726 and mu2["cap"] == 812, "Muon cap must track the read-back"
    assert m1["epochs"] == 3000 and m1["event_triggered"] is True


# ── 2) value-provenance: NO hardcoded epoch literals in the panel-model source ──
def test_no_hardcoded_epoch_literals_in_panel_code():
    """The curriculum builder must not embed epoch positions (300/450/500/726/2546…).
    Epochs come from the read-back/flags. Scan the function AST for suspicious int
    constants (small structural ints like 0/1/2/10 are allowed)."""
    src = inspect.getsource(cp.build_curriculum_panel_model)
    tree = ast.parse(src)
    allowed = {0, 1, 2, 10}  # structural (indices, sqrt-10 lives elsewhere)
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, int)
           and n.value not in allowed]
    assert not bad, f"hardcoded epoch-like literals in panel code: {bad}"
    # the mechanism-lane specs are flag-keyed (caps read via cap_flag, never inline)
    for lane in cp.MECHANISM_LANES:
        assert "cap_flag" in lane, f"lane {lane.get('key')} must declare a cap_flag key"


# ── 3) event-state honest-pending path: no row => pending (no event row yet) ──
def test_event_state_honest_pending():
    m = cp.build_curriculum_panel_model(_readback_dict(), _flags(), {})  # empty event_states
    lanes = {l["key"]: l for l in m["lanes"]}
    assert lanes["lane_band"]["status"] == "pending"
    assert lanes["lane_band"]["note"] == "no event row yet"
    assert lanes["lane_band"]["at_epoch"] is None
    # tau stage stays pending when no transition fired
    tau = next(s for s in m["stages"] if s["name"] == "tau")
    assert tau["status"] == "pending" and tau["fired_epoch"] is None


# ── 3b) fired/armed state resolves from emitted evidence ──
def test_event_state_fired_and_armed_from_evidence():
    ev = {
        "lane_render_band_engage": {"epoch": 512},   # lane-band fired
        "pose_finish_armed": {"epoch": 700},         # terminal-solve armed (not fired)
        "muon_finisher_switch": {"epoch": 731},      # Muon fired
        "curriculum_transition_fired": {"epoch": 214},  # tau fired
    }
    m = cp.build_curriculum_panel_model(_readback_dict(), _flags(), ev)
    lanes = {l["key"]: l for l in m["lanes"]}
    assert lanes["lane_band"]["status"] == "fired" and lanes["lane_band"]["at_epoch"] == 512
    assert lanes["terminal_solve"]["status"] == "armed" and lanes["terminal_solve"]["at_epoch"] == 700
    mu = next(s for s in m["stages"] if s["name"] == "Muon")
    assert mu["status"] == "fired" and mu["fired_epoch"] == 731
    tau = next(s for s in m["stages"] if s["name"] == "tau")
    assert tau["status"] == "fired" and tau["fired_epoch"] == 214


# ── 4) provenance line present + answers the cargo-cult question ──
def test_provenance_line_present():
    m = cp.build_curriculum_panel_model(_readback_dict(), _flags(), {})
    assert m["provenance"] == cp.SCHEDULE_PROVENANCE_LINE
    assert "derived" in m["provenance"]
    assert "removed-by-measurement" in m["provenance"]
    # even a failed build carries the provenance line
    bad = cp.build_curriculum_panel_model(None, None, None)
    assert bad["provenance"] == cp.SCHEDULE_PROVENANCE_LINE


# ── 4b) Muon augmented with its event trigger; tau anneal is one continuous ramp ──
def test_muon_event_trigger_and_tau_anneal_continuous():
    m = cp.build_curriculum_panel_model(_readback_dict(), _flags(), {})
    mu = next(s for s in m["stages"] if s["name"] == "Muon")
    assert mu["mode"] == "event" and "powerlaw_meat" in (mu["trigger"] or "")
    ta = m["tau_anneal"]
    assert ta["shape"] == "geometric"
    assert ta["temp_start"] == 1.0 and abs(ta["temp_end"] - 0.31) < 1e-9
    assert "tau=1 limit" in ta["ce_limit_note"]
    assert ta["span_end_epoch"] == 726  # the finishing-optimizer boundary, from read-back


# ── 5) R1 is an unselected reference, never a current-config fallback claim ──
def test_pose_readiness_r1_reference_from_artifact(tmp_path):
    art = tmp_path / "r1.json"
    art.write_text(json.dumps({
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE", "promotion_claim": False,
        "n_pairs_total": 600,
        "byte_close": {"pose_carrier_counted_bytes": 7195},
        "parity_on_inflated_frames": {"d_pose_realized_on_inflated": 0.0016095471538913576,
                                      "pairs_scored": 600},
    }))
    pm = cp.build_pose_readiness_model(_flags(), {}, {}, r1_path=art)
    b = pm["r1_reference"]
    a = b["advisory_artifact"]
    assert b["ok"] is True
    assert b["status"] == "unselected_reference"
    assert b["label"] == "full-n600 byte-closed macOS-CPU advisory"
    assert b["payload_selected"] is False
    assert b["current_config_claims"] == {"d_pose": None, "archive_bytes": None}
    assert a["d_pose"] == 0.0016095471538913576
    assert a["counted_pose_bytes"] == 7195
    assert a["promotable"] is False
    assert "NON-PROMOTABLE" in a["source_axis"]
    # contribution is DERIVED inside the advisory artifact, never a current claim
    assert abs(a["pose_term"] - math.sqrt(10.0 * a["d_pose"])) < 1e-12


def test_pose_readiness_missing_artifact_is_honest():
    pm = cp.build_pose_readiness_model(_flags(), {}, {}, r1_path="/no/such/r1.json")
    assert pm["r1_reference"]["ok"] is False
    assert "reason" in pm["r1_reference"]
    assert pm["r1_reference"]["payload_selected"] is False
    # the contract still renders (reference card degrades, contract stands)
    assert pm["contract"]["detector_mode"] == "sigma_min_plateau"


# ── 6) pose contract: detector state + explicit nonselection ──
def test_pose_contract_state_and_decision_tree():
    # pending detector, no basin probe
    pm = cp.build_pose_readiness_model(_flags(), {}, {})
    c = pm["contract"]
    assert c["detector_mode"] == "sigma_min_plateau"
    assert c["detector_cap"] == 726  # from the flag, not hardcoded in logic
    assert c["detector_state"] == "pending"
    assert c["degenerate"] is False
    assert "pose_disengaged_no_banked_payload" in c["decision_tree"]
    assert c["payload_selected"] is False
    assert c["d_pose_claim"] is None
    assert c["archive_bytes_claim"] is None
    # DEGENERATE: basin probe present with sigma at/below floor
    sensors = {"jacobian_basin": {"median_sigma_min": 1e-5, "sigma_floor": 1e-4}}
    pm2 = cp.build_pose_readiness_model(_flags(), {}, sensors)
    assert pm2["contract"]["degenerate"] is True
    assert pm2["contract"]["basin_probe_present"] is True
    # FIRED: pose_finish_engage row present
    pm3 = cp.build_pose_readiness_model(_flags(), {"pose_finish_engage": {"epoch": 812}}, {})
    assert pm3["contract"]["detector_state"] == "fired"
    assert pm3["contract"]["detector_at_epoch"] == 812


def test_pose_disengagement_alarm_current_and_legacy_read_compatibility(tmp_path):
    log = tmp_path / "run.log"
    current = {
        "stage": "confound_alarm",
        "alarm": "pose_finish_disengaged_no_banked_payload",
        "epoch": 900,
        "pose_state": "pose_disengaged_no_banked_payload",
        "payload_selected": False,
        "d_pose_claim": None,
        "archive_bytes_claim": None,
    }
    log.write_text(json.dumps(current) + "\n")
    states = cp.read_mechanism_event_states([log])
    canonical = states["pose_finish_disengaged_no_banked_payload"]
    assert canonical["legacy_read_only"] is False
    model = cp.build_pose_readiness_model(_flags(), states, {})
    assert model["contract"]["detector_state"] == "disengaged"
    assert model["contract"]["pose_state"] == "pose_disengaged_no_banked_payload"
    assert model["contract"]["payload_selected"] is False

    # Old alarm bytes remain readable, but are normalized and cannot restore selection authority.
    legacy = {
        "stage": "confound_alarm",
        "alarm": "pose_finish_disengaged_shipped_banked_r1",
        "epoch": 901,
    }
    log.write_text(json.dumps(legacy) + "\n")
    legacy_states = cp.read_mechanism_event_states([log])
    parsed = legacy_states["pose_finish_disengaged_no_banked_payload"]
    assert parsed["legacy_read_only"] is True
    assert parsed["legacy_alarm_name"] == "pose_finish_disengaged_shipped_banked_r1"
    assert parsed["payload_selected"] is False


# ── 7) integration: the live v7.5.2 run dir (if present) builds a valid model ──
def test_integration_live_run_dir_if_present():
    import glob as _g
    dirs = sorted(_g.glob("experiments/results/levelset_v752_pilot_*/"))
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "launch.sh"))]
    if not dirs:
        return  # no live run dir checked out; unit tests above cover the logic
    rd = dirs[-1]
    import render_levelset_dashboard as rld
    from tac.witness_dsl.schedule_readback import read_schedule
    flags = rld._parse_launch_sh_flags(open(os.path.join(rd, "launch.sh")).read())
    rb = read_schedule(rd)
    ev = cp.read_mechanism_event_states(_g.glob(os.path.join(rd, "*.log")))
    m = cp.build_curriculum_panel_model(rb, flags, ev)
    assert m["ok"] is True
    names = [s["name"] for s in m["stages"]]
    assert "CE" in names and "tau" in names
    assert m["provenance"] == cp.SCHEDULE_PROVENANCE_LINE
    assert len(m["lanes"]) >= 5  # the enabled mechanism lanes render
