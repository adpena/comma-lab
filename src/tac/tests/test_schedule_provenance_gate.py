# SPDX-License-Identifier: MIT
"""Tests for the SCHEDULE-PROVENANCE enforcement (operator 2026-07-09, verbatim fury:
"Fuck pr95... Never do it again. Add a gate and hook... move from hardcoded epochs to event
based and deep math governed and costate controller").

Two layers:
  * the LAUNCHER-path STRICT gate (tools/schedule_provenance_gate.py) — every emitted
    --*-start-epoch schedule TRIGGER must be EVENT-governed / LawRef-DERIVED / a TAGGED
    fail-safe CAP, else NAKED (REFUSE);
  * the drift-detector leg (tools/triality_drift_detector.py) — flags AUTHORING a naked
    hardcoded-epoch schedule param / a NEW PR95-named stage sequence.

The incumbent crucible_v6 classification (5 NAKED triggers) is pinned as the regression
fixture = the restart config's to-fix spec.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import schedule_provenance_gate as spg  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(_REPO / rel))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


tdd = _load("triality_drift_detector_under_test", "tools/triality_drift_detector.py")

_TRAINER_TEXT = """
    ap.add_argument("--tau-softplus-start-epoch", type=int, default=300)
    ap.add_argument("--l7-start-epoch", type=int, default=800)
    ap.add_argument("--muon-start-epoch", type=int, default=None)
    ap.add_argument("--lane-band-start-epoch", type=int, default=300)
    ap.add_argument("--anneal-epochs", type=int, default=None)
    ap.add_argument("--curriculum-min-stage-epochs", type=int, default=150)
    ap.add_argument("--curriculum-event-triggered", action="store_true")
    ap.add_argument("--w-seg", type=float, default=1.0)
"""


# ───────────────────────────── registry (data-driven) ─────────────────────────────
def test_registry_parses_only_start_epoch_flags():
    reg = spg.schedule_when_flags(_TRAINER_TEXT)
    assert reg == frozenset({
        "--tau-softplus-start-epoch", "--l7-start-epoch",
        "--muon-start-epoch", "--lane-band-start-epoch"})
    # duration/dwell/toggle flags are NOT WHEN-triggers -> out of registry
    assert "--anneal-epochs" not in reg
    assert "--curriculum-min-stage-epochs" not in reg
    assert "--curriculum-event-triggered" not in reg
    assert "--w-seg" not in reg


def test_flag_to_manifest_key():
    assert spg.flag_to_manifest_key("--tau-softplus-start-epoch") == "tau_softplus_start_epoch"
    assert spg.flag_to_manifest_key("--muon-start-epoch") == "muon_start_epoch"


def test_parse_epoch_value_only_positive_ints():
    assert spg.parse_epoch_value(300) == 300
    assert spg.parse_epoch_value("726") == 726
    assert spg.parse_epoch_value(0) is None      # from-ep1 / disabled — not a trigger
    assert spg.parse_epoch_value(-1) is None     # disabled
    assert spg.parse_epoch_value(None) is None
    assert spg.parse_epoch_value("cosine") is None


def test_extra_flag_pairs_parsing():
    pairs = spg.extra_flag_pairs(["--foo", "3", "--bar=7", "--flag", "--baz", "9"])
    assert pairs == [("--foo", "3"), ("--bar", "7"), ("--flag", None), ("--baz", "9")]


# ───────────────────────────── per-token classification ─────────────────────────────
_REG = frozenset({"--tau-softplus-start-epoch", "--muon-start-epoch", "--lane-band-start-epoch"})


def test_derived_class_passes():
    v = spg.classify_token("--muon-start-epoch", 726, emitted_flags={"--muon-start-epoch"},
                           manifest_keys={"muon_start_epoch"}, governance={})
    assert v.cls == spg.CLASS_DERIVED and v.ok


def test_event_class_passes_with_co_emitted_sensor():
    gov = {"--tau-softplus-start-epoch": {"class": "event",
                                          "sensor": "--curriculum-event-triggered"}}
    v = spg.classify_token(
        "--tau-softplus-start-epoch", 300,
        emitted_flags={"--tau-softplus-start-epoch", "--curriculum-event-triggered"},
        manifest_keys=set(), governance=gov)
    assert v.cls == spg.CLASS_EVENT and v.ok


def test_cap_class_passes_with_sensor_and_rationale():
    gov = {"--muon-start-epoch": {"class": "cap", "sensor": "--plateau-trigger",
                                  "rationale": "Muon finisher backstop; the plateau event fires first"}}
    v = spg.classify_token(
        "--muon-start-epoch", 726,
        emitted_flags={"--muon-start-epoch", "--plateau-trigger"},
        manifest_keys=set(), governance=gov)
    assert v.cls == spg.CLASS_CAP and v.ok


def test_naked_epoch_refused_with_named_token():
    v = spg.classify_token("--tau-softplus-start-epoch", 300,
                           emitted_flags={"--tau-softplus-start-epoch"},
                           manifest_keys=set(), governance={})
    assert v.cls == spg.CLASS_NAKED and not v.ok
    assert "no LawRef" in v.detail and "no fail-safe-cap tag" in v.detail


def test_event_decl_with_sensor_not_co_emitted_is_naked():
    gov = {"--tau-softplus-start-epoch": {"class": "event",
                                          "sensor": "--curriculum-event-triggered"}}
    v = spg.classify_token("--tau-softplus-start-epoch", 300,
                           emitted_flags={"--tau-softplus-start-epoch"},  # sensor absent!
                           manifest_keys=set(), governance=gov)
    assert v.cls == spg.CLASS_NAKED and "NOT co-emitted" in v.detail


def test_cap_decl_missing_rationale_is_naked():
    gov = {"--muon-start-epoch": {"class": "cap", "sensor": "--plateau-trigger"}}
    v = spg.classify_token("--muon-start-epoch", 726,
                           emitted_flags={"--muon-start-epoch", "--plateau-trigger"},
                           manifest_keys=set(), governance=gov)
    assert v.cls == spg.CLASS_NAKED and "rationale" in v.detail


def test_cap_decl_placeholder_rationale_is_naked():
    gov = {"--muon-start-epoch": {"class": "cap", "sensor": "--plateau-trigger",
                                  "rationale": "<rationale>"}}
    v = spg.classify_token("--muon-start-epoch", 726,
                           emitted_flags={"--muon-start-epoch", "--plateau-trigger"},
                           manifest_keys=set(), governance=gov)
    assert v.cls == spg.CLASS_NAKED


def test_unrecognised_sensor_is_naked():
    gov = {"--muon-start-epoch": {"class": "event", "sensor": "--not-a-real-sensor"}}
    v = spg.classify_token("--muon-start-epoch", 726,
                           emitted_flags={"--muon-start-epoch", "--not-a-real-sensor"},
                           manifest_keys=set(), governance=gov)
    assert v.cls == spg.CLASS_NAKED and "not a recognised event sensor" in v.detail


# ───────────────────────────── launch-level classification ─────────────────────────────
def test_classify_launch_filters_registry_and_positive():
    pairs = [("--tau-softplus-start-epoch", 300),   # naked -> classified
             ("--lane-edge-start-epoch", 0),         # value 0 -> skipped (always-on)
             ("--w-seg", 1.0),                        # not a schedule flag -> skipped
             ("--muon-start-epoch", 726)]             # naked -> classified
    verds = spg.classify_launch(pairs, registry=_REG, manifest_keys=set(), governance={})
    assert {v.flag for v in verds} == {"--tau-softplus-start-epoch", "--muon-start-epoch"}
    assert all(not v.ok for v in verds)


def test_non_schedule_flags_never_classified():
    pairs = [("--w-seg", 1.0), ("--mod-dim", 32), ("--eikonal-weight", 0.05)]
    verds = spg.classify_launch(pairs, registry=_REG, manifest_keys=set(), governance={})
    assert verds == []


def test_gate_report_ok_when_all_governed():
    pairs = [("--muon-start-epoch", 726)]
    verds = spg.classify_launch(pairs, registry=_REG, manifest_keys={"muon_start_epoch"},
                                governance={})
    ok, viol, table = spg.gate_report(verds)
    assert ok and viol == [] and "OK" in table


def test_gate_report_reports_violations_and_table():
    pairs = [("--tau-softplus-start-epoch", 300)]
    verds = spg.classify_launch(pairs, registry=_REG, manifest_keys=set(), governance={})
    ok, viol, table = spg.gate_report(verds)
    assert not ok and len(viol) == 1
    assert "FAIL" in table and "to-fix spec" in table


# ───────── the INCUMBENT crucible_v6 fixture (= the restart config's to-fix spec) ─────────
def test_incumbent_crucible_v6_naked_epoch_set_is_the_restart_to_fix_spec():
    """REGRESSION FIXTURE: the live crucible_v6 emits exactly these 5 NAKED primary-epoch
    triggers. The restart must EVENT-govern / LawRef-derive / cap-tag each to pass the gate.
    If this set changes, the memo's incumbent-violations table + the restart spec must too."""
    wac = pytest.importorskip("tac.witness_autoconfig")
    trainer = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"
    cfg = wac.derive_crucible_v6_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz", num_pairs=600, epochs=3000)
    verds = spg.classify_launch(
        list(cfg.to_trainer_flags("OUT")),
        registry=spg.schedule_when_flags(trainer.read_text()),
        manifest_keys=set(getattr(cfg, "constants_manifest", {}) or {}),
        governance=getattr(cfg, "schedule_governance", {}) or {})
    naked = {v.flag for v in verds if not v.ok}
    assert naked == {
        "--tau-softplus-start-epoch",   # CE->tau loss-form boundary (PR95's 10% position)
        "--l7-start-epoch",             # tau->l7 loss-form boundary
        "--muon-start-epoch",           # AdamW->Muon optimizer-phase switch (a fail-safe cap)
        "--lane-band-start-epoch",      # analytic lane-band lever-engage
        "--seg-chroma-boundary-start-epoch",  # chroma-boundary lever-engage
    }, naked
    # the incumbent carries NO schedule_governance surface (all 5 untagged) -> all NAKED
    assert not getattr(cfg, "schedule_governance", {})


# ───────────────────────────── drift-detector leg ─────────────────────────────
def test_leg_file_in_scope():
    assert tdd.schedule_gov_file_in_scope("src/tac/witness_autoconfig.py")
    assert tdd.schedule_gov_file_in_scope("src/tac/witness_dsl/curriculum_dsl.py")
    assert not tdd.schedule_gov_file_in_scope("experiments/train_levelset_witness.py")
    assert not tdd.schedule_gov_file_in_scope("docs/notes.md")


def test_leg_naked_epoch_additions_detected():
    added = ['    "tau_softplus_start_epoch": 300,',
             '    muon_start_epoch = 726',
             '    "lane_edge_start_epoch": 0,']   # value 0 -> not a trigger
    hits = tdd.naked_epoch_additions(added)
    assert "tau_softplus_start_epoch 300" in hits
    assert "muon_start_epoch 726" in hits
    assert not any("lane_edge" in h for h in hits)


def test_leg_naked_epoch_argparse_default_detected():
    added = ['    ap.add_argument("--foo-start-epoch", type=int, default=450)']
    assert tdd.naked_epoch_additions(added) == ["--foo-start-epoch 450"]


def test_leg_waiver_respected():
    added = ['    "tau_softplus_start_epoch": 300,  # SCHEDULE_PROVENANCE_OK: control arm of the A/B']
    assert tdd.naked_epoch_additions(added) == []


def test_leg_window_gov_cite_suppresses():
    assert tdd.window_has_schedule_gov_cite(['    key = LawRef(equation_id="tau_end_knee_v1")'])
    assert tdd.window_has_schedule_gov_cite(['    schedule_governance = {...}'])
    assert not tdd.window_has_schedule_gov_cite(['    x = 3'])


def test_leg_pr95_stage_sequence_additions():
    added = ['    stages = ["ce", "tau_softplus", "l7", "muon"]']
    assert tdd.pr95_stage_sequence_additions(added)
    # a single stage name, or a non-sequence line, does not fire
    assert not tdd.pr95_stage_sequence_additions(['    note = "muon finisher only"'])


def test_leg_violations_fire_on_naked_epoch_without_cite():
    per_file = {"src/tac/witness_autoconfig.py": ['    "tau_softplus_start_epoch": 300,']}
    viol = tdd.schedule_provenance_violations(["add crucible tau stage"], per_file)
    assert len(viol) == 1 and "naked hardcoded schedule epoch" in viol[0]


def test_leg_no_fire_when_window_has_lawref_cite():
    per_file = {"src/tac/witness_autoconfig.py": [
        '    "muon_start_epoch": 726,',
        '    schedule_governance = {"--muon-start-epoch": {"class": "cap"}}']}
    assert tdd.schedule_provenance_violations(["migrate muon to cap"], per_file) == []


def test_leg_opt_out_respected():
    per_file = {"src/tac/witness_autoconfig.py": ['    "tau_softplus_start_epoch": 300,']}
    assert tdd.schedule_provenance_violations(["chore [no-triality]"], per_file) == []


def test_leg_fail_open_wrapper():
    # a malformed per_file value must never raise -> [] (fail-open)
    assert tdd.schedule_provenance_violations_safe(["x"], {"f": None}) == []


def test_leg_pr95_sequence_violation_message():
    per_file = {"src/tac/witness_dsl/curriculum_dsl.py": [
        '    curriculum = ("ce", "tau_softplus", "muon")']}
    viol = tdd.schedule_provenance_violations(["new curriculum shape"], per_file)
    assert len(viol) == 1 and "PR95-named stage sequence" in viol[0]


def test_existing_classify_untouched_for_non_schedule_commit():
    # a plain doc commit touching no leg + no schedule surface stays clean (the schedule leg
    # is additive; it never perturbs the base classify()).
    assert tdd.classify(["docs: fix a typo"], ["docs/readme.md"]) == "clean"
