"""Tests for the FIRST-CLASS schedule/curriculum DSL objects (operator 2026-07-06 "we need
schedule and curriculum in DSL as well"; task #334).

Covers HoscSchedule / Transition / Curriculum + the WitnessProgram.curriculum wiring:
  * each object's .flags() emits the right (real) flag set;
  * store_true flags are never emitted False (review C2);
  * Curriculum.validate() enforces the hand-off enum + the tau<l7 ordering (muon FREE);
  * the sealed_205 curriculum-first path is FLAG-EQUIVALENT to the legacy sealed_205_program
    (the elevation is faithful) AND validates against the real trainer argparse;
  * handoff='event' swaps in the #315 nucleus-guard flags (the CE-didn't-plateau actuator);
  * curriculum=None (default) leaves every existing program byte-identical (legacy path).
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from tac import witness_autoconfig as wac
from tac.witness_dsl import curriculum_dsl as cd

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


# --- HoscSchedule ----------------------------------------------------------
def test_hosc_schedule_flags():
    f = cd.HoscSchedule(1.0, 4.0, "linear", 1.0).flags()
    assert f == {
        "--hosc-beta": 1.0, "--hosc-beta-end": 4.0,
        "--hosc-beta-anneal": "linear", "--hosc-omega": 1.0,
    }


# --- Transition (reheat) ---------------------------------------------------
def test_transition_flags_reset_true_emits_bare_store_true():
    f = cd.Transition(8, 0.1, "linear", reset_moments=True).flags()
    assert f["--stage-transition-rewarmup-epochs"] == 8
    assert f["--stage-transition-rewarmup-floor"] == 0.1
    assert f["--stage-transition-rewarmup-shape"] == "linear"
    # store_true → emitted True (bare), never False (review C2: a False would compile to --no-X).
    assert f["--stage-transition-reset-moments"] is True


def test_transition_flags_reset_false_omits_store_true():
    f = cd.Transition(reset_moments=False).flags()
    assert "--stage-transition-reset-moments" not in f  # never emitted False


# --- Curriculum.flags ------------------------------------------------------
def _curr(**kw):
    base = dict(
        stages=(cd.Stage("CE", None, None),
                cd.Stage("tau", "--tau-softplus-start-epoch", 300),
                cd.Stage("l7", "--l7-start-epoch", 1001),
                cd.Stage("muon", "--muon-start-epoch", 726)),
        temp=cd.Anneal(1.0, 0.05),
    )
    base.update(kw)
    return cd.Curriculum(**base)


def test_curriculum_flags_fixed_full():
    f = _curr(
        regularizers=(cd.Regularizer("--eikonal-weight", 0.0),
                      cd.Regularizer("--length-weight", 0.001)),
        hosc=cd.HoscSchedule(1.0, 4.0, "linear", 1.0),
        tau=0.3,
        transition=cd.Transition(8, 0.1, "linear", True),
        handoff="fixed",
    ).flags()
    assert f["--curriculum"] is True
    assert f["--softmax-temp-start"] == 1.0 and f["--softmax-temp-end"] == 0.05
    assert f["--tau-softplus-start-epoch"] == 300
    assert f["--l7-start-epoch"] == 1001 and f["--muon-start-epoch"] == 726
    assert f["--hosc-beta-end"] == 4.0 and f["--tau-softplus-tau"] == 0.3
    # FIXED hand-off does NOT emit the event guards.
    assert "--curriculum-event-triggered" not in f
    assert "--curriculum-nucleus-guard" not in f


def test_curriculum_flags_event_adds_guards():
    f = _curr(handoff="event").flags()
    assert f["--curriculum-event-triggered"] is True
    assert f["--curriculum-nucleus-guard"] is True


def test_curriculum_curriculum_off_omits_master_flag():
    f = _curr(curriculum_on=False).flags()
    assert "--curriculum" not in f


# --- Curriculum.validate ---------------------------------------------------
def test_curriculum_validate_ok():
    assert _curr().validate() == []


def test_curriculum_validate_bad_handoff():
    v = _curr(handoff="nope").validate()
    assert any("handoff" in p for p in v)


def test_curriculum_validate_tau_ge_l7_rejected():
    bad = cd.Curriculum(
        stages=(cd.Stage("tau", "--tau-softplus-start-epoch", 300),
                cd.Stage("l7", "--l7-start-epoch", 200)),
        temp=cd.Anneal(1.0, 0.05))
    assert any("ordering" in p for p in bad.validate())


def test_curriculum_validate_muon_before_l7_is_allowed():
    # the REAL sealed #205 places muon(726) BEFORE l7(1001) — must NOT be flagged (operator freedom).
    ok = _curr()  # tau300 < l7 1001, muon726 < l7 → allowed
    assert ok.validate() == []


# --- sealed_205 agreement (the faithfulness proof) -------------------------
def test_sealed_205_curriculum_first_is_flag_equivalent_to_legacy():
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)
    prog = cd.sealed_205_program(out_dir="experiments/results/_t")
    fd_legacy = prog.flag_dict()
    prog_cur = replace(prog, curriculum=cd.sealed_205_curriculum(cfg))
    fd_cur = prog_cur.flag_dict()
    assert fd_legacy == fd_cur, {
        k: (fd_legacy.get(k), fd_cur.get(k))
        for k in set(fd_legacy) | set(fd_cur) if fd_legacy.get(k) != fd_cur.get(k)
    }


def test_sealed_205_curriculum_validates_against_real_trainer_flags():
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)
    prog = replace(cd.sealed_205_program(out_dir="experiments/results/_t"),
                   curriculum=cd.sealed_205_curriculum(cfg))
    # never-invent-flags + curriculum ordering + preserve/contain/authority all clean.
    assert prog.validate() == []


def test_sealed_205_event_handoff_validates_and_adds_guards():
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)
    prog = replace(cd.sealed_205_program(out_dir="experiments/results/_t"),
                   curriculum=cd.sealed_205_curriculum(cfg, handoff="event"))
    fd = prog.flag_dict()
    assert fd["--curriculum-event-triggered"] is True
    assert fd["--curriculum-nucleus-guard"] is True
    # the guard flags are REAL trainer flags (never-invent-flags passes).
    assert prog.validate() == []


# --- backward-compat: curriculum=None leaves legacy programs unchanged ------
def test_curriculum_none_is_legacy_path_unchanged():
    prog = cd.sealed_205_program(out_dir="experiments/results/_t")
    assert prog.curriculum is None
    fd = prog.flag_dict()
    # legacy schedule still emitted directly (temp/stages/regularizers), no curriculum object.
    assert fd["--curriculum"] is True
    assert fd["--tau-softplus-start-epoch"] == 300
    assert fd["--softmax-temp-start"] == 1.0


def test_baseline_program_unaffected():
    fd = cd.BASELINE.flag_dict()
    assert cd.BASELINE.curriculum is None
    assert fd["--tau-softplus-start-epoch"] == 300
    assert fd["--softmax-temp-end"] == 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
