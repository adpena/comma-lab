"""owed-1 REPAIRED POSE-GATE — DSL wire-in + activation-ledger visibility (SYNTHESIS_v3_v752 §A.4).

Triality DSL leg: the ``PoseFinishConditioningGate`` Lever factory HOLDS the ``--pose-finish-engage-on``
flag (a lever is not built until it is a DSL factory), is ``--dsl-lever``-composable, compiles through the
trainer's REAL argparse, validates clean, closes the completeness gap, and surfaces in the #247 costate
duty-to-measure queue (never-fired). Mirrors ``test_feed07_dsl_wirein`` for the sibling levers.
"""
from __future__ import annotations

import pytest

from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry as LR

_LEVER = "PoseFinishConditioningGate"
_FLAG = "--pose-finish-engage-on"


def test_pose_gate_lever_is_composable_single_lever_factory():
    comp = LR.name_composable_levers()
    assert _LEVER in comp, f"{_LEVER} must be composable via --dsl-lever (zero-required-arg single Lever)"
    lv = LR.resolve_composable_lever(_LEVER)
    assert isinstance(lv, cd.Lever)
    assert lv.overrides[_FLAG] == "sigma_min_plateau"


def test_pose_gate_closes_the_dsl_completeness_gap():
    """The flag was an UNMAPPED trainer gap before the factory (config-orphan); now the DSL holds it."""
    c = LR.completeness()
    assert _FLAG not in c.unmapped, "the DSL must now hold --pose-finish-engage-on (no longer a gap)"
    assert _FLAG in LR.dsl_referenced_flags()
    assert _FLAG not in c.stale, "the flag is a REAL trainer flag (not DSL drift)"


def test_pose_gate_composes_and_validates_through_real_argparse():
    ap = cd.build_real_trainer_parser()
    lv = LR.resolve_composable_lever(_LEVER)
    prog = cd.BASELINE.with_lever(lv)
    assert prog.validate() == [], "the pose-gate lever must reference only real, type-compatible flags"
    # the compiled flags parse through the trainer's OWN argparse (never-invent-flags, end-to-end).
    argv: list[str] = []
    for flag, val in prog.flag_dict().items():
        argv.append(str(flag))
        if val is True:
            continue
        if val is False:
            argv[-1] = f"--no-{flag[2:]}"
            continue
        argv.append(str(val))
    try:
        ap.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - diagnostic
        raise AssertionError(f"pose-gate composed argv rejected by the real argparse (rc={exc.code})") from exc


def test_pose_gate_optional_backstop_and_wpose_guards():
    # optional args arm the two-phase + finish weight; the guards mirror the sibling lever discipline.
    lv = cd.PoseFinishConditioningGate(backstop_epoch=2500, w_pose=0.05)
    assert lv.overrides["--pose-finish-start-epoch"] == 2500
    assert lv.overrides["--w-pose"] == 0.05
    with pytest.raises(ValueError, match="backstop_epoch must be > 0"):
        cd.PoseFinishConditioningGate(backstop_epoch=0)
    with pytest.raises(ValueError, match="w_pose must be > 0"):
        cd.PoseFinishConditioningGate(w_pose=0.0)


def test_pose_gate_lands_in_activation_ledger_duty_to_measure(tmp_path):
    """#247 costate SENSE: the never-fired lever surfaces in the duty-to-measure queue (auto-derived from
    lever_factories() — no wiring), so a designed-but-never-measured gate cannot be orphaned."""
    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers, never_fired
    assert _LEVER in known_levers(), f"{_LEVER} missing from the activation ledger known set"
    empty = tmp_path / "activation_ledger_empty.jsonl"   # nonexistent => zero events
    assert _LEVER in never_fired(path=empty)
    assert _LEVER in duty_to_measure(path=empty)
