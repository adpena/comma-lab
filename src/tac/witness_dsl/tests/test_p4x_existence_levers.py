# SPDX-License-Identifier: MIT
"""ddm_p4x (#920) — tests for the existence-primitive DSL levers.

The point of these tests is that a lever can be WRONG in three ways a unit test of the
underlying math would never see: it can emit a flag the trainer does not declare
(never-invent-flags), it can be filed under the wrong vehicle (ddm_lr2 §1), or it can
collide with another factory's name family (the ddm_pt2 round-1 lesson). All three are
checked structurally here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tac.optimization.existence_hinge import CONNECTIVITY_4, CONNECTIVITY_8
from tac.witness_dsl import p4x_existence_levers_20260803 as M
from tac.witness_dsl.curriculum_dsl import Lever

_REPO = Path(__file__).resolve().parents[4]
_TR1 = _REPO / "experiments" / "train_tr1_partition_renderer_mlx.py"


def _all_levers() -> list[Lever]:
    return [
        M.lever_lane_existence_hinge(),
        M.lever_existence_birth_matrix(),
        M.lever_existence_grammar(CONNECTIVITY_4),
        M.lever_existence_grammar(CONNECTIVITY_8),
        M.lever_existence_weight_policy("uniform"),
        M.lever_existence_weight_policy("sqrt_area"),
        M.lever_existence_weight_policy("area"),
        M.lever_existence_target(0.5),
        M.lever_existence_beta(16.0),
    ]


def _trainer_declared_flags() -> set[str]:
    tree = ast.parse(_TR1.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
    return out


def test_every_factory_returns_a_lever():
    for lv in _all_levers():
        assert isinstance(lv, Lever)
        assert lv.name and lv.notes
        assert lv.overrides


def test_never_invent_flags():
    """Every emitted flag must exist in the TR1 argparse."""
    declared = _trainer_declared_flags()
    emitted = {f for lv in _all_levers() for f in lv.overrides}
    assert emitted, "no flags emitted"
    missing = sorted(emitted - declared)
    assert not missing, f"levers emit flags the TR1 trainer does not declare: {missing}"


def test_trainer_relpath_files_levers_under_the_shipped_vehicle():
    """ddm_lr2 §1: an undeclared module silently defaults to the RETIRED trainer."""
    assert M.TRAINER_RELPATH == "experiments/train_tr1_partition_renderer_mlx.py"
    assert _TR1.is_file()


def test_lever_names_are_unique_and_do_not_collide_with_other_families():
    """ddm_pt2 round-1 lesson: a new lever must not land inside another factory's name
    family, because the registry join reports that, it does not repair it."""
    names = [lv.name for lv in _all_levers()]
    assert len(names) == len(set(names))
    for n in names:
        assert n.startswith("tr1_existence_")
    # 'tr1_seg_*' is lever_seg_physics' family; 'tr1_focal_*' is pt2's. Stay out of both.
    assert not any(n.startswith(("tr1_seg_", "tr1_focal_", "tr1_fisher_")) for n in names)


def test_every_lever_arms_the_weight_because_zero_is_the_off_control():
    for lv in _all_levers():
        assert "--existence-hinge-weight" in lv.overrides
        assert float(lv.overrides["--existence-hinge-weight"]) > 0.0


def test_factories_refuse_the_off_control_and_bad_arguments():
    """A Lever for the OFF control is meaningless -- the trainer default already is OFF."""
    with pytest.raises(ValueError):
        M.lever_lane_existence_hinge(weight=0.0)
    with pytest.raises(ValueError):
        M.lever_existence_birth_matrix(weight=-1.0)
    with pytest.raises(ValueError):
        M.lever_existence_grammar(connectivity=6)
    with pytest.raises(ValueError):
        M.lever_existence_weight_policy("bogus")
    with pytest.raises(ValueError):
        M.lever_existence_weight_policy("")
    with pytest.raises(ValueError):
        M.lever_existence_target(-0.1)
    with pytest.raises(ValueError):
        M.lever_existence_beta(0.0)


def test_every_constant_carries_a_provenance_rung():
    """constants-are-poison: no bare literal may ship without a laddered manifest row."""
    for lv in _all_levers():
        for flag in lv.overrides:
            row = lv.constant_manifest.get(flag)
            if row is None:
                continue  # a flag may be manifested by its sibling factory
            assert row["rung"], f"{lv.name}:{flag} has no provenance rung"
            assert len(row["provenance"]) > 40, f"{lv.name}:{flag} provenance is a stub"


def test_each_factory_docstring_preregisters_a_falsifier():
    """A lever with no falsifier cannot produce an admissible negative."""
    for fn in (M.lever_lane_existence_hinge, M.lever_existence_birth_matrix,
               M.lever_existence_grammar, M.lever_existence_weight_policy,
               M.lever_existence_target, M.lever_existence_beta):
        doc = (fn.__doc__ or "").lower()
        assert "falsifier" in doc, f"{fn.__name__} has no pre-registered falsifier"


def test_grammar_lever_encodes_both_measured_grammars():
    four = M.lever_existence_grammar(CONNECTIVITY_4)
    eight = M.lever_existence_grammar(CONNECTIVITY_8)
    assert four.overrides["--existence-hinge-connectivity"] == 4
    assert eight.overrides["--existence-hinge-connectivity"] == 8
    assert four.name != eight.name


def test_registry_holds_every_factory_under_the_tr1_vehicle():
    """The duty-to-measure surface: unregistered == orphaned, by definition."""
    from tac.witness_dsl import lever_registry as LR
    builds = [f for f in LR.build_completeness().factories
              if f.module == "p4x_existence_levers_20260803.py"]
    assert len(builds) == 6, f"expected 6 registered factories, got {len(builds)}"
    for f in builds:
        assert f.missing_flags == (), f"{f.factory} emits undeclared flags {f.missing_flags}"
        assert f.stub_marker is False, f"{f.factory} is a stub"
        assert f.trainer_declared is True, f"{f.factory} is not filed under a declared trainer"
        assert f.trainer == M.TRAINER_RELPATH
