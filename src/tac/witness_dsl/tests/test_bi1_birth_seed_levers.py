# SPDX-License-Identifier: MIT
"""ddm_bi1 (#924) DSL tests for the TR1 birth seed/amplify lever."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tac.witness_dsl import bi1_birth_seed_levers_20260805 as M
from tac.witness_dsl.curriculum_dsl import Lever

_REPO = Path(__file__).resolve().parents[4]
_TR1 = _REPO / "experiments" / "train_tr1_partition_renderer_mlx.py"


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


def test_birth_seed_lever_returns_real_tr1_flags():
    lv = M.lever_tr1_birth_seed_amplify()
    assert isinstance(lv, Lever)
    assert lv.name.startswith("tr1_birth_seed_")
    assert lv.policy_contracts["score_claim"] is False
    declared = _trainer_declared_flags()
    missing = sorted(set(lv.overrides) - declared)
    assert not missing, f"BI1 lever emits undeclared TR1 flags: {missing}"


def test_birth_seed_lever_refuses_off_and_bad_arguments():
    with pytest.raises(ValueError):
        M.lever_tr1_birth_seed_amplify(seed_weight=0.0)
    with pytest.raises(ValueError):
        M.lever_tr1_birth_seed_amplify(amplify_weight=-0.1)
    with pytest.raises(ValueError):
        M.lever_tr1_birth_seed_amplify(classes="road")
    with pytest.raises(ValueError):
        M.lever_tr1_birth_seed_amplify(dilate_px=-1)
    with pytest.raises(ValueError):
        M.lever_tr1_birth_seed_amplify(persist="bogus")


def test_birth_seed_registry_files_factory_under_tr1():
    from tac.witness_dsl import lever_registry as LR

    builds = [f for f in LR.build_completeness().factories
              if f.module == "bi1_birth_seed_levers_20260805.py"]
    assert len(builds) == 1, f"expected 1 BI1 registered factory, got {len(builds)}"
    f = builds[0]
    assert f.factory == "lever_tr1_birth_seed_amplify"
    assert f.missing_flags == (), f"BI1 factory emits undeclared flags {f.missing_flags}"
    assert f.stub_marker is False
    assert f.trainer_declared is True
    assert f.trainer == M.TRAINER_RELPATH


def test_birth_seed_constant_manifest_carries_provenance():
    lv = M.lever_tr1_birth_seed_amplify(classes="lane,movable")
    assert lv.overrides["--tr1-birth-seed-classes"] == "lane,movable"
    for flag in lv.overrides:
        row = lv.constant_manifest.get(flag)
        assert row is not None, f"{flag} missing constant manifest"
        assert row["rung"]
        assert len(row["provenance"]) > 30
