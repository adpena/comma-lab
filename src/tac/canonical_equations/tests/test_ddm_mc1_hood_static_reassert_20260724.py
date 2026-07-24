# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import json

import pytest

from tac.canonical_equations.ddm_mc1_hood_static_reassert_20260724 import (
    EQUATION_ID,
    build_ddm_mc1_static_hood_reassert_joint_action_v1,
    populate_ddm_mc1_static_hood_reassert_joint_action_v1,
)
from tac.canonical_equations.registry import load_registry_events_lenient


def test_callable_path_imports_and_evaluates_stored_winner() -> None:
    equation = build_ddm_mc1_static_hood_reassert_joint_action_v1()
    module_path, function_name = equation.python_callable_module_path.split(":")
    function = getattr(importlib.import_module(module_path), function_name)
    with open(equation.canonical_producers[1]) as stream:
        receipt = json.load(stream)
    parent = receipt["input_custody"]["menu1_parent"]
    child = receipt["pool_winner"]
    derived = function(
        parent_d_seg=parent["d_seg"],
        parent_d_pose=parent["d_pose"],
        parent_bytes=parent["archive_bytes"],
        child_d_seg=child["d_seg"],
        child_d_pose=child["d_pose"],
        child_bytes=child["archive_bytes"],
    )
    assert derived == pytest.approx(child["delta_advisory_objective_vs_parent"], abs=1e-12)
    assert derived > 0.0


def test_populate_uses_locked_registry_helper(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.lock"
    equation = populate_ddm_mc1_static_hood_reassert_joint_action_v1(
        path=registry,
        lock_path=lock,
        agent="codex",
        subagent_id="test",
    )
    rows = load_registry_events_lenient(registry)
    assert equation.equation_id == EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["equation_id"] == EQUATION_ID
