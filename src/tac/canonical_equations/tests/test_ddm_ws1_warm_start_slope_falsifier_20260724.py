from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_ws1_warm_start_slope_falsifier_20260724 import (
    EQUATION_ID,
    build_ddm_ws1_warm_start_slope_falsifier_v1,
    populate_ddm_ws1_warm_start_slope_falsifier_v1,
)
from tac.optimization.ddm_warm_start_slope_falsifier import (
    critical_pose_to_seg_slope_ratio,
)

REPO = Path(__file__).resolve().parents[4]
SPEC = REPO / ".omx/research/configs/ddm_ws1_j5_slope_falsifier_20260724.json"


def test_ws1_equation_rederives_preregistered_ratio() -> None:
    equation = build_ddm_ws1_warm_start_slope_falsifier_v1()
    spec = json.loads(SPEC.read_bytes())
    starts = spec["starts"]
    observed = critical_pose_to_seg_slope_ratio(
        wseg_d_seg=starts["W_seg"]["d_seg"],
        wseg_d_pose=starts["W_seg"]["d_pose"],
        wjoint_d_seg=starts["W_joint"]["d_seg"],
        wjoint_d_pose=starts["W_joint"]["d_pose"],
    )
    assert equation.equation_id == EQUATION_ID
    assert observed == pytest.approx(
        spec["derived_gap"][
            "critical_pose_progress_to_seg_advantage_erosion_ratio"
        ]
    )
    assert equation.empirical_anchors[0].empirical_output[
        "training_outcome"
    ] == "UNMEASURED_SPEC_ONLY"


def test_ws1_equation_registers_through_locked_helper(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    equation = populate_ddm_ws1_warm_start_slope_falsifier_v1(
        path=registry,
        lock_path=tmp_path / "registry.lock",
        agent="codex",
        subagent_id="ws1-test",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    assert equation.equation_id == EQUATION_ID
    assert rows[-1]["equation_payload"]["equation_id"] == EQUATION_ID
