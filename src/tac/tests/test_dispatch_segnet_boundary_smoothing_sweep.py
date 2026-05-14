# SPDX-License-Identifier: MIT
"""Regression tests for the SegNet boundary-smoothing dispatch wrapper."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "dispatch_segnet_boundary_smoothing_sweep.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "dispatch_segnet_boundary_smoothing_sweep", TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_segnet_boundary_smoothing_sweep"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dispatch_results_path_does_not_overwrite_nonstandard_rollup_name():
    tool = _load_tool()
    rollup = Path("experiments/results/custom_rollup.json")

    out = tool.dispatch_results_path(rollup)

    assert out != rollup
    assert out.name == "custom_rollup_dispatch_results.json"


def test_dispatch_results_path_preserves_canonical_rollup_replacement():
    tool = _load_tool()
    rollup = Path("experiments/results/segnet_boundary_smoothing_rollup_20260509.json")

    out = tool.dispatch_results_path(rollup)

    assert out.name == "segnet_boundary_smoothing_dispatch_results_20260509.json"


def test_child_claim_command_uses_parallel_guardrails():
    tool = _load_tool()

    cmd = tool.build_claim_cmd(
        lane_id="lane_a1_segnet_boundary_smoothing_inflate",
        instance_job_id="a1_segnet_boundary_smoothing_v_smooth_3x3",
        agent="codex:test",
        platform="github_actions",
        status="eval_cpu",
        notes="child",
        child_of="parent_job",
        parallel_reason="bounded test sweep",
    )

    assert "claim_lane_dispatch.py" in " ".join(cmd)
    assert "--allow-parallel" in cmd
    assert cmd[cmd.index("--child-of") + 1] == "parent_job"
    assert cmd[cmd.index("--parallel-reason") + 1] == "bounded test sweep"


def test_parent_claim_command_is_active_before_dispatch():
    tool = _load_tool()

    cmd = tool.build_claim_cmd(
        lane_id="lane_a1_segnet_boundary_smoothing_inflate",
        instance_job_id="parent_job",
        agent="codex:test",
        platform="github_actions",
        status="eval_cpu_sweep",
        notes="parent",
    )

    assert "--allow-parallel" not in cmd
    assert cmd[cmd.index("--status") + 1] == "eval_cpu_sweep"
