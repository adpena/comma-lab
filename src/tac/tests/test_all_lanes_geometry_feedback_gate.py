from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from tools.build_cross_paradigm_frontier_inventory import build_inventory

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location("all_lanes_preflight_geometry_test", ALL_LANES)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_lanes_cross_paradigm_gate_accepts_fail_closed_geometry_contracts() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)

    assert module._geometry_feedback_inventory_failures(payload) == []


def test_all_lanes_cross_paradigm_gate_rejects_missing_geometry_contract() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)
    row = next(row for row in payload["rows"] if row["key"] == "raft_radial_openpilot_pose")
    row.pop("geometry_feedback_contract")

    failures = module._geometry_feedback_inventory_failures(payload)

    assert "raft_radial_openpilot_pose: geometry_feedback_contract_missing" in failures


def test_all_lanes_cross_paradigm_gate_rejects_geometry_dispatch_ready_drift() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)
    row = next(row for row in payload["rows"] if row["key"] == "lapose_motion_atom_allocator")
    row["geometry_feedback_contract"]["ready_for_exact_eval_dispatch"] = True

    failures = module._geometry_feedback_inventory_failures(payload)

    assert (
        "lapose_motion_atom_allocator: "
        "geometry_feedback_contract_ready_for_exact_eval_dispatch_false"
    ) in failures
