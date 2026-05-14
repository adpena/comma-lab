# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_ego_motion_field_atoms.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_ego_motion_field_atoms_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pose_file(tmp_path: Path, poses: torch.Tensor) -> Path:
    path = tmp_path / "optimized_poses.pt"
    torch.save(poses, path)
    return path


def test_ego_motion_plan_is_deterministic_and_non_promotable(tmp_path: Path) -> None:
    planner = _load_planner()
    poses = torch.zeros((4, 6), dtype=torch.float32)
    poses[:, 0] = torch.tensor([20.0, 20.4, 20.1, 20.8])
    poses[:, 1] = torch.tensor([0.0, 0.1, -0.2, 0.3])
    poses[:, 2] = torch.tensor([0.0, -0.1, 0.2, -0.3])
    pose_path = _write_pose_file(tmp_path, poses)

    kwargs = {
        "pose_path": pose_path,
        "expected_pairs": 4,
        "max_atoms": 16,
        "foveal_center": (256.0, 174.0),
    }
    first = planner.build_plan(output_json=tmp_path / "a.json", **kwargs)
    second = planner.build_plan(output_json=tmp_path / "b.json", **kwargs)

    assert first == second
    assert json.loads((tmp_path / "a.json").read_text()) == first
    assert first["schema"] == "ego_motion_field_atom_plan_v1"
    assert first["score_claim"] is False
    assert first["promotion_eligible"] is False
    assert first["evidence_grade"] == "planning_only"
    assert first["dynamic_foveation"]["schema"] == "ego_motion_dynamic_foveation_manifest_v1"
    assert first["dynamic_foveation"]["score_claim"] is False
    assert first["dynamic_foveation"]["promotion_eligible"] is False
    assert first["dynamic_foveation"]["provenance"]["pose_sha256"]
    assert first["dynamic_foveation"]["frame_center_count"] == 8
    assert "frame_centers_sha256" in first["dynamic_foveation"]
    assert first["atoms"]
    assert all(atom["score_claim"] is False for atom in first["atoms"])


def test_ego_motion_plan_uses_non_velocity_dimensions_for_dynamic_centers(tmp_path: Path) -> None:
    planner = _load_planner()
    poses = torch.zeros((3, 6), dtype=torch.float32)
    poses[:, 0] = torch.tensor([20.0, 20.0, 20.0])
    poses[:, 1] = torch.tensor([-1.0, 0.0, 1.0])
    poses[:, 2] = torch.tensor([1.0, 0.0, -1.0])
    pose_path = _write_pose_file(tmp_path, poses)

    plan = planner.build_plan(
        pose_path=pose_path,
        output_json=tmp_path / "plan.json",
        expected_pairs=3,
        max_atoms=12,
        foveal_center=(100.0, 50.0),
        center_gain_x=10.0,
        center_gain_y=5.0,
        frame_width=200,
        frame_height=100,
    )

    centers = plan["dynamic_foveation"]["frame_centers"]
    assert centers[0] == centers[1]
    assert centers[2] == centers[3]
    assert centers[4] == centers[5]
    assert centers[0][0] < centers[2][0] < centers[4][0]
    assert centers[0][1] > centers[2][1] > centers[4][1]
    families = {atom["family"] for atom in plan["atoms"]}
    assert "curvature_velocity_yaw_product" in families
    assert "pitch_velocity_product" in families


def test_ego_motion_plan_rejects_partial_pose_files(tmp_path: Path) -> None:
    planner = _load_planner()
    pose_path = _write_pose_file(tmp_path, torch.zeros((2, 6), dtype=torch.float32))

    try:
        planner.build_plan(
            pose_path=pose_path,
            output_json=tmp_path / "bad.json",
            expected_pairs=3,
        )
    except ValueError as exc:
        assert "expected 3" in str(exc) or "Pose count mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected partial pose rejection")
