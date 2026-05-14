# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BRIDGE_PATH = REPO_ROOT / "experiments" / "build_c067_multiresolution_stack_candidate.py"


def _load_bridge() -> Any:
    spec = importlib.util.spec_from_file_location("c067_multires_stack_bridge_test", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _assert_no_score_claim_true(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("score_claim") is True:
            raise AssertionError(f"score_claim=true found in {value}")
        for child in value.values():
            _assert_no_score_claim_true(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_score_claim_true(child)


def _cmg2_artifact_manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "cmg2_downsample_candidate_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
            "frontier_archive": {
                "path": str(path.parent / "frontier_archive.zip"),
                "bytes": 276214,
                "sha256": "a" * 64,
            },
            "decoded_mask_array": {
                "path": str(path.parent / "decoded_mask_array.npy"),
                "shape": [600, 384, 512],
                "npy_sha256": "b" * 64,
                "tensor_sha256": "c" * 64,
            },
            "output_archive": {
                "path": str(path.parent / "archive.zip"),
                "bytes": 194020,
                "sha256": "d" * 64,
            },
            "cmg2": {
                "schema": "cmg2_downsample_candidate_v1",
                "scale": [2, 2],
                "compressor": "bz2",
                "payload_bytes": 132610,
                "score_claim": False,
            },
        },
    )


def _plan(path: Path, *, cmg2_manifest: Path, include_repair: bool = True) -> Path:
    artifacts: list[dict[str, Any]] = [
        {
            "artifact_id": "pass0_anchor",
            "path": str(path.parent / "anchor_eval.json"),
            "schema": None,
            "pass_index": 0,
            "logical_members": ["masks.mkv", "renderer.bin", "optimized_poses.bin"],
            "builder_consumable": False,
            "score_claim": False,
        },
        {
            "artifact_id": "pass1_cmg2",
            "path": str(cmg2_manifest),
            "schema": "cmg2_downsample_candidate_v1",
            "pass_index": 1,
            "logical_members": ["masks.mkv"],
            "builder_consumable": True,
            "score_claim": False,
        },
    ]
    component_ids = ["pass0_anchor", "pass1_cmg2"]
    if include_repair:
        repair_manifest = _write_json(
            path.parent / "repair_plan.json",
            {
                "schema": "multimask_reconciliation_atom_plan_v1",
                "score_claim": False,
                "evidence_grade": "planning_only",
            },
        )
        artifacts.append(
            {
                "artifact_id": "pass2_repair_plan",
                "path": str(repair_manifest),
                "schema": "multimask_reconciliation_atom_plan_v1",
                "pass_index": 2,
                "logical_members": ["masks.mkv"],
                "builder_consumable": False,
                "score_claim": False,
            }
        )
        component_ids.append("pass2_repair_plan")

    _write_json(
        path.parent / "anchor_eval.json",
        {
            "score_recomputed_from_components": 0.315,
            "archive_size_bytes": 276214,
            "score_claim": False,
        },
    )
    return _write_json(
        path,
        {
            "schema": "c067_multiresolution_stack_planner_v1",
            "score_claim": False,
            "loaded_artifacts": artifacts,
            "candidate_policies": [
                {
                    "policy_id": "c067_multires_test_policy",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "existing_builder_can_consume_full_stack": False,
                    "dispatchable_from_this_plan": False,
                    "component_ids": component_ids,
                    "passes": [],
                    "pass_antagonisms": [],
                    "pass_synergies": [],
                }
            ],
        },
    )


def test_bridge_emits_valid_standalone_builder_command_and_no_score_claim(tmp_path: Path) -> None:
    bridge = _load_bridge()
    cmg2_manifest = _cmg2_artifact_manifest(tmp_path / "cmg2_build_manifest.json")
    plan_json = _plan(tmp_path / "c067_multiresolution_stack_plan.json", cmg2_manifest=cmg2_manifest)

    manifest = bridge.build_manifest(
        plan_json=plan_json,
        output_dir=tmp_path / "bridge_out",
    )

    assert manifest["schema"] == "c067_multiresolution_stack_build_manifest_v1"
    assert manifest["score_claim"] is False
    assert manifest["byte_closed_stack_archive_emitted"] is False
    _assert_no_score_claim_true(manifest)
    policy = manifest["build_policy_manifests"][0]
    assert policy["full_stack_status"] == "blocked"
    assert policy["dispatchable"] is False
    steps = policy["runnable_standalone_steps"]
    assert len(steps) == 1
    command = steps[0]["command"]
    assert command[1] == "experiments/build_cmg2_downsample_candidate.py"
    assert "--scale-y" in command
    assert "--scale-x" in command
    assert "--compressor" in command
    assert steps[0]["command_status"] == "argparse_valid"
    assert steps[0]["part_of_byte_closed_stack"] is False
    assert (tmp_path / "bridge_out" / "c067_multiresolution_stack_build_manifest.json").exists()


def test_require_byte_closed_stack_refuses_unsupported_policy_composition(tmp_path: Path) -> None:
    bridge = _load_bridge()
    cmg2_manifest = _cmg2_artifact_manifest(tmp_path / "cmg2_build_manifest.json")
    plan_json = _plan(tmp_path / "c067_multiresolution_stack_plan.json", cmg2_manifest=cmg2_manifest)

    try:
        bridge.build_manifest(
            plan_json=plan_json,
            output_dir=tmp_path / "bridge_out",
            require_byte_closed_stack=True,
        )
    except bridge.BridgeError as exc:
        message = str(exc)
        assert "byte-closed" in message
        assert "unsupported" in message
        assert "masks.mkv" in message
    else:
        raise AssertionError("BridgeError was not raised")


def test_unsupported_repair_component_is_blocker_not_fake_archive(tmp_path: Path) -> None:
    bridge = _load_bridge()
    cmg2_manifest = _cmg2_artifact_manifest(tmp_path / "cmg2_build_manifest.json")
    plan_json = _plan(tmp_path / "c067_multiresolution_stack_plan.json", cmg2_manifest=cmg2_manifest)

    manifest = bridge.build_manifest(
        plan_json=plan_json,
        output_dir=tmp_path / "bridge_out",
    )

    policy = manifest["build_policy_manifests"][0]
    assert policy["archive_path"] is None
    assert policy["byte_closed_stack_archive_emitted"] is False
    assert any(
        item["component_id"] == "pass2_repair_plan" and "no byte-closed" in item["reason"]
        for item in policy["unsupported_components"]
    )
    assert any("existing_builder_can_consume_full_stack=false" in reason for reason in policy["blocker_reasons"])


def test_rejects_score_claim_true_in_planner_input(tmp_path: Path) -> None:
    bridge = _load_bridge()
    cmg2_manifest = _cmg2_artifact_manifest(tmp_path / "cmg2_build_manifest.json")
    plan_json = _plan(
        tmp_path / "c067_multiresolution_stack_plan.json",
        cmg2_manifest=cmg2_manifest,
        include_repair=False,
    )
    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    payload["candidate_policies"][0]["score_claim"] = True
    plan_json.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    try:
        bridge.build_manifest(plan_json=plan_json, output_dir=tmp_path / "bridge_out")
    except bridge.BridgeError as exc:
        assert "score_claim=true" in str(exc)
    else:
        raise AssertionError("BridgeError was not raised")
