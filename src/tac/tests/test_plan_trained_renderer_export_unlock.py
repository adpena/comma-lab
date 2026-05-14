# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_trained_renderer_export_unlock.py"


def _load_planner(name: str = "_plan_trained_renderer_export_unlock_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _surrogate_preflight_payload() -> dict[str, Any]:
    return {
        "schema": "trained_renderer_blockfp_transplant_preflight_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "renderer_export": {
            "mode": "source_renderer_surrogate",
            "same_as_source_renderer": True,
            "dispatchable_trained_export": False,
            "sha256": "5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb",
        },
        "best_by_archive_bytes": {
            "candidate_id": "trained_qbf1_b1024",
            "archive_bytes": 283869,
            "archive_sha256": "6d331e479d961df22a2baa8b3f09722394ece7d0c194821c80c6aa354cb1449b",
            "archive_path": "out/archive.zip",
        },
        "h100_lightning_readiness": {
            "ready": False,
            "reason": "source-renderer surrogate only",
            "next_commands_if_ready": None,
        },
    }


def _ready_preflight_payload() -> dict[str, Any]:
    payload = _surrogate_preflight_payload()
    payload["renderer_export"] = {
        "mode": "trained_renderer_export",
        "same_as_source_renderer": False,
        "dispatchable_trained_export": True,
        "sha256": "a" * 64,
    }
    payload["best_by_archive_bytes"] = {
        "candidate_id": "trained_qbf1_b0064",
        "archive_bytes": 250000,
        "archive_sha256": "b" * 64,
        "archive_path": "out/ready/archive.zip",
        "pose_safety_gate": {
            "status": "pass",
            "safe_for_exact_eval_dispatch": True,
            "matching_report_path": "pose_safety.json",
            "blockers": [],
        },
    }
    payload["best_dispatchable_after_pose_safety"] = {
        "candidate_id": "trained_qbf1_b0064",
        "archive_bytes": 250000,
        "archive_sha256": "b" * 64,
        "archive_path": "out/ready/archive.zip",
        "pose_safety_gate": {
            "status": "pass",
            "safe_for_exact_eval_dispatch": True,
            "matching_report_path": "pose_safety.json",
            "blockers": [],
        },
    }
    payload["h100_lightning_readiness"] = {
        "ready": True,
        "reason": "trained renderer export is byte-closed",
        "pose_safety_required": True,
        "selected_pose_safety_gate": {
            "status": "pass",
            "safe_for_exact_eval_dispatch": True,
            "matching_report_path": "pose_safety.json",
            "blockers": [],
        },
        "next_commands_if_ready": {
            "claim_command": ["tools/claim_lane_dispatch.py", "claim"],
            "lightning_exact_eval_submit_command_shape": [
                "scripts/launch_lightning_batch_job.py",
                "exact-eval",
                "--machine",
                "g7e.4xlarge",
            ],
            "remote_gpu_dispatch_performed": False,
        },
    }
    return payload


def _legacy_ready_preflight_without_pose_safety_payload() -> dict[str, Any]:
    payload = _ready_preflight_payload()
    payload.pop("best_dispatchable_after_pose_safety", None)
    payload["best_by_archive_bytes"].pop("pose_safety_gate", None)
    payload["h100_lightning_readiness"].pop("pose_safety_required", None)
    payload["h100_lightning_readiness"].pop("selected_pose_safety_gate", None)
    return payload


def _qfaithful_export_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "score_claim": False,
        "renderer_bin": "export/renderer.bin",
        "renderer_bin_bytes": 389756,
        "renderer_bin_sha256": "86d11569d5956be778d096c6b3372144bbc14f24478a53eb22c7f648a53f51ae",
        "pose_shape": [600, 6],
    }


def _qfaithful_packed_payload() -> dict[str, Any]:
    return {
        "score_claim": False,
        "output_archive": "qzs3_pr64_qp1/archive.zip",
        "output_archive_bytes": 273103,
        "output_archive_sha256": "a34f493b77e3a2ccba7e059134127e9b3cb6e774a41862143d369fa3f5fc81af",
        "payload_format": "pr64_len_table",
        "payload_member": "p",
        "pose_codec": "pose_qp1_v1",
        "header": {
            "schema": "renderer_payload_pr64_len_table_v1",
            "members": [
                {
                    "name": "renderer.bin",
                    "bytes": 59288,
                    "sha256": "00ce395fc6495a47d26e3844537859fdc2a1bc38d121ea5f4fb2610179d26e46",
                },
                {"name": "masks.mkv", "bytes": 223738, "sha256": "1" * 64},
                {"name": "optimized_poses.bin", "bytes": 1149, "sha256": "2" * 64},
            ],
        },
    }


def test_surrogate_preflight_blocks_h100_commands(tmp_path: Path) -> None:
    planner = _load_planner()
    _write_json(
        tmp_path / "trained_renderer_blockfp_preflight.json",
        _surrogate_preflight_payload(),
    )
    (tmp_path / "candidate_renderer.bin").write_bytes(b"QZS3-not-source")
    (tmp_path / "checkpoint.pt").write_bytes(b"PK\x03\x04torch")

    plan = planner.build_plan(scan_dirs=(tmp_path,))

    assert plan["schema"] == "trained_renderer_export_unlock_plan_v1"
    assert plan["score_claim"] is False
    assert plan["remote_gpu_dispatch_performed"] is False
    assert plan["readiness"]["verdict"] == "blocked_no_h100_dispatch"
    assert plan["readiness"]["h100_lightning_commands"] is None
    assert "no non-surrogate trained-renderer archive passed preflight" in plan["readiness"]["blockers"]
    assert plan["non_surrogate_candidate_count"] == 1
    assert any(item["kind"] == "renderer_export_candidate" for item in plan["candidates"])
    assert any(item["kind"] == "checkpoint_requires_export" for item in plan["candidates"])


def test_qfaithful_exports_are_present_but_exact_negative_not_missing(tmp_path: Path) -> None:
    planner = _load_planner("_plan_trained_renderer_export_unlock_qfaithful_test")
    qf = tmp_path / "qfaithful_case"
    _write_json(qf / "export" / "export_provenance.json", _qfaithful_export_payload())
    (qf / "export" / "renderer.bin").write_bytes(b"QFAI" + b"x" * 32)
    _write_json(qf / "qzs3_pr64_qp1" / "packed_renderer_payload_provenance.json", _qfaithful_packed_payload())
    _write_json(
        qf / "qzs3_pr64_qp1" / "contest_auth_eval.json",
        {
            "archive_size_bytes": 273103,
            "final_score": 22.07,
            "avg_posenet_dist": 46.18100739,
            "avg_segnet_dist": 0.00393906,
            "n_samples": 600,
        },
    )

    plan = planner.build_plan(scan_dirs=(qf,))

    assert plan["readiness"]["verdict"] == "blocked_no_h100_dispatch"
    assert plan["non_surrogate_candidate_count"] >= 2
    assert "no non-surrogate QZS3/MQZ1/QBF1 export candidate was found" not in plan["readiness"]["blockers"]
    packed = next(item for item in plan["candidates"] if item["kind"] == "qfaithful_packed_renderer_export")
    assert packed["known_exact_negative"] is True
    assert packed["exact_eval_records"][0]["score"] == 22.07
    assert "matching Q-FAITHFUL archive has exact CUDA negative/collapse evidence" in packed["blockers"]
    raw = next(item for item in plan["candidates"] if item["kind"] == "qfaithful_raw_export_requires_packed_preflight")
    assert raw["wire_format"] == "QFAI"
    assert raw["non_surrogate_export"] is True


def test_ready_non_surrogate_preflight_is_only_h100_command_source(tmp_path: Path) -> None:
    planner = _load_planner("_plan_trained_renderer_export_unlock_ready_test")
    _write_json(tmp_path / "ready" / "trained_renderer_blockfp_preflight.json", _ready_preflight_payload())
    _write_json(
        tmp_path / "blocked" / "trained_renderer_blockfp_preflight.json",
        _surrogate_preflight_payload(),
    )

    plan = planner.build_plan(scan_dirs=(tmp_path,))

    assert plan["readiness"]["verdict"] == "h100_ready_after_claim"
    commands = plan["readiness"]["h100_lightning_commands"]
    assert commands["remote_gpu_dispatch_performed"] is False
    assert "--machine" in commands["lightning_exact_eval_submit_command_shape"]
    assert "g7e.4xlarge" in commands["lightning_exact_eval_submit_command_shape"]
    selected = plan["readiness"]["selected_preflight_candidate"]
    assert selected["candidate_id"] == "trained_qbf1_b0064"
    assert selected["archive_bytes"] == 250000


def test_legacy_ready_preflight_without_pose_safety_is_blocked(tmp_path: Path) -> None:
    planner = _load_planner("_plan_trained_renderer_export_unlock_legacy_gate_test")
    _write_json(
        tmp_path / "legacy" / "trained_renderer_blockfp_preflight.json",
        _legacy_ready_preflight_without_pose_safety_payload(),
    )

    plan = planner.build_plan(scan_dirs=(tmp_path,))

    assert plan["readiness"]["verdict"] == "blocked_no_h100_dispatch"
    candidate = next(item for item in plan["candidates"] if item["kind"] == "preflight_summary")
    assert "preflight summary predates mandatory renderer pose-safety gate" in candidate["blockers"]
    assert "renderer transplant pose-safety gate missing or failed" in candidate["blockers"]


def test_byte_targets_match_c067_frontier_formula(tmp_path: Path) -> None:
    planner = _load_planner("_plan_trained_renderer_export_unlock_targets_test")
    _write_json(
        tmp_path / "trained_renderer_blockfp_preflight.json",
        _surrogate_preflight_payload(),
    )

    plan = planner.build_plan(scan_dirs=(tmp_path,))
    targets = {
        item["target_score"]: item
        for item in plan["byte_targets"]["targets"]
    }

    assert targets[0.30]["unchanged_component_max_archive_bytes"] == 252760
    assert targets[0.30]["required_archive_byte_savings_vs_c067_if_components_unchanged"] == 23454
    assert targets[0.24]["unchanged_component_max_archive_bytes"] == 162650
    assert targets[0.24]["required_archive_byte_savings_vs_c067_if_components_unchanged"] == 113564

    preflight = next(item for item in plan["candidates"] if item["kind"] == "preflight_summary")
    reqs = {
        item["target_score"]: item
        for item in preflight["stacking_requirements"]["targets"]
    }
    assert reqs[0.30]["additional_archive_byte_savings_needed_if_components_unchanged"] == 31109
    assert reqs[0.24]["additional_archive_byte_savings_needed_if_components_unchanged"] == 121219


def test_write_plan_is_deterministic(tmp_path: Path) -> None:
    planner = _load_planner("_plan_trained_renderer_export_unlock_write_test")
    _write_json(
        tmp_path / "trained_renderer_blockfp_preflight.json",
        _surrogate_preflight_payload(),
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    plan_a = planner.write_plan(first, scan_dirs=(tmp_path,))
    plan_b = planner.write_plan(second, scan_dirs=(tmp_path,))

    assert plan_a == plan_b
    assert first.read_bytes() == second.read_bytes()
    loaded = json.loads(first.read_text())
    assert loaded["planning_constraints"]["deterministic_json"] is True
