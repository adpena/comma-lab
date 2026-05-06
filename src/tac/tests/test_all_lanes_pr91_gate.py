from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location("all_lanes_preflight_under_test", ALL_LANES)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tool_manifest(hash_value: str) -> dict[str, object]:
    return {"canonical_payload_without_tool_manifest_sha256": hash_value}


def _readiness_payload(hash_value: str = "readiness-hash") -> dict[str, object]:
    return {
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "tool_run_manifest": _tool_manifest(hash_value),
        "source_archive": {"matches_expected": True},
        "member_x": {
            "matches_expected": True,
            "zip_report": {"wire_contract": {"passed": True}},
        },
        "hpm1_mask_segment": {"matches_expected": True},
        "runtime_source_inventory": {
            "status": "passed_static_source_inventory",
            "required_source_files_present": True,
            "required_source_files": ["inflate.py", "pr86_hpac.py"],
            "missing_required_source_files": [],
            "pycache_only": False,
            "files": [
                {"path": "inflate.py", "bytes": 10, "sha256": "a" * 64},
                {"path": "pr86_hpac.py", "bytes": 10, "sha256": "b" * 64},
            ],
            "source_files": [
                {"path": "inflate.py", "bytes": 10, "sha256": "a" * 64},
                {"path": "pr86_hpac.py", "bytes": 10, "sha256": "b" * 64},
            ],
        },
        "dispatch_blockers": [
            "byte_exact_hpm1_reencode",
            "exact_cuda_auth_eval_after_parity",
            "full_hpm1_decode_600_frames",
            "runtime_hpm1_loader_without_sidecars",
        ],
    }


def _runtime_payload(hash_value: str = "runtime-hash") -> dict[str, object]:
    return {
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "tool_run_manifest": _tool_manifest(hash_value),
        "ambient_device_call_count": 2,
        "contradiction_count": 1,
        "dispatch_blockers": [
            "hpac_device_contract_resolved",
            "runtime_consumer_sidecar_free_hpm1",
        ],
    }


def test_pr91_hpm1_gate_passes_current_live_artifacts() -> None:
    module = _load_all_lanes_module()

    passed, output = module._run_pr91_hpm1_fail_closed_gate()

    assert passed is True
    assert "ready_for_exact_eval_dispatch=false" in output
    assert "artifact_hashes_match=true" in output


def test_pr91_hpm1_gate_rejects_artifact_hash_drift(monkeypatch) -> None:
    module = _load_all_lanes_module()

    def fake_json_tool(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in path.name:
            return True, _runtime_payload(), ""
        return True, _readiness_payload(), ""

    def fake_load_artifact(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in str(path):
            return True, _runtime_payload(), ""
        return True, _readiness_payload("stale-readiness-hash"), ""

    monkeypatch.setattr(module, "_json_tool", fake_json_tool)
    monkeypatch.setattr(module, "_load_json_artifact", fake_load_artifact)

    passed, output = module._run_pr91_hpm1_fail_closed_gate()

    assert passed is False
    assert "readiness_artifact_hash_matches_live" in output


def test_pr91_hpm1_gate_rejects_tampered_artifact_body_with_copied_hash(monkeypatch) -> None:
    module = _load_all_lanes_module()
    tampered_artifact = _readiness_payload()
    tampered_artifact["evidence_grade"] = "tampered_without_rehash"

    def fake_json_tool(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in path.name:
            return True, _runtime_payload(), ""
        return True, _readiness_payload(), ""

    def fake_load_artifact(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in str(path):
            return True, _runtime_payload(), ""
        return True, tampered_artifact, ""

    monkeypatch.setattr(module, "_json_tool", fake_json_tool)
    monkeypatch.setattr(module, "_load_json_artifact", fake_load_artifact)

    passed, output = module._run_pr91_hpm1_fail_closed_gate()

    assert passed is False
    assert "artifact_readiness_manifest_hash_self_consistent" in output
    assert "readiness_artifact_hash_matches_live" in output


def test_pr91_hpm1_gate_rejects_dispatch_ready_payload(monkeypatch) -> None:
    module = _load_all_lanes_module()
    readiness = _readiness_payload()
    readiness["ready_for_exact_eval_dispatch"] = True

    def fake_json_tool(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in path.name:
            return True, _runtime_payload(), ""
        return True, readiness, ""

    def fake_load_artifact(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in str(path):
            return True, _runtime_payload(), ""
        return True, readiness, ""

    monkeypatch.setattr(module, "_json_tool", fake_json_tool)
    monkeypatch.setattr(module, "_load_json_artifact", fake_load_artifact)

    passed, output = module._run_pr91_hpm1_fail_closed_gate()

    assert passed is False
    assert "live_readiness_ready_false" in output
    assert "artifact_readiness_ready_false" in output


def test_pr91_hpm1_gate_rejects_missing_required_blocker(monkeypatch) -> None:
    module = _load_all_lanes_module()
    readiness = _readiness_payload()
    readiness["dispatch_blockers"] = [
        item for item in readiness["dispatch_blockers"] if item != "full_hpm1_decode_600_frames"
    ]

    def fake_json_tool(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in path.name:
            return True, _runtime_payload(), ""
        return True, readiness, ""

    def fake_load_artifact(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in str(path):
            return True, _runtime_payload(), ""
        return True, readiness, ""

    monkeypatch.setattr(module, "_json_tool", fake_json_tool)
    monkeypatch.setattr(module, "_load_json_artifact", fake_load_artifact)

    passed, output = module._run_pr91_hpm1_fail_closed_gate()

    assert passed is False
    assert "live_readiness_required_blockers_present" in output
    assert "artifact_readiness_required_blockers_present" in output


def test_pr91_hpm1_gate_rejects_pycache_only_runtime_inventory(monkeypatch) -> None:
    module = _load_all_lanes_module()
    readiness = _readiness_payload()
    readiness["runtime_source_inventory"] = {
        "status": "failed_closed_missing_required_runtime_sources",
        "required_source_files_present": False,
        "required_source_files": ["inflate.py", "pr86_hpac.py"],
        "missing_required_source_files": ["inflate.py", "pr86_hpac.py"],
        "pycache_only": True,
        "files": [
            {
                "path": "__pycache__/inflate.cpython-312.pyc",
                "bytes": 10,
                "sha256": "c" * 64,
            }
        ],
        "source_files": [],
    }

    def fake_json_tool(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in path.name:
            return True, _runtime_payload(), ""
        return True, readiness, ""

    def fake_load_artifact(path: Path) -> tuple[bool, dict[str, object], str]:
        if "runtime" in str(path):
            return True, _runtime_payload(), ""
        return True, readiness, ""

    monkeypatch.setattr(module, "_json_tool", fake_json_tool)
    monkeypatch.setattr(module, "_load_json_artifact", fake_load_artifact)

    passed, output = module._run_pr91_hpm1_fail_closed_gate()

    assert passed is False
    assert "live_readiness_runtime_source_inventory_passed" in output
    assert "live_readiness_runtime_source_not_pycache_only" in output


def test_run_lane_respects_tools_without_verbose_flag(monkeypatch) -> None:
    module = _load_all_lanes_module()
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(args, *, capture_output: bool, text: bool):  # noqa: ANN001
        calls.append(list(args))
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    passed, output = module._run_lane(
        {
            "tool": Path("tool_without_verbose.py"),
            "args": ["--json"],
            "supports_verbose": False,
        },
        verbose=True,
    )

    assert passed is True
    assert output == "ok"
    assert calls
    assert "--verbose" not in calls[0]


def test_run_lane_passes_verbose_by_default(monkeypatch) -> None:
    module = _load_all_lanes_module()
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(args, *, capture_output: bool, text: bool):  # noqa: ANN001
        calls.append(list(args))
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    passed, output = module._run_lane({"tool": Path("verbose_tool.py"), "args": []}, verbose=True)

    assert passed is True
    assert output == "ok"
    assert calls
    assert calls[0][-1] == "--verbose"
