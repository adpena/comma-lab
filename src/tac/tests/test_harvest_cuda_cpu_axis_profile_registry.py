# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "harvest_cuda_cpu_axis_profile_registry.py"
    spec = importlib.util.spec_from_file_location(
        "harvest_cuda_cpu_axis_profile_registry", tool_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _auth_payload(
    *,
    device: str,
    score: float,
    pose: float,
    seg: float,
    archive_sha256: str = "a" * 64,
    archive_bytes: int = 178_258,
    runtime_tree_sha256: str = "b" * 64,
    platform_system: str = "Linux",
    platform_machine: str = "x86_64",
    gpu_t4_match: bool = False,
    inflated_output_aggregate_sha256: str | None = None,
) -> dict:
    payload = {
        "score_recomputed_from_components": score,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "runtime_tree_sha256": runtime_tree_sha256,
        "n_samples": 600,
        "hardware": (
            "t4-sm75"
            if device == "cuda"
            else f"{platform_system}-{platform_machine}"
        ),
        "evidence_grade": "contest-CUDA" if device == "cuda" else "contest-CPU",
        "provenance": {
            "device": device,
            "gpu_t4_match": gpu_t4_match,
            "platform_system": platform_system,
            "platform_machine": platform_machine,
            "runtime_tree_sha256": runtime_tree_sha256,
        },
    }
    if inflated_output_aggregate_sha256 is not None:
        payload["provenance"]["inflated_output_manifest"] = {
            "path": f"{device}/inflated_outputs_manifest.json",
            "sha256": f"{device}-manifest-sha",
            "payload": {
                "aggregate_sha256": inflated_output_aggregate_sha256,
                "raw_file_count": 1,
                "total_bytes": 603_979_776,
            },
        }
    return payload


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_pair_builder_requires_same_archive_and_runtime_custody(tmp_path: Path) -> None:
    mod = _load_tool()
    cpu_json = _write_json(
        tmp_path / "cpu.json",
        _auth_payload(
            device="cpu",
            score=0.196,
            pose=3.4e-5,
            seg=5.7e-4,
        ),
    )
    cuda_json = _write_json(
        tmp_path / "cuda.json",
        _auth_payload(
            device="cuda",
            score=0.229,
            pose=1.7e-4,
            seg=6.7e-4,
            gpu_t4_match=True,
        ),
    )

    combined, blockers = mod.build_combined_payload_from_pair(
        cpu_json=cpu_json,
        cuda_json=cuda_json,
        architecture_class="hnerv_ft_microcodec",
    )

    assert blockers == []
    assert combined is not None
    assert combined["sample_count"] == 600
    assert combined["cpu"]["score_axis"] == "contest_cpu"
    assert combined["cuda"]["score_axis"] == "contest_cuda"
    assert combined["inflated_outputs"]["raw_output_pairing_status"] == "raw_output_manifest_missing"
    assert combined["inflated_outputs"]["mechanism_analysis_complete"] is False
    assert combined["inflated_outputs"]["mechanism_blockers"] == ["raw_output_manifest_missing"]
    assert combined["score_claim"] is False


def test_pair_builder_preserves_different_raw_output_hashes_as_xray_signal(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    cpu_json = _write_json(
        tmp_path / "cpu.json",
        _auth_payload(
            device="cpu",
            score=0.196,
            pose=3.4e-5,
            seg=5.7e-4,
            inflated_output_aggregate_sha256="1" * 64,
        ),
    )
    cuda_json = _write_json(
        tmp_path / "cuda.json",
        _auth_payload(
            device="cuda",
            score=0.229,
            pose=1.7e-4,
            seg=6.7e-4,
            gpu_t4_match=True,
            inflated_output_aggregate_sha256="2" * 64,
        ),
    )

    combined, blockers = mod.build_combined_payload_from_pair(
        cpu_json=cpu_json,
        cuda_json=cuda_json,
        architecture_class="hnerv_ft_microcodec",
    )

    assert blockers == []
    assert combined is not None
    assert combined["inflated_outputs"]["same_inflated_output_aggregate_sha256"] is False
    assert combined["inflated_outputs"]["raw_output_pairing_status"] == "different_inflated_outputs"
    assert combined["inflated_outputs"]["mechanism_analysis_complete"] is True
    assert combined["inflated_outputs"]["mechanism_blockers"] == []
    assert combined["inflated_outputs"]["cpu"]["aggregate_sha256"] == "1" * 64


def test_harvest_updates_registry_and_audit_log_from_explicit_pair(tmp_path: Path) -> None:
    mod = _load_tool()
    cpu_json = _write_json(
        tmp_path / "cpu.json",
        _auth_payload(
            device="cpu",
            score=0.196,
            pose=3.4e-5,
            seg=5.7e-4,
        ),
    )
    cuda_json = _write_json(
        tmp_path / "cuda.json",
        _auth_payload(
            device="cuda",
            score=0.229,
            pose=1.7e-4,
            seg=6.7e-4,
            gpu_t4_match=True,
        ),
    )
    registry = tmp_path / "registry.json"
    audit_log = tmp_path / "updates.jsonl"

    report = mod.harvest_registry(
        combined_payloads=[],
        pairs=[(cpu_json, cuda_json)],
        registry_path=registry,
        audit_log_path=audit_log,
        architecture_class="hnerv_ft_microcodec",
        dry_run=False,
    )

    assert report["update_count"] == 1
    assert report["blockers"] == []
    assert registry.exists()
    assert audit_log.exists()
    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    profile = registry_payload["profiles"]["hnerv_ft_microcodec"]
    assert profile["n_anchors"] == 6
    assert registry_payload["score_claim"] is False
    audit_records = [json.loads(line) for line in audit_log.read_text().splitlines()]
    assert audit_records[0]["accepted"] is True


def test_macos_cpu_artifact_is_rejected_for_contest_cpu_learning_pair(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    cpu_json = _write_json(
        tmp_path / "macos_cpu.json",
        _auth_payload(
            device="cpu",
            score=0.196,
            pose=3.4e-5,
            seg=5.7e-4,
            platform_system="Darwin",
            platform_machine="arm64",
        ),
    )
    cuda_json = _write_json(
        tmp_path / "cuda.json",
        _auth_payload(
            device="cuda",
            score=0.229,
            pose=1.7e-4,
            seg=6.7e-4,
            gpu_t4_match=True,
        ),
    )

    combined, blockers = mod.build_combined_payload_from_pair(
        cpu_json=cpu_json,
        cuda_json=cuda_json,
    )

    assert combined is None
    assert any("expected contest_cpu" in blocker for blocker in blockers)


def test_scan_pairs_by_archive_and_runtime_sha(tmp_path: Path) -> None:
    mod = _load_tool()
    root = tmp_path / "results"
    (root / "cpu").mkdir(parents=True)
    (root / "cuda").mkdir(parents=True)
    cpu_json = _write_json(
        root / "cpu" / "contest_auth_eval.adjudicated.json",
        _auth_payload(
            device="cpu",
            score=0.196,
            pose=3.4e-5,
            seg=5.7e-4,
        ),
    )
    cuda_json = _write_json(
        root / "cuda" / "contest_auth_eval.adjudicated.json",
        _auth_payload(
            device="cuda",
            score=0.229,
            pose=1.7e-4,
            seg=6.7e-4,
            gpu_t4_match=True,
        ),
    )

    pairs, blockers = mod.discover_pairs([root])

    assert blockers == []
    assert pairs == [(cpu_json, cuda_json)]
