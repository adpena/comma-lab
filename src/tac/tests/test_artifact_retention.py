# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import comma_lab.artifact_retention as retention
from comma_lab.artifact_retention import (
    ArtifactRetentionError,
    build_retention_plan,
    execute_retention_plan,
    load_json_object,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[3]


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _load_compact_tool():
    spec = importlib.util.spec_from_file_location(
        "compact_experiment_artifacts_under_test",
        REPO / "tools" / "compact_experiment_artifacts.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_locality_candidate(
    root: Path,
    *,
    passed: bool = True,
    locality_root_name: str = "locality_work",
    manifest_name: str = "locality_controls.json",
) -> Path:
    candidate = root / "candidate_a"
    inflated = candidate / locality_root_name / "selective" / "inflated"
    raw_path = inflated / "0.raw"
    _write(raw_path, b"r" * 32)
    archive_zip = candidate / "submission_dir" / "archive.zip"
    entrypoint = candidate / "submission_dir" / "inflate.sh"
    _write(archive_zip, b"zip")
    _write(entrypoint, b"#!/bin/sh\n")
    manifest = {
        "schema": "decoder_q_selective_runtime_locality_controls.v1",
        "locality_controls_passed": passed,
        "mismatch_counts": {
            "missing_raw_file_count": 0,
            "raw_size_mismatch_count": 0,
            "selected_frame_mismatch_count": 0,
            "unselected_frame_mismatch_count": 0,
        },
        "targets": {
            "selective": {
                "archive_zip": str(archive_zip),
                "entrypoint_path": str(entrypoint),
                "output_dir": str(inflated),
                "returncode": 0,
                "archive_sha256": "a" * 64,
                "entrypoint_sha256": "b" * 64,
            }
        },
        "hashes": {
            "0.raw": {
                "raw_files": {
                    "selective": sha256_file(raw_path),
                }
            }
        },
    }
    (candidate / manifest_name).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return inflated


def _write_mlx_delta_cache(root: Path, *, stamp: bool = True) -> Path:
    cache = root / "candidate_mlx" / "mlx_delta_cache"
    _write(cache / "pair_indices.npy", b"pairs")
    _write(cache / "posenet_yuv6_pair.npy", b"pose")
    _write(cache / "segnet_last_rgb.npy", b"seg")
    local_advisory = root / "candidate_mlx" / "local_cpu_advisory.json"
    local_advisory.write_text(
        json.dumps({"score_axis": "cpu_advisory", **_false_authority()}),
        encoding="utf-8",
    )
    manifest = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "raw_sha256": "c" * 64,
        "pair_count": 1,
        "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
        "array_sha256": {
            "pair_indices": "1" * 64,
            "posenet_yuv6_pair": "2" * 64,
            "segnet_last_rgb": "3" * 64,
        },
        "artifacts": {
            "pair_indices": {"sha256": sha256_file(cache / "pair_indices.npy")},
            "posenet_yuv6_pair": {"sha256": sha256_file(cache / "posenet_yuv6_pair.npy")},
            "segnet_last_rgb": {"sha256": sha256_file(cache / "segnet_last_rgb.npy")},
        },
    }
    if stamp:
        audit = {
            "schema_version": "mlx_scorer_input_cache_local_cpu_advisory_audit.v1",
            "verdict": "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
            "passed": True,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "cache": {
                "archive_sha256": manifest["archive_sha256"],
                "inflated_outputs_aggregate_sha256": manifest["inflated_outputs_aggregate_sha256"],
                "raw_sha256": manifest["raw_sha256"],
                "pair_count": manifest["pair_count"],
                "hash_domain": manifest["hash_domain"],
                "array_sha256": manifest["array_sha256"],
            },
        }
        audit_path = root / ".omx" / "research" / "mlx_cache_identity.json"
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        manifest["eligible_for_local_mlx_local_advisory_debug"] = True
        manifest["eligible_for_local_mlx_transfer_calibration"] = False
        manifest["local_cpu_advisory_cache_identity_audit"] = {
            "schema_version": audit["schema_version"],
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "verdict": audit["verdict"],
            "passed": True,
            "local_cpu_advisory_path": str(local_advisory),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache


def test_retention_deletes_only_certified_locality_raw(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path)
    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        min_bytes=1,
    )

    assert plan.total_reclaimable_bytes == 32
    assert [row.kind for row in plan.candidates] == ["locality_inflated_raw"]
    assert plan.candidates[0].certificate["raw_sha256"] == sha256_file(inflated / "0.raw")

    execution = execute_retention_plan(plan, action="delete")

    assert execution["executed_count"] == 1
    assert not inflated.exists()
    assert (tmp_path / "candidate_a" / "locality_controls.json").is_file()
    assert (tmp_path / "candidate_a" / "submission_dir" / "archive.zip").is_file()


def test_compact_cli_execute_stdout_writes_default_journal(
    tmp_path: Path,
    capsys,
) -> None:
    inflated = _write_locality_candidate(tmp_path)
    tool = _load_compact_tool()

    rc = tool.main(
        [
            str(tmp_path),
            "--repo-root",
            str(tmp_path),
            "--min-bytes",
            "1",
            "--execute",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    journal_path = Path(payload["execution"]["journal_path"])
    assert rc == 0
    assert not inflated.exists()
    assert journal_path.is_file()
    assert journal_path.relative_to(tmp_path).parts[:3] == (
        ".omx",
        "state",
        "artifact_retention_journals",
    )
    events = [
        json.loads(line)["event"]
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["start", "candidate_start", "candidate_end"]


def test_retention_blocks_failed_locality_control(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path, passed=False)
    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        min_bytes=1,
    )

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 1
    assert "locality_controls_not_passed" in plan.blocked_candidates[0].blockers
    assert inflated.exists()


def test_retention_matches_named_locality_controls_work_manifest(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(
        tmp_path,
        locality_root_name="locality_controls_work",
        manifest_name="locality_controls_pair501.json",
    )

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert len(plan.candidates) == 1
    assert plan.candidates[0].path.endswith("locality_controls_work/selective/inflated")
    assert plan.candidates[0].certificate["manifest_path"].endswith(
        "locality_controls_pair501.json"
    )
    assert inflated.exists()


def test_retention_moves_local_cpu_advisory_scratch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    work = repo / "candidate_b" / "dqs1_pair501_cpu_advisory_work_venv"
    inflated = work / "inflated"
    _write(inflated / "0.raw", b"r" * 16)
    _write(work / "archive.zip", b"zip")
    (work / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                **_false_authority(),
                "score_claim_valid": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "n_samples": 600,
            }
        ),
        encoding="utf-8",
    )
    (work / "inflated_outputs_manifest.json").write_text(
        json.dumps(
            {
                "payload": {
                    "files": [
                        {
                            "path": "0.raw",
                            "sha256": sha256_file(inflated / "0.raw"),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (work / "provenance.json").write_text(
        json.dumps({"command": ["inflate"]}),
        encoding="utf-8",
    )
    cold_store = tmp_path / "cold"
    cold_store.mkdir()

    plan = build_retention_plan(
        [repo / "candidate_b"],
        repo_root=repo,
        min_bytes=1,
    )
    execution = execute_retention_plan(plan, action="move", cold_store_root=cold_store)

    assert execution["executed_count"] == 1
    assert execution["cold_store_contract"]["write_probe_passed"] is True
    assert not inflated.exists()
    moved = cold_store / plan.candidates[0].path
    assert (moved / "0.raw").read_bytes() == b"r" * 16
    assert execution["rows"][0]["cold_store_verification"]["source_digest"]["sha256"]
    assert load_json_object(work / "contest_auth_eval.json")["n_samples"] == 600


def test_retention_move_rejects_cold_store_inside_repo(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path)
    cold_store = tmp_path / "cold"
    cold_store.mkdir()
    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    with pytest.raises(ArtifactRetentionError, match="outside repo_root"):
        execute_retention_plan(plan, action="move", cold_store_root=cold_store)

    assert inflated.exists()


def test_retention_move_copy_failure_journals_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    inflated = _write_locality_candidate(repo)
    cold_store = tmp_path / "cold"
    cold_store.mkdir()
    journal = tmp_path / "retention.journal.jsonl"
    plan = build_retention_plan([repo], repo_root=repo, min_bytes=1)
    original_digest = retention.directory_digest

    def mismatching_destination_digest(path: Path) -> dict[str, object]:
        digest = original_digest(path)
        if ".partial-" in path.name:
            digest = dict(digest)
            digest["sha256"] = "0" * 64
        return digest

    monkeypatch.setattr(retention, "directory_digest", mismatching_destination_digest)

    with pytest.raises(ArtifactRetentionError, match="copy verification failed"):
        execute_retention_plan(
            plan,
            action="move",
            cold_store_root=cold_store,
            journal_path=journal,
        )

    assert inflated.exists()
    assert not any(cold_store.rglob("*.partial-*"))
    events = [
        json.loads(line)["event"]
        for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["start", "candidate_start", "candidate_error", "candidate_end"]


def test_retention_blocks_mutated_locality_raw_after_manifest(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path)
    (inflated / "0.raw").write_bytes(b"mutated")

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 1
    assert "locality_raw_sha_mismatch:0.raw" in plan.blocked_candidates[0].blockers


def test_retention_certifies_mlx_cache_with_external_identity_stamp(tmp_path: Path) -> None:
    cache = _write_mlx_delta_cache(tmp_path)

    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        include_kinds={"mlx_scorer_input_cache"},
        min_bytes=1,
    )

    assert plan.blocked_candidates == []
    assert len(plan.candidates) == 1
    candidate = plan.candidates[0]
    assert candidate.path == cache.relative_to(tmp_path).as_posix()
    assert candidate.kind == "mlx_scorer_input_cache"
    identity = candidate.certificate["identity_audit"]
    assert identity["stamp_key"] == "local_cpu_advisory_cache_identity_audit"
    assert identity["source"]["key"] == "local_cpu_advisory_path"
    assert candidate.certificate["archive_sha256"] == "a" * 64


def test_retention_blocks_mlx_cache_without_identity_stamp(tmp_path: Path) -> None:
    cache = _write_mlx_delta_cache(tmp_path, stamp=False)

    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        include_kinds={"mlx_scorer_input_cache"},
        min_bytes=1,
    )

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 1
    assert plan.blocked_candidates[0].path == cache.relative_to(tmp_path).as_posix()
    assert "mlx_cache_identity_audit_stamp_missing" in plan.blocked_candidates[0].blockers


def test_retention_reports_unknown_raw_surface(tmp_path: Path) -> None:
    raw_dir = tmp_path / "candidate_c" / "local_macos_cpu_eval_work" / "inflated"
    _write(raw_dir / "0.raw", b"r" * 8)

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert any(
        row.kind == "blocked_unknown_raw_surface"
        and "unknown_raw_surface_no_certifier" in row.blockers
        for row in plan.blocked_candidates
    )


def test_retention_reports_known_nested_raw_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "candidate_d" / "contest_auth_eval_cpu_workdir"
    raw_dir = workdir / "nested" / "inflated"
    _write(raw_dir / "0.raw", b"r" * 8)

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert any(
        row.path == workdir.relative_to(tmp_path).as_posix()
        and row.kind == "blocked_unknown_raw_surface"
        and "unknown_raw_workdir_no_certifier" in row.blockers
        and row.certificate["nested_raw_file_count"] == 1
        for row in plan.blocked_candidates
    )
