# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from comma_lab.artifact_retention import (
    build_retention_plan,
    execute_retention_plan,
    load_json_object,
    sha256_file,
)


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


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
    work = tmp_path / "candidate_b" / "dqs1_pair501_cpu_advisory_work_venv"
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

    plan = build_retention_plan(
        [tmp_path / "candidate_b"],
        repo_root=tmp_path,
        min_bytes=1,
    )
    execution = execute_retention_plan(plan, action="move", cold_store_root=cold_store)

    assert execution["executed_count"] == 1
    assert not inflated.exists()
    moved = cold_store / plan.candidates[0].path
    assert (moved / "0.raw").read_bytes() == b"r" * 16
    assert load_json_object(work / "contest_auth_eval.json")["n_samples"] == 600


def test_retention_blocks_mutated_locality_raw_after_manifest(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path)
    (inflated / "0.raw").write_bytes(b"mutated")

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 1
    assert "locality_raw_sha_mismatch:0.raw" in plan.blocked_candidates[0].blockers


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
