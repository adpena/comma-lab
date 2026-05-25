# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from comma_lab.scheduler.frontier_rate_attack_bootstrap import (
    BOOTSTRAP_SCHEMA,
    FrontierRateAttackBootstrapError,
    archive_record,
    build_frontier_rate_attack_payloads,
    resolve_current_frontier_archive,
)
from tac.repo_io import sha256_file


def _write_zip(path: Path, *, member: str = "x", payload: bytes = b"payload") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_build_frontier_rate_attack_queue_compiles_packet_sweeps(tmp_path: Path) -> None:
    first = tmp_path / "archives" / "frontier" / "archive.zip"
    second = tmp_path / "archives" / "pr110" / "archive.zip"
    _write_zip(first, payload=b"frontier-bytes")
    _write_zip(second, payload=b"pr110-bytes")
    records = [
        archive_record(label="current_cpu", archive_path=first, repo_root=tmp_path, source_kind="test"),
        archive_record(label="pr110_base", archive_path=second, repo_root=tmp_path, source_kind="test"),
    ]

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_attack_test",
        archive_records=records,
        results_root=tmp_path / "results",
        local_cpu_concurrency=3,
    )

    assert payloads["bootstrap"]["schema"] == BOOTSTRAP_SCHEMA
    assert payloads["bootstrap"]["archive_count"] == 2
    assert payloads["bootstrap"]["executable_target_kinds"] == [
        "packet_member_zip_header_elide_v1",
        "packet_member_recompress_v1",
    ]
    assert {row["target_kind"] for row in payloads["bootstrap"]["target_omissions"]} == {
        "archive_section_entropy_recode_v1",
        "tensor_factorize_v1",
    }
    queue = payloads["queue"]
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 3
    commands = [
        step["command"]
        for experiment in queue["experiments"]
        for step in experiment["steps"]
    ]
    assert len(commands) == 2
    assert all("tools/run_family_agnostic_materializer_sweep.py" in command for command in commands)
    assert all("current_cpu=archives/frontier/archive.zip" in command for command in commands)
    assert all("pr110_base=archives/pr110/archive.zip" in command for command in commands)
    assert all("--observation-jsonl" in command for command in commands)


def test_resolve_current_frontier_archive_from_auth_request(tmp_path: Path) -> None:
    archive = tmp_path / "experiments" / "results" / "candidate" / "submission_dir" / "archive.zip"
    _write_zip(archive, payload=b"frontier")
    digest = sha256_file(archive)
    pointer = {
        "our_local_frontier_contest_cpu": {
            "archive_sha256": digest,
            "score": 0.123,
            "evidence_grade": "[contest-CPU]",
            "hardware_substrate": "linux_x86_64_cpu",
            "measured_at_utc": "2026-05-25T00:00:00Z",
            "extra": {"archive_bytes": archive.stat().st_size},
        }
    }
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(pointer_path, pointer)
    request = {
        "archive_path": archive.as_posix(),
        "archive_sha256": digest,
        "archive_size_bytes": archive.stat().st_size,
    }
    request_path = (
        tmp_path
        / "experiments"
        / "results"
        / "modal_auth_eval_cpu"
        / "job"
        / "modal_cpu_auth_eval_local_request.json"
    )
    _write_json(request_path, request)

    resolution = resolve_current_frontier_archive(
        repo_root=tmp_path,
        pointer_path=pointer_path,
        frontier_axis="contest_cpu",
    )

    assert resolution["archive_sha256"] == digest
    assert resolution["archive_record"]["path"] == "experiments/results/candidate/submission_dir/archive.zip"
    assert resolution["match"]["request_path"] == (
        "experiments/results/modal_auth_eval_cpu/job/modal_cpu_auth_eval_local_request.json"
    )


def test_resolve_current_frontier_archive_fails_closed_on_duplicate_matches(
    tmp_path: Path,
) -> None:
    first = tmp_path / "experiments" / "results" / "candidate_a" / "submission_dir" / "archive.zip"
    second = tmp_path / "experiments" / "results" / "candidate_b" / "submission_dir" / "archive.zip"
    _write_zip(first, payload=b"same")
    _write_zip(second, payload=b"same")
    digest = sha256_file(first)
    assert sha256_file(second) == digest
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "extra": {"archive_bytes": first.stat().st_size},
            }
        },
    )
    for name, archive in (("a", first), ("b", second)):
        _write_json(
            tmp_path
            / "experiments"
            / "results"
            / "modal_auth_eval_cpu"
            / name
            / "request.json",
            {
                "archive_path": archive.as_posix(),
                "archive_sha256": digest,
                "archive_size_bytes": archive.stat().st_size,
            },
        )

    with pytest.raises(FrontierRateAttackBootstrapError, match="ambiguous"):
        resolve_current_frontier_archive(
            repo_root=tmp_path,
            pointer_path=pointer_path,
            frontier_axis="contest_cpu",
        )
