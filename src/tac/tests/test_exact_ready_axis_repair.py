# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import tac.optimizer.exact_ready_axis_repair as axis_repair
from tac.optimizer.exact_readiness import runtime_dependency_manifest
from tac.optimizer.exact_ready_axis_repair import (
    plan_exact_ready_score_axis_repairs_from_audit,
    repair_exact_ready_score_axis_queues,
)

REPO = Path(__file__).resolve().parents[3]
REPAIR_TOOL = REPO / "tools" / "repair_exact_ready_score_axis.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_claims(path: Path, rows: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows or [])
        + "\n",
        encoding="utf-8",
    )
    return path


def _axis_missing_exact_ready_queue(repo: Path, *, lane_id: str = "lane_axis") -> Path:
    submission = repo / "packet"
    submission.mkdir(parents=True)
    archive = submission / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"archive bytes")
    archive_bytes = archive.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(0o755)
    (submission / "report.txt").write_text("report\n", encoding="utf-8")
    _write_json(
        submission / "archive_manifest.json",
        {
            "archive_sha256": archive_sha,
            "archive_bytes": len(archive_bytes),
            "members": [{"name": "0.bin"}],
        },
    )
    runtime_manifest = runtime_dependency_manifest(submission, repo)
    row = {
        "candidate_id": "axis_missing_candidate",
        "lane_id": lane_id,
        "target_modes": ["contest_exact_eval"],
        "ready_for_exact_eval_dispatch": True,
        "dispatch_packet_ready": True,
        "archive_path": "packet/archive.zip",
        "candidate_archive_sha256": archive_sha,
        "archive_sha256": archive_sha,
        "archive_bytes": len(archive_bytes),
        "candidate_archive_bytes": len(archive_bytes),
        "submission_dir": "packet",
        "inflate_sh_path": "packet/inflate.sh",
        "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
        "runtime_content_tree_sha256": runtime_manifest["runtime_content_tree_sha256"],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    return _write_json(
        repo / "experiments/results/fixture/exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready_count": 1,
            "dispatch_ready": [row],
            "top_k_count": 1,
            "top_k": [row],
        },
    )


def test_repair_exact_ready_score_axis_writes_non_destructive_copy(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    queue = _axis_missing_exact_ready_queue(repo)
    claims = _write_claims(repo / ".omx/state/active_lane_dispatch_claims.md")

    report = repair_exact_ready_score_axis_queues(
        [queue],
        repo_root=repo,
        out_dir=".omx/research/axis_repair",
        dispatch_claims_path=claims,
        write_repaired_queues=True,
    )

    assert report["repairable_or_repaired_count"] == 1
    row = report["rows"][0]
    assert row["status"] == "repaired"
    assert row["source_blockers"] == ["score_axis_missing"]
    assert row["repaired_audit_stale_ready_row_count"] == 0
    repaired_path = repo / row["repaired_queue_path"]
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    assert repaired["score_axis_repair"]["source_queue_path"] == (
        "experiments/results/fixture/exact_ready_queue.json"
    )
    assert repaired["dispatch_ready"][0]["score_axis"] == "contest_cuda"
    assert repaired["dispatch_ready"][0]["target_score_axis"] == "contest_cuda"
    source = json.loads(queue.read_text(encoding="utf-8"))
    assert "score_axis" not in source["dispatch_ready"][0]
    assert report["dispatch_attempted"] is False
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_repair_exact_ready_score_axis_backfills_legacy_runtime_custody_shape(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    queue = _axis_missing_exact_ready_queue(repo)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["dispatch_ready"][0]
    row["runtime_manifest"] = {
        "runtime_tree_sha256": row.pop("runtime_tree_sha256"),
        "runtime_content_tree_sha256": row.pop("runtime_content_tree_sha256"),
    }
    proof = repo / "packet/runtime_consumption_proof.json"
    _write_json(proof, {"schema": "fixture_unsupported_but_present"})
    runtime_manifest = runtime_dependency_manifest(repo / "packet", repo)
    row["runtime_manifest"] = {
        "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
        "runtime_content_tree_sha256": runtime_manifest["runtime_content_tree_sha256"],
    }
    row["runtime_consumption_proof_path"] = "packet/runtime_consumption_proof.json"
    payload["top_k"][0] = dict(row)
    _write_json(queue, payload)
    claims = _write_claims(repo / ".omx/state/active_lane_dispatch_claims.md")

    report = repair_exact_ready_score_axis_queues(
        [queue],
        repo_root=repo,
        out_dir=".omx/research/axis_repair",
        dispatch_claims_path=claims,
        write_repaired_queues=True,
    )

    repaired_path = repo / report["rows"][0]["repaired_queue_path"]
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    repaired_row = repaired["dispatch_ready"][0]
    assert repaired_row["runtime_tree_sha256"] == row["runtime_manifest"][
        "runtime_tree_sha256"
    ]
    assert repaired_row["runtime_content_tree_sha256"] == row["runtime_manifest"][
        "runtime_content_tree_sha256"
    ]
    assert repaired_row["runtime_consumption_proof_required"] is True
    assert repaired_row["runtime_consumption_proof_status"] == "present"


def test_repair_exact_ready_score_axis_skips_terminal_negative_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    queue = _axis_missing_exact_ready_queue(repo, lane_id="lane_terminal")
    source = json.loads(queue.read_text(encoding="utf-8"))
    archive_sha = source["dispatch_ready"][0]["archive_sha256"]
    claims = _write_claims(
        repo / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-24T00:00:00Z | test | lane_terminal | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=9.9 |"
        ],
    )

    report = repair_exact_ready_score_axis_queues(
        [queue],
        repo_root=repo,
        out_dir=".omx/research/axis_repair",
        dispatch_claims_path=claims,
        write_repaired_queues=True,
    )

    assert report["repairable_or_repaired_count"] == 0
    row = report["rows"][0]
    assert row["status"] == "skipped"
    assert row["skip_reason"] == "terminal_or_duplicate_exact_eval_evidence"
    assert "score_axis_missing" in row["source_blockers"]
    assert any(
        blocker.startswith("same_lane_terminal_negative_for_same_archive")
        for blocker in row["source_blockers"]
    )


def test_repair_exact_ready_score_axis_counts_still_blocked_repairs_as_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    queue = _write_json(
        repo / "experiments/results/fixture/exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "still_blocked_candidate",
                    "target_modes": ["contest_exact_eval"],
                    "archive_sha256": "a" * 64,
                }
            ],
            "top_k": [
                {
                    "candidate_id": "still_blocked_candidate",
                    "target_modes": ["contest_exact_eval"],
                    "archive_sha256": "a" * 64,
                }
            ],
        },
    )
    calls: list[Path] = []

    def fake_audit(path: Path, **_: object) -> dict[str, object]:
        calls.append(Path(path))
        if len(calls) == 1:
            return {"stale_ready_rows": [{"blockers": ["score_axis_missing"]}]}
        return {
            "stale_ready_rows": [
                {"blockers": ["ready_row_runtime_content_tree_sha256_missing_or_invalid"]}
            ]
        }

    monkeypatch.setattr(axis_repair, "audit_exact_ready_queue", fake_audit)
    claims = _write_claims(repo / ".omx/state/active_lane_dispatch_claims.md")

    report = repair_exact_ready_score_axis_queues(
        [queue],
        repo_root=repo,
        out_dir=".omx/research/axis_repair",
        dispatch_claims_path=claims,
        write_repaired_queues=True,
    )

    assert report["repairable_or_repaired_count"] == 0
    assert report["skipped_count"] == 1
    assert report["rows"][0]["status"] == "repair_still_blocked"
    assert report["rows"][0]["repaired_audit_stale_ready_row_count"] == 1


def test_plan_exact_ready_score_axis_repairs_from_audit_is_advisory() -> None:
    payload = {
        "schema": "optimizer_exact_ready_queue_terminal_evidence_audit_v1",
        "queue_count": 1,
        "stale_ready_row_count": 1,
        "queues": [
            {
                "queue_path": "experiments/results/fixture/exact_ready_queue.json",
                "stale_ready_rows": [
                    {
                        "candidate_id": "candidate",
                        "lane_id": "lane_axis",
                        "archive_sha256": "a" * 64,
                        "runtime_tree_sha256": "b" * 64,
                        "runtime_content_tree_sha256": "c" * 64,
                        "blockers": ["score_axis_missing"],
                    }
                ],
            }
        ],
    }

    plan = plan_exact_ready_score_axis_repairs_from_audit(payload)

    assert plan["repairable_or_repaired_count"] == 1
    assert plan["automatic_mutation_count"] == 0
    row = plan["rows"][0]
    assert row["status"] == "repairable"
    assert row["proposed_fields"] == {
        "score_axis": "contest_cuda",
        "target_score_axis": "contest_cuda",
    }
    assert row["automatic_mutation_allowed"] is False
    assert row["dispatch_attempted"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_plan_exact_ready_score_axis_terminal_evidence_beats_repair() -> None:
    payload = {
        "schema": "optimizer_exact_ready_queue_terminal_evidence_audit_v1",
        "queue_count": 1,
        "stale_ready_row_count": 1,
        "queues": [
            {
                "queue_path": "experiments/results/fixture/exact_ready_queue.json",
                "stale_ready_rows": [
                    {
                        "candidate_id": "candidate",
                        "lane_id": "lane_axis",
                        "archive_sha256": "a" * 64,
                        "blockers": [
                            "score_axis_missing",
                            "same_lane_terminal_negative_for_same_archive:"
                            "lane_axis:job1:completed_contest_cuda_auth_eval_negative",
                        ],
                    }
                ],
            }
        ],
    }

    plan = plan_exact_ready_score_axis_repairs_from_audit(payload)

    assert plan["repairable_or_repaired_count"] == 0
    row = plan["rows"][0]
    assert row["status"] == "skipped"
    assert row["skip_reason"] == "terminal_or_duplicate_exact_eval_evidence"
    assert row["proposed_fields"] is None
    assert row["ready_for_exact_eval_dispatch"] is False


def test_plan_exact_ready_score_axis_custody_blocker_requires_regeneration() -> None:
    payload = {
        "schema": "optimizer_exact_ready_queue_terminal_evidence_audit_v1",
        "queue_count": 1,
        "stale_ready_row_count": 1,
        "queues": [
            {
                "queue_path": "experiments/results/fixture/exact_ready_queue.json",
                "stale_ready_rows": [
                    {
                        "candidate_id": "candidate",
                        "lane_id": "lane_axis",
                        "archive_sha256": "a" * 64,
                        "blockers": [
                            "score_axis_missing",
                            "ready_row_runtime_content_tree_sha256_missing_or_invalid",
                        ],
                    }
                ],
            }
        ],
    }

    plan = plan_exact_ready_score_axis_repairs_from_audit(payload)

    assert plan["repairable_or_repaired_count"] == 0
    assert plan["rows"][0]["skip_reason"] == "live_custody_regeneration_required"


def test_repair_exact_ready_score_axis_rejects_unsupported_axis(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    queue = _axis_missing_exact_ready_queue(repo)
    claims = _write_claims(repo / ".omx/state/active_lane_dispatch_claims.md")

    with pytest.raises(ValueError, match="only supports"):
        repair_exact_ready_score_axis_queues(
            [queue],
            repo_root=repo,
            out_dir=".omx/research/axis_repair",
            dispatch_claims_path=claims,
            score_axis="contest_cpu",
        )


def test_repair_cli_default_scan_ignores_prior_repair_outputs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _axis_missing_exact_ready_queue(repo)
    _write_claims(repo / ".omx/state/active_lane_dispatch_claims.md")

    subprocess.run(
        [
            sys.executable,
            str(REPAIR_TOOL),
            "--repo-root",
            str(repo),
            "--out-dir",
            ".omx/research/exact_ready_score_axis_repair_first",
            "--report-out",
            ".omx/research/first_report.json",
            "--write-repaired-queues",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPAIR_TOOL),
            "--repo-root",
            str(repo),
            "--out-dir",
            ".omx/research/exact_ready_score_axis_repair_second",
            "--report-out",
            ".omx/research/second_report.json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert proc.stdout == ""
    report = json.loads((repo / ".omx/research/second_report.json").read_text())
    assert report["queue_count"] == 1
    assert report["repairable_or_repaired_count"] == 1
