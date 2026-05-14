from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "audit_exact_ready_queues.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_cli_fails_on_stale_exact_ready_queue(tmp_path: Path) -> None:
    archive_sha = "a" * 64
    queue = _write_json(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "candidate",
                    "lane_id": "lane",
                    "ready_for_exact_eval_dispatch": True,
                    "candidate_archive_sha256": archive_sha,
                    "archive_bytes": 123,
                }
            ],
        },
    )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--queue",
            str(queue),
            "--dispatch-claims-path",
            str(claims),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["passed"] is False
    assert payload["stale_ready_row_count"] == 1


def test_cli_warn_only_writes_report_and_exits_zero(tmp_path: Path) -> None:
    archive_sha = "b" * 64
    _write_json(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "candidate",
                    "lane_id": "lane",
                    "ready_for_exact_eval_dispatch": True,
                    "candidate_archive_sha256": archive_sha,
                    "archive_bytes": 123,
                }
            ],
        },
    )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--scan-root",
            "experiments/results",
            "--dispatch-claims-path",
            str(claims),
            "--format",
            "json",
            "--output",
            str(report),
            "--warn-only",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["stale_ready_row_count"] == 1


def test_cli_default_scan_includes_experiments_and_omx_research(tmp_path: Path) -> None:
    for rel, archive_sha in (
        ("experiments/results/fixture/exact_ready_queue.json", "e" * 64),
        (".omx/research/fixture/research_exact_ready_queue.json", "f" * 64),
    ):
        _write_json(
            tmp_path / rel,
            {
                "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
                "dispatch_ready": [
                    {
                        "candidate_id": rel,
                        "lane_id": "lane",
                        "ready_for_exact_eval_dispatch": True,
                        "candidate_archive_sha256": archive_sha,
                        "archive_bytes": 123,
                    }
                ],
            },
        )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--dispatch-claims-path",
            str(claims),
            "--format",
            "json",
            "--warn-only",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["queue_count"] == 2
    assert {
        queue["queue_path"]
        for queue in payload["queues"]
    } == {
        "experiments/results/fixture/exact_ready_queue.json",
        ".omx/research/fixture/research_exact_ready_queue.json",
    }


def test_cli_writes_suppression_manifest_and_reports_zero_unresolved(
    tmp_path: Path,
) -> None:
    archive_sha = "c" * 64
    queue = _write_json(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "candidate",
                    "lane_id": "lane",
                    "ready_for_exact_eval_dispatch": True,
                    "candidate_archive_sha256": archive_sha,
                    "archive_bytes": 123,
                }
            ],
        },
    )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |\n",
        encoding="utf-8",
    )
    manifest = tmp_path / ".omx/research/exact_ready_suppressions.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--queue",
            str(queue),
            "--dispatch-claims-path",
            str(claims),
            "--format",
            "json",
            "--write-suppression-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["passed"] is True
    assert payload["raw_stale_ready_row_count"] == 1
    assert payload["suppressed_ready_row_count"] == 1
    assert payload["stale_ready_row_count"] == 0
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["suppression_entry_count"] == 1
    entry = saved["entries"][0]
    assert entry["dispatch_allowed"] is False
    assert entry["classification"] == "retired_by_terminal_exact_cuda_negative"
    assert entry["terminal_evidence"]["job_id"] == "job1"


def test_cli_blocks_and_classifies_terminal_refused_dispatch(
    tmp_path: Path,
) -> None:
    archive_sha = "9" * 64
    queue = _write_json(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "candidate",
                    "lane_id": "lane",
                    "ready_for_exact_eval_dispatch": True,
                    "candidate_archive_sha256": archive_sha,
                    "archive_bytes": 123,
                }
            ],
        },
    )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | lane | modal | job1 |  | refused_dispatch_missing_claim_or_custody | archive_sha={archive_sha}; blocker=missing_runtime_tree_sha |\n",
        encoding="utf-8",
    )
    manifest = tmp_path / ".omx/research/exact_ready_suppressions.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--queue",
            str(queue),
            "--dispatch-claims-path",
            str(claims),
            "--format",
            "json",
            "--write-suppression-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["raw_stale_ready_row_count"] == 1
    blockers = payload["queues"][0]["suppressed_ready_rows"][0]["blockers"]
    assert any(
        blocker.startswith("same_lane_terminal_refused_dispatch_for_same_archive:")
        for blocker in blockers
    )
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    entry = saved["entries"][0]
    assert entry["classification"] == "retracted_by_terminal_refused_dispatch"
    assert entry["terminal_evidence"]["kind"] == "terminal_refused_dispatch"


def test_cli_unmatched_suppression_manifest_leaves_residual_stale_row(
    tmp_path: Path,
) -> None:
    archive_sha = "d" * 64
    queue = _write_json(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "candidate",
                    "lane_id": "lane",
                    "ready_for_exact_eval_dispatch": True,
                    "candidate_archive_sha256": archive_sha,
                    "archive_bytes": 123,
                }
            ],
        },
    )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |\n",
        encoding="utf-8",
    )
    manifest = _write_json(
        tmp_path / ".omx/research/exact_ready_suppressions.json",
        {
            "schema": "optimizer_exact_ready_queue_suppression_manifest_v1",
            "generated_at_utc": "2026-05-10T00:00:00Z",
            "source_audit_schema": "optimizer_exact_ready_queue_terminal_evidence_audit_v1",
            "raw_stale_ready_row_count": 1,
            "suppression_entry_count": 0,
            "entries": [],
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--queue",
            str(queue),
            "--dispatch-claims-path",
            str(claims),
            "--format",
            "json",
            "--suppression-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["passed"] is False
    assert payload["raw_stale_ready_row_count"] == 1
    assert payload["suppressed_ready_row_count"] == 0
    assert payload["stale_ready_row_count"] == 1
