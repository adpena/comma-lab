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
