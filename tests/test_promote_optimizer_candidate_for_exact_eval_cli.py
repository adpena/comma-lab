from __future__ import annotations

import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "promote_optimizer_candidate_for_exact_eval.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path) -> tuple[int, str]:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"payload", compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_cli_refuses_to_write_outputs_inside_submission_runtime_tree(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    archive_bytes, archive_sha = _write_archive(submission / "archive.zip")
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text("custody report\n", encoding="utf-8")
    _write_json(
        submission / "archive_manifest.json",
        {
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": archive_bytes,
            "candidate_archive": {"member_name": "0.bin"},
            "score_claim": False,
        },
    )
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "evaluate.py").write_text("# fixture\n", encoding="utf-8")
    queue = _write_json(
        tmp_path / "queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [
                {
                    "candidate_id": "fixture",
                    "lane_id": "fixture_lane",
                    "archive_path": (submission / "archive.zip").relative_to(tmp_path).as_posix(),
                    "candidate_archive_sha256": archive_sha,
                    "candidate_archive_bytes": archive_bytes,
                    "score_affecting_payload_changed": True,
                    "charged_bits_changed": True,
                    "score_claim": False,
                    "dispatch_blockers": [
                        "optimizer_candidate_queue_is_planning_only",
                        "requires_exact_eval_readiness_gate",
                    ],
                }
            ],
            "dispatch_ready": [],
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
            "--candidate-id",
            "fixture",
            "--output",
            str(submission / "exact_ready_queue.json"),
            "--report-output",
            str(submission / "exact_ready_report.json"),
            "--active-floor-archive-bytes",
            "0",
            "--allow-above-active-floor-dispatch",
            "--operator-override-reason",
            "fixture",
            "--skip-active-claim-check",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "output_inside_submission_dir_would_mutate_runtime_tree" in proc.stderr
    assert not (submission / "exact_ready_queue.json").exists()
    assert not (submission / "exact_ready_report.json").exists()
