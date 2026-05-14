from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "parallel_dispatch_top_k.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"payload", compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_parallel_dispatch_floor_preserves_active_nonpromotional_reference() -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_floor_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.DEFAULT_ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE == 0.20636832361415344
    assert module.DEFAULT_ACTIVE_SCORE_FRONTIER_SCORE == 0.20642625334307507
    assert module.DEFAULT_ACTIVE_FLOOR_SCORE == module.DEFAULT_ACTIVE_SCORE_FRONTIER_SCORE


def test_parallel_dispatch_refuses_stale_exact_ready_terminal_claim(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    archive_bytes, archive_sha = _write_archive(submission / "archive.zip")
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text("contest custody fixture\n", encoding="utf-8")
    _write_json(
        submission / "archive_manifest.json",
        {
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": archive_bytes,
            "candidate_archive": {"member_name": "0.bin"},
            "score_claim": False,
        },
    )
    queue = _write_json(
        tmp_path / "exact_ready_queue.json",
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "fixture",
                    "lane_id": "fixture_lane",
                    "target_modes": ["contest_exact_eval"],
                    "deployment_target": "t4_contest_runtime",
                    "evidence_semantics": "byte_closed_archive_runtime_ready_for_exact_eval",
                    "ready_for_exact_eval_dispatch": True,
                    "score_claim": False,
                    "score_claim_verified": False,
                    "archive_path": str(submission / "archive.zip"),
                    "submission_dir": str(submission),
                    "inflate_sh_path": str(inflate),
                    "candidate_archive_sha256": archive_sha,
                    "candidate_archive_bytes": archive_bytes,
                    "score_affecting_payload_changed": True,
                    "charged_bits_changed": True,
                    "predicted_band": [0.19, 0.2],
                }
            ],
            "top_k": [],
        },
    )
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job-1 | | completed_contest_cuda | score=0.226 archive_sha={archive_sha} |\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--ranked-input",
            str(queue),
            "--provider",
            "vastai",
            "--dry-run",
            "--dispatch-claims-path",
            str(claims),
            "--active-floor-score",
            "0.2",
            "--active-floor-archive-bytes",
            "999999",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "ranked-input exact-ready audit failed" in proc.stderr
    assert "same_lane_terminal_score_not_below_active_floor_for_same_archive" in proc.stderr


def test_parallel_dispatch_refuses_paid_vastai_without_claim_enforcement(tmp_path: Path) -> None:
    ranked = _write_json(tmp_path / "ranked.json", {"dispatch_ready": []})

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--ranked-input",
            str(ranked),
            "--provider",
            "vastai",
            "--allow-above-active-floor-dispatch",
            "--operator-override-reason",
            "fixture",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "provider=vastai is disabled" in proc.stderr
    assert "claim_lane_dispatch.py" in proc.stderr
