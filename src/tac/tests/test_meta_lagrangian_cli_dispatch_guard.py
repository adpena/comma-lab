# SPDX-License-Identifier: MIT
"""CLI-level dispatch guards for tools/meta_lagrangian_search_cli.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "tools" / "meta_lagrangian_search_cli.py"


def test_meta_lagrangian_cli_report_is_forensic_not_dispatch_ready(tmp_path: Path) -> None:
    anchors = tmp_path / "anchors_apogee_intN.json"
    anchors.write_text(
        json.dumps(
            [
                {
                    "lane_id": "lossless",
                    "rel_err_pct_per_weight": 0.0,
                    "archive_bytes": 186239,
                    "contest_cuda_score": 0.20945673,
                    "avg_pose_dist": 3.4e-5,
                    "avg_seg_dist": 0.00067819,
                    "rate_unscaled": 186239 / 37545489,
                    "measured_utc": "2026-05-05T17:25Z",
                    "job_id": "baseline",
                    "archive_sha256": "ab",
                },
                {
                    "lane_id": "mild_lossy",
                    "rel_err_pct_per_weight": 0.5,
                    "archive_bytes": 180000,
                    "contest_cuda_score": 0.23,
                    "avg_pose_dist": 5.0e-5,
                    "avg_seg_dist": 0.0008,
                    "rate_unscaled": 180000 / 37545489,
                    "measured_utc": "2026-05-05T17:26Z",
                    "job_id": "mild",
                    "archive_sha256": "cd",
                },
                {
                    "lane_id": "lossy",
                    "rel_err_pct_per_weight": 5.0,
                    "archive_bytes": 120000,
                    "contest_cuda_score": 0.9,
                    "avg_pose_dist": 0.01,
                    "avg_seg_dist": 0.004,
                    "rate_unscaled": 120000 / 37545489,
                    "measured_utc": "2026-05-05T17:27Z",
                    "job_id": "lossy",
                    "archive_sha256": "ef",
                },
            ]
        )
    )
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "candidate",
                    "archive_bytes": 180000,
                    "rel_err_pct": 0.5,
                    "n_layers": 13,
                    "lane_class": "apogee_intN",
                }
            ]
        )
    )
    report = tmp_path / "report.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--lane-class",
            "apogee_intN",
            "--anchors-path",
            str(anchors),
            "--candidates-json",
            str(candidates),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(report.read_text())

    assert "0 exact-eval dispatch-ready" in proc.stdout
    assert payload["evidence_semantics"] == "local_proxy_prediction_forensic"
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_exact_eval_dispatch_count"] == 0
    assert payload["eligible_for_dispatch"] == 0
    assert payload["top_k"] == []
    assert payload["top_k_forensic"]
