# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _sweep() -> dict[str, object]:
    return {
        "axis": "modal-t4-cuda-proxy-prefix",
        "archive_bytes": 186395,
        "archive_sha256": "a" * 64,
        "n_pairs": 2,
        "modes": [
            {
                "mode": "none",
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.001,
                "pair_posenet_dist": [0.001, 0.001],
                "pair_segnet_dist": [0.001, 0.001],
            },
            {
                "mode": "even_checker:6",
                "avg_posenet_dist": 0.00055,
                "avg_segnet_dist": 0.001,
                "pair_posenet_dist": [0.0001, 0.001],
                "pair_segnet_dist": [0.001, 0.001],
            },
        ],
    }


def _review(*, canonical_score: float, baseline_score: float = 0.206) -> dict[str, object]:
    return {
        "schema": "tac_result_review_packet_v1",
        "technique": "hdm8_cuda_selector_sparse_top001_exact_cuda_review",
        "score_axis": "contest_cuda",
        "exact_cuda_evidence": True,
        "baseline_score": baseline_score,
        "canonical_score": canonical_score,
        "score_recomputation": {
            "archive_bytes": 186518,
            "avg_posenet_dist": 0.00003249,
            "avg_segnet_dist": 0.0006426,
        },
        "custody": {"archive_sha256": "b" * 64},
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_hdm8_selector_cuda_calibration_table_blocks_proxy_waterfill_after_negative_exact_cuda(
    tmp_path: Path,
) -> None:
    sweep_path = tmp_path / "sweep.json"
    review_path = tmp_path / "review.json"
    output_json = tmp_path / "calibration.json"
    sweep_path.write_text(json.dumps(_sweep()), encoding="utf-8")
    review_path.write_text(json.dumps(_review(canonical_score=0.207)), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hdm8_selector_cuda_calibration_table.py"),
            "--sweep-json",
            str(sweep_path),
            "--exact-review-json",
            str(review_path),
            "--output-json",
            str(output_json),
            "--max-atoms",
            "2",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    table = json.loads(output_json.read_text(encoding="utf-8"))
    assert table["schema"] == "hdm8_selector_cuda_calibration_table_v1"
    assert table["score_claim"] is False
    assert table["ready_for_broad_waterfill_dispatch"] is False
    assert table["exact_cuda_summary"]["regression_count"] == 1
    assert table["exact_cuda_summary"]["positive_or_neutral_count"] == 0
    assert "proxy_positive_calibration_rows_transferred_negative" in table["blockers"]
    assert "broad_waterfill_selector_blocked_until_transfer_model" in table["blockers"]
    assert (tmp_path / "calibration.md").is_file()


def test_hdm8_selector_cuda_calibration_table_keeps_cpu_mps_non_authoritative(
    tmp_path: Path,
) -> None:
    sweep_path = tmp_path / "sweep.json"
    review_path = tmp_path / "review.json"
    output_json = tmp_path / "calibration.json"
    sweep_path.write_text(json.dumps(_sweep()), encoding="utf-8")
    review_path.write_text(json.dumps(_review(canonical_score=0.205)), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hdm8_selector_cuda_calibration_table.py"),
            "--sweep-json",
            str(sweep_path),
            "--exact-review-json",
            str(review_path),
            "--output-json",
            str(output_json),
            "--max-atoms",
            "2",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    table = json.loads(output_json.read_text(encoding="utf-8"))
    assert table["selector_dispatch_policy"]["cpu_or_mps_rows_are_authority"] is False
    assert table["selector_dispatch_policy"]["requires_exact_cuda_positive_control"] is True
