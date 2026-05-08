from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from tools.build_pr107_cpu_lossy_candidate_matrix import (
    CandidateSpec,
    build_matrix,
    contest_score,
    parse_report_text,
)

REPORT_TEXT = """=== Evaluation config ===
  batch_size: 16
  device: cpu
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00065377
  Average SegNet Distortion: 0.00262650
  Submission file size: 142,128 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00378549
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.44
"""


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_archive(path: Path, member: bytes = b"candidate") -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(info, member)
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_report_text_recovers_precise_component_score() -> None:
    parsed = parse_report_text(REPORT_TEXT)

    assert parsed["n_samples"] == 600
    assert parsed["archive_size_bytes"] == 142128
    assert parsed["canonical_score_display"] == 0.44
    assert parsed["score_recomputed_from_components"] == contest_score(
        avg_segnet_dist=0.00262650,
        avg_posenet_dist=0.00065377,
        compression_rate=0.00378549,
    )


def test_build_matrix_marks_cpu_worse_candidate_fail_closed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline/contest_auth_eval.adjudicated.json"
    _write_json(
        baseline,
        {
            "archive_size_bytes": 178392,
            "archive_sha256": "7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb",
            "score_recomputed_from_components": 0.1966358879,
            "avg_posenet_dist": 0.00003580,
            "avg_segnet_dist": 0.00058931,
            "compression_rate": 0.00475136,
            "n_samples": 600,
            "device": "cpu",
            "hardware": "github-actions-ubuntu-latest-x86_64",
            "evidence_grade": "contest-CPU-1to1",
            "lane_tag": "[contest-CPU]",
            "is_contest_compliant": True,
            "fork_pr_number": 1,
        },
    )
    (tmp_path / "local_report.txt").write_text(REPORT_TEXT, encoding="utf-8")
    (tmp_path / "reports").mkdir()
    _write_json(
        tmp_path / "reports/cathedral_autopilot_catalog_updated_20260508.json",
        {"catalog": []},
    )
    (tmp_path / "reports/cathedral_autopilot_evidence.jsonl").write_text(
        json.dumps(
            {
                "technique": "lossy_coarsening_analytical",
                "evidence_grade": "[contest-CUDA A-negative]",
                "contest_dispatch_verdict": "measured_config_retired_exact_cuda_negative",
                "timestamp": "2026-05-08T06:31:41Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    archive = tmp_path / "experiments/results/cand/archive.zip"
    archive_bytes, archive_sha = _write_archive(archive)
    manifest = tmp_path / "experiments/results/cand/build_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "unit",
            "tool": "unit_builder",
            "evidence_grade": "[CPU-prep]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "build_target_budget": 0.07,
            "build_rel_err": 0.0618,
            "build_archive_relpath": "experiments/results/cand/archive.zip",
            "build_archive_sha256": archive_sha,
            "build_archive_size_bytes": archive_bytes,
            "build_delta_zip_vs_baseline": archive_bytes - 178392,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
        },
    )
    cpu_eval = tmp_path / "experiments/results/cand_cpu/contest_auth_eval.adjudicated.json"
    _write_json(
        cpu_eval,
        {
            "archive_size_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "avg_posenet_dist": None,
            "avg_segnet_dist": None,
            "score_recomputed_from_components": 0.0,
            "device": "cpu",
            "hardware": "github-actions-ubuntu-latest-x86_64",
            "runner_arch": "x86_64",
            "evidence_grade": "contest-CPU-1to1",
            "lane_tag": "[contest-CPU]",
            "is_contest_compliant": True,
            "report_text": REPORT_TEXT,
        },
    )

    matrix = build_matrix(
        repo_root=tmp_path,
        candidates=(CandidateSpec("unit_b070", "unit_variant", manifest.relative_to(tmp_path), cpu_eval.relative_to(tmp_path)),),
        baseline_cpu_eval=baseline.relative_to(tmp_path),
        local_baseline_report=Path("local_report.txt"),
        cathedral_evidence=Path("reports/cathedral_autopilot_evidence.jsonl"),
        cathedral_catalog=Path("reports/cathedral_autopilot_catalog_updated_20260508.json"),
        recorded_at_utc="2026-05-08T14:00:00Z",
    )

    row = matrix["candidates"][0]
    assert matrix["score_claim"] is False
    assert matrix["summary"]["cpu_axis_improver_count"] == 0
    assert row["status"] == "retired_on_pr107_cpu_axis_uniform_lossy_config"
    assert row["cpu_eval"]["report_text_fallback_used"] is True
    assert row["cpu_eval_trusted_for_score_delta"] is True
    assert row["cpu_eval_validation_blockers"] == []
    assert row["delta_vs_pr107_cpu_anchor"]["candidate_minus_baseline_score_cpu"] > 0
    assert "exact_cpu_score_not_lower_than_pr107_anchor" in row["fail_closed_blockers"]
    assert "exact_cuda_auth_eval_missing" in row["fail_closed_blockers"]
    assert row["build_manifest"]["archive_probe"]["status"] == "ok"
    assert "tools/dispatch_cpu_eval_via_github_actions.py" in row["required_validation_commands"]["exact_cpu_gha"]
    assert "scripts/launch_lightning_batch_job.py exact-eval" in row["required_validation_commands"]["exact_cuda_lightning_after_claim"]
    assert matrix["cathedral_autopilot"]["lossy_rows_count"] == 1
    assert matrix["cathedral_autopilot"]["exact_negative_lossy_rows_count"] == 1


def test_build_matrix_missing_cpu_eval_stays_fail_closed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline/contest_auth_eval.adjudicated.json"
    _write_json(
        baseline,
        {
            "archive_size_bytes": 178392,
            "archive_sha256": "7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb",
            "score_recomputed_from_components": 0.1966358879,
            "avg_posenet_dist": 0.00003580,
            "avg_segnet_dist": 0.00058931,
            "compression_rate": 0.00475136,
        },
    )
    (tmp_path / "local_report.txt").write_text(REPORT_TEXT, encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/cathedral_autopilot_evidence.jsonl").write_text("", encoding="utf-8")
    _write_json(tmp_path / "reports/cathedral_autopilot_catalog_updated_20260508.json", {})

    archive = tmp_path / "experiments/results/cand/archive.zip"
    archive_bytes, archive_sha = _write_archive(archive)
    manifest = tmp_path / "experiments/results/cand/build_manifest.json"
    _write_json(
        manifest,
        {
            "build_target_budget": 0.12,
            "build_rel_err": 0.111,
            "build_archive_relpath": "experiments/results/cand/archive.zip",
            "build_archive_sha256": archive_sha,
            "build_archive_size_bytes": archive_bytes,
            "score_claim": False,
        },
    )

    matrix = build_matrix(
        repo_root=tmp_path,
        candidates=(CandidateSpec("unit_b120", "unit_variant", manifest.relative_to(tmp_path)),),
        baseline_cpu_eval=baseline.relative_to(tmp_path),
        local_baseline_report=Path("local_report.txt"),
        cathedral_evidence=Path("reports/cathedral_autopilot_evidence.jsonl"),
        cathedral_catalog=Path("reports/cathedral_autopilot_catalog_updated_20260508.json"),
        recorded_at_utc="2026-05-08T14:00:00Z",
    )

    row = matrix["candidates"][0]
    assert row["status"] == "exact_cpu_missing_fail_closed"
    assert row["cpu_eval"]["exists"] is False
    assert "exact_cpu_auth_eval_missing" in row["fail_closed_blockers"]
    assert "exact_cuda_auth_eval_missing" in row["fail_closed_blockers"]
    assert row["ready_for_exact_eval_dispatch"] is False


def test_build_matrix_rejects_cpu_eval_for_wrong_archive(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline/contest_auth_eval.adjudicated.json"
    _write_json(
        baseline,
        {
            "archive_size_bytes": 178392,
            "archive_sha256": "7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb",
            "score_recomputed_from_components": 0.1966358879,
            "avg_posenet_dist": 0.00003580,
            "avg_segnet_dist": 0.00058931,
            "compression_rate": 0.00475136,
            "n_samples": 600,
            "device": "cpu",
            "hardware": "github-actions-ubuntu-latest-x86_64",
            "runner_arch": "x86_64",
            "evidence_grade": "contest-CPU-1to1",
            "lane_tag": "[contest-CPU]",
            "is_contest_compliant": True,
            "fork_pr_number": 1,
        },
    )
    (tmp_path / "local_report.txt").write_text(REPORT_TEXT, encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/cathedral_autopilot_evidence.jsonl").write_text(
        "", encoding="utf-8"
    )
    _write_json(tmp_path / "reports/cathedral_autopilot_catalog_updated_20260508.json", {})

    archive = tmp_path / "experiments/results/cand/archive.zip"
    archive_bytes, archive_sha = _write_archive(archive)
    manifest = tmp_path / "experiments/results/cand/build_manifest.json"
    _write_json(
        manifest,
        {
            "build_target_budget": 0.07,
            "build_rel_err": 0.0618,
            "build_archive_relpath": "experiments/results/cand/archive.zip",
            "build_archive_sha256": archive_sha,
            "build_archive_size_bytes": archive_bytes,
            "score_claim": False,
        },
    )
    cpu_eval = tmp_path / "experiments/results/cand_cpu/contest_auth_eval.adjudicated.json"
    _write_json(
        cpu_eval,
        {
            "archive_size_bytes": archive_bytes + 1,
            "archive_sha256": "0" * 64,
            "avg_posenet_dist": 0.000035,
            "avg_segnet_dist": 0.00058,
            "compression_rate": 0.0047,
            "score_recomputed_from_components": 0.19,
            "n_samples": 600,
            "device": "cpu",
            "hardware": "github-actions-ubuntu-latest-x86_64",
            "runner_arch": "x86_64",
            "evidence_grade": "contest-CPU-1to1",
            "lane_tag": "[contest-CPU]",
            "is_contest_compliant": True,
        },
    )

    matrix = build_matrix(
        repo_root=tmp_path,
        candidates=(
            CandidateSpec(
                "unit_wrong_cpu",
                "unit_variant",
                manifest.relative_to(tmp_path),
                cpu_eval.relative_to(tmp_path),
            ),
        ),
        baseline_cpu_eval=baseline.relative_to(tmp_path),
        local_baseline_report=Path("local_report.txt"),
        cathedral_evidence=Path("reports/cathedral_autopilot_evidence.jsonl"),
        cathedral_catalog=Path("reports/cathedral_autopilot_catalog_updated_20260508.json"),
        recorded_at_utc="2026-05-08T14:00:00Z",
    )

    row = matrix["candidates"][0]
    assert row["status"] == "exact_cpu_untrusted_fail_closed"
    assert row["cpu_eval_trusted_for_score_delta"] is False
    assert row["delta_vs_pr107_cpu_anchor"]["candidate_minus_baseline_score_cpu"] is None
    assert "exact_cpu_archive_sha256_mismatch_with_build" in row[
        "cpu_eval_validation_blockers"
    ]
    assert "exact_cpu_archive_bytes_mismatch_with_build" in row[
        "cpu_eval_validation_blockers"
    ]
    assert "exact_cpu_auth_eval_untrusted" in row["fail_closed_blockers"]
