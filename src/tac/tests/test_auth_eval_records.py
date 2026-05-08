from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "auth_eval_records.py"


def _load_records_module():
    spec = importlib.util.spec_from_file_location("auth_eval_records_under_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_marks_linux_cpu_as_contest_cpu_not_promotion_axis() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "canonical_score": 0.1966358879,
            "final_score": 0.20,
            "archive_size_bytes": 178_981,
            "archive_sha256": "a" * 64,
            "avg_segnet_dist": 0.00057599,
            "avg_posenet_dist": 0.00003460,
            "compression_rate": 0.004767,
            "device": "cpu",
            "n_samples": 600,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        }
    )

    assert record is not None
    assert record.score == 0.1966358879
    assert record.score_axis == "contest_cpu"
    assert record.evidence_grade == "contest-CPU"
    assert record.cpu_leaderboard_reproduction_eligible is True
    assert record.promotion_eligible is False
    assert record.score_claim_valid is False
    assert record.rank_or_kill_eligible is False
    assert record.rate_unscaled == 0.004767


def test_parser_marks_macos_cpu_as_advisory_only() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "canonical_score": 0.19664189,
            "archive_size_bytes": 178_981,
            "archive_sha256": "b" * 64,
            "device": "cpu",
            "n_samples": 600,
            "provenance": {
                "platform_system": "Darwin",
                "platform_machine": "arm64",
            },
        }
    )

    assert record is not None
    assert record.score_axis == "cpu_advisory"
    assert record.evidence_grade == "macOS-CPU advisory"
    assert record.cpu_leaderboard_reproduction_eligible is False
    assert record.promotion_eligible is False
    assert record.score_claim_valid is False


def test_parser_marks_t4_cuda_as_strict_promotion_axis() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.2283908312,
            "final_score": 0.23,
            "archive_size_bytes": 178_981,
            "archive_sha256": "c" * 64,
            "device": "cuda",
            "n_samples": 600,
            "gpu_t4_match": True,
        }
    )

    assert record is not None
    assert record.score == 0.2283908312
    assert record.score_axis == "contest_cuda"
    assert record.evidence_grade == "A++"
    assert record.promotion_eligible is True
    assert record.score_claim_valid is True
    assert record.rank_or_kill_eligible is True
    assert record.cpu_leaderboard_reproduction_eligible is False


def test_parser_recovers_components_from_gha_report_text_when_fields_are_null() -> None:
    records = _load_records_module()
    report_text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00086019
  Average SegNet Distortion: 0.00312713
  Submission file size: 136,074 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00362424
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.50
"""

    record = records.parse_auth_eval_payload(
        {
            "canonical_score": None,
            "score_recomputed_from_components": 0.0,
            "avg_segnet_dist": None,
            "avg_posenet_dist": None,
            "compression_rate": None,
            "n_samples": None,
            "archive_size_bytes": 136_074,
            "archive_sha256": "d" * 64,
            "device": "cpu",
            "hardware": "github-actions-ubuntu-latest-x86_64",
            "evidence_grade": "contest-CPU-1to1",
            "report_text": report_text,
        }
    )

    expected = 100 * 0.00312713 + math.sqrt(10 * 0.00086019) + 25 * 0.00362424
    assert record is not None
    assert abs(record.score - expected) < 1e-12
    assert record.avg_segnet_dist == 0.00312713
    assert record.avg_posenet_dist == 0.00086019
    assert record.rate_unscaled == 0.00362424
    assert record.samples == 600
    assert record.score_axis == "contest_cpu"
