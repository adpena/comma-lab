# SPDX-License-Identifier: MIT
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


def test_parser_demotes_macos_artifact_that_declares_contest_cpu() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "canonical_score": 0.19664189,
            "archive_size_bytes": 178_981,
            "archive_sha256": "e" * 64,
            "device": "cpu",
            "n_samples": 600,
            "score_axis": "contest_cpu",
            "evidence_grade": "contest-CPU",
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
    assert record.rank_or_kill_eligible is False
    assert record.hardware_compliance_blocker == "contest_cpu_requires_linux_x86_64"


def test_parser_marks_t4_cuda_raw_eval_as_score_axis_not_adjudicated() -> None:
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
    assert record.promotion_eligible is False
    assert record.score_claim_valid is False
    assert record.rank_or_kill_eligible is False
    assert record.cpu_leaderboard_reproduction_eligible is False


def test_parser_requires_explicit_booleans_for_cuda_promotion_authority() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.2283908312,
            "final_score": 0.23,
            "archive_size_bytes": 178_981,
            "archive_sha256": "9" * 64,
            "device": "cuda",
            "n_samples": 600,
            "gpu_t4_match": True,
            "promotion_eligible": True,
            "score_claim_valid": True,
            "rank_or_kill_eligible": True,
        }
    )

    assert record is not None
    assert record.score_axis == "contest_cuda"
    assert record.promotion_eligible is True
    assert record.score_claim_valid is True
    assert record.rank_or_kill_eligible is True
    assert record.cpu_leaderboard_reproduction_eligible is False


def test_parser_demotes_persisted_cuda_promotion_when_blockers_present() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.20638030907530963,
            "archive_size_bytes": 186_423,
            "archive_sha256": "8" * 64,
            "device": "cuda",
            "n_samples": 600,
            "gpu_t4_match": True,
            "promotion_eligible": True,
            "score_claim_valid": True,
            "rank_or_kill_eligible": True,
            "promotion_blockers": ["pre_submission_compliance_check_not_recorded"],
            "rank_or_kill_blockers": ["requires_adjudicated_cuda_cpu_policy_review"],
        }
    )

    assert record is not None
    assert record.score_axis == "contest_cuda"
    assert record.score_claim_valid is True
    assert record.promotion_eligible is False
    assert record.rank_or_kill_eligible is False


def test_parser_demotes_unsupported_cuda_artifact_that_declares_contest_cuda() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.2283908312,
            "archive_size_bytes": 178_981,
            "archive_sha256": "f" * 64,
            "device": "cuda",
            "n_samples": 600,
            "gpu_t4_match": False,
            "score_axis": "contest_cuda",
            "evidence_grade": "contest-CUDA",
        }
    )

    assert record is not None
    assert record.score_axis == "cuda"
    assert record.evidence_grade == "A"
    assert record.promotion_eligible is False
    assert record.score_claim_valid is False
    assert record.rank_or_kill_eligible is False
    assert record.hardware_compliance_blocker == "contest_cuda_requires_t4_a100_4090_h100_a10g_l40s"


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


def test_parser_accepts_dispatcher_top_level_gha_cpu_custody_fields() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "archive_sha256": "e" * 64,
            "archive_size_bytes": 185_578,
            "avg_posenet_dist": 0.000164,
            "avg_segnet_dist": 0.00065603,
            "canonical_score": 0.22966866346263318,
            "compression_rate": 0.00494275,
            "device": "cpu",
            "evidence_grade": "contest-CPU-1to1",
            "hardware": "github-actions-ubuntu-latest-x86_64",
            "lane_tag": "[contest-CPU]",
            "n_samples": 600,
            "score_claim_valid": True,
        }
    )

    assert record is not None
    assert record.score_axis == "contest_cpu"
    assert record.cpu_leaderboard_reproduction_eligible is True
    assert record.promotion_eligible is False
    assert record.score_claim_valid is True


def test_inflated_output_manifest_summary_reads_nested_provenance_payload() -> None:
    records = _load_records_module()

    summary = records.inflated_output_manifest_summary(
        {
            "provenance": {
                "inflated_output_manifest": {
                    "path": "work/inflated_outputs_manifest.json",
                    "sha256": "b" * 64,
                    "payload": {
                        "aggregate_sha256": "c" * 64,
                        "raw_file_count": 1,
                        "total_bytes": 603_979_776,
                        "files": [
                            {
                                "video_name": "0.mkv",
                                "relative_path": "0.raw",
                                "exists": True,
                                "bytes": 603_979_776,
                                "sha256": "d" * 64,
                            }
                        ],
                    },
                }
            }
        }
    )

    assert summary == {
        "aggregate_sha256": "c" * 64,
        "raw_file_count": 1,
        "total_bytes": 603_979_776,
        "manifest_path": "work/inflated_outputs_manifest.json",
        "manifest_sha256": "b" * 64,
        "files": [
            {
                "video_name": "0.mkv",
                "relative_path": "0.raw",
                "exists": True,
                "bytes": 603_979_776,
                "sha256": "d" * 64,
            }
        ],
    }


def test_runtime_tree_sha256_prefers_content_hash_from_nested_manifest() -> None:
    records = _load_records_module()

    value = records.runtime_tree_sha256(
        {
            "runtime_tree_sha256": "path-tree",
            "provenance": {
                "inflate_runtime_manifest": {
                    "runtime_content_tree_sha256": "content-tree",
                    "runtime_tree_sha256": "nested-path-tree",
                }
            },
        }
    )

    assert value == "content-tree"


def test_parser_downgrades_explicit_contest_cpu_on_macos() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "canonical_score": 0.1966358879,
            "archive_size_bytes": 178_981,
            "device": "cpu",
            "n_samples": 600,
            "score_axis": "contest_cpu",
            "evidence_grade": "contest-CPU-1to1",
            "promotion_eligible": True,
            "score_claim_valid": True,
            "rank_or_kill_eligible": True,
            "cpu_leaderboard_reproduction_eligible": True,
            "provenance": {
                "platform_system": "Darwin",
                "platform_machine": "arm64",
            },
        }
    )

    assert record is not None
    assert record.score_axis == "cpu_advisory"
    assert record.evidence_grade == "macOS-CPU advisory"
    assert record.hardware_compliance_blocker == "contest_cpu_requires_linux_x86_64"
    assert record.cpu_leaderboard_reproduction_eligible is False
    assert record.promotion_eligible is False
    assert record.score_claim_valid is False
    assert record.rank_or_kill_eligible is False


def test_parser_keeps_cpu_artifact_out_of_rank_or_kill_even_if_payload_claims_true() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.1966358879,
            "archive_size_bytes": 178_981,
            "device": "cpu",
            "n_samples": 600,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "promotion_eligible": True,
            "score_claim_valid": True,
            "rank_or_kill_eligible": True,
            "cpu_leaderboard_reproduction_eligible": True,
        }
    )

    assert record is not None
    assert record.score_axis == "contest_cpu"
    assert record.cpu_leaderboard_reproduction_eligible is True
    assert record.promotion_eligible is False
    assert record.score_claim_valid is True
    assert record.rank_or_kill_eligible is False


def test_parser_accepts_explicit_contest_cuda_on_a100_without_t4_match() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.2283908312,
            "archive_size_bytes": 178_981,
            "device": "cuda",
            "n_samples": 600,
            "score_axis": "contest_cuda",
            "evidence_grade": "A++",
            "gpu_t4_match": False,
            "gpu_model": "NVIDIA A100-SXM4-40GB",
            "promotion_eligible": True,
            "score_claim_valid": True,
            "rank_or_kill_eligible": True,
        }
    )

    assert record is not None
    assert record.score_axis == "contest_cuda"
    assert record.evidence_grade == "A++"
    assert record.hardware_compliance_blocker is None
    assert record.promotion_eligible is True
    assert record.score_claim_valid is True
    assert record.rank_or_kill_eligible is True


def test_parser_treats_string_booleans_as_false_for_cuda_promotion() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.2283908312,
            "archive_size_bytes": 178_981,
            "device": "cuda",
            "n_samples": 600,
            "score_axis": "contest_cuda",
            "evidence_grade": "A++",
            "gpu_t4_match": "false",
            "promotion_eligible": "true",
            "score_claim_valid": "true",
            "rank_or_kill_eligible": "true",
        }
    )

    assert record is not None
    assert record.gpu_t4_match is False
    assert record.score_axis == "cuda"
    assert record.evidence_grade == "A"
    assert record.hardware_compliance_blocker == "contest_cuda_requires_t4_a100_4090_h100_a10g_l40s"
    assert record.promotion_eligible is False
    assert record.score_claim_valid is False
    assert record.rank_or_kill_eligible is False


def test_missing_contest_cuda_text_does_not_demote_linux_cpu_axis() -> None:
    records = _load_records_module()

    record = records.parse_auth_eval_payload(
        {
            "score_recomputed_from_components": 0.1966358879,
            "archive_size_bytes": 178_981,
            "device": "cpu",
            "n_samples": 600,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "evidence_semantics": "missing_contest_cuda_auth_eval",
            "dispatch_blockers": ["missing_contest_cuda_auth_eval"],
        }
    )

    assert record is not None
    assert record.score_axis == "contest_cpu"
    assert record.evidence_grade == "contest-CPU"
    assert record.cpu_leaderboard_reproduction_eligible is True
    assert record.promotion_eligible is False
    assert record.score_claim_valid is False
    assert record.rank_or_kill_eligible is False
