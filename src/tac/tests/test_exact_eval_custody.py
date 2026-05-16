# SPDX-License-Identifier: MIT
"""Tests for shared exact-eval custody validation."""

from __future__ import annotations

from tac.exact_eval_custody import (
    contest_score,
    extract_archive_sha256,
    extract_runtime_tree_sha256,
    is_sha256_hex,
    validate_exact_eval_evidence,
)


def test_extract_runtime_tree_sha256_from_nested_manifests() -> None:
    runtime_sha = "b" * 64

    assert (
        extract_runtime_tree_sha256(
            {
                "provenance": {
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": runtime_sha,
                    },
                },
            }
        )
        == runtime_sha
    )


def test_extract_archive_sha256_accepts_common_exact_eval_fields() -> None:
    archive_sha = "a" * 64

    assert extract_archive_sha256({"archive_sha": archive_sha}) == archive_sha
    assert extract_archive_sha256({"archive_sha256": "not-a-sha"}) == ""
    assert is_sha256_hex(archive_sha) is True
    assert is_sha256_hex("g" * 64) is False


def test_validate_exact_eval_evidence_requires_formula_and_devices() -> None:
    archive_bytes = 123
    score = contest_score(seg_dist=0.001, pose_dist=0.0004, archive_bytes=archive_bytes)
    valid = validate_exact_eval_evidence(
        {
            "axis": "contest_cuda",
            "archive_sha256": "a" * 64,
            "runtime_tree_sha256": "b" * 64,
            "score": score,
            "seg_dist": 0.001,
            "pose_dist": 0.0004,
            "archive_bytes": archive_bytes,
            "n_samples": 1200,
            "hardware": "modal-t4",
            "inflate_device": "cuda",
            "eval_device": "cuda",
            "auth_eval_command": "contest_auth_eval --axis contest_cuda",
            "log_path": "experiments/results/cuda.log",
        },
        expected_axis="contest_cuda",
        expected_archive_sha256="a" * 64,
        expected_runtime_tree_sha256="b" * 64,
        require_devices=True,
    )

    assert valid.blockers == ()
    assert valid.score == score
    assert valid.archive_bytes == archive_bytes


def test_validate_exact_eval_evidence_fails_closed_on_partial_rows() -> None:
    verdict = validate_exact_eval_evidence(
        {
            "axis": "contest_cpu",
            "archive_sha256": "a" * 64,
            "runtime_tree_sha256": "c" * 64,
            "score": 0.123,
            "archive_bytes": 123,
        },
        expected_axis="contest_cuda",
        expected_archive_sha256="a" * 64,
        expected_runtime_tree_sha256="b" * 64,
        require_devices=True,
        require_artifact_path=True,
    )

    assert "axis_mismatch" in verdict.blockers
    assert "runtime_tree_sha_mismatch" in verdict.blockers
    assert "seg_dist_missing" in verdict.blockers
    assert "pose_dist_missing" in verdict.blockers
    assert "n_samples_missing" in verdict.blockers
    assert "inflate_device_missing" in verdict.blockers
    assert "eval_device_missing" in verdict.blockers
    assert "auth_eval_command_missing" in verdict.blockers
    assert "log_path_missing" in verdict.blockers
    assert "artifact_path_missing" in verdict.blockers
