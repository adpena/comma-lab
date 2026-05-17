# SPDX-License-Identifier: MIT
"""Tests for shared exact-eval custody validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.exact_eval_custody import (
    CONTEST_EXACT_SAMPLE_COUNT,
    contest_score,
    extract_archive_sha256,
    extract_expected_runtime_tree_sha256,
    extract_observed_runtime_content_tree_sha256,
    extract_observed_runtime_tree_sha256,
    extract_runtime_tree_sha256,
    extract_runtime_tree_sha256_allow_expected,
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


def test_observed_runtime_hash_extractors_reject_expected_only_fields() -> None:
    runtime_sha = "b" * 64
    content_sha = "c" * 64

    expected_only = {
        "expected_runtime_tree_sha256": runtime_sha,
        "expected_runtime_content_tree_sha256": content_sha,
        "provenance": {
            "expected_runtime_tree_sha256": runtime_sha,
            "inflate_runtime_manifest": {
                "expected_runtime_tree_sha256": runtime_sha,
                "expected_runtime_content_tree_sha256": content_sha,
            },
        },
    }

    assert extract_runtime_tree_sha256(expected_only) == ""
    assert extract_runtime_tree_sha256_allow_expected(expected_only) == runtime_sha
    assert extract_observed_runtime_tree_sha256(expected_only) == ""
    assert extract_observed_runtime_content_tree_sha256(expected_only) == ""
    assert (
        extract_expected_runtime_tree_sha256(
            {
                "auth_eval_command": (
                    "contest_auth_eval --device cuda "
                    f"--expected-runtime-tree-sha256 {runtime_sha}"
                )
            }
        )
        == runtime_sha
    )
    assert (
        extract_observed_runtime_tree_sha256(
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
    assert (
        extract_observed_runtime_content_tree_sha256(
            {
                "provenance": {
                    "inflate_runtime_manifest": {
                        "runtime_content_tree_sha256": content_sha,
                    },
                },
            }
        )
        == content_sha
    )


def test_extract_archive_sha256_accepts_common_exact_eval_fields() -> None:
    archive_sha = "a" * 64

    assert extract_archive_sha256({"archive_sha": archive_sha}) == archive_sha
    assert extract_archive_sha256({"archive_sha256": "not-a-sha"}) == ""
    assert is_sha256_hex(archive_sha) is True
    assert is_sha256_hex("g" * 64) is False


def _valid_contest_evidence(tmp_path: Path, *, axis: str = "contest_cuda") -> dict[str, object]:
    log_path = tmp_path / f"{axis}.log"
    artifact_path = tmp_path / f"{axis}.json"
    manifest_path = tmp_path / f"{axis}_inflated_outputs_manifest.json"
    log_path.write_text("ok\n", encoding="utf-8")
    artifact_path.write_text("{}\n", encoding="utf-8")
    raw_output_aggregate_sha = "c" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "contest_auth_eval_inflated_output_manifest_v1",
                "aggregate_sha256": raw_output_aggregate_sha,
                "raw_file_count": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive_bytes = 123
    score = contest_score(seg_dist=0.001, pose_dist=0.0004, archive_bytes=archive_bytes)
    if axis == "contest_cuda":
        hardware = "modal-t4"
        inflate_device = "cuda"
        eval_device = "cuda"
    else:
        hardware = "linux-x86_64"
        inflate_device = "cpu"
        eval_device = "cpu"
    return {
        "axis": axis,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "score": score,
        "seg_dist": 0.001,
        "pose_dist": 0.0004,
        "archive_bytes": archive_bytes,
        "n_samples": CONTEST_EXACT_SAMPLE_COUNT,
        "hardware": hardware,
        "inflate_device": inflate_device,
        "eval_device": eval_device,
        "auth_eval_command": f"contest_auth_eval --axis {axis}",
        "log_path": log_path.name,
        "artifact_path": artifact_path.name,
        "inflated_outputs_manifest_path": manifest_path.name,
        "inflated_outputs_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "raw_output_aggregate_sha256": raw_output_aggregate_sha,
    }


def test_validate_exact_eval_evidence_requires_formula_and_devices(tmp_path: Path) -> None:
    archive_bytes = 123
    score = contest_score(seg_dist=0.001, pose_dist=0.0004, archive_bytes=archive_bytes)
    valid = validate_exact_eval_evidence(
        _valid_contest_evidence(tmp_path),
        expected_axis="contest_cuda",
        expected_archive_sha256="a" * 64,
        expected_runtime_tree_sha256="b" * 64,
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert valid.blockers == ()
    assert valid.score == score
    assert valid.archive_bytes == archive_bytes


def test_validate_exact_eval_evidence_requires_full_frame_output_custody(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path)

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        artifact_base_dir=tmp_path,
    )

    assert verdict.blockers == ()

    missing = dict(evidence)
    missing.pop("inflated_outputs_manifest_path")
    missing.pop("raw_output_aggregate_sha256")
    missing_verdict = validate_exact_eval_evidence(
        missing,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        artifact_base_dir=tmp_path,
    )

    assert "inflated_outputs_manifest_path_missing" in missing_verdict.blockers
    assert "raw_output_aggregate_sha_invalid" in missing_verdict.blockers


def test_validate_exact_eval_evidence_rejects_manifest_aggregate_mismatch(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path)
    evidence["raw_output_aggregate_sha256"] = "d" * 64

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        artifact_base_dir=tmp_path,
    )

    assert "inflated_outputs_manifest_aggregate_mismatch" in verdict.blockers


def test_validate_exact_eval_evidence_accepts_cuda_auto_inflate_policy(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cuda")
    evidence["inflate_device"] = "auto"
    evidence["eval_device"] = "cuda"
    evidence["auth_eval_command"] = (
        "python experiments/contest_auth_eval.py --device cuda --inflate-device auto"
    )

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert verdict.blockers == ()


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


def test_validate_exact_eval_evidence_rejects_cpu_devices_for_cuda_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cuda")
    evidence["hardware"] = "linux-x86_64"
    evidence["inflate_device"] = "cpu"
    evidence["eval_device"] = "cpu"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "hardware_not_cuda" in verdict.blockers
    assert "inflate_device_not_cuda" in verdict.blockers
    assert "eval_device_not_cuda" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_negated_cuda_tokens(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cuda")
    evidence["hardware"] = "cpu-no-cuda"
    evidence["inflate_device"] = "no-cuda-cpu"
    evidence["eval_device"] = "cuda-disabled"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "hardware_not_cuda" in verdict.blockers
    assert "inflate_device_not_cuda" in verdict.blockers
    assert "eval_device_not_cuda" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_cuda_devices_for_cpu_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cpu")
    evidence["inflate_device"] = "cuda"
    evidence["eval_device"] = "cuda"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cpu",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "inflate_device_not_cpu" in verdict.blockers
    assert "eval_device_not_cpu" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_auto_inflate_for_cpu_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cpu")
    evidence["inflate_device"] = "auto"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cpu",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "inflate_device_not_cpu" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_mixed_cuda_cpu_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cpu")
    evidence["hardware"] = "linux-x86_64 T4 cuda"
    evidence["inflate_device"] = "cpu cuda"
    evidence["eval_device"] = "linux-x86_64 gpu"
    evidence["auth_eval_command"] = (
        "python experiments/contest_auth_eval.py --axis contest_cpu "
        "--device cpu --scorer-device cuda"
    )

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cpu",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "hardware_not_contest_cpu" in verdict.blockers
    assert "inflate_device_not_contest_cpu" in verdict.blockers
    assert "eval_device_not_contest_cpu" in verdict.blockers
    assert "auth_eval_command_not_contest_cpu" in verdict.blockers


def test_validate_exact_eval_evidence_accepts_linux_x86_cpu_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cpu")

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cpu",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert verdict.blockers == ()


def test_validate_exact_eval_evidence_rejects_cpu_axis_without_linux_provenance(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cpu")
    evidence["hardware"] = "x86_64 cpu host"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cpu",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "hardware_not_contest_cpu" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_macos_cpu_advisory_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cpu")
    evidence["hardware"] = "macos-apple-m2-cpu"
    evidence["inflate_device"] = "cpu-mps"
    evidence["eval_device"] = "Apple Metal CPU"
    evidence["auth_eval_command"] = (
        "python upstream/contest_auth_eval.py --axis contest_cpu --platform darwin"
    )

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cpu",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "hardware_not_contest_cpu" in verdict.blockers
    assert "inflate_device_not_contest_cpu" in verdict.blockers
    assert "eval_device_not_contest_cpu" in verdict.blockers
    assert "auth_eval_command_not_contest_cpu" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_negated_cpu_tokens_for_cpu_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cpu")
    evidence["inflate_device"] = "cpu-disabled"
    evidence["eval_device"] = "no-cpu"
    evidence["auth_eval_command"] = "experiments/contest_auth_eval.py --device no-cpu"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cpu",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "inflate_device_not_cpu" in verdict.blockers
    assert "eval_device_not_cpu" in verdict.blockers
    assert "auth_eval_command_unrecognized" in verdict.blockers


def test_validate_exact_eval_evidence_requires_canonical_auth_eval_command(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path, axis="contest_cuda")
    evidence["auth_eval_command"] = "python random_script.py --device cuda"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "auth_eval_command_unrecognized" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_partial_sample_contest_axis(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path)
    evidence["n_samples"] = CONTEST_EXACT_SAMPLE_COUNT - 1

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "n_samples_not_contest_exact" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_missing_log_artifact_file(
    tmp_path: Path,
) -> None:
    evidence = _valid_contest_evidence(tmp_path)
    evidence["log_path"] = "missing.log"
    evidence["artifact_path"] = "missing.json"

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "log_path_file_missing" in verdict.blockers
    assert "artifact_path_file_missing" in verdict.blockers


def test_validate_exact_eval_evidence_rejects_transient_or_outside_paths(
    tmp_path: Path,
) -> None:
    outside_dir = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_dir.mkdir(exist_ok=True)
    outside_artifact = outside_dir / "outside.json"
    outside_artifact.write_text("{}\n", encoding="utf-8")
    evidence = _valid_contest_evidence(tmp_path)
    evidence["log_path"] = "/tmp/transient.log"
    evidence["artifact_path"] = str(outside_artifact)

    verdict = validate_exact_eval_evidence(
        evidence,
        expected_axis="contest_cuda",
        require_artifact_path=True,
        require_devices=True,
        artifact_base_dir=tmp_path,
    )

    assert "log_path_transient" in verdict.blockers
    assert "artifact_path_outside_base_dir" in verdict.blockers
