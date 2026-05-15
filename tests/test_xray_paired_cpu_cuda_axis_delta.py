# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.xray_paired_cpu_cuda_axis_delta import (
    _load_axis_input,
    build_report,
    main,
)


def _auth_eval_payload(
    *,
    axis: str,
    device: str,
    score: float,
    seg: float,
    pose: float,
    rate: float,
    archive_sha256: str = "a" * 64,
    archive_bytes: int = 178_517,
) -> dict:
    payload = {
        "score_axis": axis,
        "evidence_grade": "contest-CUDA" if axis == "contest_cuda" else "contest-CPU",
        "score_recomputed_from_components": score,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "rate_unscaled": rate,
        "archive_size_bytes": archive_bytes,
        "n_samples": 600,
        "promotion_eligible": axis == "contest_cuda",
        "score_claim_valid": axis == "contest_cuda",
        "provenance": {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_bytes,
            "device": device,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_model": "Tesla T4" if device == "cuda" else "",
            "gpu_t4_match": device == "cuda",
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": "r" * 64,
                "runtime_content_tree_sha256": "c" * 64,
            },
        },
    }
    return payload


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def test_build_report_quantifies_pr101_fec6_cpu_cuda_gap(tmp_path: Path) -> None:
    cpu_json = _write_json(
        tmp_path / "cpu.json",
        _auth_eval_payload(
            axis="contest_cpu",
            device="cpu",
            score=0.1920513168811056,
            seg=0.00056029,
            pose=0.00002943,
            rate=0.004754685709380427,
        ),
    )
    cuda_json = _write_json(
        tmp_path / "cuda.json",
        _auth_eval_payload(
            axis="contest_cuda",
            device="cuda",
            score=0.22621002169349796,
            seg=0.00066299,
            pose=0.00016846,
            rate=0.004754685709380427,
        ),
    )

    report = build_report(
        cpu_axis=_load_axis_input(path=cpu_json, required_axis="contest_cpu"),
        cuda_axis=_load_axis_input(path=cuda_json, required_axis="contest_cuda"),
        label="pr101_fec6",
        target_score=0.192,
    )

    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["classification"] == "cpu_positive_cuda_miss_due_to_component_drift"
    assert report["components"]["dominant_score_delta_component"] == "pose"
    assert report["components"]["delta_cuda_minus_cpu"]["score_delta_cuda_minus_cpu"] == pytest.approx(
        0.03415870481239236
    )
    assert report["components"]["score_delta_byte_equivalent"] == pytest.approx(
        51_300.2,
        abs=1.0,
    )
    assert report["target_gaps"]["contest_cpu"]["byte_gap_if_components_unchanged"] == 78
    assert report["target_gaps"]["contest_cuda"]["byte_gap_if_components_unchanged"] > 51_000


def test_cli_writes_report_and_compares_inflated_output_hashes(tmp_path: Path) -> None:
    cpu_json = _write_json(
        tmp_path / "cpu.json",
        _auth_eval_payload(
            axis="contest_cpu",
            device="cpu",
            score=0.1920513168811056,
            seg=0.00056029,
            pose=0.00002943,
            rate=0.004754685709380427,
        ),
    )
    cuda_json = _write_json(
        tmp_path / "cuda.json",
        _auth_eval_payload(
            axis="contest_cuda",
            device="cuda",
            score=0.22621002169349796,
            seg=0.00066299,
            pose=0.00016846,
            rate=0.004754685709380427,
        ),
    )
    cpu_manifest = _write_json(
        tmp_path / "cpu_inflated.json",
        {"aggregate_sha256": "1" * 64, "raw_file_count": 1, "total_bytes": 10},
    )
    cuda_manifest = _write_json(
        tmp_path / "cuda_inflated.json",
        {"aggregate_sha256": "2" * 64, "raw_file_count": 1, "total_bytes": 10},
    )
    out_dir = tmp_path / "out"

    assert main([
        "--cpu-auth-eval-json",
        str(cpu_json),
        "--cuda-auth-eval-json",
        str(cuda_json),
        "--cpu-inflated-outputs-manifest",
        str(cpu_manifest),
        "--cuda-inflated-outputs-manifest",
        str(cuda_manifest),
        "--label",
        "fixture",
        "--output-dir",
        str(out_dir),
    ]) == 0

    report = json.loads((out_dir / "paired_axis_delta.json").read_text())
    assert report["raw_output_comparison"]["aggregate_sha256_match"] is False
    assert (out_dir / "paired_axis_delta.md").exists()
    assert "xray_paired_cpu_cuda_axis_delta.py" in (out_dir / "rebuild_command.txt").read_text()


def test_mismatched_archives_fail_closed(tmp_path: Path) -> None:
    cpu_json = _write_json(
        tmp_path / "cpu.json",
        _auth_eval_payload(
            axis="contest_cpu",
            device="cpu",
            score=0.19,
            seg=0.001,
            pose=0.001,
            rate=0.001,
            archive_sha256="a" * 64,
        ),
    )
    cuda_json = _write_json(
        tmp_path / "cuda.json",
        _auth_eval_payload(
            axis="contest_cuda",
            device="cuda",
            score=0.20,
            seg=0.001,
            pose=0.001,
            rate=0.001,
            archive_sha256="b" * 64,
        ),
    )

    with pytest.raises(ValueError, match="different archive SHA-256"):
        build_report(
            cpu_axis=_load_axis_input(path=cpu_json, required_axis="contest_cpu"),
            cuda_axis=_load_axis_input(path=cuda_json, required_axis="contest_cuda"),
            label="mismatch",
            target_score=0.192,
        )
