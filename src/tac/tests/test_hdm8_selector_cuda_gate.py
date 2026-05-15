# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.hdm8_selector_cuda_gate import (
    HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA,
    build_hdm8_selector_cuda_component_gate,
    validate_hdm8_selector_cuda_component_gate,
    validate_hdm8_selector_cuda_gate_context,
)


def _write_auth_eval(
    path: Path,
    *,
    archive_sha256: str,
    archive_bytes: int = 123,
    score: float = 0.20,
    pose: float = 0.001,
    seg: float = 0.002,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cuda",
                "evidence_grade": "contest-CUDA",
                "exact_cuda_eval_complete": True,
                "score_recomputed_from_components": score,
                "avg_posenet_dist": pose,
                "avg_segnet_dist": seg,
                "n_samples": 600,
                "score_claim": True,
                "score_claim_valid": True,
                "provenance": {
                    "archive_sha256": archive_sha256,
                    "archive_size_bytes": archive_bytes,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_mps_selector_gate_fails_closed_without_cuda_component_probe(tmp_path: Path) -> None:
    archive_sha = "a" * 64
    reference = _write_auth_eval(tmp_path / "reference.json", archive_sha256="b" * 64)

    gate = build_hdm8_selector_cuda_component_gate(
        proxy={
            "axis": "local-mps-proxy-prefix",
            "n_pairs": 600,
            "avg_posenet_dist": 0.0001,
            "avg_segnet_dist": 0.001,
            "baseline_avg_posenet_dist": 0.001,
            "baseline_avg_segnet_dist": 0.001,
            "delta_vs_none_charged": -0.01,
        },
        candidate_archive_sha256=archive_sha,
        candidate_archive_bytes=187_366,
        repo_root=tmp_path,
        reference_result_path=reference,
    )

    assert gate["schema"] == HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA
    assert gate["passed"] is False
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert "mps_or_local_proxy_axis_requires_cuda_component_probe" in gate["blockers"]
    assert "hdm8_selector_cuda_component_gate_not_passed" in validate_hdm8_selector_cuda_component_gate(
        gate,
        expected_archive_sha256=archive_sha,
    )


def test_cuda_prefix_selector_gate_passes_when_pose_seg_and_score_do_not_regress(
    tmp_path: Path,
) -> None:
    archive_sha = "c" * 64
    reference = _write_auth_eval(tmp_path / "reference.json", archive_sha256="d" * 64)

    gate = build_hdm8_selector_cuda_component_gate(
        proxy={
            "axis": "modal-t4-cuda-proxy-prefix",
            "n_pairs": 32,
            "avg_posenet_dist": 0.0009,
            "avg_segnet_dist": 0.002,
            "baseline_avg_posenet_dist": 0.001,
            "baseline_avg_segnet_dist": 0.002,
            "delta_vs_none_charged": -0.0001,
        },
        candidate_archive_sha256=archive_sha,
        candidate_archive_bytes=187_000,
        repo_root=tmp_path,
        reference_result_path=reference,
        min_cuda_prefix_pairs=24,
    )

    assert gate["passed"] is True
    assert gate["status"] == "passed_cuda_prefix_component_check"
    assert gate["ready_for_exact_eval_dispatch"] is True
    assert gate["component_deltas"]["pose_delta"] < 0
    assert validate_hdm8_selector_cuda_component_gate(
        gate,
        expected_archive_sha256=archive_sha,
    ) == []


def test_cuda_prefix_selector_gate_tolerates_float_noise_on_neutral_seg(
    tmp_path: Path,
) -> None:
    archive_sha = "0" * 64
    reference = _write_auth_eval(tmp_path / "reference.json", archive_sha256="1" * 64)

    gate = build_hdm8_selector_cuda_component_gate(
        proxy={
            "axis": "modal-t4-cuda-proxy-prefix",
            "n_pairs": 600,
            "avg_posenet_dist": 0.0009,
            "avg_segnet_dist": 0.002000000001,
            "baseline_avg_posenet_dist": 0.001,
            "baseline_avg_segnet_dist": 0.002,
            "delta_vs_none_charged": -0.0001,
        },
        candidate_archive_sha256=archive_sha,
        candidate_archive_bytes=187_000,
        repo_root=tmp_path,
        reference_result_path=reference,
        min_cuda_prefix_pairs=600,
    )

    assert gate["passed"] is True
    assert not any(
        blocker.startswith("segnet_delta_exceeds_threshold")
        for blocker in gate["blockers"]
    )


def test_exact_cuda_candidate_gate_blocks_posenet_regression(tmp_path: Path) -> None:
    reference_sha = "e" * 64
    candidate_sha = "f" * 64
    reference = _write_auth_eval(
        tmp_path / "reference.json",
        archive_sha256=reference_sha,
        score=0.206,
        pose=0.000032,
        seg=0.000642,
    )
    candidate = _write_auth_eval(
        tmp_path / "candidate.json",
        archive_sha256=candidate_sha,
        score=0.216,
        pose=0.000073,
        seg=0.000642,
    )

    gate = build_hdm8_selector_cuda_component_gate(
        proxy=None,
        candidate_archive_sha256=candidate_sha,
        candidate_archive_bytes=187_366,
        repo_root=tmp_path,
        reference_result_path=reference,
        candidate_exact_cuda_result_path=candidate,
    )

    assert gate["passed"] is False
    assert gate["status"] == "blocked"
    assert any(
        blocker.startswith("posenet_delta_exceeds_threshold")
        for blocker in gate["blockers"]
    )
    assert any(
        blocker.startswith("score_delta_exceeds_threshold")
        for blocker in gate["blockers"]
    )


def test_context_validator_requires_gate_for_hdm8_selector_manifest(tmp_path: Path) -> None:
    row = {"candidate_archive_sha256": "1" * 64}
    manifest = {
        "schema": "hdm8_film_grain_sidecar_packet_manifest_v1",
        "postfilter_mode": "selector",
    }

    blockers, facts = validate_hdm8_selector_cuda_gate_context(
        row,
        manifest,
        expected_archive_sha256="1" * 64,
    )

    assert "hdm8_selector_cuda_component_gate_missing" in blockers
    assert facts["hdm8_selector_cuda_component_gate_required"] is True
