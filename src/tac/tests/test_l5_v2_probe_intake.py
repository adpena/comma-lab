# SPDX-License-Identifier: MIT
"""Tests for L5 v2 probe-observation artifact intake."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.exact_eval_custody import contest_score
from tac.optimization.l5_v2_probe_intake import (
    build_l5_v2_probe_observation_intake,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_l5_v2_probe_intake_classifies_single_tt5l_cuda_as_incomplete(
    tmp_path: Path,
) -> None:
    artifact = _write_json(
        tmp_path / "tt5l_cuda_review.json",
        {
            "lane_id": "lane_time_traveler_l5_autonomy_recovered_exact_eval_20260514",
            "technique": "time_traveler_recovered_tt5l_25ep_cuda_review",
            "score_axis": "contest_cuda",
            "exact_cuda_evidence": True,
            "archive_sha256": "a" * 64,
            "runtime_tree_sha256": "b" * 64,
            "canonical_score": contest_score(
                seg_dist=0.025,
                pose_dist=0.18,
                archive_bytes=34603,
            ),
            "segnet_distortion": 0.025,
            "posenet_distortion": 0.18,
            "empirical_archive_bytes": 34603,
            "custody": {
                "command": ["experiments/contest_auth_eval.py", "--device", "cuda"],
                "device": "cuda",
                "gpu_model": "Tesla T4",
                "n_samples": 600,
            },
        },
    )

    intake = build_l5_v2_probe_observation_intake([artifact], repo_root=tmp_path)

    assert intake["score_claim"] is False
    assert intake["promotion_eligible"] is False
    assert intake["architecture_lock_allowed"] is False
    verdict = intake["verdict"]
    evaluated = {
        row["candidate_id"]: row
        for row in verdict["evaluated_observations"]
    }
    tt5l = evaluated["time_traveler_l5_autonomy"]
    assert tt5l["exact_axes"] == ["contest_cuda"]
    assert "l5_v2_probe_paired_exact_axes_missing" in tt5l["blockers"]
    assert "l5_v2_probe_sideinfo_consumption_missing" in tt5l["blockers"]
    assert "l5_v2_probe_required_candidate_ineligible:time_traveler_l5_autonomy" in (
        verdict["blockers"]
    )
    assert intake["probe_gate_artifact"]["probe_disambiguator"]["verdict"] == verdict


def test_l5_v2_probe_intake_recovers_cuda_auto_inflate_policy_from_command(
    tmp_path: Path,
) -> None:
    artifact = _write_json(
        tmp_path / "tt5l_cuda_review.json",
        {
            "lane_id": "lane_time_traveler_l5_autonomy_recovered_exact_eval_20260514",
            "technique": "time_traveler_recovered_tt5l_25ep_cuda_review",
            "score_axis": "contest_cuda",
            "exact_cuda_evidence": True,
            "archive_sha256": "a" * 64,
            "runtime_tree_sha256": "b" * 64,
            "canonical_score": 3.9,
            "segnet_distortion": 0.025,
            "posenet_distortion": 0.18,
            "empirical_archive_bytes": 34603,
            "custody": {
                "command": [
                    "experiments/contest_auth_eval.py",
                    "--device",
                    "cuda",
                    "--inflate-device",
                    "auto",
                ],
                "gpu_model": "Tesla T4",
                "n_samples": 600,
            },
        },
    )

    intake = build_l5_v2_probe_observation_intake([artifact], repo_root=tmp_path)

    source = intake["source_records"][0]
    evidence = source["axis_evidence"]
    assert evidence["inflate_device"] == "auto"
    assert evidence["eval_device"] == "cuda"
    tt5l = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }["time_traveler_l5_autonomy"]
    assert "l5_v2_probe_axis_inflate_device_missing:contest_cuda" not in tt5l[
        "blockers"
    ]
    assert "l5_v2_probe_axis_inflate_device_not_cuda:contest_cuda" not in tt5l[
        "blockers"
    ]


def test_l5_v2_probe_intake_recovers_direct_auth_eval_provenance_and_local_log(
    tmp_path: Path,
) -> None:
    artifact = _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "modal_auth_eval"
        / "tt5l_cuda"
        / "contest_auth_eval.json",
        {
            "provenance": {
                "sys_argv": [
                    "experiments/contest_auth_eval.py",
                    "--device",
                    "cuda",
                    "--inflate-device",
                    "auto",
                ],
                "device": "cuda",
                "gpu_model": "Tesla T4",
                "inflate_device_policy": "auto",
                "archive_sha256": "a" * 64,
                "inflated_output_manifest": {
                    "payload": {
                        "aggregate_sha256": "c" * 64,
                    },
                },
                "inflate_runtime_manifest": {
                    "runtime_tree_sha256": "b" * 64,
                },
            },
            "score_axis": "contest_cuda",
            "exact_cuda_evidence": True,
            "canonical_score": contest_score(
                seg_dist=0.025,
                pose_dist=0.18,
                archive_bytes=34603,
            ),
            "avg_segnet_dist": 0.025,
            "avg_posenet_dist": 0.18,
            "archive_size_bytes": 34603,
            "n_samples": 600,
            "lane_id": "lane_time_traveler_l5_autonomy_exact_cuda",
        },
    )
    log_path = artifact.parent / "contest_auth_eval.stdout.log"
    log_path.write_text("contest auth eval completed\n", encoding="utf-8")
    manifest_path = artifact.parent / "inflated_outputs_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "contest_auth_eval_inflated_output_manifest_v1",
                "aggregate_sha256": "c" * 64,
                "raw_file_count": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    intake = build_l5_v2_probe_observation_intake([artifact], repo_root=tmp_path)

    evidence = intake["source_records"][0]["axis_evidence"]
    assert evidence["auth_eval_command"] == (
        "experiments/contest_auth_eval.py --device cuda --inflate-device auto"
    )
    assert evidence["hardware"] == "Tesla T4"
    assert evidence["inflate_device"] == "auto"
    assert evidence["eval_device"] == "cuda"
    assert evidence["log_path"].endswith("contest_auth_eval.stdout.log")
    assert evidence["inflated_outputs_manifest_path"].endswith(
        "inflated_outputs_manifest.json"
    )
    assert evidence["inflated_outputs_manifest_sha256"]
    assert evidence["raw_output_aggregate_sha256"] == "c" * 64
    assert intake["source_records"][0]["recognized_for_observation"] is True
    assert intake["source_records"][0]["custody_valid_for_observation"] is True
    assert intake["source_records"][0]["accepted_for_observation"] is True
    tt5l = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }["time_traveler_l5_autonomy"]
    assert "l5_v2_probe_axis_log_path_missing:contest_cuda" not in tt5l["blockers"]
    assert "l5_v2_probe_contest_evidence_grade_missing" not in tt5l["blockers"]


def test_l5_v2_probe_intake_splits_recognition_from_custody_validity(
    tmp_path: Path,
) -> None:
    artifact = _write_json(
        tmp_path / "tt5l_cuda_partial.json",
        {
            "lane_id": "lane_time_traveler_l5_autonomy_exact_cuda",
            "score_axis": "contest_cuda",
            "exact_cuda_evidence": True,
            "archive_sha256": "a" * 64,
            "runtime_tree_sha256": "b" * 64,
            "canonical_score": 3.9,
        },
    )

    intake = build_l5_v2_probe_observation_intake([artifact], repo_root=tmp_path)

    source = intake["source_records"][0]
    assert source["recognized_for_observation"] is True
    assert source["custody_valid_for_observation"] is False
    assert source["accepted_for_observation"] is False
    assert "n_samples_missing" in source["custody_blockers"]
    assert "inflated_outputs_manifest_path_missing" in source["custody_blockers"]


def test_l5_v2_probe_intake_records_missing_candidate_sources(tmp_path: Path) -> None:
    intake = build_l5_v2_probe_observation_intake([], repo_root=tmp_path)

    evaluated = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }
    assert set(evaluated) == {
        "c1_world_model_foveation",
        "z5_predictive_coding_world_model",
        "time_traveler_l5_autonomy",
    }
    for row in evaluated.values():
        assert row["eligible_for_architecture_lock"] is False
        assert "l5_v2_probe_artifact_path_missing" in row["blockers"]


def test_l5_v2_probe_intake_cli_writes_artifacts_and_fails_closed(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_probe_intake_{tmp_path.name}"
    )
    try:
        source = _write_json(
            artifact_root / "tt5l_cuda_review.json",
            {
                "lane_id": "lane_time_traveler_l5_autonomy_recovered_exact_eval_20260514",
                "score_axis": "contest_cuda",
                "exact_cuda_evidence": True,
            },
        )
        output_json = artifact_root / "intake.json"
        output_md = artifact_root / "intake.md"
        gate_out = artifact_root / "gate.json"
        proc = subprocess.run(
            [
                "tools/audit_l5_v2_probe_observations.py",
                "--source-json",
                str(source.relative_to(root)),
                "--output-json",
                str(output_json.relative_to(root)),
                "--output-md",
                str(output_md.relative_to(root)),
                "--probe-gate-out",
                str(gate_out.relative_to(root)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert output_json.is_file()
        assert output_md.is_file()
        assert gate_out.is_file()
        assert "architecture_lock_allowed=false" in proc.stdout
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["schema"] == "l5_v2_probe_observation_intake_v1"
        assert payload["ready_for_exact_eval_dispatch"] is False
    finally:
        if artifact_root.exists():
            import shutil

            shutil.rmtree(artifact_root)
