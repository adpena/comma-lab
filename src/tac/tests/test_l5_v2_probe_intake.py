# SPDX-License-Identifier: MIT
"""Tests for L5 v2 probe-observation artifact intake."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tac.exact_eval_custody import contest_score
from tac.optimization.l5_v2_probe_intake import (
    DEFAULT_L5V2_PROBE_SOURCE_PATHS,
    build_l5_v2_probe_observation_intake,
    default_l5_v2_probe_source_paths,
    materialized_tt5l_probe_source_metadata,
    materialized_tt5l_probe_source_paths,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_l5_v2_probe_intake_defaults_include_materialized_paired_axis_paths() -> None:
    assert (
        "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/"
        "modal_auth_eval_cpu/"
        "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu/"
        "contest_auth_eval.json"
    ) in DEFAULT_L5V2_PROBE_SOURCE_PATHS


def test_l5_v2_probe_intake_prefers_materialized_plan_outputs(tmp_path: Path) -> None:
    plan_path = (
        tmp_path
        / ".omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json"
    )
    cpu_output = (
        "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/"
        "modal_auth_eval_cpu/"
        "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_abc_cpu"
    )
    cuda_output = (
        "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/"
        "modal_auth_eval/"
        "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_abc_cuda"
    )
    _write_json(
        plan_path,
        {
            "schema": "modal_paired_auth_eval_dispatch_plan_v2",
            "outputs": {
                "contest_cpu": cpu_output,
                "contest_cuda": cuda_output,
            },
            "pair_group_id": "pair-current",
            "run_id": "run-current",
        },
    )

    dynamic = materialized_tt5l_probe_source_paths(repo_root=tmp_path)
    metadata = materialized_tt5l_probe_source_metadata(repo_root=tmp_path)
    defaults = default_l5_v2_probe_source_paths(repo_root=tmp_path)

    assert dynamic == (
        f"{cpu_output}/contest_auth_eval.json",
        f"{cuda_output}/contest_auth_eval.json",
    )
    assert metadata[f"{cpu_output}/contest_auth_eval.json"]["pair_group_id"] == (
        "pair-current"
    )
    assert metadata[f"{cuda_output}/contest_auth_eval.json"]["run_id"] == "run-current"
    assert defaults[:2] == dynamic


def test_l5_v2_probe_intake_applies_materialized_plan_pair_identity(
    tmp_path: Path,
) -> None:
    cpu_output = (
        "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/"
        "modal_auth_eval_cpu/"
        "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_current_cpu"
    )
    cuda_output = (
        "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/"
        "modal_auth_eval/"
        "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_current_cuda"
    )
    _write_json(
        tmp_path
        / ".omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json",
        {
            "outputs": {
                "contest_cpu": cpu_output,
                "contest_cuda": cuda_output,
            },
            "pair_group_id": "pair-current",
            "run_id": "run-current",
        },
    )
    base_payload = {
        "archive_size_bytes": 10,
        "avg_posenet_dist": 0.2,
        "avg_segnet_dist": 0.02,
        "canonical_score": 1.0,
        "n_samples": 600,
    }
    _write_json(
        tmp_path / cpu_output / "contest_auth_eval.json",
        {
            **base_payload,
            "evidence_grade": "contest-CPU",
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cpu",
                "inflate_runtime_manifest": {"runtime_tree_sha256": "b" * 64},
                "platform_machine": "x86_64",
                "platform_system": "Linux",
                "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cpu"],
            },
            "score_axis": "contest_cpu",
        },
    )
    _write_json(
        tmp_path / cuda_output / "contest_auth_eval.json",
        {
            **base_payload,
            "evidence_grade": "contest-CUDA",
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cuda",
                "gpu_model": "Tesla T4",
                "inflate_device_policy": "auto",
                "inflate_runtime_manifest": {"runtime_tree_sha256": "b" * 64},
                "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cuda"],
            },
            "score_axis": "contest_cuda",
        },
    )

    intake = build_l5_v2_probe_observation_intake(repo_root=tmp_path)
    tt5l = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }["time_traveler_l5_autonomy"]

    assert tt5l["exact_axes"] == ["contest_cpu", "contest_cuda"]
    assert tt5l["pair_group_id"] == "pair-current"
    assert tt5l["run_id"] == "run-current"
    assert {
        (row["axis"], row["pair_group_id"], row["run_id"])
        for row in tt5l["axis_evidence"]
    } == {
        ("contest_cpu", "pair-current", "run-current"),
        ("contest_cuda", "pair-current", "run-current"),
    }


def test_l5_v2_probe_intake_does_not_mix_tt5l_axes_across_archive_sha(
    tmp_path: Path,
) -> None:
    cpu = _write_json(
        tmp_path / "tt5l_cpu" / "contest_auth_eval.json",
        {
            "archive_size_bytes": 10,
            "avg_posenet_dist": 0.2,
            "avg_segnet_dist": 0.02,
            "canonical_score": 1.0,
            "evidence_grade": "contest-CPU",
            "n_samples": 600,
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cpu",
                "inflate_runtime_manifest": {"runtime_tree_sha256": "b" * 64},
                "platform_machine": "x86_64",
                "platform_system": "Linux",
                "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cpu"],
            },
            "score_axis": "contest_cpu",
        },
    )
    cuda = _write_json(
        tmp_path / "tt5l_cuda" / "contest_auth_eval.json",
        {
            "archive_size_bytes": 10,
            "avg_posenet_dist": 0.2,
            "avg_segnet_dist": 0.02,
            "canonical_score": 1.0,
            "evidence_grade": "contest-CUDA",
            "n_samples": 600,
            "provenance": {
                "archive_sha256": "c" * 64,
                "device": "cuda",
                "gpu_model": "Tesla T4",
                "inflate_device_policy": "auto",
                "inflate_runtime_manifest": {"runtime_tree_sha256": "d" * 64},
                "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cuda"],
            },
            "score_axis": "contest_cuda",
        },
    )

    intake = build_l5_v2_probe_observation_intake([cpu, cuda], repo_root=tmp_path)
    tt5l = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }["time_traveler_l5_autonomy"]

    assert tt5l["exact_axes"] != ["contest_cpu", "contest_cuda"]
    assert "l5_v2_probe_paired_exact_axes_missing" in tt5l["blockers"]
    assert (
        "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/"
        "modal_auth_eval/"
        "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cuda/"
        "contest_auth_eval.json"
    ) in DEFAULT_L5V2_PROBE_SOURCE_PATHS


def test_l5_v2_probe_intake_does_not_mix_tt5l_axes_across_pair_group(
    tmp_path: Path,
) -> None:
    cpu = _write_json(
        tmp_path / "tt5l_cpu_pair_a" / "contest_auth_eval.json",
        {
            "archive_size_bytes": 10,
            "avg_posenet_dist": 0.2,
            "avg_segnet_dist": 0.02,
            "canonical_score": 1.0,
            "evidence_grade": "contest-CPU",
            "n_samples": 600,
            "pair_group_id": "pair-a",
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cpu",
                "inflate_runtime_manifest": {"runtime_tree_sha256": "b" * 64},
                "platform_machine": "x86_64",
                "platform_system": "Linux",
                "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cpu"],
            },
            "run_id": "run-a",
            "score_axis": "contest_cpu",
        },
    )
    cuda = _write_json(
        tmp_path / "tt5l_cuda_pair_b" / "contest_auth_eval.json",
        {
            "archive_size_bytes": 10,
            "avg_posenet_dist": 0.2,
            "avg_segnet_dist": 0.02,
            "canonical_score": 1.0,
            "evidence_grade": "contest-CUDA",
            "n_samples": 600,
            "pair_group_id": "pair-b",
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cuda",
                "gpu_model": "Tesla T4",
                "inflate_device_policy": "auto",
                "inflate_runtime_manifest": {"runtime_tree_sha256": "b" * 64},
                "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cuda"],
            },
            "run_id": "run-b",
            "score_axis": "contest_cuda",
        },
    )

    intake = build_l5_v2_probe_observation_intake([cpu, cuda], repo_root=tmp_path)
    tt5l = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }["time_traveler_l5_autonomy"]
    source_evidence = [
        row["axis_evidence"]
        for row in intake["source_records"]
        if row.get("axis_evidence")
    ]

    assert {row["pair_group_id"] for row in source_evidence} == {"pair-a", "pair-b"}
    assert tt5l["exact_axes"] != ["contest_cpu", "contest_cuda"]
    assert "l5_v2_probe_paired_exact_axes_missing" in tt5l["blockers"]
    assert "pair_group_id/run_id" in tt5l["notes"]


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
    assert evidence["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
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


def test_l5_v2_probe_intake_rejects_expected_runtime_hash_as_custody(
    tmp_path: Path,
) -> None:
    result_dir = (
        tmp_path
        / "experiments"
        / "results"
        / "modal_auth_eval"
        / "tt5l_cuda_expected_only"
    )
    raw_sha = "c" * 64
    artifact = _write_json(
        result_dir / "contest_auth_eval.json",
        {
            "archive_size_bytes": 34_603,
            "avg_posenet_dist": 0.18,
            "avg_segnet_dist": 0.025,
            "canonical_score": contest_score(
                seg_dist=0.025,
                pose_dist=0.18,
                archive_bytes=34_603,
            ),
            "exact_cuda_evidence": True,
            "lane_id": "lane_time_traveler_l5_autonomy_exact_cuda",
            "n_samples": 600,
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cuda",
                "expected_runtime_tree_sha256": "b" * 64,
                "gpu_model": "Tesla T4",
                "inflated_output_manifest": {
                    "payload": {
                        "aggregate_sha256": raw_sha,
                    },
                },
                "sys_argv": [
                    "experiments/contest_auth_eval.py",
                    "--device",
                    "cuda",
                    "--inflate-device",
                    "auto",
                ],
            },
            "raw_output_aggregate_sha256": raw_sha,
            "score_axis": "contest_cuda",
        },
    )
    (artifact.parent / "contest_auth_eval.stdout.log").write_text(
        "contest auth eval completed\n",
        encoding="utf-8",
    )
    manifest_path = artifact.parent / "inflated_outputs_manifest.json"
    manifest_path.write_text(
        json.dumps({"aggregate_sha256": raw_sha}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    intake = build_l5_v2_probe_observation_intake([artifact], repo_root=tmp_path)

    source = intake["source_records"][0]
    assert source["recognized_for_observation"] is True
    assert source["custody_valid_for_observation"] is False
    assert source["accepted_for_observation"] is False
    assert source["axis_evidence"]["runtime_tree_sha256"] == ""
    assert "runtime_tree_sha_invalid" in source["custody_blockers"]
    tt5l = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }["time_traveler_l5_autonomy"]
    assert tt5l["byte_closed_archive"] is False


def test_l5_v2_probe_intake_rejects_expected_runtime_hash_mismatch(
    tmp_path: Path,
) -> None:
    result_dir = (
        tmp_path
        / "experiments"
        / "results"
        / "modal_auth_eval"
        / "tt5l_cuda_runtime_mismatch"
    )
    raw_sha = "c" * 64
    artifact = _write_json(
        result_dir / "contest_auth_eval.json",
        {
            "archive_size_bytes": 34_603,
            "avg_posenet_dist": 0.18,
            "avg_segnet_dist": 0.025,
            "canonical_score": contest_score(
                seg_dist=0.025,
                pose_dist=0.18,
                archive_bytes=34_603,
            ),
            "exact_cuda_evidence": True,
            "lane_id": "lane_time_traveler_l5_autonomy_exact_cuda",
            "n_samples": 600,
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cuda",
                "expected_runtime_tree_sha256": "d" * 64,
                "gpu_model": "Tesla T4",
                "inflate_runtime_manifest": {
                    "runtime_tree_sha256": "b" * 64,
                },
                "inflated_output_manifest": {
                    "payload": {
                        "aggregate_sha256": raw_sha,
                    },
                },
                "sys_argv": [
                    "experiments/contest_auth_eval.py",
                    "--device",
                    "cuda",
                    "--inflate-device",
                    "auto",
                ],
            },
            "raw_output_aggregate_sha256": raw_sha,
            "score_axis": "contest_cuda",
        },
    )
    (artifact.parent / "contest_auth_eval.stdout.log").write_text(
        "contest auth eval completed\n",
        encoding="utf-8",
    )
    manifest_path = artifact.parent / "inflated_outputs_manifest.json"
    manifest_path.write_text(
        json.dumps({"aggregate_sha256": raw_sha}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    intake = build_l5_v2_probe_observation_intake([artifact], repo_root=tmp_path)

    source = intake["source_records"][0]
    assert source["recognized_for_observation"] is True
    assert source["custody_valid_for_observation"] is False
    assert source["axis_evidence"]["runtime_tree_sha256"] == "b" * 64
    assert source["axis_evidence"]["expected_runtime_tree_sha256"] == "d" * 64
    assert "runtime_tree_sha_mismatch" in source["custody_blockers"]


def test_l5_v2_probe_intake_accepts_modal_cpu_platform_custody(
    tmp_path: Path,
) -> None:
    """Modal CPU results record Linux/x86_64 platform fields, not gpu_model."""

    result_dir = (
        tmp_path
        / "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/"
        "modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu"
    )
    raw_sha = "c" * 64
    manifest = result_dir / "inflated_outputs_manifest.json"
    stdout_log = result_dir / "contest_auth_eval.stdout.log"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"aggregate_sha256": raw_sha}) + "\n",
        encoding="utf-8",
    )
    stdout_log.write_text("ok\n", encoding="utf-8")

    result_json = _write_json(
        result_dir / "contest_auth_eval.json",
        {
            "archive_size_bytes": 34_603,
            "avg_posenet_dist": 0.18,
            "avg_segnet_dist": 0.025,
            "canonical_score": contest_score(
                seg_dist=0.025,
                pose_dist=0.18,
                archive_bytes=34_603,
            ),
            "evidence_grade": "contest-CPU",
            "n_samples": 600,
            "provenance": {
                "archive_sha256": "a" * 64,
                "device": "cpu",
                "gpu_model": "<error:FileNotFoundError(2, 'No such file or directory')>",
                "inflate_device_policy": "auto",
                "inflate_runtime_manifest": {
                    "runtime_tree_sha256": "b" * 64,
                },
                "platform_machine": "x86_64",
                "platform_system": "Linux",
                "sys_argv": [
                    "experiments/contest_auth_eval.py",
                    "--archive",
                    "/tmp/archive.zip",
                    "--inflate-sh",
                    "/tmp/inflate.sh",
                    "--device",
                    "cpu",
                ],
            },
            "raw_output_aggregate_sha256": raw_sha,
            "score_axis": "contest_cpu",
        },
    )

    intake = build_l5_v2_probe_observation_intake(
        [result_json.relative_to(tmp_path)],
        repo_root=tmp_path,
    )
    source = next(
        row
        for row in intake["source_records"]
        if row["path"].endswith("contest_auth_eval.json")
    )
    assert source["custody_valid_for_observation"] is True
    assert source["accepted_for_observation"] is True
    evidence = source["axis_evidence"]
    assert evidence["hardware"] == "Linux x86_64"
    assert evidence["inflate_device"] == "cpu"
    assert evidence["eval_device"] == "cpu"

    tt5l = {
        row["candidate_id"]: row
        for row in intake["verdict"]["evaluated_observations"]
    }["time_traveler_l5_autonomy"]
    assert tt5l["exact_axes"] == ["contest_cpu"]
    assert "l5_v2_probe_axis_hardware_not_contest_cpu:contest_cpu" not in tt5l[
        "blockers"
    ]
    assert "l5_v2_probe_axis_inflate_device_not_cpu:contest_cpu" not in tt5l[
        "blockers"
    ]


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
