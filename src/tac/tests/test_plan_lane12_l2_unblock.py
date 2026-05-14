# SPDX-License-Identifier: MIT
"""Tests for the fail-closed Lane 12 L2 unblock readiness planner."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_lane12_l2_unblock.py"
SPEC = importlib.util.spec_from_file_location("plan_lane12_l2_unblock", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


PROMOTION_THRESHOLDS = {
    "global_disagreement_max": 0.001,
    "boundary_band_disagreement_max": {"1": 0.002, "2": 0.002, "3": 0.002, "5": 0.002},
    "stable_region_false_flip_rate_max": 0.002,
    "pair_transition_disagreement_max": 0.002,
    "pair_transition_f1_min": None,
    "class_recall_min": {"1": 0.999, "2": 0.999},
    "tiny_speckle_rate_max": 0.0001,
    "max_component_centroid_jump_px": 1.0,
    "missing_component_rate_max": 0.0,
}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _geometry_payload(*, overall_pass: bool, threshold_preset: str = "promotion") -> dict[str, Any]:
    if threshold_preset == "promotion":
        thresholds = PROMOTION_THRESHOLDS
    elif threshold_preset == "exploratory":
        thresholds = {
            "global_disagreement_max": 0.003,
            "boundary_band_disagreement_max": {"1": 0.005, "2": 0.005, "3": 0.005, "5": 0.005},
            "stable_region_false_flip_rate_max": 0.004,
            "pair_transition_disagreement_max": 0.004,
            "pair_transition_f1_min": None,
            "class_recall_min": {},
            "tiny_speckle_rate_max": 0.0005,
            "max_component_centroid_jump_px": 1.0,
            "missing_component_rate_max": 0.0,
        }
    else:
        thresholds = {}
    return {
        "diagnostic": "alpha_geo_0_nerv_geometry",
        "score_evidence_grade": "empirical",
        "device": "cpu",
        "scorer_proxy": False,
        "shape": {"frames": 1200, "height": 384, "width": 512, "num_classes": 5},
        "pass_fail": {
            "overall_pass": overall_pass,
            "checks": {
                "global_disagreement": {
                    "passed": overall_pass,
                    "value": 0.0007 if overall_pass else 0.0123,
                    "threshold": 0.001,
                    "op": "<=",
                },
                "boundary_band_2px_disagreement": {
                    "passed": overall_pass,
                    "value": 0.0015 if overall_pass else 0.148,
                    "threshold": 0.002,
                    "op": "<=",
                },
            },
            "thresholds": {
                **thresholds,
            },
        },
        "diagnostic_config": {
            "threshold_preset": threshold_preset,
            "thresholds": thresholds,
        },
        "global": {"global_disagreement": 0.0007 if overall_pass else 0.0123},
        "boundary_bands": {"2": {"disagreement_rate": 0.0015 if overall_pass else 0.148}},
        "temporal": {"pair_transition": {"disagreement_rate": 0.001 if overall_pass else 0.009}},
        "inputs": {
            "baseline_source": {
                "source_sha256": "b" * 64,
                "archive_member_resolved": "masks.mkv",
            },
            "candidate_source": {
                "source_sha256": "c" * 64,
                "archive_member_resolved": "masks.nrv",
                "decoded_mask_shape": [1200, 384, 512],
            },
        },
    }


def _primitive_contract_payload() -> dict[str, Any]:
    return {
        "diagnostic": "alpha_geo_primitive_contract_v1",
        "score_evidence_grade": "empirical",
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
        "source": {
            "baseline": {
                "archive_member": "masks.mkv",
                "archive_sha256": "b" * 64,
                "decoded_mask_dtype": "torch.uint8",
                "decoded_mask_sha256": "a" * 64,
                "decoded_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
                "decoded_mask_shape": [1200, 384, 512],
            },
            "failed_candidate": {
                "archive_member": "masks.nrv",
                "archive_sha256": "c" * 64,
            },
        },
        "threshold_gates": {
            "exploratory_retrain_gate": {
                "passed": False,
                "blockers": ["global_disagreement"],
                "observed": {"global_disagreement": 0.0123},
                "thresholds": {"global_disagreement_max": 0.003},
            }
        },
    }


def _exact_cuda_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "archive_size_bytes": 296478,
        "avg_posenet_dist": 49.7784996,
        "avg_segnet_dist": 0.03528685,
        "n_samples": 600,
        "score_recomputed_from_components": 26.03719330455429,
        "provenance": {
            "tool": "experiments/contest_auth_eval.py",
            "device": "cuda",
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "gpu_t4_match": False,
            "archive_sha256": "c" * 64,
        },
    }


def _pose_regen_summary_payload(
    *,
    passed: bool = True,
    candidate_archive_sha256: str = "c" * 64,
) -> dict[str, Any]:
    if not passed:
        return {
            "schema_version": 1,
            "tool": "experiments/alpha_geo0_pose_regen.py",
            "stage": "decode_candidate_masks",
            "passed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "error": "RuntimeError: diagnose_nerv_geometry failed",
            "commands": [{"name": "diagnose_nerv_geometry", "returncode": 2}],
        }
    return {
        "schema_version": 1,
        "tool": "experiments/alpha_geo0_pose_regen.py",
        "stage": "done",
        "passed": True,
        "score_claim": True,
        "promotion_eligible": False,
        "archive_sha256": "d" * 64,
        "archive_size_bytes": 296478,
        "commands": [
            {"name": "diagnose_nerv_geometry", "returncode": 0},
            {"name": "optimize_poses", "returncode": 0},
            {"name": "contest_auth_eval", "returncode": 0},
            {"name": "adjudicate_contest_auth_eval", "returncode": 0},
        ],
        "inputs": {
            "candidate_archive": {"sha256": candidate_archive_sha256},
            "baseline_archive": {"sha256": "b" * 64},
        },
    }


def test_missing_clearance_and_failed_geometry_are_reported_fail_closed(tmp_path: Path) -> None:
    geometry = _write_json(tmp_path / "alpha_geo_failed.json", _geometry_payload(overall_pass=False))
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    exact = _write_json(tmp_path / "contest_auth_eval.json", _exact_cuda_payload())
    clearance = tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json"
    output = tmp_path / "readiness.json"

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        clearance_json=clearance,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        exact_evidence_jsons=[exact],
        use_default_artifact_globs=False,
        output_json=output,
    )

    assert output.read_text() == json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert clearance.exists() is False
    assert report["clearance_state_written"] is False
    assert report["readiness_summary"]["ready_for_retraining_unblock"] is False
    assert report["readiness_summary"]["ready_for_exact_eval_dispatch"] is False

    missing_codes = {
        item["code"] for item in report["evidence_buckets"]["missing_prerequisites"]
    }
    assert "clearance_packet_invalid_or_missing" in missing_codes
    assert "no_passing_alpha_geo_geometry" in missing_codes
    assert "pose_regeneration_provenance_missing" in missing_codes
    assert report["readiness_summary"]["usable_primitive_contract_count"] == 1

    empirical = report["evidence_buckets"]["empirical_evidence"]
    assert empirical["geometry_jsons"][0]["geometry_gate_passed"] is False
    assert "failed check: global_disagreement" in empirical["geometry_jsons"][0]["blockers"]
    exact_rows = report["evidence_buckets"]["exact_evidence"]
    assert exact_rows[0]["exact_cuda"] is True
    assert exact_rows[0]["outcome_vs_current_frontier"] == "worse_than_current_frontier"
    assert "no_clearance_by_itself" in exact_rows[0]["allowed_use"]

    target = report["redesign_recipe"]["training_target"]
    assert target["gt_masks_source"] == "decoded-baseline"
    assert target["segnet_target_default_allowed"] is False
    assert target["segnet_forensic_escape_hatch"]["trainer_flag"] == "--allow-forensic-segnet-target"


def test_primitive_contract_requires_decoded_mask_custody(tmp_path: Path) -> None:
    contract_payload = _primitive_contract_payload()
    contract_payload["source"]["baseline"].pop("decoded_mask_sha256")
    contract = _write_json(tmp_path / "primitive_contract.json", contract_payload)

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        primitive_contract_jsons=[contract],
        use_default_artifact_globs=False,
        output_json=None,
    )

    row = report["evidence_buckets"]["empirical_evidence"]["primitive_contracts"][0]
    assert row["usable_for_decoded_baseline_training"] is False
    assert "source.baseline.decoded_mask_sha256 must be a SHA-256 hex digest" in row["blockers"]
    assert report["readiness_summary"]["usable_primitive_contract_count"] == 0


def test_alpha_geo_geometry_requires_candidate_and_baseline_archive_custody(
    tmp_path: Path,
) -> None:
    geometry_payload = _geometry_payload(overall_pass=True)
    geometry_payload["inputs"]["baseline_source"].pop("source_sha256")
    geometry_payload["inputs"]["candidate_source"].pop("source_sha256")
    geometry = _write_json(tmp_path / "alpha_geo_no_sha.json", geometry_payload)

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        geometry_jsons=[geometry],
        use_default_artifact_globs=False,
        output_json=None,
    )

    row = report["evidence_buckets"]["empirical_evidence"]["geometry_jsons"][0]
    assert row["geometry_gate_passed"] is False
    assert "inputs.baseline_source source SHA-256 must be recorded" in row["blockers"]
    assert "inputs.candidate_source source SHA-256 must be recorded" in row["blockers"]
    assert report["readiness_summary"]["passing_geometry_count"] == 0


def test_failed_pose_regen_summary_is_discovered_but_not_usable(tmp_path: Path) -> None:
    geometry = _write_json(tmp_path / "alpha_geo_pass.json", _geometry_payload(overall_pass=True))
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    pose = _write_json(tmp_path / "alpha_geo0_summary.json", _pose_regen_summary_payload(passed=False))

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        pose_regeneration_provenance=[pose],
        use_default_artifact_globs=False,
        output_json=None,
    )

    row = report["evidence_buckets"]["empirical_evidence"]["pose_regeneration_provenance"][0]
    assert row["usable_for_exact_eval_dispatch"] is False
    assert "alpha_geo0_pose_regen summary passed must be true" in row["blockers"]
    assert "alpha_geo0_pose_regen stage must be done" in row["blockers"]
    assert report["readiness_summary"]["usable_pose_regeneration_provenance_count"] == 0


def test_clearance_evidence_paths_must_exist(tmp_path: Path) -> None:
    geometry = _write_json(tmp_path / "alpha_geo_pass.json", _geometry_payload(overall_pass=True))
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    clearance = _write_json(
        tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json",
        {
            "lane_id": "lane_12_nerv_mask_codec",
            "cleared_for_retraining_unblock": True,
            "lane12_l2": True,
            "geometry_gate_passed": True,
            "grand_council_clean_passes": 3,
            "evidence": ["missing_l2_review.md"],
        },
    )

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        clearance_json=clearance,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        use_default_artifact_globs=False,
        output_json=None,
    )

    assert report["launcher_criteria"]["passed"] is False
    assert any(
        "evidence path does not exist as a file" in violation
        for violation in report["launcher_criteria"]["violations"]
    )
    assert report["readiness_summary"]["ready_for_retraining_unblock"] is False


def test_exploratory_geometry_cannot_clear_l2_even_if_overall_passes(tmp_path: Path) -> None:
    evidence = tmp_path / ".omx" / "research" / "lane12_l2_review.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("clean pass evidence\n")
    geometry = _write_json(
        tmp_path / "alpha_geo_exploratory_pass.json",
        _geometry_payload(overall_pass=True, threshold_preset="exploratory"),
    )
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    clearance = _write_json(
        tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json",
        {
            "lane_id": "lane_12_nerv_mask_codec",
            "cleared_for_retraining_unblock": True,
            "lane12_l2": True,
            "geometry_gate_passed": True,
            "grand_council_clean_passes": 3,
            "evidence": [".omx/research/lane12_l2_review.md"],
        },
    )

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        clearance_json=clearance,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        use_default_artifact_globs=False,
        output_json=None,
    )

    row = report["evidence_buckets"]["empirical_evidence"]["geometry_jsons"][0]
    assert row["overall_pass"] is True
    assert row["threshold_preset"] == "exploratory"
    assert row["promotion_thresholds_passed"] is False
    assert "diagnostic_config.threshold_preset must be promotion" in row["blockers"]
    assert report["readiness_summary"]["ready_for_retraining_unblock"] is False


def test_passing_clearance_geometry_contract_and_pose_make_unblock_ready(tmp_path: Path) -> None:
    evidence = tmp_path / ".omx" / "research" / "lane12_l2_review.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("clean pass evidence\n")
    geometry = _write_json(tmp_path / "alpha_geo_pass.json", _geometry_payload(overall_pass=True))
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    pose = _write_json(tmp_path / "pose_regen_provenance.json", _pose_regen_summary_payload())
    clearance = _write_json(
        tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json",
        {
            "lane_id": "lane_12_nerv",
            "cleared_for_retraining_unblock": True,
            "lane12_l2": True,
            "geometry_gate_passed": True,
            "grand_council_clean_passes": 3,
            "evidence": [".omx/research/lane12_l2_review.md"],
        },
    )

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        clearance_json=clearance,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        pose_regeneration_provenance=[pose],
        use_default_artifact_globs=False,
        output_json=None,
    )

    assert report["launcher_criteria"]["passed"] is True
    assert report["readiness_summary"]["ready_for_retraining_unblock"] is True
    assert report["readiness_summary"]["ready_for_exact_eval_dispatch"] is True
    assert report["readiness_summary"]["passing_geometry_count"] == 1
    assert report["readiness_summary"]["usable_primitive_contract_count"] == 1
    assert report["readiness_summary"]["usable_pose_regeneration_provenance_count"] == 1
    assert report["readiness_summary"]["matched_alpha_geo_pose_candidate_count"] == 1
    match = report["evidence_buckets"]["provenance_closure"]["alpha_geo_pose_candidate_match"]
    assert match["passed"] is True
    assert match["matches"][0]["candidate_archive_sha256"] == "c" * 64
    assert report["readiness_summary"]["write_clearance_packet"] is False
    assert report["clearance_state_written"] is False
    assert report["evidence_buckets"]["missing_prerequisites"] == []


def test_exact_eval_dispatch_requires_pose_provenance_for_same_geometry_candidate(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / ".omx" / "research" / "lane12_l2_review.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("clean pass evidence\n")
    geometry = _write_json(tmp_path / "alpha_geo_pass.json", _geometry_payload(overall_pass=True))
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    pose = _write_json(
        tmp_path / "pose_regen_provenance.json",
        _pose_regen_summary_payload(candidate_archive_sha256="e" * 64),
    )
    clearance = _write_json(
        tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json",
        {
            "lane_id": "lane_12_nerv",
            "cleared_for_retraining_unblock": True,
            "lane12_l2": True,
            "geometry_gate_passed": True,
            "grand_council_clean_passes": 3,
            "evidence": [".omx/research/lane12_l2_review.md"],
        },
    )

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        clearance_json=clearance,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        pose_regeneration_provenance=[pose],
        use_default_artifact_globs=False,
        output_json=None,
    )

    assert report["readiness_summary"]["ready_for_retraining_unblock"] is True
    assert report["readiness_summary"]["usable_pose_regeneration_provenance_count"] == 1
    assert report["readiness_summary"]["matched_alpha_geo_pose_candidate_count"] == 0
    assert report["readiness_summary"]["ready_for_exact_eval_dispatch"] is False
    missing_codes = {
        item["code"] for item in report["evidence_buckets"]["missing_prerequisites"]
    }
    assert "alpha_geo_pose_regen_candidate_mismatch" in missing_codes
    match = report["evidence_buckets"]["provenance_closure"]["alpha_geo_pose_candidate_match"]
    assert match["passed"] is False
    assert match["unmatched_passing_geometry"][0]["candidate_archive_sha256"] == "c" * 64
    assert match["usable_pose_candidate_archive_sha256"] == ["e" * 64]


def test_write_clearance_packet_materializes_valid_packet_when_local_evidence_is_green(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / ".omx" / "research" / "lane12_l2_review.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("three clean adversarial review passes\n")
    geometry = _write_json(tmp_path / "alpha_geo_pass.json", _geometry_payload(overall_pass=True))
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    clearance = tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json"

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        clearance_json=clearance,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        use_default_artifact_globs=False,
        output_json=None,
        write_clearance_packet=True,
        clearance_evidence=[evidence],
        grand_council_clean_passes=3,
        command=["unit-test-lane12-clearance-write"],
    )

    packet = json.loads(clearance.read_text())
    assert packet["schema"] == "lane12_nerv_l2_clearance_v1"
    assert packet["lane_id"] == "lane_12_nerv_mask_codec"
    assert packet["cleared_for_retraining_unblock"] is True
    assert packet["lane12_l2"] is True
    assert packet["geometry_gate_passed"] is True
    assert packet["grand_council_clean_passes"] == 3
    assert packet["evidence"] == [".omx/research/lane12_l2_review.md"]
    assert packet["geometry_evidence"][0]["path"] == "alpha_geo_pass.json"
    assert packet["primitive_contract_evidence"][0]["path"] == "primitive_contract.json"
    assert packet["remote_job_launched"] is False
    assert report["clearance_state_written"] is True
    assert report["launcher_criteria"]["passed"] is True
    assert report["readiness_summary"]["ready_for_retraining_unblock"] is True
    assert report["readiness_summary"]["ready_for_exact_eval_dispatch"] is False
    assert report["readiness_summary"]["state_write_performed"] is True


def test_write_clearance_packet_refuses_failed_geometry_without_state_write(tmp_path: Path) -> None:
    evidence = tmp_path / ".omx" / "research" / "lane12_l2_review.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("review exists but geometry is not green\n")
    geometry = _write_json(tmp_path / "alpha_geo_failed.json", _geometry_payload(overall_pass=False))
    contract = _write_json(tmp_path / "primitive_contract.json", _primitive_contract_payload())
    clearance = tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json"

    report = planner.plan_lane12_l2_unblock(
        repo_root=tmp_path,
        clearance_json=clearance,
        geometry_jsons=[geometry],
        primitive_contract_jsons=[contract],
        use_default_artifact_globs=False,
        output_json=None,
        write_clearance_packet=True,
        clearance_evidence=[evidence],
        grand_council_clean_passes=3,
    )

    assert clearance.exists() is False
    assert report["clearance_state_written"] is False
    assert report["launcher_criteria"]["passed"] is False
    assert report["readiness_summary"]["eligible_to_create_clearance_packet"] is False
    assert report["readiness_summary"]["ready_for_retraining_unblock"] is False
    missing_codes = {
        item["code"] for item in report["evidence_buckets"]["missing_prerequisites"]
    }
    assert "no_passing_alpha_geo_geometry" in missing_codes
