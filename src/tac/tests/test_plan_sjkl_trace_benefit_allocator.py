# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO / "experiments" / "plan_sjkl_trace_benefit_allocator.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_sjkl_trace_benefit_allocator_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _auth(path: Path, *, archive_bytes: int, score: float, pose_score: float, seg_score: float) -> None:
    _write_json(
        path,
        {
            "archive_size_bytes": archive_bytes,
            "avg_posenet_dist": 0.00049,
            "avg_segnet_dist": 0.00061,
            "n_samples": 600,
            "score_pose_contribution": pose_score,
            "score_rate_contribution": archive_bytes * (25.0 / 37_545_489),
            "score_recomputed_from_components": score,
            "score_seg_contribution": seg_score,
        },
    )


def _trace(path: Path, *, archive_bytes: int, archive_sha: str, score: float, pair0: float, pair1: float) -> None:
    samples = [
        {
            "candidate_posenet_dist": 0.0,
            "frame_indices": [0, 1],
            "frame_start": 0,
            "pair_index": 0,
            "posenet_dist": 0.0001,
            "score_combined_contribution_first_order": pair0,
            "score_pose_contribution_first_order": pair0 / 2.0,
            "score_seg_contribution_exact": pair0 / 2.0,
            "segnet_dist": 0.0002,
        },
        {
            "frame_indices": [2, 3],
            "frame_start": 2,
            "pair_index": 1,
            "posenet_dist": 0.0003,
            "score_combined_contribution_first_order": pair1,
            "score_pose_contribution_first_order": pair1 / 2.0,
            "score_seg_contribution_exact": pair1 / 2.0,
            "segnet_dist": 0.0004,
        },
    ]
    while len(samples) < 600:
        pair = len(samples)
        samples.append(
            {
                "frame_indices": [pair * 2, pair * 2 + 1],
                "frame_start": pair * 2,
                "pair_index": pair,
                "posenet_dist": 0.00001,
                "score_combined_contribution_first_order": 0.0,
                "score_pose_contribution_first_order": 0.0,
                "score_seg_contribution_exact": 0.0,
                "segnet_dist": 0.00001,
            }
        )
    _write_json(
        path,
        {
            "archive_size_bytes": archive_bytes,
            "avg_posenet_dist": 0.00049,
            "avg_segnet_dist": 0.00061,
            "n_samples": 600,
            "samples": samples,
            "score_recomputed_from_components": score,
            "trace_inputs": {"archive_sha256": archive_sha},
        },
    )


def _candidate_files(tmp_path: Path, *, archive_sha: str = "c" * 64):
    eval_dir = tmp_path / "eval"
    local_dir = tmp_path / "local"
    source_sha = "b" * 64
    _auth(eval_dir / "contest_auth_eval.adjudicated.json", archive_bytes=1100, score=0.95, pose_score=0.2, seg_score=0.05)
    _trace(eval_dir / "component_trace.json", archive_bytes=1100, archive_sha=archive_sha, score=0.95, pair0=0.08, pair1=0.20)
    _write_json(eval_dir / "eval_provenance.json", {"archive_sha256": archive_sha})
    _write_json(
        eval_dir / "adjudication_provenance.json",
        {
            "contest_cuda_archive_sha256": archive_sha,
            "contest_cuda_gpu_model": "diagnostic",
            "contest_cuda_gpu_t4_match": False,
            "evidence_grade": "A score-grade",
            "promotion_eligible": False,
            "scientific_score_eligible": True,
        },
    )
    _write_json(
        local_dir / "pack" / "sjkl_c067_archive_manifest.json",
        {
            "output_archive": {"bytes": 1100, "sha256": archive_sha},
            "score_claim": False,
            "sjkl_payload": {"bytes": 20, "sha256": "d" * 64},
            "source_archive": {"bytes": 1000, "sha256": source_sha},
        },
    )
    _write_json(
        local_dir / "repack" / "sjkl_repack_manifest.json",
        {
            "alpha_bits": 3,
            "basis_bytes": 10,
            "basis_grid_h": 2,
            "basis_grid_w": 2,
            "coefficient_block_bytes": 5,
            "k": 1,
            "out": {"bytes": 20, "sha256": "d" * 64},
            "requested_pair_indices": [0],
            "residual_gain": 0.03125,
            "score_claim": False,
            "selected_pair_count": 1,
            "selected_pair_indices": [0],
        },
    )
    return eval_dir, local_dir, source_sha


def test_planner_classifies_component_positive_score_negative_candidate(tmp_path: Path) -> None:
    planner = _load_planner()
    eval_dir, local_dir, source_sha = _candidate_files(tmp_path)
    baseline_auth = tmp_path / "baseline" / "contest_auth_eval.adjudicated.json"
    baseline_trace = tmp_path / "baseline" / "component_trace.json"
    _auth(baseline_auth, archive_bytes=1000, score=0.9, pose_score=0.3, seg_score=0.1)
    _trace(baseline_trace, archive_bytes=1000, archive_sha=source_sha, score=0.9, pair0=0.10, pair1=0.10)
    _write_json(baseline_auth.with_name("eval_provenance.json"), {"archive_sha256": source_sha, "gpu_model": "T4", "gpu_t4_match": True})

    plan = planner.build_plan(
        output_json=tmp_path / "plan.json",
        baseline_auth_json=baseline_auth,
        baseline_trace_json=baseline_trace,
        candidates=[
            planner.CandidatePaths(
                label="candidate",
                auth_json=eval_dir / "contest_auth_eval.adjudicated.json",
                trace_json=eval_dir / "component_trace.json",
                pack_manifest=local_dir / "pack" / "sjkl_c067_archive_manifest.json",
                repack_manifest=local_dir / "repack" / "sjkl_repack_manifest.json",
            )
        ],
        byte_screen_jsons=(),
        local_repack_roots=(),
    )

    candidate = plan["candidate_summaries"][0]
    assert candidate["classification"]["frontier_class"] == "score_negative_component_positive"
    assert candidate["score_terms"]["component_benefit_vs_baseline"] == pytest.approx(0.15)
    assert candidate["score_terms"]["shrink_bytes_required_to_break_even"] == 0
    assert candidate["pair_response"]["top_positive_pairs"][0]["pair_index"] == 0
    assert candidate["pair_response"]["top_negative_pairs"][0]["pair_index"] == 1
    assert plan["recommendations"][0]["kind"] == "explicit_pair_policy"
    assert plan["score_claim"] is False
    assert plan["remote_jobs_dispatched"] is False


def test_planner_fails_closed_when_trace_archive_sha_does_not_match_custody(tmp_path: Path) -> None:
    planner = _load_planner()
    eval_dir, local_dir, source_sha = _candidate_files(tmp_path, archive_sha="c" * 64)
    _trace(eval_dir / "component_trace.json", archive_bytes=1100, archive_sha="e" * 64, score=0.95, pair0=0.08, pair1=0.20)
    baseline_auth = tmp_path / "baseline" / "contest_auth_eval.adjudicated.json"
    baseline_trace = tmp_path / "baseline" / "component_trace.json"
    _auth(baseline_auth, archive_bytes=1000, score=0.9, pose_score=0.3, seg_score=0.1)
    _trace(baseline_trace, archive_bytes=1000, archive_sha=source_sha, score=0.9, pair0=0.10, pair1=0.10)
    _write_json(baseline_auth.with_name("eval_provenance.json"), {"archive_sha256": source_sha})

    with pytest.raises(planner.PlannerError, match="output archive SHA does not match trace"):
        planner.build_plan(
            output_json=tmp_path / "plan.json",
            baseline_auth_json=baseline_auth,
            baseline_trace_json=baseline_trace,
            candidates=[
                planner.CandidatePaths(
                    label="candidate",
                    auth_json=eval_dir / "contest_auth_eval.adjudicated.json",
                    trace_json=eval_dir / "component_trace.json",
                    pack_manifest=local_dir / "pack" / "sjkl_c067_archive_manifest.json",
                    repack_manifest=local_dir / "repack" / "sjkl_repack_manifest.json",
                )
            ],
            byte_screen_jsons=(),
            local_repack_roots=(),
        )


def test_planner_fails_closed_on_missing_trace(tmp_path: Path) -> None:
    planner = _load_planner()
    eval_dir, local_dir, source_sha = _candidate_files(tmp_path)
    (eval_dir / "component_trace.json").unlink()
    baseline_auth = tmp_path / "baseline" / "contest_auth_eval.adjudicated.json"
    baseline_trace = tmp_path / "baseline" / "component_trace.json"
    _auth(baseline_auth, archive_bytes=1000, score=0.9, pose_score=0.3, seg_score=0.1)
    _trace(baseline_trace, archive_bytes=1000, archive_sha=source_sha, score=0.9, pair0=0.10, pair1=0.10)
    _write_json(baseline_auth.with_name("eval_provenance.json"), {"archive_sha256": source_sha})

    with pytest.raises(planner.PlannerError, match="required JSON input is missing"):
        planner.build_plan(
            output_json=tmp_path / "plan.json",
            baseline_auth_json=baseline_auth,
            baseline_trace_json=baseline_trace,
            candidates=[
                planner.CandidatePaths(
                    label="candidate",
                    auth_json=eval_dir / "contest_auth_eval.adjudicated.json",
                    trace_json=eval_dir / "component_trace.json",
                    pack_manifest=local_dir / "pack" / "sjkl_c067_archive_manifest.json",
                    repack_manifest=local_dir / "repack" / "sjkl_repack_manifest.json",
                )
            ],
            byte_screen_jsons=(),
            local_repack_roots=(),
        )
