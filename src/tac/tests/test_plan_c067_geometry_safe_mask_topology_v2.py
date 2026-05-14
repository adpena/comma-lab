# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.plan_c067_geometry_safe_mask_topology_v2 import (
    ExactNegativeSpec,
    build_plan,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _trace(path: Path, *, archive_sha: str, archive_bytes: int, pose_by_pair: dict[int, float]) -> Path:
    samples = []
    for pair_index, pose in sorted(pose_by_pair.items()):
        samples.append(
            {
                "pair_index": pair_index,
                "frame_indices": [2 * pair_index, 2 * pair_index + 1],
                "posenet_dist": pose,
                "segnet_dist": 0.0005,
                "score_seg_contribution_exact": 0.0001,
                "score_pose_contribution_first_order": pose / 100.0,
                "score_combined_contribution_first_order": 0.0001 + pose / 100.0,
            }
        )
    return _write_json(
        path,
        {
            "schema_version": "component_trace_test",
            "trace_inputs": {"archive_sha256": archive_sha},
            "archive_size_bytes": archive_bytes,
            "samples": samples,
        },
    )


def _auth(path: Path, *, archive_sha: str, archive_bytes: int, score: float, pose: float) -> Path:
    return _write_json(
        path,
        {
            "score_recomputed_from_components": score,
            "avg_posenet_dist": pose,
            "avg_segnet_dist": 0.001,
            "archive_size_bytes": archive_bytes,
            "provenance": {
                "archive_sha256": archive_sha,
                "archive_size_bytes": archive_bytes,
                "gpu_model": "test-gpu",
                "gpu_t4_match": False,
            },
        },
    )


def _triage(path: Path, candidates: list[dict]) -> Path:
    return _write_json(
        path,
        {
            "schema": "c067_bigmove_nontrain_candidate_triage_v1",
            "score_claim": False,
            "ranked_candidates": candidates,
        },
    )


def _claims(path: Path, rows: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# claims\n\n| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + rows,
        encoding="utf-8",
    )
    return path


def _negative_spec(tmp_path: Path, *, sha: str = "neg-sha", pair: int = 5) -> ExactNegativeSpec:
    trace = _trace(
        tmp_path / "neg_trace.json",
        archive_sha=sha,
        archive_bytes=240_000,
        pose_by_pair={pair: 1.0, 10: 0.0001},
    )
    auth = _auth(
        tmp_path / "neg_auth.json",
        archive_sha=sha,
        archive_bytes=240_000,
        score=2.0,
        pose=0.5,
    )
    return ExactNegativeSpec("neg", "mask_topology_global_replacement", auth, trace)


def test_blocks_identical_exact_negative_archive_sha(tmp_path: Path) -> None:
    frontier = _trace(
        tmp_path / "frontier_trace.json",
        archive_sha="frontier",
        archive_bytes=276_214,
        pose_by_pair={5: 0.0001, 10: 0.0001},
    )
    triage = _triage(
        tmp_path / "triage.json",
        [
            {
                "candidate_id": "already_bad",
                "family": "pmg_hotspot_atomtop_byte_screen",
                "archive": {"path": "archive.zip", "sha256": "neg-sha", "bytes": 200_000},
                "builder_command_if_materialization_needed": [],
                "score_claim": False,
            }
        ],
    )
    plan = build_plan(
        repo_root=tmp_path,
        frontier_trace_json=frontier,
        triage_json=triage,
        poseguard_policy_json=None,
        active_claims_md=_claims(tmp_path / "claims.md"),
        negative_specs=[_negative_spec(tmp_path, sha="neg-sha")],
    )

    assert plan["dispatchable_candidate_count"] == 0
    blockers = plan["gated_candidates"][0]["geometry_pose_safety_gate"]["blockers"]
    assert any("identical archive SHA" in blocker for blocker in blockers)


def test_blocks_poseguard_policy_touching_catastrophic_pair(tmp_path: Path) -> None:
    frontier = _trace(
        tmp_path / "frontier_trace.json",
        archive_sha="frontier",
        archive_bytes=276_214,
        pose_by_pair={5: 0.0001, 10: 0.0001},
    )
    policy_json = _write_json(
        tmp_path / "policy.json",
        {
            "schema": "yousfi_fridrich_atom_field_allocator_v1",
            "score_claim": False,
            "candidate_policies": [
                {
                    "policy_id": "touches_bad_pair",
                    "evidence_grade": "planning_only",
                    "score_claim": False,
                    "selected_atom_count": 1,
                    "builder": "builder command",
                    "support": {"top_pair_indices_by_selected_atom_count": [5]},
                    "selected_row_run_atoms": [
                        {"frame_index": 10, "y": 1, "x0": 2, "x1_exclusive": 4, "class_id": 2}
                    ],
                }
            ],
        },
    )
    plan = build_plan(
        repo_root=tmp_path,
        frontier_trace_json=frontier,
        triage_json=_triage(tmp_path / "triage.json", []),
        poseguard_policy_json=policy_json,
        active_claims_md=_claims(tmp_path / "claims.md"),
        negative_specs=[_negative_spec(tmp_path, pair=5)],
    )

    assert plan["dispatchable_candidate_count"] == 0
    blockers = plan["gated_candidates"][0]["geometry_pose_safety_gate"]["blockers"]
    assert any("catastrophic exact-negative pairs" in blocker for blocker in blockers)


def test_can_surface_byte_closed_safe_delta_candidate(tmp_path: Path) -> None:
    frontier = _trace(
        tmp_path / "frontier_trace.json",
        archive_sha="frontier",
        archive_bytes=276_214,
        pose_by_pair={5: 0.0001, 10: 0.0001, 11: 0.0001},
    )
    triage = _triage(
        tmp_path / "triage.json",
        [
            {
                "candidate_id": "safe_delta_overlay",
                "family": "decoded_baseline_delta_overlay",
                "archive": {"path": "safe/archive.zip", "sha256": "safe-sha", "bytes": 240_000},
                "builder_command_if_materialization_needed": [
                    "--pair-indices",
                    "10,11",
                ],
                "score_claim": False,
            }
        ],
    )
    plan = build_plan(
        repo_root=tmp_path,
        frontier_trace_json=frontier,
        triage_json=triage,
        poseguard_policy_json=None,
        active_claims_md=_claims(tmp_path / "claims.md"),
        negative_specs=[_negative_spec(tmp_path, sha="neg-sha", pair=5)],
    )

    assert plan["dispatchable_candidate_count"] == 1
    candidate = plan["dispatchable_candidates"][0]
    assert candidate["candidate_id"] == "safe_delta_overlay"
    assert candidate["geometry_pose_safety_gate"]["status"] == "pass_dispatchable_after_claim"


def test_active_claim_conflict_blocks_dispatch(tmp_path: Path) -> None:
    frontier = _trace(
        tmp_path / "frontier_trace.json",
        archive_sha="frontier",
        archive_bytes=276_214,
        pose_by_pair={5: 0.0001, 10: 0.0001},
    )
    triage = _triage(
        tmp_path / "triage.json",
        [
            {
                "candidate_id": "safe_delta_overlay",
                "family": "decoded_baseline_delta_overlay",
                "archive": {"path": "safe/archive.zip", "sha256": "safe-sha", "bytes": 240_000},
                "builder_command_if_materialization_needed": ["--pair-indices", "10"],
                "score_claim": False,
            }
        ],
    )
    rows = (
        "| 2026-05-02T00:00:00Z | codex | safe_delta_overlay | lightning | job | "
        "2026-05-02T01:00:00Z | eval | active |\n"
    )
    plan = build_plan(
        repo_root=tmp_path,
        frontier_trace_json=frontier,
        triage_json=triage,
        poseguard_policy_json=None,
        active_claims_md=_claims(tmp_path / "claims.md", rows),
        negative_specs=[_negative_spec(tmp_path, sha="neg-sha", pair=5)],
    )

    assert plan["dispatchable_candidate_count"] == 0
    blockers = plan["gated_candidates"][0]["geometry_pose_safety_gate"]["blockers"]
    assert "active dispatch claim conflict exists for this lane/family" in blockers


def test_newer_terminal_claim_closes_older_eval_row(tmp_path: Path) -> None:
    frontier = _trace(
        tmp_path / "frontier_trace.json",
        archive_sha="frontier",
        archive_bytes=276_214,
        pose_by_pair={5: 0.0001, 10: 0.0001},
    )
    triage = _triage(
        tmp_path / "triage.json",
        [
            {
                "candidate_id": "safe_delta_overlay",
                "family": "decoded_baseline_delta_overlay",
                "archive": {"path": "safe/archive.zip", "sha256": "safe-sha", "bytes": 240_000},
                "builder_command_if_materialization_needed": ["--pair-indices", "10"],
                "score_claim": False,
            }
        ],
    )
    rows = (
        "| 2026-05-02T00:10:00Z | codex | safe_delta_overlay | lightning | job | "
        "2026-05-02T00:10:00Z | completed_negative | done |\n"
        "| 2026-05-02T00:00:00Z | codex | safe_delta_overlay | lightning | job | "
        "2026-05-02T01:00:00Z | eval | old active row |\n"
    )
    plan = build_plan(
        repo_root=tmp_path,
        frontier_trace_json=frontier,
        triage_json=triage,
        poseguard_policy_json=None,
        active_claims_md=_claims(tmp_path / "claims.md", rows),
        negative_specs=[_negative_spec(tmp_path, sha="neg-sha", pair=5)],
    )

    assert plan["dispatchable_candidate_count"] == 1


def test_missing_frontier_trace_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="frontier trace JSON missing"):
        build_plan(
            repo_root=tmp_path,
            frontier_trace_json=tmp_path / "missing.json",
            triage_json=_triage(tmp_path / "triage.json", []),
            poseguard_policy_json=None,
            active_claims_md=_claims(tmp_path / "claims.md"),
            negative_specs=[],
        )
