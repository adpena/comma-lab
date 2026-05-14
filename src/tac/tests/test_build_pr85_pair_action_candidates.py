# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_pair_action_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_pair_action_candidates_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _atom(pair_index: int = 7) -> dict:
    return {
        "atom_id": f"fixture_eval:pair_{pair_index:04d}",
        "pair_index": pair_index,
        "frame_indices": [pair_index * 2, pair_index * 2 + 1],
        "ranking_score": 0.0003,
        "byte_break_even": {
            "combined": {
                "max_charged_bytes_for_zero_net_change": 450.0,
                "dscore_darchive_byte": module.RATE_SCORE_PER_BYTE,
            },
            "pose_only": {
                "max_charged_bytes_for_zero_net_change": 300.0,
                "dscore_darchive_byte": module.RATE_SCORE_PER_BYTE,
            },
            "seg_only": {
                "max_charged_bytes_for_zero_net_change": 150.0,
                "dscore_darchive_byte": module.RATE_SCORE_PER_BYTE,
            },
        },
        "dispatch_gate": {
            "dispatchable": False,
            "status": "blocked_planning_only",
            "blockers": ["planner_output_is_not_an_archive"],
        },
    }


def _scorer_plan(path: Path, *, archive_sha: str, archive_bytes: int, pairs: tuple[int, ...] = (7, 9)) -> Path:
    payload = {
        "schema": module.SCORER_PLAN_SCHEMA,
        "producer": "fixture",
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "compression_time_only": True,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "inflate_time_scorer_load_allowed": False,
        "exact_eval": {
            "archive_size_bytes": archive_bytes,
            "avg_posenet_dist": 0.0002,
            "avg_segnet_dist": 0.0005,
            "n_samples": module.PAIR_COUNT,
            "reported_score": 0.25,
            "provenance": {
                "archive_sha256": archive_sha,
                "archive_size_bytes": archive_bytes,
                "device": "cuda",
                "gpu_model": "fixture",
                "tool": "experiments/contest_auth_eval.py",
            },
        },
        "atom_ranking": [_atom(pair) for pair in pairs],
    }
    payload["stable_plan_digest_sha256"] = module._stable_digest(payload)
    return _write_json(path, payload)


def _pair_planning(path: Path, *, scorer: Path, archive_sha: str, archive_bytes: int) -> Path:
    payload = {
        "schema": module.PAIR_ATOM_READINESS_SCHEMA,
        "tool": "fixture",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "dispatch_unlocked": False,
        "candidate_archive_count": 0,
        "source_archive": {
            "path": "fixture/archive.zip",
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "member_sha256": "b" * 64,
        },
        "scorer_gradient_plan": {
            "path": str(scorer),
            "sha256": module._sha256_file(scorer),
            "top_atoms": [_atom(7), _atom(9)],
        },
    }
    return _write_json(path, payload)


def _fixtures(tmp_path: Path) -> tuple[Path, Path, str, int]:
    archive_sha = "a" * 64
    archive_bytes = 1234
    scorer = _scorer_plan(tmp_path / "scorer.json", archive_sha=archive_sha, archive_bytes=archive_bytes)
    pair = _pair_planning(tmp_path / "pair_planning.json", scorer=scorer, archive_sha=archive_sha, archive_bytes=archive_bytes)
    return pair, scorer, archive_sha, archive_bytes


def _action_evidence(
    path: Path,
    *,
    scorer_sha: str,
    pair_index: int = 7,
    source_value: int = 4,
    candidate_value: int = 9,
    thresholds: list[dict] | None = None,
    duplicate_pair: bool = False,
    archive_changing_path: dict | None = None,
) -> Path:
    actions = [
        {
            "pair_index": pair_index,
            "stream": "frac2",
            "op": "set",
            "source_value": source_value,
            "candidate_value": candidate_value,
            "source_artifact_sha256": scorer_sha,
            "source_atom_id": f"fixture_eval:pair_{pair_index:04d}",
            "rationale": "fixture measured action direction",
        }
    ]
    if duplicate_pair:
        actions.append(
            {
                "pair_index": pair_index,
                "stream": "bias",
                "op": "set",
                "source_value": 13,
                "candidate_value": 12,
                "source_artifact_sha256": scorer_sha,
                "source_atom_id": f"fixture_eval:pair_{pair_index:04d}",
                "rationale": "duplicate fixture action",
            }
        )
    return _write_json(
        path,
        {
            "schema": module.ACTION_EVIDENCE_SCHEMA,
            "score_claim": False,
            "dispatch_performed": False,
            "remote_jobs_dispatched": False,
            "thresholds": thresholds or [],
            "candidates": [
                {
                    "candidate_id": "pair7_frac2_set9",
                    "targeted_component": "combined",
                    "header_mode": "explicit_30",
                    "charged_bytes_proxy": {
                        "bytes": 12,
                        "basis": "fixture action-spec byte proxy",
                        "source_artifact_sha256": scorer_sha,
                    },
                    "archive_changing_path": archive_changing_path,
                    "actions": actions,
                }
            ],
        },
    )


def test_unlowered_pair_gradient_specs_stay_dispatch_locked(tmp_path: Path) -> None:
    pair, _scorer, _archive_sha, _archive_bytes = _fixtures(tmp_path)

    summary = module.build_pair_action_candidates(
        pair_atom_planning_json=pair,
        top_n=2,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["ready_for_exact_eval_after_lane_claim_count"] == 0
    assert summary["blocker_class"] == "missing_pair_action_evidence"
    assert [row["candidate_id"] for row in summary["candidates"]] == [
        "pr85_pair_0007_unlowered",
        "pr85_pair_0009_unlowered",
    ]
    first = summary["candidates"][0]
    assert first["actions"] == []
    assert first["charged_bytes_proxy"]["status"] == "blocked_no_charged_action"
    assert first["no_op_status"]["status"] == "missing_action_delta"
    assert first["dispatch_unlocked"] is False


def test_grounded_action_evidence_emits_action_spec_but_not_exact_eval_unlock(tmp_path: Path) -> None:
    pair, scorer, _archive_sha, _archive_bytes = _fixtures(tmp_path)
    scorer_sha = module._sha256_file(scorer)
    action = _action_evidence(tmp_path / "actions.json", scorer_sha=scorer_sha)

    summary = module.build_pair_action_candidates(
        pair_atom_planning_json=pair,
        action_evidence_json=action,
        top_n=2,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["ready_for_pair_atom_archive_build_count"] == 1
    assert summary["ready_for_exact_eval_after_lane_claim_count"] == 0
    assert summary["blocker_class"] == "no_archive_changing_path"
    candidate = summary["candidates"][0]
    assert candidate["lowering_status"] == "action_spec_emitted"
    assert candidate["ready_for_pair_atom_archive_build"] is True
    assert candidate["ready_for_exact_eval_after_lane_claim"] is False
    assert candidate["selected_pair_indices"] == [7]
    assert candidate["no_op_status"] == {"status": "non_noop_value_change", "is_noop": False}
    assert candidate["charged_bytes_proxy"]["candidate_action_bytes"] == 12
    assert candidate["pair_atom_action_spec"] == {
        "schema": module.PAIR_ATOM_ACTION_SPEC_SCHEMA,
        "score_claim": False,
        "dispatch_performed": False,
        "inflate_time_scorer_load_allowed": False,
        "candidate_id": "pair7_frac2_set9",
        "header_mode": "explicit_30",
        "actions": [
            {
                "pair_index": 7,
                "stream": "frac2",
                "value": 9,
                "rationale": "fixture measured action direction",
            }
        ],
    }


def test_archive_changing_path_can_unlock_dispatch_contract(tmp_path: Path) -> None:
    pair, scorer, _archive_sha, _archive_bytes = _fixtures(tmp_path)
    action = _action_evidence(
        tmp_path / "built_actions.json",
        scorer_sha=module._sha256_file(scorer),
        archive_changing_path={
            "status": "built",
            "candidate_archive_sha256": "c" * 64,
            "candidate_archive_bytes": 1300,
            "manifest_path": "experiments/results/fixture/manifest.json",
            "non_noop_proof": {"status": "passed"},
            "lane_claim_required_before_exact_eval": True,
            "score_claim": False,
        },
    )

    summary = module.build_pair_action_candidates(
        pair_atom_planning_json=pair,
        action_evidence_json=action,
        top_n=2,
    )

    assert summary["dispatch_unlocked"] is True
    assert summary["ready_for_exact_eval_after_lane_claim_count"] == 1
    assert summary["blocker_class"] == "none"
    candidate = summary["candidates"][0]
    assert candidate["lowering_status"] == "archive_path_unlocked"
    assert candidate["ready_for_exact_eval_after_lane_claim"] is True


def test_duplicate_pair_actions_fail_closed(tmp_path: Path) -> None:
    pair, scorer, _archive_sha, _archive_bytes = _fixtures(tmp_path)
    action = _action_evidence(
        tmp_path / "actions.json",
        scorer_sha=module._sha256_file(scorer),
        duplicate_pair=True,
    )

    summary = module.build_pair_action_candidates(
        pair_atom_planning_json=pair,
        action_evidence_json=action,
        top_n=2,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["blocker_class"] == "duplicate_pair_action"
    candidate = summary["candidates"][0]
    assert candidate["lowering_status"] == "blocked"
    assert candidate["ready_for_pair_atom_archive_build"] is False
    assert candidate["pair_atom_action_spec"] is None


def test_noop_action_fails_closed(tmp_path: Path) -> None:
    pair, scorer, _archive_sha, _archive_bytes = _fixtures(tmp_path)
    action = _action_evidence(
        tmp_path / "noop_actions.json",
        scorer_sha=module._sha256_file(scorer),
        source_value=4,
        candidate_value=4,
    )

    summary = module.build_pair_action_candidates(
        pair_atom_planning_json=pair,
        action_evidence_json=action,
        top_n=2,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["blocker_class"] == "no_op_action"
    candidate = summary["candidates"][0]
    assert candidate["no_op_status"] == {"status": "contains_noop_action", "is_noop": True}
    assert candidate["pair_atom_action_spec"] is None


def test_ungrounded_threshold_fails_closed(tmp_path: Path) -> None:
    pair, scorer, _archive_sha, _archive_bytes = _fixtures(tmp_path)
    action = _action_evidence(
        tmp_path / "threshold_actions.json",
        scorer_sha=module._sha256_file(scorer),
        thresholds=[{"name": "min_break_even_bytes", "value": 100.0}],
    )

    summary = module.build_pair_action_candidates(
        pair_atom_planning_json=pair,
        action_evidence_json=action,
        top_n=2,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["blocker_class"] == "ungrounded_threshold"
    candidate = summary["candidates"][0]
    assert candidate["lowering_status"] == "blocked"
    assert candidate["ready_for_pair_atom_archive_build"] is False
    assert candidate["pair_atom_action_spec"] is None


def test_missing_pair_atom_source_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing_pair_planning.json"

    summary = module.build_pair_action_candidates(
        pair_atom_planning_json=missing,
        top_n=2,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["candidate_count"] == 0
    assert summary["blocker_class"] == "missing_pair_atom_planning_source"
