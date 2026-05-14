# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_pr85_scorer_gradient_atoms.py"
SPEC = importlib.util.spec_from_file_location("plan_pr85_scorer_gradient_atoms", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _exact_eval_payload(*, pose: float = 0.0004, seg: float = 0.0005, bytes_: int = 240_000) -> dict:
    score = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * bytes_ / 37_545_489
    return {
        "schema_version": 1,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "archive_size_bytes": bytes_,
        "n_samples": 4,
        "score_recomputed_from_components": score,
        "provenance": {
            "tool": "experiments/contest_auth_eval.py",
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "gpu_t4_match": True,
            "archive_sha256": "a" * 64,
            "archive_size_bytes": bytes_,
            "inflate_runtime_manifest": {"runtime_tree_sha256": "b" * 64},
        },
    }


def _trace_payload(*, bytes_: int = 240_000) -> dict:
    samples = [
        {
            "pair_index": 0,
            "video_name": "0.mkv",
            "video_pair_index": 0,
            "frame_indices": [0, 1],
            "posenet_dist": 0.0001,
            "segnet_dist": 0.0001,
        },
        {
            "pair_index": 1,
            "video_name": "0.mkv",
            "video_pair_index": 1,
            "frame_indices": [2, 3],
            "posenet_dist": 0.0020,
            "segnet_dist": 0.0002,
        },
        {
            "pair_index": 2,
            "video_name": "0.mkv",
            "video_pair_index": 2,
            "frame_indices": [4, 5],
            "posenet_dist": 0.0002,
            "segnet_dist": 0.0015,
        },
        {
            "pair_index": 3,
            "video_name": "0.mkv",
            "video_pair_index": 3,
            "frame_indices": [6, 7],
            "posenet_dist": 0.0003,
            "segnet_dist": 0.0001,
        },
    ]
    return {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "diagnostic_component_trace",
        "n_samples": 4,
        "avg_posenet_dist": sum(row["posenet_dist"] for row in samples) / 4,
        "avg_segnet_dist": sum(row["segnet_dist"] for row in samples) / 4,
        "archive_size_bytes": bytes_,
        "contest_auth_eval_cross_check": {
            "all_match": True,
            "contest_auth_eval_json_sha256": "c" * 64,
        },
        "samples": samples,
    }


def test_exact_formula_derivatives_and_byte_break_even(tmp_path: Path) -> None:
    exact = _write_json(tmp_path / "contest_auth_eval.json", _exact_eval_payload())
    trace = _write_json(tmp_path / "component_trace.json", _trace_payload())

    plan = planner.build_plan(
        exact_eval_json=exact,
        component_trace_jsons=[trace],
        root=tmp_path,
        auto_discover_sibling_trace=False,
        max_atoms=4,
    )

    assert plan["planning_only"] is True
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["compression_time_only"] is True
    assert plan["inflate_time_scorer_load_allowed"] is False
    assert plan["dispatch_performed"] is False
    assert plan["remote_jobs_dispatched"] is False
    assert plan["score_derivatives"]["dscore_dseg_dist"] == 100.0
    assert plan["score_derivatives"]["dscore_dpose_dist"] == 5.0 / math.sqrt(10.0 * 0.0004)
    assert plan["score_derivatives"]["dscore_darchive_byte"] == 25.0 / 37_545_489
    assert plan["formula_checks"]["abs_error_vs_reported"] < 1e-12

    top = plan["atom_ranking"][0]
    expected_combined = top["ranking_score"]
    break_even = top["byte_break_even"]["combined"]["max_charged_bytes_for_zero_net_change"]
    assert break_even == expected_combined / (25.0 / 37_545_489)
    assert top["sensitivity"]["posenet"]["status"] == "placeholder_unresolved"
    assert top["sensitivity"]["segnet"]["status"] == "placeholder_unresolved"
    assert top["dispatch_gate"]["dispatchable"] is False


def test_component_trace_atoms_rank_by_first_order_score_opportunity(tmp_path: Path) -> None:
    exact = _write_json(tmp_path / "contest_auth_eval.json", _exact_eval_payload())
    trace = _write_json(tmp_path / "component_trace.json", _trace_payload())

    plan = planner.build_plan(
        exact_eval_json=exact,
        component_trace_jsons=[trace],
        root=tmp_path,
        auto_discover_sibling_trace=False,
        max_atoms=4,
    )

    pair_indices = [atom["pair_index"] for atom in plan["atom_ranking"]]
    assert pair_indices[0] == 1
    assert [atom["ranking_score"] for atom in plan["atom_ranking"]] == sorted(
        [atom["ranking_score"] for atom in plan["atom_ranking"]],
        reverse=True,
    )
    assert plan["planning_state"] == "component_trace_ranked"
    assert plan["input_artifacts"]["component_traces"][0]["trace_cross_checked_to_exact_eval"] is True


def test_planning_only_without_component_trace_can_ingest_profile_atoms(tmp_path: Path) -> None:
    exact = _write_json(tmp_path / "contest_auth_eval.json", _exact_eval_payload())
    profile = _write_json(
        tmp_path / "profile.json",
        {
            "schema_version": 1,
            "planning_only": True,
            "score_claim": False,
            "ranked_atoms": [
                {
                    "atom_id": "profile:pose_atom",
                    "family": "pose",
                    "pair_index": 3,
                    "expected_pose_dist_saved": 0.00001,
                    "estimated_charged_bytes": 8,
                }
            ],
        },
    )

    plan = planner.build_plan(
        exact_eval_json=exact,
        profile_jsons=[profile],
        root=tmp_path,
        auto_discover_sibling_trace=False,
        max_atoms=4,
    )

    assert plan["planning_state"] == "planning_only_no_component_trace"
    assert plan["dispatch_gates"]["dispatchable"] is False
    assert "missing_component_trace_planner_uses_formula_and_profiles_only" in plan["dispatch_gates"][
        "blockers"
    ]
    atom = plan["atom_ranking"][0]
    assert atom["atom_id"] == "profile:pose_atom"
    assert atom["profile_score_detail"]["pose_dist_saved"] == 0.00001
    assert atom["estimated_score_saved"] == 0.00001 * plan["score_derivatives"]["dscore_dpose_dist"]
    assert atom["score_claim"] is False


def test_malformed_claiming_trace_fails_closed(tmp_path: Path) -> None:
    exact = _write_json(tmp_path / "contest_auth_eval.json", _exact_eval_payload())
    bad = _trace_payload()
    bad["score_claim"] = True
    trace = _write_json(tmp_path / "component_trace.json", bad)

    try:
        planner.build_plan(
            exact_eval_json=exact,
            component_trace_jsons=[trace],
            root=tmp_path,
            auto_discover_sibling_trace=False,
        )
    except planner.PR85GradientPlannerError as exc:
        assert "score_claim=false" in str(exc)
    else:
        raise AssertionError("expected score-claiming trace to fail closed")
