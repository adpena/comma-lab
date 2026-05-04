from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO / "experiments" / "plan_c067_micro_mask_reencode.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_c067_micro_mask_reencode_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(pair: int, *, pose: float, seg: float, combined: float) -> dict:
    return {
        "pair_index": pair,
        "video_pair_index": pair,
        "frame_start": pair * 2,
        "frame_indices": [pair * 2, pair * 2 + 1],
        "posenet_dist": pose,
        "segnet_dist": seg,
        "score_pose_contribution_first_order": pose / 10.0,
        "score_seg_contribution_exact": seg / 10.0,
        "score_combined_contribution_first_order": combined,
    }


def _write_trace(path: Path) -> None:
    payload = {
        "schema_version": "contest_component_trace_v1",
        "evidence_grade": "A",
        "score_claim": True,
        "n_samples": 600,
        "archive_size_bytes": 276214,
        "score_recomputed_from_components": 0.31561703078448233,
        "top_pose_samples": [
            _sample(105, pose=0.0026, seg=0.0008, combined=0.00044),
            _sample(164, pose=0.0025, seg=0.0009, combined=0.00045),
        ],
        "top_seg_samples": [
            _sample(522, pose=0.0003, seg=0.0015, combined=0.00030),
            _sample(517, pose=0.0004, seg=0.0014, combined=0.00028),
        ],
        "top_combined_samples": [
            _sample(164, pose=0.0025, seg=0.0009, combined=0.00045),
            _sample(105, pose=0.0026, seg=0.0008, combined=0.00044),
            _sample(128, pose=0.0024, seg=0.0007, combined=0.00040),
        ],
        "samples": [
            _sample(0, pose=0.00009, seg=0.00044, combined=0.00008),
            _sample(69, pose=0.0024, seg=0.00053, combined=0.00038),
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_trace_plan_emits_deterministic_micro_bands_and_builder_policies(tmp_path: Path) -> None:
    planner = _load_planner()
    trace = tmp_path / "component_trace.json"
    _write_trace(trace)
    output_json = tmp_path / "plan.json"

    plan = planner.build_plan(
        output_json=output_json,
        component_traces=[trace],
        top_pose_pairs=2,
        top_seg_pairs=2,
        top_combined_pairs=3,
        target_savings_bytes=(5000, 8000, 12000),
        savings_tolerance_bytes=250,
        frame_radius=1,
    )

    assert output_json.read_text(encoding="utf-8") == json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert plan["schema"] == "c067_micro_mask_reencode_plan_v1"
    assert plan["score_claim"] is False
    assert plan["remote_jobs_dispatched"] is False
    assert plan["frontier"]["archive_sha256"] == planner.C067_FRONTIER_ARCHIVE_SHA256
    assert plan["refused_candidate_families"][0]["family"] == "broad_whole_mask_crf_replacement"
    assert [item["target_savings_bytes"] for item in plan["candidate_configs"]] == [5000, 8000, 12000]

    first = plan["candidate_configs"][0]
    assert first["byte_screen"]["target_archive_bytes"] == 271214
    assert first["byte_screen"]["acceptable_archive_byte_range"] == [270964, 271464]
    assert first["av1_probe"]["broad_crf_replacement_allowed"] is False
    assert first["builder_policy_json"]["hard_pair_indices"] == []
    assert first["builder_policy_json"]["class_ids"] == []
    assert first["protection"]["protected_class_ids"] == [1, 2, 3, 4]
    assert 103 in first["protection"]["protected_mask_frames"]
    assert 105 in first["protection"]["protected_mask_frames"]
    assert first["protection"]["protected_regions"][0]["frames"] == first["protection"]["protected_mask_frames"]
    assert plan["protected_pair_ranking"]["ranked_pairs"][0]["pair_index"] in {105, 164}


def test_explicit_pair_json_mode_without_trace(tmp_path: Path) -> None:
    planner = _load_planner()
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps({"protected_pair_indices": [4, 9, 4]}) + "\n", encoding="utf-8")

    plan = planner.build_plan(
        output_json=tmp_path / "plan.json",
        protected_pairs=(2,),
        protected_pairs_jsons=[pairs_json],
        target_savings_bytes=(5000,),
        frame_radius=1,
        mask_frame_count=20,
    )

    candidate = plan["candidate_configs"][0]
    assert plan["protected_pair_ranking"]["pair_count"] == 3
    assert candidate["protection"]["protected_pair_indices"] == [2, 4, 9]
    assert candidate["protection"]["protected_mask_frames"] == [1, 2, 3, 4, 5, 8, 9, 10]
    assert candidate["builder_policy_json"]["planner_semantics"]["class_protection_scope"].startswith(
        "classes are protected through selected hard frames"
    )


def test_refuses_broad_crf_replacement_before_planning(tmp_path: Path) -> None:
    planner = _load_planner()

    with pytest.raises(planner.PlannerError, match="broad CRF replacement is refused"):
        planner.build_plan(
            output_json=tmp_path / "plan.json",
            protected_pairs=(1,),
            candidate_family="broad-crf-replacement",
        )


def test_screens_measured_candidates_for_micro_band_and_trust_region(tmp_path: Path) -> None:
    planner = _load_planner()
    accepted = tmp_path / "accepted_manifest.json"
    accepted.write_text(
        json.dumps(
            {
                "score_claim": False,
                "archive": {"size_bytes": 271300},
                "candidate_mask_stream": {"size_bytes": 214600},
                "policy": {"hard_frame_indices": [1, 2, 3]},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rejected = tmp_path / "rejected_manifest.json"
    rejected.write_text(
        json.dumps(
            {
                "score_claim": False,
                "archive": {"size_bytes": 250000},
                "candidate_mask_stream": {"size_bytes": 190000},
                "policy": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plan = planner.build_plan(
        output_json=tmp_path / "plan.json",
        protected_pairs=(1,),
        measured_candidate_jsons=[accepted, rejected],
        savings_tolerance_bytes=512,
    )

    screen = plan["measured_candidate_byte_screen"]
    assert screen[0]["accepted_for_local_byte_screen"] is True
    assert screen[0]["nearest_target_savings_bytes"] == 5000
    assert screen[1]["accepted_for_local_byte_screen"] is False
    assert "exceeds_micro_savings_cap_broad_reencode_risk" in screen[1]["reject_reasons"]
    assert "policy_has_no_protected_frames_or_regions" in screen[1]["reject_reasons"]
