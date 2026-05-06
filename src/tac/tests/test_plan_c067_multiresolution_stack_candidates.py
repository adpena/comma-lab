from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_c067_multiresolution_stack_candidates.py"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("c067_multires_stack_planner_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _assert_no_score_claim_true(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("score_claim") is True:
            raise AssertionError(f"score_claim=true found in {value}")
        for child in value.values():
            _assert_no_score_claim_true(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_score_claim_true(child)


def _anchor(path: Path) -> Path:
    return _write_json(
        path,
        {
            "score_recomputed_from_components": 0.31561703078448233,
            "archive_size_bytes": 276214,
            "avg_posenet_dist": 0.00049705,
            "avg_segnet_dist": 0.00061248,
        },
    )


def _cmg2(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "cmg2_downsample_candidate_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
            "canonical_score_source_required": "archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda",
            "output_archive": {
                "bytes": 194020,
                "delta_bytes_vs_frontier": -82194,
                "sha256": "a" * 64,
            },
            "cmg2": {
                "payload_bytes": 132610,
                "pixel_disagreement_vs_full": 0.02,
                "scale": [2, 2],
            },
        },
    )


def _pmg_negative_exact(path: Path) -> Path:
    return _write_json(
        path,
        {
            "score_recomputed_from_components": 30.93,
            "archive_size_bytes": 187144,
            "avg_posenet_dist": 69.2,
            "avg_segnet_dist": 0.044,
        },
    )


def _repair(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "multimask_reconciliation_atom_plan_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical",
            "candidate_policies": [
                {
                    "policy_family": "cheap_residual_over_base",
                    "score_claim": False,
                    "estimated_charged_bytes": 4096,
                }
            ],
        },
    )


def _packer(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "archive_bit_budget_profile_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical",
            "ranked_self_compression_opportunities": [],
        },
    )


def _pose(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "ego_motion_field_plan_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "planning_only",
        },
    )


def test_planner_emits_no_score_claim_and_exact_cuda_branch_rule(tmp_path: Path) -> None:
    planner = _load_planner()
    plan = planner.build_plan(
        output_json=tmp_path / "plan.json",
        include_default_artifacts=False,
        artifact_specs=[
            ("pass0_anchor_exact", _anchor(tmp_path / "anchor.json")),
            ("pass1_cmg2_downsample", _cmg2(tmp_path / "cmg2.json")),
            ("pass2_multimask_repair", _repair(tmp_path / "repair.json")),
            ("pass3_archive_bit_budget", _packer(tmp_path / "packer.json")),
        ],
    )

    assert plan["schema"] == "c067_multiresolution_stack_planner_v1"
    assert plan["score_claim"] is False
    assert plan["planning_only"] is True
    _assert_no_score_claim_true(plan)
    branch = plan["exact_eval_branch_rule"]
    assert branch["required_for_any_candidate_policy"] is True
    assert "contest_auth_eval.py --device cuda" in branch["canonical_score_source_required"]
    assert "predictions only" in plan["additive_delta_contract"]


def test_policy_components_are_typed_by_resolution_layer(tmp_path: Path) -> None:
    planner = _load_planner()
    plan = planner.build_plan(
        output_json=tmp_path / "plan.json",
        include_default_artifacts=False,
        artifact_specs=[
            ("pass0_anchor_exact", _anchor(tmp_path / "anchor.json")),
            ("pass1_cmg2_downsample", _cmg2(tmp_path / "cmg2.json")),
            ("pass2_multimask_repair", _repair(tmp_path / "repair.json")),
            ("pass3_archive_bit_budget", _packer(tmp_path / "packer.json")),
            ("pass4_ego_motion_pose_runtime", _pose(tmp_path / "pose.json")),
        ],
    )

    policy = next(
        item
        for item in plan["candidate_policies"]
        if item["policy_id"] == "c067_multires_p03_optional_pose_runtime_coadaptation"
    )
    pass_components = [
        component
        for pass_record in policy["passes"]
        for component in pass_record["components"]
    ]
    assert pass_components
    assert all(component["resolution_layer"] for component in pass_components)
    assert {
        "fixedslice_full_resolution_anchor",
        "coarse_downsample_mask_grid",
        "high_res_hard_pair_foveal_boundary_repair",
        "pose_runtime_field",
    }.issubset({component["resolution_layer"] for component in pass_components})


def test_overlapping_mask_atoms_are_antagonistic_not_additive(tmp_path: Path) -> None:
    planner = _load_planner()
    plan = planner.build_plan(
        output_json=tmp_path / "plan.json",
        include_default_artifacts=False,
        artifact_specs=[
            ("pass0_anchor_exact", _anchor(tmp_path / "anchor.json")),
            ("pass1_cmg2_downsample", _cmg2(tmp_path / "cmg2.json")),
            ("pass1_pmg_negative_exact", _pmg_negative_exact(tmp_path / "pmg_exact.json")),
            ("pass2_multimask_repair", _repair(tmp_path / "repair.json")),
        ],
    )

    policy = next(
        item
        for item in plan["candidate_policies"]
        if item["policy_id"] == "c067_multires_p01_coarse_global_with_highres_repair"
    )
    antagonisms = policy["pass_antagonisms"]
    assert antagonisms
    assert any("masks.mkv" in edge["shared_logical_members"] for edge in antagonisms)
    assert all(edge["relation"] == "antagonism" for edge in antagonisms)
    assert policy["predicted_delta"]["kind"] == "prediction"
    assert policy["predicted_delta"]["standalone_scores_are_not_stack_scores"] is True
    assert "exact CUDA auth eval" in policy["exact_eval_branch_rule"]["additive_delta_warning"]


def test_rejects_input_artifact_with_score_claim_true(tmp_path: Path) -> None:
    planner = _load_planner()
    bad = _write_json(tmp_path / "bad.json", {"schema": "bad", "score_claim": True})

    try:
        planner.build_plan(
            output_json=tmp_path / "plan.json",
            include_default_artifacts=False,
            artifact_specs=[("bad", bad)],
        )
    except planner.PlannerError as exc:
        assert "score_claim=true" in str(exc)
    else:
        raise AssertionError("PlannerError was not raised")
