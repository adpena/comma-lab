# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_c067_bigmove_nontrain_candidates.py"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("c067_bigmove_nontrain_planner_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(root: Path, rel_path: str, rows: list[dict[str, Any]]) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _exact(score: float, *, bytes_: int = 247_414, pose: float = 0.1) -> dict[str, Any]:
    return {
        "archive_size_bytes": bytes_,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": 0.002,
        "final_score": score,
        "n_samples": 600,
        "score_recomputed_from_components": score,
    }


def _archive_manifest(bytes_: int, *, sha: str = "a" * 64) -> dict[str, Any]:
    return {
        "archive": {
            "path": "/tmp/synthetic/archive.zip",
            "sha256": sha,
            "size_bytes": bytes_,
        },
        "promotion_eligible": False,
        "score_claim": False,
        "schema": "synthetic_archive_manifest_v1",
    }


def _postdecode_plan(*policy_ids: str) -> dict[str, Any]:
    policies = []
    for policy_id in policy_ids:
        policies.append(
            {
                "builder_contract": {
                    "cli_args_fragment": [
                        "--policy",
                        "pair_indices",
                        "--pair-indices",
                        "1,2",
                    ]
                },
                "expected_marginal_score_terms": {
                    "component_score_improvement_first_order": 0.02,
                    "rate_score_cost": 0.001,
                },
                "estimated_payload_bytes": 1500,
                "policy_id": policy_id,
                "promotion_eligible": False,
                "score_claim": False,
                "selected_atom_count": 2,
            }
        )
    return {
        "budget_policies": policies,
        "promotion_eligible": False,
        "schema": "c067_postdecode_mask_repair_waterfill_plan_v1",
        "score_claim": False,
    }


def _assert_no_score_claim_true(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("score_claim") is True:
            raise AssertionError(f"score_claim=true found in {value}")
        for child in value.values():
            _assert_no_score_claim_true(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_score_claim_true(child)


def test_synthetic_plan_covers_required_lanes_and_keeps_sjkl_diagnostic_out_of_candidates(
    tmp_path: Path,
) -> None:
    planner = _load_planner()
    _write_json(tmp_path, planner.MULTIRESOLUTION_PLAN, {"score_claim": False})
    _write_json(
        tmp_path,
        planner.MULTIMASK_THRESHOLD_SUMMARY,
        {
            "candidate_records": [
                {
                    "archive": {"bytes": 251_782, "path": "mm/archive.zip", "sha256": "b" * 64},
                    "evidence_grade": "empirical",
                    "score_claim": False,
                    "target_extra_runs": 68_000,
                }
            ],
            "score_claim": False,
        },
    )
    _write_json(
        tmp_path,
        planner.HOTSPOT_POSESAFE_PLAN,
        {
            "candidate_policies": [
                {
                    "builder": "experiments/build_cmg3_adaptive_runs_candidate.py --field-policy-id top8",
                    "estimated_proxy": {"first_order_score_saved_proxy": 0.01},
                    "evidence_grade": "planning_only",
                    "policy_id": "top8",
                    "score_claim": False,
                    "selected_atom_count": 8,
                }
            ],
            "score_claim": False,
        },
    )
    _write_json(
        tmp_path,
        planner.POSTDECODE_PAIR_PLAN,
        _postdecode_plan("save12k_exact_trace_pair_waterfill_budget4000"),
    )
    _write_json(tmp_path, planner.POSTDECODE_BUDGET4000_MANIFEST, _archive_manifest(243_422))
    _write_json(tmp_path, planner.SJ_KL_DIAGNOSTICS[0][1], _exact(0.5, bytes_=315_515))
    _write_json(
        tmp_path,
        planner.REVERSED_BASE_CDO1_ECONOMICS,
        {
            "best_candidates": [
                {
                    "base": {"label": "tiny_base"},
                    "candidate_id": "tiny_base__partial",
                    "cdo1_payload": {
                        "compressed_payloads": {"lzma_xz": {"bytes": 4096}}
                    },
                    "estimated_archive": {
                        "estimated_archive_bytes": 132_000,
                        "estimated_delta_vs_c067": -144_214,
                        "estimated_rate_delta_vs_c067": -0.096,
                    },
                    "gates": {
                        "byte_gate_sub0300_if_distortion_unchanged": True,
                        "joint_sub0300_geometry_gate": False,
                        "residual_geometry_gate": False,
                    },
                    "mask_disagreement": {
                        "residual_vs_target_fraction_after_overlay": 0.035
                    },
                    "policy": {"policy_id": "partial"},
                    "score_claim": False,
                }
            ],
            "score_claim": False,
        },
    )
    _write_json(
        tmp_path,
        planner.TRAINED_RENDERER_EXPORT_UNLOCK_PLAN,
        {
            "best_candidates": [
                {
                    "blockers": ["source-renderer surrogate only"],
                    "bytes": 280_000,
                    "kind": "archive_without_trained_preflight_gate",
                    "path": "renderer/archive.zip",
                    "sha256": "e" * 64,
                }
            ],
            "h100_ready_preflight_count": 0,
            "non_surrogate_candidate_count": 0,
            "score_claim": False,
        },
    )

    plan = planner.build_plan(repo_root=tmp_path, output_json=tmp_path / "out.json")

    assert plan["schema"] == "c067_bigmove_nontrain_candidate_triage_v1"
    assert plan["score_claim"] is False
    _assert_no_score_claim_true(plan)
    coverage = plan["grand_council_structural_lane_coverage"]
    assert coverage["multiresolution_multimask_reconciliation"] is True
    assert coverage["cdo1_reversed_base_mask_topology"] is True
    assert coverage["renderer_self_compression"] is True
    assert coverage["scorer_weighted_mask_topology_repair_atoms"] is True
    assert coverage["sjkl_existing_active_diagnostic_not_duplicated"] is True
    assert plan["active_diagnostics"][0]["diagnostic_id"].startswith("sjkl_")
    assert not any(
        candidate["candidate_id"].startswith("sjkl_") for candidate in plan["ranked_candidates"]
    )

    postdecode = next(
        candidate
        for candidate in plan["ranked_candidates"]
        if candidate["candidate_id"]
        == "c067_postdecode_repair_save12k_exact_trace_pair_waterfill_budget4000"
    )
    assert postdecode["dispatch"]["byte_gate_passed"] is True
    assert postdecode["dispatch"]["component_benefit_gate_passed"] is True
    assert postdecode["dispatch"]["dispatchable"] is True
    assert postdecode["score_claim"] is False
    cdo1 = next(
        candidate
        for candidate in plan["ranked_candidates"]
        if candidate["candidate_id"] == "reversed_base_cdo1_tiny_base__partial"
    )
    assert cdo1["dispatch"]["dispatchable"] is False
    assert any("residual decoded-mask geometry gate failed" in blocker for blocker in cdo1["fail_closed_blockers"])


def test_exact_negative_and_above_gate_byte_screen_fail_closed(tmp_path: Path) -> None:
    planner = _load_planner()
    _write_json(
        tmp_path,
        planner.MULTIMASK_THRESHOLD_SUMMARY,
        {
            "candidate_records": [
                {
                    "archive": {"bytes": 253_691, "path": "mm/archive.zip", "sha256": "c" * 64},
                    "evidence_grade": "empirical",
                    "score_claim": False,
                    "target_extra_runs": 69_000,
                }
            ],
            "score_claim": False,
        },
    )
    _write_json(
        tmp_path,
        planner.POSTDECODE_PAIR_PLAN,
        _postdecode_plan("save12k_exact_trace_pair_waterfill_budget8000"),
    )
    _write_json(tmp_path, planner.POSTDECODE_BUDGET8000_MANIFEST, _archive_manifest(247_414))
    _write_json(
        tmp_path,
        planner.EXACT_EVALS["c067_postdecode_repair_save12k_budget8000"],
        _exact(1.55, bytes_=247_414, pose=0.125),
    )

    plan = planner.build_plan(repo_root=tmp_path)
    candidates = {item["candidate_id"]: item for item in plan["ranked_candidates"]}

    exact_negative = candidates["c067_postdecode_repair_save12k_exact_trace_pair_waterfill_budget8000"]
    assert exact_negative["exact_eval"]["status"] == "exact_negative"
    assert exact_negative["dispatch"]["dispatchable"] is False
    assert any("exact CUDA eval" in blocker for blocker in exact_negative["fail_closed_blockers"])

    above_gate = candidates["c067_multimask_threshold_fix1_extra69000"]
    assert above_gate["dispatch"]["byte_gate_passed"] is False
    assert above_gate["dispatch"]["dispatchable"] is False
    assert any("archive bytes exceed 252760" in blocker for blocker in above_gate["fail_closed_blockers"])


def test_pairwaterfill4k_exact_negative_closes_stale_dispatch_recommendation(
    tmp_path: Path,
) -> None:
    planner = _load_planner()
    _write_json(
        tmp_path,
        planner.POSTDECODE_PAIR_PLAN,
        _postdecode_plan("save12k_exact_trace_pair_waterfill_budget4000"),
    )
    _write_json(tmp_path, planner.POSTDECODE_BUDGET4000_MANIFEST, _archive_manifest(243_422))
    _write_json(
        tmp_path,
        planner.EXACT_EVALS["c067_postdecode_repair_save12k_pairwaterfill4k"],
        _exact(1.5721465219411697, bytes_=243_422, pose=0.1304232),
    )

    plan = planner.build_plan(repo_root=tmp_path)
    candidates = {item["candidate_id"]: item for item in plan["ranked_candidates"]}
    pairwaterfill = candidates[
        "c067_postdecode_repair_save12k_exact_trace_pair_waterfill_budget4000"
    ]

    assert pairwaterfill["exact_eval"]["status"] == "exact_negative"
    assert pairwaterfill["dispatch"]["dispatchable"] is False
    assert any(
        "exact CUDA eval already measured a regression" in blocker
        for blocker in pairwaterfill["fail_closed_blockers"]
    )


def test_output_json_is_stable_and_sorted(tmp_path: Path) -> None:
    planner = _load_planner()
    _write_json(
        tmp_path,
        planner.POSTDECODE_PAIR_PLAN,
        _postdecode_plan("save12k_exact_trace_pair_waterfill_budget4000"),
    )
    _write_json(tmp_path, planner.POSTDECODE_BUDGET4000_MANIFEST, _archive_manifest(243_422))
    _write_jsonl(
        tmp_path,
        planner.PMG_BYTE_SCREEN_JSONL,
        [
            {
                "bytes": 185_006,
                "delta": -91_208,
                "n": 64,
                "rate_delta": -0.0607,
                "score_claim": False,
                "sha": "d" * 64,
                "unchanged_distortion_score": 0.255,
            }
        ],
    )

    output_json = tmp_path / "triage.json"
    plan = planner.build_plan(repo_root=tmp_path, output_json=output_json)

    assert output_json.read_text(encoding="utf-8") == (
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    ranks = [candidate["rank"] for candidate in plan["ranked_candidates"]]
    assert ranks == sorted(ranks)
    candidate_ids = [candidate["candidate_id"] for candidate in plan["ranked_candidates"]]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert plan["ranked_candidates"][0]["dispatch"]["dispatchable"] is True
