# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_multimask_reconciliation_atoms.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("multimask_reconciliation_planner_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path


def test_build_plan_is_deterministic_and_non_promotable(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.array(
        [
            [[0, 0, 1], [1, 1, 0]],
            [[2, 2, 2], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    candidate_a = source.copy()
    candidate_a[0, 0, 0] = 1
    candidate_b = source.copy()
    candidate_b[0, 0, 0] = 1
    candidate_b[1, 1, 2] = 3

    source_path = _save(tmp_path / "source.npy", source)
    a_path = _save(tmp_path / "model_a.npy", candidate_a)
    b_path = _save(tmp_path / "model_b.npy", candidate_b)

    kwargs = {
        "source_mask_array": source_path,
        "candidate_mask_arrays": [a_path, b_path],
        "source_family": "base_masks",
        "candidate_families": ["model_a", "model_b"],
        "veto_min_agreement": 1.0,
        "disagreement_byte_equivalent": 1.0,
    }
    first = planner.build_plan(output_json=tmp_path / "plan_a.json", **kwargs)
    second = planner.build_plan(output_json=tmp_path / "plan_b.json", **kwargs)

    assert first == second
    assert json.loads((tmp_path / "plan_a.json").read_text()) == first
    assert first["schema_version"] == planner.SCHEMA_VERSION
    assert first["score_claim"] is False
    assert first["promotion_eligible"] is False
    assert first["evidence_grade"] == "empirical"
    assert "No score claim is made" in first["non_promotable_warning"]
    assert first["candidate_family_names"] == ["model_a", "model_b"]
    assert all(policy["score_claim"] is False for policy in first["candidate_policies"])


def test_policy_ranking_and_required_policy_families(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((1, 2, 4), dtype=np.uint8)
    near = source.copy()
    near[0, 0, 0] = 1
    far = np.ones((1, 2, 4), dtype=np.uint8)

    plan = planner.build_plan(
        source_mask_array=_save(tmp_path / "source.npy", source),
        candidate_mask_arrays=[
            _save(tmp_path / "near.npy", near),
            _save(tmp_path / "far.npy", far),
        ],
        candidate_families=["near_model", "far_model"],
        output_json=tmp_path / "plan.json",
    )

    policy_families = {policy["policy_family"] for policy in plan["candidate_policies"]}
    assert {
        "majority_vote",
        "priority_order",
        "disagreement_gated_veto",
        "cheap_residual_over_base",
    }.issubset(policy_families)
    rank_costs = [policy["rank_cost_proxy"] for policy in plan["candidate_policies"]]
    assert rank_costs == sorted(rank_costs)
    residuals = [
        policy for policy in plan["candidate_policies"]
        if policy["policy_family"] == "cheap_residual_over_base"
    ]
    assert residuals[0]["fusion_reconciliation_policy"]["residual_family"] == "near_model"
    assert residuals[0]["estimated_charged_bytes"] < residuals[1]["estimated_charged_bytes"]


def test_vectorized_majority_preserves_source_ties_and_matches_consensus(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.array([[0, 9, 2, 3], [4, 4, 1, 1]], dtype=np.int16)
    candidate_a = np.array([[1, 8, 2, 3], [4, 7, 1, 6]], dtype=np.int16)
    candidate_b = np.array([[0, 8, 5, 3], [4, 7, 5, 6]], dtype=np.int16)
    candidate_c = np.array([[1, 9, 5, 3], [8, 7, 5, 6]], dtype=np.int16)

    fused = planner._majority_vote(  # noqa: SLF001 - behavior-specific regression
        [source, candidate_a, candidate_b, candidate_c],
        source=source,
    )
    assert fused.tolist() == [[0, 9, 2, 3], [4, 7, 1, 6]]

    consensus, mask = planner._candidate_consensus(  # noqa: SLF001
        [candidate_a, candidate_b, candidate_c],
        threshold=2.0 / 3.0,
    )
    assert consensus.tolist() == [[1, 8, 5, 3], [4, 7, 5, 6]]
    assert mask.tolist() == [[True, True, True, True], [True, True, True, True]]


def test_compact_row_run_proxy_can_beat_flat_sparse_residual(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((1, 2, 12), dtype=np.uint8)
    candidate = source.copy()
    candidate[0, 0, 1:10] = 3
    candidate[0, 1, 2:11] = 3

    plan = planner.build_plan(
        source_mask_array=_save(tmp_path / "source.npy", source),
        candidate_mask_arrays=[_save(tmp_path / "candidate.npy", candidate)],
        candidate_families=["rowrun_candidate"],
        output_json=tmp_path / "plan.json",
        disagreement_byte_equivalent=0.0,
    )
    residual = next(
        policy
        for policy in plan["candidate_policies"]
        if policy["policy_family"] == "cheap_residual_over_base"
    )
    selected = residual["estimated_byte_cost_model"]["selected"]
    alternatives = residual["estimated_byte_cost_model"]["alternatives"]

    assert selected["kind"] == "compact_row_run_residual_over_source_proxy"
    assert selected["run_count"] == 2
    assert residual["density_metrics"]["changed_elements_per_run"] == 9.0
    assert residual["density_metrics"]["estimated_bytes_per_run"] == (
        selected["estimated_charged_bytes"] / selected["run_count"]
    )
    assert residual["density_metrics"]["rate_score_cost"] > 0.0
    assert (
        residual["density_metrics"]["arithmetic_entropy_floor_bytes_no_model_overhead"]
        <= selected["estimated_charged_bytes"]
    )
    assert residual["density_metrics"]["arithmetic_estimated_bytes_with_model_overhead"] >= (
        residual["density_metrics"]["arithmetic_entropy_floor_bytes_no_model_overhead"]
    )
    assert residual["density_metrics"]["arithmetic_estimated_rate_score_cost_with_model_overhead"] >= (
        residual["density_metrics"]["arithmetic_entropy_floor_rate_score_cost_no_model_overhead"]
    )
    assert residual["density_metrics"]["break_even_component_score_improvement_per_changed_element"] > 0.0
    assert "softmax/Gumbel" in residual["differentiable_feedback_contract"]["smooth_surrogate_targets"][
        "selection_probability"
    ]
    assert alternatives["sparse_residual_over_source_proxy"]["estimated_charged_bytes"] > selected[
        "estimated_charged_bytes"
    ]
    assert alternatives["ideal_adaptive_arithmetic_row_run_lower_bound_proxy"][
        "planning_lower_bound_only"
    ] is True
    assert residual["estimated_charged_bytes"] == selected["estimated_charged_bytes"]


def test_no_op_veto_is_marked_non_dispatchable_and_not_ranked_first(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((1, 2, 6), dtype=np.uint8)
    candidate_a = source.copy()
    candidate_b = source.copy()
    candidate_a[0, 0, 1:5] = 2
    candidate_b[0, 1, 1:5] = 3

    plan = planner.build_plan(
        source_mask_array=_save(tmp_path / "source.npy", source),
        candidate_mask_arrays=[
            _save(tmp_path / "a.npy", candidate_a),
            _save(tmp_path / "b.npy", candidate_b),
        ],
        candidate_families=["a", "b"],
        output_json=tmp_path / "plan.json",
        veto_min_agreement=1.0,
        disagreement_byte_equivalent=0.0,
    )

    veto = next(
        policy
        for policy in plan["candidate_policies"]
        if policy["policy_family"] == "disagreement_gated_veto"
    )
    assert veto["dispatch_relevance"]["no_op_vs_source"] is True
    assert veto["dispatch_relevance"]["dispatchable_byte_model"] is False
    assert veto["density_metrics"]["learnable_feedback_role"] == "negative_no_op_guard"
    assert veto["density_metrics"]["estimated_bytes_per_changed_element"] is None
    assert plan["candidate_policies"][0]["policy_family"] != "disagreement_gated_veto"


def test_manifest_inputs_resolve_decoded_mask_arrays(tmp_path: Path) -> None:
    planner = _load_planner()
    source_path = _save(tmp_path / "arrays" / "source.npy", np.zeros((2, 2), dtype=np.uint8))
    candidate_path = _save(tmp_path / "arrays" / "candidate.npy", np.ones((2, 2), dtype=np.uint8))
    source_manifest = tmp_path / "source_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    source_manifest.write_text(json.dumps({"decoded_mask_array": "arrays/source.npy"}) + "\n")
    candidate_manifest.write_text(
        json.dumps({"artifacts": [{"role": "decoded_mask_array", "path": "arrays/candidate.npy"}]}) + "\n"
    )

    plan = planner.build_plan(
        source_manifest=source_manifest,
        candidate_manifests=[candidate_manifest],
        candidate_families=["candidate_from_manifest"],
        output_json=tmp_path / "plan.json",
    )

    assert plan["inputs"]["source"]["path"] == str(source_path.resolve())
    assert plan["inputs"]["candidates"][0]["path"] == str(candidate_path.resolve())
    assert plan["inputs"]["source"]["source_manifest"] == str(source_manifest.resolve())
    assert plan["inputs"]["candidates"][0]["source_manifest"] == str(candidate_manifest.resolve())


def test_missing_and_shape_mismatch_fail_closed(tmp_path: Path) -> None:
    planner = _load_planner()
    source = _save(tmp_path / "source.npy", np.zeros((2, 2), dtype=np.uint8))
    different_shape = _save(tmp_path / "different_shape.npy", np.zeros((2, 3), dtype=np.uint8))

    with pytest.raises(FileNotFoundError):
        planner.build_plan(
            source_mask_array=source,
            candidate_mask_arrays=[tmp_path / "missing.npy"],
            output_json=tmp_path / "missing.json",
        )

    with pytest.raises(planner.PlannerError, match="shape mismatch"):
        planner.build_plan(
            source_mask_array=source,
            candidate_mask_arrays=[different_shape],
            output_json=tmp_path / "shape.json",
        )

    with pytest.raises(planner.PlannerError, match="at least one candidate"):
        planner.build_plan(source_mask_array=source, output_json=tmp_path / "none.json")
