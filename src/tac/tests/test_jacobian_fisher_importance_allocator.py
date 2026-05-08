"""Tests for the Jacobian/Fisher importance-weighted allocator primitive."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tac.optimization.jacobian_fisher_importance_allocator import (
    DEFAULT_DISPATCH_BLOCKERS,
    ImportanceAllocationError,
    allocate_importance_weighted_candidates,
    build_importance_allocation_manifest,
    build_importance_weights,
)
from tac.repo_io import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
FALSE_CUSTODY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "score_affecting_payload_changed",
    "charged_bits_changed",
    "family_falsified",
)
REQUIRED_CUSTODY_BLOCKERS = (
    "cpu_mps_proxy_importance_inputs_not_score_authority",
    "requires_exact_archive_cuda_score_custody_before_rank_promotion_or_kill",
)


def _curves() -> dict[str, list[dict]]:
    return {
        "high.weight": [
            {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
            {"K": 4, "byte_proxy": 70, "rel_err": 0.2},
            {"K": 8, "byte_proxy": 30, "rel_err": 0.6},
        ],
        "low.weight": [
            {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
            {"K": 4, "byte_proxy": 70, "rel_err": 0.2},
            {"K": 8, "byte_proxy": 30, "rel_err": 0.6},
        ],
    }


def _load_tool():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "jacobian_fisher_importance_allocator_tool",
        REPO_ROOT / "tools" / "jacobian_fisher_importance_allocator.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["jacobian_fisher_importance_allocator_tool"] = module
    spec.loader.exec_module(module)
    return module


def _assert_planning_only_custody(section: dict) -> None:
    for field in FALSE_CUSTODY_FIELDS:
        assert section[field] is False
    assert section["planning_only"] is True
    assert section["proxy_or_diagnostic_evidence"] is True
    assert section["evidence_device_class"] == "cpu_mps_proxy_planning_only"
    assert section["score_authority"] == "none"
    assert section["score_authority_required"] == "exact_cuda_auth_eval_archive_path"
    assert "A/A++ exact CUDA archive evidence" in section["promotion_requires_evidence_grade"]
    assert "leaderboard_or_frontier_ranking" in section["disallowed_decision_uses"]
    assert "candidate_promotion" in section["disallowed_decision_uses"]
    assert "lane_or_family_kill" in section["disallowed_decision_uses"]
    for blocker in REQUIRED_CUSTODY_BLOCKERS:
        assert blocker in section["dispatch_blockers"]
        assert blocker in section["score_claim_blockers"]


def test_target_distortion_allocation_is_monotonic() -> None:
    loose, _, _ = allocate_importance_weighted_candidates(
        _curves(),
        per_tensor_importance={"high.weight": 1.0, "low.weight": 1.0},
        target_distortion=0.25,
    )
    tight, _, _ = allocate_importance_weighted_candidates(
        _curves(),
        per_tensor_importance={"high.weight": 1.0, "low.weight": 1.0},
        target_distortion=0.05,
    )

    assert tight.total_bytes >= loose.total_bytes
    assert tight.weighted_rms_error <= loose.weighted_rms_error
    assert max(row["K"] for row in tight.to_dict()["selected_by_tensor"]) <= max(
        row["K"] for row in loose.to_dict()["selected_by_tensor"]
    )


def test_high_importance_tensor_gets_tighter_k_under_distortion_target() -> None:
    plan, weights, _ = allocate_importance_weighted_candidates(
        _curves(),
        per_tensor_importance={"high.weight": 10.0, "low.weight": 1.0},
        target_distortion=0.3,
    )
    rows = {row["tensor_name"]: row for row in plan.to_dict()["selected_by_tensor"]}

    assert weights.weights[0] > weights.weights[1]
    assert rows["high.weight"]["K"] <= rows["low.weight"]["K"]
    assert rows["high.weight"]["error"] <= rows["low.weight"]["error"]
    assert plan.weighted_rms_error <= 0.3 + 1e-12


def test_high_importance_tensor_gets_tighter_k_under_byte_budget() -> None:
    plan, _, _ = allocate_importance_weighted_candidates(
        _curves(),
        per_tensor_importance={"high.weight": 10.0, "low.weight": 1.0},
        byte_budget=100,
    )
    rows = {row["tensor_name"]: row for row in plan.to_dict()["selected_by_tensor"]}

    assert plan.total_bytes <= 100
    assert rows["high.weight"]["K"] <= rows["low.weight"]["K"]
    assert rows["high.weight"]["error"] <= rows["low.weight"]["error"]


def test_per_weight_importance_reduces_to_tensor_weights() -> None:
    weights = build_importance_weights(
        ["high.weight", "low.weight"],
        per_weight_importance={
            "high.weight": [100.0, 100.0, 25.0],
            "low.weight": [1.0, 1.0, 1.0],
        },
        boundary_mass={"low.weight": 10.0},
        texture_capacity={"low.weight": 10.0},
    )

    rows = {row.tensor_name: row for row in weights.rows}
    assert rows["high.weight"].importance_raw > rows["low.weight"].importance_raw
    assert weights.weights[0] > weights.weights[1]
    assert weights.to_dict()["allocator_input"]["weight_semantics"].startswith(
        "higher weight protects"
    )


def test_invalid_inputs_fail_closed() -> None:
    try:
        allocate_importance_weighted_candidates(
            {},
            per_tensor_importance={"x": 1.0},
            target_distortion=0.1,
        )
    except ImportanceAllocationError as exc:
        assert "must not be empty" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("empty curves should fail closed")

    bad_curves = {"x": [{"K": 1, "byte_proxy": 10, "rel_err": -0.1}]}
    try:
        allocate_importance_weighted_candidates(
            bad_curves,
            per_tensor_importance={"x": 1.0},
            target_distortion=0.1,
        )
    except ImportanceAllocationError as exc:
        assert "non-negative" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("negative error should fail closed")

    try:
        allocate_importance_weighted_candidates(
            {"x": [{"K": 1, "byte_proxy": 10, "rel_err": 0.0}]},
            per_tensor_importance={"x": float("nan")},
            target_distortion=0.1,
        )
    except ImportanceAllocationError as exc:
        assert "finite" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("non-finite importance should fail closed")

    try:
        allocate_importance_weighted_candidates(
            _curves(),
            per_tensor_importance={"high.weight": 1.0, "low.weight": 1.0},
            target_distortion=0.1,
            byte_budget=100,
        )
    except ImportanceAllocationError as exc:
        assert "exactly one" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("dual budget objective should fail closed")

    try:
        allocate_importance_weighted_candidates(
            _curves(),
            per_tensor_importance={"high.weight": 1.0, "low.weight": 1.0},
            byte_budget=1,
        )
    except ImportanceAllocationError as exc:
        assert "infeasible" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("infeasible byte budget should fail closed")


def test_manifest_metadata_blocks_promotion_and_ranking() -> None:
    manifest = build_importance_allocation_manifest(
        _curves(),
        per_tensor_importance={"high.weight": 10.0, "low.weight": 1.0},
        target_distortion=0.3,
        producer_tool="unit-test",
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["allocation"]["metadata"]["score_claim"] is False
    _assert_planning_only_custody(manifest)
    _assert_planning_only_custody(manifest["allocation"]["metadata"])
    assert "requires_cuda_pixel_jacobian_or_fisher_pullback" in DEFAULT_DISPATCH_BLOCKERS
    assert "requires_cuda_pixel_jacobian_or_fisher_pullback" in manifest["dispatch_blockers"]
    assert "exact CUDA auth eval" in manifest["integration_point"]["next_archive_builder_requirement"]


def test_direct_allocator_plan_metadata_is_not_rank_or_kill_evidence() -> None:
    plan, _, _ = allocate_importance_weighted_candidates(
        _curves(),
        per_tensor_importance={"high.weight": 10.0, "low.weight": 1.0},
        byte_budget=100,
    )

    payload = plan.to_dict()
    _assert_planning_only_custody(payload["metadata"])
    assert payload["metadata"]["score_affecting_payload_changed"] is False
    assert payload["metadata"]["charged_bits_changed"] is False
    assert payload["metadata"]["downstream_selection_can_change_charged_bits"] is True
    assert "exact_eval_dispatch_readiness" in payload["metadata"]["disallowed_decision_uses"]


def test_cli_writes_planning_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    curves_json = tmp_path / "curves.json"
    importance_json = tmp_path / "importance.json"
    output_json = tmp_path / "manifest.json"
    write_json(curves_json, {"curves": _curves()})
    write_json(
        importance_json,
        {
            "per_tensor": {
                "high.weight": 10.0,
                "low.weight": 1.0,
            }
        },
    )

    rc = tool.main(
        [
            "--curves-json",
            str(curves_json),
            "--importance-json",
            str(importance_json),
            "--target-distortion",
            "0.3",
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    text = output_json.read_text(encoding="utf-8")
    assert '"score_claim": false' in text
    assert '"rank_or_kill_eligible": false' in text
    assert '"selected_by_tensor"' in text
    payload = json.loads(text)
    _assert_planning_only_custody(payload)
    _assert_planning_only_custody(payload["allocation"]["metadata"])
