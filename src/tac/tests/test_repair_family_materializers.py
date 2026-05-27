# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.optimization.repair_campaign_scorer import score_repair_campaign
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
    build_repair_campaign_family_materializer_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _repair_payload(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "segnet_mlx_response.json"
    ref = tmp_path / "segnet_reference_mlx_response.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "chain_id": "unit_repair_chain",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 40,
            **_false_authority(),
        },
        "typed_response_ledger": {
            "schema": "frontier_rate_attack_repair_budget_typed_response_ledger.v1",
            "available_receiver_closed_rate_credit_bytes": 40,
            "rows": [
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "segnet_region_ready",
                    "candidate_id": "segnet_class_region_waterfill",
                    "correction_family": "segnet_class_region_waterfill",
                    "targeted_dimensions": ["segnet", "region"],
                    "operation_levels": ["pixel", "boundary", "region", "frame"],
                    "entropy_position_label": (
                        "before_entropy_coder_distribution_shaping"
                    ),
                    "requested_repair_bytes": 32,
                    "objective_delta_score_units": -0.0010,
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "segnet_class_region_mask_ids": ["road_boundary"],
                    **_false_authority(),
                }
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def _plan_from_score_report(score_report: dict[str, object]) -> dict[str, object]:
    child_id = "repair_budget_spent_child_unit_segnet"
    allocation = score_report["optimizer_decision"]["selected_allocation_rows"][0]  # type: ignore[index]
    return {
        "schema": "frontier_rate_attack_repair_budget_materialization_plan.v1",
        "chain_id": "unit_repair_chain",
        "parent_candidate_chain_id": "rate_parent",
        "candidate_chain_rows": [
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_plan_row.v1",
                "candidate_kind": "rate_only_floor_parent",
                "candidate_chain_id": "rate_parent",
                "materialization_order": 1,
                "candidate_archive_materialized": False,
                "receiver_consumed": False,
                **_false_authority(),
            },
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_plan_row.v1",
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": child_id,
                "parent_candidate_chain_id": "rate_parent",
                "materialization_order": 2,
                "typed_response_id": allocation["typed_response_id"],
                "allocation_candidate_id": allocation["candidate_id"],
                "correction_family": allocation["correction_family"],
                "operation_levels": ["pixel", "boundary", "region", "frame"],
                "entropy_position_label": allocation["entropy_position_label"],
                "candidate_archive_materialized": False,
                "receiver_consumed": False,
                **_false_authority(),
            },
        ],
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **_false_authority(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def test_segnet_family_materializer_emits_ordered_fail_closed_manifest(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)

    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )

    assert manifest["schema"] == REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    assert manifest["target_kind"] == "segnet_class_region_waterfill"
    assert manifest["candidate_chain_ids"] == ["repair_budget_spent_child_unit_segnet"]
    assert manifest["active_entropy_stage"]["order"] == 10
    assert manifest["fractal_optimization_scope"]["ordered_levels"] == [
        "bit",
        "byte",
        "pixel",
        "boundary",
        "region",
        "frame",
        "pair",
        "batch",
        "full_video",
    ]
    assert manifest["component_response_replayed"] is True
    assert manifest["byte_closed_candidate_emitted"] is False
    assert (
        "segnet_class_region_waterfill_byte_closed_candidate_archive_not_materialized"
        in manifest["readiness_blockers"]
    )
    assert "segnet_class_region_mask_ids_missing" not in manifest["readiness_blockers"]
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_family_materializer_cli_writes_manifest(tmp_path: Path) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    score_report_path = _write_json(tmp_path / "score_report.json", score_report)
    plan_path = _write_json(tmp_path / "plan.json", plan)
    manifest_path = tmp_path / "family_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_repair_campaign_family_materializer_manifest.py"),
            "--materialization-plan",
            str(plan_path),
            "--score-report",
            str(score_report_path),
            "--typed-response-id",
            "segnet_region_ready",
            "--candidate-id",
            "segnet_class_region_waterfill",
            "--materializer-manifest-out",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    assert manifest["component_response_replayed"] is True
    assert manifest["score_claim"] is False
