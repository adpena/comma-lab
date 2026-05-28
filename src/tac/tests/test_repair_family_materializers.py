# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.optimization.repair_campaign_chain_contract import (
    RepairCampaignChainContractError,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
    build_repair_campaign_blocked_learning_signal_report,
    build_repair_campaign_materialization_learning_signal_report,
)
from tac.optimization.repair_campaign_posterior import (
    append_repair_campaign_blocked_learning_signal_report,
    load_repair_campaign_stackability_posterior_rows,
)
from tac.optimization.repair_campaign_scorer import (
    build_repair_campaign_posterior_prior_summary,
    score_repair_campaign,
)
from tac.optimization.repair_family_byte_transform_executor import (
    REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
    REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA,
    build_repair_family_byte_transform_execution_report,
)
from tac.optimization.repair_family_exact_ready_bridge import (
    REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA,
    build_repair_family_exact_ready_bridge,
)
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
    build_repair_campaign_family_materializer_manifest,
)
from tac.optimization.repair_family_stack_search import (
    REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA,
    REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
    REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA,
    build_repair_family_exact_handoff_plan,
    plan_repair_family_stack_search,
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


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
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


def test_repair_family_byte_transform_executor_emits_replayable_delta(
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
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)

    report, bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    assert report["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA
    assert report["family_id"] == "segnet_class_region_waterfill"
    assert report["byte_transform_delta_emitted"] is True
    assert report["byte_transform_delta"]["bytes"] > 0
    assert (tmp_path / report["byte_transform_delta"]["path"]).is_file()
    assert report["component_response_replayed"] is True
    assert report["exact_eval_handoff_gate"]["eligible_for_exact_eval_handoff"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert bundle["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA
    assert bundle["source_records_sha256"]


def test_byte_transform_executor_repacks_archive_native_candidate_when_custody_exists(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    archive_path = _write_zip(
        tmp_path / "source_archive.zip",
        {"0.bin": (b"segnet-region-waterfill" * 64)},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    plan["candidate_chain_rows"][1]["candidate_archive_path"] = str(archive_path)
    plan["candidate_chain_rows"][1]["candidate_archive_sha256"] = archive_sha
    plan["candidate_chain_rows"][1]["candidate_archive_bytes"] = archive_path.stat().st_size
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)

    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    candidate = report["candidate_archive"]
    assert report["byte_closed_candidate_emitted"] is True
    assert report["archive_native_transform_attempted"] is True
    assert report["archive_native_transform_kind"] == "zip_repack_payload_identity"
    assert candidate["runtime_consumption_proof_ready"] is True
    assert candidate["receiver_contract_satisfied"] is True
    assert (tmp_path / candidate["path"]).is_file()
    assert (tmp_path / candidate["runtime_consumption_proof_path"]).is_file()
    assert report["exact_eval_handoff_gate"][
        "archive_bound_runtime_consumption_proof_ready"
    ] is True
    assert report["exact_eval_handoff_gate"]["eligible_for_exact_eval_handoff"] is False
    report_path = _write_json(tmp_path / "byte_transform_report.json", report)
    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        byte_credit_budget=10_000,
    )
    assert stack_plan["exact_eval_handoff_candidate_count"] == 1
    assert stack_plan["archive_bound_exact_handoff_candidate_count"] == 1
    assert stack_plan["exact_eval_handoff_gate"][
        "archive_bound_custody_complete"
    ] is True
    assert (
        "byte_closed_archive_runtime_receiver_proof_required_per_stack"
        not in stack_plan["exact_eval_handoff_gate"]["blockers"]
    )
    handoff_row = stack_plan["exact_eval_handoff_candidates"][0]
    assert handoff_row["schema"] == REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA
    assert handoff_row["archive_bound_custody_complete"] is True
    assert handoff_row["candidate_archive"]["custody_complete"] is True
    assert handoff_row["runtime_consumption_proof"]["custody_complete"] is True
    assert handoff_row["ready_for_exact_eval_dispatch"] is False
    exact_handoff_plan = build_repair_family_exact_handoff_plan(
        stack_plan=stack_plan,
        stack_plan_path=report_path,
    )
    assert exact_handoff_plan["schema"] == REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA
    assert exact_handoff_plan["archive_bound_candidate_count"] == 1
    assert exact_handoff_plan["archive_bound_custody_complete"] is True
    assert exact_handoff_plan["ready_for_exact_eval_dispatch"] is False


def test_repair_family_byte_transform_cli_writes_report_and_bundle(
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
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)
    report_path = tmp_path / "byte_transform_report.json"
    bundle_path = tmp_path / "byte_transform_bundle.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_repair_family_byte_transform_executor.py"),
            "--family-materializer-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "byte_transform"),
            "--execution-report-out",
            str(report_path),
            "--replay-bundle-out",
            str(bundle_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert report["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA
    assert bundle["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA
    assert report["score_claim"] is False


@pytest.mark.parametrize(
    "family_id",
    [
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
        "frame0_k16_palette_asymmetry",
        "entropy_boundary_probe",
    ],
)
def test_byte_transform_executor_supports_all_queue_owned_repair_families(
    tmp_path: Path,
    family_id: str,
) -> None:
    archive_path = _write_zip(
        tmp_path / f"{family_id}_source_archive.zip",
        {"0.bin": (family_id.encode("utf-8") * 64)},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
        "materializer_id": f"repair_family_materializer:{family_id}",
        "target_kind": family_id,
        "family_id": family_id,
        "typed_response_id": f"{family_id}_typed",
        "candidate_chain_id": f"{family_id}_chain",
        "candidate_chain_ids": [f"{family_id}_chain"],
        "repair_budget_candidate_chain_id": f"{family_id}_chain",
        "repair_budget_candidate_chain_ids": [f"{family_id}_chain"],
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "active_entropy_stage": {
            "order": 10,
            "stage": "before_entropy_coder_distribution_shaping",
            "class": "pre_entropy_distribution_shaping",
        },
        "fractal_optimization_scope": {
            "active_levels": ["bit", "byte", "pixel", "region", "frame"],
            "declared_levels": ["bit", "byte", "pixel", "region", "frame"],
        },
        "component_response_replayed": True,
        "component_response_replay": {
            "replayed": True,
            "axis_tag": "[macOS-MLX research-signal]",
            "component_response_terms": {
                "segnet_delta_score_units": -0.0001,
                "posenet_delta_score_units": -0.0002,
                **_false_authority(),
            },
            **_false_authority(),
        },
        "receiver_contract_satisfied": False,
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": archive_path.stat().st_size,
        },
        "byte_closed_candidate_emitted": True,
        "readiness_blockers": [],
        **_false_authority(),
    }
    manifest_path = _write_json(tmp_path / f"{family_id}.json", manifest)

    report, bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / family_id,
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    assert report["family_id"] == family_id
    assert report["byte_transform_supported"] is True
    assert report["byte_transform_delta"]["transform_kind"]
    assert report["byte_closed_candidate_emitted"] is True
    assert report["candidate_archive"]["runtime_consumption_proof_ready"] is True
    assert report["exact_eval_handoff_gate"][
        "archive_bound_runtime_consumption_proof_ready"
    ] is True
    assert report["ready_for_exact_eval_dispatch"] is False
    assert bundle["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA


def test_repair_family_stack_search_demotes_negative_posterior(
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
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)
    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )
    report_path = _write_json(tmp_path / "byte_transform_report.json", report)
    posterior_path = tmp_path / "posterior.jsonl"
    posterior_path.write_text(
        json.dumps(
            {
                "schema": "repair_campaign_stackability_posterior_row.v1",
                "typed_response_id": "segnet_negative",
                "candidate_id": "segnet_class_region_waterfill",
                "family_id": "segnet_class_region_waterfill",
                "acquisition_policy_delta": {
                    "recommended_acquisition_policy": (
                        "decrease_family_priority_until_new_component_response_signal"
                    ),
                    "family_priority_direction": "decrease",
                    **_false_authority(),
                },
                "blockers": ["non_improving_local_objective_delta"],
                **_false_authority(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        posterior_path=posterior_path,
        byte_credit_budget=10_000,
    )

    assert stack_plan["schema"] == REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA
    row = stack_plan["stack_rows"][0]
    assert row["automatic_negative_result_demoted"] is True
    assert "automatic_negative_result_demotion_active" in row["blockers"]
    assert row["interaction_feature_vector"]["segnet_region_family"] is True
    assert stack_plan["interaction_tensor"]["cell_count"] == 1
    assert stack_plan["interaction_tensor"]["cells"][0]["negative_demoted_count"] == 1
    assert stack_plan["ready_for_exact_eval_dispatch"] is False


def test_repair_exact_ready_bridge_emits_blocked_source_queue(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    archive_path = _write_zip(
        tmp_path / "source_archive.zip",
        {"0.bin": (b"segnet-region-waterfill" * 64)},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    plan["candidate_chain_rows"][1]["candidate_archive_path"] = str(archive_path)
    plan["candidate_chain_rows"][1]["candidate_archive_sha256"] = archive_sha
    plan["candidate_chain_rows"][1]["candidate_archive_bytes"] = archive_path.stat().st_size
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=_write_json(
            tmp_path / "family_manifest.json",
            manifest,
        ),
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )
    report_path = _write_json(tmp_path / "byte_transform_report.json", report)
    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        byte_credit_budget=10_000,
    )
    exact_handoff_plan = build_repair_family_exact_handoff_plan(
        stack_plan=stack_plan,
        stack_plan_path=tmp_path / "repair_family_exact_handoff_plan.json",
    )

    bridge = build_repair_family_exact_ready_bridge(
        exact_handoff_plan=exact_handoff_plan,
        exact_handoff_plan_path=tmp_path / "repair_family_exact_handoff_plan.json",
        repo_root=tmp_path,
    )

    bridge_report = bridge["bridge_report"]
    source_queue = bridge["source_optimizer_queue"]
    blocked_queue = bridge["blocked_exact_ready_queue"]
    assert bridge_report["schema"] == REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA
    assert bridge_report["candidate_count"] == 1
    assert bridge_report["archive_custody_proven_count"] == 1
    assert bridge_report["runtime_proof_custody_proven_count"] == 1
    assert bridge_report["runtime_content_tree_custody_proven_count"] == 0
    assert source_queue["schema"] == "optimizer_candidate_queue_v1"
    assert source_queue["dispatch_ready"] == []
    assert blocked_queue["schema"] == "optimizer_candidate_exact_eval_ready_queue_v1"
    assert blocked_queue["dispatch_ready_count"] == 0
    source_row = source_queue["top_k"][0]
    assert source_row["target_modes"] == ["contest_exact_eval"]
    assert source_row["ready_for_exact_eval_dispatch"] is False
    assert "submission_dir_missing_for_runtime_content_tree_custody" in source_row[
        "dispatch_blockers"
    ]


def test_materialization_gate_learning_signal_updates_posterior(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    family_manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    family_manifest_path = _write_json(tmp_path / "family_manifest.json", family_manifest)
    execution_report = {
        "schema": "frontier_rate_attack_repair_budget_materialization_execution_report.v1",
        "chain_id": "unit_repair_chain",
        "candidate_archive_materialized": False,
        "runtime_consumption_proof_present": False,
        "receiver_consumed": False,
        "component_response_replayed": True,
        "execution_rows": [
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_execution_row.v1",
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": "repair_budget_spent_child_unit_segnet",
                "candidate_archive_materialized": False,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": True,
                "component_response_replay_axis_tag": "[macOS-MLX research-signal]",
                "blockers": ["candidate_archive_materialized_false"],
                **_false_authority(),
            }
        ],
        "blockers": ["candidate_archives_not_materialized"],
        **_false_authority(),
    }
    gate = {
        "schema": "repair_campaign_byte_closed_materialization_gate.v1",
        "typed_response_id": "segnet_region_ready",
        "candidate_id": "segnet_class_region_waterfill",
        "candidate_archive_materialized": False,
        "archive_bound_runtime_consumption_proof_ready": False,
        "component_response_replayed": True,
        "blockers": ["candidate_archive_materialized_false"],
        **_false_authority(),
    }
    execution_report_path = _write_json(tmp_path / "execution_report.json", execution_report)
    gate_path = _write_json(tmp_path / "gate.json", gate)

    signal_report = build_repair_campaign_materialization_learning_signal_report(
        materialization_execution_report_path=execution_report_path,
        materialization_execution_report=execution_report,
        materialization_gate_path=gate_path,
        materialization_gate=gate,
        family_materializer_manifest_path=family_manifest_path,
        family_materializer_manifest=family_manifest,
        repo_root=tmp_path,
    )

    assert signal_report["schema"] == REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
    signal = signal_report["learning_signal_rows"][0]
    update = signal["local_planning_update"]
    assert update["recommended_acquisition_policy"] == (
        "prioritize_byte_closed_family_materializer_implementation"
    )
    assert update["planner_feature_vector"]["entropy_stage_order"] == 10
    signal_report_path = _write_json(tmp_path / "signal_report.json", signal_report)
    posterior_path = tmp_path / "posterior.jsonl"
    append_report = append_repair_campaign_blocked_learning_signal_report(
        blocked_learning_signal_report_path=signal_report_path,
        blocked_learning_signal_report=signal_report,
        posterior_path=posterior_path,
        lock_path=tmp_path / ".posterior.lock",
        repo_root=tmp_path,
    )
    posterior_rows = load_repair_campaign_stackability_posterior_rows(posterior_path)
    summary = build_repair_campaign_posterior_prior_summary(
        posterior_path=posterior_path,
    )

    assert append_report["appended_count"] == 1
    assert posterior_rows[0]["typed_response_id"] == "segnet_region_ready"
    assert posterior_rows[0]["acquisition_policy_delta"][
        "family_priority_direction"
    ] == "increase"
    assert posterior_rows[0]["acquisition_policy_delta"][
        "posterior_budget_routing_hint"
    ] == "route_budget_to_byte_closed_materializer_after_custody"
    route = summary["acquisition_followup_routes"][0]
    assert route["activation_action"] == "implement_or_run_repair_family_byte_transform"


def test_blocked_credit_exhaustion_updates_posterior_budget_routing(
    tmp_path: Path,
) -> None:
    payload = _repair_payload(tmp_path)
    payload["receiver_closed_rate_credit"]["receiver_closed_saved_bytes_total"] = 0
    payload["typed_response_ledger"]["available_receiver_closed_rate_credit_bytes"] = 0
    score_report = score_repair_campaign(payload=payload, repo_root=tmp_path)
    score_report_path = _write_json(tmp_path / "score_report.json", score_report)

    signal_report = build_repair_campaign_blocked_learning_signal_report(
        score_report_path=score_report_path,
        score_report=score_report,
        repo_root=tmp_path,
    )
    signal = signal_report["learning_signal_rows"][0]
    update = signal["local_planning_update"]

    assert update["recommended_acquisition_policy"] == (
        "increase_receiver_closed_rate_credit_or_rebudget_earlier_entropy_stage"
    )
    assert update["planner_feature_vector"]["selection_blocker_class"] == (
        "receiver_credit_exhausted"
    )
    assert update["planner_feature_vector"]["receiver_credit_exhausted"] is True

    signal_report_path = _write_json(tmp_path / "blocked_signals.json", signal_report)
    posterior_path = tmp_path / "posterior.jsonl"
    append_repair_campaign_blocked_learning_signal_report(
        blocked_learning_signal_report_path=signal_report_path,
        blocked_learning_signal_report=signal_report,
        posterior_path=posterior_path,
        lock_path=tmp_path / ".posterior.lock",
        repo_root=tmp_path,
    )
    posterior_rows = load_repair_campaign_stackability_posterior_rows(posterior_path)
    summary = build_repair_campaign_posterior_prior_summary(
        posterior_path=posterior_path,
    )
    route = summary["acquisition_followup_routes"][0]

    assert posterior_rows[0]["acquisition_policy_delta"][
        "posterior_budget_routing_hint"
    ] == "rebudget_receiver_closed_credit_before_exact_axis_spend"
    assert route["activation_action"] == (
        "rebudget_receiver_credit_to_earliest_entropy_stage"
    )
    assert route["queue_artifact_key"] == "repair_budget_waterfill_queue"


def test_family_materializer_rejects_stale_optimizer_solver_contract(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    score_report["optimizer_decision"]["solver"] = "greedy_campaign_score_waterfill_v1"
    plan = _plan_from_score_report(score_report)

    with pytest.raises(RepairCampaignChainContractError, match="requires solver"):
        build_repair_campaign_family_materializer_manifest(
            repo_root=tmp_path,
            materialization_plan=plan,
            score_report=score_report,
            materialization_plan_path=tmp_path / "plan.json",
            score_report_path=tmp_path / "score_report.json",
            typed_response_id="segnet_region_ready",
            candidate_id="segnet_class_region_waterfill",
        )
