# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA
from comma_lab.scheduler.repair_campaign_materialization_queue import (
    REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA,
    REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA,
    REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_QUEUE_METADATA_SCHEMA,
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
    build_repair_campaign_byte_closed_materialization_queue,
)
from tac.optimization.repair_archive_candidate_intake import (
    REPAIR_ARCHIVE_CANDIDATE_INTAKE_SCHEMA,
    build_repair_campaign_work_order_from_archives,
)
from tac.optimization.repair_campaign_chain_contract import (
    RepairCampaignChainContractError,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import score_repair_campaign
from tac.optimizer.materializer_submission_closure import (
    build_materializer_submission_runtime_closures,
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


def _work_order(tmp_path: Path) -> dict[str, object]:
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
                    "operation_levels": ["frame", "region"],
                    "entropy_position_label": (
                        "before_entropy_coder_distribution_shaping"
                    ),
                    "requested_repair_bytes": 32,
                    "objective_delta_score_units": -0.0010,
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "segnet_class_region_mask_ids": ["road_boundary"],
                    "interaction_scope": {
                        "pair_indices": [7, 9],
                        "region_ids": ["road_boundary"],
                        **_false_authority(),
                    },
                    "stacking_interaction_terms": {
                        "must_remeasure_with_parent_and_sibling_repairs": True,
                        **_false_authority(),
                    },
                    **_false_authority(),
                }
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def _five_family_work_order(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "five_family_mlx_response.json"
    ref = tmp_path / "five_family_reference_mlx_response.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    base = {
        "schema": "frontier_rate_attack_repair_budget_typed_response_row.v1",
        "local_mlx_response_path": str(mlx),
        "reference_local_mlx_response_path": str(ref),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **_false_authority(),
    }
    rows = [
        {
            **base,
            "typed_response_id": "posenet_bottom_decile_ready",
            "candidate_id": "posenet_null_bottom_decile",
            "correction_family": "posenet_null_bottom_decile",
            "targeted_dimensions": ["posenet", "pair"],
            "operation_levels": ["pixel", "frame", "pair"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "requested_repair_bytes": 8,
            "objective_delta_score_units": -0.0015,
            "posenet_null_bottom_decile_pair_ids": ["pair_007"],
        },
        {
            **base,
            "typed_response_id": "segnet_region_ready",
            "candidate_id": "segnet_class_region_waterfill",
            "correction_family": "segnet_class_region_waterfill",
            "targeted_dimensions": ["segnet", "region"],
            "operation_levels": ["pixel", "boundary", "region", "frame"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "requested_repair_bytes": 12,
            "objective_delta_score_units": -0.0024,
            "segnet_class_region_mask_ids": ["road_boundary"],
        },
        {
            **base,
            "typed_response_id": "selector_codec_ready",
            "candidate_id": "per_region_selector_codec",
            "correction_family": "per_region_selector_codec",
            "targeted_dimensions": ["selector_stream", "region"],
            "operation_levels": ["bit", "byte", "boundary", "region", "pair"],
            "entropy_position_label": "selector_codec_entropy",
            "requested_repair_bytes": 6,
            "objective_delta_score_units": -0.0011,
            "selector_payload_bits_per_region": {"road_boundary": 16},
            "receiver_consumed_runtime_replay_proof": {
                "schema": "selector_runtime_replay_proof_stub.v1",
                "receiver_decode_only": True,
                **_false_authority(),
            },
        },
        {
            **base,
            "typed_response_id": "frame0_k16_palette_ready",
            "candidate_id": "frame0_k16_palette_asymmetry",
            "correction_family": "frame0_k16_palette_asymmetry",
            "targeted_dimensions": ["palette", "frame0"],
            "operation_levels": ["pixel", "byte", "frame", "pair"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "requested_repair_bytes": 10,
            "objective_delta_score_units": -0.0012,
            "palette_dynamics_context": {
                "canonical_k": 16,
                "mode_count": 16,
                "frame0_mode_count": 15,
                "frame1_mode_count": 0,
                "frame0_non_identity_fraction": 15 / 16,
            },
        },
        {
            **base,
            "typed_response_id": "entropy_boundary_probe_ready",
            "candidate_id": "entropy_boundary_probe",
            "correction_family": "entropy_boundary_probe",
            "targeted_dimensions": ["rate_bytes", "entropy_coder_boundary"],
            "operation_levels": ["bit", "byte"],
            "entropy_position_label": "at_entropy_coder_integer_codeword_boundary",
            "requested_repair_bytes": 4,
            "objective_delta_score_units": -0.0007,
            "entropy_boundary_probe_manifest": {
                "schema": "entropy_boundary_probe_manifest.v1",
                "probe_kind": "integer_codeword_boundary_slack",
                **_false_authority(),
            },
        },
    ]
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "chain_id": "unit_five_family_repair_chain",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 128,
            **_false_authority(),
        },
        "typed_response_ledger": {
            "schema": "frontier_rate_attack_repair_budget_typed_response_ledger.v1",
            "available_receiver_closed_rate_credit_bytes": 128,
            "rows": rows,
            **_false_authority(),
        },
        **_false_authority(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", b"PSV3" + bytes(range(32)))
        zf.writestr("inflate.py", b"print('decode-only')\n")
    return path


def _write_runtime_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    inflate_sh = path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\npython inflate.py \"$@\"\n", encoding="utf-8")
    inflate_sh.chmod(0o755)
    (path / "inflate.py").write_text("print('decode-only')\n", encoding="utf-8")
    return path


def test_byte_closed_materialization_queue_emits_archive_bound_steps(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    report_path = _write_json(tmp_path / "score_report.json", report)

    queue = build_repair_campaign_byte_closed_materialization_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        work_order_path=work_order_path,
        results_root=tmp_path / "results",
        queue_id="unit_repair_materialization",
    )

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["metadata"]["schema"] == (
        REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_QUEUE_METADATA_SCHEMA
    )
    assert queue["metadata"]["ready_experiment_count"] == 1
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert experiment["metadata"]["schema"] == (
        REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA
    )
    assert experiment["metadata"]["local_mlx_rows_are_advisory_only"] is True
    assert experiment["metadata"][
        "exact_eval_handoff_requires_complete_archive_runtime_component_custody"
    ] is True
    assert experiment["metadata"]["repair_materialization_lineage"][
        "target_queue_artifact_key"
    ] == "repair_campaign_byte_closed_materialization_queue"
    assert experiment["metadata"]["interaction_dynamics"]["active_scales"]
    assert experiment["metadata"]["entropy_pipeline_position"]["stage_index"] == 0
    assert experiment["metadata"]["entropy_pipeline_position"][
        "can_shape_coder_input_distribution"
    ] is True
    assert experiment["metadata"]["entropy_pipeline_materialization_order"] == 1
    assert queue["metadata"]["source_optimizer_solver"] == (
        "interaction_aware_entropy_stage_waterfill_v1"
    )
    assert queue["metadata"]["operator_visible_automation_rollup"][
        "exact_eval_handoff_fail_closed_until_custody_complete"
    ] is True
    assert [step["id"] for step in experiment["steps"]] == [
        "emit_repair_budget_materialization_plan",
        "emit_repair_family_materializer_manifest",
        "emit_repair_budget_child_component_replay_manifests",
        "execute_repair_family_byte_transform",
        "bind_repair_budget_materializer_execution",
        "audit_repair_budget_materialization_execution",
        "emit_selected_repair_materialization_gate",
        "build_repair_materialization_learning_signal",
        "append_repair_materialization_posterior_signal",
    ]
    assert experiment["steps"][0]["command"][1] == (
        "tools/build_frontier_repair_budget_materialization_plan.py"
    )
    assert str(work_order_path) in experiment["steps"][0]["command"]
    assert experiment["metadata"]["repair_family_materializer_manifest_schema"] == (
        REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    )
    assert experiment["metadata"][
        "repair_family_byte_transform_execution_report_schema"
    ] == "repair_family_byte_transform_execution_report.v1"
    assert "--materializer-manifest" in experiment["steps"][4]["command"]
    assert any(
        str(item).endswith("repair_family_byte_transform_execution_report.json")
        for item in experiment["steps"][4]["command"]
    )
    assert experiment["steps"][6]["postconditions"][0]["equals"] == (
        REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA
    )
    assert experiment["steps"][7]["postconditions"][0]["equals"] == (
        REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
    )
    assert experiment["steps"][8]["postconditions"][0]["equals"] == (
        REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA
    )


def test_byte_closed_materialization_queue_orders_allocations_by_entropy_stage(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    ledger = work_order["typed_response_ledger"]  # type: ignore[index]
    first = ledger["rows"][0]  # type: ignore[index]
    selector_row = {
        **first,
        "typed_response_id": "selector_codec_ready",
        "candidate_id": "per_region_selector_codec_ready",
        "correction_family": "per_region_selector_codec",
        "targeted_dimensions": ["selector_stream", "region"],
        "operation_levels": ["region", "entropy_coder"],
        "entropy_position_label": "selector_codec_entropy",
        "requested_repair_bytes": 4,
        "objective_delta_score_units": -0.01,
        "selector_payload_bits_per_region": {"road_boundary": 16},
        "receiver_consumed_runtime_replay_proof": {"passed": True},
        **_false_authority(),
    }
    ledger["rows"].append(selector_row)  # type: ignore[index]
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    report_path = _write_json(tmp_path / "score_report.json", report)

    assert [
        row["typed_response_id"]
        for row in report["optimizer_decision"]["selected_allocation_rows"]
    ] == ["segnet_region_ready", "selector_codec_ready"]
    assert report["optimizer_decision"]["selected_allocation_rows"][0][
        "selection_rationale"
    ] == "interaction_aware_entropy_stage_waterfill_under_receiver_closed_byte_credit"

    queue = build_repair_campaign_byte_closed_materialization_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        work_order_path=work_order_path,
        results_root=tmp_path / "results",
        queue_id="unit_repair_materialization",
    )

    assert [
        experiment["metadata"]["typed_response_id"]
        for experiment in queue["experiments"]
    ] == ["segnet_region_ready", "selector_codec_ready"]
    assert queue["metadata"]["entropy_pipeline_materialization_order"][0][
        "entropy_pipeline_stage_index"
    ] == 0
    assert queue["metadata"]["entropy_pipeline_materialization_order"][1][
        "entropy_pipeline_stage_index"
    ] == 2


def test_archive_candidate_intake_builds_real_archive_five_family_work_order(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "source_archive.zip")
    training_artifact = _write_json(
        tmp_path / "training_artifact.json",
        {
            "schema_version": "unit_mlx_training_artifact.v1",
            "per_epoch_metrics": [{"loss": 1.0}, {"loss": 0.02}],
            "total_epochs_completed": 2,
            "total_wall_clock_seconds": 1.0,
            "archive_path": str(archive),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    equivalence_gate = _write_json(
        tmp_path / "equivalence_gate.json",
        {
            "schema_version": "unit_equivalence_gate.v1",
            "axis_tag": "[macOS-MLX research-signal]",
            "margin_below_threshold": 0.1,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    work_order = build_repair_campaign_work_order_from_archives(
        archive_paths=[archive],
        output_dir=tmp_path / "intake",
        repo_root=REPO_ROOT,
        source_labels=["unit_psv3"],
        training_artifact_paths=[training_artifact],
        equivalence_gate_paths=[equivalence_gate],
        chain_id="unit_real_archive_repair",
        overwrite=True,
    )

    assert work_order["schema"] == (
        "frontier_rate_attack_repair_budget_waterfill_work_order.v1"
    )
    intake = work_order["archive_candidate_intake"]
    assert intake["schema"] == REPAIR_ARCHIVE_CANDIDATE_INTAKE_SCHEMA
    assert intake["archive_count"] == 1
    assert intake["typed_response_count"] == 5
    families = {
        row["correction_family"]
        for row in work_order["typed_response_ledger"]["rows"]
    }
    assert families == {
        "entropy_boundary_probe",
        "frame0_k16_palette_asymmetry",
        "per_region_selector_codec",
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
    }
    for row in work_order["typed_response_ledger"]["rows"]:
        assert row["candidate_archive_path"]
        assert row["runtime_consumption_proof_path"]
        assert row["receiver_consumed"] is True
        assert row["ready_for_exact_eval_dispatch"] is False
    palette_row = next(
        row
        for row in work_order["typed_response_ledger"]["rows"]
        if row["correction_family"] == "frame0_k16_palette_asymmetry"
    )
    assert palette_row["palette_dynamics_context"]["frame0_mode_count"] == 15
    assert palette_row["palette_dynamics_context"]["frame1_mode_count"] == 0


def test_archive_candidate_intake_overwrite_is_deterministically_rerunnable(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "source_archive.zip")
    output_dir = tmp_path / "intake"

    first = build_repair_campaign_work_order_from_archives(
        archive_paths=[archive],
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        source_labels=["unit_psv3"],
        chain_id="unit_real_archive_repair",
        overwrite=True,
    )
    second = build_repair_campaign_work_order_from_archives(
        archive_paths=[archive],
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        source_labels=["unit_psv3"],
        chain_id="unit_real_archive_repair",
        overwrite=True,
    )

    first_row = first["typed_response_ledger"]["rows"][0]
    second_row = second["typed_response_ledger"]["rows"][0]
    assert second["schema"] == first["schema"]
    assert second_row["candidate_archive_sha256"] == first_row["candidate_archive_sha256"]
    assert second_row["runtime_consumption_proof_path"] == first_row[
        "runtime_consumption_proof_path"
    ]


def test_real_archive_intake_runs_all_families_through_floor_loop(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "source_archive.zip")
    work_order = build_repair_campaign_work_order_from_archives(
        archive_paths=[archive],
        output_dir=tmp_path / "intake",
        repo_root=REPO_ROOT,
        source_labels=["unit_psv3"],
        chain_id="unit_real_archive_repair",
        overwrite=True,
    )
    work_order_path = _write_json(tmp_path / "real_archive_work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=REPO_ROOT)
    assert report["optimizer_decision"]["selected_allocation_count"] == 5
    report_path = _write_json(tmp_path / "real_archive_score_report.json", report)
    queue = build_repair_campaign_byte_closed_materialization_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        work_order_path=work_order_path,
        results_root=tmp_path / "real_archive_results",
        queue_id="unit_real_archive_repair_materialization",
    )
    queue_path = _write_json(tmp_path / "real_archive_queue.json", queue)
    summary_path = tmp_path / "real_archive_floor_loop_summary.json"
    posterior_path = tmp_path / "real_archive_floor_loop_posterior.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_repair_campaign_autonomous_floor_loop.py"),
            "--materialization-queue",
            str(queue_path),
            "--output-dir",
            str(tmp_path / "real_archive_loop"),
            "--summary-out",
            str(summary_path),
            "--posterior-path",
            str(posterior_path),
            "--posterior-lock-path",
            str(tmp_path / ".real_archive_floor_loop_posterior.lock"),
            "--execute-local",
            "--require-all-queue-families",
            "--worker-max-experiments-per-iteration",
            "5",
            "--max-steps-per-iteration",
            "80",
            "--max-iterations",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["repair_family_coverage"]["coverage_satisfied"] is True
    assert summary["archive_bound_exact_handoff_candidate_count"] == 5
    assert summary["entropy_stage_chain_execution_bundle_schema"] == (
        "repair_entropy_stage_chain_execution_bundle.v1"
    )
    assert summary["entropy_stage_chain_count"] == 1
    assert summary["entropy_stage_chain_materialized_candidate_count"] == 1
    assert (
        summary["entropy_stage_chain_runtime_consumption_proof_ready_count"]
        == 1
    )
    assert summary["posterior_learning_signal_count"] == 5
    assert summary["ready_for_exact_eval_dispatch"] is False
    source_queue_path = tmp_path / "real_archive_loop" / "repair_family_exact_ready_source_queue.json"
    source_queue = json.loads(source_queue_path.read_text(encoding="utf-8"))
    assert {
        row["receiver_contract_satisfied"] for row in source_queue["top_k"]
    } == {True}
    receiver_contract_kinds = {
        row["receiver_contract_kind"] for row in source_queue["top_k"]
    }
    assert receiver_contract_kinds <= {
        "family_agnostic_archive_zip_repack",
        "family_agnostic_packet_member_recompress",
    }
    assert "family_agnostic_packet_member_recompress" in receiver_contract_kinds

    runtime_dir = _write_runtime_dir(tmp_path / "runtime")
    closure_report = build_materializer_submission_runtime_closures(
        repo_root=REPO_ROOT,
        source_queue_path=source_queue_path,
        source_runtime_dir=runtime_dir,
        submission_dir_out=tmp_path / "submission_runtime_closure",
        closed_source_queue_out=tmp_path / "closed_repair_family_exact_ready_source_queue.json",
        closure_report_out=tmp_path / "submission_runtime_closure_report.json",
        overwrite=True,
    )
    assert closure_report["candidate_count"] == 5
    assert {
        row["materializer_submission_closure_kind"]
        for row in closure_report["rows"]
    } == {"source_runtime_static_closure_with_candidate_archive"}


def test_byte_closed_materialization_queue_rejects_stale_optimizer_solver_contract(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    report["optimizer_decision"]["solver"] = "greedy_campaign_score_waterfill_v1"

    with pytest.raises(RepairCampaignChainContractError, match="requires solver"):
        build_repair_campaign_byte_closed_materialization_queue(
            repo_root=REPO_ROOT,
            score_report=report,
            score_report_path=tmp_path / "score_report.json",
            work_order_path=work_order_path,
            results_root=tmp_path / "results",
            queue_id="stale_repair_materialization_solver",
        )


def test_byte_closed_materialization_queue_cli_writes_queue(tmp_path: Path) -> None:
    work_order = _work_order(tmp_path)
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    report_path = _write_json(tmp_path / "score_report.json", report)
    queue_path = tmp_path / "repair_materialization_queue.json"
    posterior_path = tmp_path / "repair_campaign_stackability_posterior.jsonl"
    posterior_lock_path = tmp_path / ".repair_campaign_stackability_posterior.lock"

    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tools"
                / "build_repair_campaign_byte_closed_materialization_queue.py"
            ),
            "--score-report",
            str(report_path),
            "--work-order",
            str(work_order_path),
            "--materialization-queue-out",
            str(queue_path),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "unit_repair_materialization",
            "--posterior-path",
            str(posterior_path),
            "--posterior-lock-path",
            str(posterior_lock_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["schema"] == QUEUE_SCHEMA
    assert payload["metadata"]["ready_experiment_count"] == 1
    validate_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate_result.returncode == 0, validate_result.stderr
    worker_result_path = tmp_path / "worker_result.json"
    worker_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(tmp_path / "repair_materialization_queue_state.sqlite"),
            "run-worker",
            "--noncanonical-state-rationale",
            "unit_test_uses_isolated_queue_state_to_avoid_shared_worker_collision",
            "--execute",
            "--max-steps",
            "9",
            "--max-experiments",
            "1",
            "--max-parallel",
            "1",
            "--output",
            str(worker_result_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert worker_result.returncode == 0, worker_result.stderr
    worker_payload = json.loads(worker_result_path.read_text(encoding="utf-8"))
    assert worker_payload["failure_count"] == 0
    assert posterior_path.is_file()


def test_repair_campaign_autonomous_floor_loop_cli_fails_closed(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    report_path = _write_json(tmp_path / "score_report.json", report)
    queue = build_repair_campaign_byte_closed_materialization_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        work_order_path=work_order_path,
        results_root=tmp_path / "results",
        queue_id="unit_repair_materialization",
    )
    queue_path = _write_json(tmp_path / "repair_materialization_queue.json", queue)
    summary_path = tmp_path / "floor_loop_summary.json"
    posterior_path = tmp_path / "floor_loop_posterior.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_repair_campaign_autonomous_floor_loop.py"),
            "--materialization-queue",
            str(queue_path),
            "--output-dir",
            str(tmp_path / "loop"),
            "--summary-out",
            str(summary_path),
            "--posterior-path",
            str(posterior_path),
            "--posterior-lock-path",
            str(tmp_path / ".floor_loop_posterior.lock"),
            "--max-iterations",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema"] == "repair_campaign_autonomous_floor_loop.v1"
    assert summary["stop_reason"] == "materialization_execution_reports_missing"
    assert summary["stack_search_plan"]["execution_report_count"] == 0
    assert summary["exact_handoff_plan_schema"] == "repair_family_exact_handoff_plan.v1"
    assert summary["exact_eval_handoff_candidate_count"] == 0
    assert summary["archive_bound_exact_handoff_candidate_count"] == 0
    assert summary["exact_ready_bridge_report_schema"] == (
        "repair_family_exact_ready_bridge_report.v1"
    )
    assert summary["exact_ready_bridge_candidate_count"] == 0
    assert summary["posterior_learning_signal_report_schema"] == (
        "repair_campaign_blocked_learning_signal_report.v1"
    )
    assert summary["posterior_learning_signal_count"] == 1
    assert summary["posterior_appended_count"] == 1
    assert summary["exact_axis_blocker_report_schema"] == (
        "repair_campaign_autonomous_floor_loop_blocker_report.v1"
    )
    exact_handoff_plan = json.loads(
        (tmp_path / "loop" / "repair_family_exact_handoff_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert exact_handoff_plan["schema"] == "repair_family_exact_handoff_plan.v1"
    assert exact_handoff_plan["candidate_count"] == 0
    bridge_report = json.loads(
        (tmp_path / "loop" / "repair_family_exact_ready_bridge_report.json").read_text(
            encoding="utf-8"
        )
    )
    blocked_queue = json.loads(
        (tmp_path / "loop" / "repair_family_blocked_exact_ready_queue.json").read_text(
            encoding="utf-8"
        )
    )
    learning_signal_report = json.loads(
        (tmp_path / "loop" / "repair_family_stack_learning_signal_report.json").read_text(
            encoding="utf-8"
        )
    )
    blocker_report = json.loads(
        (tmp_path / "loop" / "repair_family_floor_loop_blocker_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert bridge_report["candidate_count"] == 0
    assert blocked_queue["dispatch_ready_count"] == 0
    assert learning_signal_report["learning_signal_count"] == 1
    assert blocker_report["selected_blocker_class"] == (
        "repair_family_byte_transform_execution_reports_missing"
    )
    assert posterior_path.is_file()
    assert summary["ready_for_exact_eval_dispatch"] is False


def test_repair_campaign_autonomous_floor_loop_executes_all_required_queue_families(
    tmp_path: Path,
) -> None:
    work_order = _five_family_work_order(tmp_path)
    work_order_path = _write_json(tmp_path / "five_family_work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    assert report["optimizer_decision"]["selected_allocation_count"] == 5
    assert report["optimizer_decision"]["family_allocation_histogram"] == {
        "entropy_boundary_probe": 1,
        "frame0_k16_palette_asymmetry": 1,
        "per_region_selector_codec": 1,
        "posenet_null_bottom_decile": 1,
        "segnet_class_region_waterfill": 1,
    }
    report_path = _write_json(tmp_path / "five_family_score_report.json", report)
    queue = build_repair_campaign_byte_closed_materialization_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        work_order_path=work_order_path,
        results_root=tmp_path / "results",
        queue_id="unit_five_family_repair_materialization",
    )
    queue_path = _write_json(tmp_path / "five_family_repair_materialization_queue.json", queue)
    summary_path = tmp_path / "five_family_floor_loop_summary.json"
    posterior_path = tmp_path / "five_family_floor_loop_posterior.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_repair_campaign_autonomous_floor_loop.py"),
            "--materialization-queue",
            str(queue_path),
            "--output-dir",
            str(tmp_path / "five_family_loop"),
            "--summary-out",
            str(summary_path),
            "--posterior-path",
            str(posterior_path),
            "--posterior-lock-path",
            str(tmp_path / ".five_family_floor_loop_posterior.lock"),
            "--execute-local",
            "--require-all-queue-families",
            "--worker-max-experiments-per-iteration",
            "5",
            "--max-steps-per-iteration",
            "80",
            "--max-iterations",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coverage = summary["repair_family_coverage"]
    assert coverage["coverage_satisfied"] is True
    assert coverage["required_family_count"] == 5
    assert set(coverage["executed_family_ids"]) == {
        "entropy_boundary_probe",
        "frame0_k16_palette_asymmetry",
        "per_region_selector_codec",
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
    }
    assert coverage["missing_required_family_ids"] == []
    assert summary["stack_search_plan"]["execution_report_count"] == 5
    assert summary["stack_search_plan"]["pairwise_interaction_tensor_cell_count"] == 20
    assert summary["stack_search_plan"]["n_way_hypergraph_acquisition_enabled"] is True
    assert summary["stack_search_plan"]["hypergraph_interaction_tensor_cell_count"] == 26
    assert (
        summary["stack_search_plan"]["fractal_marginal_surface"][
            "measured_mlx_marginal_update_count"
        ]
        > 0
    )
    assert summary["fractal_marginal_surface_schema"] == (
        "repair_family_fractal_marginal_surface.v1"
    )
    assert summary["fractal_marginal_surface_cell_count"] > 0
    assert summary["top_fractal_marginal_surface_cells"]
    assert summary["stack_acquisition_frontier_count"] > 0
    assert summary["primary_stack_acquisition_frontier_path"]["source_tensor"] == (
        "hypergraph_interaction_tensor"
    )
    assert summary["measured_mlx_posterior_budget_routing_update_count"] > 0
    assert summary["entropy_stage_materializer_work_order_count"] > 0
    assert (
        summary["entropy_stage_materializer_work_orders"][
            "archive_bound_candidate_default"
        ]
        is True
    )
    assert summary["entropy_stage_chain_execution_bundle_schema"] == (
        "repair_entropy_stage_chain_execution_bundle.v1"
    )
    assert summary["entropy_stage_chain_count"] == 0
    assert summary["entropy_stage_chain_materialized_candidate_count"] == 0
    assert summary["exact_dispatch_preclaim_gate_count"] == 5
    assert summary["failure_rebudgeting_update_count"] == 5
    primary_path = summary["stack_search_plan"]["primary_stack_acquisition_path"]
    assert primary_path["path_kind"] == "n_way_hypergraph_interaction_tensor_acquisition"
    assert primary_path["source_hyperedge_order"] >= 2
    assert summary["posterior_learning_signal_count"] == 5
    assert summary["ready_for_exact_eval_dispatch"] is False


def test_repair_campaign_autonomous_floor_loop_preserves_precise_terminal_class(
    tmp_path: Path,
) -> None:
    queue_path = _write_json(
        tmp_path / "queue.json",
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": "unit_pairwise_terminal",
            "controls": {
                "mode": "running",
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": "segnet_region_ready",
                    "metadata": {
                        "queue_actuation_ready": True,
                        "family_id": "segnet_class_region_waterfill",
                        "typed_response_id": "segnet_region_ready",
                        **_false_authority(),
                    },
                    "steps": [
                        {
                            "id": "noop",
                            "command": [sys.executable, "-c", "print('segnet')"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                },
                {
                    "id": "selector_codec_ready",
                    "metadata": {
                        "queue_actuation_ready": True,
                        "family_id": "per_region_selector_codec",
                        "typed_response_id": "selector_codec_ready",
                        **_false_authority(),
                    },
                    "steps": [
                        {
                            "id": "noop",
                            "command": [sys.executable, "-c", "print('selector')"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                },
                {
                    "id": "entropy_boundary_not_selected",
                    "metadata": {
                        "queue_actuation_ready": True,
                        "family_id": "entropy_boundary_probe",
                        "typed_response_id": "entropy_boundary_not_selected",
                        **_false_authority(),
                    },
                    "steps": [
                        {
                            "id": "noop",
                            "command": [sys.executable, "-c", "print('entropy')"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                },
            ],
            "metadata": {},
            **_false_authority(),
        },
    )
    output_dir = tmp_path / "loop"
    for family_id, typed_response_id, stage, levels, delta, byte_count in (
        (
            "segnet_class_region_waterfill",
            "segnet_region_ready",
            10,
            ["pixel", "boundary", "region", "frame"],
            -0.0020,
            32,
        ),
        (
            "per_region_selector_codec",
            "selector_codec_ready",
            20,
            ["bit", "byte", "boundary", "region", "pair"],
            -0.0010,
            12,
        ),
    ):
        report_dir = output_dir / family_id
        report_dir.mkdir(parents=True)
        _write_json(
            report_dir / "repair_family_byte_transform_execution_report.json",
            {
                "schema": "repair_family_byte_transform_execution_report.v1",
                "family_id": family_id,
                "typed_response_id": typed_response_id,
                "candidate_chain_id": f"{family_id}_chain",
                "candidate_chain_ids": [f"{family_id}_chain"],
                "entropy_position_label": "before_entropy_coder_distribution_shaping",
                "active_entropy_stage": {
                    "order": stage,
                    "stage": "before_entropy_coder_distribution_shaping",
                },
                "fractal_optimization_scope": {
                    "active_levels": levels,
                    "declared_levels": levels,
                },
                "allocated_repair_bytes": byte_count,
                "byte_transform_delta": {
                    "schema": "repair_family_byte_transform_delta.v1",
                    "path": f"{family_id}.json",
                    "bytes": byte_count,
                    **_false_authority(),
                },
                "mlx_local_probe_delta": {
                    "schema": "repair_family_byte_transform_mlx_probe_delta.v1",
                    "combined_delta_score_units": delta,
                    "segnet_delta_score_units": delta,
                    "posenet_delta_score_units": 0.0,
                    **_false_authority(),
                },
                "byte_closed_candidate_emitted": False,
                "candidate_archive_materialized": False,
                "exact_eval_handoff_gate": {
                    "schema": "repair_family_exact_eval_handoff_gate.v1",
                    "archive_bound_runtime_consumption_proof_ready": False,
                    "blockers": ["byte_closed_candidate_archive_missing"],
                    **_false_authority(),
                },
                "blockers": [],
                **_false_authority(),
            },
        )
    summary_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_repair_campaign_autonomous_floor_loop.py"),
            "--materialization-queue",
            str(queue_path),
            "--output-dir",
            str(output_dir),
            "--summary-out",
            str(summary_path),
            "--byte-credit-budget",
            "64",
            "--max-iterations",
            "2",
            "--execute-local",
            "--worker-max-experiments-per-iteration",
            "3",
            "--max-steps-per-iteration",
            "10",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["stack_search_plan"]["candidate_improvement_observed"] is True
    assert summary["primary_stack_acquisition_terminal_outcome"] == (
        "precise_exact_axis_blocker"
    )
    iteration = summary["iterations"][0]
    selected_report = iteration["frontier_selected_queue_report"]
    assert summary["frontier_executable_selection_consumed"] is True
    assert selected_report["selected_experiment_count"] == 2
    assert selected_report["skipped_experiment_count"] == 1
    assert (
        selected_report["archive_bound_candidate_default_contract"][
            "candidate_archive_emission_default"
        ]
        is True
    )
    assert (
        selected_report["archive_bound_candidate_default_contract"][
            "selected_missing_archive_bound_default_count"
        ]
        == 2
    )
    assert summary["measured_mlx_posterior_budget_routing_update_count"] > 0
    assert summary["entropy_stage_materializer_work_order_count"] > 0
    assert iteration["worker_queue_path"].endswith(
        "iteration_1_frontier_selected_queue.json"
    )
    selected_queue = json.loads(
        (output_dir / "iteration_1_frontier_selected_queue.json").read_text(
            encoding="utf-8"
        )
    )
    assert [experiment["id"] for experiment in selected_queue["experiments"]] == [
        "segnet_region_ready",
        "selector_codec_ready",
    ]
    assert summary["stop_reason"] == "precise_exact_axis_blocker"
    assert summary["exact_dispatch_preclaim_gate_count"] == 2
    assert summary["failure_rebudgeting_update_count"] == 2
    blocker_report = summary["exact_axis_blocker_report"]
    assert blocker_report["stop_reason"] == "precise_exact_axis_blocker"
    assert blocker_report["selected_blocker_class"] == (
        "archive_bound_candidate_required_for_measured_mlx_marginal"
    )
    assert summary["ready_for_exact_eval_dispatch"] is False
