# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler.byte_shaving_campaign_queue import (
    MATERIALIZATION_SCHEMA,
    MATERIALIZER_BACKLOG_SCHEMA,
    MATERIALIZER_CONTEXTS_SCHEMA,
    MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA,
    MATERIALIZER_EXECUTION_STEP_ID,
    MATERIALIZER_WORK_QUEUE_SCHEMA,
    build_materializer_execution_queue,
    build_materializer_work_queue,
    compile_dqs1_byte_shaving_campaign,
    materializer_contexts_from_payload,
)
from comma_lab.scheduler.byte_shaving_materializer_registry import (
    BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
    BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID,
    BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND,
    BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
    DQS1_DROP_PAIR_MATERIALIZER,
    DQS1_PAIRSET_TARGET_KIND,
    DQS1_RECEIVER_CONTRACT_ID,
    DQS1_RECEIVER_CONTRACT_KIND,
    registry_manifest,
    resolve_materializer,
    suggest_materializer_adapters,
)
from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    _condition_passes,
    load_queue_definition,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME,
    CHAIN_SCHEMA,
)
from tac.optimization.byte_shaving_campaign import (
    SIGNAL_SURFACE_SCHEMA,
    build_byte_shaving_campaign_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_byte_shaving_campaign_queue.py"


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _pair_drop_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "dqs1_byte_shave_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_dqs1_byte_shave_fixture",
        "dqs1_base_pair_indices": [101, 320, 371, 501],
        "combo_beam_width": 16,
        "max_combo_count": 16,
        "units": [
            {
                "unit_id": "pair0371",
                "unit_kind": "pair",
                "candidate_saved_bytes": 1000,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.95,
                "operations": [
                    {
                        "operation_id": "drop_pair0371",
                        "operation_family": "drop_pair",
                        "materializer": DQS1_DROP_PAIR_MATERIALIZER,
                    }
                ],
            },
            {
                "unit_id": "pair0320",
                "unit_kind": "pair",
                "candidate_saved_bytes": 900,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.9,
                "operations": [
                    {
                        "operation_id": "drop_pair0320",
                        "operation_family": "drop_pair",
                        "materializer": DQS1_DROP_PAIR_MATERIALIZER,
                    }
                ],
            },
            {
                "unit_id": "byte_null_run_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 500,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operation_families": ["null_remove_or_seed"],
            },
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=3)


def _byte_range_entropy_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "byte_range_entropy_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_byte_range_entropy_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "zip_member_range_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 777,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operations": [
                    {
                        "operation_id": "entropy_recode_zip_member_range_a",
                        "operation_family": "entropy_recode",
                        "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
                    }
                ],
            },
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=1)


def test_byte_shaving_materializer_registry_exposes_dqs1_and_byte_range_contracts() -> None:
    manifest = registry_manifest()

    assert manifest["schema"] == "byte_shaving_materializer_registry.v1"
    assert manifest["known_target_kinds"] == [
        BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
        DQS1_PAIRSET_TARGET_KIND,
    ]
    adapters = {row["materializer_id"]: row for row in manifest["adapters"]}
    assert adapters[DQS1_DROP_PAIR_MATERIALIZER] == {
        "description": "Compile pair-unit drop operations into DQS1 pairset local-first queue rows.",
        "executable": True,
        "cooperative_receiver_required": True,
        "materializer_id": DQS1_DROP_PAIR_MATERIALIZER,
        "materialization_resource_kind": "local_cpu",
        "implementation_module": "comma_lab.scheduler.byte_shaving_campaign_queue",
        "plan_function": "",
        "materialize_function": "",
        "receiver_proof_function": "",
        "receiver_verify_function": "",
        "operation_family": "drop_pair",
        "receiver_contract_id": DQS1_RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": DQS1_RECEIVER_CONTRACT_KIND,
        "required_context_fields": ["dqs1_base_pair_indices"],
        "target_kind": DQS1_PAIRSET_TARGET_KIND,
        "unit_kind": "pair",
    }
    assert adapters[BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER] == {
        "description": (
            "Fail-closed contract for byte-range entropy recode work; requires "
            "archive-member mapping and runtime-consumption proof before queue execution."
        ),
        "executable": False,
        "cooperative_receiver_required": True,
        "materializer_id": BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
        "materialization_resource_kind": "local_cpu",
        "implementation_module": (
            "tac.optimization.byte_range_entropy_recode_materializer"
        ),
        "plan_function": "build_byte_range_entropy_recode_plan",
        "materialize_function": "materialize_byte_range_entropy_recode_candidate",
        "receiver_proof_function": "build_byte_range_entropy_recode_receiver_proof",
        "receiver_verify_function": (
            "verify_byte_range_entropy_recode_receiver_contract"
        ),
        "operation_family": "entropy_recode",
        "receiver_contract_id": BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND,
        "required_context_fields": [
            "archive_member_name",
            "archive_byte_range",
            "runtime_consumption_proof",
        ],
        "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
        "unit_kind": "byte_range",
    }
    grammar_registry = manifest["cooperative_receiver_grammar_registry"]
    assert (
        grammar_registry["schema"]
        == "cooperative_receiver_packet_grammar_registry_hook.v1"
    )
    assert grammar_registry["known_grammar_count"] >= 1
    assert grammar_registry["score_claim"] is False

    resolved = resolve_materializer(
        operation={
            "unit_id": "pair0371",
            "operation_id": "drop_pair0371",
            "operation_family": "drop_pair",
            "materializer": DQS1_DROP_PAIR_MATERIALIZER,
        },
        unit={"unit_id": "pair0371", "unit_kind": "pair"},
    )
    assert resolved.executable is True
    assert resolved.materializer_id == DQS1_DROP_PAIR_MATERIALIZER
    assert resolved.target_kind == DQS1_PAIRSET_TARGET_KIND
    assert resolved.receiver_contract_id == DQS1_RECEIVER_CONTRACT_ID
    assert resolved.receiver_contract_kind == DQS1_RECEIVER_CONTRACT_KIND
    assert resolved.cooperative_receiver_required is True
    assert resolved.blockers == ()


def test_byte_shaving_materializer_registry_refuses_implicit_dqs1_pair_drop() -> None:
    resolved = resolve_materializer(
        operation={
            "unit_id": "pair0371",
            "operation_id": "drop_pair0371",
            "operation_family": "drop_pair",
        },
        unit={"unit_id": "pair0371", "unit_kind": "pair"},
    )

    assert resolved.executable is False
    assert resolved.materializer_id is None
    assert resolved.target_kind is None
    assert "materializer_target_kind_required:pair:drop_pair" in resolved.blockers


def test_byte_shaving_materializer_registry_allows_explicit_dqs1_target_kind() -> None:
    resolved = resolve_materializer(
        operation={
            "unit_id": "pair0371",
            "operation_id": "drop_pair0371",
            "operation_family": "drop_pair",
            "target_kind": DQS1_PAIRSET_TARGET_KIND,
        },
        unit={"unit_id": "pair0371", "unit_kind": "pair"},
    )

    assert resolved.executable is True
    assert resolved.materializer_id == DQS1_DROP_PAIR_MATERIALIZER
    assert resolved.target_kind == DQS1_PAIRSET_TARGET_KIND
    assert resolved.blockers == ()


def test_byte_shaving_materializer_registry_registers_byte_range_entropy_fail_closed() -> None:
    resolved = resolve_materializer(
        operation={
            "unit_id": "zip_member_range_a",
            "operation_id": "entropy_recode_a",
            "operation_family": "entropy_recode",
            "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
        },
        unit={"unit_id": "zip_member_range_a", "unit_kind": "byte_range"},
    )

    assert resolved.executable is False
    assert resolved.materializer_id == BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER
    assert resolved.target_kind == BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND
    assert resolved.receiver_contract_id == BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID
    assert (
        resolved.receiver_contract_kind
        == BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND
    )
    assert resolved.cooperative_receiver_required is True
    assert resolved.materialization_resource_kind == "local_cpu"
    assert resolved.blockers == (
        f"materializer_not_executable:{BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER}",
    )
    suggestions = suggest_materializer_adapters(
        unit_kind="byte_range",
        operation_family="entropy_recode",
    )
    assert [adapter.materializer_id for adapter in suggestions] == [
        BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER
    ]


def test_compile_dqs1_byte_shaving_plan_preserves_explicit_target_kind(
    tmp_path: Path,
) -> None:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "dqs1_target_kind_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_dqs1_target_kind_fixture",
        "combo_beam_width": 8,
        "max_combo_count": 8,
        "dqs1_base_pair_indices": [101, 320, 371, 501],
        "units": [
            {
                "unit_id": "pair0371",
                "unit_kind": "pair",
                "candidate_saved_bytes": 1000,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.95,
                "operations": [
                    {
                        "operation_id": "drop_pair0371",
                        "operation_family": "drop_pair",
                        "target_kind": DQS1_PAIRSET_TARGET_KIND,
                    }
                ],
            },
            {
                "unit_id": "pair0320",
                "unit_kind": "pair",
                "candidate_saved_bytes": 900,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.9,
                "operations": [
                    {
                        "operation_id": "drop_pair0320",
                        "operation_family": "drop_pair",
                        "target_kind": DQS1_PAIRSET_TARGET_KIND,
                    }
                ],
            },
        ],
        **_false_authority(),
    }
    plan = build_byte_shaving_campaign_plan(surface, max_k=2)
    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    assert compiled["executable_row_count"] >= 1
    assert all(
        resolution["materializer_id"] == DQS1_DROP_PAIR_MATERIALIZER
        and resolution["target_kind"] == DQS1_PAIRSET_TARGET_KIND
        for row in compiled["executable_rows"]
        for resolution in row["materializer_resolutions"]
    )


def test_compile_dqs1_byte_shaving_plan_emits_action_summary_and_blocks_unknown_ops(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
        allow_partial_materialization=True,
        partial_materialization_rationale="unit-test mixed fixture",
    )

    assert compiled["schema"] == MATERIALIZATION_SCHEMA
    assert compiled["materializer_backlog"]["schema"] == MATERIALIZER_BACKLOG_SCHEMA
    assert compiled["materializer_work_queue"]["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert compiled["score_claim"] is False
    assert compiled["executable_row_count"] >= 1
    assert compiled["blocked_row_count"] >= 1
    backlog_row = next(
        row
        for row in compiled["materializer_backlog"]["rows"]
        if row["unit_kind"] == "byte_range"
        and row["operation_family"] == "null_remove_or_seed"
    )
    assert backlog_row["gap_class"] == "target_kind_required"
    assert (
        backlog_row["receiver_contract_status"]
        == "receiver_target_contract_required"
    )
    assert backlog_row["receiver_contract_id"] is None
    assert backlog_row["blocked_row_count"] >= 1
    assert backlog_row["candidate_saved_bytes_sum"] > 0
    assert "byte_null_run_a" in backlog_row["source_unit_ids"]
    assert compiled["action_summary"]["materializer_backlog_summary"][
        "backlog_row_count"
    ] == compiled["materializer_backlog"]["backlog_row_count"]
    assert any(
        "materializer_target_kind_required:byte_range:null_remove_or_seed"
        in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    both = next(
        row
        for row in compiled["executable_rows"]
        if row["dropped_pair_indices"] == [320, 371]
    )
    assert both["selected_pair_indices"] == [101, 501]
    assert {unit["unit_id"] for unit in both["source_units"]} == {"pair0320", "pair0371"}
    assert all(
        resolution["materializer_id"] == DQS1_DROP_PAIR_MATERIALIZER
        for resolution in both["materializer_resolutions"]
    )
    assert all(
        resolution["receiver_contract_id"] == DQS1_RECEIVER_CONTRACT_ID
        for resolution in both["materializer_resolutions"]
    )
    action = next(
        row
        for row in compiled["action_summary"]["top_operator_actions"]
        if row["candidate_id"] == both["candidate_id"]
    )
    assert action["operator_next_action"] == "materialize_pairset_archive_and_run_local_controls"
    portfolio_row = next(
        row
        for row in compiled["portfolio"]["operator_action_rows"]
        if row["candidate_id"] == both["candidate_id"]
    )
    assert portfolio_row["source_metadata"]["selected_pair_indices"] == [101, 501]
    assert portfolio_row["source_metadata"]["materializer_resolutions"] == both[
        "materializer_resolutions"
    ]
    assert portfolio_row["source_metadata"]["receiver_contracts"] == [
        DQS1_RECEIVER_CONTRACT_ID
    ]
    assert portfolio_row["source_metadata"]["cooperative_receiver_required"] is True
    assert {unit["unit_id"] for unit in portfolio_row["source_metadata"]["source_units"]} == {
        "pair0320",
        "pair0371",
    }
    assert portfolio_row["ready_for_exact_eval_dispatch"] is False


def test_compile_dqs1_byte_shaving_plan_suggests_registered_byte_range_contract(
    tmp_path: Path,
) -> None:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "byte_range_entropy_suggestion_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_byte_range_entropy_suggestion_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "zip_member_range_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 777,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operation_families": ["entropy_recode"],
            },
        ],
        **_false_authority(),
    }
    plan = build_byte_shaving_campaign_plan(surface, max_k=1)

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    entropy_backlog = next(
        row
        for row in compiled["materializer_backlog"]["rows"]
        if row["unit_kind"] == "byte_range"
        and row["operation_family"] == "entropy_recode"
    )
    assert entropy_backlog["gap_class"] == "target_kind_required"
    assert entropy_backlog["receiver_contract_status"] == (
        "receiver_target_contract_required"
    )
    assert entropy_backlog["suggested_materializers"][0]["materializer_id"] == (
        BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER
    )
    assert entropy_backlog["suggested_materializers"][0]["target_kind"] == (
        BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND
    )
    assert entropy_backlog["suggested_materializers"][0]["executable"] is False


def test_compile_dqs1_byte_shaving_plan_classifies_byte_range_entropy_contract_gap(
    tmp_path: Path,
) -> None:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "byte_range_entropy_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_byte_range_entropy_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "zip_member_range_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 777,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operations": [
                    {
                        "operation_id": "entropy_recode_zip_member_range_a",
                        "operation_family": "entropy_recode",
                        "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
                    }
                ],
            },
        ],
        **_false_authority(),
    }
    plan = build_byte_shaving_campaign_plan(surface, max_k=1)

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    assert compiled["executable_row_count"] == 0
    assert compiled["queueable_row_count"] == 0
    assert all(
        f"unsupported_materializer_target:{BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND}"
        not in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert any(
        f"materializer_not_executable:{BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER}"
        in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    backlog_row = next(
        row
        for row in compiled["materializer_backlog"]["rows"]
        if row["unit_kind"] == "byte_range"
        and row["operation_family"] == "entropy_recode"
    )
    assert backlog_row["gap_class"] == "adapter_not_executable"
    assert backlog_row["target_kind"] == BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND
    assert backlog_row["materializer_id"] == BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER
    assert (
        backlog_row["receiver_contract_id"]
        == BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID
    )
    assert (
        backlog_row["receiver_contract_kind"]
        == BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND
    )
    assert (
        backlog_row["receiver_contract_status"]
        == "receiver_contract_registered_but_adapter_not_executable"
    )
    assert backlog_row["cooperative_receiver_required"] is True
    assert backlog_row["materialization_resource_kind"] == "local_cpu"
    assert backlog_row["suggested_materializers"][0]["materializer_id"] == (
        BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER
    )
    assert "zip_member_range_a" in backlog_row["source_unit_ids"]
    assert backlog_row["blocked_resolution_count"] == 1
    assert backlog_row["selected_operation_count"] == 1
    assert backlog_row["blocker_counts"] == {
        f"materializer_not_executable:{BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER}": 1
    }
    work_row = next(
        row
        for row in compiled["materializer_work_queue"]["rows"]
        if row["unit_kind"] == "byte_range"
        and row["operation_family"] == "entropy_recode"
    )
    assert work_row["tool"] is None
    assert work_row["executable"] is False
    assert work_row["target_kind"] == BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND
    assert any(
        blocker.startswith("materializer_context_missing:")
        for blocker in work_row["materialization_blockers"]
    )


def test_compile_dqs1_byte_shaving_plan_suggests_byte_range_entropy_target_kind(
    tmp_path: Path,
) -> None:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "byte_range_entropy_suggestion_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_byte_range_entropy_suggestion_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "zip_member_range_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 777,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operation_families": ["entropy_recode"],
            },
        ],
        **_false_authority(),
    }
    plan = build_byte_shaving_campaign_plan(surface, max_k=1)

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    backlog_row = next(
        row
        for row in compiled["materializer_backlog"]["rows"]
        if row["unit_kind"] == "byte_range"
        and row["operation_family"] == "entropy_recode"
    )
    assert backlog_row["gap_class"] == "target_kind_required"
    assert backlog_row["receiver_contract_status"] == "receiver_target_contract_required"
    assert backlog_row["suggested_materializer_count"] == 1
    assert backlog_row["suggested_materializers"] == [
        {
            "description": (
                "Fail-closed contract for byte-range entropy recode work; requires "
                "archive-member mapping and runtime-consumption proof before queue execution."
            ),
            "executable": False,
            "cooperative_receiver_required": True,
            "materializer_id": BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
            "materialization_resource_kind": "local_cpu",
            "implementation_module": (
                "tac.optimization.byte_range_entropy_recode_materializer"
            ),
            "plan_function": "build_byte_range_entropy_recode_plan",
            "materialize_function": "materialize_byte_range_entropy_recode_candidate",
            "receiver_proof_function": (
                "build_byte_range_entropy_recode_receiver_proof"
            ),
            "receiver_verify_function": (
                "verify_byte_range_entropy_recode_receiver_contract"
            ),
            "receiver_contract_id": BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND,
            "required_context_fields": [
                "archive_member_name",
                "archive_byte_range",
                "runtime_consumption_proof",
            ],
            "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
        }
    ]


def test_materializer_work_queue_builds_byte_range_chain_command(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    output_dir = tmp_path / "chain_out"

    queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            backlog_row["backlog_key"]: {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json", "beam_b.json"],
                "source_runtime_dir": "runtime",
                "output_dir": str(output_dir),
                "source_archive": "archive.zip",
                "member_name": "0.bin",
                "min_free_bytes": 123,
                "fail_if_receiver_blocked": True,
            }
        },
        source_plan_path="plan.json",
    )

    assert queue["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert queue["executable_row_count"] == 1
    row = queue["rows"][0]
    assert row["executable"] is True
    assert row["tool"] == "tools/run_byte_range_entropy_recode_chain.py"
    assert row["command"][:4] == [
        ".venv/bin/python",
        "tools/run_byte_range_entropy_recode_chain.py",
        "--schema-manifest",
        "schema.json",
    ]
    assert row["command"].count("--beam-probe-report") == 2
    assert "--fail-if-receiver-blocked" in row["command"]
    assert row["postconditions"] == [
        {
            "type": "json_equals",
            "path": str(output_dir / CHAIN_MANIFEST_NAME),
            "key": "schema",
            "equals": CHAIN_SCHEMA,
        }
    ]
    manifest = output_dir / CHAIN_MANIFEST_NAME
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema": CHAIN_SCHEMA}), encoding="utf-8")
    assert _condition_passes(row["postconditions"][0], repo_root=Path("/")) is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_materializer_execution_queue_runs_executable_work_rows(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    output_dir = tmp_path / "chain_out"
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            backlog_row["backlog_key"]: {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime",
                "output_dir": str(output_dir),
                "source_archive": "archive.zip",
            }
        },
        source_plan_path="plan.json",
    )

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="materializer_exec_fixture",
        repo_root=tmp_path,
        lane_id="lane_materializer_exec_fixture",
        source_work_queue_path=tmp_path / "work_queue.json",
        local_cpu_concurrency=3,
        step_timeout_seconds=900,
    )

    assert execution_queue["schema"] == "experiment_queue.v1"
    assert execution_queue["queue_id"] == "materializer_exec_fixture"
    assert execution_queue["controls"]["max_concurrency"] == {"local_cpu": 3}
    assert len(execution_queue["experiments"]) == 1
    experiment = execution_queue["experiments"][0]
    assert experiment["lane_id"] == "lane_materializer_exec_fixture"
    assert experiment["metadata"]["schema"] == MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA
    assert experiment["metadata"]["score_claim"] is False
    assert experiment["metadata"]["promotion_eligible"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    step = experiment["steps"][0]
    assert step["id"] == MATERIALIZER_EXECUTION_STEP_ID
    assert step["resources"]["kind"] == "local_cpu"
    assert step["timeout_seconds"] == 900
    assert step["telemetry"]["artifact_paths"] == [str(output_dir)]
    assert step["telemetry"]["recursive"] is True
    assert step["command"][:2] == [
        ".venv/bin/python",
        "tools/run_byte_range_entropy_recode_chain.py",
    ]
    assert step["postconditions"] == [
        {
            "type": "json_equals",
            "path": str(output_dir / CHAIN_MANIFEST_NAME),
            "key": "schema",
            "equals": CHAIN_SCHEMA,
        }
    ]


def test_materializer_work_queue_matches_context_by_suggested_target_kind(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    output_dir = tmp_path / "chain_out"

    queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND: {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam.json"],
                "source_runtime_dir": "runtime",
                "output_dir": str(output_dir),
            }
        },
        source_plan_path="plan.json",
    )

    assert queue["executable_row_count"] == 1
    row = queue["rows"][0]
    assert row["executable"] is True
    assert row["target_kind"] == BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND
    assert row["materialization_blockers"] == []


def test_materializer_context_payload_maps_rows_to_multiple_lookup_keys(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "chain_out"
    contexts = materializer_contexts_from_payload(
        {
            "schema": MATERIALIZER_CONTEXTS_SCHEMA,
            "rows": [
                {
                    "backlog_key": "backlog_a",
                    "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
                    "source_unit_ids": ["zip_member_range_a"],
                    "context": {
                        "schema_manifest": "schema.json",
                        "beam_probe_reports": ["beam.json"],
                        "source_runtime_dir": "runtime",
                        "output_dir": str(output_dir),
                    },
                }
            ],
        }
    )

    assert set(contexts) == {
        BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
        "backlog_a",
        "zip_member_range_a",
    }
    assert contexts["zip_member_range_a"]["output_dir"] == str(output_dir)


def test_materializer_work_queue_blocks_ambiguous_multi_context_backlog_row(
    tmp_path: Path,
) -> None:
    backlog = {
        "schema": MATERIALIZER_BACKLOG_SCHEMA,
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "byte_range_entropy_shared",
                "gap_class": "adapter_not_executable",
                "unit_kind": "byte_range",
                "operation_family": "entropy_recode",
                "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
                "materializer_id": BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
                "materialization_resource_kind": "local_cpu",
                "source_unit_ids": ["unit_a", "unit_b"],
                "source_selection_ids": ["selection_a", "selection_b"],
                "candidate_saved_bytes_sum": 64,
                "expected_score_gain_sum": 0.0,
                **_false_authority(),
            }
        ],
        **_false_authority(),
    }
    queue = build_materializer_work_queue(
        backlog,
        repo_root=tmp_path,
        contexts={
            "unit_a": {
                "schema_manifest": "schema_a.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime_a",
                "output_dir": str(tmp_path / "out_a"),
            },
            "unit_b": {
                "schema_manifest": "schema_b.json",
                "beam_probe_reports": ["beam_b.json"],
                "source_runtime_dir": "runtime_b",
                "output_dir": str(tmp_path / "out_b"),
            },
        },
    )

    row = queue["rows"][0]
    assert row["executable"] is False
    assert any(
        blocker.startswith("materializer_context_ambiguous:unit_a,unit_b")
        for blocker in row["materialization_blockers"]
    )
    assert row["source_unit_ids"] == ["unit_a", "unit_b"]


def test_materializer_context_payload_rejects_truthy_authority() -> None:
    with pytest.raises(ExperimentQueueError, match="score_claim"):
        materializer_contexts_from_payload(
            {
                "schema": MATERIALIZER_CONTEXTS_SCHEMA,
                "contexts": {
                    "zip_member_range_a": {
                        "schema_manifest": "schema.json",
                        "beam_probe_reports": ["beam.json"],
                        "source_runtime_dir": "runtime",
                        "output_dir": "out",
                        "score_claim": True,
                    }
                },
            }
        )


def test_compile_dqs1_byte_shaving_plan_blocks_drop_pair_on_non_pair_unit(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    for unit in plan["ranked_units"]:
        if unit["unit_id"] == "pair0371":
            unit["unit_kind"] = "byte_range"

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
    )

    assert any(
        f"materializer_unit_kind_mismatch:{DQS1_DROP_PAIR_MATERIALIZER}:"
        "pair0371:byte_range:expected_pair" in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert all(
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]}
        for row in compiled["executable_rows"]
    )


def test_compile_dqs1_byte_shaving_plan_blocks_selected_unit_blockers(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    for unit in plan["ranked_units"]:
        if unit["unit_id"] == "pair0371":
            unit["blockers"] = ["requires_full_frame_parity"]

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
        allow_partial_materialization=True,
        partial_materialization_rationale="unit-test only",
    )

    assert any(
        "selected_unit_blocker:pair0371:requires_full_frame_parity"
        in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    blocker_backlog = next(
        row
        for row in compiled["materializer_backlog"]["rows"]
        if row["gap_class"] == "source_unit_blocked"
        and row["unit_kind"] == "pair"
        and row["operation_family"] == "drop_pair"
    )
    assert blocker_backlog["materializer_id"] == DQS1_DROP_PAIR_MATERIALIZER
    assert blocker_backlog["target_kind"] == DQS1_PAIRSET_TARGET_KIND
    assert blocker_backlog["receiver_contract_id"] == DQS1_RECEIVER_CONTRACT_ID
    assert (
        blocker_backlog["receiver_contract_status"]
        == "receiver_contract_registered_but_source_blocked"
    )
    expected_blocker_count = sum(
        "selected_unit_blocker:pair0371:requires_full_frame_parity"
        in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert blocker_backlog["blocked_row_count"] == expected_blocker_count
    assert blocker_backlog["blocker_counts"] == {
        "selected_unit_blocker:pair0371:requires_full_frame_parity": expected_blocker_count
    }
    assert all(
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]}
        for row in compiled["executable_rows"]
    )


def test_compile_dqs1_byte_shaving_plan_blocks_mixed_materialized_and_unmaterialized_ops(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
    )

    mixed = [
        row
        for row in compiled["blocked_rows"]
        if "byte_null_run_a" in row["selected_unit_ids"]
        and {"pair0320", "pair0371"} & set(row["selected_unit_ids"])
    ]
    assert mixed
    assert all(row["executable"] is False for row in mixed)
    assert compiled["queueable_row_count"] == 0
    assert compiled["action_summary"]["top_operator_actions"] == []
    assert "partial_materialization_requires_explicit_allow" in compiled[
        "partial_materialization_blockers"
    ]
    assert any(
        "materializer_target_kind_required:byte_range:null_remove_or_seed"
        in row["materialization_blockers"]
        for row in mixed
    )
    assert all(
        "byte_null_run_a" not in {
            unit["unit_id"]
            for unit in row["source_metadata"]["source_units"]
        }
        for row in compiled["portfolio"]["operator_action_rows"]
    )


def test_compile_dqs1_byte_shaving_plan_requires_partial_materialization_rationale(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExperimentQueueError, match="partial_materialization_rationale"):
        compile_dqs1_byte_shaving_campaign(
            _pair_drop_plan(),
            repo_root=tmp_path,
            base_pair_indices=[101, 320, 371, 501],
            allow_partial_materialization=True,
        )


def test_compile_dqs1_byte_shaving_plan_blocks_explicit_unknown_materializer(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    for ladder in ("combination_ladder", "sweep_ladder"):
        for row in plan[ladder]:
            for operation in row["selected_operations"]:
                if operation["unit_id"] == "pair0371":
                    operation["materializer"] = "experimental_unregistered_materializer"

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
    )

    assert any(
        "materializer_not_registered:experimental_unregistered_materializer"
        in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert all(
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]}
        for row in compiled["executable_rows"]
    )


def test_compile_dqs1_byte_shaving_plan_blocks_spoofed_registered_materializer(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    for unit in plan["ranked_units"]:
        if unit["unit_id"] == "pair0371":
            unit["unit_kind"] = "byte_range"
    for ladder in ("combination_ladder", "sweep_ladder"):
        for row in plan[ladder]:
            for operation in row["selected_operations"]:
                if operation["unit_id"] == "pair0371":
                    operation["materializer"] = DQS1_DROP_PAIR_MATERIALIZER

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
    )

    assert any(
        f"materializer_unit_kind_mismatch:{DQS1_DROP_PAIR_MATERIALIZER}:"
        "pair0371:byte_range:expected_pair" in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert all(
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]}
        for row in compiled["executable_rows"]
    )


def test_byte_shaving_campaign_queue_cli_writes_dqs1_queue(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    materialization = tmp_path / "materialization.json"
    portfolio = tmp_path / "portfolio.json"
    summary = tmp_path / "action_summary.json"
    backlog = tmp_path / "materializer_backlog.json"
    queue = tmp_path / "queue.json"
    plan_path.write_text(json.dumps(_pair_drop_plan()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--plan",
            str(plan_path),
            "--materialization-out",
            str(materialization),
            "--portfolio-out",
            str(portfolio),
            "--action-summary-out",
            str(summary),
            "--materializer-backlog-out",
            str(backlog),
            "--queue-out",
            str(queue),
            "--repo-root",
            str(tmp_path),
            "--base-pair-indices",
            "101,320,371,501",
            "--candidate-limit",
            "4",
            "--queue-candidate-limit",
            "2",
            "--allow-partial-materialization",
            "--partial-materialization-rationale",
            "unit-test mixed fixture",
            "--results-root",
            "results",
            "--local-cpu-concurrency",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    assert stdout["executable_row_count"] >= 1
    assert stdout["materializer_backlog_out"] == str(backlog)
    assert stdout["materializer_backlog_row_count"] >= 1
    assert stdout["queue"]["experiment_count"] == 2
    assert materialization.is_file()
    assert portfolio.is_file()
    assert summary.is_file()
    assert backlog.is_file()
    assert (
        json.loads(backlog.read_text(encoding="utf-8"))["schema"]
        == MATERIALIZER_BACKLOG_SCHEMA
    )
    loaded = load_queue_definition(queue)
    assert loaded["controls"]["max_concurrency"]["local_cpu"] == 2
    assert len(loaded["experiments"]) == 2
    assert all(
        experiment["id"].startswith("pairset_byte_shave_")
        for experiment in loaded["experiments"]
    )
    assert all(
        experiment["metadata"]["source_metadata"]["materializer_registry_schema"]
        == "byte_shaving_materializer_registry.v1"
        for experiment in loaded["experiments"]
    )


def test_byte_shaving_campaign_queue_cli_loads_materializer_contexts(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    contexts_path = tmp_path / "contexts.json"
    materialization = tmp_path / "materialization.json"
    portfolio = tmp_path / "portfolio.json"
    summary = tmp_path / "action_summary.json"
    work_queue = tmp_path / "materializer_work_queue.json"
    execution_queue = tmp_path / "materializer_execution_queue.json"
    output_dir = tmp_path / "chain_out"
    plan_path.write_text(json.dumps(_byte_range_entropy_plan()), encoding="utf-8")
    contexts_path.write_text(
        json.dumps(
            {
                "schema": MATERIALIZER_CONTEXTS_SCHEMA,
                "contexts": {
                    "zip_member_range_a": {
                        "schema_manifest": "schema.json",
                        "beam_probe_reports": ["beam_a.json", "beam_b.json"],
                        "source_runtime_dir": "runtime",
                        "output_dir": str(output_dir),
                        "source_archive": "archive.zip",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--plan",
            str(plan_path),
            "--materializer-contexts",
            str(contexts_path),
            "--materialization-out",
            str(materialization),
            "--portfolio-out",
            str(portfolio),
            "--action-summary-out",
            str(summary),
            "--materializer-work-queue-out",
            str(work_queue),
            "--materializer-execution-queue-out",
            str(execution_queue),
            "--materializer-execution-queue-id",
            "materializer_exec_fixture",
            "--materializer-execution-lane-id",
            "lane_materializer_exec_fixture",
            "--materializer-execution-timeout-seconds",
            "600",
            "--repo-root",
            str(tmp_path),
            "--candidate-limit",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    assert stdout["materializer_contexts"] == str(contexts_path)
    assert stdout["materializer_work_queue_out"] == str(work_queue)
    assert stdout["materializer_execution_queue"]["queue_out"] == str(execution_queue)
    assert stdout["materializer_execution_queue"]["experiment_count"] == 1
    assert stdout["materializer_work_queue_row_count"] == 1
    payload = json.loads(work_queue.read_text(encoding="utf-8"))
    assert payload["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert payload["executable_row_count"] == 1
    row = payload["rows"][0]
    assert row["executable"] is True
    assert row["command"].count("--beam-probe-report") == 2
    assert row["command"][-2:] == ["--source-archive", "archive.zip"]
    assert row["ready_for_exact_eval_dispatch"] is False
    loaded_execution_queue = load_queue_definition(execution_queue)
    assert loaded_execution_queue["queue_id"] == "materializer_exec_fixture"
    experiment = loaded_execution_queue["experiments"][0]
    assert experiment["lane_id"] == "lane_materializer_exec_fixture"
    assert experiment["metadata"]["schema"] == MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA
    step = experiment["steps"][0]
    assert step["id"] == MATERIALIZER_EXECUTION_STEP_ID
    assert step["timeout_seconds"] == 600
    assert step["telemetry"]["artifact_paths"] == [str(output_dir)]
    assert step["postconditions"][0]["type"] == "json_equals"
