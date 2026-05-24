# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import pytest

from comma_lab.scheduler.byte_shaving_campaign_queue import (
    MATERIALIZATION_SCHEMA,
    MATERIALIZER_BACKLOG_SCHEMA,
    MATERIALIZER_CONTEXTS_SCHEMA,
    MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA,
    MATERIALIZER_EXECUTION_STEP_ID,
    MATERIALIZER_HARVEST_STEP_ID,
    MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID,
    MATERIALIZER_WORK_QUEUE_SCHEMA,
    build_materializer_execution_queue,
    build_materializer_work_queue,
    compile_dqs1_byte_shaving_campaign,
    lower_packetir_operation_set_to_backlog_rows,
    materializer_contexts_from_payload,
)
from comma_lab.scheduler.byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER,
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
    ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
    ARCHIVE_SECTION_REORDER_MATERIALIZER,
    ARCHIVE_SECTION_REORDER_TARGET_KIND,
    BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
    BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID,
    BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND,
    BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
    DQS1_DROP_PAIR_MATERIALIZER,
    DQS1_PAIRSET_TARGET_KIND,
    DQS1_RECEIVER_CONTRACT_ID,
    DQS1_RECEIVER_CONTRACT_KIND,
    INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
    INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
    INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_ID,
    INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_KIND,
    INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
    INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER,
    INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_ID,
    INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_KIND,
    INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
    INVERSE_SCORER_CELL_MATERIALIZER,
    INVERSE_SCORER_CELL_RECEIVER_CONTRACT_ID,
    INVERSE_SCORER_CELL_RECEIVER_CONTRACT_KIND,
    INVERSE_SCORER_CELL_TARGET_KIND,
    PACKET_MEMBER_MERGE_MATERIALIZER,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_REORDER_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    TENSOR_FACTORIZE_MATERIALIZER,
    TENSOR_FACTORIZE_TARGET_KIND,
    TENSOR_PRUNE_MATERIALIZER,
    TENSOR_PRUNE_TARGET_KIND,
    TENSOR_QUANTIZE_MATERIALIZER,
    TENSOR_QUANTIZE_TARGET_KIND,
    TENSOR_SHARED_CODEBOOK_TARGET_KIND,
    registry_manifest,
    resolve_materializer,
    suggest_materializer_adapters,
)
from comma_lab.scheduler.dqs1_local_first_queue import DEFAULT_SCHEDULER_PREFLIGHT_EXPERIMENT_ID
from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    _condition_passes,
    load_queue_definition,
)
from comma_lab.scheduler.staircase_dag import (
    STORAGE_PREFLIGHT_DEPENDENCY_SCHEMA,
    build_staircase_dag_from_experiment_queue,
    plan_staircase_dispatch,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME,
    CHAIN_SCHEMA,
)
from tac.optimization.byte_shaving_campaign import (
    SIGNAL_SURFACE_SCHEMA,
    build_byte_shaving_campaign_plan,
    build_signal_surface_from_inverse_action_functional,
)
from tac.optimization.family_agnostic_materializers import (
    ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
    PACKET_MEMBER_RECOMPRESS_SCHEMA,
    TENSOR_FACTORIZE_SCHEMA,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_MANIFEST_NAME as INVERSE_CELL_CHAIN_MANIFEST_NAME,
)
from tac.packet_compiler.deterministic_compiler import PACKET_IR_OPERATION_SET_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_byte_shaving_campaign_queue.py"


def _write_single_member_zip(path: Path, *, payload: bytes = b"base-payload") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("x", payload)


def _load_queue_builder_tool_module():
    spec = importlib.util.spec_from_file_location(
        "build_byte_shaving_campaign_queue_tool",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def test_build_queue_tool_auto_local_cpu_concurrency_resolves_machine_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_queue_builder_tool_module()

    assert module._auto_local_cpu_concurrency(cpu_count=12) == 12
    assert module._auto_local_cpu_concurrency(cpu_count=0) == 1

    monkeypatch.setattr(module.os, "cpu_count", lambda: 9)
    assert module._parse_local_cpu_concurrency("auto") == 9
    assert module._parse_local_cpu_concurrency("7") == 7
    with pytest.raises(SystemExit):
        module._parse_local_cpu_concurrency("0")
    with pytest.raises(SystemExit):
        module._parse_local_cpu_concurrency("tiny")


def _schema_postcondition(path: Path, schema: str) -> dict[str, object]:
    return {
        "type": "json_equals",
        "path": str(path),
        "key": "schema",
        "equals": schema,
    }


def _assert_typed_postconditions(
    postconditions: list[dict[str, object]],
    *,
    path: Path,
    schema: str,
    chain: bool = False,
) -> None:
    assert postconditions[0] == _schema_postcondition(path, schema)
    assert any(
        condition.get("type") == "json_completion_contract"
        and condition.get("path") == str(path)
        and condition.get("required_equals", {}).get("schema") == schema
        for condition in postconditions
    )
    if chain:
        assert any(
            condition.get("type") == "materializer_chain_complete"
            and condition.get("path") == str(path)
            and condition.get("schema") == schema
            for condition in postconditions
        )


def _write_artifact(path: Path, payload: bytes = b"artifact") -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_constant_inflate_runtime(path: Path) -> None:
    path.mkdir(parents=True)
    inflate = path / "inflate.sh"
    inflate.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'data_dir="$1"',
                'out_dir="$2"',
                'file_list="$3"',
                'test -f "$data_dir/x"',
                'test -s "$file_list"',
                'mkdir -p "$out_dir/frames"',
                'printf frame-bytes > "$out_dir/frames/000000.raw"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    inflate.chmod(0o755)


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


def _archive_section_entropy_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "archive_section_entropy_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_archive_section_entropy_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "pr106_decoder_packed_brotli",
                "unit_kind": "archive_section",
                "candidate_saved_bytes": 321,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operations": [
                    {
                        "operation_id": "recode_pr106_decoder_packed_brotli",
                        "operation_family": "section_entropy_recode",
                        "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                    }
                ],
            },
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=1)


def _archive_section_entropy_pair_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "archive_section_entropy_pair_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_archive_section_entropy_pair_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "pr106_decoder_packed_brotli",
                "unit_kind": "archive_section",
                "candidate_saved_bytes": 321,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operations": [
                    {
                        "operation_id": "recode_pr106_decoder_packed_brotli",
                        "operation_family": "section_entropy_recode",
                        "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                    }
                ],
            },
            {
                "unit_id": "pr106_latent_packed_brotli",
                "unit_kind": "archive_section",
                "candidate_saved_bytes": 123,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.7,
                "operations": [
                    {
                        "operation_id": "recode_pr106_latent_packed_brotli",
                        "operation_family": "section_entropy_recode",
                        "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                    }
                ],
            },
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=2)


def _inverse_surface_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "inverse_surface_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_inverse_surface_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "inverse_surface_pair_0007",
                "unit_kind": "scorer_inverse_surface_cell",
                "candidate_saved_bytes": 32,
                "predicted_quality_score_delta": -0.0001,
                "confidence": 0.6,
                "operations": [
                    {
                        "operation_id": "probe_inverse_surface_pair_0007",
                        "operation_family": "probe_inverse_scorer_surface_cell",
                        "target_kind": INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
                    }
                ],
                "blockers": [
                    "inverse_surface_unit_is_planning_only",
                    "requires_materializer_before_candidate_archive",
                    "requires_exact_auth_eval_before_score_claim",
                ],
            }
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=1)


def _inverse_cell_candidate_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "inverse_cell_candidate_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_inverse_cell_candidate_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "inverse_action_pair_0007",
                "unit_kind": "scorer_inverse_surface_cell",
                "candidate_saved_bytes": 0,
                "predicted_quality_score_delta": -0.0001,
                "confidence": 0.6,
                "operations": [
                    {
                        "operation_id": "materialize_inverse_surface_pair_0007",
                        "operation_family": "materialize_inverse_scorer_cell_candidate",
                        "target_kind": INVERSE_SCORER_CELL_TARGET_KIND,
                    }
                ],
                "blockers": [
                    "inverse_action_unit_is_planning_only",
                    "requires_inverse_scorer_cell_materializer",
                    "requires_exact_auth_eval_before_score_claim",
                ],
            }
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=1)


def test_byte_shaving_materializer_registry_exposes_dqs1_and_byte_range_contracts() -> None:
    manifest = registry_manifest()

    assert manifest["schema"] == "byte_shaving_materializer_registry.v1"
    assert manifest["known_target_kinds"] == [
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
        ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
        ARCHIVE_SECTION_REORDER_TARGET_KIND,
        BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
        DQS1_PAIRSET_TARGET_KIND,
        INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
        INVERSE_SCORER_CELL_TARGET_KIND,
        INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
        PACKET_MEMBER_MERGE_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_REORDER_TARGET_KIND,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        TENSOR_FACTORIZE_TARGET_KIND,
        TENSOR_PRUNE_TARGET_KIND,
        TENSOR_QUANTIZE_TARGET_KIND,
        TENSOR_SHARED_CODEBOOK_TARGET_KIND,
    ]
    adapters = {row["materializer_id"]: row for row in manifest["adapters"]}
    assert adapters[DQS1_DROP_PAIR_MATERIALIZER] == {
        "description": "Compile pair-unit drop operations into DQS1 pairset local-first queue rows.",
        "executable": True,
        "emits_candidate_archive": True,
        "planning_only": False,
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
        "emits_candidate_archive": True,
        "planning_only": False,
        "cooperative_receiver_required": True,
        "materializer_id": BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
        "materialization_resource_kind": "local_cpu",
        "implementation_module": ("tac.optimization.byte_range_entropy_recode_materializer"),
        "plan_function": "build_byte_range_entropy_recode_plan",
        "materialize_function": "materialize_byte_range_entropy_recode_candidate",
        "receiver_proof_function": "build_byte_range_entropy_recode_receiver_proof",
        "receiver_verify_function": ("verify_byte_range_entropy_recode_receiver_contract"),
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
    assert adapters[INVERSE_SCORER_CELL_MATERIALIZER] == {
        "description": (
            "Executable inverse-scorer coordinate-cell proof chain; exact-mode "
            "queue execution requires inflate parity context before it can emit "
            "a harvestable candidate chain."
        ),
        "executable": True,
        "emits_candidate_archive": True,
        "planning_only": False,
        "cooperative_receiver_required": True,
        "materializer_id": INVERSE_SCORER_CELL_MATERIALIZER,
        "materialization_resource_kind": "local_mlx",
        "implementation_module": "tac.optimization.inverse_scorer_cell_materializer",
        "plan_function": "build_inverse_scorer_cell_candidate_plan",
        "materialize_function": "materialize_inverse_scorer_cell_candidate",
        "receiver_proof_function": "build_inverse_scorer_cell_receiver_proof",
        "receiver_verify_function": "verify_inverse_scorer_cell_receiver_contract",
        "operation_family": "materialize_inverse_scorer_cell_candidate",
        "receiver_contract_id": INVERSE_SCORER_CELL_RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": INVERSE_SCORER_CELL_RECEIVER_CONTRACT_KIND,
        "required_context_fields": [
            "raw_contest_video_digest",
            "candidate_archive_template",
            "inverse_action_functional",
            "output_dir",
            "inflate_runtime_dir_or_source_and_candidate_inflate_output_dirs",
        ],
        "target_kind": INVERSE_SCORER_CELL_TARGET_KIND,
        "unit_kind": "scorer_inverse_surface_cell",
    }
    assert adapters[INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER] == {
        "description": (
            "Compile inverse-scorer cells into a local planning-only discrete "
            "action functional artifact. This proof-chain probe is not a "
            "candidate archive materializer."
        ),
        "executable": True,
        "emits_candidate_archive": False,
        "planning_only": True,
        "cooperative_receiver_required": False,
        "materializer_id": INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER,
        "materialization_resource_kind": "local_cpu",
        "implementation_module": "comma_lab.scheduler.byte_shaving_campaign_queue",
        "plan_function": "build_inverse_steganalysis_action_functional",
        "materialize_function": "",
        "receiver_proof_function": "",
        "receiver_verify_function": "",
        "operation_family": "probe_inverse_scorer_surface_cell",
        "receiver_contract_id": INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": (INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_KIND),
        "required_context_fields": [
            "output",
            "inverse_action_source_surface",
        ],
        "target_kind": INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
        "unit_kind": "scorer_inverse_surface_cell",
    }
    assert adapters[INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER] == {
        "description": (
            "Fail-closed contract for promoting bare inverse-action cells into "
            "portfolio-level archive/runtime operation sets. Bare cells are not "
            "candidate archives until this compiler maps them to a concrete "
            "family materializer with runtime-consumption proof."
        ),
        "executable": False,
        "emits_candidate_archive": False,
        "planning_only": True,
        "cooperative_receiver_required": True,
        "materializer_id": INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
        "materialization_resource_kind": "local_mlx",
        "implementation_module": "",
        "plan_function": "",
        "materialize_function": "",
        "receiver_proof_function": "",
        "receiver_verify_function": "",
        "operation_family": INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
        "receiver_contract_id": INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_KIND,
        "required_context_fields": [
            "candidate_family",
            "archive_grammar",
            "receiver_contract_kind",
            "operation_set_compiler",
            "runtime_consumption_proof",
        ],
        "target_kind": INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
        "unit_kind": "scorer_inverse_surface_cell",
    }
    grammar_registry = manifest["cooperative_receiver_grammar_registry"]
    assert grammar_registry["schema"] == "cooperative_receiver_packet_grammar_registry_hook.v1"
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


def test_byte_shaving_materializer_registry_allows_inverse_action_probe_target_kind() -> None:
    resolved = resolve_materializer(
        operation={
            "unit_id": "inverse_surface_pair_0007",
            "operation_id": "probe_inverse_surface_pair_0007",
            "operation_family": "probe_inverse_scorer_surface_cell",
            "target_kind": INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
        },
        unit={
            "unit_id": "inverse_surface_pair_0007",
            "unit_kind": "scorer_inverse_surface_cell",
        },
    )

    assert resolved.executable is False
    assert resolved.materializer_id == INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER
    assert resolved.target_kind == INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND
    assert resolved.receiver_contract_id == (INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_ID)
    assert resolved.receiver_contract_kind == (INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_KIND)
    assert resolved.cooperative_receiver_required is False
    assert resolved.materialization_resource_kind == "local_cpu"
    assert resolved.blockers == (
        f"planning_only_materializer_not_candidate_archive:{INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER}",
    )


def test_byte_shaving_materializer_registry_blocks_bare_inverse_cell_compiler_gap() -> None:
    resolved = resolve_materializer(
        operation={
            "unit_id": "inverse_surface_pair_0007",
            "operation_id": INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
            "operation_family": INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
            "materializer": INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
            "target_kind": INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
        },
        unit={
            "unit_id": "inverse_surface_pair_0007",
            "unit_kind": "scorer_inverse_surface_cell",
        },
    )

    assert resolved.executable is False
    assert resolved.materializer_id == INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER
    assert resolved.target_kind == INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND
    assert resolved.receiver_contract_id == INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_ID
    assert resolved.receiver_contract_kind == (INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_KIND)
    assert resolved.blockers == (
        f"materializer_not_executable:{INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER}",
        f"planning_only_materializer_not_candidate_archive:{INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER}",
    )


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
    assert resolved.receiver_contract_kind == BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND
    assert resolved.cooperative_receiver_required is True
    assert resolved.materialization_resource_kind == "local_cpu"
    assert resolved.blockers == (f"materializer_not_executable:{BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER}",)
    suggestions = suggest_materializer_adapters(
        unit_kind="byte_range",
        operation_family="entropy_recode",
    )
    assert [adapter.materializer_id for adapter in suggestions] == [BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER]


def test_byte_shaving_materializer_registry_registers_inverse_scorer_exact_chain() -> None:
    resolved = resolve_materializer(
        operation={
            "unit_id": "inverse_surface_pair0007",
            "operation_id": "materialize_inverse_scorer_cell_candidate",
            "operation_family": "materialize_inverse_scorer_cell_candidate",
            "target_kind": INVERSE_SCORER_CELL_TARGET_KIND,
        },
        unit={
            "unit_id": "inverse_surface_pair0007",
            "unit_kind": "scorer_inverse_surface_cell",
        },
    )

    assert resolved.executable is True
    assert resolved.materializer_id == INVERSE_SCORER_CELL_MATERIALIZER
    assert resolved.target_kind == INVERSE_SCORER_CELL_TARGET_KIND
    assert resolved.receiver_contract_id == INVERSE_SCORER_CELL_RECEIVER_CONTRACT_ID
    assert resolved.receiver_contract_kind == INVERSE_SCORER_CELL_RECEIVER_CONTRACT_KIND
    assert resolved.materialization_resource_kind == "local_mlx"
    assert resolved.blockers == ()


def test_materializer_registry_has_family_agnostic_fail_closed_targets() -> None:
    archive_resolved = resolve_materializer(
        operation={
            "unit_id": "hnerv_decoder_section",
            "operation_id": "recode_hnerv_decoder",
            "operation_family": "section_entropy_recode",
            "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        },
        unit={
            "unit_id": "hnerv_decoder_section",
            "unit_kind": "archive_section",
        },
    )
    assert archive_resolved.executable is True
    assert archive_resolved.materializer_id == ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER
    assert archive_resolved.cooperative_receiver_required is True
    assert archive_resolved.materialization_resource_kind == "local_cpu"
    assert archive_resolved.blockers == ()
    assert archive_resolved.adapter is not None
    assert archive_resolved.adapter.implementation_module == "tac.optimization.family_agnostic_materializers"
    assert archive_resolved.adapter.materialize_function == "materialize_archive_section_entropy_recode_candidate"

    executable_cases = [
        (
            {
                "unit_id": "boostnerv_boost_weight_tensor",
                "operation_id": "factorize_boost_weight_tensor",
                "operation_family": "factorize_tensor",
                "target_kind": TENSOR_FACTORIZE_TARGET_KIND,
            },
            {
                "unit_id": "boostnerv_boost_weight_tensor",
                "unit_kind": "tensor",
            },
            TENSOR_FACTORIZE_MATERIALIZER,
            "local_cpu",
        ),
        (
            {
                "unit_id": "non_nerv_payload_member",
                "operation_id": "recompress_payload_member",
                "operation_family": "member_recompress",
                "target_kind": PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
            },
            {
                "unit_id": "non_nerv_payload_member",
                "unit_kind": "packet_member",
            },
            PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
            "local_cpu",
        ),
    ]

    for operation, unit, materializer_id, resource_kind in executable_cases:
        resolved = resolve_materializer(operation=operation, unit=unit)
        assert resolved.executable is True
        assert resolved.materializer_id == materializer_id
        assert resolved.cooperative_receiver_required is True
        assert resolved.materialization_resource_kind == resource_kind
        assert resolved.blockers == ()
        assert resolved.adapter is not None
        assert resolved.adapter.implementation_module == "tac.optimization.family_agnostic_materializers"
        assert resolved.adapter.materialize_function

    fail_closed_cases = [
        (
            {
                "unit_id": "boostnerv_inactive_tensor",
                "operation_id": "prune_boostnerv_inactive_tensor",
                "operation_family": "prune_tensor",
                "target_kind": TENSOR_PRUNE_TARGET_KIND,
            },
            {
                "unit_id": "boostnerv_inactive_tensor",
                "unit_kind": "tensor",
            },
            TENSOR_PRUNE_MATERIALIZER,
            "local_cpu",
        ),
        (
            {
                "unit_id": "hnerv_section_order",
                "operation_id": "reorder_hnerv_sections",
                "operation_family": "section_reorder",
                "target_kind": ARCHIVE_SECTION_REORDER_TARGET_KIND,
            },
            {
                "unit_id": "hnerv_section_order",
                "unit_kind": "archive_section",
            },
            ARCHIVE_SECTION_REORDER_MATERIALIZER,
            "local_cpu",
        ),
        (
            {
                "unit_id": "nerv_latent_tensor",
                "operation_id": "quantize_nerv_latent_tensor",
                "operation_family": "quantize_tensor",
                "target_kind": TENSOR_QUANTIZE_TARGET_KIND,
            },
            {
                "unit_id": "nerv_latent_tensor",
                "unit_kind": "tensor",
            },
            TENSOR_QUANTIZE_MATERIALIZER,
            "local_cpu",
        ),
        (
            {
                "unit_id": "bolton_side_member",
                "operation_id": "merge_bolton_side_member",
                "operation_family": "member_merge",
                "target_kind": PACKET_MEMBER_MERGE_TARGET_KIND,
            },
            {
                "unit_id": "bolton_side_member",
                "unit_kind": "packet_member",
            },
            PACKET_MEMBER_MERGE_MATERIALIZER,
            "local_cpu",
        ),
    ]

    for operation, unit, materializer_id, resource_kind in fail_closed_cases:
        resolved = resolve_materializer(operation=operation, unit=unit)
        assert resolved.executable is False
        assert resolved.materializer_id == materializer_id
        assert resolved.cooperative_receiver_required is True
        assert resolved.materialization_resource_kind == resource_kind
        assert resolved.blockers == (f"materializer_not_executable:{materializer_id}",)
        assert resolved.adapter is not None
        assert resolved.adapter.implementation_module == ""
        assert resolved.adapter.plan_function == ""
        assert resolved.adapter.materialize_function == ""
        assert resolved.adapter.receiver_proof_function == ""
        assert resolved.adapter.receiver_verify_function == ""

    suggestions = suggest_materializer_adapters(
        unit_kind="tensor",
        operation_family="factorize_tensor",
    )
    assert [adapter.materializer_id for adapter in suggestions] == [TENSOR_FACTORIZE_MATERIALIZER]


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
        if row["unit_kind"] == "byte_range" and row["operation_family"] == "null_remove_or_seed"
    )
    assert backlog_row["gap_class"] == "target_kind_required"
    assert backlog_row["receiver_contract_status"] == "receiver_target_contract_required"
    assert backlog_row["receiver_contract_id"] is None
    assert backlog_row["blocked_row_count"] >= 1
    assert backlog_row["candidate_saved_bytes_sum"] > 0
    assert "byte_null_run_a" in backlog_row["source_unit_ids"]
    assert (
        compiled["action_summary"]["materializer_backlog_summary"]["backlog_row_count"]
        == compiled["materializer_backlog"]["backlog_row_count"]
    )
    assert any(
        "materializer_target_kind_required:byte_range:null_remove_or_seed" in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    both = next(row for row in compiled["executable_rows"] if row["dropped_pair_indices"] == [320, 371])
    assert both["selected_pair_indices"] == [101, 501]
    assert both["selection_kind"] == "operation_set"
    assert both["operation_set_id"].startswith("opset_combo_")
    assert both["chosen_operation_sequence"]
    assert len(both["chosen_operation_sequence_sha256"]) == 64
    assert both["chosen_operation_sequence_is_permutation"] is True
    assert both["operation_set_materialization_mode"] == "ordered_dqs1_pairset_sequence"
    assert both["source_row"]["operation_set_id"] == both["operation_set_id"]
    assert both["packet_ir_operation_set"]["schema"] == PACKET_IR_OPERATION_SET_SCHEMA
    assert both["packet_ir_operation_set"]["source_operation_set_id"] == both["operation_set_id"]
    assert both["source_row"]["packet_ir_operation_set"] == both["packet_ir_operation_set"]
    assert both["packet_ir_operation_set"]["score_claim"] is False
    assert both["materialization_blockers"] == []
    assert {unit["unit_id"] for unit in both["source_units"]} == {"pair0320", "pair0371"}
    assert all(
        resolution["materializer_id"] == DQS1_DROP_PAIR_MATERIALIZER for resolution in both["materializer_resolutions"]
    )
    assert all(
        resolution["receiver_contract_id"] == DQS1_RECEIVER_CONTRACT_ID
        for resolution in both["materializer_resolutions"]
    )
    action = next(
        row for row in compiled["action_summary"]["top_operator_actions"] if row["candidate_id"] == both["candidate_id"]
    )
    assert action["operator_next_action"] == "materialize_pairset_archive_and_run_local_controls"
    portfolio_row = next(
        row for row in compiled["portfolio"]["operator_action_rows"] if row["candidate_id"] == both["candidate_id"]
    )
    assert portfolio_row["source_metadata"]["selected_pair_indices"] == [101, 501]
    assert portfolio_row["source_metadata"]["operation_set_id"] == both["operation_set_id"]
    assert portfolio_row["source_metadata"]["packet_ir_operation_set"] == both["packet_ir_operation_set"]
    assert portfolio_row["source_metadata"]["chosen_operation_sequence"] == (both["chosen_operation_sequence"])
    assert (
        portfolio_row["source_metadata"]["chosen_operation_sequence_sha256"]
        == (both["chosen_operation_sequence_sha256"])
    )
    assert portfolio_row["source_metadata"]["operation_set_materialization_mode"] == ("ordered_dqs1_pairset_sequence")
    assert portfolio_row["source_metadata"]["materializer_resolutions"] == both["materializer_resolutions"]
    assert portfolio_row["source_metadata"]["receiver_contracts"] == [DQS1_RECEIVER_CONTRACT_ID]
    assert portfolio_row["source_metadata"]["cooperative_receiver_required"] is True
    assert {unit["unit_id"] for unit in portfolio_row["source_metadata"]["source_units"]} == {
        "pair0320",
        "pair0371",
    }
    assert portfolio_row["ready_for_exact_eval_dispatch"] is False


def test_packet_ir_operation_set_lowers_to_materializer_backlog_rows(
    tmp_path: Path,
) -> None:
    plan = _archive_section_entropy_pair_plan()
    packet_ir = plan["packet_ir_operation_sets"][0]

    rows = lower_packetir_operation_set_to_backlog_rows(packet_ir)

    assert rows
    row = rows[0]
    assert row["source_packet_ir_schema"] == PACKET_IR_OPERATION_SET_SCHEMA
    assert row["source_packet_ir_operation_set_id"] == packet_ir["operation_set_id"]
    assert row["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    assert row["unit_kind"] == "archive_section"
    assert row["operation_family"] == "section_entropy_recode"
    assert row["score_claim"] is False
    assert "packetir_operation_set_requires_materializer_contexts" in row["blocker_counts"]

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
    )
    assert compiled["packet_ir_materializer_backlog_row_count"] >= 1
    assert (
        compiled["packet_ir_materializer_backlog_rows"][0]["source_packet_ir_operation_set_id"]
        == packet_ir["operation_set_id"]
    )
    merged_row = next(
        row
        for row in compiled["materializer_backlog"]["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND and row["unit_kind"] == "archive_section"
    )
    assert merged_row["packet_ir_lowered_row_count"] >= 1
    assert packet_ir["operation_set_id"] in merged_row["source_packet_ir_operation_set_ids"]
    assert "packetir_operation_set_requires_materializer_contexts" in merged_row["packet_ir_blocker_counts"]

    packet_only_plan = copy.deepcopy(plan)
    packet_only_plan["operation_set_ladder"] = []
    packet_only_plan["combination_ladder"] = []
    packet_only_plan["sweep_ladder"] = []
    packet_only = compile_dqs1_byte_shaving_campaign(
        packet_only_plan,
        repo_root=tmp_path,
    )
    assert packet_only["compiled_row_count"] == 0
    assert packet_only["materializer_backlog"]["backlog_row_count"] >= 1
    assert packet_only["materializer_backlog"]["packet_ir_lowered_row_count"] >= 1
    assert packet_only["materializer_work_queue"]["row_count"] >= 1
    assert packet_only["materializer_work_queue"]["blocked_row_count"] >= 1
    work_row = packet_only["materializer_work_queue"]["rows"][0]
    assert packet_ir["operation_set_id"] in work_row["source_packet_ir_operation_set_ids"]
    assert "packetir_operation_set_requires_materializer_contexts" in work_row["packet_ir_blocker_counts"]


def test_operation_set_execution_requires_matching_packet_ir_handoff(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    operation_set = next(
        row for row in plan["operation_set_ladder"] if set(row["selected_unit_ids"]) == {"pair0320", "pair0371"}
    )
    plan["packet_ir_operation_sets"] = []

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
        allow_partial_materialization=True,
        partial_materialization_rationale="unit-test missing packet-ir",
    )

    blocked = next(
        row
        for row in compiled["blocked_rows"]
        if row["selection_kind"] == "operation_set" and row["operation_set_id"] == operation_set["operation_set_id"]
    )
    assert blocked["packet_ir_operation_set"] is None
    assert "operation_set_packet_ir_operation_set_missing" in blocked["materialization_blockers"]
    assert blocked["executable"] is False


def test_operation_set_execution_validates_packet_ir_contract(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    operation_set = next(
        row for row in plan["operation_set_ladder"] if set(row["selected_unit_ids"]) == {"pair0320", "pair0371"}
    )
    packet_ir = next(
        row
        for row in plan["packet_ir_operation_sets"]
        if row["source_operation_set_id"] == operation_set["operation_set_id"]
    )
    packet_ir["required_proofs"] = []
    packet_ir["compiler_contract"]["required_order"] = []

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
        allow_partial_materialization=True,
        partial_materialization_rationale="unit-test packet-ir mismatch",
    )

    blocked = next(
        row
        for row in compiled["blocked_rows"]
        if row["selection_kind"] == "operation_set" and row["operation_set_id"] == operation_set["operation_set_id"]
    )
    assert "operation_set_packet_ir_required_proofs_mismatch" in blocked["materialization_blockers"]
    assert "operation_set_packet_ir_compiler_contract_mismatch:required_order" in blocked["materialization_blockers"]


def test_packet_ir_operation_set_lowering_rejects_authority_and_bad_sequence() -> None:
    plan = _archive_section_entropy_pair_plan()
    packet_ir = copy.deepcopy(plan["packet_ir_operation_sets"][0])
    packet_ir["operations"][0]["score_claim"] = True

    with pytest.raises(ExperimentQueueError, match="score_claim"):
        lower_packetir_operation_set_to_backlog_rows(packet_ir)

    packet_ir = copy.deepcopy(plan["packet_ir_operation_sets"][0])
    packet_ir["chosen_operation_sequence_is_permutation"] = False
    with pytest.raises(ExperimentQueueError, match="not a permutation"):
        lower_packetir_operation_set_to_backlog_rows(packet_ir)


def test_packet_ir_operation_set_lowering_keeps_unknown_target_blocked() -> None:
    plan = _archive_section_entropy_pair_plan()
    packet_ir = copy.deepcopy(plan["packet_ir_operation_sets"][0])
    packet_ir["operations"][0]["target_kind"] = "unknown_packetir_target_v1"

    rows = lower_packetir_operation_set_to_backlog_rows(packet_ir)

    assert any("unsupported_materializer_target:unknown_packetir_target_v1" in row["blocker_counts"] for row in rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in rows)


def test_compile_dqs1_byte_shaving_plan_blocks_operation_set_sequence_mismatch(
    tmp_path: Path,
) -> None:
    plan = copy.deepcopy(_pair_drop_plan())
    operation_set = next(
        row for row in plan["operation_set_ladder"] if set(row["selected_unit_ids"]) == {"pair0320", "pair0371"}
    )
    operation_set["chosen_operation_sequence"] = operation_set["chosen_operation_sequence"][:-1]

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
        allow_partial_materialization=True,
        partial_materialization_rationale="unit-test mixed fixture",
    )

    blocked = next(
        row for row in compiled["blocked_rows"] if row.get("operation_set_id") == operation_set["operation_set_id"]
    )
    assert blocked["chosen_operation_sequence_is_permutation"] is False
    assert "operation_set_sequence_not_permutation_of_selected_operations" in blocked["materialization_blockers"]
    assert blocked["operation_set_materialization_mode"] == ("blocked_or_requires_atomic_materializer")


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
        if row["unit_kind"] == "byte_range" and row["operation_family"] == "entropy_recode"
    )
    assert entropy_backlog["gap_class"] == "target_kind_required"
    assert entropy_backlog["receiver_contract_status"] == ("receiver_target_contract_required")
    assert entropy_backlog["suggested_materializers"][0]["materializer_id"] == (BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER)
    assert entropy_backlog["suggested_materializers"][0]["target_kind"] == (BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND)
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
        f"materializer_not_executable:{BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER}" in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    backlog_row = next(
        row
        for row in compiled["materializer_backlog"]["rows"]
        if row["unit_kind"] == "byte_range" and row["operation_family"] == "entropy_recode"
    )
    assert backlog_row["gap_class"] == "adapter_not_executable"
    assert backlog_row["target_kind"] == BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND
    assert backlog_row["materializer_id"] == BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER
    assert backlog_row["receiver_contract_id"] == BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID
    assert backlog_row["receiver_contract_kind"] == BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND
    assert backlog_row["receiver_contract_status"] == "receiver_contract_registered_but_adapter_not_executable"
    assert backlog_row["cooperative_receiver_required"] is True
    assert backlog_row["materialization_resource_kind"] == "local_cpu"
    assert backlog_row["suggested_materializers"][0]["materializer_id"] == (BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER)
    assert "zip_member_range_a" in backlog_row["source_unit_ids"]
    assert backlog_row["blocked_resolution_count"] == 1
    assert backlog_row["selected_operation_count"] == 1
    assert backlog_row["blocker_counts"] == {f"materializer_not_executable:{BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER}": 1}
    work_row = next(
        row
        for row in compiled["materializer_work_queue"]["rows"]
        if row["unit_kind"] == "byte_range" and row["operation_family"] == "entropy_recode"
    )
    assert work_row["tool"] is None
    assert work_row["executable"] is False
    assert work_row["target_kind"] == BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND
    assert any(blocker.startswith("materializer_context_missing:") for blocker in work_row["materialization_blockers"])


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
        if row["unit_kind"] == "byte_range" and row["operation_family"] == "entropy_recode"
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
            "emits_candidate_archive": True,
            "planning_only": False,
            "cooperative_receiver_required": True,
            "materializer_id": BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
            "materialization_resource_kind": "local_cpu",
            "implementation_module": ("tac.optimization.byte_range_entropy_recode_materializer"),
            "plan_function": "build_byte_range_entropy_recode_plan",
            "materialize_function": "materialize_byte_range_entropy_recode_candidate",
            "receiver_proof_function": ("build_byte_range_entropy_recode_receiver_proof"),
            "receiver_verify_function": ("verify_byte_range_entropy_recode_receiver_contract"),
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
    chain_manifest = output_dir / CHAIN_MANIFEST_NAME
    _assert_typed_postconditions(
        row["postconditions"],
        path=chain_manifest,
        schema=CHAIN_SCHEMA,
        chain=True,
    )
    manifest = output_dir / CHAIN_MANIFEST_NAME
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema": CHAIN_SCHEMA}), encoding="utf-8")
    assert _condition_passes(row["postconditions"][0], repo_root=Path("/")) is True
    assert _condition_passes(row["postconditions"][1], repo_root=Path("/")) is False
    assert _condition_passes(row["postconditions"][2], repo_root=Path("/")) is False

    archive = _write_artifact(output_dir / "candidate_archive.zip", b"candidate")
    candidate_manifest = _write_artifact(output_dir / "candidate_manifest.json")
    receiver_proof = _write_artifact(output_dir / "receiver_proof.json")
    manifest.write_text(
        json.dumps(
            {
                "schema": CHAIN_SCHEMA,
                "source_archive_bytes": archive["bytes"] - 1,
                "candidate_archive": archive,
                "candidate_archive_sha256": archive["sha256"],
                "candidate_archive_bytes": archive["bytes"],
                "serialized_archive_delta": {"status": "realized_cost"},
                "byte_closed_candidate_emitted": True,
                "runtime_adapter_ready": True,
                "receiver_proof_ready": True,
                "receiver_contract_satisfied": True,
                "candidate_runtime_adapter_blocker_cleared": True,
                "readiness_blockers": [],
                "artifacts": {
                    "candidate_manifest": candidate_manifest,
                    "receiver_proof": receiver_proof,
                },
                "chain_steps": [
                    {
                        "step_id": "materialize_candidate",
                        "status": "succeeded",
                        "artifact": candidate_manifest,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    assert _condition_passes(row["postconditions"][1], repo_root=Path("/")) is False
    assert _condition_passes(row["postconditions"][2], repo_root=Path("/")) is False

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_archive_bytes"] = archive["bytes"] + 1
    payload.pop("serialized_archive_delta")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert _condition_passes(row["postconditions"][1], repo_root=Path("/")) is False
    assert _condition_passes(row["postconditions"][2], repo_root=Path("/")) is False

    payload["serialized_archive_delta"] = {"status": "realized_saving"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert _condition_passes(row["postconditions"][1], repo_root=Path("/")) is True
    assert _condition_passes(row["postconditions"][2], repo_root=Path("/")) is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_materializer_work_queue_wraps_archive_section_entropy_recode_adapter(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _archive_section_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    assert compiled["executable_row_count"] == 0
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    assert backlog_row["gap_class"] == "materializer_work_queue_required"
    assert backlog_row["materializer_id"] == ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER
    assert backlog_row["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND

    source_archive = tmp_path / "source.zip"
    section_manifest = tmp_path / "sections.json"
    output_archive = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate.json"
    runtime_proof = manifest.with_name(
        f"{manifest.stem}.runtime_consumption_proof.json"
    )
    queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            backlog_row["backlog_key"]: {
                "archive_path": str(source_archive),
                "section_manifest": str(section_manifest),
                "output_archive": str(output_archive),
                "output_manifest": str(manifest),
                "section_names": [
                    "decoder_packed_brotli",
                    "latents_and_sidecar_brotli",
                ],
                "brotli_quality": [11],
            }
        },
        source_plan_path="plan.json",
    )

    assert queue["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert queue["executable_row_count"] == 1
    row = queue["rows"][0]
    assert row["executable"] is True
    assert row["tool"] == "tools/run_family_agnostic_materializer.py"
    assert row["command"][:6] == [
        ".venv/bin/python",
        "tools/run_family_agnostic_materializer.py",
        "--target-kind",
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        "--archive-path",
        str(source_archive),
    ]
    assert ["--section-manifest", str(section_manifest)] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert row["command"].count("--section-name") == 2
    assert ["--runtime-consumption-proof-out", str(runtime_proof)] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert row["command"][-2:] == ["--brotli-quality", "11"]
    _assert_typed_postconditions(
        row["postconditions"],
        path=manifest,
        schema=ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
    )
    completion_contract = row["postconditions"][1]
    assert completion_contract["required_true"] == ["byte_closed_candidate_emitted"]
    assert completion_contract["required_positive_int"] == [
        "candidate_archive.bytes",
        "candidate_member.bytes",
    ]
    assert completion_contract["required_sha256"] == [
        "candidate_archive.sha256",
        "candidate_member.sha256",
    ]
    assert completion_contract["required_nonempty_unless_true"] == [
        {
            "key": "readiness_blockers",
            "unless_true": "receiver_contract_satisfied",
        }
    ]
    assert row["telemetry"]["artifact_paths"] == [
        str(output_archive),
        str(manifest),
        str(runtime_proof),
    ]
    assert row["telemetry"]["input_artifact_paths"] == [
        str(source_archive),
        str(section_manifest),
    ]
    assert row["telemetry"]["family_agnostic_materializer_contract"]["target_kind"] == (
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    )
    assert row["telemetry"]["pullback_artifact_paths"] == [
        str(output_archive),
        str(manifest),
        str(runtime_proof),
    ]
    assert "archive_section_entropy_recode_requires_same_runtime_inflate_parity" in row["dispatch_blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_archive_section_entropy_recode_fails_closed_without_section_manifest(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _archive_section_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            backlog_row["backlog_key"]: {
                "archive_path": "source.zip",
                "output_archive": str(tmp_path / "candidate.zip"),
            }
        },
        source_plan_path="plan.json",
    )

    row = queue["rows"][0]
    assert row["executable"] is False
    assert row["tool"] is None
    assert "materializer_context_missing:section_manifest" in row["materialization_blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_materializer_work_queue_wraps_packet_member_and_tensor_family_adapters(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    packet_manifest = tmp_path / "packet_members.json"
    tensor_manifest = tmp_path / "tensor_manifest.json"
    factor_contract = tmp_path / "factor_contract.json"
    packet_output = tmp_path / "packet_candidate.zip"
    packet_out_manifest = tmp_path / "packet_candidate.json"
    tensor_output = tmp_path / "tensor_candidate.zip"
    tensor_out_manifest = tmp_path / "tensor_candidate.json"
    backlog = {
        "schema": MATERIALIZER_BACKLOG_SCHEMA,
        "rows": [
            {
                "backlog_key": "packet_member_recompress_fixture",
                "unit_kind": "packet_member",
                "operation_family": "member_recompress",
                "target_kind": PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
                "materializer_id": PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
            },
            {
                "backlog_key": "tensor_factorize_fixture",
                "unit_kind": "tensor",
                "operation_family": "factorize_tensor",
                "target_kind": TENSOR_FACTORIZE_TARGET_KIND,
                "materializer_id": TENSOR_FACTORIZE_MATERIALIZER,
            },
        ],
    }

    queue = build_materializer_work_queue(
        backlog,
        repo_root=tmp_path,
        contexts={
            "packet_member_recompress_fixture": {
                "archive_path": str(archive),
                "packet_member_manifest": str(packet_manifest),
                "member_name": "payload.bin",
                "output_archive": str(packet_output),
                "output_manifest": str(packet_out_manifest),
                "zip_compresslevel": [9],
            },
            "tensor_factorize_fixture": {
                "archive_path": str(archive),
                "tensor_manifest": str(tensor_manifest),
                "factorization_contract": str(factor_contract),
                "output_archive": str(tensor_output),
                "output_manifest": str(tensor_out_manifest),
            },
        },
        source_plan_path="plan.json",
    )

    assert queue["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert queue["executable_row_count"] == 2
    packet_row, tensor_row = queue["rows"]
    packet_runtime_proof = packet_out_manifest.with_name(
        f"{packet_out_manifest.stem}.runtime_consumption_proof.json"
    )
    assert packet_row["tool"] == "tools/run_family_agnostic_materializer.py"
    assert [
        "--target-kind",
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    ] in [packet_row["command"][index : index + 2] for index in range(len(packet_row["command"]) - 1)]
    assert ["--member-name", "payload.bin"] in [
        packet_row["command"][index : index + 2] for index in range(len(packet_row["command"]) - 1)
    ]
    assert ["--runtime-consumption-proof-out", str(packet_runtime_proof)] in [
        packet_row["command"][index : index + 2]
        for index in range(len(packet_row["command"]) - 1)
    ]
    _assert_typed_postconditions(
        packet_row["postconditions"],
        path=packet_out_manifest,
        schema=PACKET_MEMBER_RECOMPRESS_SCHEMA,
    )
    assert packet_row["telemetry"]["input_artifact_paths"] == [
        str(archive),
        str(packet_manifest),
    ]
    assert str(packet_runtime_proof) in packet_row["telemetry"]["artifact_paths"]
    assert str(packet_runtime_proof) in packet_row["telemetry"]["pullback_artifact_paths"]
    assert "packet_member_recompress_requires_runtime_consumption_proof" in (packet_row["dispatch_blockers"])
    assert packet_row["score_claim"] is False
    assert packet_row["ready_for_exact_eval_dispatch"] is False

    tensor_runtime_proof = tensor_out_manifest.with_name(
        f"{tensor_out_manifest.stem}.runtime_consumption_proof.json"
    )
    assert tensor_row["tool"] == "tools/run_family_agnostic_materializer.py"
    assert [
        "--target-kind",
        TENSOR_FACTORIZE_TARGET_KIND,
    ] in [tensor_row["command"][index : index + 2] for index in range(len(tensor_row["command"]) - 1)]
    assert ["--factorization-contract", str(factor_contract)] in [
        tensor_row["command"][index : index + 2] for index in range(len(tensor_row["command"]) - 1)
    ]
    assert ["--runtime-consumption-proof-out", str(tensor_runtime_proof)] in [
        tensor_row["command"][index : index + 2]
        for index in range(len(tensor_row["command"]) - 1)
    ]
    _assert_typed_postconditions(
        tensor_row["postconditions"],
        path=tensor_out_manifest,
        schema=TENSOR_FACTORIZE_SCHEMA,
    )
    assert tensor_row["telemetry"]["input_artifact_paths"] == [
        str(archive),
        str(tensor_manifest),
        str(factor_contract),
    ]
    assert str(tensor_runtime_proof) in tensor_row["telemetry"]["artifact_paths"]
    assert str(tensor_runtime_proof) in tensor_row["telemetry"]["pullback_artifact_paths"]
    assert "tensor_factorize_requires_cooperative_receiver" in tensor_row["dispatch_blockers"]
    assert tensor_row["score_claim"] is False
    assert tensor_row["ready_for_exact_eval_dispatch"] is False


def test_family_agnostic_candidate_postconditions_reject_weak_receiver_manifest(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    packet_manifest = tmp_path / "packet_members.json"
    output = tmp_path / "packet_candidate.zip"
    out_manifest = tmp_path / "packet_candidate.json"
    candidate_archive = _write_artifact(output, b"candidate-archive")
    member_sha = hashlib.sha256(b"payload").hexdigest()
    backlog = {
        "schema": MATERIALIZER_BACKLOG_SCHEMA,
        "rows": [
            {
                "backlog_key": "packet_member_recompress_fixture",
                "unit_kind": "packet_member",
                "operation_family": "member_recompress",
                "target_kind": PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
                "materializer_id": PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
            },
        ],
    }
    queue = build_materializer_work_queue(
        backlog,
        repo_root=tmp_path,
        contexts={
            "packet_member_recompress_fixture": {
                "archive_path": str(archive),
                "packet_member_manifest": str(packet_manifest),
                "member_name": "payload.bin",
                "output_archive": str(output),
                "output_manifest": str(out_manifest),
            },
        },
        source_plan_path="plan.json",
    )
    completion_contract = next(
        condition for condition in queue["rows"][0]["postconditions"] if condition["type"] == "json_completion_contract"
    )

    weak_manifest = {
        "schema": PACKET_MEMBER_RECOMPRESS_SCHEMA,
        "byte_closed_candidate_emitted": True,
        "candidate_archive": candidate_archive,
        "candidate_member": {
            "name": "payload.bin",
            "bytes": len(b"payload"),
            "sha256": member_sha,
        },
        "receiver_verification": {
            "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
            "receiver_contract_satisfied": False,
        },
        "receiver_contract_satisfied": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    out_manifest.write_text(json.dumps(weak_manifest), encoding="utf-8")
    assert not _condition_passes(completion_contract, repo_root=tmp_path)

    weak_manifest["readiness_blockers"] = ["runtime_consumption_proof_missing"]
    out_manifest.write_text(json.dumps(weak_manifest), encoding="utf-8")
    assert _condition_passes(completion_contract, repo_root=tmp_path)


def test_materializer_chain_completion_contract_rejects_schema_only_and_failure(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "chain_manifest.json"
    postconditions = [
        _schema_postcondition(manifest, CHAIN_SCHEMA),
        {
            "type": "json_completion_contract",
            "path": str(manifest),
            "required_equals": {"schema": CHAIN_SCHEMA},
            "required_true": [
                "byte_closed_candidate_emitted",
                "runtime_adapter_ready",
                "receiver_proof_ready",
                "receiver_contract_satisfied",
                "candidate_runtime_adapter_blocker_cleared",
            ],
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
            ],
            "false_or_missing": [
                "ready_for_exact_eval_dispatch",
                "dispatch_attempted",
                "gpu_launched",
            ],
            "required_sha256": ["candidate_archive_sha256"],
            "required_positive_int": ["candidate_archive_bytes"],
            "required_artifact_records": ["candidate_archive"],
            "forbidden_statuses": ["failed"],
        },
        {
            "type": "materializer_chain_complete",
            "path": str(manifest),
            "schema": CHAIN_SCHEMA,
        },
    ]

    manifest.write_text(json.dumps({"schema": CHAIN_SCHEMA}), encoding="utf-8")
    assert _condition_passes(postconditions[0], repo_root=tmp_path) is True
    assert _condition_passes(postconditions[1], repo_root=tmp_path) is False
    assert _condition_passes(postconditions[2], repo_root=tmp_path) is False

    manifest.write_text(
        json.dumps({"schema": CHAIN_SCHEMA, "status": "failed"}),
        encoding="utf-8",
    )
    assert _condition_passes(postconditions[0], repo_root=tmp_path) is True
    assert _condition_passes(postconditions[1], repo_root=tmp_path) is False
    assert _condition_passes(postconditions[2], repo_root=tmp_path) is False

    archive = _write_artifact(tmp_path / "candidate_archive.zip", b"archive-bytes")
    candidate_manifest = _write_artifact(tmp_path / "candidate_manifest.json")
    receiver_proof = _write_artifact(tmp_path / "receiver_proof.json")
    manifest.write_text(
        json.dumps(
            {
                "schema": CHAIN_SCHEMA,
                "candidate_archive": archive,
                "candidate_archive_sha256": archive["sha256"],
                "candidate_archive_bytes": archive["bytes"],
                "byte_closed_candidate_emitted": True,
                "runtime_adapter_ready": True,
                "receiver_proof_ready": True,
                "receiver_contract_satisfied": True,
                "candidate_runtime_adapter_blocker_cleared": True,
                "readiness_blockers": [],
                "artifacts": {
                    "candidate_manifest": candidate_manifest,
                    "receiver_proof": receiver_proof,
                },
                "chain_steps": [
                    {
                        "step_id": "materialize_candidate",
                        "status": "succeeded",
                        "artifact": candidate_manifest,
                    },
                    {
                        "step_id": "build_receiver_proof",
                        "status": "succeeded",
                        "artifact": receiver_proof,
                    },
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    assert _condition_passes(postconditions[1], repo_root=tmp_path) is True
    assert _condition_passes(postconditions[2], repo_root=tmp_path) is True


def test_inverse_surface_cells_compile_to_action_functional_work_queue(
    tmp_path: Path,
) -> None:
    scorer_response = tmp_path / "scorer_response.json"
    byte_shaving_plan = tmp_path / "byte_shaving_plan.json"
    action_output = tmp_path / "inverse_action.json"
    action_md = tmp_path / "inverse_action.md"
    queue_performance = tmp_path / "queue_performance_summary.json"
    runtime_identity = tmp_path / "runtime_identity.json"
    cache_identity = tmp_path / "cache_identity.json"
    candidate_map = tmp_path / "candidate_map.json"
    scorer_response.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "rows": []}),
        encoding="utf-8",
    )
    byte_shaving_plan.write_text(json.dumps(_pair_drop_plan()), encoding="utf-8")
    queue_performance.write_text(
        json.dumps({"schema": "experiment_queue_performance_summary.v1"}),
        encoding="utf-8",
    )
    runtime_identity.write_text(json.dumps({"runtime_tree_sha256": "d" * 64}), encoding="utf-8")
    cache_identity.write_text(json.dumps({"cache_sha256": "e" * 64}), encoding="utf-8")
    candidate_map.write_text(json.dumps({"candidate_a": "candidate_a"}), encoding="utf-8")
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_surface_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    assert compiled["executable_row_count"] == 0
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    assert backlog_row["unit_kind"] == "scorer_inverse_surface_cell"
    assert backlog_row["operation_family"] == "probe_inverse_scorer_surface_cell"
    assert backlog_row["target_kind"] == INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND
    assert backlog_row["materializer_id"] == (INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER)

    missing_context_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
    )
    assert missing_context_queue["executable_row_count"] == 0
    assert any(
        blocker.startswith("materializer_context_missing:")
        for blocker in missing_context_queue["rows"][0]["materialization_blockers"]
    )

    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND: {
                "scorer_response": str(scorer_response),
                "output": str(action_output),
                "md_out": str(action_md),
                "queue_performance_summary": [str(queue_performance)],
                "queue_performance_runtime_identity": str(runtime_identity),
                "queue_performance_cache_identity": str(cache_identity),
                "queue_performance_candidate_map": str(candidate_map),
                "queue_performance_axis": "[local-queue-performance advisory]",
                "total_byte_budget": 64,
                "resource_kind": "local_mlx",
                "inverse_scorer_allow_native_mlx_window_objective": True,
            }
        },
        source_plan_path="plan.json",
    )

    assert work_queue["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert work_queue["executable_row_count"] == 1
    row = work_queue["rows"][0]
    assert row["executable"] is True
    assert row["tool"] == "tools/build_inverse_steganalysis_action_functional.py"
    assert row["command"][:4] == [
        ".venv/bin/python",
        "tools/build_inverse_steganalysis_action_functional.py",
        "--output",
        str(action_output),
    ]
    assert ["--scorer-response", str(scorer_response)] == row["command"][4:6]
    runtime_index = row["command"].index("--queue-performance-runtime-identity")
    cache_index = row["command"].index("--queue-performance-cache-identity")
    candidate_map_index = row["command"].index("--queue-performance-candidate-map")
    summary_index = row["command"].index("--queue-performance-summary")
    assert row["command"][runtime_index + 1] == str(runtime_identity)
    assert row["command"][cache_index + 1] == str(cache_identity)
    assert row["command"][candidate_map_index + 1] == str(candidate_map)
    assert row["command"][summary_index + 1] == str(queue_performance)
    assert "--queue-performance-axis" in row["command"]
    assert "--inverse-scorer-allow-native-mlx-window-objective" in row["command"]
    _assert_typed_postconditions(
        row["postconditions"],
        path=action_output,
        schema="inverse_steganalysis_discrete_action_functional.v1",
    )
    assert row["telemetry"]["artifact_paths"] == [str(action_output), str(action_md)]
    assert row["telemetry"]["input_artifact_paths"] == [
        str(scorer_response),
        str(runtime_identity),
        str(cache_identity),
        str(candidate_map),
        str(queue_performance),
    ]
    assert "inverse_action_functional_is_not_candidate_archive" in row["dispatch_blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    byte_shaving_work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND: {
                "byte_shaving_campaign_plan": str(byte_shaving_plan),
                "output": str(action_output),
            }
        },
        source_plan_path="plan.json",
    )
    byte_shaving_row = byte_shaving_work_queue["rows"][0]
    assert byte_shaving_row["executable"] is True
    assert "--byte-shaving-campaign-plan" in byte_shaving_row["command"]
    assert [
        "--byte-shaving-campaign-plan",
        str(byte_shaving_plan),
    ] == byte_shaving_row["command"][4:6]
    assert byte_shaving_row["telemetry"]["input_artifact_paths"] == [str(byte_shaving_plan)]


def test_non_dqs1_executable_materializers_do_not_emit_dqs1_portfolio_rows(
    tmp_path: Path,
) -> None:
    scorer_response = tmp_path / "scorer_response.json"
    action_output = tmp_path / "inverse_action.json"
    action_md = tmp_path / "inverse_action.md"
    scorer_response.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "rows": []}),
        encoding="utf-8",
    )
    plan = _inverse_surface_plan()
    plan["ranked_units"][0]["blockers"] = []
    plan["sweep_ladder"][0]["selected_operations"][0]["blockers"] = []

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
        materializer_contexts={
            INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND: {
                "scorer_response": str(scorer_response),
                "output": str(action_output),
                "md_out": str(action_md),
                "total_byte_budget": 64,
            }
        },
    )

    assert compiled["executable_row_count"] == 0
    assert compiled["queueable_row_count"] == 0
    assert compiled["portfolio"]["operator_action_rows"] == []
    assert compiled["action_summary"]["top_operator_actions"] == []
    assert any(
        f"planning_only_materializer_not_candidate_archive:{INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER}"
        in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    assert backlog_row["gap_class"] == "materializer_work_queue_required"
    assert backlog_row["target_kind"] == INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND
    assert backlog_row["receiver_contract_status"] == ("receiver_contract_registered_for_materializer_work_queue")
    work_row = compiled["materializer_work_queue"]["rows"][0]
    assert work_row["executable"] is True
    assert work_row["tool"] == "tools/build_inverse_steganalysis_action_functional.py"
    assert "inverse_action_functional_is_not_candidate_archive" in work_row["dispatch_blockers"]


def test_inverse_action_cells_compile_to_candidate_materializer_work_queue(
    tmp_path: Path,
) -> None:
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    candidate = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate_manifest.json"
    action.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ]
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    _write_single_member_zip(template)
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_cell_candidate_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    backlog_row = compiled["materializer_backlog"]["rows"][0]
    assert backlog_row["unit_kind"] == "scorer_inverse_surface_cell"
    assert backlog_row["operation_family"] == "materialize_inverse_scorer_cell_candidate"
    assert backlog_row["target_kind"] == INVERSE_SCORER_CELL_TARGET_KIND
    assert backlog_row["materializer_id"] == INVERSE_SCORER_CELL_MATERIALIZER

    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_CELL_TARGET_KIND: {
                "candidate_archive_template": str(template),
                "inverse_action_functional": str(action),
                "raw_contest_video_digest": "f" * 64,
                "output_archive": str(candidate),
                "manifest_out": str(manifest),
                "atom_ids": ["inverse_surface_pair0007"],
                "selected_limit": 1,
            }
        },
        source_plan_path="plan.json",
    )

    assert work_queue["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert work_queue["executable_row_count"] == 1
    row = work_queue["rows"][0]
    assert row["tool"] == "tools/materialize_inverse_scorer_cell_candidate.py"
    assert row["command"][:4] == [
        ".venv/bin/python",
        "tools/materialize_inverse_scorer_cell_candidate.py",
        "--candidate-archive-template",
        str(template),
    ]
    assert "--atom-id" in row["command"]
    _assert_typed_postconditions(
        row["postconditions"],
        path=manifest,
        schema="inverse_scorer_cell_candidate_v1",
    )
    assert row["telemetry"]["artifact_paths"] == [str(candidate), str(manifest)]
    assert row["telemetry"]["input_artifact_paths"] == [str(template), str(action)]
    assert "inverse_scorer_cell_candidate_requires_receiver_proof" in row["dispatch_blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_inverse_action_cells_refuse_invalid_candidate_archive_template(
    tmp_path: Path,
) -> None:
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    candidate = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate_manifest.json"
    action.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ]
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    template.write_bytes(b"not-a-real-zip-for-command-building")
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_cell_candidate_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_CELL_TARGET_KIND: {
                "candidate_archive_template": str(template),
                "inverse_action_functional": str(action),
                "raw_contest_video_digest": "f" * 64,
                "output_archive": str(candidate),
                "manifest_out": str(manifest),
                "atom_ids": ["inverse_surface_pair0007"],
                "selected_limit": 1,
            }
        },
        source_plan_path="plan.json",
    )

    assert work_queue["executable_row_count"] == 0
    row = work_queue["rows"][0]
    assert row["command"] == []
    assert any(
        blocker.startswith("candidate_archive_template_invalid_strict_single_member_zip")
        for blocker in row["materialization_blockers"]
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_inverse_action_cells_compile_to_candidate_chain_work_queue(
    tmp_path: Path,
) -> None:
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    output_dir = tmp_path / "inverse_cell_chain"
    source_inflate_output_dir = tmp_path / "source_inflate_out"
    candidate_inflate_output_dir = tmp_path / "candidate_inflate_out"
    action.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ]
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    _write_single_member_zip(template)
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_cell_candidate_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_CELL_TARGET_KIND: {
                "candidate_archive_template": str(template),
                "inverse_action_functional": str(action),
                "raw_contest_video_digest": "f" * 64,
                "output_dir": str(output_dir),
                "source_inflate_output_dir": str(source_inflate_output_dir),
                "candidate_inflate_output_dir": str(candidate_inflate_output_dir),
                "atom_ids": ["inverse_surface_pair0007"],
                "selected_limit": 1,
                "min_free_bytes": 123,
            }
        },
        source_plan_path="plan.json",
    )

    assert work_queue["executable_row_count"] == 1
    row = work_queue["rows"][0]
    assert row["tool"] == "tools/run_inverse_scorer_cell_candidate_chain.py"
    assert row["command"][:4] == [
        ".venv/bin/python",
        "tools/run_inverse_scorer_cell_candidate_chain.py",
        "--candidate-archive-template",
        str(template),
    ]
    assert "--min-free-bytes" in row["command"]
    assert "--source-inflate-output-dir" in row["command"]
    assert "--candidate-inflate-output-dir" in row["command"]
    assert "--fail-if-inflate-parity-blocked" in row["command"]
    assert row["telemetry"]["input_artifact_paths"] == [
        str(template),
        str(action),
        str(source_inflate_output_dir),
        str(candidate_inflate_output_dir),
    ]
    _assert_typed_postconditions(
        row["postconditions"],
        path=output_dir / INVERSE_CELL_CHAIN_MANIFEST_NAME,
        schema="inverse_scorer_cell_candidate_chain_v1",
        chain=True,
    )
    completion_contract = row["postconditions"][1]
    chain_contract = row["postconditions"][2]
    assert completion_contract.get("required_equals") == {"schema": "inverse_scorer_cell_candidate_chain_v1"}
    assert "inflate_parity_satisfied" in completion_contract.get("required_true", [])
    assert completion_contract.get("required_less_than") == []
    assert chain_contract.get("required_serialized_archive_saving") in (None, False)
    assert chain_contract.get("required_inflate_parity") is True
    assert row["telemetry"]["artifact_paths"] == [
        str(output_dir),
        str(source_inflate_output_dir),
        str(candidate_inflate_output_dir),
    ]
    assert row["telemetry"]["recursive"] is True
    assert row["telemetry"]["parity_probe_required"] is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="inverse_cell_chain_exec_fixture",
        repo_root=tmp_path,
        lane_id="lane_inverse_cell_chain_exec_fixture",
        source_work_queue_path=tmp_path / "work_queue.json",
        local_cpu_concurrency=2,
        resource_concurrency={"local_mlx": 3},
        step_timeout_seconds=600,
    )
    dag = build_staircase_dag_from_experiment_queue(
        execution_queue,
        dag_id="inverse_cell_chain_dag_fixture",
    )
    plan = plan_staircase_dispatch(dag, max_nodes=1)
    task = plan["dask_task_specs"][0]
    step = execution_queue["experiments"][0]["steps"][0]

    assert task["command"][:2] == [
        ".venv/bin/python",
        "tools/run_inverse_scorer_cell_candidate_chain.py",
    ]
    assert task["target_kind"] == INVERSE_SCORER_CELL_TARGET_KIND
    assert step["telemetry"]["input_artifact_paths"] == row["telemetry"]["input_artifact_paths"]
    assert execution_queue["controls"]["max_concurrency"]["local_mlx"] == 3
    assert task["materializer_id"] == INVERSE_SCORER_CELL_MATERIALIZER
    assert task["receiver_contract_id"] == INVERSE_SCORER_CELL_RECEIVER_CONTRACT_ID
    assert task["receiver_contract_kind"] == INVERSE_SCORER_CELL_RECEIVER_CONTRACT_KIND
    assert task["telemetry"]["artifact_paths"] == [
        str(output_dir),
        str(source_inflate_output_dir),
        str(candidate_inflate_output_dir),
    ]
    assert task["telemetry"]["recursive"] is True
    assert task["telemetry"]["parity_probe_required"] is True
    assert task["experiment_metadata"]["score_claim"] is False
    assert task["experiment_metadata"]["ready_for_exact_eval_dispatch"] is False


def test_inverse_action_chain_work_queue_requires_parity_context_for_exact_mode(
    tmp_path: Path,
) -> None:
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    output_dir = tmp_path / "inverse_cell_chain"
    action.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ]
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    _write_single_member_zip(template)
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_cell_candidate_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )

    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_CELL_TARGET_KIND: {
                "candidate_archive_template": str(template),
                "inverse_action_functional": str(action),
                "raw_contest_video_digest": "f" * 64,
                "output_dir": str(output_dir),
                "atom_ids": ["inverse_surface_pair0007"],
                "selected_limit": 1,
            }
        },
        source_plan_path="plan.json",
    )

    assert work_queue["executable_row_count"] == 0
    row = work_queue["rows"][0]
    assert row["command"] == []
    assert "inverse_scorer_cell_exact_chain_requires_inflate_parity_context" in row["materialization_blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_materializer_execution_queue_can_gate_work_on_storage_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            "zip_member_range_a": {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime",
                "output_dir": str(tmp_path / "materializer_results" / "materializer_out"),
            }
        },
        source_plan_path="plan.json",
    )

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="materializer_storage_preflight_fixture",
        repo_root=tmp_path,
        local_cpu_concurrency=2,
        include_scheduler_preflight=True,
        scheduler_results_root=str(tmp_path / "materializer_results"),
        scheduler_storage_tiers=(f"fixture={tmp_path / 'VertigoDataTier'}",),
        scheduler_storage_workload_subdir="materializer_results",
        scheduler_storage_expected_workload_root=str(tmp_path / "materializer_results"),
        scheduler_storage_expected_bytes=123456,
        scheduler_proactive_cleanup_roots=("experiments/results", ".omx/tmp"),
        scheduler_proactive_cleanup_execute=True,
        scheduler_proactive_cleanup_cold_store_roots=(str(tmp_path / "cold_store"),),
    )

    assert execution_queue["controls"]["max_concurrency"]["local_cpu"] == 2
    assert execution_queue["controls"]["max_concurrency"]["local_io_heavy"] == 1
    assert [experiment["id"] for experiment in execution_queue["experiments"]] == [
        MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID,
        work_queue["rows"][0]["work_id"],
    ]
    preflight = execution_queue["experiments"][0]
    assert preflight["tags"] == [
        "byte-shaving",
        "materializer",
        "scheduler-preflight",
        "storage",
        "cleanup",
        "no-score-authority",
    ]
    storage_step, cleanup_step = preflight["steps"]
    assert storage_step["command"][:4] == [
        ".venv/bin/python",
        "tools/plan_experiment_storage.py",
        "--output",
        storage_step["postconditions"][0]["path"],
    ]
    assert "--storage-tier" in storage_step["command"]
    assert "--requested-bytes" in storage_step["command"]
    assert "123456" in storage_step["command"]
    assert cleanup_step["resources"]["kind"] == "local_io_heavy"
    assert cleanup_step["command"][:2] == [
        ".venv/bin/python",
        "tools/compact_experiment_artifacts.py",
    ]
    assert "experiments/results" in cleanup_step["command"]
    assert "--execute" in cleanup_step["command"]
    materializer_step = execution_queue["experiments"][1]["steps"][0]
    assert materializer_step["requires"] == [f"{MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.proactive_cleanup"]

    dag = build_staircase_dag_from_experiment_queue(
        execution_queue,
        dag_id="materializer_storage_preflight_dag_fixture",
    )
    materializer_node_id = f"{execution_queue['experiments'][1]['id']}.{MATERIALIZER_EXECUTION_STEP_ID}"
    by_node_id = {node["node_id"]: node for node in dag["nodes"]}
    storage_dependency = by_node_id[materializer_node_id]["metadata"]["storage_preflight_dependency"]
    assert storage_dependency["schema"] == STORAGE_PREFLIGHT_DEPENDENCY_SCHEMA
    assert storage_dependency["dependency_node_id"] == (
        f"{MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.proactive_cleanup"
    )
    assert storage_dependency["storage_plan_node_id"] == (
        f"{MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.storage_tier_plan"
    )
    assert storage_dependency["storage_plan_artifact_path"] == storage_step["postconditions"][0]["path"]
    assert storage_dependency["cleanup_plan_artifact_path"] == cleanup_step["postconditions"][0]["path"]
    assert storage_dependency["artifact_paths"] == [
        storage_step["postconditions"][0]["path"],
        cleanup_step["postconditions"][0]["path"],
    ]
    assert storage_dependency["score_claim"] is False
    assert storage_dependency["promotion_eligible"] is False
    assert storage_dependency["rank_or_kill_eligible"] is False
    assert storage_dependency["ready_for_exact_eval_dispatch"] is False
    assert "storage_preflight_dependency_is_advisory_only" in storage_dependency["dispatch_blockers"]
    monkeypatch.chdir(tmp_path)
    storage_artifact = Path(storage_dependency["storage_plan_artifact_path"])
    cleanup_artifact = Path(storage_dependency["cleanup_plan_artifact_path"])
    storage_artifact.parent.mkdir(parents=True, exist_ok=True)
    cleanup_artifact.parent.mkdir(parents=True, exist_ok=True)
    storage_artifact.write_text(
        json.dumps(
            {
                "selected_workload_root": str(tmp_path / "materializer_results"),
                "selected_workload_root_matches_expected": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    cleanup_artifact.write_text(
        json.dumps(
            {
                "plan": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "candidate_count": 0,
                    "total_reclaimable_bytes": 0,
                },
                "execution": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "executed_count": 0,
                    "local_bytes_reclaimed": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    plan = plan_staircase_dispatch(
        dag,
        status_map={
            f"{MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.storage_tier_plan": "succeeded",
            f"{MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.proactive_cleanup": "succeeded",
        },
        max_nodes=1,
    )
    task = plan["dask_task_specs"][0]
    assert task["experiment_id"] == execution_queue["experiments"][1]["id"]
    assert task["storage_preflight_dependency"] == storage_dependency
    assert task["storage_preflight_dependencies"] == [storage_dependency]
    assert task["experiment_metadata"]["score_claim"] is False
    assert task["experiment_metadata"]["ready_for_exact_eval_dispatch"] is False


def test_materializer_execution_queue_move_preflight_uses_policy_cold_store_default(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            "zip_member_range_a": {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime",
                "output_dir": ("/Volumes/VertigoDataTier/pact/materializer_results/materializer_out"),
            }
        },
        source_plan_path="plan.json",
    )

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="materializer_storage_preflight_fixture",
        repo_root=tmp_path,
        include_scheduler_preflight=True,
        scheduler_results_root="/Volumes/VertigoDataTier/pact/materializer_results",
        scheduler_storage_expected_workload_root=("/Volumes/VertigoDataTier/pact/materializer_results"),
        scheduler_proactive_cleanup_execute=True,
        scheduler_proactive_cleanup_action="move",
    )

    cleanup_command = execution_queue["experiments"][0]["steps"][1]["command"]
    assert "/Volumes/VertigoDataTier/pact/cold_store" in cleanup_command
    assert "/Volumes/APDataStore/pact/cold_store" in cleanup_command


def test_materializer_execution_queue_blocks_outputs_outside_storage_root(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            "zip_member_range_a": {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime",
                "output_dir": "materializer_out",
            }
        },
        source_plan_path="plan.json",
    )

    with pytest.raises(ExperimentQueueError, match="outside scheduler workload root"):
        build_materializer_execution_queue(
            work_queue,
            queue_id="materializer_storage_preflight_fixture",
            repo_root=tmp_path,
            include_scheduler_preflight=True,
            scheduler_results_root=str(tmp_path / "materializer_results"),
            scheduler_storage_expected_workload_root=str(tmp_path / "materializer_results"),
            scheduler_proactive_cleanup_execute=True,
            scheduler_proactive_cleanup_action="delete",
        )


def test_materializer_execution_queue_requires_bound_storage_root_for_preflight(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            "zip_member_range_a": {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime",
                "output_dir": str(tmp_path / "materializer_results" / "materializer_out"),
            }
        },
        source_plan_path="plan.json",
    )

    with pytest.raises(
        ExperimentQueueError,
        match="scheduler_storage_expected_workload_root is required",
    ):
        build_materializer_execution_queue(
            work_queue,
            queue_id="materializer_storage_preflight_fixture",
            repo_root=tmp_path,
            include_scheduler_preflight=True,
            scheduler_results_root="experiments/results",
            scheduler_proactive_cleanup_execute=True,
            scheduler_proactive_cleanup_action="delete",
        )


def test_materializer_execution_queue_rejects_dry_run_cleanup_gate(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            "zip_member_range_a": {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime",
                "output_dir": str(tmp_path / "materializer_results" / "materializer_out"),
            }
        },
        source_plan_path="plan.json",
    )

    with pytest.raises(
        ExperimentQueueError,
        match="scheduler_proactive_cleanup_execute must be true",
    ):
        build_materializer_execution_queue(
            work_queue,
            queue_id="materializer_storage_preflight_fixture",
            repo_root=tmp_path,
            include_scheduler_preflight=True,
            scheduler_results_root=str(tmp_path / "materializer_results"),
            scheduler_storage_expected_workload_root=str(tmp_path / "materializer_results"),
        )


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
    assert step["telemetry"]["input_artifact_paths"] == [
        "schema.json",
        "runtime",
        "beam_a.json",
        "archive.zip",
    ]
    assert step["telemetry"]["recursive"] is True
    assert step["command"][:2] == [
        ".venv/bin/python",
        "tools/run_byte_range_entropy_recode_chain.py",
    ]
    _assert_typed_postconditions(
        step["postconditions"],
        path=output_dir / CHAIN_MANIFEST_NAME,
        schema=CHAIN_SCHEMA,
        chain=True,
    )


def test_materializer_execution_queue_can_append_exact_readiness_followups(
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
        local_cpu_concurrency=5,
        include_exact_readiness_followup=True,
    )

    assert execution_queue["controls"]["max_concurrency"]["local_cpu"] == 5
    experiment = execution_queue["experiments"][0]
    steps = experiment["steps"]
    assert [step["id"] for step in steps] == [
        MATERIALIZER_EXECUTION_STEP_ID,
        MATERIALIZER_HARVEST_STEP_ID,
        MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    ]
    materializer_step, harvest_step, dispatch_step = steps
    assert harvest_step["requires"] == [MATERIALIZER_EXECUTION_STEP_ID]
    assert dispatch_step["requires"] == [MATERIALIZER_HARVEST_STEP_ID]
    assert harvest_step["command"][:2] == [
        ".venv/bin/python",
        "tools/harvest_materializer_chain_candidates.py",
    ]
    assert "--chain-manifest" in harvest_step["command"]
    assert "--allow-unfinished-state" not in harvest_step["command"]
    assert f"chain_out/{CHAIN_MANIFEST_NAME}" in harvest_step["command"]
    assert "chain_out/exact_eval_handoff/source_queue.json" in (harvest_step["command"])
    assert dispatch_step["command"][:2] == [
        ".venv/bin/python",
        "tools/build_materializer_exact_eval_dispatch_plan.py",
    ]
    assert "--dispatch-mode" not in dispatch_step["command"]
    assert "--allow-paid-dispatch-queue" not in dispatch_step["command"]
    assert "chain_out/exact_eval_handoff/dispatch_queue.json" in (dispatch_step["command"])

    dag = build_staircase_dag_from_experiment_queue(
        execution_queue,
        dag_id="materializer_followup_dag_fixture",
    )
    by_node_id = {node["node_id"]: node for node in dag["nodes"]}
    assert by_node_id[f"{experiment['id']}.{MATERIALIZER_HARVEST_STEP_ID}"]["dependencies"] == [
        f"{experiment['id']}.{MATERIALIZER_EXECUTION_STEP_ID}"
    ]
    assert by_node_id[f"{experiment['id']}.{MATERIALIZER_DISPATCH_PLAN_STEP_ID}"]["dependencies"] == [
        f"{experiment['id']}.{MATERIALIZER_HARVEST_STEP_ID}"
    ]


def test_materializer_execution_queue_skips_exact_followup_for_planning_only_rows(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_surface_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND: {
                "inverse_scorer_surface": "surface.json",
                "output": str(tmp_path / "inverse_action.json"),
            }
        },
        source_plan_path="plan.json",
    )

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="inverse_action_exec_fixture",
        repo_root=tmp_path,
        include_exact_readiness_followup=True,
    )

    experiment = execution_queue["experiments"][0]
    assert [step["id"] for step in experiment["steps"]] == [MATERIALIZER_EXECUTION_STEP_ID]
    assert experiment["metadata"]["exact_readiness_followup_requested"] is True
    assert experiment["metadata"]["exact_readiness_followup_enabled"] is False
    assert experiment["metadata"]["exact_readiness_followup_skipped_reason"] == (
        "planning_only_inverse_action_functional_not_candidate_archive"
    )


def test_materializer_execution_queue_skips_exact_followup_for_nonharvestable_manifest(
    tmp_path: Path,
) -> None:
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    candidate = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate_manifest.json"
    action.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ]
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    _write_single_member_zip(template)
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_cell_candidate_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_CELL_TARGET_KIND: {
                "candidate_archive_template": str(template),
                "inverse_action_functional": str(action),
                "raw_contest_video_digest": "f" * 64,
                "output_archive": str(candidate),
                "manifest_out": str(manifest),
                "selected_limit": 1,
            }
        },
        source_plan_path="plan.json",
    )

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="inverse_cell_candidate_exec_fixture",
        repo_root=tmp_path,
        include_exact_readiness_followup=True,
    )

    experiment = execution_queue["experiments"][0]
    assert [step["id"] for step in experiment["steps"]] == [MATERIALIZER_EXECUTION_STEP_ID]
    assert experiment["metadata"]["exact_readiness_followup_requested"] is True
    assert experiment["metadata"]["exact_readiness_followup_enabled"] is False
    assert experiment["metadata"]["exact_readiness_followup_skipped_reason"] == (
        "materializer_manifest_not_harvestable_for_exact_readiness"
    )


def test_materializer_execution_queue_appends_exact_followup_for_archive_section_candidate(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _archive_section_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            backlog_row["backlog_key"]: {
                "archive_path": "source.zip",
                "section_manifest": "sections.json",
                "output_archive": str(tmp_path / "candidate.zip"),
                "output_manifest": str(tmp_path / "candidate.json"),
                "section_names": ["decoder_packed_brotli"],
            }
        },
        source_plan_path="plan.json",
    )

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="packet_section_transform_exec_fixture",
        repo_root=tmp_path,
        include_exact_readiness_followup=True,
    )

    experiment = execution_queue["experiments"][0]
    assert [step["id"] for step in experiment["steps"]] == [
        MATERIALIZER_EXECUTION_STEP_ID,
        MATERIALIZER_HARVEST_STEP_ID,
        MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    ]
    assert experiment["metadata"]["exact_readiness_followup_requested"] is True
    assert experiment["metadata"]["exact_readiness_followup_enabled"] is True
    assert experiment["metadata"]["exact_readiness_followup_skipped_reason"] is None
    harvest_step = experiment["steps"][1]
    assert "--chain-manifest" in harvest_step["command"]
    assert "candidate.json" in harvest_step["command"]
    assert "exact_eval_handoff/source_queue.json" in " ".join(harvest_step["command"])


def test_materializer_execution_queue_followup_requires_chain_postcondition_for_candidates(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _byte_range_entropy_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    backlog_row = compiled["materializer_backlog"]["rows"][0]
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            backlog_row["backlog_key"]: {
                "schema_manifest": "schema.json",
                "beam_probe_reports": ["beam_a.json"],
                "source_runtime_dir": "runtime",
                "output_dir": str(tmp_path / "chain_out"),
            }
        },
        source_plan_path="plan.json",
    )
    work_queue["rows"][0]["postconditions"] = []

    with pytest.raises(ExperimentQueueError, match="harvestable materializer"):
        build_materializer_execution_queue(
            work_queue,
            queue_id="candidate_exec_fixture",
            repo_root=tmp_path,
            include_exact_readiness_followup=True,
        )


def test_materializer_execution_queue_wraps_inverse_action_work_rows(
    tmp_path: Path,
) -> None:
    compiled = compile_dqs1_byte_shaving_campaign(
        _inverse_surface_plan(),
        repo_root=tmp_path,
        candidate_limit=4,
        portfolio_json="portfolio.json",
    )
    action_output = tmp_path / "inverse_action.json"
    work_queue = build_materializer_work_queue(
        compiled["materializer_backlog"],
        repo_root=tmp_path,
        contexts={
            INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND: {
                "inverse_scorer_surface": "surface.json",
                "output": str(action_output),
            }
        },
        source_plan_path="plan.json",
    )

    execution_queue = build_materializer_execution_queue(
        work_queue,
        queue_id="inverse_action_exec_fixture",
        repo_root=tmp_path,
        lane_id="lane_inverse_action_exec_fixture",
        source_work_queue_path=tmp_path / "work_queue.json",
        local_cpu_concurrency=2,
        step_timeout_seconds=300,
    )

    assert execution_queue["schema"] == "experiment_queue.v1"
    assert execution_queue["controls"]["max_concurrency"] == {"local_cpu": 2}
    experiment = execution_queue["experiments"][0]
    assert experiment["metadata"]["target_kind"] == (INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND)
    assert experiment["metadata"]["score_claim"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    step = experiment["steps"][0]
    assert step["resources"]["kind"] == "local_cpu"
    assert step["command"][:4] == [
        ".venv/bin/python",
        "tools/build_inverse_steganalysis_action_functional.py",
        "--output",
        str(action_output),
    ]
    _assert_typed_postconditions(
        step["postconditions"],
        path=action_output,
        schema="inverse_steganalysis_discrete_action_functional.v1",
    )


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
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]} for row in compiled["executable_rows"]
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
        "selected_unit_blocker:pair0371:requires_full_frame_parity" in row["materialization_blockers"]
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
    assert blocker_backlog["receiver_contract_status"] == "receiver_contract_registered_but_source_blocked"
    expected_blocker_count = sum(
        "selected_unit_blocker:pair0371:requires_full_frame_parity" in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert blocker_backlog["blocked_row_count"] == expected_blocker_count
    assert blocker_backlog["blocker_counts"] == {
        "selected_unit_blocker:pair0371:requires_full_frame_parity": expected_blocker_count
    }
    assert all(
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]} for row in compiled["executable_rows"]
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
        if "byte_null_run_a" in row["selected_unit_ids"] and {"pair0320", "pair0371"} & set(row["selected_unit_ids"])
    ]
    assert mixed
    assert all(row["executable"] is False for row in mixed)
    assert compiled["queueable_row_count"] == 0
    assert compiled["action_summary"]["top_operator_actions"] == []
    assert "partial_materialization_requires_explicit_allow" in compiled["partial_materialization_blockers"]
    assert any(
        "materializer_target_kind_required:byte_range:null_remove_or_seed" in row["materialization_blockers"]
        for row in mixed
    )
    assert all(
        "byte_null_run_a" not in {unit["unit_id"] for unit in row["source_metadata"]["source_units"]}
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
    for ladder in ("operation_set_ladder", "combination_ladder", "sweep_ladder"):
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
        "materializer_not_registered:experimental_unregistered_materializer" in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert all(
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]} for row in compiled["executable_rows"]
    )


def test_compile_dqs1_byte_shaving_plan_blocks_spoofed_registered_materializer(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    for unit in plan["ranked_units"]:
        if unit["unit_id"] == "pair0371":
            unit["unit_kind"] = "byte_range"
    for ladder in ("operation_set_ladder", "combination_ladder", "sweep_ladder"):
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
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]} for row in compiled["executable_rows"]
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
            "--include-scheduler-preflight",
            "--scheduler-storage-tier",
            f"fixture={tmp_path / 'VertigoDataTier'}",
            "--scheduler-storage-workload-subdir",
            "results",
            "--scheduler-storage-expected-workload-root",
            str(tmp_path / "results"),
            "--scheduler-storage-expected-bytes",
            "2048",
            "--scheduler-proactive-cleanup-root",
            "experiments/results",
            "--scheduler-proactive-cleanup-execute",
            "--scheduler-proactive-cleanup-cold-store-root",
            str(tmp_path / "cold_store"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    assert stdout["executable_row_count"] >= 1
    assert stdout["dqs1_executable_row_count"] == stdout["executable_row_count"]
    assert stdout["materializer_backlog_out"] == str(backlog)
    assert stdout["materializer_backlog_row_count"] >= 1
    assert stdout["queue"]["experiment_count"] == 3
    assert materialization.is_file()
    assert portfolio.is_file()
    assert summary.is_file()
    assert backlog.is_file()
    assert json.loads(backlog.read_text(encoding="utf-8"))["schema"] == MATERIALIZER_BACKLOG_SCHEMA
    loaded = load_queue_definition(queue)
    assert loaded["controls"]["max_concurrency"]["local_cpu"] == 2
    assert len(loaded["experiments"]) == 3
    preflight = loaded["experiments"][0]
    assert preflight["id"] == DEFAULT_SCHEDULER_PREFLIGHT_EXPERIMENT_ID
    assert preflight["steps"][0]["command"][:2] == [
        ".venv/bin/python",
        "tools/plan_experiment_storage.py",
    ]
    assert "--storage-tier" in preflight["steps"][0]["command"]
    assert all(experiment["id"].startswith("pairset_byte_shave_") for experiment in loaded["experiments"][1:])
    assert all(
        experiment["steps"][0]["requires"] == [f"{DEFAULT_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.proactive_cleanup"]
        for experiment in loaded["experiments"][1:]
    )
    assert all(
        experiment["metadata"]["source_metadata"]["materializer_registry_schema"]
        == "byte_shaving_materializer_registry.v1"
        for experiment in loaded["experiments"][1:]
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
    output_dir = tmp_path / "materializer_exec" / "chain_out"
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
            "--materializer-resource-concurrency",
            "local_cpu=4",
            "--include-materializer-scheduler-preflight",
            "--materializer-scheduler-storage-tier",
            f"fixture={tmp_path / 'VertigoDataTier'}",
            "--materializer-scheduler-storage-workload-subdir",
            "materializer_exec",
            "--materializer-scheduler-storage-expected-workload-root",
            str(tmp_path / "materializer_exec"),
            "--materializer-scheduler-storage-expected-bytes",
            "4096",
            "--materializer-scheduler-proactive-cleanup-root",
            "experiments/results",
            "--materializer-scheduler-proactive-cleanup-execute",
            "--materializer-scheduler-proactive-cleanup-cold-store-root",
            str(tmp_path / "cold_store"),
            "--include-materializer-exact-readiness-followup",
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
    assert stdout["materializer_execution_queue"]["exact_readiness_followup"] is True
    assert stdout["materializer_execution_queue"]["experiment_count"] == 2
    assert stdout["materializer_work_queue_row_count"] == 1
    assert stdout["materializer_work_queue_executable_row_count"] == 1
    assert stdout["materializer_work_queue_blocked_row_count"] == 0
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
    assert loaded_execution_queue["controls"]["max_concurrency"]["local_cpu"] == 4
    assert loaded_execution_queue["controls"]["max_concurrency"]["local_io_heavy"] == 1
    preflight = loaded_execution_queue["experiments"][0]
    assert preflight["id"] == MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID
    assert preflight["steps"][0]["command"][:2] == [
        ".venv/bin/python",
        "tools/plan_experiment_storage.py",
    ]
    experiment = loaded_execution_queue["experiments"][1]
    assert experiment["lane_id"] == "lane_materializer_exec_fixture"
    assert experiment["metadata"]["schema"] == MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA
    step = experiment["steps"][0]
    assert step["id"] == MATERIALIZER_EXECUTION_STEP_ID
    assert step["requires"] == [f"{MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.proactive_cleanup"]
    assert step["timeout_seconds"] == 600
    assert step["telemetry"]["artifact_paths"] == [str(output_dir)]
    assert step["telemetry"]["input_artifact_paths"] == [
        "schema.json",
        "runtime",
        "beam_a.json",
        "beam_b.json",
        "archive.zip",
    ]
    assert step["postconditions"][0]["type"] == "json_equals"
    assert [step["id"] for step in experiment["steps"]] == [
        MATERIALIZER_EXECUTION_STEP_ID,
        MATERIALIZER_HARVEST_STEP_ID,
        MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    ]


def test_byte_shaving_campaign_queue_cli_generates_materializer_contexts_from_artifact_map(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    artifact_map_path = tmp_path / "artifact_map.json"
    contexts_path = tmp_path / "generated_contexts.json"
    materialization = tmp_path / "materialization.json"
    portfolio = tmp_path / "portfolio.json"
    summary = tmp_path / "action_summary.json"
    work_queue = tmp_path / "materializer_work_queue.json"
    output_root = tmp_path / "materializer_outputs"
    section_manifest = tmp_path / "sections.json"
    plan_path.write_text(json.dumps(_archive_section_entropy_plan()), encoding="utf-8")
    artifact_map_path.write_text(
        json.dumps(
            {
                "schema": "final_byte_artifact_map.fixture.v1",
                "artifacts": {
                    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND: {
                        "archive_path": str(tmp_path / "source.zip"),
                        "section_manifest": str(section_manifest),
                        "target_sections": ["decoder_packed_brotli"],
                        "brotli_quality": [11],
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
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
            "--materializer-artifact-map",
            str(artifact_map_path),
            "--materializer-contexts-out",
            str(contexts_path),
            "--materializer-context-default-output-root",
            str(output_root),
            "--materialization-out",
            str(materialization),
            "--portfolio-out",
            str(portfolio),
            "--action-summary-out",
            str(summary),
            "--materializer-work-queue-out",
            str(work_queue),
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
    assert stdout["materializer_contexts_generated"] is True
    assert stdout["materializer_contexts_blocked_count"] == 0
    assert stdout["materializer_work_queue_executable_row_count"] == 1
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    assert contexts["schema"] == MATERIALIZER_CONTEXTS_SCHEMA
    assert contexts["rows"][0]["context"]["output_archive"].startswith(str(output_root))
    work = json.loads(work_queue.read_text(encoding="utf-8"))
    assert work["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    row = work["rows"][0]
    assert row["executable"] is True
    assert row["tool"] == "tools/run_family_agnostic_materializer.py"
    assert "--section-manifest" in row["command"]
    assert str(section_manifest) in row["command"]


def test_inverse_action_compiler_hint_runs_family_agnostic_materializer(
    tmp_path: Path,
) -> None:
    source_archive = tmp_path / "source.zip"
    section_manifest = tmp_path / "sections.json"
    plan_path = tmp_path / "plan.json"
    artifact_map_path = tmp_path / "artifact_map.json"
    contexts_path = tmp_path / "generated_contexts.json"
    materialization = tmp_path / "materialization.json"
    portfolio = tmp_path / "portfolio.json"
    summary = tmp_path / "action_summary.json"
    work_queue = tmp_path / "materializer_work_queue.json"
    output_root = tmp_path / "materializer_outputs"

    raw = b"inverse-action-fixture" * 256
    section = brotli.compress(raw, quality=0)
    with zipfile.ZipFile(source_archive, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("0.raw", section)
    section_manifest.write_text(
        json.dumps(
            {
                "schema": "fixture_section_manifest.v1",
                "member": {"name": "0.raw"},
                "sections": [
                    {
                        "name": "decoder_packed_brotli",
                        "index": 0,
                        "offset": 0,
                        "length": len(section),
                        "sha256": hashlib.sha256(section).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    action_payload = {
        "schema": "inverse_steganalysis_discrete_action_functional.v1",
        "tool": "tac.optimization.inverse_steganalysis_acquisition",
        "math_model": {
            "representation": "discrete_action_integral_water_bucket_fixture",
            "stationarity_rule": "select positive euler_lagrange_residual cells",
            "lambda_rate": 0.0000005,
        },
        "integral_totals": {"cell_count": 1, "blocked_cell_count": 0},
        "water_bucket": {
            "schema": "inverse_steganalysis_water_bucket_plan.v1",
            "selected_count": 1,
            "selected_expected_score_gain": 0.0001,
            "selected_cells": [
                {
                    "atom_id": "compiled_archive_section_cell",
                    "candidate_id": "compiled_archive_section_candidate",
                    "scope_axis": "bytes",
                    "component": "rate",
                    "water_fill_cost_bytes": 64,
                    "expected_score_gain": 0.0001,
                    "euler_lagrange_residual": 0.00009,
                }
            ],
            **_false_authority(),
        },
        "cells": [
            {
                "atom_id": "compiled_archive_section_cell",
                "candidate_id": "compiled_archive_section_candidate",
                "scope_axis": "bytes",
                "component": "rate",
                "operation_set_compiler": {
                    "schema": "inverse_action_operation_set_compiler_hint.v1",
                    "operation_set_id": "compiled_archive_section_set",
                    "candidate_saved_bytes": 64,
                    "operation_portability": "family_agnostic",
                    "selected_operations": [
                        {
                            "unit_id": "compiled_decoder_section",
                            "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                            "archive_section": "decoder_packed_brotli",
                            "section_name": "decoder_packed_brotli",
                            "candidate_saved_bytes": 64,
                            "representation_family_class": "hnerv_variant",
                        }
                    ],
                },
            }
        ],
        **_false_authority(),
    }
    signal = build_signal_surface_from_inverse_action_functional(action_payload)
    plan = build_byte_shaving_campaign_plan(signal, max_k=1)
    assert signal["water_bucket_materialization_portfolio"]["actuation_modes"] == [
        "compiled_operation_set"
    ]
    assert plan["materialization_bridge"]["compiled_operation_set_count"] == 1
    assert plan["materialization_bridge"][
        "queue_consumable_packet_ir_operation_set_count"
    ] == 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    artifact_map_path.write_text(
        json.dumps(
            {
                "schema": "final_byte_artifact_map.fixture.v1",
                "artifacts": {
                    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND: {
                        "archive_path": str(source_archive),
                        "section_manifest": str(section_manifest),
                        "target_sections": ["decoder_packed_brotli"],
                        "brotli_quality": [0],
                        "allow_size_regression": True,
                        **_false_authority(),
                    }
                },
                **_false_authority(),
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
            "--materializer-artifact-map",
            str(artifact_map_path),
            "--materializer-contexts-out",
            str(contexts_path),
            "--materializer-context-default-output-root",
            str(output_root),
            "--materializer-contexts-fail-if-blocked",
            "--materialization-out",
            str(materialization),
            "--portfolio-out",
            str(portfolio),
            "--action-summary-out",
            str(summary),
            "--materializer-work-queue-out",
            str(work_queue),
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
    assert stdout["materializer_contexts_generated"] is True
    assert stdout["materializer_contexts_blocked_count"] == 0
    assert stdout["materializer_work_queue_executable_row_count"] == 1
    assert stdout["materializer_work_queue_blocked_row_count"] == 0
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    context = contexts["rows"][0]["context"]
    assert context["source_packet_ir_operation_set_ids"]

    payload = json.loads(work_queue.read_text(encoding="utf-8"))
    assert payload["schema"] == MATERIALIZER_WORK_QUEUE_SCHEMA
    assert payload["executable_row_count"] == 1
    row = payload["rows"][0]
    assert row["executable"] is True
    assert row["tool"] == "tools/run_family_agnostic_materializer.py"
    assert row["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    assert row["source_packet_ir_operation_set_ids"]
    assert "--runtime-consumption-proof-out" in row["command"]
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    command = [
        sys.executable,
        str(REPO_ROOT / row["command"][1]),
        *row["command"][2:],
    ]
    smoke = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    manifest = json.loads(smoke.stdout)
    persisted_manifest = json.loads(
        Path(context["output_manifest"]).read_text(encoding="utf-8")
    )
    assert persisted_manifest["schema"] == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA
    assert manifest["schema"] == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA
    assert manifest["byte_closed_candidate_emitted"] is True
    assert manifest["receiver_contract_satisfied"] is True
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert Path(manifest["candidate_archive"]["path"]).is_file()
    proof_path = Path(manifest["runtime_consumption_proof_path"])
    assert proof_path.is_file()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof["receiver_contract_satisfied"] is True
    assert proof["section_proofs"][0]["raw_payload_identical"] is True


def test_byte_shaving_campaign_queue_cli_generates_inverse_scorer_contexts_and_smoke_materializes(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    artifact_map_path = tmp_path / "artifact_map.json"
    contexts_path = tmp_path / "generated_contexts.json"
    materialization = tmp_path / "materialization.json"
    portfolio = tmp_path / "portfolio.json"
    summary = tmp_path / "action_summary.json"
    work_queue = tmp_path / "materializer_work_queue.json"
    output_root = tmp_path / "materializer_outputs"
    template = tmp_path / "template.zip"
    action = tmp_path / "inverse_action.json"
    with zipfile.ZipFile(template, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("x", b"base-payload")
    action.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "schema": "inverse_steganalysis_water_bucket_plan.v1",
                    "selected_count": 1,
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plan_path.write_text(json.dumps(_inverse_cell_candidate_plan()), encoding="utf-8")
    artifact_map_path.write_text(
        json.dumps(
            {
                "schema": "final_byte_artifact_map.fixture.v1",
                "artifacts": {
                    INVERSE_SCORER_CELL_TARGET_KIND: {
                        "candidate_archive_template": str(template),
                        "inverse_action_functional": str(action),
                        "raw_contest_video_digest": "f" * 64,
                        "atom_ids": ["inverse_surface_pair0007"],
                        "selected_limit": 1,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--plan",
            str(plan_path),
            "--materializer-artifact-map",
            str(artifact_map_path),
            "--materializer-contexts-out",
            str(contexts_path),
            "--materializer-context-default-output-root",
            str(output_root),
            "--materializer-contexts-fail-if-blocked",
            "--materialization-out",
            str(materialization),
            "--portfolio-out",
            str(portfolio),
            "--action-summary-out",
            str(summary),
            "--materializer-work-queue-out",
            str(work_queue),
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
    assert stdout["materializer_contexts_generated"] is True
    assert stdout["materializer_contexts_blocked_count"] == 0
    assert stdout["materializer_work_queue_executable_row_count"] == 1
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    context = contexts["rows"][0]["context"]
    assert context["candidate_archive_template"] == str(template)
    assert context["inverse_action_functional"] == str(action)
    assert context["output_archive"].startswith(str(output_root))
    assert context["manifest_out"].startswith(str(output_root))
    work = json.loads(work_queue.read_text(encoding="utf-8"))
    row = work["rows"][0]
    assert row["tool"] == "tools/materialize_inverse_scorer_cell_candidate.py"
    assert row["executable"] is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    command = [sys.executable, str(REPO_ROOT / row["command"][1]), *row["command"][2:]]
    smoke = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    smoke_stdout = json.loads(smoke.stdout)
    assert smoke_stdout["selected_cell_count"] == 1
    assert smoke_stdout["receiver_contract_satisfied"] is False
    assert smoke_stdout["score_claim"] is False
    manifest = json.loads(Path(context["manifest_out"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "inverse_scorer_cell_candidate_v1"
    assert manifest["byte_closed_candidate_emitted"] is True
    assert manifest["candidate_generation_only"] is True
    assert manifest["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_missing" in manifest["readiness_blockers"]
    assert Path(context["output_archive"]).is_file()


def test_byte_shaving_campaign_queue_cli_generates_inverse_scorer_chain_context_and_executes_parity(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    artifact_map_path = tmp_path / "artifact_map.json"
    contexts_path = tmp_path / "generated_contexts.json"
    materialization = tmp_path / "materialization.json"
    portfolio = tmp_path / "portfolio.json"
    summary = tmp_path / "action_summary.json"
    work_queue = tmp_path / "materializer_work_queue.json"
    output_dir = tmp_path / "inverse_cell_chain"
    inflate_runtime_dir = tmp_path / "inflate_runtime"
    inflate_work_dir = tmp_path / "inflate_work"
    template = tmp_path / "template.zip"
    action = tmp_path / "inverse_action.json"
    with zipfile.ZipFile(template, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("x", b"base-payload")
    _write_constant_inflate_runtime(inflate_runtime_dir)
    action.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "schema": "inverse_steganalysis_water_bucket_plan.v1",
                    "selected_count": 1,
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plan_path.write_text(json.dumps(_inverse_cell_candidate_plan()), encoding="utf-8")
    artifact_map_path.write_text(
        json.dumps(
            {
                "schema": "final_byte_artifact_map.fixture.v1",
                "artifacts": {
                    INVERSE_SCORER_CELL_TARGET_KIND: {
                        "candidate_archive_template": str(template),
                        "inverse_action_functional": str(action),
                        "raw_contest_video_digest": "f" * 64,
                        "output_dir": str(output_dir),
                        "inflate_runtime_dir": str(inflate_runtime_dir),
                        "source_archive_for_parity": str(template),
                        "inflate_timeout_seconds": 30,
                        "inflate_work_dir": str(inflate_work_dir),
                        "atom_ids": ["inverse_surface_pair0007"],
                        "selected_limit": 1,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--plan",
            str(plan_path),
            "--materializer-artifact-map",
            str(artifact_map_path),
            "--materializer-contexts-out",
            str(contexts_path),
            "--materializer-context-default-output-root",
            str(tmp_path / "unused_output_root"),
            "--materializer-contexts-fail-if-blocked",
            "--materialization-out",
            str(materialization),
            "--portfolio-out",
            str(portfolio),
            "--action-summary-out",
            str(summary),
            "--materializer-work-queue-out",
            str(work_queue),
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
    assert stdout["materializer_contexts_generated"] is True
    assert stdout["materializer_contexts_blocked_count"] == 0
    assert stdout["materializer_work_queue_executable_row_count"] == 1
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    context = contexts["rows"][0]["context"]
    assert context["chain_output_dir"] == str(output_dir)
    assert context["inflate_runtime_dir"] == str(inflate_runtime_dir)
    assert context["source_archive_for_parity"] == str(template)
    assert context["inflate_work_dir"] == str(inflate_work_dir)
    work = json.loads(work_queue.read_text(encoding="utf-8"))
    row = work["rows"][0]
    assert row["tool"] == "tools/run_inverse_scorer_cell_candidate_chain.py"
    assert row["executable"] is True
    assert "--inflate-runtime-dir" in row["command"]
    assert "--source-archive-for-parity" in row["command"]
    assert "--inflate-work-dir" in row["command"]
    assert "--fail-if-inflate-parity-blocked" in row["command"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    command = [sys.executable, str(REPO_ROOT / row["command"][1]), *row["command"][2:]]
    smoke = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    chain = json.loads(smoke.stdout)
    assert chain["schema"] == "inverse_scorer_cell_candidate_chain_v1"
    assert chain["byte_closed_candidate_emitted"] is True
    assert chain["receiver_contract_satisfied"] is True
    assert chain["inflate_parity_satisfied"] is True
    assert chain["score_claim"] is False
    assert chain["ready_for_exact_eval_dispatch"] is False
    assert "candidate_inflate_output_parity_missing" not in chain["readiness_blockers"]
    assert "exact_auth_eval_required_before_score_claim" in chain["readiness_blockers"]
    assert chain["artifacts"]["inflate_parity_probe"]["path"].endswith("inflate_parity_probe.json")
    assert not inflate_work_dir.exists()
    manifest = json.loads((output_dir / INVERSE_CELL_CHAIN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema"] == "inverse_scorer_cell_candidate_chain_v1"
    assert manifest["inflate_parity_satisfied"] is True
    assert (output_dir / "candidate_archive.zip").is_file()
