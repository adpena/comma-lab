# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from comma_lab.scheduler.byte_shaving_campaign_queue import (
    MATERIALIZER_CONTEXTS_SCHEMA,
    build_materializer_work_queue,
    materializer_contexts_from_payload,
)
from comma_lab.scheduler.byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER,
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    ARCHIVE_SECTION_HEADER_ELIDE_MATERIALIZER,
    ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
    ARCHIVE_SECTION_PROCEDURALIZE_MATERIALIZER,
    ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
    ARCHIVE_SECTION_REORDER_MATERIALIZER,
    ARCHIVE_SECTION_REORDER_TARGET_KIND,
    BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
    BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
    INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
    INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
    INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
    INVERSE_SCORER_CELL_MATERIALIZER,
    INVERSE_SCORER_CELL_TARGET_KIND,
    PACKET_MEMBER_MERGE_MATERIALIZER,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    RENDERER_PAYLOAD_DFL1_MATERIALIZER,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    TENSOR_FACTORIZE_MATERIALIZER,
    TENSOR_FACTORIZE_TARGET_KIND,
)
from comma_lab.scheduler.final_byte_operation_contexts import (
    CONTEXT_COMPILER_SCHEMA,
    build_final_byte_operation_contexts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_single_member_zip(path: Path, *, payload: bytes = b"base-payload") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("x", payload)


def _backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": (
                    "materializer_work_queue_required:"
                    f"{ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND}:"
                    "archive_section:section_entropy_recode:"
                    f"{ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER}"
                ),
                "backlog_rank": 1,
                "unit_kind": "archive_section",
                "operation_family": "section_entropy_recode",
                "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                "materializer_id": ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER,
                "source_unit_ids": ["pr106_decoder_packed_brotli"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _archive_section_contract_backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "archive_section_header_elide_fixture",
                "backlog_rank": 1,
                "unit_kind": "archive_section",
                "operation_family": "section_header_elide",
                "target_kind": ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
                "materializer_id": ARCHIVE_SECTION_HEADER_ELIDE_MATERIALIZER,
                "operation_params": {
                    "archive_path": "source.zip",
                    "section_manifest": "sections.json",
                },
                "source_unit_ids": ["decoder_header"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "archive_section_reorder_fixture",
                "backlog_rank": 2,
                "unit_kind": "archive_section",
                "operation_family": "section_reorder",
                "target_kind": ARCHIVE_SECTION_REORDER_TARGET_KIND,
                "materializer_id": ARCHIVE_SECTION_REORDER_MATERIALIZER,
                "operation_params": {
                    "archive_path": "source.zip",
                    "section_manifest": "sections.json",
                },
                "source_unit_ids": ["decoder_sections"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "archive_section_proceduralize_fixture",
                "backlog_rank": 3,
                "unit_kind": "archive_section",
                "operation_family": "section_proceduralize",
                "target_kind": ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
                "materializer_id": ARCHIVE_SECTION_PROCEDURALIZE_MATERIALIZER,
                "operation_params": {
                    "archive_path": "source.zip",
                    "section_manifest": "sections.json",
                },
                "source_unit_ids": ["decoder_constants"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _mixed_backlog() -> dict[str, object]:
    payload = _backlog()
    rows = payload["rows"]
    assert isinstance(rows, list)
    rows.extend(
        [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "packet_member_recompress_fixture",
                "backlog_rank": 2,
                "unit_kind": "packet_member",
                "operation_family": "member_recompress",
                "target_kind": PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
                "materializer_id": PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
                "source_unit_ids": ["payload_member"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "tensor_factorize_fixture",
                "backlog_rank": 3,
                "unit_kind": "tensor",
                "operation_family": "factorize_tensor",
                "target_kind": TENSOR_FACTORIZE_TARGET_KIND,
                "materializer_id": TENSOR_FACTORIZE_MATERIALIZER,
                "source_unit_ids": ["boost_tensor"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ]
    )
    return payload


def _unsupported_high_level_backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "inverse_action_inverse_surface_pair0007",
                "backlog_rank": 1,
                "unit_kind": "scorer_inverse_surface_cell",
                "operation_family": INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
                "target_kind": INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
                "materializer_id": INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
                "source_unit_ids": ["inverse_action_inverse_surface_pair0007"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _inverse_scorer_cell_backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "inverse_scorer_cell_candidate_fixture",
                "backlog_rank": 1,
                "unit_kind": "scorer_inverse_surface_cell",
                "operation_family": "materialize_inverse_scorer_cell_candidate",
                "target_kind": INVERSE_SCORER_CELL_TARGET_KIND,
                "materializer_id": INVERSE_SCORER_CELL_MATERIALIZER,
                "source_unit_ids": ["inverse_surface_pair0007"],
                "source_packet_ir_schemas": ["packet_ir_operation_set_v1"],
                "source_packet_ir_operation_set_ids": ["packetir_opset_0007"],
                "source_packet_ir_source_operation_set_ids": ["opset_0007"],
                "packet_ir_blocker_counts": {
                    "packetir_operation_set_requires_materializer_contexts": 1,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _packet_header_elide_backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "packet_member_zip_header_elide_fixture",
                "backlog_rank": 1,
                "unit_kind": "packet_member",
                "operation_family": "zip_header_elide",
                "target_kind": PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
                "materializer_id": PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER,
                "source_unit_ids": ["payload_member"],
                "operation_params": {"packet_member": "payload.bin"},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _packet_member_merge_backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "packet_member_merge_fixture",
                "backlog_rank": 1,
                "unit_kind": "packet_member",
                "operation_family": "member_merge",
                "target_kind": PACKET_MEMBER_MERGE_TARGET_KIND,
                "materializer_id": PACKET_MEMBER_MERGE_MATERIALIZER,
                "source_unit_ids": ["payload_members"],
                "operation_params": {"member_names": ["a.bin", "b.bin"]},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _renderer_payload_dfl1_backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "renderer_payload_dfl1_fixture",
                "backlog_rank": 1,
                "unit_kind": "packet_member",
                "operation_family": "native_renderer_payload",
                "target_kind": RENDERER_PAYLOAD_DFL1_TARGET_KIND,
                "materializer_id": RENDERER_PAYLOAD_DFL1_MATERIALIZER,
                "source_unit_ids": ["renderer_payload_members"],
                "operation_params": {
                    "member_names": [
                        "renderer.bin",
                        "masks.mkv",
                        "optimized_poses.pt",
                    ],
                    "payload_member_name": "p",
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _artifact_map(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND: {
                "source_archive": str(tmp_path / "source.zip"),
                "section_manifest": str(tmp_path / "sections.json"),
                "target_sections": ["decoder_packed_brotli"],
                "quality": [11],
                "runtime_consumption_proof": str(tmp_path / "runtime_proof.json"),
                "min_free_bytes": 512,
                "allow_size_regression": True,
                "jobs": 4,
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


def _byte_range_backlog() -> dict[str, object]:
    return {
        "schema": "byte_shaving_materializer_backlog.v1",
        "rows": [
            {
                "schema": "byte_shaving_materializer_backlog_row.v1",
                "backlog_key": "byte_range_entropy_fixture",
                "backlog_rank": 1,
                "unit_kind": "byte_range",
                "operation_family": "entropy_recode",
                "target_kind": BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
                "materializer_id": BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
                "source_unit_ids": ["payload_member_range"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _byte_range_artifact_map(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND: {
                "schema_manifest": str(tmp_path / "schema.json"),
                "beam_probe_reports": [
                    str(tmp_path / "beam_a.json"),
                    str(tmp_path / "beam_b.json"),
                ],
                "source_runtime_dir": str(tmp_path / "runtime"),
                "output_dir": str(tmp_path / "chain_out"),
                "source_archive": str(tmp_path / "source.zip"),
                "global_combo_report": str(tmp_path / "global_combo.json"),
                "member_name": "0.bin",
                "retune_brotli_section": "payload",
                "min_free_bytes": 17,
                "fail_if_receiver_blocked": True,
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


def _mixed_artifact_map(tmp_path: Path) -> dict[str, object]:
    payload = _artifact_map(tmp_path)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts[PACKET_MEMBER_RECOMPRESS_TARGET_KIND] = {
        "archive_path": str(tmp_path / "packet_source.zip"),
        "packet_member_manifest": str(tmp_path / "members.json"),
        "member_name": "payload.bin",
        "zip_compresslevel": [9],
        "runtime_consumption_proof": str(tmp_path / "packet_runtime_proof.json"),
        "min_free_bytes": 256,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    artifacts[TENSOR_FACTORIZE_TARGET_KIND] = {
        "archive_path": str(tmp_path / "tensor_source.zip"),
        "tensor_manifest": str(tmp_path / "tensor_manifest.json"),
        "factorization_contract": str(tmp_path / "factor_contract.json"),
        "rank": 1,
        "runtime_consumption_proof": str(tmp_path / "tensor_runtime_proof.json"),
        "allow_size_regression": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return payload


def _inverse_scorer_artifact_map(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            INVERSE_SCORER_CELL_TARGET_KIND: {
                "candidate_archive_template": str(tmp_path / "template.zip"),
                "inverse_action_functional": str(tmp_path / "inverse_action.json"),
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
    }


def _high_level_inverse_action_artifact_map(tmp_path: Path) -> dict[str, object]:
    payload = _artifact_map(tmp_path)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts[INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND] = {
        "candidate_family": "hnerv_variant",
        "archive_grammar": {
            "schema": "archive_grammar.fixture.v1",
            "packet": "single_member_zip",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "receiver_contract_kind": "family_agnostic_archive_section_entropy_recode",
        "runtime_consumption_proof": str(tmp_path / "runtime_proof.json"),
        "operation_set_compiler": {
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "compiled_high_level_section",
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
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return payload


def test_final_byte_context_compiler_emits_consumable_materializer_contexts(
    tmp_path: Path,
) -> None:
    payload = build_final_byte_operation_contexts(
        _backlog(),
        artifact_map=_artifact_map(tmp_path),
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["schema"] == MATERIALIZER_CONTEXTS_SCHEMA
    assert payload["generator_schema"] == CONTEXT_COMPILER_SCHEMA
    assert payload["blocked_context_count"] == 0
    row = payload["rows"][0]
    context = row["context"]
    assert context["output_archive"].endswith(".zip")
    assert context["json_out"].endswith(".json")
    assert context["output_manifest"] == context["json_out"]
    assert context["archive_path"] == context["source_archive"]
    assert context["section_manifest"].endswith("sections.json")
    assert context["target_sections"] == ["decoder_packed_brotli"]
    assert "runtime_consumption_proof" not in context
    assert "runtime_consumption_proof_out" not in context
    assert context["runtime_consumption_proof_missing_hint_ignored"].endswith(
        "runtime_proof.json"
    )
    assert context["min_free_bytes"] == 512
    assert context["allow_size_regression"] is True
    assert context["score_claim"] is False
    assert context["packetir_operation_set_contract"]["schema"] == ("packet_ir_operation_set_bridge_contract.v1")
    assert context["recommended_ir_schema"] == "packet_ir_operation_set_v1"
    assert "runtime_consumption_proof" in context["required_proofs"]
    resolved = materializer_contexts_from_payload(payload)
    assert ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND in resolved
    assert ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER in resolved
    assert "pr106_decoder_packed_brotli" in resolved
    work_queue = build_materializer_work_queue(
        _backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    assert work_queue["executable_row_count"] == 1
    assert work_queue["rows"][0]["tool"] == "tools/run_family_agnostic_materializer.py"
    command = work_queue["rows"][0]["command"]
    assert "--runtime-consumption-proof" not in command
    assert "--runtime-consumption-proof-out" in command
    proof_out_index = command.index("--runtime-consumption-proof-out")
    assert command[proof_out_index + 1].startswith(str(tmp_path / "out"))


def test_final_byte_context_compiler_wires_archive_section_contract_handoffs(
    tmp_path: Path,
) -> None:
    payload = build_final_byte_operation_contexts(
        _archive_section_contract_backlog(),
        artifact_map=None,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["unsupported_backlog_keys"] == []
    assert payload["row_count"] == 3
    header_context = next(
        row["context"]
        for row in payload["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND
    )
    reorder_context = next(
        row["context"]
        for row in payload["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_REORDER_TARGET_KIND
    )
    proceduralize_context = next(
        row["context"]
        for row in payload["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND
    )
    assert header_context["archive_path"] == "source.zip"
    assert header_context["section_manifest"] == "sections.json"
    assert "materializer_context_missing:header_elision_contract" in (
        header_context["context_blockers"]
    )
    assert "materializer_context_missing:runtime_consumption_proof" in (
        header_context["context_blockers"]
    )
    assert "materializer_context_missing:section_order_contract" in (
        reorder_context["context_blockers"]
    )
    assert "materializer_context_missing:procedural_receiver_spec" in (
        proceduralize_context["context_blockers"]
    )

    work_queue = build_materializer_work_queue(
        _archive_section_contract_backlog(),
        repo_root=tmp_path,
        contexts=materializer_contexts_from_payload(payload),
        source_plan_path="plan.json",
    )

    assert work_queue["executable_row_count"] == 0
    assert work_queue["blocked_row_count"] == 3
    for row in work_queue["rows"]:
        assert row["score_claim"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert not any(
            blocker.startswith("materializer_work_queue_adapter_missing:")
            for blocker in row["materialization_blockers"]
        )
        assert row["telemetry"]["receiver_contract_work_order"]["schema"] == (
            "archive_section_receiver_contract_work_order.v1"
        )

    header_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND
    )
    assert "archive_section_header_elide_materializer_requires_byte_closed_adapter" in (
        header_work["materialization_blockers"]
    )
    reorder_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_REORDER_TARGET_KIND
    )
    assert "archive_section_reorder_materializer_requires_byte_closed_adapter" in (
        reorder_work["materialization_blockers"]
    )
    proceduralize_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND
    )
    assert "archive_section_proceduralize_materializer_requires_byte_closed_adapter" in (
        proceduralize_work["materialization_blockers"]
    )


def test_final_byte_context_compiler_preserves_archive_section_contract_proof_signal(
    tmp_path: Path,
) -> None:
    runtime_proof = tmp_path / "runtime_consumption_proof.json"
    runtime_proof.write_text('{"runtime_consumption_proof_passed": true}\n')
    backlog = json.loads(json.dumps(_archive_section_contract_backlog()))
    rows = backlog["rows"]
    rows[0]["operation_params"].update(
        {
            "header_elision_contract": "header_elision_contract.json",
            "runtime_consumption_proof": str(runtime_proof),
        }
    )
    rows[1]["operation_params"].update(
        {
            "section_order_contract": "section_order_contract.json",
            "runtime_consumption_proof": str(runtime_proof),
        }
    )
    rows[2]["operation_params"].update(
        {
            "procedural_receiver_spec": "procedural_receiver_spec.json",
            "runtime_consumption_proof": str(runtime_proof),
        }
    )

    payload = build_final_byte_operation_contexts(
        backlog,
        artifact_map=None,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )
    for row in payload["rows"]:
        assert "materializer_context_missing:runtime_consumption_proof" not in row["context"]["context_blockers"]
        assert row["context"]["runtime_consumption_proof"] == str(runtime_proof)

    work_queue = build_materializer_work_queue(
        backlog,
        repo_root=tmp_path,
        contexts=materializer_contexts_from_payload(payload),
        source_plan_path="plan.json",
    )

    header_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND
    )
    reorder_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_REORDER_TARGET_KIND
    )
    proceduralize_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND
    )
    for row in (header_work, reorder_work, proceduralize_work):
        assert "materializer_context_missing:runtime_consumption_proof" not in (
            row["materialization_blockers"]
        )
        assert row["executable"] is False
        assert row["telemetry"]["receiver_contract_work_order"]["runtime_consumption_proof"] == (
            str(runtime_proof)
        )
    assert header_work["telemetry"]["receiver_contract_work_order"]["header_elision_contract"] == (
        "header_elision_contract.json"
    )
    assert reorder_work["telemetry"]["receiver_contract_work_order"]["section_order_contract"] == (
        "section_order_contract.json"
    )
    assert proceduralize_work["telemetry"]["receiver_contract_work_order"]["procedural_receiver_spec"] == (
        "procedural_receiver_spec.json"
    )


def test_final_byte_context_compiler_uses_existing_runtime_proof_as_input(
    tmp_path: Path,
) -> None:
    runtime_proof = tmp_path / "runtime_proof.json"
    runtime_proof.write_text('{"schema":"fixture_runtime_proof.v1"}', encoding="utf-8")

    payload = build_final_byte_operation_contexts(
        _backlog(),
        artifact_map=_artifact_map(tmp_path),
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    context = payload["rows"][0]["context"]
    assert context["runtime_consumption_proof"] == str(runtime_proof)
    assert "runtime_consumption_proof_out" not in context
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    command = work_queue["rows"][0]["command"]
    assert ["--runtime-consumption-proof", str(runtime_proof)] in [
        command[index : index + 2] for index in range(len(command) - 1)
    ]
    assert "--runtime-consumption-proof-out" not in command


@pytest.mark.parametrize(
    "manifest_key",
    ["parser_section_manifest", "packet_ir_manifest"],
)
def test_final_byte_context_compiler_normalizes_section_manifest_aliases(
    tmp_path: Path,
    manifest_key: str,
) -> None:
    artifact_map = _artifact_map(tmp_path)
    archive_hint = artifact_map["artifacts"][ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]
    assert isinstance(archive_hint, dict)
    manifest_path = archive_hint.pop("section_manifest")
    archive_hint[manifest_key] = manifest_path

    payload = build_final_byte_operation_contexts(
        _backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 0
    context = payload["rows"][0]["context"]
    assert context["section_manifest"] == manifest_path
    assert context[manifest_key] == manifest_path
    assert context["section_manifest_source_key"] == manifest_key
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    command = work_queue["rows"][0]["command"]
    assert work_queue["executable_row_count"] == 1
    assert ["--section-manifest", manifest_path] in [command[index : index + 2] for index in range(len(command) - 1)]


def test_final_byte_context_compiler_covers_packet_member_and_tensor_families(
    tmp_path: Path,
) -> None:
    payload = build_final_byte_operation_contexts(
        _mixed_backlog(),
        artifact_map=_mixed_artifact_map(tmp_path),
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["row_count"] == 3
    assert payload["blocked_context_count"] == 0
    contexts_by_target = {row["target_kind"]: row["context"] for row in payload["rows"]}
    assert (
        contexts_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["packetir_operation_set_contract"]["schema"]
        == "packet_ir_operation_set_bridge_contract.v1"
    )
    assert contexts_by_target[TENSOR_FACTORIZE_TARGET_KIND]["recommended_ir_schema"] == "packet_ir_operation_set_v1"
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _mixed_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    assert work_queue["executable_row_count"] == 3
    rows_by_target = {row["target_kind"]: row for row in work_queue["rows"]}
    assert rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["tool"] == ("tools/run_family_agnostic_materializer.py")
    assert rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["tool"] == ("tools/run_family_agnostic_materializer.py")
    assert [
        "--member-name",
        "payload.bin",
    ] in [
        rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"][index : index + 2]
        for index in range(len(rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"]) - 1)
    ]
    assert "--runtime-consumption-proof-out" in rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"]
    assert "--min-free-bytes" in rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"]
    assert [
        "--factorization-contract",
        str(tmp_path / "factor_contract.json"),
    ] in [
        rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["command"][index : index + 2]
        for index in range(len(rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["command"]) - 1)
    ]
    assert "--allow-size-regression" in rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["command"]
    assert work_queue["score_claim"] is False
    assert work_queue["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_wires_tensor_factorize_receiver_runtime(
    tmp_path: Path,
) -> None:
    artifact_map = _mixed_artifact_map(tmp_path)
    artifacts = artifact_map["artifacts"]
    assert isinstance(artifacts, dict)
    tensor_hints = artifacts[TENSOR_FACTORIZE_TARGET_KIND]
    assert isinstance(tensor_hints, dict)
    tensor_hints["tensor_factorize_source_runtime_dir"] = str(
        tmp_path / "source_runtime"
    )
    tensor_hints["tensor_factorize_runtime_dir_out"] = str(
        tmp_path / "tensor_runtime"
    )
    tensor_hints["tensor_factorize_runtime_manifest_out"] = str(
        tmp_path / "tensor_runtime.json"
    )

    payload = build_final_byte_operation_contexts(
        _mixed_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    contexts_by_target = {row["target_kind"]: row["context"] for row in payload["rows"]}
    context = contexts_by_target[TENSOR_FACTORIZE_TARGET_KIND]
    assert context["tensor_factorize_source_runtime_dir"].endswith("source_runtime")
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _mixed_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    tensor_row = {
        row["target_kind"]: row for row in work_queue["rows"]
    }[TENSOR_FACTORIZE_TARGET_KIND]
    pairs = [
        tensor_row["command"][index : index + 2]
        for index in range(len(tensor_row["command"]) - 1)
    ]
    assert [
        "--tensor-factorize-source-runtime-dir",
        str(tmp_path / "source_runtime"),
    ] in pairs
    assert ["--tensor-factorize-runtime-dir-out", str(tmp_path / "tensor_runtime")] in pairs
    assert [
        "--tensor-factorize-runtime-manifest-out",
        str(tmp_path / "tensor_runtime.json"),
    ] in pairs


def test_final_byte_context_compiler_covers_packet_member_zip_header_elide(
    tmp_path: Path,
) -> None:
    artifact_map = {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND: {
                "archive_path": str(tmp_path / "packet_source.zip"),
                "packet_member_manifest": str(tmp_path / "members.json"),
                "zip_header_contract": str(tmp_path / "zip_header_contract.json"),
                "member_names": ["payload.bin", "weights.bin"],
                "member_selection": "all",
                "min_free_bytes": 128,
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
    payload = build_final_byte_operation_contexts(
        _packet_header_elide_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["row_count"] == 1
    assert payload["blocked_context_count"] == 0
    context = payload["rows"][0]["context"]
    assert context["member_name"] == "payload.bin"
    assert context["member_names"] == ["payload.bin", "weights.bin"]
    assert context["member_selection"] == "all"
    assert context["header_elision_contract"].endswith("zip_header_contract.json")
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _packet_header_elide_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    row = work_queue["rows"][0]
    assert work_queue["executable_row_count"] == 1
    assert row["tool"] == "tools/run_family_agnostic_materializer.py"
    assert [
        "--target-kind",
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    ] in [row["command"][index : index + 2] for index in range(len(row["command"]) - 1)]
    assert [
        "--header-elision-contract",
        str(tmp_path / "zip_header_contract.json"),
    ] in [row["command"][index : index + 2] for index in range(len(row["command"]) - 1)]
    assert "--all-members" in row["command"]
    assert ["--member-names", "payload.bin"] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert ["--member-names", "weights.bin"] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert "--runtime-consumption-proof-out" in row["command"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_covers_packet_member_merge(
    tmp_path: Path,
) -> None:
    artifact_map = {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            PACKET_MEMBER_MERGE_TARGET_KIND: {
                "archive_path": str(tmp_path / "packet_source.zip"),
                "packet_member_manifest": str(tmp_path / "members.json"),
                "member_merge_contract": str(tmp_path / "merge_contract.json"),
                "packet_member_merge_source_runtime_dir": str(tmp_path / "source_runtime"),
                "member_names": ["a.bin", "b.bin"],
                "merged_member_name": "merged.packet",
                "allow_size_regression": True,
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
    payload = build_final_byte_operation_contexts(
        _packet_member_merge_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["row_count"] == 1
    assert payload["blocked_context_count"] == 0
    context = payload["rows"][0]["context"]
    assert context["member_names"] == ["a.bin", "b.bin"]
    assert context["merge_contract"].endswith("merge_contract.json")
    assert context["packet_member_merge_source_runtime_dir"].endswith("source_runtime")
    assert context["merged_member_name"] == "merged.packet"
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _packet_member_merge_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    row = work_queue["rows"][0]
    assert work_queue["executable_row_count"] == 1
    assert row["tool"] == "tools/run_family_agnostic_materializer.py"
    assert [
        "--target-kind",
        PACKET_MEMBER_MERGE_TARGET_KIND,
    ] in [row["command"][index : index + 2] for index in range(len(row["command"]) - 1)]
    assert ["--merge-contract", str(tmp_path / "merge_contract.json")] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert ["--packet-member-merge-source-runtime-dir", str(tmp_path / "source_runtime")] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert ["--merged-member-name", "merged.packet"] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert "--runtime-consumption-proof-out" in row["command"]
    assert "--allow-size-regression" in row["command"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_covers_renderer_payload_dfl1(
    tmp_path: Path,
) -> None:
    artifact_map = {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            RENDERER_PAYLOAD_DFL1_TARGET_KIND: {
                "archive_path": str(tmp_path / "packet_source.zip"),
                "packet_member_manifest": str(tmp_path / "members.json"),
                "renderer_payload_dfl1_source_runtime_dir": str(tmp_path / "source_runtime"),
                "candidate_runtime_dir": str(tmp_path / "candidate_runtime"),
                "file_list_entries": ["0.raw", "1.raw"],
                "expected_full_frame_file_list_sha256": "d" * 64,
                "expected_full_frame_entry_count": 2,
                "full_frame_file_list_source": "fixture_full_file_list",
                "renderer_payload_dfl1_inflate_parity_output_dir": str(
                    tmp_path / "parity_proof"
                ),
                "member_names": [
                    "renderer.bin",
                    "masks.mkv",
                    "optimized_poses.pt",
                ],
                "payload_member_name": "p",
                "allow_size_regression": True,
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
    payload = build_final_byte_operation_contexts(
        _renderer_payload_dfl1_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["row_count"] == 1
    assert payload["blocked_context_count"] == 0
    context = payload["rows"][0]["context"]
    assert context["member_names"] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.pt",
    ]
    assert context["payload_member_name"] == "p"
    assert context["renderer_payload_dfl1_source_runtime_dir"] == str(
        tmp_path / "source_runtime"
    )
    assert context["renderer_payload_dfl1_inflate_runtime_dir"] == str(
        tmp_path / "source_runtime"
    )
    assert context["renderer_payload_dfl1_candidate_runtime_dir"] == str(
        tmp_path / "candidate_runtime"
    )
    assert context["renderer_payload_dfl1_full_frame_file_list_entries"] == [
        "0.raw",
        "1.raw",
    ]
    assert (
        context["renderer_payload_dfl1_expected_full_frame_file_list_sha256"]
        == "d" * 64
    )
    assert context["renderer_payload_dfl1_expected_full_frame_entry_count"] == 2
    assert (
        context["renderer_payload_dfl1_full_frame_file_list_source"]
        == "fixture_full_file_list"
    )
    assert context["renderer_payload_dfl1_inflate_parity_output_dir"] == str(
        tmp_path / "parity_proof"
    )
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _renderer_payload_dfl1_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    row = work_queue["rows"][0]
    assert work_queue["executable_row_count"] == 1
    assert [
        "--target-kind",
        RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    ] in [row["command"][index : index + 2] for index in range(len(row["command"]) - 1)]
    assert ["--payload-member-name", "p"] in [
        row["command"][index : index + 2] for index in range(len(row["command"]) - 1)
    ]
    assert "--runtime-consumption-proof-out" in row["command"]
    assert row["renderer_payload_dfl1_parity_context"] == {
        "source_archive": str(tmp_path / "packet_source.zip"),
        "candidate_archive": context["output_archive"],
        "source_runtime_dir": str(tmp_path / "source_runtime"),
        "candidate_runtime_dir": str(tmp_path / "candidate_runtime"),
        "output_dir": str(tmp_path / "parity_proof"),
        "expected_full_frame_file_list_sha256": "d" * 64,
        "full_frame_file_list_source": "fixture_full_file_list",
        "expected_full_frame_entry_count": 2,
        "file_list_entries": ["0.raw", "1.raw"],
    }
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_blocks_dfl1_missing_parity_identity(
    tmp_path: Path,
) -> None:
    artifact_map = {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            RENDERER_PAYLOAD_DFL1_TARGET_KIND: {
                "archive_path": str(tmp_path / "packet_source.zip"),
                "packet_member_manifest": str(tmp_path / "members.json"),
                "member_names": [
                    "renderer.bin",
                    "masks.mkv",
                    "optimized_poses.pt",
                ],
                "payload_member_name": "p",
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

    payload = build_final_byte_operation_contexts(
        _renderer_payload_dfl1_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["row_count"] == 1
    assert payload["blocked_context_count"] == 1
    assert payload["rows"][0]["context_blockers"] == [
        "materializer_context_missing:renderer_payload_dfl1_source_runtime_dir",
        "materializer_context_missing:renderer_payload_dfl1_candidate_runtime_dir",
        "materializer_context_missing:renderer_payload_dfl1_full_frame_file_list_or_entries",
        "materializer_context_missing:renderer_payload_dfl1_expected_full_frame_file_list_sha256",
        "materializer_context_missing:renderer_payload_dfl1_expected_full_frame_entry_count",
        "materializer_context_missing:renderer_payload_dfl1_full_frame_file_list_source",
    ]
    context = payload["rows"][0]["context"]
    assert context["score_claim"] is False
    assert context["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_blocks_duplicate_dfl1_file_list_entries(
    tmp_path: Path,
) -> None:
    artifact_map = {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            RENDERER_PAYLOAD_DFL1_TARGET_KIND: {
                "archive_path": str(tmp_path / "packet_source.zip"),
                "packet_member_manifest": str(tmp_path / "members.json"),
                "renderer_payload_dfl1_source_runtime_dir": str(
                    tmp_path / "source_runtime"
                ),
                "candidate_runtime_dir": str(tmp_path / "candidate_runtime"),
                "file_list_entries": ["0.raw", "0.raw"],
                "expected_full_frame_file_list_sha256": "d" * 64,
                "expected_full_frame_entry_count": 2,
                "full_frame_file_list_source": "fixture_full_file_list",
                "member_names": ["renderer.bin"],
                "payload_member_name": "p",
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

    payload = build_final_byte_operation_contexts(
        _renderer_payload_dfl1_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 1
    assert (
        "materializer_context_duplicate:renderer_payload_dfl1_full_frame_file_list_entries"
        in payload["rows"][0]["context_blockers"]
    )


def test_final_byte_context_compiler_uses_inline_operation_params(
    tmp_path: Path,
) -> None:
    backlog = _mixed_backlog()
    rows = backlog["rows"]
    assert isinstance(rows, list)
    rows[0]["operation_params"] = {"section_name": "decoder_packed_brotli"}
    rows[1]["operation_params"] = {"member_name": "payload.bin"}
    rows[2]["operation_params"] = {"rank": 1}

    artifact_map = _mixed_artifact_map(tmp_path)
    artifacts = artifact_map["artifacts"]
    assert isinstance(artifacts, dict)
    archive_hints = artifacts[ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]
    packet_hints = artifacts[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]
    tensor_hints = artifacts[TENSOR_FACTORIZE_TARGET_KIND]
    assert isinstance(archive_hints, dict)
    assert isinstance(packet_hints, dict)
    assert isinstance(tensor_hints, dict)
    del archive_hints["target_sections"]
    del packet_hints["member_name"]
    del tensor_hints["factorization_contract"]
    del tensor_hints["rank"]

    payload = build_final_byte_operation_contexts(
        backlog,
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 0
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        backlog,
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )

    assert work_queue["executable_row_count"] == 3
    rows_by_target = {row["target_kind"]: row for row in work_queue["rows"]}
    assert [
        "--section-name",
        "decoder_packed_brotli",
    ] in [
        rows_by_target[ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]["command"][index : index + 2]
        for index in range(len(rows_by_target[ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]["command"]) - 1)
    ]
    assert [
        "--member-name",
        "payload.bin",
    ] in [
        rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"][index : index + 2]
        for index in range(len(rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"]) - 1)
    ]
    assert [
        "--rank",
        "1",
    ] in [
        rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["command"][index : index + 2]
        for index in range(len(rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["command"]) - 1)
    ]
    assert work_queue["score_claim"] is False
    assert work_queue["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_covers_inverse_scorer_cell_candidate(
    tmp_path: Path,
) -> None:
    _write_single_member_zip(tmp_path / "template.zip")
    backlog = _inverse_scorer_cell_backlog()
    rows = backlog["rows"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    row["packet_ir_blocker_counts"] = {
        "packetir_operation_set_requires_materializer_contexts": "1",
        "zero_count": 0,
        "negative_count": -3,
        "bool_count": True,
        "nonnumeric_count": "not-an-int",
    }

    payload = build_final_byte_operation_contexts(
        backlog,
        artifact_map=_inverse_scorer_artifact_map(tmp_path),
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["row_count"] == 1
    assert payload["blocked_context_count"] == 0
    row = payload["rows"][0]
    context = row["context"]
    assert row["source_packet_ir_operation_set_ids"] == ["packetir_opset_0007"]
    assert context["source_packet_ir_operation_set_ids"] == ["packetir_opset_0007"]
    assert context["packet_ir_blocker_counts"] == {
        "packetir_operation_set_requires_materializer_contexts": 1,
    }
    assert context["candidate_archive_template"].endswith("template.zip")
    assert context["inverse_action_functional"].endswith("inverse_action.json")
    assert context["output_archive"].endswith(".zip")
    assert context["manifest_out"].endswith(".json")
    assert context["atom_ids"] == ["inverse_surface_pair0007"]
    assert context["score_claim"] is False

    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _inverse_scorer_cell_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )

    assert work_queue["executable_row_count"] == 1
    work = work_queue["rows"][0]
    assert work["tool"] == "tools/materialize_inverse_scorer_cell_candidate.py"
    assert "--candidate-archive-template" in work["command"]
    assert "--inverse-action-functional" in work["command"]
    assert "--atom-id" in work["command"]
    assert work["source_packet_ir_operation_set_ids"] == ["packetir_opset_0007"]
    assert work["score_claim"] is False
    assert work["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_blocks_inverse_scorer_chain_without_parity(
    tmp_path: Path,
) -> None:
    artifact_map = _inverse_scorer_artifact_map(tmp_path)
    artifacts = artifact_map["artifacts"]
    assert isinstance(artifacts, dict)
    inverse_hints = artifacts[INVERSE_SCORER_CELL_TARGET_KIND]
    assert isinstance(inverse_hints, dict)
    inverse_hints["output_dir"] = str(tmp_path / "chain")

    payload = build_final_byte_operation_contexts(
        _inverse_scorer_cell_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 1
    assert (
        "materializer_context_missing:inverse_scorer_cell_inflate_parity_context"
        in payload["rows"][0]["context_blockers"]
    )
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _inverse_scorer_cell_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    assert work_queue["executable_row_count"] == 0
    assert (
        "materializer_context_missing:inverse_scorer_cell_inflate_parity_context"
        in work_queue["rows"][0]["materialization_blockers"]
    )


def test_final_byte_context_compiler_rejects_truthy_authority(
    tmp_path: Path,
) -> None:
    artifact_map = _artifact_map(tmp_path)
    artifact_map["artifacts"][ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]["score_claim"] = True

    with pytest.raises(ValueError, match="score_claim"):
        build_final_byte_operation_contexts(
            _backlog(),
            artifact_map=artifact_map,
            repo_root=tmp_path,
            default_output_root=tmp_path / "out",
        )


def test_final_byte_context_compiler_carries_missing_artifact_blockers(
    tmp_path: Path,
) -> None:
    artifact_map = _artifact_map(tmp_path)
    del artifact_map["artifacts"][ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]["section_manifest"]

    payload = build_final_byte_operation_contexts(
        _backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    row = payload["rows"][0]
    assert payload["blocked_context_count"] == 1
    assert "materializer_context_missing:section_manifest" in row["context_blockers"]
    assert row["context"]["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_wires_byte_range_entropy_recode_chain(
    tmp_path: Path,
) -> None:
    payload = build_final_byte_operation_contexts(
        _byte_range_backlog(),
        artifact_map=_byte_range_artifact_map(tmp_path),
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 0
    assert payload["unsupported_backlog_keys"] == []
    row = payload["rows"][0]
    assert "unsupported" not in row
    context = row["context"]
    assert context["schema_manifest"] == str(tmp_path / "schema.json")
    assert context["beam_probe_reports"] == [
        str(tmp_path / "beam_a.json"),
        str(tmp_path / "beam_b.json"),
    ]
    assert context["source_runtime_dir"] == str(tmp_path / "runtime")
    assert context["output_dir"] == str(tmp_path / "chain_out")
    assert context["member_name"] == "0.bin"
    assert context["fail_if_receiver_blocked"] is True
    assert context["score_claim"] is False
    assert context["ready_for_exact_eval_dispatch"] is False

    work_queue = build_materializer_work_queue(
        _byte_range_backlog(),
        repo_root=tmp_path,
        contexts=materializer_contexts_from_payload(payload),
        source_plan_path="plan.json",
    )

    assert work_queue["executable_row_count"] == 1
    work = work_queue["rows"][0]
    assert work["executable"] is True
    assert work["tool"] == "tools/run_byte_range_entropy_recode_chain.py"
    assert work["command"][:4] == [
        ".venv/bin/python",
        "tools/run_byte_range_entropy_recode_chain.py",
        "--schema-manifest",
        str(tmp_path / "schema.json"),
    ]
    assert work["command"].count("--beam-probe-report") == 2
    assert "--fail-if-receiver-blocked" in work["command"]
    assert work["score_claim"] is False
    assert work["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_blocks_incomplete_byte_range_context(
    tmp_path: Path,
) -> None:
    artifact_map = {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND: {
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

    payload = build_final_byte_operation_contexts(
        _byte_range_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=None,
    )

    assert payload["blocked_context_count"] == 1
    blockers = payload["rows"][0]["context_blockers"]
    assert blockers == [
        "materializer_context_missing:schema_manifest",
        "materializer_context_missing:beam_probe_reports",
        "materializer_context_missing:source_runtime_dir",
        "materializer_context_missing:output_dir",
    ]
    work_queue = build_materializer_work_queue(
        _byte_range_backlog(),
        repo_root=tmp_path,
        contexts=materializer_contexts_from_payload(payload),
        source_plan_path="plan.json",
    )
    assert work_queue["executable_row_count"] == 0
    assert set(blockers).issubset(set(work_queue["rows"][0]["materialization_blockers"]))


def test_final_byte_context_compiler_accepts_byte_range_context_aliases(
    tmp_path: Path,
) -> None:
    artifact_map = {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND: {
                "byte_range_schema_manifest": "schema.json",
                "beam_probe_report": "beam.json",
                "runtime_dir": "runtime",
                "chain_output_dir": "chain",
                "archive_path": "source.zip",
                "archive_member_name": "payload.bin",
                "target_section": "packed_payload",
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

    payload = build_final_byte_operation_contexts(
        _byte_range_backlog(),
        artifact_map=artifact_map,
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 0
    context = payload["rows"][0]["context"]
    assert context["schema_manifest"] == "schema.json"
    assert context["beam_probe_reports"] == ["beam.json"]
    assert context["source_runtime_dir"] == "runtime"
    assert context["output_dir"] == "chain"
    assert context["source_archive"] == "source.zip"
    assert context["member_name"] == "payload.bin"
    assert context["retune_brotli_section"] == "packed_payload"


def test_final_byte_context_compiler_rejects_truthy_byte_range_authority(
    tmp_path: Path,
) -> None:
    artifact_map = _byte_range_artifact_map(tmp_path)
    artifacts = artifact_map["artifacts"]
    assert isinstance(artifacts, dict)
    byte_range_context = artifacts[BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND]
    assert isinstance(byte_range_context, dict)
    byte_range_context["score_claim"] = True

    with pytest.raises(ValueError, match="score_claim"):
        build_final_byte_operation_contexts(
            _byte_range_backlog(),
            artifact_map=artifact_map,
            repo_root=tmp_path,
            default_output_root=tmp_path / "out",
        )


def test_final_byte_context_compiler_blocks_high_level_inverse_action_without_compiler_hint(
    tmp_path: Path,
) -> None:
    payload = build_final_byte_operation_contexts(
        _unsupported_high_level_backlog(),
        artifact_map={"schema": "empty.fixture.v1", "score_claim": False},
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 1
    assert payload["unsupported_backlog_keys"] == []
    row = payload["rows"][0]
    assert "unsupported" not in row
    assert "materializer_context_missing:operation_set_compiler" in row["context_blockers"]
    bridge = row["context"]["packetir_compiler_bridge"]
    assert bridge["schema"] == "final_byte_packetir_compiler_bridge_hint.v1"
    assert bridge["canonical_packet_compiler_module"] == ("tac.packet_compiler.deterministic_compiler")
    assert bridge["packetir_operation_set_contract"]["schema"] == ("packet_ir_operation_set_bridge_contract.v1")
    assert bridge["packetir_operation_set_contract"]["score_claim"] is False
    assert bridge["packetir_operation_set_contract"]["ready_for_exact_eval_dispatch"] is False
    assert bridge["recommended_ir_schema"] == "packet_ir_operation_set_v1"
    assert "runtime_consumption_proof" in bridge["required_proofs"]
    assert "packetir_bridge_requires_operation_set_ir" in bridge["blockers"]
    assert bridge["score_claim"] is False
    resolved = materializer_contexts_from_payload(payload)
    assert (
        resolved["inverse_action_inverse_surface_pair0007"]["packetir_compiler_bridge"]["recommended_ir_schema"]
        == "packet_ir_operation_set_v1"
    )
    work_queue = build_materializer_work_queue(
        _unsupported_high_level_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    work_row = work_queue["rows"][0]
    assert work_queue["executable_row_count"] == 0
    assert (
        "materializer_context_missing:operation_set_compiler"
        in work_row["materialization_blockers"]
    )
    assert not any(
        blocker.startswith("materializer_work_queue_adapter_missing:")
        for blocker in work_row["materialization_blockers"]
    )
    assert work_row["score_claim"] is False


def test_final_byte_context_compiler_lowers_high_level_inverse_action_to_packetir_context(
    tmp_path: Path,
) -> None:
    payload = build_final_byte_operation_contexts(
        _unsupported_high_level_backlog(),
        artifact_map=_high_level_inverse_action_artifact_map(tmp_path),
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 0
    assert payload["unsupported_backlog_keys"] == []
    assert payload["row_count"] == 2
    high_level_context = payload["rows"][0]["context"]
    assert high_level_context["packet_ir_operation_set"]["schema"] == "packet_ir_operation_set_v1"
    assert high_level_context["score_claim"] is False
    concrete = payload["rows"][1]
    assert concrete["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    assert concrete["context"]["archive_path"] == str(tmp_path / "source.zip")
    assert concrete["context"]["section_manifest"] == str(tmp_path / "sections.json")
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _unsupported_high_level_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    assert work_queue["executable_row_count"] == 1
    assert work_queue["blocked_row_count"] == 1
    assert not any(
        "materializer_work_queue_adapter_missing" in "\n".join(row["materialization_blockers"])
        for row in work_queue["rows"]
    )
    concrete_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    )
    assert concrete_work["executable"] is True
    assert concrete_work["tool"] == "tools/run_family_agnostic_materializer.py"
    assert concrete_work["score_claim"] is False
    high_level_work = next(
        row
        for row in work_queue["rows"]
        if row["target_kind"] == INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND
    )
    assert high_level_work["executable"] is False
    assert (
        "inverse_action_high_level_context_lowered_to_packet_ir_materializer_rows"
        in high_level_work["materialization_blockers"]
    )


def test_build_final_byte_operation_contexts_cli_writes_json(
    tmp_path: Path,
) -> None:
    backlog_path = tmp_path / "backlog.json"
    artifact_map_path = tmp_path / "artifact_map.json"
    output_path = tmp_path / "contexts.json"
    backlog_path.write_text(json.dumps(_backlog()), encoding="utf-8")
    artifact_map_path.write_text(json.dumps(_artifact_map(tmp_path)), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_final_byte_operation_contexts.py"),
            "--backlog",
            str(backlog_path),
            "--artifact-map",
            str(artifact_map_path),
            "--output",
            str(output_path),
            "--repo-root",
            str(tmp_path),
            "--default-output-root",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == MATERIALIZER_CONTEXTS_SCHEMA
    assert materializer_contexts_from_payload(payload)


def test_build_final_byte_operation_contexts_cli_fail_if_blocked_catches_unsupported(
    tmp_path: Path,
) -> None:
    backlog_path = tmp_path / "backlog.json"
    artifact_map_path = tmp_path / "artifact_map.json"
    output_path = tmp_path / "contexts.json"
    backlog_path.write_text(
        json.dumps(_unsupported_high_level_backlog()),
        encoding="utf-8",
    )
    artifact_map_path.write_text(
        json.dumps({"schema": "empty.fixture.v1", "score_claim": False}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_final_byte_operation_contexts.py"),
            "--backlog",
            str(backlog_path),
            "--artifact-map",
            str(artifact_map_path),
            "--output",
            str(output_path),
            "--repo-root",
            str(tmp_path),
            "--default-output-root",
            str(tmp_path / "out"),
            "--fail-if-blocked",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["blocked_context_count"] == 1
