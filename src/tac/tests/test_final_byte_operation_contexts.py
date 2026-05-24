# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
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
    INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
    INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
    INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    TENSOR_FACTORIZE_MATERIALIZER,
    TENSOR_FACTORIZE_TARGET_KIND,
)
from comma_lab.scheduler.final_byte_operation_contexts import (
    CONTEXT_COMPILER_SCHEMA,
    build_final_byte_operation_contexts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _artifact_map(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "final_byte_artifact_map.fixture.v1",
        "artifacts": {
            ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND: {
                "source_archive": str(tmp_path / "source.zip"),
                "section_manifest": str(tmp_path / "sections.json"),
                "target_sections": ["decoder_packed_brotli"],
                "quality": [11],
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


def _mixed_artifact_map(tmp_path: Path) -> dict[str, object]:
    payload = _artifact_map(tmp_path)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts[PACKET_MEMBER_RECOMPRESS_TARGET_KIND] = {
        "archive_path": str(tmp_path / "packet_source.zip"),
        "packet_member_manifest": str(tmp_path / "members.json"),
        "member_name": "payload.bin",
        "zip_compresslevel": [9],
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
    assert context["score_claim"] is False
    assert context["packetir_operation_set_contract"]["schema"] == (
        "packet_ir_operation_set_bridge_contract.v1"
    )
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
    assert ["--section-manifest", manifest_path] in [
        command[index : index + 2] for index in range(len(command) - 1)
    ]


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
    contexts_by_target = {
        row["target_kind"]: row["context"]
        for row in payload["rows"]
    }
    assert contexts_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND][
        "packetir_operation_set_contract"
    ]["schema"] == "packet_ir_operation_set_bridge_contract.v1"
    assert contexts_by_target[TENSOR_FACTORIZE_TARGET_KIND][
        "recommended_ir_schema"
    ] == "packet_ir_operation_set_v1"
    resolved = materializer_contexts_from_payload(payload)
    work_queue = build_materializer_work_queue(
        _mixed_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    assert work_queue["executable_row_count"] == 3
    rows_by_target = {
        row["target_kind"]: row
        for row in work_queue["rows"]
    }
    assert rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["tool"] == (
        "tools/run_family_agnostic_materializer.py"
    )
    assert rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["tool"] == (
        "tools/run_family_agnostic_materializer.py"
    )
    assert [
        "--member-name",
        "payload.bin",
    ] in [
        rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"][index : index + 2]
        for index in range(
            len(rows_by_target[PACKET_MEMBER_RECOMPRESS_TARGET_KIND]["command"]) - 1
        )
    ]
    assert [
        "--factorization-contract",
        str(tmp_path / "factor_contract.json"),
    ] in [
        rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["command"][index : index + 2]
        for index in range(len(rows_by_target[TENSOR_FACTORIZE_TARGET_KIND]["command"]) - 1)
    ]
    assert work_queue["score_claim"] is False
    assert work_queue["ready_for_exact_eval_dispatch"] is False


def test_final_byte_context_compiler_rejects_truthy_authority(
    tmp_path: Path,
) -> None:
    artifact_map = _artifact_map(tmp_path)
    artifact_map["artifacts"][ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND][
        "score_claim"
    ] = True

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
    del artifact_map["artifacts"][ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND][
        "section_manifest"
    ]

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


def test_final_byte_context_compiler_turns_unsupported_rows_into_blocked_contexts(
    tmp_path: Path,
) -> None:
    payload = build_final_byte_operation_contexts(
        _unsupported_high_level_backlog(),
        artifact_map={"schema": "empty.fixture.v1", "score_claim": False},
        repo_root=tmp_path,
        default_output_root=tmp_path / "out",
    )

    assert payload["blocked_context_count"] == 1
    assert payload["unsupported_backlog_keys"] == [
        "inverse_action_inverse_surface_pair0007"
    ]
    row = payload["rows"][0]
    assert row["unsupported"] is True
    assert "final_byte_context_compiler_unsupported_backlog_row" in row[
        "context_blockers"
    ]
    bridge = row["context"]["packetir_compiler_bridge"]
    assert bridge["schema"] == "final_byte_packetir_compiler_bridge_hint.v1"
    assert bridge["canonical_packet_compiler_module"] == (
        "tac.packet_compiler.deterministic_compiler"
    )
    assert bridge["packetir_operation_set_contract"]["schema"] == (
        "packet_ir_operation_set_bridge_contract.v1"
    )
    assert bridge["packetir_operation_set_contract"]["score_claim"] is False
    assert bridge["packetir_operation_set_contract"][
        "ready_for_exact_eval_dispatch"
    ] is False
    assert bridge["recommended_ir_schema"] == "packet_ir_operation_set_v1"
    assert "runtime_consumption_proof" in bridge["required_proofs"]
    assert "packetir_bridge_requires_operation_set_ir" in bridge["blockers"]
    assert bridge["score_claim"] is False
    resolved = materializer_contexts_from_payload(payload)
    assert resolved["inverse_action_inverse_surface_pair0007"][
        "packetir_compiler_bridge"
    ]["recommended_ir_schema"] == "packet_ir_operation_set_v1"
    work_queue = build_materializer_work_queue(
        _unsupported_high_level_backlog(),
        repo_root=tmp_path,
        contexts=resolved,
        source_plan_path="plan.json",
    )
    work_row = work_queue["rows"][0]
    assert work_queue["executable_row_count"] == 0
    assert "final_byte_context_compiler_unsupported_backlog_row" in work_row[
        "materialization_blockers"
    ]
    assert any(
        blocker.startswith("materializer_work_queue_adapter_missing:")
        for blocker in work_row["materialization_blockers"]
    )
    assert work_row["score_claim"] is False


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
