# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_acquisition_batch import (
    MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA,
    MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA,
    MLXAcquisitionBatchError,
    build_mlx_acquisition_batch_from_selection,
    validate_mlx_acquisition_batch,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS


def _selection(row_overrides: dict[str, object]) -> dict[str, object]:
    row = {
        "schema": MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA,
        **PROXY_FALSE_AUTHORITY_FIELDS,
        "candidate_generation_only": True,
        "row_id": "packet_row",
        "candidate_id": "mlx_packet_candidate",
        "family": "packetir",
        "pair_indices": [10, 11],
        "normalized_full_video_scorer_gain_vs_baseline": 0.00002,
        "added_archive_bytes": -3,
        **row_overrides,
    }
    return {
        "schema": MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA,
        **PROXY_FALSE_AUTHORITY_FIELDS,
        "candidate_generation_only": True,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "selected_rows": [row],
    }


def test_mlx_acquisition_batch_preserves_operation_set_compiler_hints() -> None:
    batch = build_mlx_acquisition_batch_from_selection(
        _selection(
            {
                "operation_set_compiler": {
                    "schema": "inverse_action_operation_set_compiler_hint.v1",
                    "selected_operations": [
                        {
                            "unit_id": "payload_member",
                            "unit_kind": "packet_member",
                            "operation_id": "recompress_payload",
                            "operation_family": "member_recompress",
                            "target_kind": "packet_member_recompress_v1",
                            "member_name": "payload.bin",
                            "candidate_saved_bytes": 3,
                            "predicted_quality_score_delta": -0.00002,
                        }
                    ],
                }
            }
        ),
        source_path="selection.json",
    )

    operation_set = batch["operation_sets"][0]
    compiler = operation_set["operation_set_compiler"]
    operation = compiler["selected_operations"][0]
    assert compiler["schema"] == "inverse_action_operation_set_compiler_hint.v1"
    assert compiler["candidate_saved_bytes"] == 3
    assert operation["target_kind"] == "packet_member_recompress_v1"
    assert operation["operation_family"] == "member_recompress"
    assert operation["params"]["member_name"] == "payload.bin"
    assert operation["params"]["source_row_id"] == "packet_row"
    assert "compiled_from_mlx_acquisition_operation_set_compiler" in operation["blockers"]
    assert operation_set["score_claim"] is False
    assert compiler["score_claim"] is False
    assert validate_mlx_acquisition_batch(batch)["operation_sets"][0] == operation_set


def test_mlx_acquisition_batch_rejects_compiler_hint_without_target_kind() -> None:
    with pytest.raises(MLXAcquisitionBatchError, match=r"target_kind is required"):
        build_mlx_acquisition_batch_from_selection(
            _selection(
                {
                    "operation_set_compiler": {
                        "schema": "inverse_action_operation_set_compiler_hint.v1",
                        "selected_operations": [{"operation_id": "missing_target"}],
                    }
                }
            )
        )


def test_mlx_acquisition_batch_rejects_truthy_authority_in_compiler_hint() -> None:
    with pytest.raises(ValueError, match=r"forbidden truthy authority"):
        build_mlx_acquisition_batch_from_selection(
            _selection(
                {
                    "operation_set_compiler": {
                        "schema": "inverse_action_operation_set_compiler_hint.v1",
                        "selected_operations": [
                            {
                                "operation_id": "bad_authority",
                                "target_kind": "packet_member_recompress_v1",
                                "promotable": True,
                            }
                        ],
                    }
                }
            )
        )
