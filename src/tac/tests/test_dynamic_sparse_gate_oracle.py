# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.dynamic_sparse_gate_oracle import (
    DynamicSparseGateOracleError,
    dynamic_sparse_skip_mixture,
    operation_set_compiler_hint_from_channel_gate_scores,
    operation_set_compiler_hint_from_gate_scores,
    operation_set_compiler_hint_from_observation_feedback,
)
from tac.optimization.inverse_steganalysis_operation_set_compiler import (
    packet_ir_operation_set_from_compiler_hint,
)
from tools import build_dynamic_sparse_gate_compiler_hint as gate_cli


def test_dynamic_sparse_skip_mixture_zero_init_is_exact_noop() -> None:
    base = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    sources = {
        "early_hidden": np.ones_like(base),
        "mid_hidden": np.full_like(base, 2.0),
    }
    w1 = np.full((3, 4), 0.125, dtype=np.float32)
    w2 = np.zeros((4, 2), dtype=np.float32)

    result = dynamic_sparse_skip_mixture(base, sources, w1=w1, w2=w2)

    assert result["schema"] == "dynamic_sparse_skip_mixture_oracle.v1"
    assert result["source_order"] == ["early_hidden", "mid_hidden"]
    assert result["zero_init_noop_proven"] is True
    np.testing.assert_array_equal(result["coefficients"], np.zeros((2, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(result["delta"], np.zeros_like(base))
    np.testing.assert_array_equal(result["mixed"], base)
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False


def test_dynamic_sparse_gate_hint_lowers_to_packetir_false_authority() -> None:
    coefficients = np.array(
        [
            [0.1, 0.9],
            [0.3, 0.2],
        ],
        dtype=np.float32,
    )
    candidates = [
        {
            "unit_id": "decoder_blob",
            "target_kind": "archive_section_entropy_recode_v1",
            "candidate_saved_bytes": 40,
        },
        {
            "unit_id": "payload_member",
            "target_kind": "packet_member_recompress_v1",
            "candidate_saved_bytes": 20,
        },
    ]

    hint = operation_set_compiler_hint_from_gate_scores(
        candidates,
        coefficients,
        operation_set_id="dynamic_sparse_gate_fixture",
        source_ids=["decoder_blob_source", "payload_member_source"],
        max_operations=1,
        lane_id="codex_dynamic_sparse_gate_oracle_20260525",
    )
    packet_ir = packet_ir_operation_set_from_compiler_hint(
        hint,
        source_backlog_key="dynamic_sparse_gate_backlog",
        source_unit_ids=["decoder_blob", "payload_member"],
    )

    selected = hint["selected_operations"][0]
    assert selected["unit_id"] == "payload_member"
    assert selected["dynamic_gate_source_id"] == "payload_member_source"
    assert selected["params"]["dynamic_sparse_gate"]["abs_mean_coefficient"] == pytest.approx(0.55)
    assert selected["score_claim"] is False
    assert hint["score_claim"] is False
    assert packet_ir["schema"] == "packet_ir_operation_set_v1"
    assert packet_ir["selected_operations"][0]["target_kind"] == "packet_member_recompress_v1"
    assert packet_ir["selected_operations"][0]["score_claim"] is False
    assert "packetir_operation_set_requires_runtime_consumption_proof" in packet_ir["blockers"]


def test_channel_gate_hint_preserves_mudd_style_source_channel_signal() -> None:
    coefficients = np.array(
        [
            [
                [0.10, 0.20, 0.80],
                [0.05, 0.40, 0.15],
            ],
            [
                [0.20, 0.10, 0.90],
                [0.10, 0.35, 0.20],
            ],
        ],
        dtype=np.float32,
    )
    candidates = [
        {
            "unit_id": "late_value_h9",
            "target_kind": "packet_member_recompress_v1",
            "dynamic_gate_channel_id": "value",
            "dynamic_gate_source_id": "h9",
            "candidate_saved_bytes": 50,
        },
        {
            "unit_id": "late_residual_h7",
            "target_kind": "archive_section_entropy_recode_v1",
            "dynamic_gate_channel_id": "residual",
            "dynamic_gate_source_id": "h7",
            "candidate_saved_bytes": 80,
        },
    ]

    hint = operation_set_compiler_hint_from_channel_gate_scores(
        candidates,
        coefficients,
        operation_set_id="mudd_sparse_topology_fixture",
        source_ids=["x0", "h7", "h9"],
        channel_ids=["value", "residual"],
        max_operations=1,
        lane_id="codex_dynamic_sparse_gate_oracle_20260525",
        shared_projection_id="mudd_pr259_shared_w1",
        topology_id="mudd_pr259_late_layer_subset",
    )
    packet_ir = packet_ir_operation_set_from_compiler_hint(
        hint,
        source_backlog_key="dynamic_sparse_channel_gate_backlog",
        source_unit_ids=["late_value_h9", "late_residual_h7"],
    )

    selected = hint["selected_operations"][0]
    nested = selected["params"]["dynamic_sparse_channel_gate"]
    assert hint["source_schema"] == "dynamic_sparse_channel_gate_operation_selection.v1"
    assert hint["selection_source"] == "dynamic_sparse_channel_gate_scores"
    assert selected["unit_id"] == "late_value_h9"
    assert selected["dynamic_gate_channel_id"] == "value"
    assert selected["dynamic_gate_source_id"] == "h9"
    assert nested["shared_projection_id"] == "mudd_pr259_shared_w1"
    assert nested["topology_id"] == "mudd_pr259_late_layer_subset"
    assert nested["abs_mean_coefficient"] == pytest.approx(0.85)
    assert selected["score_claim"] is False
    assert packet_ir["selected_operations"][0]["params"]["dynamic_sparse_channel_gate"][
        "score_claim"
    ] is False


def test_dynamic_sparse_gate_rejects_truthy_authority() -> None:
    with pytest.raises(DynamicSparseGateOracleError, match="score_claim"):
        operation_set_compiler_hint_from_gate_scores(
            [{"unit_id": "bad", "target_kind": "packet_member_recompress_v1", "score_claim": True}],
            np.array([1.0], dtype=np.float32),
            operation_set_id="bad_authority",
        )


def test_channel_gate_hint_rejects_ambiguous_unkeyed_candidates() -> None:
    with pytest.raises(DynamicSparseGateOracleError, match="len\\(channel_ids\\)"):
        operation_set_compiler_hint_from_channel_gate_scores(
            [{"unit_id": "too_few", "target_kind": "packet_member_recompress_v1"}],
            np.ones((2, 2), dtype=np.float32),
            operation_set_id="ambiguous",
            source_ids=["x0", "h9"],
            channel_ids=["value", "residual"],
        )


def test_observation_feedback_builds_channel_gate_hint() -> None:
    observations = [
        {
            "schema": "inverse_steganalysis_observation.v1",
            "observation_id": "obs_value_h9",
            "observation_kind": "family_agnostic_materializer_empirical_observation",
            "candidate_id": "candidate_value_h9",
            "axis": "[local-materializer advisory]",
            "target_kind": "packet_member_recompress_v1",
            "materializer_id": "packet_member_recompress_adapter",
            "source_unit_ids": ["mudd_value_h9"],
            "saved_bytes": 90,
            "observed_rate_gain": 0.00006,
            "rate_positive": True,
            "receiver_contract_satisfied": True,
            "inflate_parity_satisfied": True,
            "elapsed_seconds": 3.0,
            "score_claim": False,
            "promotion_eligible": False,
        },
        {
            "schema": "inverse_steganalysis_observation.v1",
            "observation_id": "obs_cost",
            "candidate_id": "candidate_cost",
            "axis": "[local-materializer advisory]",
            "target_kind": "archive_section_entropy_recode_v1",
            "source_unit_ids": ["mudd_residual_h7"],
            "saved_bytes": -8,
            "observed_rate_gain": 0.0,
            "rate_positive": False,
            "score_claim": False,
        },
    ]

    hint = operation_set_compiler_hint_from_observation_feedback(
        observations,
        operation_set_id="observation_feedback_fixture",
        max_operations=1,
        lane_id="codex_dynamic_sparse_observation_feedback_20260525",
    )
    packet_ir = packet_ir_operation_set_from_compiler_hint(
        hint,
        source_backlog_key="dynamic_sparse_observation_feedback_backlog",
        source_unit_ids=["mudd_value_h9"],
    )

    selected = hint["selected_operations"][0]
    feedback = selected["params"]["dynamic_sparse_observation_feedback"]
    gate = selected["params"]["dynamic_sparse_channel_gate"]
    assert hint["selection_source"] == "dynamic_sparse_observation_feedback"
    assert hint["observation_feedback"]["observation_count"] == 2
    assert hint["observation_feedback"]["selectable_observation_count"] == 1
    assert selected["unit_id"] == "mudd_value_h9"
    assert selected["target_kind"] == "packet_member_recompress_v1"
    assert selected["candidate_saved_bytes"] == 90
    assert feedback["observation_id"] == "obs_value_h9"
    assert feedback["channel_scores"]["rate_saving"] == pytest.approx(0.00006)
    assert gate["channel_id"] == "rate_saving"
    assert gate["source_id"] == "mudd_value_h9"
    assert selected["score_claim"] is False
    assert packet_ir["selected_operations"][0]["score_claim"] is False


def test_dynamic_sparse_gate_compiler_hint_cli_writes_channel_hint(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.json"
    coefficients = tmp_path / "coefficients.json"
    out = tmp_path / "hint.json"
    candidates.write_text(
        json.dumps(
            {
                "operation_candidates": [
                    {
                        "unit_id": "value_h9",
                        "target_kind": "packet_member_recompress_v1",
                        "dynamic_gate_channel_id": "value",
                        "dynamic_gate_source_id": "h9",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    coefficients.write_text(
        json.dumps(
            {
                "coefficients": [
                    [[0.0, 0.0, 0.7], [0.0, 0.0, 0.1]],
                    [[0.0, 0.0, 0.9], [0.0, 0.0, 0.2]],
                ],
                "source_ids": ["x0", "h7", "h9"],
                "channel_ids": ["value", "residual"],
            }
        ),
        encoding="utf-8",
    )

    assert (
        gate_cli.main(
            [
                "--operation-candidates",
                str(candidates),
                "--coefficients",
                str(coefficients),
                "--operation-set-id",
                "cli_channel_gate",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "inverse_action_operation_set_compiler_hint.v1"
    assert payload["source_schema"] == "dynamic_sparse_channel_gate_operation_selection.v1"
    assert payload["selected_operations"][0]["dynamic_gate_abs_mean_coefficient"] == pytest.approx(0.8)


def test_dynamic_sparse_gate_compiler_hint_cli_writes_observation_feedback_hint(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.json"
    out = tmp_path / "hint.json"
    observations.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "schema": "inverse_steganalysis_observation.v1",
                        "observation_id": "obs_cli",
                        "candidate_id": "candidate_cli",
                        "axis": "[local-materializer advisory]",
                        "target_kind": "packet_member_recompress_v1",
                        "source_unit_ids": ["packet_member_cli"],
                        "saved_bytes": 42,
                        "rate_positive": True,
                        "receiver_contract_satisfied": True,
                        "score_claim": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert (
        gate_cli.main(
            [
                "--observations",
                str(observations),
                "--operation-set-id",
                "cli_observation_feedback",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["selection_source"] == "dynamic_sparse_observation_feedback"
    assert payload["observation_feedback"]["selectable_observation_count"] == 1
    assert payload["selected_operations"][0]["unit_id"] == "packet_member_cli"
