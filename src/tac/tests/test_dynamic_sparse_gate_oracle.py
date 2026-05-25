# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.dynamic_sparse_gate_oracle import (
    DynamicSparseGateOracleError,
    dynamic_sparse_skip_mixture,
    operation_set_compiler_hint_from_gate_scores,
)
from tac.optimization.inverse_steganalysis_operation_set_compiler import (
    packet_ir_operation_set_from_compiler_hint,
)


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


def test_dynamic_sparse_gate_rejects_truthy_authority() -> None:
    with pytest.raises(DynamicSparseGateOracleError, match="score_claim"):
        operation_set_compiler_hint_from_gate_scores(
            [{"unit_id": "bad", "target_kind": "packet_member_recompress_v1", "score_claim": True}],
            np.array([1.0], dtype=np.float32),
            operation_set_id="bad_authority",
        )
