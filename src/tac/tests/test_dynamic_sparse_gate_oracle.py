# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
)
from tac.optimization.dynamic_sparse_gate_oracle import (
    DynamicSparseGateOracleError,
    dynamic_sparse_skip_mixture,
    operation_set_compiler_hint_from_channel_gate_scores,
    operation_set_compiler_hint_from_gate_scores,
    operation_set_compiler_hint_from_materializer_feedback,
    operation_set_compiler_hint_from_observation_feedback,
)
from tac.optimization.inverse_steganalysis_operation_set_compiler import (
    packet_ir_operation_set_from_compiler_hint,
)
from tac.optimization.materializer_feedback import materializer_observation_feedback_rows
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


def test_operation_set_compiler_rejects_nested_truthy_authority() -> None:
    hint = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "operation_set_id": "nested_truthy_authority_fixture",
        "selected_operations": [
            {
                "unit_id": "payload_member",
                "target_kind": "packet_member_recompress_v1",
                "candidate_saved_bytes": 20,
                "ready_for_exact_eval_dispatch": True,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    with pytest.raises(
        ValueError,
        match=r"selected_operations\[0\].*ready_for_exact_eval_dispatch=truthy",
    ):
        packet_ir_operation_set_from_compiler_hint(
            hint,
            source_backlog_key="nested_truthy_backlog",
            source_unit_ids=["payload_member"],
        )


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


def test_observation_feedback_rejects_stale_archive_bound_contract_fields() -> None:
    stale_contract = {
        "schema": ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
        "selected_archive_transform_variant": True,
        "runtime_consumption_proof_ready": True,
        "receiver_contract_satisfied": True,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    fresh_contract = {
        **stale_contract,
        "receiver_contract_satisfied": True,
    }
    hint = operation_set_compiler_hint_from_observation_feedback(
        [
            {
                "schema": "family_agnostic_materializer_empirical_observation.v1",
                "observation_id": "stale_contract_positive_rate",
                "candidate_id": "stale_contract_candidate",
                "target_kind": "packet_member_recompress_v1",
                "source_unit_ids": ["stale_unit"],
                "saved_bytes": 800,
                "rate_positive": True,
                "receiver_contract_satisfied": False,
                "archive_bound_candidate_contract": stale_contract,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "schema": "family_agnostic_materializer_empirical_observation.v1",
                "observation_id": "fresh_contract_smaller_rate",
                "candidate_id": "fresh_contract_candidate",
                "target_kind": "packet_member_recompress_v1",
                "source_unit_ids": ["fresh_unit"],
                "saved_bytes": 10,
                "rate_positive": True,
                "receiver_contract_satisfied": True,
                "archive_bound_candidate_contract": fresh_contract,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ],
        operation_set_id="stale_contract_feedback_fixture",
        max_operations=1,
    )

    selected = hint["selected_operations"][0]
    assert hint["observation_feedback"]["selectable_observation_count"] == 1
    assert selected["candidate_id"] == "fresh_contract_candidate"
    assert selected["params"]["archive_bound_candidate_contract"]["schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    )


def test_materializer_feedback_builds_channel_gate_hint() -> None:
    hint = operation_set_compiler_hint_from_materializer_feedback(
        {
            "target_kind": "packet_member_recompress_v1",
            "materializer_id": "packet_member_recompress_adapter",
            "receiver_contract_kind": "packet_member_receiver",
            "candidate_id": "candidate_manifest_feedback",
            "selected_member_name": "payload.bin",
            "source_archive": {"sha256": "a" * 64, "bytes": 128},
            "candidate_archive": {"sha256": "b" * 64, "bytes": 80},
            "selected_compression": {
                "source_archive_bytes": 128,
                "candidate_archive_bytes": 80,
                "saved_bytes": 48,
            },
            "runtime_consumption_proof_write": {"sha256": "c" * 64},
            "receiver_contract_satisfied": True,
            "inflate_parity_satisfied": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        operation_set_id="materializer_feedback_fixture",
        source_path="candidate_manifest.json",
        max_operations=1,
        lane_id="codex_dynamic_sparse_queue_observation_bridge_20260525",
    )

    selected = hint["selected_operations"][0]
    feedback = selected["params"]["dynamic_sparse_observation_feedback"]
    assert hint["selection_source"] == "dynamic_sparse_materializer_feedback"
    assert hint["materializer_feedback"]["normalized_observation_count"] == 1
    assert selected["unit_id"] == "payload.bin"
    assert selected["target_kind"] == "packet_member_recompress_v1"
    assert selected["candidate_saved_bytes"] == 48
    assert feedback["saved_bytes"] == 48
    assert feedback["source_path"] == "candidate_manifest.json"
    assert selected["score_claim"] is False


def test_materializer_feedback_accepts_serialized_archive_delta_contract() -> None:
    rows = materializer_observation_feedback_rows(
        {
            "schema": "future_materializer_candidate.v1",
            "target_kind": "future_byte_packer_v1",
            "materializer_id": "future_byte_packer_adapter",
            "receiver_contract_kind": "family_agnostic_future_byte_packer",
            "candidate_id": "future_candidate",
            "selected_member_name": "future/payload.bin",
            "source_archive": {"bytes": 2048, "sha256": "a" * 64},
            "candidate_archive": {"bytes": 2000, "sha256": "b" * 64},
            "serialized_archive_delta": {
                "schema": "serialized_archive_delta_contract.v1",
                "source_archive_bytes": 2048,
                "candidate_archive_bytes": 2000,
                "archive_delta_bytes": -48,
                "realized_saved_bytes": 48,
                "savings_realized": True,
                "status": "realized_saving",
                "score_claim": False,
            },
            "receiver_contract_satisfied": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        source_path="future_manifest.json",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["target_kind"] == "future_byte_packer_v1"
    assert row["selected_materialization_key"] == "serialized_archive_delta"
    assert row["serialized_archive_delta"]["schema"] == "serialized_archive_delta_contract.v1"
    assert row["saved_bytes"] == 48
    assert row["rate_positive"] is True
    assert row["source_unit_ids"] == ["future/payload.bin"]
    assert row["score_claim"] is False


def test_materializer_feedback_preserves_contradictory_receiver_evidence() -> None:
    base = {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        "observation_id": "same_candidate",
        "candidate_id": "same_candidate",
        "target_kind": "packet_member_merge_v1",
        "materializer_id": "packet_member_merge_adapter",
        "source_archive_sha256": "a" * 64,
        "candidate_archive_sha256": "b" * 64,
        "selected_member_name": "renderer.bin",
        "saved_bytes": 258,
        "rate_positive": True,
        "savings_realized": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    rows = materializer_observation_feedback_rows(
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "observations": [
                {
                    **base,
                    "receiver_contract_satisfied": False,
                    "inflate_parity_satisfied": False,
                    "readiness_blockers": ["runtime_adapter_missing"],
                },
                {
                    **base,
                    "receiver_contract_satisfied": True,
                    "inflate_parity_satisfied": True,
                    "readiness_blockers": [],
                },
            ],
        },
        source_path="merge_feedback.json",
    )

    assert len(rows) == 2
    assert {row["receiver_contract_satisfied"] for row in rows} == {False, True}


def test_materializer_feedback_rejects_truthy_serialized_archive_delta_authority() -> None:
    with pytest.raises(ValueError, match="score_claim"):
        materializer_observation_feedback_rows(
            {
                "schema": "future_materializer_candidate.v1",
                "target_kind": "future_byte_packer_v1",
                "materializer_id": "future_byte_packer_adapter",
                "receiver_contract_kind": "family_agnostic_future_byte_packer",
                "source_archive": {"bytes": 2048, "sha256": "a" * 64},
                "candidate_archive": {"bytes": 2000, "sha256": "b" * 64},
                "serialized_archive_delta": {
                    "schema": "serialized_archive_delta_contract.v1",
                    "realized_saved_bytes": 48,
                    "savings_realized": True,
                    "status": "realized_saving",
                    "score_claim": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
            },
            source_path="future_manifest.json",
        )


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


def test_dynamic_sparse_gate_compiler_hint_cli_accepts_queue_observation(
    tmp_path: Path,
) -> None:
    queue_observation = tmp_path / "queue_observation.json"
    runtime_identity = tmp_path / "runtime_identity.json"
    cache_identity = tmp_path / "cache_identity.json"
    out = tmp_path / "hint.json"
    runtime_identity.write_text(
        json.dumps({"runtime_tree_sha256": "a" * 64}),
        encoding="utf-8",
    )
    cache_identity.write_text(
        json.dumps({"cache_sha256": "b" * 64}),
        encoding="utf-8",
    )
    queue_observation.write_text(
        json.dumps(
            {
                "schema": "experiment_queue_observation.v1",
                "queue_id": "queue_gate_feedback",
                "healthy": True,
                "status_counts": {"succeeded": 1},
                "succeeded_artifact_steps": [
                    {
                        "experiment_id": "materializer_candidate_cli",
                        "step_id": "materialize_local_proof_chain",
                        "resource_kind": "local_cpu",
                        "candidate_ids": ["candidate_cli"],
                        "source_unit_ids": ["packet_member_queue"],
                        "expected_artifacts": [
                            {
                                "path": "candidate_manifest.json",
                                "candidate_id": "candidate_cli",
                                "target_kind": "packet_member_recompress_v1",
                                "materializer_id": "packet_member_recompress_adapter",
                                "receiver_contract_kind": "packet_member_receiver",
                                "serialized_archive_delta_status": "realized_saving",
                                "serialized_archive_delta_realized_saved_bytes": 64,
                                "serialized_archive_delta_savings_realized": True,
                                "receiver_contract_satisfied": True,
                                "bytes": 128,
                            }
                        ],
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    assert (
        gate_cli.main(
            [
                "--queue-observation",
                str(queue_observation),
                "--queue-performance-runtime-identity",
                str(runtime_identity),
                "--queue-performance-cache-identity",
                str(cache_identity),
                "--operation-set-id",
                "cli_queue_observation_feedback",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    selected = payload["selected_operations"][0]
    feedback = selected["params"]["dynamic_sparse_observation_feedback"]
    assert payload["selection_source"] == "dynamic_sparse_observation_feedback"
    assert payload["observation_feedback"]["selectable_observation_count"] == 1
    assert selected["unit_id"] == "packet_member_queue"
    assert feedback["queue_id"] == "queue_gate_feedback"
    assert feedback["saved_bytes"] == 64
    assert selected["score_claim"] is False


def test_dynamic_sparse_gate_compiler_hint_cli_accepts_materializer_feedback(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "candidate_manifest.json"
    out = tmp_path / "hint.json"
    manifest.write_text(
        json.dumps(
            {
                "target_kind": "packet_member_recompress_v1",
                "materializer_id": "packet_member_recompress_adapter",
                "candidate_id": "candidate_manifest_cli",
                "selected_member_name": "payload.bin",
                "source_archive": {"sha256": "d" * 64, "bytes": 144},
                "candidate_archive": {"sha256": "e" * 64, "bytes": 100},
                "selected_compression": {
                    "source_archive_bytes": 144,
                    "candidate_archive_bytes": 100,
                    "saved_bytes": 44,
                },
                "runtime_consumption_proof_write": {"sha256": "f" * 64},
                "receiver_contract_satisfied": True,
                "inflate_parity_satisfied": True,
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )

    assert (
        gate_cli.main(
            [
                "--materializer-feedback",
                str(manifest),
                "--operation-set-id",
                "cli_materializer_feedback",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    selected = payload["selected_operations"][0]
    assert payload["selection_source"] == "dynamic_sparse_materializer_feedback"
    assert payload["materializer_feedback"]["normalized_observation_count"] == 1
    assert selected["unit_id"] == "payload.bin"
    assert selected["candidate_saved_bytes"] == 44
    assert selected["score_claim"] is False


def test_dynamic_sparse_gate_compiler_hint_cli_discovers_materializer_feedback_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier_feedback"
    first = root / "packet_member_zip_header_elide_v1"
    second = root / "per_archive" / "archive_a" / "packet_member_merge_v1"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "sweep.json").write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_sweep.v1",
                "observations": [
                    {
                        "schema": "family_agnostic_materializer_empirical_observation.v1",
                        "observation_id": "header_elide_positive",
                        "target_kind": "packet_member_zip_header_elide_v1",
                        "materializer_id": "packet_member_zip_header_elide_adapter",
                        "selected_member_name": "renderer.bin",
                        "source_archive_sha256": "a" * 64,
                        "candidate_archive_sha256": "b" * 64,
                        "source_archive_bytes": 345_802,
                        "candidate_archive_bytes": 345_646,
                        "saved_bytes": 156,
                        "observed_score_gain": 0.0001,
                        "observed_rate_gain": 0.0001,
                        "rate_positive": True,
                        "receiver_contract_satisfied": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    (first / "observations.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in json.loads((first / "sweep.json").read_text(encoding="utf-8"))[
                "observations"
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (second / "observations.jsonl").write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_observation.v1",
                "observation_id": "merge_positive_receiver_blocked",
                "target_kind": "packet_member_merge_v1",
                "materializer_id": "packet_member_merge_adapter",
                "selected_member_name": "merged.bin",
                "source_archive_sha256": "c" * 64,
                "candidate_archive_sha256": "d" * 64,
                "source_archive_bytes": 345_802,
                "candidate_archive_bytes": 345_544,
                "saved_bytes": 258,
                "observed_score_gain": 0.0,
                "observed_rate_gain": 0.00017,
                "rate_positive": True,
                "receiver_contract_satisfied": False,
                "readiness_blockers": [
                    "packet_member_merge_receiver_contract_not_satisfied"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "hint.json"

    assert (
        gate_cli.main(
            [
                "--materializer-feedback-root",
                str(root),
                "--operation-set-id",
                "materializer_feedback_root_fixture",
                "--out",
                str(out),
                "--max-operations",
                "4",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    feedback = payload["materializer_feedback"]
    assert payload["selection_source"] == "dynamic_sparse_materializer_feedback"
    assert feedback["normalized_observation_count"] == 2
    assert feedback["discovered_source_count"] == 2
    assert len(feedback["source_paths"]) == 2
    assert {row["target_kind"] for row in payload["selected_operations"]} == {
        "packet_member_zip_header_elide_v1",
        "packet_member_merge_v1",
    }
    assert payload["score_claim"] is False
