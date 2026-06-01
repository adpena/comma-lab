# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.optimization.byte_shaving_campaign import build_signal_surface_from_candidate_queue
from tac.packet_compiler.pr101_per_tensor_grammar_solver import (
    PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA,
    PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA,
    PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA,
    build_grouped_optimizer_candidate_queue_from_report,
    build_optimizer_candidate_queue_from_solver_report,
    empirical_shannon_floor_bytes,
    measure_tensor_grammar_candidates,
    select_best_tensor_candidate,
    solve_grouped_brotli_packet_grammar,
    solve_state_dict_per_tensor_grammar,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA


def _tiny_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        # Keep values tiny but non-degenerate so quantization exercises signs.
        out[name] = torch.randn(shape, generator=gen) * (0.01 + idx * 0.0001)
    return out


def test_tensor_candidates_are_real_roundtrip_codecs() -> None:
    q = np.array([0, 0, 0, 1, -1, 2, -2, 4, -4], dtype=np.int8)

    candidates = measure_tensor_grammar_candidates(
        q,
        tensor_index=1,
        tensor_name="synthetic.bias",
        coders=("brotli", "lzma_raw", "canonical_huffman"),
        brotli_quality=4,
    )

    assert candidates
    assert all(row["schema"] == "pr101_tensor_grammar_candidate.v1" for row in candidates)
    assert all(row["roundtrip_exact"] for row in candidates if row["status"] == "ok")
    selected = select_best_tensor_candidate(candidates)
    assert selected["charged_bytes"] == min(
        row["charged_bytes"] for row in candidates if row["status"] == "ok"
    )
    assert selected["score_claim"] is False
    assert selected["score_claim_valid"] is False
    assert selected["promotion_eligible"] is False
    assert selected["rank_or_kill_eligible"] is False
    assert selected["ready_for_exact_eval_dispatch"] is False
    assert selected["ready_for_provider_dispatch"] is False
    assert selected["dispatch_attempted"] is False
    assert "isolated_tensor_measurement_not_grouped_archive_authority" in selected["blockers"]


def test_brotli_quality_sweep_prices_compress_time_exhaustively() -> None:
    q = np.arange(32, dtype=np.int8) - 16

    candidates = measure_tensor_grammar_candidates(
        q,
        tensor_index=1,
        tensor_name="synthetic.bias",
        byte_maps=("zig",),
        coders=("brotli",),
        brotli_quality=11,
        brotli_quality_values=(1, 11),
        brotli_lgwin_sweep=True,
    )

    qualities = {row["coder_params"]["quality"] for row in candidates}
    assert qualities == {1, 11}
    assert all(row["coder_params"]["lgwin_sweep"] is True for row in candidates)
    assert all(row["roundtrip_exact"] for row in candidates)


def test_exhaustive_conv4_storage_perms_are_measured_without_fake_branches() -> None:
    q = np.arange(16, dtype=np.int8).reshape(1, 2, 2, 4) - 8

    candidates = measure_tensor_grammar_candidates(
        q,
        tensor_index=2,
        tensor_name="blocks.0.weight",
        storage_perm_mode="exhaustive-conv4",
        byte_maps=("zig",),
        coders=("brotli",),
        brotli_quality=4,
    )

    perms = {row["storage_perm"] for row in candidates}
    assert "identity" in perms
    assert "3,0,2,1" in perms
    assert len(perms) == 24
    assert all(row["roundtrip_exact"] for row in candidates)


def test_huffman_floor_and_header_are_priced() -> None:
    payload = b"\x00" * 512
    floor = empirical_shannon_floor_bytes(payload)

    candidates = measure_tensor_grammar_candidates(
        np.zeros(512, dtype=np.int8),
        tensor_index=1,
        tensor_name="constant.bias",
        byte_maps=("zig",),
        coders=("canonical_huffman",),
        brotli_quality=4,
    )

    assert floor == 0.0
    row = candidates[0]
    assert row["side_info_bytes"] == 256
    assert row["charged_bytes"] >= 256
    assert row["roundtrip_exact"] is True
    assert row["runtime_consumption_status"] == "new_receiver_adapter_required"


def test_state_dict_solver_report_is_false_authority_and_saturation_typed() -> None:
    report = solve_state_dict_per_tensor_grammar(
        _tiny_state_dict(seed=1),
        storage_perm_mode="identity",
        coders=("brotli",),
        brotli_quality=4,
        max_tensors=2,
        include_current_grouped_pr101=False,
    )

    assert report["schema"] == PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA
    assert report["partial_schema_sample"] is True
    assert report["authority"]["score_claim"] is False
    assert report["authority"]["promotion_eligible"] is False
    assert report["authority"]["ready_for_exact_eval_dispatch"] is False
    assert report["byte_accounting"]["selected_isolated_tensor_bytes"] > 0
    assert "selected_rate_term" not in report["byte_accounting"]
    assert (
        report["byte_accounting"]["isolated_tensor_payload_rate_term_not_archive_authority"]
        > 0
    )
    assert report["planner_feedback"]["schema"] == "pr101_per_tensor_grammar_planner_feedback.v1"
    assert report["planner_feedback"]["operation_hint_count"] == 2
    assert report["planner_feedback"]["posterior_update_hooks"]
    assert report["saturation_diagnostic"]["status"] in {
        "entropy_saturated",
        "weak_entropy_gap",
        "unsaturated_entropy_gap",
        "floor_unavailable",
    }
    assert len(report["rows"]) == 2


def test_solver_report_converts_to_planning_only_optimizer_queue() -> None:
    report = solve_state_dict_per_tensor_grammar(
        _tiny_state_dict(seed=0),
        storage_perm_mode="identity",
        coders=("brotli", "lzma_raw", "canonical_huffman"),
        brotli_quality=4,
        max_tensors=2,
        include_current_grouped_pr101=False,
    )

    queue = build_optimizer_candidate_queue_from_solver_report(report)

    assert queue["schema"] == PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA
    assert queue["score_claim"] is False
    assert queue["score_claim_valid"] is False
    assert queue["ready_for_provider_dispatch"] is False
    assert queue["dispatch_attempted"] is False
    assert queue["candidate_count"] == 2
    assert queue["top_k"] == queue["candidates"]
    assert "grouped_split_brotli_packet_selection_not_run" in queue["blockers"]
    first = queue["candidates"][0]
    assert first["schema"] == "optimizer_candidate_queue_row_v1"
    assert first["operation_family"] == "pr101_per_tensor_grammar_selection"
    assert first["candidate_saved_bytes"] > 0
    assert first["consumer_payload"]["selected_operations"]
    assert "full_frame_inflate_parity_missing" in first["blockers"]
    assert first["ready_for_exact_eval_dispatch"] is False

    surface = build_signal_surface_from_candidate_queue(queue)
    assert surface["score_claim"] is False
    assert surface["ready_for_exact_eval_dispatch"] is False
    assert len(surface["units"]) == 1


def test_grouped_brotli_packet_solver_prices_split_context_not_isolated_bytes() -> None:
    report = solve_grouped_brotli_packet_grammar(
        _tiny_state_dict(seed=3),
        selected_transform_mode="best_brotli_per_tensor",
        storage_perm_mode="identity",
        exact_stream_count=2,
        max_streams=2,
        brotli_quality=4,
        max_tensors=4,
    )

    assert report["schema"] == PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["parser_roundtrip"]["stream_roundtrip_exact"] is True
    assert report["byte_accounting"]["selected_grouped_brotli_bytes"] > 0
    assert report["partition"]["stream_count"] == 2
    assert report["partition"]["stream_ends"][-1] == 4
    assert "partial_schema_sample_not_archive_authority" in report["blockers"]
    assert "byte_closed_archive_not_materialized" in report["blockers"]


def test_grouped_brotli_queue_is_consumable_only_for_positive_packet_savings() -> None:
    report = solve_grouped_brotli_packet_grammar(
        _tiny_state_dict(seed=0),
        selected_transform_mode="best_brotli_per_tensor",
        storage_perm_mode="identity",
        exact_stream_count=2,
        max_streams=2,
        brotli_quality=4,
        max_tensors=4,
    )
    queue = build_grouped_optimizer_candidate_queue_from_report(report)

    assert queue["schema"] == PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA
    assert queue["score_claim"] is False
    assert queue["ready_for_exact_eval_dispatch"] is False
    assert queue["candidate_count"] == 1
    saved = report["byte_accounting"]["grouped_saved_bytes_vs_current_stock"]
    assert queue["candidates"][0]["candidate_saved_bytes"] == saved
    assert len(queue["top_k"]) == (1 if saved > 0 else 0)
    if saved > 0:
        surface = build_signal_surface_from_candidate_queue(queue)
        assert surface["score_claim"] is False
        assert surface["units"][0]["candidate_saved_bytes"] == saved


def test_negzig_rejects_int8_min_roundtrip_instead_of_faking_exactness() -> None:
    q = np.array([-128, 0, 1], dtype=np.int8)

    candidates = measure_tensor_grammar_candidates(
        q,
        tensor_index=1,
        byte_maps=("negzig",),
        coders=("brotli",),
        brotli_quality=4,
    )

    assert candidates[0]["status"] == "transform_roundtrip_failed"
    with pytest.raises(ValueError, match="no exact tensor grammar candidate"):
        select_best_tensor_candidate(candidates)
