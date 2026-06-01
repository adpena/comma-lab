# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import struct
import zipfile

import numpy as np
import pytest
import torch

from tac.optimization.byte_shaving_campaign import build_signal_surface_from_candidate_queue
from tac.packet_compiler.pr101_per_tensor_grammar_solver import (
    DEFAULT_CODERS,
    PR101_GRAMMAR_CAMPAIGN_SUMMARY_SCHEMA,
    PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA,
    PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA,
    PR101_GROUPED_DECODER_BLOB_MATERIALIZATION_SCHEMA,
    PR101_LEN24_RECEIVER_ADAPTER_SOURCE_SCHEMA,
    PR101_LEN24_RUNTIME_TREE_MATERIALIZATION_SCHEMA,
    PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA,
    PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA,
    PR101_U32_RECEIVER_ADAPTER_SOURCE_SCHEMA,
    PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA,
    build_grouped_optimizer_candidate_queue_from_report,
    build_len24_receiver_adapter_source_from_report,
    build_len24_receiver_runtime_tree_from_report,
    build_optimizer_candidate_queue_from_solver_report,
    build_pr101_optimal_grammar_campaign_summary,
    build_u32_receiver_adapter_source_from_report,
    build_u32_receiver_runtime_tree_from_report,
    empirical_shannon_floor_bytes,
    materialize_grouped_archive_from_report,
    materialize_grouped_decoder_blob_from_report,
    measure_tensor_grammar_candidates,
    select_best_tensor_candidate,
    solve_grouped_brotli_packet_grammar,
    solve_state_dict_per_tensor_grammar,
)
from tac.pr101_split_brotli_codec import DECODER_BLOB_LEN, FIXED_STATE_SCHEMA, LATENT_BLOB_LEN


def _tiny_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        # Keep values tiny but non-degenerate so quantization exercises signs.
        out[name] = torch.randn(shape, generator=gen) * (0.01 + idx * 0.0001)
    return out


def _stored_zip(member_name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    info = zipfile.ZipInfo(member_name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (2026, 5, 4, 16, 48, 4)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(info, payload)
    return buf.getvalue()


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


def test_default_coder_universe_includes_range_ac_with_visible_status() -> None:
    q = np.array([0, 0, 1, 1, 2, -2, 7, -7], dtype=np.int8)

    candidates = measure_tensor_grammar_candidates(
        q,
        tensor_index=1,
        tensor_name="synthetic.bias",
        byte_maps=("zig",),
        brotli_quality=4,
    )

    coders = {row["coder"] for row in candidates}
    assert tuple(DEFAULT_CODERS) == (
        "brotli",
        "lzma_raw",
        "canonical_huffman",
        "range_ac_empirical_hist_u16",
    )
    assert coders == set(DEFAULT_CODERS)
    range_rows = [
        row for row in candidates if row["coder"] == "range_ac_empirical_hist_u16"
    ]
    assert range_rows
    assert range_rows[0]["status"] in {"ok", "codec_dependency_unavailable"}
    if range_rows[0]["status"] == "ok":
        assert range_rows[0]["roundtrip_exact"] is True
        assert range_rows[0]["side_info_bytes"] == 512


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
        brotli_quality=4,
        max_tensors=2,
        include_current_grouped_pr101=False,
    )

    assert report["schema"] == PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA
    assert report["coders"] == list(DEFAULT_CODERS)
    assert report["partial_schema_sample"] is True
    assert report["candidate_status_counts"]
    assert report["coder_status_counts"]
    assert any(
        key.startswith("range_ac_empirical_hist_u16:")
        for key in report["coder_status_counts"]
    )
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
    assert all(row["candidate_status_counts"] for row in report["rows"])
    assert all(row["coder_status_counts"] for row in report["rows"])


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


def test_campaign_summary_demotes_grouped_zero_saturated_pr101_grammar() -> None:
    state_dict = _tiny_state_dict(seed=11)
    per_tensor_report = solve_state_dict_per_tensor_grammar(
        state_dict,
        storage_perm_mode="identity",
        coders=("brotli",),
        brotli_quality=4,
        max_tensors=4,
        include_current_grouped_pr101=False,
    )
    per_tensor_report = dict(per_tensor_report)
    per_tensor_report["saturation_diagnostic"] = {
        **per_tensor_report["saturation_diagnostic"],
        "status": "entropy_saturated",
        "selected_over_floor_ratio": 1.01,
    }
    grouped_report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=2,
        max_streams=2,
        brotli_quality=4,
        max_tensors=4,
    )
    grouped_report = {
        **grouped_report,
        "byte_accounting": {
            **grouped_report["byte_accounting"],
            "grouped_delta_bytes_vs_current_stock": 0,
            "grouped_saved_bytes_vs_current_stock": 0,
        },
    }

    summary = build_pr101_optimal_grammar_campaign_summary(
        per_tensor_report,
        grouped_report=grouped_report,
        per_tensor_queue=build_optimizer_candidate_queue_from_solver_report(
            per_tensor_report
        ),
        grouped_queue=build_grouped_optimizer_candidate_queue_from_report(grouped_report),
    )

    assert summary["schema"] == PR101_GRAMMAR_CAMPAIGN_SUMMARY_SCHEMA
    assert summary["verdict"] == "current_substrate_grammar_saturated"
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["score_claim"] is False
    assert summary["planner_feedback"]["grouped_positive"] is False
    assert summary["planner_feedback"]["receiver_adapter_work_justified"] is False
    assert summary["planner_feedback"]["grammar_payoff_is_substrate_conditional"] is True
    assert (
        "tac.cathedral_consumers.pr101_optimal_grammar_campaign_consumer.consume_candidate"
        in summary["planner_feedback"]["consumer_surfaces"]
    )
    assert (
        summary["rate_axis"]["grouped_saved_bytes_vs_current_stock"]
        == 0
    )
    assert "full_frame_inflate_parity_missing" in summary["blockers"]


def test_campaign_summary_routes_grouped_positive_runtime_tree_to_local_replay() -> None:
    state_dict = _tiny_state_dict(seed=12)
    per_tensor_report = solve_state_dict_per_tensor_grammar(
        state_dict,
        storage_perm_mode="identity",
        coders=("brotli",),
        brotli_quality=4,
        max_tensors=4,
        include_current_grouped_pr101=False,
    )
    grouped_report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=2,
        max_streams=2,
        brotli_quality=4,
        max_tensors=4,
    )
    grouped_report = {
        **grouped_report,
        "byte_accounting": {
            **grouped_report["byte_accounting"],
            "grouped_delta_bytes_vs_current_stock": -7,
            "grouped_saved_bytes_vs_current_stock": 7,
        },
    }
    archive_manifest = {
        "schema": PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA,
        "archive_layout": "u32_decoder_len_adapter",
        "byte_closed_archive_zip_materialized": True,
        "u32_decoder_len_adapter_parse_safe": True,
        "archive_zip_delta_bytes": -7,
        "blockers": ["full_frame_inflate_parity_missing"],
    }
    runtime_manifest = {
        "schema": PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA,
        "runtime_consumption_status": "u32_receiver_runtime_tree_materialized",
        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
    }

    summary = build_pr101_optimal_grammar_campaign_summary(
        per_tensor_report,
        grouped_report=grouped_report,
        archive_manifest=archive_manifest,
        runtime_tree_manifest=runtime_manifest,
    )

    assert (
        summary["verdict"]
        == "grouped_positive_runtime_ready_for_local_replay_gate"
    )
    assert summary["planner_feedback"]["grouped_positive"] is True
    assert summary["planner_feedback"]["receiver_adapter_work_justified"] is True
    assert summary["planner_feedback"]["exact_auth_work_justified"] is False
    assert summary["rate_axis"]["archive_zip_delta_bytes"] == -7
    assert summary["rate_axis"]["rate_delta_score_if_components_unchanged"] < 0
    assert summary["artifact_status"]["archive_materialized"] is True
    assert summary["artifact_status"]["archive_receiver_parse_safe"] is True
    assert summary["artifact_status"]["runtime_tree_materialized"] is True
    assert (
        summary["artifact_status"]["runtime_tree_compatible_with_archive_layout"]
        is True
    )
    assert "archive_zip_not_materialized" not in summary["blockers"]
    assert "byte_closed_archive_not_materialized" not in summary["blockers"]
    assert "receiver_runtime_source_not_emitted" not in summary["blockers"]
    assert "runtime_consumption_proof_missing" not in summary["blockers"]
    assert "local_replay_gate_wins" in summary["planner_feedback"][
        "exact_auth_work_blocked_until"
    ]


def test_campaign_summary_refuses_incompatible_archive_runtime_layout() -> None:
    state_dict = _tiny_state_dict(seed=13)
    per_tensor_report = solve_state_dict_per_tensor_grammar(
        state_dict,
        storage_perm_mode="identity",
        coders=("brotli",),
        brotli_quality=4,
        max_tensors=4,
        include_current_grouped_pr101=False,
    )
    grouped_report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=2,
        max_streams=2,
        brotli_quality=4,
        max_tensors=4,
    )
    grouped_report = {
        **grouped_report,
        "byte_accounting": {
            **grouped_report["byte_accounting"],
            "grouped_delta_bytes_vs_current_stock": -1,
            "grouped_saved_bytes_vs_current_stock": 1,
        },
    }
    archive_manifest = {
        "schema": PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA,
        "archive_layout": "fixed_pr101",
        "byte_closed_archive_zip_materialized": True,
        "fixed_offset_stock_runtime_parse_safe": False,
        "archive_zip_delta_bytes": -1,
        "blockers": ["stock_runtime_fixed_offset_decoder_blob_length_mismatch"],
    }
    runtime_manifest = {
        "schema": PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA,
        "runtime_consumption_status": "u32_receiver_runtime_tree_materialized",
        "blockers": [],
    }

    summary = build_pr101_optimal_grammar_campaign_summary(
        per_tensor_report,
        grouped_report=grouped_report,
        archive_manifest=archive_manifest,
        runtime_tree_manifest=runtime_manifest,
    )

    assert (
        summary["verdict"]
        == "grouped_positive_archive_materialized_but_receiver_incompatible"
    )
    assert summary["artifact_status"]["archive_materialized"] is True
    assert summary["artifact_status"]["archive_receiver_parse_safe"] is False
    assert (
        summary["artifact_status"]["runtime_tree_compatible_with_archive_layout"]
        is False
    )
    assert "stock_runtime_fixed_offset_decoder_blob_length_mismatch" in summary[
        "blockers"
    ]


def test_campaign_summary_demotes_adapter_overhead_that_consumes_grouped_win() -> None:
    state_dict = _tiny_state_dict(seed=14)
    per_tensor_report = solve_state_dict_per_tensor_grammar(
        state_dict,
        storage_perm_mode="identity",
        coders=("brotli",),
        brotli_quality=4,
        max_tensors=4,
        include_current_grouped_pr101=False,
    )
    grouped_report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=2,
        max_streams=2,
        brotli_quality=4,
        max_tensors=4,
    )
    grouped_report = {
        **grouped_report,
        "byte_accounting": {
            **grouped_report["byte_accounting"],
            "grouped_delta_bytes_vs_current_stock": -1,
            "grouped_saved_bytes_vs_current_stock": 1,
        },
    }
    archive_manifest = {
        "schema": PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA,
        "archive_layout": "u32_decoder_len_adapter",
        "byte_closed_archive_zip_materialized": True,
        "u32_decoder_len_adapter_parse_safe": True,
        "archive_zip_delta_bytes": 3,
        "blockers": ["full_frame_inflate_parity_missing"],
    }
    runtime_manifest = {
        "schema": PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA,
        "runtime_consumption_status": "u32_receiver_runtime_tree_materialized",
        "blockers": [],
    }

    summary = build_pr101_optimal_grammar_campaign_summary(
        per_tensor_report,
        grouped_report=grouped_report,
        archive_manifest=archive_manifest,
        runtime_tree_manifest=runtime_manifest,
    )

    assert summary["verdict"] == "grouped_positive_consumed_by_archive_overhead"
    assert summary["rate_axis"]["archive_zip_delta_bytes"] == 3
    assert summary["rate_axis"]["archive_rate_positive"] is False
    assert summary["planner_feedback"]["grouped_positive"] is True
    assert summary["planner_feedback"]["receiver_adapter_work_justified"] is False
    assert "archive_zip_not_materialized" not in summary["blockers"]
    assert "byte_closed_archive_not_materialized" not in summary["blockers"]
    assert "runtime_consumption_proof_missing" not in summary["blockers"]


def test_grouped_decoder_blob_materializer_proves_receiver_parse_without_score_claim() -> None:
    state_dict = _tiny_state_dict(seed=4)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )

    blob, manifest = materialize_grouped_decoder_blob_from_report(state_dict, report)

    assert blob
    assert manifest["schema"] == PR101_GROUPED_DECODER_BLOB_MATERIALIZATION_SCHEMA
    assert manifest["decoder_blob_bytes"] == len(blob)
    assert manifest["reported_grouped_brotli_bytes"] == len(blob)
    assert len(manifest["decoder_blob_sha256"]) == 64
    assert manifest["byte_closed_decoder_blob_materialized"] is True
    assert manifest["proof"]["stream_roundtrip_exact"] is True
    assert manifest["proof"]["materialized_bytes_match_report"] is True
    assert manifest["proof"]["state_dict_parser_roundtrip_exact"] is True
    assert manifest["proof"]["quantized_state_dict_exact"] is True
    assert manifest["ready_for_archive_packaging"] is True
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "archive_zip_not_materialized" in manifest["blockers"]


def test_grouped_decoder_blob_materializer_rejects_partial_schema_report() -> None:
    state_dict = _tiny_state_dict(seed=5)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="best_brotli_per_tensor",
        storage_perm_mode="identity",
        exact_stream_count=2,
        max_streams=2,
        brotli_quality=4,
        max_tensors=4,
    )

    with pytest.raises(ValueError, match="full PR101 schema"):
        materialize_grouped_decoder_blob_from_report(state_dict, report)


def test_grouped_archive_materializer_preserves_sections_and_fails_closed() -> None:
    state_dict = _tiny_state_dict(seed=6)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    source_decoder = b"D" * DECODER_BLOB_LEN
    source_latent = b"L" * LATENT_BLOB_LEN
    source_sidecar = b"S" * 607
    source_zip = _stored_zip("x", source_decoder + source_latent + source_sidecar)

    archive_zip, manifest = materialize_grouped_archive_from_report(
        state_dict,
        report,
        source_archive_zip=source_zip,
    )

    assert manifest["schema"] == PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA
    assert manifest["byte_closed_archive_zip_materialized"] is True
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["proof"]["latent_blob_preserved"] is True
    assert manifest["proof"]["sidecar_blob_preserved"] is True
    assert manifest["zip_proof"]["single_member"] is True
    assert manifest["zip_proof"]["zip_stored"] is True
    assert manifest["zip_proof"]["empty_extra"] is True
    assert manifest["zip_proof"]["empty_comment"] is True
    assert manifest["zip_proof"]["deterministic_timestamp"] is True
    assert manifest["zip_proof"]["roundtrip_exact"] is True
    with zipfile.ZipFile(io.BytesIO(archive_zip), "r") as zf:
        infos = zf.infolist()
        assert [info.filename for info in infos] == ["x"]
        inner = zf.read("x")
    assert inner.endswith(source_latent + source_sidecar)
    assert manifest["inner_member_bytes"] == len(inner)
    assert "full_frame_inflate_parity_missing" in manifest["blockers"]
    if manifest["decoder_blob_bytes"] != DECODER_BLOB_LEN:
        assert "stock_runtime_fixed_offset_decoder_blob_length_mismatch" in manifest["blockers"]


def test_grouped_archive_materializer_can_emit_u32_decoder_len_adapter_layout() -> None:
    state_dict = _tiny_state_dict(seed=8)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    source_latent = b"L" * LATENT_BLOB_LEN
    source_sidecar = b"S" * 607
    source_zip = _stored_zip("x", b"D" * DECODER_BLOB_LEN + source_latent + source_sidecar)

    archive_zip, manifest = materialize_grouped_archive_from_report(
        state_dict,
        report,
        source_archive_zip=source_zip,
        archive_layout="u32_decoder_len_adapter",
    )

    with zipfile.ZipFile(io.BytesIO(archive_zip), "r") as zf:
        inner = zf.read("x")
    section_total = struct.unpack_from("<I", inner, 0)[0]
    assert section_total == 4 + manifest["decoder_blob_bytes"]
    assert inner[section_total : section_total + LATENT_BLOB_LEN] == source_latent
    assert inner[section_total + LATENT_BLOB_LEN :] == source_sidecar
    assert manifest["archive_layout"] == "u32_decoder_len_adapter"
    assert manifest["decoder_blob_offset"] == 4
    assert manifest["latent_blob_offset"] == section_total
    assert manifest["proof"]["u32_decoder_len_adapter_parse_safe"] is True
    assert "stock_runtime_fixed_offset_decoder_blob_length_mismatch" not in manifest["blockers"]
    assert "receiver_runtime_source_not_emitted" in manifest["blockers"]


def test_grouped_archive_materializer_can_emit_len24_decoder_len_adapter_layout() -> None:
    state_dict = _tiny_state_dict(seed=8)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    source_latent = b"L" * LATENT_BLOB_LEN
    source_sidecar = b"S" * 607
    source_zip = _stored_zip("x", b"D" * DECODER_BLOB_LEN + source_latent + source_sidecar)

    archive_zip, manifest = materialize_grouped_archive_from_report(
        state_dict,
        report,
        source_archive_zip=source_zip,
        archive_layout="len24_decoder_len_adapter",
    )

    with zipfile.ZipFile(io.BytesIO(archive_zip), "r") as zf:
        inner = zf.read("x")
    section_total = int.from_bytes(inner[:3], "little")
    assert section_total == 3 + manifest["decoder_blob_bytes"]
    assert inner[section_total : section_total + LATENT_BLOB_LEN] == source_latent
    assert inner[section_total + LATENT_BLOB_LEN :] == source_sidecar
    assert manifest["archive_layout"] == "len24_decoder_len_adapter"
    assert manifest["decoder_len_header_bytes"] == 3
    assert manifest["decoder_blob_offset"] == 3
    assert manifest["latent_blob_offset"] == section_total
    assert manifest["proof"]["len24_decoder_len_adapter_parse_safe"] is True
    assert manifest["proof"]["u32_decoder_len_adapter_parse_safe"] is False
    assert "stock_runtime_fixed_offset_decoder_blob_length_mismatch" not in manifest["blockers"]
    assert "receiver_runtime_source_not_emitted" in manifest["blockers"]


def test_u32_receiver_adapter_source_is_executable_parser_glue() -> None:
    state_dict = _tiny_state_dict(seed=9)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    source, manifest = build_u32_receiver_adapter_source_from_report(report)
    decoder_blob = b"decoder"
    latent_blob = b"L" * LATENT_BLOB_LEN
    sidecar_blob = b"sidecar"
    inner = struct.pack("<I", 4 + len(decoder_blob)) + decoder_blob + latent_blob + sidecar_blob
    calls: dict[str, object] = {}

    def fake_decode_decoder_compact(blob: bytes, **kwargs: object) -> dict[str, object]:
        calls["decoder_blob"] = blob
        calls["decoder_kwargs"] = kwargs
        return {"ok": True}

    def fake_decode_latents_compact(blob: bytes) -> bytes:
        calls["latent_blob"] = blob
        return b"latents"

    def fake_apply_latent_sidecar(latents: bytes, sidecar: bytes) -> tuple[bytes, bytes]:
        calls["latents"] = latents
        calls["sidecar_blob"] = sidecar
        return latents, sidecar

    namespace: dict[str, object] = {}
    exec(compile(source, "<generated_pr101_u32_adapter>", "exec"), namespace)
    parsed = namespace["parse_archive_u32_decoder_len"](
        inner,
        decode_decoder_compact=fake_decode_decoder_compact,
        decode_latents_compact=fake_decode_latents_compact,
        apply_latent_sidecar=fake_apply_latent_sidecar,
    )

    assert manifest["schema"] == PR101_U32_RECEIVER_ADAPTER_SOURCE_SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["runtime_consumption_status"] == "u32_decoder_len_adapter_source_emitted"
    assert calls["decoder_blob"] == decoder_blob
    assert calls["latent_blob"] == latent_blob
    assert calls["sidecar_blob"] == sidecar_blob
    assert "derived_stream_ends" in calls["decoder_kwargs"]
    assert parsed[0] == {"ok": True}
    assert parsed[1] == (b"latents", sidecar_blob)
    assert parsed[2]["n_pairs"] == 600


def test_len24_receiver_adapter_source_is_executable_parser_glue() -> None:
    state_dict = _tiny_state_dict(seed=9)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    source, manifest = build_len24_receiver_adapter_source_from_report(report)
    decoder_blob = b"decoder"
    latent_blob = b"L" * LATENT_BLOB_LEN
    sidecar_blob = b"sidecar"
    section_total = 3 + len(decoder_blob)
    inner = section_total.to_bytes(3, "little") + decoder_blob + latent_blob + sidecar_blob
    calls: dict[str, object] = {}

    def fake_decode_decoder_compact(blob: bytes, **kwargs: object) -> dict[str, object]:
        calls["decoder_blob"] = blob
        calls["decoder_kwargs"] = kwargs
        return {"ok": True}

    def fake_decode_latents_compact(blob: bytes) -> bytes:
        calls["latent_blob"] = blob
        return b"latents"

    def fake_apply_latent_sidecar(latents: bytes, sidecar: bytes) -> tuple[bytes, bytes]:
        calls["latents"] = latents
        calls["sidecar_blob"] = sidecar
        return latents, sidecar

    namespace: dict[str, object] = {}
    exec(compile(source, "<generated_pr101_len24_adapter>", "exec"), namespace)
    parsed = namespace["parse_archive_len24_decoder_len"](
        inner,
        decode_decoder_compact=fake_decode_decoder_compact,
        decode_latents_compact=fake_decode_latents_compact,
        apply_latent_sidecar=fake_apply_latent_sidecar,
    )

    assert manifest["schema"] == PR101_LEN24_RECEIVER_ADAPTER_SOURCE_SCHEMA
    assert manifest["archive_layout"] == "len24_decoder_len_adapter"
    assert manifest["decoder_len_header_bytes"] == 3
    assert manifest["runtime_consumption_status"] == "len24_decoder_len_adapter_source_emitted"
    assert calls["decoder_blob"] == decoder_blob
    assert calls["latent_blob"] == latent_blob
    assert calls["sidecar_blob"] == sidecar_blob
    assert "derived_stream_ends" in calls["decoder_kwargs"]
    assert parsed[0] == {"ok": True}
    assert parsed[1] == (b"latents", sidecar_blob)


def test_u32_runtime_tree_materializer_emits_submission_runtime_files() -> None:
    state_dict = _tiny_state_dict(seed=10)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    codec_source = (
        "def decode_decoder_compact(*args, **kwargs): pass\n"
        "def decode_latents_compact(*args, **kwargs): pass\n"
        "def apply_latent_sidecar(*args, **kwargs): pass\n"
    )
    model_source = "class HNeRVDecoder:\n    pass\n"

    files, manifest = build_u32_receiver_runtime_tree_from_report(
        report,
        codec_py_source=codec_source,
        model_py_source=model_source,
    )

    assert manifest["schema"] == PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["runtime_consumption_status"] == "u32_receiver_runtime_tree_materialized"
    assert set(files) == {
        "inflate.sh",
        "inflate.py",
        "pr101_u32_adapter.py",
        "src/codec.py",
        "src/model.py",
    }
    assert files["src/codec.py"] == codec_source.encode()
    assert files["src/model.py"] == model_source.encode()
    assert b"parse_archive_u32_decoder_len" in files["inflate.py"]
    assert b"ADAPTER_PARAMS_JSON" in files["pr101_u32_adapter.py"]
    assert files["inflate.sh"].startswith(b"#!/usr/bin/env bash\n")
    compile(files["inflate.py"].decode(), "<generated_inflate.py>", "exec")
    compile(files["pr101_u32_adapter.py"].decode(), "<generated_pr101_u32_adapter.py>", "exec")
    rels = {row["rel_path"]: row for row in manifest["files"]}
    assert rels["inflate.sh"]["executable"] is True
    assert rels["inflate.py"]["bytes"] == len(files["inflate.py"])
    assert "full_frame_inflate_parity_missing" in manifest["blockers"]


def test_len24_runtime_tree_materializer_emits_submission_runtime_files() -> None:
    state_dict = _tiny_state_dict(seed=10)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    codec_source = (
        "def decode_decoder_compact(*args, **kwargs): pass\n"
        "def decode_latents_compact(*args, **kwargs): pass\n"
        "def apply_latent_sidecar(*args, **kwargs): pass\n"
    )
    model_source = "class HNeRVDecoder:\n    pass\n"

    files, manifest = build_len24_receiver_runtime_tree_from_report(
        report,
        codec_py_source=codec_source,
        model_py_source=model_source,
    )

    assert manifest["schema"] == PR101_LEN24_RUNTIME_TREE_MATERIALIZATION_SCHEMA
    assert manifest["archive_layout"] == "len24_decoder_len_adapter"
    assert manifest["runtime_consumption_status"] == "len24_receiver_runtime_tree_materialized"
    assert set(files) == {
        "inflate.sh",
        "inflate.py",
        "pr101_len24_adapter.py",
        "src/codec.py",
        "src/model.py",
    }
    assert b"parse_archive_len24_decoder_len" in files["inflate.py"]
    assert b"ADAPTER_PARAMS_JSON" in files["pr101_len24_adapter.py"]
    compile(files["inflate.py"].decode(), "<generated_len24_inflate.py>", "exec")
    compile(files["pr101_len24_adapter.py"].decode(), "<generated_pr101_len24_adapter.py>", "exec")


def test_grouped_archive_materializer_rejects_multimember_source_zip() -> None:
    state_dict = _tiny_state_dict(seed=7)
    report = solve_grouped_brotli_packet_grammar(
        state_dict,
        selected_transform_mode="stock_pr101",
        exact_stream_count=7,
        max_streams=7,
        brotli_quality=4,
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x", b"D" * DECODER_BLOB_LEN + b"L" * LATENT_BLOB_LEN)
        zf.writestr("y", b"extra")

    with pytest.raises(ValueError, match="exactly one member"):
        materialize_grouped_archive_from_report(
            state_dict,
            report,
            source_archive_zip=buf.getvalue(),
        )


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
