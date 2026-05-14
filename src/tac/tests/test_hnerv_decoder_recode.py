from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    build_structural_recode_profile,
    decode_global_prev_symbol_context_range_fixture,
    decode_global_prev_symbol_mixed_context_fixture,
    decode_hdm3_q_brotli_split_fixture,
    decode_hdm6_q_brotli_tuned_fixture,
    decode_hdm5_q_brotli_split_planning_fixture,
    decode_prev_symbol_context_range_fixture,
    encode_global_prev_symbol_context_range_fixture,
    encode_global_prev_symbol_mixed_context_fixture,
    encode_hdm3_q_brotli_split_fixture,
    encode_hdm4_q_brotli_split_fixture,
    encode_hdm6_q_brotli_tuned_fixture,
    encode_hdm5_q_brotli_split_planning_fixture,
    encode_prev_symbol_context_range_fixture,
    parse_decoder_section_for_recode,
    parse_packed_decoder_brotli,
    search_hdm5_q_brotli_split_recipes,
)
from tac.hnerv_lowlevel_packer import parse_ff_packed_brotli_hnerv, write_stored_single_member_zip
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106SidecarPacket,
    emit_pr106_sidecar_packet,
)

REPO = Path(__file__).resolve().parents[3]


def test_parse_packed_decoder_brotli_roundtrips_raw() -> None:
    raw = _synthetic_decoder_raw()
    parsed = parse_packed_decoder_brotli(brotli.compress(raw, quality=5))

    assert parsed.to_raw() == raw
    assert len(parsed.records) == len(PACKED_STATE_SCHEMA)
    assert len(parsed.scale_stream) == 4 * len(PACKED_STATE_SCHEMA)


def test_structural_recode_profile_is_planning_only_and_raw_equal() -> None:
    packed = parse_ff_packed_brotli_hnerv(_packed_payload(brotli.compress(_synthetic_decoder_raw(), quality=5)))
    profile = build_structural_recode_profile(
        packed,
        source_label="fixture",
        source_archive_sha256="a" * 64,
    )

    assert profile["score_claim"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["ready_for_archive_preflight"] is False
    assert profile["record_count"] == len(PACKED_STATE_SCHEMA)
    assert all(row["raw_equal"] is True for row in profile["variants"])
    entropy = profile["entropy_summary"]
    assert entropy["score_claim"] is False
    assert entropy["q_stream_symbols"] == profile["q_stream_bytes"]
    assert entropy["global_q_entropy_floor_bytes"] <= profile["q_stream_bytes"]
    assert entropy["global_q_entropy_floor_plus_raw_scales_bytes"] >= entropy["global_q_entropy_floor_bytes"]
    assert entropy["current_static_model_interpretation"] in {
        "zero_order_q_symbol_floor_loses_to_current_brotli",
        "zero_order_q_symbol_floor_has_byte_headroom",
    }
    aq_global = next(
        row for row in profile["variants"] if row["variant"] == "aq_global_q_stream_plus_raw_scales"
    )
    assert "byte_gap_vs_global_q_entropy_floor_plus_raw_scales" in aq_global
    context_range = next(
        row
        for row in profile["variants"]
        if row["variant"] == "range_prev_symbol_per_tensor_q_streams_plus_raw_scales"
    )
    assert context_range["codec"] == "HDC1_prev_symbol_per_tensor_range_uint8"
    assert context_range["parity_fixture"] is True
    assert context_range["archive_ready"] is False
    assert context_range["raw_equal"] is True
    assert context_range["q_roundtrip_equal"] is True
    assert context_range["scale_roundtrip_equal"] is True
    assert context_range["context_count"] > 0
    assert context_range["header_bytes"] > 0
    assert context_range["range_payload_bytes"] > 0
    assert "byte_gap_vs_per_tensor_q_entropy_floor_plus_raw_scales" in context_range
    assert "byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales" in context_range
    assert entropy["per_tensor_prev_symbol_contexts"] == context_range["context_count"]
    assert entropy["per_tensor_prev_symbol_tokens"] == context_range["context_token_count"]
    assert entropy["per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"] >= entropy[
        "per_tensor_prev_symbol_entropy_floor_bytes"
    ]
    global_context = next(
        row
        for row in profile["variants"]
        if row["variant"] == "range_prev_symbol_global_q_streams_plus_raw_scales"
    )
    assert global_context["codec"] == "HDC2_global_prev_symbol_range_uint8"
    assert global_context["parity_fixture"] is True
    assert global_context["archive_ready"] is False
    assert global_context["raw_equal"] is True
    assert global_context["q_roundtrip_equal"] is True
    assert global_context["context_count"] <= context_range["context_count"]
    assert global_context["context_token_count"] == context_range["context_token_count"]
    assert "byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales" in global_context
    mixed_context = next(
        row
        for row in profile["variants"]
        if row["variant"]
        == "mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales"
    )
    assert mixed_context["codec"] == "HDM2_global_prev_symbol_mixed_range_raw_schema_indexed_uint8"
    assert mixed_context["parity_fixture"] is True
    assert mixed_context["archive_ready"] is False
    assert mixed_context["raw_equal"] is True
    assert mixed_context["q_roundtrip_equal"] is True
    assert mixed_context["schema_indexed"] is True
    assert mixed_context["raw_context_count"] + mixed_context["range_context_count"] == mixed_context[
        "context_count"
    ]
    assert mixed_context["mixed_payload_bytes"] == (
        mixed_context["range_payload_bytes"] + mixed_context["raw_payload_bytes"]
    )
    assert "byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales" in mixed_context
    hdm3 = next(
        row
        for row in profile["variants"]
        if row["variant"]
        == "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"
    )
    assert hdm3["codec"] == "HDM3_fixed_schema_q_brotli_raw_scales"
    assert hdm3["parity_fixture"] is True
    assert hdm3["archive_ready"] is False
    assert hdm3["raw_equal"] is True
    assert hdm3["q_roundtrip_equal"] is True
    assert hdm3["scale_roundtrip_equal"] is True
    assert hdm3["q_stream_bytes"] == profile["q_stream_bytes"]
    assert hdm3["raw_scale_bytes"] == profile["scale_stream_bytes"]
    assert hdm3["header_bytes"] == 7
    assert "byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales" in hdm3
    hdm6 = next(
        row
        for row in profile["variants"]
        if row["variant"]
        == "hdm6_q_brotli_split_fixed_recipe_tuned_lgwin_plus_raw_scales"
    )
    assert hdm6["codec"] == "HDM6_fixed_recipe_tuned_q_brotli_raw_scales"
    assert hdm6["parity_fixture"] is True
    assert hdm6["archive_ready"] is False
    assert hdm6["raw_equal"] is True
    assert hdm6["q_roundtrip_equal"] is True
    assert hdm6["scale_roundtrip_equal"] is True
    assert hdm6["q_stream_bytes"] == profile["q_stream_bytes"]
    assert hdm6["raw_scale_bytes"] == profile["scale_stream_bytes"]
    assert hdm6["header_bytes"] == 17
    assert len(hdm6["brotli_params_by_chunk"]) == 4
    assert "byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales" in hdm6
    plan = profile["context_overhead_plan"]
    assert plan["score_claim"] is False
    assert plan["planning_only"] is True
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["hdc1_to_hdc2_header_savings_bytes"] == (
        context_range["header_bytes"] - global_context["header_bytes"]
    )
    assert plan["hdc2_gap_to_prev_symbol_entropy_floor_plus_raw_scales_bytes"] == (
        global_context["bytes"]
        - entropy["per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"]
    )
    assert plan["break_even_reduction_vs_source_from_hdc2_bytes"] == max(
        0,
        global_context["byte_delta_vs_source_section"] + 1,
    )
    assert plan["largest_accounted_gap"]["status"] == "already_realized_by_hdc2_fixture"
    assert plan["largest_remaining_safe_target"] == plan["remaining_target_ranking"][0]
    bounded_candidate = plan["bounded_hdc2_mixed_context_candidate"]
    assert bounded_candidate["variant"] == mixed_context["variant"]
    assert bounded_candidate["score_claim"] is False
    assert bounded_candidate["ready_for_exact_eval_dispatch"] is False
    assert bounded_candidate["byte_reduction_vs_hdc2_bytes"] == (
        global_context["bytes"] - mixed_context["bytes"]
    )
    assert bounded_candidate["static_context_header_reduction_vs_hdc2_bytes"] == (
        global_context["header_bytes"] - mixed_context["header_bytes"]
    )
    assert bounded_candidate["payload_delta_vs_hdc2_bytes"] == (
        mixed_context["mixed_payload_bytes"] - global_context["range_payload_bytes"]
    )


def test_context_range_fixture_roundtrips_and_is_deterministic() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))

    first, first_stats = encode_prev_symbol_context_range_fixture(parsed)
    second, second_stats = encode_prev_symbol_context_range_fixture(parsed)
    restored = decode_prev_symbol_context_range_fixture(first)

    assert first == second
    assert first_stats == second_stats
    assert first.startswith(b"HDC1")
    assert restored.to_raw() == parsed.to_raw()
    assert restored.q_stream == parsed.q_stream
    assert restored.scale_stream == parsed.scale_stream
    assert first_stats["context_count"] > 0
    assert first_stats["context_token_count"] == len(parsed.q_stream) - len(parsed.records)
    assert first_stats["header_bytes"] + first_stats["range_payload_bytes"] + len(parsed.scale_stream) <= len(first)


def test_global_context_range_fixture_roundtrips_and_amortizes_context_table() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))

    per_tensor, per_tensor_stats = encode_prev_symbol_context_range_fixture(parsed)
    global_payload, global_stats = encode_global_prev_symbol_context_range_fixture(parsed)
    restored = decode_global_prev_symbol_context_range_fixture(global_payload)

    assert global_payload.startswith(b"HDC2")
    assert restored.to_raw() == parsed.to_raw()
    assert restored.q_stream == parsed.q_stream
    assert restored.scale_stream == parsed.scale_stream
    assert global_stats["context_token_count"] == per_tensor_stats["context_token_count"]
    assert global_stats["context_count"] <= per_tensor_stats["context_count"]
    assert global_stats["header_bytes"] < per_tensor_stats["header_bytes"]
    assert len(global_payload) < len(per_tensor)


def test_mixed_global_context_fixture_roundtrips_and_reduces_hdc2_bytes() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))

    global_payload, global_stats = encode_global_prev_symbol_context_range_fixture(parsed)
    mixed_payload, mixed_stats = encode_global_prev_symbol_mixed_context_fixture(parsed)
    restored = decode_global_prev_symbol_mixed_context_fixture(mixed_payload)

    assert mixed_payload.startswith(b"HDM2")
    assert restored.to_raw() == parsed.to_raw()
    assert restored.q_stream == parsed.q_stream
    assert restored.scale_stream == parsed.scale_stream
    assert mixed_stats["context_token_count"] == global_stats["context_token_count"]
    assert mixed_stats["raw_context_count"] + mixed_stats["range_context_count"] == mixed_stats[
        "context_count"
    ]
    assert mixed_stats["mixed_payload_bytes"] == (
        mixed_stats["range_payload_bytes"] + mixed_stats["raw_payload_bytes"]
    )
    assert mixed_stats["schema_metadata_elided_vs_hdc2_bytes"] > 0
    assert mixed_stats["header_bytes"] < global_stats["header_bytes"]
    assert len(mixed_payload) < len(global_payload)


def test_hdm3_q_brotli_split_fixture_roundtrips_and_reduces_raw_q_bytes() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))

    payload, stats = encode_hdm3_q_brotli_split_fixture(parsed)
    restored = decode_hdm3_q_brotli_split_fixture(payload)

    assert payload.startswith(b"HDM3")
    assert restored.to_raw() == parsed.to_raw()
    assert restored.q_stream == parsed.q_stream
    assert restored.scale_stream == parsed.scale_stream
    assert stats["header_bytes"] == 7
    assert stats["q_stream_bytes"] == len(parsed.q_stream)
    assert stats["raw_scale_bytes"] == len(parsed.scale_stream)
    assert stats["q_brotli_bytes"] < len(parsed.q_stream)
    assert len(payload) == 7 + stats["q_brotli_bytes"] + stats["raw_scale_bytes"]


def test_hdm6_tuned_q_brotli_fixture_roundtrips_with_fixed_recipe_params() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))

    payload, stats = encode_hdm6_q_brotli_tuned_fixture(parsed)
    restored = decode_hdm6_q_brotli_tuned_fixture(payload)

    assert payload.startswith(b"HDM6")
    assert restored.to_raw() == parsed.to_raw()
    assert restored.q_stream == parsed.q_stream
    assert restored.scale_stream == parsed.scale_stream
    assert stats["header_bytes"] == 17
    assert stats["recipe_id"] == 1
    assert stats["split_points"] == [6, 9, 26, 28]
    assert stats["brotli_params_by_chunk"] == [
        {"quality": 11, "lgwin": 18, "mode": brotli.MODE_GENERIC},
        {"quality": 11, "lgwin": 16, "mode": brotli.MODE_GENERIC},
        {"quality": 11, "lgwin": 16, "mode": brotli.MODE_GENERIC},
        {"quality": 10, "lgwin": 16, "mode": brotli.MODE_GENERIC},
    ]
    assert stats["non_arbitrary_selection"]["selected_mode_by_chunk"] == ["generic"] * 4
    assert len(payload) == stats["header_bytes"] + stats["q_brotli_bytes"] + stats["raw_scale_bytes"]


def test_parse_decoder_section_for_recode_accepts_hdm4_source_sections() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    hdm4_payload, _stats = encode_hdm4_q_brotli_split_fixture(parsed, quality=5)

    restored, raw, codec = parse_decoder_section_for_recode(hdm4_payload)

    assert codec == "hdm4_q_brotli_split"
    assert raw == parsed.to_raw()
    assert restored.to_raw() == parsed.to_raw()


def test_parse_decoder_section_for_recode_accepts_hdm6_source_sections() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    hdm6_payload, _stats = encode_hdm6_q_brotli_tuned_fixture(parsed)

    restored, raw, codec = parse_decoder_section_for_recode(hdm6_payload)

    assert codec == "hdm6_q_brotli_tuned_split"
    assert raw == parsed.to_raw()
    assert restored.to_raw() == parsed.to_raw()


def test_hdm5_planning_fixture_roundtrips_with_self_describing_byte_accounting() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    ordered_names = tuple(name for name, _shape in reversed(PACKED_STATE_SCHEMA))
    split_points = (4, 12, len(PACKED_STATE_SCHEMA))

    payload, stats = encode_hdm5_q_brotli_split_planning_fixture(
        parsed,
        ordered_record_names=ordered_names,
        split_points=split_points,
        quality=5,
    )
    restored = decode_hdm5_q_brotli_split_planning_fixture(payload)

    assert payload.startswith(b"HDM5")
    assert restored.to_raw() == parsed.to_raw()
    assert stats["planning_only"] is True
    assert stats["self_describing_order"] is True
    assert stats["record_order_metadata_bytes"] == len(PACKED_STATE_SCHEMA)
    assert stats["split_metadata_bytes"] == len(split_points)
    assert stats["length_prefix_bytes"] == 3 * len(split_points)
    assert len(payload) == (
        stats["header_bytes"] + stats["q_brotli_bytes"] + stats["raw_scale_bytes"]
    )


def test_hdm5_search_is_planning_only_and_compares_against_hdm4() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    hdm4_payload, _stats = encode_hdm4_q_brotli_split_fixture(parsed, quality=5)

    search = search_hdm5_q_brotli_split_recipes(
        parsed,
        baseline_section_bytes=len(hdm4_payload),
        hdm4_section_bytes=len(hdm4_payload),
        quality=5,
        max_parts=3,
        workers=1,
        top_k=4,
    )

    assert search["score_claim"] is False
    assert search["planning_only"] is True
    assert search["archive_ready"] is False
    assert search["ready_for_exact_eval_dispatch"] is False
    assert search["hdm4_section_bytes"] == len(hdm4_payload)
    assert search["candidate_count"] >= search["order_family_count"]
    assert len(search["top_candidates"]) == 4
    best = search["best_candidate"]
    assert best["raw_equal"] is True
    assert best["byte_delta_vs_hdm4_section"] == best["bytes"] - len(hdm4_payload)
    assert best["record_order_metadata_bytes"] == len(PACKED_STATE_SCHEMA)
    assert best["fixed_recipe_projected_bytes"] <= best["bytes"]
    assert search["best_fixed_recipe_projection"]["fixed_recipe_projection_contract"] == (
        "planning_only_assumes_runtime_hardcodes_order_and_split_recipe"
    )
    assert search["best_fixed_recipe_projection"]["fixed_recipe_projected_bytes"] <= best[
        "fixed_recipe_projected_bytes"
    ]


def test_hdm5_search_is_deterministic_across_serial_and_parallel_workers() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    hdm4_payload, _stats = encode_hdm4_q_brotli_split_fixture(parsed, quality=5)

    serial = search_hdm5_q_brotli_split_recipes(
        parsed,
        baseline_section_bytes=len(hdm4_payload),
        hdm4_section_bytes=len(hdm4_payload),
        quality=5,
        max_parts=4,
        workers=1,
        top_k=5,
    )
    parallel = search_hdm5_q_brotli_split_recipes(
        parsed,
        baseline_section_bytes=len(hdm4_payload),
        hdm4_section_bytes=len(hdm4_payload),
        quality=5,
        max_parts=4,
        workers=2,
        top_k=5,
    )

    assert parallel["best_candidate"] == serial["best_candidate"]
    assert parallel["top_candidates"] == serial["top_candidates"]
    assert parallel["family_summaries"] == serial["family_summaries"]


def test_hdm5_planning_fixture_rejects_non_permutation_order_indices() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    ordered_names = tuple(name for name, _shape in PACKED_STATE_SCHEMA)
    payload, _stats = encode_hdm5_q_brotli_split_planning_fixture(
        parsed,
        ordered_record_names=ordered_names,
        split_points=(len(PACKED_STATE_SCHEMA),),
        quality=5,
    )
    broken = bytearray(payload)
    broken[8] = broken[7]

    import pytest

    with pytest.raises(Exception, match="order indices are not a schema permutation"):
        decode_hdm5_q_brotli_split_planning_fixture(bytes(broken))


def test_structural_recode_profile_can_search_from_hdm4_archive_surface() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    hdm4_payload, _stats = encode_hdm4_q_brotli_split_fixture(parsed, quality=5)
    packed = parse_ff_packed_brotli_hnerv(_packed_payload(hdm4_payload))

    profile = build_structural_recode_profile(
        packed,
        source_label="fixture-hdm4",
        source_archive_sha256="b" * 64,
        include_hdm5_search=True,
        hdm5_max_parts=3,
        hdm5_workers=1,
        hdm5_top_k=3,
    )

    assert profile["source_decoder_section_codec"] == "hdm4_q_brotli_split"
    assert profile["source_decoder_section_bytes"] == len(hdm4_payload)
    assert profile["hdm5_search"]["score_claim"] is False
    assert profile["hdm5_search"]["best_candidate"]["raw_equal"] is True
    assert "best_fixed_recipe_projection" in profile["hdm5_search"]


def test_profile_hnerv_decoder_structural_recode_cli_includes_hdm5_search(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    hdm4_payload, _stats = encode_hdm4_q_brotli_split_fixture(parsed, quality=5)
    write_stored_single_member_zip(
        archive,
        member_name="x",
        payload=_packed_payload(hdm4_payload),
    )
    json_out = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "profile_hnerv_decoder_structural_recode.py"),
            "--source-archive",
            str(archive),
            "--source-label",
            "fixture-hdm4-cli",
            "--include-hdm5-search",
            "--hdm5-max-parts",
            "2",
            "--hdm5-workers",
            "1",
            "--hdm5-top-k",
            "2",
            "--json-out",
            str(json_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["source_label"] == "fixture-hdm4-cli"
    assert payload["source_decoder_section_codec"] == "hdm4_q_brotli_split"
    assert payload["hdm5_search"]["score_claim"] is False
    assert payload["hdm5_search"]["planning_only"] is True
    assert len(payload["hdm5_search"]["top_candidates"]) == 2


def test_mixed_global_context_fixture_rejects_schema_record_count_mismatch() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    payload, _stats = encode_global_prev_symbol_mixed_context_fixture(parsed)
    broken = bytearray(payload)
    broken[5] = broken[5] - 1

    import pytest

    with pytest.raises(Exception, match="fixed schema record count mismatch"):
        decode_global_prev_symbol_mixed_context_fixture(bytes(broken))


def test_context_range_fixture_rejects_duplicate_previous_symbol_context() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    payload, _stats = encode_prev_symbol_context_range_fixture(parsed)
    # Construct a targeted duplicate by replacing the second context key with
    # the first context key in the first record. The decoder must reject this
    # before any context overwrite can hide malformed custody.
    first_key, second_key = _first_two_hdc1_context_key_offsets(payload)
    duplicate = bytearray(payload)
    duplicate[second_key] = duplicate[first_key]

    import pytest

    with pytest.raises(Exception, match="duplicate previous-symbol context"):
        decode_prev_symbol_context_range_fixture(bytes(duplicate))


def test_profile_hnerv_decoder_structural_recode_cli(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(
        archive,
        member_name="x",
        payload=_packed_payload(brotli.compress(_synthetic_decoder_raw(), quality=5)),
    )
    json_out = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "profile_hnerv_decoder_structural_recode.py"),
            "--source-archive",
            str(archive),
            "--source-label",
            "fixture",
            "--json-out",
            str(json_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["source_label"] == "fixture"
    assert payload["score_claim"] is False
    assert payload["source_payload_kind"] == "raw_ff_hnerv"
    assert payload["source_decoder_section_name"] == "decoder_packed_brotli"
    assert payload["best_variant"]["raw_equal"] is True


def test_profile_hnerv_decoder_structural_recode_cli_supports_pr106_sidecar(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    inner = _packed_payload(brotli.compress(_synthetic_decoder_raw(), quality=5))
    sidecar = emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=PR106_SIDECAR_FORMAT_BROTLI,
            pr106_bytes=inner,
            sidecar_payload=brotli.compress(b"\x00\x00", quality=5),
        )
    )
    write_stored_single_member_zip(archive, member_name="0.bin", payload=sidecar)
    json_out = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "profile_hnerv_decoder_structural_recode.py"),
            "--source-archive",
            str(archive),
            "--source-label",
            "fixture-sidecar",
            "--json-out",
            str(json_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["source_label"] == "fixture-sidecar"
    assert payload["score_claim"] is False
    assert payload["source_payload_kind"] == "pr106_sidecar_wrapper"
    assert payload["source_decoder_section_name"] == "inner_decoder_packed_brotli"
    assert payload["best_variant"]["raw_equal"] is True


def _synthetic_decoder_raw() -> bytes:
    q_parts = []
    scale_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = 1
        for dim in shape:
            count *= dim
        q_parts.append(bytes((i + index) % 256 for i in range(count)))
        scale_parts.append((index + 1).to_bytes(4, "little"))
    return b"".join(q_parts) + b"".join(scale_parts)


def _synthetic_context_decoder_raw() -> bytes:
    q_parts = []
    scale_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = 1
        for dim in shape:
            count *= dim
        pattern = bytes(((index + i // 5) % 17) for i in range(64))
        repeats, remainder = divmod(count, len(pattern))
        q_parts.append(pattern * repeats + pattern[:remainder])
        scale_parts.append((index + 1).to_bytes(4, "little"))
    return b"".join(q_parts) + b"".join(scale_parts)


def _first_two_hdc1_context_key_offsets(payload: bytes) -> tuple[int, int]:
    def read_varint(cursor: int) -> tuple[int, int]:
        value = 0
        shift = 0
        while True:
            byte = payload[cursor]
            cursor += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value, cursor
            shift += 7

    cursor = 4
    _version, cursor = read_varint(cursor)
    _record_count, cursor = read_varint(cursor)
    name_len, cursor = read_varint(cursor)
    cursor += name_len
    _value_count, cursor = read_varint(cursor)
    cursor += 1  # first symbol
    context_count, cursor = read_varint(cursor)
    assert context_count >= 2
    offsets = []
    for _ in range(2):
        offsets.append(cursor)
        cursor += 1  # previous symbol
        _token_count, cursor = read_varint(cursor)
        unique_count, cursor = read_varint(cursor)
        payload_len, cursor = read_varint(cursor)
        for _symbol_index in range(unique_count):
            _delta, cursor = read_varint(cursor)
            _frequency, cursor = read_varint(cursor)
        cursor += payload_len
    return offsets[0], offsets[1]


def _packed_payload(decoder_brotli: bytes) -> bytes:
    latents = brotli.compress(b"latents", quality=5)
    return bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents
