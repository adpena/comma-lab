from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    build_structural_recode_profile,
    decode_prev_symbol_context_range_fixture,
    encode_prev_symbol_context_range_fixture,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import parse_ff_packed_brotli_hnerv, write_stored_single_member_zip

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
