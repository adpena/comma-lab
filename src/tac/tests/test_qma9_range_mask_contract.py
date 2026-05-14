# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
import struct
import zipfile
from pathlib import Path

import pytest

from tac.qma9_range_mask_contract import (
    QMA9ContractError,
    analyze_qma9_vertical_block_copy_opportunities,
    decode_qma9_first_row_specialization_mask,
    decode_qma9_mask,
    decode_qma9_prefix_frames,
    decode_qma9_vertical_block_escape_mask,
    encode_qma9_first_row_specialization_mask,
    encode_qma9_mask,
    encode_qma9_vertical_block_escape_mask,
    pack_router_actions,
    parse_qma9_first_row_specialization_header,
    parse_qma9_header,
    parse_qma9_vertical_block_escape_header,
    rate_break_even,
    read_single_member_zip,
    sha256_bytes,
    slice_payload_segments,
    split_qma9_pr81_payload,
    trace_qma9_prefix,
    unpack_router_actions,
)


def _write_zip(path: Path, members: dict[str, bytes], *, compress_type: int = zipfile.ZIP_STORED) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = compress_type
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload)


def test_parse_qma9_header_records_static_mask_contract() -> None:
    payload = struct.pack("<4sIIII", b"QMA9", 600, 512, 384, 3) + b"abc" + b"tail"

    header = parse_qma9_header(payload)

    assert header.magic == "QMA9"
    assert header.frame_count == 600
    assert header.width == 512
    assert header.height == 384
    assert header.bitstream_bytes == 3
    assert header.packed_bytes == 23
    assert header.decoded_mask_bytes == 117_964_800
    assert header.bitstream_sha256 == sha256_bytes(b"abc")
    assert header.payload_sha256 == sha256_bytes(payload[:23])


def test_parse_qma9_header_fails_closed_on_bad_magic_or_overrun() -> None:
    bad_magic = struct.pack("<4sIIII", b"NOPE", 1, 2, 3, 0)
    with pytest.raises(QMA9ContractError, match="expected QMA9 magic"):
        parse_qma9_header(bad_magic)

    overrun = struct.pack("<4sIIII", b"QMA9", 1, 2, 3, 4) + b"abc"
    with pytest.raises(QMA9ContractError, match="declares 4 bytes"):
        parse_qma9_header(overrun)


def test_read_single_member_zip_requires_exact_stored_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(archive, {"p": b"payload"})

    payload, custody = read_single_member_zip(archive)

    assert payload == b"payload"
    assert custody.member_name == "p"
    assert custody.member_bytes == 7
    assert custody.member_sha256 == sha256_bytes(b"payload")
    assert custody.zip_overhead_bytes == custody.archive_bytes - 7

    extra = tmp_path / "extra.zip"
    _write_zip(extra, {"p": b"payload", "sidecar": b"bad"})
    with pytest.raises(QMA9ContractError, match="exactly one file member"):
        read_single_member_zip(extra)

    compressed = tmp_path / "compressed.zip"
    _write_zip(compressed, {"p": b"payload"}, compress_type=zipfile.ZIP_DEFLATED)
    with pytest.raises(QMA9ContractError, match="ZIP_STORED"):
        read_single_member_zip(compressed)


def test_slice_payload_segments_requires_exact_coverage() -> None:
    segments = slice_payload_segments(
        b"aaabbbbcc",
        [
            ("range_mask.qma9", 3, "qma9"),
            ("model", 4, "brotli"),
            ("router", 2, "packed"),
        ],
    )

    assert [(s.name, s.offset, s.size_bytes, s.sha256) for s in segments] == [
        ("range_mask.qma9", 0, 3, sha256_bytes(b"aaa")),
        ("model", 3, 4, sha256_bytes(b"bbbb")),
        ("router", 7, 2, sha256_bytes(b"cc")),
    ]

    with pytest.raises(QMA9ContractError, match="consumed 3 bytes"):
        slice_payload_segments(b"aaaa", [("one", 3, "raw")])


def test_rate_break_even_is_static_rate_math_only() -> None:
    result = rate_break_even(
        candidate_bytes=215_960,
        reference_bytes=277_321,
        reference_label="PR79_S2",
    )

    expected_delta = -61_361 * 25.0 / 37_545_489
    assert result.delta_bytes == -61_361
    assert math.isclose(result.rate_score_delta_if_components_unchanged, expected_delta)
    assert math.isclose(result.component_worsening_budget_before_equal_score, -expected_delta)


def test_pure_python_qma9_codec_roundtrips_storage_order_mask() -> None:
    raw = bytes(
        (t + y + 2 * x) % 5
        for t in range(3)
        for y in range(5)
        for x in range(4)
    )

    encoded = encode_qma9_mask(raw, frame_count=3, width=5, height=4)
    decoded = decode_qma9_mask(encoded)

    assert parse_qma9_header(encoded).frame_count == 3
    assert decoded.data == raw
    assert decoded.sha256 == sha256_bytes(raw)
    assert encoded == encode_qma9_mask(raw, frame_count=3, width=5, height=4)


def test_decode_qma9_prefix_frames_returns_complete_prefix() -> None:
    raw = bytes((t + y + x) % 5 for t in range(3) for y in range(4) for x in range(3))
    encoded = encode_qma9_mask(raw, frame_count=3, width=4, height=3)

    prefix = decode_qma9_prefix_frames(encoded, frame_count=2)

    assert prefix == raw[: 2 * 4 * 3]
    with pytest.raises(QMA9ContractError, match="exceeds payload frames"):
        decode_qma9_prefix_frames(encoded, frame_count=4)


def test_qma9_vertical_block_escape_roundtrips_copy_blocks() -> None:
    frame0 = bytes([0, 0, 1, 1, 2, 2] * 4)
    frame1 = frame0
    raw = frame0 + frame1

    encoded = encode_qma9_vertical_block_escape_mask(raw, frame_count=2, width=4, height=6, block_width=2)
    decoded = decode_qma9_vertical_block_escape_mask(encoded)
    header = parse_qma9_vertical_block_escape_header(encoded)
    opportunities = analyze_qma9_vertical_block_copy_opportunities(
        raw,
        frame_count=2,
        width=4,
        height=6,
        block_width=2,
    )

    assert header.magic == "QMB1"
    assert header.block_width == 2
    assert decoded.data == raw
    assert decoded.sha256 == sha256_bytes(raw)
    assert opportunities["copied_blocks"] == opportunities["eligible_blocks"]
    assert opportunities["copied_pixel_fraction"] == pytest.approx(0.75)
    assert encoded == encode_qma9_vertical_block_escape_mask(raw, frame_count=2, width=4, height=6, block_width=2)


def test_qma9_vertical_block_escape_fails_closed_on_bad_input() -> None:
    with pytest.raises(QMA9ContractError, match="block width"):
        encode_qma9_vertical_block_escape_mask(b"\x00\x01", frame_count=1, width=1, height=2, block_width=0)

    bad = struct.pack("<4sIIIII", b"QMB1", 1, 1, 1, 1, 4) + b"abc"
    with pytest.raises(QMA9ContractError, match="declares 4 bytes"):
        parse_qma9_vertical_block_escape_header(bad)


def test_qma9_first_row_specialization_modes_roundtrip_storage_order_mask() -> None:
    raw = bytes(
        (t + y + x + (1 if y == 0 and x % 3 == 0 else 0)) % 5
        for t in range(3)
        for y in range(4)
        for x in range(5)
    )

    for mode_id in (1, 2, 3):
        encoded = encode_qma9_first_row_specialization_mask(raw, frame_count=3, width=4, height=5, mode_id=mode_id)
        decoded = decode_qma9_first_row_specialization_mask(encoded)
        header = parse_qma9_first_row_specialization_header(encoded)

        assert header.magic == "QMF1"
        assert header.mode_id == mode_id
        assert header.mode_name
        assert decoded.data == raw
        assert decoded.sha256 == sha256_bytes(raw)
        assert encoded == encode_qma9_first_row_specialization_mask(raw, frame_count=3, width=4, height=5, mode_id=mode_id)


def test_qma9_first_row_specialization_fails_closed_on_bad_mode() -> None:
    with pytest.raises(QMA9ContractError, match="unknown QMF1"):
        encode_qma9_first_row_specialization_mask(b"\x00\x01", frame_count=1, width=1, height=2, mode_id=99)

    bad = struct.pack("<4sIIIII", b"QMF1", 1, 1, 1, 99, 0)
    with pytest.raises(QMA9ContractError, match="unknown QMF1"):
        parse_qma9_first_row_specialization_header(bad)


def test_trace_qma9_prefix_records_decoder_state_and_prefix_roundtrip() -> None:
    raw = bytes(
        [
            0, 0, 1, 1,
            0, 2, 2, 1,
            3, 3, 2, 4,
        ]
    )
    encoded = encode_qma9_mask(raw, frame_count=1, width=3, height=4)

    trace = trace_qma9_prefix(encoded, max_pixels=len(raw), checkpoint_pixels=[0, 5, 11])

    assert trace["schema"] == "qma9_pure_python_prefix_trace_v1"
    assert trace["payload_sha256"] == parse_qma9_header(encoded).payload_sha256
    assert trace["decoded_prefix_pixels"] == len(raw)
    assert trace["decoded_prefix_sha256"] == sha256_bytes(raw)
    assert trace["prefix_self_roundtrip"]["matches_prefix"] is True
    assert trace["stage_counts"]["up_gate"] == len(raw)
    assert sum(trace["class_counts"].values()) == len(raw)
    assert [row["pixel_index"] for row in trace["checkpoints"]] == [0, 5, 11]
    assert trace["checkpoints"][0]["decoder"]["bits_consumed"] >= 32
    assert trace["decoder_state_after_prefix"]["low"] <= trace["decoder_state_after_prefix"]["high"]


def test_qma9_encoder_fails_closed_on_bad_shape_or_class() -> None:
    with pytest.raises(QMA9ContractError, match="expected 6"):
        encode_qma9_mask(b"\x00" * 5, frame_count=1, width=2, height=3)

    with pytest.raises(QMA9ContractError, match="class out of range"):
        encode_qma9_mask(b"\x00\x05", frame_count=1, width=1, height=2)


def test_router_action_pack_unpack_uses_pr81_little_endian_bit_order() -> None:
    actions = tuple([0, 1, 7, 3, 4, 5, 2, 6, 1, 0, 7])

    packed = pack_router_actions(actions)

    assert unpack_router_actions(packed, count=len(actions)) == actions
    with pytest.raises(QMA9ContractError, match="outside 3-bit range"):
        pack_router_actions([8])


def test_pr81_runtime_router_action_unpack_matches_contract() -> None:
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[3]
    runtime_path = repo / "submissions" / "robust_current" / "inflate_renderer.py"
    spec = importlib.util.spec_from_file_location("inflate_renderer_pr81_router_test", runtime_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    actions = [(idx * 5) % 8 for idx in range(600)]
    packed = pack_router_actions(actions)

    decoded = module._unpack_pr81_router_actions(packed)

    assert decoded.tolist() == actions


def test_split_qma9_pr81_payload_names_and_hashes_segments() -> None:
    payload = b"mask" + b"model" + b"pose" + b"router"

    split = split_qma9_pr81_payload(
        payload,
        range_mask_bytes=4,
        model_bytes=5,
        pose_bytes=4,
        router_bytes=6,
    )

    assert split.range_mask == b"mask"
    assert split.model == b"model"
    assert split.pose == b"pose"
    assert split.router == b"router"
    assert [(s.name, s.offset, s.size_bytes) for s in split.segments] == [
        ("range_mask.qma9", 0, 4),
        ("split_model_reordered.br_bundle", 4, 5),
        ("optimized_poses.qp1.br", 9, 4),
        ("router_actions.3bit", 13, 6),
    ]
