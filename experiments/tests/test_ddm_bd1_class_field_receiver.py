# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import sys

import brotli
import numpy as np
import pytest

from experiments import ddm_bd1_class_field_receiver as bd1
from experiments import ddm_bf1_band_field_repr_race as bf1


def test_bitpack_padding_is_checked() -> None:
    payload = bd1.pack_bits(np.array([1, 0, 1], dtype=bool))
    assert bd1.unpack_bits(payload, 3).tolist() == [True, False, True]
    corrupted = bytes([payload[0] | 0b0001])
    with pytest.raises(bd1.BD1Error, match="nonzero padding"):
        bd1.unpack_bits(corrupted, 3)


def test_bd1_empty_n600_section_roundtrips_and_closes() -> None:
    band_bytes = (bd1.SEG_H * bd1.SEG_W + 7) // 8
    raw = b"\x00" * (bd1.N_PAIRS * band_bytes)
    body = brotli.compress(raw, quality=11)
    section = (
        bd1.BD1_HEADER.pack(
            bd1.BD1_MAGIC,
            bd1.BD1_VERSION,
            bd1.SEG_H,
            bd1.SEG_W,
            bd1.N_PAIRS,
            1,
            bd1.ROAD,
            bd1.LANE,
            1,
            bd1.BD1_BROTLI_Q11,
            len(raw),
            band_bytes,
            hashlib.sha256(raw).digest(),
        )
        + body
    )
    parsed = bd1.parse_class_field_section(section)
    assert parsed["codec"] == "brotli-q11"
    assert parsed["raw_bytes"] == len(raw)
    assert parsed["band_pixels"] == 0
    assert parsed["per_pair_band_pixels"] == [0] * bd1.N_PAIRS

    broken = bytearray(section)
    broken[8] = 99
    with pytest.raises(bd1.BD1Error, match="version"):
        bd1.parse_class_field_section(bytes(broken))


def test_receiver_source_accepts_tagged_optional_bd1_class_field() -> None:
    source = (bd1._REPO / "experiments/inflate_runner_v4d.py").read_text()
    assert "BD1_CLASS_FIELD_MAGIC = b\"BD1CLF1!\"" in source
    assert "BD1_CLASS_FIELD_HEADER_V2 = struct.Struct" in source
    assert "BD1_CLASS_FIELD_VERSION_V2 = 2" in source
    assert "_BD1_REPR_LANE_CROP = 2" in source
    assert "self._bd1_class_field = _bd1_parse_class_field(extra)" in source
    assert "elif extra[:8] == BD1_CLASS_FIELD_MAGIC" in source
    assert "frame = _bd1_apply_class_field(frame, self._bd1_class_field, i)" in source
    assert "PE1_EDGE_MAGIC = b\"PE1EDGE1\"" in source
    assert "self._pe1_edge_field = _pe1_parse_edge_field(extra)" in source
    assert "elif extra[:8] == PE1_EDGE_MAGIC" in source
    assert "frame = _pe1_apply_edge_field(frame, self._pe1_edge_field, i)" in source
    assert "PE3_EDGE_MAGIC = b\"PE3EDGE1\"" in source
    assert "self._pe3_edge_field = _pe3_parse_edge_field(extra)" in source
    assert "elif extra[:8] == PE3_EDGE_MAGIC" in source
    assert "frame = _pe1_apply_edge_field(frame, self._pe3_edge_field, i)" in source


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _zigzag(value: int) -> bytes:
    return _varint((value << 1) ^ (value >> 63))


def _build_pe1_section(runner, raw: bytes, kind: int) -> bytes:
    body = brotli.compress(raw, quality=11)
    return (
        runner.PE1_EDGE_HEADER.pack(
            runner.PE1_EDGE_MAGIC,
            runner.PE1_EDGE_VERSION,
            384,
            512,
            600,
            kind,
            runner._BD1_BROTLI_Q11,
            len(raw),
            600,
            hashlib.sha256(raw).digest(),
        )
        + body
    )


def _build_pe3_section(runner, raw: bytes) -> bytes:
    body = brotli.compress(raw, quality=11)
    return (
        runner.PE3_EDGE_HEADER.pack(
            runner.PE3_EDGE_MAGIC,
            runner.PE3_EDGE_VERSION,
            384,
            512,
            600,
            runner._PE3_HYBRID,
            runner._BD1_BROTLI_Q11,
            len(raw),
            600,
            hashlib.sha256(raw).digest(),
        )
        + body
    )


def _import_v4d_runner():
    runtime_path = str(bd1._REPO / "src/tac/optimization")
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    from experiments import inflate_runner_v4d as runner

    return runner


def test_pe1_empty_n600_section_roundtrips_and_closes() -> None:
    runner = _import_v4d_runner()

    raw = b"\x00" * 600
    section = _build_pe1_section(runner, raw, runner._PE1_CURVE)
    parsed = runner._pe1_parse_edge_field(section)
    reparsed = runner._pe1_parse_edge_field(section)

    assert parsed["kind_name"] == "explicit_curve_spline"
    assert parsed["raw_bytes"] == len(raw)
    assert parsed["component_records"] == 0
    assert sum(parsed["pair_counts"]) == 0
    assert parsed["raster_sha256"] == reparsed["raster_sha256"]

    broken = bytearray(section)
    broken[8] = 99
    with pytest.raises(SystemExit, match="version"):
        runner._pe1_parse_edge_field(bytes(broken))


def test_pe1_curve_record_paints_frame1_private_support() -> None:
    runner = _import_v4d_runner()

    record = (
        bytes([0, 1, 0])
        + _varint(1)
        + _varint(3)
        + _varint(1)
        + _varint(1)
        + _varint(1)
        + _zigzag(5)
    )
    raw = _varint(1) + _varint(len(record)) + record + (b"\x00" * 599)
    section = _build_pe1_section(runner, raw, runner._PE1_CURVE)
    parsed = runner._pe1_parse_edge_field(section)

    assert parsed["component_records"] == 1
    assert parsed["pair_counts"][0] == 3
    assert set(parsed["classes"][0].tolist()) == {0, 1}

    before = np.zeros((874, 1164, 3), dtype=np.uint8)
    after = runner._pe1_apply_edge_field(before, parsed, 0)
    assert np.any(after != before)


def test_pe3_hybrid_record_paints_mixed_curve_and_generator_modes() -> None:
    runner = _import_v4d_runner()

    curve_record = (
        bytes([0, 1, 0])
        + _varint(1)
        + _varint(3)
        + _varint(1)
        + _varint(1)
        + _varint(1)
        + _zigzag(5)
    )
    generator_record = runner._PE1_GENERATOR_RECORD.pack(0, 1, 8, 8, 10, 10, 32, 32, 64, 64)
    mixed_records = [
        bytes([runner._PE3_MODE_CURVE]) + curve_record,
        bytes([runner._PE3_MODE_GENERATOR]) + generator_record,
    ]
    first_frame = _varint(len(mixed_records)) + b"".join(
        _varint(len(record)) + record for record in mixed_records
    )
    raw = first_frame + (b"\x00" * 599)
    section = _build_pe3_section(runner, raw)
    parsed = runner._pe3_parse_edge_field(section)

    assert parsed["kind_name"] == "hybrid_per_regime"
    assert parsed["mode_counts"] == {
        "depth_conditioned_curve": 1,
        "generator_pair_bisector": 1,
    }
    assert parsed["component_records"] == 2
    assert parsed["pair_counts"][0] > 3

    before = np.zeros((874, 1164, 3), dtype=np.uint8)
    after = runner._pe1_apply_edge_field(before, parsed, 0)
    assert np.any(after != before)


def test_bf1_v2_lane_crop_empty_section_roundtrips() -> None:
    raw = b"\x00" * (bf1.N_PAIRS * 8)
    body = brotli.compress(raw, quality=11)
    section = (
        bf1.BD1_HEADER_V2.pack(
            bd1.BD1_MAGIC,
            bf1.BD1_VERSION_V2,
            bf1.SEG_H,
            bf1.SEG_W,
            bf1.N_PAIRS,
            1,
            bf1.ROAD,
            bf1.LANE,
            1,
            bf1.REPR_LANE_CROP,
            bd1.BD1_BROTLI_Q11,
            len(raw),
            0,
            hashlib.sha256(raw).digest(),
        )
        + body
    )
    codec, fields = bf1.parse_section_v2(section)
    assert codec == "brotli-q11"
    assert len(fields) == bf1.N_PAIRS
    assert all(field.indices.size == 0 for field in fields)
