# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

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
    assert "self._bd1_class_field = _bd1_parse_class_field(class_field)" in source
    assert "elif extra[:8] == BD1_CLASS_FIELD_MAGIC" in source
    assert "frame = _bd1_apply_class_field(frame, self._bd1_class_field, i)" in source
    assert "len(IX2_JOINT_ORDER) + 2" in source


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
