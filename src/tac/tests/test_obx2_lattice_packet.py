# SPDX-License-Identifier: MIT
"""Behaviour tests for the OBX2 packet grammar and reference lattice receiver."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from tac import obx2_lattice_packet as pkt


def _spec(**overrides: object) -> pkt.LatticeSpec:
    base = {
        "levels": ((4, 3, 5), (8, 6, 10)),
        "channels": 2,
        "bits": (8, 4),
        "gate_kind": 1,
        "quantizer_kind": 0,
        "entropy_model": 0,
        "condition_channels": 3,
        "hidden": 7,
        "outputs": 6,
    }
    base.update(overrides)
    return pkt.LatticeSpec(**base)  # type: ignore[arg-type]


def _payload(spec: pkt.LatticeSpec, *, seed: int = 3) -> tuple[list[np.ndarray], list[float], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    codes = []
    for count, bits in zip(spec.codes_per_level(), spec.bits, strict=True):
        limit = 1 << (bits - 1)
        codes.append(rng.integers(-limit, limit, size=count, dtype=np.int64))
    scales = [0.25, 0.125]
    fusion = {
        "hidden_w": rng.standard_normal((spec.feature_width, spec.hidden)).astype(np.float32),
        "hidden_b": rng.standard_normal(spec.hidden).astype(np.float32),
        "out_w": rng.standard_normal((spec.hidden, spec.outputs)).astype(np.float32),
        "out_b": rng.standard_normal(spec.outputs).astype(np.float32),
    }
    return codes, scales, fusion


def test_lattice_spec_refuses_malformed_geometry() -> None:
    with pytest.raises(pkt.OBX2PacketError):
        _spec(levels=())
    with pytest.raises(pkt.OBX2PacketError):
        _spec(bits=(8,))
    with pytest.raises(pkt.OBX2PacketError):
        _spec(bits=(8, 7))
    with pytest.raises(pkt.OBX2PacketError):
        _spec(levels=((0, 3, 5), (8, 6, 10)))
    with pytest.raises(pkt.OBX2PacketError):
        _spec(gate_kind=9)
    with pytest.raises(pkt.OBX2PacketError):
        _spec(quantizer_kind=9)
    with pytest.raises(pkt.OBX2PacketError):
        _spec(entropy_model=9)
    with pytest.raises(pkt.OBX2PacketError):
        _spec(channels=0)


def test_feature_width_and_code_counts_follow_the_declared_geometry() -> None:
    spec = _spec()
    assert spec.feature_width == 2 * 2 + 3
    assert spec.codes_per_level() == (4 * 3 * 5 * 2, 8 * 6 * 10 * 2)
    described = spec.describe()
    assert described["gate_kind"] == "interface_softgate"
    assert described["total_codes"] == sum(spec.codes_per_level())


@pytest.mark.parametrize("bits", pkt.SUPPORTED_BITS)
@pytest.mark.parametrize("count", [1, 2, 7, 64])
def test_code_packing_round_trips_exactly(bits: int, count: int) -> None:
    limit = 1 << (bits - 1)
    rng = np.random.default_rng(bits * 100 + count)
    codes = rng.integers(-limit, limit, size=count, dtype=np.int64)
    packed = pkt.pack_codes(codes, bits)
    assert len(packed) == pkt.packed_length(count, bits)
    assert np.array_equal(pkt.unpack_codes(packed, count, bits), codes)


@pytest.mark.parametrize("bits", pkt.SUPPORTED_BITS)
def test_code_packing_refuses_out_of_range_values(bits: int) -> None:
    limit = 1 << (bits - 1)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.pack_codes(np.asarray([limit], dtype=np.int64), bits)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.pack_codes(np.asarray([-limit - 1], dtype=np.int64), bits)


def test_unpack_refuses_a_wrong_length_payload() -> None:
    with pytest.raises(pkt.OBX2PacketError):
        pkt.unpack_codes(b"\x00\x00", 4, 8)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.unpack_codes(b"\x00", 4, 4)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.unpack_codes(b"\x00", 4, 16)


def test_lattice_section_round_trips_every_declared_field() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    raw = pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=fusion)
    got_spec, got_codes, got_scales, got_fusion = pkt.decode_lattice_section(raw)
    assert got_spec == spec
    assert got_scales == scales
    for want, have in zip(codes, got_codes, strict=True):
        assert np.array_equal(want, have)
    for name in fusion:
        assert np.array_equal(fusion[name], got_fusion[name])


def test_lattice_section_refuses_crc_trailing_and_truncation() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    raw = pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=fusion)
    corrupted = bytearray(raw)
    corrupted[-1] ^= 0xFF
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_lattice_section(bytes(corrupted))
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_lattice_section(raw + b"\x00")
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_lattice_section(raw[:-4])
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_lattice_section(b"NOTLAT01" + raw[8:])


def test_lattice_section_refuses_a_nonfinite_or_nonpositive_scale() -> None:
    spec = _spec()
    codes, _, fusion = _payload(spec)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.encode_lattice_section(spec, codes=codes, scales=[0.0, 0.5], fusion=fusion)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.encode_lattice_section(spec, codes=codes, scales=[float("nan"), 0.5], fusion=fusion)


def test_lattice_section_refuses_wrong_fusion_shapes_and_nonfinite_values() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    broken = dict(fusion)
    broken["out_b"] = np.zeros(spec.outputs + 1, dtype=np.float32)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=broken)
    missing = {name: value for name, value in fusion.items() if name != "out_b"}
    with pytest.raises(pkt.OBX2PacketError):
        pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=missing)
    nonfinite = dict(fusion)
    nonfinite["hidden_b"] = np.full(spec.hidden, np.inf, dtype=np.float32)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=nonfinite)


def test_lattice_section_refuses_a_level_code_count_mismatch() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    codes[0] = codes[0][:-1]
    with pytest.raises(pkt.OBX2PacketError):
        pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=fusion)


def _sections() -> list[pkt.RacedSection]:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    lattice = pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=fusion)
    raws = {
        pkt.SECTION_CONFIG: b"config-bytes" * 4,
        pkt.SECTION_MODEL: bytes(range(256)) * 3,
        pkt.SECTION_LATENT_META: b"\x01\x02\x03\x04",
        pkt.SECTION_LATENTS: b"latent" * 16,
        pkt.SECTION_LATTICE: lattice,
    }
    return [pkt.race_section(section_id, raw) for section_id, raw in raws.items()]


def test_race_section_verifies_every_coder_and_keeps_the_smallest() -> None:
    raw = b"".join(bytes([value % 251]) for value in range(5_000))
    raced = pkt.race_section(pkt.SECTION_LATTICE, raw)
    assert raced.raw_bytes == len(raw)
    assert len(raced.candidates) == len(pkt.CODEC_IDS)
    assert len(raced.payload) == min(size for _, size in raced.candidates)
    assert pkt.decompress(pkt.CODEC_NAMES[raced.codec_id], raced.payload) == raw
    with pytest.raises(pkt.OBX2PacketError):
        pkt.race_section(99, raw)


def test_packet_round_trips_and_preserves_every_section_body() -> None:
    sections = _sections()
    packet = pkt.pack_obx2_packet(sections)
    decoded = pkt.decode_obx2_packet(packet)
    assert set(decoded) == set(pkt.SECTION_NAMES)
    for section in sections:
        assert pkt.sha256_bytes(decoded[section.section_id]) == section.raw_sha256


def test_packet_refuses_trailing_content_and_a_bad_magic() -> None:
    packet = pkt.pack_obx2_packet(_sections())
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(packet + b"\x00")
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(b"XXXXXXXX" + packet[8:])
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(packet[: pkt._PACKET_HEADER.size - 1])


def test_packet_refuses_a_missing_section() -> None:
    sections = [row for row in _sections() if row.section_id != pkt.SECTION_LATTICE]
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(pkt.pack_obx2_packet(sections))


def test_packet_refuses_duplicate_sections_at_pack_time() -> None:
    sections = _sections()
    with pytest.raises(pkt.OBX2PacketError):
        pkt.pack_obx2_packet([*sections, sections[0]])
    with pytest.raises(pkt.OBX2PacketError):
        pkt.pack_obx2_packet([])


def test_packet_refuses_a_corrupted_section_payload() -> None:
    packet = bytearray(pkt.pack_obx2_packet(_sections()))
    packet[-1] ^= 0xFF
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(bytes(packet))


def test_packet_refuses_an_unknown_section_id() -> None:
    sections = _sections()
    packet = bytearray(pkt.pack_obx2_packet(sections))
    struct.pack_into(">B", packet, pkt._PACKET_HEADER.size, 9)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(bytes(packet))


def test_packet_refuses_an_unknown_codec_id() -> None:
    sections = _sections()
    packet = bytearray(pkt.pack_obx2_packet(sections))
    struct.pack_into(">B", packet, pkt._PACKET_HEADER.size + 1, 7)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(bytes(packet))


def test_packet_crc_guard_is_independent_of_the_body_hash() -> None:
    sections = _sections()
    packet = bytearray(pkt.pack_obx2_packet(sections))
    offset = pkt._PACKET_HEADER.size + pkt._SECTION_HEADER.size - 4
    (crc,) = struct.unpack_from(">I", packet, offset)
    struct.pack_into(">I", packet, offset, (crc + 1) & 0xFFFFFFFF)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.decode_obx2_packet(bytes(packet))
    assert zlib.crc32(b"") == 0


def test_trilinear_sample_is_exact_at_grid_corners() -> None:
    grid = np.arange(2 * 2 * 2 * 1, dtype=np.float32).reshape(2, 2, 2, 1)
    corners = {
        (-1.0, -1.0, -1.0): 0.0,
        (-1.0, -1.0, 1.0): 1.0,
        (-1.0, 1.0, -1.0): 2.0,
        (1.0, 1.0, 1.0): 7.0,
    }
    for (t, y, x), want in corners.items():
        got = pkt.trilinear_sample(grid, np.asarray(t), np.asarray(y), np.asarray(x))
        assert float(got.reshape(-1)[0]) == pytest.approx(want)
    centre = pkt.trilinear_sample(grid, np.asarray(0.0), np.asarray(0.0), np.asarray(0.0))
    assert float(centre.reshape(-1)[0]) == pytest.approx(3.5)


def test_trilinear_sample_clamps_out_of_range_coordinates() -> None:
    grid = np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1)
    inside = pkt.trilinear_sample(grid, np.asarray(1.0), np.asarray(1.0), np.asarray(1.0))
    outside = pkt.trilinear_sample(grid, np.asarray(4.0), np.asarray(4.0), np.asarray(4.0))
    assert float(inside.reshape(-1)[0]) == float(outside.reshape(-1)[0])


def test_query_lattice_emits_one_output_row_per_queried_point() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    grids = [
        pkt.dequantize_level(code, scale, (*level, spec.channels))
        for code, scale, level in zip(codes, scales, spec.levels, strict=True)
    ]
    points = 11
    rng = np.random.default_rng(5)
    t = rng.uniform(-1.0, 1.0, points).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, points).astype(np.float32)
    x = rng.uniform(-1.0, 1.0, points).astype(np.float32)
    condition = rng.standard_normal((points, spec.condition_channels)).astype(np.float32)
    out = pkt.query_lattice_numpy(spec, grids, fusion, t=t, y=y, x=x, condition=condition)
    assert out.shape == (points, spec.outputs)
    assert np.isfinite(out).all()
    repeat = pkt.query_lattice_numpy(spec, grids, fusion, t=t, y=y, x=x, condition=condition)
    assert np.array_equal(out, repeat)


def test_query_lattice_refuses_a_condition_or_grid_mismatch() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    grids = [
        pkt.dequantize_level(code, scale, (*level, spec.channels))
        for code, scale, level in zip(codes, scales, spec.levels, strict=True)
    ]
    coords = np.zeros(3, dtype=np.float32)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.query_lattice_numpy(
            spec, grids, fusion, t=coords, y=coords, x=coords, condition=np.zeros((3, 1), dtype=np.float32)
        )
    with pytest.raises(pkt.OBX2PacketError):
        pkt.query_lattice_numpy(
            spec,
            grids[:1],
            fusion,
            t=coords,
            y=coords,
            x=coords,
            condition=np.zeros((3, spec.condition_channels), dtype=np.float32),
        )


def test_lattice_grids_rebuild_every_level_in_declaration_order() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    raw = pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=fusion)
    got_spec, got_codes, got_scales, _ = pkt.decode_lattice_section(raw)
    grids = pkt.lattice_grids(got_spec, got_codes, got_scales)
    assert [grid.shape for grid in grids] == [(*level, spec.channels) for level in spec.levels]
    assert np.allclose(grids[0].reshape(-1), codes[0].astype(np.float32) * np.float32(scales[0]))
    with pytest.raises(pkt.OBX2PacketError):
        pkt.lattice_grids(got_spec, got_codes[:1], got_scales)


def test_encode_refuses_a_shaped_level_array_with_the_wrong_geometry() -> None:
    spec = _spec()
    codes, scales, fusion = _payload(spec)
    depth, height, width = spec.levels[0]
    codes[0] = codes[0].reshape(depth, height, width, spec.channels).transpose(0, 2, 1, 3)
    with pytest.raises(pkt.OBX2PacketError):
        pkt.encode_lattice_section(spec, codes=codes, scales=scales, fusion=fusion)
