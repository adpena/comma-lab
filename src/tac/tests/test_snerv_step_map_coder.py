# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the compact SNeRV step-map packet coder."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable

import numpy as np
import pytest

from tac.analysis.snerv_step_map_coder import (
    ADAPTIVE_MAGIC,
    MAGIC,
    SnervStepMapCoderError,
    decode_step_maps,
    encode_step_maps,
    encode_step_maps_adaptive,
    encode_step_maps_waterfill,
)


def _smooth_step_maps() -> list[np.ndarray]:
    yy, xx = np.mgrid[0:48, 0:64].astype(np.float32)
    base = 0.5 + 0.2 * np.sin(xx / 12.0) + 0.1 * np.cos(yy / 9.0)
    maps = []
    for i in range(8):
        maps.append(np.exp2(base + i * 0.02).astype(np.float32))
    return maps


def test_step_map_packet_roundtrips_shapes_and_positive_values() -> None:
    maps = _smooth_step_maps()
    packet = encode_step_maps(maps, bins=128)
    decoded = decode_step_maps(packet.packet)

    assert packet.packet.startswith(MAGIC)
    assert len(decoded) == len(maps)
    for ref, got in zip(maps, decoded, strict=True):
        assert got.shape == ref.shape
        assert np.all(got > 0)
    assert packet.max_relative_error < 0.01


def test_compact_packet_beats_fp32_lzma_on_smooth_maps() -> None:
    packet = encode_step_maps(_smooth_step_maps(), bins=64)

    assert packet.total_bytes < packet.fp32_lzma_baseline_bytes
    assert packet.score_claim is False
    assert packet.ready_for_exact_eval_dispatch is False


def test_bundled_packet_reports_savings_over_per_map_packets() -> None:
    packet = encode_step_maps(_smooth_step_maps(), bins=64)

    assert packet.per_map_packet_baseline_bytes > packet.total_bytes
    assert packet.bundle_savings_bytes == (
        packet.per_map_packet_baseline_bytes - packet.total_bytes
    )
    assert 0.0 < packet.bundle_savings_ratio < 1.0
    assert packet.unique_code_count > 1
    assert packet.code_entropy_bits_per_symbol > 0.0
    assert packet.code_entropy_ideal_bytes <= packet.packed_code_bytes


def test_packet_uses_shared_quantizer_so_more_bins_reduce_error() -> None:
    maps = _smooth_step_maps()
    coarse = encode_step_maps(maps, bins=16)
    fine = encode_step_maps(maps, bins=128)

    assert coarse.bits_per_code == 4
    assert fine.bits_per_code == 7
    assert fine.max_relative_error < coarse.max_relative_error
    assert fine.mean_relative_error < coarse.mean_relative_error


def test_subbyte_packet_packs_int4_and_int2_codes() -> None:
    maps = _smooth_step_maps()
    int4 = encode_step_maps(maps, bins=16)
    int2 = encode_step_maps(maps, bins=4)

    assert int4.code_storage == "packed_bits_lsb"
    assert int2.code_storage == "packed_bits_lsb"
    assert int4.packed_code_bytes * 2 == int4.code_count
    assert int2.packed_code_bytes * 4 == int2.code_count
    assert int2.packed_code_bytes < int4.packed_code_bytes
    assert decode_step_maps(int4.packet)[0].shape == maps[0].shape
    assert decode_step_maps(int2.packet)[0].shape == maps[0].shape


def test_adaptive_packet_bundles_maps_by_precision_groups() -> None:
    maps = _smooth_step_maps()
    importance = np.linspace(0.0, 1.0, len(maps))
    packet = encode_step_maps_adaptive(
        maps,
        map_importance=importance,
        bin_choices=(128, 16, 4),
    )
    decoded = decode_step_maps(packet.packet)

    assert packet.packet.startswith(ADAPTIVE_MAGIC)
    assert len(decoded) == len(maps)
    assert {group["bins"] for group in packet.groups} == {128, 16, 4}
    assert all(group["payload_bytes"] > 0 for group in packet.groups)
    assert packet.total_bytes < packet.fp32_lzma_baseline_bytes
    assert packet.score_claim is False
    for ref, got in zip(maps, decoded, strict=True):
        assert got.shape == ref.shape
        assert np.all(got > 0)


def test_adaptive_packet_can_encode_low_importance_maps_as_constant_rle() -> None:
    maps = _smooth_step_maps()
    importance = np.linspace(0.0, 1.0, len(maps))
    packet = encode_step_maps_adaptive(
        maps,
        map_importance=importance,
        bin_choices=(128, 16, 4),
        constant_importance_quantile=0.25,
    )
    decoded = decode_step_maps(packet.packet)
    constant_group = next(group for group in packet.groups if group["bins"] == 0)

    assert constant_group["kind"] == "constant_log2_fill"
    assert constant_group["payload_bytes"] == 0
    assert constant_group["packed_code_bytes"] == 0
    assert constant_group["code_storage"] == "run_length_constant_log2_f32"
    assert len(constant_group["map_indices"]) == 2
    assert constant_group["map_indices"] == [0, 1]
    for idx in constant_group["map_indices"]:
        assert decoded[idx].shape == maps[idx].shape
        assert np.all(decoded[idx] > 0)
        assert np.unique(decoded[idx]).size == 1


def test_adaptive_decode_rejects_duplicate_map_ownership() -> None:
    packet = encode_step_maps_adaptive(
        _smooth_step_maps(),
        map_importance=np.linspace(0.0, 1.0, 8),
        bin_choices=(128, 16, 4),
    )

    def mutate(header: dict[str, object]) -> None:
        groups = header["groups"]
        assert isinstance(groups, list)
        first_indices = groups[0]["map_indices"]
        assert isinstance(first_indices, list)
        second_indices = groups[1]["map_indices"]
        assert isinstance(second_indices, list)
        second_indices[0] = first_indices[0]

    with pytest.raises(SnervStepMapCoderError, match="duplicate adaptive map index"):
        decode_step_maps(_rewrite_adaptive_header(packet.packet, mutate))


def test_adaptive_decode_rejects_unconsumed_payload_bytes() -> None:
    packet = encode_step_maps_adaptive(
        _smooth_step_maps(),
        map_importance=np.linspace(0.0, 1.0, 8),
        bin_choices=(128, 16, 4),
    )

    with pytest.raises(SnervStepMapCoderError, match="unused adaptive payload bytes"):
        decode_step_maps(packet.packet + b"x")


def test_adaptive_packet_rejects_bad_constant_quantile() -> None:
    with pytest.raises(SnervStepMapCoderError, match="constant_importance_quantile"):
        encode_step_maps_adaptive(
            _smooth_step_maps(),
            map_importance=np.linspace(0.0, 1.0, 8),
            constant_importance_quantile=1.5,
        )


def test_waterfill_packet_uses_fp16_and_constant_extremes() -> None:
    maps = _smooth_step_maps()[:5]
    importance = np.asarray([1e6, 1e3, 10.0, 1.0, 1e-6], dtype=np.float64)
    packet = encode_step_maps_waterfill(
        maps,
        map_importance=importance,
        target_bits_per_coeff=6.0,
    )
    decoded = decode_step_maps(packet.packet)

    assert packet.packet.startswith(ADAPTIVE_MAGIC)
    assert len(decoded) == len(maps)
    group_kinds = {group.get("kind") for group in packet.groups}
    assert "fp16_steps_lzma" in group_kinds
    assert "constant_log2_fill" in group_kinds
    assert all(group["payload_bytes"] > 0 for group in packet.groups if group["bins"] != 0)
    assert any(group["payload_bytes"] == 0 for group in packet.groups if group["bins"] == 0)
    assert packet.total_bytes < packet.fp32_lzma_baseline_bytes
    for ref, got in zip(maps, decoded, strict=True):
        assert got.shape == ref.shape
        assert np.all(got > 0)


def test_waterfill_zero_budget_is_all_constant_fill() -> None:
    maps = _smooth_step_maps()[:4]
    packet = encode_step_maps_waterfill(
        maps,
        map_importance=np.linspace(1.0, 4.0, len(maps)),
        target_bits_per_coeff=0.0,
    )
    decoded = decode_step_maps(packet.packet)

    assert len(packet.groups) == 1
    group = packet.groups[0]
    assert group["kind"] == "constant_log2_fill"
    assert group["map_indices"] == [0, 1, 2, 3]
    assert group["payload_bytes"] == 0
    for ref, got in zip(maps, decoded, strict=True):
        assert got.shape == ref.shape
        assert np.unique(got).size == 1


def test_adaptive_decode_accepts_shared_shape_constant_group() -> None:
    maps = _smooth_step_maps()[:4]
    packet = encode_step_maps_waterfill(
        maps,
        map_importance=np.linspace(1.0, 4.0, len(maps)),
        target_bits_per_coeff=0.0,
    )

    def mutate(header: dict[str, object]) -> None:
        groups = header["groups"]
        assert isinstance(groups, list)
        group = groups[0]
        shape = group["shapes"][0]
        log2_value = group["log2_values"][0]
        group.clear()
        group.update(
            {
                "kind": "constant_log2_shared_shape",
                "precision_label": "constant",
                "bins": 0,
                "bits_per_code": 0,
                "code_storage": "run_length_constant_log2_shared_shape",
                "map_indices": [0, 1, 2, 3],
                "payload_offset": 0,
                "payload_bytes": 0,
                "packed_code_bytes": 0,
                "log2_value": log2_value,
                "shape": shape,
                "code_count": 0,
            }
        )

    decoded = decode_step_maps(_rewrite_adaptive_header(packet.packet, mutate))

    assert len(decoded) == len(maps)
    for got in decoded:
        assert got.shape == maps[0].shape
        assert np.unique(got).size == 1
        assert np.all(got > 0)


def test_large_constant_group_uses_binary_shared_shape_payload() -> None:
    maps = [
        np.full((8, 12), np.exp2(0.25 + index * 0.01), dtype=np.float32)
        for index in range(12)
    ]
    packet = encode_step_maps_waterfill(
        maps,
        map_importance=np.linspace(1.0, 12.0, len(maps)),
        target_bits_per_coeff=0.0,
    )
    decoded = decode_step_maps(packet.packet)

    assert len(packet.groups) == 1
    group = packet.groups[0]
    assert group["kind"] == "constant_log2_shared_shape_f64_lzma"
    assert group["code_storage"] == "run_length_constant_log2_shared_shape_f64_lzma"
    assert group["shape"] == [8, 12]
    assert "shapes" not in group
    assert "log2_values" not in group
    assert group["payload_bytes"] > 0
    assert group["raw_bytes"] == len(maps) * 8
    legacy_group = {
        **{
            key: value
            for key, value in group.items()
            if key
            not in {
                "kind",
                "code_storage",
                "shape",
                "payload_bytes",
                "packed_code_bytes",
                "raw_bytes",
                "log2_dtype",
            }
        },
        "kind": "constant_log2_fill",
        "code_storage": "run_length_constant_log2_f32",
        "payload_offset": 0,
        "payload_bytes": 0,
        "packed_code_bytes": 0,
        "log2_values": [
            float(np.mean(np.log2(arr.astype(np.float64)))) for arr in maps
        ],
        "shapes": [list(arr.shape) for arr in maps],
    }
    compact_json = json.dumps(group, separators=(",", ":"), sort_keys=True)
    legacy_json = json.dumps(legacy_group, separators=(",", ":"), sort_keys=True)
    assert len(compact_json) < len(legacy_json)
    for ref, got in zip(decoded, decode_step_maps(packet.packet), strict=True):
        np.testing.assert_array_equal(got, ref)
    for ref, got in zip(maps, decoded, strict=True):
        assert got.shape == ref.shape
        assert np.unique(got).size == 1
        assert np.all(got > 0)


def test_waterfill_rejects_negative_target_bits() -> None:
    with pytest.raises(SnervStepMapCoderError, match="target_bits_per_coeff"):
        encode_step_maps_waterfill(
            _smooth_step_maps(),
            map_importance=np.linspace(0.0, 1.0, 8),
            target_bits_per_coeff=-0.1,
        )


def test_decode_rejects_bad_magic() -> None:
    with pytest.raises(SnervStepMapCoderError):
        decode_step_maps(b"BAD!!")


def test_encode_rejects_nonpositive_steps() -> None:
    bad = np.ones((4, 4), dtype=np.float32)
    bad[0, 0] = 0.0
    with pytest.raises(SnervStepMapCoderError):
        encode_step_maps([bad])


def _rewrite_adaptive_header(
    packet: bytes,
    mutate: Callable[[dict[str, object]], None],
) -> bytes:
    offset = len(ADAPTIVE_MAGIC)
    header_len_size = struct.calcsize("<I")
    (header_len,) = struct.unpack("<I", packet[offset : offset + header_len_size])
    header_start = offset + header_len_size
    header_end = header_start + int(header_len)
    header = json.loads(packet[header_start:header_end].decode("utf-8"))
    mutate(header)
    new_header = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return (
        packet[:offset]
        + struct.pack("<I", len(new_header))
        + new_header
        + packet[header_end:]
    )
