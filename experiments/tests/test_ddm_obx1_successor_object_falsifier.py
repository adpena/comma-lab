# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import torch

from experiments import ddm_obx1_successor_object_falsifier as obx1


def test_lane_edge_band_is_lane_boundary_plus_one_cell_annulus() -> None:
    field = np.zeros((1, obx1.EVAL_H, obx1.EVAL_W), dtype=np.uint8)
    field[0, 100:103, 200:203] = obx1.LANE_CLASS
    band = obx1.lane_edge_band(field)

    assert band.dtype == np.bool_
    assert band[0, 101, 201]
    assert band[0, 99, 201]
    assert band[0, 103, 201]
    assert band[0, 101, 199]
    assert band[0, 101, 203]
    assert not band[0, 98, 201]


def test_delta_chunk_and_catalog_roundtrip_bit_exact() -> None:
    delta = np.zeros((2, obx1.CHANNELS, obx1.EVAL_H, obx1.EVAL_W), dtype=np.int8)
    delta[0, 0, 4, 5] = -127
    delta[1, 2, 6, 7] = 126

    chunk = obx1.encode_delta_chunk(delta, start=40)
    repeat = obx1.encode_delta_chunk(delta, start=40)
    start, decoded = obx1.decode_delta_chunk(chunk)
    catalog = obx1.pack_catalog([chunk, repeat])

    assert chunk == repeat
    assert start == 40
    assert np.array_equal(decoded, delta)
    assert obx1.unpack_catalog(catalog) == [chunk, repeat]


def test_object_and_stored_zip_roundtrip_bit_exact() -> None:
    base = b"base-packet"
    carrier = obx1.pack_catalog([obx1.encode_delta_chunk(np.zeros((1, 3, 384, 512), dtype=np.int8), start=0)])
    packet = obx1.pack_object(base, carrier)
    archive = obx1.qbf1.deterministic_archive(packet, member_name="0.obx1")

    assert obx1.unpack_object(packet) == (base, carrier)
    assert obx1.qbf1.read_deterministic_archive(archive, member_name="0.obx1") == packet


def test_fuse_camera_changes_only_second_frame() -> None:
    born = torch.zeros((1, 2, 3, obx1.CAMERA_H, obx1.CAMERA_W), dtype=torch.float32)
    delta = np.zeros((1, 3, obx1.EVAL_H, obx1.EVAL_W), dtype=np.int8)
    delta[:, :, 100:102, 200:202] = 7

    fused = obx1.fuse_camera(born, delta)

    assert torch.equal(fused[:, 0], born[:, 0])
    assert int(fused[:, 1].max()) == 7
    assert int(torch.count_nonzero(fused[:, 1])) > 0


def test_strict_byte_cap_is_strict() -> None:
    cap = obx1.strict_byte_cap(obx1.POINTER_DISTORTION)

    assert obx1.POINTER_DISTORTION + 25.0 * cap / obx1.RATE_DENOMINATOR < obx1.TARGET_SCORE
    assert obx1.POINTER_DISTORTION + 25.0 * (cap + 1) / obx1.RATE_DENOMINATOR >= obx1.TARGET_SCORE
