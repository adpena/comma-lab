from __future__ import annotations

import hashlib

import numpy as np
import pytest

from experiments import ddm_gc1_generator_capacity_control as gc1


def test_node_address_round_trip_across_depths() -> None:
    for pair in (0, 17, gc1.N_PAIRS - 1):
        for depth in range(5):
            edge = 2**depth - 1
            for row, col in ((0, 0), (edge, edge)):
                address = gc1.node_id(pair, depth, row, col, max_depth=4)
                assert gc1.decode_node_id(address, pairs=gc1.N_PAIRS, max_depth=4) == (pair, depth, row, col)


def test_overlay_round_trip_and_receiver_application() -> None:
    records = [
        (0, 1, 0, 0, 1),
        (0, 2, 0, 2, 2),
        (1, 0, 0, 0, 4),
    ]
    payload = gc1.encode_overlay(records, shape=(2, 8, 8), max_depth=2)
    shape, depth, decoded = gc1.decode_overlay(payload)
    assert shape == (2, 8, 8)
    assert depth == 2
    assert decoded == records
    assert gc1.encode_overlay(decoded, shape=shape, max_depth=depth) == payload

    output = np.zeros(shape, dtype=np.uint8)
    assert gc1.apply_overlay(payload, output) == 3
    assert np.all(output[0, :4, :4] == 1)
    assert np.all(output[0, :2, 4:6] == 2)
    assert np.all(output[1] == 4)


def test_overlay_rejects_overlap_and_trailing_bytes() -> None:
    overlapping = [(0, 0, 0, 0, 1), (0, 1, 0, 0, 2)]
    with pytest.raises(gc1.GC1Error, match="overlap"):
        gc1.encode_overlay(overlapping, shape=(1, 8, 8), max_depth=2)

    payload = gc1.encode_overlay([(0, 0, 0, 0, 1)], shape=(1, 8, 8), max_depth=2)
    with pytest.raises(gc1.GC1Error, match="trailing"):
        gc1.decode_overlay(payload + b"x")


def test_fit_capacity_control_changes_real_receiver_output() -> None:
    baseline = np.zeros((1, 8, 8), dtype=np.uint8)
    target = np.empty_like(baseline)
    target[0, :4, :4] = 1
    target[0, :4, 4:] = 2
    target[0, 4:, :4] = 3
    target[0, 4:, 4:] = 4

    expensive_records, expensive = gc1.fit_dyadic_overlay(target, baseline, max_depth=2, penalty=1_000)
    assert expensive_records == []
    assert expensive["dynamic_program_mismatches"] == 64

    free_records, free = gc1.fit_dyadic_overlay(target, baseline, max_depth=2, penalty=0)
    assert len(free_records) == 4
    assert free["dynamic_program_mismatches"] == 0
    payload = gc1.encode_overlay(free_records, shape=target.shape, max_depth=2)
    received = baseline.copy()
    gc1.apply_overlay(payload, received)
    np.testing.assert_array_equal(received, target)


def test_fit_breaks_objective_tie_toward_lower_distortion() -> None:
    baseline = np.zeros((1, 8, 8), dtype=np.uint8)
    target = np.ones_like(baseline)

    tied, tied_fit = gc1.fit_dyadic_overlay(target, baseline, max_depth=2, penalty=64)
    assert tied == [(0, 0, 0, 0, 1)]
    assert tied_fit["dynamic_program_mismatches"] == 0

    expensive, expensive_fit = gc1.fit_dyadic_overlay(target, baseline, max_depth=2, penalty=65)
    assert expensive == []
    assert expensive_fit["dynamic_program_mismatches"] == 64


def test_node_id_rejects_invalid_depth() -> None:
    with pytest.raises(gc1.GC1Error, match="depth"):
        gc1.node_id(0, -1, 0, 0, max_depth=2)
    with pytest.raises(gc1.GC1Error, match="depth"):
        gc1.node_id(0, 3, 0, 0, max_depth=2)


def test_compact_generator_packet_round_trip(tmp_path) -> None:
    rows = bytearray()
    bodies = bytearray()
    expected = {}
    coder = "zlib_9"
    for name in gc1.hg1.GENERATOR_STREAMS:
        raw = f"fixture:{name}".encode()
        coded = gc1.hg1.et1.compress_payload(raw, coder)
        rows.extend(
            gc1.hg1.PACKET_ROW.pack(
                gc1.hg1.STREAM_IDS[name],
                gc1.hg1.et1.CODER_IDS[coder],
                len(raw),
                len(coded),
                hashlib.sha256(raw).digest(),
                hashlib.sha256(coded).digest(),
            )
        )
        bodies.extend(coded)
        expected[name] = raw
    baseline = (
        gc1.hg1.PACKET_HEADER.pack(
            gc1.hg1.PACKET_MAGIC,
            gc1.hg1.PACKET_VERSION,
            len(gc1.hg1.GENERATOR_STREAMS),
            0,
        )
        + rows
        + bodies
    )
    overlay = gc1.encode_overlay([(0, 0, 0, 0, 1)], shape=(1, 8, 8), max_depth=2)
    raw_path = tmp_path / "overlay.raw"
    coded_path = tmp_path / "overlay.coded"
    raw_path.write_bytes(overlay)
    coded = gc1.hg1.et1.compress_payload(overlay, coder)
    coded_path.write_bytes(coded)
    race = {
        "winner": coder,
        "raw": gc1.file_fact(raw_path),
        "coders": {coder: {"coded": gc1.file_fact(coded_path)}},
    }

    packet_path = tmp_path / "generator.packet"
    fact = gc1.build_generator_packet(baseline, race, packet_path)
    streams, decoded_overlay = gc1.parse_generator_packet(packet_path.read_bytes())
    assert streams == expected
    assert decoded_overlay == overlay
    assert fact["bytes"] == gc1.GENERATOR_HEADER.size + len(baseline) + len(coded)
    assert gc1.GENERATOR_HEADER.size == 20
