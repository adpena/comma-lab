from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments import ddm_gdc1_scanline_partition_program as gdc1


def test_optimal_segments_is_exact_at_sufficient_capacity() -> None:
    row = np.asarray([0, 0, 1, 1, 1, 4, 4, 2, 2], dtype=np.uint8)
    endpoints, labels = gdc1.optimal_segments(row, 4)
    assert np.array_equal(gdc1.render_segments(endpoints, labels, row.size), row)


def test_optimal_segments_minimizes_hamming_for_two_runs() -> None:
    row = np.asarray([0, 0, 1, 1, 4, 4, 1, 1, 1], dtype=np.uint8)
    endpoints, labels = gdc1.optimal_segments(row, 2)
    rendered = gdc1.render_segments(endpoints, labels, row.size)
    assert np.count_nonzero(rendered != row) == 2


def test_optimal_segments_matches_small_bruteforce() -> None:
    row = np.asarray([0, 1, 1, 4, 2, 2], dtype=np.uint8)
    endpoints, labels = gdc1.optimal_segments(row, 2)
    achieved = int(np.count_nonzero(gdc1.render_segments(endpoints, labels, row.size) != row))
    brute = row.size
    for endpoint in range(1, row.size):
        for left, right in itertools.product(range(gdc1.CLASSES), repeat=2):
            candidate = np.asarray([left] * endpoint + [right] * (row.size - endpoint), dtype=np.uint8)
            brute = min(brute, int(np.count_nonzero(candidate != row)))
    assert achieved == brute


def test_raw_program_roundtrip() -> None:
    counts = np.asarray([2, 3, 1, 2], dtype=np.uint8)
    starts = np.asarray([0, 2, 4, 1], dtype=np.uint8)
    endpoints = np.asarray([[3, 0], [1, 4], [0, 0], [2, 0]], dtype=np.uint16)
    labels = np.asarray([[1, 255], [3, 0], [255, 255], [4, 255]], dtype=np.uint8)
    payload = gdc1.build_raw_program(
        counts, starts, endpoints, labels, pairs=2, height=2, width=5, maximum_runs=3
    )
    parsed = gdc1.parse_raw_program(payload)
    assert np.array_equal(parsed["counts"], counts)
    assert np.array_equal(parsed["starts"], starts)
    assert np.array_equal(parsed["endpoints"], endpoints)
    assert np.array_equal(parsed["labels"], labels)


def test_raw_program_rejects_noncanonical_inactive_slots() -> None:
    counts = np.asarray([1], dtype=np.uint8)
    starts = np.asarray([0], dtype=np.uint8)
    endpoints = np.asarray([[2]], dtype=np.uint16)
    labels = np.asarray([[255]], dtype=np.uint8)
    payload = gdc1.build_raw_program(
        counts, starts, endpoints, labels, pairs=1, height=1, width=5, maximum_runs=2
    )
    with pytest.raises(gdc1.GDC1Error, match="inactive"):
        gdc1.parse_raw_program(payload)


def test_packet_rejects_corrupted_coded_bytes() -> None:
    counts = np.asarray([1], dtype=np.uint8)
    starts = np.asarray([0], dtype=np.uint8)
    endpoints = np.empty((1, 0), dtype=np.uint16)
    labels = np.empty((1, 0), dtype=np.uint8)
    raw = gdc1.build_raw_program(
        counts, starts, endpoints, labels, pairs=1, height=1, width=5, maximum_runs=1
    )
    coder = "zlib_9"
    coded = gdc1.hg1.et1.compress_payload(raw, coder)
    packet = bytearray(gdc1.build_packet(raw, coder, coded))
    packet[-1] ^= 1
    with pytest.raises(gdc1.GDC1Error, match="coded identity"):
        gdc1.parse_packet(bytes(packet))
