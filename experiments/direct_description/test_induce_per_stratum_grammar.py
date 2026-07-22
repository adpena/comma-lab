from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from experiments.direct_description.induce_per_stratum_grammar import (
    GrammarError,
    PolygonTrackCorpus,
    combined_projection,
    decode_arc_streams,
    decode_envelope,
    decode_lane_streams,
    decode_polygon_tracks,
    decode_row_runs,
    decode_signed_values,
    encode_arc_streams,
    encode_envelope,
    encode_polygon_tracks,
    encode_row_runs,
    encode_signed_values,
    lane_streams,
    stored_npy_memmap,
)


def test_signed_varints_round_trip_and_reject_trailing_bytes() -> None:
    values = np.asarray([0, -1, 1, -127, 127, -65_535, 65_535], dtype=np.int64)
    payload = encode_signed_values(values)
    np.testing.assert_array_equal(decode_signed_values(payload, len(values)), values)
    with pytest.raises(GrammarError, match="trailing"):
        decode_signed_values(payload + b"\x00", len(values))


def test_actual_codec_envelope_counts_and_round_trips() -> None:
    streams = [
        ("EVENT", bytes(range(64)) * 16),
        ("SHAPE", b"persistent-shape-production" * 100),
    ]
    payload, rows = encode_envelope(streams)
    assert decode_envelope(payload) == streams
    assert len(payload) == 5 + sum(row["counted_bytes"] for row in rows)
    assert all(row["winner"] in {"brotli_q11", "lzma1_raw_1m", "zlib9"} for row in rows)


def test_row_run_lossless_parse_back() -> None:
    mask = np.zeros((3, 7, 13), dtype=bool)
    mask[0, 1, 2:5] = True
    mask[1, 3, [0, 5, 6, 12]] = True
    mask[2, :, 4:11] = True
    np.testing.assert_array_equal(decode_row_runs(encode_row_runs(mask)), mask)


def test_stored_npy_member_memmap(tmp_path: Path) -> None:
    source = np.arange(60, dtype=np.int64).reshape(3, 4, 5)
    path = tmp_path / "stored.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        npy = tmp_path / "lstars.npy"
        np.save(npy, source)
        archive.write(npy, "lstars.npy")
    mapped = stored_npy_memmap(path, "lstars")
    np.testing.assert_array_equal(mapped, source)


@pytest.mark.parametrize("morph_delta", [False, True])
def test_movable_polygon_semantic_parse_back(morph_delta: bool) -> None:
    polygon = np.asarray([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32)
    shifted = polygon + np.asarray([3, 2], dtype=np.int32)
    corpus = PolygonTrackCorpus(
        presence=np.asarray([[True], [True], [False]], dtype=bool),
        polygons=[{0: polygon}, {0: shifted}, {}],
        births=1,
        persists=1,
        deaths=1,
        max_slots=1,
    )
    streams, rendered, metadata = encode_polygon_tracks(corpus, morph_delta=morph_delta)
    np.testing.assert_array_equal(decode_polygon_tracks(streams), rendered)
    assert metadata["morph_delta_events"] == int(morph_delta)


def test_boundary_arc_semantic_parse_back() -> None:
    mask = np.zeros((2, 384, 512), dtype=bool)
    mask[0, 10, 10:100] = True
    mask[1, 20:80, 40] = True
    streams, rendered, _vertices = encode_arc_streams(mask, 1.0)
    np.testing.assert_array_equal(decode_arc_streams(streams), rendered)


@pytest.mark.parametrize("persist_dash", [False, True])
def test_lane_production_streams_decode_event_and_values(persist_dash: bool) -> None:
    presence = np.asarray([[True], [False], [True]], dtype=bool)
    steps = np.ones(11, dtype=np.float64)
    matrix = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],
        dtype=np.float64,
    )
    streams, quantized = lane_streams(matrix, presence, steps, persist_dash=persist_dash)
    lines, decoded_presence = decode_lane_streams(streams, steps)
    np.testing.assert_array_equal(decoded_presence, presence)
    assert [len(pair) for pair in lines] == [1, 0, 1]
    assert quantized.shape == matrix.shape


def test_projection_is_derived_upper_bound_not_score() -> None:
    lane = [{"candidate": "lane", "exact": False, "counted_bytes": 35_000, "fidelity": {"errors": 400_000}}]
    movable = [{"candidate": "movable", "exact": False, "counted_bytes": 20_000, "fidelity": {"errors": 100_000}}]
    projection = combined_projection(lane, movable)
    assert projection["score_claim"] is False
    assert projection["receiver_closed"] is False
    assert projection["best_under_budget"]["counted_bytes"] == 55_000
    assert projection["joint_gate_passed"] is True
