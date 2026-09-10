"""Tests for the ddm_gdc4 run-native endpoint primitives and packet codecs."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tac.gdc4_run_native_endpoint import (
    decode_field_runs,
    decode_field_transitions,
    encode_field_runs,
    encode_field_transitions,
    frame_run_counts,
    optimal_bounded_endpoint_cost,
    plan_field_transitions,
    raster_row,
    row_runs,
)


def _random_field(seed: int, shape=(3, 6, 24), block: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f, h, w = shape
    out = np.zeros(shape, dtype=np.uint8)
    for i in range(f):
        for y in range(h):
            out[i, y] = np.repeat(rng.integers(0, 5, w // block + 1), block)[:w]
    return out


def test_row_runs_basic() -> None:
    row = np.array([1, 1, 2, 2, 2, 0], dtype=np.uint8)
    values, stops = row_runs(row)
    assert values.tolist() == [1, 2, 0]
    assert stops.tolist() == [2, 5, 6]


def test_row_runs_single_value() -> None:
    values, stops = row_runs(np.full(9, 3, dtype=np.uint8))
    assert values.tolist() == [3]
    assert stops.tolist() == [9]


def test_row_runs_rejects_2d() -> None:
    with pytest.raises(ValueError):
        row_runs(np.zeros((2, 2), dtype=np.uint8))


def test_raster_row_roundtrips_row_runs() -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        row = np.repeat(rng.integers(0, 5, 8), 4).astype(np.uint8)
        values, stops = row_runs(row)
        assert np.array_equal(raster_row(values, stops, row.shape[0]), row)


def test_raster_row_rejects_bad_stops() -> None:
    with pytest.raises(ValueError):
        raster_row(np.array([1, 2]), np.array([4, 3]), 3)
    with pytest.raises(ValueError):
        raster_row(np.array([1]), np.array([3]), 4)
    with pytest.raises(ValueError):
        raster_row(np.array([1, 2]), np.array([2]), 4)


def _brute_force_bounded_cost(row: np.ndarray, budget: int) -> int:
    """Exhaustive minimum mismatch over all <= budget piecewise-constant rows."""
    n = int(row.shape[0])
    best = n
    for cuts in range(0, min(budget, n) ):
        for breaks in itertools.combinations(range(1, n), cuts):
            bounds = (0, *breaks, n)
            total = 0
            for a, b in itertools.pairwise(bounds):
                seg = row[a:b]
                total += (b - a) - int(np.bincount(seg, minlength=5).max())
            best = min(best, total)
    return best


def test_optimal_bounded_endpoint_cost_matches_brute_force() -> None:
    rng = np.random.default_rng(11)
    budgets = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    for _ in range(12):
        row = rng.integers(0, 4, 9).astype(np.uint8)
        got = optimal_bounded_endpoint_cost(row, budgets)
        for i, budget in enumerate(budgets.tolist()):
            assert int(got[i]) == _brute_force_bounded_cost(row, budget)


def test_optimal_bounded_endpoint_cost_is_zero_at_full_budget() -> None:
    rng = np.random.default_rng(3)
    row = rng.integers(0, 5, 32).astype(np.uint8)
    n_runs = int(row_runs(row)[0].shape[0])
    got = optimal_bounded_endpoint_cost(row, np.array([n_runs], dtype=np.int64))
    assert int(got[0]) == 0


def test_optimal_bounded_endpoint_cost_is_monotone_non_increasing() -> None:
    rng = np.random.default_rng(5)
    budgets = np.array([1, 2, 3, 4, 6, 8, 12], dtype=np.int64)
    for _ in range(20):
        row = np.repeat(rng.integers(0, 5, 10), 3).astype(np.uint8)
        got = optimal_bounded_endpoint_cost(row, budgets).tolist()
        assert all(a >= b for a, b in itertools.pairwise(got))


def test_frame_run_counts() -> None:
    frame = np.array([[1, 1, 2], [3, 3, 3]], dtype=np.uint8)
    assert frame_run_counts(frame).tolist() == [2, 1]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_v1_codec_roundtrip_is_exact(seed: int) -> None:
    field = _random_field(seed)
    streams = encode_field_runs(field)
    assert np.array_equal(decode_field_runs(streams, field.shape), field)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_v2_codec_roundtrip_is_exact(seed: int) -> None:
    field = _random_field(seed)
    streams = encode_field_transitions(field)
    assert np.array_equal(decode_field_transitions(streams, field.shape), field)


def test_v2_codec_roundtrip_on_pathological_fields() -> None:
    cases = [
        np.zeros((2, 3, 8), dtype=np.uint8),
        np.full((2, 3, 8), 4, dtype=np.uint8),
        np.tile(np.arange(8, dtype=np.uint8) % 5, (2, 3, 1)),
    ]
    alternating = np.zeros((2, 4, 10), dtype=np.uint8)
    alternating[:, :, 1::2] = 1
    cases.append(alternating)
    for field in cases:
        streams = encode_field_transitions(field)
        assert np.array_equal(decode_field_transitions(streams, field.shape), field)


def test_both_codecs_are_deterministic() -> None:
    field = _random_field(9)
    assert encode_field_runs(field) == encode_field_runs(field)
    assert encode_field_transitions(field) == encode_field_transitions(field)


def test_decode_fails_closed_on_stream_residue() -> None:
    field = _random_field(4)
    streams = dict(encode_field_transitions(field))
    streams["dx"] = streams["dx"] + b"\x00"
    with pytest.raises(ValueError):
        decode_field_transitions(streams, field.shape)


def test_decode_fails_closed_on_short_mode_stream() -> None:
    field = _random_field(4)
    streams = dict(encode_field_runs(field))
    streams["mode"] = streams["mode"][:-1]
    with pytest.raises(ValueError):
        decode_field_runs(streams, field.shape)


def test_planner_and_encoder_agree_on_op_counts() -> None:
    field = _random_field(6)
    planned = sum(
        len(script) + 1 for *_x, script, _t, _r, _p in plan_field_transitions(field)
    )
    assert planned == len(encode_field_transitions(field)["op"])


def test_planner_row_count_matches_field() -> None:
    field = _random_field(2, shape=(4, 5, 12))
    assert sum(1 for _ in plan_field_transitions(field)) == 4 * 5
