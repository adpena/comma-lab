"""Behavioral tests for the bounded population-global V2 exact recode."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.codec.v10_population_predictor_residual_v2 import (
    PAIR_PREFIX,
    PREFIX,
    PopulationPair,
    PopulationPredictorResidualV2Error,
    decode_population_predictor_residual_v2,
    encode_population_predictor_residual_v2,
    verify_population_identity,
)


def _rows(count: int = 5) -> tuple[PopulationPair, ...]:
    base = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    return tuple(
        PopulationPair(
            pair_id=pair_id,
            frame0=np.ascontiguousarray(base + pair_id * 7),
            frame1=np.ascontiguousarray(np.roll(base, pair_id + 1, axis=1) + pair_id * 11),
        )
        for pair_id in range(count)
    )


def _encode(path: Path, rows: tuple[PopulationPair, ...], reset_interval: int = 3):
    return encode_population_predictor_residual_v2(
        rows,
        output_path=path,
        pair_count=len(rows),
        height=4,
        width=6,
        reset_interval=reset_interval,
    )


def test_v2_roundtrip_is_exact_streaming_and_byte_deterministic(tmp_path: Path) -> None:
    rows = _rows()
    first = _encode((tmp_path / "first.bin").resolve(), rows)
    second = _encode((tmp_path / "second.bin").resolve(), rows)
    assert (tmp_path / "first.bin").read_bytes() == (tmp_path / "second.bin").read_bytes()
    assert first.payload_sha256 == second.payload_sha256
    assert first.reset_count == 2
    assert first.peak_retained_pairs == 2

    observed: list[tuple[int, np.ndarray, np.ndarray]] = []
    decoded = decode_population_predictor_residual_v2(
        (tmp_path / "first.bin").resolve(),
        on_pair=lambda pair_id, frame0, frame1: observed.append((pair_id, frame0.copy(), frame1.copy())),
    )
    assert decoded.payload_sha256 == first.payload_sha256
    assert len(observed) == len(rows)
    for expected, (pair_id, frame0, frame1) in zip(rows, observed, strict=True):
        assert pair_id == expected.pair_id
        np.testing.assert_array_equal(frame0, expected.frame0)
        np.testing.assert_array_equal(frame1, expected.frame1)


def test_v2_identity_callback_requires_complete_exact_population(tmp_path: Path) -> None:
    rows = _rows(3)
    path = (tmp_path / "identity.bin").resolve()
    _encode(path, rows, reset_interval=2)
    hashes = {
        (row.pair_id, slot): hashlib.sha256(frame.tobytes(order="C")).hexdigest()
        for row in rows
        for slot, frame in enumerate((row.frame0, row.frame1))
    }
    assert (
        verify_population_identity(
            path,
            expected_frame_sha256=hashes,
        ).pair_count
        == 3
    )
    hashes.pop((2, 1))
    with pytest.raises(PopulationPredictorResidualV2Error, match="identity callback"):
        verify_population_identity(path, expected_frame_sha256=hashes)


def test_v2_refuses_component_corruption_trailing_and_immutable_overwrite(
    tmp_path: Path,
) -> None:
    rows = _rows(2)
    path = (tmp_path / "payload.bin").resolve()
    _encode(path, rows, reset_interval=2)
    with pytest.raises(PopulationPredictorResidualV2Error, match="already exists"):
        _encode(path, rows, reset_interval=2)

    corrupted = bytearray(path.read_bytes())
    corrupted[PREFIX.size + PAIR_PREFIX.size] ^= 1
    corrupt_path = (tmp_path / "corrupt.bin").resolve()
    corrupt_path.write_bytes(corrupted)
    with pytest.raises(PopulationPredictorResidualV2Error, match="component hash"):
        decode_population_predictor_residual_v2(corrupt_path, on_pair=lambda *_: None)

    trailing_path = (tmp_path / "trailing.bin").resolve()
    trailing_path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(PopulationPredictorResidualV2Error, match="trailing"):
        decode_population_predictor_residual_v2(trailing_path, on_pair=lambda *_: None)

    oversized = bytearray(path.read_bytes())
    pair_header = list(PAIR_PREFIX.unpack_from(oversized, PREFIX.size))
    pair_header[2] = len(oversized) + 1
    oversized[PREFIX.size : PREFIX.size + PAIR_PREFIX.size] = PAIR_PREFIX.pack(*pair_header)
    oversized_path = (tmp_path / "oversized.bin").resolve()
    oversized_path.write_bytes(oversized)
    with pytest.raises(PopulationPredictorResidualV2Error, match="bounded payload"):
        decode_population_predictor_residual_v2(oversized_path, on_pair=lambda *_: None)
