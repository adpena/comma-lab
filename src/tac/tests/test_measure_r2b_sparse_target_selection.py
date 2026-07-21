from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "measure_r2b_sparse_target_selection",
    ROOT / "tools" / "measure_r2b_sparse_target_selection.py",
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_sparse_stream_roundtrip_is_canonical_and_strict() -> None:
    indices = [0, 1, 130, 999_999]
    signs = [-1, 1, -1, 1]
    payload = mod.encode_stream(indices, signs)
    assert mod.decode_stream(payload) == (indices, signs)
    assert mod.encode_stream(*mod.decode_stream(payload)) == payload


def test_canonical_archive_is_reproducible_and_charged() -> None:
    stream = mod.encode_stream([4, 9], [1, -1])
    first = mod.canonical_archive(stream)
    second = mod.canonical_archive(stream)
    assert first == second
    assert len(first) > len(stream)


def test_signed_rounding_block_is_exact_and_keeps_rounded_byte() -> None:
    op = mod.DisjointResizeOperator.build(camera_h=6, camera_w=8, scorer_h=3, scorer_w=4)
    rounded = np.array([23, 127, 231], dtype=np.uint8)
    for sign in (-1, 1):
        block, targets = mod.signed_rounding_block(op, rounded, 1, 2, sign)
        rs, cs = op.row_supports[1], op.col_supports[2]
        coefficients = np.outer(rs.numerators, cs.numerators).astype(np.int64).reshape(-1)
        denominator = int(rs.denominator) * int(cs.denominator)
        assert block.dtype == np.uint8
        for channel in range(3):
            assert int(np.dot(coefficients, block[:, :, channel].reshape(-1).astype(np.int64))) == int(targets[channel])
            assert (int(targets[channel]) + denominator // 2) // denominator == int(rounded[channel])


def test_source_distance_uses_wide_integer_arithmetic() -> None:
    # Regression: int16 squaring overflows for |delta| > 181 and can reverse
    # the selected sign.  The production helper must agree with int64 SSE.
    op = mod.DisjointResizeOperator.build(camera_h=6, camera_w=8, scorer_h=3, scorer_w=4)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    rounded = np.array([127, 127, 127], dtype=np.uint8)
    sign, block, _targets = mod.choose_source_closest_sign(op, rounded, frame, 1, 2)
    alternatives = {
        candidate_sign: mod.signed_rounding_block(op, rounded, 1, 2, candidate_sign)[0]
        for candidate_sign in (-1, 1)
    }
    expected = min(
        alternatives,
        key=lambda candidate_sign: (
            int(np.sum(alternatives[candidate_sign].astype(np.int64) ** 2)),
            candidate_sign,
        ),
    )
    assert sign == expected
    assert np.array_equal(block, alternatives[expected])


def test_curve_kkt_stop_uses_exact_archive_bytes() -> None:
    indices = list(range(200))
    signs = [1 if i % 2 else -1 for i in indices]
    curve, stop = mod.build_curve(indices, signs)
    assert curve[0]["decisions"] == 0
    assert curve[-1]["decisions"] == 200
    assert 0 <= stop <= 200
    assert curve[0]["charged_archive_bytes"] == 0
    assert all(row["charged_archive_bytes"] > 0 for row in curve[1:])
