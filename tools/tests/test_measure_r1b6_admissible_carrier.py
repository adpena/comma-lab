from __future__ import annotations

from functools import reduce
from math import gcd

import numpy as np

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tools.measure_r1b6_admissible_carrier import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    _signed_rounding_block,
    _source_closest_block,
    breakeven_bytes,
)


def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )


def test_signed_singleton_endpoints_are_exact_and_keep_the_rounded_cell() -> None:
    operator = _operator()
    row, col = 123, 234
    rounded = np.asarray([64, 128, 192], dtype=np.uint8)
    negative, negative_numerators = _signed_rounding_block(
        operator, rounded, row, col, -1
    )
    positive, positive_numerators = _signed_rounding_block(
        operator, rounded, row, col, 1
    )

    row_support = operator.row_supports[row]
    col_support = operator.col_supports[col]
    coefficients = np.outer(row_support.numerators, col_support.numerators).astype(
        np.int64
    ).reshape(-1)
    denominator = int(row_support.denominator) * int(col_support.denominator)
    common_gcd = reduce(gcd, (int(value) for value in coefficients))
    expected_separation = 2 * ((denominator - 1) // 2 // common_gcd) * common_gcd

    assert negative.dtype == positive.dtype == np.uint8
    assert np.all(positive_numerators - negative_numerators == expected_separation)
    for channel in range(3):
        for block, numerator in (
            (negative, negative_numerators[channel]),
            (positive, positive_numerators[channel]),
        ):
            assert int(
                np.dot(coefficients, block[:, :, channel].reshape(-1).astype(np.int64))
            ) == int(numerator)
            assert (int(numerator) + denominator // 2) // denominator == int(
                rounded[channel]
            )


def test_source_closest_singleton_sign_is_measured_not_hardcoded() -> None:
    operator = _operator()
    row, col = 123, 234
    rounded = np.asarray([64, 128, 192], dtype=np.uint8)
    negative, _ = _signed_rounding_block(operator, rounded, row, col, -1)
    positive, _ = _signed_rounding_block(operator, rounded, row, col, 1)
    row_support = operator.row_supports[row]
    col_support = operator.col_supports[col]

    source_negative = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    source_negative[
        np.ix_(row_support.indices, col_support.indices, range(3))
    ] = negative
    sign_negative, selected_negative, _, distance_negative = _source_closest_block(
        operator, rounded, source_negative, row, col
    )
    assert sign_negative == -1
    assert distance_negative == 0
    assert np.array_equal(selected_negative, negative)

    source_positive = source_negative.copy()
    source_positive[
        np.ix_(row_support.indices, col_support.indices, range(3))
    ] = positive
    sign_positive, selected_positive, _, distance_positive = _source_closest_block(
        operator, rounded, source_positive, row, col
    )
    assert sign_positive == 1
    assert distance_positive == 0
    assert np.array_equal(selected_positive, positive)


def test_break_even_uses_the_canonical_realized_recovery_law() -> None:
    assert breakeven_bytes(0.0) == 0.0
    assert breakeven_bytes(0.0012332316583976016) == 1852.0914265927563
