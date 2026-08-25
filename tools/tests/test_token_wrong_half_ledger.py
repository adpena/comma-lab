from __future__ import annotations

import numpy as np

from tools.token_wrong_half_ledger import (
    MARGIN_EDGES,
    TOTAL_FREQUENCY,
    binary_entropy_bits,
    integer_decomposition,
)


def test_integer_decomposition_closes_for_correct_and_wrong_symbols() -> None:
    coding = np.asarray(
        [[0.70, 0.10, 0.08, 0.07, 0.05], [0.55, 0.25, 0.10, 0.06, 0.04]],
        dtype=np.float64,
    )
    symbols = np.asarray([0, 1], dtype=np.int64)
    split = integer_decomposition(coding, symbols)
    values = np.ascontiguousarray(coding, dtype=np.float32)
    frequency = (values.astype(np.float64) * TOTAL_FREQUENCY).astype(np.uint64)
    np.maximum(frequency, 1, out=frequency)
    winners = values.argmax(axis=1)
    row = np.arange(2)
    frequency[row, winners] = (
        frequency[row, winners].astype(np.int64)
        + TOTAL_FREQUENCY
        - frequency.sum(axis=1, dtype=np.uint64).astype(np.int64)
    ).astype(np.uint64)
    expected = 31.0 - np.log2(frequency[row, symbols].astype(np.float64))
    np.testing.assert_allclose(split["indicator"] + split["which"], expected, rtol=0, atol=2e-12)
    assert split["which"][0] == 0.0
    assert split["which"][1] > 0.0


def test_margin_bucket_is_defined_by_exact_balanced_integer_frequencies() -> None:
    coding = np.asarray([[0.5, 0.25, 0.125, 0.075, 0.05]], dtype=np.float64)
    split = integer_decomposition(coding, np.asarray([0]))
    assert 0 <= int(split["margin_bucket"][0]) < len(MARGIN_EDGES) - 1
    assert np.isclose(split["margin"][0], 1.0, atol=1e-6)


def test_binary_entropy_bound_handles_pure_and_balanced_cells() -> None:
    result = binary_entropy_bits(np.asarray([0, 5, 10]), np.asarray([10, 10, 10]))
    np.testing.assert_allclose(result, np.asarray([0.0, 10.0, 0.0]))
