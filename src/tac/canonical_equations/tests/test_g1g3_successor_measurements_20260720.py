# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.g1g3_successor_measurements_20260720 import (
    ideal_cell_stream_bytes,
    ideal_cell_stream_bytes_from_bit_sum,
    transport_event_fraction,
)


def test_transport_event_fraction_is_exact_count_ratio() -> None:
    assert transport_event_fraction(3, 12) == 0.25
    with pytest.raises(ValueError, match="0 <= event_count"):
        transport_event_fraction(13, 12)
    with pytest.raises(TypeError, match="exact integers"):
        transport_event_fraction(3.0, 12)  # type: ignore[arg-type]


def test_ideal_cell_stream_bytes_sums_self_information() -> None:
    assert ideal_cell_stream_bytes([0.25] * 4) == 1.0
    assert ideal_cell_stream_bytes_from_bit_sum(8.0) == 1.0
    assert math.isclose(ideal_cell_stream_bytes([0.2]), math.log2(5.0) / 8.0)


def test_ideal_cell_stream_refuses_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ideal_cell_stream_bytes([])
    with pytest.raises(ValueError, match=r"\(0,1\]"):
        ideal_cell_stream_bytes([0.0])
    with pytest.raises(ValueError, match="nonnegative"):
        ideal_cell_stream_bytes_from_bit_sum(-1.0)
