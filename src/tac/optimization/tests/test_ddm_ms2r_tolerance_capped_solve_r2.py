# SPDX-License-Identifier: MIT

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tac.optimization.ddm_ms2r_tolerance_capped_solve_r2 import (
    MS2RToleranceSolveError,
    quantize_uint8_half_up,
    solve_binary_pair_lattice,
)


def test_quantize_uint8_half_up_clips_terminal_level() -> None:
    value = np.asarray([0, 1, 2, 3, 4, 252, 253, 254, 255], dtype=np.uint8)
    assert quantize_uint8_half_up(value, 4).tolist() == [
        0,
        0,
        4,
        4,
        4,
        252,
        252,
        255,
        255,
    ]


def test_binary_solve_matches_exhaustive_minimum() -> None:
    rows = [
        {"pair_id": 0, "q4_errors": 1, "q8_errors": 8, "q4_record_bytes": 9, "q8_record_bytes": 1},
        {"pair_id": 1, "q4_errors": 2, "q8_errors": 7, "q4_record_bytes": 7, "q8_record_bytes": 2},
        {"pair_id": 2, "q4_errors": 1, "q8_errors": 5, "q4_record_bytes": 8, "q8_record_bytes": 3},
    ]
    result = solve_binary_pair_lattice(rows, allowed_errors=10)
    exhaustive = []
    for choices in itertools.product((4, 8), repeat=3):
        errors = sum(row[f"q{choice}_errors"] for row, choice in zip(rows, choices, strict=True))
        size = sum(row[f"q{choice}_record_bytes"] for row, choice in zip(rows, choices, strict=True))
        if errors <= 10:
            exhaustive.append((size, errors, list(choices)))
    assert (
        result["additive_predictor_record_bytes"],
        result["realized_errors"],
        result["selected_steps"],
    ) == min(exhaustive)


def test_binary_solve_refuses_missing_pair_identity() -> None:
    with pytest.raises(MS2RToleranceSolveError, match=r"0\.\.n-1"):
        solve_binary_pair_lattice(
            [{"pair_id": 1, "q4_errors": 0, "q8_errors": 1, "q4_record_bytes": 2, "q8_record_bytes": 1}],
            allowed_errors=1,
        )
