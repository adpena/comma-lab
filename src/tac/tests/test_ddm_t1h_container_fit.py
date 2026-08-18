# SPDX-License-Identifier: MIT
"""Tests for the t1h packed-CAP1 container-fit repair.

The packed CAP1 carrier section is dispatched by the receiver on an EXACT length, so the
Rice residual payload inside it has a fixed byte width and the composition must fit a hard
bit budget.  ``fit_to_bit_budget`` trades the least MEASURED pose gain per bit saved until it
fits, substituting among moves the sweep already evaluated exactly.

The repair takes its encoder by injection, so these tests exercise the selection logic
directly with a synthetic bit model -- no archive, no scorer, no brotli.

Lane: ddm_t1h_pose_coeff_resolve_headroom_20260817.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "experiments" / "ddm_t1h_compose_pass1.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("ddm_t1h_compose_under_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(per_pair: list[list[tuple[int, int, float]]]) -> list[dict]:
    """Build sweep-shaped rows: per pair, a list of (coord, best_delta, energy)."""
    return [
        {
            "pair": pair,
            "per_coord": [
                {"coord": coord, "best_delta": delta, "energy": energy}
                for coord, delta, energy in options
            ],
        }
        for pair, options in enumerate(per_pair)
    ]


def _magnitude_encoder(base_bits: int):
    """A monotone stand-in for Rice: every unit of code magnitude costs one bit."""

    def encode_bits(codes: np.ndarray):
        return base_bits + int(np.abs(codes).sum()), b"", np.zeros(1, dtype=np.int64)

    return encode_bits


def test_no_repair_when_composition_already_fits() -> None:
    tool = _load_tool()
    candidate = np.array([[1, 0], [0, 0]], dtype=np.int32)
    choice = [(0, 1, 0.5), (-1, 0, 2.0)]
    rows = _rows([[(0, 1, 0.5)], [(0, 1, 2.0)]])
    base = np.array([1.0, 2.0])

    out, out_choice, bits, log = tool.fit_to_bit_budget(
        candidate, list(choice), rows, base, _magnitude_encoder(0), max_bits=100
    )

    assert log == []
    assert bits == 1
    assert out_choice == choice
    assert np.array_equal(out, candidate)


def test_repair_picks_the_cheapest_gain_per_bit_substitution() -> None:
    """Both pairs can shed a bit by reverting; the one that costs less pose must be chosen."""
    tool = _load_tool()
    # Pair 0 reverting costs 0.1 energy; pair 1 reverting costs 5.0.  Each saves one bit.
    candidate = np.array([[1, 0], [1, 0]], dtype=np.int32)
    choice = [(0, 1, 1.0), (0, 1, 1.0)]
    rows = _rows([[(0, 1, 1.0)], [(0, 1, 1.0)]])
    base = np.array([1.1, 6.0])

    out, out_choice, bits, log = tool.fit_to_bit_budget(
        candidate, list(choice), rows, base, _magnitude_encoder(0), max_bits=1
    )

    assert bits == 1
    assert len(log) == 1
    entry = log[0]
    assert entry["pair"] == 0
    assert entry["substituted_to_coord"] == -1
    assert entry["bits_saved"] == 1
    assert entry["energy_cost"] == pytest.approx(0.1)
    # Pair 0 reverted to base; pair 1 kept its move.
    assert out_choice[0] == (-1, 0, 1.1)
    assert out_choice[1] == (0, 1, 1.0)
    assert np.array_equal(out, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_repair_prefers_a_cheaper_coordinate_over_dropping_the_move() -> None:
    """Substituting onto another measured coordinate must beat reverting when it costs less."""
    tool = _load_tool()
    # Coordinate 1's move has magnitude 1 (saves nothing) -- only reverting or a smaller
    # magnitude sheds bits, so give coordinate 1 a delta of 0 magnitude via a negative move
    # that lands on zero.
    candidate = np.array([[2, 0]], dtype=np.int32)
    choice = [(0, 2, 1.0)]
    # Option on coord 0 with delta 1 lands the code at 1 (one bit cheaper) at energy 1.2;
    # reverting entirely costs energy 9.0.
    rows = _rows([[(0, 1, 1.2)]])
    base = np.array([9.0])

    out, out_choice, bits, log = tool.fit_to_bit_budget(
        candidate, list(choice), rows, base, _magnitude_encoder(0), max_bits=1
    )

    assert bits == 1
    assert len(log) == 1
    assert log[0]["substituted_to_coord"] == 0
    assert log[0]["substituted_to_delta"] == 1
    assert log[0]["energy_after"] == pytest.approx(1.2)
    assert out_choice[0] == (0, 1, 1.2)
    assert np.array_equal(out, np.array([[1, 0]], dtype=np.int32))


def test_repair_refuses_when_no_substitution_sheds_a_bit() -> None:
    """A container that cannot be fitted must fail closed, never silently mis-price."""
    tool = _load_tool()
    candidate = np.array([[1, 0]], dtype=np.int32)
    choice = [(0, 1, 1.0)]
    rows = _rows([[(0, 1, 1.0)]])
    base = np.array([2.0])

    def constant_encoder(codes: np.ndarray):
        return 500, b"", np.zeros(1, dtype=np.int64)

    with pytest.raises(SystemExit, match="CONTAINER INFEASIBLE"):
        tool.fit_to_bit_budget(
            candidate, list(choice), rows, base, constant_encoder, max_bits=100
        )


def test_repair_keeps_moves_inside_signed_int12() -> None:
    """A substitution that would leave signed-int12 must never be selected, even if cheap.

    The encoder here charges per NONZERO cell rather than per unit of magnitude, so the
    out-of-range substitution saves exactly as many bits as the legal revert while costing
    LESS pose energy.  Without the bounds guard the repair would prefer it and emit a lattice
    holding 2049 -- outside the carrier's signed-int12 alphabet.  With the guard it must fall
    back to the legal revert.
    """
    tool = _load_tool()

    def nonzero_encoder(codes: np.ndarray):
        return 100 * int(np.count_nonzero(codes)), b"", np.zeros(1, dtype=np.int64)

    candidate = np.array([[2047, 3]], dtype=np.int32)
    choice = [(1, 3, 1.0)]
    # Coordinate 0's option would land the code at 2047 + 2 = 2049 and looks cheapest on
    # energy (0.1 against the revert's 9.0); it is illegal and must be rejected.
    rows = _rows([[(0, 2, 0.1), (1, 3, 1.0)]])
    base = np.array([9.0])

    out, out_choice, bits, log = tool.fit_to_bit_budget(
        candidate, list(choice), rows, base, nonzero_encoder, max_bits=100
    )

    assert bits == 100
    assert len(log) == 1
    assert log[0]["substituted_to_coord"] == -1
    assert out_choice[0] == (-1, 0, 9.0)
    assert np.array_equal(out, np.array([[2047, 0]], dtype=np.int32))
    assert int(np.abs(out).max()) <= 2047
