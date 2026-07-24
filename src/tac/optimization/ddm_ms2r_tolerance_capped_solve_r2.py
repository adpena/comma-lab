# SPDX-License-Identifier: MIT
"""Exact finite-family solver for the DDM MS2R box-tolerance control.

This module deliberately separates two claims:

* the binary q4/q8 quotient-plane family is an exact finite control family;
* it is not the full per-dimension Fisher/G4 waterfill promised by MS2R.

The dynamic program is exact for measured per-pair errors and predictor-record
bytes.  It does not impute the missing stratum x visibility x G4 byte homes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

SCHEMA: Final = "ddm_ms2r_binary_quotient_control_solve.v1"
ALLOWED_STEPS: Final = (4, 8)


class MS2RToleranceSolveError(ValueError):
    """A finite-family row or exact dynamic-program invariant differs."""


def quantize_uint8_half_up(value: np.ndarray, step: int) -> np.ndarray:
    """Quantize uint8 values to a dyadic lattice with nonnegative ties up."""

    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise MS2RToleranceSolveError("quotient plane must be exact uint8")
    if type(step) is not int or step not in ALLOWED_STEPS:
        raise MS2RToleranceSolveError("quantum must be one of the sealed q4/q8 controls")
    widened = array.astype(np.uint16)
    quantized = np.minimum(((widened + step // 2) // step) * step, 255)
    return np.ascontiguousarray(quantized.astype(np.uint8))


def _exact_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise MS2RToleranceSolveError(f"{field} must be an exact nonnegative integer")
    return value


def solve_binary_pair_lattice(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_errors: int,
) -> dict[str, Any]:
    """Minimize additive record bytes over exact q4/q8 per-pair choices.

    The state is the realized Seg error total.  Prefix errors can only grow,
    so states above the global cap may be discarded without approximation.
    Ties resolve toward q4, then toward fewer realized errors at the terminal
    state.  The returned selection is independently recomputed before return.
    """

    cap = _exact_nonnegative_int(allowed_errors, "allowed_errors")
    if not rows:
        raise MS2RToleranceSolveError("binary lattice requires at least one pair")
    normalized: list[tuple[int, int, int, int, int]] = []
    expected_pair_ids = list(range(len(rows)))
    observed_pair_ids: list[int] = []
    for row in rows:
        pair_id = _exact_nonnegative_int(row.get("pair_id"), "pair_id")
        observed_pair_ids.append(pair_id)
        normalized.append(
            (
                pair_id,
                _exact_nonnegative_int(row.get("q4_errors"), "q4_errors"),
                _exact_nonnegative_int(row.get("q8_errors"), "q8_errors"),
                _exact_nonnegative_int(row.get("q4_record_bytes"), "q4_record_bytes"),
                _exact_nonnegative_int(row.get("q8_record_bytes"), "q8_record_bytes"),
            )
        )
    if observed_pair_ids != expected_pair_ids:
        raise MS2RToleranceSolveError("pair rows must be the ordered exact 0..n-1 identity")

    infinity = np.iinfo(np.int64).max // 4
    current = np.full(cap + 1, infinity, dtype=np.int64)
    current[0] = 0
    choices = np.full((len(normalized), cap + 1), 255, dtype=np.uint8)
    for row_index, (_pair_id, q4_errors, q8_errors, q4_bytes, q8_bytes) in enumerate(normalized):
        next_cost = np.full(cap + 1, infinity, dtype=np.int64)
        next_choice = choices[row_index]
        for choice, errors, record_bytes in (
            (4, q4_errors, q4_bytes),
            (8, q8_errors, q8_bytes),
        ):
            if errors > cap:
                continue
            source = current[: cap + 1 - errors]
            candidate = source + record_bytes
            target = next_cost[errors:]
            improves = candidate < target
            target[improves] = candidate[improves]
            next_choice[errors:][improves] = choice
        current = next_cost

    feasible = np.flatnonzero(current < infinity)
    if feasible.size == 0:
        raise MS2RToleranceSolveError("q4/q8 family has no member inside the error cap")
    minimum_bytes = int(current[feasible].min())
    terminal_errors = int(feasible[current[feasible] == minimum_bytes].min())
    selected = [0] * len(normalized)
    cursor = terminal_errors
    for row_index in range(len(normalized) - 1, -1, -1):
        choice = int(choices[row_index, cursor])
        if choice not in ALLOWED_STEPS:
            raise MS2RToleranceSolveError("dynamic-program predecessor chain is incomplete")
        selected[row_index] = choice
        errors = normalized[row_index][1 if choice == 4 else 2]
        cursor -= errors
        if cursor < 0:
            raise MS2RToleranceSolveError("dynamic-program predecessor underflow")
    if cursor != 0:
        raise MS2RToleranceSolveError("dynamic-program predecessor did not reach the origin")

    recomputed_errors = sum(row[1] if choice == 4 else row[2] for row, choice in zip(normalized, selected, strict=True))
    recomputed_bytes = sum(row[3] if choice == 4 else row[4] for row, choice in zip(normalized, selected, strict=True))
    if recomputed_errors != terminal_errors or recomputed_bytes != minimum_bytes or recomputed_errors > cap:
        raise MS2RToleranceSolveError("dynamic-program result failed independent recomputation")
    return {
        "schema": SCHEMA,
        "family": "C1_SCORER_QUOTIENT_BINARY_Q4_Q8_CONTROL",
        "family_scope": (
            "[naive-uniform-quantum upper bound] exact over the finite q4/q8 "
            "per-pair family; not the full Fisher/G4 per-dimension waterfill"
        ),
        "allowed_errors": cap,
        "realized_errors": recomputed_errors,
        "additive_predictor_record_bytes": recomputed_bytes,
        "selected_steps": selected,
        "q4_pair_count": selected.count(4),
        "q8_pair_count": selected.count(8),
        "exact_dynamic_program": True,
        "score_claim": False,
    }


__all__ = [
    "ALLOWED_STEPS",
    "SCHEMA",
    "MS2RToleranceSolveError",
    "quantize_uint8_half_up",
    "solve_binary_pair_lattice",
]
