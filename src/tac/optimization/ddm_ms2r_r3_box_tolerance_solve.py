# SPDX-License-Identifier: MIT
"""Diagnostics for the fresh MS2R-R3 BOX finite-family solve.

The exact optimizer remains :func:`solve_binary_pair_lattice`.  This module
adds dual diagnostics without pretending that C1 predictor-record bytes have
an RD1 ``stratum x scorer_visibility x G4`` ownership decomposition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
SCHEMA: Final = "ddm_ms2r_r3_binary_dual_diagnostics.v1"


class MS2RR3DiagnosticError(ValueError):
    """A measured row or finite-family selection is malformed."""


def _exact_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise MS2RR3DiagnosticError(f"{field} must be an exact nonnegative integer")
    return value


def build_binary_dual_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    selected_steps: Sequence[int],
) -> dict[str, Any]:
    """Return exact local q8->q4 exchanges and scoped stratum diagnostics.

    Each pair has an exact rate and Seg-error edge because both endpoints were
    measured with the real predictor coder.  Class-stratum slopes are
    deliberately marked nonadditive: the same record-byte delta cannot be
    assigned independently to five class-error deltas.
    """

    if not rows or len(rows) != len(selected_steps):
        raise MS2RR3DiagnosticError("rows and selected_steps must have equal nonzero length")
    pair_rows: list[dict[str, Any]] = []
    aggregate_delta_bytes = 0
    aggregate_corrected_errors = 0
    aggregate_stratum = dict.fromkeys(CLASS_NAMES, 0)
    for expected_pair, (row, selected) in enumerate(zip(rows, selected_steps, strict=True)):
        pair_id = _exact_nonnegative_int(row.get("pair_id"), "pair_id")
        if pair_id != expected_pair:
            raise MS2RR3DiagnosticError("pair rows must preserve ordered 0..n-1 identity")
        if selected not in (4, 8):
            raise MS2RR3DiagnosticError("selected step must be q4 or q8")
        q4_bytes = _exact_nonnegative_int(row.get("q4_record_bytes"), "q4_record_bytes")
        q8_bytes = _exact_nonnegative_int(row.get("q8_record_bytes"), "q8_record_bytes")
        q4_errors = _exact_nonnegative_int(row.get("q4_errors"), "q4_errors")
        q8_errors = _exact_nonnegative_int(row.get("q8_errors"), "q8_errors")
        delta_bytes = q4_bytes - q8_bytes
        corrected_errors = q8_errors - q4_errors
        q4_strata = row.get("q4_stratum_errors")
        q8_strata = row.get("q8_stratum_errors")
        if not isinstance(q4_strata, Mapping) or not isinstance(q8_strata, Mapping):
            raise MS2RR3DiagnosticError("per-class stratum errors are required")
        stratum_corrections: dict[str, int] = {}
        for name in CLASS_NAMES:
            correction = _exact_nonnegative_int(q8_strata.get(name), f"q8_{name}") - _exact_nonnegative_int(
                q4_strata.get(name), f"q4_{name}"
            )
            stratum_corrections[name] = correction
            aggregate_stratum[name] += correction
        pair_rows.append(
            {
                "pair_id": pair_id,
                "selected_step": selected,
                "q8_to_q4_delta_record_bytes": delta_bytes,
                "q8_to_q4_corrected_seg_errors": corrected_errors,
                "lambda_record_bytes_per_corrected_error": (
                    delta_bytes / corrected_errors if corrected_errors > 0 else None
                ),
                "stratum_corrected_errors": stratum_corrections,
                "lambda_status": (
                    "MEASURED_LOCAL_BINARY_EDGE"
                    if corrected_errors > 0
                    else "NULL_NO_POSITIVE_SEG_ERROR_CORRECTION"
                ),
                "actionable_scope": "same-pair q8/q4 finite control only",
                "score_claim": False,
            }
        )
        aggregate_delta_bytes += delta_bytes
        aggregate_corrected_errors += corrected_errors
    return {
        "schema": SCHEMA,
        "pair_count": len(rows),
        "pair_exchange_rows": pair_rows,
        "aggregate_q8_to_q4": {
            "delta_record_bytes": aggregate_delta_bytes,
            "corrected_seg_errors": aggregate_corrected_errors,
            "lambda_record_bytes_per_corrected_error": (
                aggregate_delta_bytes / aggregate_corrected_errors
                if aggregate_corrected_errors > 0
                else None
            ),
            "stratum_corrected_errors": aggregate_stratum,
        },
        "per_stratum_dual_table": [
            {
                "stratum": name,
                "corrected_errors": aggregate_stratum[name],
                "shared_edge_bytes": aggregate_delta_bytes,
                "diagnostic_shared_edge_bytes_per_corrected_error": (
                    aggregate_delta_bytes / aggregate_stratum[name]
                    if aggregate_stratum[name] > 0
                    else None
                ),
                "actionable_for_allocator": False,
                "measurement_status": (
                    "MEASURED_ERRORS_NONADDITIVE_SHARED_RATE_EDGE"
                    if aggregate_stratum[name] > 0
                    else "NULL_NO_POSITIVE_STRATUM_CORRECTION"
                ),
            }
            for name in CLASS_NAMES
        ],
        "verdict_scope": (
            "Exact local dual diagnostics for measured q4/q8 predictor records. "
            "The shared record-byte edge is not an RD1 cellwise byte allocation."
        ),
        "score_claim": False,
    }


def backfill_rd1_cells_null_preserving(
    rd1_rows: Sequence[Mapping[str, Any]],
    *,
    ev1_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Carry all 162 RD1 identities and EV1 homes without inventing duals."""

    if len(rd1_rows) != 162:
        raise MS2RR3DiagnosticError("RD1 source must contain exactly 162 cells")
    ev1_by_key: dict[tuple[Any, Any, Any, Any], Mapping[str, Any]] = {}
    if ev1_rows is not None:
        if len(ev1_rows) != 162:
            raise MS2RR3DiagnosticError("EV1 source must contain exactly 162 cells")
        for row in ev1_rows:
            key = (
                row.get("dual_index"),
                row.get("stratum"),
                row.get("scorer_visibility"),
                row.get("g4_temporal_class"),
            )
            if key in ev1_by_key:
                raise MS2RR3DiagnosticError("EV1 cell identities must be unique")
            ev1_by_key[key] = row
    cells: list[dict[str, Any]] = []
    for row in rd1_rows:
        key = (
            row.get("dual_index"),
            row.get("stratum"),
            row.get("scorer_visibility"),
            row.get("g4_temporal_class"),
        )
        ev1 = ev1_by_key.get(key)
        if ev1_rows is not None and ev1 is None:
            raise MS2RR3DiagnosticError("EV1 and RD1 cell identities differ")
        cells.append(
            {
                "dual_index": row.get("dual_index"),
                "stratum": row.get("stratum"),
                "scorer_visibility": row.get("scorer_visibility"),
                "g4_temporal_class": row.get("g4_temporal_class"),
                "lambda_bytes_per_D_dimension": None,
                "effective_quantum_D": row.get("effective_quantum_D"),
                "ev1_accounting_home": (
                    None
                    if ev1 is None
                    else {
                        "delta_D_dimension": ev1.get("delta_D_dimension"),
                        "delta_counted_bytes_dimension": ev1.get(
                            "delta_counted_bytes_dimension"
                        ),
                        "byte_home_ranges": ev1.get("byte_home_ranges"),
                        "byte_home_epistemic_status": ev1.get(
                            "byte_home_epistemic_status"
                        ),
                        "receiver_changed_channel_values": ev1.get(
                            "receiver_changed_channel_values"
                        ),
                    }
                ),
                "binary_control_lambda_bytes_per_corrected_error": None,
                "measurement_status": (
                    "EV1_ACCOUNTING_HOME_CONSUMED_BUT_STILL_NULL_NO_C1_PAIR_FOREIGN_KEY"
                    if ev1 is not None
                    else "STILL_NULL_NO_C1_PREDICTOR_BYTE_HOME_FOR_RD1_CELL"
                ),
                "actionable_for_train_decision": False,
                "score_claim": False,
            }
        )
    return {
        "schema": "ddm_ms2r_r3_rd1_162_dual_backfill.v1",
        "source_cell_count": 162,
        "measured_cell_count": 0,
        "still_null_cell_count": 162,
        "ev1_accounting_home_cell_count": len(ev1_by_key),
        "ev1_nonzero_accounting_byte_cell_count": sum(
            int(row.get("delta_counted_bytes_dimension", 0) != 0)
            for row in ev1_by_key.values()
        ),
        "ev1_nonzero_distortion_cell_count": sum(
            int(float(row.get("delta_D_dimension", 0.0)) != 0.0)
            for row in ev1_by_key.values()
        ),
        "cells": cells,
        "blocker": (
            "C1 predictor records have no lawful stratum x scorer_visibility x "
            "G4 temporal pair-coordinate ownership map. EV1's exclusive "
            "accounting homes are consumed, but its v19_pair_join explicitly "
            "has per_pair_byte_allocation=null; J8F likewise exposes counted "
            "applications without composable C1 coordinate foreign keys."
        ),
        "score_claim": False,
    }


__all__ = [
    "CLASS_NAMES",
    "SCHEMA",
    "MS2RR3DiagnosticError",
    "backfill_rd1_cells_null_preserving",
    "build_binary_dual_diagnostics",
]
