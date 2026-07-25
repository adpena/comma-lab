from __future__ import annotations

import pytest

from tac.optimization.ddm_ms2r_r3_box_tolerance_solve import (
    MS2RR3DiagnosticError,
    backfill_rd1_cells_null_preserving,
    build_binary_dual_diagnostics,
)
from tools.run_ddm_ms2r_r3_box_tolerance_solve import CONFIG_PATH, _load_config


def _row(pair_id: int, q4_errors: int, q8_errors: int) -> dict[str, object]:
    q4 = {"Road": q4_errors, "Lane": 0, "Undrivable": 0, "Movable": 0, "MyCar": 0}
    q8 = {"Road": q8_errors, "Lane": 0, "Undrivable": 0, "Movable": 0, "MyCar": 0}
    return {
        "pair_id": pair_id,
        "q4_errors": q4_errors,
        "q8_errors": q8_errors,
        "q4_record_bytes": 11,
        "q8_record_bytes": 7,
        "q4_stratum_errors": q4,
        "q8_stratum_errors": q8,
    }


def test_binary_duals_keep_shared_stratum_rate_nonactionable() -> None:
    result = build_binary_dual_diagnostics([_row(0, 2, 4), _row(1, 3, 6)], [4, 8])
    assert result["aggregate_q8_to_q4"]["delta_record_bytes"] == 8
    assert result["aggregate_q8_to_q4"]["corrected_seg_errors"] == 5
    assert result["pair_exchange_rows"][0]["lambda_record_bytes_per_corrected_error"] == 2
    assert all(not row["actionable_for_allocator"] for row in result["per_stratum_dual_table"])


def test_binary_duals_reject_unordered_pair_identity() -> None:
    with pytest.raises(MS2RR3DiagnosticError, match="ordered"):
        build_binary_dual_diagnostics([_row(1, 2, 4)], [4])


def test_rd1_backfill_preserves_all_nulls() -> None:
    source = [
        {
            "dual_index": index,
            "stratum": "Road",
            "scorer_visibility": "visible",
            "g4_temporal_class": "static",
            "effective_quantum_D": None,
        }
        for index in range(162)
    ]
    ev1 = [
        {
            **row,
            "delta_D_dimension": -0.25 if row["dual_index"] == 0 else 0.0,
            "delta_counted_bytes_dimension": 3 if row["dual_index"] == 0 else 0,
            "byte_home_ranges": [],
            "byte_home_epistemic_status": "DERIVED",
            "receiver_changed_channel_values": 0,
        }
        for row in source
    ]
    result = backfill_rd1_cells_null_preserving(source, ev1_rows=ev1)
    assert result["source_cell_count"] == 162
    assert result["measured_cell_count"] == 0
    assert result["still_null_cell_count"] == 162
    assert result["ev1_accounting_home_cell_count"] == 162
    assert result["ev1_beneficial_accounting_slope_count"] == 1
    assert result["cells"][0][
        "observed_accounting_slope_full_bytes_per_D_improvement"
    ] == 12
    assert all(row["lambda_bytes_per_D_dimension"] is None for row in result["cells"])


def test_rd1_backfill_requires_complete_cube() -> None:
    with pytest.raises(MS2RR3DiagnosticError, match="162"):
        backfill_rd1_cells_null_preserving([])


def test_typed_config_separates_rounded_charter_from_exact_custody() -> None:
    config, _sha256 = _load_config(CONFIG_PATH)
    assert config.charter_rounded_q1_errors == 17_931
    assert config.expected_measured_q1_errors == 17_927
