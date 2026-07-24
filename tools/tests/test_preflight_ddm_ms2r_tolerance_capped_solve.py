from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TOOL = Path("tools/preflight_ddm_ms2r_tolerance_capped_solve.py")
SPEC = importlib.util.spec_from_file_location(
    "preflight_ddm_ms2r_tolerance_capped_solve",
    TOOL,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_partial_bundle_emits_no_rungs_and_162_null_cells(tmp_path: Path) -> None:
    receipt, receipt_path, dual_path = MODULE.build_fail_closed_receipt(
        bundle_path=MODULE.DEFAULT_BUNDLE,
        rd1_duals_path=MODULE.DEFAULT_RD1_DUALS,
        rd1_frontier_path=MODULE.DEFAULT_RD1_FRONTIER,
        rg3_summary_path=MODULE.DEFAULT_RG3_SUMMARY,
        output_root=tmp_path,
        finished_at_utc="2026-07-24T15:40:00Z",
    )
    assert receipt["metric_bundle_gate"]["status"] == "PARTIAL"
    assert receipt["metric_bundle_gate"]["blockers"] == [
        "PF2_BUCKET_INPUT_ASSIGNMENT_ABSENT"
    ]
    assert receipt["homotopy"]["launched"] is False
    assert receipt["homotopy"]["rungs"] == []
    assert receipt["actuation"] == dict.fromkeys(receipt["actuation"], False)
    assert receipt["rd1_dual_backfill"]["measured_cell_count"] == 0
    assert receipt["rd1_dual_backfill"]["still_null_cell_count"] == 162
    assert receipt["rg3_assignment_gate"]["missing_block_count"] == 25
    assert receipt_path.is_file()

    duals = json.loads(dual_path.read_text(encoding="utf-8"))
    assert duals["source_cell_count"] == 162
    assert duals["measured_cell_count"] == 0
    assert duals["still_null_cell_count"] == 162
    assert len(duals["cells"]) == 162
    assert all(row["lambda_bytes_per_D_dimension"] is None for row in duals["cells"])
    assert all(row["actionable_for_train_decision"] is False for row in duals["cells"])


def test_exact_error_count_is_rederived_not_rounded_charter_value(
    tmp_path: Path,
) -> None:
    receipt, _, _ = MODULE.build_fail_closed_receipt(
        bundle_path=MODULE.DEFAULT_BUNDLE,
        rd1_duals_path=MODULE.DEFAULT_RD1_DUALS,
        rd1_frontier_path=MODULE.DEFAULT_RD1_FRONTIER,
        rg3_summary_path=MODULE.DEFAULT_RG3_SUMMARY,
        output_root=tmp_path,
        finished_at_utc="2026-07-24T15:40:00Z",
    )
    arithmetic = receipt["exact_row_arithmetic"]
    assert arithmetic["scored_pixels"] == 117_964_800
    assert arithmetic["measured_error_count"] == 17_927
    assert arithmetic["charter_rounded_error_count"] == 17_931
    assert arithmetic["rounding_difference_errors"] == 4
    assert arithmetic["box_allowed_errors"] == 136_839


def test_new_operator_directives_are_fail_closed_next_run_columns(
    tmp_path: Path,
) -> None:
    receipt, _, _ = MODULE.build_fail_closed_receipt(
        bundle_path=MODULE.DEFAULT_BUNDLE,
        rd1_duals_path=MODULE.DEFAULT_RD1_DUALS,
        rd1_frontier_path=MODULE.DEFAULT_RD1_FRONTIER,
        rg3_summary_path=MODULE.DEFAULT_RG3_SUMMARY,
        output_root=tmp_path,
        finished_at_utc="2026-07-24T15:40:00Z",
    )
    homotopy = receipt["homotopy"]
    assert homotopy["waterfill_required"] is True
    assert homotopy["uniform_homotopy_role"] == "LABELED_CONTROL_ONLY"
    assert "ORDER1_CONTEXT_ARITHMETIC" in homotopy["coder_race_required"]
    assert "best_coded_bytes" in homotopy["per_rung_required_columns"]
    partition = receipt["visibility_partition_next_run_contract"]
    assert partition["both_blind_gauge"]["counted_bytes"] == 0
    assert partition["both_blind_gauge"]["required_check"].startswith("REVERIFY")
    assert partition["pose_only"]["coordinate_family"] == "frame_0"
    assert partition["per_type_mass"] is None
    assert partition["per_type_byte_spend"] is None
