from __future__ import annotations

import copy
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pytest

from tac.optimization.ddm_ev2_per_pair_allocation import (
    EXPECTED_HEADLINE_BLOCKERS,
    MEASURED_C1_BYTES,
    UNALLOCATED_STATUS,
    EV2AllocationError,
    build_ev2_allocation,
)
from tac.optimization.ddm_pf2_bucket_assignment import validate_assignment_table

REPO = Path(__file__).resolve().parents[4]
C1_ARCHIVE = REPO / (
    ".omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/"
    "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)
LP1 = REPO / (
    ".omx/research/ddm_lp1_layer_pricing_20260725T031654Z/"
    "ddm_lp1_layer_pricing_receipt.json"
)
EV1 = REPO / (
    ".omx/research/ddm_ev1_campaign_evidence_joins_20260724T191623Z/"
    "ddm_ev1_campaign_evidence_join_receipt.json"
)
RD1 = REPO / (
    ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "typed_dimension_duals_effective_quantum.json"
)
R3 = REPO / (
    ".omx/research/ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/"
    "receipt.json"
)
MS5 = REPO / (
    ".omx/research/ddm_ms5_pf2_bucket_assignment_20260724T044736Z/"
    "pf2_bucket_assignment_table.json"
)
BUNDLE = REPO / (
    ".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/"
    "BUNDLE-COMPLETE.json"
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _build(**overrides: object):
    arguments = {
        "c1_archive": C1_ARCHIVE.read_bytes(),
        "lp1": _json(LP1),
        "ev1": _json(EV1),
        "rd1": _json(RD1),
        "r3": _json(R3),
        "ms5": _json(MS5),
        "ms5_sha256": hashlib.sha256(MS5.read_bytes()).hexdigest(),
        "bundle_path": BUNDLE,
        "repository_root": REPO,
    }
    arguments.update(overrides)
    return build_ev2_allocation(**arguments)


def test_exact_construction_lineage_fires_preregistered_falsifier() -> None:
    result = _build()
    conservation = result.receipt["mass_conservation"]
    assert conservation == {
        "lp1_measured_bytes": MEASURED_C1_BYTES,
        "assigned_pair_cell_bytes": 0,
        "unallocated_bytes": MEASURED_C1_BYTES,
        "separable_fraction": 0.0,
        "unallocated_fraction": 1.0,
        "conserved": True,
    }
    assert result.receipt["falsifier"] == {
        "threshold_unallocated_fraction": 0.3,
        "observed_unallocated_fraction": 1.0,
        "fired": True,
        "verdict": "FORMULATION_MISPOSED_FOR_CURRENT_C1_COMPOSITION",
        "verdict_scope": "FORMULATION",
    }
    assert result.receipt["score_claim"] is False
    assert result.receipt["promotion_eligible"] is False
    assert result.receipt["pointer_moved"] is False


def test_all_mass_is_typed_unallocated_at_the_coarsest_lawful_partition() -> None:
    result = _build()
    table = result.allocation_table
    assert len(table["rows"]) == 162
    assert len(table["pair_rows"]) == 600
    assert all(row["assigned_counted_bytes"] == 0 for row in table["rows"])
    assert all(row["pair_ids"] == [] for row in table["rows"])
    assert all(
        row["assignment_status"] == UNALLOCATED_STATUS for row in table["rows"]
    )
    assert all(
        row["exclusive_archive_section_counted_bytes"] == 0
        and row["exclusive_archive_section_present"] is False
        for row in table["pair_rows"]
    )
    coarse = table["coarse_lawful_partition"]["rows"]
    assert len(coarse) == 7
    assert sum(row["counted_bytes"] for row in coarse) == MEASURED_C1_BYTES
    assert {row["derivation_method"] for row in coarse} == {
        "EXACT_CONSTRUCTION_LINEAGE"
    }
    archive_ranges = sorted(
        row["byte_range"] for row in coarse if row["byte_range"] is not None
    )
    assert archive_ranges[0][0] == 0
    assert archive_ranges[-1][1] == 133_941
    assert all(
        left[1] == right[0]
        for left, right in pairwise(archive_ranges)
    )


def test_exact_ms5_loader_schema_projection_remains_strictly_loadable() -> None:
    result = _build()
    projection = result.ms5_loader_table
    assert projection["schema"] == "ddm_ms5_pf2_bucket_assignment_table.v1"
    assert len(projection["rows"]) == 1_200
    assert projection["ev2_rate_home_extension"]["cell_count"] == 162
    assert projection["ev2_rate_home_extension"]["assigned_counted_bytes"] == 0
    validate_assignment_table(
        projection,
        expected_pf2_sha256=projection["pf2_receipt_sha256"],
    )


def test_ev1_accounting_cube_is_conserved_without_cross_object_smearing() -> None:
    result = _build()
    conservation = result.receipt["ev1_accounting_home_conservation"]
    assert conservation["same_object_as_c1_allocation"] is False
    assert conservation["all_exclusive_accounting_homes_reconciled"] is True
    assert {
        dual: row["cell_home_sum_bytes"]
        for dual, row in conservation["per_dual"].items()
    } == {"1": 16, "2": 962, "3": 409_388_124}
    assert result.receipt["same_object_firewall"]["cross_object_byte_smearing"] == 0


def test_all_costates_remain_null_and_headline_blockers_are_unchanged() -> None:
    result = _build()
    assert result.rd1_backfill["computable_cell_count"] == 0
    assert result.rd1_backfill["still_null_cell_count"] == 162
    assert all(
        row["lambda_bytes_per_D_dimension"] is None
        for row in result.rd1_backfill["cells"]
    )
    assert result.headline_replay["remaining_blockers"] == list(
        EXPECTED_HEADLINE_BLOCKERS
    )
    assert result.headline_replay["edge_cleared_blockers"] == []


def test_lp1_mass_or_ev1_cell_identity_drift_fails_closed() -> None:
    lp1 = copy.deepcopy(_json(LP1))
    lp1["c1_corrected_waterfill"]["corrected_measured_allocated_bytes"] -= 1
    with pytest.raises(EV2AllocationError, match="LP1 measured C1 allocation"):
        _build(lp1=lp1)

    ev1 = copy.deepcopy(_json(EV1))
    ev1["rd1_evidence"]["bucket_rows"][1]["dual_index"] = ev1[
        "rd1_evidence"
    ]["bucket_rows"][0]["dual_index"]
    ev1["rd1_evidence"]["bucket_rows"][1]["stratum"] = ev1["rd1_evidence"][
        "bucket_rows"
    ][0]["stratum"]
    ev1["rd1_evidence"]["bucket_rows"][1]["scorer_visibility"] = ev1[
        "rd1_evidence"
    ]["bucket_rows"][0]["scorer_visibility"]
    ev1["rd1_evidence"]["bucket_rows"][1]["g4_temporal_class"] = ev1[
        "rd1_evidence"
    ]["bucket_rows"][0]["g4_temporal_class"]
    with pytest.raises(EV2AllocationError, match="do not cover the sealed cube"):
        _build(ev1=ev1)
