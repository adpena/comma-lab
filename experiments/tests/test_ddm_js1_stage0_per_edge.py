from __future__ import annotations

import numpy as np

from experiments.ddm_js1_stage0_per_edge import (
    C1_BATCH16_REFERENCE_FLIPS,
    CLASSES,
    TERMINAL_BASE_FLIPS,
    adjudicate_axis,
    confusion,
    matrix_summary,
)


def test_confusion_is_directed_gt_by_rendered() -> None:
    gt = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.uint8)
    predicted = np.array([[0, 1, 0], [1, 2, 3]], dtype=np.uint8)
    matrix = confusion(gt, predicted)
    assert matrix.sum() == gt.size
    assert matrix[0, 1] == 1
    assert matrix[1, 0] == 1
    assert matrix[2, 3] == 1
    assert int(matrix.trace()) == 3


def test_matrix_summary_preserves_asymmetry_and_denominator() -> None:
    matrix = np.zeros((5, 5), dtype=np.int64)
    matrix[0, 0] = 100
    matrix[0, 1] = 3
    matrix[1, 0] = 9
    matrix[2, 0] = 2
    result = matrix_summary(matrix)
    assert result["total_flips"] == 14
    assert result["road_incident_flips"] == 14
    road_lane = next(
        row for row in result["undirected_edges"] if row["edge"] == "Road<->Lane"
    )
    assert road_lane["flips"] == 12
    assert road_lane["Road->Lane"] == 3
    assert road_lane["Lane->Road"] == 9
    assert road_lane["asymmetry_ratio"] == 3.0
    assert {row["gt_class"] for row in result["directed_cells"]} == set(CLASSES)


def test_axis_adjudication_fails_closed_on_either_reference_mismatch() -> None:
    assert adjudicate_axis(
        TERMINAL_BASE_FLIPS, C1_BATCH16_REFERENCE_FLIPS
    )["admitted_for_stage0_rho"]
    base_mismatch = adjudicate_axis(
        TERMINAL_BASE_FLIPS + 1, C1_BATCH16_REFERENCE_FLIPS
    )
    target_mismatch = adjudicate_axis(
        TERMINAL_BASE_FLIPS, C1_BATCH16_REFERENCE_FLIPS + 1
    )
    assert base_mismatch["status"] == "BLOCKED_AXIS_MISMATCH"
    assert target_mismatch["status"] == "BLOCKED_AXIS_MISMATCH"
    assert not base_mismatch["admitted_for_stage0_rho"]
    assert not target_mismatch["admitted_for_stage0_rho"]
