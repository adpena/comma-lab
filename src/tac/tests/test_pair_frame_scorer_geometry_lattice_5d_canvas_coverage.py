# SPDX-License-Identifier: MIT
"""Tests for the 5D canvas coverage audit."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from comma_lab.scheduler.pair_frame_5d_extended_operator_queue import (
    build_pair_frame_5d_extended_operator_queue,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage import (
    COVERAGE_AUDIT_SCHEMA,
    WORK_ORDER_SCHEMA,
    audit_5d_canvas_coverage,
    load_5d_canvas_json,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL = _REPO_ROOT / "tools" / "audit_5d_canvas_coverage.py"


def _cell(
    pair_idx: int,
    *,
    axis: CpuCudaAxis = CpuCudaAxis.CONTEST_CPU,
    runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL,
    delta: float = 0.01,
) -> PairFrameScorerGeometryCell:
    return PairFrameScorerGeometryCell(
        pair_idx=pair_idx,
        frame_idx=2 * pair_idx,
        scorer_axis=ScorerAxis.SEGNET_5CLASS,
        receiver_runtime=runtime,
        cpu_cuda_axis=axis,
        predicted_delta_score=delta,
        predicted_byte_cost=0,
        receiver_feasibility=True,
    )


def _canvas(cells: list[PairFrameScorerGeometryCell]) -> PairFrameScorerGeometryLattice:
    return PairFrameScorerGeometryLattice(
        archive_sha256="d" * 64,
        cells={cell.coordinate: cell for cell in cells},
    )


def _write_canvas(path: pathlib.Path, cells: list[PairFrameScorerGeometryCell]) -> None:
    canvas = _canvas(cells)
    path.write_text(
        json.dumps(
            {
                "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
                "archive_sha256": canvas.archive_sha256,
                "cells": [cell.as_dict() for cell in canvas._cells.values()],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_sparse_positive_canvas_emits_densification_orders() -> None:
    audit = audit_5d_canvas_coverage(_canvas([_cell(0)]))

    assert audit["schema"] == COVERAGE_AUDIT_SCHEMA
    assert audit["verdict"] == "densification_required"
    assert audit["score_claim"] is False
    assert audit["ready_for_exact_eval_dispatch"] is False
    assert "missing_cpu_cuda_axis:contest_cuda_t4" in audit["blockers"]
    assert "receiver_runtime_diversity_missing" in audit["blockers"]
    assert "no_negative_predicted_delta_cells" in audit["blockers"]
    work_order_ids = {order["id"] for order in audit["work_orders"]}
    assert "populate_missing_paired_cpu_cuda_axis_anchors" in work_order_ids
    assert "acquire_negative_delta_cells_before_operator_fanout" in work_order_ids
    for order in audit["work_orders"]:
        assert order["schema"] == WORK_ORDER_SCHEMA
        assert order["promotable"] is False


def test_audit_dense_enough_negative_biaxial_canvas_is_grouped_search_ready() -> None:
    cells: list[PairFrameScorerGeometryCell] = []
    for pair_idx in range(60):
        for axis in (CpuCudaAxis.CONTEST_CPU, CpuCudaAxis.CONTEST_CUDA_T4):
            for runtime in (
                ReceiverRuntime.RAW_RESIDUAL,
                ReceiverRuntime.FEATHERED,
            ):
                cells.append(_cell(pair_idx, axis=axis, runtime=runtime, delta=-0.01))

    audit = audit_5d_canvas_coverage(_canvas(cells))

    assert audit["verdict"] == "ready_for_grouped_search"
    assert audit["blockers"] == []
    assert audit["unique_pair_count"] == 60
    assert audit["negative_pair_count"] == 60
    assert audit["cpu_cuda_axis_counts"] == {
        "contest_cpu": 120,
        "contest_cuda_t4": 120,
    }


def test_load_5d_canvas_json_round_trips_populated_sidecar(
    tmp_path: pathlib.Path,
) -> None:
    canvas_path = tmp_path / "canvas.json"
    _write_canvas(canvas_path, [_cell(0), _cell(1, delta=-0.01)])

    canvas = load_5d_canvas_json(canvas_path)
    audit = audit_5d_canvas_coverage(
        canvas,
        min_pair_coverage_for_grouped_search=0.0,
        min_frame_coverage_for_grouped_search=0.0,
        require_both_contest_axes=False,
        require_receiver_runtime_diversity=False,
    )

    assert canvas.cell_count() == 2
    assert audit["negative_cell_count"] == 1


def test_audit_5d_canvas_coverage_cli(tmp_path: pathlib.Path) -> None:
    canvas_path = tmp_path / "canvas.json"
    audit_path = tmp_path / "audit.json"
    _write_canvas(canvas_path, [_cell(0)])

    subprocess.run(
        [
            sys.executable,
            str(_TOOL),
            "--canvas-path",
            str(canvas_path),
            "--output",
            str(audit_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["schema"] == COVERAGE_AUDIT_SCHEMA
    assert payload["work_order_count"] >= 1
    assert payload["score_claim"] is False


def test_extended_operator_queue_carries_canvas_coverage_audit(
    tmp_path: pathlib.Path,
) -> None:
    canvas_path = tmp_path / "canvas.json"
    _write_canvas(canvas_path, [_cell(0)])

    queue = build_pair_frame_5d_extended_operator_queue(
        repo_root=_REPO_ROOT,
        canvas_path=canvas_path,
        output_root=tmp_path / "operator_outputs",
        queue_id="unit_5d_extended_operator_queue_with_coverage",
        top_n=4,
    )

    assert len(queue["experiments"]) == 8
    for experiment in queue["experiments"]:
        audit = experiment["metadata"]["canvas_coverage_audit"]
        assert audit["schema"] == COVERAGE_AUDIT_SCHEMA
        assert audit["verdict"] == "densification_required"
        assert audit["score_claim"] is False
        assert "no_negative_predicted_delta_cells" in audit["blockers"]
