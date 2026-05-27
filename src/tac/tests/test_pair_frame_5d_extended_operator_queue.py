# SPDX-License-Identifier: MIT
"""Tests for the 5D extended-operator local queue builder."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from comma_lab.scheduler.pair_frame_5d_extended_operator_queue import (
    PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA,
    build_pair_frame_5d_extended_operator_queue,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators import (
    ExtendedOperation,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL = _REPO_ROOT / "tools" / "build_5d_extended_operator_queue.py"


def _cell(pair_idx: int) -> PairFrameScorerGeometryCell:
    return PairFrameScorerGeometryCell(
        pair_idx=pair_idx,
        frame_idx=2 * pair_idx,
        scorer_axis=ScorerAxis.SEGNET_5CLASS,
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        cpu_cuda_axis=CpuCudaAxis.CONTEST_CPU,
        predicted_delta_score=-0.01,
        predicted_byte_cost=0,
        receiver_feasibility=True,
    )


def _write_canvas(path: pathlib.Path) -> None:
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256="c" * 64,
        cells={_cell(0).coordinate: _cell(0)},
    )
    path.write_text(
        json.dumps(
            {
                "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
                "archive_sha256": canvas.archive_sha256,
                "cells": [cell.as_dict() for cell in canvas._cells.values()],
            }
        ),
        encoding="utf-8",
    )


def test_build_pair_frame_5d_extended_operator_queue_shape(
    tmp_path: pathlib.Path,
) -> None:
    canvas_path = tmp_path / "canvas.json"
    _write_canvas(canvas_path)

    queue = build_pair_frame_5d_extended_operator_queue(
        repo_root=_REPO_ROOT,
        canvas_path=canvas_path,
        output_root=tmp_path / "operator_outputs",
        queue_id="unit_5d_extended_operator_queue",
        top_n=4,
        local_cpu_concurrency=3,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 3
    assert len(queue["experiments"]) == 8
    operations = {
        experiment["metadata"]["operation"] for experiment in queue["experiments"]
    }
    assert operations == {operation.value for operation in ExtendedOperation}
    for experiment in queue["experiments"]:
        assert experiment["status"] == "queued"
        assert experiment["metadata"]["schema"] == (
            PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA
        )
        assert experiment["metadata"]["score_claim"] is False
        assert experiment["metadata"]["promotable"] is False
        command = experiment["steps"][0]["command"]
        assert "tools/apply_8_extended_operators_to_5d_canvas_cli.py" in command
        assert "--output" in command
        false_authority = experiment["steps"][0]["postconditions"][1]
        assert false_authority["type"] == "json_false_authority"
        assert false_authority["required_false"] == []


def test_build_5d_extended_operator_queue_cli(tmp_path: pathlib.Path) -> None:
    canvas_path = tmp_path / "canvas.json"
    queue_path = tmp_path / "queue.json"
    _write_canvas(canvas_path)

    subprocess.run(
        [
            sys.executable,
            str(_TOOL),
            "--canvas-path",
            str(canvas_path),
            "--output-root",
            str(tmp_path / "operator_outputs"),
            "--queue-out",
            str(queue_path),
            "--queue-id",
            "unit_5d_extended_operator_queue_cli",
            "--top-n",
            "4",
            "--local-cpu-concurrency",
            "2",
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue_id"] == "unit_5d_extended_operator_queue_cli"
    assert payload["controls"]["max_concurrency"]["local_cpu"] == 2
    assert len(payload["experiments"]) == 8
