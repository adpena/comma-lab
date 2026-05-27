# SPDX-License-Identifier: MIT
"""Coverage audit for the pair-frame scorer-geometry 5D canvas.

The 5D canvas is only useful for grouped inverse-steganalysis search when it is
dense enough to expose scorer/runtime interactions. A sparse canvas that emits
zero operator candidates is not a dead end; it is a typed request for more
anchors. This module converts that condition into machine-readable work orders
instead of leaving it as manual diagnosis.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CANONICAL_FRAME_COUNT,
    CANONICAL_PAIR_COUNT,
    CANVAS_SCHEMA,
    CELL_SCHEMA,
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.proxy_candidate_contract import (
    require_no_truthy_authority_fields,
)

COVERAGE_AUDIT_SCHEMA = (
    "pair_frame_scorer_geometry_lattice_5d_canvas_coverage_audit.v1"
)
WORK_ORDER_SCHEMA = (
    "pair_frame_scorer_geometry_lattice_5d_canvas_coverage_work_order.v1"
)
POPULATED_CANVAS_SCHEMA = "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1"
CONSUMER_NAME = "pair_frame_scorer_geometry_lattice_5d_canvas_coverage"
CONSUMER_VERSION = "0.1.0"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_packet_ready": False,
    "reproduction_claim": False,
    "reproduction_equivalence": False,
}


class CanvasCoverageAuditError(ValueError):
    """Raised when a 5D canvas coverage audit cannot parse its input."""


def _as_counter(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _cell_from_mapping(row: Mapping[str, Any], *, source: Path | str) -> PairFrameScorerGeometryCell:
    try:
        cell_schema = row.get("schema")
        if cell_schema not in (None, CELL_SCHEMA):
            raise CanvasCoverageAuditError(
                f"{source}: cell schema mismatch: got {cell_schema!r}, "
                f"expected {CELL_SCHEMA!r}"
            )
        return PairFrameScorerGeometryCell(
            pair_idx=int(row["pair_idx"]),
            frame_idx=int(row["frame_idx"]),
            scorer_axis=ScorerAxis(row["scorer_axis"]),
            receiver_runtime=ReceiverRuntime(row["receiver_runtime"]),
            cpu_cuda_axis=CpuCudaAxis(row["cpu_cuda_axis"]),
            predicted_delta_score=float(row["predicted_delta_score"]),
            predicted_byte_cost=int(row["predicted_byte_cost"]),
            receiver_feasibility=bool(row["receiver_feasibility"]),
            catalog_323_provenance=dict(row.get("catalog_323_provenance", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CanvasCoverageAuditError):
            raise
        raise CanvasCoverageAuditError(
            f"{source}: failed to parse 5D canvas cell: {exc!r}"
        ) from exc


def load_5d_canvas_json(canvas_path: Path) -> PairFrameScorerGeometryLattice:
    """Load a canvas sidecar or populated-canvas JSON manifest.

    The loader is intentionally independent from the operator CLIs so audit,
    queue construction, and future cathedral consumers share one parser.
    """

    if not isinstance(canvas_path, Path):
        raise CanvasCoverageAuditError(
            f"canvas_path must be Path, got {type(canvas_path).__name__}"
        )
    if not canvas_path.exists():
        raise CanvasCoverageAuditError(f"canvas JSON does not exist: {canvas_path}")
    try:
        with canvas_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise CanvasCoverageAuditError(
            f"{canvas_path}: corrupt JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CanvasCoverageAuditError(
            f"{canvas_path}: expected JSON object, got {type(payload).__name__}"
        )

    schema = payload.get("schema")
    if schema not in (CANVAS_SCHEMA, POPULATED_CANVAS_SCHEMA):
        raise CanvasCoverageAuditError(
            f"{canvas_path}: schema mismatch: got {schema!r}, expected "
            f"{CANVAS_SCHEMA!r} or {POPULATED_CANVAS_SCHEMA!r}"
        )
    archive_sha256 = payload.get("archive_sha256")
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        raise CanvasCoverageAuditError(
            f"{canvas_path}: missing valid archive_sha256"
        )
    cells_payload = payload.get("cells", [])
    if not isinstance(cells_payload, list):
        raise CanvasCoverageAuditError(
            f"{canvas_path}: cells must be a list, got "
            f"{type(cells_payload).__name__}"
        )

    cells: dict[tuple, PairFrameScorerGeometryCell] = {}
    for row in cells_payload:
        if not isinstance(row, Mapping):
            raise CanvasCoverageAuditError(
                f"{canvas_path}: cell row must be object, got "
                f"{type(row).__name__}"
            )
        cell = _cell_from_mapping(row, source=canvas_path)
        cells[cell.coordinate] = cell
    return PairFrameScorerGeometryLattice(archive_sha256=archive_sha256, cells=cells)


def _work_order(
    *,
    order_id: str,
    priority: int,
    reason: str,
    consumer: str,
    target: Mapping[str, Any],
    suggested_next_tools: Sequence[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": WORK_ORDER_SCHEMA,
        "id": order_id,
        "priority": int(priority),
        "reason": reason,
        "consumer": consumer,
        "target": dict(target),
        "suggested_next_tools": [str(tool) for tool in suggested_next_tools],
        "allowed_use": "experiment_queue_v1_planning_input_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context=f"5d_canvas_work_order:{order_id}")
    return payload


def audit_5d_canvas_coverage(
    canvas: PairFrameScorerGeometryLattice,
    *,
    min_pair_coverage_for_grouped_search: float = 0.05,
    min_frame_coverage_for_grouped_search: float = 0.05,
    require_both_contest_axes: bool = True,
    require_receiver_runtime_diversity: bool = True,
) -> dict[str, Any]:
    """Summarize whether a populated canvas can drive grouped search.

    The audit is false-authority: it is a local planning surface. Its job is to
    tell the acquisition stack what evidence is missing before the next queue
    fan-out, not to promote or kill any candidate.
    """

    if not isinstance(canvas, PairFrameScorerGeometryLattice):
        raise CanvasCoverageAuditError(
            "canvas must be PairFrameScorerGeometryLattice, got "
            f"{type(canvas).__name__}"
        )
    for label, value in (
        ("min_pair_coverage_for_grouped_search", min_pair_coverage_for_grouped_search),
        ("min_frame_coverage_for_grouped_search", min_frame_coverage_for_grouped_search),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise CanvasCoverageAuditError(f"{label} must be finite in [0, 1]")

    cells = list(canvas._cells.values())
    feasible_cells = [cell for cell in cells if cell.receiver_feasibility]
    negative_cells = [
        cell
        for cell in feasible_cells
        if math.isfinite(float(cell.predicted_delta_score))
        and float(cell.predicted_delta_score) < 0.0
    ]
    finite_deltas = [
        float(cell.predicted_delta_score)
        for cell in cells
        if math.isfinite(float(cell.predicted_delta_score))
    ]
    pairs = {cell.pair_idx for cell in cells}
    frames = {cell.frame_idx for cell in cells}
    feasible_pairs = {cell.pair_idx for cell in feasible_cells}
    feasible_frames = {cell.frame_idx for cell in feasible_cells}
    negative_pairs = {cell.pair_idx for cell in negative_cells}
    negative_frames = {cell.frame_idx for cell in negative_cells}
    cpu_cuda_axes_present = {cell.cpu_cuda_axis for cell in cells}
    receiver_runtimes_present = {cell.receiver_runtime for cell in cells}

    full_cell_count = (
        CANONICAL_PAIR_COUNT
        * CANONICAL_FRAME_COUNT
        * len(ScorerAxis)
        * len(ReceiverRuntime)
        * len(CpuCudaAxis)
    )
    pair_coverage = _ratio(len(pairs), CANONICAL_PAIR_COUNT)
    frame_coverage = _ratio(len(frames), CANONICAL_FRAME_COUNT)

    blockers: list[str] = []
    if not cells:
        blockers.append("canvas_empty")
    if not feasible_cells:
        blockers.append("no_receiver_feasible_cells")
    if require_both_contest_axes:
        missing_axes = [
            axis.value for axis in CpuCudaAxis if axis not in cpu_cuda_axes_present
        ]
        blockers.extend(f"missing_cpu_cuda_axis:{axis}" for axis in missing_axes)
    if require_receiver_runtime_diversity and len(receiver_runtimes_present) < 2:
        blockers.append("receiver_runtime_diversity_missing")
    if pair_coverage < float(min_pair_coverage_for_grouped_search):
        blockers.append(
            "pair_coverage_below_grouped_search_floor:"
            f"{pair_coverage:.6f}<{float(min_pair_coverage_for_grouped_search):.6f}"
        )
    if frame_coverage < float(min_frame_coverage_for_grouped_search):
        blockers.append(
            "frame_coverage_below_grouped_search_floor:"
            f"{frame_coverage:.6f}<{float(min_frame_coverage_for_grouped_search):.6f}"
        )
    if not negative_cells:
        blockers.append("no_negative_predicted_delta_cells")
    if len(feasible_pairs) <= 1 and feasible_cells:
        blockers.append("only_single_feasible_pair_anchor")
    if len(feasible_frames) <= 1 and feasible_cells:
        blockers.append("only_single_feasible_frame_anchor")

    work_orders: list[dict[str, Any]] = []
    priority = 1
    if "canvas_empty" in blockers:
        work_orders.append(
            _work_order(
                order_id="populate_5d_canvas_from_master_gradient_anchors",
                priority=priority,
                reason="canvas has zero populated cells",
                consumer="tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator",
                target={"archive_sha256": canvas.archive_sha256},
                suggested_next_tools=[
                    ".venv/bin/python tools/populate_5d_canvas_cli.py --latest"
                ],
            )
        )
        priority += 1
    missing_axis_values = [
        axis.value for axis in CpuCudaAxis if axis not in cpu_cuda_axes_present
    ]
    if missing_axis_values:
        work_orders.append(
            _work_order(
                order_id="populate_missing_paired_cpu_cuda_axis_anchors",
                priority=priority,
                reason="5D canvas cannot calibrate scorer-axis drift without both contest axes",
                consumer="paired_auth_eval_consumer",
                target={
                    "archive_sha256": canvas.archive_sha256,
                    "missing_cpu_cuda_axes": missing_axis_values,
                },
                suggested_next_tools=[
                    ".venv/bin/python tools/paired_auth_eval_cli.py",
                    ".venv/bin/python tools/populate_5d_canvas_cli.py --latest",
                ],
            )
        )
        priority += 1
    if pair_coverage < float(min_pair_coverage_for_grouped_search):
        work_orders.append(
            _work_order(
                order_id="densify_pair_coverage_for_grouped_search",
                priority=priority,
                reason="pair coverage too sparse for replace-many or temporal interaction search",
                consumer="master_gradient_pair_acquisition",
                target={
                    "archive_sha256": canvas.archive_sha256,
                    "current_unique_pairs": len(pairs),
                    "target_unique_pairs_floor": math.ceil(
                        CANONICAL_PAIR_COUNT
                        * float(min_pair_coverage_for_grouped_search)
                    ),
                },
                suggested_next_tools=[
                    "tac.master_gradient_anchors targeted per-pair acquisition",
                    ".venv/bin/python tools/populate_5d_canvas_cli.py --latest",
                ],
            )
        )
        priority += 1
    if frame_coverage < float(min_frame_coverage_for_grouped_search):
        work_orders.append(
            _work_order(
                order_id="densify_frame_coverage_for_masked_and_feathered_search",
                priority=priority,
                reason="frame coverage too sparse for SegNet/PoseNet region waterfill",
                consumer="per_frame_sensitivity_consumer",
                target={
                    "archive_sha256": canvas.archive_sha256,
                    "current_unique_frames": len(frames),
                    "target_unique_frames_floor": math.ceil(
                        CANONICAL_FRAME_COUNT
                        * float(min_frame_coverage_for_grouped_search)
                    ),
                },
                suggested_next_tools=[
                    "tac.cathedral_consumers.per_frame_sensitivity_consumer",
                    ".venv/bin/python tools/populate_5d_canvas_cli.py --latest",
                ],
            )
        )
        priority += 1
    if not negative_cells:
        work_orders.append(
            _work_order(
                order_id="acquire_negative_delta_cells_before_operator_fanout",
                priority=priority,
                reason=(
                    "all feasible cells are score-worsening or neutral; "
                    "fanout will be empty until acquisition targets scorer-null directions"
                ),
                consumer="inverse_steganalysis_acquisition",
                target={
                    "archive_sha256": canvas.archive_sha256,
                    "feasible_cell_count": len(feasible_cells),
                    "best_observed_delta": min(finite_deltas) if finite_deltas else None,
                },
                suggested_next_tools=[
                    "tac.optimization.scorer_inverse_decision_surface",
                    "master-gradient constrained PoseNet-null / SegNet-region waterfill",
                    ".venv/bin/python tools/apply_8_extended_operators_to_5d_canvas_cli.py --operator all",
                ],
            )
        )
        priority += 1
    if require_receiver_runtime_diversity and len(receiver_runtimes_present) < 2:
        work_orders.append(
            _work_order(
                order_id="populate_receiver_runtime_mode_diversity",
                priority=priority,
                reason="only one receiver_runtime mode is present; repair/masked/feathered interactions cannot be compared",
                consumer="cooperative_receiver_integration",
                target={
                    "archive_sha256": canvas.archive_sha256,
                    "present_receiver_runtimes": sorted(
                        runtime.value for runtime in receiver_runtimes_present
                    ),
                },
                suggested_next_tools=[
                    "tac.optimization.cooperative_receiver_integration",
                    ".venv/bin/python tools/populate_5d_canvas_cli.py --latest",
                ],
            )
        )

    payload: dict[str, Any] = {
        "schema": COVERAGE_AUDIT_SCHEMA,
        "consumer_name": CONSUMER_NAME,
        "consumer_version": CONSUMER_VERSION,
        "archive_sha256": canvas.archive_sha256,
        "canvas_schema": CANVAS_SCHEMA,
        "canonical_pair_count": CANONICAL_PAIR_COUNT,
        "canonical_frame_count": CANONICAL_FRAME_COUNT,
        "total_possible_cells": full_cell_count,
        "populated_cell_count": len(cells),
        "populated_cell_coverage_fraction": _ratio(len(cells), full_cell_count),
        "feasible_cell_count": len(feasible_cells),
        "negative_cell_count": len(negative_cells),
        "nonnegative_feasible_cell_count": len(feasible_cells) - len(negative_cells),
        "unique_pair_count": len(pairs),
        "unique_frame_count": len(frames),
        "feasible_pair_count": len(feasible_pairs),
        "feasible_frame_count": len(feasible_frames),
        "negative_pair_count": len(negative_pairs),
        "negative_frame_count": len(negative_frames),
        "pair_coverage_fraction": pair_coverage,
        "frame_coverage_fraction": frame_coverage,
        "min_pair_coverage_for_grouped_search": float(
            min_pair_coverage_for_grouped_search
        ),
        "min_frame_coverage_for_grouped_search": float(
            min_frame_coverage_for_grouped_search
        ),
        "scorer_axis_counts": _as_counter(
            [cell.scorer_axis.value for cell in cells]
        ),
        "receiver_runtime_counts": _as_counter(
            [cell.receiver_runtime.value for cell in cells]
        ),
        "cpu_cuda_axis_counts": _as_counter(
            [cell.cpu_cuda_axis.value for cell in cells]
        ),
        "feasible_by_cpu_cuda_axis": _as_counter(
            [cell.cpu_cuda_axis.value for cell in feasible_cells]
        ),
        "negative_by_cpu_cuda_axis": _as_counter(
            [cell.cpu_cuda_axis.value for cell in negative_cells]
        ),
        "feasible_by_receiver_runtime": _as_counter(
            [cell.receiver_runtime.value for cell in feasible_cells]
        ),
        "best_predicted_delta_score": min(finite_deltas) if finite_deltas else None,
        "worst_predicted_delta_score": max(finite_deltas) if finite_deltas else None,
        "negative_cells_total_predicted_byte_cost": sum(
            int(cell.predicted_byte_cost) for cell in negative_cells
        ),
        "blockers": _ordered_unique(blockers),
        "work_order_count": len(work_orders),
        "work_orders": work_orders,
        "verdict": "ready_for_grouped_search" if not blockers else "densification_required",
        "allowed_use": "local_planning_and_experiment_queue_acquisition_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="5d_canvas_coverage_audit")
    return payload


def audit_5d_canvas_coverage_json(
    canvas_path: Path,
    *,
    min_pair_coverage_for_grouped_search: float = 0.05,
    min_frame_coverage_for_grouped_search: float = 0.05,
) -> dict[str, Any]:
    """Load a canvas JSON file and return its coverage audit payload."""

    canvas = load_5d_canvas_json(canvas_path)
    return audit_5d_canvas_coverage(
        canvas,
        min_pair_coverage_for_grouped_search=min_pair_coverage_for_grouped_search,
        min_frame_coverage_for_grouped_search=min_frame_coverage_for_grouped_search,
    )


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "COVERAGE_AUDIT_SCHEMA",
    "FALSE_AUTHORITY",
    "POPULATED_CANVAS_SCHEMA",
    "WORK_ORDER_SCHEMA",
    "CanvasCoverageAuditError",
    "audit_5d_canvas_coverage",
    "audit_5d_canvas_coverage_json",
    "load_5d_canvas_json",
]
