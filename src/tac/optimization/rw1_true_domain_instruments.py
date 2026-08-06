# SPDX-License-Identifier: MIT
"""rw1 true-domain instrument helpers.

These helpers are intentionally scorer-free.  They convert existing solver
diagnostics into typed cap receipts, carry the rw1 element-grade vector, and
apply dk1's lattice-native private-block realizer to q3-style scorer paints.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.optimization.lattice_native_pose_null_realizer import (
    add_private_delta_to_frame,
    build_default_operator,
    extract_private_camera_block,
    private_block_geometry,
    realize_lattice_native_block,
)
from tac.optimization.trajectory_stopping import CapStopReceipt, build_cap_stop_receipt

RW1_INSTRUMENT_SCHEMA = "ddm_rw1_true_domain_instrument.v1"
GRADE_VALUES = ("OPTIMAL-RECEIPT", "NAIVE-NAMED", "UNKNOWN")
ELEMENT_GRADE_KEYS = (
    "init",
    "step_rule",
    "stopping_rule",
    "metric",
    "subset",
    "realization",
    "projection",
    "tie_breaks",
    "seed",
    "caches",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=1, default=jsonable, allow_nan=False)
        fh.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=jsonable, allow_nan=False))
        fh.write("\n")


def parse_cap_ladder(text: str | None, *, fallback: int) -> tuple[int, ...]:
    """Parse a comma-separated positive integer cap ladder."""

    raw = text if text is not None and text.strip() else str(int(fallback))
    caps: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        cap = int(item)
        if cap < 1:
            raise ValueError(f"cap ladder entries must be positive: {cap}")
        caps.append(cap)
    if not caps:
        raise ValueError("cap ladder must contain at least one positive cap")
    return tuple(caps)


def _curve_last_step(curve: Any) -> int | None:
    if not isinstance(curve, list) or not curve:
        return None
    steps = [int(row["step"]) for row in curve if isinstance(row, Mapping) and "step" in row]
    return max(steps) if steps else None


def _selected_start_diagnostics(diagnostics: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = diagnostics.get("selected", {})
    start_name = selected.get("start") if isinstance(selected, Mapping) else None
    starts = diagnostics.get("starts", [])
    if isinstance(starts, list):
        for row in starts:
            if isinstance(row, Mapping) and row.get("start") == start_name:
                return row
    return selected if isinstance(selected, Mapping) else {}


def solver_best_proxy_flips(diagnostics: Mapping[str, Any]) -> int | None:
    selected = diagnostics.get("selected", {})
    if not isinstance(selected, Mapping):
        return None
    for key in ("best_proxy_flips", "best_proxy_phase_target_flips"):
        if key in selected and selected[key] is not None:
            return int(selected[key])
    start_diag = _selected_start_diagnostics(diagnostics)
    for key in ("best_proxy_flips", "best_proxy_phase_target_flips"):
        if key in start_diag and start_diag[key] is not None:
            return int(start_diag[key])
    return None


def cap_receipt_from_solver_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    cap: int,
) -> CapStopReceipt:
    """Convert sq1/sw1-style diagnostics into CA1's typed cap receipt."""

    selected = diagnostics.get("selected", {})
    selected = selected if isinstance(selected, Mapping) else {}
    start_diag = _selected_start_diagnostics(diagnostics)
    stop_reason = str(
        selected.get("stop_reason")
        or start_diag.get("stop_reason")
        or "UNKNOWN_STOP_REASON"
    )
    steps_run = selected.get("steps_run")
    if steps_run is None:
        steps_run = start_diag.get("steps_run")
    if steps_run is None:
        steps_run = _curve_last_step(selected.get("curve")) or _curve_last_step(start_diag.get("curve"))
    steps = int(steps_run) if steps_run is not None else 0

    if stop_reason.startswith("iteration_cap"):
        still_descending = stop_reason in {
            "iteration_cap_best_at_cap",
            "iteration_cap_no_convergence_test",
        }
        return build_cap_stop_receipt(
            stop_reason="cap_bound",
            steps_run=max(steps, int(cap)),
            cap=int(cap),
            still_descending=still_descending,
        )
    if stop_reason in {
        "plateau_no_proxy_improvement",
        "converged_projected",
        "marginal_below_bar",
    }:
        return build_cap_stop_receipt(
            stop_reason="converged",
            steps_run=steps,
            cap=int(cap),
            still_descending=False,
        )
    return build_cap_stop_receipt(
        stop_reason="failed",
        steps_run=steps,
        cap=int(cap),
        still_descending=None,
    )


def element_grade_vector(
    *,
    chain_name: str,
    overrides: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    """Build the full rw1 element grade vector, defaulting missing legs to UNKNOWN."""

    elements: dict[str, dict[str, str]] = {}
    for key in ELEMENT_GRADE_KEYS:
        grade, note = overrides.get(key, ("UNKNOWN", "not graded in this receipt"))
        if grade not in GRADE_VALUES:
            raise ValueError(f"bad grade for {key}: {grade}")
        elements[key] = {"grade": grade, "note": note}
    status = (
        "CURED"
        if all(row["grade"] == "OPTIMAL-RECEIPT" for row in elements.values())
        else "PARTIALLY-CURED"
    )
    return {
        "schema": "ddm_rw1_element_grade_vector.v1",
        "chain_name": chain_name,
        "elements": elements,
        "status": status,
    }


def block_mask_from_scorer_mask(mask: np.ndarray) -> np.ndarray:
    x = np.asarray(mask).astype(bool)
    if x.ndim != 2 or x.shape[0] % 2 or x.shape[1] % 2:
        raise ValueError(f"expected even 2-D scorer mask, got {x.shape}")
    return x.reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2).any(axis=(1, 3))


def q3_block_coverage_payload(
    *,
    total_blocks: int,
    realized_blocks: int,
    block_limit: int | None,
) -> dict[str, Any]:
    """Grade whether a Q3 realization covered the full requested block mask."""

    total = int(total_blocks)
    realized = int(realized_blocks)
    if total < 0:
        raise ValueError("total_blocks must be non-negative")
    if realized < 0:
        raise ValueError("realized_blocks must be non-negative")
    if realized > total:
        raise ValueError("realized_blocks cannot exceed total_blocks")
    if block_limit is not None and int(block_limit) < 1:
        raise ValueError("block_limit must be positive when supplied")
    full = realized == total and block_limit is None
    return {
        "blocks_total_requested_by_mask": total,
        "blocks_realized": realized,
        "blocks_unrealized": total - realized,
        "block_limit": None if block_limit is None else int(block_limit),
        "block_coverage_fraction": (float(realized) / float(total) if total else 1.0),
        "block_coverage_status": "FULL_REQUESTED_MASK" if full else "PARTIAL_OR_CAPPED",
        "coverage_form_grade": "OPTIMAL-RECEIPT" if full else "NAIVE-NAMED",
    }


def scorer_hwc_from_tensor(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        if arr.shape[:2] == (1, 3):
            return arr[0].transpose(1, 2, 0).astype(np.float64)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.float64)
    raise ValueError(f"cannot convert scorer value with shape {arr.shape} to HWC")


def realize_q3_delta_lattice_native(
    *,
    camera_frame: np.ndarray,
    base_scorer: torch.Tensor | np.ndarray,
    target_paint_hwc: np.ndarray,
    block_mask: np.ndarray,
    method: str = "cvp",
    dykstra_iterations: int = 8,
    cvp_tap_radius: int = 1,
    max_blocks: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply dk1 private-block realization for q3 projected scorer paint."""

    if method not in {"cvp", "dykstra", "naive"}:
        raise ValueError(f"unknown dk1 method: {method}")
    frame = np.asarray(camera_frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("camera_frame must be uint8 HWC RGB")
    base = scorer_hwc_from_tensor(base_scorer)
    target = np.asarray(target_paint_hwc, dtype=np.float64)
    if target.shape != base.shape:
        raise ValueError(f"target/base shape mismatch: {target.shape} != {base.shape}")
    blocks = np.asarray(block_mask).astype(bool)
    if blocks.shape != (base.shape[0] // 2, base.shape[1] // 2):
        raise ValueError(f"bad block mask shape: {blocks.shape}")

    coords = [(int(by), int(bx)) for by, bx in zip(*np.nonzero(blocks), strict=False)]
    total_blocks = len(coords)
    if max_blocks is not None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be positive when supplied")
        coords = coords[: int(max_blocks)]
    operator = build_default_operator()
    out = frame.copy()
    receipts: list[dict[str, Any]] = []
    aggregate = {
        **q3_block_coverage_payload(
            total_blocks=int(total_blocks),
            realized_blocks=len(coords),
            block_limit=max_blocks,
        ),
        "pose_leakage_sq_sum": 0.0,
        "seg_discrepancy_sum": 0.0,
        "changed_camera_values_sum": 0,
        "exact_declared_scope_count": 0,
        "global_integer_optimum_claim": False,
    }

    for by, bx in coords:
        sy = by * 2
        sx = bx * 2
        target_delta = target[sy : sy + 2, sx : sx + 2] - base[sy : sy + 2, sx : sx + 2]
        geometry = private_block_geometry(operator, sy, sx)
        base_block = extract_private_camera_block(out, geometry)
        results = realize_lattice_native_block(
            target_delta,
            geometry,
            base_block=base_block,
            dykstra_iterations=int(dykstra_iterations),
            cvp_tap_radius=int(cvp_tap_radius),
        )
        result = results[method]
        out = add_private_delta_to_frame(out, geometry, result.camera_delta)
        result_payload = result.to_dict()
        aggregate["pose_leakage_sq_sum"] += float(result.pose_leakage_sq)
        aggregate["seg_discrepancy_sum"] += float(result.seg_discrepancy)
        aggregate["changed_camera_values_sum"] += int(result.changed_camera_values)
        diag = result_payload.get("diagnostics", {})
        if isinstance(diag, Mapping) and bool(diag.get("exact_declared_scope")):
            aggregate["exact_declared_scope_count"] += 1
        if len(receipts) < 16:
            receipts.append(
                {
                    "scorer_row": sy,
                    "scorer_col": sx,
                    "selected_method": method,
                    "result": result_payload,
                    "geometry": {
                        "assumes_uniform_025": False,
                        "denominator": int(geometry.denominator),
                    },
                }
            )

    receipt = {
        "schema": RW1_INSTRUMENT_SCHEMA,
        "instrument": "dk1_lattice_native_q3_realizer",
        "selected_method": method,
        "dk1_cvp_tap_radius": int(cvp_tap_radius),
        "dk1_dykstra_iterations": int(dykstra_iterations),
        "score_claim": False,
        "promotion_eligible": False,
        "aggregate": aggregate,
        "sampled_block_receipts": receipts,
    }
    return out, receipt
