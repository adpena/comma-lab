# SPDX-License-Identifier: MIT
"""Integer-aware near-margin proposals for reopened FD zero-accept probes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.optimization.lattice_native_pose_null_realizer import (
    add_private_delta_to_frame,
    build_default_operator,
    extract_private_camera_block,
    private_block_geometry,
    realize_lattice_native_block,
)

FD_INTEGER_NEAR_MARGIN_SCHEMA = "ddm_rw1_fd_integer_near_margin_proposals.v1"


@dataclass(frozen=True)
class NearMarginSite:
    scorer_row: int
    scorer_col: int
    target_class: int
    current_class: int
    margin_current_minus_target: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "scorer_row": int(self.scorer_row),
            "scorer_col": int(self.scorer_col),
            "target_class": int(self.target_class),
            "current_class": int(self.current_class),
            "margin_current_minus_target": float(self.margin_current_minus_target),
        }


Validator = Callable[[np.ndarray, Mapping[str, Any]], Mapping[str, Any]]


def near_margin_sites(
    *,
    logits_chw: np.ndarray,
    realized_argmax: np.ndarray,
    target_argmax: np.ndarray,
    max_sites: int,
) -> list[NearMarginSite]:
    """Return wrong-label sites sorted by the realized target margin."""

    logits = np.asarray(logits_chw, dtype=np.float64)
    if logits.ndim != 3:
        raise ValueError(f"logits must be CHW, got {logits.shape}")
    current = np.asarray(realized_argmax)
    target = np.asarray(target_argmax)
    if current.shape != target.shape or logits.shape[1:] != current.shape:
        raise ValueError(
            f"shape mismatch logits={logits.shape} current={current.shape} target={target.shape}"
        )
    wrong = current != target
    rows, cols = np.nonzero(wrong)
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    current_classes = current[rows, cols].astype(np.int64)
    target_classes = target[rows, cols].astype(np.int64)
    margins = logits[current_classes, rows, cols] - logits[target_classes, rows, cols]
    order = np.lexsort((cols, rows, margins))
    sites: list[NearMarginSite] = []
    seen_blocks: set[tuple[int, int]] = set()
    for idx in order:
        row = int(rows[idx])
        col = int(cols[idx])
        scorer_row = row - (row % 2)
        scorer_col = col - (col % 2)
        key = (scorer_row, scorer_col)
        if key in seen_blocks:
            continue
        seen_blocks.add(key)
        sites.append(
            NearMarginSite(
                scorer_row=scorer_row,
                scorer_col=scorer_col,
                target_class=int(target_classes[idx]),
                current_class=int(current_classes[idx]),
                margin_current_minus_target=float(margins[idx]),
            )
        )
        if len(sites) >= int(max_sites):
            break
    return sites


class IntegerNearMarginProposalGenerator:
    """Generate FD proposals on the uint8 camera lattice and validate argmax."""

    def __init__(
        self,
        *,
        method: str = "cvp",
        dykstra_iterations: int = 8,
        cvp_tap_radius: int = 1,
    ) -> None:
        if method not in {"cvp", "dykstra", "naive"}:
            raise ValueError(f"bad method: {method}")
        self.method = method
        self.dykstra_iterations = int(dykstra_iterations)
        self.cvp_tap_radius = int(cvp_tap_radius)
        self.operator = build_default_operator()

    def generate(
        self,
        *,
        camera_frame: np.ndarray,
        base_scorer_hwc: np.ndarray,
        target_scorer_hwc: np.ndarray,
        logits_chw: np.ndarray,
        realized_argmax: np.ndarray,
        target_argmax: np.ndarray,
        max_proposals: int,
        validator: Validator,
    ) -> dict[str, Any]:
        frame = np.asarray(camera_frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("camera_frame must be uint8 HWC RGB")
        base = np.asarray(base_scorer_hwc, dtype=np.float64)
        target = np.asarray(target_scorer_hwc, dtype=np.float64)
        if base.shape != target.shape or base.ndim != 3 or base.shape[2] != 3:
            raise ValueError(f"bad scorer RGB shapes: {base.shape} {target.shape}")

        sites = near_margin_sites(
            logits_chw=logits_chw,
            realized_argmax=realized_argmax,
            target_argmax=target_argmax,
            max_sites=max_proposals,
        )
        rows: list[dict[str, Any]] = []
        accepted = 0
        for site in sites:
            sy = site.scorer_row
            sx = site.scorer_col
            geometry = private_block_geometry(self.operator, sy, sx)
            target_delta = target[sy : sy + 2, sx : sx + 2] - base[sy : sy + 2, sx : sx + 2]
            base_block = extract_private_camera_block(frame, geometry)
            results = realize_lattice_native_block(
                target_delta,
                geometry,
                base_block=base_block,
                dykstra_iterations=self.dykstra_iterations,
                cvp_tap_radius=self.cvp_tap_radius,
            )
            result = results[self.method]
            candidate = add_private_delta_to_frame(frame, geometry, result.camera_delta)
            context = {
                "schema": FD_INTEGER_NEAR_MARGIN_SCHEMA,
                "site": site.to_payload(),
                "selected_method": self.method,
                "dk1_result": result.to_dict(),
            }
            validation = dict(validator(candidate, context))
            is_accepted = bool(validation.get("accepted"))
            if is_accepted:
                accepted += 1
            rows.append(
                {
                    **context,
                    "validation": validation,
                    "accepted": is_accepted,
                }
            )
        return {
            "schema": FD_INTEGER_NEAR_MARGIN_SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "generated_on": "uint8 camera lattice via dk1 private-block integer realizer",
            "proposal_order": "ascending realized logit margin current_class-minus-target_class",
            "validation_gate": "realized SegNet argmax must improve in the proposal loop",
            "selected_method": self.method,
            "dk1_cvp_tap_radius": self.cvp_tap_radius,
            "dk1_dykstra_iterations": self.dykstra_iterations,
            "proposals": rows,
            "n_proposals": len(rows),
            "n_accepted": int(accepted),
            "accept_rate": accepted / len(rows) if rows else None,
        }
