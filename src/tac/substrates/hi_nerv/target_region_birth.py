# SPDX-License-Identifier: MIT
"""Torch-free worst-target-region selection for HiNeRV hard class birth.

The contest SegNet term is an argmax-disagreement *rate*: a class can gain
probability mass everywhere while winning the argmax nowhere, so per-class
aggregate losses can improve while ``target_min_ratio`` stays ``0.0`` and the
receiver argmax probe still fails.  Birth only counts when a specific
connected component of wrong pixels flips its argmax on the receiver surface.

This module prices every connected wrong-region of every target class in
exact contest score units (``100 * unsolved_pixels / total_scored_pixels``),
selects the worst one deterministically, and builds the receipts a scoped
birth actuator must emit.  It is numpy-only on purpose: the same selector and
receipt math serve the live MLX forward, fake-quant forward, archive
parse-back, and inflate replay surfaces without backend drift.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

TARGET_REGION_DEBT_SCHEMA = "hi_nerv_target_region_debt.v1"
TARGET_REGION_BIRTH_RECEIPT_SCHEMA = "hi_nerv_target_region_birth_receipt.v1"

# Archive-charged tensors the birth actuator may update.  This mirrors the
# live-SegNet-scoped bootstrap list in ``mlx_renderer`` (late surfaces with
# local spatial leverage); early coarse/stem tensors move many pairs at once
# and can break PoseNet geometry, so they stay frozen during birth.
ALLOWED_BIRTH_UPDATE_EXACT: tuple[str, ...] = ("latents_fine",)
ALLOWED_BIRTH_UPDATE_PREFIXES: tuple[str, ...] = (
    "latents_fine.",
    "feature_grids.",
    "fine_injector.",
    "head_rgb_1.",
)


def allowed_birth_update_name(name: Any) -> bool:
    """Return True when a flattened parameter name is birth-updatable."""

    flat = (
        ".".join(str(part) for part in name)
        if isinstance(name, (tuple, list))
        else str(name)
    )
    if flat in ALLOWED_BIRTH_UPDATE_EXACT:
        return True
    return any(flat.startswith(prefix) for prefix in ALLOWED_BIRTH_UPDATE_PREFIXES)


@dataclass(frozen=True)
class TargetRegionDebt:
    """One connected target-class region priced in exact contest score units."""

    batch_index: int
    class_index: int
    region_label: int
    region_pixel_count: int
    region_unsolved_pixel_count: int
    region_hard_ratio: float
    frame_pixel_count: int
    total_scored_pixels: int
    score_debt_units: float
    frame_fraction: float
    bbox_y0: int
    bbox_y1: int
    bbox_x0: int
    bbox_x1: int

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = TARGET_REGION_DEBT_SCHEMA
        payload.update(PROXY_FALSE_AUTHORITY_FIELDS)
        return payload


def _validate_label_pair(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target_labels)
    candidate = np.asarray(candidate_argmax)
    if target.ndim == 4 and target.shape[-1] == 1:
        target = target[..., 0]
    if candidate.ndim == 4 and candidate.shape[-1] == 1:
        candidate = candidate[..., 0]
    if target.ndim != 3 or candidate.ndim != 3:
        raise ValueError(
            "target_labels and candidate_argmax must be BHW (or BHW1); got "
            f"target={target.shape} candidate={candidate.shape}"
        )
    if target.shape != candidate.shape:
        raise ValueError(
            "target_labels and candidate_argmax shapes must match; got "
            f"target={target.shape} candidate={candidate.shape}"
        )
    if target.size == 0:
        raise ValueError("target_labels must be non-empty")
    return target.astype(np.int64, copy=False), candidate.astype(np.int64, copy=False)


def find_target_region_debts(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
    *,
    min_region_pixels: int = 1,
) -> list[TargetRegionDebt]:
    """Price every 4-connected target-class component in exact score units.

    ``score_debt_units = 100 * unsolved_pixels / total_scored_pixels`` where
    ``total_scored_pixels = B * H * W`` for the supplied batch.  The
    normalizer is batch-local and recorded on every row so a 4-pair smoke
    cannot silently masquerade as full-video score units.
    """

    if min_region_pixels < 1:
        raise ValueError(f"min_region_pixels must be >= 1; got {min_region_pixels}")
    target, candidate = _validate_label_pair(target_labels, candidate_argmax)
    batch, height, width = target.shape
    total_scored = int(batch * height * width)
    frame_pixels = int(height * width)
    rows: list[TargetRegionDebt] = []
    for batch_index in range(batch):
        frame_target = target[batch_index]
        frame_candidate = candidate[batch_index]
        for class_index in np.unique(frame_target).tolist():
            class_mask = frame_target == class_index
            # scipy default 2-D structure is the 4-connected cross.
            labeled, component_count = ndimage.label(class_mask)
            if component_count == 0:
                continue
            unsolved_mask = class_mask & (frame_candidate != class_index)
            for region_label in range(1, component_count + 1):
                region_mask = labeled == region_label
                region_pixels = int(np.count_nonzero(region_mask))
                if region_pixels < min_region_pixels:
                    continue
                unsolved_pixels = int(np.count_nonzero(region_mask & unsolved_mask))
                ys, xs = np.nonzero(region_mask)
                rows.append(
                    TargetRegionDebt(
                        batch_index=int(batch_index),
                        class_index=int(class_index),
                        region_label=int(region_label),
                        region_pixel_count=region_pixels,
                        region_unsolved_pixel_count=unsolved_pixels,
                        region_hard_ratio=float(
                            (region_pixels - unsolved_pixels) / region_pixels
                        ),
                        frame_pixel_count=frame_pixels,
                        total_scored_pixels=total_scored,
                        score_debt_units=float(100.0 * unsolved_pixels / total_scored),
                        frame_fraction=float(region_pixels / frame_pixels),
                        bbox_y0=int(ys.min()),
                        bbox_y1=int(ys.max()) + 1,
                        bbox_x0=int(xs.min()),
                        bbox_x1=int(xs.max()) + 1,
                    )
                )
    return rows


def select_worst_target_region(
    debts: Sequence[TargetRegionDebt],
) -> TargetRegionDebt:
    """Return the highest-score-debt region with a deterministic tie-break."""

    if not debts:
        raise ValueError("at least one target-region debt row is required")
    for row in debts:
        if not math.isfinite(row.score_debt_units) or row.score_debt_units < 0.0:
            raise ValueError(
                "score_debt_units must be finite and non-negative; got "
                f"{row.score_debt_units} for batch={row.batch_index} "
                f"class={row.class_index} region={row.region_label}"
            )
    return min(
        debts,
        key=lambda row: (
            -row.score_debt_units,
            row.batch_index,
            row.class_index,
            row.region_label,
        ),
    )


def select_worst_target_region_with_mask(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
    *,
    min_region_pixels: int = 1,
) -> tuple[TargetRegionDebt, np.ndarray]:
    """Return the worst region row plus its full-batch BHW float32 mask."""

    target, _ = _validate_label_pair(target_labels, candidate_argmax)
    debts = find_target_region_debts(
        target_labels,
        candidate_argmax,
        min_region_pixels=min_region_pixels,
    )
    worst = select_worst_target_region(debts)
    frame_target = target[worst.batch_index]
    labeled, _count = ndimage.label(frame_target == worst.class_index)
    mask = np.zeros(target.shape, dtype=np.float32)
    mask[worst.batch_index] = (labeled == worst.region_label).astype(np.float32)
    if int(np.count_nonzero(mask)) != worst.region_pixel_count:
        raise RuntimeError(
            "worst-region mask reconstruction drifted from the priced row; "
            f"mask={int(np.count_nonzero(mask))} row={worst.region_pixel_count}"
        )
    return worst, mask


def region_margin_stats(
    logits_bhwc: np.ndarray,
    region_mask_bhw: np.ndarray,
    class_index: int,
) -> dict[str, float]:
    """Return raw frontier-margin stats for one class within one region.

    The margin convention is PR95's ``impostor - class`` (no floor, no relu):
    negative at a pixel means the class wins the argmax there.  ``p50`` is the
    median over region pixels — the quantity whose *delta* the crux-trace
    consumer ingests as ``worst_region_margin_p50_delta``.
    """

    logits = np.asarray(logits_bhwc, dtype=np.float64)
    mask = np.asarray(region_mask_bhw)
    if logits.ndim != 4:
        raise ValueError(f"logits must be BHWC; got shape {logits.shape}")
    if mask.shape != logits.shape[:3]:
        raise ValueError(
            "region mask BHW must match logits BHW; got "
            f"mask={mask.shape} logits={logits.shape[:3]}"
        )
    class_count = int(logits.shape[-1])
    if not 0 <= int(class_index) < class_count:
        raise ValueError(
            f"class_index {class_index} outside logits classes {class_count}"
        )
    flat_mask = mask.reshape(-1) > 0.0
    region_pixels = int(np.count_nonzero(flat_mask))
    if region_pixels == 0:
        raise ValueError("region mask selects zero pixels")
    flat_logits = logits.reshape(-1, class_count)[flat_mask]
    class_logit = flat_logits[:, int(class_index)]
    impostor = np.copy(flat_logits)
    impostor[:, int(class_index)] = -np.inf
    impostor_logit = impostor.max(axis=1)
    margin = impostor_logit - class_logit
    hard_won = int(np.count_nonzero(margin < 0.0))
    return {
        "region_pixel_count": float(region_pixels),
        "region_hard_ratio": float(hard_won / region_pixels),
        "region_hard_won_pixels": float(hard_won),
        "margin_min": float(np.min(margin)),
        "margin_p50": float(np.median(margin)),
        "margin_mean": float(np.mean(margin)),
    }


def build_target_region_birth_receipt(
    *,
    debt: TargetRegionDebt,
    before_margin_stats: Mapping[str, float],
    after_margin_stats: Mapping[str, float],
    receiver_uint8_changed_pixels_region: int,
    receiver_uint8_delta_abs_max: float,
    receiver_float_rgb_delta_linf: float,
    argmax_flipped_pixels_region: int,
    accepted_step_count: int,
    rejected_step_count: int,
    blockers: Sequence[str],
    grad_norm_by_group: Mapping[str, float],
    update_norm_by_group: Mapping[str, float],
    updated_parameter_names: Sequence[str],
    pose_guard: Mapping[str, Any],
    runtime_sidecar_bytes: int = 0,
) -> dict[str, Any]:
    """Assemble the actuation receipt with crux-trace-compatible keys.

    The ``receiver_surface_*`` aliases match the trace contract consumed by
    ``tools/trace_nerv_crux.py`` rows so accepted birth updates populate the
    receiver-surface evidence the witness-readiness DAG requires, without the
    consumer needing producer-specific adapters.
    """

    margin_p50_delta = float(
        after_margin_stats["margin_p50"] - before_margin_stats["margin_p50"]
    )
    disallowed = [
        name for name in updated_parameter_names if not allowed_birth_update_name(name)
    ]
    if disallowed:
        raise ValueError(
            f"updated parameter names escape the birth scope: {sorted(disallowed)}"
        )
    receipt: dict[str, Any] = {
        "schema": TARGET_REGION_BIRTH_RECEIPT_SCHEMA,
        "actuator_id": "hinerv_target_region_birth",
        "family": "hinerv",
        "frame_scope": "frame1_seg_pose_joint",
        "pair_index": int(debt.batch_index),
        "worst_region": debt.as_dict(),
        "before_region_margin_stats": dict(before_margin_stats),
        "after_region_margin_stats": dict(after_margin_stats),
        "before_region_hard_ratio": float(before_margin_stats["region_hard_ratio"]),
        "after_region_hard_ratio": float(after_margin_stats["region_hard_ratio"]),
        "worst_region_margin_p50_delta": margin_p50_delta,
        "receiver_surface_worst_region_margin_p50_delta": margin_p50_delta,
        "receiver_surface_uint8_changed_pixels": int(
            receiver_uint8_changed_pixels_region
        ),
        "receiver_surface_uint8_delta_abs_max": float(receiver_uint8_delta_abs_max),
        "receiver_surface_float_rgb_delta_linf": float(receiver_float_rgb_delta_linf),
        "receiver_surface_argmax_flipped_pixels": int(argmax_flipped_pixels_region),
        "accepted_step_count": int(accepted_step_count),
        "rejected_step_count": int(rejected_step_count),
        "blockers": [str(item) for item in blockers],
        "grad_norm_by_group": {
            str(name): float(value) for name, value in grad_norm_by_group.items()
        },
        "update_norm_by_group": {
            str(name): float(value) for name, value in update_norm_by_group.items()
        },
        "updated_parameter_names": sorted(str(name) for name in updated_parameter_names),
        "allowed_update_prefixes": list(
            ALLOWED_BIRTH_UPDATE_EXACT + ALLOWED_BIRTH_UPDATE_PREFIXES
        ),
        "pose_guard": dict(pose_guard),
        "runtime_sidecar_bytes": int(runtime_sidecar_bytes),
        "human_visual_fidelity_objective": False,
    }
    receipt.update(PROXY_FALSE_AUTHORITY_FIELDS)
    return receipt
