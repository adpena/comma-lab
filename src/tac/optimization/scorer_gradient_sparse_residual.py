# SPDX-License-Identifier: MIT
"""Scorer-gradient sparse residual targeting for charged raw-output probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tac.optimization.inflate_postprocess_surface import RawVideoShape
from tac.optimization.sparse_residual_oracle import (
    SparseResidualPlan,
    build_sparse_residual_plan_from_global_values,
)

GradientComponent = Literal["pose", "seg", "combined"]


@dataclass(frozen=True)
class ScorerGradientSparseConfig:
    top_k_pixels: int
    max_abs_delta: int = 1
    component: GradientComponent = "pose"
    quantize_bits: int = 8
    compression: str = "zlib"
    rate_cap_bytes: int | None = None

    def validate(self) -> None:
        if self.top_k_pixels <= 0:
            raise ValueError("top_k_pixels must be positive")
        if self.max_abs_delta <= 0:
            raise ValueError("max_abs_delta must be positive")
        if self.component not in {"pose", "seg", "combined"}:
            raise ValueError(f"unsupported component={self.component!r}")
        if self.quantize_bits not in {4, 8, 16}:
            raise ValueError("quantize_bits must be one of 4, 8, 16")
        if self.compression not in {"zlib", "none"}:
            raise ValueError("compression must be 'zlib' or 'none'")
        if self.rate_cap_bytes is not None and self.rate_cap_bytes <= 0:
            raise ValueError("rate_cap_bytes must be positive when supplied")

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_k_pixels": self.top_k_pixels,
            "max_abs_delta": self.max_abs_delta,
            "component": self.component,
            "quantize_bits": self.quantize_bits,
            "compression": self.compression,
            "rate_cap_bytes": self.rate_cap_bytes,
        }


@dataclass(frozen=True)
class GradientAlignedSelection:
    indices: np.ndarray
    values: np.ndarray
    gains: np.ndarray
    candidate_count: int
    rejected_non_descent_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_selected": int(self.indices.size),
            "candidate_count": self.candidate_count,
            "rejected_non_descent_count": self.rejected_non_descent_count,
            "gain_sum": float(self.gains.sum()) if self.gains.size else 0.0,
            "gain_max": float(self.gains.max()) if self.gains.size else 0.0,
            "gain_mean": float(self.gains.mean()) if self.gains.size else 0.0,
        }


@dataclass(frozen=True)
class BudgetedGradientSelection:
    indices: np.ndarray
    values: np.ndarray
    gains: np.ndarray
    utility: np.ndarray
    byte_costs: np.ndarray
    candidate_count: int
    rejected_non_descent_count: int
    rejected_by_saliency_count: int
    budget_used: float
    budget_limit: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_selected": int(self.indices.size),
            "candidate_count": self.candidate_count,
            "rejected_non_descent_count": self.rejected_non_descent_count,
            "rejected_by_saliency_count": self.rejected_by_saliency_count,
            "gain_sum": float(self.gains.sum()) if self.gains.size else 0.0,
            "gain_max": float(self.gains.max()) if self.gains.size else 0.0,
            "utility_sum": float(self.utility.sum()) if self.utility.size else 0.0,
            "utility_max": float(self.utility.max()) if self.utility.size else 0.0,
            "budget_used": float(self.budget_used),
            "budget_limit": None if self.budget_limit is None else float(self.budget_limit),
        }


def _validate_gradient_frame_space(
    *,
    gradient: np.ndarray,
    shape: RawVideoShape,
    frame_indices: list[int],
) -> None:
    if gradient.shape[1] != shape.height or gradient.shape[2] != shape.width:
        raise ValueError(
            "gradient frame-space shape must match RawVideoShape height/width: "
            f"{gradient.shape[1:3]} vs {(shape.height, shape.width)}"
        )
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= shape.frames:
            raise ValueError(f"frame_index out of range for shape.frames={shape.frames}: {frame_index}")


def select_gradient_aligned_residuals(
    *,
    gradient: np.ndarray,
    residual: np.ndarray,
    shape: RawVideoShape,
    frame_indices: list[int],
    top_k_pixels: int,
    max_abs_delta: int,
) -> GradientAlignedSelection:
    """Select residual pixels whose GT-bounded delta descends scorer loss."""

    if top_k_pixels <= 0:
        raise ValueError("top_k_pixels must be positive")
    if max_abs_delta <= 0:
        raise ValueError("max_abs_delta must be positive")
    grad = np.asarray(gradient, dtype=np.float32)
    raw_resid = np.asarray(residual, dtype=np.int16)
    if grad.shape != raw_resid.shape:
        raise ValueError(f"gradient/residual shape mismatch: {grad.shape} vs {raw_resid.shape}")
    if grad.ndim != 4 or grad.shape[-1] != 3:
        raise ValueError(f"expected gradient shape (N,H,W,3), got {grad.shape}")
    if len(frame_indices) != grad.shape[0]:
        raise ValueError("frame_indices length must match gradient frame dimension")
    if shape.channels != 3:
        raise ValueError("shape.channels must be 3")
    _validate_gradient_frame_space(gradient=grad, shape=shape, frame_indices=frame_indices)

    clipped = np.clip(raw_resid, -max_abs_delta, max_abs_delta).astype(np.int16)
    dot = (grad * clipped.astype(np.float32)).sum(axis=-1)
    gains = -dot.reshape(-1)
    flat_values = clipped.reshape(-1, 3)
    candidate_mask = np.any(flat_values != 0, axis=-1)
    descent_mask = candidate_mask & (gains > 0.0)
    candidate_count = int(np.count_nonzero(candidate_mask))
    rejected = int(np.count_nonzero(candidate_mask & ~descent_mask))
    if not np.any(descent_mask):
        return GradientAlignedSelection(
            indices=np.asarray([], dtype=np.uint32),
            values=np.zeros((0, 3), dtype=np.int16),
            gains=np.asarray([], dtype=np.float32),
            candidate_count=candidate_count,
            rejected_non_descent_count=rejected,
        )

    eligible = np.flatnonzero(descent_mask)
    local_k = min(top_k_pixels, int(eligible.size))
    eligible_gains = gains[eligible]
    chosen_local = np.argpartition(eligible_gains, -local_k)[-local_k:]
    chosen = eligible[chosen_local]
    order = np.argsort(-gains[chosen], kind="stable")
    chosen = chosen[order]

    pixels_per_frame = shape.height * shape.width
    local_frame_slot = chosen // pixels_per_frame
    local_pixel_offset = chosen % pixels_per_frame
    frame_arr = np.asarray(frame_indices, dtype=np.uint64)
    global_indices = frame_arr[local_frame_slot] * np.uint64(pixels_per_frame) + local_pixel_offset.astype(np.uint64)

    return GradientAlignedSelection(
        indices=global_indices.astype(np.uint32),
        values=flat_values[chosen].astype(np.int16),
        gains=gains[chosen].astype(np.float32),
        candidate_count=candidate_count,
        rejected_non_descent_count=rejected,
    )


def select_budgeted_gradient_residuals(
    *,
    gradient: np.ndarray,
    residual: np.ndarray,
    shape: RawVideoShape,
    frame_indices: list[int],
    top_k_pixels: int,
    max_abs_delta: int,
    saliency_mask: np.ndarray | None = None,
    byte_costs: np.ndarray | None = None,
    budget_limit: float | None = None,
    saliency_floor: float = 0.0,
) -> BudgetedGradientSelection:
    """Select descent residuals by score-impact utility per byte cost.

    The gradient source may be the full scorer, a Hinton-distilled surrogate,
    or an empirical master-gradient map. This selector only requires that a
    positive ``-dot(gradient, residual_delta)`` means lower scorer objective.
    """

    if top_k_pixels <= 0:
        raise ValueError("top_k_pixels must be positive")
    if max_abs_delta <= 0:
        raise ValueError("max_abs_delta must be positive")
    if budget_limit is not None and budget_limit <= 0:
        raise ValueError("budget_limit must be positive when supplied")
    if saliency_floor < 0.0:
        raise ValueError("saliency_floor must be non-negative")

    grad = np.asarray(gradient, dtype=np.float32)
    raw_resid = np.asarray(residual, dtype=np.int16)
    if grad.shape != raw_resid.shape:
        raise ValueError(f"gradient/residual shape mismatch: {grad.shape} vs {raw_resid.shape}")
    if grad.ndim != 4 or grad.shape[-1] != 3:
        raise ValueError(f"expected gradient shape (N,H,W,3), got {grad.shape}")
    if len(frame_indices) != grad.shape[0]:
        raise ValueError("frame_indices length must match gradient frame dimension")
    if shape.channels != 3:
        raise ValueError("shape.channels must be 3")
    _validate_gradient_frame_space(gradient=grad, shape=shape, frame_indices=frame_indices)

    clipped = np.clip(raw_resid, -max_abs_delta, max_abs_delta).astype(np.int16)
    flat_values = clipped.reshape(-1, 3)
    candidate_mask = np.any(flat_values != 0, axis=-1)
    dot = (grad * clipped.astype(np.float32)).sum(axis=-1)
    gains = -dot.reshape(-1)
    descent_mask = candidate_mask & (gains > 0.0)

    if saliency_mask is None:
        saliency = np.ones(grad.shape[:-1], dtype=np.float32)
    else:
        saliency = np.asarray(saliency_mask, dtype=np.float32)
        if saliency.shape != grad.shape[:-1]:
            raise ValueError(f"saliency_mask shape mismatch: {saliency.shape} vs {grad.shape[:-1]}")
    flat_saliency = saliency.reshape(-1)
    saliency_ok = flat_saliency > saliency_floor

    if byte_costs is None:
        costs = np.ones(grad.shape[:-1], dtype=np.float32)
    else:
        costs = np.asarray(byte_costs, dtype=np.float32)
        if costs.shape != grad.shape[:-1]:
            raise ValueError(f"byte_costs shape mismatch: {costs.shape} vs {grad.shape[:-1]}")
        if np.any(costs <= 0.0):
            raise ValueError("byte_costs must be strictly positive")
    flat_costs = costs.reshape(-1)

    eligible_mask = descent_mask & saliency_ok
    candidate_count = int(np.count_nonzero(candidate_mask))
    rejected_non_descent = int(np.count_nonzero(candidate_mask & ~descent_mask))
    rejected_by_saliency = int(np.count_nonzero(descent_mask & ~saliency_ok))
    if not np.any(eligible_mask):
        return BudgetedGradientSelection(
            indices=np.asarray([], dtype=np.uint32),
            values=np.zeros((0, 3), dtype=np.int16),
            gains=np.asarray([], dtype=np.float32),
            utility=np.asarray([], dtype=np.float32),
            byte_costs=np.asarray([], dtype=np.float32),
            candidate_count=candidate_count,
            rejected_non_descent_count=rejected_non_descent,
            rejected_by_saliency_count=rejected_by_saliency,
            budget_used=0.0,
            budget_limit=budget_limit,
        )

    utility = gains * flat_saliency / flat_costs
    eligible = np.flatnonzero(eligible_mask)
    order = eligible[np.argsort(-utility[eligible], kind="stable")]
    chosen_parts: list[int] = []
    budget_used = 0.0
    for idx in order:
        if len(chosen_parts) >= top_k_pixels:
            break
        cost = float(flat_costs[idx])
        if budget_limit is not None and budget_used + cost > budget_limit:
            continue
        chosen_parts.append(int(idx))
        budget_used += cost
    if not chosen_parts:
        chosen = np.asarray([], dtype=np.int64)
    else:
        chosen = np.asarray(chosen_parts, dtype=np.int64)

    pixels_per_frame = shape.height * shape.width
    if chosen.size:
        local_frame_slot = chosen // pixels_per_frame
        local_pixel_offset = chosen % pixels_per_frame
        frame_arr = np.asarray(frame_indices, dtype=np.uint64)
        global_indices = frame_arr[local_frame_slot] * np.uint64(pixels_per_frame) + local_pixel_offset.astype(np.uint64)
    else:
        global_indices = np.asarray([], dtype=np.uint64)

    return BudgetedGradientSelection(
        indices=global_indices.astype(np.uint32),
        values=flat_values[chosen].astype(np.int16),
        gains=gains[chosen].astype(np.float32),
        utility=utility[chosen].astype(np.float32),
        byte_costs=flat_costs[chosen].astype(np.float32),
        candidate_count=candidate_count,
        rejected_non_descent_count=rejected_non_descent,
        rejected_by_saliency_count=rejected_by_saliency,
        budget_used=budget_used,
        budget_limit=budget_limit,
    )


def build_plan_from_gradient_selection(
    *,
    selection: GradientAlignedSelection,
    shape: RawVideoShape,
    config: ScorerGradientSparseConfig,
) -> SparseResidualPlan:
    config.validate()
    return build_sparse_residual_plan_from_global_values(
        shape=shape,
        indices=selection.indices,
        values=selection.values,
        quantize_bits=config.quantize_bits,
        compression=config.compression,
        rate_cap_bytes=config.rate_cap_bytes,
        gains=selection.gains,
    )


def compute_pair_scorer_gradient(
    *,
    baseline_pair_hwc: np.ndarray,
    target_pair_hwc: np.ndarray,
    posenet: Any,
    segnet: Any,
    device: str,
    component: GradientComponent,
    seg_ce_weight: float = 0.05,
    pose_eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return d(objective)/d(candidate pixels) for one two-frame pair."""

    import torch
    import torch.nn.functional as F

    if component not in {"pose", "seg", "combined"}:
        raise ValueError(f"unsupported component={component!r}")
    cand_np = np.asarray(baseline_pair_hwc, dtype=np.float32)
    gt_np = np.asarray(target_pair_hwc, dtype=np.float32)
    if cand_np.shape != gt_np.shape or cand_np.ndim != 4 or cand_np.shape[0] != 2 or cand_np.shape[-1] != 3:
        raise ValueError(f"expected pair arrays with shape (2,H,W,3), got {cand_np.shape} and {gt_np.shape}")

    torch_device = torch.device(device)
    cand = torch.from_numpy(cand_np).to(torch_device).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    gt = torch.from_numpy(gt_np).to(torch_device).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    cand = cand.detach().clone().requires_grad_(True)
    gt = gt.detach()

    pose_dist = torch.zeros((), device=torch_device)
    pose_term = torch.zeros((), device=torch_device)
    if component in {"pose", "combined"}:
        pose_pred = posenet(posenet.preprocess_input(cand))
        with torch.no_grad():
            pose_gt = posenet(posenet.preprocess_input(gt))
        pose_dim = pose_pred["pose"].shape[-1] // 2
        pose_dist = (
            pose_pred["pose"][..., :pose_dim] - pose_gt["pose"][..., :pose_dim]
        ).pow(2).mean()
        pose_term = torch.sqrt(10.0 * pose_dist.clamp_min(0.0) + pose_eps)

    seg_ce = torch.zeros((), device=torch_device)
    if component in {"seg", "combined"}:
        seg_logits = segnet(segnet.preprocess_input(cand))
        with torch.no_grad():
            seg_target = segnet(segnet.preprocess_input(gt)).argmax(dim=1)
        seg_ce = F.cross_entropy(seg_logits, seg_target, reduction="mean")

    if component == "pose":
        objective = pose_term
    elif component == "seg":
        objective = seg_ce
    else:
        objective = pose_term + float(seg_ce_weight) * seg_ce
    objective.backward()

    grad = cand.grad.detach().squeeze(0).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    metrics = {
        "objective": float(objective.detach().cpu().item()),
        "pose_dist": float(pose_dist.detach().cpu().item()),
        "pose_term": float(pose_term.detach().cpu().item()),
        "seg_cross_entropy": float(seg_ce.detach().cpu().item()),
        "seg_ce_weight": float(seg_ce_weight),
        "grad_abs_max": float(np.abs(grad).max()) if grad.size else 0.0,
        "grad_abs_mean": float(np.abs(grad).mean()) if grad.size else 0.0,
    }
    return grad, metrics


def compute_pair_component_distortions(
    *,
    candidate_pair_hwc: np.ndarray,
    target_pair_hwc: np.ndarray,
    posenet: Any,
    segnet: Any,
    device: str,
) -> dict[str, float]:
    """Compute official-style PoseNet/SegNet components for one raw pair."""

    import torch

    cand_np = np.asarray(candidate_pair_hwc, dtype=np.float32)
    gt_np = np.asarray(target_pair_hwc, dtype=np.float32)
    if cand_np.shape != gt_np.shape or cand_np.ndim != 4 or cand_np.shape[0] != 2 or cand_np.shape[-1] != 3:
        raise ValueError(f"expected pair arrays with shape (2,H,W,3), got {cand_np.shape} and {gt_np.shape}")

    torch_device = torch.device(device)
    cand = torch.from_numpy(cand_np).to(torch_device).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    gt = torch.from_numpy(gt_np).to(torch_device).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    with torch.inference_mode():
        pose_pred = posenet(posenet.preprocess_input(cand))
        pose_gt = posenet(posenet.preprocess_input(gt))
        pose_dim = pose_pred["pose"].shape[-1] // 2
        pose_dist = (
            pose_pred["pose"][..., :pose_dim] - pose_gt["pose"][..., :pose_dim]
        ).pow(2).mean()
        seg_logits_pred = segnet(segnet.preprocess_input(cand))
        seg_logits_gt = segnet(segnet.preprocess_input(gt))
        seg_dist = (seg_logits_pred.argmax(dim=1) != seg_logits_gt.argmax(dim=1)).float().mean()
    return {
        "pose_dist": float(pose_dist.detach().cpu().item()),
        "seg_dist": float(seg_dist.detach().cpu().item()),
    }


def pair_component_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Return signed after-minus-before deltas for one-pair components."""

    return {
        "pose_dist_delta": float(after["pose_dist"]) - float(before["pose_dist"]),
        "seg_dist_delta": float(after["seg_dist"]) - float(before["seg_dist"]),
    }


def local_pair_eval_worse_or_null(delta: dict[str, float], *, eps: float = 0.0) -> bool:
    """True when neither axis improves and at least one axis worsens."""

    pose = float(delta["pose_dist_delta"])
    seg = float(delta["seg_dist_delta"])
    improves = pose < -eps or seg < -eps
    worsens = pose > eps or seg > eps
    return (not improves) and worsens
