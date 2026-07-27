# SPDX-License-Identifier: MIT
"""Authority-separated exact coarse costates for the G90 V2 population atlas."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.optimization.taskspace_projected_population_costates_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_BATCH_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    BatchPopulationCostatesV1,
    PopulationScorePointV1,
    ProjectedPopulationCostateError,
)

CHANNELS = 3


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _camera_batch(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.uint8
        or array.ndim != 5
        or not 1 <= array.shape[0] <= MAX_BATCH_PAIRS
        or array.shape[1:] != (2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
    ):
        raise ProjectedPopulationCostateError(
            f"{label} must be exact uint8 [B,2,{CAMERA_HEIGHT},{CAMERA_WIDTH},3], B<=16"
        )
    return np.ascontiguousarray(array).copy()


def _cells(value: np.ndarray, *, batch: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != (batch, SCORER_HEIGHT, SCORER_WIDTH):
        raise ProjectedPopulationCostateError(f"{label} must be exact uint8 [B,{SCORER_HEIGHT},{SCORER_WIDTH}]")
    if array.size and int(array.max()) >= 5:
        raise ProjectedPopulationCostateError(f"{label} escaped the frozen five-class head")
    return np.ascontiguousarray(array).copy()


@dataclass(frozen=True, slots=True)
class DifferentiableAuthorityDriftV2:
    """Observed difference between the scorer authority and autograd surrogate."""

    axis: str
    expected_cells_sha256: str
    authority_cells_sha256: str
    differentiable_cells_sha256: str
    mismatch_cell_count: int
    mismatch_pair_ids: tuple[int, ...]
    minimum_top_two_margin_at_drift: float | None

    def __post_init__(self) -> None:
        if self.axis not in {"current", "target"}:
            raise ProjectedPopulationCostateError("unknown authority-drift axis")
        if self.mismatch_cell_count < 0:
            raise ProjectedPopulationCostateError("negative authority-drift count")
        if (self.mismatch_cell_count == 0) != (self.minimum_top_two_margin_at_drift is None):
            raise ProjectedPopulationCostateError("authority-drift margin presence differs from mismatch count")

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "expected_cells_sha256": self.expected_cells_sha256,
            "authority_cells_sha256": self.authority_cells_sha256,
            "differentiable_cells_sha256": self.differentiable_cells_sha256,
            "mismatch_cell_count": self.mismatch_cell_count,
            "mismatch_pair_ids": list(self.mismatch_pair_ids),
            "minimum_top_two_margin_at_drift": self.minimum_top_two_margin_at_drift,
        }


@dataclass(frozen=True, slots=True)
class PoseDifferentiableAuthorityDriftV2:
    """Pose-output drift between inference authority and the gradient surface."""

    authority_current_pose6_sha256: str
    differentiable_current_pose6_sha256: str
    authority_target_pose6_sha256: str
    differentiable_target_pose6_sha256: str
    maximum_abs_current_delta: float
    maximum_abs_target_delta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_current_pose6_sha256": self.authority_current_pose6_sha256,
            "differentiable_current_pose6_sha256": (self.differentiable_current_pose6_sha256),
            "authority_target_pose6_sha256": self.authority_target_pose6_sha256,
            "differentiable_target_pose6_sha256": (self.differentiable_target_pose6_sha256),
            "maximum_abs_current_delta": self.maximum_abs_current_delta,
            "maximum_abs_target_delta": self.maximum_abs_target_delta,
        }


@dataclass(frozen=True, slots=True)
class ExactCoarseBatchCostatesV2:
    """Dense ephemeral costates plus durable authority-drift annotations."""

    costates: BatchPopulationCostatesV1
    current_drift: DifferentiableAuthorityDriftV2
    target_drift: DifferentiableAuthorityDriftV2
    pose_drift: PoseDifferentiableAuthorityDriftV2

    def drift_dict(self) -> dict[str, object]:
        return {
            "current": self.current_drift.to_dict(),
            "target": self.target_drift.to_dict(),
            "pose": self.pose_drift.to_dict(),
            "authority_cells_drive_exact_replay": True,
            "authority_pose_targets_and_base_mse_drive_exact_replay": True,
            "differentiable_argmax_has_no_authority": True,
        }


def _drift(
    *,
    axis: str,
    pair_ids: tuple[int, ...],
    expected: np.ndarray,
    authority: np.ndarray,
    differentiable: np.ndarray,
    logits: Any,
) -> DifferentiableAuthorityDriftV2:
    import torch

    if not np.array_equal(authority, expected):
        mismatch = authority != expected
        raise ProjectedPopulationCostateError(
            f"{axis} scorer-authority inference cells differ from expected custody",
            context={
                "failing_pair_range": [pair_ids[0], pair_ids[-1] + 1],
                "mismatch_cell_count": int(np.count_nonzero(mismatch)),
                "authority_cells_sha256": _sha256_array(authority),
                "expected_cells_sha256": _sha256_array(expected),
            },
        )
    mismatch = differentiable != authority
    mismatch_counts = mismatch.reshape(len(pair_ids), -1).sum(axis=1)
    mismatch_pair_ids = tuple(
        pair_id for pair_id, count in zip(pair_ids, mismatch_counts, strict=True) if int(count) != 0
    )
    minimum_margin: float | None = None
    if np.any(mismatch):
        with torch.no_grad():
            top_two = torch.topk(logits, k=2, dim=1).values
            margins = (top_two[:, 0] - top_two[:, 1]).detach().cpu().numpy()
        minimum_margin = float(np.min(margins[mismatch]))
    return DifferentiableAuthorityDriftV2(
        axis=axis,
        expected_cells_sha256=_sha256_array(expected),
        authority_cells_sha256=_sha256_array(authority),
        differentiable_cells_sha256=_sha256_array(differentiable),
        mismatch_cell_count=int(np.count_nonzero(mismatch)),
        mismatch_pair_ids=mismatch_pair_ids,
        minimum_top_two_margin_at_drift=minimum_margin,
    )


def compute_batch_exact_coarse_costates_v2(
    *,
    candidate_pairs_hwc: np.ndarray,
    target_pairs_hwc: np.ndarray,
    expected_target_cells: np.ndarray,
    expected_current_cells: np.ndarray,
    authority_target_cells: np.ndarray,
    authority_current_cells: np.ndarray,
    pair_ids: tuple[int, ...],
    posenet: Any,
    segnet: Any,
    device: str,
    score_point: PopulationScorePointV1,
) -> ExactCoarseBatchCostatesV2:
    """Differentiate at one pair state while keeping inference cells authoritative."""

    import torch

    candidate_np = _camera_batch(candidate_pairs_hwc, label="candidate pairs")
    target_np = _camera_batch(target_pairs_hwc, label="target pairs")
    if candidate_np.shape != target_np.shape:
        raise ProjectedPopulationCostateError("candidate and target camera batches differ")
    batch = candidate_np.shape[0]
    if len(pair_ids) != batch:
        raise ProjectedPopulationCostateError("pair IDs do not match scorer batch")
    expected_target_np = _cells(
        expected_target_cells,
        batch=batch,
        label="expected target cells",
    )
    expected_current_np = _cells(
        expected_current_cells,
        batch=batch,
        label="expected current cells",
    )
    authority_target_np = _cells(
        authority_target_cells,
        batch=batch,
        label="authority target cells",
    )
    authority_current_np = _cells(
        authority_current_cells,
        batch=batch,
        label="authority current cells",
    )

    torch_device = torch.device(device)
    candidate = (
        torch.from_numpy(candidate_np).to(torch_device).permute(0, 1, 4, 2, 3).float().contiguous().requires_grad_(True)
    )
    target = torch.from_numpy(target_np).to(torch_device).permute(0, 1, 4, 2, 3).float().contiguous()
    with torch.inference_mode():
        authority_current_pose = posenet(posenet.preprocess_input(candidate.detach()))["pose"][..., :6]
        authority_target_pose = posenet(posenet.preprocess_input(target))["pose"][..., :6]
    authority_current_pose_np = authority_current_pose.detach().cpu().numpy().astype(np.float32)
    authority_target_pose_np = authority_target_pose.detach().cpu().numpy().astype(np.float32)
    pose_pred = posenet(posenet.preprocess_input(candidate))
    current_logits = segnet(segnet.preprocess_input(candidate))
    with torch.no_grad():
        target_pose = posenet(posenet.preprocess_input(target))
        target_logits = segnet(segnet.preprocess_input(target))
    if pose_pred["pose"].shape[-1] < 6 or target_pose["pose"].shape != pose_pred["pose"].shape:
        raise ProjectedPopulationCostateError("PoseNet output changed frozen pose-head ABI")
    differentiable_current_pose_np = pose_pred["pose"][..., :6].detach().cpu().numpy().astype(np.float32)
    differentiable_target_pose_np = target_pose["pose"][..., :6].detach().cpu().numpy().astype(np.float32)
    pose_drift = PoseDifferentiableAuthorityDriftV2(
        authority_current_pose6_sha256=_sha256_array(authority_current_pose_np),
        differentiable_current_pose6_sha256=_sha256_array(differentiable_current_pose_np),
        authority_target_pose6_sha256=_sha256_array(authority_target_pose_np),
        differentiable_target_pose6_sha256=_sha256_array(differentiable_target_pose_np),
        maximum_abs_current_delta=float(
            np.max(
                np.abs(differentiable_current_pose_np.astype(np.float64) - authority_current_pose_np.astype(np.float64))
            )
        ),
        maximum_abs_target_delta=float(
            np.max(
                np.abs(differentiable_target_pose_np.astype(np.float64) - authority_target_pose_np.astype(np.float64))
            )
        ),
    )

    differentiable_current_np = current_logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
    differentiable_target_np = target_logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
    current_drift = _drift(
        axis="current",
        pair_ids=pair_ids,
        expected=expected_current_np,
        authority=authority_current_np,
        differentiable=differentiable_current_np,
        logits=current_logits,
    )
    target_drift = _drift(
        axis="target",
        pair_ids=pair_ids,
        expected=expected_target_np,
        authority=authority_target_np,
        differentiable=differentiable_target_np,
        logits=target_logits,
    )

    target_cells_t = torch.from_numpy(expected_target_np.astype(np.int64)).to(torch_device)
    current_cells_t = torch.from_numpy(expected_current_np.astype(np.int64)).to(torch_device)
    authority_target_pose_t = torch.from_numpy(authority_target_pose_np).to(torch_device)
    pose_error = pose_pred["pose"][..., :6] - authority_target_pose_t
    pair_pose_mse = pose_error.square().mean(dim=1)
    pose_objective = pair_pose_mse.sum() * score_point.pair_pose_mse_vjp_scale
    target_logit = current_logits.gather(1, target_cells_t[:, None]).squeeze(1)
    current_logit = current_logits.gather(1, current_cells_t[:, None]).squeeze(1)
    authority_mismatch = target_cells_t != current_cells_t
    gap = target_logit - current_logit
    gap_objective = gap[authority_mismatch].sum()
    pose_grad = torch.autograd.grad(
        pose_objective,
        candidate,
        retain_graph=True,
        create_graph=False,
    )[0]
    seg_grad = torch.autograd.grad(
        gap_objective,
        candidate,
        retain_graph=False,
        create_graph=False,
    )[0]
    pose_hwc = pose_grad.detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.float32)
    seg_hwc = seg_grad.detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.float32)
    authority_base_pair_pose_mse = np.mean(
        (authority_current_pose_np.astype(np.float64) - authority_target_pose_np.astype(np.float64)) ** 2,
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    costates = BatchPopulationCostatesV1(
        pair_ids=pair_ids,
        pose_costate_hwc=np.ascontiguousarray(pose_hwc),
        seg_gap_costate_hwc=np.ascontiguousarray(seg_hwc),
        base_pair_pose_mse=np.ascontiguousarray(authority_base_pair_pose_mse),
        target_pose6=np.ascontiguousarray(authority_target_pose_np),
        base_mismatch_count=int(authority_mismatch.sum().detach().cpu().item()),
        base_gap_sum=float(gap_objective.detach().cpu().item()),
        score_point=score_point,
        candidate_sha256=_sha256_array(candidate_np),
        target_sha256=_sha256_array(target_np),
        target_cells_sha256=_sha256_array(expected_target_np),
        described_cells_sha256=_sha256_array(expected_current_np),
    )
    return ExactCoarseBatchCostatesV2(
        costates=costates,
        current_drift=current_drift,
        target_drift=target_drift,
        pose_drift=pose_drift,
    )


__all__ = [
    "DifferentiableAuthorityDriftV2",
    "ExactCoarseBatchCostatesV2",
    "PoseDifferentiableAuthorityDriftV2",
    "compute_batch_exact_coarse_costates_v2",
]
