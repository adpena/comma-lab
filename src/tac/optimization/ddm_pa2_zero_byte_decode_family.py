# SPDX-License-Identifier: MIT
"""Deterministic zero-counted-byte decode transforms for DDM PA2.

The transforms in this module may inspect only the decoded camera frames and
frozen, video-independent geometry.  They never accept a per-pair table,
learned payload, scorer target, or ground-truth label.  That makes the callable
surface suitable for rule-118 receiver code while keeping every video-derived
quantity in the counted archive.

The spatial and temporal members are deliberately scorer-recursive:

* spatial work is performed after the exact ``874x1164 -> 384x512`` resize and
  on the SegNet stride-2 stem lattice;
* temporal displacement is estimated from decoded-frame luminance gradients,
  then applied at the same scorer resolution;
* residuals are lifted to camera resolution with the same deterministic
  bilinear convention used by the frozen scorers.

These are candidate actuators, not admitted score claims.  Admission requires
fresh frozen-scorer measurement on the exact receiver output and exact archive
byte custody.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np

from tac.through_r.blind_coordinate import build_blind_mask

CAMERA_H: Final = 874
CAMERA_W: Final = 1164
SCORER_H: Final = 384
SCORER_W: Final = 512
CHANNELS: Final = 3
FRAMES_PER_PAIR: Final = 2


class PA2TransformError(ValueError):
    """A PA2 transform was asked to cross its typed receiver boundary."""


class PA2Member(StrEnum):
    """Stable IDs for the PA2 family and its two typed blocked members."""

    BLIND_ZERO_FILL = "pa2_blind_coordinate_zero_fill_v1"
    SPATIAL_STEM_RESIDUAL = "pa2_spatial_stride2_stem_residual_v1"
    TEMPORAL_XIHAT_FRAME0 = "pa2_temporal_xihat_frame0_companion_v1"
    TEMPORAL_XIHAT_FRAME1 = "pa2_temporal_xihat_frame1_proposal_v1"
    GAUGE_ORBIT = "pa2_gauge_orbit_rgb_pullback_v1"
    RANK4_CLASS_TONE = "pa2_rank4_class_tone_gamma_v1"


@dataclass(frozen=True)
class TypedBlocker:
    """A machine-readable reason a family member cannot be a free actuator."""

    member: PA2Member
    blocker_code: str
    missing_surface: str
    counted_if_supplied: str
    verdict_scope: str
    authority_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "member": self.member.value,
            "blocker_code": self.blocker_code,
            "missing_surface": self.missing_surface,
            "counted_if_supplied": self.counted_if_supplied,
            "verdict_scope": self.verdict_scope,
            "authority_paths": list(self.authority_paths),
        }


BLOCKERS: Final[tuple[TypedBlocker, ...]] = (
    TypedBlocker(
        member=PA2Member.GAUGE_ORBIT,
        blocker_code="missing_generic_rgb_uint8_receiver_pullback",
        missing_surface=(
            "The measured gauge energy is sample-specific and no deterministic "
            "decoded-frame-only map from the gauge coordinate to legal RGB uint8 "
            "camera residuals is registered."
        ),
        counted_if_supplied=(
            "Per-pair gauge coefficients, residuals, or selected orbit positions "
            "are video-derived payload and therefore COUNTED."
        ),
        verdict_scope=(
            "Blocks only the current zero-byte RGB realization; it does not reject "
            "the gauge family under a future generic receiver pullback."
        ),
        authority_paths=(
            "src/tac/optimization/predictor_r4_tailrace.py",
            ".omx/research/prereq_surfaces_flush_20260720/surface_2_rank4_prototype_bank.json",
        ),
    ),
    TypedBlocker(
        member=PA2Member.RANK4_CLASS_TONE,
        blocker_code="rank4_feature_prototypes_lack_rgb_pullback",
        missing_surface=(
            "The frozen rank-4 artifact contains feature-space prototypes and "
            "hashes, not a class assignment plus RGB/uint8 receiver pullback."
        ),
        counted_if_supplied=(
            "Per-pair class maps, tone values, gamma values, or RGB prototype "
            "tables are video-derived payload and therefore COUNTED."
        ),
        verdict_scope=(
            "Blocks only free decoded-frame application against the present "
            "feature-only artifact; the prototype family remains viable after a "
            "frozen video-independent RGB pullback is proved."
        ),
        authority_paths=(
            "src/tac/optimization/predictor_r4_tailrace.py",
            ".omx/research/prereq_surfaces_flush_20260720/surface_2_rank4_prototype_bank.json",
        ),
    ),
)

EXECUTABLE_MEMBERS: Final[tuple[PA2Member, ...]] = (
    PA2Member.BLIND_ZERO_FILL,
    PA2Member.SPATIAL_STEM_RESIDUAL,
    PA2Member.TEMPORAL_XIHAT_FRAME0,
    PA2Member.TEMPORAL_XIHAT_FRAME1,
)


def _camera_batch(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.uint8
        or array.ndim != 5
        or array.shape[1:] != (FRAMES_PER_PAIR, CAMERA_H, CAMERA_W, CHANNELS)
    ):
        raise PA2TransformError(
            "camera batch must be uint8 [B,2,874,1164,3]"
        )
    return np.ascontiguousarray(array)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment guard
        raise PA2TransformError("PA2 scorer-recursive transforms require torch") from error
    return torch


def _camera_tensor(camera: np.ndarray) -> Any:
    torch = _torch()
    return (
        torch.from_numpy(np.ascontiguousarray(camera))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .to(torch.float32)
    )


def _to_camera(tensor: Any) -> np.ndarray:
    return np.ascontiguousarray(
        tensor.clamp(0.0, 255.0)
        .round()
        .to(_torch().uint8)
        .permute(0, 1, 3, 4, 2)
        .cpu()
        .numpy()
    )


def scorer_resize(camera: np.ndarray) -> Any:
    """Apply the exact frozen scorer resize to a validated camera batch."""

    torch = _torch()
    tensor = _camera_tensor(_camera_batch(camera))
    return torch.nn.functional.interpolate(
        tensor.reshape(-1, CHANNELS, CAMERA_H, CAMERA_W),
        size=(SCORER_H, SCORER_W),
        mode="bilinear",
        align_corners=False,
    ).reshape(
        int(tensor.shape[0]),
        FRAMES_PER_PAIR,
        CHANNELS,
        SCORER_H,
        SCORER_W,
    )


def _lift_low_residual(camera: np.ndarray, residual: Any) -> np.ndarray:
    torch = _torch()
    source = _camera_tensor(_camera_batch(camera))
    if tuple(residual.shape) != (
        int(source.shape[0]),
        FRAMES_PER_PAIR,
        CHANNELS,
        SCORER_H,
        SCORER_W,
    ):
        raise PA2TransformError("low-resolution residual geometry differs")
    lifted = torch.nn.functional.interpolate(
        residual.reshape(-1, CHANNELS, SCORER_H, SCORER_W),
        size=(CAMERA_H, CAMERA_W),
        mode="bilinear",
        align_corners=False,
    ).reshape_as(source)
    return _to_camera(source + lifted)


def blind_zero_fill(camera: np.ndarray) -> np.ndarray:
    """Fill #401 blind coordinates with a generic constant and store no table.

    The value zero is video-independent.  It changes only exact zero-weight
    camera coordinates of the frozen resize, so both scorer inputs remain
    unchanged by construction.  Pure-generator bases save zero archive bytes;
    the member is retained because it becomes a real rate lever as soon as a
    camera-resolution residual section is counted.
    """

    source = _camera_batch(camera)
    output = source.copy()
    output[:, :, build_blind_mask().mask, :] = 0
    return output


def spatial_stride2_stem_residual(camera: np.ndarray) -> np.ndarray:
    """Apply a boundary-only residual on SegNet's exact stride-2 stem lattice.

    A 2x2 scorer block is a boundary iff any RGB component differs inside the
    block.  On those blocks, the decoded frame-1 plane is reflected halfway
    away from its block mean.  Flat blocks and frame 0 are unchanged.  The
    one-half coefficient is the symmetric midpoint on the two-scale
    resize-to-stem chain; no sample-specific threshold or coefficient exists.
    """

    torch = _torch()
    source = _camera_batch(camera)
    low = scorer_resize(source)
    frame1 = low[:, 1]
    blocks = frame1.reshape(
        int(frame1.shape[0]),
        CHANNELS,
        SCORER_H // 2,
        2,
        SCORER_W // 2,
        2,
    )
    minimum = blocks.amin(dim=(3, 5), keepdim=True)
    maximum = blocks.amax(dim=(3, 5), keepdim=True)
    boundary = (maximum > minimum).any(dim=1, keepdim=True)
    mean = blocks.mean(dim=(3, 5), keepdim=True)
    sharpened = torch.where(boundary, blocks + (blocks - mean) * 0.5, blocks)
    candidate = sharpened.reshape_as(frame1).clamp(0.0, 255.0)
    residual = torch.zeros_like(low)
    residual[:, 1] = candidate - frame1
    return _lift_low_residual(source, residual)


def _luma_gradient_centroid(frame: Any) -> tuple[Any, Any]:
    """Return a parameter-free decoded-content centroid of luma variation."""

    torch = _torch()
    red, green, blue = frame[:, 0], frame[:, 1], frame[:, 2]
    luma = red * 0.299 + green * 0.587 + blue * 0.114
    gradient = torch.zeros_like(luma)
    gradient[:, :, 1:] += (luma[:, :, 1:] - luma[:, :, :-1]).abs()
    gradient[:, 1:, :] += (luma[:, 1:, :] - luma[:, :-1, :]).abs()
    weights = gradient.to(torch.float64)
    total = weights.sum(dim=(1, 2)).clamp_min(1.0)
    rows = torch.arange(SCORER_H, dtype=torch.float64, device=frame.device)
    cols = torch.arange(SCORER_W, dtype=torch.float64, device=frame.device)
    row = (weights * rows.reshape(1, -1, 1)).sum(dim=(1, 2)) / total
    col = (weights * cols.reshape(1, 1, -1)).sum(dim=(1, 2)) / total
    return row, col


def estimate_xihat(low: Any) -> tuple[Any, Any]:
    """Estimate integer frame0->frame1 displacement from decoded frames only."""

    if tuple(low.shape[1:]) != (
        FRAMES_PER_PAIR,
        CHANNELS,
        SCORER_H,
        SCORER_W,
    ):
        raise PA2TransformError("xi-hat expects [B,2,3,384,512]")
    row0, col0 = _luma_gradient_centroid(low[:, 0])
    row1, col1 = _luma_gradient_centroid(low[:, 1])
    return (row1 - row0).round().to(_torch().int64), (
        col1 - col0
    ).round().to(_torch().int64)


def _translate(frame: Any, row_shift: Any, col_shift: Any) -> Any:
    """Translate with edge replication, never wraparound."""

    torch = _torch()
    batch = int(frame.shape[0])
    rows = torch.arange(SCORER_H, device=frame.device).reshape(1, -1)
    cols = torch.arange(SCORER_W, device=frame.device).reshape(1, -1)
    source_rows = (rows - row_shift.reshape(-1, 1)).clamp(0, SCORER_H - 1)
    source_cols = (cols - col_shift.reshape(-1, 1)).clamp(0, SCORER_W - 1)
    batch_indices = torch.arange(batch, device=frame.device).reshape(-1, 1, 1)
    return frame[
        batch_indices,
        :,
        source_rows.reshape(batch, SCORER_H, 1),
        source_cols.reshape(batch, 1, SCORER_W),
    ].permute(0, 3, 1, 2).contiguous()


def temporal_xihat(camera: np.ndarray, *, target_frame: int) -> np.ndarray:
    """Blend one frame halfway toward its xi-hat-aligned companion.

    ``target_frame=0`` is Seg-safe because the frozen SegNet reads frame 1
    only.  ``target_frame=1`` is the literal proposal arm and must earn back
    any Seg movement in the joint score.  Both use only decoded frames.
    """

    if target_frame not in (0, 1):
        raise PA2TransformError("target_frame must be 0 or 1")
    source = _camera_batch(camera)
    low = scorer_resize(source)
    row_shift, col_shift = estimate_xihat(low)
    if target_frame == 0:
        aligned = _translate(low[:, 1], -row_shift, -col_shift)
    else:
        aligned = _translate(low[:, 0], row_shift, col_shift)
    candidate = (low[:, target_frame] + aligned) * 0.5
    residual = _torch().zeros_like(low)
    residual[:, target_frame] = candidate - low[:, target_frame]
    return _lift_low_residual(source, residual)


def apply_member(camera: np.ndarray, member: PA2Member | str) -> np.ndarray:
    """Apply one executable member, refusing typed blocked members."""

    try:
        resolved = member if isinstance(member, PA2Member) else PA2Member(member)
    except ValueError as error:
        raise PA2TransformError(f"unknown PA2 member: {member!r}") from error
    if resolved == PA2Member.BLIND_ZERO_FILL:
        return blind_zero_fill(camera)
    if resolved == PA2Member.SPATIAL_STEM_RESIDUAL:
        return spatial_stride2_stem_residual(camera)
    if resolved == PA2Member.TEMPORAL_XIHAT_FRAME0:
        return temporal_xihat(camera, target_frame=0)
    if resolved == PA2Member.TEMPORAL_XIHAT_FRAME1:
        return temporal_xihat(camera, target_frame=1)
    blocker = next(row for row in BLOCKERS if row.member == resolved)
    raise PA2TransformError(
        f"{blocker.blocker_code}: {blocker.missing_surface} "
        f"{blocker.counted_if_supplied}"
    )


def apply_stack(
    camera: np.ndarray,
    members: Iterable[PA2Member | str],
) -> np.ndarray:
    """Compose a fixed ordered receiver stack without hidden state."""

    output = _camera_batch(camera)
    for member in members:
        output = apply_member(output, member)
    return output


def family_inventory() -> dict[str, Any]:
    """Return the durable rule-118 boundary and typed member inventory."""

    return {
        "schema": "ddm_pa2_zero_byte_decode_family.v1",
        "geometry": {
            "camera_hw": [CAMERA_H, CAMERA_W],
            "scorer_hw": [SCORER_H, SCORER_W],
            "segnet_frame": 1,
            "posenet_frames": [0, 1],
            "stem_stride": 2,
        },
        "rate_boundary": {
            "free": (
                "fixed transform code and quantities derived solely from decoded "
                "frames at receiver runtime"
            ),
            "counted": (
                "per-pair tables, target labels, learned/video-derived "
                "coefficients, selected orbit positions, or residual payloads"
            ),
        },
        "executable_members": [member.value for member in EXECUTABLE_MEMBERS],
        "blocked_members": [blocker.to_dict() for blocker in BLOCKERS],
        "score_claim": False,
    }
