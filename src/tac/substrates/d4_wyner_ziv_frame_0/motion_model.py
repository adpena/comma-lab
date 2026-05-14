# SPDX-License-Identifier: MIT
"""D4 motion models — SE(3) parametric AND optical-flow non-parametric.

Per the probe-disambiguator contract (Catalog #125 hook #6) BOTH motion modes
ship as callable interfaces; `tools/probe_d4_motion_model_disambiguator.py`
empirically arbitrates per substrate which mode gives the lowest pose-residual
error.

Mode A — SE(3) parametric motion
--------------------------------

6 floats per pair = 3 translation + 3 axis-angle rotation. The warp is

::

    K_inv * frame_0_pixel = R * (K_inv * frame_1_pixel) + t * (1/depth)

We approximate depth ≈ const at the contest scorer resolution (384×512) and
fold the translation magnitude into a global scale; the resulting per-pair
affine warp has 6 free parameters AND a per-pair scalar depth (7 floats).
For byte budget we round depth to 0.001 precision in a fixed range and
encode 6 float16 + 1 uint16 = 14 B/pair. At 600 pairs → 8.4 KB.

For training simplicity we use the affine-flow approximation where the SE(3)
motion is linearized at the image center (the contest dashcam has small
per-frame rotation; 20 Hz capture). The 6-parameter affine flow matrix is
fitted via differentiable bilinear warping (``torch.nn.functional.grid_sample``).

Mode B — Optical-flow field
---------------------------

Per-pixel ``(u, v)`` flow at coarse resolution (default 12×16 = 192 cells).
Quantized to int8 (range ±64 pixels at scorer-res ÷ 4 = ±16 pixel/cell) +
brotli-packed. Better for parallax-heavy scenes (vehicles, lane changes).

Byte cost: 2 int8 × 192 cells × 600 pairs = 230,400 B raw; brotli typically
closes to ~30-50 KB on highly-correlated flow fields.

Both motion modes are differentiable w.r.t. their parameters AND w.r.t.
frame_1 pixels (gradient flows back through ``grid_sample`` and through the
flow field). Training jointly optimizes motion + residual to minimize
PoseNet pose error on the reconstructed pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import nn

# Scorer resolution (frame_0, frame_1 reach this size after the contest preprocess).
EVAL_HW: tuple[int, int] = (384, 512)


class MotionModelMode(str, Enum):
    """Two defensible interpretations per probe-disambiguator contract."""

    SE3_PARAMETRIC = "se3_parametric"
    OPTICAL_FLOW = "optical_flow"


@dataclass
class SE3MotionParams:
    """Per-pair SE(3) motion: 6 floats encoding translation + axis-angle rotation.

    The contest dashcam captures at 20 Hz so the per-frame rotation is small
    (typically < 5 degrees). The affine-flow approximation linearizes the
    SE(3) twist at the image center and is sufficient for ego-motion modeling.

    Args:
        translation: shape ``(num_pairs, 3)`` — t_x, t_y, t_z in normalized
            image coordinates (range ~ [-0.1, 0.1] for 20 Hz dashcam).
        axis_angle: shape ``(num_pairs, 3)`` — rotation in axis-angle form
            (range ~ [-0.1, 0.1] radians for 20 Hz dashcam).
    """

    translation: torch.Tensor
    axis_angle: torch.Tensor

    def __post_init__(self) -> None:
        if self.translation.dim() != 2 or self.translation.shape[1] != 3:
            raise ValueError(
                f"translation shape must be (num_pairs, 3); got "
                f"{tuple(self.translation.shape)}"
            )
        if self.axis_angle.shape != self.translation.shape:
            raise ValueError(
                f"axis_angle shape {tuple(self.axis_angle.shape)} != "
                f"translation shape {tuple(self.translation.shape)}"
            )

    @property
    def num_pairs(self) -> int:
        return int(self.translation.shape[0])

    def to_flat(self) -> torch.Tensor:
        """Return ``(num_pairs, 6)`` concatenated parameter tensor."""
        return torch.cat([self.translation, self.axis_angle], dim=1)

    @classmethod
    def from_flat(cls, flat: torch.Tensor) -> "SE3MotionParams":
        if flat.dim() != 2 or flat.shape[1] != 6:
            raise ValueError(
                f"from_flat expects (num_pairs, 6); got {tuple(flat.shape)}"
            )
        return cls(translation=flat[:, :3].contiguous(), axis_angle=flat[:, 3:].contiguous())


@dataclass
class OpticalFlowField:
    """Per-pair optical flow at a coarse grid.

    Args:
        flow_uv: shape ``(num_pairs, 2, grid_h, grid_w)`` — (u, v) flow in
            normalized image coordinates (range typically [-0.05, 0.05]).
        grid_h: coarse flow grid height (default 12 ≈ 384/32).
        grid_w: coarse flow grid width (default 16 ≈ 512/32).
    """

    flow_uv: torch.Tensor
    grid_h: int = 12
    grid_w: int = 16

    def __post_init__(self) -> None:
        if self.flow_uv.dim() != 4:
            raise ValueError(
                f"flow_uv must be 4D (num_pairs, 2, grid_h, grid_w); got "
                f"{self.flow_uv.dim()}D"
            )
        if self.flow_uv.shape[1] != 2:
            raise ValueError(
                f"flow_uv channels must be 2 (u, v); got {self.flow_uv.shape[1]}"
            )
        if self.flow_uv.shape[2] != self.grid_h or self.flow_uv.shape[3] != self.grid_w:
            raise ValueError(
                f"flow_uv shape {tuple(self.flow_uv.shape)} does not match "
                f"grid_h={self.grid_h} grid_w={self.grid_w}"
            )

    @property
    def num_pairs(self) -> int:
        return int(self.flow_uv.shape[0])


def _axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: axis-angle ``(N, 3)`` -> rotation matrix ``(N, 3, 3)``.

    Differentiable; backprop-safe via the standard Rodrigues series.
    """
    theta = torch.linalg.norm(axis_angle, dim=1, keepdim=True).clamp_min(1e-9)
    axis = axis_angle / theta  # (N, 3)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    one_minus_cos = 1.0 - cos_t

    # Skew matrices
    n = axis_angle.shape[0]
    zero = torch.zeros(n, device=axis.device, dtype=axis.dtype)
    K = torch.stack(
        [
            torch.stack([zero, -axis[:, 2], axis[:, 1]], dim=1),
            torch.stack([axis[:, 2], zero, -axis[:, 0]], dim=1),
            torch.stack([-axis[:, 1], axis[:, 0], zero], dim=1),
        ],
        dim=1,
    )  # (N, 3, 3)

    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(n, -1, -1)
    K_sq = torch.bmm(K, K)
    R = I + sin_t.unsqueeze(-1) * K + one_minus_cos.unsqueeze(-1) * K_sq
    return R


def _se3_to_affine_warp(
    se3_params: SE3MotionParams,
    *,
    output_hw: tuple[int, int] = EVAL_HW,
) -> torch.Tensor:
    """Convert SE(3) motion to per-pair 2D affine matrices suitable for grid_sample.

    Returns:
        Tensor of shape ``(num_pairs, 2, 3)`` — the 2D affine matrix per pair.

    The full SE(3) -> 2D affine projection at the image center is:

    1. Rotate around the principal axis (small angle approximation).
    2. Translate in normalized image coordinates.

    For grid_sample's ``align_corners=False`` convention the affine matrix
    transforms output pixel coordinates back to input sampling coordinates;
    we approximate the SE(3) twist at image center.
    """
    n = se3_params.num_pairs
    R3 = _axis_angle_to_rotation_matrix(se3_params.axis_angle)  # (N, 3, 3)
    # Project to 2D affine: take the upper-left 2x2 (in-plane rotation) plus
    # the (x, y) translation (z-translation absorbed into a global scale; we
    # use a small forward-driving approximation where z-translation produces
    # a small isotropic zoom).
    R2 = R3[:, :2, :2].contiguous()  # (N, 2, 2)
    tx = se3_params.translation[:, 0]
    ty = se3_params.translation[:, 1]
    tz = se3_params.translation[:, 2]
    # z-translation -> zoom factor; small forward-driving model
    zoom = (1.0 + tz * 0.5).unsqueeze(-1).unsqueeze(-1)
    R2_scaled = R2 / zoom
    affine = torch.zeros(n, 2, 3, device=R2.device, dtype=R2.dtype)
    affine[:, :, :2] = R2_scaled
    affine[:, 0, 2] = tx
    affine[:, 1, 2] = ty
    return affine


def apply_se3_motion(
    frame_1: torch.Tensor,
    se3_params: SE3MotionParams,
    *,
    output_hw: tuple[int, int] = EVAL_HW,
) -> torch.Tensor:
    """Differentiably warp frame_1 by per-pair SE(3) motion to predict frame_0.

    Args:
        frame_1: ``(num_pairs, 3, H, W)`` RGB tensor in unit range [0, 1].
        se3_params: per-pair motion parameters.
        output_hw: target spatial resolution (default 384x512).

    Returns:
        ``(num_pairs, 3, H_out, W_out)`` predicted frame_0.
    """
    if frame_1.dim() != 4:
        raise ValueError(
            f"frame_1 must be 4D (N, C, H, W); got {frame_1.dim()}D shape "
            f"{tuple(frame_1.shape)}"
        )
    if frame_1.shape[0] != se3_params.num_pairs:
        raise ValueError(
            f"frame_1 batch {frame_1.shape[0]} != se3 num_pairs "
            f"{se3_params.num_pairs}"
        )
    n = frame_1.shape[0]
    affine = _se3_to_affine_warp(se3_params, output_hw=output_hw)
    grid = torch.nn.functional.affine_grid(
        affine,
        size=[n, 3, output_hw[0], output_hw[1]],
        align_corners=False,
    )
    return torch.nn.functional.grid_sample(
        frame_1,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def apply_optical_flow(
    frame_1: torch.Tensor,
    flow_field: OpticalFlowField,
    *,
    output_hw: tuple[int, int] = EVAL_HW,
) -> torch.Tensor:
    """Differentiably warp frame_1 by per-pair coarse optical flow.

    The coarse flow grid is bilinearly upsampled to the output resolution
    and applied via grid_sample. Backprop-safe through both the flow field
    and frame_1.

    Args:
        frame_1: ``(num_pairs, 3, H, W)`` RGB tensor in unit range.
        flow_field: per-pair coarse flow field.
        output_hw: target spatial resolution.

    Returns:
        ``(num_pairs, 3, H_out, W_out)`` predicted frame_0.
    """
    if frame_1.dim() != 4:
        raise ValueError(
            f"frame_1 must be 4D; got {frame_1.dim()}D shape "
            f"{tuple(frame_1.shape)}"
        )
    if frame_1.shape[0] != flow_field.num_pairs:
        raise ValueError(
            f"frame_1 batch {frame_1.shape[0]} != flow num_pairs "
            f"{flow_field.num_pairs}"
        )
    n = frame_1.shape[0]
    h_out, w_out = output_hw
    # Upsample coarse flow grid to full resolution
    flow_full = torch.nn.functional.interpolate(
        flow_field.flow_uv,
        size=(h_out, w_out),
        mode="bilinear",
        align_corners=False,
    )  # (N, 2, H_out, W_out)
    # Build the sample grid: base (mesh) + flow
    ys = torch.linspace(-1.0, 1.0, h_out, device=frame_1.device, dtype=frame_1.dtype)
    xs = torch.linspace(-1.0, 1.0, w_out, device=frame_1.device, dtype=frame_1.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
    flow_xy = flow_full.permute(0, 2, 3, 1).contiguous()  # (N, H, W, 2) with (u, v)
    sample_grid = base_grid + flow_xy
    return torch.nn.functional.grid_sample(
        frame_1,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


class MotionModelModule(nn.Module):
    """Wraps the chosen motion mode as a trainable nn.Module.

    Per the probe-disambiguator contract both modes ship as callable
    interfaces. The MotionModelModule selects between them via the
    ``mode`` constructor argument and exposes a unified ``forward(frame_1)``
    surface so both substrate trainer paths reuse the same skeleton.
    """

    def __init__(
        self,
        *,
        mode: MotionModelMode,
        num_pairs: int,
        flow_grid_h: int = 12,
        flow_grid_w: int = 16,
        output_hw: tuple[int, int] = EVAL_HW,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_pairs = num_pairs
        self.flow_grid_h = flow_grid_h
        self.flow_grid_w = flow_grid_w
        self.output_hw = output_hw
        if mode == MotionModelMode.SE3_PARAMETRIC:
            # 6 trainable parameters per pair, initialized near identity.
            self.se3_flat = nn.Parameter(
                torch.zeros(num_pairs, 6) + 1e-4 * torch.randn(num_pairs, 6)
            )
            self.flow_uv = None
        elif mode == MotionModelMode.OPTICAL_FLOW:
            self.flow_uv = nn.Parameter(
                torch.zeros(num_pairs, 2, flow_grid_h, flow_grid_w)
            )
            self.se3_flat = None
        else:  # pragma: no cover - guarded by Enum
            raise ValueError(f"unknown motion mode: {mode!r}")

    def forward(self, frame_1: torch.Tensor) -> torch.Tensor:
        if self.mode == MotionModelMode.SE3_PARAMETRIC:
            assert self.se3_flat is not None
            params = SE3MotionParams.from_flat(self.se3_flat)
            return apply_se3_motion(frame_1, params, output_hw=self.output_hw)
        else:
            assert self.flow_uv is not None
            field = OpticalFlowField(
                flow_uv=self.flow_uv,
                grid_h=self.flow_grid_h,
                grid_w=self.flow_grid_w,
            )
            return apply_optical_flow(frame_1, field, output_hw=self.output_hw)


__all__ = [
    "EVAL_HW",
    "MotionModelMode",
    "MotionModelModule",
    "OpticalFlowField",
    "SE3MotionParams",
    "apply_optical_flow",
    "apply_se3_motion",
]
