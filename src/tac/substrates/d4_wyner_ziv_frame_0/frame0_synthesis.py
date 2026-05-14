# SPDX-License-Identifier: MIT
"""D4 frame-0 synthesis — warp frame_1 + add photometric residual.

The core operation:

::

    frame_0_predicted = warp(frame_1, motion_params) + residual

Both terms are decoded at inflate time:

* ``warp(frame_1, motion_params)`` — applies the chosen motion model
  (SE(3) or optical flow) to a base-substrate-provided frame_1.
* ``residual`` — per-pair int8-quantized photometric residual decoded from
  the archive's residual section.

The synthesis preserves the frame_1 bytes by construction (SegNet sees
frame_1 unchanged; PoseNet sees the reconstructed pair).
"""

from __future__ import annotations

import torch

from tac.substrates.d4_wyner_ziv_frame_0.motion_model import (
    EVAL_HW,
    MotionModelMode,
    OpticalFlowField,
    SE3MotionParams,
    apply_optical_flow,
    apply_se3_motion,
)


def synthesize_frame_0(
    *,
    frame_1: torch.Tensor,
    motion_mode: MotionModelMode,
    se3_params: SE3MotionParams | None = None,
    flow_field: OpticalFlowField | None = None,
    residual: torch.Tensor,
    output_hw: tuple[int, int] = EVAL_HW,
    clamp_unit: bool = True,
) -> torch.Tensor:
    """Reconstruct frame_0 from frame_1 + motion + residual.

    Args:
        frame_1: ``(num_pairs, 3, H, W)`` RGB tensor in unit range [0, 1].
            Provided by the BASE substrate's reconstruction at inflate time.
            At training time this is the GT frame_1 (so the residual can be
            trained against the GT frame_0).
        motion_mode: which motion model to apply.
        se3_params: required if mode is SE3_PARAMETRIC.
        flow_field: required if mode is OPTICAL_FLOW.
        residual: ``(num_pairs, 3, H, W)`` photometric residual in unit-range
            difference space (i.e. residual = frame_0_gt - warp(frame_1)).
        output_hw: target spatial resolution.
        clamp_unit: if True, clamp the output to [0, 1] (default True; matches
            contest preprocess range expectations).

    Returns:
        ``(num_pairs, 3, H_out, W_out)`` reconstructed frame_0.

    Raises:
        ValueError: if shapes mismatch or required motion params are missing.
    """
    if frame_1.dim() != 4 or frame_1.shape[1] != 3:
        raise ValueError(
            f"frame_1 must be (N, 3, H, W); got {tuple(frame_1.shape)}"
        )
    if residual.shape != frame_1.shape:
        # Allow residual to be at output_hw if frame_1 already is; otherwise
        # we resize residual to match the warped output.
        if residual.shape[0] != frame_1.shape[0] or residual.shape[1] != 3:
            raise ValueError(
                f"residual batch/channel mismatch: residual={tuple(residual.shape)} "
                f"frame_1={tuple(frame_1.shape)}"
            )
    if motion_mode == MotionModelMode.SE3_PARAMETRIC:
        if se3_params is None:
            raise ValueError("se3_params is required for SE3_PARAMETRIC mode")
        warped = apply_se3_motion(frame_1, se3_params, output_hw=output_hw)
    elif motion_mode == MotionModelMode.OPTICAL_FLOW:
        if flow_field is None:
            raise ValueError("flow_field is required for OPTICAL_FLOW mode")
        warped = apply_optical_flow(frame_1, flow_field, output_hw=output_hw)
    else:  # pragma: no cover — guarded by Enum
        raise ValueError(f"unknown motion mode: {motion_mode!r}")

    # If residual spatial dims differ from warped, bilinear-resize.
    if residual.shape[-2:] != warped.shape[-2:]:
        residual = torch.nn.functional.interpolate(
            residual,
            size=warped.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    reconstructed = warped + residual
    if clamp_unit:
        reconstructed = reconstructed.clamp(0.0, 1.0)
    return reconstructed


__all__ = ["synthesize_frame_0"]
