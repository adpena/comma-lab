"""Lane CG: calibrated viewing-ray positional encoding.

The encoding is derived analytically from fixed comma camera intrinsics rather
than learned as a lookup table.  For each output pixel, we map the pixel center
back to the native 1164x874 camera grid, construct the pinhole-camera ray, and
return its unit direction.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

NATIVE_W: int = 1164
NATIVE_H: int = 874
SCORER_W: int = 512
SCORER_H: int = 384
CAMERA_FX: float = 910.0
CAMERA_FY: float = 910.0
CAMERA_CX: float = 582.0
CAMERA_CY: float = 437.0

__all__ = ["CalibratedPositionalEncoding"]


class CalibratedPositionalEncoding(nn.Module):
    """Fixed per-pixel unit viewing-ray encoding.

    Args:
        learnable_scale: if true, a scalar calibrates the angular spread of
            the x/y ray components and receives gradients.
        require_cuda: reject CPU generation when CUDA-only production behavior
            is requested.
    """

    def __init__(
        self,
        *,
        learnable_scale: bool = False,
        require_cuda: bool = False,
    ) -> None:
        super().__init__()
        self.require_cuda = bool(require_cuda)
        init = torch.tensor(1.0, dtype=torch.float32)
        if learnable_scale:
            self.ray_scale = nn.Parameter(init)
        else:
            self.register_buffer("ray_scale", init)

    def forward(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if batch_size <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                f"batch_size, height, and width must be positive; got "
                f"{batch_size}, {height}, {width}"
            )
        device = torch.device(device)
        if self.require_cuda and device.type != "cuda":
            raise RuntimeError("CUDA is required for calibrated positional encoding")

        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        # Pixel centers mapped to native camera coordinates.
        u = (xx + 0.5) * (NATIVE_W / width) - 0.5
        v = (yy + 0.5) * (NATIVE_H / height) - 0.5

        scale = self.ray_scale.to(device=device, dtype=dtype)
        x = ((u - CAMERA_CX) / CAMERA_FX) * scale
        y = ((v - CAMERA_CY) / CAMERA_FY) * scale
        z = torch.ones_like(x)
        rays = torch.stack([x, y, z], dim=0)
        rays = rays / rays.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return rays.unsqueeze(0).expand(batch_size, -1, -1, -1)


def _patch_renderer_mask_renderer() -> None:
    """Allow the existing renderer constructor to accept the CG flag."""
    try:
        import tac.renderer as renderer
    except Exception:
        return

    current = renderer.MaskRenderer
    if getattr(current, "_cg_accepts_calibrated_flag", False):
        return

    class CalibratedMaskRenderer(current):  # type: ignore[misc, valid-type]
        _cg_accepts_calibrated_flag = True

        def __init__(
            self,
            *args: Any,
            use_calibrated_positional_encoding: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.use_calibrated_positional_encoding = bool(
                use_calibrated_positional_encoding
            )
            if self.use_calibrated_positional_encoding:
                self.calibrated_positional_encoding = CalibratedPositionalEncoding()
            else:
                self.calibrated_positional_encoding = None

    CalibratedMaskRenderer.__name__ = "MaskRenderer"
    CalibratedMaskRenderer.__qualname__ = "MaskRenderer"
    renderer._cg_original_mask_renderer = current
    renderer.MaskRenderer = CalibratedMaskRenderer


_patch_renderer_mask_renderer()
