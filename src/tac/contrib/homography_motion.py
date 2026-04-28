"""Lane HM: analytical road-plane homography motion.

This replaces the learned motion CNN with a differentiable perspective-zoom
field centered on the scorer-space focus of expansion.  The implementation is
small, deterministic, and accepts both hard class-index masks and soft class
probability tensors.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

SCORER_H: int = 384
SCORER_W: int = 512
FOE_X: float = 256.0
FOE_Y: float = 174.0

__all__ = ["HomographyMotionModule", "FOE_X", "FOE_Y"]


class HomographyMotionModule(nn.Module):
    """Analytical perspective-zoom motion module.

    Args:
        output_channels: ``2`` for flow only, ``6`` for renderer-compatible
            ``flow(2)+gate(1)+residual(3)`` output.
        max_flow_px: cap for the perspective-zoom flow magnitude.
        learn_velocity_scale: if true, exposes ``velocity_scale`` as a trainable
            scalar.
        require_cuda: reject CPU inputs when production CUDA-only behavior is
            requested.
    """

    def __init__(
        self,
        *,
        output_channels: int = 6,
        max_flow_px: float = 20.0,
        dt: float = 0.05,
        foe_distance: float = 30.0,
        learn_velocity_scale: bool = False,
        require_cuda: bool = False,
    ) -> None:
        super().__init__()
        if output_channels < 2:
            raise ValueError(f"output_channels must be at least 2; got {output_channels}")
        self.output_channels = int(output_channels)
        self.max_flow_px = float(max_flow_px)
        self.dt = float(dt)
        self.foe_distance = float(foe_distance)
        self.require_cuda = bool(require_cuda)

        init = torch.tensor(1.0, dtype=torch.float32)
        if learn_velocity_scale:
            self.velocity_scale = nn.Parameter(init)
        else:
            self.register_buffer("velocity_scale", init)

    def _mask_scalar(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            return mask.float()
        if mask.ndim == 4:
            classes = torch.arange(mask.shape[1], device=mask.device, dtype=mask.dtype)
            view_shape = (1, mask.shape[1], 1, 1)
            return (mask * classes.view(view_shape)).sum(dim=1)
        raise ValueError(
            f"mask must be hard (B, H, W) or soft (B, C, H, W); got {tuple(mask.shape)}"
        )

    def forward(self, mask_t: torch.Tensor, mask_t1: torch.Tensor) -> torch.Tensor:
        if self.require_cuda and mask_t.device.type != "cuda":
            raise RuntimeError("CUDA is required for homography motion")
        if mask_t.shape != mask_t1.shape:
            raise ValueError(
                f"mask shape mismatch: {tuple(mask_t.shape)} vs {tuple(mask_t1.shape)}"
            )

        scalar_t = self._mask_scalar(mask_t)
        scalar_t1 = self._mask_scalar(mask_t1)
        if scalar_t.ndim != 3:
            raise ValueError(f"internal mask scalar must be (B, H, W); got {tuple(scalar_t.shape)}")

        b, h, w = scalar_t.shape
        device = scalar_t.device
        dtype = torch.promote_types(scalar_t.dtype, self.velocity_scale.dtype)
        scalar_t = scalar_t.to(dtype=dtype)
        scalar_t1 = scalar_t1.to(dtype=dtype)

        # A compact speed proxy from the mask-pair change.  Hard masks and soft
        # probabilities both keep gradients through this path where possible.
        delta = (scalar_t1 - scalar_t).abs()
        speed = delta.mean(dim=(1, 2), keepdim=True)
        zoom = torch.tanh(speed * self.velocity_scale.to(device=device, dtype=dtype))
        zoom = zoom * (self.dt / max(self.foe_distance, 1e-6))

        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        foe_x = FOE_X * (w / SCORER_W)
        foe_y = FOE_Y * (h / SCORER_H)

        raw_x = zoom * (xx.unsqueeze(0) - foe_x)
        raw_y = zoom * (yy.unsqueeze(0) - foe_y)
        radius = torch.sqrt(raw_x.square() + raw_y.square()).amax(dim=(1, 2), keepdim=True)
        cap = self.max_flow_px / radius.clamp_min(1e-6)
        scale = torch.minimum(torch.ones_like(cap), cap)
        flow_px_x = raw_x * scale
        flow_px_y = raw_y * scale

        denom_w = max(w - 1, 1)
        denom_h = max(h - 1, 1)
        flow = torch.stack(
            [
                flow_px_x * (2.0 / denom_w),
                flow_px_y * (2.0 / denom_h),
            ],
            dim=1,
        )

        if self.output_channels == 2:
            return flow

        extras = torch.zeros(
            b,
            self.output_channels - 2,
            h,
            w,
            device=device,
            dtype=dtype,
        )
        if extras.shape[1] >= 1:
            extras[:, 0:1] = -2.0
        return torch.cat([flow, extras], dim=1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _patch_renderer_dispatch() -> None:
    """Register ``motion_type='homography_analytical'`` without editing renderer.py."""
    try:
        import tac.renderer as renderer
    except Exception:
        return

    if getattr(renderer, "_hm_dispatch_patched", False):
        return

    original = renderer.build_renderer

    def build_renderer_with_hm(*args: Any, **kwargs: Any) -> Any:
        motion_type = kwargs.get("motion_type", "learned_cnn")
        if motion_type != "homography_analytical":
            return original(*args, **kwargs)

        patched_kwargs = dict(kwargs)
        patched_kwargs["motion_type"] = "none"
        model = original(*args, **patched_kwargs)
        max_flow_px = float(kwargs.get("max_flow_px", 20.0))
        model.motion = HomographyMotionModule(output_channels=6, max_flow_px=max_flow_px)
        return model

    renderer._hm_dispatch_patched = True
    renderer._hm_original_build_renderer = original
    renderer.build_renderer = build_renderer_with_hm


_patch_renderer_dispatch()
