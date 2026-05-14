# SPDX-License-Identifier: MIT
"""Lane SI-V3: UNIWARD-style texture probability masks."""
from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["compute_texture_probability"]


def _gradient_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernels = torch.tensor(
        [
            [[[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernels


def compute_texture_probability(
    frames: torch.Tensor,
    scorers: list[nn.Module] | tuple[nn.Module, ...],
    *,
    detach: bool = True,
    require_cuda: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a non-negative UNIWARD texture probability map.

    ``frames`` is ``(N, 3, H, W)``.  The returned ``(H, W)`` tensor is high
    where horizontal, vertical, and diagonal residual energy is high; those
    textured regions are suitable for more aggressive compression.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must have shape (N, 3, H, W); got {tuple(frames.shape)}")
    if require_cuda and frames.device.type != "cuda":
        raise RuntimeError("CUDA is required for UNIWARD texture probability")

    x = frames.to(dtype=torch.float32)
    n, c, h, w = x.shape
    flat = x.reshape(n * c, 1, h, w)
    kernels = _gradient_kernels(x.device, x.dtype)
    residuals = F.conv2d(flat, kernels, padding=1)
    energy = residuals.square().sum(dim=1).reshape(n, c, h, w)

    mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mean_sq = F.avg_pool2d(x.square(), kernel_size=3, stride=1, padding=1)
    local_var = (mean_sq - mean.square()).clamp_min(0.0)

    sigma2 = (energy * (1.0 + local_var)).mean(dim=(0, 1))
    sigma2 = torch.nan_to_num(sigma2, nan=0.0, posinf=0.0, neginf=0.0)
    sigma2 = sigma2.clamp_min(eps)
    if detach:
        sigma2 = sigma2.detach()
    return sigma2


def _patched_apply_saliency_weighted_compression(
    masks: torch.Tensor,
    saliency_inv: torch.Tensor | None = None,
    high_crf: int = 50,
    low_crf: int = 30,
    *,
    encoder: Any = None,
    saliency: torch.Tensor | None = None,
    target_bytes: int | None = None,
    target_bytes_tolerance: float = 256.0,
    mode: str | None = None,
    texture_probability: torch.Tensor | None = None,
    texture_quantile: float = 0.5,
) -> bytes:
    if mode is None:
        # The code object is swapped onto tac.saliency_inversion's function, so
        # globals() resolves through tac.saliency_inversion rather than this module.
        original_apply = globals().get("_UNIWARD_ORIGINAL_APPLY")
        if not callable(original_apply):
            raise RuntimeError("UNIWARD patch missing original saliency compressor")
        return original_apply(
            masks,
            saliency_inv,
            high_crf,
            low_crf,
            encoder=encoder,
            saliency=saliency,
            target_bytes=target_bytes,
            target_bytes_tolerance=target_bytes_tolerance,
        )

    if mode != "uniward_texture":
        raise ValueError(f"unsupported saliency compression mode {mode!r}")
    if texture_probability is None:
        raise ValueError("mode='uniward_texture' requires texture_probability")
    if masks.ndim != 3 or masks.dtype != torch.uint8:
        raise ValueError(
            f"masks must be (N, H, W) uint8; got {tuple(masks.shape)} {masks.dtype}"
        )
    if texture_probability.ndim != 2:
        raise ValueError(
            f"texture_probability must be 2-D; got {tuple(texture_probability.shape)}"
        )
    if tuple(texture_probability.shape) != tuple(masks.shape[1:]):
        raise ValueError(
            f"shape mismatch: masks {tuple(masks.shape)} vs texture_probability "
            f"{tuple(texture_probability.shape)}"
        )
    if not 0.0 <= float(texture_quantile) <= 1.0:
        raise ValueError(f"texture_quantile must be in [0, 1]; got {texture_quantile}")
    if not 0 <= low_crf <= high_crf <= 63:
        raise ValueError(
            f"CRF must satisfy 0 <= low_crf ({low_crf}) <= high_crf "
            f"({high_crf}) <= 63"
        )

    tex = texture_probability.detach().float().cpu()
    threshold = torch.quantile(tex.flatten(), float(texture_quantile))
    aggressive_region = (tex >= threshold).bool()
    if encoder is None:
        encoder = globals().get("_default_zlib_encoder")
        if not callable(encoder):
            raise RuntimeError("UNIWARD patch missing default saliency encoder")
    encode_with_inv = globals().get("_encode_with_inv")
    if not callable(encode_with_inv):
        raise RuntimeError("UNIWARD patch missing saliency inverse encoder")
    return encode_with_inv(
        masks=masks,
        saliency_inv=aggressive_region,
        high_crf=high_crf,
        low_crf=low_crf,
        encoder=encoder,
    )


def _patch_saliency_inversion() -> None:
    try:
        import tac.saliency_inversion as saliency_inversion
    except Exception:
        return

    fn = saliency_inversion.apply_saliency_weighted_compression
    if getattr(fn, "_uniward_texture_patched", False):
        return

    original = types.FunctionType(
        fn.__code__,
        fn.__globals__,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    original.__kwdefaults__ = fn.__kwdefaults__

    saliency_inversion.__dict__["_UNIWARD_ORIGINAL_APPLY"] = original
    fn.__globals__["_UNIWARD_ORIGINAL_APPLY"] = original
    fn.__code__ = _patched_apply_saliency_weighted_compression.__code__
    fn.__defaults__ = _patched_apply_saliency_weighted_compression.__defaults__
    fn.__kwdefaults__ = _patched_apply_saliency_weighted_compression.__kwdefaults__
    fn.__annotations__ = _patched_apply_saliency_weighted_compression.__annotations__
    fn._uniward_texture_patched = True


_patch_saliency_inversion()
