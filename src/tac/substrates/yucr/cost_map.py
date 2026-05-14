# SPDX-License-Identifier: MIT
"""YUCR cost-map computation — Atick-Redlich orthogonal-complement projector.

The cost map ``C(x, y)`` is the per-pixel inverse-detectability of the
contest scorer:

.. math::

    C(x, y) = 1 / (||grad_pixel d_seg|| + sqrt(10) * ||grad_pixel d_pose|| + eps)

where ``d_seg`` and ``d_pose`` come from the canonical
:func:`tac.substrates.score_aware_common.score_pair_components` per
Catalog #164. Pixels with HIGH ``C`` are scorer-BLIND — quantization noise
placed there is "free" in score-space. Pixels with LOW ``C`` are
scorer-SENSITIVE — quantization noise there moves the score.

This is mathematically equivalent to the Atick-Redlich 1990 (Neural
Computation 2:308-320) orthogonal-complement projector. The encoder needs
``H(X | scorer)``; ``1/C`` is the LOCAL importance density. UNIWARD-STC
allocation (sister module :mod:`tac.substrates.yucr.stc_encoder`) does the
constructive water-fill.

All cost-map computation routes through the canonical scorer contract and
honors :class:`tac.differentiable_eval_roundtrip.DifferentiableEvalRoundtrip`
gradient-reachability per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE".

NO score claims. The cost map is a PROXY signal — the resulting STC payload
must still be auth-evaluated on contest-CUDA + contest-CPU before any
promotion / ranking / kill verdict.
"""

from __future__ import annotations

import math
from enum import Enum

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_SEG_WEIGHT,
    score_pair_components,
)


class YUCRCostMapMode(Enum):
    """How to derive per-pixel cost from the scorer pair components."""

    SCORE_GRADIENT = "score_gradient"
    """Atick-Redlich canonical: compute ||grad_pixel d_seg|| + sqrt(10) * ||grad_pixel d_pose||."""

    UNIFORM = "uniform"
    """Uniform cost — degenerates STC to plain water-fill."""

    DUMMY_CONSTANT = "dummy_constant"
    """Test-only constant cost map; ``compute_cost_map_dummy`` is the canonical helper."""


def compute_cost_map(
    *,
    seg_scorer: torch.nn.Module,
    pose_scorer: torch.nn.Module,
    rgb_0_rt: torch.Tensor,
    rgb_1_rt: torch.Tensor,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    target_resolution: tuple[int, int] = (384, 512),
    pose_sqrt_weight: float = CONTEST_POSE_SQRT_WEIGHT,
    seg_weight: float = CONTEST_SEG_WEIGHT,
    eps: float = 1e-6,
    reduce_pair: str = "mean",
    detach_grad: bool = True,
) -> torch.Tensor:
    """Compute per-pixel cost map via Atick-Redlich orthogonal-complement projector.

    Routes through canonical :func:`score_pair_components` per Catalog #164
    so the scorer preprocessing pipeline (rgb_to_yuv6, eval_roundtrip,
    differentiable preprocess) is consistent with every sister substrate.

    Args:
        seg_scorer, pose_scorer: Upstream contest scorers (must expose
            ``preprocess_input``).
        rgb_0_rt, rgb_1_rt: Roundtripped reconstruction of the two paired
            frames, ``(B, 3, H, W)``. Must have ``requires_grad=True`` so we
            can backprop the score against the pixel inputs.
        gt_rgb_0, gt_rgb_1: Ground-truth pair, ``(B, 3, H, W)``.
        target_resolution: Spatial resolution of the returned cost map.
            Defaults to scorer eval resolution (384, 512) — bilinear-resampled
            if ``rgb_*`` is at a different resolution.
        pose_sqrt_weight: Pose contribution weight per the contest formula.
            ``sqrt(10)`` matches :data:`CONTEST_POSE_SQRT_WEIGHT`.
        seg_weight: Segmentation contribution weight per the contest formula.
            ``100.0`` matches :data:`CONTEST_SEG_WEIGHT`.
        eps: Eps-clamp to avoid div-by-zero in the inverse-detectability map.
        reduce_pair: How to reduce the per-pair cost into a single map.
            ``"mean"`` averages frame_0 + frame_1 grads; ``"max"`` takes the
            elementwise max (more conservative on pose-sensitive pixels).
        detach_grad: If True (default), the returned cost map has no grad —
            it's used for STC allocation, not for backprop. Set False only
            for a research probe of cost-map-aware loss.

    Returns:
        Cost map ``(B, H, W)`` float32 (or shape ``target_resolution``).
        Higher values = scorer-blind = cheaper to embed noise.

    Raises:
        ValueError: when input shapes don't match.
        RuntimeError: when scorers don't propagate gradients to RGB inputs.
    """

    if rgb_0_rt.shape != rgb_1_rt.shape:
        raise ValueError(
            f"compute_cost_map: rgb_0_rt and rgb_1_rt shapes differ: "
            f"{tuple(rgb_0_rt.shape)} vs {tuple(rgb_1_rt.shape)}"
        )
    if rgb_0_rt.dim() != 4 or rgb_0_rt.shape[1] != 3:
        raise ValueError(
            f"compute_cost_map: rgb_0_rt must be (B, 3, H, W); got {tuple(rgb_0_rt.shape)}"
        )
    if not (rgb_0_rt.requires_grad and rgb_1_rt.requires_grad):
        raise RuntimeError(
            "compute_cost_map: rgb_0_rt and rgb_1_rt must have requires_grad=True; "
            "the cost map is derived from per-pixel score gradients."
        )
    if reduce_pair not in {"mean", "max"}:
        raise ValueError(f"reduce_pair must be 'mean' or 'max'; got {reduce_pair!r}")

    seg_term, pose_term = score_pair_components(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        rgb_0_rt=rgb_0_rt,
        rgb_1_rt=rgb_1_rt,
        gt_rgb_0=gt_rgb_0,
        gt_rgb_1=gt_rgb_1,
    )
    # Combine seg + pose into the contest-formula weighted distortion. We
    # use sqrt(pose) per the contest aggregation rule (sqrt(10 * pose_avg))
    # so the per-pixel grads reflect the marginal score impact at the
    # frontier operating point (per CLAUDE.md "SegNet vs PoseNet importance —
    # operating-point dependent" non-negotiable).
    pose_sqrt = torch.sqrt(pose_term.clamp_min(eps))
    distortion_scalar = seg_weight * seg_term + pose_sqrt_weight * pose_sqrt

    # Backprop distortion against the RGB inputs to get per-pixel sensitivity.
    grads = torch.autograd.grad(
        outputs=distortion_scalar.sum(),
        inputs=[rgb_0_rt, rgb_1_rt],
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    grad_0, grad_1 = grads  # each (B, 3, H, W)

    # Per-pixel L2 norm across channels.
    sens_0 = grad_0.pow(2).sum(dim=1).clamp_min(0.0).sqrt()  # (B, H, W)
    sens_1 = grad_1.pow(2).sum(dim=1).clamp_min(0.0).sqrt()

    if reduce_pair == "mean":
        sensitivity = 0.5 * (sens_0 + sens_1)
    else:
        sensitivity = torch.maximum(sens_0, sens_1)

    # Cost = 1 / (sensitivity + eps). Higher cost = scorer-blind = cheap to embed.
    # We CLAMP rather than divide-by-tiny so cost map stays bounded for STC.
    sens_clamped = sensitivity.clamp_min(eps)
    cost_map = 1.0 / sens_clamped

    # Resample to target resolution if needed.
    target_h, target_w = target_resolution
    if cost_map.shape[-2:] != (target_h, target_w):
        cost_map = torch.nn.functional.interpolate(
            cost_map.unsqueeze(1),  # (B, 1, H, W)
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    if detach_grad:
        cost_map = cost_map.detach()

    return cost_map.contiguous()


def compute_cost_map_dummy(
    resolution: tuple[int, int] = (384, 512),
    constant_value: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Test-only constant cost map. NEVER use in a real training path.

    Uniform cost degenerates STC to plain water-fill (every pixel equally
    cheap). Useful for archive grammar roundtrip tests + STC encoder unit
    tests where we want to isolate the allocator from the cost-map-grad
    path. NEVER pass through :func:`compose_with_base` for a real archive.
    """
    if constant_value <= 0:
        raise ValueError(f"constant_value must be > 0; got {constant_value}")
    h, w = resolution
    return torch.full((h, w), constant_value, device=device, dtype=dtype)


def quantize_cost_map_int8(
    cost_map: torch.Tensor,
    *,
    scale: float = 127.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Quantize cost map to int8 for archive storage.

    The returned ``(int8_map, recovered_scale)`` lets inflate-time recover
    the float cost map deterministically:
    ``cost_map_recovered = int8_map.float() / recovered_scale * cost_max``.

    Args:
        cost_map: Float tensor (typically (H, W) or (B, H, W)).
        scale: Target int8 dynamic range (typically 127 for full-range int8).
        eps: Eps-clamp for div-by-zero.

    Returns:
        Tuple ``(int8_quantized, recovered_scale)`` where
        ``recovered_scale = max_value / scale`` so dequantization is
        ``int8_value * recovered_scale``.

    Raises:
        ValueError: when ``scale`` is out of range.
    """
    if scale <= 0 or scale > 127:
        raise ValueError(f"scale must be in (0, 127]; got {scale}")
    cost_max = cost_map.detach().abs().max().clamp_min(eps).item()
    recovered_scale = cost_max / scale
    int8 = (
        (cost_map.detach() / recovered_scale)
        .clamp(-128.0, 127.0)
        .round()
        .to(torch.int8)
    )
    return int8, recovered_scale


def dequantize_cost_map_int8(
    int8_map: torch.Tensor,
    *,
    recovered_scale: float,
) -> torch.Tensor:
    """Inverse of :func:`quantize_cost_map_int8`."""
    if recovered_scale <= 0:
        raise ValueError(
            f"recovered_scale must be > 0; got {recovered_scale}. The encoder side "
            "stored a non-positive cost-map scale. Refuse the archive."
        )
    return int8_map.to(torch.float32) * recovered_scale


__all__ = [
    "YUCRCostMapMode",
    "compute_cost_map",
    "compute_cost_map_dummy",
    "dequantize_cost_map_int8",
    "quantize_cost_map_int8",
]
