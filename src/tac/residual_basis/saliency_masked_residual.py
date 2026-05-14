# SPDX-License-Identifier: MIT
"""Saliency-masked residual encoder upgrade (L2 score-aware composition layer).

Per W's DEFERRED reactivation criterion #2 (memory
``feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md``) +
N's D2 council verdict + the sparse PacketIR codec (S's landing):

The L2 residual encoders currently emit DENSE residuals that the L1 inflate
runtime reads byte-for-byte. The sparse PacketIR family (0x20/0x21/0x22)
admits sparse-pattern-friendly residuals. This module composes a SCORE-AWARE
SALIENCY MASK over per-pixel residual coefficients: high-saliency pixels keep
their residual; low-saliency pixels are zeroed; bytes saved by the zeroed
regions appear as RLE-of-zeros in the sparse PacketIR output.

Mathematical formulation
------------------------

Per-pixel saliency::

    S(t, h, w) = || ∇_{decoded(t, h, w)} L_score(decoded, gt) ||

where ``L_score`` is the score-domain surrogate (KL-distill SegNet + MSE
PoseNet — i.e. the Hinton-distilled-scorer surrogate output, NOT a
weight-domain proxy per Catalog #123).

Saliency-masked residual::

    residual_masked(t, h, w, c) = {
        residual(t, h, w, c) if S(t, h, w) >= threshold
        0                    otherwise
    }

The threshold can be either a fixed scalar (per-archive global) or a
percentile (per-frame; e.g. keep top-25% saliency).

Per Catalog #123 (``check_no_weight_domain_saliency_on_score_gradient_substrate``):
saliency MUST be computed via score gradient on the substrate, NOT via
``mean(theta^2)`` / ``norm(theta)`` / ``var(theta)`` proxies. This module's
``compute_score_aware_saliency`` honors that constraint by routing through
``tac.residual_basis.hinton_distilled_scorer_surrogate.compute_distortion_via_distilled_scorer``
(which itself is gradient-reachable per HNeRV parity discipline lesson 8).

Composition with sparse PacketIR
--------------------------------

* Saliency masking ZEROS pixels in the residual before sparse-encoding.
* Sparse PacketIR (S's codec at 0x20/0x21/0x22) encodes zero runs efficiently
  (RLE of zeros + brotli on the remaining nonzero coefficients).
* Bytes saved by saliency masking are realized as compressed RLE runs.

Per the sparse-aware L2 encoder Lagrangian (W's landing), the inner loop
optimizes byte-cost-AFTER-sparse-encoding. Saliency masking is a NEW source
of sparsity the encoder can exploit alongside coefficient-magnitude
thresholding.

Public API
----------
``SaliencyMaskingConfig`` — frozen dataclass holding masking parameters
``compute_score_aware_saliency`` — per-pixel saliency map via score gradient
``mask_residual_by_saliency`` — apply saliency mask to a residual tensor
``compose_saliency_with_threshold_mask`` — combine per-pixel saliency with
                                          per-coefficient magnitude threshold
``SaliencyMaskedResidualError`` — typed error raised on contract violations

Cross-references
----------------
* W's DEFERRED memo: ``feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md``
* Sister Hinton-distilled surrogate:
  ``tac.residual_basis.hinton_distilled_scorer_surrogate``
* Sister score-gradient saliency (PARAMETER space):
  ``tac.score_gradient_param_saliency``
* Sister per-pixel saliency (older, contest-scorer-direct path):
  ``tac.saliency``
* Catalog #123 (forbidden weight-domain saliency):
  ``feedback_track4_bug_class_fix_self_protect_landed_20260509.md``
* HNeRV parity discipline lessons 6/8 (CLAUDE.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Optional

import torch

# Default saliency percentile keeping the top 25% most-score-relevant pixels.
# Source: PR106 r2 operating-point analysis — pose marginal value is 2.79x
# SegNet's, so keeping the top-quartile pixels by score-gradient magnitude is
# the operating-point-aware default.
DEFAULT_SALIENCY_PERCENTILE: Final[float] = 0.75  # 25th percentile = keep top 75%

# Default saliency threshold (absolute). Used when ``percentile`` is None.
# Conservative default: the threshold the L2 encoder's saliency analysis used
# 2026-05-11 (per W's reactivation criterion 2).
DEFAULT_SALIENCY_THRESHOLD: Final[float] = 0.0  # 0 = no mask by default

# Camera + scorer dimensions (canonical contest values).
CAMERA_H: Final[int] = 874
CAMERA_W: Final[int] = 1164
RGB_CHANNELS: Final[int] = 3


class SaliencyMaskedResidualError(ValueError):
    """Raised on contract violations in saliency-masked residual encoding."""


@dataclass(frozen=True)
class SaliencyMaskingConfig:
    """Frozen dataclass holding saliency-masking parameters.

    Per CLAUDE.md "no silent defaults" rule: every field is required at
    construction; ``__post_init__`` validates ranges.

    Attributes
    ----------
    threshold
        Absolute saliency threshold. Pixels with saliency >= threshold keep
        their residual; pixels below threshold are zeroed. Used when
        ``percentile`` is None.
    percentile
        If not None: per-frame keep-top-percentile-of-saliency. E.g. 0.75 keeps
        the top 25% most-score-relevant pixels per frame. Overrides ``threshold``
        when set. Must be in [0.0, 1.0).
    minimum_kept_fraction
        Refusal floor: if applying the threshold/percentile would leave fewer
        than this fraction of pixels kept, the encoder refuses (signals a
        contract violation: the saliency mask cannot legitimately zero ALL
        pixels). Default = 0.01 (must keep at least 1% of pixels per frame).
    per_channel_aggregation
        How to aggregate per-channel residual saliency into a single
        per-pixel scalar. One of:
        * ``"max"`` (default): max over channels (any one channel's saliency
          counts; conservative)
        * ``"mean"``: mean over channels
        * ``"sum"``: sum over channels
    """

    threshold: float
    percentile: Optional[float]
    minimum_kept_fraction: float
    per_channel_aggregation: str

    def __post_init__(self) -> None:
        if math.isnan(self.threshold) or math.isinf(self.threshold) or self.threshold < 0.0:
            raise SaliencyMaskedResidualError(
                f"threshold must be finite >= 0; got {self.threshold}"
            )
        if self.percentile is not None:
            if (
                math.isnan(self.percentile)
                or math.isinf(self.percentile)
                or not (0.0 <= self.percentile < 1.0)
            ):
                raise SaliencyMaskedResidualError(
                    f"percentile must be in [0.0, 1.0); got {self.percentile}"
                )
        if (
            math.isnan(self.minimum_kept_fraction)
            or math.isinf(self.minimum_kept_fraction)
            or not (0.0 < self.minimum_kept_fraction <= 1.0)
        ):
            raise SaliencyMaskedResidualError(
                f"minimum_kept_fraction must be in (0.0, 1.0]; "
                f"got {self.minimum_kept_fraction}"
            )
        if self.per_channel_aggregation not in ("max", "mean", "sum"):
            raise SaliencyMaskedResidualError(
                f"per_channel_aggregation must be 'max', 'mean', or 'sum'; "
                f"got {self.per_channel_aggregation!r}"
            )

    @classmethod
    def council_canonical(cls) -> "SaliencyMaskingConfig":
        """Return the council-canonical config (top-25% percentile, max-channel).

        This is the configuration L2 encoders default to when
        ``use_saliency_masking=True``.
        """
        return cls(
            threshold=DEFAULT_SALIENCY_THRESHOLD,
            percentile=DEFAULT_SALIENCY_PERCENTILE,
            minimum_kept_fraction=0.01,
            per_channel_aggregation="max",
        )


def compute_score_aware_saliency(
    decoded_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    distilled_segnet,
    distilled_posenet,
    eval_roundtrip: bool = True,
    distill_temperature: float = 2.0,
    score_seg_weight: float = 100.0,
    score_pose_weight: float = 1.0,
) -> torch.Tensor:
    """Per-pixel score-aware saliency via score gradient through distilled scorer.

    Computes the saliency map ``S(t, h, w) = || ∇_{decoded(t, h, w)} L_score ||``
    where ``L_score`` is the weighted sum of:

    * ``score_seg_weight * d_seg_distilled`` (KL-distill seg loss)
    * ``score_pose_weight * sqrt(10 * d_pose_distilled)`` (matches contest's
      pose contribution form)

    Per Catalog #123 (``check_no_weight_domain_saliency_on_score_gradient_substrate``):
    saliency is computed via INPUT-SPACE gradient — NOT via weight-domain
    proxies like ``mean(theta^2)``. The gradient flows from the score-domain
    surrogate back to the per-pixel decoded RGB inputs via autograd.

    Per HNeRV parity discipline lesson 8: routes through the canonical
    eval_roundtrip + differentiable YUV6 helpers via
    ``compute_distortion_via_distilled_scorer``.

    Parameters
    ----------
    decoded_rgb
        ``(B, 3, H, W)`` or ``(B, H, W, 3)`` or ``(H, W, 3)`` decoded frames.
        B must be even (frame pairs). The tensor will be cloned + given
        ``requires_grad_(True)`` so callers may pass a detached tensor.
    gt_rgb
        Same shape, ground-truth frames.
    distilled_segnet
        Loaded ``DistilledSegNet`` instance (frozen).
    distilled_posenet
        Loaded ``DistilledPoseNet`` instance (frozen).
    eval_roundtrip
        If True (default), routes through ``apply_eval_roundtrip_during_training``.
    distill_temperature
        Hinton T for the SegNet KL distill term. Council canon = 2.0.
    score_seg_weight
        Weight on the SegNet contribution. Default 100.0 (contest β).
    score_pose_weight
        Weight on the PoseNet contribution. Default 1.0 (contest γ).

    Returns
    -------
    torch.Tensor
        ``(B, H, W)`` per-pixel saliency map (non-negative; gradient norm
        across the 3 RGB channels). The saliency tensor is DETACHED (no
        gradient flows back through it — it's a per-pixel scalar map).
    """
    from tac.residual_basis.hinton_distilled_scorer_surrogate import (
        compute_distortion_via_distilled_scorer,
    )
    from tac.residual_basis.l2_score_aware_loss import (
        _coerce_rgb_to_bchw_float,
    )

    decoded_bchw = _coerce_rgb_to_bchw_float(decoded_rgb).clone().detach()
    gt_bchw = _coerce_rgb_to_bchw_float(gt_rgb).detach()
    if decoded_bchw.shape != gt_bchw.shape:
        raise SaliencyMaskedResidualError(
            f"decoded/gt shape mismatch: decoded={tuple(decoded_bchw.shape)} "
            f"gt={tuple(gt_bchw.shape)}"
        )
    if decoded_bchw.shape[0] % 2 != 0:
        raise SaliencyMaskedResidualError(
            f"decoded B must be even (frame pairs); got B={decoded_bchw.shape[0]}"
        )

    decoded_bchw.requires_grad_(True)

    d_seg, d_pose, _ = compute_distortion_via_distilled_scorer(
        decoded_bchw,
        gt_bchw,
        distilled_segnet=distilled_segnet,
        distilled_posenet=distilled_posenet,
        eval_roundtrip=eval_roundtrip,
        distill_temperature=distill_temperature,
    )

    # Score-domain loss = β · d_seg + γ · sqrt(10 · d_pose). Matches the
    # contest's pose contribution form so the saliency reflects the
    # operating-point marginal value.
    pose_term = torch.sqrt(torch.clamp(10.0 * d_pose, min=1e-12))
    score_loss = score_seg_weight * d_seg + score_pose_weight * pose_term
    score_loss.backward()

    if decoded_bchw.grad is None:
        raise SaliencyMaskedResidualError(
            "score-aware saliency: backward did not produce a gradient on "
            "decoded_rgb. The distilled scorer's gradient path is severed."
        )

    # Per-pixel saliency = gradient norm across RGB channels.
    grad_bchw = decoded_bchw.grad.detach()  # (B, 3, H, W)
    saliency_bhw = grad_bchw.abs().sum(dim=1)  # (B, H, W), positive scalar map
    return saliency_bhw


def _aggregate_channels(residual_bthwc: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce ``(B, H, W, C)`` channel dim to ``(B, H, W)`` via ``mode``."""
    abs_residual = residual_bthwc.abs()
    if mode == "max":
        return abs_residual.max(dim=-1).values
    if mode == "mean":
        return abs_residual.mean(dim=-1)
    if mode == "sum":
        return abs_residual.sum(dim=-1)
    raise SaliencyMaskedResidualError(f"unknown per-channel aggregation mode: {mode!r}")


def mask_residual_by_saliency(
    residual: torch.Tensor,
    saliency: torch.Tensor,
    *,
    config: SaliencyMaskingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply saliency mask to a residual tensor.

    Returns the masked residual + diagnostics dict. Per CLAUDE.md HNeRV parity
    discipline lesson 6 (score-domain Lagrangian): the masking is at the
    SCORE-DOMAIN level (pixels score-irrelevant per the distilled scorer are
    zeroed), NOT at the weight-domain level.

    Parameters
    ----------
    residual
        ``(B, H, W, C)`` (preferred) or ``(B, C, H, W)`` residual tensor.
        Will be returned in the same layout.
    saliency
        ``(B, H, W)`` per-pixel saliency map from ``compute_score_aware_saliency``.
        Must match residual's spatial dims.
    config
        Validated ``SaliencyMaskingConfig``.

    Returns
    -------
    (masked_residual, diagnostics)
        ``masked_residual``: same layout as ``residual``; pixels below threshold
                             zeroed.
        ``diagnostics``: dict[str, float] with kept-fraction + saliency stats
    """
    if not isinstance(config, SaliencyMaskingConfig):
        raise SaliencyMaskedResidualError(
            f"config must be SaliencyMaskingConfig; got {type(config).__name__}"
        )

    # Detect layout: (B, H, W, C) vs (B, C, H, W).
    if residual.dim() != 4:
        raise SaliencyMaskedResidualError(
            f"residual must be 4-dim; got {tuple(residual.shape)}"
        )
    if residual.shape[-1] == RGB_CHANNELS:
        residual_bthwc = residual
        layout = "bhwc"
    elif residual.shape[1] == RGB_CHANNELS:
        residual_bthwc = residual.permute(0, 2, 3, 1).contiguous()
        layout = "bchw"
    else:
        raise SaliencyMaskedResidualError(
            f"residual must have RGB channel dim ({RGB_CHANNELS}); "
            f"got shape {tuple(residual.shape)}"
        )

    if saliency.dim() != 3:
        raise SaliencyMaskedResidualError(
            f"saliency must be (B, H, W); got {tuple(saliency.shape)}"
        )
    if saliency.shape != residual_bthwc.shape[:3]:
        raise SaliencyMaskedResidualError(
            f"saliency shape {tuple(saliency.shape)} does not match "
            f"residual spatial {tuple(residual_bthwc.shape[:3])}"
        )

    # Compute per-pixel threshold mask.
    saliency_flat = saliency.reshape(saliency.shape[0], -1)  # (B, H*W)
    if config.percentile is not None:
        # Per-frame top-(1-percentile) keep.
        # Find the threshold value per frame such that exactly
        # (1 - percentile) fraction of pixels are kept (above threshold).
        n_pixels = saliency_flat.shape[1]
        # We want to keep (1 - percentile) fraction = the top scoring pixels.
        # quantile(percentile) is the cut-point: values >= quantile are kept.
        # Compute per-frame quantile.
        per_frame_threshold = torch.quantile(
            saliency_flat,
            q=float(config.percentile),
            dim=1,
            keepdim=True,
        )  # (B, 1)
        per_frame_threshold = per_frame_threshold.unsqueeze(-1)  # (B, 1, 1)
        mask_bhw = saliency >= per_frame_threshold
    else:
        mask_bhw = saliency >= config.threshold

    # Compute kept fraction per frame.
    kept_per_frame = mask_bhw.float().mean(dim=(1, 2))  # (B,)
    min_kept = float(kept_per_frame.min().item())
    if min_kept < config.minimum_kept_fraction:
        # Refuse: the mask would zero too many pixels (per the minimum_kept_fraction
        # contract). Either the saliency is degenerate or the threshold is too
        # aggressive — caller must back off.
        raise SaliencyMaskedResidualError(
            f"saliency mask kept_fraction={min_kept:.4f} < "
            f"minimum_kept_fraction={config.minimum_kept_fraction:.4f} "
            "(saliency mask too aggressive; the masked residual cannot legitimately "
            "zero this many pixels). Either reduce threshold, increase percentile, "
            "or relax minimum_kept_fraction."
        )

    # Apply the mask across channels.
    mask_bhwc = mask_bhw.unsqueeze(-1).expand(-1, -1, -1, residual_bthwc.shape[-1])
    masked_residual_bthwc = residual_bthwc * mask_bhwc.to(residual_bthwc.dtype)

    # Restore original layout.
    if layout == "bchw":
        masked_residual = masked_residual_bthwc.permute(0, 3, 1, 2).contiguous()
    else:
        masked_residual = masked_residual_bthwc

    # Diagnostics.
    saliency_min = float(saliency.min().item())
    saliency_max = float(saliency.max().item())
    saliency_mean = float(saliency.mean().item())
    kept_fraction_overall = float(mask_bhw.float().mean().item())
    n_zeroed_pixels = int((~mask_bhw).sum().item())
    n_total_pixels = int(mask_bhw.numel())

    diagnostics = {
        "saliency_min": saliency_min,
        "saliency_max": saliency_max,
        "saliency_mean": saliency_mean,
        "saliency_min_kept_fraction_per_frame": min_kept,
        "saliency_kept_fraction_overall": kept_fraction_overall,
        "saliency_n_zeroed_pixels": float(n_zeroed_pixels),
        "saliency_n_total_pixels": float(n_total_pixels),
        "use_saliency_masking": 1.0,
        "saliency_threshold": float(config.threshold),
        "saliency_percentile": float(config.percentile)
        if config.percentile is not None
        else -1.0,
    }
    return masked_residual, diagnostics


def compose_saliency_with_threshold_mask(
    residual: torch.Tensor,
    saliency: torch.Tensor,
    *,
    saliency_config: SaliencyMaskingConfig,
    magnitude_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compose saliency masking with per-coefficient magnitude thresholding.

    The two masks compose multiplicatively: a pixel is kept iff its saliency
    is above threshold AND its residual magnitude is above the magnitude
    threshold.

    Per the sparse-aware L2 encoder Lagrangian (W's landing), the inner loop
    optimizes byte-cost-AFTER-sparse-encoding. Both saliency masking and
    magnitude thresholding produce zero-runs that compress to RLE in the
    sparse PacketIR family (0x20/0x21/0x22).

    Parameters
    ----------
    residual
        ``(B, H, W, C)`` or ``(B, C, H, W)`` residual tensor.
    saliency
        ``(B, H, W)`` per-pixel saliency map.
    saliency_config
        Saliency masking config.
    magnitude_threshold
        Absolute residual coefficient magnitude threshold. Coefficients with
        absolute value below this are zeroed.

    Returns
    -------
    (composed_masked_residual, diagnostics)
        Both masks applied; diagnostics include kept-fractions for each mask.
    """
    if magnitude_threshold < 0.0 or math.isnan(magnitude_threshold) or math.isinf(
        magnitude_threshold
    ):
        raise SaliencyMaskedResidualError(
            f"magnitude_threshold must be finite >= 0; got {magnitude_threshold}"
        )

    # Step 1: apply saliency mask.
    saliency_masked, saliency_diag = mask_residual_by_saliency(
        residual, saliency, config=saliency_config
    )

    # Step 2: apply magnitude threshold mask.
    if saliency_masked.shape[-1] == RGB_CHANNELS:
        layout = "bhwc"
        residual_bthwc = saliency_masked
    else:
        layout = "bchw"
        residual_bthwc = saliency_masked.permute(0, 2, 3, 1).contiguous()

    magnitude_mask = residual_bthwc.abs() >= magnitude_threshold
    composed_bthwc = residual_bthwc * magnitude_mask.to(residual_bthwc.dtype)

    if layout == "bchw":
        composed = composed_bthwc.permute(0, 3, 1, 2).contiguous()
    else:
        composed = composed_bthwc

    n_zeroed_by_magnitude = int((~magnitude_mask).sum().item())
    n_total_coefficients = int(magnitude_mask.numel())

    diagnostics = dict(saliency_diag)
    diagnostics["magnitude_threshold"] = float(magnitude_threshold)
    diagnostics["magnitude_n_zeroed_coefficients"] = float(n_zeroed_by_magnitude)
    diagnostics["magnitude_n_total_coefficients"] = float(n_total_coefficients)
    diagnostics["composed_kept_fraction"] = float(
        (composed != 0).float().mean().item()
    )
    return composed, diagnostics


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "DEFAULT_SALIENCY_PERCENTILE",
    "DEFAULT_SALIENCY_THRESHOLD",
    "RGB_CHANNELS",
    "SaliencyMaskedResidualError",
    "SaliencyMaskingConfig",
    "compose_saliency_with_threshold_mask",
    "compute_score_aware_saliency",
    "mask_residual_by_saliency",
]
