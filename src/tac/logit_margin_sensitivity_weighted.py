"""Lane 19 β-variant — sensitivity-weighted SegNet logit-margin loss.

Paradigm β extension of :mod:`tac.losses_logit_margin`. Vanilla Lane 19
weights the cross-entropy loss by ``(threshold − margin)/threshold`` so
boundary pixels (small top1−top2 margin) receive maximum weight. This
β-variant additionally consumes a per-pixel score-sensitivity tensor and
multiplies the boundary weight with it:

    L_β = mean over pixels [w_margin(margin) * w_sens(p) * CE(z, gt)]

where ``w_sens(p) = (sensitivity(p) / sensitivity_norm)`` is a normalized
per-pixel sensitivity. The composite weight ``w_margin × w_sens`` only
fires for pixels that are SIMULTANEOUSLY (a) close to a boundary AND (b)
high score-sensitivity. This concentrates compress-time gradient updates
on the pixels that actually move the contest score — Fridrich's UNIWARD
plus Yousfi's detector-informed embedding, applied at the loss level.

Math foundation
---------------
For a pixel ``p`` with logits ``z`` and ground-truth class ``gt``:

    margin(p)       = z[top1] − z[top2]
    w_margin(p)     = clamp((threshold − margin) / threshold, 0, 1)   ∈ [0, 1]
    w_sens(p)       = sensitivity(p) / max(sensitivity)                 ∈ [0, 1]
    w_combined(p)   = w_margin(p) * w_sens(p)
    L_β             = mean_p [ w_combined(p) * CE(z, gt) ]

Without ``w_sens``, vanilla Lane 19 spends gradient updates on boundary
pixels uniformly. But not every boundary pixel is equally score-relevant:
pixels in low-sensitivity regions of the SegNet input are score-blind even
when classified ambiguously. The β weighting suppresses those updates so
the loss budget concentrates on high-sensitivity boundaries.

CLAUDE.md compliance
--------------------
* Compress-time training-only loss; never loaded at inflate time.
* No silent defaults — threshold + reduction + sensitivity_norm are required.
* No scorer load — operates on pre-computed sensitivity tensor.
* Sensitivity tensor is DETACHED before multiplication — no gradient flows
  back through the per-pixel sensitivity (it is data, not a parameter).

References
----------
* :mod:`tac.losses_logit_margin` — vanilla Lane 19 loss.
* :mod:`tac.sensitivity_map` — sensitivity contract (per-channel, but the
  per-pixel variant here follows the same non-negative-finite contract).
* ``.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md``
  §"Paradigm Shift β" — math foundation.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from tac.losses_logit_margin import fragility_weights


class SensitivityWeightedLogitMarginError(ValueError):
    """Raised when sensitivity-weighted logit-margin loss inputs are malformed."""


def _validate_pixel_sensitivity(
    sensitivity: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    """Validate per-pixel sensitivity and detach to float32."""
    if not torch.is_tensor(sensitivity):
        raise SensitivityWeightedLogitMarginError(
            f"{name}: sensitivity must be a torch.Tensor, got "
            f"{type(sensitivity).__name__}"
        )
    if tuple(sensitivity.shape) != tuple(expected_shape):
        raise SensitivityWeightedLogitMarginError(
            f"{name}: sensitivity shape {tuple(sensitivity.shape)} does not "
            f"match expected {expected_shape}"
        )
    out = sensitivity.detach().to(torch.float32)
    if not torch.isfinite(out).all():
        n_bad = int((~torch.isfinite(out)).sum().item())
        raise SensitivityWeightedLogitMarginError(
            f"{name}: sensitivity contains {n_bad} NaN/Inf value(s)"
        )
    if (out < 0).any():
        raise SensitivityWeightedLogitMarginError(
            f"{name}: sensitivity must be non-negative"
        )
    return out


def normalize_sensitivity(
    sensitivity: torch.Tensor,
    *,
    mode: Literal["max", "sum", "none"] = "max",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize per-pixel sensitivity to a [0, 1] (or [0, ∞)) weight.

    Args:
        sensitivity: non-negative tensor (any shape).
        mode: ``"max"`` → divide by max (gives [0, 1]); ``"sum"`` → divide by
            sum (gives weights summing to 1); ``"none"`` → return unchanged.
        eps: safety floor on the divisor.

    Returns:
        Normalized tensor with the same shape.

    Raises:
        SensitivityWeightedLogitMarginError: bad mode / NaN-Inf input /
        negative values.
    """
    if mode not in {"max", "sum", "none"}:
        raise SensitivityWeightedLogitMarginError(
            f"normalize_sensitivity: mode must be max/sum/none; got {mode!r}"
        )
    if not torch.is_tensor(sensitivity):
        raise SensitivityWeightedLogitMarginError(
            "normalize_sensitivity: input must be a torch.Tensor"
        )
    s = sensitivity.detach().to(torch.float32)
    if not torch.isfinite(s).all():
        raise SensitivityWeightedLogitMarginError(
            "normalize_sensitivity: input contains NaN/Inf"
        )
    if (s < 0).any():
        raise SensitivityWeightedLogitMarginError(
            "normalize_sensitivity: input must be non-negative"
        )
    if mode == "none":
        return s
    if mode == "max":
        denom = s.max().clamp_min(eps)
    else:
        denom = s.sum().clamp_min(eps)
    return s / denom


def sensitivity_weighted_logit_margin_loss(
    logits: torch.Tensor | None = None,
    gt_argmax: torch.Tensor | None = None,
    *,
    pixel_sensitivity: torch.Tensor | None = None,
    threshold: float | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
    sensitivity_norm: Literal["max", "sum", "none"] = "max",
) -> torch.Tensor:
    """Lane 19 β-variant: per-pixel sensitivity × margin-weighted CE.

    L_β = mean / sum / none over pixels of:
        w_margin(margin) * w_sens(pixel_sensitivity) * CE(logits, gt_argmax)

    Args:
        logits: (N, K, ...) class logits. Required.
        gt_argmax: (N, ...) integer ground-truth labels. Required.
        pixel_sensitivity: (N, ...) per-pixel score sensitivity, matching
            the spatial shape of ``gt_argmax``. Required (no silent default
            — Check 81 STRICT). Must be non-negative and finite.
        threshold: positive scalar threshold for the margin clamping.
            Required.
        reduction: "mean" / "sum" / "none". Required.
        sensitivity_norm: how to normalize the sensitivity map; default
            ``"max"``.

    Returns:
        Scalar (mean/sum) or per-pixel (none) tensor.

    Raises:
        SensitivityWeightedLogitMarginError: bad inputs.
    """
    if logits is None or gt_argmax is None:
        raise SensitivityWeightedLogitMarginError(
            "sensitivity_weighted_logit_margin_loss: logits and gt_argmax "
            "are required"
        )
    if pixel_sensitivity is None:
        raise SensitivityWeightedLogitMarginError(
            "sensitivity_weighted_logit_margin_loss: pixel_sensitivity is "
            "required (no silent default — Check 81 STRICT)"
        )
    if threshold is None:
        raise SensitivityWeightedLogitMarginError(
            "sensitivity_weighted_logit_margin_loss: threshold is required"
        )
    if reduction not in {"mean", "sum", "none"}:
        raise SensitivityWeightedLogitMarginError(
            f"reduction must be mean/sum/none; got {reduction!r}"
        )
    if logits.ndim < 2:
        raise SensitivityWeightedLogitMarginError(
            f"logits must be >= 2-D (N, K, ...); got {tuple(logits.shape)}"
        )
    if gt_argmax.ndim != logits.ndim - 1:
        raise SensitivityWeightedLogitMarginError(
            f"gt_argmax shape {tuple(gt_argmax.shape)} does not match "
            f"logits without K dim ({tuple(logits.shape)})"
        )

    expected_pixel_shape = tuple(gt_argmax.shape)
    sens = _validate_pixel_sensitivity(
        pixel_sensitivity,
        expected_shape=expected_pixel_shape,
        name="pixel_sensitivity",
    )
    sens = normalize_sensitivity(sens, mode=sensitivity_norm)

    # Move sens onto logits' device for elementwise multiplication.
    sens = sens.to(logits.device)

    # Compute fragility (margin) weights using the existing Lane 19 helper.
    w_margin = fragility_weights(logits, threshold=threshold).detach()
    if w_margin.shape != expected_pixel_shape:
        raise SensitivityWeightedLogitMarginError(
            f"fragility weight shape {tuple(w_margin.shape)} does not match "
            f"gt_argmax shape {expected_pixel_shape}"
        )

    # Cross-entropy per-pixel.
    gt_long = gt_argmax.long()
    ce = F.cross_entropy(logits, gt_long, reduction="none")  # (N, ...)

    weighted = ce * w_margin * sens

    if reduction == "mean":
        # MEAN over the full tensor matches the vanilla Lane 19 convention.
        return weighted.mean()
    if reduction == "sum":
        return weighted.sum()
    return weighted


__all__ = [
    "SensitivityWeightedLogitMarginError",
    "normalize_sensitivity",
    "sensitivity_weighted_logit_margin_loss",
]
