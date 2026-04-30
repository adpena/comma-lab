"""Lane 19 — SegNet logit-margin boundary loss (Fridrich + Yousfi lead).

Per Phase 2 Lane 19 spec (memory project_phases_2_3_4_*):

    Compute SegNet logit gradients at compress time (UNLIMITED). Identify
    pixels where the second-best class is within margin ε of argmax — these
    are FRAGILE pixels where score-loss is largest if encoding flips them.
    Allocate compression bits proportional to fragility.

    Specifically: for each pixel, compute `margin = top1_logit - top2_logit`.
    Sort pixels ascending by margin. Encode top-N% (smallest margin = most
    fragile) with highest fidelity, encode rest with low fidelity.

This module implements the loss-function side of the Lane 19 idea: instead
of CE loss on argmax mismatches, use a logit-margin-weighted loss that
concentrates learning on uncertain boundaries (Fridrich-style: where the
steganalysis-equivalent SegNet is least confident).

Math foundation
---------------
For each pixel, given logits z ∈ R^K:
    top1 = argmax(z)
    top2 = argmax(z) excluding top1
    margin = z[top1] - z[top2]

Define a fragility weight w(margin):
    w(margin) = clamp((threshold - margin) / threshold, 0, 1)

Pixels with margin >= threshold get weight 0 (confident; no learning signal).
Pixels with margin < threshold get weight ∈ (0, 1] linearly increasing as
margin shrinks. Pixels with margin = 0 (tied logits) get weight 1.

The loss is:
    L = mean over pixels [w(margin) * CE(z, gt)]

Where CE(z, gt) is standard cross-entropy. The Fridrich principle: spend
gradient updates on the boundary tokens that ACTUALLY matter; ignore confident
correct predictions; ignore confident wrong predictions (they are likely
already saturated and not worth the gradient).

CLAUDE.md compliance
--------------------
- Compress-time training-only loss; never loaded at inflate time
- No silent defaults — threshold + reduction are required keywords
- No scorer load (loss is computed on PROVIDED logits + gt; this module does
  not load SegNet or PoseNet itself)
- All claims tagged [synthetic] / [prediction]
- No GPU dependency for the loss computation; runs on whatever device the
  input logits are on

Predicted Phase 2 EV: 80-300 bp — score-aware compression via gradient
margins; novel paper section per memory entry. Empirical confirm needed
on a real Lane G v3 archive.

References
----------
* Fridrich UNIWARD framework: errors in textured/uncertain regions are
  undetectable; concentrate signal where the detector is least confident
* Yousfi 2022: detector-informed embedding (TTO uses SegNet as the
  detector; this lane operationalises that via the logit margin)
* memory: project_phases_2_3_4_design_implementation_math_provenance §"Lane 19"
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def fragility_weights(
    logits: torch.Tensor,
    threshold: float | None = None,
) -> torch.Tensor:
    """Compute per-pixel fragility weights based on top1-top2 logit margin.

    Args:
        logits: (N, K, ...) tensor — class logits per pixel. Required.
            N is batch (or batch*pixels), K is number of classes. Trailing
            dims are spatial; preserved in output.
        threshold: positive scalar — margin above which weight=0 (confident
            pixel; no learning signal). Required (no silent default per
            Check 81 STRICT). Typical values: 0.5 to 2.0 depending on
            logit scale.

    Returns:
        (N, ...) tensor of weights in [0, 1]. Spatial dims match input
        (the K dim is contracted via the top-2 calculation).

    Raises:
        ValueError: bad input shape, non-positive threshold.
    """
    if threshold is None:
        raise ValueError(
            "fragility_weights: threshold is required (no silent default — "
            "Check 81 STRICT). Pass an explicit positive scalar matching "
            "the logit scale."
        )
    if threshold <= 0:
        raise ValueError(
            f"fragility_weights: threshold must be > 0; got {threshold}"
        )
    if logits.ndim < 2:
        raise ValueError(
            f"fragility_weights: logits must be >= 2-D (N, K, ...); "
            f"got shape {tuple(logits.shape)}"
        )
    K = int(logits.shape[1])
    if K < 2:
        raise ValueError(
            f"fragility_weights: K (num_classes) must be >= 2; got {K}"
        )
    # top-2 across class dim. torch.topk returns sorted desc; values: (..., 2)
    # We move K to last dim, topk, then move back.
    # Alternative (simpler): use logits.transpose(1, -1) then topk on -1.
    # Robust path: flatten spatial → (N, K, P), topk on dim=1.
    orig_shape = logits.shape
    if logits.ndim == 2:
        # (N, K) — no spatial dims
        top2_vals, _ = torch.topk(logits, k=2, dim=1)  # (N, 2)
        margin = top2_vals[:, 0] - top2_vals[:, 1]  # (N,)
    else:
        # (N, K, ...) — flatten spatial
        flat = logits.reshape(orig_shape[0], K, -1)  # (N, K, P)
        top2_vals, _ = torch.topk(flat, k=2, dim=1)  # (N, 2, P)
        margin = top2_vals[:, 0, :] - top2_vals[:, 1, :]  # (N, P)
        margin = margin.reshape(orig_shape[0], *orig_shape[2:])  # (N, ...)

    # Clamp margin to [0, threshold]; weight = (threshold - margin) / threshold ∈ [0, 1]
    clamped = margin.clamp(min=0.0, max=float(threshold))
    weights = (float(threshold) - clamped) / float(threshold)
    return weights


def logit_margin_loss(
    logits: torch.Tensor | None = None,
    gt_argmax: torch.Tensor | None = None,
    *,
    threshold: float | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Logit-margin-weighted cross-entropy.

    Loss = mean/sum over pixels [w(margin) * CE(logits, gt_argmax)]

    Where w(margin) = clamp((threshold - margin) / threshold, 0, 1) — pixels
    with confident predictions (margin >= threshold) contribute zero loss;
    pixels with ambiguous predictions (margin < threshold) contribute
    proportional to their ambiguity.

    Args:
        logits: (N, K, ...) class logits. Required.
        gt_argmax: (N, ...) ground-truth class IDs (integer dtype). Required.
        threshold: positive scalar; pixels with margin >= threshold get weight 0.
            Required (no silent default — Check 81 STRICT).
        reduction: "mean" / "sum" / "none". Required as keyword (no silent
            default for the loss-aggregation choice — Check 81 STRICT).

    Returns:
        Scalar loss (mean / sum) or (N, ...) per-pixel loss (reduction="none").

    Raises:
        ValueError: bad shapes / dtypes / threshold.
    """
    if logits is None or gt_argmax is None:
        raise ValueError(
            "logit_margin_loss: logits and gt_argmax are required (no silent "
            "default — Check 81 STRICT)."
        )
    if threshold is None:
        raise ValueError(
            "logit_margin_loss: threshold is required (no silent default — "
            "Check 81 STRICT)."
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"logit_margin_loss: reduction must be 'mean'/'sum'/'none'; got {reduction!r}"
        )
    if logits.ndim < 2:
        raise ValueError(
            f"logit_margin_loss: logits must be >= 2-D (N, K, ...); got {tuple(logits.shape)}"
        )
    if gt_argmax.ndim != logits.ndim - 1:
        raise ValueError(
            f"logit_margin_loss: gt_argmax must be (N, ...) matching "
            f"logits without K; got logits {tuple(logits.shape)} + "
            f"gt {tuple(gt_argmax.shape)}"
        )
    if not gt_argmax.dtype.is_floating_point:
        gt_argmax_long = gt_argmax.long()
    else:
        gt_argmax_long = gt_argmax.long()
    # CE per-pixel
    ce = F.cross_entropy(logits, gt_argmax_long, reduction="none")  # (N, ...)
    # Fragility weights per-pixel
    weights = fragility_weights(logits, threshold=threshold)  # (N, ...)
    if ce.shape != weights.shape:
        raise ValueError(
            f"logit_margin_loss: internal shape mismatch ce={tuple(ce.shape)} "
            f"weights={tuple(weights.shape)}"
        )
    weighted = ce * weights
    if reduction == "mean":
        return weighted.mean()
    if reduction == "sum":
        return weighted.sum()
    return weighted


__all__ = [
    "fragility_weights",
    "logit_margin_loss",
]
