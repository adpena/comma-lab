# SPDX-License-Identifier: MIT
"""Lovász hinge mask geometry — convex envelope of IoU/argmax disagreement.

T11 — Dykstra (CO-LEAD) eureka shower-thought (Fields-medal council 2026-05-09).
Complement to T7 (Fisher-Rao normalized simplex distance, already landed in
:mod:`tac.losses`):

* T7 measures Riemannian/information-geometric distance on the simplex.
* T11 measures the convex envelope of the IoU loss — i.e. the **tightest
  convex upper bound** on the contest scorer's argmax-disagreement metric.

The two are mathematically complementary:

* T7 gradients are well-behaved at distribution interiors; they vanish at
  the simplex boundary (where one class has p=1).
* T11 gradients are exactly the SUBGRADIENT of the (non-smooth, non-convex)
  IoU loss at the optimum, which means a trainer using T11 sees a gradient
  signal that is **geometrically tight to the contest scorer's argmax IoU**.

The Lovász extension of a submodular set function ``f: 2^[n] → ℝ`` is the
unique extension to the unit cube ``[0, 1]^n`` that is convex when ``f`` is
submodular and piecewise-linear with at most ``n!`` pieces. For the IoU
indicator loss, the closed-form extension is the **Lovász hinge** of Berman,
Triki & Blaschke 2018:

    L_lovasz(p, y) = ⟨m, ĝ_lovasz⟩

where:

* ``m_i ∈ [0, 1]^n`` are the per-pixel IoU "errors" (1 - p when y=1, p when
  y=0), sorted in DESCENDING order;
* ``ĝ_lovasz_i = J(π_1 ∪ … ∪ π_i) − J(π_1 ∪ … ∪ π_{i-1})`` is the discrete
  derivative of the Jaccard index ``J`` along the sorted permutation ``π``;
* ``⟨·, ·⟩`` is the standard inner product.

For the binary-foreground case, ``ĝ_lovasz`` admits a 4-line closed form
(Berman 2018 §3.1):

    ints(0)        = num_positives
    union(0)       = num_positives
    ints(i)        = num_positives - cumsum(errors_sorted * y_sorted)[i]
    union(i)       = num_positives + cumsum(errors_sorted * (1-y_sorted))[i]
    jaccard(i)     = 1 - ints(i) / union(i)
    g_lovasz_i     = jaccard(i) - jaccard(i-1)
    L_lovasz       = sum(errors_sorted * g_lovasz_i)

For the multi-class SegNet case (5 classes), we use the Lovász-Softmax
probability error **one-vs-rest per class** (the canonical multi-class
extension; Berman 2018 §3.2) and average. This preserves convexity per class
while avoiding the accidental factor-of-two that appears if probabilities are
first re-centered and then fed to a binary hinge.

Cross-references
----------------

* T7 (Fisher-Rao): :func:`tac.losses.segnet_fisher_rao_per_pixel`.
* Council memo: ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md``
  §"Dykstra (CO-LEAD)".
* Math citation: Berman, M., Triki, A. R. & Blaschke, M. B. 2018. *The Lovász-Softmax
  loss: A tractable surrogate for the optimization of the intersection-over-union
  measure in neural networks*, CVPR 2018.
* Original Lovász extension: Lovász, L. 1983. *Submodular functions and
  convexity*. Mathematical Programming: The State of the Art.

CLAUDE.md compliance
--------------------

* Pure-PyTorch differentiable surrogate; no scorer load, no MPS-falsification
  hazard (this is a TRAINING SIGNAL, not an authoritative score).
* All score-impact estimates are TAGGED ``[predicted; T11 Lovász hinge mask
  surrogate]`` per Forbidden Score Claims rule.
* The function is a TRAINING SIGNAL only; contest score still requires exact
  CUDA auth eval on the archive bytes.
"""
from __future__ import annotations

import math

import torch

DEFAULT_SEGNET_NUM_CLASSES = 5


def _validate_lovasz_eps(eps: float) -> float:
    """Validate the Lovász denominator-stability epsilon."""
    if isinstance(eps, bool):
        raise ValueError("lovasz_eps must be a finite number in (0, 1e-3)")
    try:
        value = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("lovasz_eps must be a finite number in (0, 1e-3)") from exc
    if not math.isfinite(value) or value <= 0.0 or value >= 1e-3:
        raise ValueError("lovasz_eps must be a finite number in (0, 1e-3)")
    return value


def lovasz_grad_jaccard(gt_sorted: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Compute the Lovász gradient of the Jaccard (IoU) loss.

    Implements the closed-form discrete derivative of the Jaccard index along
    a sorted-error permutation, per Berman 2018 §3.1 eqn (1).

    Args:
        gt_sorted: 1-D tensor of binary {0, 1} ground-truth labels sorted
            in DESCENDING order of per-pixel error magnitude. Shape ``(n,)``.
            Must be float (so the cumsum is differentiable through the
            autograd graph; gradients are passed through the SORT INDICES,
            not through the gt values).
        eps: Numerical-stability floor on the union denominator. Required
            because the union starts at zero when there are no positives;
            ``eps`` keeps the division finite.

    Returns:
        1-D tensor ``g_lovasz`` of shape ``(n,)``, the Lovász discrete
        derivative of the Jaccard index. Inner-product with the sorted
        error vector gives the Lovász hinge.

    Raises:
        ValueError: ``gt_sorted`` is not 1-D, or contains values outside
            ``{0, 1}``, or is empty.
    """
    eps_value = _validate_lovasz_eps(eps)
    if gt_sorted.ndim != 1:
        raise ValueError(
            f"gt_sorted must be 1-D; got shape {tuple(gt_sorted.shape)}"
        )
    if gt_sorted.numel() == 0:
        raise ValueError("gt_sorted must be non-empty")
    # Detach to scalar check; allow autograd-tracked tensors that happen
    # to hold {0,1} values (after .detach() they remain numerically valid).
    gt_check = gt_sorted.detach()
    invalid = ((gt_check != 0) & (gt_check != 1)).any().item()
    if invalid:
        raise ValueError(
            "gt_sorted must contain only {0, 1} values (binary labels)"
        )
    n_positives = gt_sorted.sum()
    # Cumulative true positives + false positives along sorted permutation.
    # As we walk down the sorted-error list, each pixel either is a positive
    # (decreases intersection) or a negative (increases union). This matches
    # the canonical Berman 2018 LovaszSoftmax reference implementation
    # (https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/
    # lovasz_losses.py::lovasz_grad).
    cum_tp = gt_sorted.cumsum(0)
    cum_fp = (1.0 - gt_sorted).cumsum(0)
    # Intersection at step i = (total positives) - (positives covered above i).
    intersection = n_positives - cum_tp
    # Union at step i = (total positives) + (negatives covered above i).
    union = n_positives + cum_fp
    jaccard = 1.0 - intersection / union.clamp_min(eps_value)
    # First element: jaccard(0) is the gradient at the lead position.
    # Remaining: discrete derivative jaccard(i) - jaccard(i-1).
    grad = jaccard.clone()
    if grad.numel() > 1:
        grad[1:] = jaccard[1:] - jaccard[:-1]
    return grad


def lovasz_hinge_binary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-pair Lovász hinge for one binary class.

    Implements Berman 2018 §3.1 (binary case): for predicted logits and
    binary {0, 1} labels, the Lovász hinge is the inner product of the
    sorted hinge errors with the Lovász gradient of the Jaccard index.

    Args:
        logits: Predicted class logits, shape ``(B, *)`` with B leading
            batch dim. Higher logit = stronger "foreground" prediction.
        labels: Ground-truth binary labels of the SAME shape as ``logits``,
            with values in ``{0, 1}``.
        eps: Forwarded to :func:`lovasz_grad_jaccard`.

    Returns:
        Scalar tensor: mean Lovász hinge over the batch.

    Raises:
        ValueError: shape mismatch, non-binary labels, or empty input.
    """
    eps_value = _validate_lovasz_eps(eps)
    if logits.shape != labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != labels shape "
            f"{tuple(labels.shape)}"
        )
    if logits.numel() == 0:
        raise ValueError("logits/labels must be non-empty")
    if labels.ndim < 1:
        raise ValueError(
            "labels must have at least 1 dimension (batch dim required); "
            f"got shape {tuple(labels.shape)}"
        )
    # Validate binary labels — fail loud per CLAUDE.md "fail loud, not silent".
    # Detach for the value check so autograd-tracked tensors aren't mutated.
    labels_check = labels.detach()
    invalid = ((labels_check != 0) & (labels_check != 1)).any().item()
    if invalid:
        raise ValueError(
            "labels must contain only {0, 1} values (binary labels); "
            "convert multi-class labels to one-vs-rest first"
        )
    # Per-pixel hinge error: max(0, 1 - signs * logits) where
    # signs = 2*labels - 1 ∈ {-1, +1}. This is the standard SVM-style
    # margin error; high-confidence correct predictions give zero error,
    # high-confidence wrong predictions give large error.
    signs = 2.0 * labels.float() - 1.0
    errors = (1.0 - signs * logits).clamp_min(0.0)
    return _lovasz_loss_from_errors(errors, labels.float(), eps=eps_value)


def _lovasz_loss_from_errors(
    errors: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Mean Lovász extension from precomputed per-pixel errors."""
    losses = []
    flat_errors = errors.reshape(errors.shape[0], -1)
    flat_labels = labels.reshape(labels.shape[0], -1).float()
    for b in range(flat_errors.shape[0]):
        err_b = flat_errors[b]
        lab_b = flat_labels[b]
        err_sorted, perm = err_b.sort(0, descending=True)
        lab_sorted = lab_b[perm]
        grad_b = lovasz_grad_jaccard(lab_sorted, eps=eps)
        losses.append(torch.dot(err_sorted, grad_b))
    return torch.stack(losses).mean()


def lovasz_hinge_mask_distortion(
    pred_probs: torch.Tensor,
    gt_probs: torch.Tensor,
    *,
    eps: float = 1e-6,
    num_classes: int = DEFAULT_SEGNET_NUM_CLASSES,
) -> torch.Tensor:
    """Multi-class Lovász hinge (one-vs-rest mean) on the SegNet probability simplex.

    The official scorer uses hard SegNet argmax disagreement. This helper is a
    differentiable convex-envelope proxy: the mean per-class Lovász hinge of
    the IoU between predicted softmax and ground-truth argmax (one-hot).

    The Lovász hinge is the **tightest convex upper bound** on the IoU
    indicator loss. Where T7 (Fisher-Rao) measures Riemannian-geometric
    distance on the simplex, T11 measures the convex envelope of the actual
    contest scorer's argmax-IoU metric.

    Args:
        pred_probs: ``(B, C, H, W)`` predicted softmax probabilities.
        gt_probs: ``(B, C, H, W)`` ground-truth softmax (or one-hot)
            probabilities. The argmax over class dim is used as the
            per-class binary label.
        eps: Numerical-stability floor on Jaccard denominators.
        num_classes: Expected channel count.

    Returns:
        Scalar tensor: mean per-class Lovász hinge.

    Raises:
        ValueError: shape mismatch, wrong class count, or invalid eps.
    """
    eps_value = _validate_lovasz_eps(eps)
    if pred_probs.shape != gt_probs.shape:
        raise ValueError(
            f"pred_probs shape {tuple(pred_probs.shape)} does not match "
            f"gt_probs shape {tuple(gt_probs.shape)}"
        )
    if pred_probs.ndim != 4:
        raise ValueError(
            "Lovász hinge expects BCHW probability tensors; "
            f"got shape {tuple(pred_probs.shape)}"
        )
    if pred_probs.shape[1] != num_classes:
        raise ValueError(
            f"Lovász hinge expects {num_classes} classes, "
            f"got {pred_probs.shape[1]}"
        )
    # Per-class one-vs-rest Lovász-Softmax. The binary hinge helper is for
    # real logits. Here the inputs are already probabilities, so the canonical
    # Berman 2018 multiclass error is |1{y=c} - p_c|, not a re-centered hinge.
    gt_argmax = gt_probs.argmax(dim=1)  # (B, H, W)
    per_class_losses = []
    for c in range(num_classes):
        label_c = (gt_argmax == c).float()
        errors_c = (label_c - pred_probs[:, c, ...]).abs()
        per_class_losses.append(_lovasz_loss_from_errors(errors_c, label_c, eps=eps_value))
    return torch.stack(per_class_losses).mean()


__all__ = [
    "lovasz_grad_jaccard",
    "lovasz_hinge_binary",
    "lovasz_hinge_mask_distortion",
]
