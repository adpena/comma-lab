"""Lane J-JBL: Jaccard Metric Loss + Boundary Label Smoothing distillation.

Implements the Jack-from-skunkworks Cycle 1 TOP-1 SegNet attack
(see ``.omx/research/jack_skunkworks_segnet_rate_research_20260428.md``
section S1).

Reference: Wang et al., "Jaccard Metric Losses: Optimizing the Jaccard
Index with Soft Labels" (arXiv 2302.05666, NeurIPS 2023, refined 2024).
JML is a soft-label-compatible Jaccard surrogate that subsumes
Lovász-Softmax. Combined with Boundary Label Smoothing (BLS) which
injects "dark knowledge" at class boundaries, the paper reports +2-5
mIoU across 13 architectures including EfficientNet — directly
relevant to the comma SegNet (EfficientNet-B2 backbone) whose argmax
distortion is the LARGEST score wedge (38% of Lane G v3 = 1.05).

Math (Wang et al. Eq. 4): the soft Jaccard surrogate is the quadratic
relaxation

    JML(p, q) = 1 - mean_c [ <p_c, q_c> / (||p_c||² + ||q_c||² - <p_c, q_c>) ]

where p, q are softmax distributions, c indexes classes, and the inner
product / squared norms are over spatial pixels. At p == q the
intersection equals each squared norm so JML == 0 (perfect match);
disagreement increases the loss monotonically.

BLS:
    boundary_mask = (max_pool_kxk(target_hard) != min_pool_kxk(target_hard))
    soft_target[boundary] = (1-eps) * one_hot + eps/C * uniform
    soft_target[interior] = one_hot  (unchanged)

The combined ``combined_jbl_distill_loss`` consumes:
  (a) JML on student vs teacher soft labels (the "dark knowledge"
      distillation channel);
  (b) cross-entropy on student vs GT-with-BLS, with boundary pixels
      weighted ``boundary_pixel_weight`` × interior pixels (the
      "boundary precision" channel).

Both channels are distillation-family auxiliaries around the teacher / GT.
They do not replace exact component gates: JBL can only be treated as
promotion-capable after exact CUDA archive eval proves PoseNet non-collapse
and SegNet behavior under the canonical scorer path. JBL is designed to STACK
with the standard scorer loss, NOT replace it — see ``train_renderer.py``'s
``args.loss_mode == "jbl"`` dispatch which auxiliary-adds the JBL term
alongside the existing ``scorer_loss``.

CLAUDE.md compliance:
  * No CUDA/MPS device defaults — pure functional API; caller picks device.
  * No subprocess calls — no flag-invention surface.
  * eval_roundtrip is the caller's responsibility (consistent with
    every other helper in ``tac.losses``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jaccard_metric_loss(
    pred_logits: torch.Tensor,
    target_soft: torch.Tensor,
    num_classes: int = 5,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Soft Jaccard surrogate (Wang et al. NeurIPS 2023, Eq. 4).

    Args:
        pred_logits: ``(B, C, H, W)`` student logits (gradients flow).
        target_soft: ``(B, C, H, W)`` teacher soft labels — already a
            probability distribution along ``dim=1`` (i.e. softmax
            output). Caller must softmax the teacher logits before
            passing them in. This makes the distillation temperature
            an explicit caller choice rather than a hidden default.
        num_classes: expected channel dim of both tensors. Used as a
            structural guard — a mismatch raises ``ValueError``.
        epsilon: numerical floor on the union to prevent /0 when both
            distributions are effectively empty for some class.

    Returns:
        Scalar loss in ``[0, 1]``. Zero when ``softmax(pred_logits) ==
        target_soft`` exactly, monotone in disagreement.

    The quadratic relaxation makes intersection ``<p, q>`` and union
    ``||p||² + ||q||² - <p, q>`` differentiable in soft-label space.
    Reduces to Lovász-Softmax in the hard-label limit (Wang et al. §3.2).
    """
    if pred_logits.shape[1] != num_classes:
        raise ValueError(
            f"pred_logits channel dim {pred_logits.shape[1]} != num_classes "
            f"{num_classes}; pass logits in (B, C, H, W) layout."
        )
    if target_soft.shape != pred_logits.shape:
        raise ValueError(
            f"target_soft shape {target_soft.shape} must match pred_logits "
            f"shape {pred_logits.shape}."
        )

    # Student softmax along class dim.
    pred_soft = F.softmax(pred_logits, dim=1)

    # Per-class intersection / squared norms summed over (B, H, W).
    # Shape after sum: (C,)
    intersection = (pred_soft * target_soft).sum(dim=(0, 2, 3))
    pred_sq = pred_soft.pow(2).sum(dim=(0, 2, 3))
    target_sq = target_soft.pow(2).sum(dim=(0, 2, 3))

    union = pred_sq + target_sq - intersection
    # Floor union to avoid /0 when both distributions are empty for class c.
    # When a class is absent from BOTH distributions (union ~= 0), its IoU is
    # undefined — by Lovász/Jaccard convention we exclude these classes from
    # the mean (mIoU is averaged over present classes only). Without this,
    # absent classes contribute 0 to the numerator and pull JML toward 1.0
    # spuriously even at perfect prediction-vs-teacher match — see test
    # ``test_jaccard_metric_loss_increases_with_disagreement`` which pins
    # the perfect-match case to 0.
    iou_per_class = intersection / union.clamp(min=epsilon)
    present = (union > epsilon).float()
    n_present = present.sum().clamp(min=1.0)
    iou_mean = (iou_per_class * present).sum() / n_present
    return 1.0 - iou_mean


def boundary_label_smoothing(
    target_hard: torch.Tensor,
    num_classes: int = 5,
    smoothing: float = 0.1,
    boundary_width: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply label smoothing only at class boundaries.

    A pixel is a "boundary" pixel iff any pixel within ``boundary_width``
    of it has a different class label. Implemented via ``max_pool -
    min_pool`` of the integer label map: where the max and min within
    the kxk neighbourhood differ, there is a class transition. This
    matches the morphological-gradient definition used by Wang et al.
    and the wider segmentation literature.

    Boundary pixels receive a smoothed distribution:
        target[boundary] = (1 - smoothing) * one_hot + smoothing / C
    Interior pixels keep their hard one-hot label (no smoothing applied).

    Args:
        target_hard: ``(B, H, W)`` integer class labels in ``[0, C)``.
        num_classes: number of classes ``C``.
        smoothing: epsilon in the standard label-smoothing formula.
            ``0.0`` => hard labels everywhere (no smoothing). The Wang
            et al. recipe uses ``0.1`` as the canonical default.
        boundary_width: half-window radius for the morphological
            neighbourhood. ``2`` => 5x5 window (kernel = 2*w+1).

    Returns:
        ``(soft_target, boundary_mask)`` where:
          * ``soft_target`` is ``(B, C, H, W)`` float — one-hot at
            interior pixels, smoothed at boundary pixels.
          * ``boundary_mask`` is ``(B, H, W)`` float in ``{0, 1}`` —
            ``1`` at boundary pixels. Returned so the caller can do
            boundary-pixel weighting on a separate loss term without
            recomputing the morphological pool.

    The boundary mask uses ``max_pool - min_pool`` on the float-cast
    integer labels — when this is > 0 anywhere in the neighbourhood,
    there is a class transition.
    """
    if target_hard.dim() != 3:
        raise ValueError(
            f"target_hard must be (B, H, W); got shape {target_hard.shape}."
        )
    if not (0.0 <= smoothing <= 1.0):
        raise ValueError(f"smoothing must be in [0, 1]; got {smoothing}.")
    if boundary_width < 1:
        raise ValueError(f"boundary_width must be >= 1; got {boundary_width}.")

    B, H, W = target_hard.shape
    device = target_hard.device

    # One-hot baseline. Cast to float for the smoothing arithmetic.
    one_hot = F.one_hot(target_hard.long(), num_classes=num_classes)  # (B,H,W,C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Morphological boundary detection on the integer label map.
    kernel = 2 * boundary_width + 1
    pad = boundary_width
    label_float = target_hard.float().unsqueeze(1)  # (B, 1, H, W)
    max_pool = F.max_pool2d(label_float, kernel_size=kernel, stride=1, padding=pad)
    # min_pool via -max_pool(-x) (PyTorch lacks a direct min_pool2d).
    min_pool = -F.max_pool2d(-label_float, kernel_size=kernel, stride=1, padding=pad)
    boundary_mask = (max_pool - min_pool > 0).float().squeeze(1)  # (B, H, W)

    if smoothing == 0.0:
        return one_hot, boundary_mask

    # Smoothed distribution at boundary pixels:
    #   (1 - smoothing) * one_hot + smoothing / C * uniform
    smoothed = (1.0 - smoothing) * one_hot + (smoothing / num_classes)
    # Where boundary_mask == 1, use smoothed; else keep one_hot.
    bm_expanded = boundary_mask.unsqueeze(1).expand_as(one_hot)
    soft_target = torch.where(bm_expanded > 0, smoothed, one_hot)
    # Sanity: distribution rows must sum to 1 within float tolerance. We
    # don't assert at runtime (cost) but the construction is correct by
    # arithmetic — see test_boundary_label_smoothing_only_affects_boundary
    # for the structural gate.
    _ = device  # silence unused (kept for potential future use)
    return soft_target, boundary_mask


def combined_jbl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    num_classes: int = 5,
    jaccard_weight: float = 1.0,
    bls_weight: float = 0.5,
    boundary_pixel_weight: float = 3.0,
    bls_smoothing: float = 0.1,
    boundary_width: int = 2,
    teacher_temperature: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Combined JBL distillation loss.

    Returns the auxiliary loss to add to the standard scorer_loss. The
    JBL channel is designed to STACK on top of the existing scorer
    loss — passing the resulting tensor as
    ``loss = scorer_loss + args.kl_distill_weight * jbl_loss`` is the
    expected wiring (mirrors how ``kl_distill_segnet_only`` is used in
    ``train_renderer.py``). DO NOT replace ``scorer_loss`` with this.

    Two channels:
      1. JML on student soft vs teacher soft (distillation channel).
         ``teacher_logits`` are softened with ``teacher_temperature``
         before softmax — this is the Hinton 2015 "dark knowledge"
         channel adapted for Jaccard space.
      2. Cross-entropy on student vs ``boundary_label_smoothing(gt_mask)``
         with boundary pixels weighted ``boundary_pixel_weight`` × the
         interior weight (the "boundary precision" channel).

    Args:
        student_logits: ``(B, C, H, W)`` — gradients flow.
        teacher_logits: ``(B, C, H, W)`` — typically detached scorer
            output. Caller is responsible for ``no_grad`` if desired.
        gt_mask: ``(B, H, W)`` integer GT class labels.
        num_classes: ``C`` (5 for the comma SegNet).
        jaccard_weight: scalar on the JML channel. Default ``1.0``
            (matches Wang et al. §4 unit-weight setup).
        bls_weight: scalar on the BLS+CE channel. Default ``0.5`` —
            ratio matching the JML weight is reasonable but BLS is the
            tighter signal; halve it to avoid over-pulling on boundaries.
        boundary_pixel_weight: per-pixel weight multiplier inside the
            CE term for boundary pixels. Default ``3.0`` — middle of
            the paper's 3-5x recommendation.
        bls_smoothing: epsilon in label smoothing on boundary pixels.
        boundary_width: half-window radius for the boundary mask.
        teacher_temperature: softmax temperature applied to
            ``teacher_logits`` before producing soft targets for JML.

    Returns:
        ``(loss, telemetry_dict)`` — telemetry contains scalar Python
        floats for the JML term, the CE term, and the boundary pixel
        fraction (useful for confirming the morphological pool fired).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student_logits {student_logits.shape} must match teacher_logits "
            f"{teacher_logits.shape}."
        )
    if student_logits.dim() != 4:
        raise ValueError(
            f"student_logits must be (B, C, H, W); got {student_logits.shape}."
        )

    # Channel 1: JML on softened teacher.
    teacher_soft = F.softmax(teacher_logits / teacher_temperature, dim=1)
    jml = jaccard_metric_loss(
        student_logits, teacher_soft, num_classes=num_classes,
    )

    # Channel 2: BLS + boundary-weighted CE.
    soft_target, boundary_mask = boundary_label_smoothing(
        gt_mask, num_classes=num_classes,
        smoothing=bls_smoothing, boundary_width=boundary_width,
    )
    # Cross-entropy with soft targets: -sum_c (target_c * log_softmax_c)
    # per pixel. Manual implementation because F.cross_entropy demands
    # hard integer targets.
    log_softmax = F.log_softmax(student_logits, dim=1)
    ce_per_pixel = -(soft_target * log_softmax).sum(dim=1)  # (B, H, W)

    # Boundary-pixel weighting: interior weight = 1, boundary weight =
    # boundary_pixel_weight. Normalize so the mean weight is 1, keeping
    # the CE scale comparable to the unweighted case (= the "loss scale
    # invariance under reweighting" rule from kl_distill_scorer_loss).
    weight_map = 1.0 + (boundary_pixel_weight - 1.0) * boundary_mask
    weight_map = weight_map / weight_map.mean().clamp(min=1e-8)
    ce_loss = (ce_per_pixel * weight_map).mean()

    loss = jaccard_weight * jml + bls_weight * ce_loss

    telemetry = {
        "jml": jml.item(),
        "ce_bls": ce_loss.item(),
        "boundary_pixel_fraction": boundary_mask.mean().item(),
    }
    return loss, telemetry
