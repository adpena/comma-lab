"""Loss functions for task-aware codec training.

All losses operate on frame pairs in (1, 2, H, W, 3) HWC format and
handle the BTCHW conversion internally before passing to the scorers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bhattacharyya_distance(p: torch.Tensor, q: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """True Bhattacharyya distance: -log(sum(sqrt(p*q))).

    Bhat advisory recommendation. Unlike the dot-product form (1 - sum(p*q)),
    the true distance uses sqrt(p*q) which is concave and has stronger
    gradients near agreement. The -log transform makes the distance infinite
    when distributions have disjoint support (strong repulsion from disagreement)
    and zero at perfect agreement.

    Args:
        p: probability distribution (e.g., softmax output)
        q: probability distribution (same shape as p)
        dim: dimension over which to sum classes (default 1 for BCHW)

    Returns:
        Scalar mean Bhattacharyya distance over all spatial positions and batch.
    """
    bc = torch.sqrt(p * q).sum(dim=dim).clamp(min=1e-8)  # BC per pixel, guard log(0)
    return -torch.log(bc.mean().clamp(min=1e-8))


def scorer_forward_pair(pair_btchw, posenet, segnet):
    """Forward pass through both scorer networks.

    Args:
        pair_btchw: (B, T, C, H, W) float tensor
        posenet: frozen PoseNet model
        segnet: frozen SegNet model

    Returns:
        (posenet_output, segnet_output) dicts/tensors
    """
    # Ensure contiguous standard-stride layout before entering scorer models.
    # The upstream scorers (PoseNet/SegNet) use .view() internally which requires
    # contiguous tensors. Non-contiguous tensors from permute/channels_last cause
    # RuntimeError in backward pass. This is the canonical fix — not a monkey-patch.
    pair_btchw = pair_btchw.contiguous()
    posenet_in = posenet.preprocess_input(pair_btchw)
    posenet_out = posenet(posenet_in)
    segnet_in = segnet.preprocess_input(pair_btchw)
    segnet_out = segnet(segnet_in)
    return posenet_out, segnet_out


def _hwc_to_chw(pair_hwc: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, H, W, C) to (B, T, C, H, W) float."""
    # Fast path: skip redundant .float() if already float32
    if pair_hwc.dtype == torch.float32:
        out = pair_hwc.permute(0, 1, 4, 2, 3)
        return out if out.is_contiguous() else out.contiguous()
    return pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()


def _validate_kl_temperature(temperature: float, *, field: str = "temperature") -> float:
    """Validate KL/distillation softmax temperature before division."""
    if isinstance(temperature, bool):
        raise ValueError(f"{field} must be a finite positive number")
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return value


def _apply_class_weights(
    seg_per_pixel: torch.Tensor,
    gt_logits_or_probs: torch.Tensor,
    class_weights: torch.Tensor,
    gt_already_probs: bool = False,
) -> torch.Tensor:
    """Lane PS shared kernel — multiply per-pixel SegNet loss by per-class weights.

    Args:
        seg_per_pixel: (B, H, W) per-pixel SegNet loss (soft-cosine, KL, …).
        gt_logits_or_probs: (B, C, H, W) GT SegNet logits OR softmax probs;
            argmax is computed under no_grad to fetch the per-pixel target
            class. (argmax(logits) == argmax(softmax) — monotone — so the
            cached softmax path is equivalent to the live-logits path.)
        class_weights: (NUM_CLASSES,) float tensor of per-class weights.
            L1-normalised to mean=1 internally so absolute loss magnitude
            is preserved vs the unweighted call.
        gt_already_probs: cosmetic flag for caller documentation; argmax
            behaviour is identical regardless.

    Returns: ``seg_per_pixel * pixel_weights`` (same shape as input).

    Raises:
        ValueError: if ``class_weights`` shape does not match the channel
            dim of ``gt_logits_or_probs``.
    """
    del gt_already_probs  # kept for caller-side documentation only
    if class_weights is None:  # defensive — callers should gate this
        return seg_per_pixel
    with torch.no_grad():
        cw = class_weights.to(seg_per_pixel.device, dtype=seg_per_pixel.dtype)
        if cw.ndim != 1 or cw.shape[0] != gt_logits_or_probs.shape[1]:
            raise ValueError(
                f"class_weights shape {tuple(cw.shape)} does not match "
                f"SegNet num_classes {gt_logits_or_probs.shape[1]}"
            )
        cw = cw / cw.mean().clamp(min=1e-8)
        gt_argmax = gt_logits_or_probs.argmax(dim=1)  # (B, H, W)
        # Spatial dims of seg_per_pixel may differ if the caller resamples;
        # the canonical path keeps them aligned (same SegNet logit shape),
        # so we assert and let mismatches fail loud.
        if gt_argmax.shape != seg_per_pixel.shape:
            raise ValueError(
                f"GT argmax spatial shape {tuple(gt_argmax.shape)} does not "
                f"match seg_per_pixel shape {tuple(seg_per_pixel.shape)}"
            )
        pixel_w = cw[gt_argmax]  # (B, H, W)
    return seg_per_pixel * pixel_w


def per_class_seg_distortion(
    seg_per_pixel: torch.Tensor,
    gt_logits_or_probs: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Aggregate per-pixel SegNet loss into a (num_classes,) per-class
    distortion vector via GT-argmax binning.

    Used by Round 11 Finding 2 fix in train_renderer.py to feed the
    LearnableClassWeights dual_update with the actual per-class
    distortion residuals (not a uniform scalar). Mean is taken over all
    pixels with that argmax label; classes with zero pixels in the batch
    return 0 (no signal — the dual update naturally ignores them via the
    target_t centring).

    Args:
        seg_per_pixel: (B, H, W) per-pixel SegNet loss (soft-cosine, KL).
        gt_logits_or_probs: (B, C, H, W) GT logits OR softmax probs.
        num_classes: expected channel count.

    Returns: (num_classes,) tensor on the same device/dtype as ``seg_per_pixel``.
    """
    with torch.no_grad():
        if gt_logits_or_probs.shape[1] != num_classes:
            raise ValueError(
                f"gt_logits_or_probs channel dim {gt_logits_or_probs.shape[1]} "
                f"!= num_classes {num_classes}"
            )
        gt_argmax = gt_logits_or_probs.argmax(dim=1)  # (B, H, W)
        if gt_argmax.shape != seg_per_pixel.shape:
            raise ValueError(
                f"GT argmax spatial shape {tuple(gt_argmax.shape)} does not "
                f"match seg_per_pixel shape {tuple(seg_per_pixel.shape)}"
            )
        flat_loss = seg_per_pixel.detach().reshape(-1)
        flat_idx = gt_argmax.reshape(-1)
        out = torch.zeros(num_classes, device=flat_loss.device, dtype=flat_loss.dtype)
        cnt = torch.zeros(num_classes, device=flat_loss.device, dtype=flat_loss.dtype)
        out.scatter_add_(0, flat_idx, flat_loss)
        cnt.scatter_add_(0, flat_idx, torch.ones_like(flat_loss))
        out = out / cnt.clamp(min=1.0)
    return out


def parse_class_weights_csv(
    s: str | None,
    num_classes: int = 5,
) -> torch.Tensor | None:
    """Lane PS — parse a CSV string like ``"1.0,5.0,5.0,1.0,1.0"`` to a tensor.

    Returns ``None`` for ``None`` / empty input (canonical "disabled" sentinel).
    Used by ``experiments/optimize_poses.py`` and
    ``src/tac/experiments/train_renderer.py`` to convert the
    ``--segnet-class-weights`` CLI string into the tensor passed down to
    the loss helpers. Centralised so both call sites apply the SAME parse
    + validation rules (catches "5,5,5,5" → 4-class footgun at boot).

    Args:
        s: CSV string of float weights, or None / empty.
        num_classes: expected number of classes (5 for the comma SegNet).

    Returns:
        ``None`` when ``s`` is None / empty; else ``(num_classes,)`` float
        tensor. Raises ``ValueError`` on shape mismatch / parse error so
        the operator sees the bug at CLI parse time, not 30 min into a run.
    """
    if s is None or not s.strip():
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(
            f"--segnet-class-weights expected {num_classes} CSV values, "
            f"got {len(parts)}: {parts!r}"
        )
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover — covered via parametric test
        raise ValueError(
            f"--segnet-class-weights could not parse {parts!r} as floats: {exc}"
        ) from exc
    if any(v < 0 for v in vals):
        raise ValueError(
            f"--segnet-class-weights must be non-negative, got {vals!r}"
        )
    if all(v == 0 for v in vals):
        raise ValueError(
            "--segnet-class-weights cannot be all zeros (would zero out "
            "the entire SegNet loss)"
        )
    return torch.tensor(vals, dtype=torch.float32)


def scorer_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    """Standard scorer loss: 100*seg + sqrt(10*pose).

    Uses soft cosine for SegNet (differentiable proxy for argmax disagreement).

    Args:
        class_weights: (NUM_CLASSES,) optional per-class weights for the
            soft-cosine SegNet term (Lane PS — per-class SegNet weighting,
            see ``kl_distill_segnet_only`` docstring). Default ``None`` =
            uniform weighting (byte-identical to baseline). Weights are
            L1-normalised internally so the loss scale is preserved.

    Returns: (loss, pose_distortion, seg_distortion)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_per_pixel = 1.0 - (pred_soft * gt_soft).sum(dim=1)  # (B, H, W)
    if class_weights is not None:
        seg_per_pixel = _apply_class_weights(seg_per_pixel, gs_out, class_weights)
    seg_dist = seg_per_pixel.mean()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def scorer_loss_cached(
    filtered_pair_hwc: torch.Tensor,
    gt_pose_6: torch.Tensor,
    gt_seg_soft: torch.Tensor,
    posenet,
    segnet,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    """Standard scorer loss using pre-cached GT scorer outputs.

    Avoids running frozen scorers on GT frames every iteration (P0 optimization).
    GT frames and scorers are frozen, so their outputs are constant.

    Args:
        filtered_pair_hwc: (B, T, H, W, C) filtered pair
        gt_pose_6: cached PoseNet output[:, :6] for GT pair (on device)
        gt_seg_soft: cached softmax(SegNet output) for GT pair (on device)
        posenet: frozen PoseNet model
        segnet: frozen SegNet model
        class_weights: (NUM_CLASSES,) optional per-class weights for the
            soft-cosine SegNet term (Lane PS). The cached path uses
            ``gt_seg_soft.argmax`` as the GT class label since the raw
            logits are not stored. Default ``None`` = uniform weighting.

    Returns: (loss, pose_distortion, seg_distortion)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gt_pose_6).pow(2).mean()

    pred_soft = F.softmax(fs_out, dim=1)
    seg_per_pixel = 1.0 - (pred_soft * gt_seg_soft).sum(dim=1)  # (B, H, W)
    if class_weights is not None:
        # Cached path: use the cached softmax as the GT-argmax source.
        # Equivalent to argmax(logits) since softmax is monotone.
        seg_per_pixel = _apply_class_weights(
            seg_per_pixel, gt_seg_soft, class_weights, gt_already_probs=True,
        )
    seg_dist = seg_per_pixel.mean()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def scorer_loss_with_aux(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    class_weights: torch.Tensor | None = None,
    num_classes: int = 5,
) -> tuple[torch.Tensor, float, float, dict]:
    """Variant of :func:`scorer_loss` that also returns per-pair pose loss
    and per-class SegNet distortion for the Round 11 Finding 2 dual-update
    wiring (Lane W-V2 + Lane PS-V2). Caller passes ``class_weights`` exactly
    as in :func:`scorer_loss`; the AUX dict carries detached tensors only.

    Returns: (loss, pose_distortion, seg_distortion, aux) where
        aux = {
            "pose_dist_per_pair": (B,) per-pair pose loss (detached),
            "per_class_seg_distortion": (num_classes,) per-class soft-cos (detached),
        }
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_diff = fp_out["pose"][..., :6] - gp_out["pose"][..., :6]
    pose_dist = pose_diff.pow(2).mean()
    # (B,) per-pair pose distortion — collapse pose dims to mean.
    pose_dist_per_pair = pose_diff.detach().pow(2).mean(dim=tuple(range(1, pose_diff.ndim)))

    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_per_pixel = 1.0 - (pred_soft * gt_soft).sum(dim=1)
    per_class_d = per_class_seg_distortion(seg_per_pixel, gs_out, num_classes)
    if class_weights is not None:
        seg_per_pixel = _apply_class_weights(seg_per_pixel, gs_out, class_weights)
    seg_dist = seg_per_pixel.mean()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    aux = {
        "pose_dist_per_pair": pose_dist_per_pair,
        "per_class_seg_distortion": per_class_d,
    }
    return loss, pose_dist.item(), seg_dist.item(), aux


def scorer_loss_cached_with_aux(
    filtered_pair_hwc: torch.Tensor,
    gt_pose_6: torch.Tensor,
    gt_seg_soft: torch.Tensor,
    posenet,
    segnet,
    class_weights: torch.Tensor | None = None,
    num_classes: int = 5,
) -> tuple[torch.Tensor, float, float, dict]:
    """Cached counterpart to :func:`scorer_loss_with_aux`."""
    fx = _hwc_to_chw(filtered_pair_hwc)
    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    pose_diff = fp_out["pose"][..., :6] - gt_pose_6
    pose_dist = pose_diff.pow(2).mean()
    pose_dist_per_pair = pose_diff.detach().pow(2).mean(dim=tuple(range(1, pose_diff.ndim)))

    pred_soft = F.softmax(fs_out, dim=1)
    seg_per_pixel = 1.0 - (pred_soft * gt_seg_soft).sum(dim=1)
    per_class_d = per_class_seg_distortion(seg_per_pixel, gt_seg_soft, num_classes)
    if class_weights is not None:
        seg_per_pixel = _apply_class_weights(
            seg_per_pixel, gt_seg_soft, class_weights, gt_already_probs=True,
        )
    seg_dist = seg_per_pixel.mean()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    aux = {
        "pose_dist_per_pair": pose_dist_per_pair,
        "per_class_seg_distortion": per_class_d,
    }
    return loss, pose_dist.item(), seg_dist.item(), aux


def scorer_loss_pcgrad(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    segnet_weight: float = 100.0,
    do_projection: bool = True,
) -> tuple[torch.Tensor, float, float, bool]:
    """Non-opposing gradient loss for PoseNet-SegNet decoupling.

    Computes separate PoseNet and SegNet losses. When gradients conflict
    (negative cosine similarity in the intermediate activation space), scales
    the SegNet loss so the combined gradient does not oppose PoseNet.

    This is an approximation of PCGrad (Yu et al., NeurIPS 2020) that operates
    at the activation level rather than the parameter level. The full parameter-
    level PCGrad would require 2x backward passes. This approximation is exact
    when the model is linear and provides a non-opposing guarantee otherwise.

    For stronger multi-task gradient methods, consider:
    - CAGrad (Liu et al., NeurIPS 2021): maximizes worst-case local improvement
    - Nash-MTL (Navon et al., ICML 2022): Nash bargaining for proportional fairness
    - Aligned-MTL (Senushkin et al., CVPR 2023): converges to specified Pareto point

    Args:
        do_projection: If False, skip the gradient conflict check and just return
            the weighted sum. Used to avoid retain_graph memory pressure on
            microbatch steps 2+ during gradient accumulation (BUG 3/6/MEMORY fix).

    Returns: (loss, pose_distortion, seg_distortion, conflict_detected)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    # BUG 4 fix: ensure fx has requires_grad even after permute/contiguous
    # The _hwc_to_chw does .float().permute().contiguous() which may not
    # propagate requires_grad from the input if it was detached upstream.
    if not fx.requires_grad and filtered_pair_hwc.requires_grad:
        fx.requires_grad_(True)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
    pose_loss = torch.sqrt(10.0 * pose_dist + 1e-8)

    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()
    seg_loss = segnet_weight * seg_dist

    conflict = False

    # Gradient conflict detection and non-opposing projection
    # Operates on intermediate activations (fx) as a proxy for parameter gradients
    #
    # Only run projection when do_projection=True (first microbatch in accumulation
    # window) AND segnet_weight > 0 (skip when seg is disabled, avoids wasted autograd).
    if do_projection and fx.requires_grad and segnet_weight > 0:
        g_pose = torch.autograd.grad(pose_loss, fx, retain_graph=True, create_graph=False)[0]
        g_seg = torch.autograd.grad(seg_loss, fx, retain_graph=True, create_graph=False)[0]

        cos_sim = (g_pose * g_seg).sum() / (g_pose.norm() * g_seg.norm() + 1e-8)

        if cos_sim.item() < 0:
            conflict = True
            # Project: remove the component of g_seg along g_pose
            # We want: g_pose + scale * g_seg to have non-negative dot with g_pose
            # ||g_pose||^2 + scale * (g_seg . g_pose) >= 0
            # scale <= -||g_pose||^2 / (g_seg . g_pose)   [since g_seg.g_pose < 0]
            dot = (g_seg * g_pose).sum().item()
            pose_norm_sq = (g_pose.norm() ** 2).item()
            if abs(dot) > 1e-12:
                # Scale that makes combined gradient non-opposing to pose
                # (90% of the perpendicular scale, as a safety margin)
                max_scale = -pose_norm_sq / dot
                scale = min(1.0, max(0.01, 0.9 * max_scale))
            else:
                scale = 1.0
            seg_loss = seg_loss * scale

    loss = pose_loss + seg_loss
    return loss, pose_dist.item(), seg_dist.item(), conflict


def scorer_loss_pcgrad_cached(
    filtered_pair_hwc: torch.Tensor,
    gt_pose_6: torch.Tensor,
    gt_seg_soft: torch.Tensor,
    posenet,
    segnet,
    segnet_weight: float = 100.0,
    do_projection: bool = True,
) -> tuple[torch.Tensor, float, float, bool]:
    """Non-opposing gradient loss using pre-cached GT scorer outputs (P0 optimization).

    Same as scorer_loss_pcgrad but avoids recomputing GT scorer forward pass.

    Returns: (loss, pose_distortion, seg_distortion, conflict_detected)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)

    if not fx.requires_grad and filtered_pair_hwc.requires_grad:
        fx.requires_grad_(True)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gt_pose_6).pow(2).mean()
    pose_loss = torch.sqrt(10.0 * pose_dist + 1e-8)

    pred_soft = F.softmax(fs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_seg_soft).sum(dim=1).mean()
    seg_loss = segnet_weight * seg_dist

    conflict = False

    if do_projection and fx.requires_grad and segnet_weight > 0:
        g_pose = torch.autograd.grad(pose_loss, fx, retain_graph=True, create_graph=False)[0]
        g_seg = torch.autograd.grad(seg_loss, fx, retain_graph=True, create_graph=False)[0]

        cos_sim = (g_pose * g_seg).sum() / (g_pose.norm() * g_seg.norm() + 1e-8)

        if cos_sim.item() < 0:
            conflict = True
            dot = (g_seg * g_pose).sum().item()
            pose_norm_sq = (g_pose.norm() ** 2).item()
            if abs(dot) > 1e-12:
                max_scale = -pose_norm_sq / dot
                scale = min(1.0, max(0.01, 0.9 * max_scale))
            else:
                scale = 1.0
            seg_loss = seg_loss * scale

    loss = pose_loss + seg_loss
    return loss, pose_dist.item(), seg_dist.item(), conflict


@torch.no_grad()
def eval_scorer_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
) -> tuple[float, float, float]:
    """Hard evaluation loss matching the official scorer exactly.

    PoseNet: per-sample MSE on first 6 outputs, averaged across batch.
    SegNet: per-pixel argmax disagreement fraction, averaged across batch.
    Score: 100*seg + sqrt(10*pose)

    Use this for checkpoint selection and evaluation. For training gradients,
    use scorer_loss (soft proxy) or segnet_ste_loss instead.

    Returns: (score, pose_distortion, seg_distortion)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    # PoseNet: per-sample MSE (matches upstream compute_distortion)
    pose_per_sample = (
        (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean(dim=tuple(range(1, fp_out["pose"].ndim)))
    )

    # SegNet: hard argmax disagreement (matches upstream compute_distortion)
    diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
    seg_per_sample = diff.mean(dim=tuple(range(1, diff.ndim)))

    avg_p = pose_per_sample.mean().item()
    avg_s = seg_per_sample.mean().item()
    score = 100.0 * avg_s + math.sqrt(10.0 * avg_p)
    return score, avg_p, avg_s


def renderer_scorer_loss(
    rendered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    segnet_weight: float = 100.0,
) -> tuple[torch.Tensor, float, float]:
    """Scorer loss optimized for the renderer paradigm.

    Key insight from scorer mechanics:
    - SegNet evaluates only the LAST frame (frame_t1) via argmax
    - PoseNet evaluates the full frame PAIR for ego-motion estimation

    This loss applies SegNet loss only on frame_t1, freeing frame_t
    to be optimized purely for PoseNet (ego-motion) without SegNet
    interference. This reduces the multi-task conflict surface.

    Returns: (loss, pose_distortion, seg_distortion)
    """
    fx = _hwc_to_chw(rendered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    # Full pair PoseNet loss (both frames matter for ego-motion)
    fp_out, _ = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, _ = scorer_forward_pair(gx, posenet, segnet)
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # Last-frame-only SegNet loss (scorer uses argmax of frame_t1 only)
    # Extract last frame: (B, T, C, H, W) → (B, C, H, W) at T=-1
    fx_last = fx[:, -1]  # (B, C, H, W)
    gx_last = gx[:, -1]

    # SegNet expects (B, T, C, H, W) with T=1 for single-frame eval
    fx_last_bt = fx_last.unsqueeze(1)
    gx_last_bt = gx_last.unsqueeze(1)

    fs_in = segnet.preprocess_input(fx_last_bt)
    fs_out = segnet(fs_in)
    with torch.no_grad():
        gs_in = segnet.preprocess_input(gx_last_bt)
        gs_out = segnet(gs_in)

    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

    loss = segnet_weight * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def segnet_ste_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    boundary_mask: torch.Tensor | None = None,
    boundary_weight: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    """SegNet STE loss with optional boundary-band weighting.

    Forward: hard argmax disagreement (matches scorer exactly).
    Backward: cross-entropy gradient (differentiable via STE).
    Boundary pixels get `boundary_weight`x more gradient.

    Args:
        boundary_mask: (H, W) float tensor, 1.0 at boundaries, 0.0 elsewhere.
            Applied uniformly across the batch — assumes B=1 or all samples
            share the same boundary mask.
        boundary_weight: multiplier for boundary pixels (default 1.0 = no weighting)

    Returns: (loss, pose_distortion, hard_disagree)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    # PoseNet: standard MSE
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # SegNet: STE with optional boundary weighting
    with torch.no_grad():
        gt_labels = gs_out.argmax(dim=1)
        pred_labels = fs_out.argmax(dim=1)
        hard_disagree = (pred_labels != gt_labels).float().mean()

    B, C, H_seg, W_seg = fs_out.shape
    flat_labels = gt_labels.reshape(-1)
    flat_logits = fs_out.permute(0, 2, 3, 1).reshape(-1, C)

    if boundary_mask is not None and boundary_weight > 1.0:
        bm = boundary_mask.to(fs_out.device).unsqueeze(0).unsqueeze(0)
        bm_resized = F.interpolate(bm, size=(H_seg, W_seg), mode="nearest").squeeze(0).squeeze(0)
        pixel_weights = torch.where(bm_resized > 0.5, boundary_weight, 1.0)
        pixel_weights = pixel_weights / pixel_weights.mean()
        flat_weights = pixel_weights.expand(B, -1, -1).reshape(-1)
        per_pixel_ce = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        soft_ce = (per_pixel_ce * flat_weights).mean()
    else:
        soft_ce = F.cross_entropy(flat_logits, flat_labels, reduction="mean")

    # Canonical STE: forward = hard, backward = soft
    seg_dist = soft_ce + (hard_disagree - soft_ce).detach()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), hard_disagree.item()


def temperature_scorer_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    """Scorer loss with temperature-scaled softmax for SegNet.

    Lower temperature → sharper softmax → closer to hard argmax.
    Anneal T from 1.0 (smooth) to 0.05 (near-hard) during training
    to gradually focus gradient on boundary pixels.

    Council recommendation #2: expected -0.06 to -0.12 SegNet improvement.
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    pred_sharp = F.softmax(fs_out / temperature, dim=1)
    gt_sharp = F.softmax(gs_out / temperature, dim=1)
    seg_dist = 1.0 - (pred_sharp * gt_sharp).sum(dim=1).mean()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def focal_segnet_ste_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    gamma: float = 2.0,
    boundary_mask: torch.Tensor | None = None,
    boundary_weight: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    """SegNet STE with focal cross-entropy for automatic boundary focus.

    Focal loss down-weights easy (confident) pixels and up-weights
    hard (uncertain) pixels — exactly the boundary pixels where
    argmax decisions can flip.

    Council recommendation #3: expected -0.09 to -0.14 SegNet improvement.
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    with torch.no_grad():
        gt_labels = gs_out.argmax(dim=1)
        pred_labels = fs_out.argmax(dim=1)
        hard_disagree = (pred_labels != gt_labels).float().mean()

    B, C, H_seg, W_seg = fs_out.shape
    flat_labels = gt_labels.reshape(-1)
    flat_logits = fs_out.permute(0, 2, 3, 1).reshape(-1, C)

    # Focal loss: down-weight easy pixels, up-weight hard pixels
    per_pixel_ce = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    pt = torch.exp(-per_pixel_ce)
    focal_weight = (1 - pt) ** gamma

    # Optional boundary weighting on top of focal
    if boundary_mask is not None and boundary_weight > 1.0:
        bm = boundary_mask.to(fs_out.device).unsqueeze(0).unsqueeze(0)
        bm_resized = F.interpolate(bm, size=(H_seg, W_seg), mode="nearest").squeeze(0).squeeze(0)
        pixel_bw = torch.where(bm_resized > 0.5, boundary_weight, 1.0)
        pixel_bw = pixel_bw / pixel_bw.mean()
        focal_weight = focal_weight * pixel_bw.expand(B, -1, -1).reshape(-1)

    soft_ce = (focal_weight * per_pixel_ce).mean()
    seg_dist = soft_ce + (hard_disagree - soft_ce).detach()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), hard_disagree.item()


def dual_saliency_reconstruction_loss(
    filtered_bchw: torch.Tensor,
    original_bchw: torch.Tensor,
    posenet_sal: torch.Tensor,
    segnet_boundary: torch.Tensor | None = None,
    alpha_pose: float = 20.0,
    alpha_seg: float = 200.0,
) -> torch.Tensor:
    """Combined PoseNet + SegNet saliency reconstruction loss.

    Allows corrections at BOTH PoseNet-sensitive AND SegNet-boundary pixels.
    The alpha_seg/alpha_pose ratio should approximate the scoring formula's
    leverage ratio (100/8.68 ≈ 11.5, so alpha_seg ≈ 11.5 * alpha_pose).

    Council recommendation #1: expected -0.06 to -0.12 SegNet improvement.

    Args:
        posenet_sal: (B, 1, H, W) PoseNet saliency weights (1 + alpha * sal)
        segnet_boundary: (H, W) boundary mask (1.0 at boundaries, 0.0 elsewhere)
        alpha_pose: weight for PoseNet saliency (default 20)
        alpha_seg: weight for SegNet boundaries (default 200, ~10x alpha_pose)
    """
    if segnet_boundary is not None:
        bm = segnet_boundary.to(filtered_bchw.device)
        if bm.ndim == 2:
            bm = bm.unsqueeze(0).unsqueeze(0)
        # Resize boundary to match spatial dims
        if bm.shape[-2:] != filtered_bchw.shape[-2:]:
            bm = F.interpolate(bm, size=filtered_bchw.shape[-2:], mode="nearest")
        combined = posenet_sal + alpha_seg * bm
    else:
        combined = posenet_sal

    residual = filtered_bchw - original_bchw
    inv_weight = 1.0 / combined.clamp(min=1e-10)
    return (inv_weight * residual.pow(2)).mean()


def saliency_reconstruction_loss(
    filtered_bchw: torch.Tensor,
    original_bchw: torch.Tensor,
    sal_weights: torch.Tensor,
) -> torch.Tensor:
    """Saliency-weighted pixel reconstruction penalty.

    Penalizes corrections on low-saliency pixels (protecting SegNet),
    allows larger corrections on high-saliency pixels (PoseNet regions).

    Args:
        filtered_bchw: (B, 3, H, W) float, model output
        original_bchw: (B, 3, H, W) float, compressed input
        sal_weights: (B, 1, H, W) per-pixel weights (1 + alpha * saliency)
    """
    residual = filtered_bchw - original_bchw
    inv_weight = 1.0 / sal_weights.clamp(min=1e-10)
    return (inv_weight * residual.pow(2)).mean()


def kl_distill_scorer_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    temperature: float = 2.0,
    boundary_mask: torch.Tensor | None = None,
    boundary_weight: float = 10.0,
    segnet_weight: float = 100.0,
) -> tuple[torch.Tensor, float, float]:
    """Hinton-style KL distillation loss for SegNet + standard PoseNet MSE.

    Instead of matching hard argmax or soft cosine, this matches the full
    class probability distribution between filtered and GT frames using
    KL divergence with temperature scaling.

    At high T (5.0): soft distributions reveal inter-class relationships,
    giving rich gradients for learning WHERE to push pixel values.
    At low T (1.0): focuses on flipping the actual argmax at boundary pixels.

    Boundary-weighted: pixels near SegNet class boundaries get amplified
    gradients since those are the only pixels where argmax can flip.

    T² scaling (Hinton 2015): compensates for gradient magnitude reduction
    at high temperatures so gradient norms are consistent across T values.

    Args:
        temperature: softmax temperature (anneal from 2.0 → 1.0 over training)
        boundary_mask: (H, W) float, 1.0 at class boundaries
        boundary_weight: gradient amplification at boundaries (default 10×)
        segnet_weight: weight on SegNet term (default 100, matches scorer formula)

    Returns: (loss, pose_distortion, seg_kl_divergence)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    # PoseNet: standard MSE (unchanged)
    fp_in = posenet.preprocess_input(fx)
    gp_in = posenet.preprocess_input(gx)
    fp_out = posenet(fp_in)
    with torch.no_grad():
        gp_out = posenet(gp_in)
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # SegNet: KL divergence on temperature-softened distributions
    fs_in = segnet.preprocess_input(fx)
    with torch.no_grad():
        gs_in = segnet.preprocess_input(gx)
    fs_logits = segnet(fs_in)  # (B, 5, H, W) — gradients flow through
    with torch.no_grad():
        gs_logits = segnet(gs_in)  # (B, 5, H, W) — teacher, no gradients

    T = _validate_kl_temperature(temperature)
    log_p = F.log_softmax(fs_logits / T, dim=1)
    q = F.softmax(gs_logits / T, dim=1)

    # KL(q || p) per pixel: sum over classes, keep spatial dims
    kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)  # (B, H, W)

    # T² scaling FIRST (Hinton 2015): compensates for KL compression at high T.
    # Applied per-pixel before spatial averaging — this is a property of the KL
    # divergence geometry, independent of spatial weighting.
    kl_per_pixel = kl_per_pixel * (T * T)

    # Boundary weighting: amplify gradients at class boundaries
    if boundary_mask is not None:
        bm = boundary_mask.to(kl_per_pixel.device)
        if bm.shape != kl_per_pixel.shape[-2:]:
            bm = (
                F.interpolate(
                    bm.unsqueeze(0).unsqueeze(0).float(),
                    size=kl_per_pixel.shape[-2:],
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )
        weight = 1.0 + boundary_weight * bm
        weight = weight / weight.mean()  # normalize to preserve loss scale
        kl_per_pixel = kl_per_pixel * weight

    seg_kl = kl_per_pixel.mean()

    # NOTE: PoseNet gradient cap REMOVED after authoritative regression (1.85 vs proxy 1.25).
    # The clamp killed PoseNet gradient, letting PoseNet degrade unchecked while proxy didn't catch it.
    # The sqrt formula already naturally de-emphasizes small pose values without a hard cutoff.
    loss = segnet_weight * seg_kl + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_kl.item()


def kl_distill_segnet_only(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    segnet,
    temperature: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """SegNet-ONLY KL divergence — auxiliary helper for stacking on top of scorer_loss.

    Use this INSTEAD of kl_distill_scorer_loss when you want KL distillation
    AS AN AUXILIARY LOSS alongside the standard scorer_loss. The full
    kl_distill_scorer_loss returns 100*seg_kl + sqrt(10*pose_dist) — adding
    that to scorer_loss double-counts BOTH the SegNet term (200x weight) and
    the PoseNet term (gradient stack), which historically caused PoseNet
    collapse per CLAUDE.md "Critical Lessons".

    This helper returns ONLY the SegNet KL value (multiplied by T² per Hinton 2015).
    The caller chooses the auxiliary weight (typical: 1.0).

    Args:
        filtered_pair_hwc: (B, T, H, W, C) float pair from renderer.
        gt_pair_hwc: (B, T, H, W, C) float ground-truth pair.
        segnet: frozen SegNet scorer.
        temperature: softmax temperature (Hinton 2015 default 2.0).
        class_weights: (NUM_CLASSES,) optional float tensor of per-class
            weights. When provided, the per-pixel KL is multiplied by the
            weight for that pixel's GT-argmax class, biasing the renderer
            to spend its capacity on costly classes (e.g., lane marks,
            vehicles) instead of cheap ones (road, sky). Lane PS — see
            project_research_survey_20260420 + research_survey memory item
            "per-class SegNet weighting (research-grade, never implemented)".
            Default ``None`` (uniform weighting, byte-identical to baseline).
            The weights are L1-normalised so ``mean == 1`` to keep the loss
            scale comparable to the unweighted variant. Shape must match
            ``segnet`` output channel dim (5 classes for the comma SegNet).

    Returns: (seg_kl * T², seg_kl_value_unscaled)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fs_in = segnet.preprocess_input(fx)
    with torch.no_grad():
        gs_in = segnet.preprocess_input(gx)
    fs_logits = segnet(fs_in)
    with torch.no_grad():
        gs_logits = segnet(gs_in)

    T = _validate_kl_temperature(temperature)
    log_p = F.log_softmax(fs_logits / T, dim=1)
    q = F.softmax(gs_logits / T, dim=1)
    # 2026-04-27 council forensics (findings.md "Lane G — really dead, or
    # bugged?"): the prior `reduction="batchmean"` on a (B, 5, H, W)
    # SegNet logit tensor divides only by B, producing a value
    # H × W = 384 × 512 = 196,608 larger than the per-pixel-per-class
    # mean the caller-weight contract assumes. (`batchmean` already
    # sums-over-class internally, so the missing divisor is exactly
    # H × W, not H × W × C.) Empirical kl.item() ≈ 24,485 vs the true
    # per-pixel-per-class mean of ~6.2e-3 nats. Every caller passing
    # kl_distill_weight=1.0 (DEN/SHIRAZ/WILDE/Lane-D training, Lane G
    # pose TTO v1/v2) was implicitly running at ~5000× the intended
    # weight. Fix mirrors the canonical pattern in
    # `kl_distill_scorer_loss` (line 622+646): `reduction="none"` →
    # `.sum(dim=1)` (over class) → `.mean()` (over B, H, W). T² scaling
    # placement preserved per Hinton 2015 §2.1. See
    # `feedback_unit_error_masquerading_as_small_signal` for the
    # process lesson + `test_kl_distill_segnet_only_reduction_is_per_pixel_mean`
    # for the structural gate.
    kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)  # (B, H, W)

    # Lane PS (per-class SegNet weighting): when class_weights is supplied,
    # multiply per-pixel KL by the weight at the GT-argmax class. Uses the
    # shared kernel _apply_class_weights so the same L1-normalisation +
    # argmax indexing rules apply across kl_distill_segnet_only / scorer_loss
    # / scorer_loss_cached. The argmax is taken under no_grad inside the
    # kernel so the indexing path stays non-differentiable (matches the
    # upstream SegNet distortion which is itself argmax-based).
    if class_weights is not None:
        kl_per_pixel = _apply_class_weights(kl_per_pixel, gs_logits, class_weights)

    kl = kl_per_pixel.mean() * (T * T)
    return kl, kl.item()


def segnet_uncertainty_weighted_loss(
    rendered_frame_hwc: torch.Tensor,
    gt_frame_hwc: torch.Tensor,
    segnet,
    *,
    weight_floor: float = 0.1,
) -> torch.Tensor:
    """Yousfi #5: weight L1 by inverse-SegNet-entropy.

    Codex R-Lane-D-Issue1 (2026-04-27, INCIDENTAL FIX): this function was
    referenced from train_renderer.py:62 + ~1612 since commit c9ef0884
    ("council R3 fixes: Yousfi uncertainty_loss_floor wired") but was
    NEVER actually defined in tac.losses. The HEAD `from tac.losses import
    segnet_uncertainty_weighted_loss` raised ImportError on every fresh
    import (test runs that "passed" did so because of stale .pyc caches).
    Cleared caches → broken imports → broken tests → broken Lane D
    deployments. Stub added here to unblock; weight_floor parameter is
    consumed so the resolver remains correct. The semantics target Yousfi
    council #5: focus L1 on pixels where SegNet is most CERTAIN (low
    entropy → high inverse-entropy weight), so the renderer biases its
    error budget toward pixels the scorer has high confidence about.

    Args:
        rendered_frame_hwc: (B, H, W, 3) renderer output for the SegNet-
            evaluated frame (index 1 of the pair per upstream's
            `x[:, -1, ...]` indexing).
        gt_frame_hwc: (B, H, W, 3) GT frame, same indexing.
        segnet: the frozen SegNet scorer (call as
            ``segnet(segnet.preprocess_input(_hwc_to_chw(...)))``).
        weight_floor: minimum value of the inverse-entropy weight
            (default 0.1 = matches WILDE/SHIRAZ/DEN profile setting).
            Bounds the weight so high-entropy regions still contribute
            ~10% of L1 — without this, totally uncertain pixels (e.g.
            class boundaries) get zero gradient.

    Returns:
        Scalar loss = inverse_entropy_weight-weighted L1 reconstruction.
    """
    def _frame_to_bchw(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"{name} must be 4D BCHW or BHWC; got {tuple(x.shape)}")
        if x.shape[1] == 3:
            return x.contiguous()
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"{name} must have 3 RGB channels; got {tuple(x.shape)}")

    rx = _frame_to_bchw(rendered_frame_hwc, "rendered_frame")
    gx = _frame_to_bchw(gt_frame_hwc, "gt_frame")
    with torch.no_grad():
        gs_in = segnet.preprocess_input(gx.unsqueeze(1))
        gs_logits = segnet(gs_in)
        # Compute SegNet entropy per pixel (lower entropy = more confident).
        probs = F.softmax(gs_logits, dim=1).clamp_min(1e-6)
        entropy = -(probs * probs.log()).sum(dim=1)  # (B, H_seg, W_seg)
        H_r, W_r = rx.shape[-2:]
        if entropy.shape[-2:] != (H_r, W_r):
            entropy = F.interpolate(
                entropy.unsqueeze(1).float(), size=(H_r, W_r),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        # Inverse-entropy weight, bounded below by weight_floor (so
        # high-entropy pixels still produce gradient).
        max_ent = entropy.max().clamp_min(1e-6)
        inv_ent = 1.0 - (entropy / max_ent)  # in [0, 1]
        weight = inv_ent.clamp_min(float(weight_floor))
    diff = (rx.float() - gx.float()).abs().mean(dim=1)  # (B, H, W) — mean over 3 RGB
    return (diff * weight).mean()


def uniward_quant_noise_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    base_std: float = 2.0,
    kernel_size: int = 8,
    mode: str = "variance",
) -> torch.Tensor:
    """L2 reconstruction loss after UNIWARD-shaped quantization noise.

    The noise field is generated from ``target.detach()`` so the proposal
    field is a fixed scorer-aligned profile, while gradients still flow
    through ``reconstructed``. Accepts either ``(B, 3, H, W)`` or
    ``(B, H, W, 3)`` tensors.
    """
    from tac.fridrich import variance_weighted_noise

    if reconstructed.ndim == 4 and reconstructed.shape[-1] == 3:
        recon_chw = reconstructed.permute(0, 3, 1, 2).contiguous()
        target_chw = target.permute(0, 3, 1, 2).contiguous()
    else:
        recon_chw = reconstructed
        target_chw = target

    noise = variance_weighted_noise(
        target_chw.detach(),
        base_std=base_std,
        kernel_size=kernel_size,
        mode=mode,
    )
    return F.mse_loss(recon_chw + noise, target_chw)


def feature_matching_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    feature_layer: str = "stages.2",
    segnet_weight: float = 100.0,
) -> tuple[torch.Tensor, float, float]:
    """Loss that matches PoseNet INTERMEDIATE features, not outputs.

    .. deprecated::
        This function is superseded by :func:`tac.feature_matching.compute_feature_matching_loss`,
        which supports multi-layer hooking, batched forward passes, configurable
        normalization, and proper layer-weight configuration.  New code should
        import from ``tac.feature_matching`` instead.  This function is kept for
        backward compatibility with existing callers and will be removed in a
        future release.

    Steganalysis insight: match what the detector LOOKS AT (texture statistics
    in mid-layers), not what it OUTPUTS (6 pose values). This is strictly
    more informative because intermediate features are higher-dimensional.

    FastViT-T12 stages:
        stages.0: 64 channels (low-level)
        stages.1: 128 channels (low-mid)
        stages.2: 256 channels (mid — texture statistics, DEFAULT)
        stages.3: 512 channels (high-level, pre-classification)

    The loss = MSE between intermediate features of filtered vs GT frames,
    extracted from the frozen PoseNet's vision backbone. SegNet loss uses
    the standard soft cosine proxy.

    Args:
        feature_layer: dot-separated path into posenet.vision (default "stages.2")
        segnet_weight: weight for SegNet term (default 100, matches scorer formula)

    Returns: (loss, pose_feature_mse, seg_distortion)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    # --- PoseNet feature extraction via hook ---
    # Navigate to the target layer in the vision backbone
    target_module = posenet.vision
    for part in feature_layer.split("."):
        target_module = target_module[int(part)] if part.isdigit() else getattr(target_module, part)

    features_filtered = []
    features_gt = []

    def _hook_fn(storage):
        def hook(module, input, output):
            storage.append(output)

        return hook

    handle = target_module.register_forward_hook(_hook_fn(features_filtered))

    # Forward pass for filtered frames (with gradients)
    posenet_in_f = posenet.preprocess_input(fx)
    fp_out = posenet(posenet_in_f)
    handle.remove()

    # Forward pass for GT frames (no gradients)
    handle_gt = target_module.register_forward_hook(_hook_fn(features_gt))
    with torch.no_grad():
        posenet_in_g = posenet.preprocess_input(gx)
        gp_out = posenet(posenet_in_g)
    handle_gt.remove()

    feat_f = features_filtered[0]
    feat_g = features_gt[0]

    # Feature MSE (higher-dimensional than 6-value pose output)
    pose_feat_mse = (feat_f - feat_g).pow(2).mean()

    # --- SegNet loss (standard soft cosine) ---
    segnet_in_f = segnet.preprocess_input(fx)
    fs_out = segnet(segnet_in_f)
    with torch.no_grad():
        segnet_in_g = segnet.preprocess_input(gx)
        gs_out = segnet(segnet_in_g)

    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

    # Scale feature MSE via sqrt (analogous to sqrt(10*pose) in scorer)
    # The feature space is much higher-dimensional, so raw MSE is larger.
    # Use sqrt to compress dynamic range, with a tunable coefficient.
    loss = segnet_weight * seg_dist + torch.sqrt(pose_feat_mse + 1e-8)
    return loss, pose_feat_mse.item(), seg_dist.item()


def frequency_aware_loss(
    rendered_hwc: torch.Tensor,
    gt_hwc: torch.Tensor,
    band_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Frequency-domain shaping loss via Haar DWT.

    PoseNet reads textures in the mid-high frequency band (1/16 to 1/4 Nyquist).
    Decompose into wavelet bands and weight the loss per-band according to
    PoseNet's sensitivity profile.

    Default weights (from steganalysis literature):
        LL (low-freq structure): 0.5  — PoseNet cares less about DC/coarse
        LH/HL (mid-freq edges):  2.0  — PoseNet texture-sensitive band
        HH (high-freq noise):    1.0  — some sensitivity, but less than mid

    Args:
        rendered_hwc: (B, T, H, W, C) rendered/filtered frames in [0, 255]
        gt_hwc: (B, T, H, W, C) ground truth frames in [0, 255]
        band_weights: optional dict with keys 'll', 'lh', 'hl', 'hh'

    Returns:
        Scalar weighted wavelet-domain loss.
    """
    from .wavelet_renderer import haar_dwt2d

    if band_weights is None:
        band_weights = {"ll": 0.5, "lh": 2.0, "hl": 2.0, "hh": 1.0}

    # Convert to BCHW float
    r = rendered_hwc.float().reshape(-1, *rendered_hwc.shape[2:])  # (B*T, H, W, C)
    g = gt_hwc.float().reshape(-1, *gt_hwc.shape[2:])
    r = r.permute(0, 3, 1, 2) / 255.0  # (B*T, C, H, W)
    g = g.permute(0, 3, 1, 2) / 255.0

    # Ensure even spatial dims for DWT
    if r.shape[-2] % 2 != 0:
        r = r[..., :-1, :]
        g = g[..., :-1, :]
    if r.shape[-1] % 2 != 0:
        r = r[..., :-1]
        g = g[..., :-1]

    r_ll, r_lh, r_hl, r_hh = haar_dwt2d(r)
    g_ll, g_lh, g_hl, g_hh = haar_dwt2d(g)

    loss = (
        band_weights["ll"] * F.mse_loss(r_ll, g_ll)
        + band_weights["lh"] * F.mse_loss(r_lh, g_lh)
        + band_weights["hl"] * F.mse_loss(r_hl, g_hl)
        + band_weights["hh"] * F.mse_loss(r_hh, g_hh)
    )
    return loss


# ── Migrated legacy loss functions ────────────────────────────────────────


def segnet_kl_divergence_loss(
    gt_logprobs: torch.Tensor,
    filtered_frames_hwc: torch.Tensor,
    segnet,
    posenet=None,
    seg_weight: float = 50.0,
) -> tuple[torch.Tensor, float, float]:
    """Direct semantic preservation via KL divergence on SegNet logits.

    Migrated from experiments/train_postfilter_segaware.py (compute_pair_loss_segaware).
    Precomputes GT SegNet log-probabilities, then minimizes KL(GT_logprobs || filtered_logprobs).
    This directly measures semantic map corruption, unlike the softmax-dot-product surrogate.

    NOTE: Not validated on authoritative scorer for comma -- may be more effective than
    softmax surrogate for other codecs. For comma competition, standard loss is proven.

    Args:
        gt_logprobs: (B, C, H_seg, W_seg) pre-computed log-softmax from GT SegNet pass
        filtered_frames_hwc: (B, T, H, W, C) filtered frame pair (float, with gradients)
        segnet: frozen SegNet model
        posenet: optional frozen PoseNet model (if provided, includes PoseNet MSE term)
        seg_weight: weight for SegNet KL term (default 50.0)

    Returns: (loss, seg_kl_value, seg_argmax_disagree)
    """
    fx = _hwc_to_chw(filtered_frames_hwc)

    # SegNet forward on filtered frames
    seg_in = segnet.preprocess_input(fx)
    fs_out = segnet(seg_in)
    filtered_log_probs = F.log_softmax(fs_out, dim=1)
    gt_log_probs_dev = gt_logprobs.to(fx.device)

    # KL(GT || filtered) with log_target=True since both are log-probabilities.
    # 2026-04-27 council forensics (same bug class as kl_distill_segnet_only,
    # see findings.md "Lane G — really dead, or bugged?" + Check M in
    # preflight.py): `reduction="batchmean"` on a (B, C, H_seg, W_seg)
    # tensor under-divides by H_seg × W_seg vs the canonical per-pixel
    # mean. Likely contributor to the historical "KL distill caused
    # PoseNet collapse as primary loss" failure mode in CLAUDE.md
    # Critical Lessons. Switched to per-pixel-per-class mean to match
    # `kl_distill_scorer_loss` line 622+646 + `kl_distill_segnet_only`.
    kl_per_pixel = F.kl_div(
        filtered_log_probs, gt_log_probs_dev,
        reduction="none", log_target=True,
    ).sum(dim=1)  # (B, H_seg, W_seg) — sum over class dim
    seg_kl = kl_per_pixel.mean()

    # Argmax mismatch for monitoring
    with torch.no_grad():
        gt_probs = gt_log_probs_dev.exp()
        seg_argmax_dist = (fs_out.argmax(dim=1) != gt_probs.argmax(dim=1)).float().mean().item()

    loss = seg_weight * seg_kl

    # SegNet-only by design: this function deliberately returns ONLY the
    # segmentation KL term. Callers wanting a combined loss should add a
    # PoseNet term explicitly (see scorer_loss / kl_distill_scorer_loss for
    # combined variants). Keeping the SegNet path isolated lets callers
    # weight or schedule the two terms independently — required by Quantizr's
    # 5-stage training where PoseNet pressure is added/removed per phase.

    return loss, seg_kl.item(), seg_argmax_dist


def saliency_reconstruction_loss_alpha(
    filtered_bchw: torch.Tensor,
    original_bchw: torch.Tensor,
    saliency_map: torch.Tensor,
    alpha: float = 20.0,
) -> torch.Tensor:
    """Per-pixel inverse saliency weighting with explicit alpha parameter.

    Migrated from experiments/train_postfilter_saliency.py (compute_saliency_reconstruction_loss).
    Alpha controls emphasis on SegNet-critical regions. The weight formula is:
        weight = 1.0 + alpha * saliency_map
    Then inverse weighting penalizes corrections on low-saliency pixels.

    IMPORTANT: Alpha is applied exactly once here. The caller must NOT pre-bake alpha
    into saliency_map -- this was the double-alpha bug found in the sweep where alpha
    was applied both in load_saliency_weights() and again in the loss function.

    Args:
        filtered_bchw: (B, C, H, W) float, model output
        original_bchw: (B, C, H, W) float, compressed input
        saliency_map: (B, 1, H, W) raw saliency values in [0, 1] (NOT pre-weighted)
        alpha: saliency emphasis factor (default 20.0)

    Returns: scalar weighted MSE loss
    """
    # Apply alpha exactly once -- this is the single authoritative location
    weights = 1.0 + alpha * saliency_map
    residual = filtered_bchw - original_bchw
    inv_weight = 1.0 / weights.clamp(min=1e-10)
    return (inv_weight * residual.pow(2)).mean()


def posenet_embedding_loss(
    gt_frames_hwc: torch.Tensor,
    filtered_frames_hwc: torch.Tensor,
    posenet,
    layer: str = "summary",
) -> torch.Tensor:
    """Perceptual loss on PoseNet's internal 512-d embeddings.

    Migrated from experiments/train_postfilter_featmatch.py (PoseNetFeatureCapture + compute_loss_with_featmatch).
    Hooks PoseNet's summarizer to capture the 512-dimensional embedding vector, then
    computes normalized L2 loss between GT and filtered embeddings.

    Provides richer gradient signal than 6-dim pose output (512 vs 6 dimensions).
    Analogous to VGG-feature losses in super-resolution, but tuned to the exact model
    that judges us.

    NOTE: Not adopted for comma competition -- standard loss with scorer MSE performed
    better in practice. Potentially useful for other perceptual compression tasks.

    Args:
        gt_frames_hwc: (B, T, H, W, C) ground truth frames
        filtered_frames_hwc: (B, T, H, W, C) filtered frames (requires grad)
        posenet: frozen PoseNet model
        layer: which layer to hook -- 'summary' for summarizer (512-d),
               or dot-path like 'stages.2' for vision backbone (256-d)

    Returns: normalized L2 feature distance (scalar)
    """
    fx = _hwc_to_chw(filtered_frames_hwc)
    gx = _hwc_to_chw(gt_frames_hwc)

    # Set up hook on the target layer
    if layer == "summary":
        target_module = posenet.summarizer
    else:
        target_module = posenet.vision
        for part in layer.split("."):
            target_module = target_module[int(part)] if part.isdigit() else getattr(target_module, part)

    features_filtered = []
    features_gt = []

    def _hook_fn(storage):
        def hook(module, input, output):
            storage.append(output)
        return hook

    # Forward filtered frames (with gradients)
    handle = target_module.register_forward_hook(_hook_fn(features_filtered))
    posenet_in_f = posenet.preprocess_input(fx)
    posenet(posenet_in_f)
    handle.remove()

    # Forward GT frames (no gradients)
    handle_gt = target_module.register_forward_hook(_hook_fn(features_gt))
    with torch.no_grad():
        posenet_in_g = posenet.preprocess_input(gx)
        posenet(posenet_in_g)
    handle_gt.remove()

    feat_f = features_filtered[0]
    feat_g = features_gt[0]

    # Normalized L2: divide by GT feature norm so magnitude is comparable across examples
    feat_diff = feat_f - feat_g
    feat_norm = feat_g.detach().pow(2).mean().sqrt() + 1e-6
    return feat_diff.pow(2).mean() / feat_norm


def band_orthogonality_loss(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
    num_bands: int = 8,
) -> torch.Tensor:
    """DCT-space band masking for disjoint frequency registers in multi-voice ensembles.

    Migrated from experiments/train_postfilter_counterpoint.py (dct2_of_residual + band penalty).
    Enforces that two filter voices operate in complementary frequency bands by penalizing
    the element-wise product of their frequency spectra. Uses rFFT as a cheap DCT proxy.

    Harmonic ensemble loss from Jacob Collier-inspired dual-voice architecture. Experimental.
    The hypothesis: explicit band orthogonality prevents phase-cancellation that killed
    naive weight averaging (scored 2.0469). Two voices should form a "chord, not a doubled note."

    Args:
        output_a: (B, C, H, W) residual output from voice A
        output_b: (B, C, H, W) residual output from voice B
        num_bands: number of frequency bands (unused in current impl, reserved for
                   future band-specific weighting)

    Returns: scalar orthogonality penalty (lower = more disjoint)
    """
    # Guard against zero residuals (epoch 0 with zero-init)
    eps_active = 1e-8
    a_norm_sq = output_a.pow(2).sum()
    b_norm_sq = output_b.pow(2).sum()
    if a_norm_sq.item() <= eps_active or b_norm_sq.item() <= eps_active:
        return torch.zeros((), device=output_a.device, dtype=output_a.dtype)

    # Convert to luma via BT.601 coefficients
    def _to_luma_fft(residual_bchw):
        r = residual_bchw[:, 0]
        g = residual_bchw[:, 1]
        b = residual_bchw[:, 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        return torch.fft.rfft2(luma).abs()

    spec_a = _to_luma_fft(output_a)
    spec_b = _to_luma_fft(output_b)

    # Normalize each spectrum so scale differences don't dominate
    a_scale = spec_a.pow(2).sum().sqrt() + 1e-6
    b_scale = spec_b.pow(2).sum().sqrt() + 1e-6
    spec_a_n = spec_a / a_scale
    spec_b_n = spec_b / b_scale

    return (spec_a_n * spec_b_n).pow(2).sum()


def output_decorrelation_loss(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
) -> torch.Tensor:
    """Cosine-similarity decorrelation penalty for multi-voice ensemble outputs.

    Migrated from experiments/train_postfilter_counterpoint.py (decorrelation term).
    Prevents two filter voices from collapsing to the same residual by penalizing
    the squared cosine similarity between their flattened outputs.

    Harmonic ensemble loss from Jacob Collier-inspired dual-voice architecture. Experimental.

    Args:
        output_a: (B, C, H, W) residual output from voice A
        output_b: (B, C, H, W) residual output from voice B

    Returns: squared cosine similarity (0 = perfectly decorrelated, 1 = identical)
    """
    eps_active = 1e-8
    a_norm_sq = output_a.pow(2).sum()
    b_norm_sq = output_b.pow(2).sum()
    if a_norm_sq.item() <= eps_active or b_norm_sq.item() <= eps_active:
        return torch.zeros((), device=output_a.device, dtype=output_a.dtype)

    a_flat = output_a.reshape(-1)
    b_flat = output_b.reshape(-1)
    a_norm = a_flat / (a_flat.pow(2).sum().sqrt() + 1e-6)
    b_norm = b_flat / (b_flat.pow(2).sum().sqrt() + 1e-6)
    cos_sim = (a_norm * b_norm).sum()
    return cos_sim.pow(2)


def compute_boundary_mask(
    gt_pair: torch.Tensor,
    segnet,
    device: str | torch.device = "cpu",
    kernel_size: int = 5,
) -> torch.Tensor:
    """Compute SegNet class boundary mask from a single GT pair.

    Returns: (H, W) float tensor, 1.0 at boundaries, 0.0 elsewhere.
    """
    p = gt_pair.to(device).float()
    frame = p[:, 1]  # SegNet uses last frame — (B, H, W, C)
    frame_chw = frame.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    with torch.no_grad():
        # preprocess_input expects (B, T, C, H, W) — add T dim, then it does x[:, -1, ...]
        frame_btchw = frame_chw.unsqueeze(1)  # (B, 1, C, H, W)
        seg_in = segnet.preprocess_input(frame_btchw)  # resizes to (384, 512)
        seg_out = segnet(seg_in)
        labels = seg_out.argmax(dim=1).float().unsqueeze(1)

    pad = kernel_size // 2
    max_pool = F.max_pool2d(labels, kernel_size=kernel_size, stride=1, padding=pad)
    min_pool = -F.max_pool2d(-labels, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (max_pool != min_pool).float().squeeze()
    return boundary.cpu()


# ── Trick 24: Boundary-aware loss with morphological dilation ──────────


def boundary_aware_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    boundary_weight: float = 5.0,
    dilation_kernel: int = 5,
    segnet_weight: float = 100.0,
) -> tuple[torch.Tensor, float, float]:
    """SegNet loss with boundary pixels upweighted via morphological dilation.

    SegNet's failure mode is at class boundaries where the receptive field
    straddles two classes and the softmax is near-uniform. These boundary
    pixels contribute disproportionately to mIoU because they are the ONLY
    pixels that can change class under small perturbations.

    This loss identifies boundary pixels (via dilate XOR erode on GT labels),
    then applies boundary_weight multiplier to the per-pixel cross-entropy
    at those locations. Interior pixels (where argmax is confident) get
    weight 1.0.

    A 1-pixel-wide correction band around every class boundary can dominate
    the SegNet score delta with negligible rate cost.

    Args:
        filtered_pair_hwc: (B, T, H, W, C) filtered frame pair.
        gt_pair_hwc: (B, T, H, W, C) ground truth frame pair.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        boundary_weight: multiplier for boundary pixels (default 5.0).
        dilation_kernel: kernel size for morphological ops (default 5).
        segnet_weight: weight for SegNet term (default 100).

    Returns: (loss, pose_distortion, seg_hard_disagree)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)
    gx = _hwc_to_chw(gt_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    # PoseNet: standard MSE
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # SegNet: boundary-weighted cross-entropy with STE
    with torch.no_grad():
        gt_labels = gs_out.argmax(dim=1)  # (B, H_seg, W_seg)
        pred_labels = fs_out.argmax(dim=1)
        hard_disagree = (pred_labels != gt_labels).float().mean()

        # Morphological boundary detection on GT labels
        # Dilate: max_pool; Erode: -max_pool(-x); Boundary = dilate != erode
        labels_float = gt_labels.float().unsqueeze(1)  # (B, 1, H, W)
        pad = dilation_kernel // 2
        dilated = F.max_pool2d(labels_float, dilation_kernel, stride=1, padding=pad)
        eroded = -F.max_pool2d(-labels_float, dilation_kernel, stride=1, padding=pad)
        boundary = (dilated != eroded).float().squeeze(1)  # (B, H_seg, W_seg)

        # Build per-pixel weight map
        pixel_weights = torch.where(boundary > 0.5, boundary_weight, 1.0)
        pixel_weights = pixel_weights / pixel_weights.mean()  # normalize

    B, C, H_seg, W_seg = fs_out.shape
    flat_labels = gt_labels.reshape(-1)
    flat_logits = fs_out.permute(0, 2, 3, 1).reshape(-1, C)
    flat_weights = pixel_weights.reshape(-1)

    per_pixel_ce = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    soft_ce = (per_pixel_ce * flat_weights).mean()

    # STE: forward = hard argmax disagree, backward = weighted cross-entropy
    seg_dist = soft_ce + (hard_disagree - soft_ce).detach()

    loss = segnet_weight * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), hard_disagree.item()


def boundary_aware_loss_cached(
    filtered_pair_hwc: torch.Tensor,
    gt_pose_6: torch.Tensor,
    gt_seg_soft: torch.Tensor,
    gt_labels: torch.Tensor,
    boundary_mask: torch.Tensor,
    posenet,
    segnet,
    boundary_weight: float = 5.0,
    segnet_weight: float = 100.0,
) -> tuple[torch.Tensor, float, float]:
    """Boundary-aware loss with pre-cached GT outputs (P0 optimization).

    Pre-compute gt_labels and boundary_mask once per GT pair and reuse
    across all training iterations.

    Args:
        gt_pose_6: cached PoseNet[:, :6] for GT pair.
        gt_seg_soft: cached softmax(SegNet(GT)).
        gt_labels: cached argmax of GT SegNet logits, (B, H_seg, W_seg) long.
        boundary_mask: cached boundary pixel mask, (B, H_seg, W_seg) float.
        boundary_weight: multiplier for boundary pixels.
        segnet_weight: weight for SegNet term.

    Returns: (loss, pose_distortion, seg_hard_disagree)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gt_pose_6).pow(2).mean()

    with torch.no_grad():
        pred_labels = fs_out.argmax(dim=1)
        hard_disagree = (pred_labels != gt_labels).float().mean()

        pixel_weights = torch.where(boundary_mask > 0.5, boundary_weight, 1.0)
        pixel_weights = pixel_weights / pixel_weights.mean()

    B, C, H_seg, W_seg = fs_out.shape
    flat_labels = gt_labels.reshape(-1)
    flat_logits = fs_out.permute(0, 2, 3, 1).reshape(-1, C)
    flat_weights = pixel_weights.reshape(-1)

    per_pixel_ce = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    soft_ce = (per_pixel_ce * flat_weights).mean()

    seg_dist = soft_ce + (hard_disagree - soft_ce).detach()

    loss = segnet_weight * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), hard_disagree.item()


# ── Trick 29: GAN Discriminator as Scorer Proxy ──────────────────────────


class ScorerProxyDiscriminator(torch.nn.Module):
    """Lightweight PatchGAN discriminator trained to predict scorer loss.

    Unlike a standard GAN discriminator that classifies real/fake, this one
    is trained to REGRESS the actual scorer loss from input frames. This
    makes it a fast, differentiable proxy for the full scorer pipeline.

    Architecture: 4-layer PatchGAN with spectral normalization.
    Input: (B, 3, H, W) float [0, 255] frames.
    Output: (B, 1, H', W') per-patch scorer loss predictions.

    Use cases:
        (a) Faster TTO: replace full scorer with proxy in inner loop.
        (b) Frame-level difficulty estimation: which frames need more bits.
        (c) Architecture search: evaluate candidate postfilters cheaply.

    Args:
        input_channels: number of input channels (default 3 for RGB).
        base_channels: channel width of first conv (default 32).
    """

    def __init__(self, input_channels: int = 3, base_channels: int = 32):
        super().__init__()
        ch = base_channels

        def _disc_block(in_c, out_c, stride=2, norm=True):
            layers = [torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=not norm)
            )]
            if norm:
                layers.append(torch.nn.InstanceNorm2d(out_c))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return torch.nn.Sequential(*layers)

        self.layers = torch.nn.Sequential(
            _disc_block(input_channels, ch, norm=False),     # /2
            _disc_block(ch, ch * 2),                          # /4
            _disc_block(ch * 2, ch * 4),                      # /8
            _disc_block(ch * 4, ch * 4, stride=1),            # /8 (same)
            torch.nn.Conv2d(ch * 4, 1, 4, padding=1),        # regression head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-patch scorer loss.

        Args:
            x: (B, 3, H, W) float tensor in [0, 255].

        Returns:
            (B, 1, H', W') predicted scorer loss per patch.
        """
        return self.layers(x / 255.0)  # normalize to [0, 1] for stability

    def predict_frame_difficulty(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar predicted scorer loss per frame.

        Args:
            x: (B, 3, H, W) float tensor in [0, 255].

        Returns:
            (B,) predicted scorer loss per frame (mean over patches).
        """
        patch_scores = self.forward(x)
        return patch_scores.mean(dim=(1, 2, 3))


def train_scorer_proxy(
    discriminator: ScorerProxyDiscriminator,
    frames_bchw: torch.Tensor,
    scorer_losses: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-4,
) -> list[float]:
    """Train the scorer proxy discriminator on (frame, scorer_loss) pairs.

    Collects training pairs by running the full scorer on frames, then
    trains the discriminator to regress the scorer loss from frames alone.

    This function expects pre-computed scorer losses (from the real scorer)
    paired with their corresponding frames.

    Args:
        discriminator: ScorerProxyDiscriminator to train.
        frames_bchw: (N, 3, H, W) float tensor of frames.
        scorer_losses: (N,) float tensor of real scorer losses for each frame.
        epochs: number of training epochs (default 50).
        lr: learning rate (default 1e-4).

    Returns:
        List of per-epoch mean L1 losses (training curve).
    """
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    N = frames_bchw.shape[0]
    device = frames_bchw.device
    discriminator = discriminator.to(device)
    discriminator.train()

    # Normalize target to [0, 1] for stable regression
    target_min = scorer_losses.min()
    target_range = scorer_losses.max() - target_min + 1e-8
    targets_norm = (scorer_losses - target_min) / target_range

    losses_history: list[float] = []
    batch_size = min(8, N)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            batch_frames = frames_bchw[idx]
            batch_targets = targets_norm[idx]

            pred_patches = discriminator(batch_frames)  # (B, 1, H', W')
            pred_mean = pred_patches.mean(dim=(1, 2, 3))  # (B,)

            loss = F.l1_loss(pred_mean, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses_history.append(epoch_loss / max(n_batches, 1))

    discriminator.eval()
    return losses_history
