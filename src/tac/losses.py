"""Loss functions for task-aware codec training.

All losses operate on frame pairs in (1, 2, H, W, 3) HWC format and
handle the BTCHW conversion internally before passing to the scorers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scorer_forward_pair(pair_btchw, posenet, segnet):
    """Forward pass through both scorer networks.

    Args:
        pair_btchw: (B, T, C, H, W) float tensor
        posenet: frozen PoseNet model
        segnet: frozen SegNet model

    Returns:
        (posenet_output, segnet_output) dicts/tensors
    """
    posenet_in = posenet.preprocess_input(pair_btchw)
    posenet_out = posenet(posenet_in)
    segnet_in = segnet.preprocess_input(pair_btchw)
    segnet_out = segnet(segnet_in)
    return posenet_out, segnet_out


def _hwc_to_chw(pair_hwc: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, H, W, C) to (B, T, C, H, W) float."""
    return pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()


def scorer_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
) -> tuple[torch.Tensor, float, float]:
    """Standard scorer loss: 100*seg + sqrt(10*pose).

    Uses soft cosine for SegNet (differentiable proxy for argmax disagreement).

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
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def scorer_loss_cached(
    filtered_pair_hwc: torch.Tensor,
    gt_pose_6: torch.Tensor,
    gt_seg_soft: torch.Tensor,
    posenet,
    segnet,
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

    Returns: (loss, pose_distortion, seg_distortion)
    """
    fx = _hwc_to_chw(filtered_pair_hwc)

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gt_pose_6).pow(2).mean()

    pred_soft = F.softmax(fs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_seg_soft).sum(dim=1).mean()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


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
    temperature: float = 5.0,
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
        temperature: softmax temperature (anneal from 5.0 → 1.0 over training)
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

    T = temperature
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


def feature_matching_loss(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet,
    segnet,
    feature_layer: str = "stages.2",
    segnet_weight: float = 100.0,
) -> tuple[torch.Tensor, float, float]:
    """Loss that matches PoseNet INTERMEDIATE features, not outputs.

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
