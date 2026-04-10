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
    pose_per_sample = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean(
        dim=tuple(range(1, fp_out["pose"].ndim))
    )

    # SegNet: hard argmax disagreement (matches upstream compute_distortion)
    diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
    seg_per_sample = diff.mean(dim=tuple(range(1, diff.ndim)))

    avg_p = pose_per_sample.mean().item()
    avg_s = seg_per_sample.mean().item()
    score = 100.0 * avg_s + math.sqrt(10.0 * avg_p)
    return score, avg_p, avg_s


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
        boundary_mask: (H, W) float tensor, 1.0 at boundaries, 0.0 elsewhere
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
    inv_weight = 1.0 / combined
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
    inv_weight = 1.0 / sal_weights
    return (inv_weight * residual.pow(2)).mean()


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
