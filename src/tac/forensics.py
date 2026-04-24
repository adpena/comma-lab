"""Forensic analysis tools for detecting scorer-visible artifacts.

These tools reveal what the SegNet and PoseNet scorers actually see,
helping us optimise in scorer space rather than pixel space.

Yousfi (challenge creator, Fridrich's PhD student) is a forensic image
analyst.  His SegNet IS a steganalysis detector.  This module provides the
analytical tools to detect the kinds of artifacts Yousfi would notice:

* **Boundary artifacts** from zero-padded convolutions (systematic intensity
  shifts near frame edges).
* **SegNet class-boundary errors** (the argmax decision boundary is where
  SegNet distortion concentrates).
* **PoseNet sensitivity maps** (per-pixel Jacobian norm — which pixels
  PoseNet cares about most).
* **Eval roundtrip distortion maps** (where the 384 -> 874 -> uint8 -> 384
  chain introduces the most error).

Council origin
--------------
* Fridrich: "steganalysis detectors respond to local statistics — boundary
  artifacts are exactly what a forensic detector flags."
* Yousfi: "zero-padding introduces artificial boundary statistics that EfficientNet
  picks up in the first stem conv."
* Hotz: "just compute the Jacobian, it tells you everything."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.camera import CAMERA_H, CAMERA_W, SEGNET_INPUT_H, SEGNET_INPUT_W

__all__ = [
    "boundary_artifact_score",
    "segnet_class_boundary_analysis",
    "posenet_sensitivity_map",
    "eval_roundtrip_distortion_map",
]


# ======================================================================
# 1. Boundary artifact detection
# ======================================================================


def boundary_artifact_score(
    frames: torch.Tensor,
    border_width: int = 8,
) -> dict[str, float]:
    """Measure statistical difference between border and interior pixels.

    Yousfi flagged boundary artifacts from zero-padded convolutions as a
    primary detection signal.  A steganalysis detector like SegNet sees
    these as anomalous local statistics.

    Computes three metrics comparing the ``border_width``-pixel border
    region against the interior:

    * **mean_diff**: absolute difference of mean intensity.
    * **var_ratio**: ratio of border variance to interior variance
      (values far from 1.0 indicate artifacts).
    * **grad_ratio**: ratio of mean gradient magnitude at the border to
      the interior (high values = sharp boundary artifacts).

    The overall ``artifact_score`` is the geometric mean of the normalised
    per-metric scores, where 0.0 = no detectable artifact and higher is worse.

    Parameters
    ----------
    frames : torch.Tensor
        ``(B, C, H, W)`` or ``(B, H, W, C)`` float frames in ``[0, 255]``.
        If last dim is 3, auto-detected as HWC and transposed.
    border_width : int
        Width of the border region in pixels (default 8).

    Returns
    -------
    dict[str, float]
        Keys: ``mean_diff``, ``var_ratio``, ``grad_ratio``,
        ``artifact_score``.
    """
    # Normalise to (B, C, H, W)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.shape[-1] == 3 and frames.shape[1] != 3:
        frames = frames.permute(0, 3, 1, 2)

    B, C, H, W = frames.shape
    bw = border_width

    if bw * 2 >= H or bw * 2 >= W:
        raise ValueError(
            f"border_width={bw} too large for frame size ({H}, {W})"
        )

    # Build border mask (True = border pixel)
    border_mask = torch.zeros(H, W, dtype=torch.bool, device=frames.device)
    border_mask[:bw, :] = True     # top
    border_mask[-bw:, :] = True    # bottom
    border_mask[:, :bw] = True     # left
    border_mask[:, -bw:] = True    # right
    interior_mask = ~border_mask

    # Flatten spatial dims for masked selection
    flat = frames.reshape(B * C, H, W)

    border_pixels = flat[:, border_mask]    # (B*C, n_border)
    interior_pixels = flat[:, interior_mask]  # (B*C, n_interior)

    # ── Metric 1: mean intensity difference ───────────────────────
    border_mean = border_pixels.mean().item()
    interior_mean = interior_pixels.mean().item()
    mean_diff = abs(border_mean - interior_mean)

    # ── Metric 2: variance ratio ─────────────────────────────────
    border_var = border_pixels.var().item()
    interior_var = interior_pixels.var().item()
    # Avoid division by zero; add small epsilon
    var_ratio = border_var / (interior_var + 1e-8)

    # ── Metric 3: gradient magnitude ratio ────────────────────────
    # Compute spatial gradients (Sobel-like: just finite differences)
    grad_h = (frames[:, :, 1:, :] - frames[:, :, :-1, :]).abs()  # (B, C, H-1, W)
    grad_w = (frames[:, :, :, 1:] - frames[:, :, :, :-1]).abs()  # (B, C, H, W-1)

    # Mean gradient magnitude per pixel (average over both directions)
    # Pad to original size for mask alignment
    grad_h_padded = F.pad(grad_h, (0, 0, 0, 1))  # (B, C, H, W)
    grad_w_padded = F.pad(grad_w, (0, 1, 0, 0))   # (B, C, H, W)
    grad_mag = (grad_h_padded + grad_w_padded) / 2.0  # (B, C, H, W)

    grad_flat = grad_mag.reshape(B * C, H, W)
    border_grad = grad_flat[:, border_mask].mean().item()
    interior_grad = grad_flat[:, interior_mask].mean().item()
    grad_ratio = border_grad / (interior_grad + 1e-8)

    # ── Overall artifact score ────────────────────────────────────
    # Normalise each metric to a 0+ range where 0 = no artifact:
    #   mean_diff / 255 (max possible range)
    #   |log(var_ratio)| (0 when ratio=1)
    #   |log(grad_ratio)| (0 when ratio=1)
    import math

    norm_mean = mean_diff / 255.0
    norm_var = abs(math.log(var_ratio + 1e-8))
    norm_grad = abs(math.log(grad_ratio + 1e-8))

    # Geometric mean (all three must be low for a good score)
    artifact_score = (norm_mean * norm_var * norm_grad) ** (1.0 / 3.0)

    return {
        "mean_diff": mean_diff,
        "var_ratio": var_ratio,
        "grad_ratio": grad_ratio,
        "artifact_score": artifact_score,
    }


# ======================================================================
# 2. SegNet class-boundary analysis
# ======================================================================


def segnet_class_boundary_analysis(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    segnet: nn.Module,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float | dict]:
    """Analyse SegNet disagreement at class boundaries vs interior.

    SegNet distortion is defined as the hard argmax disagreement rate.
    Almost all disagreement concentrates at semantic class boundaries
    (road/lane, road/vehicle, sky/background) — the logit margins are
    smallest there.  This analysis quantifies that concentration and
    identifies the worst class transitions.

    Parameters
    ----------
    pred_frames : torch.Tensor
        ``(N, H, W, 3)`` float candidate frames in ``[0, 255]``.
    gt_frames : torch.Tensor
        ``(N, H, W, 3)`` float ground-truth frames in ``[0, 255]``.
    segnet : nn.Module
        Frozen SegNet model.  Expects ``(B, T, C, H, W)`` input with
        ``T=1``.  Returns ``(B, 5, H', W')`` class logits.
    device : torch.device
        Compute device.
    batch_size : int
        Frames per forward pass (memory management).

    Returns
    -------
    dict
        Keys:

        * ``boundary_error_rate``: disagreement rate at class boundaries.
        * ``interior_error_rate``: disagreement rate in class interiors.
        * ``boundary_interior_ratio``: concentration ratio (higher =
          errors more concentrated at boundaries).
        * ``overall_error_rate``: total disagreement rate.
        * ``n_boundary_pixels``: count of boundary pixels analysed.
        * ``n_interior_pixels``: count of interior pixels analysed.
        * ``worst_transitions``: list of (class_a, class_b, error_count)
          tuples for the worst GT class transitions.
    """
    N = pred_frames.shape[0]
    H_in, W_in = pred_frames.shape[1], pred_frames.shape[2]

    boundary_errors = 0
    boundary_total = 0
    interior_errors = 0
    interior_total = 0
    transition_errors: dict[tuple[int, int], int] = {}

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            # Prepare inputs: (B, 1, C, H, W) — SegNet uses last frame of seq
            pred_batch = pred_frames[start:end].to(device)
            gt_batch = gt_frames[start:end].to(device)

            # Convert HWC -> CHW and add time dim
            pred_chw = pred_batch.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
            gt_chw = gt_batch.permute(0, 3, 1, 2).contiguous()

            # Resize to scorer input
            if H_in != SEGNET_INPUT_H or W_in != SEGNET_INPUT_W:
                pred_chw = F.interpolate(
                    pred_chw,
                    size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                    mode="bilinear",
                    align_corners=False,
                )
                gt_chw = F.interpolate(
                    gt_chw,
                    size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                    mode="bilinear",
                    align_corners=False,
                )

            # SegNet expects (B, T, C, H, W) — use T=1, feeds x[:, -1, ...]
            pred_5d = pred_chw.unsqueeze(1)  # (B, 1, 3, H', W')
            gt_5d = gt_chw.unsqueeze(1)

            pred_logits = segnet(pred_5d)  # (B, 5, H', W')
            gt_logits = segnet(gt_5d)

            pred_cls = pred_logits.argmax(dim=1)  # (B, H', W')
            gt_cls = gt_logits.argmax(dim=1)

            disagree = (pred_cls != gt_cls)  # (B, H', W')

            # Find class boundaries in GT: pixels where any 4-connected
            # neighbour has a different class
            gt_shifted_h = torch.zeros_like(gt_cls, dtype=torch.bool)
            gt_shifted_w = torch.zeros_like(gt_cls, dtype=torch.bool)
            gt_shifted_h[:, 1:, :] |= (gt_cls[:, 1:, :] != gt_cls[:, :-1, :])
            gt_shifted_h[:, :-1, :] |= (gt_cls[:, :-1, :] != gt_cls[:, 1:, :])
            gt_shifted_w[:, :, 1:] |= (gt_cls[:, :, 1:] != gt_cls[:, :, :-1])
            gt_shifted_w[:, :, :-1] |= (gt_cls[:, :, :-1] != gt_cls[:, :, 1:])
            is_boundary = gt_shifted_h | gt_shifted_w  # (B, H', W')
            is_interior = ~is_boundary

            boundary_errors += disagree[is_boundary].sum().item()
            boundary_total += is_boundary.sum().item()
            interior_errors += disagree[is_interior].sum().item()
            interior_total += is_interior.sum().item()

            # Track worst transitions: for each disagreeing boundary pixel,
            # find the GT class pair at that boundary
            for b in range(B):
                err_locs = (disagree[b] & is_boundary[b]).nonzero(as_tuple=False)
                if err_locs.shape[0] == 0:
                    continue
                for r, c in err_locs[:100].tolist():  # cap per frame for speed
                    cls_here = gt_cls[b, r, c].item()
                    # Find the neighbouring class that differs
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        h_dim, w_dim = gt_cls.shape[1], gt_cls.shape[2]
                        if 0 <= nr < h_dim and 0 <= nc < w_dim:
                            cls_nbr = gt_cls[b, nr, nc].item()
                            if cls_nbr != cls_here:
                                key = (min(cls_here, cls_nbr), max(cls_here, cls_nbr))
                                transition_errors[key] = transition_errors.get(key, 0) + 1
                                break  # one transition per pixel

    boundary_error_rate = boundary_errors / max(boundary_total, 1)
    interior_error_rate = interior_errors / max(interior_total, 1)
    overall_n = boundary_total + interior_total
    overall_error_rate = (boundary_errors + interior_errors) / max(overall_n, 1)

    ratio = boundary_error_rate / max(interior_error_rate, 1e-8)

    # Sort transitions by error count descending
    worst = sorted(transition_errors.items(), key=lambda kv: kv[1], reverse=True)
    worst_transitions = [
        {"class_a": k[0], "class_b": k[1], "error_count": v}
        for k, v in worst[:10]
    ]

    return {
        "boundary_error_rate": boundary_error_rate,
        "interior_error_rate": interior_error_rate,
        "boundary_interior_ratio": ratio,
        "overall_error_rate": overall_error_rate,
        "n_boundary_pixels": boundary_total,
        "n_interior_pixels": interior_total,
        "worst_transitions": worst_transitions,
    }


# ======================================================================
# 3. PoseNet sensitivity map
# ======================================================================


def posenet_sensitivity_map(
    frame_pair: torch.Tensor,
    posenet: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-pixel PoseNet sensitivity via Jacobian.

    Returns an ``(H, W)`` map of ``||d(PoseNet_output) / d(pixel)||``
    — shows which pixels PoseNet cares about most.  High-sensitivity
    regions need more rendering fidelity.

    The Jacobian is computed for the first 6 pose dimensions (the scored
    ones) with respect to frame pixel values.

    Parameters
    ----------
    frame_pair : torch.Tensor
        ``(2, H, W, 3)`` float frame pair in ``[0, 255]`` (HWC format).
        These are the two consecutive frames PoseNet evaluates.
    posenet : nn.Module
        Frozen PoseNet model with differentiable ``preprocess_input``.
    device : torch.device
        Compute device.

    Returns
    -------
    torch.Tensor
        ``(H, W)`` float sensitivity map (L2 norm of per-pixel Jacobian).
    """
    H, W = frame_pair.shape[1], frame_pair.shape[2]

    # Convert to CHW, add batch dim
    pair_chw = frame_pair.permute(0, 3, 1, 2).contiguous()  # (2, 3, H, W)
    pair_chw = pair_chw.unsqueeze(0).to(device)  # (1, 2, 3, H, W)

    # Resize to scorer input if needed
    if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
        flat = pair_chw.reshape(2, 3, H, W)
        flat = F.interpolate(
            flat,
            size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
            mode="bilinear",
            align_corners=False,
        )
        pair_chw = flat.reshape(1, 2, 3, SEGNET_INPUT_H, SEGNET_INPUT_W)
        H, W = SEGNET_INPUT_H, SEGNET_INPUT_W

    # Require grad on the frame pair to get the Jacobian
    pair_chw = pair_chw.detach().requires_grad_(True)

    # Forward through PoseNet
    posenet.eval()
    preprocessed = posenet.preprocess_input(pair_chw)

    # Verify gradients flow through preprocess (same gradient bug check as radial_zoom)
    assert preprocessed.requires_grad, (
        "PoseNet preprocess_input kills gradients. Use load_differentiable_scorers()."
    )

    pose_out_raw = posenet(preprocessed)  # dict{'pose': (1, 12)} or tensor
    pose_vals = pose_out_raw["pose"] if isinstance(pose_out_raw, dict) else pose_out_raw

    # Compute gradient for each of the 6 scored pose dims
    # and accumulate the squared gradient per-pixel
    sensitivity_sq = torch.zeros(1, 2, 3, H, W, device=device)

    for dim_idx in range(6):
        pair_chw.grad = None
        pose_vals[0, dim_idx].backward(retain_graph=(dim_idx < 5))
        if pair_chw.grad is not None:
            sensitivity_sq += pair_chw.grad.detach() ** 2

    # L2 norm across pose dims, channels, and both frames: sqrt(sum of squares)
    # Sum over (frames=dim1, channels=dim2), keep spatial (H, W)
    sensitivity = sensitivity_sq.sum(dim=(0, 1, 2)).sqrt()  # (H, W)

    return sensitivity.cpu()


# ======================================================================
# 4. Eval roundtrip distortion map
# ======================================================================


def eval_roundtrip_distortion_map(
    frames: torch.Tensor,
    target_h: int = CAMERA_H,
    target_w: int = CAMERA_W,
) -> torch.Tensor:
    """Compute per-pixel distortion from the eval roundtrip chain.

    The contest scorer takes rendered frames at scorer resolution
    (384 x 512), upscales to camera resolution (874 x 1164), quantises
    to uint8, then downscales back to scorer resolution.  This function
    shows *where* that chain introduces the most error, so the renderer
    can learn to compensate in high-distortion regions.

    Parameters
    ----------
    frames : torch.Tensor
        ``(B, 3, H, W)`` or ``(B, H, W, 3)`` float frames in ``[0, 255]``
        at scorer resolution.  If last dim is 3 and dim 1 is not 3,
        auto-detected as HWC and transposed.
    target_h : int
        Camera native height (874).
    target_w : int
        Camera native width (1164).

    Returns
    -------
    torch.Tensor
        ``(B, H, W)`` float distortion map (mean absolute error per pixel
        across channels after the roundtrip).
    """
    # Normalise to (B, C, H, W)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.shape[-1] == 3 and frames.shape[1] != 3:
        frames = frames.permute(0, 3, 1, 2)

    frames = frames.float()
    orig_h, orig_w = frames.shape[2], frames.shape[3]

    # Step 1: upscale to camera resolution
    up = F.interpolate(
        frames,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    # Step 2: uint8 quantisation (hard, no STE — this is analysis, not training)
    up_quantised = up.round().clamp(0, 255)

    # Step 3: downscale back to scorer resolution
    down = F.interpolate(
        up_quantised,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )

    # Per-pixel absolute error, averaged over channels
    distortion = (down - frames).abs().mean(dim=1)  # (B, H, W)

    return distortion
