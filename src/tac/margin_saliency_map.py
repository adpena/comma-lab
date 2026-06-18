# SPDX-License-Identifier: MIT
"""Detector margin-saliency map — the unified Yousfi lever asset.

The contest's ``d_seg`` is the output of a frozen DETECTOR (EfficientNet-B2
SegNet, argmax-flip rate on a downsampled 512x384 decision grid). The
detector's OWN per-pixel sensitivity — how much its top1-top2 logit margin
moves per unit input perturbation — is the single map that drives all three
sub-0.15 levers identified in the Yousfi council check-in
(``.omx/research/yousfi_council_checkin_unified_margin_saliency_20260618.md``):

1. **d_seg lever** — a margin-hinge weighted by the detector's own gradient
   is first-order (vs a bare hinge = zeroth-order). This is the
   Yousfi-Fridrich 2022 detector-informed embedding cost, computed from the
   detector's OWN gradient (sharper than UNIWARD, which is the wrong domain
   for a SEMANTIC detector).
2. **rate lever** — the COMPLEMENT (low |saliency| pixels) is where
   quantization / dead-zone bytes can be shed at certified ~0 d_seg cost.
3. **survival certification** — the map lives on the detector's decision grid,
   so it certifies which repairs survive the downsample.

This module is the PRODUCER of that map (``∂margin/∂input`` via autograd
through the frozen scorer). The consumer side already exists:
:mod:`tac.logit_margin_sensitivity_weighted` (sensitivity-weighted logit-margin
loss) takes a per-pixel sensitivity tensor as input — this module produces it.

NO FAKE: the saliency is the REAL autograd gradient of the REAL frozen SegNet's
top1-top2 margin w.r.t. its input frame. ``segnet.preprocess_input`` is a clean
slice + bilinear interpolate (verified gradient-reachable: no ``@torch.no_grad``,
no in-place, no round), so the gradient flows end-to-end. No surrogate.

AUTHORITY: this is a DIAGNOSTIC / loss-weighting asset, never a score claim.
The only authoritative ``d_seg`` is ``upstream/evaluate.py`` on a byte-closed
archive. Run on CPU (MPS would 2x-corrupt the SegNet per CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class MarginSaliencyError(ValueError):
    """Raised when margin-saliency inputs are malformed."""


def _topk_margin(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel top1-top2 logit margin, shape (B, H, W), kept differentiable.

    ``logits`` is (B, C, H, W). We use ``torch.topk(k=2)`` along the class dim;
    the difference of the two largest values is a piecewise-linear (hence a.e.
    differentiable) function of the logits, so autograd flows for the
    ``∂margin/∂input`` saliency.
    """
    if logits.dim() != 4:
        raise MarginSaliencyError(
            f"_topk_margin: expected (B, C, H, W) logits, got shape {tuple(logits.shape)}"
        )
    top2 = torch.topk(logits, 2, dim=1)
    return top2.values[:, 0] - top2.values[:, 1]  # (B, H, W)


@dataclass
class MarginSaliency:
    """A computed margin-saliency map on the SegNet decision grid.

    Attributes
    ----------
    saliency : torch.Tensor
        ``|∂(sum_p margin_p) / ∂input|`` reduced over the input RGB channels,
        shape (H, W) on the scorer's decision grid (e.g. 384x512). The detector
        cost: HIGH = the detector is sensitive here (a small input change moves
        the margin a lot = SHARP); LOW = score-blind (the rate-shed band).
    margin : torch.Tensor
        The per-pixel top1-top2 margin of the input frame, shape (H, W). The
        detector's confidence (distance-to-flip) on this specific frame.
    grid_hw : tuple[int, int]
        The (H, W) of the decision grid the maps live on.
    """

    saliency: torch.Tensor
    margin: torch.Tensor
    grid_hw: tuple[int, int]


def compute_margin_saliency_map(
    segnet,
    frame_chw_camera: torch.Tensor,
    *,
    flip_pixel_mask: torch.Tensor | None = None,
) -> MarginSaliency:
    """Compute ``|∂margin/∂input|`` for one camera-res frame through a frozen SegNet.

    Parameters
    ----------
    segnet
        A frozen SegNet exposing ``preprocess_input(x)`` (slice last frame +
        bilinear to the decision grid) and ``__call__`` returning
        ``(1, C, H, W)`` logits. Must be on CPU (CLAUDE.md: never MPS for the
        scorer).
    frame_chw_camera : torch.Tensor
        One camera-res frame, shape (3, H_cam, W_cam), float in [0, 255].
    flip_pixel_mask : torch.Tensor | None
        Optional boolean mask on the decision grid (H, W). When given, the
        scalar that is back-propagated is the margin summed over ONLY those
        pixels (e.g. the current flip pixels), so the saliency answers
        "what input change repairs the flips" rather than the global map.
        When ``None``, all decision-grid pixels contribute (the global map).

    Returns
    -------
    MarginSaliency
        ``saliency`` (H, W) = ``|∂(sum margin)/∂input|`` reduced over RGB,
        ``margin`` (H, W) = the per-pixel margin of this frame.

    Notes
    -----
    The saliency is the gradient of a SUM of per-pixel margins w.r.t. the input
    image. By linearity of ``∂/∂input``, ``∂(sum_p margin_p)/∂input_q`` is the
    superposed influence of input pixel ``q`` on every margin; for the flip-mask
    variant it is the influence on exactly the flip pixels. Reducing over the
    input RGB channels with an L2 norm gives a single per-input-pixel cost.
    """
    import einops

    if frame_chw_camera.dim() != 3 or frame_chw_camera.shape[0] != 3:
        raise MarginSaliencyError(
            "compute_margin_saliency_map: frame_chw_camera must be (3, H, W), "
            f"got {tuple(frame_chw_camera.shape)}"
        )

    frame = frame_chw_camera.detach().clone().requires_grad_(True)  # (3, H_cam, W_cam)
    hwc = frame.permute(1, 2, 0)  # (H_cam, W_cam, 3)
    pair = torch.stack([hwc, hwc]).unsqueeze(0)  # (1, 2, H_cam, W_cam, 3)
    x = einops.rearrange(pair, "b t h w c -> b t c h w")
    seg_in = segnet.preprocess_input(x)  # (1, 3, H, W) on decision grid
    logits = segnet(seg_in)  # (1, C, H, W) WITH grad
    margin = _topk_margin(logits)[0]  # (H, W)

    if flip_pixel_mask is not None:
        if flip_pixel_mask.shape != margin.shape:
            raise MarginSaliencyError(
                "compute_margin_saliency_map: flip_pixel_mask shape "
                f"{tuple(flip_pixel_mask.shape)} != margin grid {tuple(margin.shape)}"
            )
        scalar = margin[flip_pixel_mask.bool()].sum()
    else:
        scalar = margin.sum()

    (grad,) = torch.autograd.grad(scalar, frame, retain_graph=False, create_graph=False)
    # grad: (3, H_cam, W_cam) on the camera grid. Reduce RGB (L2) -> (H_cam, W_cam).
    sal_cam = grad.detach().pow(2).sum(0).sqrt()  # (H_cam, W_cam)

    # Resize the camera-grid saliency onto the decision grid so it aligns 1:1
    # with the margin map (the grid where the argmax actually decides).
    H, W = margin.shape
    sal = torch.nn.functional.interpolate(
        sal_cam.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )[0, 0]
    return MarginSaliency(saliency=sal.detach(), margin=margin.detach(), grid_hw=(H, W))


def saliency_boundary_concentration(
    saliency: torch.Tensor,
    margin: torch.Tensor,
    *,
    boundary_margin: float = 0.5,
) -> dict:
    """Characterize how concentrated the saliency is at the detector boundary.

    Sanity asset for the Yousfi seam: a well-formed margin-saliency map should
    place its mass at LOW-margin (boundary) pixels (where the detector is on a
    class wall and a small input change flips it), not at high-margin interior
    pixels. Returns the fraction of total saliency mass that falls in the
    low-margin band and the ratio of mean-saliency boundary/interior.
    """
    sal = saliency.detach().reshape(-1).double()
    mg = margin.detach().reshape(-1).double()
    boundary = mg < boundary_margin
    total = sal.sum().clamp_min(1e-30)
    mass_boundary = sal[boundary].sum() if boundary.any() else torch.zeros((), dtype=sal.dtype)
    mean_b = sal[boundary].mean() if boundary.any() else torch.zeros((), dtype=sal.dtype)
    mean_i = sal[~boundary].mean() if (~boundary).any() else torch.zeros((), dtype=sal.dtype)
    return {
        "boundary_margin_threshold": float(boundary_margin),
        "frac_pixels_boundary": float(boundary.float().mean()),
        "frac_saliency_mass_in_boundary": float(mass_boundary / total),
        "mean_saliency_boundary": float(mean_b),
        "mean_saliency_interior": float(mean_i),
        "boundary_over_interior_ratio": (
            float(mean_b / mean_i) if float(mean_i) > 1e-30 else float("inf")
        ),
        "saliency_mean": float(sal.mean()),
        "saliency_max": float(sal.max()) if sal.numel() else 0.0,
        "saliency_gini": _gini(sal),
    }


def _gini(x: torch.Tensor) -> float:
    """Gini concentration of a non-negative vector (0=uniform, 1=concentrated)."""
    v = x.detach().reshape(-1).double()
    v = v.clamp_min(0)
    n = v.numel()
    if n == 0 or float(v.sum()) <= 1e-30:
        return 0.0
    sv, _ = torch.sort(v)
    idx = torch.arange(1, n + 1, dtype=torch.double)
    return float((2.0 * (idx * sv).sum()) / (n * sv.sum()) - (n + 1.0) / n)
