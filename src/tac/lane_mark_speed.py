"""Lane-marking-derived per-pair zoom scalar (Hotz analytical, zero archive cost).

Per project_lane_marking_speed_estimation.md and project_posenet_rank1_discovery.md:

PoseNet's first dim accounts for 99.8% of variance and corresponds geometrically
to scalar radial zoom from the Focus of Expansion (FoE). Lane markings (SegNet
class 1) appear in EVERY frame at known physical dimensions (MUTCD: 3m × 15cm).
Their inter-frame radial displacement encodes vehicle speed, which IS the zoom
scalar — derivable from the masks alone at inflate time.

The strategic win: zoom_scalars.bin (2.4KB) becomes COMPUTABLE at inflate from
the masks already in the archive. Net archive saving: 2.4KB. More importantly,
the renderer's pose conditioning is now a function of the masks themselves —
no separate stored signal — which matches the inverse-steganalysis principle
that the scorer only sees the rendered output, not the conditioning input.

Camera (verified from /Users/adpena/Projects/pact/CLAUDE.md project_comma_hardware.md):
    fx = 910 px, principal point pp = (582, 437) at native (1164, 874)
    FoE at scorer scale (256, 174) — verified empirically via radial_zoom.py
    20 fps, 1200 frames per clip

Usage:
    from tac.lane_mark_speed import zoom_from_masks
    zoom_scalars = zoom_from_masks(masks_t, masks_t1)  # (n_pairs,) tensor

The output drops directly into RadialZoomWarp.zoom_scalars.

Validation: estimate_correlation_with_optimized() compares against the output
of optimize_zoom_scalars() — if correlation > 0.9, the analytical estimate is
good enough to skip TTO entirely.
"""
from __future__ import annotations

import torch

from tac.radial_zoom import RadialZoomWarp

# FoE in scorer (512, 384) coordinates (matches RadialZoomWarp default)
DEFAULT_FOE_W = 256.0
DEFAULT_FOE_H = 174.0

# SegNet class index for lane markings (verified from comma2k19 spec)
LANE_MARK_CLASS = 1


def _lane_centroid(mask: torch.Tensor) -> tuple[float, float, int]:
    """Compute the centroid (col, row) of lane-marking pixels in a single mask.

    Args:
        mask: (H, W) integer tensor with class indices in [0, 4]

    Returns:
        (centroid_col, centroid_row, n_pixels). Returns FoE if no lane pixels
        present (degenerate case = no displacement signal).
    """
    is_lane = (mask == LANE_MARK_CLASS)
    n = int(is_lane.sum().item())
    if n == 0:
        return DEFAULT_FOE_W, DEFAULT_FOE_H, 0
    rows, cols = is_lane.nonzero(as_tuple=True)
    return float(cols.float().mean().item()), float(rows.float().mean().item()), n


def _radial_distance(centroid_col: float, centroid_row: float,
                     foe_w: float = DEFAULT_FOE_W,
                     foe_h: float = DEFAULT_FOE_H) -> float:
    """Euclidean distance from FoE in pixels."""
    dx = centroid_col - foe_w
    dy = centroid_row - foe_h
    return (dx * dx + dy * dy) ** 0.5


def lane_displacement_per_pair(mask_t: torch.Tensor, mask_t1: torch.Tensor,
                                foe_w: float = DEFAULT_FOE_W,
                                foe_h: float = DEFAULT_FOE_H) -> float:
    """Radial displacement of lane-mark centroid from t to t1.

    Positive = mask drift away from FoE (forward motion). Negative = drift
    toward FoE (backward — should be rare on a forward-driving clip).

    Args:
        mask_t, mask_t1: (H, W) class-index masks for the pair
        foe_w, foe_h: FoE coordinates

    Returns:
        Radial displacement in pixels (signed).
    """
    cw_t, ch_t, n_t = _lane_centroid(mask_t)
    cw_t1, ch_t1, n_t1 = _lane_centroid(mask_t1)
    if n_t == 0 or n_t1 == 0:
        # Degenerate: no signal → assume no zoom (identity)
        return 0.0
    rad_t = _radial_distance(cw_t, ch_t, foe_w, foe_h)
    rad_t1 = _radial_distance(cw_t1, ch_t1, foe_w, foe_h)
    return rad_t1 - rad_t


def zoom_from_masks(masks: torch.Tensor,
                    foe_w: float = DEFAULT_FOE_W,
                    foe_h: float = DEFAULT_FOE_H,
                    smoothing: float = 0.3) -> torch.Tensor:
    """Compute per-pair scalar log-zoom from a sequence of masks.

    Pairs are formed as (masks[2k], masks[2k+1]) for k in [0, n_pairs).

    The log-zoom is the log of the ratio of post-zoom to pre-zoom radial
    distances. Per RadialZoomWarp's parametrisation:
        grid = foe + exp(s) * (coord_grid - foe)
    so larger s → larger radial expansion. We invert from displacement:
        s ≈ log(rad_t1 / rad_t) ≈ displacement / mean_radial_distance

    Smoothing (default 0.3) blends each pair toward the local 3-pair median to
    suppress lane-change spikes that aren't actually forward motion.

    Args:
        masks: (N, H, W) integer mask tensor (N = 2 * n_pairs)
        foe_w, foe_h: Focus of Expansion in mask coordinates
        smoothing: 0..1, weight of local-median blend (0 = no smoothing)

    Returns:
        zoom_scalars: (n_pairs,) float tensor of log-zoom values
    """
    if masks.ndim != 3:
        raise ValueError(f"Expected (N, H, W) masks, got shape {masks.shape}")
    n_frames = masks.shape[0]
    if n_frames % 2 != 0:
        raise ValueError(f"Need even frame count, got {n_frames}")
    n_pairs = n_frames // 2

    # Per-pair displacement
    disps = torch.zeros(n_pairs, dtype=torch.float32)
    rad_means = torch.zeros(n_pairs, dtype=torch.float32)
    for k in range(n_pairs):
        m_t, m_t1 = masks[2 * k], masks[2 * k + 1]
        cw_t, ch_t, n_t = _lane_centroid(m_t)
        cw_t1, ch_t1, n_t1 = _lane_centroid(m_t1)
        if n_t == 0 or n_t1 == 0:
            disps[k] = 0.0
            rad_means[k] = 1.0  # avoid div-by-zero in log estimate
            continue
        rad_t = _radial_distance(cw_t, ch_t, foe_w, foe_h)
        rad_t1 = _radial_distance(cw_t1, ch_t1, foe_w, foe_h)
        disps[k] = rad_t1 - rad_t
        rad_means[k] = max((rad_t + rad_t1) * 0.5, 1.0)

    # Log-zoom estimate. Approximation: log(1 + d/r) ≈ d/r for small d/r.
    raw_zoom = disps / rad_means

    # Local-median smoothing (3-pair window) to suppress lane-change spikes
    if smoothing > 0:
        smoothed = raw_zoom.clone()
        for k in range(1, n_pairs - 1):
            window = torch.stack([raw_zoom[k - 1], raw_zoom[k], raw_zoom[k + 1]])
            smoothed[k] = (1 - smoothing) * raw_zoom[k] + smoothing * window.median()
        raw_zoom = smoothed

    return raw_zoom


def build_zoom_warp_from_masks(masks: torch.Tensor,
                                foe_w: float = DEFAULT_FOE_W,
                                foe_h: float = DEFAULT_FOE_H,
                                smoothing: float = 0.3) -> RadialZoomWarp:
    """Construct a RadialZoomWarp pre-loaded with mask-derived zoom scalars.

    This is the zero-archive-cost path: at inflate time, given only the masks
    (which are already in the archive as masks.mkv), construct the zoom warp
    on the fly without needing zoom_scalars.bin.

    The renderer's forward pass uses the resulting RadialZoomWarp identically
    to one loaded from zoom_scalars.bin — same module, same outputs.
    """
    n_pairs = masks.shape[0] // 2
    zoom_warp = RadialZoomWarp(n_pairs=n_pairs, foe_h=foe_h, foe_w=foe_w)
    estimated = zoom_from_masks(masks, foe_w=foe_w, foe_h=foe_h, smoothing=smoothing)
    # Clamp to RadialZoomWarp's |s| <= max_zoom_log bound (default 0.1)
    estimated = estimated.clamp(-zoom_warp.max_zoom_log, zoom_warp.max_zoom_log)
    with torch.no_grad():
        zoom_warp.zoom_scalars.data.copy_(estimated)
    return zoom_warp


def estimate_correlation_with_optimized(
    masks: torch.Tensor,
    optimized_scalars: torch.Tensor,
    foe_w: float = DEFAULT_FOE_W,
    foe_h: float = DEFAULT_FOE_H,
    smoothing: float = 0.3,
) -> dict[str, float]:
    """Validate: how close is the analytical estimate to the optimized values?

    Computes Pearson correlation, RMSE, and per-pair max absolute error.
    Correlation > 0.9 means analytical is good enough; can skip TTO + storage.

    Args:
        masks: (N, H, W) class masks
        optimized_scalars: (n_pairs,) values from optimize_zoom_scalars()
        foe_w, foe_h: FoE coords used for both estimates
        smoothing: smoothing applied to the analytical estimate

    Returns:
        dict with "correlation", "rmse", "max_abs_err", "estimated", "optimized"
    """
    estimated = zoom_from_masks(masks, foe_w=foe_w, foe_h=foe_h, smoothing=smoothing)
    if estimated.shape != optimized_scalars.shape:
        raise ValueError(
            f"shape mismatch: estimated {estimated.shape} vs optimized {optimized_scalars.shape}"
        )
    # Pearson correlation. Use unbiased=False (population std, div by N) so
    # numerator and denominator use the same N normalization — otherwise the
    # default Bessel correction (div by N-1) introduces a (N-1)/N factor and
    # corr-of-x-with-itself returns 1 - 1/N instead of exactly 1.
    e_centered = estimated - estimated.mean()
    o_centered = optimized_scalars - optimized_scalars.mean()
    e_std = e_centered.std(unbiased=False).clamp(min=1e-10)
    o_std = o_centered.std(unbiased=False).clamp(min=1e-10)
    corr = (e_centered * o_centered).mean() / (e_std * o_std)
    rmse = (estimated - optimized_scalars).pow(2).mean().sqrt()
    max_err = (estimated - optimized_scalars).abs().max()
    return {
        "correlation": float(corr.item()),
        "rmse": float(rmse.item()),
        "max_abs_err": float(max_err.item()),
        "estimated": estimated,
        "optimized": optimized_scalars,
    }
