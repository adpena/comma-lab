# SPDX-License-Identifier: MIT
"""Canonical camera model and coordinate conventions for comma driving video.

All modules in tac must use these conventions:
- Depth: meters (z-forward from camera)
- Focal length: pixels
- Principal point: pixels from top-left corner
- Flow: pixels per frame
- Angles: radians
- Image coordinates: (x=right, y=down) from top-left

The reference frame matches openpilot's ecef_from_car convention
projected through the camera model.

Usage::

    from tac.camera import (
        COMMA_INTRINSICS, COMMA_EXTRINSICS,
        DEPTH_PRIORS_METERS, CLASS_MEAN_COLORS,
        SEGNET_CLASS_NAMES,
        FRAME_H, FRAME_W, FRAME_C, NUM_CLASSES,
        SEGNET_INPUT_H, SEGNET_INPUT_W,
        CAMERA_H, CAMERA_W,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASS_NAME_TUPLE,
    NUM_CONTEST_SEGNET_CLASSES,
)

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "COMMA_INTRINSICS",
    "COMMA_EXTRINSICS",
    "DEPTH_PRIORS_METERS",
    "CLASS_MEAN_COLORS",
    "SEGNET_CLASS_NAMES",
    "FRAME_H",
    "FRAME_W",
    "FRAME_C",
    "NUM_CLASSES",
    "SEGNET_INPUT_H",
    "SEGNET_INPUT_W",
    "CAMERA_H",
    "CAMERA_W",
    "VANISHING_POINT",
    "HORIZON_BAND",
    "vanishing_point_saliency",
]


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Attributes:
        fx: focal length along x-axis, in pixels.
        fy: focal length along y-axis, in pixels.
        cx: principal point x-coordinate, in pixels from left edge.
        cy: principal point y-coordinate, in pixels from top edge.
    """

    fx: float  # focal length x, pixels
    fy: float  # focal length y, pixels
    cx: float  # principal point x, pixels
    cy: float  # principal point y, pixels


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters relative to road surface.

    Attributes:
        height: camera height above road plane, in meters.
        pitch: camera pitch angle in radians (negative = looking down).
    """

    height: float  # meters above road
    pitch: float  # radians, negative = looking down


# ── Default values for Comma EON dashcam (2018) ──────────────────────────
# Device: Comma EON with OnSemi AR0231AT sensor (1/2.7" CMOS BSI, 3.0μm pixels)
# Source: comma2k19 dataset camera_intrinsics.txt
# Video segment: b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc
#
# Native resolution intrinsics (1164x874):
COMMA_INTRINSICS_NATIVE = CameraIntrinsics(fx=910.0, fy=910.0, cx=582.0, cy=437.0)

# Scorer resolution intrinsics (512x384) — ALL values scaled from native:
# fx_scorer = 910 * (512/1164) ≈ 400.3, fy_scorer = 910 * (384/874) ≈ 399.5
# cx_scorer = 582 * (512/1164) ≈ 256.0, cy_scorer = 437 * (384/874) ≈ 192.0
# Using fx=fy=910 at 512px wide would inflate flow magnitudes by ~2.3x,
# causing depth/focal entanglement in DepthAwareMotionPredictor.
COMMA_INTRINSICS = CameraIntrinsics(fx=400.3, fy=399.5, cx=256.0, cy=192.0)

COMMA_EXTRINSICS = CameraExtrinsics(height=1.2, pitch=-0.02)


# ── Per-class depth priors in METERS ────────────────────────────────────
# These are heuristic depth seeds used across tac geometry modules. Class IDs
# follow tac.semantic_label_contract:
# 0=road, 1=lane_markings, 2=undrivable, 3=movable, 4=my_car.
# Do not infer semantic labels from the numeric depth values; exact archive
# evidence must decide whether these priors help a lane.

DEPTH_PRIORS_METERS: dict[int, float] = {
    0: 30.0,   # road — ground plane, far
    1: 30.0,   # lane markings — same plane as road
    2: 15.0,   # undrivable — legacy medium-distance seed
    3: 1000.0, # movable — legacy far-field seed; verify before promotion
    4: 20.0,   # my_car — legacy medium-distance seed
}


# ── Semantic class mean colors (RGB order, [0, 255]) ───────────────────
# Used for initial frame generation from masks. These are empirical means
# observed in comma driving video for each segmentation class.

CLASS_MEAN_COLORS = torch.tensor(
    [
        [128.0, 128.0, 128.0],  # class 0: road (gray)
        [170.0, 170.0, 170.0],  # class 1: lane markings (light gray)
        [100.0, 80.0, 60.0],    # class 2: undrivable (brown)
        [120.0, 140.0, 160.0],  # class 3: movable objects (blue-gray)
        [180.0, 200.0, 230.0],  # class 4: my_car / ego-car interior seed
    ],
    dtype=torch.float32,
)  # (NUM_CLASSES, 3)


# ── Frame / image dimensions ───────────────────────────────────────────

# SegNet scorer input resolution
FRAME_H: int = 384
FRAME_W: int = 512
FRAME_C: int = 3
NUM_CLASSES: int = NUM_CONTEST_SEGNET_CLASSES
SEGNET_CLASS_NAMES: tuple[str, ...] = CONTEST_SEGNET_CLASS_NAME_TUPLE

# Aliases matching constrained_gen.py naming
SEGNET_INPUT_H: int = FRAME_H
SEGNET_INPUT_W: int = FRAME_W

# Camera native resolution (comma challenge spec)
CAMERA_H: int = 874
CAMERA_W: int = 1164

# ── Vanishing point and saliency prior ────────────────────────────────
# The vanishing point (VP) is where all ego-motion converges. PoseNet's
# tz estimate is maximally sensitive at the VP because forward translation
# maps to radial optical flow centered on the VP. Pixels near the VP
# contribute disproportionately to PoseNet loss.
#
# VP location in scorer coordinates (512x384):
#   cx = 256 (from COMMA_INTRINSICS), cy ~174 (slightly above center —
#   camera is tilted slightly downward, so the horizon and VP sit above
#   the image midpoint). Empirical calibration from comma2k19 highway
#   sequences places the VP at row ~174 in the 384-tall scorer frame.
#
# Horizon band: rows 155-195 in scorer coords (the sky/road boundary)
# contain the sharpest semantic edges and highest SegNet gradient magnitude.

VANISHING_POINT: tuple[int, int] = (256, 174)  # (x, y) in scorer coords (512x384)
HORIZON_BAND: tuple[int, int] = (155, 195)  # (y_top, y_bottom) in scorer coords


def vanishing_point_saliency(
    H: int = 384,
    W: int = 512,
    sigma: float = 40.0,
    min_weight: float = 0.3,
    horizon_boost: float = 2.0,
    horizon_band: tuple[int, int] = HORIZON_BAND,
) -> torch.Tensor:
    """Generate a VP-weighted saliency map for training loss weighting.

    Pixels near the vanishing point get weight 1.0 (maximum PoseNet
    sensitivity). Pixels far from the VP get ``min_weight``. The horizon
    band (sky/road boundary) gets an additional multiplicative boost
    because that is where the sharpest semantic edges live.

    The map is a 2D Gaussian centered on the VP, normalized to
    [min_weight, 1.0], with an additive horizon-band bump.

    Args:
        H: frame height in scorer coordinates (default 384).
        W: frame width in scorer coordinates (default 512).
        sigma: Gaussian spread in pixels (default 40.0).
        min_weight: minimum weight for far-from-VP pixels (default 0.3).
        horizon_boost: multiplicative weight for the horizon band (default 2.0).
        horizon_band: (y_top, y_bottom) row range for horizon boost.

    Returns:
        (1, 1, H, W) float32 tensor of per-pixel weights.
    """
    vp_x, vp_y = VANISHING_POINT

    # Create coordinate grids
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # Gaussian centered on VP
    dist_sq = (xx - vp_x) ** 2 + (yy - vp_y) ** 2
    gaussian = torch.exp(-dist_sq / (2.0 * sigma ** 2))

    # Normalize to [min_weight, 1.0]
    saliency = min_weight + (1.0 - min_weight) * gaussian

    # Horizon band boost: rows in [y_top, y_bottom] get extra weight
    y_top, y_bottom = horizon_band
    horizon_mask = torch.zeros(H, W, dtype=torch.float32)
    y_top_clamped = max(0, min(y_top, H))
    y_bottom_clamped = max(0, min(y_bottom, H))
    if y_bottom_clamped > y_top_clamped:
        horizon_mask[y_top_clamped:y_bottom_clamped, :] = 1.0

    # Smooth the horizon band edges with a small Gaussian taper
    # to avoid hard boundaries in the loss landscape
    if y_bottom_clamped > y_top_clamped:
        band_center = (y_top_clamped + y_bottom_clamped) / 2.0
        band_half = (y_bottom_clamped - y_top_clamped) / 2.0
        band_sigma = band_half * 0.8  # taper within 80% of band width
        band_dist = (yy - band_center).abs()
        band_weight = torch.exp(-((band_dist - band_half).clamp(min=0.0) ** 2) / (2.0 * (band_sigma * 0.3) ** 2))
        horizon_mask = band_weight * horizon_mask.clamp(0, 1) + (1.0 - horizon_mask.clamp(0, 1))
        # Apply boost: multiply saliency by horizon factor where band is active
        saliency = saliency * (1.0 + (horizon_boost - 1.0) * horizon_mask.clamp(0, 1))

    # Re-normalize so max is 1.0 (the VP + horizon intersection)
    saliency = saliency / saliency.max()
    # Ensure minimum weight
    saliency = saliency.clamp(min=min_weight)

    return saliency.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
