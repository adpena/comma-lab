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
        FRAME_H, FRAME_W, FRAME_C, NUM_CLASSES,
        SEGNET_INPUT_H, SEGNET_INPUT_W,
        CAMERA_H, CAMERA_W,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "COMMA_INTRINSICS",
    "COMMA_EXTRINSICS",
    "DEPTH_PRIORS_METERS",
    "CLASS_MEAN_COLORS",
    "FRAME_H",
    "FRAME_W",
    "FRAME_C",
    "NUM_CLASSES",
    "SEGNET_INPUT_H",
    "SEGNET_INPUT_W",
    "CAMERA_H",
    "CAMERA_W",
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

# Scorer resolution intrinsics (512x384) — scaled from native:
# fx_scorer = 910 * (512/1164) ≈ 400.3, but scorers resize internally
# via preprocess_input, so geometric computations on scorer-resolution
# frames use the scaled principal point: (582/1164*512, 437/874*384) ≈ (256, 192)
COMMA_INTRINSICS = CameraIntrinsics(fx=910.0, fy=910.0, cx=256.0, cy=192.0)

COMMA_EXTRINSICS = CameraExtrinsics(height=1.2, pitch=-0.02)


# ── Per-class depth priors in METERS ────────────────────────────────────
# These are the canonical depth values used across all tac modules.
# road=far ground plane, lane=same as road, vehicle=medium,
# sky=effectively infinity (large value), background=medium-far.

DEPTH_PRIORS_METERS: dict[int, float] = {
    0: 30.0,   # road — ground plane, far
    1: 30.0,   # lane markings — same plane as road
    2: 15.0,   # vehicle / undrivable — medium distance
    3: 1000.0, # sky — effectively infinity
    4: 20.0,   # background — medium-far
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
        [180.0, 200.0, 230.0],  # class 4: sky (light blue)
    ],
    dtype=torch.float32,
)  # (NUM_CLASSES, 3)


# ── Frame / image dimensions ───────────────────────────────────────────

# SegNet scorer input resolution
FRAME_H: int = 384
FRAME_W: int = 512
FRAME_C: int = 3
NUM_CLASSES: int = 5

# Aliases matching constrained_gen.py naming
SEGNET_INPUT_H: int = FRAME_H
SEGNET_INPUT_W: int = FRAME_W

# Camera native resolution (comma challenge spec)
CAMERA_H: int = 874
CAMERA_W: int = 1164
