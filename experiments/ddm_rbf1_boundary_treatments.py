# SPDX-License-Identifier: MIT
"""Generic post-render operators; no scorer, fitted values, or video tables.

Research-only until n600 receiver/scorer and cross-host parity proofs exist.
Guided filtering: He/Sun/Tang, https://people.csail.mit.edu/kaiming/eccv10/ .
Categorical one-hot guidance uses the zero-ridge local least-squares solution:
each window predicts its per-class RGB mean, and overlapping predictions average.
SSAA: four centred samples of the rendered bilinear RGB reconstruction.
SDF: local signed Euclidean distance, analogous to the rasterization primitive
in https://github.com/Chlumsky/msdfgen (algorithm anchor, no code copied).
All arithmetic executes in NumPy fp32 on CPU, including on a CUDA host.
"""
from __future__ import annotations

import numpy as np


def edges(labels: np.ndarray) -> np.ndarray:
    """Mark both endpoints of every four-connected inter-class edge."""
    out = np.zeros(labels.shape, dtype=bool)
    change = labels[1:] != labels[:-1]
    out[1:] |= change
    out[:-1] |= change
    change = labels[:, 1:] != labels[:, :-1]
    out[:, 1:] |= change
    out[:, :-1] |= change
    return out


def box3(values: np.ndarray) -> np.ndarray:
    """Fixed-order replicate-boundary 3x3 fp32 mean; no backend reduction."""
    pads = ((1, 1), (1, 1)) + ((0, 0),) * (values.ndim - 2)
    padded = np.pad(values.astype(np.float32), pads, mode="edge")
    total = np.zeros_like(values, dtype=np.float32)
    h, w = values.shape[:2]
    for dy in range(3):
        for dx in range(3):
            total += padded[dy:dy + h, dx:dx + w]
    return total / np.float32(9)


def sample(image: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Border-clamped bilinear sampling with explicit fp32 operation order."""
    h, w = image.shape[:2]
    yy = np.clip(yy, 0, h - 1).astype(np.float32)
    xx = np.clip(xx, 0, w - 1).astype(np.float32)
    y0, x0 = np.floor(yy).astype(np.int32), np.floor(xx).astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = yy - y0.astype(np.float32), xx - x0.astype(np.float32)
    if image.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    a = image[y0, x0].astype(np.float32) * (np.float32(1) - fx)
    a += image[y0, x1].astype(np.float32) * fx
    b = image[y1, x0].astype(np.float32) * (np.float32(1) - fx)
    b += image[y1, x1].astype(np.float32) * fx
    return a * (np.float32(1) - fy) + b * fy


def camera_geometry(tokens: np.ndarray, shape: tuple[int, int]):
    """Map pixel centres with the receiver's align_corners=False convention."""
    h, w = shape
    th, tw = tokens.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    ty = (yy + np.float32(.5)) * np.float32(th / h) - np.float32(.5)
    tx = (xx + np.float32(.5)) * np.float32(tw / w) - np.float32(.5)
    iy = np.clip(np.floor(ty + np.float32(.5)).astype(int), 0, th - 1)
    ix = np.clip(np.floor(tx + np.float32(.5)).astype(int), 0, tw - 1)
    guide = tokens[iy, ix]
    band = edges(tokens)[iy, ix]
    return yy, xx, ty, tx, guide, band


def categorical_guided(rgb: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """One-hot guided linear regression, zero ridge, one-camera-pixel radius."""
    out = np.empty_like(rgb, dtype=np.float32)
    for label in np.unique(guide):
        mask = (guide == label).astype(np.float32)
        count_fraction = box3(mask)
        local_mean = box3(rgb.astype(np.float32) * mask[..., None])
        local_mean /= np.maximum(count_fraction[..., None], np.float32(1 / 9))
        # Windows containing the centre necessarily contain its class. Values
        # in empty-class windows never contribute at a centre of that class.
        prediction = box3(local_mean)
        selected = guide == label
        out[selected] = prediction[selected]
    return out


def signed_distance_local(mask: np.ndarray) -> np.ndarray:
    """Signed Euclidean pixel-centre distance, truncated at three cells.

    Exact within the radius used by the edge-band derivative. No arbitrary
    class ordering: sign is positive inside each independently processed class.
    """
    h, w = mask.shape
    padded = np.pad(mask, 3, mode="edge")
    distance = np.full((h, w), 3, dtype=np.float32)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if not 0 < dy * dy + dx * dx < 9:
                continue
            other = padded[3 + dy:3 + dy + h, 3 + dx:3 + dx + w]
            np.minimum(distance, np.where(other != mask, np.float32(np.sqrt(dy * dy + dx * dx)), np.float32(3)), out=distance)
    return np.where(mask, distance, -distance).astype(np.float32)


def sdf_displacement(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel displacement from signed-distance curvature compensation.

    The central-difference Laplacian shifts a bilinearly reconstructed level
    set; subtract its half-pixel footprint bias (variance 1/12), bounded by a
    half cell. Constants are lattice geometry, never fitted to this video.
    """
    dy = np.zeros(tokens.shape, dtype=np.float32)
    dx = np.zeros(tokens.shape, dtype=np.float32)
    for label in np.unique(tokens):
        mask = tokens == label
        phi = signed_distance_local(mask)
        p = np.pad(phi, 1, mode="edge")
        gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * np.float32(.5)
        gx = (p[1:-1, 2:] - p[1:-1, :-2]) * np.float32(.5)
        lap = p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - np.float32(4) * phi
        amount = lap / (np.float32(24) * np.maximum(gx * gx + gy * gy, np.float32(1)))
        dy[mask] = np.clip(amount * gy, -.5, .5)[mask]
        dx[mask] = np.clip(amount * gx, -.5, .5)[mask]
    return dy, dx


def treat(rgb: np.ndarray, tokens: np.ndarray, mode: str) -> np.ndarray:
    """Treat one camera uint8 RGB master; retain all off-band bytes exactly."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("expected camera HWC uint8 RGB")
    if tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("expected a two-dimensional integer token plane")
    if mode == "baseline":
        return rgb.copy()
    if mode == "composition":
        out = rgb
        for component in ("guided", "ssaa", "sdf"):
            out = treat(out, tokens, component)
        return out
    yy, xx, ty, tx, guide, band = camera_geometry(tokens, rgb.shape[:2])
    if mode == "guided":
        filtered = categorical_guided(rgb, guide)
    elif mode == "ssaa":
        filtered = np.zeros(rgb.shape, dtype=np.float32)
        # Exact centred 2x2 quadrature of the camera-pixel bilinear footprint.
        for oy in (-.25, .25):
            for ox in (-.25, .25):
                filtered += sample(rgb, yy + np.float32(oy), xx + np.float32(ox))
        filtered *= np.float32(.25)
    elif mode == "sdf":
        dy, dx = sdf_displacement(tokens)
        cy = sample(dy, ty, tx) * np.float32(rgb.shape[0] / tokens.shape[0])
        cx = sample(dx, ty, tx) * np.float32(rgb.shape[1] / tokens.shape[1])
        filtered = sample(rgb, yy + cy, xx + cx)
    else:
        raise ValueError(f"unknown boundary treatment: {mode}")
    out = rgb.copy()
    out[band] = np.clip(np.rint(filtered[band]), 0, 255).astype(np.uint8)
    return out
