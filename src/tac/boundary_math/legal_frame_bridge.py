# SPDX-License-Identifier: MIT
"""The legal-frame variational bridge — synthesize a scorer-free frame in the argmax cell.

THE CAMPAIGN CRUX (verdict memo §6, the "burning question"): the lever-B generator outputs
LOGITS / an argmax label map, but the contest archive scores FRAMES. The bridge solves the
direct variational problem: the minimum-byte raw-[0,255] camera-resolution frame ``y1`` whose
SegNet-argmax lands in the cell { F : argmax SegNet(F) == A_target } AND whose 2-frame YUV6
holds the PoseNet tube.

THE KEY CONSTRAINT: the inflate-time synthesizer MUST be SCORER-FREE (no SegNet/PoseNet at
inflate per the contest rule + CLAUDE.md). So the bridge is a deterministic, palette-driven
rasterizer: given a target label map, paint each pixel its class's learned palette color, then
(optionally) apply the boundary-solver's correction field. The palette is learned ONCE at
compress time (with the scorer, the only place the scorer is legal) and stored as a tiny
per-class color table (5 classes x 3 channels = 15 values).

THE FEASIBLE CELL is large (closed-spec §4): 95% of pixels carry > 2 logits of free margin, so a
piecewise-constant palette frame that maps each class region to a SegNet-robust color lands in the
cell over the region interiors; the residual is the boundary band, which the boundary solver
repairs. This module supplies the RASTERIZER + the PALETTE LEARNER; the boundary solver
(`tac.boundary_math.boundary_solver`) supplies the residual correction.

POSE: a palette frame carries no motion, so PoseNet collapses if both frames are palette-painted.
The bridge supports a ``pose_carrier`` strategy: frame0 is SegNet-invisible (SegNet reads frame1
only) so it can be the GT frame0 (carrying the real motion for PoseNet) while frame1 is the
synthesized palette frame. The HONEST cost of this is measured by the exact PoseNet at smoke time
(the verdict's "appearance section" prediction).

COMPUTE-SUBSTRATE LAW: palette learned with the exact CPU-torch SegNet (NEVER MPS); GT decode via
``frame_utils.yuv420_to_rgb`` ONLY. Inflate-time rasterizer is pure numpy (scorer-free, portable).
``[local CPU-torch advisory]`` — non-promotable.

NO-FAKE (class 2): the palette is FIT to the real GT regions and the rasterizer ACTUALLY paints
by label map; a stub that returns a constant frame FAILS the tests (which assert the rasterized
frame's per-class mean matches the palette + the SegNet argmax recovery improves over a flat frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512


class LegalFrameBridgeError(ValueError):
    """Raised on malformed bridge inputs / contract violations."""


# ---------------------------------------------------------------------------
# Palette learning (compress-time, scorer-aware).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassPalette:
    """A per-class RGB color table the rasterizer paints by label map.

    ``colors`` is ``(n_classes, 3)`` float in [0, 255]. ``source`` records how it was
    fit (gt_region_mean / segnet_robust). The palette is the entire learned appearance
    section (tiny: 5*3 = 15 values).
    """

    colors: np.ndarray  # (n_classes, 3)
    source: str

    def __post_init__(self) -> None:
        c = np.asarray(self.colors)
        if c.ndim != 2 or c.shape[1] != 3:
            raise LegalFrameBridgeError(f"palette colors must be (n_classes, 3); got {c.shape}")

    def to_bytes(self) -> bytes:
        """uint8 palette = n_classes*3 bytes (the appearance-section byte cost)."""
        return np.clip(np.round(self.colors), 0, 255).astype(np.uint8).tobytes()

    @property
    def n_classes(self) -> int:
        return int(self.colors.shape[0])


def fit_palette_gt_region_mean(
    gt_frames_hwc: np.ndarray,
    gt_argmax_seg: np.ndarray,
    n_classes: int = 5,
) -> ClassPalette:
    """Fit the per-class palette = the MEAN GT camera-RGB over each class's SegNet region.

    ``gt_frames_hwc`` is ``(P, camera_h, camera_w, 3)`` uint8/float GT frame1s; ``gt_argmax_seg``
    is ``(P, SEG_H, SEG_W)`` the SegNet argmax (the label map at scorer resolution). The argmax is
    upsampled (nearest) to camera resolution to index the GT pixels. This is a real fit (the mean
    color SegNet associates with each class on THIS video), not a guess.
    """

    frames = np.asarray(gt_frames_hwc, dtype=np.float64)
    am = np.asarray(gt_argmax_seg)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise LegalFrameBridgeError(f"gt_frames must be (P,H,W,3); got {frames.shape}")
    if am.ndim != 3:
        raise LegalFrameBridgeError(f"gt_argmax must be (P,SEG_H,SEG_W); got {am.shape}")
    P, H, W, _ = frames.shape
    colors = np.zeros((n_classes, 3), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for pi in range(P):
        am_cam = _nearest_upsample(am[pi], H, W)  # (H, W)
        for c in range(n_classes):
            m = am_cam == c
            n = int(m.sum())
            if n:
                colors[c] += frames[pi][m].sum(axis=0)
                counts[c] += n
    for c in range(n_classes):
        if counts[c]:
            colors[c] /= counts[c]
        else:
            colors[c] = 128.0  # unobserved class -> mid-gray
    return ClassPalette(colors=colors, source="gt_region_mean")


def _nearest_upsample(label_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour upsample a (h, w) label map to (out_h, out_w)."""

    h, w = label_hw.shape
    ys = (np.arange(out_h) * h / out_h).astype(np.int64).clip(0, h - 1)
    xs = (np.arange(out_w) * w / out_w).astype(np.int64).clip(0, w - 1)
    return label_hw[ys][:, xs]


# ---------------------------------------------------------------------------
# The scorer-free rasterizer (inflate-time, pure numpy, portable).
# ---------------------------------------------------------------------------
def rasterize_palette_frame(
    target_argmax_seg: np.ndarray,
    palette: ClassPalette,
    *,
    camera_h: int = CAMERA_H,
    camera_w: int = CAMERA_W,
    correction_hw_seg: np.ndarray | None = None,
) -> np.ndarray:
    """Paint a camera-res frame by upsampling the target label map + palette lookup.

    ``target_argmax_seg`` is the (SEG_H, SEG_W) target label map (the generator's argmax). The
    label map is nearest-upsampled to camera resolution and each pixel painted its class color.
    An optional ``correction_hw_seg`` (the boundary solver's luma field at SEG resolution) is
    bilinearly upsampled and added (all 3 channels) before clamp+round. SCORER-FREE: no SegNet at
    inflate. Returns ``(camera_h, camera_w, 3)`` uint8.
    """

    am = np.asarray(target_argmax_seg)
    if am.ndim != 2:
        raise LegalFrameBridgeError(f"target_argmax must be (H, W); got {am.shape}")
    am_cam = _nearest_upsample(am, camera_h, camera_w)  # (H, W)
    frame = palette.colors[am_cam.clip(0, palette.n_classes - 1)]  # (H, W, 3) float
    if correction_hw_seg is not None:
        corr = _bilinear_upsample(np.asarray(correction_hw_seg, dtype=np.float64), camera_h, camera_w)
        frame = frame + corr[:, :, None]
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def _bilinear_upsample(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear upsample a (h, w) plane to (out_h, out_w) (align_corners=False)."""

    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(plane).float()[None, None]
    up = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return up[0, 0].numpy().astype(np.float64)


def lowres_appearance_carrier(
    gt_frame_hwc: np.ndarray,
    *,
    factor: int,
) -> tuple[np.ndarray, int]:
    """Downsample a GT camera frame by ``factor`` (the pose-carrying appearance section).

    The palette frame carries NO motion (pose collapses); a low-res RGB carrier preserves the
    real luma/chroma structure for PoseNet at a fraction of the bytes. Returns
    ``(lowres_hwc_uint8, coded_bytes)`` — the small stored payload (brotli-q11) the inflate
    upsamples. This is the verdict's predicted "appearance section" realized at the cheapest
    pose-preserving granularity; the reactivation-path probe.
    """

    import brotli
    import torch
    import torch.nn.functional as F

    f = np.asarray(gt_frame_hwc, dtype=np.float32)
    if f.ndim != 3 or f.shape[-1] != 3:
        raise LegalFrameBridgeError(f"gt_frame must be (H, W, 3); got {f.shape}")
    h, w, _ = f.shape
    lh, lw = max(1, h // factor), max(1, w // factor)
    t = torch.from_numpy(f.transpose(2, 0, 1))[None]
    down = F.interpolate(t, size=(lh, lw), mode="area")[0].numpy()  # (3, lh, lw)
    lowres = np.clip(np.round(down.transpose(1, 2, 0)), 0, 255).astype(np.uint8)
    coded = len(brotli.compress(lowres.tobytes(), quality=11))
    return lowres, coded


def upsample_appearance(lowres_hwc: np.ndarray, camera_h: int, camera_w: int) -> np.ndarray:
    """Bilinear-upsample a low-res appearance carrier to camera resolution (inflate-time)."""

    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(lowres_hwc, dtype=np.float32).transpose(2, 0, 1))[None]
    up = F.interpolate(t, size=(camera_h, camera_w), mode="bilinear", align_corners=False)[0]
    return np.clip(np.round(up.numpy().transpose(1, 2, 0)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The byte-accounting of the synthesized appearance section.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BridgeByteAccount:
    """The bridge's appearance-section byte cost (honest, brotli-measured)."""

    palette_bytes: int
    label_map_bytes: int  # the per-pair label map, coded (the generator already carries this)
    correction_bytes: int
    pose_carrier_bytes: int
    total_appearance_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "palette_bytes": self.palette_bytes,
            "label_map_bytes": self.label_map_bytes,
            "correction_bytes": self.correction_bytes,
            "pose_carrier_bytes": self.pose_carrier_bytes,
            "total_appearance_bytes": self.total_appearance_bytes,
        }


def measure_label_map_coded_bytes(argmax_maps: np.ndarray) -> int:
    """Brotli-q11 coded bytes of a stack of (P, H, W) uint8 label maps (the seg carrier).

    This is the rate the GENERATOR already amortizes (the generator blob IS the coded label maps).
    Stored here as the explicit floor a raw label-map carrier would cost (the generator beats it).
    """

    import brotli

    a = np.asarray(argmax_maps).astype(np.uint8)
    return len(brotli.compress(a.tobytes(), quality=11))


__all__ = [
    "BridgeByteAccount",
    "ClassPalette",
    "LegalFrameBridgeError",
    "fit_palette_gt_region_mean",
    "measure_label_map_coded_bytes",
    "rasterize_palette_frame",
]
