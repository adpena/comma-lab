# SPDX-License-Identifier: MIT
"""R-SURVIVAL PHYSICS probe (DAG FEED-iw / GAP-2 — the binding d_seg wall).

Isolates the RENDERING-SURVIVAL term of contest ``d_seg``: a *geometrically
correct* partition (the GT SegNet argmax ``L*``) can FLIP under the contest
render operator R::

    render (sub-camera) -> bicubic UP to camera 874x1164 -> uint8 @ camera
                        -> bilinear DOWN to scorer 384x512 -> argmax

R is the EXACT chain in
``tac.local_acceleration.pr95_hnerv_mlx_training.apply_contest_faithful_roundtrip_nhwc``
(verified against ``upstream/evaluate.py`` + ``modules.py:108-113``).  We
reproduce it with PyTorch ``align_corners=False`` bicubic/bilinear (the torch
authority the witness trainer uses, ``_torch_R_to_camera_uint8``).

WHAT THIS MEASURES (and what it does NOT):
  * MEASURES: how a *boundary representation* of a partition survives the
    double-resample + uint8 knife-edge.  The partition is encoded as a K=5
    channel membership carrier in [0,255]; R is applied per channel; the
    recovered partition is ``argmax_k R(C_k)``.  This is the topology-matched
    model of "the partition's continuous representation surviving R".
  * DOES NOT run the frozen SegNet.  This is a SegNet-FREE isolation of the
    *resampling/quantization* survival physics (the Nyquist/Gibbs/uint8 term),
    NOT the SegNet's RGB-reading nonlinearity.  It is an ADVISORY
    ``[macOS research-signal]`` measurement, never a contest score (pointer
    0.19110 unmoved).  The SegNet-in-loop confirmation is the witness trainer's
    realized-through-R verdict (``cpu_verdict_d_seg``).

DECOMPOSITION (the task's "isolate rendering-survival vs pre-R geometric cost"):
  * pre-R geometric cost = d_seg of ``argmax_k C_k`` (carrier at render res,
    nearest-resized to 384) vs L* -> for hard/sdf reps this is ~0 by
    construction (argmax preserves L*), so the ENTIRE measured d_seg is the
    rendering-survival term.
  * rendering-survival cost = d_seg of ``argmax_k R(C_k)`` vs L*.

NO FAKE: every number is a real argmax-disagreement (canonical
``tac.boundary_math.bitmask_dseg.d_seg_reference``) on the real cached GT
partition; reps that reproduce L* exactly pre-R are asserted to do so.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np

# Contest geometry (CLAUDE.md "Exact scorer architectures" + R chain above).
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
N_CLASSES = 5
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}


def _torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def signed_distance_fields(labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Per-class signed distance ``phi_k`` (H,W,K): +EDT inside, -EDT outside.

    Canonical builder (copied from ``boundary_math.lever_b_levelset_generator``).
    ``argmax_k phi_k == labels`` EXACTLY.  1-Lipschitz (|grad phi|=1).
    """

    from scipy import ndimage

    a = np.asarray(labels)
    h, w = a.shape
    out = np.zeros((h, w, int(n_classes)), np.float32)
    for k in range(int(n_classes)):
        inside = a == k
        if inside.all():
            out[..., k] = float(max(h, w))
            continue
        if not inside.any():
            out[..., k] = -float(max(h, w))
            continue
        d_in = ndimage.distance_transform_edt(inside)
        d_out = ndimage.distance_transform_edt(~inside)
        out[..., k] = (d_in - d_out).astype(np.float32)
    return out


def _resize(x_kchw, size_hw, mode):
    """Per-channel resize of (K,H,W) float carrier with align_corners=False."""

    torch, F = _torch()
    t = torch.from_numpy(np.ascontiguousarray(x_kchw)).float().unsqueeze(0)  # (1,K,H,W)
    kw = {} if mode == "nearest" else {"align_corners": False}
    t = F.interpolate(t, size=size_hw, mode=mode, **kw)
    return t.squeeze(0).numpy()


def apply_R(carrier_khw: np.ndarray, render_hw: tuple[int, int]) -> np.ndarray:
    """Contest-EXACT R per channel: [bicubic UP to camera] -> uint8 -> bilinear DOWN to 384.

    ``carrier_khw``: (K, Hr, Wr) float in [0,255].  If render res == camera res,
    the bicubic-up is skipped (native sub-pixel render at 874).  Returns
    (K, 384, 512) float (the scorer-res carrier the argmax reads).
    """

    torch, F = _torch()
    x = carrier_khw
    if render_hw != (CAMERA_H, CAMERA_W):
        x = _resize(x, (CAMERA_H, CAMERA_W), "bicubic")
    # uint8 knife-edge at CAMERA res (the stored video frame).
    x = np.clip(np.round(x), 0.0, 255.0)
    x = _resize(x, (SEG_H, SEG_W), "bilinear")
    return x


# --------------------------------------------------------------------------- #
# Boundary representations: L* (384) -> carrier (K, Hr, Wr) in [0,255].
# --------------------------------------------------------------------------- #
def carrier_hard(lstar_384: np.ndarray, render_hw: tuple[int, int]) -> np.ndarray:
    """Hard 0/1 class indicator (the palette-paint / hard-argmax analog).

    Render@384: indicator at 384.  Render@874: NN-upsample labels to 874 then
    indicator (STAIRCASE boundary, quantized to the 384 grid).
    """

    rh, rw = render_hw
    if (rh, rw) != (SEG_H, SEG_W):
        lab = _resize(lstar_384[None].astype(np.float32), (rh, rw), "nearest")[0].round().astype(np.int64)
    else:
        lab = lstar_384
    K = N_CLASSES
    C = np.zeros((K, rh, rw), np.float32)
    for k in range(K):
        C[k] = 255.0 * (lab == k)
    return C


def carrier_sdf(lstar_384: np.ndarray, render_hw: tuple[int, int], slope: float) -> np.ndarray:
    """1-Lipschitz SDF ramp carrier: C_k = clamp(128 + slope*phi_k, 0, 255).

    Render@384: SDF computed at 384, ramped.  Render@874: SDF computed at 384
    then BICUBIC-upsampled to 874 (sub-pixel boundary placement: the zero-crossing
    of the smooth SDF lands between 384-grid samples), then ramped.

    ``slope`` (per-pixel) sets the boundary ramp half-width ~= 127/slope px.
    """

    rh, rw = render_hw
    phi = signed_distance_fields(lstar_384, N_CLASSES)  # (384,512,K)
    phi = np.transpose(phi, (2, 0, 1))  # (K,384,512)
    if (rh, rw) != (SEG_H, SEG_W):
        phi = _resize(phi, (rh, rw), "bicubic")  # sub-pixel zero-crossing at 874
    C = np.clip(128.0 + slope * phi, 0.0, 255.0).astype(np.float32)
    return C


def carrier_palette(lstar_384: np.ndarray, render_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """SINGLE shared channel: paint class k at gray level ``levels[k]`` (the naive
    palette-paint the witness would do in ONE image).  Returns (carrier (1,Hr,Wr),
    levels).  Recovery = nearest level.  This is the SHARED-channel coupling that
    box-averages a thin minority class into its majority neighbor -> the death.
    """

    rh, rw = render_hw
    levels = np.linspace(0.0, 255.0, N_CLASSES).astype(np.float32)  # evenly spaced gray levels
    if (rh, rw) != (SEG_H, SEG_W):
        lab = _resize(lstar_384[None].astype(np.float32), (rh, rw), "nearest")[0].round().astype(np.int64)
    else:
        lab = lstar_384
    img = levels[lab][None]  # (1,Hr,Wr)
    return img.astype(np.float32), levels.reshape(N_CLASSES, 1)  # protos (K,1)


def carrier_rgb3(lstar_384: np.ndarray, render_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """3-channel RGB, 5 maximally-separated prototype colors (the REAL witness
    constraint: 3 channels, 5 classes).  Recovery = nearest prototype color.
    """

    rh, rw = render_hw
    # 5 maximally-separated colors on the RGB cube (greedy max-min over corners/centroid).
    proto = np.array([
        [0, 0, 0],        # 0 Road
        [255, 255, 255],  # 1 Lane (bright -> still averaged when thin)
        [255, 0, 0],      # 2 Undrivable
        [0, 255, 0],      # 3 Movable
        [0, 0, 255],      # 4 MyCar
    ], np.float32)
    if (rh, rw) != (SEG_H, SEG_W):
        lab = _resize(lstar_384[None].astype(np.float32), (rh, rw), "nearest")[0].round().astype(np.int64)
    else:
        lab = lstar_384
    img = np.transpose(proto[lab], (2, 0, 1))  # (3,Hr,Wr)
    return img.astype(np.float32), proto


def _classify_nearest(field_chw: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Nearest-prototype classify: field (C,H,W), protos (K,C) -> argmin dist (H,W)."""

    C, H, W = field_chw.shape
    x = field_chw.reshape(C, -1).T  # (HW, C)
    d = ((x[:, None, :] - protos[None, :, :]) ** 2).sum(-1)  # (HW, K)
    return d.argmin(1).reshape(H, W).astype(np.int64)


@dataclass
class RepResult:
    rep: str
    render_hw: tuple[int, int]
    slope: float | None
    pre_r_dseg: float
    survival_dseg: float
    per_class_flip_rate: dict[int, float] = field(default_factory=dict)
    per_class_present_frac: dict[int, float] = field(default_factory=dict)


def _per_class_flip(recovered: np.ndarray, lstar: np.ndarray) -> dict[int, float]:
    """Per-class flip rate = (# pixels whose GT class is k AND recovered != k) / (# GT class-k px)."""

    out = {}
    for k in range(N_CLASSES):
        m = lstar == k
        n = int(m.sum())
        if n == 0:
            out[k] = float("nan")
            continue
        out[k] = float(np.count_nonzero(recovered[m] != k)) / n
    return out


def measure(
    lstars: np.ndarray,
    reps: list[tuple[str, tuple[int, int], float | None]],
) -> dict:
    """Run every (rep, render_hw, slope) config over all frames in ``lstars`` (N,384,512)."""

    from tac.boundary_math.bitmask_dseg import d_seg_reference

    N = lstars.shape[0]
    agg: dict[str, dict] = {}
    for (rep, render_hw, slope) in reps:
        key = f"{rep}@{render_hw[0]}" + (f"|s{slope:g}" if slope is not None else "")
        pre_list, surv_list = [], []
        pc_flip = {k: [] for k in range(N_CLASSES)}
        pc_present = {k: 0 for k in range(N_CLASSES)}
        for i in range(N):
            lstar = lstars[i]
            for k in range(N_CLASSES):
                if (lstar == k).any():
                    pc_present[k] += 1
            protos = None
            if rep == "hard":
                C = carrier_hard(lstar, render_hw)
            elif rep == "sdf":
                C = carrier_sdf(lstar, render_hw, float(slope))
            elif rep == "palette":
                C, protos = carrier_palette(lstar, render_hw)
            elif rep == "rgb3":
                C, protos = carrier_rgb3(lstar, render_hw)
            else:
                raise ValueError(rep)

            def _recover(field):
                if protos is None:  # multi-channel membership -> argmax over class channels
                    return np.argmax(field, axis=0).astype(np.int64)
                # shared-channel / RGB -> nearest prototype
                return _classify_nearest(field, protos)

            # pre-R geometric: recover from carrier nearest-resized to 384.
            pre = _resize(C, (SEG_H, SEG_W), "nearest") if render_hw != (SEG_H, SEG_W) else C
            pre_list.append(d_seg_reference(_recover(pre), lstar))
            # rendering-survival: recover from R(C).
            R = apply_R(C, render_hw)
            rec = _recover(R)
            surv_list.append(d_seg_reference(rec, lstar))
            f = _per_class_flip(rec, lstar)
            for k in range(N_CLASSES):
                if not np.isnan(f[k]):
                    pc_flip[k].append(f[k])
        agg[key] = RepResult(
            rep=rep,
            render_hw=render_hw,
            slope=slope,
            pre_r_dseg=float(np.mean(pre_list)),
            survival_dseg=float(np.mean(surv_list)),
            per_class_flip_rate={k: (float(np.mean(v)) if v else float("nan")) for k, v in pc_flip.items()},
            per_class_present_frac={k: pc_present[k] / N for k in range(N_CLASSES)},
        )
    return agg


def lane_geometry_stats(lstars: np.ndarray, lane_cls: int = 1) -> dict:
    """Lane (class 1) width distribution at 384 (grounds the Nyquist argument)."""

    from scipy import ndimage

    widths = []
    areas = []
    for i in range(lstars.shape[0]):
        m = lstars[i] == lane_cls
        if not m.any():
            continue
        areas.append(float(m.mean()))
        # local width ~ 2 * EDT(inside) at the medial axis; use per-pixel 2*EDT over lane px.
        edt = ndimage.distance_transform_edt(m)
        widths.extend((2.0 * edt[m]).tolist())
    w = np.asarray(widths) if widths else np.zeros(1)
    return {
        "lane_area_frac_mean": float(np.mean(areas)) if areas else 0.0,
        "lane_width_px_median": float(np.median(w)),
        "lane_width_px_mean": float(np.mean(w)),
        "lane_width_px_p90": float(np.percentile(w, 90)),
        "lane_width_le_2px_frac": float(np.mean(w <= 2.0)),
        "lane_width_le_1px_frac": float(np.mean(w <= 1.0)),
    }


def estimate_R_kernel_sigma(render_h: int) -> dict:
    """Effective Gaussian sigma of the R chain at a given render res, via edge-spread.

    Push a vertical step edge (0|255) at render res through R; the 10-90 rise of the
    edge-spread-function at 384 gives the effective kernel sigma (10-90 ~= 2.563*sigma
    for a Gaussian).  Measures R's OWN low-pass (it is small/benign: ~0.4-1.2px).
    """

    rw = int(round(render_h * SEG_W / SEG_H))
    f = np.zeros((1, render_h, rw), np.float32)
    f[:, :, rw // 2:] = 255.0
    out = apply_R(f, (render_h, rw))[0][SEG_H // 2]  # mid-row edge-spread at 384

    def _cross(val):
        idx = int(np.argmax(out >= val))
        if idx == 0:
            return 0.0
        y0, y1 = out[idx - 1], out[idx]
        return idx - 1 + (val - y0) / (y1 - y0 + 1e-9)

    rise = _cross(229.5) - _cross(25.5)
    return {"render_h": render_h, "rise_10_90_px": float(rise), "eff_sigma_px": float(rise / 2.563)}


def heat_vs_interp_lane_survival(lstars: np.ndarray, sigmas: list[float]) -> dict:
    """DECISIVE test: is the SDF's R-survival a HEAT-kernel (diffusion) effect or an
    INTERPOLATION (subsample->reconstruct) effect?

    HEAT path: Gaussian-blur the carrier (exact heat kernel, t=sigma^2/2) then argmax.
    INTERP path: the actual R probe (bicubic subsample to render res -> R reconstruct).

    Finding: under HEAT the SDF does NOT beat hard for the thin lane (blur averages the
    thin class's small-magnitude phi into neighbors); under INTERP the SDF wins ~8x.
    -> R is interpolation-dominant; the SDF wins via interpolation-exactness on the
    1-Lipschitz linear ramp, NOT heat-kernel level-set stability.
    """

    from scipy import ndimage

    from tac.boundary_math.bitmask_dseg import d_seg_reference

    N = lstars.shape[0]
    out = {"heat": {}, "interp_note": "see capacity grid (sdf vs hard survival_dseg)"}
    for sg in sigmas:
        hl, sl = [], []
        for i in range(N):
            lab = lstars[i]
            hard = np.stack([(lab == k).astype(np.float32) for k in range(N_CLASSES)])
            hard_b = np.stack([ndimage.gaussian_filter(hard[k], sg) for k in range(N_CLASSES)])
            phi = np.transpose(signed_distance_fields(lab), (2, 0, 1))
            phi_b = np.stack([ndimage.gaussian_filter(phi[k], sg) for k in range(N_CLASSES)])
            m = lab == 1
            if m.sum():
                hl.append(float((hard_b.argmax(0)[m] != 1).mean()))
                sl.append(float((phi_b.argmax(0)[m] != 1).mean()))
        out["heat"][f"sigma{sg:g}"] = {
            "hard_lane_flip": float(np.mean(hl)) if hl else float("nan"),
            "sdf_lane_flip": float(np.mean(sl)) if sl else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #
# MSDF (multi-channel signed distance field; Chlumsky 2018) — LANE corner carrier.
#
# A SINGLE SDF rounds sharp corners (dash-ends): the iso-line near a convex corner
# follows the distance-to-the-corner-VERTEX cone, not the sharp edge-line
# intersection.  MSDF stores 3 channels, each the signed PSEUDO-distance (distance to
# the edge's infinite SUPPORTING LINE, not the clamped segment) of a 2-colored SUBSET
# of the contour edges, arranged so the two edges meeting at a sharp corner land in
# DIFFERENT channels; median(R,G,B) then selects the sharp intersection instead of the
# rounded cone.  This is a faithful port of msdfgen's ``edgeColoringSimple`` +
# ``LinearSegment::signedDistance`` + ``distanceToPseudoDistance`` (Chlumsky, CGF
# 37(1) 2018; OSS github.com/Chlumsky/msdfgen).  Validated against a synthetic sharp
# corner (``validate_msdf_synthetic``) before any lane number is trusted (NO FAKE).
#
# rule-118: the MSDF GENERATION (edge coloring + per-channel pseudo-distance + median)
# is a FREE deterministic geometric algorithm legal to expand inside inflate.py; the
# stored lane CONTOUR is COUNTED.  The SAME contour is what single-SDF stores, so the
# 3-channel decomposition adds 0 COUNTED bytes IF the witness stores the vector contour
# (channels generated at decode); it costs ~3x the lane channel only IF the witness
# stores rendered fields (see memo byte-cost note).
# --------------------------------------------------------------------------- #

_RED, _GREEN, _BLUE, _WHITE, _BLACK = 1, 2, 4, 7, 0  # msdfgen EdgeColor bit flags


def _switch_color(color: int, seed: int, banned: int = _BLACK) -> tuple[int, int]:
    """Port of msdfgen ``switchColor`` (coloring/EdgeColoringSimple.cpp)."""

    combined = color & banned
    if combined in (_RED, _GREEN, _BLUE):
        return combined ^ _WHITE, seed
    if color in (_BLACK, _WHITE):
        start = (6, 5, 3)  # CYAN(G,B), MAGENTA(R,B), YELLOW(R,G)
        return start[seed % 3], seed // 3
    shifted = color << (1 + (seed & 1))
    return (shifted | (shifted >> 3)) & _WHITE, seed >> 1


def _poly_edge_dirs(verts: np.ndarray) -> np.ndarray:
    """Unit direction of edge i = verts[i]->verts[(i+1)%V]. ``verts`` (V,2)."""

    V = len(verts)
    d = verts[(np.arange(V) + 1) % V] - verts
    n = np.hypot(d[:, 0], d[:, 1]) + 1e-12
    return d / n[:, None]


def _polygon_corner_indices(verts: np.ndarray, angle_threshold_deg: float) -> list[int]:
    """Edge indices i whose preceding junction (edge i-1 -> edge i) is a sharp corner.

    msdfgen ``isCorner``: dot(prevDir, dir) <= 0  OR  |cross(prevDir, dir)| > sin(theta).
    """

    V = len(verts)
    if V < 2:
        return []
    dirs = _poly_edge_dirs(verts)
    cross_thr = float(np.sin(np.deg2rad(angle_threshold_deg)))
    corners = []
    for i in range(V):
        a = dirs[(i - 1) % V]
        b = dirs[i]
        dot = float(a @ b)
        crs = float(a[0] * b[1] - a[1] * b[0])
        if dot <= 0.0 or abs(crs) > cross_thr:
            corners.append(i)
    return corners


def _edge_coloring_simple(verts: np.ndarray, angle_threshold_deg: float = 3.0, seed: int = 0) -> list[int]:
    """2-color a closed polygon's edges so sharp-corner edges differ in one channel.

    Faithful port of msdfgen ``edgeColoringSimple`` (multi-corner branch; teardrop and
    smooth special cases handled).  Returns per-edge color (len V; edge i =
    verts[i]->verts[(i+1)%V]).
    """

    V = len(verts)
    if V < 2:
        return [_WHITE] * V
    corners = _polygon_corner_indices(verts, angle_threshold_deg)
    colors = [_WHITE] * V
    if not corners:
        return colors  # fully smooth contour -> no corners to sharpen (single color set)
    if len(corners) == 1:
        c = corners[0]
        cols3 = []
        col = _WHITE
        for _ in range(3):
            col, seed = _switch_color(col, seed)
            cols3.append(col)
        for j in range(V):
            colors[(c + j) % V] = cols3[min(2, (3 * j) // max(1, V))]
        return colors
    spline = 0
    start = corners[0]
    color, seed = _switch_color(_WHITE, seed)
    initial = color
    m = len(corners)
    for i in range(V):
        idx = (start + i) % V
        if spline + 1 < m and corners[spline + 1] == idx:
            spline += 1
            banned = initial if spline == m - 1 else _BLACK
            color, seed = _switch_color(color, seed, banned)
        colors[idx] = color
    return colors


def _lane_contours(mask_bool: np.ndarray, approx_eps: float = 1.0) -> list[np.ndarray]:
    """Lane connected-component polygons via cv2 (vertices in (x,y)=(col,row) float)."""

    import cv2

    m = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cnts:
        if len(c) < 3:
            continue
        ap = cv2.approxPolyDP(c, float(approx_eps), True).reshape(-1, 2).astype(np.float64)
        if len(ap) >= 2:
            polys.append(ap)
    return polys


def _segment_signed_distances(px: np.ndarray, py: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Vectorized msdfgen ``LinearSegment::signedDistance`` + ``distanceToPseudoDistance``.

    Returns ``(true_signed (P,), value (P,))``:
      * ``true_signed`` = the clamped signed distance (perpendicular within the segment,
        else signed endpoint distance).  Selection of the NEAREST edge is by |true_signed|.
      * ``value`` = the pseudo-distance-aware signed distance written to the texture for the
        winner: identical to ``true_signed`` EXCEPT where the perpendicular foot falls
        BEYOND the segment (near a corner) AND the line-extension distance is smaller in
        magnitude, in which case it is the signed distance to the INFINITE supporting line
        (this is the corner-extension that, via median-of-3, recovers sharp corners).
    Single sign convention: ``cross(aq, ab)`` (interior + for one winding); a global flip
    aligns inside=+ (polygon-winding ambiguity resolved empirically by the caller).
    """

    abx = B[0] - A[0]
    aby = B[1] - A[1]
    L2 = abx * abx + aby * aby + 1e-12
    L = np.sqrt(L2)
    aqx = px - A[0]
    aqy = py - A[1]
    param = (aqx * abx + aqy * aby) / L2
    c = aqx * aby - aqy * abx          # cross(aq, ab)
    ortho_line = c / L                  # signed orthogonal distance to the infinite line
    sgn = np.sign(c)
    sgn[sgn == 0] = 1.0
    ex = np.where(param > 0.5, B[0], A[0])
    ey = np.where(param > 0.5, B[1], A[1])
    endp = np.hypot(px - ex, py - ey)
    inside_seg = (param > 0.0) & (param < 1.0)
    use_ortho = inside_seg & (np.abs(ortho_line) < endp)
    true_signed = np.where(use_ortho, ortho_line, sgn * endp)
    # msdfgen SignedDistance.dot tie-breaker: 0 within segment, else |dot(ab_norm, eq_norm)|
    # (how ALIGNED the point->endpoint direction is with the edge; ties at a shared corner
    # vertex are resolved toward the more-perpendicular edge -> the geometrically-correct
    # sharp extension, killing the convex-corner false-inside cone).
    dot = np.where(
        use_ortho,
        0.0,
        np.abs(abx * (ex - px) + aby * (ey - py)) / (L * endp + 1e-12),
    )
    value = true_signed.copy()
    # distanceToPseudoDistance near point0 (foot before A): replace with infinite-line dist.
    ts0 = (aqx * abx + aqy * aby) / L
    repl0 = (param < 0.0) & (ts0 < 0.0) & (np.abs(ortho_line) <= np.abs(true_signed))
    value = np.where(repl0, ortho_line, value)
    # near point1 (foot beyond B).
    aq1x = px - B[0]
    aq1y = py - B[1]
    c1 = aq1x * aby - aq1y * abx       # cross(aq1, ab)
    pseudo1 = c1 / L
    ts1 = (aq1x * abx + aq1y * aby) / L
    repl1 = (param > 1.0) & (ts1 > 0.0) & (np.abs(pseudo1) <= np.abs(true_signed))
    value = np.where(repl1, pseudo1, value)
    return true_signed, value, dot


def _channel_pseudo_field(px: np.ndarray, py: np.ndarray, edges_AB: list, chunk: int = 4096) -> np.ndarray:
    """Per-point pseudo-aware value of the nearest channel edge (msdfgen SignedDistance order).

    Selection minimizes ``(|true_signed|, dot)`` lexicographically: smaller |distance| wins;
    on a tie (e.g. two edges sharing a corner vertex) the smaller ``dot`` (more perpendicular)
    wins -> the geometrically-correct sharp edge.
    """

    P = px.shape[0]
    out = np.zeros(P, np.float64)
    eps = 1e-6
    for s in range(0, P, chunk):
        e = s + chunk
        bx, by = px[s:e], py[s:e]
        best_abs = np.full(bx.shape[0], np.inf)
        best_dot = np.full(bx.shape[0], np.inf)
        best_val = np.zeros(bx.shape[0])
        for (A, B) in edges_AB:
            ts, val, dot = _segment_signed_distances(bx, by, A, B)
            a = np.abs(ts)
            better = (a < best_abs - eps) | ((a <= best_abs + eps) & (dot < best_dot))
            best_abs = np.where(better, a, best_abs)
            best_dot = np.where(better, dot, best_dot)
            best_val = np.where(better, val, best_val)
        out[s:e] = best_val
    return out


def msdf_lane_channels(lab_hw: np.ndarray, band_px: int, approx_eps: float = 1.0,
                       angle_threshold_deg: float = 3.0, lane_cls: int = 1):
    """3-channel lane MSDF pseudo-distance fields (3,H,W), deep-negative outside a band.

    ``lab_hw`` is an integer label map at any resolution; the lane mask is ``==lane_cls``.
    The 3 channels are sign-aligned so inside-lane is + (global flip resolved empirically).
    Returns ``(chans (3,H,W) float, stats dict)``.
    """

    from scipy import ndimage

    mask = lab_hw == lane_cls
    H, W = mask.shape
    chans = np.full((3, H, W), -1e4, np.float32)
    stats = {"n_polys": 0, "n_edges": 0, "n_corners": 0, "band_px": 0}
    if not mask.any():
        return chans, stats
    polys = _lane_contours(mask, approx_eps)
    edges_by_ch = {1: [], 2: [], 4: []}
    for verts in polys:
        cols = _edge_coloring_simple(verts, angle_threshold_deg)
        stats["n_polys"] += 1
        stats["n_corners"] += len(_polygon_corner_indices(verts, angle_threshold_deg))
        Vn = len(verts)
        for i in range(Vn):
            A = verts[i]
            B = verts[(i + 1) % Vn]
            col = cols[i]
            stats["n_edges"] += 1
            for ch in (1, 2, 4):
                if col & ch:
                    edges_by_ch[ch].append((A, B))
    band = ndimage.binary_dilation(mask, iterations=int(band_px))
    ys, xs = np.nonzero(band)
    stats["band_px"] = int(ys.shape[0])
    px = xs.astype(np.float64)
    py = ys.astype(np.float64)
    order = [1, 2, 4]
    vals = []
    for ch in order:
        eAB = edges_by_ch[ch]
        if eAB:
            vals.append(_channel_pseudo_field(px, py, eAB))
        else:
            vals.append(None)
    # Fill any empty channel with the elementwise mean of non-empty channels
    # (graceful degradation -> median ~ single-SDF; biases AGAINST an MSDF win).
    present = [v for v in vals if v is not None]
    if not present:
        return chans, stats
    fallback = np.mean(np.stack(present), axis=0)
    vals = [v if v is not None else fallback for v in vals]
    band_stack = np.stack(vals)  # (3, P)
    med = np.median(band_stack, axis=0)
    inside = mask[ys, xs]
    if float(np.mean((med > 0) == inside)) < 0.5:
        band_stack = -band_stack
    for j in range(3):
        chans[j, ys, xs] = band_stack[j].astype(np.float32)
    return chans, stats


def _argmax_through_R_sdf(lstar_384: np.ndarray, render_hw, slope: float) -> np.ndarray:
    """Single-SDF baseline: argmax_k R(carrier_sdf_k). Returns recovered (384,512)."""

    C = carrier_sdf(lstar_384, render_hw, slope)
    R = apply_R(C, render_hw)
    return np.argmax(R, axis=0).astype(np.int64)


def _argmax_through_R_msdf(lstar_384: np.ndarray, render_hw, slope: float, band_px: int,
                           approx_eps: float, angle_threshold_deg: float) -> np.ndarray:
    """MSDF lane carrier: non-lane single-SDF + 3 lane MSDF channels, median-of-3 AFTER R.

    The 3 lane channels are built at 384 (the geometry res, matching carrier_sdf), bicubic
    -resized to render res, ramped, pushed through R, then median-combined at decode -> the
    literal 3-channel decode order.  argmax over [road, lane_msdf, undriv, movable, mycar].
    """

    rh, rw = render_hw
    sdf5 = carrier_sdf(lstar_384, render_hw, slope)  # (5,rh,rw)
    chans384, _ = msdf_lane_channels(lstar_384, band_px, approx_eps, angle_threshold_deg)
    if (rh, rw) != (SEG_H, SEG_W):
        chans_r = _resize(chans384, (rh, rw), "bicubic")
    else:
        chans_r = chans384
    lane3 = np.clip(128.0 + slope * chans_r, 0.0, 255.0).astype(np.float32)  # (3,rh,rw)
    carrier7 = np.concatenate([sdf5[0:1], lane3, sdf5[2:5]], axis=0)  # (7,rh,rw)
    R7 = apply_R(carrier7, render_hw)  # (7,384,512)
    lane_field = np.median(R7[1:4], axis=0)
    fields5 = np.stack([R7[0], lane_field, R7[4], R7[5], R7[6]], axis=0)
    return np.argmax(fields5, axis=0).astype(np.int64)


def measure_msdf(lstars: np.ndarray, render_hs: list[int], slope: float, band_px: int,
                 approx_eps: float = 1.0, angle_threshold_deg: float = 3.0) -> dict:
    """A/B/C lane-survival: hard vs single-SDF vs MSDF through R, at each render res."""

    from tac.boundary_math.bitmask_dseg import d_seg_reference

    def _hw(h):
        return (CAMERA_H, CAMERA_W) if h == CAMERA_H else (h, int(round(h * SEG_W / SEG_H)))

    out = {}
    N = lstars.shape[0]
    for h in render_hs:
        hw = _hw(h)
        rows = {"hard": {"lane": [], "tot": []}, "sdf": {"lane": [], "tot": []},
                "msdf": {"lane": [], "tot": []}}
        for i in range(N):
            lstar = lstars[i]
            lane_gt = lstar == 1
            n_lane = int(lane_gt.sum())
            # hard
            rh = np.argmax(apply_R(carrier_hard(lstar, hw), hw), axis=0).astype(np.int64)
            # single-sdf
            rs = _argmax_through_R_sdf(lstar, hw, slope)
            # msdf
            rm = _argmax_through_R_msdf(lstar, hw, slope, band_px, approx_eps, angle_threshold_deg)
            for tag, rec in (("hard", rh), ("sdf", rs), ("msdf", rm)):
                rows[tag]["tot"].append(d_seg_reference(rec, lstar))
                if n_lane:
                    rows[tag]["lane"].append(float(np.count_nonzero(rec[lane_gt] != 1)) / n_lane)
        out[h] = {tag: {"lane_flip": float(np.mean(v["lane"])) if v["lane"] else float("nan"),
                        "total_dseg": float(np.mean(v["tot"]))} for tag, v in rows.items()}
    return out


def decompose_single_sdf_residual(lstars: np.ndarray, render_h: int, slope: float,
                                  corner_radius_px: float = 3.0, angle_threshold_deg: float = 30.0,
                                  approx_eps: float = 1.0) -> dict:
    """Classify single-SDF lane flips as corner-region (MSDF-addressable) vs thin (Nyquist).

    corner-region: within ``corner_radius_px`` (at 384) of a sharp lane-contour corner.
    thin: lane local width * (render_h/384) <= 2px (sub-Nyquist at the witness render res).
    """

    from scipy import ndimage

    hw = (CAMERA_H, CAMERA_W) if render_h == CAMERA_H else (render_h, int(round(render_h * SEG_W / SEG_H)))
    agg = {"n_flip": 0, "corner": 0, "thin": 0, "both": 0, "neither": 0, "n_lane_corners": 0}
    scale = render_h / float(SEG_H)
    for i in range(lstars.shape[0]):
        lstar = lstars[i]
        lane_gt = lstar == 1
        if not lane_gt.any():
            continue
        rec = _argmax_through_R_sdf(lstar, hw, slope)
        flip = lane_gt & (rec != 1)
        if not flip.any():
            continue
        # corner map: distance to nearest sharp lane corner vertex
        polys = _lane_contours(lane_gt, approx_eps)
        corner_xy = []
        for verts in polys:
            for idx in _polygon_corner_indices(verts, angle_threshold_deg):
                corner_xy.append(verts[idx])
        agg["n_lane_corners"] += len(corner_xy)
        near_corner = np.zeros(lane_gt.shape, bool)
        if corner_xy:
            cmask = np.zeros(lane_gt.shape, bool)
            for (cx, cy) in corner_xy:
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= iy < lane_gt.shape[0] and 0 <= ix < lane_gt.shape[1]:
                    cmask[iy, ix] = True
            dist = ndimage.distance_transform_edt(~cmask)
            near_corner = dist <= corner_radius_px
        width = 2.0 * ndimage.distance_transform_edt(lane_gt)
        thin = (width * scale) <= 2.0
        fc = near_corner[flip]
        ft = thin[flip]
        agg["n_flip"] += int(flip.sum())
        agg["corner"] += int(np.count_nonzero(fc & ~ft))
        agg["thin"] += int(np.count_nonzero(ft & ~fc))
        agg["both"] += int(np.count_nonzero(fc & ft))
        agg["neither"] += int(np.count_nonzero(~fc & ~ft))
    nf = max(1, agg["n_flip"])
    agg["frac_corner_only"] = agg["corner"] / nf
    agg["frac_thin_only"] = agg["thin"] / nf
    agg["frac_both"] = agg["both"] / nf
    agg["frac_neither"] = agg["neither"] / nf
    agg["frac_corner_any"] = (agg["corner"] + agg["both"]) / nf
    agg["frac_thin_any"] = (agg["thin"] + agg["both"]) / nf
    return agg


def validate_msdf_synthetic(canvas: int = 256, down: int = 24, corner_band: int = 4) -> dict:
    """NO-FAKE proof the MSDF impl actually recovers a SHARP corner single-SDF rounds.

    Draws sharp convex shapes (rotated square 90-deg corners + thin triangle ~25-deg),
    coarsens geometry to ``down`` px then magnifies back to ``canvas`` (the corner-rounding
    regime), thresholds, and compares corner-region error of single-SDF vs MSDF.  PASS iff
    MSDF corner-region error < single-SDF for every shape.
    """

    import cv2
    from scipy import ndimage

    def _shape(name):
        img = np.zeros((canvas, canvas), np.uint8)
        c = canvas // 2
        if name == "square":
            r = canvas // 4
            theta = np.deg2rad(30.0)
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            pts = (np.array([[-r, -r], [r, -r], [r, r], [-r, r]], np.float64) @ R.T + c).astype(np.int32)
        else:  # sharp triangle
            r = canvas // 3
            pts = np.array([[c, c - r], [c - r // 3, c + r], [c + r // 3, c + r]], np.int32)
        cv2.fillPoly(img, [pts], 1)
        return img.astype(bool), pts.astype(np.float64)

    res = {}
    ok_all = True
    for name in ("square", "triangle"):
        mask, pts = _shape(name)
        lab = mask.astype(np.int64)  # 0 outside, 1 inside (lane_cls=1)
        # single-SDF (1-Lipschitz) signed field, + inside
        phi = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
        phi = phi.astype(np.float32)[None]  # (1,H,W)
        # MSDF channels (lane_cls=1) on full image (band covers all)
        chans, st = msdf_lane_channels(lab, band_px=max(canvas, 8), approx_eps=0.8, angle_threshold_deg=3.0)

        def _coarse_then_magnify(field_khw):
            t = _resize(field_khw, (down, down), "bicubic")
            t = _resize(t, (canvas, canvas), "bicubic")
            return t

        recon_single = _coarse_then_magnify(phi)[0] > 0.0
        chans_cm = _coarse_then_magnify(chans)
        recon_msdf = np.median(chans_cm, axis=0) > 0.0
        # corner region = within corner_band px of any true polygon vertex
        cm = np.zeros(mask.shape, bool)
        for (cx, cy) in pts:
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= iy < canvas and 0 <= ix < canvas:
                cm[iy, ix] = True
        cregion = ndimage.distance_transform_edt(~cm) <= corner_band
        err_single = float(np.mean(recon_single[cregion] != mask[cregion]))
        err_msdf = float(np.mean(recon_msdf[cregion] != mask[cregion]))
        iou_single = float((recon_single & mask).sum() / max(1, (recon_single | mask).sum()))
        iou_msdf = float((recon_msdf & mask).sum() / max(1, (recon_msdf | mask).sum()))
        passed = err_msdf < err_single
        ok_all = ok_all and passed
        res[name] = {"corner_err_single": err_single, "corner_err_msdf": err_msdf,
                     "iou_single": iou_single, "iou_msdf": iou_msdf,
                     "n_corners_detected": st["n_corners"], "n_edges": st["n_edges"],
                     "PASS": passed}
    res["ALL_PASS"] = bool(ok_all)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="R-survival physics probe (advisory [macOS research-signal]).")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--n", type=int, default=96, help="# frames (subset for speed).")
    ap.add_argument("--slopes", default="255,127,64,32", help="SDF ramp slopes (per px).")
    ap.add_argument("--render-res", default="48,96,192,384,874",
                    help="Witness native render heights (capacity sweep). Width = round(h*512/384).")
    ap.add_argument("--stress-res", type=int, default=192,
                    help="Render height for the SDF ramp-slope sweep (where survival bites).")
    ap.add_argument("--out", default=None, help="JSON output path.")
    ap.add_argument("--scale-space", action="store_true",
                    help="Also run scale-space diagnostics (R kernel sigma + heat-vs-interp).")
    ap.add_argument("--msdf", action="store_true",
                    help="Run the MSDF lane-carrier A/B/C (hard vs single-SDF vs MSDF) + "
                         "corner-vs-thin residual decomposition + synthetic-corner validation.")
    ap.add_argument("--msdf-render-res", default="192,320",
                    help="Render heights for the MSDF A/B/C (F1 recipe: MSDF at render>=320).")
    ap.add_argument("--msdf-slope", type=float, default=48.0,
                    help="SDF/MSDF ramp slope (per px). 48 -> half-width ~2.6px (F1 best @192).")
    ap.add_argument("--band-px", type=int, default=12,
                    help="Lane MSDF band dilation (px); must exceed the ramp half-width.")
    ap.add_argument("--approx-eps", type=float, default=1.0, help="cv2.approxPolyDP epsilon (px).")
    ap.add_argument("--corner-radius-px", type=float, default=3.0,
                    help="Corner-region radius for the residual decomposition (at 384).")
    args = ap.parse_args(argv)

    d = np.load(args.gt_cache)
    lstars = d["lstars"][: args.n]
    print(f"[r_survival] loaded {lstars.shape[0]} frames; classes present: "
          f"{sorted(np.unique(lstars).tolist())}", flush=True)

    geo = lane_geometry_stats(lstars)
    print(f"[r_survival] lane geometry: {json.dumps(geo)}", flush=True)

    slopes = [float(s) for s in args.slopes.split(",")]
    render_hs = [int(h) for h in args.render_res.split(",")]
    def _hw(h):
        if h == CAMERA_H:
            return (CAMERA_H, CAMERA_W)
        return (h, int(round(h * SEG_W / SEG_H)))
    # Capacity x rep grid: at each render res, compare palette / hard / sdf(best slope).
    best_slope = slopes[len(slopes) // 2] if slopes else 64.0
    reps: list[tuple[str, tuple[int, int], float | None]] = []
    for h in render_hs:
        hw = _hw(h)
        reps.append(("palette", hw, None))
        reps.append(("hard", hw, None))
        reps.append(("sdf", hw, best_slope))
    # SDF ramp-slope sweep at the STRESS render res (where survival bites): which
    # ramp half-width best preserves the thin lane through R?
    stress_hw = _hw(int(args.stress_res))
    for s in slopes:
        reps.append(("sdf", stress_hw, s))

    t0 = time.time()
    agg = measure(lstars, reps)
    secs = time.time() - t0

    # Pretty table.
    print(f"\n[r_survival] === SURVIVAL TABLE (n={lstars.shape[0]}, {secs:.1f}s) ===", flush=True)
    hdr = f"{'config':<18}{'preR':>9}{'survival':>10}  " + "".join(f"{CLASS_NAMES[k][:4]:>7}" for k in range(N_CLASSES))
    print(hdr, flush=True)
    rows = {}
    for key, r in agg.items():
        pc = "".join(
            (f"{r.per_class_flip_rate[k]*100:>6.2f}%" if not np.isnan(r.per_class_flip_rate[k]) else f"{'--':>7}")
            for k in range(N_CLASSES)
        )
        print(f"{key:<18}{r.pre_r_dseg:>9.5f}{r.survival_dseg:>10.5f}  {pc}", flush=True)
        rows[key] = {
            "rep": r.rep, "render_h": r.render_hw[0], "slope": r.slope,
            "pre_r_dseg": r.pre_r_dseg, "survival_dseg": r.survival_dseg,
            "per_class_flip_rate": {str(k): r.per_class_flip_rate[k] for k in range(N_CLASSES)},
            "per_class_present_frac": {str(k): r.per_class_present_frac[k] for k in range(N_CLASSES)},
        }

    scale_space = None
    if args.scale_space:
        ker = {h: estimate_R_kernel_sigma(h) for h in [874, 384, 320, 256, 192, 128, 96]}
        heat = heat_vs_interp_lane_survival(lstars[: min(48, lstars.shape[0])], [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        scale_space = {"R_kernel_sigma": ker, "heat_vs_interp": heat}
        print("\n[r_survival] === SCALE-SPACE: R kernel eff sigma (edge-spread @384) ===", flush=True)
        for h, v in ker.items():
            print(f"  render {h:>3}: eff_sigma={v['eff_sigma_px']:.3f}px (10-90 rise {v['rise_10_90_px']:.3f}px)", flush=True)
        print("[r_survival] === HEAT (gaussian blur->argmax) vs INTERP (R): lane flip% ===", flush=True)
        print("  (SDF beats hard under INTERP/R but NOT under HEAT -> R is interpolation-dominant)", flush=True)
        for s, v in heat["heat"].items():
            print(f"  {s}: hard {v['hard_lane_flip']*100:.2f}%  sdf {v['sdf_lane_flip']*100:.2f}%", flush=True)

    msdf_block = None
    if args.msdf:
        msdf_render_hs = [int(h) for h in args.msdf_render_res.split(",")]
        print("\n[r_survival] === MSDF synthetic-corner VALIDATION (NO-FAKE impl proof) ===", flush=True)
        val = validate_msdf_synthetic()
        for shp in ("square", "triangle"):
            v = val[shp]
            print(f"  {shp:<9}: corner_err single {v['corner_err_single']*100:.2f}%  "
                  f"msdf {v['corner_err_msdf']*100:.2f}%  (iou {v['iou_single']:.3f}->{v['iou_msdf']:.3f})  "
                  f"corners={v['n_corners_detected']}  {'PASS' if v['PASS'] else 'FAIL'}", flush=True)
        print(f"  ALL_PASS={val['ALL_PASS']}", flush=True)

        print(f"\n[r_survival] === MSDF LANE A/B/C (n={lstars.shape[0]}, slope={args.msdf_slope:g}, "
              f"band={args.band_px}px) ===", flush=True)
        abc = measure_msdf(lstars, msdf_render_hs, args.msdf_slope, args.band_px, args.approx_eps)
        print(f"{'render':>7}  {'hard lane%':>11}{'sdf lane%':>11}{'msdf lane%':>11}  "
              f"{'hard tot':>10}{'sdf tot':>10}{'msdf tot':>10}", flush=True)
        for h in msdf_render_hs:
            r = abc[h]
            print(f"{h:>7}  {r['hard']['lane_flip']*100:>10.2f}%{r['sdf']['lane_flip']*100:>10.2f}%"
                  f"{r['msdf']['lane_flip']*100:>10.2f}%  {r['hard']['total_dseg']:>10.5f}"
                  f"{r['sdf']['total_dseg']:>10.5f}{r['msdf']['total_dseg']:>10.5f}", flush=True)

        print("\n[r_survival] === single-SDF lane residual DECOMPOSITION (corner vs thin) ===", flush=True)
        decomp = {}
        for h in msdf_render_hs:
            dec = decompose_single_sdf_residual(lstars, h, args.msdf_slope,
                                                corner_radius_px=args.corner_radius_px)
            decomp[h] = dec
            print(f"  render {h:>3}: flips={dec['n_flip']}  corner-any={dec['frac_corner_any']*100:.1f}%  "
                  f"thin-any={dec['frac_thin_any']*100:.1f}%  (corner-only {dec['frac_corner_only']*100:.1f}%  "
                  f"thin-only {dec['frac_thin_only']*100:.1f}%  both {dec['frac_both']*100:.1f}%  "
                  f"neither {dec['frac_neither']*100:.1f}%)", flush=True)

        msdf_block = {
            "synthetic_validation": val,
            "abc_lane_survival": {str(h): abc[h] for h in msdf_render_hs},
            "single_sdf_residual_decomposition": {str(h): decomp[h] for h in msdf_render_hs},
            "params": {"slope": args.msdf_slope, "band_px": args.band_px,
                       "approx_eps": args.approx_eps, "corner_radius_px": args.corner_radius_px},
        }

    payload = {
        "evidence_grade": "macOS research-signal",
        "scale_space": scale_space,
        "msdf": msdf_block,
        "score_claim": False,
        "promotable": False,
        "n_frames": int(lstars.shape[0]),
        "R_chain": "bicubic_up_874x1164 -> uint8 @ camera -> bilinear_down_384x512 (align_corners=False)",
        "lane_geometry": geo,
        "rows": rows,
        "secs": secs,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[r_survival] wrote {args.out}", flush=True)
    return payload


if __name__ == "__main__":
    main()
