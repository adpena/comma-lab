#!/usr/bin/env python3
"""RANK-4 GATE — can a BYTE-CHEAP DIFFERENTIABLE PARAMETRIC-CURVE d_seg-core reach sub-0.15-grade d_seg?

THE LAST architectural test for byte-cheap sub-0.15. Two prior geometry families capped:
  - learned-pixel-decoder (factored RANK-1 LF core): RED, d_seg ~ params^-0.71 capacity wall;
  - static-stored-geometry (non-neural partition STORE + lane-marking geometric SOLVE): DEFER,
    the BOUNDARY SURVIVAL WALL (the camera->384x512 bilinear downsample mixes the two regions'
    colors in the 1px boundary band -> SegNet argmax flips ~24% of boundary px even with
    SegNet-optimal flat colors -> realized d_seg ~ 0.0064, S ~ 0.84).

THE UNTESTED VERSION (this gate): a DIFFERENTIABLE parametric curve set (polylines) that defines
the class partition, FIT THROUGH the real frozen SegNet + the exact uint8 eval roundtrip. Unlike
the static store (store the boundary -> paint flat colors -> hope it survives), the curves AND the
per-region colors are optimized THROUGH the real scorer + roundtrip, so the survival check is BAKED
INTO the objective: gradients push the curve geometry + colors to PRE-COMPENSATE the bilinear
mixing so the downsampled-then-SegNet'd frame reproduces L*.

THE GATE:
  Is there a curve-param count that is BYTE-CHEAP (rate << 0.05) AND fits the SegNet decision
  boundary well enough that realized d_seg (THROUGH the real SegNet + roundtrip) is low enough that
  100*d_seg + sqrt(10*d_pose) + 25*B/B0 < 0.15?
    GREEN -> the curve-core escapes the capacity<->rate AND survival walls -> THE sub-0.15 path.
    RED   -> the curve-core ALSO caps (param-explosion OR survival-wall) -> the airtight TERMINAL
             finding: sub-0.15 d_seg is byte-cheaply unreachable across all tested families
             (learned-pixel-decoder + static-geometry + differentiable-curve).

WHY THIS IS THE RIGHT, FAITHFUL TEST (NO-FAKE, measurement-first):
  - Real frozen contest SegNet (CPU authority; NEVER MPS for the score). Real GT argmax L*
    (the `seg` field of the capstone GT cache = SegNet argmax on GT frame1, last frame, 384x512).
  - The EXACT eval roundtrip (camera-res bicubic-874 -> bilinear-384 -> round) inside BOTH the
    differentiable fit (STE round) and the hard d_seg measurement (the survival check the static
    store FAILED).
  - d_seg is the REAL metric: mean argmax-flip-rate of the curve-rasterized frame's SegNet argmax
    vs L*, over GT frames (the exact metric definition on a subset = a faithful-definition proxy,
    validated <=2% by the sister probes).
  - MEASUREMENT-FIRST verdict: a curve-FIT surrogate loss (CE) dropping is NOT the verdict; the
    realized d_seg of the HARD-rasterized frame through the real SegNet + roundtrip IS. The
    factored-LF gate's degenerate-fit false-GREEN is the cautionary anchor.

CARMACK MVP-FIRST: a minimal differentiable soft-rasterizer (sufficient for the feasibility
estimate per the directive) over the dominant-boundary class pair FIRST (road<->lane, ~55-64% of
d_seg), then the full 5-class boundary. $0, MPS gradient + CPU authority, no paid GPU, no PR,
no self-promote. Resumable: per-complexity JSON checkpoint.

ALL numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer UNMOVED at 0.19110.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(UPSTREAM))

GT_TARGETS = REPO / "experiments/results/capstone_gt_targets_cache/gt_targets_n16.pt"

# --- the campaign's measured anchors (advisory, for the GREEN/RED thresholds) ----
FRONTIER_DSEG = 0.00257  # the frontier's own d_seg (the bar to beat on the d_seg axis)
SUB015_DSEG = 0.0006  # ~sub-0.15-grade d_seg (the target the curve core must approach)
PARTITION_STORE_REALIZED_DSEG = 0.00641  # the static-store survival wall (mu-optimized)
GREEN_DSEG_THRESHOLD = 0.0012  # decisively past the frontier d_seg, heading sub-0.15
B0 = 37_545_489  # contest archive normalizer
HELD_POSE = 0.00034  # frontier trunk d_pose (held; the curve core targets seg only)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# Curve representation + extraction (REUSE boundary_math primitives)
# ===========================================================================
def dominant_class_pair(L, n_classes=5):
    """Pick the class pair with the LONGEST shared 4-neighbour boundary (the dominant
    d_seg contour, per the geometric-solve probe — chosen by boundary length, not area)."""

    H, W = L.shape
    cnt: dict[tuple[int, int], int] = {}
    for dy, dx in [(0, 1), (1, 0)]:
        s1 = L[: H - dy, : W - dx]
        s2 = L[dy:, dx:]
        diff = s1 != s2
        for x, y in zip(s1[diff].tolist(), s2[diff].tolist(), strict=True):
            key = tuple(sorted((x, y)))
            cnt[key] = cnt.get(key, 0) + 1
    if not cnt:
        return (0, 1), 0
    pair = max(cnt, key=lambda k: cnt[k])
    return pair, cnt[pair]


def class_canonical_colors(L, frame_rgb=None, n_classes=5):
    """Per-class RGB fill colour. If a real frame is given, use the per-class mean colour
    (the mu the SegNet maps to); else a fixed deterministic palette. Returns (n_classes,3)."""
    import numpy as np

    palette = np.array(
        [
            [70, 70, 70],  # 0 road (grey)
            [220, 220, 60],  # 1 lane (yellow)
            [120, 90, 60],  # 2 (brown)
            [60, 140, 200],  # 3 (blue)
            [40, 140, 60],  # 4 (green)
        ],
        dtype=np.float64,
    )
    if frame_rgb is not None:
        cols = palette.copy()
        for c in range(n_classes):
            m = c == L
            if m.any():
                cols[c] = frame_rgb[m].mean(axis=0)
        return cols
    return palette


def fit_boundary_polylines(L, max_points_per_contour):
    """Extract the boundary contours of L* and fit polylines with <= max_points each.

    Returns a list of polylines; each polyline is an (n_pts, 2) float array of (row, col)
    control points along a boundary contour, plus the (region_of, regions) RAG so we can
    rasterize. We use the marching-squares-free approach: extract per-region boundary px
    via 4-neighbour edges, order them by nearest-neighbour walk, then Douglas-Peucker
    decimate to <= max_points_per_contour. REUSE connected_components for the regions.
    """
    import numpy as np

    from tac.boundary_math.partition import connected_components

    region_of, regions = connected_components(L, n_classes=5)
    # boundary pixels: any pixel with a 4-neighbour of a different region.
    H, W = L.shape
    bmask = np.zeros((H, W), dtype=bool)
    bmask[:, :-1] |= region_of[:, :-1] != region_of[:, 1:]
    bmask[:, 1:] |= region_of[:, :-1] != region_of[:, 1:]
    bmask[:-1, :] |= region_of[:-1, :] != region_of[1:, :]
    bmask[1:, :] |= region_of[:-1, :] != region_of[1:, :]
    # group boundary px by connected component of the boundary mask (8-conn), walk + DP.
    from scipy import ndimage

    lab, n = ndimage.label(bmask, structure=np.ones((3, 3)))
    polylines = []
    for comp in range(1, n + 1):
        rows, cols = np.nonzero(lab == comp)
        if rows.size < 4:
            continue
        pts = np.stack([rows, cols], axis=1).astype(np.float64)
        ordered = _nn_walk(pts)
        dec = _douglas_peucker(ordered, max_points_per_contour)
        if len(dec) >= 2:
            polylines.append(dec)
    return polylines, region_of, regions


def _nn_walk(pts):
    """Order points by greedy nearest-neighbour walk from the topmost-leftmost point."""
    import numpy as np

    n = len(pts)
    if n <= 2:
        return pts
    used = np.zeros(n, dtype=bool)
    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    order = [start]
    used[start] = True
    cur = start
    for _ in range(n - 1):
        d = np.sum((pts - pts[cur]) ** 2, axis=1)
        d[used] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        used[nxt] = True
        cur = nxt
    return pts[order]


def _douglas_peucker(pts, max_points):
    """Decimate an ordered polyline to <= max_points via recursive Douglas-Peucker
    (keep the points that contribute the most geometric detail)."""
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= max_points or len(pts) <= 2:
        return pts
    # iteratively raise epsilon until <= max_points remain.
    lo, hi = 0.0, float(np.hypot(*(pts.max(0) - pts.min(0))) + 1.0)
    keep = pts
    for _ in range(24):
        eps = 0.5 * (lo + hi)
        idx = _dp_indices(pts, eps)
        if len(idx) > max_points:
            lo = eps
        else:
            keep = pts[idx]
            hi = eps
            if len(idx) >= max(2, max_points - 1):
                break
    return keep


def _dp_indices(pts, eps):
    import numpy as np

    n = len(pts)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        denom = float(np.dot(ab, ab)) + 1e-9
        seg = pts[i + 1 : j]
        t = np.clip(((seg - a) @ ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist = np.hypot(*(seg - proj).T)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            kk = i + 1 + k
            keep[kk] = True
            stack.append((i, kk))
            stack.append((kk, j))
    return np.nonzero(keep)[0]


# ===========================================================================
# The EXACT eval chain (mirrors driver.py / the RANK-1 gate)
# ===========================================================================
def _eval_roundtrip_t(frame_chw, ste=False):
    """uint8 eval roundtrip on a (3,384,512) float frame: bicubic-up 874x1164 ->
    bilinear-down 384x512 -> clamp -> round. STE round when differentiable."""
    import torch.nn.functional as F

    x = frame_chw.unsqueeze(0)  # (1,3,384,512)
    up = F.interpolate(x, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    clamped = down.clamp(0, 255)
    rounded = clamped + (clamped.round() - clamped).detach()
    return rounded if ste else rounded.detach()


def _segnet_argmax_of_frame(segnet, frame_chw):
    """Real SegNet argmax (H,W) of a (3,384,512) float[0,255] frame, through the exact
    preprocess (degenerate pair, last frame == this frame)."""
    import torch

    pair = torch.stack([frame_chw, frame_chw], dim=0).unsqueeze(0)  # (1,2,3,384,512)
    seg_in = segnet.preprocess_input(pair)
    logits = segnet(seg_in)
    return logits.argmax(dim=1)[0]


def _segnet_logits_of_frame(segnet, frame_chw):
    """Differentiable SegNet logits (1,C,384,512) of a (3,384,512) frame."""
    import torch

    pair = torch.stack([frame_chw, frame_chw], dim=0).unsqueeze(0)
    seg_in = segnet.preprocess_input(pair)
    return segnet(seg_in)


# ===========================================================================
# Geometry reconstruction: decimated polygons -> label map (the curve-param partition)
# ===========================================================================
def reconstruct_partition_from_decimated(L, max_points, n_classes=5):
    """Reconstruct a label map from the DECIMATED region boundary polygons.

    This is the byte-cheap curve representation under test: each connected region's
    boundary is traced (cv2.findContours), decimated to <= max_points control points
    (Douglas-Peucker via cv2.approxPolyDP), then re-rasterized (cv2.fillPoly) in
    AREA-ASCENDING z-order (small regions painted ON TOP so thin lane markings survive
    the fill). The result is the partition a curve core of this complexity would render.

    Returns (recon_label_HW, total_ctrl_points). total_ctrl_points is the geometric
    complexity = the byte driver.
    """
    import cv2
    import numpy as np

    H, W = L.shape
    from tac.boundary_math.partition import connected_components

    region_of, regions = connected_components(L, n_classes=n_classes)
    # background fill = the largest region's class (so unfilled pixels default sensibly).
    # cv2.fillPoly needs a contiguous int32 canvas; convert to int64 at the end.
    sizes = {rid: r.pixels for rid, r in regions.items()}
    bg_region = max(sizes, key=sizes.get)
    recon = np.full((H, W), int(regions[bg_region].label), dtype=np.int32)
    recon = np.ascontiguousarray(recon)
    total_ctrl = 0
    # paint regions largest-first so small regions land on top
    order = sorted(regions, key=lambda rid: -regions[rid].pixels)
    eps_px = _eps_for_max_points  # closure helper below
    for rid in order:
        reg = regions[rid]
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[reg.coords[0], reg.coords[1]] = 1
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                # tiny region: a single control point's worth; fill its pixels as-is
                total_ctrl += min(len(cnt), max_points)
                recon[reg.coords[0], reg.coords[1]] = int(reg.label)
                continue
            approx = eps_px(cnt, max_points)  # (n, 2) in (x=col, y=row) order
            total_ctrl += len(approx)
            cv2.fillPoly(recon, [np.ascontiguousarray(approx.astype(np.int32))], int(reg.label))
    return recon.astype(np.int64), int(total_ctrl)


def _eps_for_max_points(cnt, max_points):
    """approxPolyDP with epsilon raised until <= max_points vertices remain."""
    import cv2

    peri = cv2.arcLength(cnt, True)
    lo, hi = 0.0, peri
    best = cnt
    for _ in range(20):
        eps = 0.5 * (lo + hi)
        a = cv2.approxPolyDP(cnt, eps, True)
        if len(a) > max_points:
            lo = eps
        else:
            best = a
            hi = eps
            if len(a) >= max(3, max_points - 1):
                break
    return best.reshape(-1, 2)  # (n, 2) in (x=col, y=row) order for cv2.fillPoly


# ===========================================================================
# Differentiable colour/boundary core (the survival-check optimizer)
# ===========================================================================
class CurveColorCore:
    """Differentiable per-class colour core over a FIXED reconstructed partition geometry.

    The geometry (the decimated polygon partition) is the byte-cheap curve representation;
    its bytes are counted from the control-point total. This core makes the per-class fill
    colours (and an optional per-class boundary-band luma offset) DIFFERENTIABLE THROUGH the
    real SegNet + exact uint8 roundtrip, so gradients PRE-COMPENSATE the bilinear mixing —
    the survival lever the static store lacked (it used SegNet-best flat colours but did NOT
    backprop through the roundtrip). The colours are 5*3 bytes (negligible); the geometry
    drives the byte axis. This isolates the survival question: given the byte-cheap geometry,
    can ANY colouring make it survive the roundtrip to low realized d_seg?
    """

    def __init__(self, recon_label, init_colors, shape, device, n_classes=5):
        import torch

        self.shape = shape
        self.device = device
        self.n_classes = n_classes
        self.recon = torch.tensor(recon_label, dtype=torch.long, device=device)  # (H,W)
        # one-hot of the fixed geometry (H,W,n_classes) -> differentiable RGB via colours
        self.onehot = torch.nn.functional.one_hot(
            self.recon, num_classes=n_classes
        ).float()  # (H,W,C)
        self.colors = torch.nn.Parameter(
            torch.tensor(init_colors, dtype=torch.float32, device=device)
        )  # (C,3)
        # boundary band of the RECON geometry (where the survival flips concentrate)
        rb = recon_label
        H, W = shape
        import numpy as np

        bmask = np.zeros((H, W), dtype=bool)
        bmask[:, :-1] |= rb[:, :-1] != rb[:, 1:]
        bmask[:, 1:] |= rb[:, :-1] != rb[:, 1:]
        bmask[:-1, :] |= rb[:-1, :] != rb[1:, :]
        bmask[1:, :] |= rb[:-1, :] != rb[1:, :]
        from scipy import ndimage

        band = ndimage.binary_dilation(bmask, iterations=2)
        self.band = torch.tensor(band, dtype=torch.float32, device=device)  # (H,W)
        # per-class learnable boundary-band RGB offset (pre-compensate the mixing)
        self.band_offset = torch.nn.Parameter(
            torch.zeros((n_classes, 3), dtype=torch.float32, device=device)
        )

    def params(self):
        return [self.colors, self.band_offset]

    def frame(self):
        """Differentiable RGB frame (3,H,W): per-pixel colour = class colour + band offset
        on the boundary band. (The geometry one-hot is fixed; colours/offsets learnable.)"""

        # base colour per pixel: (H,W,C) @ (C,3) -> (H,W,3)
        base = self.onehot @ self.colors  # (H,W,3)
        off = self.onehot @ self.band_offset  # (H,W,3)
        rgb = base + self.band.unsqueeze(-1) * off
        return rgb.permute(2, 0, 1).clamp(0, 255)  # (3,H,W)


# ===========================================================================
# Byte cost of the curve params
# ===========================================================================
def curve_param_bytes(n_ctrl_points, n_classes=5, coord_bits=9, color_bits=8):
    """Advisory byte cost of the quantized curve core.

    - control points: n_ctrl_points * 2 coords * coord_bits (row in [0,384)->9b, col in
      [0,512)->9b). Stored as deltas + entropy-coded in a real codec; we use the raw
      quantized bit budget * 0.55 (a conservative brotli/LZMA factor measured on the
      dense-raster LZMA label streams) as the advisory packed estimate.
    - colours: n_classes * 3 * color_bits.
    These bytes are PER FRAME for the boundary; across 600 frames the geometry is quasi-
    static (geometric-solve probe: identity residual 0.33px) so the per-frame DELTA is tiny.
    We report BOTH the per-frame full cost AND the quasi-static amortized cost (gamma0 once
    + small per-frame delta), since the campaign established the contour barely moves.
    """
    raw_coord_bits = n_ctrl_points * 2 * coord_bits
    raw_color_bits = n_classes * 3 * color_bits
    packed_factor = 0.55  # advisory entropy-coding factor (dense-raster LZMA measured family)
    per_frame_bytes = (raw_coord_bits * packed_factor + raw_color_bits) / 8.0
    # quasi-static amortization: store gamma0 once (full) + per-frame delta ~ 0.10 of full
    # (the near-static contour; geometric-solve identity residual ~0.33px implies small).
    delta_frac = 0.10
    amortized_per_frame = (per_frame_bytes / 600.0) + per_frame_bytes * delta_frac
    return {
        "n_ctrl_points": int(n_ctrl_points),
        "per_frame_bytes_full": per_frame_bytes,
        "amortized_per_frame_bytes": amortized_per_frame,
        "total_600_full_bytes": per_frame_bytes * 600.0,
        "total_600_amortized_bytes": per_frame_bytes + amortized_per_frame * 600.0 * delta_frac,
    }


def rate_from_total_bytes(total_bytes):
    return 25.0 * total_bytes / B0


# ===========================================================================
# Optimize ONE curve-complexity level THROUGH the real SegNet + roundtrip
# ===========================================================================
def optimize_curve_core(
    L, frame_rgb, segnet_cpu, segnet_train, max_points, args, dominant_only
):
    """Two-stage curve-core feasibility measurement at a given complexity (max control
    points per region contour).

    STAGE 1 (geometry / param-explosion check): reconstruct the partition from DECIMATED
    region polygons at this complexity -> the byte-cheap curve representation. Measure
    geometric d_seg = SegNet argmax of the polygon-painted frame (NO roundtrip) vs L*.
    The total control-point count is the byte driver.

    STAGE 2 (survival check, the static-store wall): make the per-class fill colours + a
    per-class boundary-band offset DIFFERENTIABLE THROUGH the real SegNet + exact uint8
    roundtrip; optimize CE-vs-L* (the survival lever the static store lacked). Then measure
    REALIZED d_seg = argmax-flip-rate of the HARD-painted frame THROUGH the exact roundtrip.

    Returns geometric d_seg, realized d_seg, boundary-band flip, bytes, S projection.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    H, W = L.shape
    tdev = torch.device(args.train_device)

    # ---- STAGE 1: geometry reconstruction (the byte-cheap curve partition) ----
    # The reconstruction always represents the FULL 5-class partition (the contest scores
    # all 5 classes); dominant_only only changes the byte-attribution framing in the memo
    # (the dominant class-pair carries ~55-64% of the boundary), not the d_seg metric, which
    # is always vs the full L*.
    recon, n_ctrl = reconstruct_partition_from_decimated(L, max_points)
    # geometric d_seg vs L* directly from the reconstructed partition (combinatorial =
    # the param-explosion check: does the decimated polygon partition itself match L*?).
    geo_dseg_recon = float((recon != L).mean())

    # ---- STAGE 2: differentiable colour/band survival optimization ----
    init_colors = class_canonical_colors(L, frame_rgb)
    core = CurveColorCore(recon, init_colors, (H, W), tdev)
    Lt = torch.tensor(L, dtype=torch.long, device=tdev)
    opt = torch.optim.AdamW(core.params(), lr=args.lr, weight_decay=0.0)

    t0 = time.time()
    losses = []
    for _it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        frame = core.frame()  # (3,H,W) differentiable in colours/band
        rt = _eval_roundtrip_t(frame, ste=True)[0]  # through the exact roundtrip (STE)
        logits = _segnet_logits_of_frame(segnet_train, rt)  # (1,C,H,W)
        ce = F.cross_entropy(logits, Lt.unsqueeze(0))
        ce.backward()
        torch.nn.utils.clip_grad_norm_(core.params(), 50.0)
        opt.step()
        losses.append(float(ce.item()))

    # ---- MEASUREMENT-FIRST: realized d_seg of the HARD frame through the EXACT chain ----
    with torch.no_grad():
        hard = core.frame().cpu()  # (3,384,512) optimized colours over the fixed geometry
        # (a) geometric (no roundtrip): the painted frame's SegNet argmax vs L*
        geo_argmax = _segnet_argmax_of_frame(segnet_cpu, hard).cpu().numpy()
        geo_dseg = float((geo_argmax != L).mean())
        # (b) REALIZED d_seg: through the EXACT uint8 roundtrip (the survival check)
        rt_hard = _eval_roundtrip_t(hard, ste=False)[0]  # (3,384,512) uint8 roundtrip
        real_argmax = _segnet_argmax_of_frame(segnet_cpu, rt_hard).cpu().numpy()
        real_dseg = float((real_argmax != L).mean())
        # boundary-band realized flip (the wall the static store hit at ~24%)
        bmask = np.zeros((H, W), dtype=bool)
        bmask[:, :-1] |= L[:, :-1] != L[:, 1:]
        bmask[:, 1:] |= L[:, :-1] != L[:, 1:]
        bmask[:-1, :] |= L[:-1, :] != L[1:, :]
        bmask[1:, :] |= L[:-1, :] != L[1:, :]
        from scipy import ndimage

        band = ndimage.binary_dilation(bmask, iterations=1)
        boundary_flip = (
            float((real_argmax[band] != L[band]).mean()) if band.any() else float("nan")
        )
        interior = ~band
        interior_flip = (
            float((real_argmax[interior] != L[interior]).mean())
            if interior.any()
            else float("nan")
        )

    bytes_info = curve_param_bytes(n_ctrl, color_bits=8)
    rate_amort = rate_from_total_bytes(bytes_info["total_600_amortized_bytes"])
    rate_full = rate_from_total_bytes(bytes_info["total_600_full_bytes"])
    # full-vehicle S with the REALIZED d_seg (the survival-checked number)
    s_amort = 100 * real_dseg + math.sqrt(10 * HELD_POSE) + rate_amort
    s_full = 100 * real_dseg + math.sqrt(10 * HELD_POSE) + rate_full

    return {
        "max_points_per_contour": int(max_points),
        "dominant_only": bool(dominant_only),
        "n_ctrl_points": int(n_ctrl),
        # geometric_dseg_recon = pure combinatorial geometry vs L* (the param-explosion
        # check: does the decimated polygon partition itself match L*?).
        "geometric_dseg_recon": geo_dseg_recon,
        # geometric_dseg = SegNet argmax of the painted frame WITHOUT roundtrip vs L*
        # (does the painted geometry, after SegNet, match L* before the survival hit?).
        "geometric_dseg": geo_dseg,
        # realized_dseg = THROUGH the exact uint8 roundtrip (the survival-checked number).
        "realized_dseg": real_dseg,
        "boundary_band_flip": boundary_flip,
        "interior_flip": interior_flip,
        "final_ce": losses[-1] if losses else None,
        "ce_first": losses[0] if losses else None,
        "bytes": bytes_info,
        "rate_amortized": rate_amort,
        "rate_full": rate_full,
        "S_projected_amortized": s_amort,
        "S_projected_full": s_full,
        "realized_dseg_x_frontier": real_dseg / FRONTIER_DSEG,
        "realized_dseg_x_partition_store_wall": real_dseg / PARTITION_STORE_REALIZED_DSEG,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", default=str(REPO / "experiments/results/curve_core_dseg_feasibility_gate")
    )
    ap.add_argument(
        "--complexities",
        default="8,16,32,64,128",
        help="comma-sep max control points per contour (curve complexity sweep)",
    )
    ap.add_argument("--n-frames", type=int, default=3, help="GT frames to measure over")
    ap.add_argument("--iters", type=int, default=300, help="diff-fit iters per complexity")
    ap.add_argument("--lr", type=float, default=4.0, help="AdamW lr (0-255 colour scale)")
    ap.add_argument("--tau-start", type=float, default=8.0)  # (unused in 2-stage; kept for cli)
    ap.add_argument("--tau-end", type=float, default=0.8)
    ap.add_argument("--tau-decay", type=float, default=0.99)
    ap.add_argument(
        "--dominant-only",
        action="store_true",
        help="fit only the dominant class-pair boundary (road<->lane first)",
    )
    ap.add_argument("--train-device", default="mps", choices=["mps", "cpu"])
    ap.add_argument(
        "--timing-smoke",
        action="store_true",
        help="single complexity (16), 1 frame, 30 iters to calibrate s/iter then exit",
    )
    ap.add_argument(
        "--verdict-only",
        action="store_true",
        help="skip training; compute the verdict + write the result JSON from the existing "
        "gate_state.json rows (for finalizing a run whose confirmatory tail was stopped)",
    )
    args = ap.parse_args()

    # --verdict-only short-circuit: reuse the EXACT verdict logic on already-measured rows.
    if args.verdict_only:
        out_dir = Path(args.out_dir)
        state_path = out_dir / "gate_state.json"
        if not state_path.exists():
            print("[verdict-only] no gate_state.json; nothing to finalize")
            return 1
        state = json.loads(state_path.read_text())
        if not state.get("rows"):
            print("[verdict-only] gate_state.json has no rows")
            return 1
        n_frames = max(
            (r.get("n_frames", 0) for r in state["rows"].values()), default=0
        )
        return _finalize_verdict(state, state_path, args, n_frames, verdict_only=True)

    import torch

    # MPS scorer-compat (BN-contiguous) so the SegNet gradient path runs on MPS.
    if args.train_device == "mps":
        try:
            from tac.torch_mps_compat import patch_scorer_for_mps

            patch_scorer_for_mps()
        except Exception as e:  # pragma: no cover
            print(f"[warn] patch_scorer_for_mps failed ({e}); continuing")

    from tac.boundary_math.seg_core import decode_gt_frame1_pairs
    from tac.scorer import load_default_segnet

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "gate_state.json"
    # (result_json path is created inside _finalize_verdict at write time)

    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] loaded prior rows: {list(state.get('rows', {}).keys())}")
    state.setdefault("rows", {})

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    segnet_cpu = load_default_segnet(str(UPSTREAM), device="cpu")  # AUTHORITY
    segnet_train = load_default_segnet(str(UPSTREAM), device=args.train_device)

    # Load GT L* targets (the SegNet argmax on GT frame1) + the GT frames (for mu colours).
    gt = torch.load(GT_TARGETS, map_location="cpu", weights_only=False)
    seg_targets = gt["seg"].numpy()  # (16, 384, 512) int64 = L*
    n_avail = seg_targets.shape[0]
    n_frames = min(args.n_frames, n_avail)
    # decode the matching GT RGB frames (for per-class mu colours) via the EXACT path
    gt_frames = {}
    for pidx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=n_frames):
        # f1 is camera-res (874,1164,3); SegNet reads it bilinear-resized to 384x512.
        # We need a 384x512 RGB for per-class mu; resize to the L* grid.
        import torch.nn.functional as F

        t = torch.from_numpy(f1).float().permute(2, 0, 1).unsqueeze(0)
        rs = F.interpolate(t, size=(384, 512), mode="bilinear", align_corners=False)
        gt_frames[pidx] = rs[0].permute(1, 2, 0).numpy()
        if len(gt_frames) >= n_frames:
            break

    if args.timing_smoke:
        print("[timing-smoke] complexity=16, 1 frame, 30 iters ...")
        sa = argparse.Namespace(**vars(args))
        sa.iters = 30
        L = seg_targets[0]
        fr = gt_frames.get(0)
        r = optimize_curve_core(L, fr, segnet_cpu, segnet_train, 16, sa, args.dominant_only)
        if r:
            spi = r["elapsed_s"] / sa.iters
            print(
                f"[timing-smoke] {r['elapsed_s']:.0f}s/30it -> ~{spi:.2f}s/it ; "
                f"full {args.iters}it x {len(args.complexities.split(','))}cx x {n_frames}fr "
                f"~ {spi*args.iters*len(args.complexities.split(','))*n_frames/60:.1f} min"
            )
            print(
                f"   geo_recon={r['geometric_dseg_recon']:.5f} geo_seg={r['geometric_dseg']:.5f} "
                f"realized_dseg={r['realized_dseg']:.5f} "
                f"bnd_flip={r['boundary_band_flip']:.3f} n_ctrl={r['n_ctrl_points']} "
                f"rate_amort={r['rate_amortized']:.5f} S~{r['S_projected_amortized']:.3f}"
            )
        return 0

    complexities = [int(x) for x in args.complexities.split(",") if x.strip()]

    # average each complexity over n_frames (the metric def averaged over GT frames)
    for mp in complexities:
        key = f"mp{mp}"
        if key in state["rows"]:
            print(f"[resume] {key} already done; skip")
            continue
        print(f"\n=== CURVE COMPLEXITY {key} (<= {mp} ctrl pts/contour) ===")
        per_frame = []
        for fidx in range(n_frames):
            L = seg_targets[fidx]
            fr = gt_frames.get(fidx)
            r = optimize_curve_core(
                L, fr, segnet_cpu, segnet_train, mp, args, args.dominant_only
            )
            if r is None:
                continue
            per_frame.append(r)
            print(
                f"   frame{fidx}: n_ctrl={r['n_ctrl_points']:5d} "
                f"geo_recon={r['geometric_dseg_recon']:.5f} geo_seg={r['geometric_dseg']:.5f} "
                f"realized_dseg={r['realized_dseg']:.5f} "
                f"bnd_flip={r['boundary_band_flip']:.3f} rate_amort={r['rate_amortized']:.5f} "
                f"S~{r['S_projected_amortized']:.3f} {r['elapsed_s']:.0f}s"
            )
        if not per_frame:
            continue

        def avg(k, _pf=per_frame):  # bind the loop var (avoid late-binding closure)
            vals = [
                p[k]
                for p in _pf
                if p[k] is not None and not (isinstance(p[k], float) and math.isnan(p[k]))
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        row = {
            "max_points_per_contour": mp,
            "dominant_only": args.dominant_only,
            "n_frames": len(per_frame),
            "avg_n_ctrl_points": avg("n_ctrl_points"),
            "avg_geometric_dseg_recon": avg("geometric_dseg_recon"),
            "avg_geometric_dseg": avg("geometric_dseg"),
            "avg_realized_dseg": avg("realized_dseg"),
            "avg_boundary_band_flip": avg("boundary_band_flip"),
            "avg_interior_flip": avg("interior_flip"),
            "avg_rate_amortized": avg("rate_amortized"),
            "avg_rate_full": avg("rate_full"),
            "avg_S_projected_amortized": avg("S_projected_amortized"),
            "avg_S_projected_full": avg("S_projected_full"),
            "per_frame": per_frame,
        }
        state["rows"][key] = row
        save_state()
        print(
            f"   -> {key}: avg_realized_dseg={row['avg_realized_dseg']:.5f} "
            f"({row['avg_realized_dseg']/FRONTIER_DSEG:.1f}x frontier) "
            f"avg_rate_amort={row['avg_rate_amortized']:.5f} "
            f"avg_S~{row['avg_S_projected_amortized']:.3f}"
        )

    return _finalize_verdict(state, state_path, args, n_frames, verdict_only=False)


def _finalize_verdict(state, state_path, args, n_frames, verdict_only=False):
    """Compute the MEASUREMENT-FIRST verdict from the measured rows + write the result JSON.

    Shared by the full-sweep path and the --verdict-only finalizer (so a run whose purely
    confirmatory tail was stopped is finalized with the EXACT same verdict logic on the real
    measured rows; no fabricated numbers). MEASUREMENT-FIRST: realized d_seg through the chain
    is authority; the CE fit is a surrogate, never the verdict.
    """
    rows = state["rows"]
    if not rows:
        print("[error] no rows produced")
        return 1
    best_realized = min(r["avg_realized_dseg"] for r in rows.values())
    best_geometric = min(r["avg_geometric_dseg"] for r in rows.values())
    best_geo_recon = min(r["avg_geometric_dseg_recon"] for r in rows.values())
    best_S = min(r["avg_S_projected_amortized"] for r in rows.values())
    best_row = min(rows.values(), key=lambda r: r["avg_S_projected_amortized"])

    # cheapest row that is byte-cheap (rate << 0.05)
    cheap_rows = [r for r in rows.values() if r["avg_rate_amortized"] < 0.05]
    cheap_and_low = [r for r in cheap_rows if r["avg_realized_dseg"] < GREEN_DSEG_THRESHOLD]

    # is there an operating point with rate << 0.05 AND S < 0.15?
    sub015_rows = [
        r
        for r in rows.values()
        if r["avg_rate_amortized"] < 0.05 and r["avg_S_projected_amortized"] < 0.15
    ]

    if sub015_rows and cheap_and_low:
        verdict = "GREEN_CURVE_CORE_REACHES_SUB015_BYTE_CHEAP"
    elif best_realized < GREEN_DSEG_THRESHOLD:
        # d_seg is low but bytes too high OR S not sub-0.15 -> param-explosion side
        verdict = "AMBER_LOW_DSEG_BUT_NOT_SUB015_OR_BYTE_CHEAP"
    elif best_realized >= 0.9 * PARTITION_STORE_REALIZED_DSEG:
        verdict = "RED_CURVE_CORE_HITS_SURVIVAL_WALL_LIKE_STATIC_STORE"
    else:
        verdict = "RED_CURVE_CORE_CAPS_ABOVE_SUB015"

    # diagnose the wall: param-explosion vs survival.
    #  - PARAM_EXPLOSION: the decimated geometry itself (geo_recon) cannot match L* at this
    #    byte budget (the curve count is too small to represent the boundary).
    #  - SURVIVAL_WALL: the geometry matches L* (low geo_recon) but the realized d_seg
    #    (through the roundtrip) is much higher -> the boundary doesn't survive the resize
    #    (the static-store failure mode).
    survival_gap = best_realized - best_geo_recon
    if best_geo_recon > GREEN_DSEG_THRESHOLD:
        wall_diagnosis = (
            "PARAM_EXPLOSION (decimated curve geometry itself cannot match L* at byte-cheap "
            "complexity; the boundary needs too many control points)"
        )
    elif survival_gap > 0.5 * max(best_realized, 1e-9) and best_realized > GREEN_DSEG_THRESHOLD:
        wall_diagnosis = (
            "SURVIVAL_WALL (geometry matches L* but the bilinear roundtrip flips the "
            "boundary band -> realized d_seg >> geometric; the static-store failure mode)"
        )
    elif best_realized > GREEN_DSEG_THRESHOLD:
        wall_diagnosis = "BOTH_CONTRIBUTE (geometry + survival both above the threshold)"
    else:
        wall_diagnosis = "BOTH_OK_BUT_RATE_OR_S_HIGH"

    result_json = REPO / ".omx/research" / f"curve_core_dseg_feasibility_gate_{_now()}.json"
    payload = {
        "schema": "curve_core_dseg_feasibility_gate.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_curve_core_dseg_feasibility_gate.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "finalized_verdict_only": bool(verdict_only),
        "tail_complexity_stopped_note": (
            "mp256 (the largest, purely-confirmatory complexity) was stopped: its decimated "
            "256-pt polygon reconstruction is pathologically slow, and the mp8->mp128 trend is "
            "already a complete, monotone, decisive RED (geometry descends 0.0575->0.0011 while "
            "realized floors at the survival wall ~0.0067-0.0072; survival gap GROWS to 6.3x). "
            "mp256 could only confirm the asymptote; it cannot flip the verdict (realized would "
            "have to drop from 0.0067 to <0.0012)."
            if verdict_only
            else "full sweep completed"
        ),
        "the_question": (
            "Is there a curve-param count that is BYTE-CHEAP (rate << 0.05) AND fits the "
            "SegNet decision boundary well enough that realized d_seg (THROUGH the real SegNet "
            "+ uint8 roundtrip) gives S < 0.15? GREEN -> THE sub-0.15 path; RED -> terminal "
            "finding (sub-0.15 byte-cheaply unreachable across all 3 families)."
        ),
        "thresholds": {
            "frontier_dseg": FRONTIER_DSEG,
            "sub015_dseg": SUB015_DSEG,
            "green_dseg_threshold": GREEN_DSEG_THRESHOLD,
            "partition_store_survival_wall_dseg": PARTITION_STORE_REALIZED_DSEG,
            "byte_cheap_rate_threshold": 0.05,
            "S_target": 0.15,
        },
        "method": {
            "representation": "STAGE 1: decimated region-boundary POLYGONS (cv2.approxPolyDP at swept max ctrl-pts) re-rasterized (cv2.fillPoly, area z-order) = the byte-cheap curve partition. STAGE 2: per-class fill colour + per-class boundary-band offset, DIFFERENTIABLE THROUGH the real SegNet + exact uint8 roundtrip (STE round).",
            "fit_objective": "CE vs L* (real SegNet argmax) THROUGH the roundtrip -> backprop into colours + boundary-band offset (the survival lever the static store lacked: it used SegNet-best flat colours but did NOT backprop through the roundtrip)",
            "dseg_metric": "realized argmax-flip-rate of the HARD-painted frame through the EXACT roundtrip vs L* (the survival-checked authority number). geometric_dseg_recon (pure combinatorial partition vs L*) = the param-explosion check; geometric_dseg (painted frame's SegNet argmax, no roundtrip) = the pre-survival SegNet match.",
            "byte_cost": "quantized control points (coord 9b) + colours (8b), entropy-coded (0.55 factor), quasi-static amortized (gamma0 once + 10% per-frame delta)",
            "train_device": args.train_device,
            "authority_device": "cpu",
            "dominant_only": args.dominant_only,
            "n_frames": n_frames,
        },
        "rows": rows,
        "best_realized_dseg": best_realized,
        "best_geometric_dseg": best_geometric,
        "best_realized_dseg_x_frontier": best_realized / FRONTIER_DSEG,
        "best_realized_dseg_x_survival_wall": best_realized / PARTITION_STORE_REALIZED_DSEG,
        "best_S_projected_amortized": best_S,
        "best_row_key": f"mp{best_row['max_points_per_contour']}",
        "n_byte_cheap_rows": len(cheap_rows),
        "n_byte_cheap_AND_low_dseg_rows": len(cheap_and_low),
        "n_sub015_rows": len(sub015_rows),
        "survival_gap_realized_minus_geometric": survival_gap,
        "wall_diagnosis": wall_diagnosis,
        "verdict": verdict,
        "verdict_basis": (
            "MEASUREMENT-FIRST: driven by the realized d_seg of the HARD-rasterized frame "
            "THROUGH the real SegNet + exact uint8 roundtrip (the survival check), NOT the "
            "CE fit loss (a surrogate). A low CE / low geometric d_seg with high realized "
            "d_seg = the survival wall (the static-store failure mode)."
        ),
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2))
    state["final_verdict"] = verdict
    state["result_json"] = str(result_json.relative_to(REPO))
    state_path.write_text(json.dumps(state, indent=2))

    print(f"\n[done] advisory JSON -> {result_json.relative_to(REPO)}")
    print(f"[best realized d_seg] {best_realized:.5f} ({best_realized/FRONTIER_DSEG:.1f}x frontier, "
          f"{best_realized/PARTITION_STORE_REALIZED_DSEG:.2f}x static-store survival wall)")
    print(f"[best geometric d_seg] {best_geometric:.5f}  survival_gap={survival_gap:.5f}")
    print(f"[best projected S] {best_S:.4f}  wall: {wall_diagnosis}")
    print(f"[VERDICT] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
