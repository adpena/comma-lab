#!/usr/bin/env python3
"""$0 d_seg-SIDE feasibility probe for the from-scratch task-space rep (#155) — the TWO
untested closed-form corners the recursive adversarial review surfaced.

THE QUESTION (measurement-first, NO-FAKE): can the d_seg-CRITICAL boundary be coded
CHEAPER than the frontier's effective d_seg contribution? The 0.19110 frontier is near
its task-RD floor (p_suff_task_ablation RED); #155 must BEAT it on the d_seg axis. The
frontier's own breakdown: rate 0.11797 + d_seg 0.056 (d_seg=0.00056) + d_pose 0.01715.

TWO CORNERS (each reuses the curve-gate's EXACT roundtrip + realized-d_seg + GT-L*
harness; no training, NO MPS for authority, $0 CPU):

  CORNER 1 (#149) — CAMERA-RES SUB-PIXEL BOUNDARY PLACEMENT.
    The eval loads the recon as raw uint8 CAMERA-RES (874x1164) and applies D (bilinear
    down to 384x512) INSIDE the SegNet preprocess (modules.py:113). The 384-grid gates all
    render at 384 then up->Q->down, eating the up-then-down blur; but a camera-res rep skips
    U entirely (eval has NO up; the up is the decoder's inflate choice). At camera-res the
    boundary has ~3x the pixels and can be ANTI-ALIASED sub-pixel so D's weighted average
    lands the argmax on the correct side. We SOLVE the camera-res boundary-band pixel values
    (closed-form-flavoured: gradient descent through D+SegNet on CPU, real frozen scorer) so
    the flip is SET at 874x1164 BEFORE D averages.
    GATE: does sub-pixel camera-res placement beat the bnd_flip the 384-grid flat-paint hits?
    Quantify the byte cost of the sub-pixel boundary code.

  CORNER 2 (#148) — CROSS-FRAME KEYFRAME + TINY WARP.
    d_seg is scored on the LAST frame of each pair; consecutive last-frames drift only ~1%
    (measured). Take N consecutive GT frames; code ONE keyframe boundary/texture + a per-frame
    warp (translation / affine / projective), and measure per-frame realized d_seg through the
    real roundtrip+SegNet + the TRUE amortized rate (keyframe ONCE + tiny per-frame warp).
    GATE: does paying the boundary ONCE amortize d_seg across frames at near-zero per-frame
    bytes, sidestepping the per-frame-from-scratch problem?

VERDICT (for each corner): the realized-d_seg-vs-byte result + whether it codes the
d_seg-critical boundary BELOW the frontier's effective d_seg-byte share.
  GREEN = a piece of #155 that beats the frontier on d_seg-rate.
  RED   = doesn't (the frontier's d_seg coding is already near-minimal on this axis).
False-GREEN guard: realized d_seg through the EXACT chain is authority, NOT a fit loss.
False-RED guard: report the BEST of {flat-paint, camera-res-native, sub-pixel-solved,
mu-init, gt-texture-band} so an under-powered solve can't fake a RED.

NO-FAKE: real frozen contest SegNet (CPU AUTHORITY; NEVER MPS for the score). Real GT
camera frames (yuv420_to_rgb) -> the cached L* IS D(GT_cam)->SegNet->argmax (verified
match==1.0). The EXACT eval chain. d_seg = real argmax-flip-rate vs L*. Resumable per-corner
JSON. All numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer UNMOVED at 0.19110.
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

# --- contest geometry (verified from upstream/frame_utils.py + modules.py) ---
CAM_H, CAM_W = 874, 1164  # camera_size=(W=1164,H=874)
MH, MW = 384, 512  # segnet_model_input_size=(W=512,H=384)
B0 = 37_545_489  # contest archive normalizer

# --- frontier anchors (advisory; the BAR to beat on the d_seg axis) ---
FRONTIER_DSEG = 0.00056  # frontier realized d_seg (full-600 report)
FRONTIER_DSEG_TERM = 100 * FRONTIER_DSEG  # 0.056 = the frontier's d_seg contribution to S
FRONTIER_RATE_TERM = 0.11797  # 25*177169/B0
SUB015_DSEG = 0.0006  # ~sub-0.15-grade d_seg
HELD_POSE = 0.00034  # frontier trunk d_pose (held; these corners target d_seg only)
GREEN_DSEG_THRESHOLD = 0.0012  # decisively past the frontier d_seg, heading sub-0.15
CONTOUR_BYTES_PER_FRAME = 914  # measured dense-raster LZMA length of one L* (not a boundary store)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# Shared harness (reuses the exact GT decode + SegNet authority)
# ===========================================================================
def _load_segnet_cpu():
    from tac.boundary_math.seg_core import load_real_segnet

    return load_real_segnet("cpu")  # CPU AUTHORITY; never MPS for the score


def _load_gt_cache():
    import torch

    gt = torch.load(GT_TARGETS, map_location="cpu", weights_only=False)
    return gt["seg"].numpy()  # (16,384,512) int64 = L*


def _decode_gt_camera_frames(n_frames):
    """Decode the first n_frames GT camera frames (the last-frame of each non-overlapping
    pair) via the EXACT upstream path. Returns dict pidx -> (874,1164,3) uint8."""
    from tac.boundary_math.seg_core import decode_gt_frame1_pairs

    frames = {}
    for pidx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=n_frames):
        frames[pidx] = f1  # camera-res uint8
        if len(frames) >= n_frames:
            break
    return frames


def _segnet_argmax_cam(segnet, cam_hwc):
    """Real SegNet argmax (384,512) of a CAMERA-RES (874,1164,3) float[0,255] frame, through
    the EXACT preprocess (D = bilinear-down inside preprocess_input). This is the eval-native
    chain for a recon stored at camera-res: NO bicubic-up (that is the inflate decoder's
    choice, not an eval step)."""
    import numpy as np

    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    am, _ = measure_segnet_argmax(segnet, np.asarray(cam_hwc, dtype=np.float64))
    return am


def _segnet_argmax_via_384_roundtrip(segnet, frame_384_hwc):
    """The OLD 384-grid path: render at 384, bicubic-up to camera-res, clamp+round (uint8),
    then the eval D+SegNet. This is what the prior gates measured (up->Q->down)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(frame_384_hwc, dtype=np.float64)).permute(2, 0, 1)[None].float()
    up = F.interpolate(t, size=(CAM_H, CAM_W), mode="bicubic", align_corners=False)
    clamped = up.clamp(0, 255)
    q = (clamped + (clamped.round() - clamped).detach()).detach()  # uint8 quant at camera res
    cam = q[0].permute(1, 2, 0).numpy()
    return _segnet_argmax_cam(segnet, cam)


def _per_class_mu_colors(L_384, frame_cam, n_classes=5):
    """Per-class mean GT camera color (the mu the SegNet maps to), via L* upsampled NN to
    camera grid. Returns (n_classes,3) and the camera-grid label map Lup."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    Lt = torch.from_numpy(L_384.astype(np.int64))[None, None].float()
    Lup = F.interpolate(Lt, size=(CAM_H, CAM_W), mode="nearest")[0, 0].numpy().astype(np.int64)
    cols = np.zeros((n_classes, 3), dtype=np.float64)
    fr = frame_cam.astype(np.float64)
    for c in range(n_classes):
        m = Lup == c
        if m.any():
            cols[c] = fr[m].mean(axis=0)
    return cols, Lup


def _d_seg(argmax_384, L_384):

    return float((argmax_384 != L_384).mean())


def _boundary_band_384(L, iters=1):
    import numpy as np
    from scipy import ndimage

    bmask = np.zeros((MH, MW), dtype=bool)
    bmask[:, :-1] |= L[:, :-1] != L[:, 1:]
    bmask[:, 1:] |= L[:, :-1] != L[:, 1:]
    bmask[:-1, :] |= L[:-1, :] != L[1:, :]
    bmask[1:, :] |= L[:-1, :] != L[1:, :]
    return ndimage.binary_dilation(bmask, iterations=iters)


def _band_flip(argmax_384, L_384, band):

    if band.any():
        return float((argmax_384[band] != L_384[band]).mean())
    return float("nan")


def rate_from_total_bytes(total_bytes):
    return 25.0 * total_bytes / B0


# ===========================================================================
# CORNER 1 — camera-res sub-pixel boundary placement
# ===========================================================================
def corner1_one_frame(segnet, L, frame_cam, args):
    """Measure CORNER 1 for one frame. Compares, all THROUGH the real SegNet authority:

      (a) flat@384 -> up->Q->down            (the old 384-grid path; baseline)
      (b) flat@CAMERA-RES (eval-native D)    (skip the up-then-down blur)
      (c) sub-pixel SOLVED camera-res frame  (gradient through D+SegNet anti-aliases the
                                              boundary band so D lands the argmax right)

    False-RED guard: report the BEST realized d_seg across (a),(b),(c) so an under-powered
    solve cannot fake a RED. The sub-pixel solve optimizes ONLY the camera-res frame
    (continuous) through D+SegNet CE vs L* — the realized d_seg of the HARD uint8 frame is
    the authority, the CE is a surrogate.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    band1 = _boundary_band_384(L, iters=1)
    cols, Lup = _per_class_mu_colors(L, frame_cam)

    # (a) flat @ 384 then up->Q->down
    flat_384 = cols[L]
    am_a = _segnet_argmax_via_384_roundtrip(segnet, flat_384)
    dseg_a = _d_seg(am_a, L)
    bnd_a = _band_flip(am_a, L, band1)

    # (b) flat @ camera-res (eval-native: just D + SegNet, no up)
    cam_flat = cols[Lup]
    am_b = _segnet_argmax_cam(segnet, cam_flat)
    dseg_b = _d_seg(am_b, L)
    bnd_b = _band_flip(am_b, L, band1)

    # (c) sub-pixel SOLVE: optimize the camera-res frame through D + SegNet.
    # init from the flat camera-res frame; the gradient anti-aliases the boundary band.
    tdev = torch.device(args.train_device)
    if args.train_device == "mps":
        try:
            from tac.torch_mps_compat import patch_scorer_for_mps

            patch_scorer_for_mps()
        except Exception:
            tdev = torch.device("cpu")
    seg_train = (
        segnet
        if args.train_device == "cpu"
        else _load_segnet_for_device(args.train_device)
    )

    Lt = torch.tensor(L, dtype=torch.long, device=tdev)
    # we optimize a camera-res RGB frame; to keep memory sane, optimize a residual on the
    # boundary band region only (interior is fixed flat = its mu, which is already optimal).
    cam0 = torch.tensor(cam_flat, dtype=torch.float32, device=tdev).permute(2, 0, 1)  # (3,Hc,Wc)
    # camera-res boundary band (dilate the 384 band, upsample NN, then dilate a bit at cam res)
    band_cam = _boundary_band_camera(L, iters_384=2, iters_cam=args.cam_band_dilate, device=tdev)
    delta = torch.zeros_like(cam0, requires_grad=True)
    opt = torch.optim.AdamW([delta], lr=args.lr, weight_decay=0.0)
    t0 = time.time()
    ce_first = ce_last = None
    for _it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        cam = (cam0 + band_cam * delta).clamp(0, 255)  # only the band moves
        # D = bilinear-down to (384,512) (the eval's preprocess), then SegNet logits
        down = F.interpolate(cam.unsqueeze(0), size=(MH, MW), mode="bilinear", align_corners=False)
        logits = _segnet_logits(seg_train, down[0])
        ce = F.cross_entropy(logits, Lt.unsqueeze(0))
        ce.backward()
        torch.nn.utils.clip_grad_norm_([delta], 50.0)
        opt.step()
        if ce_first is None:
            ce_first = float(ce.item())
        ce_last = float(ce.item())

    with torch.no_grad():
        cam_solved = (cam0 + band_cam * delta).clamp(0, 255).round()  # HARD uint8 camera frame
        cam_solved_np = cam_solved.permute(1, 2, 0).cpu().numpy()
    am_c = _segnet_argmax_cam(segnet, cam_solved_np)  # AUTHORITY on CPU
    dseg_c = _d_seg(am_c, L)
    bnd_c = _band_flip(am_c, L, band1)

    # byte cost of the sub-pixel boundary code: the contour boundary (914 B) + the band
    # residual delta. The delta lives only on the band; advisory packed size = quantized
    # band-residual bits (8b/px on band pixels) * entropy factor + the contour bytes.
    n_band_cam = int((band_cam[0] > 0).sum().item())
    band_residual_bits = n_band_cam * 3 * 8  # 3 channels, 8b, all band cam px
    packed = 0.30  # advisory entropy factor for a sparse smooth residual
    subpixel_extra_bytes = band_residual_bits * packed / 8.0
    per_frame_bytes = CONTOUR_BYTES_PER_FRAME + subpixel_extra_bytes

    best_dseg = min(dseg_a, dseg_b, dseg_c)
    best_label = ["flat_384_roundtrip", "flat_camera_native", "subpixel_solved"][
        int(np.argmin([dseg_a, dseg_b, dseg_c]))
    ]
    # S projections (full-vehicle, realized best d_seg + this corner's per-frame boundary rate)
    rate_full = rate_from_total_bytes(per_frame_bytes * 600.0)
    rate_amort = rate_from_total_bytes(
        CONTOUR_BYTES_PER_FRAME + subpixel_extra_bytes * 600.0 * 0.10 + CONTOUR_BYTES_PER_FRAME * 600.0 * 0.10
    )
    s_full = 100 * best_dseg + math.sqrt(10 * HELD_POSE) + rate_full
    s_amort = 100 * best_dseg + math.sqrt(10 * HELD_POSE) + rate_amort

    return {
        "dseg_flat_384_roundtrip": dseg_a,
        "dseg_flat_camera_native": dseg_b,
        "dseg_subpixel_solved": dseg_c,
        "bnd_flip_flat_384": bnd_a,
        "bnd_flip_camera_native": bnd_b,
        "bnd_flip_subpixel": bnd_c,
        "best_dseg": best_dseg,
        "best_path": best_label,
        "best_dseg_x_frontier": best_dseg / FRONTIER_DSEG,
        "n_band_cam_px": n_band_cam,
        "subpixel_extra_bytes": subpixel_extra_bytes,
        "per_frame_boundary_bytes": per_frame_bytes,
        "rate_full": rate_full,
        "rate_amortized": rate_amort,
        "S_projected_full": s_full,
        "S_projected_amortized": s_amort,
        "ce_first": ce_first,
        "ce_last": ce_last,
        "elapsed_s": round(time.time() - t0, 1),
    }


def _load_segnet_for_device(device):
    from tac.boundary_math.seg_core import _ensure_upstream_on_path

    _ensure_upstream_on_path()
    from modules import SegNet  # type: ignore
    from safetensors.torch import load_file

    ckpt = UPSTREAM / "models" / "segnet.safetensors"
    seg = SegNet().eval().to(device)
    seg.load_state_dict(load_file(str(ckpt), device=device))
    return seg


def _segnet_logits(segnet, frame_384_chw):
    """Differentiable SegNet logits (1,C,384,512) of a (3,384,512) float frame already at
    model res (the D-downsampled camera frame)."""
    import torch

    # build a degenerate pair (1,2,3,384,512); SegNet reads last frame, resizes (no-op at 384)
    pair = torch.stack([frame_384_chw, frame_384_chw], dim=0).unsqueeze(0)
    seg_in = segnet.preprocess_input(pair)
    return segnet(seg_in)


def _boundary_band_camera(L, iters_384, iters_cam, device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import ndimage

    band384 = _boundary_band_384(L, iters=iters_384)
    bt = torch.from_numpy(band384.astype(np.float32))[None, None]
    band_cam = F.interpolate(bt, size=(CAM_H, CAM_W), mode="nearest")[0, 0].numpy() > 0.5
    if iters_cam > 0:
        band_cam = ndimage.binary_dilation(band_cam, iterations=iters_cam)
    return torch.tensor(band_cam[None].astype(np.float32), device=device)  # (1,Hc,Wc)


def run_corner1(segnet, seg_cache, frames, args, state):
    rows = []
    n = min(args.n_frames, len(frames))
    for fidx in range(n):
        if fidx not in frames:
            continue
        key = f"c1_frame{fidx}"
        if key in state.get("c1_rows", {}):
            print(f"[resume] {key} done; skip")
            rows.append(state["c1_rows"][key])
            continue
        L = seg_cache[fidx]
        r = corner1_one_frame(segnet, L, frames[fidx], args)
        r["frame"] = fidx
        state.setdefault("c1_rows", {})[key] = r
        _save_state(state, args)
        print(
            f"  C1 frame{fidx}: flat384={r['dseg_flat_384_roundtrip']:.5f} "
            f"cam-native={r['dseg_flat_camera_native']:.5f} "
            f"subpixel={r['dseg_subpixel_solved']:.5f} "
            f"BEST={r['best_dseg']:.5f}({r['best_path']}) "
            f"{r['best_dseg_x_frontier']:.1f}x-frontier "
            f"bnd:[384={r['bnd_flip_flat_384']:.3f} cam={r['bnd_flip_camera_native']:.3f} "
            f"sub={r['bnd_flip_subpixel']:.3f}] {r['elapsed_s']:.0f}s"
        )
    return rows


# ===========================================================================
# CORNER 2 — cross-frame keyframe + tiny warp
# ===========================================================================
def _warp_label_map(L_keyframe_384, params, mode):
    """Apply a 2D geometric warp to the keyframe's label map at 384 (the d_seg-critical
    grid). mode in {translate, affine}. params is a small float vector. Returns warped L
    (nearest, label-preserving)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    Lt = torch.from_numpy(L_keyframe_384.astype(np.float32))[None, None]  # (1,1,H,W)
    if mode == "translate":
        dy, dx = params  # in pixels
        theta = torch.tensor(
            [[1.0, 0.0, 2.0 * dx / MW], [0.0, 1.0, 2.0 * dy / MH]], dtype=torch.float32
        )[None]
    elif mode == "affine":
        a, b, c, d, e, f = params  # 2x3 affine on normalized grid
        theta = torch.tensor([[a, b, c], [d, e, f]], dtype=torch.float32)[None]
    else:
        raise ValueError(mode)
    grid = F.affine_grid(theta, Lt.shape, align_corners=False)
    warped = F.grid_sample(Lt, grid, mode="nearest", padding_mode="border", align_corners=False)
    return warped[0, 0].numpy().astype(np.int64)


def _solve_warp(L_key, L_target, mode, args):
    """Find the warp params (translate or affine) minimizing argmax-diff of warped keyframe
    vs the target frame's L*. Coarse grid search (translate) / Nelder-Mead (affine) — closed
    over the COMBINATORIAL label-diff, $0, no scorer needed (this is a geometry solve)."""
    import numpy as np

    if mode == "none":
        return None, float((L_key != L_target).mean())
    if mode == "translate":
        best = (0.0, 0.0)
        best_d = float((L_key != L_target).mean())
        # coarse-to-fine pixel search in +/- range
        for rng, step in [(6.0, 1.0), (1.5, 0.25)]:
            cy, cx = best
            cand = [
                (cy + dy, cx + dx)
                for dy in np.arange(-rng, rng + 1e-9, step)
                for dx in np.arange(-rng, rng + 1e-9, step)
            ]
            for dy, dx in cand:
                w = _warp_label_map(L_key, (dy, dx), "translate")
                d = float((w != L_target).mean())
                if d < best_d:
                    best_d, best = d, (dy, dx)
        return best, best_d
    if mode == "affine":
        from scipy.optimize import minimize

        def loss(p):
            w = _warp_label_map(L_key, tuple(p), "affine")
            return float((w != L_target).mean())

        x0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": args.affine_maxiter, "xatol": 1e-4, "fatol": 1e-5})
        return tuple(res.x), float(res.fun)
    raise ValueError(mode)


def run_corner2(segnet, seg_cache, frames, args, state):
    """CORNER 2: keyframe boundary + per-frame tiny warp. Pay the boundary (914 B) ONCE,
    then a tiny per-frame warp code; measure per-frame realized d_seg of the WARPED keyframe
    PARTITION through the real SegNet (we render the warped partition with mu colors and
    measure realized d_seg vs each frame's own L*), and the TRUE amortized rate.

    NO-FAKE: realized d_seg through the real SegNet is authority. The warp is solved on the
    combinatorial label-diff (a geometry solve, $0); the realized number then runs the warped
    flat-painted partition through D+SegNet vs the target L*. We ALSO report the pure
    combinatorial warp-diff (the partition-level amortization) so the survival gap is visible.
    """
    import numpy as np

    n = min(args.n_frames, len(frames))
    if n < 2:
        return [{"note": "need >=2 frames for cross-frame corner", "n": n}]
    key_idx = 0
    L_key = seg_cache[key_idx]
    rows = []
    for fidx in range(1, n):
        if fidx not in frames:
            continue
        rkey = f"c2_frame{fidx}_{args.warp_mode}"
        if rkey in state.get("c2_rows", {}):
            print(f"[resume] {rkey} done; skip")
            rows.append(state["c2_rows"][rkey])
            continue
        t0 = time.time()
        L_tgt = seg_cache[fidx]
        # combinatorial baseline: keyframe with NO warp vs target
        d_nowarp = float((L_key != L_tgt).mean())
        params, d_warp_combinatorial = _solve_warp(L_key, L_tgt, args.warp_mode, args)
        # render the WARPED keyframe partition with mu colors of the TARGET frame, measure
        # realized d_seg through the real SegNet (the survival check).
        L_warped = L_key if params is None else _warp_label_map(L_key, params, args.warp_mode)
        cols_tgt, Lup_tgt = _per_class_mu_colors(L_tgt, frames[fidx])
        # paint the WARPED partition (camera-res) with target mu colors, measure realized d_seg
        import torch
        import torch.nn.functional as F

        Lw = torch.from_numpy(L_warped.astype(np.int64))[None, None].float()
        Lw_cam = F.interpolate(Lw, size=(CAM_H, CAM_W), mode="nearest")[0, 0].numpy().astype(np.int64)
        cam_painted = cols_tgt[Lw_cam]
        am = _segnet_argmax_cam(segnet, cam_painted)
        realized_dseg = _d_seg(am, L_tgt)

        # ALSO: the realized d_seg of the TARGET's OWN flat partition (per-frame-from-scratch
        # baseline) so we can see if the keyframe+warp AMORTIZATION is cheaper than per-frame.
        cam_own = cols_tgt[Lup_tgt]
        am_own = _segnet_argmax_cam(segnet, cam_own)
        realized_dseg_own = _d_seg(am_own, L_tgt)

        # TRUE amortized rate: keyframe boundary (914 B) shared once + per-frame warp code.
        warp_bytes = {"none": 0, "translate": 2 * 2, "affine": 6 * 2}[args.warp_mode]  # quantized
        per_frame_amort_bytes = CONTOUR_BYTES_PER_FRAME / 600.0 + warp_bytes
        rate_amort = rate_from_total_bytes(per_frame_amort_bytes * 600.0)
        # per-frame-from-scratch comparison rate (store the boundary EVERY frame)
        rate_per_frame = rate_from_total_bytes(CONTOUR_BYTES_PER_FRAME * 600.0)

        s_amort = 100 * realized_dseg + math.sqrt(10 * HELD_POSE) + rate_amort
        s_per_frame = 100 * realized_dseg_own + math.sqrt(10 * HELD_POSE) + rate_per_frame

        r = {
            "frame": fidx,
            "warp_mode": args.warp_mode,
            "warp_params": list(params) if params is not None else None,
            "d_nowarp_combinatorial": d_nowarp,
            "d_warp_combinatorial": d_warp_combinatorial,
            "realized_dseg_keyframe_warp": realized_dseg,
            "realized_dseg_own_flat": realized_dseg_own,
            "realized_dseg_kw_x_frontier": realized_dseg / FRONTIER_DSEG,
            "per_frame_amort_bytes": per_frame_amort_bytes,
            "rate_amortized": rate_amort,
            "rate_per_frame": rate_per_frame,
            "S_projected_amortized": s_amort,
            "S_projected_per_frame_own": s_per_frame,
            "elapsed_s": round(time.time() - t0, 1),
        }
        state.setdefault("c2_rows", {})[rkey] = r
        _save_state(state, args)
        print(
            f"  C2 frame{fidx}: nowarp_comb={d_nowarp:.5f} warp_comb={d_warp_combinatorial:.5f} "
            f"realized_kw={realized_dseg:.5f}({realized_dseg/FRONTIER_DSEG:.1f}x) "
            f"own_flat={realized_dseg_own:.5f} rate_amort={rate_amort:.5f} "
            f"S_amort~{s_amort:.3f} {r['elapsed_s']:.0f}s"
        )
        rows.append(r)
    return rows


# ===========================================================================
# state / verdict
# ===========================================================================
def _save_state(state, args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "gate_state.json").write_text(json.dumps(state, indent=2))


def _verdict_corner1(state):
    import numpy as np

    rows = list(state.get("c1_rows", {}).values())
    if not rows:
        return None
    best_dseg = min(r["best_dseg"] for r in rows)
    avg_best = float(np.mean([r["best_dseg"] for r in rows]))
    avg_subpixel = float(np.mean([r["dseg_subpixel_solved"] for r in rows]))
    avg_cam_native = float(np.mean([r["dseg_flat_camera_native"] for r in rows]))
    avg_flat384 = float(np.mean([r["dseg_flat_384_roundtrip"] for r in rows]))
    avg_S = float(np.mean([r["S_projected_amortized"] for r in rows]))
    # GREEN: best realized d_seg below the green threshold AND S projection < 0.15
    if best_dseg < GREEN_DSEG_THRESHOLD and avg_S < 0.15:
        v = "GREEN_C1_SUBPIXEL_BEATS_FRONTIER_DSEG_BYTE_CHEAP"
    elif avg_best < FRONTIER_DSEG:
        v = "AMBER_C1_BELOW_FRONTIER_DSEG_BUT_NOT_SUB015_OR_RATE"
    else:
        v = "RED_C1_SUBPIXEL_DOES_NOT_BEAT_FRONTIER_DSEG"
    return {
        "verdict": v,
        "best_dseg": best_dseg,
        "avg_best_dseg": avg_best,
        "avg_dseg_subpixel_solved": avg_subpixel,
        "avg_dseg_flat_camera_native": avg_cam_native,
        "avg_dseg_flat_384_roundtrip": avg_flat384,
        "avg_best_dseg_x_frontier": avg_best / FRONTIER_DSEG,
        "avg_S_projected_amortized": avg_S,
        "subpixel_helped_vs_384": avg_flat384 - avg_subpixel,
        "n_frames": len(rows),
    }


def _verdict_corner2(state):
    import numpy as np

    rows = [r for r in state.get("c2_rows", {}).values() if "realized_dseg_keyframe_warp" in r]
    if not rows:
        return None
    avg_kw = float(np.mean([r["realized_dseg_keyframe_warp"] for r in rows]))
    avg_own = float(np.mean([r["realized_dseg_own_flat"] for r in rows]))
    avg_warp_comb = float(np.mean([r["d_warp_combinatorial"] for r in rows]))
    avg_nowarp_comb = float(np.mean([r["d_nowarp_combinatorial"] for r in rows]))
    avg_S_amort = float(np.mean([r["S_projected_amortized"] for r in rows]))
    rate_amort = rows[0]["rate_amortized"]
    # GREEN: amortized keyframe+warp realized d_seg below green threshold AND S<0.15
    if avg_kw < GREEN_DSEG_THRESHOLD and avg_S_amort < 0.15:
        v = "GREEN_C2_KEYFRAME_WARP_AMORTIZES_DSEG_BYTE_CHEAP"
    elif avg_kw < FRONTIER_DSEG:
        v = "AMBER_C2_BELOW_FRONTIER_DSEG_BUT_NOT_SUB015"
    else:
        v = "RED_C2_KEYFRAME_WARP_DOES_NOT_AMORTIZE_DSEG_TO_FRONTIER"
    return {
        "verdict": v,
        "avg_realized_dseg_keyframe_warp": avg_kw,
        "avg_realized_dseg_own_flat": avg_own,
        "avg_realized_dseg_kw_x_frontier": avg_kw / FRONTIER_DSEG,
        "avg_warp_combinatorial_diff": avg_warp_comb,
        "avg_nowarp_combinatorial_diff": avg_nowarp_comb,
        "warp_closed_combinatorial_gap": avg_nowarp_comb - avg_warp_comb,
        "avg_S_projected_amortized": avg_S_amort,
        "rate_amortized": rate_amort,
        "amortization_helped_realized": avg_own - avg_kw,
        "n_frames": len(rows),
    }


def _write_result_json(state, args, n_frames):
    v1 = _verdict_corner1(state)
    v2 = _verdict_corner2(state)
    result = REPO / ".omx/research" / f"dseg_side_feasibility_corners_{_now()}.json"
    payload = {
        "schema": "dseg_side_feasibility_corners.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_dseg_side_feasibility_corners.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "the_question": (
            "Can the d_seg-critical boundary be coded CHEAPER than the frontier's effective "
            "d_seg contribution (frontier d_seg-term 0.056 @ d_seg=0.00056, near task-RD floor)? "
            "GREEN -> a piece of #155 that beats the frontier on d_seg-rate; RED -> doesn't."
        ),
        "frontier_bar": {
            "frontier_dseg": FRONTIER_DSEG,
            "frontier_dseg_term": FRONTIER_DSEG_TERM,
            "frontier_rate_term": FRONTIER_RATE_TERM,
            "green_dseg_threshold": GREEN_DSEG_THRESHOLD,
            "sub015_dseg": SUB015_DSEG,
            "held_pose": HELD_POSE,
            "contour_bytes_per_frame": CONTOUR_BYTES_PER_FRAME,
        },
        "method": {
            "corner1": "camera-res sub-pixel boundary placement: flat@384->up->Q->down vs "
            "flat@camera-res(eval-native D) vs sub-pixel SOLVED (gradient through D+SegNet "
            "anti-aliases the boundary band). realized d_seg of HARD uint8 frame through real "
            "SegNet = authority; report BEST of 3 (false-RED guard).",
            "corner2": "cross-frame keyframe + tiny warp: keyframe boundary (914B) once + "
            "per-frame warp (translate/affine, solved on combinatorial label-diff); realized "
            "d_seg of WARPED partition (target mu colors) through real SegNet vs target L* = "
            "authority; compared to per-frame-from-scratch own-flat realized d_seg.",
            "authority_device": "cpu",
            "train_device": args.train_device,
            "warp_mode": args.warp_mode,
            "n_frames": n_frames,
        },
        "corner1_verdict": v1,
        "corner2_verdict": v2,
        "c1_rows": state.get("c1_rows", {}),
        "c2_rows": state.get("c2_rows", {}),
        "overall_dseg_side_feasibility_for_155": _overall(v1, v2),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2))
    state["result_json"] = str(result.relative_to(REPO))
    state["corner1_verdict"] = v1["verdict"] if v1 else None
    state["corner2_verdict"] = v2["verdict"] if v2 else None
    _save_state(state, args)
    return result, v1, v2, payload["overall_dseg_side_feasibility_for_155"]


def _overall(v1, v2):
    g1 = v1 and v1["verdict"].startswith("GREEN")
    g2 = v2 and v2["verdict"].startswith("GREEN")
    a1 = v1 and v1["verdict"].startswith("AMBER")
    a2 = v2 and v2["verdict"].startswith("AMBER")
    if g1 or g2:
        return "GREEN_A_DSEG_CORE_CORNER_BEATS_THE_FRONTIER"
    if a1 or a2:
        return "AMBER_A_CORNER_BELOW_FRONTIER_DSEG_BUT_NOT_SUB015_BYTE_CHEAP"
    return "RED_NEITHER_CORNER_BEATS_THE_FRONTIER_DSEG_CODING"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", default=str(REPO / "experiments/results/dseg_side_feasibility_corners")
    )
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--iters", type=int, default=200, help="sub-pixel solve iters (corner 1)")
    ap.add_argument("--lr", type=float, default=8.0, help="AdamW lr for sub-pixel solve (0-255)")
    ap.add_argument("--cam-band-dilate", type=int, default=2, help="extra camera-res band dilate")
    ap.add_argument("--warp-mode", default="affine", choices=["none", "translate", "affine"])
    ap.add_argument("--affine-maxiter", type=int, default=400)
    ap.add_argument("--train-device", default="cpu", choices=["cpu", "mps"],
                    help="device for the sub-pixel GRADIENT only; authority is ALWAYS cpu")
    ap.add_argument("--corners", default="1,2", help="which corners to run (1,2)")
    ap.add_argument("--timing-smoke", action="store_true",
                    help="1 frame corner-1 + 1 frame corner-2 with reduced iters, then exit")
    ap.add_argument("--verdict-only", action="store_true",
                    help="recompute verdict from existing gate_state.json rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    state_path = out_dir / "gate_state.json"
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] loaded prior state with c1={len(state.get('c1_rows',{}))} "
              f"c2={len(state.get('c2_rows',{}))} rows")

    if args.verdict_only:
        if not state:
            print("[verdict-only] no gate_state.json")
            return 1
        result, v1, v2, overall = _write_result_json(state, args, args.n_frames)
        _print_summary(v1, v2, overall, result)
        return 0

    segnet = _load_segnet_cpu()
    seg_cache = _load_gt_cache()

    if args.timing_smoke:
        print("[timing-smoke] corner1 1 frame x 30 iters + corner2 1 frame ...")
        sa = argparse.Namespace(**vars(args))
        sa.iters = 30
        sa.n_frames = 2
        frames = _decode_gt_camera_frames(2)
        t0 = time.time()
        r1 = corner1_one_frame(segnet, seg_cache[0], frames[0], sa)
        print(f"  C1: best={r1['best_dseg']:.5f}({r1['best_path']}) subpixel={r1['dseg_subpixel_solved']:.5f} "
              f"{r1['elapsed_s']:.0f}s ; ~{r1['elapsed_s']/sa.iters:.2f}s/it")
        params, dc = _solve_warp(seg_cache[0], seg_cache[1], sa.warp_mode, sa)
        print(f"  C2 warp solve: combinatorial diff {dc:.5f} in {time.time()-t0:.0f}s total")
        return 0

    frames = _decode_gt_camera_frames(args.n_frames)
    corners = {x.strip() for x in args.corners.split(",")}
    if "1" in corners:
        print("\n=== CORNER 1 — camera-res sub-pixel boundary placement ===")
        run_corner1(segnet, seg_cache, frames, args, state)
    if "2" in corners:
        print("\n=== CORNER 2 — cross-frame keyframe + tiny warp ===")
        run_corner2(segnet, seg_cache, frames, args, state)

    result, v1, v2, overall = _write_result_json(state, args, args.n_frames)
    _print_summary(v1, v2, overall, result)
    return 0


def _print_summary(v1, v2, overall, result):
    print(f"\n[done] advisory JSON -> {result.relative_to(REPO)}")
    if v1:
        print(f"[CORNER 1] {v1['verdict']}")
        print(f"  avg best d_seg {v1['avg_best_dseg']:.5f} ({v1['avg_best_dseg_x_frontier']:.1f}x frontier); "
              f"subpixel helped vs 384 by {v1['subpixel_helped_vs_384']:+.5f}; avg S~{v1['avg_S_projected_amortized']:.3f}")
    if v2:
        print(f"[CORNER 2] {v2['verdict']}")
        print(f"  avg realized d_seg (keyframe+warp) {v2['avg_realized_dseg_keyframe_warp']:.5f} "
              f"({v2['avg_realized_dseg_kw_x_frontier']:.1f}x frontier); warp closed combinatorial gap "
              f"{v2['warp_closed_combinatorial_gap']:+.5f}; amortization helped realized "
              f"{v2['amortization_helped_realized']:+.5f}; avg S~{v2['avg_S_projected_amortized']:.3f}")
    print(f"[OVERALL d_seg-side feasibility for #155] {overall}")
    print("[contest-CPU advisory] NON-PROMOTABLE. Exact pointer UNMOVED at 0.19110.")


if __name__ == "__main__":
    raise SystemExit(main())
