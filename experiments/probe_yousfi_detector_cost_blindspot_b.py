# SPDX-License-Identifier: MIT
"""YOUSFI BLIND-SPOT PROBE B — is ``d_seg cheap by construction`` a real lever?

Inverse-steganalysis lens (Yousfi 2022 / Holub-Fridrich 2014 S-UNIWARD): the
contest's ``d_seg`` is the output of a DETECTOR (EfficientNet-B2 SegNet,
argmax-flip rate). Rather than grind ``d_seg`` down with days of gradient
descent, put the renderer's reconstruction error WHERE THE DETECTOR IS BLIND
(high-margin / high-texture pixels — UNIWARD principle #1) and AVOID spending it
where the detector is SHARP (low-margin class boundaries) — keeping the
quantization error inside the per-pixel margin-polytope INTERIOR so the argmax
never flips.

This probe MEASURES (no claims the code does not honor) on the REAL basin:

  M1. DETECTABILITY COST MAP — per-pixel SegNet top1-top2 margin (distance to
      flip, the detector's SHARPNESS) + the S-UNIWARD inverse-texture cost (the
      embedding capacity). Where is SegNet BLIND vs SHARP?
  M2. ERROR-VS-COST ALLOCATION — does the basin decoder's reconstruction error
      land wastefully in SHARP (low-margin) regions, vs the BLIND regions where
      it is free? (Pearson + binned profile of |error| vs margin.)
  M3. ARGMAX-CELL FREE BUDGET — of the basin's ACTUAL d_seg flips, what fraction
      are AVOIDABLE — i.e. flips that occurred at pixels whose GT-side margin was
      DEEP-INTERIOR (large slack), so a smaller / better-placed error would have
      stayed in-cell and kept d_seg=0 there. Quantifies the "wasted" d_seg.
  M4. $0 SMOKE — a cost-weighted seg loss (down-weight error in BLIND pixels, up-
      weight near boundaries) + a polytope-interior regularizer (penalize error
      that crosses the GT argmax cell). Does allocating error by the cost map +
      staying in-cell reduce d_seg vs the basin's plain reconstruction, on a few
      pairs? (Compress-time delta optimized on rendered frames, scored by the
      REAL SegNet argmax-flip.)

AUTHORITY: ``[contest-CPU advisory] / [macOS-CPU advisory]`` — NON-PROMOTABLE.
The only authoritative ``d_seg`` is ``upstream/evaluate.py`` on a byte-closed
archive. This probe's numbers are a DIAGNOSTIC of where the lever lives, not a
score claim. CPU-only (MPS owns the live train; MPS would 2x-corrupt SegNet).

NO FAKE: real frozen SegNet (``load_frozen_distortion_net(device='cpu')``), real
basin archive (parse-back of ``best/best_archive.bin`` — the contest-visible
bytes), real GT via ``frame_utils.yuv420_to_rgb`` (PyAV rgb24 manufactures
phantom pose; never used). The d_seg here is the EXACT vendored pipeline:
native(384x512) -> bicubic camera(874x1164) -> uint8 -> SegNet preprocess
(bilinear 384x512) -> argmax-disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ── exact vendored constants (score.py) ─────────────────────────────────────
CAMERA_H, CAMERA_W = 874, 1164
EVAL_H, EVAL_W = 384, 512

BASIN_DIR = "experiments/results/torch_vehicle_full_mps_basin_bc20_n600"
VIDEO_PATH = "upstream/videos/0.mkv"


# ─────────────────────────────────────────────────────────────────────────────
# Real-input plumbing (parse-back basin decoder + frozen SegNet + GT decode)
# ─────────────────────────────────────────────────────────────────────────────
def _load_basin_decoder(basin_dir: str):
    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    model = import_vendored("model")
    arch = Path(basin_dir, "best", "best_archive.bin").read_bytes()
    dec_sd, latents, meta = codec.parse_archive(arch)
    dec = model.HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    )
    dec.load_state_dict(dec_sd)
    dec.eval()
    return dec, latents.float(), meta, len(arch)


def _decode_gt_pairs(n_pairs: int):
    """Stream the first ``n_pairs`` non-overlapping GT pairs (camera-res uint8).

    Exact vendored pairing (seq_len=2, non-overlapping) + the MANDATORY
    ``frame_utils.yuv420_to_rgb`` decode (CLAUDE.md non-negotiable).
    """
    from tac.torch_vehicle.vendored_imports import resolve_challenge_root

    root = resolve_challenge_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import av
    from frame_utils import yuv420_to_rgb

    cont = av.open(VIDEO_PATH)
    pairs: list[torch.Tensor] = []
    prev = None
    for fr in cont.decode(cont.streams.video[0]):
        f = yuv420_to_rgb(fr)
        if prev is None:
            prev = f
            continue
        pairs.append(torch.stack([prev, f]))  # (2, H, W, 3) uint8 camera-res
        prev = None
        if len(pairs) >= n_pairs:
            break
    cont.close()
    return pairs


def _decoded_to_camera(decoded_native: torch.Tensor) -> torch.Tensor:
    """(N, 3, EVAL_H, EVAL_W) -> (N, 3, CAMERA_H, CAMERA_W) bicubic (vendored)."""
    return F.interpolate(
        decoded_native, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
    )


def _camera_to_uint8_lastframe(frame_chw_camera: torch.Tensor) -> torch.Tensor:
    """One camera-res f1 (3, 874, 1164) float -> the SegNet uint8 round-trip
    that the contest sees, returned as (1, 2, H, W, 3) uint8 for preprocess_input
    (SegNet uses only the last frame so we duplicate)."""
    u = frame_chw_camera.clamp(0, 255).round().to(torch.uint8)  # (3, 874, 1164)
    hwc = u.permute(1, 2, 0)  # (874, 1164, 3)
    return torch.stack([hwc, hwc]).unsqueeze(0)  # (1, 2, 874, 1164, 3)


def _segnet_logits_from_camera_f1(segnet, frame_chw_camera: torch.Tensor) -> torch.Tensor:
    """Run the EXACT vendored SegNet pipeline on a single camera-res f1 frame.

    camera(874x1164) -> uint8 -> einops b t h w c -> b t c h w -> preprocess_input
    (use last frame, bilinear -> 384x512) -> SegNet -> (1, 5, 384, 512) logits.
    """
    import einops

    pair_uint = _camera_to_uint8_lastframe(frame_chw_camera).float()  # (1,2,H,W,3)
    x = einops.rearrange(pair_uint, "b t h w c -> b t c h w")
    seg_in = segnet.preprocess_input(x)  # (1, 3, 384, 512)
    with torch.inference_mode():
        return segnet(seg_in)  # (1, 5, 384, 512)


def _argmax_margin(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    top2 = torch.topk(logits, 2, dim=1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)[0]
    am = logits.argmax(1)[0]
    return am.cpu().numpy().astype(np.int64), margin.cpu().numpy().astype(np.float64)


def _uniward_cost_camera_to_seg(frame_chw_camera: torch.Tensor) -> np.ndarray:
    """S-UNIWARD inverse-texture cost (high=textured=BLIND/safe) on the camera-res
    f1, resized to the SegNet decision grid (384x512) so it aligns 1:1 with the
    margin map. Reuses ``tac.uniward_delta.compute_uniward_cost_map`` (real
    directional-Haar wavelet math; NO duplication)."""
    from tac.uniward_delta import compute_uniward_cost_map

    fr = frame_chw_camera.unsqueeze(0)  # (1, 3, 874, 1164) in [0,255]
    cost_cam = compute_uniward_cost_map(fr)  # (1, 874, 1164)
    cost_seg = F.interpolate(
        cost_cam.unsqueeze(1), size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False
    )[0, 0]
    return cost_seg.cpu().numpy().astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair core: render basin f1, measure rendered/GT argmax+margin+cost
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _PairFields:
    rendered_argmax: np.ndarray  # (384, 512) int64 — basin f1 SegNet argmax
    gt_argmax: np.ndarray
    gt_margin: np.ndarray  # (384, 512) — GT-side detector sharpness (distance to flip)
    rendered_margin: np.ndarray
    gt_cost: np.ndarray  # (384, 512) — S-UNIWARD capacity (high=textured=blind)
    seg_error: np.ndarray  # (384, 512) — |rendered_f1 - gt_f1| mean over RGB, in SegNet grid
    flip: np.ndarray  # (384, 512) bool — rendered argmax != gt argmax (the d_seg flips)


def _seg_grid_error(rendered_cam_f1: torch.Tensor, gt_cam_f1: torch.Tensor) -> np.ndarray:
    """Mean-abs reconstruction error, downsampled to the SegNet 384x512 grid.

    Both inputs are camera-res (3, 874, 1164) float [0,255]. We bilinear-resize
    the per-pixel |Δ| to (384, 512) so it lives in the same grid the margin/cost
    maps do (the grid where the detector's argmax actually decides)."""
    err = (rendered_cam_f1 - gt_cam_f1).abs().mean(0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
    err_seg = F.interpolate(err, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
    return err_seg[0, 0].cpu().numpy().astype(np.float64)


def _measure_pair(dec, latents, segnet, gt_pair, pair_idx: int) -> _PairFields:
    with torch.inference_mode():
        out = dec(latents[pair_idx : pair_idx + 1])  # (1, 2, 3, 384, 512)
    native_f1 = out[0, 1]  # (3, 384, 512) — the LAST frame SegNet scores
    cam_f1 = _decoded_to_camera(native_f1.unsqueeze(0))[0]  # (3, 874, 1164)

    gt_cam_f1 = gt_pair[1].permute(2, 0, 1).float()  # (3, 874, 1164) from uint8

    r_logits = _segnet_logits_from_camera_f1(segnet, cam_f1)
    g_logits = _segnet_logits_from_camera_f1(segnet, gt_cam_f1)
    r_am, r_mg = _argmax_margin(r_logits)
    g_am, g_mg = _argmax_margin(g_logits)

    cost = _uniward_cost_camera_to_seg(gt_cam_f1)
    err = _seg_grid_error(cam_f1, gt_cam_f1)
    flip = r_am != g_am
    return _PairFields(
        rendered_argmax=r_am, gt_argmax=g_am, gt_margin=g_mg, rendered_margin=r_mg,
        gt_cost=cost, seg_error=err, flip=flip,
    )


# ─────────────────────────────────────────────────────────────────────────────
# M1-M3 aggregation
# ─────────────────────────────────────────────────────────────────────────────
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _quantile_bins(x: np.ndarray, y: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Bin ``y`` by quantiles of ``x``; return mean(y) per bin (a monotone-check
    profile)."""
    x = x.ravel()
    y = y.ravel()
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        sel = (x >= lo) & (x <= hi) if i == n_bins - 1 else (x >= lo) & (x < hi)
        if sel.sum() == 0:
            out.append({"bin": i, "x_lo": float(lo), "x_hi": float(hi),
                        "y_mean": 0.0, "n": 0})
            continue
        out.append({"bin": i, "x_lo": float(lo), "x_hi": float(hi),
                    "y_mean": float(y[sel].mean()), "n": int(sel.sum())})
    return out


def aggregate(fields: list[_PairFields], boundary_margin: float = 0.5) -> dict:
    margins = np.concatenate([f.gt_margin.ravel() for f in fields])
    costs = np.concatenate([f.gt_cost.ravel() for f in fields])
    errs = np.concatenate([f.seg_error.ravel() for f in fields])
    flips = np.concatenate([f.flip.ravel() for f in fields])

    # ── M1: detectability cost map stats ────────────────────────────────────
    boundary = margins < boundary_margin  # SHARP — detector decisive, small slack
    cost_med = float(np.median(costs))
    blind = costs >= cost_med  # high texture = high UNIWARD capacity = BLIND
    m1 = {
        "n_pixels": int(margins.size),
        "margin_mean": float(margins.mean()),
        "margin_median": float(np.median(margins)),
        "frac_boundary_margin_lt_0p5": float(boundary.mean()),
        "frac_boundary_margin_lt_1p0": float((margins < 1.0).mean()),
        "uniward_cost_mean": float(costs.mean()),
        "uniward_cost_median": cost_med,
        "frac_blind_high_texture": float(blind.mean()),
        # do BLIND (textured) pixels also tend to be detector-confident (high margin)?
        "pearson_cost_vs_margin": _pearson(costs, margins),
    }

    # ── M2: error-vs-cost allocation ────────────────────────────────────────
    # FINDING (measured): this basin renderer is a DETECTOR-MATCHER, not an image
    # reconstructor — its raw RGB error to GT is ~104/255 yet d_seg is 0.0026
    # (the frames look nothing like GT in pixels but their SegNet argmax matches).
    # So "raw recon error vs cost" is NOT the lever surface; the lever surface is
    # WHERE THE FLIPS LAND vs the cost/margin maps. We report BOTH: (a) the raw
    # recon-error profile (for completeness; expected to be ~flat = no recon
    # signal) and (b) the FLIP-DENSITY profile (the d_seg-relevant allocation).
    err_in_boundary = float(errs[boundary].mean()) if boundary.any() else 0.0
    err_in_interior = float(errs[~boundary].mean()) if (~boundary).any() else 0.0
    # Flip density (the actual d_seg allocation): fraction of pixels in each
    # region that flip. If flips concentrate in SHARP/boundary pixels, the d_seg
    # is detector-intrinsic, not a wasteful-error-placement artifact.
    flip_rate_boundary = float(flips[boundary].mean()) if boundary.any() else 0.0
    flip_rate_interior = float(flips[~boundary].mean()) if (~boundary).any() else 0.0
    flip_rate_blind = float(flips[blind].mean()) if blind.any() else 0.0
    flip_rate_sharp_texture = float(flips[~blind].mean()) if (~blind).any() else 0.0
    m2 = {
        "_finding": (
            "basin renderer is a DETECTOR-MATCHER not a reconstructor "
            "(raw RGB err ~104/255 with d_seg 0.0026); flip-density is the "
            "d_seg-relevant allocation, not raw recon error"
        ),
        # (a) raw recon error (expected ~flat = renderer carries no recon signal)
        "raw_recon_error_mean": float(errs.mean()),
        "pearson_recon_error_vs_margin": _pearson(errs, margins),
        "pearson_recon_error_vs_uniward_cost": _pearson(errs, costs),
        "mean_recon_error_boundary_margin_lt_0p5": err_in_boundary,
        "mean_recon_error_interior_margin_ge_0p5": err_in_interior,
        "recon_error_by_margin_quintile": _quantile_bins(margins, errs, 5),
        # (b) FLIP-density allocation (the d_seg lever surface)
        "flip_rate_boundary_margin_lt_0p5": flip_rate_boundary,
        "flip_rate_interior_margin_ge_0p5": flip_rate_interior,
        "flip_rate_boundary_over_interior_ratio": (
            flip_rate_boundary / flip_rate_interior if flip_rate_interior > 1e-12 else float("inf")
        ),
        "flip_rate_blind_high_texture": flip_rate_blind,
        "flip_rate_sharp_low_texture": flip_rate_sharp_texture,
        "flip_density_by_margin_quintile": _quantile_bins(margins, flips.astype(np.float64), 5),
        "flip_density_by_uniward_cost_quintile": _quantile_bins(costs, flips.astype(np.float64), 5),
    }

    # ── M3: avoidable d_seg (cell free budget) ──────────────────────────────
    # A flip is AVOIDABLE-by-staying-interior when the GT-side margin at that pixel
    # was DEEP INTERIOR (large slack): the cell had room, so a smaller / better-
    # placed error would have stayed in-cell and kept d_seg=0 there. A flip at a
    # tiny-GT-margin pixel is HARD (already on a class wall; even tiny error flips).
    total_flips = int(flips.sum())
    d_seg_pixelrate = float(flips.mean())  # ~ the per-pixel d_seg over these pairs
    flip_margins = margins[flips]
    # interior threshold = overall median margin (deep-interior = above median)
    interior_thr = float(np.median(margins))
    avoidable = flip_margins >= interior_thr
    # also a stricter "deep interior" = top-quartile margin
    deep_thr = float(np.quantile(margins, 0.75))
    deep_avoidable = flip_margins >= deep_thr
    m3 = {
        "total_flips": total_flips,
        "d_seg_pixelrate_advisory": d_seg_pixelrate,
        "interior_margin_threshold_median": interior_thr,
        "deep_interior_threshold_q75": deep_thr,
        "frac_flips_avoidable_interior": (
            float(avoidable.mean()) if total_flips else 0.0
        ),
        "frac_flips_deep_interior_avoidable": (
            float(deep_avoidable.mean()) if total_flips else 0.0
        ),
        "frac_flips_hard_boundary_lt_0p5": (
            float((flip_margins < boundary_margin).mean()) if total_flips else 0.0
        ),
        "flip_margin_mean": float(flip_margins.mean()) if total_flips else 0.0,
        "flip_margin_median": float(np.median(flip_margins)) if total_flips else 0.0,
        # interpretation: avoidable-interior flips are "wasted" d_seg — error that
        # crossed a cell with slack to spare. Reallocating that error into the
        # BLIND budget (or shrinking it) recovers them at ~0 byte cost.
        "wasted_d_seg_fraction": (
            float(avoidable.mean()) if total_flips else 0.0
        ),
    }
    return {"m1_cost_map": m1, "m2_error_vs_cost": m2, "m3_avoidable_d_seg": m3}


# ─────────────────────────────────────────────────────────────────────────────
# M4: $0 cost-weighted + interior-regularized seg-error smoke
# ─────────────────────────────────────────────────────────────────────────────
def _d_seg_for_native_f1(segnet, native_f1: torch.Tensor, gt_am: np.ndarray) -> float:
    """Exact-pipeline d_seg (argmax-flip rate vs gt_am) for a native (3,384,512)
    f1 candidate. Used to SCORE the smoke (real SegNet, real flip rate)."""
    cam = _decoded_to_camera(native_f1.unsqueeze(0))[0]
    logits = _segnet_logits_from_camera_f1(segnet, cam)
    am, _ = _argmax_margin(logits)
    return float((am != gt_am).mean())


def run_smoke(
    dec, latents, segnet, gt_pairs, fields: list[_PairFields], *,
    n_smoke_pairs: int, steps: int, l_inf: float, lr: float, seed: int,
) -> dict:
    """Optimize a compress-time additive delta on the rendered NATIVE f1 under THREE
    regimes and compare the resulting REAL d_seg (exact SegNet argmax-flip):

      A. NAIVE recon-to-GT descent (uniform |cand - GT| over RGB). The "just match
         the GT image" lever. EXPECTED to HURT d_seg here: the basin is a
         detector-matcher whose frames are RGB-far from GT, so pulling toward GT
         RGB breaks the detector match (M2 finding). A is the honest baseline.
      B. INTERIOR-ONLY (in-cell) descent: cross-entropy of the rendered f1 SegNet
         logits toward the GT argmax. This is the polytope-interior objective —
         drive the argmax back into the GT cell. No recon term. This is the
         direct "stay in-cell / repair the flip" lever.
      C. COST-WEIGHTED interior: same in-cell CE but the per-pixel CE is weighted
         toward BOUNDARY (sharp, low-margin) pixels and away from BLIND (textured)
         ones — the Yousfi cost-map allocation. Tests whether ALLOCATING the
         in-cell effort by detector-cost beats the uniform in-cell effort.

    All regimes share L_inf / steps / lr (apples-to-apples). The delta is a pure
    additive carrier (Yousfi PR#35: no scorer at inflate). NOTE: this is a
    COMPRESS-TIME per-pair delta on the rendered frame; it is NOT byte-closed and
    NOT a score claim — it measures whether the LEVER moves the real argmax-flip
    d_seg at fixed (here zero) byte cost, advisory only.
    """
    import einops

    torch.manual_seed(seed)
    boundary_margin = 0.5
    results_baseline: list[float] = []
    results_A: list[float] = []
    results_B: list[float] = []
    results_C: list[float] = []

    def _seg_logits_with_grad(native_f1: torch.Tensor) -> torch.Tensor:
        cam = _decoded_to_camera(native_f1.unsqueeze(0))[0]
        u = cam.clamp(0, 255)  # keep grad (no round in the surrogate)
        hwc = u.permute(1, 2, 0)
        pair = torch.stack([hwc, hwc]).unsqueeze(0)
        x = einops.rearrange(pair, "b t h w c -> b t c h w")
        seg_in = segnet.preprocess_input(x)
        return segnet(seg_in)  # (1, 5, 384, 512) WITH grad

    for k in range(min(n_smoke_pairs, len(fields))):
        f = fields[k]
        gt_am = torch.from_numpy(f.gt_argmax).long()  # (384, 512)
        gt_cam_f1 = gt_pairs[k][1].permute(2, 0, 1).float()  # (3, 874, 1164)
        gt_seg_target = F.interpolate(  # GT f1 at native grid (the recon target)
            gt_cam_f1.unsqueeze(0), size=(EVAL_H, EVAL_W), mode="bilinear",
            align_corners=False,
        )[0]

        with torch.inference_mode():
            base_native = dec(latents[k : k + 1])[0, 1].clone()  # (3, 384, 512)
        results_baseline.append(_d_seg_for_native_f1(segnet, base_native, f.gt_argmax))

        # Cost-map weight (SegNet grid): high near boundary (sharp), low at BLIND.
        margin = torch.from_numpy(f.gt_margin).float()
        cost = torch.from_numpy(f.gt_cost).float()
        cost_n = cost / cost.max().clamp_min(1e-8)
        w_boundary = torch.exp(-margin / boundary_margin)  # ~1 at the wall
        w_cost = (1.0 - cost_n).clamp_min(0.05)
        seg_weight = (w_boundary * w_cost)
        seg_weight = seg_weight / seg_weight.mean().clamp_min(1e-8)  # (384, 512)

        def optimize(mode: str) -> float:
            delta = torch.zeros_like(base_native, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                cand = base_native + delta
                if mode == "A":  # naive recon-to-GT
                    loss = (cand - gt_seg_target).abs().mean()
                elif mode == "B":  # interior-only CE (uniform)
                    logits = _seg_logits_with_grad(cand)
                    loss = F.cross_entropy(logits, gt_am.unsqueeze(0))
                else:  # mode == "C": cost-weighted interior CE
                    logits = _seg_logits_with_grad(cand)
                    ce_px = F.cross_entropy(logits, gt_am.unsqueeze(0), reduction="none")[0]
                    loss = (ce_px * seg_weight).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    delta.clamp_(-l_inf, l_inf)
            with torch.no_grad():
                cand = (base_native + delta).clamp(0, 255)
            return _d_seg_for_native_f1(segnet, cand, f.gt_argmax)

        results_A.append(optimize("A"))
        results_B.append(optimize("B"))
        results_C.append(optimize("C"))

    base = float(np.mean(results_baseline))
    a, b, c = float(np.mean(results_A)), float(np.mean(results_B)), float(np.mean(results_C))
    return {
        "n_smoke_pairs": int(min(n_smoke_pairs, len(fields))),
        "steps": steps, "l_inf_budget": l_inf, "lr": lr,
        "d_seg_basin_baseline": base,
        "d_seg_after_naive_recon_to_gt_A": a,
        "d_seg_after_interior_ce_uniform_B": b,
        "d_seg_after_interior_ce_cost_weighted_C": c,
        "delta_d_seg_A_minus_baseline": a - base,
        "delta_d_seg_B_minus_baseline": b - base,
        "delta_d_seg_C_minus_baseline": c - base,
        "delta_d_seg_C_minus_B": c - b,
        "pct_d_seg_reduction_best_interior": (
            100.0 * (base - min(b, c)) / base if base > 1e-12 else 0.0
        ),
        "note": (
            "All d_seg are EXACT SegNet argmax-flip rates on the real pipeline. "
            "A (recon-to-GT) HURTING d_seg confirms the detector-matcher finding. "
            "B/C < baseline => the in-cell (interior) lever reduces d_seg at ~0 "
            "byte cost; C < B => cost-map allocation adds value beyond uniform "
            "in-cell effort. NON-PROMOTABLE advisory (not byte-closed)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin-dir", default=BASIN_DIR)
    ap.add_argument("--n-pairs", type=int, default=24,
                    help="pairs for M1-M3 cost-map / allocation / avoidable-d_seg stats")
    ap.add_argument("--n-smoke-pairs", type=int, default=4,
                    help="pairs for the M4 paired delta-optimization smoke")
    ap.add_argument("--smoke-steps", type=int, default=60)
    ap.add_argument("--l-inf", type=float, default=6.0, help="L_inf delta budget [0,255]")
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():  # CPU-only by design; never MPS/CUDA scorer drift here
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    dec, latents, meta, archive_bytes = _load_basin_decoder(args.basin_dir)
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    net = load_frozen_distortion_net(device="cpu")
    segnet = net.segnet

    n = min(args.n_pairs, latents.shape[0])
    gt_pairs = _decode_gt_pairs(n)
    n = min(n, len(gt_pairs))

    fields = [_measure_pair(dec, latents, segnet, gt_pairs[i], i) for i in range(n)]
    agg = aggregate(fields)

    smoke = None
    if not args.skip_smoke:
        smoke = run_smoke(
            dec, latents, segnet, gt_pairs, fields,
            n_smoke_pairs=args.n_smoke_pairs, steps=args.smoke_steps,
            l_inf=args.l_inf, lr=args.lr, seed=args.seed,
        )

    report = {
        "probe": "yousfi_detector_cost_blindspot_b",
        "authority": "[contest-CPU advisory] / [macOS-CPU advisory] — NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "basin_dir": args.basin_dir,
        "basin_meta": meta,
        "basin_archive_bytes": archive_bytes,
        "n_pairs_measured": n,
        "wall_clock_s": round(time.time() - t0, 2),
        **agg,
        "m4_smoke": smoke,
    }

    print(json.dumps(report, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\n[wrote] {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
