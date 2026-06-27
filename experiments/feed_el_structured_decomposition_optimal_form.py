"""FEED-el: structured-decomposition OPTIMAL-FORM re-open ($0 CPU-only, realized-through-R).

Catalog #307 re-open: the FEED-ef/ek structured-prior NEGATIVE (realized 0.586) was a NAIVE
impl (phi init'd to the structured partition but out_tex RANDOM -> SegNet read garbage). The
PARADIGM (render the deterministic static core FREE, spend learned capacity only on the lane+
movable residual) is intact. This measures the OPTIMAL FORM as a DECOMPOSITION (not an init):

  static core  = {road, sky, hood} (+ optional static lane)  -- DETERMINISTIC, rule-118 FREE
  residual     = {lane, movable}                              -- the learned hard residual
  partition    = static_core  COMPOSE  residual   -> ONE argmax -> realized d_seg

The crux: SegNet is trained on REAL textured driving images, so a FLAT-color render has a
realized floor (SegNet misreads flat regions). We MEASURE which static-core render is SegNet-
LEGIBLE cheaply: flat-generic (free) / flat-gtmean (tiny counted) / deterministic-realistic
texture (free) / real-GT-pixels (ideal-texture upper bound).

Authority = realized-through-R: paint partition -> R [bicubic up 384x512->874x1164, uint8] ->
frozen CPU-torch SegNet.preprocess (bilinear down to scorer) -> argmax -> flip-rate vs L*.
NEVER MLX, NEVER MPS. The L* cache (gt_n96.npz) is the EXACT frozen CPU-torch authority.

$0 CPU-only, additive/default-off, standalone (no import of the MLX trainer). Does NOT touch
the running GPU row. pointer UNMOVED 0.19110; advisory training-signal / geometry measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# CPU-ONLY, NEVER MPS/CUDA. Set before torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512  # scorer / render / lstar resolution

# Generic FREE palette by comma10k role (rule-118: generic priors, 0 video-derived bytes).
GENERIC_PALETTE = {
    "road": (95, 95, 98),       # asphalt gray
    "lane": (235, 235, 235),    # white markings
    "sky": (180, 195, 215),     # light blue-gray undrivable/sky
    "movable": (120, 90, 80),   # generic car
    "hood": (28, 28, 30),       # dark ego hood
}


def _torch_R_to_camera_uint8(rgb_render_np: np.ndarray):
    """TORCH-authority R: render-res float RGB (SEG_H,SEG_W,3) -> bicubic up to CAMERA -> uint8.
    Op-identical to train_*_realized_through_R_mlx._torch_R_to_camera_uint8."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(rgb_render_np)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def _cpu_realized_argmax(segnet, frame1_uint8_camera: np.ndarray) -> np.ndarray:
    """Frozen CPU-torch SegNet realized argmax (384x512) on a CAMERA uint8 frame.
    Op-identical to cpu_verdict_d_seg's forward (SegNet uses only the last frame)."""
    import torch

    r = np.asarray(frame1_uint8_camera)
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1,2,H,W,3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(xp)
        logits = segnet(seg_in)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)


def d_seg_from_argmax(realized: np.ndarray, lstar: np.ndarray) -> float:
    return float(np.count_nonzero(realized != lstar)) / lstar.size


def paint_flat(part: np.ndarray, palette_by_class: dict[int, tuple]) -> np.ndarray:
    """Paint an int partition (H,W) -> (H,W,3) float32 with a per-class flat color."""
    h, w = part.shape
    out = np.zeros((h, w, 3), np.float32)
    for c, col in palette_by_class.items():
        out[part == c] = np.asarray(col, np.float32)
    return out


def paint_det_realistic(part: np.ndarray, roles) -> np.ndarray:
    """Deterministic FREE generic texture (rule-118): sky vertical gradient, road asphalt
    with a deterministic ripple, hood flat dark, lane white, movable mid. Coord-deterministic,
    no RNG -> reproducible + free."""
    h, w = part.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :] * np.ones((h, 1), np.float32)
    out = np.zeros((h, w, 3), np.float32)
    # sky/undrivable: bright top -> dimmer mid, bluish
    sky = np.stack([
        215 - 70 * yy, 225 - 65 * yy, 235 - 55 * yy,
    ], axis=-1)
    # road: asphalt mid-gray + deterministic ripple (perspective-ish horizontal banding)
    ripple = 6.0 * np.sin(40.0 * yy) * np.cos(18.0 * xx)
    road = np.stack([95 + ripple, 95 + ripple, 99 + ripple], axis=-1)
    # hood: flat dark with faint vertical shading
    hood = np.stack([26 + 8 * yy, 26 + 8 * yy, 30 + 8 * yy], axis=-1)
    lane = np.full((h, w, 3), 235.0, np.float32)
    mov = np.stack([120 + 0 * yy, 90 + 0 * yy, 80 + 0 * yy], axis=-1)
    out[part == roles.sky] = sky[part == roles.sky]
    out[part == roles.road] = road[part == roles.road]
    out[part == roles.hood] = hood[part == roles.hood]
    out[part == roles.lane] = lane[part == roles.lane]
    out[part == roles.movable] = mov[part == roles.movable]
    return np.clip(out, 0, 255).astype(np.float32)


def compute_gtmean_palette(gt_f1_cam: list, lstars: list, n_classes: int = 5) -> dict[int, tuple]:
    """Per-class MEAN RGB at scorer res (tiny COUNTED ~60 bytes, advisory). bilinear-resize each
    camera frame to (SEG_H,SEG_W), index by L*, accumulate. The 'optimal flat color' per class."""
    import torch

    sums = np.zeros((n_classes, 3), np.float64)
    cnts = np.zeros(n_classes, np.float64)
    for cam, ls in zip(gt_f1_cam, lstars):
        x = torch.from_numpy(np.ascontiguousarray(cam)).permute(2, 0, 1)[None].float()
        with torch.inference_mode():
            dn = torch.nn.functional.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
        small = dn[0].permute(1, 2, 0).numpy()  # (384,512,3)
        for c in range(n_classes):
            m = ls == c
            if m.any():
                sums[c] += small[m].sum(axis=0)
                cnts[c] += int(m.sum())
    pal = {}
    for c in range(n_classes):
        pal[c] = tuple((sums[c] / max(cnts[c], 1.0)).tolist())
    return pal


def role_palette(generic: dict, roles) -> dict[int, tuple]:
    return {
        roles.road: generic["road"], roles.lane: generic["lane"], roles.sky: generic["sky"],
        roles.movable: generic["movable"], roles.hood: generic["hood"],
    }


def measure_config(segnet, paint_fn, parts: list, lstars: list, label: str, *,
                   per_class_flip: bool = False, n_classes: int = 5) -> dict:
    """paint_fn(part_i, idx) -> (SEG_H,SEG_W,3) float render; R -> verdict per frame."""
    t0 = time.time()
    dsegs = []
    flip_by_gtclass = np.zeros(n_classes, np.float64)  # flips where L*==c, normalized by total px
    px_by_gtclass = np.zeros(n_classes, np.float64)
    for i, (part, ls) in enumerate(zip(parts, lstars)):
        rgb = paint_fn(part, i)
        cam = _torch_R_to_camera_uint8(rgb)
        realized = _cpu_realized_argmax(segnet, cam)
        dsegs.append(d_seg_from_argmax(realized, ls))
        if per_class_flip:
            flips = realized != ls
            for c in range(n_classes):
                mc = ls == c
                flip_by_gtclass[c] += int(np.count_nonzero(flips & mc))
                px_by_gtclass[c] += int(mc.sum())
    out = {
        "label": label, "n": len(dsegs),
        "d_seg_mean": float(np.mean(dsegs)), "d_seg_std": float(np.std(dsegs)),
        "d_seg_min": float(np.min(dsegs)), "d_seg_max": float(np.max(dsegs)),
        "wall_s": round(time.time() - t0, 1),
    }
    if per_class_flip:
        tot = px_by_gtclass.sum()
        # fraction of ALL pixels that are flips falling on each GT class (sums to d_seg_mean approx)
        out["flip_frac_by_gtclass"] = {int(c): round(float(flip_by_gtclass[c] / tot), 5) for c in range(n_classes)}
        out["flip_rate_within_gtclass"] = {int(c): round(float(flip_by_gtclass[c] / max(px_by_gtclass[c], 1)), 5) for c in range(n_classes)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--n", type=int, default=48, help="num frames (subset of cache)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(int(args.threads))
    from tac.boundary_math.lever_b_levelset_generator import build_static_core_partition
    from tac.boundary_math.road_horizon_component import classify_segnet_regions
    from tac.boundary_math.seg_core import load_real_segnet

    seg = load_real_segnet("cpu")
    z = np.load(args.cache, allow_pickle=False)
    P = min(int(args.n), int(z["n_pairs"]))
    lstars = [np.asarray(z["lstars"][i], dtype=np.int64) for i in range(P)]
    gt_f1 = [np.asarray(z["gt_f1"][i], dtype=np.uint8) for i in range(P)]
    lst_stack = np.stack(lstars, axis=0)

    roles = classify_segnet_regions(lst_stack, n_classes=5)
    rd = roles.as_dict()
    print(f"[roles] {rd}  (canonical: road sky hood movable lane self-detected)")

    # static-core partitions (per-frame uses the SHARED majority static mask; same for all frames)
    part_sc_nolane, _r, meta_nl = build_static_core_partition(lst_stack, n_classes=5, include_lane=False)
    part_sc_lane, _r2, meta_l = build_static_core_partition(lst_stack, n_classes=5, include_lane=True)
    print(f"[static-core] frac no-lane={meta_nl.get('part_frac')}  lane-incl-px={meta_l.get('lane_px')}")

    # static approximation error: static-core vs per-frame L* IN STATIC REGIONS only
    static_classes = {roles.road, roles.sky, roles.hood}
    sae_num = 0
    sae_den = 0
    for ls in lstars:
        static_mask = np.isin(ls, list(static_classes)) | np.isin(part_sc_lane, list(static_classes))
        sae_num += int(np.count_nonzero((part_sc_lane != ls) & static_mask))
        sae_den += int(ls.size)
    static_approx_err = sae_num / sae_den
    print(f"[static-approx-err] static-core vs per-frame L* in static regions: {static_approx_err:.6f}")

    # composed best-case partition: static-core, overridden by GT-L* wherever L* in {lane,movable}
    res_classes = (roles.lane, roles.movable)
    parts_compose = []
    for ls in lstars:
        pc = part_sc_lane.copy()
        ov = np.isin(ls, res_classes)
        pc[ov] = ls[ov]
        parts_compose.append(pc)

    # palettes
    gen_pal = role_palette(GENERIC_PALETTE, roles)
    t0 = time.time()
    gtmean_pal = compute_gtmean_palette(gt_f1, lstars, n_classes=5)
    print(f"[gtmean-palette] {{ {', '.join(f'{k}:({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})' for k,v in sorted(gtmean_pal.items()))} }}  ({time.time()-t0:.1f}s)")

    # GT camera frames downsampled to render-res once (for gt_pixels ideal-texture upper bound)
    def gt_small_render(i):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(gt_f1[i])).permute(2, 0, 1)[None].float()
        with torch.inference_mode():
            dn = torch.nn.functional.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
        return dn[0].permute(1, 2, 0).numpy().astype(np.float32)

    results = []

    def add(r):
        results.append(r)
        print(f"  {r['label']:<42s} d_seg={r['d_seg_mean']:.6f} (+/-{r['d_seg_std']:.4f}) [{r['wall_s']}s]"
              + (f"  flip_frac={r.get('flip_frac_by_gtclass')}" if 'flip_frac_by_gtclass' in r else ""))

    print("\n=== M0 baselines ===")
    # random texture (reproduce FEED-ek ~0.506)
    rng = np.random.default_rng(0)
    rand_tex = [rng.uniform(0, 255, (SEG_H, SEG_W, 3)).astype(np.float32) for _ in range(P)]
    add(measure_config(seg, lambda p, i: rand_tex[i], [part_sc_lane] * P, lstars, "M0 random-texture(static-core part)"))
    # GT pixels at render res (R-survival floor of a perfect render-res renderer)
    add(measure_config(seg, lambda p, i: gt_small_render(i), [None] * P, lstars, "M0 GT-pixels@render-res(full)", per_class_flip=True))

    print("\n=== D: full GT L* painted (flat-palette legibility ceiling over whole frame) ===")
    add(measure_config(seg, lambda p, i: paint_flat(lstars[i], gen_pal), [None] * P, lstars, "D full-L* flat-generic", per_class_flip=True))
    add(measure_config(seg, lambda p, i: paint_flat(lstars[i], gtmean_pal), [None] * P, lstars, "D full-L* flat-gtmean", per_class_flip=True))
    add(measure_config(seg, lambda p, i: paint_det_realistic(lstars[i], roles), [None] * P, lstars, "D full-L* det-realistic", per_class_flip=True))

    print("\n=== B: static-core ONLY (legibility + missing-residual cost) ===")
    add(measure_config(seg, lambda p, i: paint_flat(part_sc_lane, gen_pal), [None] * P, lstars, "B static-core flat-generic", per_class_flip=True))
    add(measure_config(seg, lambda p, i: paint_flat(part_sc_lane, gtmean_pal), [None] * P, lstars, "B static-core flat-gtmean"))
    add(measure_config(seg, lambda p, i: paint_det_realistic(part_sc_lane, roles), [None] * P, lstars, "B static-core det-realistic"))
    add(measure_config(seg, lambda p, i: paint_flat(part_sc_nolane, gtmean_pal), [None] * P, lstars, "B static-core(no-lane) flat-gtmean"))

    print("\n=== C: DECOMPOSITION BEST-CASE (static-core + perfect GT residual) ===")
    add(measure_config(seg, lambda p, i: paint_flat(parts_compose[i], gen_pal), [None] * P, lstars, "C decomp-best flat-generic", per_class_flip=True))
    add(measure_config(seg, lambda p, i: paint_flat(parts_compose[i], gtmean_pal), [None] * P, lstars, "C decomp-best flat-gtmean", per_class_flip=True))
    add(measure_config(seg, lambda p, i: paint_det_realistic(parts_compose[i], roles), [None] * P, lstars, "C decomp-best det-realistic", per_class_flip=True))

    print("\n=== B': static-core with IDEAL (real GT) texture in static region (legibility upper bound) ===")
    # static-core region gets real GT pixels (render-res), residual+rest gets flat-gtmean
    def paint_static_gtpixels(i):
        base = paint_flat(part_sc_lane, gtmean_pal)
        gtsmall = gt_small_render(i)
        sc_mask = np.isin(part_sc_lane, [roles.road, roles.sky, roles.hood])
        base[sc_mask] = gtsmall[sc_mask]
        return base
    add(measure_config(seg, lambda p, i: paint_static_gtpixels(i), [None] * P, lstars, "B' static-core REAL-texture+flat-resid", per_class_flip=True))

    summary = {
        "feed": "el", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": P, "cache": args.cache, "roles": rd,
        "static_core_frac_nolane": meta_nl.get("part_frac"),
        "static_approx_err_in_static_regions": round(static_approx_err, 6),
        "gtmean_palette": {int(k): [round(x, 1) for x in v] for k, v in gtmean_pal.items()},
        "results": results,
        "authority": "realized-through-R: paint->bicubic-up-874->uint8->SegNet.preprocess->argmax vs L* (frozen CPU-torch). NEVER MLX/MPS.",
        "pointer": "UNMOVED 0.19110; advisory geometry/training-signal measurement.",
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\n[wrote] {args.out_json}")
    print("\n=== SUMMARY JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
