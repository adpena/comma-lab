#!/usr/bin/env python3
"""Minimal-SegNet-fooling render ladder (operator 2026-06-29: "what would fool SegNet
into reading correctly?").

THE QUESTION: what is the CHEAPEST input render that makes the FROZEN CPU-torch SegNet
argmax match the GT argmax ``lstars`` THROUGH the R operator, given that a stored
label / flat class-color is the known-FAIL baseline (FEED-ko: SegNet argmax is
receptive-field/context dependent)?

R OPERATOR (faithful, prompt-exact): render at SEG res 384x512 -> bicubic UP to native
874x1164 -> uint8 -> [SegNet.preprocess_input: bilinear DOWN to 384x512] -> frozen
CPU-torch SegNet argmax (NEVER MPS).  ``lstars`` itself is SegNet of the NATIVE 874x1164
GT frame down to 384 (so a 384-native render carries an inherent resolution penalty --
measured here as REF-B).

NO-FAKE: a self-check asserts SegNet(gt_f1)==lstars and ABORTS otherwise.  Every number
is real argmax-disagreement (d_seg).  This moves NO pointer (advisory, pre-submission,
bulk-only on n<600); only an n600 byte-closed exact eval is a score.

RUNGS (each rendered at 384, pushed THROUGH R, d_seg + honest byte estimate):
  REF-A  native-full-store (== lstars; the 874-native ceiling, d_seg 0 by self-check)
  REF-B  384-full-store (gt downsampled to 384; the 384-native ceiling = resolution penalty)
  R0     flat class-mean color per GT region (the known-FAIL baseline)
  R1     R0 + anti-Gibbs ramp (pre-blur the flat map so bicubic-up does not ring at edges)
  R2     R0 + procedural per-class TEXTURE (class mean + class-std noise, matched low-level stats)
  R3     R-robust margin-saliency delta in the fragile thin-margin band (fragile-vs-robust test)
  R4     procedural STATIC geometric template (aggregate road/sky/hood structure; ~free per frame)

Authority: numpy/CPU-torch.  Usage:
  .venv/bin/python tools/measure_segnet_fooling_ladder.py --n 6   # smoke
  .venv/bin/python tools/measure_segnet_fooling_ladder.py --n 96  # report (bulk advisory)
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
B0 = 37_545_489  # contest archive normalizer (25*bytes/B0 = rate term)
NAT_H, NAT_W = 874, 1164
SEG_H, SEG_W = 384, 512
BULK_IDX = [0, 2, 4]  # Road, Undrivable, MyCar (canonical comma10k order)
CLASS_NAMES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]


def _refuse_tmp(path: Path) -> None:
    if "/tmp" in str(path).replace("\\", "/").lower().split("/private")[-1] or str(path).startswith("/tmp"):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _brotli_or_zlib(buf: bytes) -> int:
    b = len(zlib.compress(buf, 9))
    try:
        import brotli
        b = min(b, len(brotli.compress(buf, quality=11)))
    except Exception:
        pass
    return b


def _resize_bilinear(pixel_map: np.ndarray, th: int, tw: int) -> np.ndarray:
    """numpy-portable separable bilinear (align_corners=False); mirrors _resize_map."""
    src = np.asarray(pixel_map, dtype=np.float64)
    sh, sw = src.shape[:2]
    if (sh, sw) == (th, tw):
        return src
    ys = np.clip((np.arange(th) + 0.5) * sh / th - 0.5, 0, sh - 1)
    xs = np.clip((np.arange(tw) + 0.5) * sw / tw - 0.5, 0, sw - 1)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1); x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0); wx = (xs - x0)
    if src.ndim == 2:
        wyc = wy[:, None]; wxc = wx[None, :]
        top = src[y0][:, x0] * (1 - wxc) + src[y0][:, x1] * wxc
        bot = src[y1][:, x0] * (1 - wxc) + src[y1][:, x1] * wxc
        return top * (1 - wyc) + bot * wyc
    out = np.empty((th, tw, src.shape[2]), dtype=np.float64)
    for c in range(src.shape[2]):
        out[:, :, c] = _resize_bilinear(src[:, :, c], th, tw)
    return out


def _seg2nat_nearest(idx_map_seg: np.ndarray) -> np.ndarray:
    ys = (np.arange(NAT_H) * SEG_H / NAT_H).astype(np.int64).clip(0, SEG_H - 1)
    xs = (np.arange(NAT_W) * SEG_W / NAT_W).astype(np.int64).clip(0, SEG_W - 1)
    return idx_map_seg[ys][:, xs]


def _gauss_blur(img_hw3: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur on a HxWx3 float image (numpy-portable)."""
    if sigma <= 0:
        return img_hw3
    rad = max(1, int(round(3 * sigma)))
    xs = np.arange(-rad, rad + 1)
    k = np.exp(-(xs ** 2) / (2 * sigma * sigma)); k /= k.sum()
    out = img_hw3.astype(np.float64)
    # horizontal then vertical, reflect padding
    for ax in (1, 0):
        padw = [(0, 0)] * out.ndim
        padw[ax] = (rad, rad)
        p = np.pad(out, padw, mode="reflect")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(i, i + out.shape[ax])
            acc += w * p[tuple(sl)]
        out = acc
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="number of pairs (n6 smoke / n96 report)")
    ap.add_argument("--n-r3", type=int, default=24, help="pairs for the heavier R3 gradient rung")
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--rungs", default="all", help="comma list or 'all'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck-pairs", type=int, default=4)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    d = np.load(cache)
    P = min(args.n, int(d["gt_f1"].shape[0]))
    gt_f1 = d["gt_f1"][:P].astype(np.uint8)         # (P,874,1164,3)
    lstars = d["lstars"][:P].astype(np.int64)        # (P,384,512)
    margins = d["margins"][:P].astype(np.float32)    # (P,384,512)
    assert gt_f1.shape[1:3] == (NAT_H, NAT_W) and lstars.shape[1:] == (SEG_H, SEG_W)

    seg = load_real_segnet("cpu")
    fwd_times = []

    def seg_native(frame_native_float):  # native HWC -> argmax(384), margin(384)
        t = time.time()
        am, mg = measure_segnet_argmax(seg, np.asarray(frame_native_float, dtype=np.float64))
        fwd_times.append(time.time() - t)
        return am, mg

    def bicubic_up(render384_f):  # (384,512,3) float 0..255 -> native uint8 (874,1164,3)
        x = torch.from_numpy(np.asarray(render384_f, dtype=np.float32)).permute(2, 0, 1)[None]
        xu = F.interpolate(x, size=(NAT_H, NAT_W), mode="bicubic", align_corners=False)
        nat = xu[0].permute(1, 2, 0).clamp(0, 255).round().numpy()
        return nat

    def through_R(render384_f, do_uint8=True):
        nat = bicubic_up(render384_f)
        if not do_uint8:
            # skip the uint8 cliff: feed the float native straight to SegNet
            x = torch.from_numpy(np.asarray(render384_f, dtype=np.float32)).permute(2, 0, 1)[None]
            xu = F.interpolate(x, size=(NAT_H, NAT_W), mode="bicubic", align_corners=False)
            nat = xu[0].permute(1, 2, 0).clamp(0, 255).numpy()
        am, _ = seg_native(nat)
        return am

    def dseg(am, lstar, mask=None):
        ne = (am != lstar)
        if mask is not None:
            return float(ne[mask].mean()) if mask.any() else 0.0
        return float(ne.mean())

    def bulk_mask(lstar):
        return np.isin(lstar, BULK_IDX)

    # ---- NO-FAKE self-check (REF-A native full store) ----
    scn = min(args.selfcheck_pairs, P)
    maxd = 0
    for p in range(scn):
        am, _ = seg_native(gt_f1[p].astype(np.float64))
        maxd = max(maxd, int(np.count_nonzero(am != lstars[p])))
    if maxd != 0:
        raise SystemExit(f"NO-FAKE self-check FAILED (max_disagree_px={maxd}).")
    print(f"[ladder] NO-FAKE self-check PASS ({scn} pairs, REF-A native d_seg=0)", flush=True)

    # ---- compress-time per-class stats (mean + std RGB) at SEG res ----
    # map gt_f1 (native) down to 384 once per pair; accumulate per-class mean/var.
    csum = np.zeros((5, 3)); csq = np.zeros((5, 3)); ccnt = np.zeros(5)
    gt384 = np.empty((P, SEG_H, SEG_W, 3))
    for p in range(P):
        gt384[p] = _resize_bilinear(gt_f1[p].astype(np.float64), SEG_H, SEG_W)
        for c in range(5):
            m = lstars[p] == c
            if m.any():
                px = gt384[p][m]
                csum[c] += px.sum(0); csq[c] += (px ** 2).sum(0); ccnt[c] += px.shape[0]
    class_mean = np.where(ccnt[:, None] > 0, csum / np.maximum(ccnt[:, None], 1), 127.0)
    class_var = np.maximum(csq / np.maximum(ccnt[:, None], 1) - class_mean ** 2, 0.0)
    class_std = np.sqrt(class_var)

    want = set(["REF_B", "R0", "R1", "R2", "R3", "R4"]) if args.rungs == "all" else set(args.rungs.split(","))
    results = {}

    def acc(label, render_fn, n=P, do_uint8=True, byte_note="", store_bytes=None):
        fs = []; fs_b = []
        for p in range(n):
            r384 = render_fn(p)
            am = through_R(r384, do_uint8=do_uint8)
            fs.append(dseg(am, lstars[p]))
            fs_b.append(dseg(am, lstars[p], bulk_mask(lstars[p])))
        results[label] = {
            "n": n, "d_seg_full": float(np.mean(fs)), "d_seg_bulk": float(np.mean(fs_b)),
            "byte_note": byte_note, "store_bytes_per600_est": store_bytes,
        }
        print(f"[ladder] {label:7s} n={n} d_seg_full={np.mean(fs):.5f} d_seg_bulk={np.mean(fs_b):.5f}", flush=True)

    # ---------- REF-B : 384 full store (resolution penalty ceiling) ----------
    if "REF_B" in want:
        acc("REF_B", lambda p: gt384[p], byte_note="store full 384x512x3 frame (compressed) per frame",
            store_bytes=None)
        # byte estimate: compress gt384 frames
        raw = np.stack([gt384[p].clip(0, 255).round().astype(np.uint8) for p in range(P)])
        results["REF_B"]["store_bytes_per600_est"] = int(_brotli_or_zlib(raw.tobytes()) * 600 / P)

    # ---------- R0 : flat class-mean ----------
    flat384_cache = {}

    def flat384(p):
        if p not in flat384_cache:
            img = class_mean[lstars[p]]  # (384,512,3)
            flat384_cache[p] = img
        return flat384_cache[p]

    # partition-map byte estimate (the dominant counted cost of R0/R1): compress the 384 class-index image
    part_raw = np.stack([lstars[p].astype(np.uint8) for p in range(P)])
    part_bytes_600 = int(_brotli_or_zlib(part_raw.tobytes()) * 600 / P)
    if "R0" in want:
        acc("R0", flat384, byte_note=f"full per-frame partition map (compressed ~{part_bytes_600}B/600) + 5 mean RGB(15B)",
            store_bytes=part_bytes_600 + 15)

    # ---------- R1 : anti-Gibbs ramp (pre-blur the flat map) ----------
    if "R1" in want:
        best = None
        for sigma in (0.5, 1.0, 1.5):
            fs = []; fsb = []
            for p in range(P):
                r = _gauss_blur(flat384(p), sigma)
                am = through_R(r)
                fs.append(dseg(am, lstars[p])); fsb.append(dseg(am, lstars[p], bulk_mask(lstars[p])))
            cand = (float(np.mean(fsb)), sigma, float(np.mean(fs)))
            print(f"[ladder] R1 sigma={sigma} d_seg_full={np.mean(fs):.5f} d_seg_bulk={np.mean(fsb):.5f}", flush=True)
            if best is None or cand[0] < best[0]:
                best = cand
        results["R1"] = {
            "n": P, "d_seg_full": best[2], "d_seg_bulk": best[0], "best_sigma": best[1],
            "byte_note": "same partition cost as R0; ramp is DETERMINISTIC (free)",
            "store_bytes_per600_est": part_bytes_600 + 15,
        }

    # ---------- R2 : procedural per-class texture ----------
    if "R2" in want:
        rng = np.random.default_rng(0)
        # one fixed procedural noise field per class (seeded -> free to regenerate at decode)
        noise = rng.standard_normal((SEG_H, SEG_W, 3))

        def tex384(p):
            img = class_mean[lstars[p]].copy()
            std_map = class_std[lstars[p]]  # (384,512,3)
            return img + noise * std_map

        acc("R2", tex384,
            byte_note=f"partition (~{part_bytes_600}B/600) + per-class mean+std (30B) + seed(free)",
            store_bytes=part_bytes_600 + 30)

    # ---------- R4 : static aggregate geometric template (~free per frame) ----------
    if "R4" in want:
        # per (row,col) per-class frequency over the n frames -> static argmax template
        freq = np.zeros((5, SEG_H, SEG_W))
        for p in range(P):
            for c in range(5):
                freq[c] += (lstars[p] == c)
        templ = freq.argmax(0).astype(np.int64)  # (384,512) ONE template for all frames
        templ_render = class_mean[templ]
        # byte: ONE template compressed, amortized over 600 -> per-frame ~0; report the one-time cost
        templ_bytes = _brotli_or_zlib(templ.astype(np.uint8).tobytes())

        def geom384(p):
            return templ_render

        # measure full + bulk vs each frame's lstars
        fs = []; fsb = []
        for p in range(P):
            am = through_R(geom384(p))
            fs.append(dseg(am, lstars[p])); fsb.append(dseg(am, lstars[p], bulk_mask(lstars[p])))
        results["R4"] = {
            "n": P, "d_seg_full": float(np.mean(fs)), "d_seg_bulk": float(np.mean(fsb)),
            "byte_note": f"ONE static template ({templ_bytes}B one-time, ~0/frame) + 5 mean RGB; per-frame pose-warp(#138/#145) would refine",
            "store_bytes_per600_est": templ_bytes + 15,
        }
        print(f"[ladder] R4      n={P} d_seg_full={np.mean(fs):.5f} d_seg_bulk={np.mean(fsb):.5f} (static template {templ_bytes}B one-time)", flush=True)

    # ---------- R3 : R-robust margin-saliency delta (fragile vs robust) ----------
    if "R3" in want:
        nr3 = min(args.n_r3, P)
        tau = 2.0  # fragile band: margin < tau (top1-top2 logit gap)
        results["R3"] = _run_r3(seg, F, torch, gt384, lstars, margins, flat384, class_mean,
                                bicubic_up, seg_native, dseg, bulk_mask, nr3, tau)

    avg_fwd = float(np.mean(fwd_times)) if fwd_times else None
    out = {
        "schema": "segnet_fooling_render_ladder.v1",
        "authority": "numpy/CPU-torch (NEVER MPS)",
        "score_claim": False, "promotable": False, "ready_for_exact_eval_dispatch": False,
        "frontier_pointer": "UNMOVED 0.19110",
        "R_operator": "render@384 -> bicubic UP 874x1164 -> uint8 -> bilinear DOWN 384 -> frozen CPU SegNet argmax",
        "d_seg_def": "mean argmax-disagreement vs lstars; d_seg_full=all px, d_seg_bulk=GT class in {Road,Undriv,MyCar}",
        "n_pairs": P, "n_frames": 2 * P, "selfcheck": "PASS (REF-A native d_seg=0)",
        "calibration": "GROUNDED through-R d_seg on n%d; bulk-only advisory; lane/movables/pose stack on top; only n600 byte-closed is a score" % P,
        "rungs": results, "avg_segnet_fwd_sec": avg_fwd,
    }
    outpath = (REPO / args.out) if args.out else (
        REPO / f"experiments/results/segnet_fooling_ladder/ladder_n{P}.json")
    _refuse_tmp(outpath); outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"[ladder] wrote {outpath}", flush=True)
    return 0


def _run_r3(seg, F, torch, gt384, lstars, margins, flat384, class_mean,
            bicubic_up, seg_native, dseg, bulk_mask, nr3, tau):
    """R-robust margin-saliency delta: compute the R-aware input gradient of a
    GT-class-margin hinge in the fragile band, add delta, and test whether it
    survives through R (uint8 + resample) vs only pre-uint8 (the fragile-vs-robust crux)."""
    import numpy as np
    NAT_H, NAT_W = 874, 1164
    eps_list = [4.0, 12.0, 32.0]
    base_full, base_bulk = [], []
    rob_thru = {e: [] for e in eps_list}      # delta via R-aware grad, measured THROUGH-R (uint8)
    rob_nouint8 = {e: [] for e in eps_list}   # same delta measured pre-uint8 (float native)
    frag_frac = []
    delta_active_px = []

    for p in range(nr3):
        base = flat384(p).copy()  # start from R0 flat (it has the residual fragile flips)
        # base through-R baseline
        am0 = seg_native(bicubic_up(base))[0]
        base_full.append(dseg(am0, lstars[p])); base_bulk.append(dseg(am0, lstars[p], bulk_mask(lstars[p])))
        frag = (margins[p] < tau)  # (384,512) fragile thin-margin band
        frag_frac.append(float(frag.mean()))

        # ----- R-aware input gradient of a GT-class margin hinge -----
        x = torch.tensor(base, dtype=torch.float32, requires_grad=True)  # (384,512,3)
        xup = F.interpolate(x.permute(2, 0, 1)[None], size=(NAT_H, NAT_W), mode="bicubic", align_corners=False)
        pair = xup.unsqueeze(1).repeat(1, 2, 1, 1, 1)  # (1,2,3,874,1164)
        seg_in = seg.preprocess_input(pair)            # (1,3,384,512)
        logits = seg(seg_in)[0]                         # (5,384,512)
        gt = torch.from_numpy(lstars[p]).long()        # (384,512)
        gt_logit = logits.gather(0, gt[None])[0]       # (384,512)
        big = logits.clone()
        big.scatter_(0, gt[None], -1e9)
        runnerup = big.max(0).values                   # (384,512)
        # hinge: want gt_logit - runnerup >= tau ; loss = relu(tau - (gt_logit-runnerup)) on fragile band
        fragt = torch.from_numpy(frag)
        hinge = torch.clamp(tau - (gt_logit - runnerup), min=0.0)
        loss = (hinge * fragt.float()).sum()
        loss.backward()
        g = x.grad.detach().numpy()                    # (384,512,3) ; -grad increases gt margin
        direction = -g
        # texture weight (UNIWARD): concentrate delta where the base render has local variance
        gray = base.mean(2)
        lv = np.abs(gray - _box3(gray))
        tw = (lv / (lv.max() + 1e-6))[:, :, None]
        # normalize direction to unit per-pixel sign*magnitude, restrict to fragile band
        dn = np.sign(direction) * (np.abs(direction) / (np.abs(direction).max() + 1e-12))
        fmask = frag[:, :, None]
        for e in eps_list:
            delta = e * dn * tw * fmask
            rr = np.clip(base + delta, 0, 255)
            am_t = seg_native(bicubic_up(rr))[0]                 # THROUGH-R (uint8)
            rob_thru[e].append(dseg(am_t, lstars[p], bulk_mask(lstars[p])))
            # pre-uint8: bicubic up float, NO round
            x2 = torch.from_numpy(rr.astype(np.float32)).permute(2, 0, 1)[None]
            xu2 = F.interpolate(x2, size=(NAT_H, NAT_W), mode="bicubic", align_corners=False)
            nat2 = xu2[0].permute(1, 2, 0).clamp(0, 255).numpy()
            am_n = seg_native(nat2)[0]
            rob_nouint8[e].append(dseg(am_n, lstars[p], bulk_mask(lstars[p])))
        delta_active_px.append(int((frag).sum()))
        print(f"[ladder] R3 pair {p+1}/{nr3} base_bulk={base_bulk[-1]:.5f} frag_frac={frag_frac[-1]:.3f}", flush=True)

    # byte estimate for delta: fragile px per 384 frame * ~ (sign:1 bit + which-channel/mag ~ a few bits)
    frag_px = float(np.mean(delta_active_px))
    bits_per_px = 4.0  # honest estimate: AR-coded fragile-band residual (sign + small magnitude)
    delta_bytes_600 = int(frag_px * bits_per_px / 8 * 600)
    res = {
        "n": nr3, "tau_fragile": tau, "eps_units_0_255": eps_list,
        "base_R0_flat_d_seg_bulk": float(np.mean(base_bulk)),
        "base_R0_flat_d_seg_full": float(np.mean(base_full)),
        "fragile_band_frac_mean": float(np.mean(frag_frac)),
        "delta_through_R_uint8_d_seg_bulk": {str(e): float(np.mean(rob_thru[e])) for e in eps_list},
        "delta_pre_uint8_d_seg_bulk": {str(e): float(np.mean(rob_nouint8[e])) for e in eps_list},
        "byte_note": f"per-frame fragile-band residual ~{frag_px:.0f}px * {bits_per_px}b/px -> ~{delta_bytes_600}B/600 (the LEARNED part)",
        "store_bytes_per600_est": delta_bytes_600,
        "interpretation": "fragile-vs-robust: compare pre_uint8 (best case) vs through_R_uint8 (real); gap = R/uint8 destruction",
    }
    print(f"[ladder] R3 base_bulk={res['base_R0_flat_d_seg_bulk']:.5f} "
          f"thru_R={res['delta_through_R_uint8_d_seg_bulk']} pre_uint8={res['delta_pre_uint8_d_seg_bulk']}", flush=True)
    return res


def _box3(a):
    import numpy as np
    p = np.pad(a, 1, mode="reflect")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + p[1:-1, 1:-1]) / 5.0


if __name__ == "__main__":
    raise SystemExit(main())
