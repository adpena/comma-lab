# SPDX-License-Identifier: MIT
"""Precompute the through-R fragility-weighted margin-Jacobian reachability map S_R
and store it as an ``sR`` key in an existing MLX-fleet GT cache npz (LEVER-4 REACHABILITY).

WHY
---
LEVER-4 (``--margin-saliency-*`` in ``experiments/train_levelset_witness_realized_through_R_mlx.py``)
optionally down-weights its per-pixel saliency by a UNIWARD texture proxy ``1/(1+beta*tex)``.
A $0 Phase-0 measurement (memory ``[[msal-uni-texture-proxy-inert-build-exact-sR-reachability-weight]]``)
PROVED that texture proxy does NOT predict through-R detector reachability (Pearson -0.033,
top-5% Jaccard 0.024 -- statistical chance, mildly MISDIRECTS). The EXACT signal is the
**through-R fragility-weighted margin-Jacobian**

    S_R = | d( sum_p w_p * margin_p ) / dx |,   w = exp(-margin / tau)

evaluated at the GT **target** frame x = downsample(gt_f1). This is theta-INDEPENDENT (it is the
reachability of the CORRECT answer through the contest R operator + frozen SegNet), so it is
precomputed ONCE and cached like ``margins``. S_R lives right on the fragile margin-boundary band
where the d_seg debt is.

METHOD (verbatim from the proven $0 scratch probe ``measure_SR_vs_texproxy.py``; CPU-ONLY torch
autograd, per-frame -- NEVER batch n600, the verdict-batch OOM trap):
  x       = bilinear-downsample(gt_f1 874x1164 -> 384x512), float32 [0,255]   ("correct" render)
  Rx      = R(x): bicubic-up->(874,1164 a_c=False) -> clip[0,255]+round STE @ camera -> bilinear-down->(384,512)
            (uint8-STE grad = identity => dR/dx is the pure resample low-pass; the exact contest R)
  margin  = (top1-top2).clamp_min(0) of frozen CPU-torch SegNet(Rx)            (realized through-R margin)
  w       = exp(-margin/tau).detach()                                          (fragility cotangent, stop-grad)
  S_R     = |d(sum_p w_p*margin_p)/dx|.sum(channels)  via margin.backward(gradient=w)

NORMALIZATION (stored, per-frame): a clean bounded [0,1] weight the trainer multiplies onto the
saliency directly (no percentile op needed in the MLX train step -> strictly cheaper at train time):
  sR_norm = clip( S_R / (percentile(S_R, 99) + eps), 0, 1 )   -- robust (p99, not max: outlier-safe).

Stored ``sR`` shape ``(P, 384, 512)`` float32 in [0,1], ONE per GT pair, alongside ``margins``.

DISK / CONTAINMENT
------------------
* CPU-ONLY torch autograd. NO MLX/MPS -- never contends with a live MLX-GPU training run.
* Atomic write (tmp + os.replace); refuses /tmp-class durable paths (CLAUDE.md hygiene).
* ``--mode inplace`` (default) re-serializes ALL original cache keys + ``sR`` back to the SAME path.
  Adding a member does NOT change any existing member read by ``load_gt_from_cache`` (it reads the 5
  named keys), so an augmented cache is byte-identical for the trainer's GT reads.
* ``--mode sidecar`` writes ONLY ``sR`` + ``n_pairs`` to ``<stem>_sR.npz`` -- NEVER touches the main
  cache (use when the main cache is a live-run dependency you must not rewrite, e.g. gt_n600 during a
  running n600 arm).
* The map is deterministic + rebuildable (this command + frozen SegNet + the cache's gt_f1); the cache
  path is its own provenance.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TAU_DEFAULT = 0.5  # == the trainer's --margin-saliency-tau default (sal = exp(-gt_margin/tau)).
CAMERA_HW = (874, 1164)
SEG_HW = (384, 512)
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class durable path; use the SSD/repo tier per CLAUDE.md.")


# --------------------------------------------------------------------------------------------------
# Proven compute primitives (VERBATIM from the $0 scratch probe measure_SR_vs_texproxy.py that
# produced the Pearson -0.033 texture-proxy-inert measurement). Torch CPU autograd, differentiable R.
# --------------------------------------------------------------------------------------------------
def _load_segnet():
    from tac.boundary_math.seg_core import load_real_segnet

    seg = load_real_segnet("cpu")
    seg.eval()
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg


def _bilin_down_to_render(gt_f1_uint8):
    """gt_f1 (874,1164,3) uint8 -> x (384,512,3) float32 [0,255], bilinear a_c=False."""
    import torch

    t = torch.from_numpy(np.asarray(gt_f1_uint8)).float().permute(2, 0, 1)[None]  # (1,3,874,1164)
    d = torch.nn.functional.interpolate(t, size=SEG_HW, mode="bilinear", align_corners=False)
    return d[0].permute(1, 2, 0).contiguous()  # (384,512,3)


def _R_torch(x_hwc):
    """x (384,512,3) [0,255] -> Rx (384,512,3). Differentiable (uint8-STE identity grad). Exact contest R."""
    import torch

    xc = x_hwc.permute(2, 0, 1)[None]  # (1,3,384,512)
    up = torch.nn.functional.interpolate(xc, size=CAMERA_HW, mode="bicubic", align_corners=False)
    clipped = up.clamp(0.0, 255.0)
    rounded = clipped.round()
    q = clipped + (rounded - clipped).detach()  # STE round @ camera res
    down = torch.nn.functional.interpolate(q, size=SEG_HW, mode="bilinear", align_corners=False)
    return down[0].permute(1, 2, 0).contiguous()  # (384,512,3)


def _segnet_margin(seg, rx_hwc):
    """rx (384,512,3) [0,255] -> margin (384,512) realized top1-top2, differentiable."""
    import torch

    pair = torch.stack([rx_hwc, rx_hwc], dim=0)[None]  # (1,2,384,512,3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous()      # (1,2,3,384,512)
    seg_in = seg.preprocess_input(xp)                  # (1,3,384,512) [no-op resize @384x512]
    logits = seg(seg_in)                               # (1,5,384,512)
    top2 = torch.topk(logits, k=2, dim=1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)[0]  # (384,512)
    return margin


def _compute_SR(seg, x_hwc, tau: float):
    import torch

    x = x_hwc.clone().requires_grad_(True)
    rx = _R_torch(x)
    margin = _segnet_margin(seg, rx)
    w = torch.exp(-margin / tau).detach()  # fragility cotangent (stop-grad)
    x.grad = None
    margin.backward(gradient=w)
    S_R = x.grad.abs().sum(dim=-1).detach().numpy()  # (384,512) float32-ish
    return S_R.astype(np.float32)


def _normalize_sR(S_R: np.ndarray, *, pct: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    """Per-frame robust [0,1] weight: clip(S_R / (percentile(S_R, pct) + eps), 0, 1)."""
    scale = float(np.percentile(S_R, pct)) + eps
    out = np.clip(S_R / scale, 0.0, 1.0).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------------------
def compute_sR_for_cache(cache_path: Path, *, tau: float, num_pairs: int | None,
                         pct: float) -> tuple[np.ndarray, dict]:
    """Return (sR (P,384,512) float32 in [0,1], stats). Reads ONLY gt_f1 from the cache (lazy npz)."""
    z = np.load(cache_path, allow_pickle=False)
    if "gt_f1" not in z.files:
        raise ValueError(f"--gt-cache {cache_path} has no 'gt_f1' member (not an MLX-fleet GT cache?).")
    cached_P = int(z["n_pairs"]) if "n_pairs" in z.files else int(z["gt_f1"].shape[0])
    P = cached_P if num_pairs is None else min(int(num_pairs), cached_P)
    gt_f1_all = z["gt_f1"]  # inflate the gt_f1 member ONCE (the only large member we read)
    seg = _load_segnet()
    sR = np.empty((P, SEG_HW[0], SEG_HW[1]), dtype=np.float32)
    raw_means = np.empty(P, dtype=np.float64)
    t0 = time.time()
    for i in range(P):
        x = _bilin_down_to_render(gt_f1_all[i])
        S_R_raw = _compute_SR(seg, x, tau)
        raw_means[i] = float(S_R_raw.mean())
        sR[i] = _normalize_sR(S_R_raw, pct=pct)
        del x, S_R_raw
        gc.collect()
        if (i + 1) % 50 == 0 or i == P - 1:
            dt = time.time() - t0
            print(json.dumps({"stage": "sR_progress", "done": i + 1, "of": P,
                              "secs": round(dt, 1), "s_per_frame": round(dt / (i + 1), 3)}), flush=True)
    stats = {
        "n_pairs": int(P), "tau": float(tau), "norm_percentile": float(pct),
        "sR_norm_mean": round(float(sR.mean()), 5),
        "sR_norm_frac_gt_0p5": round(float((sR > 0.5).mean()), 5),
        "sR_raw_mean_over_frames": round(float(raw_means.mean()), 6),
        "compute_secs": round(time.time() - t0, 1),
    }
    return sR, stats


def _atomic_savez(out_path: Path, arrays: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, out_path)


def write_inplace(cache_path: Path, sR: np.ndarray) -> None:
    """Re-serialize ALL original cache members + the new ``sR`` key, atomically, to the SAME path.
    Adding a member does not change any existing member -> byte-identical for load_gt_from_cache reads."""
    z = np.load(cache_path, allow_pickle=False)
    arrays = {k: z[k] for k in z.files}  # materialize existing members (bounded by the cache footprint)
    arrays["sR"] = sR
    _atomic_savez(cache_path, arrays)


def write_sidecar(cache_path: Path, sR: np.ndarray) -> Path:
    """Write ONLY sR + n_pairs to ``<stem>_sR.npz`` -- NEVER touch the main cache."""
    sidecar = cache_path.with_name(cache_path.stem + "_sR.npz")
    _atomic_savez(sidecar, {"n_pairs": np.int64(sR.shape[0]), "sR": sR})
    return sidecar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Precompute the through-R margin-Jacobian reachability map S_R and cache it as 'sR'.")
    ap.add_argument("--gt-cache", type=Path, required=True,
                    help="Existing MLX-fleet GT cache npz (must contain gt_f1); "
                    "e.g. experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--num-pairs", type=int, default=None,
                    help="Compute sR for only the first N cached pairs (default: all cached pairs).")
    ap.add_argument("--tau", type=float, default=TAU_DEFAULT,
                    help="Fragility softness w=exp(-margin/tau) (default 0.5 == trainer --margin-saliency-tau).")
    ap.add_argument("--norm-percentile", type=float, default=99.0,
                    help="Per-frame normalization: sR_norm = clip(S_R / percentile(S_R, this), 0, 1).")
    ap.add_argument("--mode", choices=("inplace", "sidecar"), default="inplace",
                    help="inplace: re-write the cache with an added 'sR' key (default). "
                    "sidecar: write ONLY '<stem>_sR.npz' (never touch the main cache).")
    args = ap.parse_args(argv)

    cache_path = Path(args.gt_cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"--gt-cache {cache_path} does not exist.")
    if args.mode == "inplace":
        _refuse_tmp(cache_path)

    sR, stats = compute_sR_for_cache(
        cache_path, tau=args.tau, num_pairs=args.num_pairs, pct=args.norm_percentile)

    if args.mode == "inplace":
        write_inplace(cache_path, sR)
        out = str(cache_path)
    else:
        out = str(write_sidecar(cache_path, sR))

    info = {"stage": "sR_written", "mode": args.mode, "out_path": out,
            "sR_shape": list(sR.shape), **stats,
            "note": "LEVER-4 reachability weight; activate with "
            "--margin-saliency-weight >0 --margin-saliency-reachability; pointer 0.19110 UNMOVED"}
    print(json.dumps(info, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
