#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""$0 DM3 axis test — per-position SPATIAL LATENT GRID vs GLOBAL per-pair CODE for the witness.

[macOS advisory / research-signal] score_claim=false promotable=false. NOT a contest score;
pointer claim FORBIDDEN. This sizes the witness-v2 conditioning axis BEFORE any GPU spend.

QUESTION (witness-v2 axis verdict, DAG FEED-ip): the proposed binding d_seg axis is DM3
(a per-position spatial latent grid) REPLACING the global per-pair FiLM code. Does spatial
conditioning move the spatially-LOCALIZED annulus d_seg long-tail (96.8% flip-mass in a 2px
annulus) that a global per-pair code supposedly cannot?

DECISIVE LINEAR TEST (no INR training; closed-form; matched per-pair bytes):
  shared partition   S(x)   = mean_p phi*_p(x)        (what decoder+static-core give for FREE)
  per-pair residual  R_p(x) = phi*_p(x) - S(x)        (the COUNTED per-pair payload's job)
  GLOBAL-CODE CEILING  = best rank-D factorization of R across pairs (truncated SVD on the
                         pair Gram): R_p ~= code_p(D) @ Dict(D, N*K). The STRONGEST possible
                         global per-pair code of dim D (the witness FiLM is BELOW it: PR collapse).
  SPATIAL GRID         = per-pair per-class block-mean (and LSQ-optimal) of R_p on a (Gh,Gw)
                         grid + bilinear up. Per-pair latent = Gh*Gw*K, integer-decodable.
  recon_p = S + correction_p ; d_seg_p = argmax-disagree(recon_p, lstar_p), FULL + 2px ANNULUS.
  -> RD curve: per-pair floats (proxy bytes) vs d_seg. Winner = lower d_seg at matched bytes,
     ESPECIALLY in the annulus.

FALSIFICATION (pre-registered): if a coarse spatial grid does NOT beat the global-code CEILING
on the annulus residual at matched per-pair bytes, the literal DM3 (per-pair grid) premise is
WRONG and the per-pair partition variation is globally low-rank -> redirect the design.

MEASURED VERDICT (n96 gt_n96, 2026-06-29, build of this tool): FALSIFIED. cross-pair variation
is globally LOW-RANK (rank-8 = 95.6% / rank-16 = 98.6% of variance: ego-motion coherence). At
~60 floats/pair the global-code CEILING reaches annulus 0.026 while the grid (block OR LSQ-
optimal) sits at 0.21 / 0.20 -> ~8x worse. The grid only "beats" a rank-1-collapsed FiLM by
spending ~100x more bytes to match a working rank-8 global code. The witness FiLM PR collapse is
a CONDITIONING-MECHANISM failure, not evidence a global code is insufficient. See design memo
.omx/research/witness_v2_dm2_dm3_spatial_conditioning_design_*.md.

NO-FAKE: phi* via the REAL scipy EDT (signed_distance_fields), asserted argmax-roundtrip == L*;
d_seg via the canonical d_seg_reference; the SVD/QR are the ACTUAL closed-form solves (a stub
would FAIL the full-rank reconstruction == phi check). Advisory tag; no score claim.

Disk hygiene: writes one small JSON to experiments/results/dm3_spatial_grid_val_run/ (gitignored,
rebuildable from this script + the committed gt cache). No bulk artifacts; no /tmp.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.boundary_math.bitmask_dseg import d_seg_reference  # noqa: E402
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields  # noqa: E402
from tac.contest_score import UNCOMPRESSED_SIZE_BYTES  # noqa: E402

H, W, K = 384, 512, 5
SEG_FRAMES_FULL = 600
DEFAULT_NPZ = str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz")


def boundary_band(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """Pixels within ``radius`` of an inter-class boundary (the d_seg annulus long-tail)."""
    from scipy import ndimage

    a = np.asarray(labels)
    bnd = np.zeros(a.shape, bool)
    bnd[:-1, :] |= a[:-1, :] != a[1:, :]
    bnd[1:, :] |= a[:-1, :] != a[1:, :]
    bnd[:, :-1] |= a[:, :-1] != a[:, 1:]
    bnd[:, 1:] |= a[:, :-1] != a[:, 1:]
    return ndimage.binary_dilation(bnd, iterations=int(radius))


def block_grid_correction(R_p: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """R_p (H,W,K) -> block-mean to (gh,gw,K) -> bilinear up (H,W,K). Integer-decodable in deploy."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(R_p).permute(2, 0, 1).unsqueeze(0).float()
    coarse = F.adaptive_avg_pool2d(t, (gh, gw))
    up = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def upsample_operator(gh: int, gw: int) -> np.ndarray:
    """(N, gh*gw) bilinear-up operator: column c = up(onehot grid cell c). For the LSQ-optimal grid."""
    import torch
    import torch.nn.functional as F

    eye = torch.eye(gh * gw).reshape(gh * gw, 1, gh, gw)
    up = F.interpolate(eye, size=(H, W), mode="bilinear", align_corners=False)
    return up[:, 0].reshape(gh * gw, H * W).t().contiguous().numpy().astype(np.float64)


def dseg_full_annulus(recon_hwk: np.ndarray, lstar: np.ndarray, band: np.ndarray) -> tuple[float, float]:
    pred = recon_hwk.argmax(axis=-1).astype(np.int64)
    full = float(d_seg_reference(pred, lstar))
    nb = int(band.sum())
    ann = float(np.count_nonzero((pred != lstar) & band)) / max(nb, 1)
    return full, ann


def rate_term(floats_per_pair: int) -> float:
    """int8 UPPER bound on the rate term for D floats/pair x 600 pairs (brotli/range-code lower)."""
    return 25.0 * floats_per_pair * SEG_FRAMES_FULL / UNCOMPRESSED_SIZE_BYTES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--global-dims", default="4,8,16,32,64")
    ap.add_argument("--grids", default="3x4,4x5,6x8,8x11,12x16")
    ap.add_argument("--optimal-grids", default="6x8,12x16", help="LSQ-optimal grid steelman arms")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.npz)
    lstars = d["lstars"][: args.frames].astype(np.int64)
    P = lstars.shape[0]
    print(f"[load] {P} frames {H}x{W} K={K}  ({time.time()-t0:.1f}s)")

    phi = np.empty((P, H, W, K), np.float32)
    bands = np.empty((P, H, W), bool)
    for i in range(P):
        phi[i] = signed_distance_fields(lstars[i], K)
        bands[i] = boundary_band(lstars[i], 2)
    rt = int(np.count_nonzero(phi.argmax(-1).reshape(P, H, W) != lstars))
    if rt != 0:
        raise AssertionError(f"NO-FAKE: argmax(phi*) != L* on {rt} px — the EDT round-trip failed.")
    band_frac = float(bands.mean())
    print(f"[edt] phi* roundtrip OK; annulus band frac={band_frac:.4f}  ({time.time()-t0:.1f}s)")

    S = phi.mean(axis=0)
    Rres = phi - S[None]
    floor = [dseg_full_annulus(S, lstars[i], bands[i]) for i in range(P)]
    floor_full = float(np.mean([f for f, _ in floor]))
    floor_ann = float(np.mean([a for _, a in floor]))
    print(f"[floor S] d_seg full={floor_full:.5f} annulus={floor_ann:.5f}")

    X = Rres.reshape(P, H * W * K).astype(np.float64)
    with np.errstate(all="ignore"):
        G = X @ X.T
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    cum = np.cumsum(evals / evals.sum())
    print(f"[svd] cum-var rank[1,2,4,8,16,32]="
          f"{[round(float(cum[min(r-1, P-1)]),4) for r in (1,2,4,8,16,32)]}")

    def global_code_recon(Dn: int) -> np.ndarray:
        Wd = evecs[:, :Dn]
        with np.errstate(all="ignore"):
            Rrec = Wd @ (Wd.T @ X)
        return (S[None] + Rrec.reshape(P, H, W, K)).astype(np.float32)

    results = {
        "provenance": {
            "npz": args.npz, "frames": P, "shape": [H, W, K], "annulus_band_frac": band_frac,
            "phi_roundtrip_mismatch_px": rt, "floor_S_dseg_full": floor_full,
            "floor_S_dseg_annulus": floor_ann,
            "eig_cum_var": {str(r): float(cum[min(r - 1, P - 1)]) for r in (1, 2, 4, 8, 16, 32, 64)},
            "break_even_dseg_per_byte_per_frame": 25.0 * SEG_FRAMES_FULL / (100.0 * UNCOMPRESSED_SIZE_BYTES),
            "tag": "[macOS advisory/research-signal] score_claim=false promotable=false",
        },
        "global_code_ceiling": [], "spatial_grid": [], "spatial_grid_optimal": [],
    }

    for Dn in [int(x) for x in args.global_dims.split(",")]:
        recon = global_code_recon(Dn)
        res = [dseg_full_annulus(recon[i], lstars[i], bands[i]) for i in range(P)]
        ff, aa = float(np.mean([f for f, _ in res])), float(np.mean([a for _, a in res]))
        results["global_code_ceiling"].append(
            {"D": Dn, "per_pair_floats": Dn, "rate_term": rate_term(Dn), "dseg_full": ff, "dseg_annulus": aa})
        print(f"[global D={Dn:3d}] floats/pair={Dn:4d} full={ff:.5f} ann={aa:.5f}")

    def parse_grid(s: str) -> tuple[int, int]:
        a, b = s.lower().split("x")
        return int(a), int(b)

    for gh, gw in [parse_grid(s) for s in args.grids.split(",")]:
        res = []
        for i in range(P):
            recon = S + block_grid_correction(Rres[i], gh, gw)
            res.append(dseg_full_annulus(recon, lstars[i], bands[i]))
        fl = gh * gw * K
        ff, aa = float(np.mean([f for f, _ in res])), float(np.mean([a for _, a in res]))
        results["spatial_grid"].append(
            {"grid": [gh, gw], "channels_K": K, "per_pair_floats": fl, "rate_term": rate_term(fl),
             "dseg_full": ff, "dseg_annulus": aa})
        print(f"[grid {gh}x{gw}] floats/pair={fl:4d} full={ff:.5f} ann={aa:.5f}")

    for gh, gw in [parse_grid(s) for s in args.optimal_grids.split(",") if s.strip()]:
        U = upsample_operator(gh, gw)
        with np.errstate(all="ignore"):
            Q, _ = np.linalg.qr(U)
        res = []
        for i in range(P):
            Rk = Rres[i].reshape(H * W, K).astype(np.float64)
            with np.errstate(all="ignore"):
                corr = (Q @ (Q.T @ Rk)).reshape(H, W, K)
            res.append(dseg_full_annulus(S + corr.astype(np.float32), lstars[i], bands[i]))
        fl = gh * gw * K
        ff, aa = float(np.mean([f for f, _ in res])), float(np.mean([a for _, a in res]))
        results["spatial_grid_optimal"].append(
            {"grid": [gh, gw], "channels_K": K, "per_pair_floats": fl, "rate_term": rate_term(fl),
             "dseg_full": ff, "dseg_annulus": aa})
        print(f"[grid* {gh}x{gw}] floats/pair={fl:4d} full={ff:.5f} ann={aa:.5f} (LSQ-optimal)")

    results["elapsed_sec"] = time.time() - t0
    out = args.out or str(_REPO / "experiments/results/dm3_spatial_grid_val_run/results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {out}  ({results['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()
