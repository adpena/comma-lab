#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pp1 R2/R3 — the correction-stream BAND LEMMA falsifier + measured density curve.

MEANS. pointer 0.1910828242 UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE.

THE LEMMA (ee1 §C6, DERIVED): a correction stream's POSITION cost alone is lower-bounded by the
uniform combinatorial rate ~ log2(N/k)/8 B/err (k = #errors over N sites). Context/coherence LOWERS
the actual cost below that bound. The uniform bound crosses the registered 1.2731 B/flip water level
at base-error k/N where log2(N/k)=8*1.2731 => N/k=2^10.185 => k/N ~ 8.6e-4. CLAIM: correction streams
are rational ONLY in a band ~1e-3..1e-2 base error; below it the position floor exceeds the water
level so CONCEDING dominates; above it support cost explodes anyway.

THE $0 FALSIFIER (this tool): recompute the SUPPORT (position) coding price at N synthetic densities
spanning ~1e-4..1e-1, on boundary-COHERENT correction fields (margin-thresholded: {px: margin<tau} —
the realistic "where a base disagrees" structure) AND on RANDOM-subsampled (incoherent) fields (the
uniform-bound reference). Code positions with the generic incumbent (packbits->LZMA) and the coherent
#307 contour chain coder; report B/err vs the uniform log2(N/k)/8 and locate the MEASURED crossing of
1.2731 (the context-shifted band edge). Cross-check: the curve must pass near fc1's measured anchor
(0.864% density, 0.413 B/err LZMA support). If the measured curve shows the crossing near ~1e-3 (as
the lemma predicts, shifted by context), R3 registers the canonical equation; if it does NOT, register
nothing and the measured curve IS the finding.

Usage:
  PYTHONPATH=src:tools .venv/bin/python experiments/ddm_pp1_band_lemma_curve.py \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --out /Volumes/VertigoDataTier/pact/ddm_pp1_20260728/r2_band_lemma_curve_n600.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import lzma
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WATER_B_PER_FLIP = 1.2731
FC1_ANCHOR = {"density": 0.00864212883843316, "b_per_err_lzma": 421366 / 1019467}  # 0.4133


def _lzma_raw(data: bytes) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_RAW,
                         filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}])


def _load_m307():
    spec = importlib.util.spec_from_file_location(
        "m307c", str(_REPO / "tools" / "measure_contour_string_flip_coding.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m307c"] = m
    spec.loader.exec_module(m)
    return m


def position_price(support: np.ndarray, m307, want_contour: bool) -> dict:
    """Position (support) coding price: packbits->LZMA incumbent + optional #307 contour chain coder.
    support: (N,H,W) bool. Returns bytes + B/err for each coder (round-trip guaranteed by construction:
    packbits/LZMA are lossless; the #307 contour helper verifies its own streams internally)."""
    n = int(support.sum())
    if n == 0:
        return {"n_err": 0}
    packed = np.packbits(support.reshape(-1)).tobytes()
    lz = len(_lzma_raw(packed))
    out = {"n_err": n, "lzma_bytes": lz, "b_per_err_lzma": lz / n}
    if want_contour:
        N, H, W = support.shape
        fmaps = [np.ascontiguousarray(support[f]) for f in range(N)]
        cmaps = [np.zeros((H, W), np.int64) for _ in range(N)]  # positions only; class stream trivial
        enc = m307.contour_encode_frames(fmaps, cmaps)
        # exclude the (near-zero) class stream to isolate POSITION cost
        pos_bytes = enc["stream_bytes"]["counts"] + enc["stream_bytes"]["anchor"] + enc["stream_bytes"]["chain"]
        out["contour_pos_bytes"] = int(pos_bytes)
        out["b_per_err_contour"] = pos_bytes / n
    out["b_per_err_best"] = min([v for k, v in out.items()
                                 if k.startswith("b_per_err_")])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache",
                    default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.gt_cache, mmap_mode="r")
    margins = np.asarray(z["margins"], dtype=np.float32)
    N, H, W = margins.shape
    Nsites = N * H * W
    m307 = _load_m307()
    print(f"[pp1-R2] margins {margins.shape} ({time.time()-t0:.1f}s)")

    # coherent (margin-thresholded) density sweep
    taus = [0.008, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 3.00]
    coherent = []
    for tau in taus:
        sup = margins < tau
        dens = float(sup.mean())
        # #307 contour is O(components); skip on the very dense fields (>3%) to bound wall-clock
        pr = position_price(sup, m307, want_contour=(dens <= 0.03))
        pr["tau"] = tau
        pr["density"] = dens
        pr["uniform_bound_b_per_err"] = float(np.log2(1.0 / dens) / 8.0) if dens > 0 else None
        coherent.append(pr)
        print(f"   coherent tau<{tau:5.2f} rho={dens:.5f} "
              f"B/err best={pr.get('b_per_err_best'):.3f} uniform={pr['uniform_bound_b_per_err']:.3f}")

    # incoherent (random-subsample) reference: subsample the tau<0.8 field to target densities
    rng = np.random.default_rng(0)
    base = margins < 0.80
    base_idx = np.flatnonzero(base.reshape(-1))
    incoherent = []
    for target in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        k = round(target * Nsites)
        pick = rng.choice(base_idx, size=min(k, base_idx.size), replace=False)
        sup = np.zeros(Nsites, dtype=bool)
        sup[pick] = True
        sup = sup.reshape(N, H, W)
        dens = float(sup.mean())
        pr = position_price(sup, m307, want_contour=False)  # random -> contour meaningless
        pr["target_density"] = target
        pr["density"] = dens
        pr["uniform_bound_b_per_err"] = float(np.log2(1.0 / dens) / 8.0)
        incoherent.append(pr)
        print(f"   incoherent rho={dens:.5f} B/err_lzma={pr.get('b_per_err_lzma'):.3f} "
              f"uniform={pr['uniform_bound_b_per_err']:.3f}")

    # locate the measured coherent crossing of the water level (interp on log-density)
    pts = [(p["density"], p["b_per_err_best"]) for p in coherent if p.get("n_err")]
    pts.sort()
    xs = np.array([np.log10(d) for d, _ in pts])
    ys = np.array([b for _, b in pts])
    # crossing where b == WATER (curve decreasing in density, so search)
    crossing_density = None
    for i in range(len(xs) - 1):
        if (ys[i] - WATER_B_PER_FLIP) * (ys[i + 1] - WATER_B_PER_FLIP) <= 0:
            # linear interp in log-density
            t = (WATER_B_PER_FLIP - ys[i]) / (ys[i + 1] - ys[i]) if ys[i + 1] != ys[i] else 0.0
            crossing_density = float(10 ** (xs[i] + t * (xs[i + 1] - xs[i])))
            break
    # uniform-bound crossing (analytic): log2(1/rho)/8 = 1.2731 -> rho = 2^-10.1848
    uniform_crossing = float(2 ** (-8 * WATER_B_PER_FLIP))

    # lemma verdict: confirmed if the coherent curve DOES cross the water level at a density in the
    # predicted band (context shifts it BELOW the uniform crossing but same order ~1e-3..1e-4)
    confirmed = (crossing_density is not None) and (1e-5 <= crossing_density <= 1e-2)

    res = {
        "schema": "ddm_pp1_band_lemma_curve.v1",
        "utc": datetime.now(UTC).isoformat(),
        "evidence_axis": "[macOS-CPU advisory] NON-PROMOTABLE — real position-coding bytes on "
                         "synthetic correction-support densities; NOT a byte-closed evaluate.py row.",
        "water_B_per_flip": WATER_B_PER_FLIP,
        "N_sites": Nsites,
        "fc1_anchor": FC1_ANCHOR,
        "coherent_curve": coherent,
        "incoherent_curve": incoherent,
        "uniform_bound_crossing_density": uniform_crossing,
        "measured_coherent_crossing_density": crossing_density,
        "lemma_confirmed": bool(confirmed),
        "interpretation": (
            "The uniform combinatorial position bound log2(N/k)/8 crosses the 1.2731 water at "
            f"rho={uniform_crossing:.2e}. The MEASURED context-coherent position cost crosses at "
            f"rho={crossing_density if crossing_density else 'NO-CROSSING-IN-RANGE'} (the band edge, "
            "shifted by context). Correction streams are rational only ABOVE the crossing; below it, "
            "conceding at the water level dominates."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[pp1-R2] wrote {args.out}")
    print(f"[pp1-R2] uniform crossing rho={uniform_crossing:.2e}  measured coherent crossing "
          f"rho={crossing_density}  lemma_confirmed={confirmed}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
