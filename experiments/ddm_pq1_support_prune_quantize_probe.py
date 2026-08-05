#!/usr/bin/env python3
"""ddm_pq1 — support-prune x quantize surface on the persisted sq2 solves.

The operator's naive-engineering critique (2026-08-05) exposed that the carriage
"family closed" verdict priced the SOLVER'S artifacts, not the family's optimum:
(a) positions were priced as shipped, but the band is the DECODER-DERIVABLE
``label_boundary_band(seg_argmax(dec), 1)`` (gp1's FREE band, zero bytes);
(b) the full 10.3k-site support is the solver's initialization, not a necessity;
(c) continuous per-site RGB is the solver's choice, not the feasible set's.

This probe measures the family at its real corner: for a few persisted pairs,
prune the paint support to the top-k salient band sites, quantize kept values to
b bits/channel, realize through the SAME camera path, and measure flip RETENTION
through the real SegNet + d_pose. Break-even (derived, ledger 218f0e0855):
~320 B/pair against the -0.128 S population prize.

Gate 0 (runs first): verify band_flat == derivable band, per pair — if this
fails the positions-free premise dies and the probe reports it honestly.

Axis: [macOS-CPU frozen-scorer advisory], score_claim=false. Resumable per pair.
"""

from __future__ import annotations

import json
import lzma
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "upstream", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ddm_sq1_eta_seg_realization import Scorer, label_boundary_band  # noqa: E402
from ddm_sq1_stage_decomposition_and_solved_paint import (  # noqa: E402
    realize_scorer_paint_to_camera,
)
from ddm_sl2_sq2_persist_and_compose import _d_pose  # noqa: E402

SUB_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2")
ARGMAX_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
FRAMES_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_sl2_20260805/sq2_persisted_frames")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pq1_20260805/pq1_prune_quantize_rows.json")
CAM_H, CAM_W, SEQ_LEN, N_PAIRS_TOTAL = 874, 1164, 2, 600
PAIRS = (0, 179, 370, 514)
K_LADDER = (0, 4096, 2048, 1024, 512, 256)  # 0 = full band
B_LADDER = (8, 3, 2)  # bits per channel on kept paint
BREAK_EVEN_B_PER_PAIR = 320.0  # derived: 0.128 S prize / (25/37,545,489 per byte) / 600


def quantize(vals: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 8:
        return vals.copy()
    step = 256.0 / (1 << bits)
    q = np.round(vals.astype(np.float64) / step) * step + step / 2.0
    return np.clip(q, 0, 255).astype(np.uint8)


def coded_value_bytes(vals: np.ndarray, bits: int) -> int:
    if bits >= 8:
        payload = vals.tobytes()
    else:
        step = 256.0 / (1 << bits)
        payload = np.round(vals.astype(np.float64) / step).astype(np.uint8).tobytes()
    return len(lzma.compress(payload, preset=9))


def subset_position_bytes(n_band: int, k: int) -> float:
    """Combinatorial bound for shipping WHICH derivable-band sites are kept."""
    if k >= n_band:
        return 0.0
    return (math.lgamma(n_band + 1) - math.lgamma(k + 1) - math.lgamma(n_band - k + 1)) / (
        math.log(2) * 8.0)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if OUT.exists():
        rows = json.loads(OUT.read_text()).get("rows", [])
    done = {(r["pair"], r["k"], r["bits"]) for r in rows}

    sc = Scorer(threads=4)
    raw = np.memmap(SUB_DIR / "inflated/0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * SEQ_LEN, CAM_H, CAM_W, 3))
    cx1 = np.load(ARGMAX_CACHE / "cx1_argmax_n600.npy", mmap_mode="r")
    gtc = np.load(ARGMAX_CACHE / "gt_argmax_n600.npy", mmap_mode="r")

    for pair in PAIRS:
        z = np.load(FRAMES_DIR / f"pair_{pair:04d}_sq2_solved_frame.npz")
        band_flat, paint_vals = z["band_flat"], z["paint_rgb"]
        target_pose6 = z["target_pose6"]
        dec = np.stack([raw[SEQ_LEN * pair], raw[SEQ_LEN * pair + 1]]).astype(np.uint8)
        current = np.asarray(cx1[pair], dtype=np.uint8)
        target = np.asarray(gtc[pair], dtype=np.uint8)

        # GATE 0 — derivability: the shipped band must equal the decoder-derivable band
        derived = label_boundary_band(current, 1)
        derived_flat = np.flatnonzero(derived.reshape(-1))
        band_derivable = bool(np.array_equal(derived_flat, band_flat))
        print(f"[pq1] pair {pair}: band_derivable={band_derivable} n_band={band_flat.size}",
              flush=True)

        sh, sw = current.shape[-2], current.shape[-1]
        band_mask = np.zeros(sh * sw, dtype=bool)
        band_mask[band_flat] = True
        band_mask = band_mask.reshape(sh, sw)
        paint_img = np.zeros((sh, sw, 3), dtype=np.uint8)
        paint_img.reshape(-1, 3)[band_flat] = paint_vals

        base_flips = int((current != target).sum())
        # saliency: |paint - base scorer-plane value| (mean of D's 4 private camera px per m86)
        ii, jj = np.unravel_index(band_flat, (sh, sw))
        from ddm_sq1_stage_decomposition_and_solved_paint import ROW_SUP, COL_SUP  # noqa: E402
        base_at_band = np.zeros((band_flat.size, 3), dtype=np.float64)
        for a in range(2):
            for b in range(2):
                base_at_band += dec[1][ROW_SUP[ii, a], COL_SUP[jj, b]].astype(np.float64)
        base_at_band /= 4.0
        saliency = np.abs(paint_vals.astype(np.float64) - base_at_band).sum(axis=1)
        order = np.argsort(-saliency)

        d_pose_base = _d_pose(sc, dec, target_pose6)
        full_solved_flips = None
        for k in K_LADDER:
            keep = band_flat if k == 0 else band_flat[order[:min(k, band_flat.size)]]
            n_keep = int(keep.size)
            sub_mask = np.zeros(sh * sw, dtype=bool)
            sub_mask[keep] = True
            for bits in B_LADDER:
                if (pair, k, bits) in done:
                    continue
                t0 = time.time()
                pq = paint_img.reshape(-1, 3).copy()
                kept_vals = quantize(paint_vals[np.isin(band_flat, keep, assume_unique=True)]
                                     if k else paint_vals, bits)
                pq[keep] = kept_vals
                edited = realize_scorer_paint_to_camera(
                    dec[1], sub_mask.reshape(sh, sw), pq.reshape(sh, sw, 3))
                pair_u8 = np.stack([dec[0], edited])
                cell_argmax = sc.seg_argmax(pair_u8)
                cell_flips = int((cell_argmax != target).sum())
                if k == 0 and bits == 8:
                    full_solved_flips = cell_flips
                d_pose_cell = _d_pose(sc, pair_u8, target_pose6)
                vb = coded_value_bytes(kept_vals, bits)
                pb = subset_position_bytes(int(band_flat.size), n_keep)
                row = {
                    "pair": pair, "k": k, "bits": bits, "n_keep": n_keep,
                    "band_derivable": band_derivable,
                    "base_flips": base_flips, "cell_flips": cell_flips,
                    "flips_fixed": base_flips - cell_flips,
                    "d_pose_base": d_pose_base, "d_pose_cell": d_pose_cell,
                    "value_bytes_lzma": vb, "subset_pos_bytes_bound": round(pb, 1),
                    "total_bytes_est": round(vb + pb, 1),
                    "under_break_even": bool(vb + pb < BREAK_EVEN_B_PER_PAIR),
                    "sec": round(time.time() - t0, 1),
                }
                rows.append(row)
                print(f"[pq1] {row}", flush=True)
                OUT.write_text(json.dumps({
                    "schema": "ddm_pq1_prune_quantize.v1",
                    "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                    "score_claim": False,
                    "break_even_bytes_per_pair": BREAK_EVEN_B_PER_PAIR,
                    "note": "retention denominator = full-solve flips_fixed (k=0,b=8 row per pair)",
                    "rows": rows}, indent=1))
    print("[pq1] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
