#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_da1 D1 -- SUPPORT GEOMETRY 421 KB decomposition (TEMPORAL + PER-CLASS + STRUCTURAL).

Decomposes the fc1 support-geometry wall (421,366 B LZMA over the 117.9M binary flip field) into:
  (a) PER-CLASS support bytes: each flip owned by its TRUE label (lstars) -> 5 binary planes coded
      separately with the SAME LZMA1-x9e-FORMAT_RAW settings fc1 used. Which class owns the bytes?
  (b) CROSS-PAIR conditional coding: baseline LZMA(concat flip[t]) reproduces the 421 KB; then
      delta LZMA(concat flip[t] XOR flip[t-1]); then g4-style static-frequency predictor residual
      LZMA(flip XOR (freqmap>0.5)) + the one-time freqmap cost. REAL coder bytes.
      FALSIFIER: conditional >= 421 KB => temporal redundancy not exploitable at mask granularity.

Reuses cached copy_argmax (SegNet on copy base f0) + lstars (GT argmax). No SegNet re-run.
`[macOS-CPU advisory]` -- real LZMA1 coder bytes; NOT a byte-closed evaluate.py row.
"""
from __future__ import annotations

import argparse
import json
import lzma
import time
from pathlib import Path

import numpy as np

SCHEMA = "ddm_da1_d1_support_decomp.v1"
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
FC1_SUPPORT_LZMA_BYTES = 421366  # stage2 n600 baseline we must reproduce
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}]


def _lzma_len(packed: np.ndarray) -> int:
    comp = lzma.compress(packed.tobytes(), format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    return len(comp)


def _load_flips(ctx_dir: Path, gt_cache: Path, max_pairs: int):
    gt = np.load(str(gt_cache))
    lstars = gt["lstars"]  # (600,384,512) int64
    chunks = sorted(ctx_dir.glob("ctx_*.npz"))
    argmax_list, flip_list = [], []
    for ch in chunks:
        d = np.load(str(ch))
        c_arg = d["copy_argmax"]  # (m,384,512) uint8
        s0, s1 = int(d["start"]), int(d["end"])
        s1 = min(s1, max_pairs)
        if s0 >= max_pairs:
            continue
        ls = lstars[s0:s1].astype(np.uint8)
        ca = c_arg[: s1 - s0]
        argmax_list.append(ca)
        flip_list.append(ca != ls)
    argmax = np.concatenate(argmax_list, axis=0)  # (P,384,512) uint8
    flip = np.concatenate(flip_list, axis=0)  # (P,384,512) bool
    ls_all = lstars[: argmax.shape[0]].astype(np.uint8)
    return argmax, flip, ls_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx-dir", type=Path, required=True)
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=600)
    args = ap.parse_args(argv)

    t0 = time.time()
    argmax, flip, lstars = _load_flips(args.ctx_dir, args.gt_cache, args.max_pairs)
    P, H, W = flip.shape
    total_sites = P * H * W
    total_flips = int(flip.sum())
    print(f"[d1] loaded P={P} flips={total_flips} frac={total_flips/total_sites:.8f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- baseline: reproduce fc1 421 KB (concat packbits of all flip masks) ----
    baseline_packed = np.packbits(flip.reshape(-1))
    baseline_bytes = _lzma_len(baseline_packed)
    print(f"[d1] baseline concat-LZMA = {baseline_bytes} B (fc1={FC1_SUPPORT_LZMA_BYTES})", flush=True)

    # ---- (a) PER-CLASS support bytes (owned by TRUE label lstars at flip sites) ----
    per_class = {}
    class_sum = 0
    for k in range(5):
        plane = flip & (lstars == k)  # (P,384,512) bool
        b = _lzma_len(np.packbits(plane.reshape(-1)))
        fk = int(plane.sum())
        per_class[CLASS_NAMES[k]] = {
            "flips": fk,
            "lzma_bytes": b,
            "bytes_per_flip": (b / fk) if fk else 0.0,
            "share_of_flips": fk / max(1, total_flips),
        }
        class_sum += b
        print(f"[d1a] {CLASS_NAMES[k]:11s} flips={fk:8d} lzma={b:7d} B  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[d1a] sum-of-per-class={class_sum} vs joint={baseline_bytes} (separation overhead={class_sum-baseline_bytes})", flush=True)

    # ---- (b) CROSS-PAIR conditional coding ----
    # delta: flip[0] as-is; flip[t]^flip[t-1] for t>=1
    delta = flip.copy()
    delta[1:] = flip[1:] ^ flip[:-1]
    delta_bytes = _lzma_len(np.packbits(delta.reshape(-1)))
    print(f"[d1b] XOR-delta concat-LZMA = {delta_bytes} B ({time.time()-t0:.0f}s)", flush=True)

    # g4-style static frequency predictor: freqmap = per-pixel flip frequency across P
    freqmap = flip.mean(axis=0)  # (384,512) float
    freq_total = None
    freq_detail = {}
    for thr in (0.5,):
        pred = freqmap > thr  # (384,512) bool
        resid = flip ^ pred[None, :, :]
        resid_bytes = _lzma_len(np.packbits(resid.reshape(-1)))
        # one-time cost to transmit the freqmap predictor as a binary plane (freq>thr)
        predmap_bytes = _lzma_len(np.packbits(pred.reshape(-1)))
        freq_total = resid_bytes + predmap_bytes
        freq_detail = {"thr": thr, "resid_bytes": resid_bytes, "predmap_bytes": predmap_bytes, "total": freq_total}
        print(f"[d1b] freq>{thr} residual-LZMA = {resid_bytes} B + predmap {predmap_bytes} B = {freq_total} ({time.time()-t0:.0f}s)", flush=True)

    result = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory] REAL LZMA1-x9e-FORMAT_RAW coder bytes over cached copy-base flip field; NOT a byte-closed evaluate.py row",
        "pairs": P,
        "total_sites": total_sites,
        "total_flips": total_flips,
        "baseline_joint_lzma_bytes": baseline_bytes,
        "fc1_reference_bytes": FC1_SUPPORT_LZMA_BYTES,
        "reproduces_fc1": abs(baseline_bytes - FC1_SUPPORT_LZMA_BYTES) < 2000,
        "D1a_per_class": per_class,
        "D1a_sum_per_class_bytes": class_sum,
        "D1a_separation_overhead_bytes": class_sum - baseline_bytes,
        "D1b_temporal": {
            "baseline_independent_bytes": baseline_bytes,
            "xor_delta_bytes": delta_bytes,
            "xor_delta_vs_baseline": delta_bytes / baseline_bytes,
            "staticfreq_detail": freq_detail,
            "staticfreq_residual_plus_predmap_bytes": freq_total,
            "staticfreq_vs_baseline": freq_total / baseline_bytes,
            "FALSIFIER_conditional_ge_baseline": min(delta_bytes, freq_total) >= baseline_bytes,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[d1] done ({time.time()-t0:.0f}s) -> {args.out}", flush=True)
    print(json.dumps(result["D1b_temporal"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
