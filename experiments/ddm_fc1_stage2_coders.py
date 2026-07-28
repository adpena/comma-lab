#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-FC1 STAGE 2 -- CORRECTION-STREAM REAL CODERS (compiled cores only; decode wall-clock timed).

Confirms the STAGE-1 tier-mover with a REAL coder, not an entropy proxy (iv4 A14: proxies != coder):

  (a) LABEL stream: constriction (Rust range coder) context-categorical over the 1,019,467 corrected
      labels, model = the decoder-derivable STAGE-1 cell frequencies (transmitted table, counted).
      Reports coded bytes, model-table bytes (real LZMA), encode+decode wall-clock, lossless roundtrip.
  (b) SUPPORT geometry: packbits + LZMA1-x9e over the 117.9M binary flip field (reproduces r2s's
      421,496 B measured floor) with decode wall-clock. The per-pixel context-arith floor (STAGE-1
      H_support=0.034 b/px -> ~499 KB) LOSES to LZMA because the flip field is spatially contiguous
      (contour/boundary coding, ~100-200 KB, is the named next lever -- NOT built here).

Decode wall-clock is checked against the 30-min (1800 s) CPU-4-core budget. `[macOS-CPU advisory]`.
"""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import time
from pathlib import Path

import numpy as np

import ddm_fc1_flip_entropy as S1  # context helpers + edges (same dir)

SCHEMA = "ddm_fc1_stage2_coders.v1"


def _context_at_flips(ctx_dir: Path, lstars_all: np.ndarray, max_pairs: int):
    """Return (cell_idx, labels) arrays over all flip sites, using the LABEL context cells."""
    chunks = sorted(ctx_dir.glob("ctx_*.npz"))
    cells: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    for ch in chunks:
        d = np.load(str(ch))
        c_arg = d["copy_argmax"]
        s0, s1 = int(d["start"]), int(d["end"])
        s1 = min(s1, max_pairs)
        if s0 >= max_pairs:
            continue
        ls = lstars_all[s0:s1]
        for i in range(s1 - s0):
            a = c_arg[i].astype(np.int64)
            l = ls[i]
            flip = a != l
            if not flip.any():
                continue
            bdist = S1._boundary_distance(a)
            adj = S1._adjacent_class(a)
            db = S1._bucket(bdist, S1.BDIST_EDGES)
            fa = a[flip]; fdb = db[flip]; fadj = adj[flip]; fl = l[flip]
            cell = (fa * S1.N_BDIST + fdb) * S1.N_CLASS + fadj
            cells.append(cell.astype(np.int32))
            labs.append(fl.astype(np.int32))
    return np.concatenate(cells), np.concatenate(labs)


def _code_labels(cell: np.ndarray, lab: np.ndarray) -> dict:
    import constriction

    n_cells = S1.N_CLASS * S1.N_BDIST * S1.N_CLASS
    counts = np.zeros((n_cells, S1.N_CLASS), dtype=np.float64)
    np.add.at(counts, (cell, lab), 1)
    smoothed = counts + 0.5  # KT smoothing -> strictly positive -> lossless
    cell_probs = smoothed / smoothed.sum(axis=1, keepdims=True)  # (n_cells,5)
    probs = cell_probs[cell]  # (N,5) per-symbol model

    model = constriction.stream.model.Categorical(perfect=False)
    t0 = time.time()
    enc = constriction.stream.queue.RangeEncoder()
    enc.encode(lab.astype(np.int32), model, probs)
    comp = enc.get_compressed()
    enc_s = time.time() - t0
    label_bytes = len(comp) * 4

    t1 = time.time()
    dec = constriction.stream.queue.RangeDecoder(comp)
    out = dec.decode(model, probs)
    dec_s = time.time() - t1
    lossless = bool((out == lab).all())

    # model table: transmit the integer counts of OCCUPIED cells (decoder rebuilds probs identically).
    occupied = counts.sum(axis=1) > 0
    table_payload = pickle.dumps({
        "cells": np.nonzero(occupied)[0].astype(np.int32),
        "counts": counts[occupied].astype(np.int64),
    })
    table_bytes = len(lzma.compress(table_payload, preset=9 | lzma.PRESET_EXTREME))

    return {
        "n_labels": int(lab.size),
        "label_coded_bytes": label_bytes,
        "label_bits_per_flip_REALCODER": label_bytes * 8.0 / lab.size,
        "model_table_bytes_lzma": table_bytes,
        "occupied_cells": int(occupied.sum()),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "lossless_roundtrip": lossless,
    }


def _code_support(ctx_dir: Path, lstars_all: np.ndarray, max_pairs: int) -> dict:
    """packbits + LZMA over the full binary flip field (reproduces r2s support-geometry floor)."""
    chunks = sorted(ctx_dir.glob("ctx_*.npz"))
    bit_parts: list[np.ndarray] = []
    total_sites = 0
    total_flips = 0
    for ch in chunks:
        d = np.load(str(ch))
        c_arg = d["copy_argmax"]
        s0, s1 = int(d["start"]), int(d["end"])
        s1 = min(s1, max_pairs)
        if s0 >= max_pairs:
            continue
        ls = lstars_all[s0:s1]
        flip = (c_arg[: s1 - s0].astype(np.int64) != ls)  # (m,384,512)
        bit_parts.append(np.packbits(flip.reshape(-1)))
        total_sites += flip.size
        total_flips += int(flip.sum())
    packed = np.concatenate(bit_parts)
    raw_bytes = packed.nbytes
    t0 = time.time()
    comp = lzma.compress(packed.tobytes(), format=lzma.FORMAT_RAW,
                         filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}])
    enc_s = time.time() - t0
    t1 = time.time()
    _ = lzma.decompress(comp, format=lzma.FORMAT_RAW,
                       filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}])
    dec_s = time.time() - t1
    return {
        "total_sites": total_sites,
        "total_flips": total_flips,
        "packbits_raw_bytes": raw_bytes,
        "support_coded_bytes_lzma": len(comp),
        "support_bits_per_pixel_REALCODER": len(comp) * 8.0 / total_sites,
        "encode_s": enc_s,
        "decode_s": dec_s,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx-dir", type=Path, required=True)
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=600)
    args = ap.parse_args(argv)

    t0 = time.time()
    lstars_all = np.load(str(args.gt_cache))["lstars"]
    print("[stage2] computing label context at flip sites...", flush=True)
    cell, lab = _context_at_flips(args.ctx_dir, lstars_all, args.max_pairs)
    print(f"[stage2] {lab.size} flip labels; coding with constriction...", flush=True)
    labels = _code_labels(cell, lab)
    print(f"[stage2] label coder: {labels['label_coded_bytes']} B, roundtrip={labels['lossless_roundtrip']}, dec={labels['decode_s']:.3f}s", flush=True)
    print("[stage2] coding support geometry (packbits+LZMA)...", flush=True)
    support = _code_support(args.ctx_dir, lstars_all, args.max_pairs)
    print(f"[stage2] support coder: {support['support_coded_bytes_lzma']} B, dec={support['decode_s']:.3f}s", flush=True)

    correction_total = labels["label_coded_bytes"] + labels["model_table_bytes_lzma"] + support["support_coded_bytes_lzma"]
    result = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory] REAL compiled coders (constriction Rust range coder + LZMA1); decode wall-clock timed; NOT a byte-closed evaluate.py row",
        "pairs": args.max_pairs,
        "labels": labels,
        "support": support,
        "correction_stream_total_bytes": correction_total,
        "decode_budget_s": 1800,
        "composed_decode_s_correction": labels["decode_s"] + support["decode_s"],
        "bar_bytes_0p172": 187_727,
        "bar_bytes_0p15": 154_522,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"[done] stage2 in {time.time()-t0:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
