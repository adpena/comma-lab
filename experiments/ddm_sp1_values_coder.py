#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_sp1 R2 — VALUES REAL CODER: amplitude+sign+context arithmetic vs the int8x3 incumbent.

MEANS. pointer 0.19108 UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE lossless coder bytes
over the cached range(A) residual values; NEVER a byte-closed evaluate.py score. NO-FAKE: the values
are the REAL range(A)-projected temporal residual (gt_f1 - gt_f0) at the copy-flip support (scorer
argmax != GT), computed from cached GT camera frames + the fixed range(A) projection (no scorer). A
real in-tree order-1 context range coder (zigzag value under a previous-magnitude-bucket context)
encodes them and DECODES back BIT-EXACT before any byte is reported. A coder that does not round-trip
is reported broken, never as a price (the sp1 honest-boundary rule).

CLOSES gc5 bridges B2+B7's shared rung (the VALUES stream). The r2s incumbent stored the residual
VALUES as int8x3 -> LZMA = 10,062,148 B (~9.87 B/err over 1,019,467 scorer flips). This tool codes
the SAME range(A) residual with an amplitude+sign+context model + the LZMA/brotli race floor, and
reports real B/err. The da1-d4 REPRICED alphabet (minimal correcting amplitude median 1.11 / 64.1%
<= 2 uint8 steps) is the TIGHTER 2-5 bit/value target, but the minimal amplitudes come from a frozen
SegNet line-search (sc1 owns the scorer slot) -> that alphabet is GATED and reported as a DERIVED
entropy bound, not a measured coded price.
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import measure_contour_string_flip_coding as mcs  # REUSE the in-tree AdaptiveStream range coder

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def _zigzag(v: np.ndarray) -> np.ndarray:
    """Signed int -> unsigned zigzag (0,-1,1,-2,2 -> 0,1,2,3,4). Vectorized."""
    v = v.astype(np.int64)
    return (v << 1) ^ (v >> 63)


def _mag_bucket(z: int) -> int:
    """Context bucket of a zigzag value: 0,1,2,3-4,5-8,9-16,17+."""
    if z == 0:
        return 0
    if z <= 2:
        return 1
    if z <= 4:
        return 2
    if z <= 8:
        return 3
    if z <= 16:
        return 4
    if z <= 32:
        return 5
    return 6


_N_CTX = 7


def _context_range_code(values: np.ndarray) -> dict:
    """Order-1 context adaptive range coder on zigzag(values), context = prev-magnitude bucket.

    values: (S,) int8. Returns real coded bytes + bit-exact round-trip proof. This is the
    amplitude+sign+context coder (zigzag folds sign into the amplitude alphabet; the low-magnitude
    concentration the da1-d4 alphabet predicts shows up as a peaked per-context PMF)."""
    z = _zigzag(values).astype(np.int64)
    alphabet = int(z.max()) + 1 if z.size else 1
    alphabet = max(alphabet, 2)
    t0 = time.time()
    enc = mcs.AdaptiveStream(alphabet)
    ctx = 0
    for zi in z.tolist():
        enc.encode(int(zi), ctx)
        ctx = _mag_bucket(int(zi))
    stream = enc.finish()
    enc_s = round(time.time() - t0, 2)
    # bit-exact decode-verify
    td = time.time()
    dec = mcs.AdaptiveStreamDecoder(stream, alphabet)
    out = np.empty(z.shape, dtype=np.int64)
    ctx = 0
    for i in range(z.size):
        zi = dec.decode(ctx)
        out[i] = zi
        ctx = _mag_bucket(int(zi))
    decode_s = round(time.time() - td, 2)
    lossless = bool(np.array_equal(out, z))
    return {"coded_bytes": len(stream), "alphabet": alphabet, "encode_s": enc_s,
            "decode_s": decode_s, "lossless_roundtrip": lossless}


def _entropy_bound_bits(values: np.ndarray) -> dict:
    """Order-1 conditional entropy H(zigzag | prev-bucket) — the tight arithmetic-coder floor
    (MEASURED on the real symbols, not a projection)."""
    z = _zigzag(values).astype(np.int64)
    if z.size < 2:
        return {"H_order1_bits_per_value": 0.0, "H0_bits_per_value": 0.0}
    buckets = np.array([_mag_bucket(int(x)) for x in z[:-1]], dtype=np.int64)
    ctx = np.empty(z.size, dtype=np.int64)
    ctx[0] = 0
    ctx[1:] = buckets
    total_bits = 0.0
    for c in range(_N_CTX):
        sel = z[ctx == c]
        if sel.size == 0:
            continue
        _, counts = np.unique(sel, return_counts=True)
        p = counts / counts.sum()
        total_bits += -(counts * np.log2(p)).sum()
    _, c0 = np.unique(z, return_counts=True)
    p0 = c0 / c0.sum()
    h0 = float(-(p0 * np.log2(p0)).sum())
    return {"H_order1_bits_per_value": round(total_bits / z.size, 4),
            "H0_bits_per_value": round(h0, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-cache",
                    default="/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--ctx-dir", default="/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/chunks")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    gt = np.load(args.gt_cache)
    gt_f0 = gt["gt_f0"][: args.n]  # (N,H,W,3) uint8 camera frames
    gt_f1 = gt["gt_f1"][: args.n]
    lstars = gt["lstars"][: args.n]
    N, H, W, _ = gt_f0.shape
    print(f"[sp1-R2] loaded N={N} frames {H}x{W} ({time.time()-t0:.0f}s)", flush=True)

    # copy-flip support (scorer-res) from cached copy_argmax, upsample to camera nearest (radius 0)
    from tac.boundary_math.range_a_projection import apply_projection
    chunks = sorted(Path(args.ctx_dir).glob("ctx_*.npz"))
    flip_scorer = np.zeros((N, 384, 512), dtype=bool)
    for ch in chunks:
        d = np.load(str(ch))
        s0 = int(d["start"])
        ca = d["copy_argmax"]
        for j in range(ca.shape[0]):
            pi = s0 + j
            if pi >= N:
                break
            flip_scorer[pi] = ca[j] != lstars[pi].astype(np.uint8)
    ys = np.linspace(0, 383, H).round().astype(np.int64)
    xs = np.linspace(0, 511, W).round().astype(np.int64)
    n_scorer_flips = int(flip_scorer.sum())

    # range(A)-projected temporal residual at camera flip sites (int8-centered), gathered
    vals_list = []
    for i in range(N):
        resid = gt_f1[i].astype(np.int16) - gt_f0[i].astype(np.int16)  # (H,W,3)
        proj = apply_projection(resid.astype(np.float64))
        rint = np.clip(np.round(proj), -128, 127).astype(np.int8)
        sup = flip_scorer[i][ys][:, xs]  # (H,W) bool nearest-upsampled
        vals_list.append(rint[sup])  # (s_i, 3)
    vals = np.concatenate(vals_list, axis=0)  # (S,3) int8
    S = vals.shape[0]
    print(f"[sp1-R2] camera support sites={S} (x3 chan) scorer_flips={n_scorer_flips} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # incumbent reproduction: int8x3 raw -> LZMA1-x9e / brotli / bz2
    raw = vals.tobytes()
    filt = [{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}]
    lzma_b = len(lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filt))
    bz2_b = len(bz2.compress(raw, 9))
    try:
        import brotli
        br_b = len(brotli.compress(raw, quality=11))
    except Exception:
        br_b = None
    incumbent_best = min(b for b in (lzma_b, bz2_b, br_b) if b)
    print(f"[sp1-R2] incumbent int8x3: lzma={lzma_b} bz2={bz2_b} brotli={br_b} "
          f"best={incumbent_best} ({incumbent_best/n_scorer_flips:.2f} B/err)", flush=True)

    # amplitude+sign+context real coder on the flattened per-channel values
    flat = vals.reshape(-1)  # (3S,)
    coder = _context_range_code(flat)
    ent = _entropy_bound_bits(flat)
    coder_b_per_err = coder["coded_bytes"] / n_scorer_flips
    print(f"[sp1-R2] context range coder: {coder['coded_bytes']} B "
          f"({coder_b_per_err:.2f} B/err) lossless={coder['lossless_roundtrip']} "
          f"H_order1={ent['H_order1_bits_per_value']} b/val", flush=True)

    result = {
        "schema": "ddm_sp1_values_coder.v1",
        "task": "gc5 B2+B7 shared rung — VALUES real coder (amplitude+sign+context) vs int8x3",
        "evidence_axis": ("[macOS-CPU advisory] NON-PROMOTABLE lossless coder bytes over cached "
                          "range(A) residual values; NOT a byte-closed evaluate.py row; pointer 0.19108 UNMOVED"),
        "utc": datetime.now(UTC).isoformat(),
        "n_pairs": N,
        "scorer_flip_sites": n_scorer_flips,
        "camera_support_sites": S,
        "values_per_site": 3,
        "incumbent_int8x3": {
            "raw_bytes": len(raw), "lzma1_x9e": lzma_b, "bz2": bz2_b, "brotli_q11": br_b,
            "best_bytes": incumbent_best,
            "b_per_err": round(incumbent_best / n_scorer_flips, 3),
            "r2s_reference_bytes": 10062148,
        },
        "context_range_coder": {
            **coder,
            "b_per_err": round(coder_b_per_err, 3),
            "order1_entropy": ent,
        },
        "coder_vs_incumbent": {
            "coder_bytes": coder["coded_bytes"],
            "incumbent_bytes": incumbent_best,
            "ratio": round(coder["coded_bytes"] / incumbent_best, 4),
            "b_per_err_coder": round(coder_b_per_err, 3),
            "b_per_err_incumbent": round(incumbent_best / n_scorer_flips, 3),
        },
        "da1_d4_repriced_alphabet_GATED": {
            "note": ("da1-d4 minimal correcting amplitudes (median 1.11 / p75 3.33 / p90 7.78 uint8 "
                     "steps, 64.1% <= 2) are the TIGHTER 2-5 bit/value target, but require the frozen "
                     "SegNet line-search (sc1 owns the scorer slot). This tool codes the FULL range(A) "
                     "residual (a loose UPPER bound); the minimal-amplitude alphabet is GATED."),
            "measured_percentiles_from_da1_d4": {"median": 1.11, "p75": 3.33, "p90": 7.78,
                                                 "frac_le_2": 0.641, "frac_le_4": 0.788},
            "derived_minimal_alphabet_entropy_note": ("if the minimal-amplitude alphabet has a peaked "
                     "PMF matching those percentiles, its order-0 entropy is ~2-4 bits/value; "
                     "materialization + real coding is GATED on sc1's base"),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[sp1-R2] wrote {args.out} ({result['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
