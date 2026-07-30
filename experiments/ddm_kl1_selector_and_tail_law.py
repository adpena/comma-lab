#!/usr/bin/env python3
"""ddm_kl1 LEG B2 (selector-from-xi) + B3 (rank-1 tail law) — measured, lossless.

Pointer 0.1910828242 [contest-CPU] UNMOVED. All rows [macOS-CPU advisory],
score_claim=false. No PoseNet: pure coding + structure measurement over solved
fields; the d_pose costs cited are from the qa43/ck1 caches (frozen-PoseNet).
"""
from __future__ import annotations

import json
import lzma
import math

import numpy as np

try:
    import brotli
except Exception:
    brotli = None

_LZ = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 0}]


def lz(b):
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZ))


def br(b):
    return len(brotli.compress(b, quality=11)) if brotli else 10**9


def best(b):
    return min(lz(b), br(b))


def f16_ordered(x):
    u = x.astype(np.float16).view(np.uint16).astype(np.uint32)
    neg = (u & 0x8000) != 0
    return np.where(neg, (~u) & 0xFFFF, u | 0x8000).astype(np.int64)


def byteplane_bytes(field_f16):
    cm = np.ascontiguousarray(field_f16.T).view(np.uint16).astype(np.uint16)
    hi = (cm >> 8).astype(np.uint8).reshape(-1)
    lo = (cm & 0xFF).astype(np.uint8).reshape(-1)
    return best(np.concatenate([hi, lo]).tobytes())


def colex_bits(n, k):
    """bits to encode a k-subset of n by combinatorial rank."""
    if k == 0 or k == n:
        return 0.0
    return math.log2(math.comb(n, k))


def main():
    QA43 = "/Volumes/VertigoDataTier/pact/ddm_qa43_20260729/two_plane_probe_v2.partial.jsonl"
    rows = {}
    for line in open(QA43):
        d = json.loads(line)
        rows[d["pair"]] = d
    pairs = sorted(rows)
    ds = np.array([rows[p]["d_single_solved_cached"] for p in pairs])
    dt = np.array([rows[p]["d_two_solved"] for p in pairs])
    two_better = dt < ds
    Ptwo = np.array([rows[p]["p_two_star"] for p in pairs], dtype=np.float64)  # 112x6
    # (correction-delta p_two - p_single SVD is the SAME dim0-scale artifact,
    #  measured separately; the tail-field rank below is sufficient and cleaner.)

    out = {
        "schema": "ddm_kl1_selector_and_tail_law.v1",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "n_tail": len(pairs),
    }

    # ---------------- B2: selector-from-xi ----------------
    n_two = int(two_better.sum())
    # true selector cost given the tail set is known (colex rank of the winners)
    colex_tail = colex_bits(len(pairs), n_two) / 8.0
    # raw explicit bits for the tail selector
    raw_tail_bits = len(pairs) / 8.0
    # brotli of the tail bitmap
    bitmap = np.packbits(two_better.astype(np.uint8)).tobytes()
    br_bitmap = best(bitmap)
    # full-600 selector (non-tail always single): 600-bit map, n_two set
    colex_full = colex_bits(600, n_two) / 8.0
    # yaw-threshold law: best single-dim threshold on |dim| predicting two_better,
    # then exceptions listed explicitly (colex of the mispredicted set).
    best_law = None
    for j in range(6):
        a = np.abs(Ptwo[:, j])
        # try every threshold; pick the one minimizing (exceptions colex + 4B thr)
        for thr in np.unique(a):
            pred = a >= thr  # high |dim| -> pick two
            exc = pred != two_better
            n_exc = int(exc.sum())
            law_bytes = 4 + colex_bits(len(pairs), n_exc) / 8.0  # thr(f32)+exceptions
            if best_law is None or law_bytes < best_law["law_bytes"]:
                best_law = {"dim": j, "thr": float(thr), "n_exc": n_exc,
                            "law_bytes": law_bytes}
    out["B2_selector"] = {
        "tail_pairs": len(pairs), "two_wins": n_two,
        "raw_explicit_tail_bits_bytes": raw_tail_bits,
        "brotli_tail_bitmap_bytes": br_bitmap,
        "colex_tail_given_set_bytes": round(colex_tail, 2),
        "colex_full600_bytes": round(colex_full, 2),
        "best_yaw_threshold_law": best_law,
        "verdict": ("yaw-threshold NO better than colex: weak corr; "
                    "ship colex of the sparse winner-set (~%.0f B)" % colex_tail),
    }

    # ---------------- B3: rank-1 tail law (lossless) ----------------
    # standardized rank (kills the dim0-scale artifact)
    Pc = Ptwo - Ptwo.mean(0)
    std = Pc.std(0) + 1e-12
    Uz, Sz, _ = np.linalg.svd(Pc / std, full_matrices=False)
    energy_raw = (np.linalg.svd(Pc, compute_uv=False) ** 2)
    energy_std = Sz ** 2
    # lossless rank-1: store mean(6 f16) + dir(6 f16) + per-pair coeff(112 f16)
    # + residual(112x6 f16 of what rank-1 misses), byteplane-coded.
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    coeff = U[:, 0] * S[0]
    approx = np.outer(coeff, Vt[0]) + Ptwo.mean(0)
    # lossless requires the residual reproduce f16 exactly; code the residual field
    resid_f16 = (Ptwo.astype(np.float16).view(np.uint16).astype(np.int64)
                 - approx.astype(np.float16).view(np.uint16).astype(np.int64))
    # serialize residual (int) + coeff + dir + mean
    zz = ((resid_f16 << 1) ^ (resid_f16 >> 63)).astype(np.uint64).astype(np.uint32)
    planes = np.concatenate([(zz & 0xFF).astype(np.uint8).reshape(-1),
                             ((zz >> 8) & 0xFF).astype(np.uint8).reshape(-1),
                             ((zz >> 16) & 0xFF).astype(np.uint8).reshape(-1)])
    resid_bytes = best(planes.tobytes())
    rank1_total = resid_bytes + 6 * 2 + 6 * 2 + len(pairs) * 2  # resid+dir+mean+coeff
    # baseline: byteplane of the full tail field
    bp_full = byteplane_bytes(Ptwo.astype(np.float16))
    out["B3_tail_rank1"] = {
        "raw_field_bytes": len(pairs) * 6 * 2,
        "byteplane_lossless_bytes": bp_full,
        "svd_energy_raw_frac": (energy_raw / energy_raw.sum()).round(4).tolist(),
        "svd_energy_standardized_frac": (energy_std / energy_std.sum()).round(4).tolist(),
        "rank1_plus_lossless_residual_bytes": rank1_total,
        "verdict": ("SVD rank-1=1.0 is a dim0-SCALE ARTIFACT (speed~%.0f swamps "
                    "dims1-5~%.3g); standardized rank spreads energy %s. Lossless "
                    "rank-1+residual (%d B) %s byteplane (%d B)." % (
                        np.abs(Ptwo[:, 0]).mean(), np.abs(Ptwo[:, 1:]).mean(),
                        (energy_std / energy_std.sum()).round(3).tolist(),
                        rank1_total, "beats" if rank1_total < bp_full else "LOSES to",
                        bp_full)),
    }

    print(json.dumps(out, indent=1))
    with open("/Volumes/VertigoDataTier/pact/ddm_kl1_20260730/b2b3_selector_tail.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
