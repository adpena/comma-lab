#!/usr/bin/env python3
"""ddm_kl1 follow-up (coordinator feeds) — effective-quantum column (pi2 reframe)
+ exposure (a,b) stream characterization (pm1 QA44 rung-B, seed item 6).

Pointer 0.1910828242 UNMOVED; [macOS-CPU advisory]; score_claim=false. Cites
pi2 (1 f16 ULP @ dim0 ~= 0.040 S) and pm1 (QA44 rung-B -0.1039 S composed);
does NOT re-measure their d_pose. No PoseNet here.
"""
from __future__ import annotations

import json
import lzma
import math

import brotli
import numpy as np

_LZ = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 0}]


def best(b):
    return min(len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZ)),
              len(brotli.compress(b, quality=11)))


def byteplane(field_f16):
    cm = np.ascontiguousarray(field_f16.T).view(np.uint16).astype(np.uint16)
    hi = (cm >> 8).astype(np.uint8).reshape(-1)
    lo = (cm & 0xFF).astype(np.uint8).reshape(-1)
    return best(np.concatenate([hi, lo]).tobytes())


def f16_ulp(x):
    x = abs(float(x))
    if x == 0:
        return 2.0 ** -24
    return 2.0 ** (math.floor(math.log2(x)) - 10)


def main():
    D2 = "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl"
    QA44 = "/Volumes/VertigoDataTier/pact/ddm_qa44_20260730/photometric_rungs_probe.partial.jsonl"
    d2 = {json.loads(l)["pair"]: json.loads(l) for l in open(D2)}
    P = np.array([d2[p]["p_star"] for p in sorted(d2)], dtype=np.float64)

    out = {"schema": "ddm_kl1_quantum_and_exposure.v1",
           "pointer": "0.1910828242 [contest-CPU] UNMOVED",
           "axis": "[macOS-CPU advisory] NON-PROMOTABLE", "score_claim": False}

    # ---- (1) effective quantum per dim (pi2 distortion reframe) ----
    eq = []
    for j in range(6):
        c = P[:, j]
        mu = float(c.mean())
        resid = c - mu
        typ = np.abs(c).mean() if j == 0 else max(np.abs(c).max(), 1e-9)
        ulp_raw = f16_ulp(typ)
        ulp_off = f16_ulp(max(np.abs(resid).mean(), 1e-9))
        eq.append({"dim": j, "mean": mu, "raw_f16_ulp": ulp_raw,
                   "offset_resid_ulp": ulp_off,
                   "effective_bits_gained": round(math.log2(ulp_raw / max(ulp_off, 1e-30)), 2)})
    out["effective_quantum"] = {
        "per_dim": eq,
        "pi2_cite": "1 f16 ULP @ dim0 (|val|~32.6) ~= 0.040 S at operating point; dims1-5 f16-free",
        "mechanism": ("shared-mean + f16 residual (B3 shared-axis coding) stores dim0 at ~16x "
                      "finer effective quantum (4 bits) at ~same bytes (mean is 2 B shared/600). "
                      "DISTORTION lever, not just rate."),
        "realizability": ("REALIZABLE ONLY at a v4c RE-SOLVE that keeps dim0 above single-f16 "
                          "precision (the shipped p_star is ALREADY f16 -> offset-coding the "
                          "existing f16 is distortion-neutral). Potential recovery of pi2's ~0.040 "
                          "S/ULP -> ~0.03 S d_pose OWED to measure through PoseNet; NOT realized here."),
    }

    # ---- (2) exposure (a,b) stream (pm1 QA44 rung-B) ----
    rows = [json.loads(l) for l in open(QA44) if l.strip()]
    by = {r["pair"]: r for r in rows if "rungB_a" in r}
    pairs = sorted(by)
    a = np.array([by[p]["rungB_a"] for p in pairs])
    b = np.array([by[p]["rungB_b"] for p in pairs])
    af16 = a.astype(np.float16)
    bf16 = b.astype(np.float16)
    ab = np.stack([af16, bf16], axis=1)  # (n,2)
    n = len(pairs)
    out["exposure_ab_stream"] = {
        "pm1_cite": "QA44 rung-B 112/112 improvement, -0.1039 S composed with rung-A at <=4 B/pair",
        "available": {"n_pairs": n, "span": [pairs[0], pairs[-1]],
                      "contiguous_full600": False,
                      "note": "QA44 fit (a,b) only on the AIMED/tail pairs (scattered in time)"},
        "a_dist": {"mean": float(a.mean()), "std": float(a.std()),
                   "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
                   "frac_gain_near_1_(|a-1|<0.1)": float((np.abs(a - 1) < 0.1).mean())},
        "b_dist": {"mean": float(b.mean()), "std": float(b.std()),
                   "median": float(np.median(b)), "min": float(b.min()), "max": float(b.max())},
        "bytes_raw_f16": n * 2 * 2,
        "bytes_byteplane_lossless": byteplane(ab),
        "smooth_curve_law_testable": False,
        "verdict": ("The AE-control-loop smooth-curve law (AR(1)/spline vs raw 2.4KB) CANNOT be "
                    "tested here: QA44 produced (a,b) ONLY on ~112 SCATTERED tail pairs (not the "
                    "full-600 contiguous series), and those fits are NOISY (a in [%.2f,%.2f], b in "
                    "[%.1f,%.1f]) — a scattered noisy subset cannot exhibit temporal smoothness. "
                    "The 2.4KB full-600 (a,b) stream + its AR/spline race is OWED on a FULL-600 "
                    "rung-B fit (v4c build). Coding the tail-112 (a,b): raw %d B -> byteplane %d B."
                    % (a.min(), a.max(), b.min(), b.max(), n * 4, byteplane(ab))),
    }

    print(json.dumps(out, indent=1))
    with open("/Volumes/VertigoDataTier/pact/ddm_kl1_20260730/quantum_and_exposure.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
