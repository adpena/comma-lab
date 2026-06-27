#!/usr/bin/env python3
"""DAG FEED-fl (LENS C): coding-theory rate-slack measurement of the level-set witness byte-close.

CPU-only, deterministic, GPU UNTOUCHED. Reads a SETTLED (non-live) trained checkpoint and measures:
  (1) base-weight int8+brotli bytes vs the i.i.d. Shannon entropy floor  -> range/AC-coder slack
  (2) per-tensor bits/param                                              -> over-bit-allocated tensors
  (3) the per-frame `code` matrix effective rank (SVD)                   -> mod_dim over-width
  (4) post-hoc SVD factorization rate (fold Vt into film.weight, FREE)  -> the recoverable rate
  (5) temporal-delta-coded code bytes                                   -> PR95-L25 applicability
  (6) low-bit (int4..int7) entropy floors                               -> bit-allocation lever bound

NO-FAKE: every number is MEASURED via the SAME int8-symmetric+brotli-q11 grammar the canonical
byte-close (`tools/levelset_byte_close_and_eval.py` / `quantize_levelset_blob`) uses. macOS-CPU =
advisory; this characterizes the REPRESENTATION's rate structure, it is a MEANS (moves nothing).
"""
from __future__ import annotations
import sys
import numpy as np
import brotli

CKPT = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/witness_capstone_v3_n600/witness_ema_mlx.npz"


def i8(a, bits=8):
    a = np.asarray(a, np.float32)
    s = float(np.abs(a).max()) + 1e-8
    m = 2 ** (bits - 1) - 1
    q = np.clip(np.round(a / s * m), -m, m).astype(np.int16)
    return q, s / m


def bz(b):
    return len(brotli.compress(b, quality=11))


def ent(q):
    v, c = np.unique(q, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    d = np.load(CKPT, allow_pickle=True)
    base, code = {}, None
    for k in d.files:
        a = np.asarray(d[k])
        if (not np.issubdtype(a.dtype, np.number) or a.size <= 4
                or "::" in k or k.startswith(("cfg", "__"))):
            continue
        a = a.astype(np.float32)
        if k == "code" or k.endswith(".code"):
            code = a
        else:
            base[k] = a

    q8 = np.concatenate([i8(a, 8)[0].ravel() for a in base.values()])
    n = q8.size
    H8 = ent(q8)
    br8 = bz(q8.astype(np.int8).tobytes()) * 8 / n
    print(f"[BASE] params={n} iid_entropy={H8:.2f} b/p  brotli={br8:.2f} b/p  "
          f"AC_slack={br8 - H8:+.2f} b/p ({(br8 - H8) * n / 8 / 1024:.1f}KB)")
    for bits in (4, 5, 6, 7):
        qb = np.concatenate([i8(a, bits)[0].ravel() for a in base.values()])
        print(f"  PTQ int{bits}: AC_floor={ent(qb) * n / 8 / 1024:.1f}KB (DISTORTION lever)")

    if code is not None:
        qc = i8(code, 8)[0]
        nc = code.size
        brc = bz(qc.astype(np.int8).tobytes())
        print(f"[CODE] shape={code.shape} brotli={brc / 1024:.1f}KB "
              f"({brc * 8 / nc:.2f} b/s, entropy={ent(qc.ravel()):.2f})")
        C = code.reshape(code.shape[0], -1).astype(np.float64)
        C0 = C - C.mean(0)
        U, S, Vt = np.linalg.svd(C0, full_matrices=False)
        e = (S ** 2) / (S ** 2).sum()
        cum = np.cumsum(e)
        pr = (S ** 2).sum() ** 2 / (S ** 4).sum()
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        print(f"  SVD: PR(eff_rank)={pr:.1f}  50%@{np.searchsorted(cum, .5)+1} "
              f"90%@{k90}  99%@{np.searchsorted(cum, .99)+1}  (mod_dim={code.shape[1]} "
              f"-> {code.shape[1]/k90:.1f}x over)")
        basisB = bz(i8(Vt[:k90])[0].astype(np.int8).tobytes())
        coefB = bz(i8(U[:, :k90] * S[:k90])[0].astype(np.int8).tobytes())
        print(f"  factorized@k={k90}: basis={basisB/1024:.2f}KB + coeffs={coefB/1024:.2f}KB "
              f"= {(basisB+coefB)/1024:.2f}KB  vs {brc/1024:.1f}KB  "
              f"-> SAVE {(brc-basisB-coefB)/1024:.1f}KB ({100*(brc-basisB-coefB)/brc:.0f}%)")
        dlt = np.diff(qc.astype(np.int16), axis=0).astype(np.int8).tobytes()
        print(f"  temporal-delta brotli={bz(dlt)/1024:.1f}KB vs {brc/1024:.1f}KB direct "
              f"({100*(1-bz(dlt)/brc):+.0f}%; <0 => delta HURTS)")


if __name__ == "__main__":
    main()
