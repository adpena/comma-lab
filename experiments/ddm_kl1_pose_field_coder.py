#!/usr/bin/env python3
"""ddm_kl1 LEG B1 — pose-field law coder (LOSSLESS at f16).

Charter (operator 2026-07-30): "Name everything we still store as a list of
numbers that is really one law plus noise. Code the law."

This codes the 600x6 f16 pose-warp field (RUNG P0 = pfs1 D2 p_star, and the
ck1 knee-base field) as {law predictor} + {entropy-coded residual}, races the
predictor FORM (generic-triple / constants-are-poison discipline: never assume
the law, MEASURE it), and PROVES bit-exact f16 round-trip (zero d_pose risk).

Pointer 0.1910828242 [contest-CPU] UNMOVED. All outputs
[macOS-CPU advisory], score_claim=false, research_only. No PoseNet here — this
is a pure lossless-coding measurement over the already-solved field values.
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
from dataclasses import asdict, dataclass

import numpy as np

try:
    import brotli
except Exception:  # pragma: no cover
    brotli = None

# e4/L24 coder stack: FORMAT_RAW + FILTER_LZMA1 dict_size=4096, lc=3, lp=0, pb=0
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 0}]


def lzma1_size(b: bytes) -> int:
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))


def brotli_size(b: bytes) -> int:
    if brotli is None:
        return 10**9
    return len(brotli.compress(b, quality=11))


def best_coder(b: bytes) -> tuple[int, str]:
    lz = lzma1_size(b)
    br = brotli_size(b)
    return (lz, "lzma1") if lz <= br else (br, "brotli")


# ---- f16 <-> total-order uint16 (monotonic key; lossless integer domain) ----
def f16_to_ordered(x: np.ndarray) -> np.ndarray:
    """IEEE total order: positive -> u|0x8000 ; negative -> ~u. Monotonic."""
    u = x.astype(np.float16).view(np.uint16).astype(np.uint32)
    neg = (u & 0x8000) != 0
    key = np.where(neg, (~u) & 0xFFFF, u | 0x8000).astype(np.int64)
    return key


def ordered_to_f16(key: np.ndarray) -> np.ndarray:
    key = key.astype(np.uint32) & 0xFFFF
    neg = (key & 0x8000) == 0  # after mapping, high bit 0 => was negative
    u = np.where(neg, (~key) & 0xFFFF, key & 0x7FFF).astype(np.uint16)
    return u.view(np.float16)


def _roundtrip_ok(field_f16: np.ndarray) -> bool:
    key = f16_to_ordered(field_f16)
    back = ordered_to_f16(key)
    return np.array_equal(back.view(np.uint16), field_f16.astype(np.float16).view(np.uint16))


# ---- residual serialization (int -> bytes, LOSSLESS, entropy-friendly) ----
def resid_to_bytes(resid: np.ndarray) -> bytes:
    """Zigzag varint-ish: split into low byte + high plane so the entropy coder
    sees mostly-zero high bytes. resid can be any int (int64). We store as two
    int16 planes after zigzag on int32 (residuals fit int32 for f16 fields)."""
    r = resid.astype(np.int64)
    # zigzag map to unsigned
    z = ((r << 1) ^ (r >> 63)).astype(np.uint64)
    assert (z < (1 << 32)).all(), "residual exceeds 32 bits"
    z32 = z.astype(np.uint32)
    b0 = (z32 & 0xFF).astype(np.uint8)
    b1 = ((z32 >> 8) & 0xFF).astype(np.uint8)
    b2 = ((z32 >> 16) & 0xFF).astype(np.uint8)
    b3 = ((z32 >> 24) & 0xFF).astype(np.uint8)
    # plane-split (all b0, then all b1, ...) so near-constant high planes RLE
    return np.concatenate([b0, b1, b2, b3]).tobytes()


# ---- predictors (return prediction in ordered-int space, + coeff bytes) ----
def pred_zero(key_col: np.ndarray) -> tuple[np.ndarray, int, str]:
    """No predictor: 'residual' = value itself (pure alphabet coding)."""
    return np.zeros_like(key_col), 0, "none"


def pred_delta1(key_col: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = np.empty_like(key_col)
    p[0] = 0
    p[1:] = key_col[:-1]
    return p, 0, "delta1"


def pred_delta2(key_col: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = np.empty_like(key_col)
    p[0] = 0
    p[1] = key_col[0]
    p[2:] = 2 * key_col[1:-1] - key_col[:-2]
    return p, 0, "delta2"


def pred_poly(key_col: np.ndarray, deg: int, colf: np.ndarray) -> tuple[np.ndarray, int, str]:
    """Low-order polynomial in t on the FLOAT column, round to f16 -> ordered.
    Ships (deg+1) float32 coeffs."""
    n = len(colf)
    t = np.arange(n) / max(n - 1, 1)
    c = np.polyfit(t, colf, deg)
    predf = np.polyval(c, t)
    p = f16_to_ordered(predf.astype(np.float16))
    return p, 4 * (deg + 1), f"poly{deg}"


@dataclass
class ColResult:
    dim: int
    predictor: str
    coeff_bytes: int
    resid_coder: str
    resid_bytes: int
    total_bytes: int


def code_column(colf: np.ndarray, dim: int) -> tuple[ColResult, dict]:
    """Race predictors x coders on one column; return the best (lossless)."""
    key = f16_to_ordered(colf.astype(np.float16))
    candidates = [pred_zero(key), pred_delta1(key), pred_delta2(key)]
    for deg in (1, 2, 3, 5):
        candidates.append(pred_poly(key, deg, colf))
    best = None
    per_pred = {}
    for pred, coeff_b, name in candidates:
        resid = key - pred
        # lossless check for this predictor
        recon = ordered_to_f16((pred + resid).astype(np.int64))
        assert np.array_equal(
            recon.view(np.uint16), colf.astype(np.float16).view(np.uint16)
        ), f"predictor {name} not lossless on dim {dim}"
        rb = resid_to_bytes(resid)
        sz, coder = best_coder(rb)
        total = sz + coeff_b
        per_pred[name] = total
        cr = ColResult(dim, name, coeff_b, coder, sz, total)
        if best is None or total < best.total_bytes:
            best = cr
    return best, per_pred


def load_field(path: str, key: str, n_expect: int | None) -> np.ndarray:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    # dedup by pair, keep last (resume-tolerant), then sort by pair
    by_pair = {}
    for r in rows:
        by_pair[r["pair"]] = r
    ordered = [by_pair[p] for p in sorted(by_pair)]
    P = np.array([r[key] for r in ordered], dtype=np.float64)
    if n_expect is not None and len(P) != n_expect:
        print(f"[warn] loaded {len(P)} rows, expected {n_expect}", file=sys.stderr)
    return P.astype(np.float16).astype(np.float64), sorted(by_pair)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--key", default="p_star")
    ap.add_argument("--n-expect", type=int, default=None)
    ap.add_argument("--label", default="field")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P, pairs = load_field(args.jsonl, args.key, args.n_expect)
    n, d = P.shape
    Pf16 = P.astype(np.float16)
    assert _roundtrip_ok(Pf16), "ordered-int mapping not lossless!"

    raw = Pf16.view(np.uint16).tobytes()  # row-major 2*n*d
    raw_bytes = len(raw)

    # ---- no-law generic controls ----
    row_major = Pf16.copy().view(np.uint16)
    col_major = np.ascontiguousarray(Pf16.T).view(np.uint16)
    controls = {
        "raw": raw_bytes,
        "row_major_lzma1": lzma1_size(row_major.tobytes()),
        "row_major_brotli": brotli_size(row_major.tobytes()),
        "col_major_lzma1": lzma1_size(col_major.tobytes()),
        "col_major_brotli": brotli_size(col_major.tobytes()),
    }
    # byte-plane split, column-major (hi byte plane then lo byte plane)
    cm = np.ascontiguousarray(Pf16.T).view(np.uint16).astype(np.uint16)  # (d, n)
    hi = (cm >> 8).astype(np.uint8)
    lo = (cm & 0xFF).astype(np.uint8)
    plane = np.concatenate([hi.reshape(-1), lo.reshape(-1)]).tobytes()
    controls["byteplane_colmajor_lzma1"] = lzma1_size(plane)
    controls["byteplane_colmajor_brotli"] = brotli_size(plane)

    # ---- per-column law race (LOSSLESS) ----
    col_results = []
    per_pred_all = {}
    law_total = 0
    for j in range(d):
        cr, per_pred = code_column(P[:, j], j)
        col_results.append(cr)
        per_pred_all[f"dim{j}"] = per_pred
        law_total += cr.total_bytes
    # add small framing: 1 byte predictor id per column + coeff already counted
    law_total_framed = law_total + d  # d bytes predictor-id header

    # also: single-stream residual coding (all columns' best-predictor residuals
    # concatenated column-major, then ONE coder pass -- often beats per-col)
    best_pred_names = [cr.predictor for cr in col_results]
    all_resid = []
    coeff_bytes = 0
    for j in range(d):
        key = f16_to_ordered(P[:, j].astype(np.float16))
        name = best_pred_names[j]
        if name == "none":
            pred = np.zeros_like(key)
        elif name == "delta1":
            pred, _, _ = pred_delta1(key)
        elif name == "delta2":
            pred, _, _ = pred_delta2(key)
        elif name.startswith("poly"):
            deg = int(name[4:])
            pred, cb, _ = pred_poly(key, deg, P[:, j])
            coeff_bytes += cb
        all_resid.append(key - pred)
    all_resid = np.concatenate(all_resid)
    single_stream = resid_to_bytes(all_resid)
    ss_sz, ss_coder = best_coder(single_stream)
    single_stream_total = ss_sz + coeff_bytes + d

    result = {
        "schema": "ddm_kl1_pose_field_coder.v1",
        "label": args.label,
        "jsonl": args.jsonl,
        "key": args.key,
        "n": n,
        "d": d,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "lossless_verified": True,
        "raw_bytes": raw_bytes,
        "controls_no_law": controls,
        "per_column_best": [asdict(cr) for cr in col_results],
        "per_column_law_total_framed": law_total_framed,
        "single_stream_total": single_stream_total,
        "single_stream_coder": ss_coder,
        "per_predictor_per_dim": per_pred_all,
        "best_overall": min(
            [
                ("row_major_lzma1", controls["row_major_lzma1"]),
                ("row_major_brotli", controls["row_major_brotli"]),
                ("col_major_lzma1", controls["col_major_lzma1"]),
                ("col_major_brotli", controls["col_major_brotli"]),
                ("byteplane_colmajor_lzma1", controls["byteplane_colmajor_lzma1"]),
                ("byteplane_colmajor_brotli", controls["byteplane_colmajor_brotli"]),
                ("per_column_law", law_total_framed),
                ("single_stream_law", single_stream_total),
            ],
            key=lambda kv: kv[1],
        ),
    }
    js = json.dumps(result, indent=1)
    print(js)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)


if __name__ == "__main__":
    main()
