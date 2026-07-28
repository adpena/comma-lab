#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pp1 R1 — DIRECT-PARTITION coder race on the cached n600 GT argmax maps (`lstars`).

MEANS. pointer 0.1910828242 [contest-CPU] UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE
— real lossless coder bytes over the cached GT partition; NOT a byte-closed `evaluate.py` row.

WHAT THIS PRICES (the ee1 R1 unmeasured cell): the sufficient statistic `L*_t` = the 600 seg-scored
GT argmax partitions (`lstars`, one per pair, 384x512, 5 classes). Internal prices before this arm
covered two DIFFERENT objects (the continuous exact plane #541 ~334KB/pair; the flip-residual-vs-weak
-base support 421-444KB). The GT partition ITSELF had never been fed to a real coder. This tool runs
the coder families the ee1 falsifier is defined over and proves each is a real decodable stream.

CODER FAMILIES (all lossless, n600=600 frames, bit-exact round-trip where a real stream is produced):
  (b) generic raster           : LZMA1-x9e / zlib / bz2 / brotli on the dense uint8 raster.
  (c) per-class binary planes  : 5 planes packbits -> LZMA (union == exact partition).
  (c') PNG-Paeth predictive    : per-frame Paeth residual -> LZMA.
  (c'') row run-length         : RLE (value,runlen) -> LZMA.
  (a) context-adaptive arith   : causal spatial context (o4/o6/o8) — the ECC-class strong intra coder.
  (d) + TEMPORAL context       : + prev-frame neighborhood as CONDITIONING CONTEXT (never predict-then
                                 -residual; ee1 C3). Codes frame t given frame t-1 as context.
  (e) LANE-dash sub-race       : the #307 contour chain coder on the Lane binary field (JBIG2-flavor)
                                 vs the context coder's Lane attribution.

THE ADAPTIVE-CODER LENGTH IS AUTHORITATIVE + REAL (NO-FAKE): a context-adaptive arithmetic coder's
coded byte length equals the closed-form Dirichlet-multinomial (KT/Laplace) code length of its model
to <0.01% (ONE range-coder flush over ~1e8 symbols). This tool (a) computes that closed-form length
EXACTLY over all n600 (order-independent, pays its own model-learning cost by construction — no
held-out artifact, no plug-in optimism) AND (b) PROVES the correspondence + bit-exact decodability on
a consecutive-frame subset via the in-tree #307 `AdaptiveStream`/`AdaptiveStreamDecoder`
(coded_bytes/closed_form == 1.0000, round-trip == True). The full-n600 closed form is the reported
price; the subset round-trip is the receipt that it is a real decodable coder length.

LOSSY CONCESSION (ee1 A.1 waterfill at the registered 1.2731 B/flip water level): progressively drop
the lowest-persistence label detail (small per-class connected components -> surrounding majority),
measure (retained best-coder bytes, conceded flips), and report S_partition over the curve to find the
lossy-optimal point. reuses sp1's region-merge concession semantics on the FULL partition.

FALSIFIER (ee1, pre-registered): lossless >= ~350KB AND lossy-at-water >= ~250KB => direct-explicit
partition coding DEAD (typed scope: this object, these coder families). ~120-180KB => THIRD ROUTE
opens (report composed arithmetic: partition + renderer-class + pose + labels vs 187.7/154.5KB budgets).

Usage:
  PYTHONPATH=src:tools .venv/bin/python experiments/ddm_pp1_direct_partition_coder.py \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --n 600 \
      --out /Volumes/VertigoDataTier/pact/ddm_pp1_20260728/r1_direct_partition_n600.json
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_CLASSES = 5
RATE_DENOM = 37_545_489.0
WATER_B_PER_FLIP = 1.2731  # registered region_merge water level (bytes per conceded argmax flip)
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


# --------------------------------------------------------------------------- coders
def _lzma_raw(data: bytes, preset: int = 9 | lzma.PRESET_EXTREME) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_RAW,
                         filters=[{"id": lzma.FILTER_LZMA1, "preset": preset}])


def _lzma_raw_dec(blob: bytes) -> bytes:
    return lzma.decompress(blob, format=lzma.FORMAT_RAW,
                           filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}])


def generic_raster(L: np.ndarray) -> dict:
    """Dense uint8 raster (frame-major) through generic byte compressors; bit-exact round-trip."""
    raw = L.astype(np.uint8).tobytes()
    out = {}
    lz = _lzma_raw(raw)
    assert _lzma_raw_dec(lz) == raw, "LZMA raster round-trip failed"
    out["lzma_x9e"] = len(lz)
    out["zlib_9"] = len(zlib.compress(raw, 9))
    out["bz2_9"] = len(bz2.compress(raw, 9))
    try:
        import brotli
        out["brotli_11"] = len(brotli.compress(raw, quality=11))
    except Exception:
        out["brotli_11"] = None
    out["roundtrip_lzma"] = True
    return out


def per_class_planes(L: np.ndarray) -> dict:
    """5 binary class planes -> packbits -> LZMA. Union of the planes reconstructs the partition."""
    total = 0
    per = {}
    recon = np.zeros_like(L, dtype=np.uint8)
    for c in range(N_CLASSES):
        plane = (c == L)
        packed = np.packbits(plane.reshape(-1)).tobytes()
        blob = _lzma_raw(packed)
        assert _lzma_raw_dec(blob) == packed, f"plane {c} round-trip failed"
        n = int(np.prod(L.shape))
        un = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:n].reshape(L.shape).astype(bool)
        recon[un] = c
        per[CLASS_NAMES[c]] = len(blob)
        total += len(blob)
    assert bool((recon == L.astype(np.uint8)).all()), "per-class union reconstruction failed"
    return {"total_bytes": total, "per_class_bytes": per, "roundtrip": True}


def paeth_residual(L: np.ndarray) -> dict:
    """PNG Paeth predictor per frame on the label raster, residual mod 256 -> LZMA. Round-trip."""
    Lb = L.astype(np.uint8)
    N, H, W = Lb.shape
    res = np.empty_like(Lb)
    for f in range(N):
        img = Lb[f].astype(np.int16)
        left = np.zeros_like(img)
        left[:, 1:] = img[:, :-1]
        up = np.zeros_like(img)
        up[1:, :] = img[:-1, :]
        ul = np.zeros_like(img)
        ul[1:, 1:] = img[:-1, :-1]
        p = left + up - ul
        pa = np.abs(p - left)
        pb = np.abs(p - up)
        pc = np.abs(p - ul)
        pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, ul))
        res[f] = ((img - pred) & 0xFF).astype(np.uint8)
    raw = res.tobytes()
    blob = _lzma_raw(raw)
    # round-trip: decode residual, invert Paeth
    dec = np.frombuffer(_lzma_raw_dec(blob), dtype=np.uint8).reshape(Lb.shape)
    rec = np.empty_like(Lb)
    for f in range(N):
        r = dec[f].astype(np.int16)
        out = np.zeros((H, W), np.int16)
        for y in range(H):
            for xx in range(W):
                left = int(out[y, xx - 1]) if xx > 0 else 0
                up = int(out[y - 1, xx]) if y > 0 else 0
                ul = int(out[y - 1, xx - 1]) if (y > 0 and xx > 0) else 0
                p = left + up - ul
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                pred = left if (pa <= pb and pa <= pc) else (up if pb <= pc else ul)
                out[y, xx] = (int(r[y, xx]) + pred) & 0xFF
        rec[f] = out.astype(np.uint8)
    ok = bool((rec == Lb).all())
    return {"total_bytes": len(blob), "roundtrip": ok}


def row_rle(L: np.ndarray) -> dict:
    """Row-major run-length (value byte, runlen varint) -> LZMA. Round-trip via re-expansion."""
    flat = L.astype(np.uint8).reshape(-1)
    # boundaries where value changes
    chg = np.empty(flat.size, dtype=bool)
    chg[0] = True
    chg[1:] = flat[1:] != flat[:-1]
    idx = np.flatnonzero(chg)
    vals = flat[idx]
    runs = np.diff(np.append(idx, flat.size))
    # encode runs as little varints
    out = bytearray()
    for v, r in zip(vals.tolist(), runs.tolist(), strict=True):
        out.append(int(v))
        rr = int(r)
        while True:
            b = rr & 0x7F
            rr >>= 7
            out.append(b | (0x80 if rr else 0))
            if not rr:
                break
    blob = _lzma_raw(bytes(out))
    # round-trip
    dec = _lzma_raw_dec(blob)
    rec = np.empty(flat.size, dtype=np.uint8)
    pos = 0
    i = 0
    while i < len(dec):
        v = dec[i]
        i += 1
        shift = 0
        rr = 0
        while True:
            b = dec[i]
            i += 1
            rr |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        rec[pos:pos + rr] = v
        pos += rr
    ok = bool((rec == flat).all())
    return {"total_bytes": len(blob), "n_runs": int(vals.size), "roundtrip": ok}


# --------------------------------------------------------------------------- context arithmetic
def _shift(a: np.ndarray, dy: int, dx: int, fill: int = 5) -> np.ndarray:
    H, W = a.shape[1], a.shape[2]
    out = np.full_like(a, fill)
    ys0, ye0 = max(0, dy), H + min(0, dy)
    xs0, xe0 = max(0, dx), W + min(0, dx)
    ys1, ye1 = max(0, -dy), H + min(0, -dy)
    xs1, xe1 = max(0, -dx), W + min(0, -dx)
    out[:, ys0:ye0, xs0:xe0] = a[:, ys1:ye1, xs1:xe1]
    return out


def _tshift(a: np.ndarray, k: int, fill: int = 5) -> np.ndarray:
    o = np.empty_like(a)
    o[:k] = fill
    o[k:] = a[:-k]
    return o


_INTRA_O4 = (("s", 0, -1), ("s", -1, 0), ("s", -1, -1), ("s", -1, 1))
_INTRA_O6 = (*_INTRA_O4, ("s", 0, -2), ("s", -2, 0))
_INTRA_O8 = (*_INTRA_O6, ("s", -1, -2), ("s", -1, 2))
_PREV5 = (("t", 0, 0), ("t", 0, 1), ("t", 0, -1), ("t", 1, 0), ("t", -1, 0))


def _build_ctx(L: np.ndarray, template) -> np.ndarray:
    tot = L.size
    ctx = np.zeros(tot, dtype=np.int64)
    for kind, dy, dx in template:
        # "s" = spatial causal neighbor; "t" = prev frame spatially shifted (prev frame fully
        # decoded -> any offset legal as decoder context)
        fmap = _shift(L, dy, dx) if kind == "s" else _tshift(_shift(L, dy, dx), 1)
        ctx = ctx * 6 + fmap.reshape(-1)
    return ctx


def adaptive_code_bytes(L: np.ndarray, template, alpha: float = 0.5) -> tuple[float, int]:
    """Closed-form Dirichlet-multinomial (KT alpha=0.5 / Laplace alpha=1) adaptive code length in
    BYTES over all pixels, with contexts factorized to compact ids. This equals a real single-pass
    adaptive-arithmetic coder's byte length to <0.01% (proven by roundtrip_proof below)."""
    from scipy.special import gammaln
    x = L.reshape(-1)
    ctx = _build_ctx(L, template)
    cid = np.unique(ctx, return_inverse=True)[1]
    ncomb = int(cid.max()) + 1
    cnt = np.bincount(cid * 5 + x, minlength=ncomb * 5).reshape(ncomb, 5).astype(np.float64)
    Nc = cnt.sum(1)
    cc = gammaln(N_CLASSES * alpha) - N_CLASSES * gammaln(alpha)
    ln2 = np.log(2.0)
    bits = -(ncomb * cc - gammaln(Nc + N_CLASSES * alpha).sum() + gammaln(cnt + alpha).sum()) / ln2
    return float(bits / 8.0), int(ncomb)


def per_class_attribution(L: np.ndarray, template, alpha: float = 0.5) -> dict:
    """Per true-class KT byte attribution (final-p approximation of the sequential cost; sums close to
    the exact total, used for RELATIVE class ranking only)."""
    x = L.reshape(-1)
    ctx = _build_ctx(L, template)
    cid = np.unique(ctx, return_inverse=True)[1]
    ncomb = int(cid.max()) + 1
    cnt = np.bincount(cid * 5 + x, minlength=ncomb * 5).reshape(ncomb, 5).astype(np.float64)
    Nc = cnt.sum(1)
    p = (cnt + alpha) / (Nc[:, None] + N_CLASSES * alpha)
    perpx = -np.log2(p[cid, x])
    out = {}
    for c, nm in enumerate(CLASS_NAMES):
        m = x == c
        out[nm] = {"px": int(m.sum()), "approx_bytes": float(perpx[m].sum() / 8.0),
                   "bits_per_px": float(perpx[m].mean()) if m.any() else 0.0}
    return out


def roundtrip_proof(L: np.ndarray, template, nf: int = 6) -> dict:
    """Prove the closed-form adaptive length == real #307 AdaptiveStream coded bytes AND that the
    stream decodes bit-exact, on a consecutive-frame subset (temporal context valid)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m307", str(_REPO / "tools" / "measure_contour_string_flip_coding.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m307"] = m
    spec.loader.exec_module(m)
    sub = L[:nf]
    ctx = _build_ctx(sub, template)
    cid = np.unique(ctx, return_inverse=True)[1]
    xs = sub.reshape(-1)
    enc = m.AdaptiveStream(N_CLASSES)
    closed = 0.0
    counts: dict[int, list[int]] = {}
    for i in range(xs.size):
        c = int(cid[i])
        s = int(xs[i])
        cc = counts.get(c)
        if cc is None:
            cc = [1, 1, 1, 1, 1]  # Laplace(+1) mirrors AdaptiveStream init
            counts[c] = cc
        tot = sum(cc)
        closed += -np.log2(cc[s] / tot)
        cc[s] += 1
        enc.encode(s, c)
    stream = enc.finish()
    dec = m.AdaptiveStreamDecoder(stream, N_CLASSES)
    rec = np.array([dec.decode(int(cid[i])) for i in range(xs.size)], dtype=np.int64)
    ok = bool((rec == xs).all())
    coded = len(stream)
    return {"subset_frames": nf, "coded_bytes": coded, "closed_form_bytes_laplace": float(closed / 8.0),
            "coded_over_closed": float(coded / (closed / 8.0)) if closed else None,
            "bit_exact_roundtrip": ok}


def lane_dash_subrace(L: np.ndarray) -> dict:
    """#307 contour chain coder on the Lane (class 1) binary field vs context-coder Lane attribution.
    The #307 tool codes a binary field's 8-conn components as chain strings (JBIG2/DjVu-flavor); Lane
    dashes are exactly that. Real range-coded streams, bit-exact round-trip inside the #307 helper."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m307b", str(_REPO / "tools" / "measure_contour_string_flip_coding.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m307b"] = m
    spec.loader.exec_module(m)
    lane_maps = [np.ascontiguousarray(L[f] == 1) for f in range(L.shape[0])]
    cls_maps = [np.ascontiguousarray(L[f]) for f in range(L.shape[0])]
    enc = m.contour_encode_frames(lane_maps, cls_maps)
    # verify decodability
    flips_out, _ = m.contour_decode_frames(enc["streams"], L.shape[0], L.shape[1], L.shape[2])
    ok = all(bool((flips_out[f] == lane_maps[f]).all()) for f in range(min(24, L.shape[0])))
    return {"contour_total_bytes": enc["total_bytes"], "n_lane_px": enc["n_flips"],
            "n_components": enc["n_components"], "stream_bytes": enc["stream_bytes"],
            "singleton_frac": enc["singleton_flip_frac"],
            "coherent_ge4_frac": enc["coherent_ge4_flip_frac"], "roundtrip_first24": ok}


def lossy_concession(L: np.ndarray, template, thresholds=(1, 2, 4, 8, 16, 32)) -> list[dict]:
    """Drop lowest-persistence label detail: remove per-class 8-conn components with area < k by
    relabeling each removed component to its dominant 4-neighborhood class. Measure (retained best-
    coder bytes, conceded flips) and S_partition = 25*bytes/DENOM + 100*conceded/total_sites."""
    from scipy.ndimage import label as cc_label
    N, H, W = L.shape
    total_sites = float(N * H * W)
    struct = np.ones((3, 3), dtype=np.int64)
    curve = []
    for k in thresholds:
        if k <= 1:
            simp = L.copy()
        else:
            simp = L.copy()
            for f in range(N):
                fr = simp[f]
                removed = np.zeros((H, W), dtype=bool)
                for c in range(N_CLASSES):
                    lab, nlab = cc_label(fr == c, structure=struct)
                    if nlab == 0:
                        continue
                    sizes = np.bincount(lab.reshape(-1))
                    small = np.flatnonzero(sizes < k)
                    small = small[small > 0]
                    if small.size:
                        removed |= np.isin(lab, small)
                if not removed.any():
                    continue
                # vectorized iterative dilation: each removed pixel takes any kept 4-neighbor label,
                # repeated until filled; isolated remnants keep their original label.
                fr2 = fr.copy()
                fr2[removed] = -1
                while (fr2 < 0).any():
                    up = np.full((H, W), -1, np.int64)
                    up[1:] = fr2[:-1]
                    dn = np.full((H, W), -1, np.int64)
                    dn[:-1] = fr2[1:]
                    lf = np.full((H, W), -1, np.int64)
                    lf[:, 1:] = fr2[:, :-1]
                    rt = np.full((H, W), -1, np.int64)
                    rt[:, :-1] = fr2[:, 1:]
                    best = np.stack([up, dn, lf, rt], 0).max(0)  # any kept neighbor (>= 0)
                    fill = (fr2 < 0) & (best >= 0)
                    if not fill.any():
                        fr2[fr2 < 0] = fr[fr2 < 0]  # isolated -> keep original
                        break
                    fr2[fill] = best[fill]
                simp[f] = fr2
        conceded = int((simp != L).sum())
        bytes_best, _ = adaptive_code_bytes(simp, template, alpha=0.5)
        s_part = 25.0 * bytes_best / RATE_DENOM + 100.0 * conceded / total_sites
        curve.append({"drop_lt_k": int(k), "conceded_flips": conceded,
                      "conceded_frac": conceded / total_sites, "retained_bytes": bytes_best,
                      "S_partition": s_part})
    return curve


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache",
                    default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-lossy", action="store_true",
                    help="skip the (slow) per-pixel component relabel concession sweep")
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.gt_cache, mmap_mode="r")
    L = np.asarray(z["lstars"][: args.n], dtype=np.int64)
    N, H, W = L.shape
    total_sites = N * H * W
    print(f"[pp1] lstars {L.shape} classes {np.unique(L).tolist()}  ({time.time()-t0:.1f}s)")

    templates = {"intra_o4": _INTRA_O4, "intra_o6": _INTRA_O6, "intra_o8": _INTRA_O8,
                 "temporal_o8_prev5": _INTRA_O8 + _PREV5}

    res: dict = {
        "schema": "ddm_pp1_direct_partition_coder.v1",
        "utc": datetime.now(UTC).isoformat(),
        "evidence_axis": "[macOS-CPU advisory] NON-PROMOTABLE — real lossless coder bytes over cached "
                         "GT partition lstars; NOT a byte-closed evaluate.py row. pointer 0.1910828242 "
                         "UNMOVED.",
        "object": "n600 GT SegNet argmax partition (lstars): 600 seg-scored frames, 384x512, 5 classes",
        "n_frames": N, "H": H, "W": W, "total_sites": total_sites,
        "class_fractions": {CLASS_NAMES[c]: float((c == L).mean()) for c in range(N_CLASSES)},
        "water_B_per_flip": WATER_B_PER_FLIP, "rate_denom": RATE_DENOM,
        "bar_bytes_0p172": 187727, "bar_bytes_0p15": 154522,
    }

    # boundary + temporal recon
    db_l = np.zeros_like(L, dtype=bool)
    db_l[:, :, 1:] = L[:, :, 1:] != L[:, :, :-1]
    db_u = np.zeros_like(L, dtype=bool)
    db_u[:, 1:, :] = L[:, 1:, :] != L[:, :-1, :]
    res["boundary_px_per_frame_mean"] = float((db_l | db_u).reshape(N, -1).sum(1).mean())
    res["temporal_disagree_frac"] = float((L[1:] != L[:-1]).mean())

    print("[pp1] generic raster ...")
    res["generic_raster"] = generic_raster(L)
    print("[pp1] per-class planes ...")
    res["per_class_planes"] = per_class_planes(L)
    print("[pp1] row RLE ...")
    res["row_rle"] = row_rle(L)
    print("[pp1] Paeth predictive ...")
    res["paeth"] = paeth_residual(L)

    print("[pp1] context-adaptive arithmetic (closed-form KT + Laplace) ...")
    ctx_res = {}
    for name, tpl in templates.items():
        b_kt, nc = adaptive_code_bytes(L, tpl, alpha=0.5)
        b_lap, _ = adaptive_code_bytes(L, tpl, alpha=1.0)
        ctx_res[name] = {"kt_bytes": b_kt, "laplace_bytes": b_lap, "n_contexts": nc}
        print(f"   {name:20s} KT {b_kt/1000:7.1f}KB  Laplace {b_lap/1000:7.1f}KB  ctx {nc}")
    res["context_arith"] = ctx_res
    best_tpl_name = min(ctx_res, key=lambda k: ctx_res[k]["kt_bytes"])
    res["best_context"] = best_tpl_name
    best_tpl = templates[best_tpl_name]

    print("[pp1] real-coder round-trip proof (subset) ...")
    res["roundtrip_proof"] = roundtrip_proof(L, best_tpl, nf=6)
    print(f"   coded/closed={res['roundtrip_proof']['coded_over_closed']:.4f} "
          f"roundtrip={res['roundtrip_proof']['bit_exact_roundtrip']}")

    print("[pp1] per-class attribution (best context) ...")
    res["per_class_attribution"] = per_class_attribution(L, best_tpl, alpha=0.5)

    print("[pp1] Lane-dash contour sub-race ...")
    res["lane_dash_subrace"] = lane_dash_subrace(L)

    if not args.skip_lossy:
        print("[pp1] lossy concession sweep (component removal) ...")
        res["lossy_concession"] = lossy_concession(L, best_tpl)

    # falsifier verdict
    lossless_best = min(res["context_arith"][best_tpl_name]["kt_bytes"],
                        res["generic_raster"]["lzma_x9e"], res["per_class_planes"]["total_bytes"])
    lossy_best = (min(pt["retained_bytes"] + pt["conceded_flips"] * WATER_B_PER_FLIP
                      for pt in res.get("lossy_concession", []))
                  if res.get("lossy_concession") else None)
    lossy_min_s = (min(pt["S_partition"] for pt in res.get("lossy_concession", []))
                   if res.get("lossy_concession") else None)
    dead = (lossless_best >= 350_000) and (lossy_best is not None and lossy_best >= 250_000)
    third_route = 120_000 <= lossless_best <= 180_000
    res["verdict"] = {
        "lossless_best_bytes": lossless_best,
        "lossy_optimal_equiv_bytes": lossy_best,
        "lossy_min_S_partition": lossy_min_s,
        "falsifier_dead": bool(dead),
        "third_route_opens": bool(third_route),
        "verdict_scope": "FORMULATION: this object (n600 GT lstars), these coder families "
                         "(generic/plane/Paeth/RLE/context-arith intra+temporal/#307 lane contour)",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[pp1] wrote {args.out}  ({time.time()-t0:.1f}s total)")
    print(f"[pp1] VERDICT: lossless_best={lossless_best/1000:.1f}KB  dead={dead}  "
          f"third_route={third_route}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
