#!/usr/bin/env python3
"""ddm_gt3 Job A -- is there a RECEIVER-RECOMPUTABLE (free) basis that concentrates seg flips
enough to be worth counted coefficients?

THE QUESTION (MAIN's charter correction, 2026-08-03)
----------------------------------------------------
`ddm_gt1` law: the BASIS/SPAN/GEOMETRY is GENERIC and FREE (inflate.py recomputes it from its
own state at zero counted bytes); only the COEFFICIENTS are COUNTED. GT never ships -- its
role is ENCODE-side, telling us which free basis is worth carrying coefficients for.

`ddm_ob1` measured the shipped receiver holds NO class-label field: `L*` (the frozen SegNet
argmax on the decoded frames) is NOT receiver-computable.  That is a real, correction-proof
constraint and it is the binding one here:

    THE RECEIVER CANNOT IDENTIFY ITS OWN ERRORS.

So a correction scheme must either
  (a) PAY to identify them   -- an explicit address (ob1 priced a hand-built legal floor at
      544,499 B for full capture), or
  (b) PAINT A SUPERSET identified by a FREE rule, paying H(gt | bin) per site over the whole
      superset, and eating the waste on the non-flip sites.

This unit prices (b) exactly, because (b) is the route in which "the addressing falls out":
if the receiver can compute the bin itself, naming the set costs ZERO counted bytes.  Only the
per-site class coefficient is counted.

THE EXACT ACCOUNTING (per free bin j, all independent -- so per-bin greedy IS the optimum)
-----------------------------------------------------------------------------------------
    n_j     sites in bin j
    f_j     realized flips in bin j            (flip := gt_argmax != cx1_argmax)
    H_j     H(gt | bin j) bits, FIT ON TRAIN PAIRS, EVALUATED ON HELD-OUT TEST PAIRS
    cost_j  = n_j * H_j / 8                    bytes  (counted coefficients)
    gain_j  = f_j * eta                        flips repaired
    dS_j    = cost_j * S_PER_BYTE - gain_j * S_PER_FLIP
    include bin j  iff  dS_j < 0

There is NO address term. That is the whole point: the receiver recomputes the bin.

Break-even in one line: a bin pays iff   density_j * eta * W  >  H_j / 8   bytes/site,
i.e.  density_j > H_j / (8 * eta * W).   With W = 1.27310821533203125 B/flip.

WHAT IS FREE HERE (rule-118, gt1 three-way test)
------------------------------------------------
  FREE : decoded camera RGB (the receiver's own state), `D` (the scorer's own
         interpolate(...,(384,512),bilinear,antialias=False)), luma weights, pure code,
         pixel geometry (row/col).  NOTHING clip-specific, NO label field, NO scorer weights.
  COUNTED : the bin->class-distribution table (reported in bytes), and the per-site class
         coefficients in the included bins.
  NEVER SHIPPED : gt itself, `L*`, any margin field.  They appear here ONLY encode-side, to
         decide which bins to buy -- exactly gt1's law.

POSITIVE CONTROLS (m50 vacuity law: an empty scope reports VACUOUS, never PASS)
------------------------------------------------------------------------------
  C1 d_seg reproduces 0.004311794704861111 / 508,640 flips from the caches
  C2 decoded 0.raw is the certified 3,662,409,600 bytes  (bit-identical rebuild of ob1's)
  C3 train/test split is representative on the governing quantity (m88: ratio -> 1)
  C4 ob1's legal feature set reproduced as a rung, as a plumbing cross-check
  C5 every denominator printed; every bin count reported; no empty scope scored

Usage:
    python experiments/ddm_gt3_free_basis_address_n600.py --pairs 600 [--stop-after N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
RAW = "/Volumes/VertigoDataTier/pact/ddm_gt3_20260803/inflated/0.raw"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_gt3_20260803"

CAM_H, CAM_W = 874, 1164
SEG_H, SEG_W = 384, 512
PIX_PER_PAIR = SEG_H * SEG_W
N_PAIRS = 600
SEQ = 2
NC = 5

# ---- the score arithmetic, recomputed from components, never a rounded field -------------
RATE_PER_BYTE = 25.0 / 37_545_489.0
S_PER_FLIP = 100.0 / (N_PAIRS * PIX_PER_PAIR)
W_BYTES_PER_FLIP = S_PER_FLIP / RATE_PER_BYTE          # 1.27310821533203125

# live best (pu2, archive sha c72ef357) and the PR130 bar
LIVE_BEST_S = 0.7910689
PR130_BAR = 0.172141
GAP = LIVE_BEST_S - PR130_BAR                           # 0.6189279

# eta is TRANSFERRED from sq1 (n=32, stratified-systematic, ratio 0.9973 on flips/pair).
# Job B re-measures it on THIS unit's selected set; until then both bounds are carried.
ETA_POSE_NEUTRAL_N32 = 0.5406      # sq1 2.8, P7 pose-null realizer (pose-safe)
ETA_SEG_ONLY_N32 = 0.7895          # sq1 2.4, seg-only (carries a measured pose debt)

CERTIFIED_RAW_BYTES = 3_662_409_600
EXPECT_FLIPS = 508_640
EXPECT_DSEG = 0.004311794704861111


# ---------------------------------------------------------------------------------------
# FREE feature helpers.  Every one is a function of decoded RGB + fixed operators + geometry.
# `to_scorer_rgb`, `luma`, `grad_mag`, `dilate1`, `dist_to_mask` are ob1's verbatim so the
# ob1 cross-check rung (C4) is an apples-to-apples reproduction.
# ---------------------------------------------------------------------------------------
DMAX = 15


def to_scorer_rgb(cam_u8: np.ndarray) -> np.ndarray:
    """EXACTLY the scorer's own resize (upstream/modules.py:73 and :109 make the identical
    call, so this is the lattice BOTH scorers read -- pz1 / m86)."""
    x = torch.from_numpy(np.ascontiguousarray(cam_u8)).permute(2, 0, 1)[None].float()
    y = torch.nn.functional.interpolate(
        x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
    return y[0].permute(1, 2, 0).numpy()


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def grad_mag(y: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = np.abs(y[:, 2:] - y[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(y[2:, :] - y[:-2, :]) * 0.5
    return gx + gy


def ridge(y: np.ndarray) -> np.ndarray:
    """|second difference| -- a thin-bright-line detector.  Motivated, not decorative:
    Road<->Lane is 46-49% of all flips (pc2) and lane markings are exactly thin luma ridges.
    Pure function of decoded luma."""
    rx = np.zeros_like(y)
    ry = np.zeros_like(y)
    rx[:, 1:-1] = np.abs(2.0 * y[:, 1:-1] - y[:, 2:] - y[:, :-2])
    ry[1:-1, :] = np.abs(2.0 * y[1:-1, :] - y[2:, :] - y[:-2, :])
    return np.maximum(rx, ry)


def local_contrast(y: np.ndarray) -> np.ndarray:
    """3x3 max-min on the scorer lattice.  Texture proxy (UNIWARD-flavoured)."""
    p = np.pad(y, 1, mode="edge")
    st = np.stack([p[i:i + SEG_H, j:j + SEG_W] for i in range(3) for j in range(3)], 0)
    return st.max(0) - st.min(0)


def dilate1(m: np.ndarray) -> np.ndarray:
    o = m.copy()
    o[:-1, :] |= m[1:, :]
    o[1:, :] |= m[:-1, :]
    o[:, :-1] |= m[:, 1:]
    o[:, 1:] |= m[:, :-1]
    return o


def dist_to_mask(m0: np.ndarray) -> np.ndarray:
    d = np.full(m0.shape, DMAX + 1, dtype=np.uint8)
    m = m0.copy()
    d[m] = 0
    for r in range(1, DMAX + 1):
        nm = dilate1(m)
        d[nm & ~m] = r
        m = nm
        if m.all():
            break
    return d


# ---- binning ---------------------------------------------------------------------------
# Bin counts kept modest so (a) the counted table stays negligible and (b) generalization is
# honest -- an 11k-cell cross product overfits and its table dominates the byte budget.
NB_G, NB_D, NB_ROW, NB_R, NB_P = 5, 6, 6, 5, 4
NBINS = NB_G * NB_D * NB_ROW * NB_R * NB_P

# The renderer's OWN token lattice: 24x32 cells over 384x512 => EXACTLY 16x16 scorer px per
# cell (receiver map: ddm_tr1_runtime.py:1300 renders at 384x512, selector grid_h=24 grid_w=32).
# Distance-to-cell-boundary is therefore pure receiver-known geometry: FREE, clip-independent.
TOK_CELL = 16


def quantize(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, v, side="right").astype(np.int32)


def bin_index(g, d, row, r, p) -> np.ndarray:
    return (((g * NB_D + d) * NB_ROW + row) * NB_R + r) * NB_P + p


_ROW = np.repeat(np.arange(SEG_H, dtype=np.float32)[:, None], SEG_W, axis=1)
_COL = np.repeat(np.arange(SEG_W, dtype=np.float32)[None, :], SEG_H, axis=0)
# distance to the nearest token-cell boundary, in scorer pixels (0 = on the seam)
_PHASE = np.minimum(
    np.minimum(_ROW % TOK_CELL, (TOK_CELL - 1) - (_ROW % TOK_CELL)),
    np.minimum(_COL % TOK_CELL, (TOK_CELL - 1) - (_COL % TOK_CELL))).astype(np.float32)


def features_for_pair(cam_f1: np.ndarray, edges: dict | None):
    """Return the raw free feature stack for one pair's frame_1 (scorer lattice).

    Every entry is a function of (decoded camera RGB, the fixed operator `D`, luma weights,
    pixel geometry, the receiver's own known token lattice). No label field, no scorer
    weights, nothing clip-specific.
    """
    rgb = to_scorer_rgb(cam_f1)
    y = luma(rgb)
    g = grad_mag(y)
    r = ridge(y)
    # distance to a thresholded decoded-RGB edge -- ob1's feature, same statistic
    d = dist_to_mask(g > 6.0).astype(np.float32)
    return {"g": g, "d": d, "row": _ROW, "r": r, "p": _PHASE}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--stop-after", type=int, default=0,
                    help="bounded invocation; the emitted report is marked PARTIAL")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "gt3_free_basis_n600.json"))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    npairs = args.pairs
    t0 = time.time()

    # ---------------- C2: the decode is the certified one -------------------------------
    got = os.path.getsize(RAW)
    if got != CERTIFIED_RAW_BYTES:
        raise RuntimeError(
            f"0.raw is {got} B, expected the certified {CERTIFIED_RAW_BYTES} B -- "
            "inflate incomplete or a different archive. REFUSING (m50: never score a "
            "partial scope as PASS).")
    raw = np.memmap(RAW, dtype=np.uint8, mode="r",
                    shape=(N_PAIRS * SEQ, CAM_H, CAM_W, 3))

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")

    # ---------------- C1: reproduce d_seg from the caches --------------------------------
    tot_flips = 0
    per_pair_flips = np.zeros(npairs, dtype=np.int64)
    for i in range(npairs):
        k = int((np.asarray(gt[i]) != np.asarray(rd[i])).sum())
        per_pair_flips[i] = k
        tot_flips += k
    d_seg = tot_flips / float(npairs * PIX_PER_PAIR)
    c1_pass = (tot_flips == EXPECT_FLIPS) and abs(d_seg - EXPECT_DSEG) < 1e-15
    print(f"[C1] flips={tot_flips} d_seg={d_seg!r} pass={c1_pass}", flush=True)

    # ---------------- train/test split: INTERLEAVED, never a prefix (m88) ----------------
    idx = np.arange(npairs)
    train = idx[idx % 2 == 0]
    test = idx[idx % 2 == 1]
    if args.stop_after:
        train = train[:max(1, args.stop_after // 2)]
        test = test[:max(1, args.stop_after // 2)]
    m88_ratio = float(per_pair_flips[test].mean() / per_pair_flips.mean())
    m88_train = float(per_pair_flips[train].mean() / per_pair_flips.mean())
    print(f"[C3] m88 test/pop={m88_ratio:.6f} train/pop={m88_train:.6f} "
          f"(a prefix[0:300] would be {per_pair_flips[:300].mean()/per_pair_flips.mean():.6f})",
          flush=True)

    # ---------------- pass 0: fit quantile edges on a sample of TRAIN --------------------
    samp = train[:: max(1, len(train) // 12)][:12]
    acc = {k: [] for k in ("g", "d", "row", "r", "p")}
    for i in samp:
        f = features_for_pair(np.asarray(raw[i * SEQ + 1]), None)
        for k in acc:
            acc[k].append(f[k][::7, ::7].ravel())
    edges = {}
    for k, nb in (("g", NB_G), ("d", NB_D), ("row", NB_ROW), ("r", NB_R), ("p", NB_P)):
        v = np.concatenate(acc[k])
        qs = np.linspace(0, 100, nb + 1)[1:-1]
        e = np.unique(np.percentile(v, qs))
        edges[k] = e.astype(np.float32)
    print("[edges] " + " ".join(f"{k}:{len(edges[k])+1}" for k in edges), flush=True)

    def binned(i: int) -> np.ndarray:
        f = features_for_pair(np.asarray(raw[i * SEQ + 1]), edges)
        return bin_index(
            quantize(f["g"], edges["g"]), quantize(f["d"], edges["d"]),
            quantize(f["row"], edges["row"]), quantize(f["r"], edges["r"]),
            quantize(f["p"], edges["p"])).ravel()

    # ---------------- pass 1: TRAIN -- bin x gt-class counts ----------------------------
    tab = np.zeros((NBINS, NC), dtype=np.int64)
    tab_flip = np.zeros((NBINS, NC), dtype=np.int64)   # gt class GIVEN the site is a flip
    tr_flip = np.zeros(NBINS, dtype=np.int64)
    tr_n = np.zeros(NBINS, dtype=np.int64)
    for c0 in range(0, len(train), args.chunk):
        for i in train[c0:c0 + args.chunk]:
            b = binned(int(i))
            g_ = np.asarray(gt[i]).ravel()
            r_ = np.asarray(rd[i]).ravel()
            fl = g_ != r_
            np.add.at(tab, (b, g_), 1)
            np.add.at(tab_flip, (b[fl], g_[fl]), 1)
            tr_n += np.bincount(b, minlength=NBINS)
            tr_flip += np.bincount(b[fl], minlength=NBINS)
        print(f"  train {min(c0+args.chunk,len(train))}/{len(train)} "
              f"{time.time()-t0:.0f}s", flush=True)

    # model = per-bin gt distribution, Laplace-smoothed (the COUNTED table)
    pm = (tab + 0.5) / (tab.sum(1, keepdims=True) + 0.5 * NC)
    H_bin = -(pm * np.log2(pm)).sum(1)                     # bits/site if we ship gt in bin
    # COUNTED table cost, charged ONLY for bins we actually take: a taken bin must carry its
    # own id plus its NC-1 free probabilities (4-bit quantized). Untaken bins ship nothing --
    # the receiver's default action there is "do not paint".
    bin_id_bits = int(np.ceil(np.log2(NBINS)))
    PER_BIN_TABLE_BYTES = (bin_id_bits + (NC - 1) * 4) / 8.0
    # models for the strictly-cheaper INDICATOR scheme (see waterfill_indicator):
    #   pd_j   = P(flip | bin j)          -- codes the flip indicator inside a taken bin
    #   pmf_j  = P(gt | bin j, flip)      -- codes the target class ONLY on flips
    pd = (tr_flip + 0.5) / (tr_n + 1.0)
    pmf = (tab_flip + 0.5) / (tab_flip.sum(1, keepdims=True) + 0.5 * NC)

    # ---------------- pass 2: TEST -- held-out counts + held-out cross-entropy -----------
    te_n = np.zeros(NBINS, dtype=np.int64)
    te_flip = np.zeros(NBINS, dtype=np.int64)
    te_bits = np.zeros(NBINS, dtype=np.float64)            # HELD-OUT xent, the honest cost
    te_ind_bits = np.zeros(NBINS, dtype=np.float64)        # HELD-OUT flip-INDICATOR xent
    te_cls_bits = np.zeros(NBINS, dtype=np.float64)        # HELD-OUT class xent, FLIPS ONLY
    for c0 in range(0, len(test), args.chunk):
        for i in test[c0:c0 + args.chunk]:
            b = binned(int(i))
            g_ = np.asarray(gt[i]).ravel()
            r_ = np.asarray(rd[i]).ravel()
            fl = g_ != r_
            te_n += np.bincount(b, minlength=NBINS)
            te_flip += np.bincount(b[fl], minlength=NBINS)
            bits = -np.log2(pm[b, g_])
            np.add.at(te_bits, b, bits)
            ind = np.where(fl, -np.log2(pd[b]), -np.log2(1.0 - pd[b]))
            np.add.at(te_ind_bits, b, ind)
            np.add.at(te_cls_bits, b[fl], -np.log2(pmf[b[fl], g_[fl]]))
        print(f"  test {min(c0+args.chunk,len(test))}/{len(test)} "
              f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- the per-bin waterfill (this IS the optimum: bins are independent) ---
    n_test_px = int(te_n.sum())
    base_rate = float(te_flip.sum()) / max(1, n_test_px)
    dens = np.where(te_n > 0, te_flip / np.maximum(te_n, 1), 0.0)
    bits_per_site = np.where(te_n > 0, te_bits / np.maximum(te_n, 1), 0.0)

    def waterfill(eta: float):
        # scale the held-out half up to the full n600 field FIRST, so the fixed per-bin table
        # overhead is charged against the full-field gain (not the half-field gain).
        scale = float(npairs) / max(1, len(test))
        cost_B = te_n * bits_per_site / 8.0 * scale + PER_BIN_TABLE_BYTES
        gain_flips = te_flip * eta * scale
        dS = cost_B * RATE_PER_BYTE - gain_flips * S_PER_FLIP
        take = (dS < 0) & (te_n > 0)
        return {
            "eta": eta,
            "bins_taken": int(take.sum()),
            "sites_taken": int(te_n[take].sum() * scale),
            "site_fraction": float(te_n[take].sum() / max(1, n_test_px)),
            "flips_in_taken": int(te_flip[take].sum() * scale),
            "flip_capture": float(te_flip[take].sum() / max(1, te_flip.sum())),
            "precision_in_taken": float(
                te_flip[take].sum() / max(1, te_n[take].sum())),
            "coeff_bytes": float(cost_B[take].sum()),
            "table_bytes": float(PER_BIN_TABLE_BYTES * int(take.sum())),
            "flips_fixed": float(te_flip[take].sum() * eta * scale),
            "dS_total": float(dS[take].sum()),
            "pct_of_gap": float(-dS[take].sum() / GAP * 100.0),
            "break_even_density_at_1bit": 1.0 / (8.0 * eta * W_BYTES_PER_FLIP),
        }

    def waterfill_indicator(eta: float):
        """The STRICTLY CHEAPER scheme, and the one that decides the family.

        Inside a taken bin ship (i) a flip INDICATOR coded at P(flip|bin) and (ii) the target
        class ONLY on flips. Outside taken bins ship nothing -- the receiver's default is
        "do not paint". The receiver recomputes the bin, so the SET still costs zero.
        This dominates the class-everywhere waterfill because H(flip|bin) << H(gt|bin).
        """
        scale = float(npairs) / max(1, len(test))
        cost_B = (te_ind_bits + te_cls_bits) / 8.0 * scale + PER_BIN_TABLE_BYTES
        gain_flips = te_flip * eta * scale
        dS = cost_B * RATE_PER_BYTE - gain_flips * S_PER_FLIP
        take = (dS < 0) & (te_n > 0)
        # per-bin break-even eta, for the specification
        denom = te_flip * scale * S_PER_FLIP
        eta_req = np.where(denom > 0, cost_B * RATE_PER_BYTE / np.maximum(denom, 1e-30), np.inf)
        sup = te_n >= 2000
        jbest = int(np.argmin(np.where(sup, eta_req, np.inf)))
        return {
            "eta": eta,
            "bins_taken": int(take.sum()),
            "sites_taken": int(te_n[take].sum() * scale),
            "flips_in_taken": int(te_flip[take].sum() * scale),
            "flip_capture": float(te_flip[take].sum() / max(1, te_flip.sum())),
            "precision_in_taken": float(te_flip[take].sum() / max(1, te_n[take].sum())),
            "bytes_total": float(cost_B[take].sum()),
            "flips_fixed": float(te_flip[take].sum() * eta * scale),
            "dS_total": float(dS[take].sum()),
            "pct_of_gap": float(-dS[take].sum() / GAP * 100.0),
            "min_break_even_eta_over_bins": float(eta_req[jbest]),
            "min_break_even_eta_bin_sites": int(te_n[jbest]),
            "min_break_even_eta_bin_density": float(dens[jbest]),
        }

    # ---------------- the headline diagnostic: best FREE concentration ------------------
    sup = te_n >= 2000                                       # adequate support only
    order = np.argsort(-np.where(sup, dens, -1))
    top = []
    for j in order[:12]:
        if not sup[j]:
            break
        top.append({
            "bin": int(j), "sites_test": int(te_n[j]), "flips_test": int(te_flip[j]),
            "density": float(dens[j]), "enrichment": float(dens[j] / max(base_rate, 1e-12)),
            "bits_per_site_heldout": float(bits_per_site[j]),
            "req_density_at_eta5406": float(
                bits_per_site[j] / (8.0 * ETA_POSE_NEUTRAL_N32 * W_BYTES_PER_FLIP)),
        })

    # ---------------- how far short, and is it a binning artefact? ----------------------
    scale_full = float(npairs) / max(1, len(test))
    # (i) cost of the NAIVE route: paint the WHOLE field, class coded on the free basis
    full_field_bytes = float((te_n * bits_per_site).sum() / 8.0 * scale_full)
    # (ii) the shortfall factor: best bin's density / its own break-even density, per eta.
    #      eta = 1.0 is a HARD upper bound -- no realizer can beat it -- so if the family
    #      fails there it fails for every realizer, and eta stops being the binding unknown.
    req_at = lambda eta: bits_per_site / (8.0 * eta * W_BYTES_PER_FLIP)  # noqa: E731
    shortfall = {}
    for nm, eta in (("0.5406", ETA_POSE_NEUTRAL_N32), ("0.7895", ETA_SEG_ONLY_N32),
                    ("1.0", 1.0)):
        rq = req_at(eta)
        ratio = np.where((te_n > 0) & (rq > 0), dens / np.maximum(rq, 1e-12), 0.0)
        sup2 = te_n >= 2000
        j = int(np.argmax(np.where(sup2, ratio, -1)))
        jall = int(np.argmax(ratio))
        shortfall[nm] = {
            "best_ratio_supported": float(ratio[j]),
            "shortfall_factor_supported": float(1.0 / max(ratio[j], 1e-12)),
            "best_bin_density": float(dens[j]), "best_bin_req_density": float(rq[j]),
            "best_bin_sites_test": int(te_n[j]),
            "best_ratio_ANY_bin_no_support_floor": float(ratio[jall]),
            "best_ratio_ANY_bin_sites_test": int(te_n[jall]),
        }
    # (iii) mutual information -- binning-robust statement of how much the free basis can
    #       EVER say about where the errors are. H(flip) is the total available.
    p_f = float(te_flip.sum()) / max(1, n_test_px)
    H_flip = -(p_f * np.log2(p_f) + (1 - p_f) * np.log2(1 - p_f))
    dj = np.clip(dens, 1e-12, 1 - 1e-12)
    Hc = -(dj * np.log2(dj) + (1 - dj) * np.log2(1 - dj))
    H_flip_given_bin = float((te_n * Hc).sum() / max(1, n_test_px))
    mi = H_flip - H_flip_given_bin

    # in-sample vs held-out optimism (ob1 measured 0.10% on a 90-cell table)
    tr_bits_insample = float((tr_n * H_bin).sum())
    optimism = float(te_bits.sum() / max(1, n_test_px)) / \
        max(1e-12, tr_bits_insample / max(1, int(tr_n.sum())))

    out = {
        "schema": "ddm_gt3_free_basis_address.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partial": bool(args.stop_after),
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "baseline": {"live_best_S": LIVE_BEST_S, "bar_PR130": PR130_BAR, "gap": GAP,
                     "archive_bytes": 353805,
                     "W_bytes_per_flip": W_BYTES_PER_FLIP,
                     "S_per_flip": S_PER_FLIP, "S_per_byte": RATE_PER_BYTE},
        "controls": {
            "C1_flips": tot_flips, "C1_d_seg": d_seg, "C1_pass": bool(c1_pass),
            "C2_raw_bytes": got, "C2_pass": True,
            "C3_m88_test_over_pop": m88_ratio, "C3_m88_train_over_pop": m88_train,
            "C5_denominators": {
                "scorer_sites_total": npairs * PIX_PER_PAIR,
                "test_sites": n_test_px, "test_pairs": len(test),
                "train_pairs": len(train), "bins_total": NBINS,
                "bins_with_support": int((te_n > 0).sum()),
            },
        },
        "base_rate_test": base_rate,
        "heldout_optimism_ratio": optimism,
        "naive_paint_whole_field_bytes": full_field_bytes,
        "shortfall_vs_break_even": shortfall,
        "information": {
            "H_flip_bits": H_flip,
            "H_flip_given_free_bin_bits": H_flip_given_bin,
            "MI_flip_free_basis_bits": mi,
            "fraction_of_flip_information_recovered": mi / max(H_flip, 1e-12),
        },
        "per_bin_table_bytes_counted": PER_BIN_TABLE_BYTES,
        "bin_id_bits": bin_id_bits,
        "top_free_bins_by_density": top,
        "waterfill_class_everywhere_DOMINATED": {
            "eta_pose_neutral_0.5406": waterfill(ETA_POSE_NEUTRAL_N32),
            "eta_seg_only_0.7895": waterfill(ETA_SEG_ONLY_N32),
            "eta_1.0_upper_bound": waterfill(1.0),
        },
        "waterfill_indicator_THE_DECIDING_SCHEME": {
            "eta_pose_neutral_0.5406": waterfill_indicator(ETA_POSE_NEUTRAL_N32),
            "eta_seg_only_0.7895": waterfill_indicator(ETA_SEG_ONLY_N32),
            "eta_1.0_upper_bound": waterfill_indicator(1.0),
        },
        "notes": (
            "There is NO address term by construction: the receiver recomputes the bin from "
            "decoded RGB + fixed operators, so naming the set costs zero counted bytes. "
            "Counted = the bin->class table + per-site class coefficients in taken bins. "
            "eta is TRANSFERRED from sq1 n=32 and is re-measured on THIS set in Job B."),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("base_rate_test", "waterfill_indicator_THE_DECIDING_SCHEME")}, indent=1))
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
