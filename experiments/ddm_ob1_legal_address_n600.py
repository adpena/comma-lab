#!/usr/bin/env python
"""ddm_ob1 Job C -- what does a RECEIVER-LEGAL feature set know about the flips?

ZERO SegNet/PoseNet forwards (consumes the argmax caches other arms produced).
Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE. score_claim=False.
pointer 0.1910828242 [contest-CPU] UNMOVED.

WHY THIS EXISTS
---------------
gp1 and sq1 both price an address band they call FREE / "the receiver's own label boundary" /
"legal today, zero compliance question".  A receiver-side audit of the SHIPPED decoder
(/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2/) finds NO class label field of
any kind:

  * inflate_runner.py + ddm_tr1_runtime.py + ddm_r7_token_coder.py + pfs1_warp_receiver.py +
    ddm_ix2_archive_container.py import no torch and load no network weights;
  * token_codes are (600,24,32,4) uint8 int4 LATENTS, dequantized straight to float in
    [-1,1] and fed to a conv stack -- never argmaxed, no code->class table;
  * the renderer emits 3-channel sigmoid*255 RGB, not 5-class logits;
  * the only free per-pixel partitions are GEOMETRIC: a horizon row split (v_row=437) and a
    warp-validity mask.

So `L*` is an artifact of running the 73 MB frozen SegNet, which the receiver cannot ship.
gp1's band and sq1's band are BOTH illegal as measured.  This unit prices the address using
ONLY things the shipped decoder actually computes: its own decoded RGB, plus pure code.

THE LADDER THIS COMPLETES (all priced identically, n600, same denominators)
--------------------------------------------------------------------------
  rung 0  uniform, no features                            LEGAL, useless
  rung 1  decoded-RGB features (THIS UNIT)                LEGAL  <- the honest free floor
  rung 2  L*-derived (d x own x edge)  [ob1 Job B]        ILLEGAL (needs the 73 MB scorer)
  rung 3  frozen-margin oracle                            ILLEGAL, not measured here

rung1 -> rung2 is exactly the job gp1's R5 "micro-student" would have to do, and it is much
larger than gp1's 106,954 B estimate, which was only the gap between two hard bands.

PAYLOAD, honestly
-----------------
Job B could price payload at H(gt | own_class, edge) = 0.2633 bits/flip because it assumed the
receiver knows its own class.  Without a label field it does not, so the legal payload is
H(gt | X_legal) -- the target class must be sent outright.  Both are reported.

DENOMINATORS (m66)
------------------
live best S = 0.7910689 @ 353,805 B (pu2, sha c72ef357) [macOS-CPU advisory]
gap to the PR130 bar 0.172141 = 0.6189279 ; 1% of gap = 0.0061893 S = 9,295 B
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
RAW = "/Volumes/VertigoDataTier/pact/ddm_ob1_20260803/inflated/0.raw"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_ob1_20260803"

CAM_H, CAM_W = 874, 1164
SEG_H, SEG_W = 384, 512
PIX_PER_PAIR = SEG_H * SEG_W
N_PAIRS = 600
SEQ = 2
RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689
GAP = 0.6189279
S_PER_FLIP = 100.0 / (N_PAIRS * PIX_PER_PAIR)
NC = 5
DMAX = 15
NB_D = DMAX + 2      # distance-to-edge buckets 0..15, 16 = further
NB_ROW = 12          # row // 32
NB_G = 4             # gradient-magnitude quartile buckets
ETA_POSE_NEUTRAL_N32 = 0.5406
ETA_UNCONSTRAINED_N32 = 0.7895


def to_scorer_rgb(cam_u8: np.ndarray) -> np.ndarray:
    """EXACTLY the scorer's own resize: interpolate(x,(384,512),bilinear,antialias=False).

    upstream/modules.py:73 (PoseNet) and :109 (SegNet) make the identical call, so this is the
    lattice both scorers read (pz1 / m86).
    """
    x = torch.from_numpy(np.ascontiguousarray(cam_u8)).permute(2, 0, 1)[None].float()
    y = torch.nn.functional.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear",
                                        align_corners=False)
    return y[0].permute(1, 2, 0).numpy()


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def grad_mag(y: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = np.abs(y[:, 2:] - y[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(y[2:, :] - y[:-2, :]) * 0.5
    return gx + gy


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


def cond_bits(n: np.ndarray, k: np.ndarray) -> float:
    n = n.astype(np.float64)
    k = k.astype(np.float64)
    ok = (n > 0) & (k > 0) & (k < n)
    p = np.zeros_like(n)
    p[ok] = k[ok] / n[ok]
    h = np.where(ok, -(p * np.log2(np.where(p > 0, p, 1))
                       + (1 - p) * np.log2(np.where(p < 1, 1 - p, 1))), 0.0)
    return float((n * h).sum())


def xent_bits(n: np.ndarray, k: np.ndarray, pm: np.ndarray) -> float:
    p = np.clip(pm, 1e-9, 1 - 1e-9)
    n = n.astype(np.float64)
    k = k.astype(np.float64)
    return float(-(k * np.log2(p) + (n - k) * np.log2(1 - p)).sum())


def class_bits(tab: np.ndarray) -> float:
    """SUM over cells of count * H(gt | cell): the payload cost of naming the target class."""
    t = tab.astype(np.float64)
    tot = t.sum(axis=-1, keepdims=True)
    safe = np.where(tot > 0, tot, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(t > 0, np.log2(t / safe), 0.0)
    return float(-(t * lg).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--calib-stride", type=int, default=20,
                    help="stratified stride for the gradient-quantile calibration (never a prefix)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ob1_legal_address_n600.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")
    want = os.path.getsize(RAW)
    need = N_PAIRS * SEQ * CAM_H * CAM_W * 3
    if want != need:
        raise RuntimeError(f"0.raw is {want} bytes, expected {need} -- inflate incomplete")
    raw = np.memmap(RAW, dtype=np.uint8, mode="r", shape=(N_PAIRS * SEQ, CAM_H, CAM_W, 3))

    # ---- calibration pass: gradient quantiles on a STRATIFIED subsample (m88: never a prefix)
    calib_pairs = list(range(0, n, args.calib_stride))
    gs = []
    for p in calib_pairs:
        g = grad_mag(luma(to_scorer_rgb(np.asarray(raw[SEQ * p + 1]))))
        gs.append(g[::4, ::4].ravel())
    gcat = np.concatenate(gs)
    qs = [50.0, 75.0, 90.0]
    gq = np.percentile(gcat, qs).tolist()
    edge_thr = float(np.percentile(gcat, 90.0))
    gq_txt = ", ".join(f"{v:.3f}" for v in gq)
    print(f"[ob1C] calib on {len(calib_pairs)} stratified pairs; grad quantiles "
          f"[{gq_txt}]; edge threshold(p90) {edge_thr:.3f}", flush=True)

    # cells: [split, d_edge, row_bucket, grad_bucket]
    cn = np.zeros((2, NB_D, NB_ROW, NB_G), dtype=np.int64)
    ck = np.zeros((2, NB_D, NB_ROW, NB_G), dtype=np.int64)
    pay = np.zeros((2, NB_D, NB_ROW, NB_G, NC), dtype=np.int64)
    total_flips = 0
    rows_idx = (np.arange(SEG_H) // 32).astype(np.int64)
    rows_idx = np.clip(rows_idx, 0, NB_ROW - 1)
    row_map = np.repeat(rows_idx[:, None], SEG_W, axis=1)

    t0 = time.time()
    for p in range(n):
        s = p & 1
        g_ = np.asarray(gt[p])
        lstar = np.asarray(rd[p])
        f = g_ != lstar
        total_flips += int(f.sum())

        rgb = to_scorer_rgb(np.asarray(raw[SEQ * p + 1]))
        gm = grad_mag(luma(rgb))
        d = dist_to_mask(gm > edge_thr)
        gb = np.digitize(gm, gq).astype(np.int64)   # 0..3

        idx = (d.astype(np.int64) * NB_ROW + row_map) * NB_G + gb
        m = NB_D * NB_ROW * NB_G
        cn[s] += np.bincount(idx.ravel(), minlength=m).reshape(NB_D, NB_ROW, NB_G)
        ck[s] += np.bincount(idx[f].ravel(), minlength=m).reshape(NB_D, NB_ROW, NB_G)
        pidx = idx[f] * NC + g_[f].astype(np.int64)
        pay[s] += np.bincount(pidx, minlength=m * NC).reshape(NB_D, NB_ROW, NB_G, NC)
        if (p + 1) % 50 == 0:
            print(f"  ob1-C {p+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    N = n * PIX_PER_PAIR
    CN, CK, PAY = cn.sum(0), ck.sum(0), pay.sum(0)

    feats = {}
    feats["none (uniform)"] = cond_bits(np.array([CN.sum()]), np.array([CK.sum()]))
    feats["row only"] = cond_bits(CN.sum(axis=(0, 2)), CK.sum(axis=(0, 2)))
    feats["grad only"] = cond_bits(CN.sum(axis=(0, 1)), CK.sum(axis=(0, 1)))
    feats["d_edge only"] = cond_bits(CN.sum(axis=(1, 2)), CK.sum(axis=(1, 2)))
    feats["d_edge x row"] = cond_bits(CN.sum(axis=2), CK.sum(axis=2))
    feats["d_edge x row x grad"] = cond_bits(CN, CK)

    ho = 0.0
    for s in (0, 1):
        pm = (ck[1 - s] + 0.5) / (cn[1 - s] + 1.0)
        ho += xent_bits(cn[s], ck[s], pm)

    pay_legal = class_bits(PAY)
    pay_marginal = class_bits(PAY.sum(axis=(0, 1, 2))[None, :])

    addr = feats["d_edge x row x grad"]
    gross = S_PER_FLIP * total_flips
    rows = []
    for aname, ab in feats.items():
        for pname, pb in (("H(gt|X_legal) LEGAL", pay_legal),
                          ("H(gt) marginal", pay_marginal)):
            byts = (ab + pb) / 8.0
            rows.append({
                "address_feature": aname, "payload_feature": pname, "legal": True,
                "address_bits": ab, "address_bits_per_field_px": ab / N,
                "payload_bits": pb, "payload_bits_per_flip": pb / total_flips,
                "total_bytes": byts, "rate_cost_S": byts * RATE_PER_BYTE,
                "capture_rate": 1.0,
                "net_dS_eta1": byts * RATE_PER_BYTE - gross,
                "net_dS_eta_unconstrained_n32": byts * RATE_PER_BYTE
                - ETA_UNCONSTRAINED_N32 * gross,
                "net_dS_eta_pose_neutral_n32": byts * RATE_PER_BYTE
                - ETA_POSE_NEUTRAL_N32 * gross,
                "break_even_eta": (byts * RATE_PER_BYTE) / gross,
            })

    best = min(rows, key=lambda r: r["net_dS_eta_pose_neutral_n32"])
    out = {
        "schema": "ddm_ob1_legal_address.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "scorer_forwards": 0, "n_pairs": n,
        "legality": {
            "features": "decoded RGB at the scorer lattice + pure code; NO label field, "
                        "NO scorer weights",
            "shipped_constants": {"grad_quantiles": gq, "edge_threshold_p90": edge_thr,
                                  "bytes_upper": 4 * 4},
            "receiver_audit": "shipped inflate loads zero NN weights; token_codes are int4 "
                              "latents never argmaxed; renderer emits 3-ch RGB; only free "
                              "per-pixel partitions are geometric (horizon row, warp validity)",
        },
        "baseline": {"live_best_S": LIVE_BEST_S, "gap_to_floor": GAP,
                     "rate_S_per_byte": RATE_PER_BYTE, "S_per_flip": S_PER_FLIP,
                     "one_pct_of_gap_bytes": (GAP / 100.0) / RATE_PER_BYTE},
        "totals": {"flips": total_flips, "field_pixels": N,
                   "d_seg_reproduced": total_flips / N,
                   "gross_dS_at_eta1_full_capture": gross},
        "held_out": {"address_bits_heldout": ho, "address_bits_insample": addr,
                     "optimism_bits": addr - ho,
                     "model_cells_nonempty": int((CN > 0).sum())},
        "calibration_pairs": calib_pairs,
        "rows": rows,
        "best_legal_row": best,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nwrote {args.out}")
    print(f"flips {total_flips}  d_seg {total_flips/N:.18f}  gross@eta1 {gross:.5f}\n")
    print("  address feature          bits/px   payload                bytes     net@1     "
          "net@.5406   be_eta")
    for r in rows:
        print(f"  {r['address_feature']:<22s} {r['address_bits_per_field_px']:.5f}   "
              f"{r['payload_feature']:<20s} {r['total_bytes']:8.0f}  "
              f"{r['net_dS_eta1']:+.5f}  {r['net_dS_eta_pose_neutral_n32']:+.5f}  "
              f"{r['break_even_eta']:.4f}")
    print(f"\nheld-out {ho:,.0f} bits vs in-sample {addr:,.0f} "
          f"(optimism {addr-ho:,.0f} bits = {(addr-ho)/8:,.0f} B)")
    print(f"legal payload H(gt|X)   = {pay_legal/total_flips:.4f} bits/flip")
    print(f"marginal payload H(gt)  = {pay_marginal/total_flips:.4f} bits/flip")
    print("ILLEGAL reference (ob1 Job B, L*-derived): 0.02107 bits/px address, "
          "0.2633 bits/flip payload, 327,405 B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
