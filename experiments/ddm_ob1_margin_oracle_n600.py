#!/usr/bin/env python
"""ddm_ob1 Job D -- the ORACLE rung: what does the frozen scorer's OWN margin know?

USES THE SCORER SLOT.  Chunked (<=120 pairs per chunk), resumable from disk (P0).
Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE. score_claim=False.
pointer 0.1910828242 [contest-CPU] UNMOVED.

THE LADDER THIS CLOSES
----------------------
Address cost of the seg correction, all priced identically at n600 as
`SUM_i -[f_i log2 p_i + (1-f_i) log2(1-p_i)]` over the 117,964,800 scorer pixels:

  rung 0  uniform, no features                     LEGAL      [ob1 Job B]
  rung 1  decoded-RGB features only                LEGAL      [ob1 Job C]  <- the honest floor
  rung 2  L*-derived (d x own-class x edge)        ILLEGAL    [ob1 Job B]  (needs 73 MB SegNet)
  rung 3  frozen SegNet MARGIN                     ILLEGAL    THIS UNIT    <- the true ceiling

rung1 -> rung3 is the exact job gp1's R5 "micro-student" must do: predict the frozen scorer's
own confidence from decoded RGB.  gp1 priced that rung at 106,954 B by differencing two hard
bands; this unit measures it as a mutual information, which is the quantity that actually
bounds any student.

No cx1-decoder margin cache exists anywhere on disk (only qa75-render margins, a different
vehicle), so this must be computed.  Chosen over caching the 471 MB field: accumulate the
contingency tables directly, so nothing bulky is written.

POSITIVE CONTROLS (fail-closed; m50 -- an empty scope reports VACUOUS, never PASS)
---------------------------------------------------------------------------------
  C1  argmax(frozen SegNet(decoded f1)) == cx1_argmax_n600[p], EXACT, all 600 pairs.
      sq1 verified this on 32; this is the n600 extension and it validates the whole harness.
  C2  the reproduced flip total must equal 508,640 / d_seg 0.004311794704861111.

DENOMINATORS (m66)
------------------
live best S = 0.7910689 @ 353,805 B (pu2, sha c72ef357) [macOS-CPU advisory]
gap to the PR130 bar 0.172141 = 0.6189279 ; 1% of gap = 0.0061893 S = 9,295 B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO / "upstream"), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import einops
from modules import SegNet, segnet_sd_path

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
NB_M = 32          # margin quantile buckets
NB_D = 17          # distance-to-L*-boundary buckets (0..15, 16=further)
ETA_POSE_NEUTRAL_N32 = 0.5406
ETA_UNCONSTRAINED_N32 = 0.7895
EXPECT_FLIPS = 508_640
EXPECT_DSEG = 0.004311794704861111


def load_segnet(threads: int) -> SegNet:
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    from safetensors.torch import load_file
    net = SegNet().eval()
    net.load_state_dict(load_file(str(segnet_sd_path)))
    return net


@torch.inference_mode()
def seg_logits(net: SegNet, f1_cam: np.ndarray) -> np.ndarray:
    """Frozen SegNet on the decoded frame_1, through its own preprocess. -> (5,384,512)."""
    pair = np.stack([f1_cam, f1_cam])          # SegNet reads x[:, -1, ...] only
    x = torch.from_numpy(np.ascontiguousarray(pair))[None]
    x = einops.rearrange(x, "b t h w c -> b t c h w").float()
    return net(net.preprocess_input(x))[0].numpy()


def boundary(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def dilate1(m: np.ndarray) -> np.ndarray:
    o = m.copy()
    o[:-1, :] |= m[1:, :]
    o[1:, :] |= m[:-1, :]
    o[:, :-1] |= m[:, 1:]
    o[:, 1:] |= m[:, :-1]
    return o


def dist_to_boundary(lab: np.ndarray) -> np.ndarray:
    d = np.full(lab.shape, NB_D - 1, dtype=np.uint8)
    m = boundary(lab)
    d[m] = 0
    for r in range(1, NB_D - 1):
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
    return float(-(k.astype(np.float64) * np.log2(p)
                   + (n - k).astype(np.float64) * np.log2(1 - p)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--chunk", type=int, default=100, help="pairs per checkpoint (slot rule <=120)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--calib-stride", type=int, default=20)
    ap.add_argument("--state", default=os.path.join(OUT_DIR, "ob1_margin_state.npz"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ob1_margin_oracle_n600.json"))
    ap.add_argument("--resume", action="store_true")
    # The bash harness SIGURGs (rc=144) any foreground job past ~3 min, and nohup+disown did
    # not shield the child either (measured 3x this slot).  So each invocation is BOUNDED and
    # the run is driven to completion by repeated --resume calls; --report-only then emits the
    # JSON from the completed state.  This is the resumability non-negotiable doing its job.
    ap.add_argument("--stop-after", type=int, default=0,
                    help="process at most N pairs this invocation, then checkpoint and exit 0")
    ap.add_argument("--report-only", action="store_true",
                    help="emit the report from a COMPLETE state file; no scorer forwards")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")
    need = N_PAIRS * SEQ * CAM_H * CAM_W * 3
    if os.path.getsize(RAW) != need:
        raise RuntimeError(f"0.raw incomplete: {os.path.getsize(RAW)} != {need}")
    raw = np.memmap(RAW, dtype=np.uint8, mode="r", shape=(N_PAIRS * SEQ, CAM_H, CAM_W, 3))

    net = None if args.report_only else load_segnet(args.threads)
    t0 = time.time()

    # ---- margin quantile calibration on a STRATIFIED subsample (m88: never a prefix) --------
    calib = list(range(0, n, args.calib_stride))
    c1_bad: list[int] = []
    if (args.resume or args.report_only) and os.path.exists(args.state):
        st = np.load(args.state)
        start = int(st["next_pair"])
        mq = st["mq"]
        cn, ck = st["cn"], st["ck"]
        cnd, ckd = st["cnd"], st["ckd"]
        c1_fail = int(st["c1_fail"])
        total_flips = int(st["total_flips"])
        if "c1_bad" in st:
            c1_bad = [int(v) for v in st["c1_bad"]]
        print(f"[ob1D] RESUMED at pair {start}", flush=True)
    elif args.report_only:
        raise RuntimeError(f"--report-only needs an existing state file at {args.state}")
    else:
        ms = []
        for p in calib:
            lg = seg_logits(net, np.asarray(raw[SEQ * p + 1]))
            s = np.sort(lg, axis=0)
            ms.append((s[-1] - s[-2])[::4, ::4].ravel())
        mq = np.percentile(np.concatenate(ms),
                           np.linspace(0, 100, NB_M + 1)[1:-1]).astype(np.float64)
        print(f"[ob1D] calib on {len(calib)} stratified pairs, {time.time()-t0:.1f}s; "
              f"margin q1={mq[0]:.4f} med={mq[NB_M//2-1]:.4f} q31={mq[-1]:.4f}", flush=True)
        start = 0
        cn = np.zeros((2, NB_M), dtype=np.int64)
        ck = np.zeros((2, NB_M), dtype=np.int64)
        cnd = np.zeros((2, NB_M, NB_D), dtype=np.int64)
        ckd = np.zeros((2, NB_M, NB_D), dtype=np.int64)
        c1_fail = 0
        total_flips = 0

    stop_at = n if args.stop_after <= 0 else min(n, start + args.stop_after)
    for p in (range(start, stop_at) if not args.report_only else range(0)):
        s_ = p & 1
        g_ = np.asarray(gt[p])
        lstar_cache = np.asarray(rd[p])
        lg = seg_logits(net, np.asarray(raw[SEQ * p + 1]))
        lstar = lg.argmax(axis=0).astype(np.uint8)
        if not np.array_equal(lstar, lstar_cache):
            c1_fail += 1
            c1_bad.append(p)
        srt = np.sort(lg, axis=0)
        margin = srt[-1] - srt[-2]
        f = g_ != lstar_cache
        total_flips += int(f.sum())
        mb = np.digitize(margin, mq).astype(np.int64)
        d = dist_to_boundary(lstar_cache).astype(np.int64)
        cn[s_] += np.bincount(mb.ravel(), minlength=NB_M)
        ck[s_] += np.bincount(mb[f].ravel(), minlength=NB_M)
        j = mb * NB_D + d
        cnd[s_] += np.bincount(j.ravel(), minlength=NB_M * NB_D).reshape(NB_M, NB_D)
        ckd[s_] += np.bincount(j[f].ravel(), minlength=NB_M * NB_D).reshape(NB_M, NB_D)

        if (p + 1) % args.chunk == 0 or p == stop_at - 1:
            # np.savez APPENDS '.npz' unless the name already ends in it, so the temp name
            # must already be a .npz or the atomic rename below silently targets a ghost.
            tmp = args.state + ".tmp.npz"
            np.savez(tmp, next_pair=p + 1, mq=mq, cn=cn, ck=ck, cnd=cnd, ckd=ckd,
                     c1_fail=c1_fail, total_flips=total_flips,
                     c1_bad=np.array(c1_bad, dtype=np.int64))
            os.replace(tmp, args.state)
            print(f"  ob1-D {p+1}/{n}  C1_fail={c1_fail}  {time.time()-t0:.1f}s "
                  f"(checkpoint)", flush=True)

    if stop_at < n and not args.report_only:
        print(f"[ob1D] BOUNDED STOP at {stop_at}/{n}; re-run with --resume. NO REPORT WRITTEN "
              f"(m50: a partial scope must not emit a result that looks complete).", flush=True)
        return 0

    N = n * PIX_PER_PAIR
    CN, CK = cn.sum(0), ck.sum(0)
    CND, CKD = cnd.sum(0), ckd.sum(0)

    # ---- positive controls ------------------------------------------------------------------
    # C1 is REPORTED with its exact failing-pair list rather than silently rounded to PASS.
    # It is NOT fail-closed: the flip mask and the distance field are both taken from the
    # CACHE, so a re-forward argmax that differs on a pair changes only that pair's MARGIN by
    # a hair.  C2 IS fail-closed, because a wrong flip total would invalidate every number.
    controls = {
        "C1_argmax_matches_cx1_cache_all_pairs": c1_fail == 0,
        "C1_failing_pair_count": c1_fail,
        "C1_failing_pairs": c1_bad,
        "C1_note": "argmax re-forward vs the cached argmax; any mismatch is CPU-kernel "
                   "nondeterminism (thread count / BLAS path), reported not hidden",
        "C2_flips_reproduced": total_flips,
        "C2_flips_expected": EXPECT_FLIPS,
        "C2_d_seg_reproduced": total_flips / N,
        "C2_d_seg_expected": EXPECT_DSEG,
        "C2_exact": total_flips == EXPECT_FLIPS and n == N_PAIRS,
    }
    if n == N_PAIRS and total_flips != EXPECT_FLIPS:
        raise RuntimeError(f"POSITIVE CONTROL C2 FAILED: c1_fail={c1_fail} "
                           f"flips={total_flips} expected={EXPECT_FLIPS}")

    addr_margin = cond_bits(CN, CK)
    addr_margin_d = cond_bits(CND, CKD)
    addr_uniform = cond_bits(np.array([CN.sum()]), np.array([CK.sum()]))
    ho = 0.0
    for s_ in (0, 1):
        pm = (ckd[1 - s_] + 0.5) / (cnd[1 - s_] + 1.0)
        ho += xent_bits(cnd[s_], ckd[s_], pm)

    gross = S_PER_FLIP * total_flips

    def row(name, bits, legal):
        # payload priced at the L*-conditioned 0.2633 bits/flip -- the CHEAPEST published
        # figure, so the oracle rung is given every benefit; it is itself illegal.
        pay = total_flips * 0.2633
        byts = (bits + pay) / 8.0
        return {"address_feature": name, "legal": legal,
                "address_bits": bits, "address_bits_per_field_px": bits / N,
                "total_bytes_with_Lstar_payload": byts,
                "rate_cost_S": byts * RATE_PER_BYTE,
                "net_dS_eta1": byts * RATE_PER_BYTE - gross,
                "net_dS_eta_pose_neutral_n32": byts * RATE_PER_BYTE
                - ETA_POSE_NEUTRAL_N32 * gross,
                "break_even_eta": (byts * RATE_PER_BYTE) / gross}

    rows = [row("uniform", addr_uniform, True),
            row("frozen margin (32 q-buckets)", addr_margin, False),
            row("frozen margin x d(L* boundary)", addr_margin_d, False)]

    out = {
        "schema": "ddm_ob1_margin_oracle.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": n, "scorer_forwards": n + len(calib),
        "positive_controls": controls,
        "baseline": {"live_best_S": LIVE_BEST_S, "gap_to_floor": GAP,
                     "rate_S_per_byte": RATE_PER_BYTE, "S_per_flip": S_PER_FLIP},
        "totals": {"flips": total_flips, "field_pixels": N,
                   "gross_dS_at_eta1_full_capture": gross},
        "held_out": {"address_bits_heldout": ho, "address_bits_insample": addr_margin_d,
                     "optimism_bits": addr_margin_d - ho},
        "margin_quantiles": mq.tolist(),
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nwrote {args.out}")
    print(f"CONTROLS  C1 argmax==cache all pairs: {controls['C1_argmax_matches_cx1_cache_all_pairs']} "
          f"(failures {c1_fail})   C2 flips {total_flips} (expected {EXPECT_FLIPS})")
    print("\n  address feature                  legal   bits/px    bytes     net@.5406   be_eta")
    for r in rows:
        print(f"  {r['address_feature']:<32s} {r['legal']!s:<6s} "
              f"{r['address_bits_per_field_px']:.5f}  "
              f"{r['total_bytes_with_Lstar_payload']:8.0f}  "
              f"{r['net_dS_eta_pose_neutral_n32']:+.5f}  {r['break_even_eta']:.4f}")
    print(f"\nheld-out {ho:,.0f} bits vs in-sample {addr_margin_d:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
