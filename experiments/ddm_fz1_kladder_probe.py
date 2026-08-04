# SPDX-License-Identifier: MIT
"""ddm_fz1 k-ladder probe -- locate the capacity cliff between k=4 and k=16 on cr2_ep854.

fz1's census measured, on the REAL byte-closed seg base (#827 cr2_ep854, delivered d_pose
O(80)/pair): k=4 repairs to O(1-3) (300-1000x over bar) while k=16 reaches 5.28e-4 and
free-form 1.73e-4 (BELOW the zero-byte bar 0.0042659).  The verdict therefore hinges on WHERE
between k=4 and k=16 the cliff falls, because rate admissibility closes the window from above:

    stream bytes (uniform int16) = 600*(k*k*3*2) + 17 + 2400
    banks iff  sqrt(10*d_pose_n600) < 0.2065399 - RATE_PER_BYTE*stream

    k=5:  150 B/pair ->  92,417 B -> need d < 2.100e-3
    k=6:  216 B/pair -> 132,017 B -> need d < 1.427e-3
    k=8:  384 B/pair -> 232,817 B -> need d < 2.654e-4
    k=10: 600 B/pair -> 362,417 B -> rate 0.24133 > headroom: INADMISSIBLE at int16
    (12-bit packing loosens each by 0.75x bytes; k=10 at 12-bit -> 0.18137 -> d < 6.3e-6)

By basis containment d(k) is monotone nonincreasing in k, so measuring k in {6,8,10} with an
lr ladder brackets the cliff and bounds every admissible k at once.  Coefficients at the best
QUANTISED iterate are persisted per (pair,k) and the REAL stream bytes are measured through
`tac.optimization.frame0_pose_repair_stream.encode_pose_repair_stream` (LZMA1-vs-stored raced)
-- actual section bytes, never hand-arithmetic (charter step 3).

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=False.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

from ddm_js1_staging_discriminator import (
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    RATE_PER_BYTE,
    SEG_H,
    SEG_W,
    Scorer,
    d_pose_t,
    decode_gt_frames,
    patch_yuv6_and_assert,
    pose_forward_grad,
    realize_full_frame,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import resize_to_scorer

from tac.optimization.frame0_pose_repair_stream import (
    dct_atoms,
    encode_pose_repair_stream,
    section_ledger,
)

HEADROOM_S = 0.7910689 - 0.3944070 - 0.1901220   # 0.2065399 (pre-registered, fz1 census)


def solve_dct_quantised_capture(sc, dec_f0, f1, pose_gt, *, k, steps, lr, eval_every):
    """js1's quantised-best-iterate k-DCT solve, returning the int16 coefficients at best.

    Best-iterate is selected on the int16-QUANTISED synthesis (the payload that would ship),
    per the quant-toolbox rung-3 discipline (solve with the quantiser in-loop).
    """
    posenet = sc.net.posenet
    base0 = resize_to_scorer(dec_f0)
    f1_s = resize_to_scorer(f1).detach()
    atoms = dct_atoms(k, SEG_H, SEG_W)
    A = torch.from_numpy(atoms).reshape(k * k, -1)

    def realized_dpose(t0):
        q = torch.round(torch.clamp(t0, 0.0, 255.0)).detach()
        with torch.no_grad():
            return float(d_pose_t(posenet, pose_gt, pose_forward_grad(posenet, q, f1_s)))

    ident = realized_dpose(base0)
    best = (ident, None, "identity@0", None)
    coef = torch.zeros(3, k * k, requires_grad=True)
    opt = torch.optim.Adam([coef], lr=lr)
    with torch.enable_grad():
        for it in range(steps + 1):
            d = (coef @ A).reshape(1, 3, SEG_H, SEG_W)
            cur = torch.clamp(base0 + d, 0.0, 255.0)
            if it % eval_every == 0 or it == steps:
                cq = torch.round(coef.detach()).clamp(-32768, 32767)
                dq = (cq @ A).reshape(1, 3, SEG_H, SEG_W)
                curq = torch.clamp(base0 + dq, 0.0, 255.0)
                dp = realized_dpose(curq)
                if dp < best[0]:
                    q = torch.round(curq).detach()
                    best = (dp, q[0].permute(1, 2, 0).numpy().astype(np.uint8),
                            f"dct{k}q@{it}", cq.numpy().astype(np.int16))
            if it == steps:
                break
            out = pose_forward_grad(posenet, cur, f1_s)
            loss = d_pose_t(posenet, pose_gt, out)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return best  # (d_pose_scorer, paint_u8|None, tag, coef_i16|None)


def stream_bytes_for(coefs: dict[int, np.ndarray], k: int, n_pairs_extrapolate: int) -> dict:
    """REAL section bytes via the reusable module (LZMA1 vs stored raced), plus the honest
    n600 extrapolation (per-pair mean of the coded stream scaled to 600 repaired pairs)."""
    blob = encode_pose_repair_stream(coefs, k=k, seg_h=SEG_H, seg_w=SEG_W)
    led = section_ledger(blob)
    per_pair = led["counted_bytes_per_repaired_pair"]
    n600_bytes = per_pair * n_pairs_extrapolate
    rate_s = RATE_PER_BYTE * n600_bytes
    d_bar = ((HEADROOM_S - rate_s) ** 2) / 10.0 if rate_s < HEADROOM_S else 0.0
    return {"ledger": led, "n600_bytes_extrapolated": n600_bytes,
            "n600_rate_S": rate_s,
            "d_pose_bar_at_this_rate": d_bar,
            "rate_admissible": bool(rate_s < HEADROOM_S)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coef-npz", type=Path, required=True)
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--ks", type=str, default="6,8,10")
    ap.add_argument("--lrs", type=str, default="50,200,800")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    pairs = [int(x) for x in args.pairs.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]

    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    patch = patch_yuv6_and_assert(sc)
    print(f"[fz1-k] ready t={time.time()-t0:.1f}s {patch}", flush=True)

    rows: list[dict] = []
    coef_store: dict[str, np.ndarray] = {}
    if args.resume and args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])
        if args.coef_npz.exists():
            coef_store = dict(np.load(args.coef_npz))
    done = {(int(r["pair"]), int(r["k"])) for r in rows}

    def flush() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # per-k pooled summary + REAL stream ledger over the measured pairs
        summary = {}
        for k in ks:
            krows = [r for r in rows if r["k"] == k and r["best_coef_saved"]]
            if not krows:
                continue
            coefs = {int(r["pair"]): coef_store[f"p{r['pair']}_k{k}"] for r in krows}
            sb = stream_bytes_for(coefs, k, 600)
            d_vals = [r["d_pose_camera_best"] for r in krows]
            summary[f"k{k}"] = {
                "n_pairs": len(krows), "d_pose_mean_measured": float(np.mean(d_vals)),
                "d_pose_max": float(np.max(d_vals)), "d_pose_values": d_vals,
                "stream": sb,
                "banks_on_measured_mean": bool(
                    sb["rate_admissible"]
                    and np.mean(d_vals) < sb["d_pose_bar_at_this_rate"]),
            }
        with open(args.out, "w") as f:
            json.dump({"schema": "ddm_fz1_kladder_probe.v1",
                       "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                       "score_claim": False,
                       "headroom_S": HEADROOM_S, "yuv6_patch": patch,
                       "summary": summary, "rows": rows}, f, indent=1)
        np.savez_compressed(args.coef_npz, **coef_store)

    for p in pairs:
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        pose_gt = sc.pose_out(gt)
        lstar1 = sc.seg_argmax(dec)
        dp_delivered = sc.d_pose(pose_gt, sc.pose_out(dec))
        for k in ks:
            if (p, k) in done:
                continue
            tp = time.time()
            best = None
            per_lr = {}
            for lr in lrs:
                dp_s, paint, tag, coef = solve_dct_quantised_capture(
                    sc, dec[0], dec[1], pose_gt, k=k, steps=args.steps, lr=lr,
                    eval_every=args.eval_every)
                if paint is not None:
                    pair_f = np.stack([realize_full_frame(dec[0], paint), dec[1]])
                    dp_cam = sc.d_pose(pose_gt, sc.pose_out(pair_f))
                    seg_ok = bool((sc.seg_argmax(pair_f) == lstar1).all())
                else:
                    dp_cam, seg_ok, coef = dp_delivered, True, None
                per_lr[f"lr{lr:g}"] = {"tag": tag, "d_pose_camera": dp_cam,
                                       "cmax": float(np.abs(coef).max()) if coef is not None
                                       else 0.0, "seg_exact": seg_ok}
                if best is None or dp_cam < best[0]:
                    best = (dp_cam, coef, tag, lr)
            dp_best, coef_best, tag_best, lr_best = best
            saved = coef_best is not None
            if saved:
                coef_store[f"p{p}_k{k}"] = coef_best
            rows.append({"pair": int(p), "k": int(k),
                         "d_pose_delivered": dp_delivered,
                         "d_pose_camera_best": float(dp_best),
                         "tag": tag_best, "lr_best": lr_best,
                         "best_coef_saved": saved, "per_lr": per_lr})
            flush()
            print(f"[fz1-k] pair {p:3d} k={k:2d} delivered {dp_delivered:8.3f} -> "
                  f"best {dp_best:10.6f} ({tag_best}, lr{lr_best:g}) [{time.time()-tp:.1f}s]",
                  flush=True)

    flush()
    print(f"[fz1-k] done {len(rows)} rows t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
