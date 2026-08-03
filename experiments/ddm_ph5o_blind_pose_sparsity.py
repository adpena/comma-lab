#!/usr/bin/env python
"""ddm_ph5o -- the SUPPORT sweep that decides O1 properly.

WHY THIS EXISTS
---------------
The first ``ph5o`` pass produced a discriminator, not a verdict:

  * the aimed direction is REAL -- the measured gradient has
    ``sum|g| = 2.784e-03`` per pair over the blind set, which is nonzero and
    was obtained through an adjoint verified to 1.9e-15;
  * but a FULL-FIELD +-1 LSB sign step (the smallest amplitude the uint8
    camera raster can express, applied to all 230,904 blind pixels at once)
    RAISES ``d_pose`` by 4.26e-01 -- **153x the entire first-order drop the
    gradient promised**, and 4.5x larger than the base ``d_pose`` itself.

A linearisation whose promised drop exceeds the whole objective is simply
outside its trust region.  So "the subspace is misaligned" is NOT yet
established: with the quantum pinned at +-1 LSB, the ONLY remaining free
parameter is the SUPPORT -- how many blind pixels the step touches.

THE MEASUREMENT
---------------
Rank the blind pixels by |g| and apply the exact descent step ``-sign(g)`` at
+-1 LSB to the top-n of them, for n on a log grid from 1 to 230,904.  This
holds the quantum fixed at the smallest realisable value and sweeps the
perturbation magnitude, which is the only knob left.

  * If some n lowers d_pose, the actuator is USABLE and O1 becomes a byte
    question: addressing n pixels costs ~log2(230904)/8 = 2.23 B each.
  * If d_pose rises monotonically from n=1, the aimed direction carries no
    usable descent at the realisable quantum and the actuator is retired.

POSITIVE CONTROL ON THE AIMING ITSELF
-------------------------------------
Both signs are measured at every n.  If the gradient carries real information
the two must be ASYMMETRIC at small support (``-sign(g)`` better than
``+sign(g)``).  If they are symmetric the response is purely second-order and
the gradient is uninformative -- a distinction no single-sign sweep can make.

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ORIGINAL_BYTES = 37545489
N_PAIRS = 600
#: the LIVE gap decomposition at the pu2 row (``ddm_ph4`` §0, recomputed there
#: against the PR130 bar 0.172141).  Named because a Delta-S without its
#: baseline is unanchored and baselines move (memory ``m46``/``qd1``).
LIVE_TOTAL_GAP = 0.6189279
LIVE_POSE_GAP = 0.1090357


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--supports", type=int, nargs="+", default=None,
                    help="explicit support sizes; default = 16-point log grid "
                         "over the whole blind set")
    ap.add_argument("--rows", type=Path, default=None,
                    help="resumable per-pair JSONL (required for n600)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sub = args.submission_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    for p in (str(sub), str(root / "upstream"), str(root / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    torch.set_num_threads(4)
    import ddm_p3v2_optimal_form_pose_resolve as p3v2
    import ddm_tr1_runtime as shipped
    from ddm_ph5o_blind_pose_solve import (
        PairWarpChain,
        apply_D,
        dpose_grad_wrt_f0cam,
    )
    from inflate_runner import Decoder

    from tac.optimization.ddm_ll1_window_solve import blind_mask

    posenet, _ = p3v2.load_posenet()
    targets = p3v2.load_targets(600)
    dec = Decoder(sub / "archive")
    blind = blind_mask()
    hgt, wid = blind.shape
    n_blind = int(blind.sum())
    blind_flat = np.flatnonzero(blind.ravel())
    if args.supports:
        supports = sorted({int(v) for v in args.supports})
    else:
        supports = sorted({int(v) for v in np.unique(np.round(np.logspace(
            0, np.log10(n_blind), 16)).astype(np.int64))} | {n_blind})

    idx = np.unique(np.linspace(0, int(dec.n_pairs) - 1,
                                args.pairs).round().astype(int))
    done: dict[int, dict] = {}
    handle = None
    if args.rows is not None:
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        if args.rows.exists():
            for line in args.rows.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if [e["support"] for e in rec["sweep"]] == supports:
                        done[int(rec["pair"])] = rec
            print(f"[sp] RESUME: {len(done)} rows already on disk", flush=True)
        handle = args.rows.open("a")
    print(f"[sp] base={sub.name} pairs={len(idx)} supports={supports}",
          flush=True)

    rows = []
    t_0 = time.time()
    for pos, pidx in enumerate(int(v) for v in idx):
        if pidx in done:
            rows.append(done[pidx])
            continue
        f1b = np.asarray(shipped.render_frame1_camera_uint8(dec.packet, pidx))
        f1_f = f1b.astype(np.float64)
        f0b = dec.f0(pidx, f1b)
        dp_base = p3v2.d_pose_u8(posenet, f0b, f1b, targets[pidx])
        chain = PairWarpChain(dec, pidx, hgt, wid)
        g_f0 = dpose_grad_wrt_f0cam(posenet, f0b, f1b, targets[pidx])
        f0f = chain.forward_f0f(f1_f)
        ste = ((f0f >= -0.5) & (f0f <= 255.5)).astype(np.float64)
        g_lum = chain.adjoint_to_f1(g_f0 * ste).sum(axis=2) * blind
        g_flat = g_lum.ravel()[blind_flat]
        order = blind_flat[np.argsort(-np.abs(g_flat))]
        sgn_flat = np.sign(g_lum.ravel())

        def evaluate(delta_flat, pidx=pidx, f1b=f1b):
            f1t = f1b.astype(np.float64).copy()
            f1t.reshape(-1, 3)[:] += delta_flat[:, None]
            f1t = np.clip(np.round(f1t), 0.0, 255.0).astype(np.uint8)
            return p3v2.d_pose_u8(
                posenet, dec.f0(pidx, f1t), f1t, targets[pidx]), f1t

        sweep = []
        for n_sup in supports:
            sel = order[:n_sup]
            entry = {"support": int(n_sup)}
            for label, sign in (("descent", -1.0), ("ascent", +1.0)):
                delta = np.zeros(hgt * wid, np.float64)
                delta[sel] = sign * sgn_flat[sel]
                val, f1t = evaluate(delta)
                entry[label] = val
                entry[label + "_rel"] = val / dp_base - 1.0
                if label == "descent":
                    entry["seg_plane_delta_max"] = float(np.abs(
                        apply_D(f1t) - apply_D(f1b)).max())
            sweep.append(entry)
        best = min(sweep, key=lambda e: e["descent"])
        rec = {
            "pair": pidx,
            "d_pose_base": dp_base,
            "grad_abs_sum_blind": float(np.abs(g_flat).sum()),
            "sweep": sweep,
            "best_support": best["support"],
            "best_descent_d_pose": best["descent"],
            "improved": bool(best["descent"] < dp_base),
        }
        rows.append(rec)
        if handle is not None:
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
        print(f"[sp] {pos + 1:3d}/{len(idx)} pair {pidx:4d} base "
              f"{dp_base:.8f} | best n={best['support']:,} -> "
              f"{best['descent']:.8f} ({100 * (best['descent'] / dp_base - 1):+.3f}%) "
              f"improved={best['descent'] < dp_base} | "
              f"{time.time() - t_0:.0f}s", flush=True)

    if handle is not None:
        handle.close()
    base = np.array([r["d_pose_base"] for r in rows])
    bst = np.array([r["best_descent_d_pose"] for r in rows])
    per_support = []
    for k, n_sup in enumerate(supports):
        des = np.array([r["sweep"][k]["descent"] for r in rows])
        asc = np.array([r["sweep"][k]["ascent"] for r in rows])
        segd = float(max(r["sweep"][k]["seg_plane_delta_max"] for r in rows))
        # THE BYTE ARITHMETIC.  The decoder cannot run PoseNet (CLAUDE.md "no
        # scorers at inflate time"), so BOTH the pixel address and its sign
        # must be carried: log2(230904) = 17.82 bits + 1 sign bit each.  This
        # is a combinatorial cost, not a statistical estimate.
        b_pair = n_sup * (np.log2(n_blind) + 1.0) / 8.0
        b_tot = b_pair * N_PAIRS
        ds_pose = float(np.sqrt(10.0 * des.mean())
                        - np.sqrt(10.0 * base.mean()))
        ds_rate = 25.0 * b_tot / ORIGINAL_BYTES
        per_support.append({
            "support": int(n_sup),
            "support_frac_of_blind": n_sup / n_blind,
            "addressing_bytes_per_pair": float(b_pair),
            "total_bytes_all_pairs": float(b_tot),
            "descent_mean_rel": float(des.mean() / base.mean() - 1.0),
            "ascent_mean_rel": float(asc.mean() / base.mean() - 1.0),
            "descent_frac_pairs_improved": float(np.mean(des < base)),
            "ascent_frac_pairs_improved": float(np.mean(asc < base)),
            "sign_asymmetry_mean": float(np.mean(asc - des)),
            "worst_seg_plane_delta": segd,
            "delta_S_pose": ds_pose,
            "delta_S_rate": ds_rate,
            "delta_S_joint": ds_pose + ds_rate,
            "pct_of_total_gap_pose": 100.0 * ds_pose / LIVE_TOTAL_GAP,
            "pct_of_total_gap_rate": 100.0 * ds_rate / LIVE_TOTAL_GAP,
            "pct_of_total_gap_joint": 100.0 * (ds_pose + ds_rate)
            / LIVE_TOTAL_GAP,
            "byte_cut_factor_needed_to_break_even": (
                float(ds_rate / -ds_pose) if ds_pose < 0 else None),
        })

    summary = {
        "arm": "ddm_ph5o",
        "probe": "support sweep along the aimed blind direction at fixed +-1 LSB",
        "base_submission": str(sub),
        "n_pairs": len(rows),
        "pairs": [int(v) for v in idx],
        "n_blind_px": n_blind,
        "d_pose_base_mean": float(base.mean()),
        "best_over_support_mean": float(bst.mean()),
        "best_over_support_rel": float(bst.mean() / base.mean() - 1.0),
        "frac_pairs_any_support_improves": float(np.mean(bst < base)),
        "per_support": per_support,
        "KILL_seg_bit_identical_every_cell": bool(
            max(e["worst_seg_plane_delta"] for e in per_support) == 0.0),
        "rows": rows,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n[sp] ===== SUPPORT SWEEP =====", flush=True)
    print(f"[sp] base d_pose {base.mean():.10f}   KILL(seg) bit-identical "
          f"every cell: {summary['KILL_seg_bit_identical_every_cell']}",
          flush=True)
    print(f"[sp] {'support':>9s} {'addr B/pr':>10s} {'total B':>10s} "
          f"{'descent':>11s} {'ascent':>11s} {'impr':>6s} "
          f"{'dS_pose':>10s} {'dS_rate':>10s} {'dS_JOINT':>10s} {'need x':>7s}",
          flush=True)
    for e in per_support:
        need = e["byte_cut_factor_needed_to_break_even"]
        print(f"[sp] {e['support']:9,d} {e['addressing_bytes_per_pair']:10.1f} "
              f"{e['total_bytes_all_pairs']:10.0f} "
              f"{100 * e['descent_mean_rel']:+10.4f}% "
              f"{100 * e['ascent_mean_rel']:+10.4f}% "
              f"{100 * e['descent_frac_pairs_improved']:5.1f}% "
              f"{e['delta_S_pose']:+10.6f} {e['delta_S_rate']:+10.6f} "
              f"{e['delta_S_joint']:+10.6f} "
              f"{(f'{need:.1f}x' if need else '--'):>7s}", flush=True)
    print(f"[sp] best-over-support (oracle n per pair): "
          f"{bst.mean():.10f} = {100 * (bst.mean() / base.mean() - 1):+.4f}%  "
          f"pairs improved {100 * np.mean(bst < base):.1f}%", flush=True)
    print(f"[sp] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
