# SPDX-License-Identifier: MIT
"""ddm_tt1 — joint payload gradient-TTO on the v4c continuous DOF (QA71).

THE REVIVED FORM (ph3 doctrine): the frozen-scorer gradient is the PROPOSAL
ENGINE, never the acceptor.  Per pair, warm-started from the v4c shipped point
(pose 6-vector + photometric a,b), take an Adam step on the JOINT (pose, a, b)
gradient from the differentiable twin (ONE backward, MPS-capable), then REALIZE
each backtracking-LR candidate through the EXACT numpy decode + CPU PoseNet and
ACCEPT iff realized d_pose descends at the shipped f16 quantization.  Monotone by
construction (rejected steps roll back — the archive can only improve).

FROZEN-SCORER FACTORIZATION (upstream/modules.py:108): pose+(a,b) touch ONLY
frame_0; SegNet reads frame_1 only, so d_seg + rate are INVARIANT on these DOF.
=> per-pair realized ΔS < 0  <=>  realized d_pose decreases (the contribution
sqrt(10*mean) is monotone in every pair's d_pose).  d_seg/rate are cited frozen.

ABLATION (freeze-one-stream) — the cross-stream pose×gain coupling attribution:
  --mode pose_only : optimize pose6, hold (a,b) at the v4c shipped value
  --mode ab_only   : optimize (a,b), hold pose at the v4c shipped value
  --mode joint     : optimize (pose6, a, b) jointly
  C = pose_only gain ; B = joint gain ; A = v4c baseline.
  B - C = the pose×gain coupling win ; C - A = analytic-vs-numerical-GN win.

Axis: [macOS advisory].  MPS = gradient/proposal device ONLY (fp32; NEVER a
score).  Realized authority = CPU numpy decode + CPU PoseNet (the gate
instrument, proven bit-exact in the acceptor control).  Pointer 0.1910828242
[contest-CPU] UNMOVED.  score_claim=false.  ONE bounded scorer job at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

import numpy as np

_REPO = Path("/Users/adpena/Projects/pact")
_V4C = Path("/Volumes/VertigoDataTier/pact/ddm_v4c_20260730")
_SSD = Path("/Volumes/VertigoDataTier/pact/ddm_tt1_20260731")
for _p in (_REPO / "src", _REPO / "experiments", _REPO / "src/tac/optimization",
           Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ddm_tt1_twin import WarpTwin


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def contribution(mean_d: float) -> float:
    return float(np.sqrt(10.0 * float(mean_d)))


def load_v4c_baseline() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    jl = _V4C / "photo_celldrop50_resolve.partial.jsonl"
    for ln in jl.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


def stratified_subset(base: dict[int, dict], n_hard: int, n_ctrl: int) -> list[int]:
    """worst-joint pairs (highest shipped d_rungB) + spread controls."""
    order = sorted(base, key=lambda i: base[i]["d_rungB"], reverse=True)
    hard = order[:n_hard]
    rest = order[n_hard:]
    if n_ctrl and rest:
        idx = np.linspace(0, len(rest) - 1, n_ctrl).round().astype(int)
        ctrl = [rest[j] for j in dict.fromkeys(idx.tolist())]
    else:
        ctrl = []
    return sorted(set(hard) | set(ctrl))


def _f16(x) -> np.ndarray:
    return np.asarray(x, np.float64).astype(np.float16).astype(np.float64)


def tto_pair(tw: WarpTwin, i: int, mode: str, *, steps: int, lrs: list[float],
             max_seconds: float) -> dict:
    """Warm-start from v4c shipped; Adam-propose (twin) + realized-accept."""
    import torch

    p0 = tw.dec.p_best[i].astype(np.float64).copy()
    a0 = float(tw.dec.ab[i][0])
    b0 = float(tw.dec.ab[i][1])
    d_base = tw.acceptor_d_pose(i, p0, a0, b0)

    best_p, best_a, best_b, best_d = p0.copy(), a0, b0, d_base
    n_accept = 0
    n_realize = 0
    acc_family = {"both": 0, "pose": 0, "ab": 0}
    t0 = time.time()

    opt_pose = mode in ("joint", "pose_only")
    opt_ab = mode in ("joint", "ab_only")
    # per-BLOCK unit-normalization so wildly-different DOF scales (pose rotation
    # grad ~10 vs a,b grad ~0.1) each get a meaningful step at the SAME lr.  The
    # (a,b) block lives at O(1); scale it up so lr=1e-2 moves it ~0.5 while lr
    # moves pose ~1e-2.  joint mode = BEST-OF {both, pose-only, ab-only} realized
    # candidates per step (order-independent steepest realized descent) so joint
    # dominates each block AND captures superadditive coupling (both-family
    # accepts); pose_only / ab_only run their single block.
    AB_SCALE = 50.0
    if mode == "joint":
        families = ("both", "pose", "ab")
    elif mode == "pose_only":
        families = ("pose",)
    else:
        families = ("ab",)

    pt = torch.tensor(best_p, dtype=torch.float32, device=tw.dev, requires_grad=opt_pose)
    at = torch.tensor(best_a, dtype=torch.float32, device=tw.dev, requires_grad=opt_ab)
    bt = torch.tensor(best_b, dtype=torch.float32, device=tw.dev, requires_grad=opt_ab)
    n_noaccept = 0
    for _step in range(steps):
        if max_seconds and (time.time() - t0) > max_seconds:
            break
        for pp in (pt, at, bt):
            if pp.grad is not None:
                pp.grad = None
        d = tw.d_pose_diff(i, pt, at, bt)
        d.backward()
        base_p, base_a, base_b = best_p.copy(), best_a, best_b
        gp = pt.grad.detach().cpu().numpy().copy() if opt_pose else np.zeros(6)
        ga = float(at.grad.detach().cpu()) if opt_ab else 0.0
        gb = float(bt.grad.detach().cpu()) if opt_ab else 0.0
        pnorm = float(np.linalg.norm(gp)) if opt_pose else 0.0
        abnorm = float(np.hypot(ga, gb)) if opt_ab else 0.0
        pose_dir = gp / pnorm if pnorm > 1e-30 else np.zeros(6)
        a_dir = (ga / abnorm) if abnorm > 1e-30 else 0.0
        b_dir = (gb / abnorm) if abnorm > 1e-30 else 0.0
        if pnorm < 1e-30 and abnorm < 1e-30:
            break
        best_cand = None  # (d_real, fam, p, a, b)
        for lr in lrs:
            dp = lr * pose_dir
            da = lr * AB_SCALE * a_dir
            db = lr * AB_SCALE * b_dir
            for fam in families:
                cand_p = base_p - dp if fam in ("both", "pose") else base_p
                cand_a = base_a - da if fam in ("both", "ab") else base_a
                cand_b = base_b - db if fam in ("both", "ab") else base_b
                n_realize += 1
                d_real = tw.acceptor_d_pose(i, cand_p, cand_a, cand_b)
                if best_cand is None or d_real < best_cand[0]:
                    best_cand = (d_real, fam, cand_p, cand_a, cand_b)
        if best_cand is not None and best_cand[0] < best_d - 1e-9:
            best_d, fam, cp, ca, cb = best_cand
            best_p = _f16(cp) if opt_pose else best_p
            best_a = float(_f16(ca)) if opt_ab else best_a
            best_b = float(_f16(cb)) if opt_ab else best_b
            n_accept += 1
            acc_family[fam] += 1
            n_noaccept = 0
        else:
            n_noaccept += 1
        with torch.no_grad():
            pt.copy_(torch.tensor(best_p, dtype=torch.float32, device=tw.dev))
            at.copy_(torch.tensor(best_a, dtype=torch.float32, device=tw.dev))
            bt.copy_(torch.tensor(best_b, dtype=torch.float32, device=tw.dev))
        if n_noaccept >= 2:
            break  # fully stuck at this operating point
    return {
        "pair": i, "mode": mode, "d_base": float(d_base), "d_final": float(best_d),
        "delta_d": float(best_d - d_base), "n_accept": n_accept,
        "acc_family": acc_family, "n_realize": n_realize,
        "wall_s": round(time.time() - t0, 2),
        "a_base": a0, "a_final": float(best_a), "b_base": b0, "b_final": float(best_b),
        "pose_moved_l2": float(np.linalg.norm(_f16(best_p) - p0)),
    }


def run(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(4)
    base = load_v4c_baseline()
    subset = (stratified_subset(base, args.n_hard, args.n_ctrl)
              if args.pairs is None else [int(x) for x in args.pairs.split(",")])
    tw = WarpTwin(_SSD / "archive_v4c", device=args.device)
    lrs = [float(x) for x in args.lrs.split(",")]
    work = _SSD / f"tto_{args.mode}.partial.jsonl"
    cache: dict[int, dict] = {}
    if work.exists() and args.resume:
        for ln in work.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                cache[int(r["pair"])] = r
        print(f"[tt1 {args.mode}] resume: {len(cache)} cached", flush=True)
    fj = open(work, "a")  # noqa: SIM115
    t0 = time.time()
    for i in subset:
        if i in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[tt1 {args.mode}] --max-seconds at pair {i}; re-run --resume", flush=True)
            break
        row = tto_pair(tw, i, args.mode, steps=args.steps, lrs=lrs,
                       max_seconds=args.per_pair_seconds)
        fj.write(json.dumps(row) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[i] = row
        tag = "WIN " if row["delta_d"] < -1e-6 else "----"
        print(f"  [{tag} {i:3d}] d {row['d_base']:.6f} -> {row['d_final']:.6f} "
              f"(Δ{row['delta_d']:+.6f}) acc={row['n_accept']}/{row['n_realize']} "
              f"{row['wall_s']:.0f}s", flush=True)
    fj.close()

    done = [cache[i] for i in subset if i in cache]
    if done:
        db = np.array([r["d_base"] for r in done])
        df = np.array([r["d_final"] for r in done])
        wins = int((df < db - 1e-6).sum())
        # subset-level realized contribution + ΔS (seg/rate frozen on these DOF)
        c_base = contribution(float(db.mean()))
        c_final = contribution(float(df.mean()))
        # full-600 projection: replace these pairs' d in the 600-mean, hold rest
        d600 = np.array([base[i]["d_rungB"] for i in range(600)], np.float64)
        d600_new = d600.copy()
        for r in done:
            d600_new[r["pair"]] = r["d_final"]
        S_seg = 100 * 0.00431179
        S_rate = 25 * 359750 / 37545489
        S600_base = S_seg + contribution(float(d600.mean())) + S_rate
        S600_new = S_seg + contribution(float(d600_new.mean())) + S_rate
        wall_h = (time.time() - t0) / 3600.0
        receipt = {
            "schema": "ddm_tt1_joint_tto.v1", "utc": _utc(), "mode": args.mode,
            "axis": "[macOS advisory - realized CPU decode+PoseNet]",
            "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "device_proposal": args.device, "n_pairs_done": len(done),
            "n_wins": wins, "lrs": lrs, "steps": args.steps,
            "subset_d_base_mean": float(db.mean()), "subset_d_final_mean": float(df.mean()),
            "subset_contribution_base": c_base, "subset_contribution_final": c_final,
            "subset_delta_contribution": c_final - c_base,
            "full600_projection": {
                "d_pose_mean_base": float(d600.mean()),
                "d_pose_mean_new": float(d600_new.mean()),
                "S600_base": S600_base, "S600_new": S600_new,
                "delta_S600": S600_new - S600_base,
                "note": "seg 0.00431179 + rate(359,750B) FROZEN (frame_1-only "
                        "factorization); only pose contribution moves.",
            },
            "wall_hours": wall_h,
            "delta_S600_per_hour": (S600_new - S600_base) / wall_h if wall_h > 0 else 0.0,
        }
        (_SSD / f"tto_{args.mode}_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
        print(json.dumps(receipt, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("joint", "pose_only", "ab_only"), required=True)
    ap.add_argument("--device", default="mps", help="twin PROPOSAL device (mps|cpu)")
    ap.add_argument("--n-hard", type=int, default=35)
    ap.add_argument("--n-ctrl", type=int, default=15)
    ap.add_argument("--pairs", default=None, help="explicit comma pair list (overrides subset)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3,1e-2",
                    help="absolute step magnitudes on the unit-normalized joint gradient")
    ap.add_argument("--per-pair-seconds", type=float, default=90.0)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
