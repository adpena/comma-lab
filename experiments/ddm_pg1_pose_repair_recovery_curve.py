#!/usr/bin/env python3
"""ddm_pg1 -- the POSE REPAIR TOOL and its RECOVERY CURVE.

WHY THIS EXISTS.  Operator 2026-08-02: *"Rate and SEG are extremely important,
and pose falls out if everything else is done the right way in the right
order"* and *"don't judge moves that produce significant wins on segment and
rate if they hurt pose because pose can be fixed later using techniques and
methods that we already know."*

That ordering claim is only safe to act on if the repair is PRICED.  This tool
answers exactly one question, with a number:

    Given a base whose pose was degraded by a seg/rate move, how far back does
    the repaired solve bring d_pose, at what scorer-evaluation and wall-clock
    cost?

It measures the photometric (a,b) rung-B solve under two policies on the SAME
pairs, same base, same oracle:

  * ``shipped``  -- ``ab_damped_gn`` exactly as the live chain runs it:
                    relins=4, damp_levels=4, float64 search, one float16
                    rounding at the very end.
  * ``repair``   -- ddm_pg1: relins=32, damp_levels=12 (both DERIVED, see
                    ``ddm_v4c_resolve.AB_DAMP_LEVELS_DERIVED``) and
                    ``realized_acceptance=True`` so every candidate is
                    float16-rounded BEFORE it is scored.

The second is monotone on the shipped lattice by construction, so ``repair``
cannot be worse than its own start; it CAN be worse than ``shipped``, because
``shipped`` searches off-lattice and may land luckily.  Both are reported per
pair so that is visible rather than averaged away.

AXIS.  Every number here is ``[macOS-CPU frozen-PoseNet advisory]``,
``score_claim=false``, ``promotable=false``.  d_pose is measured through the
real receiver compose + uint8 + the frozen CPU-torch PoseNet -- never a proxy --
but only ``upstream/evaluate.py`` on byte-closed archive bytes is a score.

SCOPE NOTE THAT TRAVELS WITH EVERY NUMBER.  d_pose is RELATIVE between the two
DELIVERED frames (measured by ddm_bp2: splicing a TRUE GT frame_0 into our pair
scores 3.05-16.66 against the decoded pair's 0.0008 -- four orders WORSE for
being more correct).  So "repair" here means restoring the frame-to-frame
RELATION the frozen PoseNet reads, not making frame_0 resemble anything.

RESUMABILITY (P0).  Per-pair rows are appended and fsynced to a JSONL; re-run
with the same --out to continue.  Nothing is held only in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
import ddm_v4c_resolve as v4c

AXIS = "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE"


def _load_jl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[int(r["pair"])] = r
    return rows


def _pair_site(oracle, comp, ship, solve_rows, pidx: int, pose_source: str):
    """Rebuild the EXACT rung-B site the live chain builds for one pair.

    Mirrors ``ddm_v4c_resolve.run_photo`` so the comparison is in-lineage: same
    pose source, same selector, same s_t, same static compose.
    """
    s_t = float(v4c._d2_row(pidx)["s_t"])
    tp = oracle.targets64[pidx].copy()
    if pose_source == "resolve" and pidx in solve_rows:
        theta = v4c.q16(np.asarray(solve_rows[pidx]["p_best_static"], np.float64))
        sel = int(solve_rows[pidx]["selector"])
    else:
        theta = v4c.q16(np.asarray(ship[pidx]["p"], np.float64))
        sel = int(ship[pidx]["selector"])
    f1_u8 = oracle.f1(pidx)
    f1_f = f1_u8.astype(np.float64)
    wg, wf = comp.warps(f1_f, theta, s_t)

    def pose6(gain: float, bias: float) -> np.ndarray:
        return comp.o.p3v2.pose6_u8(
            comp.o.posenet, comp.compose(wg, wf, sel, gain=gain, bias=bias), f1_u8)

    def mse(p6: np.ndarray) -> float:
        return float(np.mean((p6 - tp) ** 2))

    return pose6, mse, tp, sel, s_t


def run(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(int(args.threads))

    base, base_archive = v4c.resolve_base(args.base, args.base_archive)
    oracle = v4c.build_oracle(base, s_r=1.0, archive=base_archive)
    comp = v4c.StaticComposer(oracle)
    ship = v4c.load_ship_table()
    solve_rows = _load_jl(v4c.OUT / f"solve_{base}.partial.jsonl")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Sharding: disjoint pair subsets, each to its OWN append-only JSONL, so N
    # workers never contend for a file and each stays independently resumable.
    # `_all_shard_rows` re-unions them for the receipt, so a partial fleet still
    # summarizes honestly over whatever pairs actually landed.
    suffix = "" if args.shards == 1 else f".shard{args.shard}of{args.shards}"
    jl = out / f"pg1_recovery_{base}{suffix}.partial.jsonl"
    cache = _load_jl(jl)

    # Pair set.  Default = every pair the pose source can serve, in index order.
    # --hardest-from ranks by an existing rung-B cache so a bounded run measures
    # the pairs that actually carry the pose mass (and says so in the receipt).
    pairs = list(range(int(args.n_pairs)))
    order = "index"
    if args.hardest_from:
        prior = _load_jl(Path(args.hardest_from))
        if not prior:
            raise SystemExit(f"--hardest-from has no rows: {args.hardest_from}")
        key = "d_rungB" if "d_rungB" in next(iter(prior.values())) else "d_ctrl"
        pairs = [p for p, _ in sorted(prior.items(),
                                      key=lambda kv: -float(kv[1][key]))][:int(args.n_pairs)]
        order = f"hardest_by_{key}"
    if args.shards > 1:
        # Stride, not block: every shard gets an interleaved sample of the pair
        # index, so a shard that dies leaves a spread-out hole rather than a
        # contiguous block of the (possibly non-uniform) pair population.
        pairs = pairs[args.shard::args.shards]
    if args.pose_source == "resolve":
        missing = [p for p in pairs if p not in solve_rows]
        if missing:
            print(f"[pg1] pose-source=resolve: {len(missing)}/{len(pairs)} pairs have no "
                  f"solve row; they will use the v4b ship-table pose (recorded per row)",
                  flush=True)

    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[pg1] base={base} n={len(pairs)} order={order} cached={len(cache)} "
          f"repair(relins={v4c.AB_RELINS_DERIVED},damp={v4c.AB_DAMP_LEVELS_DERIVED},"
          f"realized=True)", flush=True)

    for k, pidx in enumerate(pairs):
        if pidx in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[pg1] --max-seconds at {k}/{len(pairs)}; re-run to resume", flush=True)
            break
        pose6, mse, tp, sel, s_t = _pair_site(
            oracle, comp, ship, solve_rows, pidx, args.pose_source)
        cur6 = pose6(1.0, 0.0)
        d_ctrl = mse(cur6)

        # ---- arm 1: the SHIPPED policy, reproduced exactly -------------------
        ts = time.time()
        a_s, b_s, _c6s, _cvs, tr_s = v4c.ab_damped_gn(
            pose6, mse, 1.0, 0.0, cur6, d_ctrl, tp)
        a_sq, b_sq = float(np.float16(a_s)), float(np.float16(b_s))
        d_shipped = mse(pose6(a_sq, b_sq))      # the honest end-quantized re-score
        wall_s = time.time() - ts
        n_ev_s = int(tr_s["n_pose6"]) + 1       # + the end-quantized re-score

        # ---- arm 2: the ddm_pg1 REPAIR policy -------------------------------
        tr0 = time.time()
        a_r, b_r, _c6r, d_repair, tr_r = v4c.ab_damped_gn(
            pose6, mse, 1.0, 0.0, cur6, d_ctrl, tp,
            relins=v4c.AB_RELINS_DERIVED,
            damp_levels=v4c.AB_DAMP_LEVELS_DERIVED,
            realized_acceptance=True)
        wall_r = time.time() - tr0
        n_ev_r = int(tr_r["n_pose6"])
        # realized_acceptance keeps (a,b) ON the f16 lattice throughout, so the
        # returned objective IS the shipped one.  Assert rather than trust.
        assert (float(np.float16(a_r)), float(np.float16(b_r))) == (a_r, b_r), \
            "repair arm left the f16 lattice"

        rec = {
            "pair": int(pidx), "selector": int(sel), "s_t": float(s_t),
            "d_ctrl": float(d_ctrl),
            "d_shipped": float(d_shipped), "d_repair": float(d_repair),
            "a_shipped": a_sq, "b_shipped": b_sq,
            "a_repair": float(a_r), "b_repair": float(b_r),
            "stop_shipped": tr_s["stop_reason"], "stop_repair": tr_r["stop_reason"],
            "n_relin_shipped": int(tr_s["n_relin"]), "n_relin_repair": int(tr_r["n_relin"]),
            "n_pose6_shipped": n_ev_s, "n_pose6_repair": n_ev_r,
            "wall_shipped_s": float(wall_s), "wall_repair_s": float(wall_r),
            "obj_traj_repair": [float(x) for x in tr_r["obj_traj"]],
            "obj_traj_shipped": [float(x) for x in tr_s["obj_traj"]],
            "pose_from_solve_row": bool(args.pose_source == "resolve" and pidx in solve_rows),
            "base": base, "axis": AXIS, "score_claim": False,
        }
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if (k + 1) % 10 == 0 or k == 0:
            print(f"[pg1 {k+1}/{len(pairs)}] pair {pidx} ctrl {d_ctrl:.6g} "
                  f"shipped {d_shipped:.6g} ({tr_s['stop_reason']}) "
                  f"repair {d_repair:.6g} ({tr_r['stop_reason']}) "
                  f"{time.time()-t0:.0f}s", flush=True)
    fj.close()
    _summarize(base, base_archive, _all_shard_rows(out, base), out, order, args)


def _all_shard_rows(out: Path, base: str) -> dict[int, dict]:
    """Union every shard's rows for the receipt. Duplicates are impossible
    (shards are disjoint by construction) but a later row wins if one appears,
    so a re-run never double-counts a pair into the denominator."""
    rows: dict[int, dict] = {}
    for p in sorted(out.glob(f"pg1_recovery_{base}*.partial.jsonl")):
        rows.update(_load_jl(p))
    return rows


def recovery_curve(rows: list[dict], arm: str) -> list[dict]:
    """Mean objective vs relinearization index -- THE recovery curve.

    Each pair's trajectory is held at its final value once it terminates, so
    the mean is over a CONSTANT denominator (all pairs) at every index.  A
    curve that silently drops terminated pairs would show a fake late descent.
    """
    trajs = [r[f"obj_traj_{arm}"] for r in rows if r.get(f"obj_traj_{arm}")]
    if not trajs:
        return []
    width = max(len(t) for t in trajs)
    curve = []
    for i in range(width):
        vals = [t[i] if i < len(t) else t[-1] for t in trajs]
        still = sum(1 for t in trajs if i < len(t) - 1)
        curve.append({"relin": i, "mean_d_pose": float(np.mean(vals)),
                      "sum_d_pose": float(np.sum(vals)),
                      "n_pairs": len(trajs), "n_still_descending": int(still)})
    return curve


def _summarize(base, base_archive, cache, out: Path, order: str,
               args: argparse.Namespace) -> None:
    rows = [cache[k] for k in sorted(cache)]
    if not rows:
        print("[pg1] no rows", flush=True)
        return
    d_ctrl = np.array([r["d_ctrl"] for r in rows])
    d_s = np.array([r["d_shipped"] for r in rows])
    d_r = np.array([r["d_repair"] for r in rows])

    def census(key: str) -> dict:
        by: dict[str, int] = {}
        for r in rows:
            by[r[key]] = by.get(r[key], 0) + 1
        return by

    receipt = {
        "schema": "ddm_pg1_recovery.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": "ddm_pg1",
        "base": base, "base_archive": str(base_archive),
        "base_archive_bytes": int(Path(base_archive).stat().st_size),
        "axis": AXIS, "score_claim": False, "promotable": False,
        "n_pairs": len(rows), "pair_order": order,
        "pose_source": args.pose_source,
        "repair_bounds": {"relins": v4c.AB_RELINS_DERIVED,
                          "damp_levels": v4c.AB_DAMP_LEVELS_DERIVED,
                          "realized_acceptance": True},
        "shipped_bounds": {"relins": v4c.GN_RELINS_PHOTO,
                           "damp_levels": v4c.AB_DAMP_LEVELS,
                           "realized_acceptance": False},
        "mean_d_ctrl": float(d_ctrl.mean()),
        "mean_d_shipped": float(d_s.mean()),
        "mean_d_repair": float(d_r.mean()),
        "sum_d_shipped": float(d_s.sum()),
        "sum_d_repair": float(d_r.sum()),
        # Recovery fraction: how much of the shipped solve's REMAINING distance
        # the repair closes.  Reported against the control (a=1,b=0) start so it
        # is a property of the repair, not of the base's absolute difficulty.
        "recovered_vs_shipped": float(d_s.sum() - d_r.sum()),
        "recovered_frac_of_shipped": (
            float((d_s.sum() - d_r.sum()) / d_s.sum()) if d_s.sum() > 0 else None),
        "pairs_repair_better": int((d_r < d_s).sum()),
        "pairs_repair_equal": int((d_r == d_s).sum()),
        "pairs_repair_worse": int((d_r > d_s).sum()),
        "monotone_vs_ctrl_repair": bool((d_r <= d_ctrl + 1e-15).all()),
        "monotone_vs_ctrl_shipped": bool((d_s <= d_ctrl + 1e-15).all()),
        "stop_census_shipped": census("stop_shipped"),
        "stop_census_repair": census("stop_repair"),
        "mean_n_pose6_shipped": float(np.mean([r["n_pose6_shipped"] for r in rows])),
        "mean_n_pose6_repair": float(np.mean([r["n_pose6_repair"] for r in rows])),
        "cost_ratio_evals": float(np.mean([r["n_pose6_repair"] for r in rows])
                                  / max(np.mean([r["n_pose6_shipped"] for r in rows]), 1e-9)),
        "wall_shipped_s": float(np.sum([r["wall_shipped_s"] for r in rows])),
        "wall_repair_s": float(np.sum([r["wall_repair_s"] for r in rows])),
        "recovery_curve_repair": recovery_curve(rows, "repair"),
        "recovery_curve_shipped": recovery_curve(rows, "shipped"),
        "bytes_added_by_repair": 0,
        "bytes_note": "the repair changes the VALUES of the already-shipped f16 "
                      "(a,b) pair per pair; it adds no section and no bytes.",
        "note": "d_pose measured through the real receiver compose + uint8 + frozen "
                "CPU-torch PoseNet at the rung-B site. ADVISORY; only "
                "upstream/evaluate.py on byte-closed bytes is a score.",
    }
    p = out / f"pg1_recovery_{base}_receipt.json"
    p.write_text(json.dumps(receipt, indent=1) + "\n")
    slim = {k: v for k, v in receipt.items() if not k.startswith("recovery_curve")}
    print(json.dumps(slim, indent=1), flush=True)
    print(f"[pg1] receipt -> {p}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help=f"base label; {tuple(v4c.BASES)} resolve automatically")
    ap.add_argument("--base-archive", default=None,
                    help="solve against any other v3_warp archive (--base names it)")
    ap.add_argument("--pose-source", choices=("resolve", "ship"), default="resolve")
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--hardest-from", default=None,
                    help="rank pairs by d_rungB from an existing rung-B JSONL")
    ap.add_argument("--out", default=str(v4c.OUT / "pg1"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--summarize-only", action="store_true",
                    help="re-union the shard JSONLs and rewrite the receipt")
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise SystemExit(f"--shard must be in [0,{args.shards})")
    if args.summarize_only:
        base, base_archive = v4c.resolve_base(args.base, args.base_archive)
        out = Path(args.out)
        _summarize(base, base_archive, _all_shard_rows(out, base), out,
                   "hardest" if args.hardest_from else "index", args)
        return
    run(args)


if __name__ == "__main__":
    main()
