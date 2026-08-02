#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_mq1 -- monotone-safe CHAINED refine of the v4d pose coordinates, and emit.

WHY THIS AND NOT A MENU REDESIGN.  ddm_mq1 measured both halves of the pose
payload separately on the same 48 mass-ordered pairs (86.5% of the population
d_pose mass), decomposing each coordinate's recoverable distortion into what a
finer STORAGE LATTICE could buy versus what a better SEARCH could buy at
today's lattice:

    coordinate   gap_LATTICE (% of gap)   gap_SEARCH (% of gap)
    p0 forward            0.0213                  0.141      <- negative control
    p1 lateral            0.0128                  0.469
    p2 vertical           0.0107                  0.874
    beta                    --                    0.336
    ---------------------------------------------------------
    TOTAL                 0.045                   1.82       (33x)

Plus the entropy-coding ceiling on all three per-pair index streams: 123 B =
0.0106% of the gap.  So every FORMAT lever -- lattice, codebook, menu
placement, conditional entropy coding -- is collectively worth <=0.06% of the
gap, while the SEARCH over the same variables is worth >=1.8%.  The menu was
never the binding constraint.

This tool banks the search half.  ``p1`` and ``p2`` ship as plain f16 columns
already (``ddm_v4d_build_composed_archive.py:176-184``), so a better value in
those columns costs **ZERO additional archive bytes**.  ``beta`` ships as an
index into the manifest table ``rs_beta_mags`` (``inflate_runner_v4d.py:127,
177-180``), which accepts any float, so a polished beta costs only the widened
table and index entropy -- and no receiver change whatsoever.

MECHANISM.  Sequential coordinate refine (chained, not independent): each
coordinate is bracketed with a self-terminating Swann outward search and
polished by golden section, and the pose is UPDATED before the next coordinate
is refined, so the reported joint gain is realized rather than assumed additive.
Every step accepts only a STRICT decrease of the realized scorer starting from
the shipped solution, so the emitted row can never be worse than the shipped
row, and pairs that are not visited keep their shipped solution unchanged.

THE HONEST LIMIT (ddm_mq1 positive control, 16 pairs, wrong-init restart).  The
objective largely agrees across starts (median |recovery| 0.35%, 11/16 within
1%) but the ARGMIN does not (only 5/16 agree on the location within 2 fine
steps), and 3/16 wrong-init restarts found a STRICTLY BETTER optimum than the
from-shipped search.  The objective is flat and multi-modal in these
coordinates.  Consequence: the values emitted here are a monotone IMPROVEMENT,
they are NOT the continuous optimum, and the distribution of emitted values is
NOT an unbiased reference density -- it must not be used to fit a codebook.

[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE; score_claim=false.
d_seg is untouched by construction: frame_1 is never modified and SegNet reads
``x[:, -1]`` only (upstream/modules.py:108).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/Projects/pact")
SCHEMA = "ddm_mq1_joint_pose_refine.v1"
V4D = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
N_PAIRS = 600
BETA_MAGS = (0.0, 0.5, 1.0)
BETA_STEP0 = 0.5
GOLDEN_TOL_FRAC = 0.02


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="celldrop50")
    ap.add_argument("--final-jsonl", type=Path, default=V4D / "pw1/final_pw1.jsonl")
    ap.add_argument("--out-dir", type=Path, default=V4D / "mq1_emit")
    ap.add_argument("--pairs", type=int, default=150,
                    help="refine the N highest-d_pose pairs (mass-ordered); "
                         "unvisited pairs keep their shipped solution")
    ap.add_argument("--order", default="p2,p1,beta",
                    help="chained refine order; measured gap_search descending")
    ap.add_argument("--beta-fine-step", type=float, default=BETA_STEP0 / 10.0)
    ap.add_argument("--max-minutes", type=float, default=600.0)
    ap.add_argument("--max-expand", type=int, default=12)
    ap.add_argument("--emit-final-jsonl", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c
    from mq1_pose_lattice_resolution_probe import bracket_out, golden, ulp16

    steps = [s.strip() for s in args.order.split(",") if s.strip()]
    if any(s not in ("p0", "p1", "p2", "beta") for s in steps):
        raise SystemExit("--order entries must be from p0,p1,p2,beta")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shipped = {int(r["pair"]): r for r in
               (json.loads(ln) for ln in
                args.final_jsonl.read_text().splitlines() if ln.strip())}
    if len(shipped) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} shipped rows, got {len(shipped)}")

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    dim0_offset = float(json.loads(
        (V4D / "v4d_composed_pw1_build_receipt.json").read_text())["dim0_offset"])

    def score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g):
        """Realized d_pose; mirrors ddm_v4d_resolve._beta_select (:245-259)."""
        beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
        wg_t, wf_t = comp.warps(f1_f, pose, s_t, 1.0 - beta / 2.0)
        wg_b, wf_b = comp.warps(f1_f, pose, s_t, 1.0 + beta / 2.0)
        f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
        f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
        f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
        if a != 1.0 or b != 0.0:
            f0 = a * f0 + b
        p6 = comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0), f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    def shippable(col: int, x: float) -> float:
        """The value the RECEIVER reconstructs for this pose column."""
        if col == 0:
            return dim0_offset + float(np.float16(x - dim0_offset))
        return float(np.float16(x))

    order = sorted(range(N_PAIRS), key=lambda p: -shipped[p]["d_final"])
    seq = order[: args.pairs]
    jl = args.out_dir / "mq1_emit.jsonl"
    cache = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in (jl.read_text().splitlines() if jl.exists() else [])
             if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[mq1emit] base={args.base} pairs={len(seq)} order={steps} "
          f"cached={len(cache)}", flush=True)

    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[mq1emit] wall cap; {len(cache)} done; rerun to resume",
                  flush=True)
            break
        sh = shipped[pidx]
        pose = np.asarray(sh["p"], np.float64).copy()
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        g = (float(sh["beta_mag"]) if "beta_mag" in sh
             else float(BETA_MAGS[int(sh["beta_idx"])]))
        s_t = float(v4c._d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)

        d_cur = score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g)
        d_ctrl = d_cur
        n_eval = 1

        # Explicit evaluator factories: every per-pair variable is bound at
        # CALL time, so no closure reads a mutated loop variable.  ``g`` is
        # deliberately bound when the factory is built, which is what makes a
        # ``beta`` step earlier in the chain visible to a later column step.
        ctx = (s_t, sel, f1_u8, f1_f, tp, a, b)

        def make_beta_eval(p, _ctx=ctx):
            def _ev(x):
                return score(p, *_ctx, x)
            return _ev

        def make_col_eval(p, c, gnow, _ctx=ctx):
            def _ev(x):
                q = p.copy()
                q[c] = x
                return score(q, *_ctx, gnow)
            return _ev

        for step in steps:
            if step == "beta":
                ev_b = make_beta_eval(pose)
                lo, hi, bx, bd, nb = bracket_out(ev_b, g, d_cur,
                                                 args.beta_fine_step,
                                                 args.max_expand)
                gx, gd, ng = golden(ev_b, lo, hi, bx, bd,
                                    tol=GOLDEN_TOL_FRAC * args.beta_fine_step)
                n_eval += nb + ng
                if gd < d_cur:          # STRICT decrease only
                    g, d_cur = float(gx), gd
                continue
            col = int(step[1])
            x0 = float(pose[col])
            st0 = ulp16(x0 - dim0_offset) if col == 0 else ulp16(x0)
            ev_c = make_col_eval(pose, col, g)

            lo, hi, bx, bd, nb = bracket_out(ev_c, x0, d_cur, st0, args.max_expand)
            gx, gd, ng = golden(ev_c, lo, hi, bx, bd, tol=0.05 * st0)
            # Commit only what the receiver can actually reconstruct.
            xq = shippable(col, gx)
            dq = ev_c(xq)
            n_eval += nb + ng + 1
            if dq < d_cur:              # STRICT decrease only
                pose[col], d_cur = xq, dq

        rec = {"pair": int(pidx), "rank": rank,
               "d_shipped": float(sh["d_final"]), "d_ctrl": d_ctrl,
               "canary_abs_err": abs(d_ctrl - float(sh["d_final"])),
               "p": [float(v) for v in pose], "a": a, "b": b,
               "selector": sel, "beta_mag": float(g),
               "d_final": float(d_cur), "gain": max(d_ctrl - d_cur, 0.0),
               "s_t": s_t, "n_eval": n_eval, "source": "mq1_joint_refine"}
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if len(cache) % 5 == 0 or rank < 3:
            done = list(cache.values())
            print(f"[mq1emit {len(cache):4d}/{len(seq)}] pair {pidx} "
                  f"{d_ctrl:.5f} -> {d_cur:.5f} gain {rec['gain']:.3e} | "
                  f"sum {sum(r['gain'] for r in done):.5f} "
                  f"canary {max(r['canary_abs_err'] for r in done):.1e} "
                  f"{time.time() - t0:.0f}s", flush=True)

    fj.close()
    summarize(args, cache, shipped, steps)


def summarize(args, cache: dict, shipped: dict, steps: list[str]) -> None:
    rows = [cache[p] for p in sorted(cache)]
    if not rows:
        print("[mq1emit] no rows")
        return
    d_ship = np.array([shipped[p]["d_final"] for p in range(N_PAIRS)])
    base_mean = float(d_ship.mean())
    floor = max(r["canary_abs_err"] for r in rows)
    # Count the gain that is ACTUALLY BANKED, i.e. measured against the SHIPPED
    # value and only where the emit will really replace the row.  Using the
    # in-instrument `gain` (against d_ctrl) would over-count by the canary floor
    # on any pair where d_ctrl drifted above d_shipped.
    gain = np.array([max(shipped[r["pair"]]["d_final"] - r["d_final"], 0.0)
                     for r in rows])
    new_mean = base_mean - float(gain.sum()) / N_PAIRS
    s_now, s_bar = 0.9476091, 0.172141
    gap = s_now - s_bar
    dS = float(np.sqrt(10.0 * new_mean) - np.sqrt(10.0 * base_mean))

    # Emit the merged n600 JSONL: refined rows replace shipped rows, every
    # other pair keeps its shipped solution verbatim.
    n_replaced = 0
    if args.emit_final_jsonl:
        with open(args.emit_final_jsonl, "w") as out:
            for p in range(N_PAIRS):
                src = shipped[p]
                if p in cache and cache[p]["d_final"] < src["d_final"]:
                    r = cache[p]
                    row = {"pair": p, "p": r["p"], "a": r["a"], "b": r["b"],
                           "selector": r["selector"], "beta_idx": 0,
                           "beta_mag": r["beta_mag"], "d_final": r["d_final"],
                           "source": "mq1_joint_refine"}
                    n_replaced += 1
                else:
                    row = dict(src)
                out.write(json.dumps(row) + "\n")

    betas = sorted({round(float(r.get("beta_mag", 0.0)), 4) for r in
                    ([cache[p] for p in cache] +
                     [shipped[p] for p in range(N_PAIRS)])})
    out = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False, "promotion_eligible": False, "research_only": True,
        "pointer_moved": False,
        "base": args.base, "order": steps,
        "n_pairs_refined": len(rows), "n_population": N_PAIRS,
        "n_pairs_improved": int((gain > 0).sum()),
        "n_pairs_improved_above_floor": int((gain > floor).sum()),
        "canary_max_abs_err": floor,
        "d_pose_mean_shipped": base_mean,
        "d_pose_mean_refined": new_mean,
        "pose_contribution_shipped": float(np.sqrt(10.0 * base_mean)),
        "pose_contribution_refined": float(np.sqrt(10.0 * new_mean)),
        "delta_S_distortion_only": dS,
        "pct_of_gap_distortion_only": 100.0 * abs(dS) / gap,
        "gap_to_bar": gap,
        "emitted_final_jsonl": (str(args.emit_final_jsonl)
                                if args.emit_final_jsonl else None),
        "emitted_rows_replaced": n_replaced,
        "distinct_beta_values": len(betas),
        # ddm_v4d_build_composed_archive.derive_beta_table:134 fails closed above
        # 256 entries because beta_idx is uint8.  A continuous per-pair beta
        # therefore CANNOT scale past ~256 refined pairs without quantisation --
        # which is the one condition under which a codebook over beta becomes
        # necessary here, forced by the format rather than chosen by RD.
        "beta_table_uint8_headroom": 256 - len(betas),
        "beta_table_exceeds_uint8": len(betas) > 256,
        "note": "delta_S_distortion_only EXCLUDES the archive-byte delta from "
                "the widened rs_beta_mags table; the byte-closed row is the "
                "authority.  p1/p2 changes are byte-free (plain f16 columns).",
        "generated_by": "tools/mq1_joint_pose_refine_emit.py",
    }
    (args.out_dir / "mq1_emit_receipt.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
