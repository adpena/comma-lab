#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pw1 — price the TWO saturating menus in the live v4d pose solve.

The live v4d refine (``experiments/ddm_v4d_resolve.py``) bounds two searches
with hardcoded constants, and the shipped solution SATURATES both:

* ``_refine_dim0`` (``:177-184``) searches dim0 only over ``d0 +- 0.048``
  (coarse) then ``+- 0.006`` (fine).  MEASURED on the shipped
  ``final_refine.jsonl``: 124/600 pairs moved >= 0.0475, i.e. they ran into
  the bound, and those pairs carry 37.4% of the population d_pose mass.
* ``_beta_select`` (``:240-262``) picks the rolling-shutter magnitude from
  ``BETA_MAGS = (0.0, 0.5, 1.0)`` (``:63``).  MEASURED: 76/600 pairs select
  the TOP entry 1.0 and carry 26.4% of the mass; the menu has nothing above.

Both bounds are rate-free to remove: dim0 ships as an f16 residual off a
single manifest offset (``ddm_v4d_build_composed_archive.py:123-133``) and
``beta_idx`` ships as a ``uint8`` column (``:116-121``), so a wider dim0
range and a longer beta menu cost ZERO additional archive bytes per pair.

This tool measures what those two constants cost, as a monotone-safe
CONTINUATION of the shipped solve rather than a re-solve: every arm starts
from the shipped per-pair solution and accepts only a STRICT improvement at
the same realized scorer, so an arm can never report a win it did not
realize, and the delta is attributable to exactly one removed bound.

Neither replacement is a bigger constant.  Both are self-terminating
bracketing searches: step outward while the objective improves, doubling the
step, and stop at the first non-improving probe (Swann-style bracketing).
Termination is a proof, not a budget -- the objective is bounded below by 0
and the reachable dim0 values live on a finite float16 lattice.

Arms (change ONE thing per measurement):
  CTRL  re-score the shipped (pose, a, b, beta_idx).  This is the CANARY:
        it must reproduce ``d_final`` from the shipped JSONL.
  A     dim0 bracket-expansion past the +-0.048/0.006 bound, shipped beta.
  B     beta menu extension past 1.0, shipped dim0.
  AB    A then B (B re-run at A's dim0).

[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE; score_claim=false.
d_seg is untouched by construction: frame_1 is never modified and SegNet
reads ``x[:, -1]`` only (upstream/modules.py:108).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_pw1_pose_menu_saturation_ab.v1"
V4D = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
N_PAIRS = 600
# The live coarse step; used as the FIRST outward step of the bracketing
# search so arm A continues the shipped search on its own lattice.
DIM0_STEP0 = 0.012
# The live beta menu spacing; used as the FIRST outward step past 1.0.
BETA_STEP0 = 0.5
# The v4d seed menu (experiments/ddm_v4d_resolve.py:63) that beta_idx indexes.
BETA_MAGS = (0.0, 0.5, 1.0)


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)


def bracket_out(evaluate, x0: float, d0: float, step0: float,
                max_expand: int, quantize=None):
    """Swann-style outward bracketing; self-terminating, no budget knob.

    Probe +step and -step from x0.  If neither improves, x0 already brackets
    the minimum on this lattice and the search STOPS.  Otherwise commit to the
    improving direction and double the step until a probe fails to improve.

    MONOTONE-SAFE by construction: ``best_d`` starts at ``d0`` and is only
    replaced on a STRICT decrease, so the returned d is never worse than the
    starting point.  Returns (best_x, best_d, n_eval, probes).
    """
    best_x, best_d, n = x0, d0, 0
    probes: list[dict] = []
    direction, step = 0.0, step0
    for sign in (1.0, -1.0):
        cand = x0 + sign * step0
        cand = quantize(cand) if quantize else cand
        if cand == best_x:
            continue
        dv = evaluate(cand)
        n += 1
        probes.append({"x": cand, "d": dv, "phase": "probe"})
        if dv < best_d:
            best_x, best_d, direction = cand, dv, sign
            break
    if direction == 0.0:
        return best_x, best_d, n, probes
    for _ in range(max_expand):
        step *= 2.0
        cand = best_x + direction * step
        cand = quantize(cand) if quantize else cand
        if cand == best_x:
            break
        dv = evaluate(cand)
        n += 1
        probes.append({"x": cand, "d": dv, "phase": "expand"})
        if dv >= best_d:
            break
        best_x, best_d = cand, dv
    return best_x, best_d, n, probes


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="celldrop50",
                    help="v4c BASES key the shipped solve was fit on")
    ap.add_argument("--final-jsonl", type=Path,
                    default=V4D / "final_refine.jsonl")
    ap.add_argument("--out-dir", type=Path, default=V4D / "pw1")
    ap.add_argument("--pairs", type=int, default=N_PAIRS,
                    help="solve the N highest-d_pose pairs (mass-ordered)")
    ap.add_argument("--max-minutes", type=float, default=180.0)
    ap.add_argument("--emit-final-jsonl", type=Path, default=None,
                    help="write a v4d final JSONL carrying the arm-AB "
                         "solution so the win can be byte-closed")
    ap.add_argument("--max-expand", type=int, default=12,
                    help="hard safety stop on outward probes; the search "
                         "terminates on its own bracket well before this")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c
    import ddm_v4d_resolve as v4d

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shipped = {int(r["pair"]): r for r in
               (json.loads(ln) for ln in
                args.final_jsonl.read_text().splitlines() if ln.strip())}
    if len(shipped) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} shipped rows, got {len(shipped)}")

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    photo = v4d.load_photo(v4d.PHOTO_JL)
    offset = v4d._dim0_offset(photo)

    def lattice(x: float) -> float:
        """The value the RECEIVER reconstructs for a requested dim0."""
        return offset + float(np.float16(x - offset))

    def score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g):
        """Realized d_pose at rolling-shutter magnitude g.

        Mirrors ``ddm_v4d_resolve._beta_select`` (:245-259) exactly; at g=0
        the two warps coincide and this reduces to ``_score_at``.
        """
        beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
        wg_t, wf_t = comp.warps(f1_f, pose, s_t, 1.0 - beta / 2.0)
        wg_b, wf_b = comp.warps(f1_f, pose, s_t, 1.0 + beta / 2.0)
        f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
        f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
        f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
        if a != 1.0 or b != 0.0:
            f0 = a * f0 + b
        p6 = comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0),
                                  f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    order = sorted(range(N_PAIRS), key=lambda p: -shipped[p]["d_final"])
    seq = order[: args.pairs]
    jl = args.out_dir / "pw1_arms.jsonl"
    cache = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in (jl.read_text().splitlines() if jl.exists() else [])
             if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[pw1] base={args.base} offset={offset:.6f} pairs={len(seq)} "
          f"cached={len(cache)}", flush=True)

    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[pw1] wall cap; {len(cache)} done; rerun to resume",
                  flush=True)
            break
        sh = shipped[pidx]
        pose = np.asarray(sh["p"], np.float64).copy()
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        # Same class as the emit path: prefer an explicit beta_mag; a
        # beta_idx of -1 must NOT wrap to the last seed entry.
        if "beta_mag" in sh:
            g_ship = float(sh["beta_mag"])
        else:
            _bi = int(sh["beta_idx"])
            if not 0 <= _bi < len(BETA_MAGS):
                raise SystemExit(f"pair {pidx}: beta_idx={_bi} with no beta_mag")
            g_ship = float(BETA_MAGS[_bi])
        s_t = float(v4c._d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        n_eval = 0

        # CTRL: the canary.  Must reproduce the shipped d_final.
        d_ctrl = score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g_ship)
        n_eval += 1

        # Per-pair evaluator factories; every pair variable is bound at call
        # time (no closure over the loop variable).
        def make_dim0_eval(p, g, ctx=(s_t, sel, f1_u8, f1_f, tp, a, b)):
            def _ev(x):
                q = p.copy()
                q[0] = x
                return score(q, *ctx, g)
            return _ev

        def make_beta_eval(p, ctx=(s_t, sel, f1_u8, f1_f, tp, a, b)):
            def _ev(g):
                return score(p, *ctx, g)
            return _ev

        # ARM A: dim0 bracket-expansion at the shipped beta.
        x_a, d_a, na, probes_a = bracket_out(
            make_dim0_eval(pose, g_ship), float(pose[0]), d_ctrl,
            DIM0_STEP0, args.max_expand, quantize=lattice)
        n_eval += na

        # ARM B: beta menu extension at the shipped dim0.
        g_b, d_b, nb, probes_b = bracket_out(
            make_beta_eval(pose), g_ship, d_ctrl, BETA_STEP0, args.max_expand)
        n_eval += nb

        # ARM AB: B re-run at A's dim0.
        pose_ab = pose.copy()
        pose_ab[0] = x_a
        g_ab, d_ab, nab, _ = bracket_out(
            make_beta_eval(pose_ab), g_ship, d_a, BETA_STEP0, args.max_expand)
        n_eval += nab

        rec = {
            "pair": int(pidx), "rank": rank,
            "d_shipped": float(sh["d_final"]), "d_ctrl": d_ctrl,
            "canary_abs_err": abs(d_ctrl - float(sh["d_final"])),
            "dim0_shipped": float(pose[0]), "g_shipped": float(g_ship),
            "s_t": s_t, "selector": sel,
            "arm_a_dim0": float(x_a), "arm_a_d": float(d_a),
            "arm_a_probes": probes_a,
            "arm_b_g": float(g_b), "arm_b_d": float(d_b),
            "arm_b_probes": probes_b,
            "arm_ab_dim0": float(x_a), "arm_ab_g": float(g_ab),
            "arm_ab_d": float(d_ab),
            "n_eval": n_eval,
        }
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if len(cache) % 10 == 0 or rank < 5:
            arr = list(cache.values())
            print(f"[pw1 {len(cache):3d}/{len(seq)}] pair {pidx} "
                  f"ship {sh['d_final']:.5f} ctrl {d_ctrl:.5f} "
                  f"A {d_a:.5f} B {d_b:.5f} AB {d_ab:.5f} | "
                  f"canary_max {max(r['canary_abs_err'] for r in arr):.2e} "
                  f"{time.time()-t0:.0f}s", flush=True)

    fj.close()
    summarize(args, cache, shipped)


def summarize(args, cache: dict, shipped: dict) -> None:
    rows = [cache[p] for p in sorted(cache)]
    if not rows:
        print("[pw1] no rows")
        return
    d_ship_all = np.array([shipped[p]["d_final"] for p in range(N_PAIRS)])
    base_mean = float(d_ship_all.mean())

    # P2 NOISE FLOOR.  CTRL re-scores the shipped solution through this tool's
    # beta path; at g=0 that path computes (1-a)*x + a*x, which is not
    # bit-identical to x in floating point and can flip a pixel across the
    # uint8 rounding boundary.  The resulting CTRL-vs-shipped disagreement is
    # the instrument's floor, and every arm-vs-shipped delta carries it.
    # ``delta_vs_ctrl`` is measured entirely inside the instrument (same path
    # for baseline and arm) and is therefore floor-free.
    floor = max(r["canary_abs_err"] for r in rows)

    def composed(key: str) -> dict:
        d = d_ship_all.copy()
        dc = d_ship_all.copy()
        for r in rows:
            # monotone-safe: an arm ships only a STRICT improvement
            d[r["pair"]] = min(d[r["pair"]], r[key])
            dc[r["pair"]] = min(r["d_ctrl"], r[key])
        m, mc = float(d.mean()), float(dc.mean())
        base_ctrl = d_ship_all.copy()
        for r in rows:
            base_ctrl[r["pair"]] = r["d_ctrl"]
        bc = float(base_ctrl.mean())
        return {"d_pose_mean": m,
                "pose_contribution": float(np.sqrt(10.0 * m)),
                "delta_S_vs_shipped": float(np.sqrt(10.0 * m)
                                            - np.sqrt(10.0 * base_mean)),
                "delta_S_vs_ctrl_floor_free": float(np.sqrt(10.0 * mc)
                                                    - np.sqrt(10.0 * bc)),
                "n_pairs_improved": int(sum(
                    1 for r in rows
                    if r[key] < shipped[r["pair"]]["d_final"] - 1e-15)),
                "n_pairs_improved_above_floor": int(sum(
                    1 for r in rows
                    if r[key] < shipped[r["pair"]]["d_final"] - floor))}

    # ATTRIBUTION (arm A): was the winning dim0 REACHABLE by the shipped
    # search?  The shipped reach is 0.048 (coarse) + 0.006 (fine) from the
    # v4c start.  A win beyond that is unreachable at ANY budget, so it is
    # attributable to the removed bound and not to "more search".
    reach = 0.048 + 0.006
    # The v4c dim0 each shipped search STARTED from (recorded, confusingly,
    # as "dim0_coarse" by ddm_v4d_resolve.py:311 = float(pose0[0])).
    starts: dict[int, float] = {}
    partial = args.final_jsonl.parent / "refine.partial.jsonl"
    if partial.exists():
        for ln in partial.read_text().splitlines():
            if ln.strip():
                pr = json.loads(ln)
                starts[int(pr["pair"])] = float(pr["dim0_coarse"])
    a_out = a_in = 0.0
    n_out = n_in = 0
    for r in rows:
        if r["arm_a_d"] >= r["d_shipped"] - 1e-15:
            continue
        gain = r["d_shipped"] - r["arm_a_d"]
        start = starts.get(r["pair"], r["dim0_shipped"])
        if abs(r["arm_a_dim0"] - start) > reach + 1e-9:
            a_out += gain
            n_out += 1
        else:
            a_in += gain
            n_in += 1

    # DECOMPOSITION (arm B): the shipped menu is BETA_MAGS=(0,0.5,1.0) used as
    # a MAGNITUDE with the sign forced from yaw, so a win separates into
    # magnitude-extension (g > 1.0), sign-freedom (g < 0), or both.
    gmax = 1.0
    b = {"magnitude_only": [0, 0.0], "sign_only": [0, 0.0], "both": [0, 0.0]}
    for r in rows:
        if r["arm_b_d"] >= r["d_shipped"] - 1e-15:
            continue
        g, gain = r["arm_b_g"], r["d_shipped"] - r["arm_b_d"]
        key = ("magnitude_only" if g > gmax
               else "both" if g < -gmax
               else "sign_only" if g < 0 else "within_menu")
        if key in b:
            b[key][0] += 1
            b[key][1] += gain

    out = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False, "promotion_eligible": False,
        "pointer_moved": False, "research_only": True,
        "base": args.base,
        "n_pairs_measured": len(rows),
        "canary_max_abs_err": floor,
        "noise_floor_note": ("CTRL-vs-shipped disagreement from the g=0 "
                             "beta-blend fp path; arm deltas vs CTRL are "
                             "floor-free"),
        "arm_a_attribution": {
            "shipped_search_reach": reach,
            "wins_outside_reach": n_out, "d_gain_outside_reach": a_out,
            "wins_inside_reach": n_in, "d_gain_inside_reach": a_in,
            "frac_gain_bound_attributable": (a_out / (a_out + a_in)
                                             if (a_out + a_in) > 0 else None),
        },
        "arm_b_decomposition": {k: {"wins": v[0], "d_gain": v[1]}
                                for k, v in b.items()},
        "shipped": {"d_pose_mean": base_mean,
                    "pose_contribution": float(np.sqrt(10.0 * base_mean))},
        "arm_a_dim0_bracket": composed("arm_a_d"),
        "arm_b_beta_menu": composed("arm_b_d"),
        "arm_ab_both": composed("arm_ab_d"),
        "total_scorer_evals": int(sum(r["n_eval"] for r in rows)),
        "generated_by": "tools/pw1_pose_menu_saturation_ab.py",
        "note": ("arms are monotone-safe CONTINUATIONS of the shipped solve; "
                 "d_seg untouched (frame_1 never modified). Pairs not "
                 "measured keep their shipped d_final."),
    }
    # Emit a final JSONL carrying the arm-AB solution so the win can be
    # BYTE-CLOSED through the real builder + receiver instead of being
    # reported as a d_pose delta with an assumed rate.  Monotone-safe: a pair
    # keeps its shipped row unless AB strictly improved it.
    if args.emit_final_jsonl:
        kept = 0
        with open(args.emit_final_jsonl, "w") as fh:
            for p in range(N_PAIRS):
                row = dict(shipped[p])
                # The shipped row's magnitude: prefer an explicit beta_mag
                # (a JSONL from the extended solve) over the seed-table index,
                # which is only meaningful for pre-extension rows.  Reading the
                # index unconditionally would CORRUPT an extended input.
                if "beta_mag" in row:
                    mag = float(row["beta_mag"])
                else:
                    idx = int(row["beta_idx"])
                    if not 0 <= idx < len(BETA_MAGS):
                        raise SystemExit(
                            f"pair {p}: beta_idx={idx} with no beta_mag")
                    mag = float(BETA_MAGS[idx])
                r = cache.get(p)
                if r is not None and r["arm_ab_d"] < row["d_final"] - floor:
                    pose = list(row["p"])
                    pose[0] = r["arm_ab_dim0"]
                    row["p"] = pose
                    mag = float(r["arm_ab_g"])
                    row["d_final"] = float(r["arm_ab_d"])
                    row["source"] = "pw1_arm_ab"
                    kept += 1
                row["beta_mag"] = mag
                # keep beta_idx honest: -1 means "not in the seed table"
                row["beta_idx"] = (BETA_MAGS.index(mag) if mag in BETA_MAGS
                                   else -1)
                fh.write(json.dumps(row) + "\n")
        out["emitted_final_jsonl"] = str(args.emit_final_jsonl)
        out["emitted_rows_replaced"] = kept

    path = args.out_dir / "pw1_receipt.json"
    _atomic_write(path, json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("n_pairs_measured", "canary_max_abs_err", "shipped",
                       "arm_a_dim0_bracket", "arm_b_beta_menu",
                       "arm_ab_both", "total_scorer_evals")}, indent=1))
    print(f"[pw1] receipt {path}", flush=True)


if __name__ == "__main__":
    main()
