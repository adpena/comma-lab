#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4d — the composed rung stack on TOP of the MEASURED v4c gate.

v4c gate (n600 evaluate.py) = d_pose 0.01038450 + d_seg 0.00431179 + 359,750 B
=> S 0.992972.  v4d composes three rungs, each REALIZED-accepted through the
real receiver + frozen PoseNet, each priced through the real byte-closed member:

  MODE qa66  (DATA-READY, no scorers)  QA66 per-pair rolling-shutter shear.
      Read the v4c photo JSONL.  For each pair pick the beta magnitude
      argmin_g d(rungA_by_g) over g in {0.0, 0.5, 1.0} (SIGN free from yaw).
      The per-pair-best d is ALREADY realized (v4c measured rungA_by_g through
      the receiver) => the composed d_pose is EXACT with zero new evals.  Emit
      the final JSONL (pose/a/b/selector/beta_idx).  Measured effect (v4c
      receipt): mean d_pose 0.010384 -> 0.009533, contribution -0.013485 S.

  MODE refine  (REALIZED, PoseNet)  QA65 dim0 offset-coded re-solve + (a,b)
      re-fit + beta select, integrated per pair (greedy joint compose):
        1. dim0 refine on the OFFSET lattice (m + f16 residual, ~0.001 quantum
           vs the 0.02 f16-at-magnitude quantum => 19.3x finer; pi2 law).  A
           bounded 1-D grid+polish line search, realized ACCEPT delta-S<=0.
        2. (a,b) 2-param GN re-fit at the new dim0 (cheap; reuses the warp).
        3. beta select {0.0,0.5,1.0} at the new (dim0,a,b).
      Emits the final JSONL + the per-pair dim0 REFINEMENT CURVE (d at coarse
      vs fine dim0) = the QA69/MacKay pose-floor reading (precision-starved vs
      content-limited).  Resumable JSONL; detached ppid-1.

  MODE qa70  (REALIZED, bounded tail)  min-entropy member selection probe
      (ph3 Kolmogorov): re-solve dim0 of the hardest K pairs with a tiny
      tie-break penalty toward the temporal-neighbor dim0 prediction, INSIDE
      realized acceptance (accept only d-through-PoseNet <= current).  Measure
      the dim0 FIELD ENTROPY (kl1 byte-plane member bytes) with vs without the
      tie-break at EQUAL realized d_pose.  One number: gauge-artifact (codable
      after gauge-fix) vs intrinsic whiteness.

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  ONE n600 scorer job at a time.
tac HIJACK guarded via launch_v4d_detached.sh (assert main src on the path).
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

np.seterr(all="ignore")

sys.path.insert(0, "experiments")
import ddm_v4c_resolve as v4c  # build_oracle, StaticComposer, q16, contribution, _d2_row
from tac.canonical_equations.ddm_fs1_coordinate_fit_staleness_20260802 import (
    FIT_CONTEXT_KEY, stamp_fit_context,
)

V4C = Path("/Volumes/VertigoDataTier/pact/ddm_v4c_20260730")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
PHOTO_JL = V4C / "photo_celldrop50_resolve.partial.jsonl"
BETA_MAGS = (0.0, 0.5, 1.0)  # the SEED menu; ddm_pw1 extends it per pair
GAIN_FD, BIAS_FD = 0.02, 2.0
GN_RELINS_PHOTO = 4
# ddm_pw1 outward-bracket steps: the first outward step of each search is the
# search's OWN native spacing (the dim0 coarse grid step; the beta menu
# spacing), so the continuation resumes the shipped search on its own lattice
# instead of introducing a new scale.
DIM0_STEP0 = 0.012
BETA_STEP0 = 0.5
# ceil(log2(float16_max / DIM0_STEP0)) = ceil(log2(65504/0.012)) = 23: past
# this many doublings the candidate leaves the float16 residual range and can
# no longer move on the lattice, so the bracket cannot run longer.
DIM0_MAX_DOUBLINGS = 23
BETA_MAX_DOUBLINGS = 17  # ceil(log2(65504/0.5)) = 17, same argument


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def q16(p):
    return v4c.q16(p)


def contribution(dm: float) -> float:
    return float(np.sqrt(10.0 * float(dm)))


def load_photo(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


# --------------------------------------------------------------------------- #
# MODE qa66 — DATA-READY per-pair beta selection (no scorers)
# --------------------------------------------------------------------------- #
def run_qa66(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    photo = load_photo(PHOTO_JL)
    n = len(photo)
    gset = [str(g) for g in BETA_MAGS]
    out = OUT / "final_qa66.jsonl"
    d_final = np.zeros(n)
    d_v4c = np.zeros(n)
    beta_counts = [0, 0, 0]
    with open(out, "w") as fj:
        for i in range(n):
            r = photo[i]
            g = r["rungA_by_g"]
            vals = {k: float(g.get(k, r["d_rungB"])) for k in gset}
            k_best = min(vals, key=vals.get)
            bidx = gset.index(k_best)
            beta_counts[bidx] += 1
            d_final[i] = vals[k_best]
            # v4c SHIPPED global g*=0.0 => its realized d_pose IS rung-B
            # (gate-measured 0.01038450 == mean d_rungB); that is the reference.
            d_v4c[i] = float(r["d_rungB"])
            row = {"pair": i, "p": [float(v) for v in r["p"]],
                   "a": float(r["a"]), "b": float(r["b"]),
                   "selector": int(r["selector"]), "beta_idx": int(bidx),
                   "d_final": vals[k_best], "d_ref_v4c": float(r["d_rungB"]),
                   "source": "v4c_photo_qa66"}
            fj.write(json.dumps(row) + "\n")
    receipt = {
        "schema": "ddm_v4d_qa66.v1", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": n, "beta_mags": list(BETA_MAGS),
        "beta_idx_counts": beta_counts,
        "mean_d_final": float(d_final.mean()),
        "contribution_final": contribution(float(d_final.mean())),
        "mean_d_ref_v4c": float(d_v4c.mean()),
        "contribution_ref_v4c": contribution(float(d_v4c.mean())),
        "delta_S_vs_v4c": (contribution(float(d_final.mean()))
                           - contribution(float(d_v4c.mean()))),
        "note": "per-pair beta = argmin over g in {0,0.5,1.0} of the v4c "
                "realized rungA_by_g; sign from yaw (free). d_final REALIZED "
                "in the v4c photo pass (zero new evals). Build with "
                "--dim0-offset none for a clean QA66-only candidate.",
    }
    (OUT / "final_qa66_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


# --------------------------------------------------------------------------- #
# MODE refine — QA65 dim0 offset re-solve + (a,b) re-fit + beta select
# --------------------------------------------------------------------------- #
def _dim0_offset(photo: dict[int, dict]) -> float:
    d0 = np.array([photo[i]["p"][0] for i in range(len(photo))], np.float64)
    return float(np.float16(d0.mean()))


def _score_at(comp, pose, s_t, sel, f1_u8, f1_f, tp, a=1.0, b=0.0):
    """One realized compose+score at a given pose/exposure (re-warps)."""
    if sel:
        wg, wf = comp.warps(f1_f, pose, s_t)
        f0 = comp.compose(wg, wf, 1, gain=a, bias=b)
    else:
        wg = comp.warp_ground(f1_f, pose, s_t)
        f0 = comp.compose(wg, wg, 0, gain=a, bias=b)
    return comp.score(f0, f1_u8, tp)


def _refine_dim0(comp, pose0, s_t, sel, f1_u8, f1_f, tp, offset, a, b):
    """Bounded 1-D grid+polish of dim0 on the OFFSET lattice (m + f16 residual).
    Realized accept d<=cur.  Returns (best_dim0, d_best, d_coarse, n_fwd)."""
    d0 = float(pose0[0])
    pose = pose0.copy()

    def eval_dim0(x):
        r = np.float16(x - offset)          # store as f16 residual...
        xq = offset + float(r)              # ...receiver reconstructs this exactly
        pose[0] = xq
        return _score_at(comp, pose, s_t, sel, f1_u8, f1_f, tp, a, b), xq

    d_coarse, x_coarse = eval_dim0(d0)      # current dim0 (offset-quantized)
    best_d, best_x = d_coarse, x_coarse
    n_fwd = 1
    # coarse grid over +-0.048 (2.4x the 0.02 f16-at-magnitude quantum), then
    # a fine polish over +-0.006 at the ~0.0015 offset quantum.
    for x in (d0 + np.array([-0.048, -0.036, -0.024, -0.012, 0.012, 0.024,
                             0.036, 0.048])):
        d, xq = eval_dim0(x)
        n_fwd += 1
        if d < best_d:
            best_d, best_x = d, xq
    for x in (best_x + np.array([-0.006, -0.004, -0.002, -0.0015, 0.0015,
                                 0.002, 0.004, 0.006])):
        d, xq = eval_dim0(x)
        n_fwd += 1
        if d < best_d:
            best_d, best_x = d, xq
    # ddm_pw1: the +-0.048/+-0.006 window above is a fixed REACH, and the
    # shipped n600 solve SATURATED it -- the |move| histogram decays
    # monotonically (103,93,67,39,51,34,46,28,15) and then jumps to 124 at
    # the bound, and those 124 pairs carry 37.4% of the population d_pose
    # mass.  Continue outward with a self-terminating Swann bracket rather
    # than a wider constant: probe both directions at the coarse step, commit
    # to the improving one, and double until a probe fails to improve.
    # TERMINATION IS A PROOF, not a budget: an accepted step strictly
    # decreases a quantity bounded below by 0, and the doubling step leaves
    # the float16 residual range after at most ceil(log2(65504/0.012)) = 23
    # doublings, at which point the candidate stops moving on the lattice.
    # Costs ZERO extra evaluations on pairs where the window did not bind.
    step, direction = DIM0_STEP0, 0.0
    for sign in (1.0, -1.0):
        d, xq = eval_dim0(best_x + sign * step)
        n_fwd += 1
        if d < best_d:
            best_d, best_x, direction = d, xq, sign
            break
    for _ in range(DIM0_MAX_DOUBLINGS):
        if direction == 0.0:
            break
        step *= 2.0
        d, xq = eval_dim0(best_x + direction * step)
        n_fwd += 1
        if d >= best_d or xq == best_x:
            break
        best_d, best_x = d, xq
    return best_x, best_d, d_coarse, n_fwd


def _refit_ab(comp, pose, s_t, sel, f1_u8, f1_f, tp, a0, b0):
    """2-param (a,b) GN re-fit reusing the warp at the given pose (cheap).

    Returns ``(a_q, b_q, d, trace)``.  The solve itself is
    ``v4c.ab_damped_gn`` -- this site and the v4c rung-B site carried
    byte-identical copies of the same loop, and neither recorded WHY it
    stopped; see that function for the measured censoring.
    """
    if sel:
        wg, wf = comp.warps(f1_f, pose, s_t)
    else:
        wg = comp.warp_ground(f1_f, pose, s_t)
        wf = wg

    def pose6(a, b):
        return comp.o.p3v2.pose6_u8(comp.o.posenet,
                                    comp.compose(wg, wf, sel, gain=a, bias=b), f1_u8)

    def mse(p6):
        return float(np.mean((p6 - tp) ** 2))

    cur6 = pose6(a0, b0)
    a_p, b_p, cur6, curB, trace = v4c.ab_damped_gn(
        pose6, mse, a0, b0, cur6, mse(cur6), tp, relins=GN_RELINS_PHOTO)
    a_q, b_q = float(np.float16(a_p)), float(np.float16(b_p))
    return a_q, b_q, mse(pose6(a_q, b_q)), trace


def _beta_select(comp, pose, s_t, sel, f1_u8, f1_f, tp, a, b, d_ab):
    """Select the rolling-shutter beta at the current (pose,a,b). d0 = d_ab.

    Returns the signed MAGNITUDE g (not an index): the receiver applies
    ``beta = rs_beta_mags[idx] * yaw_sign`` (inflate_runner_v4d.py:177-180)
    and reads its table from the manifest, so a NEGATIVE or >1.0 entry needs
    no receiver change and no extra per-pair bytes (beta_idx is uint8).

    ddm_pw1 MEASURED that the seed menu (0.0, 0.5, 1.0) bound both ways: 76
    of 600 shipped pairs selected the TOP entry (26.4% of the pose mass), and
    decomposing the recoverable gain showed it needs BOTH a magnitude past
    1.0 AND freedom to oppose the yaw sign.  The seed menu is still swept
    first (so the shipped choice is always reachable), then a self-terminating
    Swann bracket continues along the signed beta line.
    """
    yaw_sign = 1.0 if pose[5] >= 0.0 else -1.0

    def at(g: float) -> float:
        if g == 0.0:
            return float(d_ab)
        beta = g * yaw_sign
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

    by_g: dict[str, float] = {}

    def probe(g: float) -> float:
        key = str(g)
        if key not in by_g:
            by_g[key] = at(g)
        return by_g[key]

    best_d, best_g = float(d_ab), 0.0
    by_g["0.0"] = float(d_ab)
    for g in BETA_MAGS:                       # the shipped sweep, unchanged
        if g == 0.0:
            continue
        d = probe(g)
        if d < best_d:
            best_d, best_g = d, g
    # Outward continuation along the SIGNED beta line.  Same termination proof
    # as the dim0 bracket: accepted steps strictly decrease a quantity bounded
    # below by 0, and the doubling step exhausts the representable range in at
    # most BETA_MAX_DOUBLINGS steps.
    step, direction = BETA_STEP0, 0.0
    for sign in (1.0, -1.0):
        d = probe(best_g + sign * step)
        if d < best_d:
            best_d, best_g, direction = d, best_g + sign * step, sign
            break
    for _ in range(BETA_MAX_DOUBLINGS):
        if direction == 0.0:
            break
        step *= 2.0
        cand = best_g + direction * step
        d = probe(cand)
        if d >= best_d:
            break
        best_d, best_g = d, cand
    return best_g, best_d, by_g


def run_refine(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = v4c.build_oracle("celldrop50", s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    photo = load_photo(PHOTO_JL)
    n = len(photo)
    offset = _dim0_offset(photo)
    # order hardest-first by v4c d_rungB so decisive pairs land early
    order = sorted(range(n), key=lambda p: -float(photo[p]["d_rungB"]))
    seq = order[: args.k] if args.k else order
    jl = OUT / "refine.partial.jsonl"
    cache = {int(r["pair"]): r for ln in (jl.read_text().splitlines()
             if jl.exists() else []) if ln.strip() for r in [json.loads(ln)]}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[v4d-refine] offset(dim0 mean f16)={offset:.5f} n={len(seq)} "
          f"cached={len(cache)}", flush=True)
    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[v4d-refine] --max-seconds; {len(cache)} done; --resume",
                  flush=True)
            break
        r = photo[pidx]
        pose0 = v4c.q16(np.asarray(r["p"], np.float64))
        sel = int(r["selector"])
        s_t = float(v4c._d2_row(pidx)["s_t"])
        a0, b0 = float(np.float16(r["a"])), float(np.float16(r["b"]))
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        # 1. dim0 refine on the offset lattice
        best_x, d_dim0, d_coarse, nfwd = _refine_dim0(
            comp, pose0, s_t, sel, f1_u8, f1_f, tp, offset, a0, b0)
        pose = pose0.copy()
        pose[0] = best_x
        # 2. (a,b) re-fit at the new dim0
        a_q, b_q, d_ab, ab_trace = _refit_ab(
            comp, pose, s_t, sel, f1_u8, f1_f, tp, a0, b0)
        # 3. beta select
        beta_mag, d_final, by_g = _beta_select(
            comp, pose, s_t, sel, f1_u8, f1_f, tp, a_q, b_q, d_ab)
        # beta_idx stays for readers that predate the extended menu; it is
        # only meaningful when the chosen magnitude is in the seed table.
        beta_idx = (BETA_MAGS.index(beta_mag) if beta_mag in BETA_MAGS
                    else -1)
        d_ref = float(r["d_rungAB"])
        rec = {"pair": int(pidx), "rank": rank,
               "p": [float(v) for v in pose], "a": a_q, "b": b_q,
               "selector": sel, "beta_idx": int(beta_idx),
               "beta_mag": float(beta_mag),
               "dim0_offset": offset, "dim0_coarse": float(pose0[0]),
               "dim0_fine": float(best_x),
               "d_coarse_dim0": float(d_coarse), "d_dim0": float(d_dim0),
               "d_ab": float(d_ab), "d_final": float(d_final),
               "d_ref_v4c": d_ref, "rungA_by_g": by_g, "n_fwd": int(nfwd),
               # ddm_sf1: step 2 re-fits (a,b) at the REFINED dim0 with beta=0;
               # step 3 then selects beta with (a,b) FROZEN, and pw1/mq1 later
               # move beta (out to |7.5|) and p1/p2 without ever re-solving
               # (a,b) -- MEASURED 0 rows re-solved, 244 shipped stale, 100
               # outside the fitted set.  This stamp is what makes that
               # gradeable from the artifact.
               FIT_CONTEXT_KEY: stamp_fit_context(
                   coefficient="ab_gain_bias",
                   partners={"beta": 0.0, "p0": float(pose[0]),
                             "p1": float(pose[1]), "p2": float(pose[2])},
                   base="celldrop50",
                   fit_menu=BETA_MAGS,
                   # _beta_select pins beta = g*yaw_sign, g >= 0 (see its
                   # yaw_sign line): the menu was sampled at ONE sign, so an
                   # opposing-sign shipment escapes the fitted set even at an
                   # in-range magnitude (63 pairs, MEASURED by ft1).
                   fit_sign=(1.0 if pose[5] >= 0.0 else -1.0))}
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        done = len(cache)
        if done % 5 == 0 or rank == len(seq) - 1:
            df = np.asarray([cache[p]["d_final"] for p in cache])
            dc = np.asarray([cache[p]["d_coarse_dim0"] for p in cache])
            dv = np.asarray([cache[p]["d_ref_v4c"] for p in cache])
            print(f"[v4d-refine {done:3d}/{len(seq)}] pair {pidx} coarse "
                  f"{d_coarse:.5f} dim0 {d_dim0:.5f} ab {d_ab:.5f} final "
                  f"{d_final:.5f} (g{beta_mag:+.2f}) | means v4c {dv.mean():.5f} "
                  f"coarse {dc.mean():.5f} final {df.mean():.5f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    fj.close()
    _summarize_refine(photo, cache, offset)


def _row_beta_mag(row: dict) -> float:
    """The signed beta magnitude a refine-cache row selected.

    Rows written before the extended menu carry only ``beta_idx`` into the
    seed table.  A row that chose an EXTENDED magnitude records beta_idx=-1,
    and Python would silently wrap that to BETA_MAGS[-1]=1.0, so the index
    path fails closed instead.
    """
    if "beta_mag" in row:
        return float(row["beta_mag"])
    idx = int(row["beta_idx"])
    if not 0 <= idx < len(BETA_MAGS):
        raise SystemExit(f"row has beta_idx={idx} and no beta_mag; the "
                         "magnitude it chose is not recoverable")
    return float(BETA_MAGS[idx])


def _qa66_dfinal(pr: dict) -> float:
    """The QA66 per-pair-best d (argmin over g of the realized rungA_by_g)."""
    g = pr["rungA_by_g"]
    return min(float(g.get(str(x), pr["d_rungB"])) for x in BETA_MAGS)


def _summarize_refine(photo, cache, offset) -> None:
    n = len(photo)
    # composed field over 600: refined where solved, else QA66 per-pair best
    d_final = np.array([cache[i]["d_final"] if i in cache
                        else _qa66_dfinal(photo[i]) for i in range(n)])
    # references: v4c SHIPPED (rung-B, global g*=0) and QA66 (per-pair beta)
    d_v4c = np.array([float(photo[i]["d_rungB"]) for i in range(n)])
    d_qa66 = np.array([_qa66_dfinal(photo[i]) for i in range(n)])
    solved = [i for i in cache]
    curve = None
    if solved:
        dc = np.array([cache[i]["d_coarse_dim0"] for i in solved])
        dd = np.array([cache[i]["d_dim0"] for i in solved])
        curve = {"n_solved": len(solved),
                 "mean_d_coarse_dim0": float(dc.mean()),
                 "mean_d_fine_dim0": float(dd.mean()),
                 "dim0_precision_gain_d": float(dc.mean() - dd.mean()),
                 "dim0_precision_gain_S": float(contribution(float(dc.mean()))
                                                - contribution(float(dd.mean()))),
                 "frac_pairs_dim0_improved": float(np.mean(dd < dc - 1e-9))}
    receipt = {
        "schema": "ddm_v4d_refine.v1", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": n, "n_refined": len(solved), "dim0_offset": offset,
        "mean_d_v4c_shipped": float(d_v4c.mean()),
        "contribution_v4c_shipped": contribution(float(d_v4c.mean())),
        "mean_d_qa66": float(d_qa66.mean()),
        "contribution_qa66": contribution(float(d_qa66.mean())),
        "mean_d_final": float(d_final.mean()),
        "contribution_final": contribution(float(d_final.mean())),
        "delta_S_vs_v4c_shipped": (contribution(float(d_final.mean()))
                                   - contribution(float(d_v4c.mean()))),
        "delta_S_vs_qa66": (contribution(float(d_final.mean()))
                            - contribution(float(d_qa66.mean()))),
        "dim0_refinement_curve": curve,
        "note": "integrated per-pair dim0-offset re-solve + (a,b) re-fit + "
                "beta select, realized-accepted. dim0_precision_gain_S = the "
                "QA69/MacKay pose-floor reading (precision-starved fraction). "
                "Unrefined pairs carry the v4c rungAB proxy.",
    }
    # MONOTONE-SAFE composition: per refined pair ship min(refine, QA66); every
    # pair is <= its QA66 floor by construction (best-of, like the v4c selector).
    gset = [str(x) for x in BETA_MAGS]

    def _qa66_row(i):
        pr = photo[i]
        g = pr["rungA_by_g"]
        vals = {k: float(g.get(k, pr["d_rungB"])) for k in gset}
        k_best = min(vals, key=vals.get)
        return {"pair": i, "p": [float(v) for v in pr["p"]],
                "a": float(pr["a"]), "b": float(pr["b"]),
                "selector": int(pr["selector"]), "beta_idx": gset.index(k_best),
                "beta_mag": float(k_best),
                "d_final": vals[k_best], "source": "v4c_photo_qa66"}, vals[k_best]

    n_refine_win = 0
    final = OUT / "final_refine.jsonl"
    with open(final, "w") as ff:
        for i in range(n):
            qa66_row, d_qa66_i = _qa66_row(i)
            if i in cache and float(cache[i]["d_final"]) <= d_qa66_i:
                c = cache[i]
                row = {"pair": i, "p": c["p"], "a": c["a"], "b": c["b"],
                       "selector": c["selector"], "beta_idx": c["beta_idx"],
                       "beta_mag": _row_beta_mag(c),
                       "d_final": c["d_final"], "source": "v4d_refine"}
                n_refine_win += 1
            else:
                row = qa66_row
            ff.write(json.dumps(row) + "\n")
    receipt["monotone_safe_n_refine_win"] = n_refine_win
    receipt["monotone_safe_n_qa66_kept"] = n - n_refine_win
    # recompute the monotone-safe composed field (what the build ships)
    d_ship = np.array([min(float(cache[i]["d_final"]),
                           _qa66_row(i)[1]) if i in cache else _qa66_row(i)[1]
                       for i in range(n)])
    receipt["mean_d_ship_monotone"] = float(d_ship.mean())
    receipt["contribution_ship_monotone"] = contribution(float(d_ship.mean()))
    receipt["delta_S_ship_vs_v4c_shipped"] = (
        contribution(float(d_ship.mean())) - contribution(float(d_v4c.mean())))
    receipt["delta_S_ship_vs_qa66"] = (
        contribution(float(d_ship.mean())) - contribution(float(d_qa66.mean())))
    (OUT / "refine_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)
    print(f"[v4d-refine] wrote {final} (refine-win {n_refine_win}/{n})", flush=True)


# --------------------------------------------------------------------------- #
# MODE qa70 — min-entropy member selection probe (bounded tail)
# --------------------------------------------------------------------------- #
def _kl1_member_bytes(field_f16: np.ndarray) -> int:
    """Byte-plane brotli member size (kl1 KL1PWF01), for entropy measurement."""
    import struct as _st

    import brotli
    n, d = field_f16.shape
    cm = np.ascontiguousarray(field_f16.T).view(np.uint16).astype(np.uint16)
    hi = (cm >> 8).astype(np.uint8).reshape(-1)
    lo = (cm & 0xFF).astype(np.uint8).reshape(-1)
    payload = brotli.compress(np.concatenate([hi, lo]).tobytes(), quality=11)
    return 8 + len(_st.pack("<HHI", n, d, len(payload))) + len(payload)  # magic(8)+hdr+payload


def run_qa70(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = v4c.build_oracle("celldrop50", s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    photo = load_photo(PHOTO_JL)
    n = len(photo)
    offset = _dim0_offset(photo)
    order = sorted(range(n), key=lambda p: -float(photo[p]["d_rungB"]))
    seq = order[: args.k] if args.k else order
    # temporal-neighbor dim0 prediction = previous pair's dim0 (index-1)
    dim0_all = np.array([photo[i]["p"][0] for i in range(n)], np.float64)
    jl = OUT / "qa70.partial.jsonl"
    cache = {int(r["pair"]): r for ln in (jl.read_text().splitlines()
             if jl.exists() else []) if ln.strip() for r in [json.loads(ln)]}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    lam = float(args.tiebreak_lambda)
    print(f"[v4d-qa70] tie-break lambda={lam:.2e} n={len(seq)} cached={len(cache)}",
          flush=True)
    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            break
        r = photo[pidx]
        pose0 = v4c.q16(np.asarray(r["p"], np.float64))
        sel = int(r["selector"])
        s_t = float(v4c._d2_row(pidx)["s_t"])
        a0, b0 = float(np.float16(r["a"])), float(np.float16(r["b"]))
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        pred = dim0_all[pidx - 1] if pidx > 0 else dim0_all[pidx]
        # plain refine (no tie-break)
        bx_plain, d_plain, d_coarse, _ = _refine_dim0(
            comp, pose0, s_t, sel, f1_u8, f1_f, tp, offset, a0, b0)
        # tie-break refine: among candidates with realized d<=d_plain+eps, pick
        # the one closest to `pred` (min |dim0-pred|).  Realized acceptance.
        pose = pose0.copy()
        best_x, best_d, best_pen = bx_plain, d_plain, abs(bx_plain - pred)
        eps = args.accept_eps

        def sc(x):
            rq = np.float16(x - offset)
            xq = offset + float(rq)
            pose[0] = xq
            return _score_at(comp, pose, s_t, sel, f1_u8, f1_f, tp, a0, b0), xq
        for x in (bx_plain + np.linspace(-0.05, 0.05, 21)):
            d, xq = sc(x)
            if d <= d_plain + eps:
                pen = abs(xq - pred)
                # prefer smaller penalty at (near-)equal d; strict realized guard
                if pen < best_pen and d <= best_d + eps:
                    best_x, best_d, best_pen = xq, d, pen
        rec = {"pair": int(pidx), "rank": rank, "pred_dim0": float(pred),
               "dim0_plain": float(bx_plain), "dim0_tiebreak": float(best_x),
               "d_plain": float(d_plain), "d_tiebreak": float(best_d),
               "d_coarse": float(d_coarse)}
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if len(cache) % 5 == 0 or rank == len(seq) - 1:
            print(f"[v4d-qa70 {len(cache):3d}/{len(seq)}] pair {pidx} plain "
                  f"{bx_plain:.4f}(d{d_plain:.5f}) tie {best_x:.4f}"
                  f"(d{best_d:.5f}) pred {pred:.4f} {time.time()-t0:.0f}s",
                  flush=True)
    fj.close()
    _summarize_qa70(photo, cache, offset)


def _summarize_qa70(photo, cache, offset) -> None:
    if not cache:
        print("[qa70] no rows", flush=True)
        return
    idx = sorted(cache)
    d_plain = np.array([cache[i]["d_plain"] for i in idx])
    d_tie = np.array([cache[i]["d_tiebreak"] for i in idx])
    x_plain = np.array([cache[i]["dim0_plain"] for i in idx], np.float64)
    x_tie = np.array([cache[i]["dim0_tiebreak"] for i in idx], np.float64)
    pred = np.array([cache[i]["pred_dim0"] for i in idx], np.float64)
    # The gauge question: does re-picking dim0 among realized-EQUAL optima toward
    # the temporal-neighbor prediction make the field PREDICTABLE from neighbors
    # (=> codable as a small prediction-residual) at EQUAL realized d?
    # Metric 1: kl1 member bytes of the PREDICTION-RESIDUAL field (dim0 - pred).
    pr_plain = (x_plain - pred).astype(np.float16)
    pr_tie = (x_tie - pred).astype(np.float16)
    bytes_plain = _kl1_member_bytes(pr_plain.reshape(-1, 1))
    bytes_tie = _kl1_member_bytes(pr_tie.reshape(-1, 1))
    # Metric 2 (robust on small subsets): mean |dim0 - pred| (prediction-residual
    # L1 energy) — scale-free, not brotli-header-dominated.
    mad_plain = float(np.mean(np.abs(x_plain - pred)))
    mad_tie = float(np.mean(np.abs(x_tie - pred)))
    d_equal = float(np.max(np.abs(d_tie - d_plain)))
    gauge = (mad_tie < 0.5 * mad_plain) and (bytes_tie < bytes_plain - 2)
    receipt = {
        "schema": "ddm_v4d_qa70.v1", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n": len(idx), "dim0_offset": offset,
        "mean_d_plain": float(d_plain.mean()),
        "mean_d_tiebreak": float(d_tie.mean()),
        "d_equal_within": d_equal,
        "pred_residual_member_bytes_plain": bytes_plain,
        "pred_residual_member_bytes_tiebreak": bytes_tie,
        "pred_residual_entropy_reduction_bytes": bytes_plain - bytes_tie,
        "mean_abs_dev_from_pred_plain": mad_plain,
        "mean_abs_dev_from_pred_tiebreak": mad_tie,
        "mad_reduction_frac": (mad_plain - mad_tie) / mad_plain if mad_plain else 0.0,
        "verdict": ("gauge-artifact (dim0 codable from temporal neighbors after "
                    "gauge-fix, at equal realized d)" if gauge
                    else "intrinsic whiteness at this chart (tie-break cannot "
                    "pull dim0 to neighbors at equal d; close honestly)"),
        "note": "min-entropy member selection (ph3 Kolmogorov). tie-break re-picks "
                "dim0 among realized-equal optima (accept only d<=plain) toward the "
                "temporal-neighbor prediction. Metric = kl1 bytes + L1 energy of "
                "the PREDICTION-RESIDUAL (dim0-neighbor). Reduction at equal d => "
                "solver-gauge artifact (codable); else intrinsic whiteness.",
    }
    (OUT / "qa70_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def _dim0_golden(comp, pose0, s_t, sel, f1_u8, f1_f, tp, a, b, span=0.12):
    """Continuous (fp32, NO f16 snap) 1-D minimize of d over dim0 by golden
    section around pose0[0]±span.  Returns (dim0_cont, d_cont, n_fwd)."""
    gr = 0.6180339887
    pose = pose0.copy()

    def f(x):
        pose[0] = x
        return _score_at(comp, pose, s_t, sel, f1_u8, f1_f, tp, a, b)
    x0 = float(pose0[0])
    lo, hi = x0 - span, x0 + span
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    nf = 2
    for _ in range(22):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
        nf += 1
    xbest = c if fc < fd else d
    dbest = min(fc, fd)
    # also check the exact interval endpoints/center against the incumbent
    for xx in (x0, 0.5 * (lo + hi)):
        v = f(xx)
        nf += 1
        if v < dbest:
            xbest, dbest = xx, v
    return float(xbest), float(dbest), nf


def run_qa72a(args: argparse.Namespace) -> None:
    """QA72a stage-attribution (ja1 #2, the $0 decider): solve dim0 CONTINUOUS
    (fp32, through the uint8 render), then re-quantize plain-f16 vs offset-f16 and
    attribute the realized pose loss per stage.  Decides QA65 (dim0 precision)
    priority: large f16-storage penalty => storage-binder (QA65 justified);
    small => content/re-solve-limited (QA65 demoted; QA68 tail outranks)."""
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = v4c.build_oracle("celldrop50", s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    photo = load_photo(PHOTO_JL)
    n = len(photo)
    offset = _dim0_offset(photo)
    order = sorted(range(n), key=lambda p: -float(photo[p]["d_rungB"]))
    seq = order[: args.k] if args.k else order
    jl = OUT / "qa72a.partial.jsonl"
    cache = {int(r["pair"]): r for ln in (jl.read_text().splitlines()
             if jl.exists() else []) if ln.strip() for r in [json.loads(ln)]}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[v4d-qa72a] offset={offset:.5f} n={len(seq)} cached={len(cache)}",
          flush=True)
    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            break
        r = photo[pidx]
        pose0 = v4c.q16(np.asarray(r["p"], np.float64))
        sel = int(r["selector"])
        s_t = float(v4c._d2_row(pidx)["s_t"])
        a0, b0 = float(np.float16(r["a"])), float(np.float16(r["b"]))
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        # v4c-coarse incumbent (dim0 at pose0[0], the shipped f16-at-magnitude)
        d_v4c = _score_at(comp, pose0, s_t, sel, f1_u8, f1_f, tp, a0, b0)
        # continuous dim0 solve (fp32, through uint8 render)
        x_cont, d_cont, nf = _dim0_golden(comp, pose0, s_t, sel, f1_u8, f1_f, tp, a0, b0)
        # re-quantize the CONTINUOUS optimum two ways, re-score through-R
        pose = pose0.copy()
        pose[0] = float(np.float16(x_cont))                    # plain f16 (0.02 quantum)
        d_f16 = _score_at(comp, pose, s_t, sel, f1_u8, f1_f, tp, a0, b0)
        pose[0] = offset + float(np.float16(x_cont - offset))  # offset f16 (0.001 quantum)
        d_off = _score_at(comp, pose, s_t, sel, f1_u8, f1_f, tp, a0, b0)
        rec = {"pair": int(pidx), "rank": rank, "d_v4c_coarse": float(d_v4c),
               "x_cont": float(x_cont), "d_cont": float(d_cont),
               "d_plain_f16": float(d_f16), "d_offset_f16": float(d_off),
               "n_fwd": int(nf)}
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if len(cache) % 5 == 0 or rank == len(seq) - 1:
            print(f"[v4d-qa72a {len(cache):3d}/{len(seq)}] pair {pidx} v4c "
                  f"{d_v4c:.5f} cont {d_cont:.5f} f16 {d_f16:.5f} off {d_off:.5f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    fj.close()
    _summarize_qa72a(cache)


def _summarize_qa72a(cache) -> None:
    if not cache:
        print("[qa72a] no rows", flush=True)
        return
    idx = sorted(cache)
    d_v4c = np.array([cache[i]["d_v4c_coarse"] for i in idx])
    d_cont = np.array([cache[i]["d_cont"] for i in idx])
    d_f16 = np.array([cache[i]["d_plain_f16"] for i in idx])
    d_off = np.array([cache[i]["d_offset_f16"] for i in idx])
    # per-pair non-negative penalties (clip tiny numerical negatives)
    f16_pen = float(np.mean(np.maximum(d_f16 - d_cont, 0.0)))
    off_pen = float(np.mean(np.maximum(d_off - d_cont, 0.0)))
    resolve_gain = float(np.mean(np.maximum(d_v4c - d_cont, 0.0)))
    # attribution: what fraction of the recoverable pose loss (v4c->cont) is
    # ATTRIBUTABLE to f16 STORAGE (survives only with finer precision)?
    total_recoverable = resolve_gain  # d_v4c - d_cont
    storage_frac = f16_pen / total_recoverable if total_recoverable > 1e-12 else 0.0
    dS = lambda a, b: contribution(float(a)) - contribution(float(b))
    receipt = {
        "schema": "ddm_v4d_qa72a.v1", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n": len(idx),
        "mean_d_v4c_coarse": float(d_v4c.mean()),
        "mean_d_continuous": float(d_cont.mean()),
        "mean_d_plain_f16": float(d_f16.mean()),
        "mean_d_offset_f16": float(d_off.mean()),
        "resolve_gain_d (v4c->continuous)": resolve_gain,
        "f16_storage_penalty_d (plain_f16 - continuous)": f16_pen,
        "offset_storage_penalty_d (offset_f16 - continuous)": off_pen,
        "storage_fraction_of_recoverable": storage_frac,
        "offset_recovers_over_plain_d": float(np.mean(np.maximum(d_f16 - d_off, 0.0))),
        "S_v4c_to_continuous": dS(d_v4c.mean(), d_cont.mean()),
        "S_plain_f16_to_offset": dS(d_f16.mean(), d_off.mean()),
        "verdict": ("STORAGE-BINDER: f16 storage penalty is a large fraction of "
                    "recoverable loss => QA65 offset-coding JUSTIFIED"
                    if storage_frac > 0.33 and off_pen < 0.5 * f16_pen
                    else "CONTENT/RE-SOLVE-LIMITED: f16 storage penalty small vs "
                    "recoverable loss => QA65 DEMOTED (ja1 #4 confirmed); the "
                    "re-solve gain is a POSE-CONTENT lever (QA68 tail direction), "
                    "storable at plain f16"),
        "note": "ja1 #2 stage-attribution. Bounded hardest-K. Continuous fp32 "
                "dim0 solve through the uint8 render; plain-f16 (0.02 quantum) vs "
                "offset-f16 (0.001 quantum) re-quantization. storage_fraction = "
                "f16_penalty/(v4c->continuous). Decides QA65 (precision) vs QA68 "
                "(content) priority.",
    }
    (OUT / "qa72a_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def run_resummarize(args: argparse.Namespace) -> None:
    """Re-emit the monotone-safe final JSONL + receipt from a completed
    refine.partial.jsonl WITHOUT building the oracle/scorer (no torch)."""
    photo = load_photo(PHOTO_JL)
    jl = OUT / "refine.partial.jsonl"
    cache = {int(r["pair"]): r for ln in jl.read_text().splitlines()
             if ln.strip() for r in [json.loads(ln)]}
    offset = float(next(iter(cache.values()))["dim0_offset"]) if cache \
        else _dim0_offset(photo)
    print(f"[v4d-resummarize] cache={len(cache)} offset={offset:.6f}", flush=True)
    _summarize_refine(photo, cache, offset)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",
                    choices=("qa66", "refine", "qa70", "qa72a", "resummarize"),
                    required=True)
    ap.add_argument("--k", type=int, default=0, help="limit to first K (hardest)")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--tiebreak-lambda", type=float, default=1.0,
                    help="qa70: tie-break weight (documentation only; realized"
                         " acceptance is the gate)")
    ap.add_argument("--accept-eps", type=float, default=1e-9,
                    help="qa70: realized d equality tolerance for member swap")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if args.mode == "qa66":
        run_qa66(args)
    elif args.mode == "refine":
        run_refine(args)
    elif args.mode == "resummarize":
        run_resummarize(args)
    elif args.mode == "qa72a":
        run_qa72a(args)
    else:
        run_qa70(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
