#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_mq1 -- is the v4d pose TRANSLATION LATTICE precision-starved, per column?

MOTIVATION (measured, ddm_mq1 2026-08-01).  The v4d pose payload ships the six
pose columns as f16, with ONE column (``p0``, forward speed) carrying the QA65
offset+residual device that was MEASURED worth ``dim0_precision_gain_S =
0.009196`` (``refine_receipt.json``).  Auditing the RESOLUTION each column
actually receives at the effective translation ``t = s_t * (p2, p1, p0)``
(``pfs1_warp_receiver.pose_to_homography:45``) shows the allocation is lopsided:

    column   stored as              ulp(effective t)   RELATIVE ulp
    p0       offset + f16 residual  1.00e-04           3.28e-05
    p1       plain f16              8.63e-06           7.33e-04   (22.4x coarser)
    p2       plain f16              1.84e-05           7.05e-04   (21.5x coarser)

``p1``/``p2`` receive ~22x coarser RELATIVE resolution than ``p0``.  Under the
waterfill principle (bits allocated so marginal distortion is equalised) that
disparity is only correct if the objective is ~22x less sensitive to the lateral
and vertical translation than to the forward one -- which nobody has measured.

The QA65 cure does NOT transfer: ``p1``/``p2`` are already zero-centred
(mean -0.0409 / -0.0240 against std 0.226 / 0.487), so subtracting their mean
shrinks the f16 residual by 1.01x / 1.00x -- MEASURED, and a dead end.  The open
question is therefore not "apply the offset trick" but the prior one: **is the
plain-f16 lattice on p1/p2 binding on d_pose at all?**

WHAT THIS TOOL MEASURES.  Per pair, holding every other shipped variable fixed
(pose, exposure (a,b), selector, beta magnitude, s_t), it refines one
translation column to its CONTINUOUS optimum and decomposes the recoverable
distortion into two disjoint parts:

    gap_lattice = d(nearest shippable f16) - d(continuous optimum)
        -> what a FINER LATTICE could buy.  Cannot be reached by more search.
    gap_search  = d(shipped) - d(nearest shippable f16)
        -> what a BETTER SEARCH could buy at TODAY's lattice.  Free of format.

This is the same decomposition ``qa72a`` ran for ``p0``, whose answer closed the
``p0`` lattice: offset-f16 sat 0.00008063 above the continuous optimum, i.e.
**0.12% of d** on the 80 highest-mass pairs.

MONOTONE-SAFE: every arm starts at the shipped solution and accepts only a
STRICT decrease of the realized scorer, so no arm can report a win it did not
realize.  CTRL re-scores the shipped solution and is the canary.

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
SCHEMA = "ddm_mq1_pose_lattice_resolution.v1"
V4D = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
N_PAIRS = 600
# The v4d seed menu (experiments/ddm_v4d_resolve.py:63) that beta_idx indexes
# when a row predates the pw1 explicit beta_mag column.
BETA_MAGS = (0.0, 0.5, 1.0)
# Golden-section ratio; the polish terminates when the bracket is narrower than
# GOLDEN_TOL_ULP lattice steps, so the returned x is inside the f16 cell that
# contains the continuous optimum.  Termination is a proof (the bracket shrinks
# by a fixed factor 0.618 each step), not a budget.
_INV_PHI = 0.6180339887498949
GOLDEN_TOL_ULP = 0.05


def ulp16(x: float) -> float:
    """Width of the float16 cell containing x (the shippable lattice step)."""
    h = np.float16(x)
    nxt = np.nextafter(h, np.float16(np.inf))
    step = abs(float(nxt) - float(h))
    return step if step > 0.0 else 6e-8


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)


def bracket_out(evaluate, x0: float, d0: float, step0: float, max_expand: int):
    """Swann-style outward bracketing; self-terminating, no budget knob.

    Returns (lo, hi, best_x, best_d, n_eval): the outer bracket that contains
    the improving direction, plus the best point found.  MONOTONE-SAFE:
    ``best_d`` starts at ``d0`` and is replaced only on a STRICT decrease.
    """
    best_x, best_d, n = x0, d0, 0
    direction, step = 0.0, step0
    lo, hi = x0 - step0, x0 + step0
    for sign in (1.0, -1.0):
        dv = evaluate(x0 + sign * step0)
        n += 1
        if dv < best_d:
            best_x, best_d, direction = x0 + sign * step0, dv, sign
            break
    if direction == 0.0:
        # x0 already brackets the minimum between its two neighbours.
        return lo, hi, best_x, best_d, n
    prev = x0
    for _ in range(max_expand):
        step *= 2.0
        cand = best_x + direction * step
        dv = evaluate(cand)
        n += 1
        if dv >= best_d:
            lo, hi = sorted((prev, cand))
            return lo, hi, best_x, best_d, n
        prev, best_x, best_d = best_x, cand, dv
    lo, hi = sorted((prev, best_x + direction * step))
    return lo, hi, best_x, best_d, n


def golden(evaluate, lo: float, hi: float, best_x: float, best_d: float,
           tol: float, max_iter: int = 40):
    """Golden-section polish inside [lo, hi]; terminates when width < tol.

    Monotone-safe: the incumbent (best_x, best_d) is only replaced on a strict
    decrease, so the polish can never return a worse point than it was given.
    """
    n = 0
    a, b = lo, hi
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = evaluate(c), evaluate(d)
    n += 2
    for f, x in ((fc, c), (fd, d)):
        if f < best_d:
            best_d, best_x = f, x
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = evaluate(c)
            n += 1
            if fc < best_d:
                best_d, best_x = fc, c
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = evaluate(d)
            n += 1
            if fd < best_d:
                best_d, best_x = fd, d
    return best_x, best_d, n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="celldrop50")
    ap.add_argument("--final-jsonl", type=Path, default=V4D / "pw1/final_pw1.jsonl")
    ap.add_argument("--out-dir", type=Path, default=V4D / "mq1")
    ap.add_argument("--pairs", type=int, default=48,
                    help="probe the N highest-d_pose pairs (mass-ordered)")
    ap.add_argument("--columns", default="0,1,2",
                    help="pose translation columns to probe (0=fwd,1=lat,2=vert). "
                         "Column 0 is the NEGATIVE CONTROL: ddm_pw1 already ran a "
                         "self-terminating bracket on it, so a large gap_search "
                         "on column 0 would mean this instrument is finding "
                         "floating-point noise rather than real basins.")
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--max-expand", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c

    cols = [int(c) for c in args.columns.split(",") if c.strip()]
    if any(c not in (0, 1, 2) for c in cols):
        raise SystemExit("--columns must be a subset of 0,1,2 (translation only)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Column 0 rides the QA65 offset lattice, columns 1/2 ship as plain f16
    # (ddm_v4d_build_composed_archive.py:176-184).  The shippable-point
    # quantizer must match the format the receiver reconstructs, per column.
    dim0_offset = float(json.loads(
        (V4D / "v4d_composed_pw1_build_receipt.json").read_text())["dim0_offset"])

    def shippable(col: int, x: float) -> float:
        if col == 0:
            return dim0_offset + float(np.float16(x - dim0_offset))
        return float(np.float16(x))
    shipped = {int(r["pair"]): r for r in
               (json.loads(ln) for ln in
                args.final_jsonl.read_text().splitlines() if ln.strip())}
    if len(shipped) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} shipped rows, got {len(shipped)}")

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)

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

    order = sorted(range(N_PAIRS), key=lambda p: -shipped[p]["d_final"])
    seq = order[: args.pairs]
    jl = args.out_dir / "mq1_lattice.jsonl"
    cache = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in (jl.read_text().splitlines() if jl.exists() else [])
             if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[mq1] base={args.base} pairs={len(seq)} cols={cols} "
          f"cached={len(cache)}", flush=True)

    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[mq1] wall cap; {len(cache)} done; rerun to resume", flush=True)
            break
        sh = shipped[pidx]
        pose = np.asarray(sh["p"], np.float64).copy()
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        if "beta_mag" in sh:
            g = float(sh["beta_mag"])
        else:
            bi = int(sh["beta_idx"])
            if not 0 <= bi < len(BETA_MAGS):
                raise SystemExit(f"pair {pidx}: beta_idx={bi} with no beta_mag")
            g = float(BETA_MAGS[bi])
        s_t = float(v4c._d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)

        d_ctrl = score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g)
        n_eval = 1
        rec = {"pair": int(pidx), "rank": rank,
               "d_shipped": float(sh["d_final"]), "d_ctrl": d_ctrl,
               "canary_abs_err": abs(d_ctrl - float(sh["d_final"])),
               "s_t": s_t, "selector": sel, "beta_mag": g, "cols": {}}

        for col in cols:
            x0 = float(pose[col])
            # The lattice step is the width of the cell the RECEIVER can
            # address for this column: the offset residual for column 0,
            # the value itself for columns 1 and 2.
            step0 = ulp16(x0 - dim0_offset) if col == 0 else ulp16(x0)

            def make_eval(p, c, ctx=(s_t, sel, f1_u8, f1_f, tp, a, b, g)):
                def _ev(x):
                    q = p.copy()
                    q[c] = x
                    return score(q, *ctx)
                return _ev

            ev = make_eval(pose, col)
            lo, hi, bx, bd, nb = bracket_out(ev, x0, d_ctrl, step0,
                                             args.max_expand)
            gx, gd, ng = golden(ev, lo, hi, bx, bd,
                                tol=GOLDEN_TOL_ULP * step0)
            # Nearest SHIPPABLE point to the continuous optimum, in the format
            # the receiver actually reconstructs for this column.
            xq = shippable(col, gx)
            dq = ev(xq)
            n_eval += nb + ng + 1
            rec["cols"][str(col)] = {
                "x_shipped": x0, "ulp": step0,
                "x_cont": gx, "d_cont": gd,
                "x_f16": xq, "d_f16": dq,
                # what a FINER LATTICE could buy (unreachable by more search)
                "gap_lattice": max(dq - gd, 0.0),
                # what a BETTER SEARCH could buy at TODAY's lattice
                "gap_search": max(d_ctrl - dq, 0.0),
                # Approximate: step0 is the ulp AT x0, and f16 cell width grows
                # with magnitude, so this over-counts for large moves.  It is a
                # descriptive scale readout, never an input to a verdict.
                "moved_ulps": abs(gx - x0) / step0,
            }
        rec["n_eval"] = n_eval
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if len(cache) % 5 == 0 or rank < 3:
            c0 = rec["cols"][str(cols[0])]
            print(f"[mq1 {len(cache):3d}/{len(seq)}] pair {pidx} "
                  f"ctrl {d_ctrl:.5f} col{cols[0]} lat {c0['gap_lattice']:.2e} "
                  f"srch {c0['gap_search']:.2e} | "
                  f"canary {rec['canary_abs_err']:.2e} "
                  f"{time.time() - t0:.0f}s", flush=True)

    fj.close()
    summarize(args, cache, shipped, cols)


def summarize(args, cache: dict, shipped: dict, cols: list[int]) -> None:
    rows = [cache[p] for p in sorted(cache)]
    if not rows:
        print("[mq1] no rows")
        return
    d_all = np.array([shipped[p]["d_final"] for p in range(N_PAIRS)])
    base_mean = float(d_all.mean())
    floor = max(r["canary_abs_err"] for r in rows)
    d_probe = np.array([r["d_ctrl"] for r in rows])
    out: dict = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False, "promotion_eligible": False, "research_only": True,
        "pointer_moved": False,
        "base": args.base, "n_pairs_probed": len(rows), "n_population": N_PAIRS,
        "canary_max_abs_err": floor,
        "population_mean_d_shipped": base_mean,
        "probe_mass_fraction": float(d_probe.sum() / d_all.sum()),
        "generated_by": "tools/mq1_pose_lattice_resolution_probe.py",
        "columns": {},
    }
    def delta_S(total_gain: float) -> float:
        """EXACT score delta from removing ``total_gain`` of summed d_pose.

        NOT linearised: sqrt is concave, so ``ds_dd * delta`` overstates the
        gain, and the overstatement grows with the gain.  Computing both
        endpoints avoids that.

        SCOPE OF THIS NUMBER: it credits ONLY the probed pairs, so as an
        estimate of the same treatment applied to all 600 pairs it is a
        monotone LOWER bound on the achievable summed gain -- every unprobed
        pair can only add, because each arm accepts only a strict decrease.
        It is NOT a lower bound on the realized score: it is measured at the
        frozen local PoseNet on advisory hardware, and only a byte-closed
        exact eval settles the realized value.
        """
        new_mean = base_mean - total_gain / N_PAIRS
        return float(np.sqrt(10.0 * new_mean) - np.sqrt(10.0 * base_mean))

    # Gap to the demonstrated floor (PR130 0.172141) that every delta is
    # reported against, per the ddm charter; S_now is the pw1 exact-eval row.
    s_now, s_bar = 0.9476091, 0.172141
    gap = s_now - s_bar
    out["gap_to_bar"] = gap
    for col in cols:
        k = str(col)
        gl = np.array([r["cols"][k]["gap_lattice"] for r in rows])
        gs = np.array([r["cols"][k]["gap_search"] for r in rows])
        mv = np.array([r["cols"][k]["moved_ulps"] for r in rows])
        dS_lat, dS_srch = delta_S(float(gl.sum())), delta_S(float(gs.sum()))
        out["columns"][k] = {
            "gap_lattice_sum": float(gl.sum()),
            "gap_lattice_frac_of_probe_d": float(gl.sum() / d_probe.sum()),
            "gap_lattice_median_rel": float(np.median(gl / np.maximum(d_probe, 1e-30))),
            "gap_search_sum": float(gs.sum()),
            "gap_search_frac_of_probe_d": float(gs.sum() / d_probe.sum()),
            "moved_ulps_median": float(np.median(mv)),
            "moved_ulps_max": float(mv.max()),
            "n_above_noise_floor": int((gl > floor).sum()),
            "delta_S_probed_pairs_lattice": dS_lat,
            "delta_S_probed_pairs_search": dS_srch,
            "pct_of_gap_lattice": 100.0 * abs(dS_lat) / gap,
            "pct_of_gap_search": 100.0 * abs(dS_srch) / gap,
        }
    _atomic_write(args.out_dir / "mq1_receipt.json",
                  json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True), flush=True)
    print(f"[mq1] receipt {args.out_dir / 'mq1_receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
