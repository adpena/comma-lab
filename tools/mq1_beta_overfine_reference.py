#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_mq1 -- the OVER-RESOLUTION reference for the v4d rolling-shutter beta menu.

WHY THIS EXISTS.  A quantizer must never be designed from quantized data.  The
13-entry ``rs_beta_mags`` table that ``ddm_pw1`` shipped LOOKS like a codebook,
so its occupancy histogram looks like a solution density.  It is neither.
DERIVED FROM SOURCE: ``bracket_out`` (``tools/pw1_pose_menu_saturation_ab.py``
:75-107) probes ``x0 +- step0`` and then doubles, so from a seed ``g0`` it can
only ever reach ``g0 +- BETA_STEP0 * (2^k - 1)``.  With ``BETA_STEP0 = 0.5``
(``ddm_v4d_resolve.py:71``) and the seed sweep ``BETA_MAGS = (0.0, 0.5, 1.0)``
that orbit is ``{0, +-0.5, +-1.5, +-2.5, +-3.5, +-4.5, +-7.5, ...}`` -- and
EVERY ONE of the 13 shipped values is a seed or an orbit point, with no
exceptions.  The shipped "menu" is the SEARCH'S REACHABLE SET, and its spacing
doubles with distance from the seed, so it is coarsest exactly where ddm_pw1
measured its largest wins (the 29 pairs that needed |g| > 1.0 AND the sign
opposed to yaw).

WHAT THIS TOOL DOES.  It establishes the over-resolution reference the design
of any beta menu must be fitted to, by re-running the SAME search at a step
10x finer and then polishing to the continuous optimum:

  ARM F   bracket at ``--fine-step`` (default 0.05 = BETA_STEP0/10), then a
          golden-section polish inside the final bracket.  The polish
          terminates by proof (the bracket contracts by 0.618 per step), not
          by a budget.
  ARM W   THE POSITIVE CONTROL the reference is worthless without.  The same
          ARM-F search restarted from a DELIBERATELY WRONG initialisation
          (``--wrong-init``, default the far end of the shipped table).  Over-
          resolution removes MENU censoring; it does NOT remove SOLVER bias.
          If ARM W does not recover ARM F's optimum, the reference is a solver
          artifact and every menu derived from it inherits that artifact.

RATE NOTE.  beta ships as ``rs_beta_mags`` in the archive manifest and is
applied as ``beta_mags[idx] * yaw_sign`` (``inflate_runner_v4d.py:127,177-180``),
so the table may hold ANY float and a finer beta costs no receiver change.  The
over-fine search is encoder-side only and NEVER SHIPS: only its designed
descendant does.

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
SCHEMA = "ddm_mq1_beta_overfine_reference.v1"
V4D = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
N_PAIRS = 600
BETA_MAGS = (0.0, 0.5, 1.0)
BETA_STEP0 = 0.5          # ddm_v4d_resolve.py:71 -- the SHIPPED bracket step
GOLDEN_TOL_FRAC = 0.02    # polish until the bracket is < 2% of the fine step


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="celldrop50")
    ap.add_argument("--final-jsonl", type=Path, default=V4D / "pw1/final_pw1.jsonl")
    ap.add_argument("--out-dir", type=Path, default=V4D / "mq1_beta")
    ap.add_argument("--pairs", type=int, default=48,
                    help="probe the N highest-d_pose pairs (mass-ordered)")
    ap.add_argument("--fine-step", type=float, default=BETA_STEP0 / 10.0,
                    help="over-resolution bracket step; the shipped search "
                         "used BETA_STEP0=0.5, so the default is 10x finer")
    ap.add_argument("--wrong-init", type=float, default=-7.5,
                    help="ARM W initialisation: a deliberately wrong start, "
                         "default the far negative end of the shipped table")
    ap.add_argument("--control-every", type=int, default=3,
                    help="run the ARM W positive control on every Nth pair")
    ap.add_argument("--max-minutes", type=float, default=75.0)
    ap.add_argument("--max-expand", type=int, default=12)
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
    from mq1_pose_lattice_resolution_probe import bracket_out, golden

    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    jl = args.out_dir / "mq1_beta.jsonl"
    cache = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in (jl.read_text().splitlines() if jl.exists() else [])
             if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[mq1beta] base={args.base} pairs={len(seq)} fine_step={args.fine_step} "
          f"wrong_init={args.wrong_init} cached={len(cache)}", flush=True)

    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[mq1beta] wall cap; {len(cache)} done; rerun to resume", flush=True)
            break
        sh = shipped[pidx]
        pose = np.asarray(sh["p"], np.float64).copy()
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        g_ship = (float(sh["beta_mag"]) if "beta_mag" in sh
                  else float(BETA_MAGS[int(sh["beta_idx"])]))
        s_t = float(v4c._d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)

        def ev(g, _c=(pose, s_t, sel, f1_u8, f1_f, tp, a, b)):
            return score(*_c, g)

        d_ctrl = ev(g_ship)
        n_eval = 1
        tol = GOLDEN_TOL_FRAC * args.fine_step

        # ARM F: over-resolution bracket + golden polish from the shipped point.
        lo, hi, bx, bd, nb = bracket_out(ev, g_ship, d_ctrl, args.fine_step,
                                         args.max_expand)
        gx, gd, ng = golden(ev, lo, hi, bx, bd, tol=tol)
        n_eval += nb + ng

        rec = {
            "pair": int(pidx), "rank": rank,
            "d_shipped": float(sh["d_final"]), "d_ctrl": d_ctrl,
            "canary_abs_err": abs(d_ctrl - float(sh["d_final"])),
            "g_shipped": g_ship, "s_t": s_t, "selector": sel,
            "arm_f_g": float(gx), "arm_f_d": float(gd),
            # what the shipped DOUBLING ORBIT could not express
            "gain_over_orbit": max(d_ctrl - gd, 0.0),
            "g_move": abs(float(gx) - g_ship),
        }

        # ARM W: the positive control -- same search, deliberately wrong start.
        if rank % args.control_every == 0:
            d_w0 = ev(float(args.wrong_init))
            lw, hw, bw, dw, nw = bracket_out(ev, float(args.wrong_init), d_w0,
                                             args.fine_step, args.max_expand)
            wx, wd, nwg = golden(ev, lw, hw, bw, dw, tol=tol)
            n_eval += 1 + nw + nwg
            rec["arm_w_g"] = float(wx)
            rec["arm_w_d"] = float(wd)
            # Recovery: does the wrong-init search reach ARM F's objective?
            # Relative, because d spans four orders of magnitude across pairs.
            rec["arm_w_recovery_rel"] = float((wd - gd) / max(gd, 1e-30))
            rec["arm_w_g_agrees"] = bool(abs(wx - gx) <= 2.0 * args.fine_step)

        rec["n_eval"] = n_eval
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if len(cache) % 5 == 0 or rank < 3:
            w = rec.get("arm_w_recovery_rel")
            print(f"[mq1beta {len(cache):3d}/{len(seq)}] pair {pidx} "
                  f"ctrl {d_ctrl:.5f} g {g_ship:+.3f}->{gx:+.4f} "
                  f"gain {rec['gain_over_orbit']:.3e} "
                  f"| ctrlW {'--' if w is None else f'{w:+.2e}'} "
                  f"canary {rec['canary_abs_err']:.1e} {time.time() - t0:.0f}s",
                  flush=True)

    fj.close()
    summarize(args, cache, shipped)


def summarize(args, cache: dict, shipped: dict) -> None:
    rows = [cache[p] for p in sorted(cache)]
    if not rows:
        print("[mq1beta] no rows")
        return
    d_all = np.array([shipped[p]["d_final"] for p in range(N_PAIRS)])
    base_mean = float(d_all.mean())
    d_probe = np.array([r["d_ctrl"] for r in rows])
    gain = np.array([r["gain_over_orbit"] for r in rows])
    mv = np.array([r["g_move"] for r in rows])
    floor = max(r["canary_abs_err"] for r in rows)
    ctl = [r for r in rows if "arm_w_recovery_rel" in r]

    def delta_S(total_gain: float) -> float:
        """EXACT (not linearised) score delta from removing summed d_pose.

        Credits ONLY the probed pairs, so as an estimate of the same treatment
        at n600 it is a monotone LOWER bound on the achievable summed gain --
        every unprobed pair can only add, since each arm accepts only a strict
        decrease.  It is NOT a claim about the realized score: it is measured
        at the frozen local PoseNet on advisory hardware.
        """
        return float(np.sqrt(10.0 * (base_mean - total_gain / N_PAIRS))
                     - np.sqrt(10.0 * base_mean))

    s_now, s_bar = 0.9476091, 0.172141
    gap = s_now - s_bar
    dS = delta_S(float(gain.sum()))
    out = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False, "promotion_eligible": False, "research_only": True,
        "pointer_moved": False,
        "base": args.base, "fine_step": args.fine_step,
        "shipped_step": BETA_STEP0,
        "n_pairs_probed": len(rows), "n_population": N_PAIRS,
        "canary_max_abs_err": floor,
        "population_mean_d_shipped": base_mean,
        "probe_mass_fraction": float(d_probe.sum() / d_all.sum()),
        "gain_over_orbit_sum": float(gain.sum()),
        "gain_over_orbit_frac_of_probe_d": float(gain.sum() / d_probe.sum()),
        "n_pairs_improved": int((gain > 0).sum()),
        "n_pairs_improved_above_floor": int((gain > floor).sum()),
        "g_move_median": float(np.median(mv)),
        "g_move_max": float(mv.max()),
        "delta_S_probed_pairs": dS,
        "pct_of_gap": 100.0 * abs(dS) / gap,
        "gap_to_bar": gap,
        "generated_by": "tools/mq1_beta_overfine_reference.py",
    }
    if ctl:
        rec = np.array([r["arm_w_recovery_rel"] for r in ctl])
        # The reference is trustworthy only if the two starts AGREE.  A large
        # NEGATIVE recovery_rel is a failure too, not a pass: it means the
        # wrong-init search beat the from-shipped search, i.e. ARM F is itself
        # start-dependent and is NOT the continuous optimum.  Judge |.|.
        adverse = float(np.median(np.abs(rec)))
        out["positive_control"] = {
            "n": len(ctl),
            "wrong_init": args.wrong_init,
            "recovery_rel_median": float(np.median(rec)),
            "recovery_rel_abs_median": adverse,
            "recovery_rel_min": float(rec.min()),
            "recovery_rel_max": float(rec.max()),
            "n_agree_within_1pct": int((np.abs(rec) <= 0.01).sum()),
            "n_wrong_init_STRICTLY_BETTER": int((rec < -0.01).sum()),
            "n_wrong_init_worse": int((rec > 0.01).sum()),
            "n_g_agrees": int(sum(r["arm_w_g_agrees"] for r in ctl)),
            "verdict": ("REFERENCE_TRUSTWORTHY" if adverse <= 0.01
                        else "SOLVER_BIASED_REFERENCE_NOT_TRUSTWORTHY"),
            "note": "recovery_rel = (d_wrong_init - d_from_shipped)/d_from_shipped. "
                    "0 means the two starts agree. POSITIVE means the wrong start "
                    "did worse (classic trapping). NEGATIVE means the wrong start "
                    "did BETTER, i.e. the from-shipped reference is itself trapped. "
                    "Both are disqualifying, so the verdict reads |recovery_rel|.",
        }
    (args.out_dir / "mq1_beta_receipt.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
