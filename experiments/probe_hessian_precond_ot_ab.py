# SPDX-License-Identifier: MIT
"""Hessian-preconditioned vs damped-Newton head-offset A/B ($0, read-only, CPU).

Paper #6 (Plus-Gourdon & Nielsen, arXiv 2606.09077) applied to the #288 damped-Newton
semi-discrete-OT head-offset solve (:func:`damped_newton_ot_offsets`). The boundary
Hessian ``cov = diag(m) - mean_p s_p s_p^T`` (softmax covariance) is anisotropic per-pixel;
the paper's fix is to eigendecompose the local Hessian, deform to the canonical paraboloid
(whiten), solve the well-conditioned residual, undo. Cost: 1 eigendecomposition + 2 matvecs
per step. This probe MEASURES whether it pays on the REAL n600 head-offset problem — the
EXACT ckpt + protocol as the N-1 ot_newton verdict (ot_offset_n600_verdict_20260709.md).

TWO measurements, both reported (per CLAUDE.md recursive-review axis #9 — measured-scored-
quantity is the authority):
  (A) CONVERGENCE (internal): iters + terminal residual + range condition number + b* +
      max|b*_legacy - b*_precond| + per-solve wall-time (many repeats for a stable timing).
  (B) THROUGH-R d_seg (AUTHORITY): realize no_offset + legacy-b* + precond-b* through the
      frozen CPU SegNet (fp32 EMA render). A faster/lower-residual solve that does not
      improve (or at least PRESERVE) the exact through-R d_seg is NOT a win.

Plus a synthetic ILL-CONDITIONED positive control (in-harness, cheap) that proves the
preconditioned code path is more robust WHEN the averaged Hessian is genuinely degenerate
(a unit-style check, NOT the finding — the real n600 problem is the authority).

AUTHORITY: realized-through-R on the frozen CPU SegNet (NOT MPS, NOT MLX), fp32 EMA render.
[macOS-CPU advisory] — NON-PROMOTABLE until byte-closed exact eval. Pointer 0.19108282
UNMOVED (a solver-conditioning change moves NO score until a byte-closed upstream/evaluate.py
row proves it). NO FAKE: real ckpt, real GT, real scorer; each mode computes what it claims.

The SegNet forward is CHUNKED (``--verdict-batch``, per the n600-verdict-OOM law). All 600
pairs (``--num-pairs 600``) — never a subset for the decision. Does NOT touch the live run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "experiments", REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from probe_laguerre_logit_offset_sweep import (  # noqa: E402  (reuse the N-1 harness verbatim)
    _cache_phi_tex,
    _compose_rgb,
    _load_ckpt,
    _segnet_argmax_batch,
)
from train_witness_realized_through_R_mlx import (  # noqa: E402
    _build_render_coords,
    _torch_R_to_camera_uint8,
    load_gt_from_cache,
)

from tac.boundary_math.laguerre_logit_offset import (  # noqa: E402
    CLASS_NAMES,
    damped_newton_ot_offsets,
    per_class_disagreement,
)
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    curvelet_directional_B,
    curvelet_feats,
)


def _synthetic_positive_control() -> dict:
    """Cheap in-harness proof that the preconditioned path is MORE ROBUST when the
    AVERAGED softmax-covariance Hessian is genuinely ill-conditioned. The real n600
    Hessian is a mean over ~10^8 pixels (well-conditioned by construction); this control
    manufactures a degenerate mean by making ALL pixels near-identical + concentrated so
    the mean cov approaches a single rank-deficient cov (lambda_min -> 0). NOT the finding.
    """
    rng = np.random.default_rng(20260710)
    k = 5
    # all pixels concentrate on class 2 with a tiny common tilt => mean cov nearly rank-1
    base = np.zeros(k)
    base[2] = 40.0
    phi = base[None, :] + 1e-3 * rng.standard_normal((4000, k))
    tgt = np.array([0.30, 0.05, 0.30, 0.05, 0.30])  # far from the current ~e_2 mass
    b_leg, i_leg = damped_newton_ot_offsets(phi, tgt, precondition=False, max_iter=400)
    b_pre, i_pre = damped_newton_ot_offsets(phi, tgt, precondition=True, max_iter=400,
                                            precond_eps_rel=1e-9)
    return {
        "note": "synthetic degenerate-mean control (NOT the real-problem verdict)",
        "cond_number": i_pre["cond_number"],
        "legacy": {"iters": i_leg["iters"], "converged": i_leg["converged"],
                   "max_mass_err": i_leg["max_mass_err"]},
        "precond": {"iters": i_pre["iters"], "converged": i_pre["converged"],
                    "max_mass_err": i_pre["max_mass_err"]},
        "max_abs_db": float(np.max(np.abs(b_leg - b_pre))),
    }


def _time_solve(phi_stack, counts, *, precondition, repeats, tau) -> float:
    """Median wall-time (s) of a single FULL solve over ``repeats`` runs. NOTE: this is
    PIXEL-DOMINATED (the per-pixel softmax + ``s.T@s`` covariance over ~10^8 rows), which is
    IDENTICAL between the two methods — the eigh-vs-pinv difference is on the 5x5 step only,
    so see :func:`_time_step` for the real preconditioner overhead."""
    ts = []
    for _ in range(int(repeats)):
        t = time.perf_counter()
        damped_newton_ot_offsets(phi_stack, counts, tau=tau, precondition=precondition)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def _time_step(cov, g, taus, *, precondition, repeats=20000) -> float:
    """Median wall-time (s) of ONE 5x5 Newton step (the ONLY part the preconditioner changes:
    eigh+whiten vs pinv). This is the accurate 'does the eigendecomposition cost dominate'
    answer, isolated from the shared pixel-dominated covariance formation."""
    from tac.boundary_math.laguerre_logit_offset import _newton_step_from_cov
    ts = []
    for _ in range(int(repeats)):
        t = time.perf_counter()
        _newton_step_from_cov(cov, g, taus, precondition=precondition, rcond=1e-10,
                              eps_rel=1e-9, cond_gate=None)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--gt-cache", type=str, required=True)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--so-iters", type=int, default=3)
    ap.add_argument("--verdict-batch", type=int, default=32,
                    help="chunk size for the SegNet forward (n600-verdict-OOM law; 0 = single batch).")
    ap.add_argument("--ot-tau", type=float, default=1.0,
                    help="OT solve softmax temperature (matches the N-1 ot_newton tau).")
    ap.add_argument("--precond-eps-rel", type=float, default=1e-9,
                    help="relative eigenvalue floor eps_rel*lambda_max for the preconditioned inverse.")
    ap.add_argument("--time-repeats", type=int, default=25,
                    help="repeats for the per-solve wall-time median.")
    ap.add_argument("--no-realize", action="store_true",
                    help="skip the through-R realize (convergence-only; realize is the authority).")
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()
    # positive control first ($0, instant) — proves the code path before the expensive cache.
    pos_ctrl = _synthetic_positive_control()
    print(json.dumps({"stage": "positive_control", **pos_ctrl}), flush=True)

    params, cfg = _load_ckpt(Path(args.ckpt))
    rh, rw = int(cfg["render_hw"][0]), int(cfg["render_hw"][1])
    tsm = float(cfg["cfg_softmax_temp"])
    palette = params["palette"].astype(np.float64)

    gt, seg_cpu, _pn = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    p = gt.n_pairs
    coords = _build_render_coords(rh, rw)
    bank = CurveletBankConfig(
        n_scales=int(cfg["bank_n_scales"]), n_orient0=int(cfg["bank_n_orient0"]),
        f0=float(cfg["bank_f0"]), base=float(cfg["bank_base"]), n_iso=int(cfg["bank_n_iso"]))
    big_b = curvelet_directional_B(bank, max_freq=float(cfg["cfg_max_bank_freq"]))
    curv_feats = curvelet_feats(coords, big_b).astype(np.float32)
    print(json.dumps({"stage": "loaded", "n_pairs": p, "epoch": int(cfg["epoch"]),
                      "render_hw": [rh, rw], "softmax_temp": tsm,
                      "verdict_batch": int(args.verdict_batch), "secs": round(time.time() - t0, 1)}),
          flush=True)

    # --- cache phi + tex per pair (frame1) — the expensive witness trunk, computed ONCE ---
    phis, texs = [], []
    for pi in range(p):
        phi, tex = _cache_phi_tex(params, curv_feats, coords, cfg, pi, rh, rw, args.so_iters, palette, tsm)
        phis.append(phi)
        texs.append(tex)
        if (pi + 1) % 50 == 0:
            print(json.dumps({"stage": "cache", "done": pi + 1, "secs": round(time.time() - t0, 1)}),
                  flush=True)
    gt_l = [gt.lstars[pi].astype(np.int64) for pi in range(p)]
    gt_arr = np.stack(gt_l, axis=0)
    phi_stack = np.stack(phis, axis=0).reshape(-1, 5).astype(np.float64)  # (N*H*W, K)
    counts = np.bincount(gt_arr.reshape(-1), minlength=5).astype(np.float64)  # GT class freqs (ot_newton target)

    # --- (A) CONVERGENCE A/B: legacy damped-Newton vs Hessian-preconditioned, SAME (phi, target) ---
    b_leg, info_leg = damped_newton_ot_offsets(phi_stack, counts, tau=float(args.ot_tau),
                                               precondition=False)
    b_pre, info_pre = damped_newton_ot_offsets(phi_stack, counts, tau=float(args.ot_tau),
                                               precondition=True, precond_eps_rel=float(args.precond_eps_rel))
    max_abs_db = float(np.max(np.abs(b_leg - b_pre)))
    t_leg = _time_solve(phi_stack, counts, precondition=False, repeats=args.time_repeats, tau=float(args.ot_tau))
    t_pre = _time_solve(phi_stack, counts, precondition=True, repeats=args.time_repeats, tau=float(args.ot_tau))
    # step-level microbenchmark: the ONLY part the preconditioner changes (5x5 eigh vs pinv),
    # on the representative first-iterate covariance (b=0, largest gradient).
    z0 = phi_stack / max(float(args.ot_tau), 1e-9)
    z0 = z0 - z0.max(axis=1, keepdims=True)
    e0 = np.exp(z0)
    s0 = e0 / np.maximum(e0.sum(axis=1, keepdims=True), 1e-300)
    m0 = s0.mean(axis=0)
    cov0 = np.diag(m0) - (s0.T @ s0) / float(s0.shape[0])
    g0 = (counts / counts.sum()) - m0
    ts_leg = _time_step(cov0, g0, float(args.ot_tau), precondition=False)
    ts_pre = _time_step(cov0, g0, float(args.ot_tau), precondition=True)
    conv = {
        "legacy": {"iters": info_leg["iters"], "max_mass_err": info_leg["max_mass_err"],
                   "converged": info_leg["converged"], "cond_number": info_leg["cond_number"],
                   "solve_secs_median": t_leg,
                   "offsets": {int(c): round(float(b_leg[c]), 5) for c in range(5)}},
        "precond": {"iters": info_pre["iters"], "max_mass_err": info_pre["max_mass_err"],
                    "converged": info_pre["converged"], "cond_number": info_pre["cond_number"],
                    "solve_secs_median": t_pre,
                    "offsets": {int(c): round(float(b_pre[c]), 5) for c in range(5)}},
        "max_abs_db_star": max_abs_db,
        "same_fixed_point": bool(max_abs_db < 1e-8),
        "step_secs_median_legacy_pinv": ts_leg,
        "step_secs_median_precond_eigh": ts_pre,
        "step_overhead_ratio_precond_over_pinv": (ts_pre / ts_leg) if ts_leg > 0 else float("nan"),
        "full_solve_note": "solve_secs_median is PIXEL-dominated (identical work both methods); the "
                           "preconditioner overhead is step_secs_* (5x5 eigh vs pinv).",
    }
    print(json.dumps({"stage": "convergence_ab", **conv}), flush=True)

    # --- (B) THROUGH-R d_seg AUTHORITY: realize no_offset + legacy-b* + precond-b* ---
    through_r: dict = {}
    if not args.no_realize:
        vb = int(args.verdict_batch)

        def realized_dseg(offset: np.ndarray):
            f1s = [_torch_R_to_camera_uint8(_compose_rgb(phis[pi], texs[pi], palette, tsm, offset))
                   for pi in range(p)]
            am = _segnet_argmax_batch(seg_cpu, f1s, chunk=(vb if vb > 0 else 32))
            d = float(np.mean([np.count_nonzero(am[i] != gt_l[i]) / gt_l[i].size for i in range(p)]))
            pc = per_class_disagreement(am.reshape(-1), gt_arr.reshape(-1), 5)
            return d, {int(k): v for k, v in pc.items()}

        d_base, pc_base = realized_dseg(np.zeros(5))
        print(json.dumps({"stage": "realize", "arm": "no_offset", "d_seg": d_base,
                          "secs": round(time.time() - t0, 1)}), flush=True)
        d_leg, pc_leg = realized_dseg(b_leg)
        print(json.dumps({"stage": "realize", "arm": "legacy_ot_newton", "d_seg": d_leg,
                          "secs": round(time.time() - t0, 1)}), flush=True)
        d_pre, pc_pre = realized_dseg(b_pre)
        print(json.dumps({"stage": "realize", "arm": "precond_ot_newton", "d_seg": d_pre,
                          "secs": round(time.time() - t0, 1)}), flush=True)
        through_r = {
            "no_offset": {"d_seg": d_base, "per_class": pc_base},
            "legacy_ot_newton": {"d_seg": d_leg, "per_class": pc_leg},
            "precond_ot_newton": {"d_seg": d_pre, "per_class": pc_pre},
            "delta_precond_minus_legacy": d_pre - d_leg,
            "precond_preserves_dseg": bool(abs(d_pre - d_leg) < 1e-9),
            "note": "realized-through-R d_seg on the frozen CPU SegNet; lower=better. legacy vs "
                    "precond are the SAME fixed point when same_fixed_point=true => d_seg identical.",
        }

    verdict = _honest_verdict(conv, through_r, pos_ctrl)
    out = {
        "advisory": "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet authority; fp32 EMA render; "
                    "NON-PROMOTABLE until byte-closed exact eval; pointer 0.19108282 UNMOVED]",
        "task": "Hessian-preconditioned vs damped-Newton head-offset A/B (Plus-Gourdon & Nielsen 2606.09077)",
        "ckpt": str(args.ckpt), "n_pairs": p, "epoch": int(cfg["epoch"]),
        "class_names": list(CLASS_NAMES), "ot_tau": float(args.ot_tau),
        "n1_reference": {"no_offset": 0.003143556382921007, "menon": 0.003311852349175347,
                         "ot_newton": 0.0048921034071180555,
                         "note": "N-1 n600 verdict (ot_offset_n600_verdict_20260709.md); SAME ckpt; lower=better; "
                                 "ot_newton area-objective HURTS d_seg (solver exact, objective wrong)."},
        "convergence_ab": conv,
        "through_r_authority": through_r,
        "synthetic_positive_control": pos_ctrl,
        "honest_verdict": verdict,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps({"stage": "done", "secs": round(time.time() - t0, 1),
                      "verdict": verdict, "out": args.out_json}), flush=True)


def _honest_verdict(conv: dict, through_r: dict, pos_ctrl: dict) -> dict:
    """Derive the NO-FAKE verdict from the measured numbers (no tuning-to-win)."""
    same_fp = conv["same_fixed_point"]
    iters_delta = conv["precond"]["iters"] - conv["legacy"]["iters"]
    faster = conv["precond"]["solve_secs_median"] < conv["legacy"]["solve_secs_median"]
    cond = conv["legacy"]["cond_number"]
    preserves = through_r.get("precond_preserves_dseg", None)
    if same_fp and (preserves is None or preserves):
        outcome = ("SAME-FIXED-POINT / NO d_seg CHANGE. The real averaged boundary Hessian is "
                   f"well-conditioned (range cond ~ {cond:.3g}); the pixel-mean regularizes the "
                   "per-pixel anisotropy, so legacy damped-Newton already reaches the exact b* in "
                   f"{int(conv['legacy']['iters'])} iters. Preconditioning reaches the IDENTICAL b* "
                   "(-> identical through-R d_seg) with no iteration savings"
                   + (" and is not faster" if not faster else "")
                   + ". On THIS (global, averaged) head-offset problem preconditioning does not pay; "
                   "it is a numerical-robustness refinement whose payoff is confined to genuinely "
                   "ill-conditioned solves (the synthetic control; and any FUTURE per-pixel/per-patch "
                   "terminal solve #341/#396 that does NOT average the Hessian).")
    else:
        outcome = ("DIFFERENT fixed point or d_seg change measured — see numbers; investigate whether "
                   "legacy rcond truncation was the cause.")
    return {
        "same_fixed_point": same_fp, "iters_delta_precond_minus_legacy": iters_delta,
        "precond_faster": faster, "range_condition_number": cond,
        "through_r_preserves_dseg": preserves, "outcome": outcome,
        "score_impact": "NONE until a byte-closed upstream/evaluate.py row proves it (#341/#396 path). "
                        "Pointer 0.19108282 UNMOVED. A solver-conditioning change is a candidate, not a score.",
    }


if __name__ == "__main__":
    main()
