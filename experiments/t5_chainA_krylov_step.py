#!/usr/bin/env python
"""T5 PURSUIT-CHAIN-A — curvature-aware Krylov-TR step from a saved Lanczos state ($0 solve step).

[macOS-CPU/MLX advisory] NON-PROMOTABLE — NOT a score claim. Pointer contest-CPU 0.19110 UNMOVED.

Consumes the Lanczos state (V, alphas, betas, g) produced by experiments/t5_s3_hvp_lanczos_probe.py
and builds step candidates at ZERO extra HVPs:
  - exact trust-region steps on the tridiagonal T over a radius ladder (hard case handled via the
    lambda_min eigenvector), Delta = V^T y*;
  - pure +/- s * u_min negative-curvature line-search steps (u_min = extreme Ritz vector).

Every candidate is screened by MEASURED loss (never predicted reduction — the through-R uint8-STE
FD gap makes model-trust invalid):
  - solve-subset loss (the subset the Lanczos state was measured on) — reference only;
  - HOLDOUT subset loss (different seed) — the selection signal (subset-overfit-aware per #341);
  - int8-dequant deploy loss on the holdout (quantization survival — the verdict path quantizes).

Winner (by holdout deploy loss) is saved as a stepped checkpoint npz (cfg keys copied) ready for
tools/quadratic_basin_finisher_probe.py stage `verdict` (chunked n600, same reconstruction path).

Foreground, bounded; GPU = throughput only (research signal).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "experiments/results/t5_pursuit_chainA_20260707"
RAM_FLOOR_GIB = 10.0


def _load_probe_mod():
    spec = importlib.util.spec_from_file_location(
        "quadratic_basin_finisher_probe", REPO / "tools/quadratic_basin_finisher_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tr_step_tridiag(alphas: np.ndarray, betas: np.ndarray, gnorm: float, radius: float
                    ) -> tuple[np.ndarray, dict]:
    """Exact TR solution y* of min 0.5 y^T T y + gnorm*e1^T y  s.t. ||y|| <= radius.

    Returns (y*, info). Standard More-Sorensen via eigendecomposition of the small tridiagonal
    (k <= 32 — exact eig is free). Hard case: fill the radius along the lambda_min eigenvector.
    """
    k = len(alphas)
    T = np.diag(np.asarray(alphas, np.float64))
    for i, b in enumerate(np.asarray(betas, np.float64)[: k - 1]):
        T[i, i + 1] = T[i + 1, i] = b
    w, Y = np.linalg.eigh(T)
    c = gnorm * Y[0, :]  # components of the linear term in the eigenbasis

    def y_of(lam: float) -> np.ndarray:
        return Y @ (-c / (w + lam))

    lam_lo = max(0.0, -float(w[0])) + 1e-12
    # interior solution if T is PD and unconstrained Newton step fits
    if w[0] > 0:
        y0 = y_of(0.0)
        if np.linalg.norm(y0) <= radius:
            return y0, {"case": "interior", "lam": 0.0, "pred_red": float(0.5 * np.sum(c**2 / w))}
    # boundary: find lam > lam_lo with ||y(lam)|| = radius (norm decreasing in lam)
    lo, hi = lam_lo, lam_lo + 1.0
    while np.linalg.norm(y_of(hi)) > radius and hi < 1e12:
        hi *= 10.0
    if np.linalg.norm(y_of(lo + 1e-9)) < radius:
        # hard case: even at lam -> -lambda_min the norm is short; fill along v_min
        yh = y_of(lo + 1e-9)
        gap = radius**2 - float(np.dot(yh, yh))
        tau = np.sqrt(max(gap, 0.0))
        y = yh + tau * Y[:, 0]
        pred = -(float(np.dot(c, y)) + 0.5 * float(y @ (Y @ (w * (Y.T @ y)))))
        return y, {"case": "hard", "lam": float(lo), "pred_red": pred}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(y_of(mid)) > radius:
            lo = mid
        else:
            hi = mid
    y = y_of(hi)
    pred = -(float(np.dot(c, y)) + 0.5 * float(y @ (Y @ (w * (Y.T @ y)))))
    return y, {"case": "boundary", "lam": float(hi), "pred_red": pred}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True, help="lanczos_state_*.npz")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--feats-tag", type=str, default="main_gt1")
    ap.add_argument("--out-tag", type=str, required=True)
    ap.add_argument("--radii", type=str, default="0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--negcurv-steps", type=str, default="0.01,0.03,0.1,0.3")
    ap.add_argument("--holdout-k", type=int, default=64)
    ap.add_argument("--holdout-seed", type=int, default=7)
    ap.add_argument("--solve-k", type=int, default=32, help="k_pairs of the solve subset (for ref loss)")
    ap.add_argument("--solve-seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--skip-solve-ref", action="store_true")
    args = ap.parse_args(argv)

    mod = _load_probe_mod()
    ok, avail = mod.ram_floor_ok(RAM_FLOOR_GIB)
    if not ok:
        print(json.dumps({"stage": "refuse_ram", "avail_gib": round(avail, 1)}))
        return 4
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = mod.Ctx(args.ckpt)
    theta = {k: ctx.params[k] for k in ctx.keys}

    st = np.load(args.state)
    V = np.asarray(st["V"], np.float64)
    alphas = np.asarray(st["alphas"], np.float64)
    betas = np.asarray(st["betas"], np.float64)
    g = np.asarray(st["g"], np.float64)
    k = len(alphas)
    gnorm = float(np.linalg.norm(g))
    T = np.diag(alphas)
    for i, b in enumerate(betas[: k - 1]):
        T[i, i + 1] = T[i + 1, i] = b
    w, Y = np.linalg.eigh(T)
    u_min = V[:k].T @ Y[:, 0]
    u_min /= np.linalg.norm(u_min)

    # ---- candidates ----
    cands: list[tuple[str, np.ndarray]] = []
    for r in [float(x) for x in args.radii.split(",")]:
        y, info = tr_step_tridiag(alphas, betas, gnorm, r)
        d = V[:k].T @ y
        cands.append((f"tr_r{r:g}_{info['case']}", d))
    for s in [float(x) for x in args.negcurv_steps.split(",")]:
        cands.append((f"negcurv+{s:g}", s * u_min))
        cands.append((f"negcurv-{s:g}", -s * u_min))

    # ---- screens ----
    sc_hold = mod.SolveCtx(ctx, args.feats_tag, args.holdout_k, args.holdout_seed, theta)
    if args.device == "gpu":
        sc_hold.mx.set_default_device(sc_hold.mx.gpu)
    sc_solve = None
    if not args.skip_solve_ref:
        sc_solve = mod.SolveCtx(ctx, args.feats_tag, args.solve_k, args.solve_seed, theta)
        if args.device == "gpu":
            sc_solve.mx.set_default_device(sc_solve.mx.gpu)

    def unflat(v: np.ndarray) -> dict:
        return mod.unflatten_masked(v.astype(np.float32), ctx.shapes, list(ctx.keys))

    def theta_plus(d: np.ndarray) -> dict:
        dd = unflat(d)
        return {kk: theta[kk] + dd[kk] for kk in ctx.keys}

    rows = []
    t0 = time.time()
    L0_hold = sc_hold.loss_at(theta)
    L0_hold_q = sc_hold.loss_at(ctx.deploy(theta))
    L0_solve = sc_solve.loss_at(theta) if sc_solve else None
    base = {"stage": "baselines", "L0_holdout": L0_hold, "L0_holdout_int8deploy": L0_hold_q,
            "L0_solve_subset": L0_solve, "gnorm": gnorm, "k_lanczos": k,
            "ritz_min": float(w[0]), "ritz_max": float(w[-1]),
            "holdout": {"k": args.holdout_k, "seed": args.holdout_seed}}
    print(json.dumps(base), flush=True)
    rows.append(base)
    for name, d in cands:
        tp = theta_plus(d)
        l_h = sc_hold.loss_at(tp)
        l_q = sc_hold.loss_at(ctx.deploy(tp))
        l_s = sc_solve.loss_at(tp) if sc_solve else None
        row = {"stage": "candidate", "name": name, "step_norm": float(np.linalg.norm(d)),
               "holdout_loss": l_h, "holdout_delta": l_h - L0_hold,
               "holdout_int8deploy_loss": l_q, "holdout_int8deploy_delta": l_q - L0_hold_q,
               "solve_subset_loss": l_s,
               "solve_subset_delta": (l_s - L0_solve) if l_s is not None else None,
               "secs": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        rows.append(row)

    cand_rows = [r for r in rows if r["stage"] == "candidate"]
    winner = min(cand_rows, key=lambda r: r["holdout_int8deploy_loss"])
    summary = {"stage": "winner", "name": winner["name"],
               "holdout_delta": winner["holdout_delta"],
               "holdout_int8deploy_delta": winner["holdout_int8deploy_delta"],
               "improves_fp32": winner["holdout_delta"] < 0,
               "improves_int8deploy": winner["holdout_int8deploy_delta"] < 0}
    print(json.dumps(summary), flush=True)
    rows.append(summary)
    (OUT_DIR / f"krylov_step_screen_{args.out_tag}.json").write_text(json.dumps(rows, indent=1))

    # save stepped checkpoint for the n600 verdict stage
    d = dict(cands)[winner["name"]]
    tp = theta_plus(d)
    arrays = {kk: tp[kk] for kk in ctx.keys}
    zsrc = np.load(args.ckpt, allow_pickle=True)
    for ck in zsrc.files:
        if ck.startswith("__"):
            arrays[ck] = zsrc[ck]
    outp = OUT_DIR / f"stepped_theta_{args.out_tag}.npz"
    np.savez(outp, **arrays)
    print(json.dumps({"stage": "saved", "stepped_ckpt": str(outp), "winner": winner["name"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
