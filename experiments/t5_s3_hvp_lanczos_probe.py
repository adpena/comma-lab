#!/usr/bin/env python
"""T5-crucible seat S3 — $0 HVP-Lanczos GN/Fisher-spectrum probe on a saved #205 checkpoint.

[macOS-CPU/MLX advisory] NON-PROMOTABLE — NOT a score claim. Pointer contest-CPU 0.19110 UNMOVED.

Implements the pinned D-3/4/5 verdict's FIRST measurement: top Hessian/GN Ritz values of the
tau-stage training loss at a saved mod32cap checkpoint (clean T3 baseline — no lever confounds),
via Lanczos with full reorthogonalization. HVPs REUSE tools/quadratic_basin_finisher_probe.py
(#341) SolveCtx.hvp_pair (mx.vjp of mx.grad; exact symmetric HVP; MLX jvp lacks SliceUpdate so
forward-over-reverse is not available — vjp-of-grad is). De-orphan, don't rebuild.

Outputs per checkpoint tag:
  - Ritz values theta_1..theta_k (subset-loss Hessian spectrum, K pairs, seed-deterministic)
  - lambda_max, most-negative Ritz (saddle indicator), n_negative
  - Newton-decrement estimate 0.5*||g||^2 * (e1^T T^-1 e1)  [only reported when T is PD]
    with v1 = g/||g|| so the Krylov space is gradient-aligned (remaining-quadratic-descent).
  Units: the SOLVE loss (100*tau_softplus surrogate + 0.001*length), NOT d_seg — the
  surrogate<->d_seg mapping is a separate SENSE calibration; do not conflate.

Chunked resumable foreground: each invocation does bounded Lanczos iterations within
--max-seconds and persists (V, alphas, betas, g) to a durable npz; rerun until "done".
Envelope: mx.cpu, mx.eval + clear_cache per pair (inherited from SolveCtx), RAM floor check.

Usage:
  .venv/bin/python experiments/t5_s3_hvp_lanczos_probe.py \
      --ckpt experiments/results/basin_finisher_probe_20260707/ema_best_snapshot.npz \
      --tag ep650 --feats-tag main_gt1 --k-pairs 8 --subset-seed 0 --iters 12 --max-seconds 480
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "experiments/results/t5_s3_hvp_lanczos_20260707"
RAM_FLOOR_GIB = 10.0


def _load_probe_mod():
    spec = importlib.util.spec_from_file_location(
        "quadratic_basin_finisher_probe", REPO / "tools/quadratic_basin_finisher_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flat(mod, ctx, d: dict) -> np.ndarray:
    return mod.flatten_masked(d, list(ctx.keys))


def _unflat(mod, ctx, v: np.ndarray) -> dict:
    return mod.unflatten_masked(v, ctx.shapes, list(ctx.keys))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--feats-tag", type=str, default="main_gt1")
    ap.add_argument("--k-pairs", type=int, default=8)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--max-seconds", type=float, default=480.0)
    ap.add_argument("--timing-only", action="store_true",
                    help="measure one grad + one HVP wall-time and exit (feasibility gate)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="gpu = MLX-GPU THROUGHPUT ONLY (never bit-exact cross-process; memory "
                         "L70) — use --verify-device for a CPU spot-check of the HVP")
    ap.add_argument("--verify-device", action="store_true",
                    help="compute one HVP on cpu AND gpu, print rel error, exit")
    args = ap.parse_args(argv)

    mod = _load_probe_mod()
    ok, avail = mod.ram_floor_ok(RAM_FLOOR_GIB)
    if not ok:
        print(json.dumps({"stage": "refuse_ram", "avail_gib": round(avail, 1)}))
        return 4

    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st_path = OUT_DIR / f"lanczos_state_{args.tag}_K{args.k_pairs}_s{args.subset_seed}.npz"
    out_path = OUT_DIR / f"spectrum_{args.tag}_K{args.k_pairs}_s{args.subset_seed}.json"

    ctx = mod.Ctx(args.ckpt)
    theta = {k: ctx.params[k] for k in ctx.keys}
    sctx = mod.SolveCtx(ctx, args.feats_tag, args.k_pairs, args.subset_seed, theta)
    if args.device == "gpu":
        # THROUGHPUT ONLY (research-signal; MLX-GPU is never bit-exact cross-process).
        sctx.mx.set_default_device(sctx.mx.gpu)
    n_dim = int(sum(np.prod(ctx.shapes[k]) for k in ctx.keys))

    def hvp_flat(v: np.ndarray) -> np.ndarray:
        vd = _unflat(mod, ctx, v.astype(np.float32))
        acc = np.zeros(n_dim, np.float64)
        for pi in sctx.subset:
            acc += _flat(mod, ctx, sctx.hvp_pair(theta, vd, pi)).astype(np.float64)
        return acc / len(sctx.subset)

    def grad_flat() -> np.ndarray:
        acc = np.zeros(n_dim, np.float64)
        for pi in sctx.subset:
            acc += _flat(mod, ctx, sctx.grad_pair(theta, pi)).astype(np.float64)
        return acc / len(sctx.subset)

    if args.timing_only:
        t0 = time.time(); g = grad_flat(); tg = time.time() - t0
        v = g / (np.linalg.norm(g) + 1e-30)
        t0 = time.time(); _ = hvp_flat(v); th = time.time() - t0
        row = {"stage": "timing", "tag": args.tag, "k_pairs": args.k_pairs,
               "device": args.device,
               "grad_seconds": round(tg, 2), "hvp_seconds": round(th, 2),
               "grad_norm": float(np.linalg.norm(g)), "n_dim": n_dim,
               "epoch": int(ctx.cfg["__epoch"])}
        print(json.dumps(row)); return 0

    if args.verify_device:
        # One-pair HVP on cpu vs gpu: rel error gate for using gpu as throughput device.
        rng = np.random.default_rng(0)
        v = rng.standard_normal(n_dim).astype(np.float32)
        v /= np.linalg.norm(v)
        vd = _unflat(mod, ctx, v)
        pi = sctx.subset[0]
        sctx.mx.set_default_device(sctx.mx.cpu)
        h_cpu = _flat(mod, ctx, sctx.hvp_pair(theta, vd, pi)).astype(np.float64)
        sctx.mx.set_default_device(sctx.mx.gpu)
        h_gpu = _flat(mod, ctx, sctx.hvp_pair(theta, vd, pi)).astype(np.float64)
        rel = float(np.linalg.norm(h_cpu - h_gpu) / (np.linalg.norm(h_cpu) + 1e-30))
        row = {"stage": "verify_device", "pair": int(pi),
               "hvp_norm_cpu": float(np.linalg.norm(h_cpu)),
               "hvp_norm_gpu": float(np.linalg.norm(h_gpu)), "rel_err": rel}
        print(json.dumps(row)); return 0

    # ---- resume or init Lanczos state (full reorthogonalization; V kept: n_dim*k*8B tiny) ----
    if st_path.exists():
        st = np.load(st_path)
        V = [np.asarray(st["V"][i], np.float64) for i in range(st["V"].shape[0])]
        alphas = list(np.asarray(st["alphas"], np.float64))
        betas = list(np.asarray(st["betas"], np.float64))
        g = np.asarray(st["g"], np.float64)
    else:
        g = grad_flat()
        v1 = g / (np.linalg.norm(g) + 1e-30)
        V, alphas, betas = [v1], [], []

    done = False
    while len(alphas) < args.iters:
        if time.time() - t_start > args.max_seconds:
            break
        v = V[-1]
        w = hvp_flat(v)
        a = float(np.dot(v, w))
        alphas.append(a)
        w = w - a * v - (betas[-1] * V[-2] if len(V) > 1 else 0.0)
        for u in V:  # full reorthogonalization (twice for fp hygiene)
            w -= np.dot(u, w) * u
        for u in V:
            w -= np.dot(u, w) * u
        b = float(np.linalg.norm(w))
        if b < 1e-12 or len(alphas) >= args.iters:
            done = True
            break
        betas.append(b)
        V.append(w / b)
        np.savez(st_path.with_suffix(".tmp.npz"), V=np.stack(V), alphas=np.asarray(alphas),
                 betas=np.asarray(betas), g=g)
        st_path.with_suffix(".tmp.npz").replace(st_path)
    if len(alphas) >= args.iters:
        done = True

    k = len(alphas)
    T = np.diag(np.asarray(alphas, np.float64))
    for i, b in enumerate(betas[: k - 1]):
        T[i, i + 1] = T[i + 1, i] = b
    ritz = np.linalg.eigvalsh(T) if k else np.array([])
    gnorm = float(np.linalg.norm(g))
    decrement = None
    if k and ritz.min() > 0:
        e1 = np.zeros(k); e1[0] = 1.0
        decrement = float(0.5 * gnorm**2 * np.dot(e1, np.linalg.solve(T, e1)))
    row = {
        "stage": "lanczos", "done": bool(done), "tag": args.tag,
        "ckpt": str(args.ckpt), "epoch": int(ctx.cfg["__epoch"]),
        "k_pairs": args.k_pairs, "subset_seed": args.subset_seed,
        "subset_pairs": [int(p) for p in sctx.subset],
        "iters_done": k, "iters_target": args.iters,
        "ritz": [float(x) for x in ritz],
        "lambda_max": float(ritz.max()) if k else None,
        "lambda_min_ritz": float(ritz.min()) if k else None,
        "n_negative_ritz": int((ritz < 0).sum()) if k else None,
        "grad_norm": gnorm,
        "newton_decrement_est_surrogate_units": decrement,
        "units": "SOLVE loss = 100*tau_softplus(tau=0.3) + 0.001*length (NOT d_seg)",
        "axis": "[macOS-CPU/MLX advisory] NON-PROMOTABLE",
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out_path.write_text(json.dumps(row, indent=1))
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
