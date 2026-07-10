#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MLX-GPU cross-process bit-identity probe (task #348, 2026-07-07).

Measures, per op class, whether MLX Metal compute is bit-identical when the
SAME seeded computation runs in N SEPARATE processes. This is the localization
instrument behind the refined L70 verdict:

  * ALL core forwards (GEMM incl. huge-K, conv, reductions, softmax, random,
    fused-R fwd, custom grouped kernels) are cross-process BIT-IDENTICAL.
  * `mx.<arr>.at[idx].add` scatter with duplicate indices is NONDETERMINISTIC
    (atomics; not even in-process repeatable), and the take-VJP inherits this
    whenever its cotangent is not the trivial fused case.
  * The gather-based `_resize_axis_nhwc` bicubic-UP backward (the witness R
    operator's default reference path) is exactly that op => it poisoned every
    witness-trainer gradient from epoch 1 (the historical 28/28 divergence).
  * The in-tree fused-R Metal kernel (`--fused-r-kernel`, #252) replaces both
    directions with fixed-order kernels (no atomics) => the FULL witness
    trainer becomes cross-process bit-identical on GPU (measured N=10, 0/28).

Usage:
    .venv/bin/python tools/mlx_gpu_determinism_probe.py                # all cells, N=10
    .venv/bin/python tools/mlx_gpu_determinism_probe.py --n 5 --ops scatter_add_dup r_up_grad
    .venv/bin/python tools/mlx_gpu_determinism_probe.py --child <op> <device>   # internal

Authority: [macOS-MLX research-signal] — bit-IDENTITY verdicts are valid
on-device facts; NEVER a score claim (MPS/MLX never a score authority).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from tac import witness_run_artifacts as _wra  # noqa: E402  (tac is pip-install-editable; no sys.path shim needed)

OPS = (
    "random_normal",
    "elementwise",
    "matmul_square",
    "matmul_bigK",
    "gemv_bigK",
    "sum_reduce",
    "mean_axis",
    "softmax_big",
    "conv2d_s2",
    "conv2d_grouped_s2",
    "custom_grouped_backward",
    "persistence_pool",   # fused 3x3 min/max/mean pool (#212; forward, no atomics)
    "mlp_grad",
    "take_grad_dup",
    "scatter_add_dup",
    "take_axis_grad",
    "fused_r_forward",
    "fused_r_vjp",
    "r_up_grad",          # reference-R bicubic-UP backward (the divergent witness op)
    "r_full_grad",        # full reference-R backward
)


def _h(*arrays) -> str:
    import numpy as np

    m = hashlib.sha256()
    for a in arrays:
        m.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    return m.hexdigest()[:16]


def run_op(op: str, seed: int = 1234):
    import mlx.core as mx
    import numpy as np

    def _inputs(sd, shapes):
        outs = []
        for i, s in enumerate(shapes):
            outs.append(mx.random.normal(s, key=mx.random.key(sd + i), dtype=mx.float32))
        mx.eval(*outs)
        return outs

    if op == "random_normal":
        return _inputs(seed, [(1024, 1024)])
    if op == "elementwise":
        (a,) = _inputs(seed, [(1024, 1024)])
        r = mx.tanh(mx.sin(a) * mx.exp(a * 0.01) + a * a)
        mx.eval(r)
        return [r]
    if op == "matmul_square":
        a, b = _inputs(seed, [(1024, 1024), (1024, 1024)])
        r = a @ b
        mx.eval(r)
        return [r]
    if op == "matmul_bigK":
        a, b = _inputs(seed, [(32, 1 << 18), (1 << 18, 32)])
        r = a @ b
        mx.eval(r)
        return [r]
    if op == "gemv_bigK":
        a, b = _inputs(seed, [(1, 1 << 20), (1 << 20, 64)])
        r = a @ b
        mx.eval(r)
        return [r]
    if op == "sum_reduce":
        (a,) = _inputs(seed, [(1 << 24,)])
        r = mx.sum(a)
        mx.eval(r)
        return [r]
    if op == "mean_axis":
        (a,) = _inputs(seed, [(4096, 4096)])
        r = mx.mean(a, axis=0)
        mx.eval(r)
        return [r]
    if op == "softmax_big":
        (a,) = _inputs(seed, [(4096, 4096)])
        r = mx.softmax(a, axis=-1)
        mx.eval(r)
        return [r]
    if op == "conv2d_s2":
        x, w = _inputs(seed, [(4, 96, 128, 32), (64, 3, 3, 32)])
        r = mx.conv2d(x, w, stride=2, padding=1)
        mx.eval(r)
        return [r]
    if op == "conv2d_grouped_s2":
        x, w = _inputs(seed, [(4, 96, 128, 32), (32, 3, 3, 1)])
        r = mx.conv2d(x, w, stride=2, padding=1, groups=32)
        mx.eval(r)
        return [r]
    if op == "custom_grouped_backward":
        from tac.local_acceleration.metal_grouped_conv_backward import make_grouped_conv2d_nhwc

        x, w = _inputs(seed, [(2, 64, 64, 32), (32, 3, 3, 1)])
        conv = make_grouped_conv2d_nhwc(stride=2, padding=1, dilation=1, groups=32)
        gx, gw = mx.grad(lambda x_, w_: mx.sum(conv(x_, w_) ** 2), argnums=(0, 1))(x, w)
        mx.eval(gx, gw)
        return [gx, gw]
    if op == "persistence_pool":
        from tac.local_acceleration.metal_persistence_pool import pool3x3_metal

        (x,) = _inputs(seed, [(4, 384, 512)])
        r0 = pool3x3_metal(x, "min")
        r1 = pool3x3_metal(x, "max")
        r2 = pool3x3_metal(x, "mean")
        mx.eval(r0, r1, r2)
        return [r0, r1, r2]
    if op == "mlp_grad":
        x, w1, w2, w3, t = _inputs(seed, [(256, 128), (128, 512), (512, 512), (512, 8), (256, 8)])

        def loss(w1_, w2_, w3_):
            h = mx.tanh(x @ w1_)
            h = mx.tanh(h @ w2_)
            return mx.mean((h @ w3_ - t) ** 2)

        g1, g2, g3 = mx.grad(loss, argnums=(0, 1, 2))(w1, w2, w3)
        mx.eval(g1, g2, g3)
        return [g1, g2, g3]
    if op == "take_grad_dup":
        (table,) = _inputs(seed, [(64, 128)])
        idx = mx.array((np.arange(4096) % 7).astype(np.int32))
        (w,) = _inputs(seed + 99, [(128, 1)])
        gt = mx.grad(lambda t_: mx.sum((t_[idx] @ w) ** 2))(table)
        mx.eval(gt)
        return [gt]
    if op == "scatter_add_dup":
        (src,) = _inputs(seed, [(65536, 32)])
        idx = mx.array((np.arange(65536) % 13).astype(np.uint32))
        r = mx.zeros((13, 32), dtype=mx.float32).at[idx].add(src)
        mx.eval(r)
        return [r]
    if op == "take_axis_grad":
        (x,) = _inputs(seed, [(1, 96, 128, 3)])
        idx = mx.array((np.arange(874 * 4) % 96).astype(np.int32))
        gx = mx.grad(lambda x_: mx.sum(mx.take(x_, idx, axis=1) ** 2))(x)
        mx.eval(gx)
        return [gx]
    if op == "fused_r_forward":
        from tac.local_acceleration.metal_fused_r_operator import make_fused_r_roundtrip

        (x,) = _inputs(seed, [(2, 96, 128, 3)])
        x = mx.abs(x) * 60.0
        fn = make_fused_r_roundtrip(camera_hw=(874, 1164), output_hw=(384, 512))
        r = fn(x)
        mx.eval(r)
        return [r]
    if op == "fused_r_vjp":
        from tac.local_acceleration.metal_fused_r_operator import make_fused_r_roundtrip

        (x,) = _inputs(seed, [(2, 96, 128, 3)])
        x = mx.abs(x) * 60.0
        fn = make_fused_r_roundtrip(camera_hw=(874, 1164), output_hw=(384, 512))
        g = mx.grad(lambda x_: mx.sum(fn(x_) ** 2))(x)
        mx.eval(g)
        return [g]
    if op == "r_up_grad":
        from tac.local_acceleration.pr95_hnerv_mlx_training import resize_nhwc_align_corners_false

        (x,) = _inputs(seed, [(1, 96, 128, 3)])
        x = mx.abs(x) * 80.0
        g = mx.grad(lambda t: mx.sum(
            resize_nhwc_align_corners_false(t, size=(874, 1164), mode="bicubic") ** 2))(x)
        mx.eval(g)
        return [g]
    if op == "r_full_grad":
        from tac.local_acceleration.pr95_hnerv_mlx_training import (
            apply_contest_faithful_roundtrip_nhwc,
        )

        (x,) = _inputs(seed, [(1, 96, 128, 3)])
        x = mx.abs(x) * 80.0
        g = mx.grad(lambda t: mx.sum(
            apply_contest_faithful_roundtrip_nhwc(t, output_hw=(384, 512), ste_round=True) ** 2))(x)
        mx.eval(g)
        return [g]
    raise SystemExit(f"unknown op {op!r}")


def _child(op: str, device: str) -> None:
    import mlx.core as mx

    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    outs1 = run_op(op)
    h1 = _h(*outs1)
    outs2 = run_op(op)  # in-process repeat (fresh graph, same seed)
    h2 = _h(*outs2)
    print(json.dumps({"op": op, "device": device, "hash": h1,
                      "in_process_identical": h1 == h2}))


def probe_cell(op: str, device: str = "gpu", n: int = 10, arch: str = "") -> dict:
    """Run ``op`` in ``n`` separate processes; return the bit-identity verdict."""
    env = dict(os.environ)
    env.pop("MLX_METAL_GPU_ARCH", None)
    if arch:
        env["MLX_METAL_GPU_ARCH"] = arch
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO, "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    hashes, in_proc = [], []
    for _ in range(n):
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", op, device],
            env=env, capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            return {"op": op, "error": p.stderr.strip()[-400:]}
        row = json.loads(p.stdout.strip().splitlines()[-1])
        hashes.append(row["hash"])
        in_proc.append(row["in_process_identical"])
    return {
        "op": op, "device": device, "arch": arch or "default", "n": n,
        "unique_hashes": len(set(hashes)),
        "cross_process_identical": len(set(hashes)) == 1,
        "in_process_identical_all": all(in_proc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 (task #350): COMPOSITE witness-step determinism cell.
#
# The op cells above localize determinism per PRIMITIVE. This cell answers the
# COMPOSITE question the #348 doc explicitly left owed: is the FULL real
# level-set witness training step cross-process bit-identical at the n600
# self-orient config with ``--fused-r-kernel`` ON? (Op-class determinism is
# scale-invariant per the #348 localization table — huge-K matmul already ruled
# out kernel-selection-by-size — so a reduced-pairs run answers the OP-CLASS
# question definitively; the n600 run adds only the memory/self-orient-cache
# axis, re-confirmed at true scale when budget allows.) This doubles as run-1's
# launch-preflight: if the composite DIVERGES with fused-R ON, the trainer's
# gradients are untrustworthy and NO run should launch.
#
# Mechanism: spawn the REAL trainer (train_levelset_witness_realized_through_R_mlx.py)
# in N separate processes (identical seed+config), each from-scratch for a few
# epochs into a fresh scratch out-dir, then sha256 the EMA-shadow deploy npz
# (``levelset_witness_ema_mlx.npz``). Reuses the crash-resume smoke's argv
# discipline. Resumable: per-child hashes append to a JSONL so a re-invocation
# continues (foreground, chunkable — spawned long-runners die ~5min, so each
# child is a bounded subprocess.run, never backgrounded).
# ─────────────────────────────────────────────────────────────────────────────

_TRAINER = os.path.join(_REPO, "experiments", "train_levelset_witness_realized_through_R_mlx.py")


_SAFE_RUN = os.path.join(_REPO, "tools", "safe_run.py")


def _composite_trainer_argv(out_dir: str, *, num_pairs: int, epochs: int, gt_cache: str,
                            device: str, fused_r: bool, seed: int = 0,
                            rss_mb: int = 74000, timeout_s: int = 560,
                            eval_every: int | None = None,
                            extra_flags: list[str] | None = None) -> list[str]:
    """The mod32cap n600 launch config (verbatim levers) at bounded pairs/epochs, with the
    ``--fused-r-kernel`` toggle exposed. self-orient ON (the unverified composite element).
    Verdict is skipped (``--eval-every`` past the horizon) — determinism is about the WEIGHT
    bytes, not the CPU-torch verdict.

    Routed through ``tools/safe_run.py`` (the sanctioned governed-admission path): it stamps
    ``TAC_GOVERNED_ADMISSION=1`` on the child so the P0 memory-admission guard passes, runs the
    system admission gate, and RSS-caps the process group at ``rss_mb`` (memory-bounded per the
    #350 charter — the box is protected, no raw-python heavy bypass)."""
    projected_gib = max(2.0, 4.0 + num_pairs * 0.085)  # ~52 GiB at n600 self-orient; ~4 at n=2
    trainer = [
        sys.executable, _TRAINER,
        "--out-dir", out_dir,
        "--gt-cache", gt_cache,
        "--num-pairs", str(num_pairs),
        "--mlx-device", device,
        "--seed", str(seed),
        "--epochs", str(epochs),
        # default skips the (CPU-torch) verdict entirely; the exact-AB harness sets eval_every=1 to
        # capture the per-epoch d_seg history it uses to locate the divergence STEP.
        "--eval-every", str(eval_every if eval_every is not None else epochs + 999),
        "--verdict-pairs", "0",
        "--curriculum",
        "--tau-softplus-start-epoch", "300", "--tau-softplus-tau", "0.3",
        "--l7-start-epoch", "1001",
        "--w-seg", "100", "--w-pose", "0", "--score-domain-loss",
        "--mod-dim", "32", "--hidden-dim", "96", "--n-hidden", "4",
        "--activation", "hosc",
        "--hosc-beta", "1.0", "--hosc-beta-end", "4.0", "--hosc-beta-anneal", "linear",
        "--hosc-omega", "1.0", "--siren-init",
        "--softmax-temp-start", "1.0", "--softmax-temp-end", "0.05",
        "--self-orient", "--n-dir-freqs", "4", "--freq-across", "32", "--freq-along", "8",
        "--reorient-every", "50", "--max-bank-freq", "64",
        "--chroma", "--palette-anchor",
        "--eikonal-weight", "0", "--length-weight", "0.001",
        "--render-h", "384", "--render-w", "512",
        "--accum-pairs", "8", "--grad-clip", "1.0", "--ema-decay", "0.997",
        "--structured-init", "--structured-init-include-lane",
        "--ckpt-every", "1", "--stage-checkpoints", "--per-group-grad-clip",
    ]
    trainer.append("--fused-r-kernel" if fused_r else "--no-fused-r-kernel")
    if extra_flags:
        trainer.extend(extra_flags)
    return [
        sys.executable, _SAFE_RUN,
        "--rss-mb", str(rss_mb), "--timeout", str(timeout_s),
        "--projected-gib", f"{projected_gib:.1f}", "--label", f"b_det_comp_np{num_pairs}",
        "--", *trainer,
    ]


def _hash_npz(path: str) -> str:
    """sha256 of an npz's fp32 arrays in SORTED key order (npz internal order is not stable)."""
    import numpy as np

    m = hashlib.sha256()
    with np.load(path, allow_pickle=True) as z:
        for k in sorted(z.files):
            m.update(k.encode())
            a = z[k]
            if a.dtype.kind in "fiu":
                m.update(np.ascontiguousarray(a).tobytes())
            else:  # cfg scalars stored as object/str — hash their repr (deterministic)
                m.update(repr(a.tolist()).encode())
    return m.hexdigest()[:16]


def probe_composite_cell(*, num_pairs: int, epochs: int, gt_cache: str, device: str,
                         fused_r: bool, n: int, scratch: str, ledger: str,
                         per_proc_timeout: int = 570) -> dict:
    """Spawn the real trainer ``n`` times (fresh scratch dirs, identical seed+config); return the
    cross-process bit-identity verdict on the EMA deploy npz. Resumable via ``ledger`` JSONL."""
    import numpy as np  # noqa: F401  (only to fail early if the venv lacks it)

    os.makedirs(scratch, exist_ok=True)
    done: dict[int, dict] = {}
    if os.path.exists(ledger):
        with open(ledger) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("num_pairs") == num_pairs and r.get("fused_r") == fused_r:
                        done[r["idx"]] = r
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO, "src"), os.path.join(_REPO, "experiments"),
         os.path.join(_REPO, "upstream")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    for i in range(n):
        if i in done and done[i].get("hash"):
            continue
        out_dir = os.path.join(scratch, f"comp_np{num_pairs}_fr{int(fused_r)}_{i}")
        argv = _composite_trainer_argv(out_dir, num_pairs=num_pairs, epochs=epochs,
                                       gt_cache=gt_cache, device=device, fused_r=fused_r,
                                       timeout_s=max(30, per_proc_timeout - 12))
        t0 = __import__("time").time()
        try:
            p = subprocess.run(argv, env=env, cwd=_REPO, capture_output=True, text=True,
                               timeout=per_proc_timeout)
        except subprocess.TimeoutExpired:
            row = {"idx": i, "num_pairs": num_pairs, "fused_r": fused_r,
                   "error": f"timeout>{per_proc_timeout}s"}
            with open(ledger, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            done[i] = row
            continue
        ema = os.path.join(out_dir, _wra.EMA_NPZ)
        if p.returncode != 0 or not os.path.exists(ema):
            row = {"idx": i, "num_pairs": num_pairs, "fused_r": fused_r,
                   "error": (p.stderr or "no-ema")[-500:]}
        else:
            row = {"idx": i, "num_pairs": num_pairs, "fused_r": fused_r,
                   "hash": _hash_npz(ema), "wall_s": round(__import__("time").time() - t0, 1)}
        with open(ledger, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        done[i] = row
    rows = [done[i] for i in range(n) if i in done]
    hashes = [r["hash"] for r in rows if r.get("hash")]
    errs = [r for r in rows if not r.get("hash")]
    return {
        "cell": "composite_witness_step", "num_pairs": num_pairs, "epochs": epochs,
        "device": device, "fused_r": fused_r, "n_requested": n, "n_hashed": len(hashes),
        "unique_hashes": len(set(hashes)),
        "cross_process_identical": len(hashes) >= 2 and len(set(hashes)) == 1,
        "errors": errs[:3],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAFE-COMPILE region certification (#252, 2026-07-08).
#
# The op cells above localize determinism per PRIMITIVE; the composite cell
# answers the full-step question. This cell answers a THIRD question the
# determinism-first ``mx.compile`` layer needs: for each NOMINATED elementwise
# region, is ``mx.compile(region)`` BIT-IDENTICAL to the uncompiled region
# (max|Δ|=0 on real inputs) AND cross-process deterministic? It delegates to
# ``tac.mlx_safe_compile`` (the certification harness) so the probe stays the
# single certification instrument. A region ships ENABLED in the trainer only via
# a CERTIFIED manifest row — never on vibes.
# ─────────────────────────────────────────────────────────────────────────────


def probe_safe_compile_regions(region_ids: list[str] | None, *, device: str, n: int,
                               n_inputs: int, out: str | None) -> dict:
    """Certify the nominated SAFE-COMPILE regions and (optionally) write the manifest."""
    from datetime import datetime, timezone

    sys.path.insert(0, os.path.join(_REPO, "src"))
    from tac import mlx_safe_compile as sc

    ids = region_ids or sorted(sc.REGION_BUILDERS)
    man = sc.CertificationManifest(device=device,
                                   created_utc=datetime.now(timezone.utc).isoformat())
    for rid in ids:
        region = sc.get_region(rid)
        if region.classify().region_class is sc.RegionClass.UNSAFE_REDUCTION:
            cert = sc.certify_fixed_point_reduction(rid, n_determinism=n, device=device)
        else:
            cert = sc.certify_region(rid, n_inputs=n_inputs, n_determinism=n, device=device)
        man.add(cert)
    if out:
        man.save(out)
    return man.as_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", nargs=2, metavar=("OP", "DEVICE"), default=None,
                    help="internal: run one op in this process and print its hash")
    ap.add_argument("--safe-compile", action="store_true",
                    help="certify the SAFE-COMPILE regions (tac.mlx_safe_compile) on this device")
    ap.add_argument("--safe-compile-regions", nargs="*", default=None,
                    help="subset of region ids to certify (default: all registered)")
    ap.add_argument("--safe-compile-out", default=None,
                    help="write the certification manifest JSON here")
    ap.add_argument("--n", type=int, default=10, help="processes per cell (default 10)")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--arch", default="", help="MLX_METAL_GPU_ARCH override (e.g. applegpu_g15)")
    ap.add_argument("--ops", nargs="*", default=None, help=f"subset of: {', '.join(OPS)}")
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    # STAGE 0 composite cell (task #350) / launch-preflight:
    ap.add_argument("--composite", action="store_true",
                    help="run the COMPOSITE real-trainer determinism cell (n600 self-orient config)")
    ap.add_argument("--comp-pairs", type=int, default=600, help="composite: --num-pairs (default 600)")
    ap.add_argument("--comp-epochs", type=int, default=2, help="composite: from-scratch epochs")
    ap.add_argument("--comp-gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--comp-fused-r", action=argparse.BooleanOptionalAction, default=True,
                    help="composite: --fused-r-kernel (default ON — the run-1 deterministic config)")
    ap.add_argument("--comp-scratch", default=os.path.join(_REPO, "experiments", "results",
                    "b_det_composite_350"), help="composite: scratch out-dir base (durable, not /tmp)")
    ap.add_argument("--comp-timeout", type=int, default=570, help="composite: per-process seconds")
    args = ap.parse_args()
    if args.child:
        _child(args.child[0], args.child[1])
        return 0
    if args.safe_compile:
        r = probe_safe_compile_regions(args.safe_compile_regions, device=args.device,
                                       n=args.n, n_inputs=32, out=args.safe_compile_out)
        print(json.dumps(r, indent=1))
        rows = r["rows"]
        failed = [rid for rid, row in rows.items() if row["verdict"] != "CERTIFIED"]
        print(f"\n[verdict] safe-compile: {len(rows) - len(failed)}/{len(rows)} regions CERTIFIED "
              f"on {args.device}; not-certified: {failed}", file=sys.stderr)
        return 0 if not failed else 1
    if args.composite:
        gt = args.comp_gt_cache
        if not os.path.isabs(gt):
            gt = os.path.join(_REPO, gt)
        ledger = os.path.join(args.comp_scratch, "composite_hashes.jsonl")
        os.makedirs(args.comp_scratch, exist_ok=True)
        r = probe_composite_cell(
            num_pairs=args.comp_pairs, epochs=args.comp_epochs, gt_cache=gt,
            device=args.device, fused_r=args.comp_fused_r, n=args.n,
            scratch=args.comp_scratch, ledger=ledger, per_proc_timeout=args.comp_timeout)
        print(json.dumps(r, indent=1))
        if r["n_hashed"] < 2:
            print("[verdict] INCONCLUSIVE — <2 successful children (see errors); re-run to resume",
                  file=sys.stderr)
            return 2
        print(f"[verdict] composite n={r['num_pairs']} fused_r={r['fused_r']}: "
              f"{'BIT-IDENTICAL' if r['cross_process_identical'] else 'DIVERGES'} "
              f"({r['n_hashed']} procs, {r['unique_hashes']} unique hash)")
        return 0 if r["cross_process_identical"] else 1
    results = []
    for op in (args.ops or OPS):
        r = probe_cell(op, device=args.device, n=args.n, arch=args.arch)
        results.append(r)
        if not args.json:
            print(json.dumps(r), flush=True)
    if args.json:
        print(json.dumps(results, indent=1))
    bad = [r for r in results if not r.get("cross_process_identical", False)]
    if not args.json:
        print(f"\n[verdict] {len(results) - len(bad)}/{len(results)} op cells cross-process "
              f"bit-identical on {args.device}; nondeterministic: {[r['op'] for r in bad]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
