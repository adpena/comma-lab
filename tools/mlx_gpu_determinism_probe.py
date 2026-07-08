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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", nargs=2, metavar=("OP", "DEVICE"), default=None,
                    help="internal: run one op in this process and print its hash")
    ap.add_argument("--n", type=int, default=10, help="processes per cell (default 10)")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--arch", default="", help="MLX_METAL_GPU_ARCH override (e.g. applegpu_g15)")
    ap.add_argument("--ops", nargs="*", default=None, help=f"subset of: {', '.join(OPS)}")
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    args = ap.parse_args()
    if args.child:
        _child(args.child[0], args.child[1])
        return 0
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
