#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_dt1 (#903) — isolate WHERE MLX non-determinism enters, WITHIN the GPU path.

Established before this probe runs (MEASURED, ``.omx/research/ddm_dt1_determinism_floor_20260803.md``):
  * ``--mlx-device cpu``  -> 3/3 TR1 windows BIT-IDENTICAL (41/41 ckpt arrays, 134/134 telemetry).
  * ``--mlx-device gpu``  -> 40/41 ckpt arrays DIFFER, with or without
    ``TAC_MLX_CUSTOM_GROUPED_BACKWARD``. So the mechanism is the MLX **GPU** path and it is
    NOT the custom grouped-backward Metal kernel.

This probe splits the remaining space with two orthogonal questions, each of which
discriminates a different mechanism class:

  Q1 WITHIN-PROCESS repeat: run the identical op graph twice in ONE process on the same
     input buffers.
       differs  => the KERNEL/SCHEDULER is nondeterministic (varying reduction order,
                   atomics, split-k / threadgroup count chosen per-dispatch).
       same     => the kernel is deterministic given its inputs; the divergence must be
                   seeded by something that varies BETWEEN processes.

  Q2 CROSS-PROCESS repeat: emit a digest of the same graph so two invocations can be diffed.
       differs while Q1 is same => process-level state (JIT/kernel-variant selection,
                   buffer alignment, device state, env) picks a different-but-internally-
                   stable numeric path per process.

Each op family is probed separately (elementwise / matmul / reduction / conv2d / grouped
conv2d / softmax-CE-like) so the answer names an OP, not just "the GPU".

Reported for every family: bit-identical yes/no, max |delta|, max ULP, and the DENOMINATOR
(n arrays / n elements compared) — an empty family reports VACUOUS, never a bare pass.

score_claim=false. This is apparatus measurement only.

Usage:
    .venv/bin/python experiments/ddm_dt1_mlx_gpu_determinism_probe.py --device gpu --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any

import numpy as np


def _ulp(a: np.ndarray, b: np.ndarray) -> int:
    if a.dtype.kind != "f" or a.dtype.itemsize != 4:
        return 0
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    sign_bit = np.int64(1) << np.int64(31)
    ia = np.where(ia < 0, sign_bit - ia, ia)
    ib = np.where(ib < 0, sign_bit - ib, ib)
    d = np.abs(ia - ib)
    return int(d.max()) if d.size else 0


def _cmp(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    same = a.tobytes() == b.tobytes()
    out: dict[str, Any] = {
        "bit_identical": bool(same),
        "n_elem": int(a.size),
        "max_abs": 0.0, "max_ulp": 0, "frac_differ": 0.0,
    }
    if not same:
        d = a.astype(np.float64) - b.astype(np.float64)
        out["max_abs"] = float(np.abs(d).max())
        out["max_ulp"] = _ulp(a, b)
        out["frac_differ"] = float(np.count_nonzero(d) / max(a.size, 1))
    return out


def build_families(mx, seed: int) -> dict[str, Any]:
    """Deterministic host-side inputs (numpy, fixed seed) -> mx arrays.

    Inputs come from numpy so the probe never confounds "MLX RNG differs" with
    "MLX arithmetic differs". RNG is tested as its OWN family.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 64, 64, 32)).astype(np.float32)
    w = rng.standard_normal((32, 3, 3, 32)).astype(np.float32)
    wg = rng.standard_normal((32, 3, 3, 1)).astype(np.float32)  # depthwise, groups=32
    m1 = rng.standard_normal((512, 512)).astype(np.float32)
    m2 = rng.standard_normal((512, 512)).astype(np.float32)
    big = rng.standard_normal((1 << 20,)).astype(np.float32)
    logits = rng.standard_normal((8, 5, 96, 128)).astype(np.float32)

    X, W, WG = mx.array(x), mx.array(w), mx.array(wg)
    M1, M2, BIG, LOG = mx.array(m1), mx.array(m2), mx.array(big), mx.array(logits)

    def elementwise():
        return mx.tanh(X * 1.0001 + 0.5) * mx.sigmoid(X)

    def matmul():
        return M1 @ M2

    def reduction_sum():
        # A LARGE 1-D sum is the classic non-associative-accumulation surface.
        return mx.sum(BIG)

    def reduction_mean_axis():
        return mx.mean(X, axis=(0, 1, 2))

    def conv2d_dense():
        return mx.conv2d(X, W, stride=(1, 1), padding=(1, 1))

    def conv2d_grouped_strided():
        return mx.conv2d(X, WG, stride=(2, 2), padding=(1, 1), groups=32)

    def softmax_ce_like():
        lse = mx.logsumexp(LOG, axis=1)
        return mx.mean(lse - LOG[:, 0])

    def grad_conv_chain():
        """Forward+backward through a small conv stack: the shape the trainer actually runs."""
        def f(w_):
            y = mx.conv2d(X, w_, stride=(1, 1), padding=(1, 1))
            y = mx.maximum(y, 0.0)
            y = mx.conv2d(y, w_, stride=(2, 2), padding=(1, 1))
            return mx.mean(y * y)
        return mx.grad(f)(W)

    def mlx_rng():
        # seeded INSIDE the callable so repeat-in-process re-seeds identically; a differing
        # result here would mean mx.random itself is not reproducible from a fixed seed.
        mx.random.seed(1234)
        return mx.random.normal((256, 256))

    return {
        "elementwise": elementwise,
        "matmul": matmul,
        "reduction_sum_1d_1M": reduction_sum,
        "reduction_mean_axis": reduction_mean_axis,
        "conv2d_dense": conv2d_dense,
        "conv2d_grouped_strided": conv2d_grouped_strided,
        "softmax_ce_like": softmax_ce_like,
        "grad_conv_chain": grad_conv_chain,
        "mlx_random_normal_seeded": mlx_rng,
    }


def run(device: str, seed: int, repeats: int) -> dict[str, Any]:
    import mlx.core as mx

    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    mx.random.seed(seed)
    fams = build_families(mx, seed)

    report: dict[str, Any] = {
        "device": device, "seed": seed, "repeats": repeats,
        "platform": platform.platform(), "python": sys.version.split()[0],
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "n_families": len(fams),
        "families": {},
    }
    for name, fn in fams.items():
        vals: list[np.ndarray] = []
        for _ in range(repeats):
            v = fn()
            mx.eval(v)
            vals.append(np.array(v, copy=True))
        # every repeat is compared against repeat 0 -> denominator = repeats-1
        cmps = [_cmp(vals[0], v) for v in vals[1:]]
        n_pairs = len(cmps)
        n_differ = sum(1 for c in cmps if not c["bit_identical"])
        report["families"][name] = {
            "verdict": ("VACUOUS" if n_pairs == 0
                        else ("DIFFER" if n_differ else "IDENTICAL")),
            "n_pairs_compared": n_pairs,
            "n_pairs_differ": n_differ,
            "shape": list(vals[0].shape),
            "n_elem": int(vals[0].size),
            "max_abs": max((c["max_abs"] for c in cmps), default=0.0),
            "max_ulp": max((c["max_ulp"] for c in cmps), default=0),
            "max_frac_differ": max((c["frac_differ"] for c in cmps), default=0.0),
            # digest lets a SECOND PROCESS diff the same graph (the Q2 cross-process axis)
            "digest_repeat0": float(np.float64(vals[0].astype(np.float64).sum())),
            "bytes_sha_prefix": __import__("hashlib").sha256(
                np.ascontiguousarray(vals[0]).tobytes()).hexdigest()[:16],
        }
    n_fam_differ = sum(1 for f in report["families"].values() if f["verdict"] == "DIFFER")
    report["within_process_verdict"] = (
        "VACUOUS" if not fams else ("DIFFER" if n_fam_differ else "IDENTICAL"))
    report["n_families_differ"] = n_fam_differ
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5,
                    help="within-process repeats of each op family (>=2 to be non-vacuous)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    rep = run(args.device, args.seed, args.repeats)
    print(f"device={rep['device']} mlx={rep['mlx_version']} repeats={rep['repeats']} "
          f"families={rep['n_families']}")
    for name, f in rep["families"].items():
        print(f"  {name:<28} {f['verdict']:<10} "
              f"{f['n_pairs_differ']}/{f['n_pairs_compared']} pairs differ  "
              f"max_abs={f['max_abs']:.3e} max_ulp={f['max_ulp']} "
              f"fracdiff={f['max_frac_differ']:.4f}  sha={f['bytes_sha_prefix']}")
    print(f"WITHIN-PROCESS VERDICT: {rep['within_process_verdict']} "
          f"({rep['n_families_differ']} of {rep['n_families']} families differ)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
