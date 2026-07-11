# SPDX-License-Identifier: MIT
"""#356 whole-step-megakernel decisive smoke (2026-07-11) — MEASURED NO-GO evidence.

Question: is ``mx.compile`` of the witness trunk+loss closure (the whole-step megakernel
candidate) EXACTLY bit-identical (fwd + bwd, fp32) to the eager multi-dispatch path, and
what wall-clock does it buy?  CPU + GPU, small-P (launch-overhead regime) and
full-frame-P=196608 (real 384x512 render regime).  $0, bounded, brief.

MEASURED VERDICT (M5 Max, macOS, MLX; run 2026-07-11 — memo
.omx/research/whole_step_megakernel_356_20260711.md):
  bit-identity FAIL everywhere: grad max|delta| 2.3e-7 (CPU fullframe) .. 2.3e-5 (CPU smallP);
  compiled determinism EXACT 0.0 (deterministic-but-DIFFERENT = fp reorder/contraction);
  speedup GPU 1.12x/1.21x (smallP/fullframe), CPU 0.79x/0.83x (SLOWER).
Both legs independently kill the mx.compile whole-step megakernel for the pointer run
(the #410 micro-batch class: not bit-identical -> cannot ride a score-faithful lineage).
Equation: witness_fp_reorder_transform_bit_identity_wall_v1.  Advisory, non-promotable;
pointer 0.19108282 [contest-CPU] UNMOVED."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src")  # repo-root invocation: .venv/bin/python experiments/megakernel_compile_bit_identity_smoke_356.py

import mlx.core as mx
import numpy as np

from tac.local_acceleration.mlx_compile_step import (
    build_representative_dseg_trunk,
    compile_loss_and_grad,
)


def make_case(P: int, seed: int = 0, mod_dim: int = 32, hidden: int = 96, n_hidden: int = 4):
    """Representative trunk + CE loss at pixel-count P (mirrors the witness op kinds)."""
    import mlx.nn as nn

    model = build_representative_dseg_trunk(
        in_feat=88, hidden_dim=hidden, n_hidden=n_hidden, n_classes=5, mod_dim=mod_dim, seed=seed
    )
    rng = np.random.default_rng(seed)
    feats = mx.array(rng.standard_normal((P, 88)).astype(np.float32))
    code = mx.array(rng.standard_normal((mod_dim,)).astype(np.float32))
    scorer_w = mx.array(rng.standard_normal((3, 5)).astype(np.float32) * 0.01)
    target = mx.array(rng.integers(0, 5, size=(P,)).astype(np.int32))
    tgt_oh = mx.eye(5)[target]

    def loss_fn(f, c):
        rgb, phi = model(f, c)
        logits = rgb @ scorer_w
        logsum = mx.logsumexp(logits, axis=-1)
        tgt = mx.sum(logits * tgt_oh, axis=-1)
        # add an eikonal-style grid term on phi to widen op coverage (finite diff + square)
        d = phi[1:] - phi[:-1]
        return mx.mean(logsum - tgt) + 0.01 * mx.mean(d * d)

    lag = nn.value_and_grad(model, loss_fn)
    return lag, (feats, code)


def flatten(tree):
    from mlx.utils import tree_flatten

    return [np.asarray(v) for _, v in tree_flatten(tree)]


def bit_delta(lag, args, runs: int = 3):
    """Exact max|Δ| eager-vs-compiled (loss + every grad leaf) + compiled determinism."""
    compiled = compile_loss_and_grad(lag, enabled=True)
    lu, gu = lag(*args)
    lc, gc = compiled(*args)
    mx.eval(lu, gu, lc, gc)
    dl = abs(float(np.asarray(lu)) - float(np.asarray(lc)))
    fu, fc = flatten(gu), flatten(gc)
    dg = max((float(np.max(np.abs(a - b))) if a.size else 0.0) for a, b in zip(fu, fc))
    # compiled determinism (repeat, exact)
    det = 0.0
    base_l, base_g = float(np.asarray(lc)), [v.copy() for v in fc]
    for _ in range(runs - 1):
        li, gi = compiled(*args)
        mx.eval(li, gi)
        det = max(det, abs(float(np.asarray(li)) - base_l))
        for r, v in zip(base_g, flatten(gi)):
            det = max(det, float(np.max(np.abs(r - v))) if r.size else 0.0)
    return dl, dg, det, compiled


def bench(fn, args, iters: int = 30, warmup: int = 5):
    for _ in range(warmup):
        l, g = fn(*args)
        mx.eval(l, g)
    t0 = time.perf_counter()
    for _ in range(iters):
        l, g = fn(*args)
        mx.eval(l, g)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/step


def run(device_name: str):
    dev = mx.gpu if device_name == "gpu" else mx.cpu
    mx.set_default_device(dev)
    out = {"device": device_name}
    for P, tag in ((4096, "smallP"), (196608, "fullframeP")):
        lag, args = make_case(P)
        dl, dg, det, compiled = bit_delta(lag, args)
        t_eager = bench(lag, args, iters=20 if P > 100000 else 50)
        t_comp = bench(compiled, args, iters=20 if P > 100000 else 50)
        out[tag] = {
            "P": P,
            "loss_max_abs_delta": dl,
            "grad_max_abs_delta": dg,
            "compiled_determinism_delta": det,
            "eager_ms_per_step": round(t_eager, 3),
            "compiled_ms_per_step": round(t_comp, 3),
            "speedup_x": round(t_eager / t_comp, 3) if t_comp > 0 else None,
            "BIT_IDENTICAL_0p0": bool(dl == 0.0 and dg == 0.0),
        }
    return out


if __name__ == "__main__":
    results = [run("cpu"), run("gpu")]
    print(json.dumps(results, indent=2))
