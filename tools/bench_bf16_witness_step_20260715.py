# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""M5-unconstrained constraint-1 $0 gate (charter item 2, MVP-first): does bf16/fp16
COMPUTE speed up the REAL witness value_and_grad step on MLX M5 Max at all?

Falsifiable prediction: if the step is memory-bandwidth-bound, half-width compute
gives >1.3x; if it is occupancy/launch-bound (many small kernels), ~1.0x and the
bf16 lever is NOT worth building (honest negative, ledger row).

Casts the WITNESS module params (+feats) to the low dtype; scorers stay fp32
(mixed promotes at the boundary). Also reports gradient-quality: rel-err of the
low-dtype grads (upcast) vs the fp32 grads — the n24 gate only matters if the
speedup exists. [macOS-MLX advisory] throughput-only; NO score claim.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path("/Users/adpena/Projects/pact")
for p in (REPO / "src", REPO / "experiments", REPO / "upstream"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_map  # noqa: E402

import tac.local_acceleration.mlx_scorer_adapters as A  # noqa: E402
import train_witness_realized_through_R_mlx as T  # noqa: E402

RENDER_H, RENDER_W = 384, 512
ITERS, WARM = 20, 5


def timeit(fn, n=ITERS, warm=WARM):
    for _ in range(warm):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def grads_flat(g):
    return {k: np.asarray(v.astype(mx.float32)) for k, v in tree_flatten(g)}


def main() -> int:
    out = {"axis": "[macOS-MLX advisory] throughput-only; promotable=False",
           "render": [RENDER_H, RENDER_W], "iters": ITERS}
    with A.temporary_mlx_device("gpu"):
        adapter = A.load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        rng = np.random.default_rng(0)
        lstar_oh = mx.array(np.eye(5, dtype=np.float32)[rng.integers(0, 5, size=(T.SEG_H, T.SEG_W))][None])
        margin = mx.array(rng.uniform(0, 1, size=(T.SEG_H, T.SEG_W)).astype(np.float32))
        pose_tgt = mx.array(rng.uniform(-1, 1, size=(6,)).astype(np.float32))
        w_seg, w_pose, hinge, mtgt = mx.array(100.0), mx.array(1.0), mx.array(0.0), mx.array(1.0)
        loss_fn = T.make_loss_fn(adapter, RENDER_H, RENDER_W, score_domain=True,
                                 pose_eps=1e-2, seg_loss="ce")

        ref_grads = None
        for dt_name, dt in (("fp32", mx.float32), ("bf16", mx.bfloat16), ("fp16", mx.float16)):
            mx.random.seed(0)
            model = T.build_witness_module(num_pairs=1, n_fourier=24, hidden_dim=128, n_hidden=4,
                                           mod_dim=48, fourier_sigma=8.0, activation="relu",
                                           chroma=True, siren_init=False)
            mx.eval(model.parameters())
            if dt is not mx.float32:
                model.update(tree_map(lambda p: p.astype(dt), model.parameters()))
                mx.eval(model.parameters())
            coords = mx.array(T._build_render_coords(RENDER_H, RENDER_W))
            feats = model.build_feats(coords)
            if dt is not mx.float32:
                feats = feats.astype(dt)
            mx.eval(feats)
            vg = nn.value_and_grad(model, loss_fn)

            def step():
                l, g = vg(model, feats, 0, 1, lstar_oh, margin, pose_tgt,
                          w_seg, w_pose, hinge, mtgt)
                mx.eval(l, g)
                return l, g

            try:
                l, g = step()
            except Exception as exc:
                out[dt_name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            ms = timeit(step)
            row = {"value_and_grad_ms": round(ms, 3), "loss": float(mx.array(l).astype(mx.float32))}
            gf = grads_flat(g)
            if dt_name == "fp32":
                ref_grads = gf
            elif ref_grads is not None:
                rel = []
                for k, v in gf.items():
                    r = ref_grads.get(k)
                    if r is None or r.shape != v.shape:
                        continue
                    denom = np.abs(r).max()
                    if denom > 0:
                        rel.append(float(np.abs(v - r).max() / denom))
                row["grad_relmax_vs_fp32"] = round(max(rel), 6) if rel else None
            out[dt_name] = row

        base = out.get("fp32", {}).get("value_and_grad_ms")
        for low in ("bf16", "fp16"):
            low_ms = out.get(low, {}).get("value_and_grad_ms")
            if base and low_ms:
                out[f"{low}_speedup_x"] = round(base / low_ms, 3)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
