#!/usr/bin/env python3
"""Per-pair TRAINING-step wall-clock profiler for the witness/level-set
render-through-R vehicle (``experiments/train_witness_realized_through_R_mlx.py``
+ ``train_levelset_witness_realized_through_R_mlx.py``).

DECOMPOSES one ``value_and_grad`` pair-step into components and A/B's the already-
built custom Metal grouped-conv backward (``TAC_MLX_CUSTOM_GROUPED_BACKWARD``):

  * render_through_R forward   (witness MLP fwd + contest-exact R: bicubic-up ->
    uint8-STE @ camera -> bilinear-down)
  * frozen MLX SegNet forward
  * frozen MLX PoseNet forward
  * full loss forward (2 renders + seg + pose)
  * full value_and_grad (fwd + bwd) == the per-pair step

It also reports the MLX-GPU DETERMINISM FLOOR (render/loss forward run-to-run +
gradient run-to-run) so any future "exact-preserving speedup" can be judged
against the real bit-reproducibility limit of this hardware.

THROUGHPUT TOOL ONLY — false authority. Random GT targets (shapes faithful); the
render+R+scorer COMPUTE is value-independent so wall-clock is faithful. No score,
frontier, promotion, or kill claim is produced (CLAUDE.md "MPS/MLX NEVER authority"
+ "NO FAKE"). Use it to (a) verify the ~18x custom-backward speedup, (b) confirm a
render/R/grad-step change did NOT perturb the FORWARD (loss bit-identical), and
(c) re-profile after any change.

Usage::

    .venv/bin/python tools/profile_witness_through_R_step_throughput.py \\
        --render-h 192 --render-w 256 --iters 30 --backward both \\
        --out .omx/research/witness_step_throughput_<utc>.json

Disk hygiene: writes only a small JSON manifest (no large artifacts); refuses /tmp
durable paths per AGENTS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src"), str(REPO / "upstream")):
    if p not in sys.path:
        sys.path.insert(0, p)

FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "evidence_grade": "macOS-MLX research-signal (throughput only)",
}


def _utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _timeit(fn, n: int, warm: int, mx) -> float:
    for _ in range(warm):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0  # ms


def _profile_one(*, custom_backward: bool, render_h: int, render_w: int,
                 iters: int, warm: int) -> dict:
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1" if custom_backward else "0"
    import importlib

    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    import tac.local_acceleration.mlx_scorer_adapters as A
    importlib.reload(A)  # re-read the env gate at adapter-build time
    import experiments.train_witness_realized_through_R_mlx as T
    from tac.local_acceleration.metal_grouped_conv_backward import (
        metal_grouped_conv2d_backend_available,
    )

    out: dict = {
        "custom_backward_requested": custom_backward,
        "metal_grouped_backend_available": bool(metal_grouped_conv2d_backend_available()),
    }
    with A.temporary_mlx_device("gpu"):
        adapter = A.load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        mx.random.seed(0)
        model = T.build_witness_module(num_pairs=1, n_fourier=24, hidden_dim=128, n_hidden=4,
                                       mod_dim=48, fourier_sigma=8.0, activation="relu",
                                       chroma=True, siren_init=False)
        mx.eval(model.parameters())
        coords = mx.array(T._build_render_coords(render_h, render_w))
        feats = model.build_feats(coords)
        mx.eval(feats)
        rng = np.random.default_rng(0)
        lstar_oh = mx.array(np.eye(5, dtype=np.float32)[rng.integers(0, 5, size=(T.SEG_H, T.SEG_W))][None])
        margin = mx.array(rng.uniform(0, 1, size=(T.SEG_H, T.SEG_W)).astype(np.float32))
        pose_tgt = mx.array(rng.uniform(-1, 1, size=(6,)).astype(np.float32))
        w_seg, w_pose, hinge, mtgt = mx.array(100.0), mx.array(1.0), mx.array(0.0), mx.array(1.0)

        def render():
            f = T.render_through_R_mlx(model, feats, 0, render_h, render_w)
            mx.eval(f)
            return f

        f1 = render()
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx
        pair = mx.stack([f1[0], f1[0]], axis=0)[None]
        yuv = rgb_to_yuv6_mlx(pair)
        b, t, h2, w2, c6 = yuv.shape
        yuv_nhwc = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (b, h2, w2, t * c6))
        mx.eval(yuv_nhwc)

        loss_fn = T.make_loss_fn(adapter, render_h, render_w, score_domain=True, pose_eps=1e-2, seg_loss="ce")
        vg = nn.value_and_grad(model, loss_fn)

        def loss_fwd():
            l = loss_fn(model, feats, 0, 1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt)
            mx.eval(l)
            return l

        def step():
            l, g = vg(model, feats, 0, 1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt)
            mx.eval(l, g)
            return l, g

        comp = {
            "render_through_R_fwd": round(_timeit(render, iters, warm, mx), 3),
            "segnet_fwd": round(_timeit(lambda: (mx.eval(adapter.segnet(f1)), None)[1], iters, warm, mx), 3),
            "posenet_fwd": round(_timeit(lambda: (mx.eval(adapter.posenet(yuv_nhwc)["pose"]), None)[1], iters, warm, mx), 3),
            "loss_FWD_per_pair": round(_timeit(loss_fwd, iters, warm, mx), 3),
            "value_and_grad_per_pair": round(_timeit(step, iters, warm, mx), 3),
        }
        out["ms_per_call"] = comp

        # determinism floor (forward must be 0.0; gradient is the GPU nondeterminism band)
        r1, r2 = render(), render()
        l1, g1 = step()
        l2, g2 = step()
        gd = max(float(mx.max(mx.abs(a - b)))
                 for (_, a), (_, b) in zip(tree_flatten(g1), tree_flatten(g2)))
        out["determinism"] = {
            "render_fwd_runtorun_maxabs": float(mx.max(mx.abs(r1 - r2))),
            "loss_fwd_runtorun_diff": abs(float(l2) - float(l1)),
            "grad_runtorun_maxabs": gd,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render-h", type=int, default=192)
    ap.add_argument("--render-w", type=int, default=256)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--backward", choices=["off", "on", "both"], default="both",
                    help="profile reference backward (off), custom Metal backward (on), or both for the A/B speedup")
    ap.add_argument("--out", type=str, default=None, help="JSON manifest path (refuses /tmp)")
    args = ap.parse_args(argv)

    manifest: dict = {
        "schema": "witness_through_R_step_throughput_v1",
        "utc": _utc(),
        "render": [args.render_h, args.render_w],
        "iters": args.iters,
        "configs": {},
        **FALSE_AUTHORITY,
    }
    if args.backward in ("off", "both"):
        manifest["configs"]["reference_backward"] = _profile_one(
            custom_backward=False, render_h=args.render_h, render_w=args.render_w,
            iters=args.iters, warm=args.warm)
    if args.backward in ("on", "both"):
        manifest["configs"]["custom_metal_backward"] = _profile_one(
            custom_backward=True, render_h=args.render_h, render_w=args.render_w,
            iters=args.iters, warm=args.warm)
    if args.backward == "both":
        off = manifest["configs"]["reference_backward"]["ms_per_call"]["value_and_grad_per_pair"]
        on = manifest["configs"]["custom_metal_backward"]["ms_per_call"]["value_and_grad_per_pair"]
        manifest["custom_backward_step_speedup_factor"] = round(off / on, 2) if on > 0 else None

    print(json.dumps(manifest, indent=2))
    if args.out:
        out_path = Path(args.out)
        if "/tmp" in str(out_path) or str(out_path).startswith("/private/tmp"):
            raise SystemExit(f"refusing /tmp durable evidence path: {out_path} (AGENTS.md)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2))
        print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
