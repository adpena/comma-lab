#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Per-stage MLX-GPU scorer breakdown — localize WHY the MLX port is slower.

Companion of ``bench_mlx_vs_mps_scorer_ceiling.py``. The head-to-head answers
"which backend is faster"; THIS probe answers "WHERE the MLX time goes" by timing
the two scorer heads (SegNet / PoseNet) forward and forward+backward SEPARATELY on
MLX-GPU, in both the DEFAULT (reference strided-grouped conv) and CUSTOM (Metal
backward kernel) configurations. It also reports how many strided-grouped Conv2d
layers each head routes through the slow fixed-order reference accumulator (the
suspected MLX bottleneck per the adapter dispatch + the 2026-06-09 throughput memo
which found mx.compile=0 across the adapter).

Authority: ``[macOS-MLX advisory]``. No score claim. Concurrent training load noted
by the sister head-to-head script; this is a pure relative per-stage breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = (
    "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
)


def _count_strided_grouped_convs(dist) -> dict:
    """Count Conv2d layers that hit the slow reference path (groups>1 AND stride!=1)
    in each torch head — the layers the MLX dispatch routes to the manual
    fixed-order accumulator (or the custom Metal kernel when enabled)."""
    import torch

    def _walk(mod):
        n_strided_grouped = 0
        n_total_conv = 0
        for m in mod.modules():
            if isinstance(m, torch.nn.Conv2d):
                n_total_conv += 1
                s = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
                if int(m.groups) > 1 and tuple(s) != (1, 1):
                    n_strided_grouped += 1
        return {"strided_grouped": n_strided_grouped, "total_conv2d": n_total_conv}

    return {"segnet": _walk(dist.segnet), "posenet": _walk(dist.posenet)}


def _time(fn, *, warmup, iters):
    import mlx.core as mx

    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    ts = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return round(1000 * ts[len(ts) // 2], 2)


def _run(pose_np, seg_np, *, custom: bool, warmup, iters):
    import mlx.core as mx

    # #224 Wave D FIX: the reference (default) column MUST force the env to "0", NOT pop it. The
    # adapter reads ``os.environ.get("TAC_MLX_CUSTOM_GROUPED_BACKWARD", "1")`` with DEFAULT "1"
    # (mlx_scorer_adapters.py ~L1142), so popping the var leaves the custom path ENABLED and BOTH
    # columns silently run the custom Metal backward — the "ref" numbers were never the reference.
    # Setting "0" makes the get() return "0" (not in {"1","true","True"}) => the true reference path.
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1" if custom else "0"

    from tac.local_acceleration.mlx_scorer_adapters import (
        nchw_to_nhwc,
        temporary_mlx_device,
        torch_distortion_net_to_mlx,
    )
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    dist = load_frozen_distortion_net(upstream_dir="upstream", device="cpu")
    with temporary_mlx_device("gpu"):
        adapter = torch_distortion_net_to_mlx(dist)
    seg_nhwc = mx.array(nchw_to_nhwc(seg_np))
    pose_nhwc = mx.array(nchw_to_nhwc(pose_np))

    out = {}
    with temporary_mlx_device("gpu"):
        # SegNet
        out["segnet_fwd"] = _time(
            lambda: adapter.segnet(seg_nhwc), warmup=warmup, iters=iters
        )

        def seg_loss(x):
            return (adapter.segnet(x).astype(mx.float32) ** 2).mean()

        out["segnet_fwd_bwd"] = _time(
            lambda: mx.value_and_grad(seg_loss)(seg_nhwc), warmup=warmup, iters=iters
        )
        # PoseNet
        out["posenet_fwd"] = _time(
            lambda: adapter.posenet(pose_nhwc)["pose"], warmup=warmup, iters=iters
        )

        def pose_loss(x):
            return (adapter.posenet(x)["pose"].astype(mx.float32) ** 2).mean()

        out["posenet_fwd_bwd"] = _time(
            lambda: mx.value_and_grad(pose_loss)(pose_nhwc), warmup=warmup, iters=iters
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    from tac.local_acceleration.mlx_scorer_response import load_scorer_input_cache
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    cache = load_scorer_input_cache(args.cache_dir)
    seg_np = np.asarray(cache.segnet_last_rgb[: args.batch], dtype=np.float32)
    pose_np = np.asarray(cache.posenet_yuv6_pair[: args.batch], dtype=np.float32)

    dist = load_frozen_distortion_net(upstream_dir="upstream", device="cpu")
    conv_counts = _count_strided_grouped_convs(dist)
    print("[conv-routing] strided-grouped (slow ref/custom path) counts:", flush=True)
    print(f"   segnet  {conv_counts['segnet']}", flush=True)
    print(f"   posenet {conv_counts['posenet']}", flush=True)

    default = _run(pose_np, seg_np, custom=False, warmup=args.warmup, iters=args.iters)
    custom = _run(pose_np, seg_np, custom=True, warmup=args.warmup, iters=args.iters)

    print(f"\n[per-stage ms/step  B={args.batch}]  default | custom-backward", flush=True)
    for k in ("segnet_fwd", "segnet_fwd_bwd", "posenet_fwd", "posenet_fwd_bwd"):
        print(f"   {k:18s} {default[k]:9.2f} | {custom[k]:9.2f}", flush=True)

    payload = {
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "evidence_grade": "macOS-MLX advisory; torch-CPU is the only authority",
        "score_claim": False,
        "promotable": False,
        "batch": args.batch,
        "conv_routing": conv_counts,
        "default": default,
        "custom_backward": custom,
    }
    out_json = args.out_json or (
        REPO / ".omx/research"
        / f"mlx_scorer_stage_breakdown_{payload['stamp']}.json"
    )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"\n[json] {out_json}", flush=True)


if __name__ == "__main__":
    main()
