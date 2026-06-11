# SPDX-License-Identifier: MIT
"""Profile the capstone score-aware training step + per-eval cost (throughput lane).

This is the reusable, $0, local-CPU profiler for the capstone carrier-A/B loop.
It answers the operator's "where does the wall-clock go?" question with a
per-section breakdown of ONE real training step (the #82 MLX-render -> torch
frozen-scorer -> pixel-cotangent -> mx.vjp loop) and ONE full per-eval pass,
plus a torch-thread sweep and a scorer-batch amortization sweep.

It runs the REAL bridge (``tac.mlx_pr95_port.score_bridge.TorchScorerBridge``)
over the REAL frozen contest scorer + REAL GT targets (``tac.score_aware_loop.
targets``) on a small seeded slice of the real video — NO synthetic fixtures
(CLAUDE.md "Synthetic-fixture-instead-of-real-input" is FORBIDDEN; the
``_slow_conv2d`` amortization curve degenerates on toy inputs). The torch CPU
scorer is the TRUSTED authority per CLAUDE.md "local CPU + MLX GPU good"; MPS is
NEVER touched.

Empirical anchors this tool was built to measure (M5 Max, torch 2.11 arm64):
  - The scorer fwd+bwd is >97% of the step. SegNet (EfficientNet-B2) forward is
    the single biggest sink; the backward through it is ~2x the forward.
  - On arm64, torch has NO mkldnn / NO MKL -> EfficientNet-B2's depthwise convs
    dispatch to ``aten::_slow_conv2d_forward`` (the naive reference kernel,
    ~85% of SegNet self-CPU). channels_last does NOT help that path.
  - The ONLY large numerics-preserving lever is BATCH AMORTIZATION: the
    ``_slow_conv2d`` per-call fixed cost amortizes over a larger scorer batch
    (per-frame cost drops ~3.5x from batch=1 to batch=16). The eval_roundtrip
    (bicubic-up 874x1164) is ~1% of the step; the MLX render/copy/permute are
    <0.1%.
  - torch threads: ~6-8 is the sweet spot on the 6-perf-core M5 Max; >=14
    thrashes (worse than 6).

Usage (quick):
    .venv/bin/python tools/profile_capstone_training_throughput.py \
        --max-pairs 8 --base-channels 16 --out .omx/tmp/throughput_quick.json

Usage (full breakdown + sweeps, the canonical profiling run):
    .venv/bin/python tools/profile_capstone_training_throughput.py \
        --max-pairs 24 --base-channels 20 \
        --thread-sweep 4,6,8,10 --batch-sweep 1,4,8,16 \
        --out .omx/research/throughput_profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _median_ms(fn: Callable[[], Any], *, repeats: int) -> float:
    xs: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs) * 1000.0


def _build_real_stack(max_pairs: int, base_channels: int, codebook_size: int, seed: int):
    """Build the REAL frozen scorer + GT targets + bundle + bridge (no fixtures)."""
    import numpy as np
    import torch  # noqa: F401  (thread config already applied by caller)

    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
    from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net

    net = load_frozen_distortion_net(device="cpu")
    seg_t, pose_t, n = build_gt_targets(net, max_pairs=max_pairs, device="cpu")
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n,
            base_channels=base_channels,
            codebook_size=codebook_size,
            carrier="vq_index",
            seed=seed,
        )
    )
    bridge = TorchScorerBridge(
        net,
        seg_t,
        pose_t,
        seg_loss_form="ce_seg_loss",
        seg_weight=100.0,
        pose_weight=1.0,
        eval_roundtrip=True,
    )
    pose_store = pose_t.float().cpu().numpy().astype(np.float32)
    bundle.set_pose_stats(pose_store.mean(0), pose_store.std(0))
    return net, bundle, bridge, pose_store, int(n)


def _section_breakdown(
    net, bundle, bridge, pose_store, n: int, *, repeats: int
) -> dict[str, float]:
    """Per-section timing of ONE real training step over all ``n`` pairs."""
    import mlx.core as mx
    import numpy as np
    import torch
    import torch.nn.functional as F

    from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
    from tac.mlx_pr95_port.score_bridge import SCORER_HW

    idx = np.arange(n)
    indices = mx.array(idx.astype(np.int32))
    pose = mx.array(pose_store[idx])
    idx_t = torch.from_numpy(idx.astype(np.int64))

    # Warmup (one full fwd+bwd so lazy graphs/allocators are hot).
    r = bundle(indices, pose=pose)
    mx.eval(r)
    bridge.loss_and_pixel_grad(r, idx_t)

    def render_and_sync() -> Any:
        rr = bundle(indices, pose=pose)
        mx.eval(rr)
        return rr

    t_render = _median_ms(render_and_sync, repeats=repeats)
    r = render_and_sync()
    np_r = np.asarray(r, dtype=np.float32)

    t_npcopy = _median_ms(lambda: np.asarray(r, dtype=np.float32), repeats=repeats)
    leaf = torch.tensor(np_r, requires_grad=True)
    b = leaf.shape[0]
    flat0 = leaf.reshape(b * 2, 3, leaf.shape[-2], leaf.shape[-1])
    if (int(flat0.shape[-2]), int(flat0.shape[-1])) != SCORER_HW:
        flat0 = F.interpolate(flat0, size=SCORER_HW, mode="bilinear", align_corners=False)
    cam_h = max(round(SCORER_HW[0] * 874 / 384), SCORER_HW[0] + 1)
    cam_w = max(round(SCORER_HW[1] * 1164 / 512), SCORER_HW[1] + 1)

    def roundtrip() -> Any:
        return apply_eval_roundtrip_during_training(
            flat0,
            simulate_uint8=True,
            simulate_resize=True,
            ste_round=True,
            target_h=cam_h,
            target_w=cam_w,
        )

    t_rt = _median_ms(roundtrip, repeats=repeats)
    flat = roundtrip()
    bchw = flat.reshape(b, 2, 3, SCORER_HW[0], SCORER_HW[1])

    def permute() -> Any:
        return bchw.permute(0, 1, 3, 4, 2).contiguous()

    t_perm = _median_ms(permute, repeats=repeats)
    bhwc = permute()

    def prep() -> Any:
        return net.preprocess_input(bhwc)

    t_prep = _median_ms(prep, repeats=repeats)
    posenet_in, segnet_in = prep()

    def seg_fwd() -> Any:
        with torch.no_grad():
            return net.segnet(segnet_in)

    t_segf = _median_ms(seg_fwd, repeats=repeats)

    def pose_fwd() -> Any:
        with torch.no_grad():
            return net.posenet(posenet_in)

    t_posef = _median_ms(pose_fwd, repeats=repeats)

    def full_step() -> Any:
        rr = bundle(indices, pose=pose)
        mx.eval(rr)
        return bridge.loss_and_pixel_grad(rr, idx_t)

    t_full = _median_ms(full_step, repeats=repeats)

    fwd_subtotal = t_rt + t_perm + t_prep + t_segf + t_posef
    backward_est = max(t_full - t_render - t_npcopy - fwd_subtotal, 0.0)
    return {
        "mlx_render_and_sync_ms": round(t_render, 2),
        "np_copy_render_to_leaf_ms": round(t_npcopy, 2),
        "eval_roundtrip_bicubic_ms": round(t_rt, 2),
        "permute_contiguous_ms": round(t_perm, 2),
        "preprocess_input_ms": round(t_prep, 2),
        "segnet_forward_ms": round(t_segf, 2),
        "posenet_forward_ms": round(t_posef, 2),
        "forward_subtotal_ms": round(fwd_subtotal, 2),
        "backward_estimate_ms": round(backward_est, 2),
        "full_step_fwd_bwd_ms": round(t_full, 2),
        "scorer_fwd_bwd_fraction": round(
            (t_segf + t_posef + backward_est) / max(t_full, 1e-9), 4
        ),
    }


def _eval_pass_cost(
    net, bundle, bridge, pose_store, n: int, eval_batch: int, *, repeats: int
) -> dict[str, float]:
    """Cost of ONE full per-eval pass (exact_d_seg + exact_d_pose) over all pairs.

    Measures both the SEPARATE path (the current trainer: d_seg loop then d_pose
    loop, two renders + two preprocesses) AND a FUSED path (one render +
    preprocess per batch, SegNet and PoseNet on the same batch). The fused path
    is what ``TorchScorerBridge.fused_d_seg_d_pose`` lands; the savings are the
    avoided second render + preprocess (small relative to the SegNet forward,
    but real and numerics-identical for d_seg / numerically-equivalent for
    d_pose mean).
    """
    import mlx.core as mx
    import numpy as np
    import torch

    def render(idx_np):
        rr = bundle(mx.array(idx_np.astype(np.int32)), pose=mx.array(pose_store[idx_np]))
        mx.eval(rr)
        return rr

    # warmup
    render(np.arange(min(eval_batch, n)))

    def separate():
        for s in range(0, n, eval_batch):
            ix = np.arange(s, min(s + eval_batch, n))
            r = render(ix)
            bridge.exact_d_seg(r, torch.from_numpy(ix.astype(np.int64)))
        for s in range(0, n, eval_batch):
            ix = np.arange(s, min(s + eval_batch, n))
            r = render(ix)
            bridge.exact_d_pose(r, torch.from_numpy(ix.astype(np.int64)))

    def fused():
        for s in range(0, n, eval_batch):
            ix = np.arange(s, min(s + eval_batch, n))
            r = render(ix)
            bridge.fused_d_seg_d_pose(r, torch.from_numpy(ix.astype(np.int64)))

    has_fused = hasattr(bridge, "fused_d_seg_d_pose")
    t_sep = _median_ms(separate, repeats=repeats)
    t_fused = _median_ms(fused, repeats=repeats) if has_fused else float("nan")
    out = {
        "eval_batch": eval_batch,
        "separate_d_seg_then_d_pose_ms": round(t_sep, 1),
    }
    if has_fused:
        out["fused_d_seg_d_pose_ms"] = round(t_fused, 1)
        out["fused_speedup"] = round(t_sep / max(t_fused, 1e-9), 3)
    return out


def _thread_sweep(max_pairs, base_channels, codebook_size, seed, threads, repeats):
    """Re-import torch per thread setting (set BEFORE torch parallel init)."""
    import subprocess
    import sys

    results = []
    reps = max(repeats, 2)
    for th in threads:
        snippet = (
            f"import os; os.environ['OMP_NUM_THREADS']=str({th})\n"
            f"import torch; torch.set_num_threads({th})\n"
            "import numpy as np, time, statistics, mlx.core as mx\n"
            "from tools.profile_capstone_training_throughput import _build_real_stack\n"
            f"net,bundle,bridge,ps,n=_build_real_stack({max_pairs},{base_channels},"
            f"{codebook_size},{seed})\n"
            "idx=np.arange(n); indices=mx.array(idx.astype(np.int32)); pose=mx.array(ps[idx])\n"
            "idx_t=torch.from_numpy(idx.astype(np.int64))\n"
            "r=bundle(indices,pose=pose); mx.eval(r); bridge.loss_and_pixel_grad(r,idx_t)\n"
            "xs=[]\n"
            f"for _ in range({reps}):\n"
            "    r=bundle(indices,pose=pose); mx.eval(r)\n"
            "    t0=time.perf_counter(); bridge.loss_and_pixel_grad(r,idx_t); "
            "xs.append(time.perf_counter()-t0)\n"
            f"print('RESULT', {th}, statistics.median(xs)*1000)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        ms = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT"):
                ms = float(line.split()[2])
        results.append({"threads": th, "full_step_ms": round(ms, 1) if ms else None})
    return results


def _batch_sweep(net, bundle_fn, bridge, pose_store, n, batches, repeats):
    """SegNet-forward per-frame cost vs scorer batch size (the amortization curve)."""
    import mlx.core as mx
    import numpy as np
    import torch

    from tac.mlx_pr95_port.score_bridge import SCORER_HW

    results = []
    for npairs in batches:
        npairs = min(npairs, n)
        idx = np.arange(npairs)
        bundle = bundle_fn()
        bundle.set_pose_stats(pose_store.mean(0), pose_store.std(0))
        r = bundle(mx.array(idx.astype(np.int32)), pose=mx.array(pose_store[idx]))
        mx.eval(r)
        np_r = np.asarray(r, dtype=np.float32)
        leaf = torch.tensor(np_r)
        b = leaf.shape[0]
        flat = leaf.reshape(b * 2, 3, SCORER_HW[0], SCORER_HW[1])
        bchw = flat.reshape(b, 2, 3, SCORER_HW[0], SCORER_HW[1])
        bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
        _, seg_in = net.preprocess_input(bhwc)

        def seg_fwd(seg_in=seg_in):  # bind per-iteration (no loop-closure capture)
            with torch.no_grad():
                net.segnet(seg_in)

        seg_fwd()  # warmup
        ms = _median_ms(seg_fwd, repeats=max(repeats, 2))
        nframes = npairs * 2
        results.append(
            {
                "pairs": npairs,
                "frames": nframes,
                "segnet_fwd_ms": round(ms, 1),
                "per_frame_ms": round(ms / nframes, 1),
            }
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument(
        "--thread-sweep",
        type=str,
        default="",
        help="Comma list of torch thread counts to sweep the full step over "
        "(spawns a fresh process per count; e.g. '4,6,8,10'). Empty = skip.",
    )
    ap.add_argument(
        "--batch-sweep",
        type=str,
        default="",
        help="Comma list of scorer batch sizes for the SegNet amortization curve "
        "(e.g. '1,4,8,16'). Empty = skip.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Pin torch threads for the section breakdown (default: torch default).",
    )
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.threads is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        import torch

        torch.set_num_threads(int(args.threads))

    import torch

    t_start = time.time()
    net, bundle, bridge, pose_store, n = _build_real_stack(
        args.max_pairs, args.base_channels, args.codebook_size, args.seed
    )

    report: dict[str, Any] = {
        "axis": "[macOS-CPU advisory]",
        "tool": "profile_capstone_training_throughput",
        "n_pairs": n,
        "base_channels": args.base_channels,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "torch_mkl_available": bool(torch.backends.mkl.is_available()),
    }

    print(f"[profile] n={n} base_ch={args.base_channels} "
          f"threads={torch.get_num_threads()} "
          f"mkldnn={torch.backends.mkldnn.is_available()} "
          f"mkl={torch.backends.mkl.is_available()}", flush=True)

    report["section_breakdown"] = _section_breakdown(
        net, bundle, bridge, pose_store, n, repeats=args.repeats
    )
    print("[profile] section breakdown:", flush=True)
    for k, v in report["section_breakdown"].items():
        print(f"    {k:34s} {v}", flush=True)

    report["eval_pass_cost"] = _eval_pass_cost(
        net, bundle, bridge, pose_store, n, args.eval_batch, repeats=args.repeats
    )
    print(f"[profile] eval pass cost: {report['eval_pass_cost']}", flush=True)

    if args.batch_sweep:
        batches = [int(x) for x in args.batch_sweep.split(",") if x.strip()]
        from tac.capstone_vq_nerv.vq_nerv_bundle import (
            CapstoneVqNervBundle,
            CapstoneVqNervConfig,
        )

        def bundle_fn():
            return CapstoneVqNervBundle(
                CapstoneVqNervConfig(
                    num_pairs=n,
                    base_channels=args.base_channels,
                    codebook_size=args.codebook_size,
                    carrier="vq_index",
                    seed=args.seed,
                )
            )

        report["batch_amortization_sweep"] = _batch_sweep(
            net, bundle_fn, bridge, pose_store, n, batches, args.repeats
        )
        print("[profile] SegNet batch amortization (per_frame_ms drops with batch):",
              flush=True)
        for row in report["batch_amortization_sweep"]:
            print(f"    {row}", flush=True)

    if args.thread_sweep:
        threads = [int(x) for x in args.thread_sweep.split(",") if x.strip()]
        report["thread_sweep"] = _thread_sweep(
            args.max_pairs, args.base_channels, args.codebook_size, args.seed,
            threads, args.repeats,
        )
        print("[profile] full-step thread sweep:", flush=True)
        for row in report["thread_sweep"]:
            print(f"    {row}", flush=True)

    report["wall_s"] = round(time.time() - t_start, 1)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"[profile] wrote {out_path}", flush=True)
    else:
        print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
