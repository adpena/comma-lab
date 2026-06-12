"""Profile the MLX SegNet backward: how much is the Python-loop strided-grouped
fallback vs the native-MLX backward floor?

This sizes the prize for a custom Metal depthwise/grouped backward kernel.

Authority note: torch-CPU exact ``modules.py`` is the d_seg authority. EVERY
number here is `[macOS-MLX research-signal]` / `[macOS-CPU advisory]` throughput
only. NO score claim. NO MPS.

Method:
  1. Build the real DistortionNet (frozen SegNet), enumerate every Conv2d in the
     SegNet encoder+decoder, classify which hit the strided-grouped reference
     fallback (groups>1 AND stride!=1).
  2. Time the full MLX SegNet forward + VJP (value_and_grad through the render
     cotangent) on a real-shaped input.
  3. Monkeypatch ``mlx_reference_conv2d_nhwc`` to count calls + accumulate
     wall-time, isolating the Python-loop cost inside the full backward.
  4. Micro-benchmark a single strided-depthwise layer (reference fwd+bwd) vs an
     equivalent native-MLX conv fwd+bwd at the SAME shape, to size the per-layer
     speedup ceiling a custom kernel could capture.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "upstream"


def _load_distortion_net() -> Any:
    import sys

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import torch
    from modules import DistortionNet  # type: ignore

    net = DistortionNet().eval()
    net.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        device="cpu",
    )
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _pair(v: Any) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def enumerate_conv_layers(torch_segnet: Any) -> list[dict[str, Any]]:
    """Walk the torch SegNet, return per-Conv2d metadata + fallback classification."""
    import torch.nn as nn

    rows: list[dict[str, Any]] = []
    for name, mod in torch_segnet.named_modules():
        if isinstance(mod, nn.Conv2d):
            groups = int(mod.groups)
            stride = _pair(mod.stride)
            is_strided_grouped = groups > 1 and stride != (1, 1)
            is_depthwise = groups == int(mod.in_channels) and groups > 1
            rows.append(
                {
                    "name": name,
                    "in_channels": int(mod.in_channels),
                    "out_channels": int(mod.out_channels),
                    "kernel": _pair(mod.kernel_size),
                    "stride": stride,
                    "padding": _pair(mod.padding),
                    "groups": groups,
                    "is_depthwise": is_depthwise,
                    "hits_strided_grouped_fallback": is_strided_grouped,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4, help="SegNet batch (frame-1 RGB)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--out-json",
        default=str(REPO / "experiments" / "results" / "mlx_segnet_backward_profile.json"),
    )
    args = ap.parse_args()

    import mlx.core as mx

    from tac.local_acceleration import mlx_scorer_adapters as A

    net = _load_distortion_net()
    segnet = net.segnet

    conv_rows = enumerate_conv_layers(segnet)
    n_fallback = sum(1 for r in conv_rows if r["hits_strided_grouped_fallback"])
    print(f"[enum] {len(conv_rows)} Conv2d layers; {n_fallback} hit strided-grouped fallback")
    for r in conv_rows:
        if r["hits_strided_grouped_fallback"]:
            print(
                f"   FALLBACK {r['name']}: in={r['in_channels']} out={r['out_channels']} "
                f"k={r['kernel']} s={r['stride']} g={r['groups']} dw={r['is_depthwise']}"
            )

    # Build the MLX SegNet adapter.
    mlx_segnet = A.torch_segnet_to_mlx(segnet)

    # SegNet input is frame-1 RGB resized to (384, 512) -> the model input
    # (segnet_model_input_size = (512, 384) WxH per modules.py). NCHW in torch,
    # but the MLX adapter takes NHWC. The adapter's __call__ contract:
    # check what shape MLXSegNetAdapter expects.
    B = args.batch
    H, W = 384, 512  # model input HxW
    rng = np.random.default_rng(0)
    x_nhwc = mx.array(rng.standard_normal((B, H, W, 3)).astype(np.float32))

    # --- Instrument the fallback: count calls + accumulate time ---
    orig_ref = A.mlx_reference_conv2d_nhwc
    fallback_stats = {"calls": 0, "wall_s": 0.0, "shapes": []}

    def _instrumented(x, weight, bias=None, **kw):
        t0 = time.perf_counter()
        out = orig_ref(x, weight, bias, **kw)
        mx.eval(out)  # force materialization to time it honestly
        dt = time.perf_counter() - t0
        fallback_stats["calls"] += 1
        fallback_stats["wall_s"] += dt
        if len(fallback_stats["shapes"]) < 12:
            fallback_stats["shapes"].append(
                {
                    "x": list(x.shape),
                    "w": list(weight.shape),
                    "stride": kw.get("stride"),
                    "groups": kw.get("groups"),
                    "dt_ms": round(dt * 1e3, 3),
                }
            )
        return out

    # --- Forward timing (full SegNet) ---
    def fwd_loss(x):
        out = mlx_segnet(x)
        # scalar surrogate loss so VJP runs through the whole net
        return mx.sum(out * out)

    # warm
    _ = fwd_loss(x_nhwc)
    mx.eval(_)

    # full-forward time (no instrumentation)
    fwd_times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = fwd_loss(x_nhwc)
        mx.eval(out)
        fwd_times.append(time.perf_counter() - t0)
    fwd_s = float(np.median(fwd_times))
    print(f"[fwd] full SegNet forward: {fwd_s*1e3:.1f} ms (B={B})")

    # full-forward time WITH fallback instrumentation
    A.mlx_reference_conv2d_nhwc = _instrumented
    # also patch the symbol the adapter calls via module global
    try:
        fallback_stats_fwd = {"calls": 0, "wall_s": 0.0, "shapes": []}
        fallback_stats.update(fallback_stats_fwd)
        t0 = time.perf_counter()
        out = fwd_loss(x_nhwc)
        mx.eval(out)
        fwd_instr_s = time.perf_counter() - t0
        fwd_fallback_calls = fallback_stats["calls"]
        fwd_fallback_s = fallback_stats["wall_s"]
        print(
            f"[fwd-instr] {fwd_instr_s*1e3:.1f} ms; fallback {fwd_fallback_calls} calls, "
            f"{fwd_fallback_s*1e3:.1f} ms ({100*fwd_fallback_s/max(fwd_instr_s,1e-9):.1f}% of fwd)"
        )
    finally:
        A.mlx_reference_conv2d_nhwc = orig_ref

    # --- Backward (VJP) timing: full SegNet ---
    grad_fn = mx.grad(fwd_loss)
    g = grad_fn(x_nhwc)
    mx.eval(g)  # warm
    bwd_times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        g = grad_fn(x_nhwc)
        mx.eval(g)
        bwd_times.append(time.perf_counter() - t0)
    bwd_s = float(np.median(bwd_times))
    print(f"[bwd] full SegNet value_and_grad: {bwd_s*1e3:.1f} ms (B={B})")

    # backward WITH fallback instrumentation
    A.mlx_reference_conv2d_nhwc = _instrumented
    try:
        fallback_stats["calls"] = 0
        fallback_stats["wall_s"] = 0.0
        fallback_stats["shapes"] = []
        t0 = time.perf_counter()
        g = grad_fn(x_nhwc)
        mx.eval(g)
        bwd_instr_s = time.perf_counter() - t0
        bwd_fallback_calls = fallback_stats["calls"]
        bwd_fallback_s = fallback_stats["wall_s"]
        print(
            f"[bwd-instr] {bwd_instr_s*1e3:.1f} ms; fallback {bwd_fallback_calls} calls, "
            f"{bwd_fallback_s*1e3:.1f} ms ({100*bwd_fallback_s/max(bwd_instr_s,1e-9):.1f}% of bwd)"
        )
        bwd_fallback_shapes = list(fallback_stats["shapes"])
    finally:
        A.mlx_reference_conv2d_nhwc = orig_ref

    result = {
        "evidence_grade": "macOS-MLX research-signal",
        "batch": B,
        "input_hw": [H, W],
        "n_conv_layers": len(conv_rows),
        "n_strided_grouped_fallback_layers": n_fallback,
        "conv_layers": conv_rows,
        "forward": {
            "full_ms": round(fwd_s * 1e3, 3),
            "instrumented_ms": round(fwd_instr_s * 1e3, 3),
            "fallback_calls": fwd_fallback_calls,
            "fallback_ms": round(fwd_fallback_s * 1e3, 3),
            "fallback_frac": round(fwd_fallback_s / max(fwd_instr_s, 1e-9), 4),
        },
        "backward": {
            "full_ms": round(bwd_s * 1e3, 3),
            "instrumented_ms": round(bwd_instr_s * 1e3, 3),
            "fallback_calls": bwd_fallback_calls,
            "fallback_ms": round(bwd_fallback_s * 1e3, 3),
            "fallback_frac": round(bwd_fallback_s / max(bwd_instr_s, 1e-9), 4),
            "fallback_shapes": bwd_fallback_shapes,
        },
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}")
    print(
        f"\n=== PRIZE SIZING ===\n"
        f"  full backward:        {bwd_s*1e3:8.1f} ms\n"
        f"  fallback within bwd:  {bwd_fallback_s*1e3:8.1f} ms ({100*result['backward']['fallback_frac']:.1f}%)\n"
        f"  native-floor within:  {(bwd_instr_s-bwd_fallback_s)*1e3:8.1f} ms ({100*(1-result['backward']['fallback_frac']):.1f}%)\n"
    )


if __name__ == "__main__":
    main()
