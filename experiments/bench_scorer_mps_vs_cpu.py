#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Upstream-faithful throughput probe: the REAL PyTorch DistortionNet scorer
forward+backward on torch-MPS (Apple GPU) vs torch-CPU vs torch-MPS-fp16.

The operator's redirect ("do whatever upstream did but for macOS"): upstream's
own ``modules.py`` selects ``cuda -> mps -> cpu`` and carries a
``# TODO run in bfloat16?`` on the scorer forward. We have been running the
scorer on CPU (19 s/step) and a hand-built FP32-exact MLX port (26 s/step,
handbraked) — NOT the device upstream itself would use on a Mac. This measures
the literal upstream approach: the same PyTorch modules on the Apple GPU.

The scorer is FROZEN; the gradient flows to the rendered-frame INPUT (then to
the decoder). So we time forward+backward-to-input, the actual training cost.

Authority discipline (CLAUDE.md "MPS auth eval is NOISE"): MPS is used here ONLY
as a TRAINING-GRADIENT throughput backend. The EXACT d_seg/d_pose for any
decision remain torch-CPU. This probe reports (a) wall-time CPU vs MPS vs
MPS-fp16, and (b) the input-gradient cosine MPS-vs-CPU — a FIRST descent-direction
sanity, NOT a descent-equivalence verdict (that needs the bounded both-terms A/B).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _patch_batchnorm_contiguous_for_mps():
    """FRONTIER runtime patch (NOT editing the pinned upstream snapshot): force the
    BatchNorm input contiguous on MPS so the BatchNorm BACKWARD does not hit the
    PyTorch-MPS `.view`-on-non-contiguous-saved-tensor bug (NativeBatchNormBackward0).
    Same pattern as ``patch_upstream_yuv6_globally``. Forward numerics unchanged."""
    import torch.nn.functional as F

    if getattr(F, "_tac_bn_mps_contig_patched", False):
        return
    _orig = F.batch_norm

    def _bn(inp, *a, **k):
        if inp.device.type == "mps":
            inp = inp.contiguous()
        return _orig(inp, *a, **k)

    F.batch_norm = _bn
    F._tac_bn_mps_contig_patched = True


def _faithful_loss(posenet_out, segnet_out):
    # A deterministic differentiable scalar that exercises BOTH heads end-to-end
    # (so the backward traverses the full PoseNet + SegNet graph, the real cost).
    # PoseNet.forward returns a dict of hydra heads; SegNet returns a tensor.
    if isinstance(posenet_out, dict):
        pose_term = sum(v.float().pow(2).mean() for v in posenet_out.values())
    else:
        pose_term = posenet_out.float().pow(2).mean()
    return pose_term + segnet_out.float().pow(2).mean()


def _bench(net, x0, device, *, autocast_dtype=None, warmup=2, iters=5, label=""):
    is_mps = device.type == "mps"
    times = []
    grad_ref = None
    for i in range(warmup + iters):
        x = x0.clone().to(device).requires_grad_(True)
        if is_mps:
            torch.mps.synchronize()
        t0 = time.perf_counter()
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                pose_out, seg_out = net(x)
                loss = _faithful_loss(pose_out, seg_out)
        else:
            pose_out, seg_out = net(x)
            loss = _faithful_loss(pose_out, seg_out)
        loss.backward()
        if is_mps:
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
            if grad_ref is None:
                grad_ref = x.grad.detach().float().cpu().flatten()
    times.sort()
    med = times[len(times) // 2]
    return med, grad_ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--upstream-dir", default="upstream")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    _patch_batchnorm_contiguous_for_mps()
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    sys.path.insert(0, args.upstream_dir)
    import frame_utils

    W, H = frame_utils.camera_size  # (1164, 874)
    T = frame_utils.seq_len         # 2
    B = args.batch_size
    print(f"[probe] input (B,T,H,W,C)=({B},{T},{H},{W},3) torch={torch.__version__} "
          f"mps={torch.backends.mps.is_available()}", flush=True)

    torch.manual_seed(0)
    # Frame batch in [0,1], the rendered-frame domain the decoder emits.
    x0 = torch.rand(B, T, H, W, 3)

    # --- torch-CPU (the authority + current basin path) ---
    net_cpu = load_frozen_distortion_net(upstream_dir=args.upstream_dir, device="cpu")
    cpu_t, cpu_grad = _bench(net_cpu, x0, torch.device("cpu"),
                             warmup=args.warmup, iters=args.iters, label="cpu")
    print(f"[cpu      fp32] {cpu_t*1000:8.1f} ms/step (fwd+bwd, B={B})", flush=True)

    if not torch.backends.mps.is_available():
        print("[mps] UNAVAILABLE — cannot probe the upstream macOS path.")
        return

    # --- torch-MPS fp32 (the literal upstream macos device) ---
    results = {}
    try:
        net_mps = load_frozen_distortion_net(upstream_dir=args.upstream_dir, device="mps")
        mps_t, mps_grad = _bench(net_mps, x0, torch.device("mps"),
                                 warmup=args.warmup, iters=args.iters, label="mps")
        cos = torch.nn.functional.cosine_similarity(cpu_grad, mps_grad, dim=0).item()
        results["mps_fp32"] = (mps_t, cos)
        print(f"[mps      fp32] {mps_t*1000:8.1f} ms/step  speedup={cpu_t/mps_t:5.2f}x  "
              f"grad_cosine_vs_cpu={cos:.6f}", flush=True)
    except Exception as e:  # noqa: BLE001 — report the unsupported op honestly
        print(f"[mps fp32] FAILED: {type(e).__name__}: {e}", flush=True)

    # --- torch-MPS fp16 autocast (upstream's own bfloat16 TODO, but fp16) ---
    for dt_name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        try:
            net_mps2 = load_frozen_distortion_net(upstream_dir=args.upstream_dir, device="mps")
            t, g = _bench(net_mps2, x0, torch.device("mps"), autocast_dtype=dt,
                          warmup=args.warmup, iters=args.iters, label=f"mps-{dt_name}")
            cos = torch.nn.functional.cosine_similarity(cpu_grad, g, dim=0).item()
            results[f"mps_{dt_name}"] = (t, cos)
            print(f"[mps autocast {dt_name}] {t*1000:8.1f} ms/step  speedup={cpu_t/t:5.2f}x  "
                  f"grad_cosine_vs_cpu={cos:.6f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[mps {dt_name}] FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n[verdict] wall-time is the throughput signal; grad_cosine_vs_cpu is a "
          "FIRST descent-direction sanity (NOT a descent-equivalence verdict — that "
          "needs the bounded both-terms A/B). Higher cosine => MPS gradient points "
          "the same way as the CPU authority gradient.", flush=True)


if __name__ == "__main__":
    main()
