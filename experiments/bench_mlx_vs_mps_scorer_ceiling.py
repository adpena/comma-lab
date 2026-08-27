#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Head-to-head scorer fwd+bwd benchmark: torch-CPU vs torch-MPS vs MLX-GPU.

THE SIDE-QUEST: empirically close the loop on "why is torch-MPS faster than our
MLX scorer port, and is MLX's ceiling just MPS?" (operator 2026-06-12).

This is the FIRST script that times the SAME frozen scorer (EfficientNet-B2
SegNet + FastViT-T12 PoseNet) forward+backward-to-input on ALL THREE backends on
the SAME real scorer-input batch — so the relative comparison is apples-to-apples
(CLAUDE.md "Apples-to-apples evidence discipline"). Prior art measured torch
CPU-vs-MPS (``bench_scorer_mps_vs_cpu.py``, the 104x path) and MLX-GPU *forward*
throughput (``mlx_scorer_drift_audit`` = 8.7 pairs/s) SEPARATELY; nobody ran the
MLX-vs-MPS fwd+bwd head-to-head. That gap is what this fills.

WHAT IS TIMED. Forward+backward-to-input through the two scorer heads on a batch
of B real pairs, at the cached scorer-input boundary:
  - SegNet head: in ``(B,3,384,512)`` RGB; out ``(B,5,384,512)`` logits.
  - PoseNet head: in ``(B,12,192,256)`` YUV6 (2 frames x 6 ch, [0,255]); out pose.
A deterministic scalar loss (sum of squared outputs across BOTH heads) is
backproped to the INPUT pixels — the exact quantity training charges (the scorer
is frozen; the gradient flows to the render). The cached inputs are IDENTICAL
across backends, so the only difference is the backend.

BACKENDS / VARIANTS (the A/B/C of the investigation):
  - ``torch-cpu``    : the authority + current-basin training path.
  - ``torch-mps``    : the literal upstream macOS device (the 104x path), fp32.
  - ``mlx-gpu``      : OUR MLX port, DEFAULT path (strided-grouped conv uses the
                       fixed-order reference accumulator — slow but parity-safe).
  - ``mlx-gpu-custom``: OUR MLX port with the opt-in custom Metal grouped-conv
                       BACKWARD kernel (``TAC_MLX_CUSTOM_GROUPED_BACKWARD=1``,
                       gradient-EXONERATED 2026-06-12, ~5.5x backward win). This
                       is the MLX-MAX candidate.
  - ``mlx-gpu-fwd``  : MLX-GPU FORWARD ONLY (isolates fwd cost vs the bwd cost).

AUTHORITY DISCIPLINE (CLAUDE.md "MPS auth eval is NOISE" + "NO FAKE"): every
number here is a REAL measured wall-time on the REAL frozen scorer + REAL 0.mkv
scorer-input cache. This is a SPEED + relative-fidelity investigation, NOT a
score claim. No d_seg/d_pose is promoted. All numbers are
``[macOS-MLX/MPS advisory]``. Concurrent training load is noted in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = (
    "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
)


# --------------------------------------------------------------------------- #
# Concurrent-load probe (so every result is honest about contention)
# --------------------------------------------------------------------------- #
def _concurrent_load_note() -> dict:
    """Best-effort snapshot of OTHER heavy python procs sharing the Metal GPU /
    CPU (the live training daemons). The benchmark CONTENDS with them; the
    RELATIVE comparison stays valid because all backends are measured under the
    same load in the same run, but absolute numbers are depressed."""
    try:
        out = subprocess.run(  # subprocess-no-check-OK: best-effort ps load census; failure degrades via the except arm
            ["ps", "-Ao", "pid,pcpu,comm,args"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return {"probe_failed": True}
    hot = []
    mypid = str(os.getpid())
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, pcpu, _comm, args = parts
        if pid == mypid:
            continue
        if "python" not in args.lower():
            continue
        try:
            cpu = float(pcpu)
        except ValueError:
            cpu = 0.0
        if cpu < 20.0:
            continue
        if any(k in args for k in ("launch_", "basin", "lever", "train", "run_")):
            hot.append({"pid": pid, "pcpu": cpu, "args": args[:140]})
    return {"heavy_concurrent_python": hot, "count": len(hot)}


# --------------------------------------------------------------------------- #
# Input cache
# --------------------------------------------------------------------------- #
def _load_inputs(cache_dir: str, batch: int):
    sys.path.insert(0, str(REPO / "src"))
    from tac.local_acceleration.mlx_scorer_response import load_scorer_input_cache

    cache = load_scorer_input_cache(cache_dir)
    seg = np.asarray(cache.segnet_last_rgb[:batch], dtype=np.float32)  # (B,3,384,512)
    pose = np.asarray(cache.posenet_yuv6_pair[:batch], dtype=np.float32)  # (B,12,192,256)
    return pose, seg


# --------------------------------------------------------------------------- #
# torch backend (cpu / mps)
# --------------------------------------------------------------------------- #
def _bench_torch(pose_np, seg_np, *, device: str, warmup: int, iters: int):
    import torch

    if device == "mps":
        from tac.torch_mps_compat import patch_scorer_for_mps

        patch_scorer_for_mps()

    from tac.score_aware_loop.targets import load_frozen_distortion_net

    net = load_frozen_distortion_net(upstream_dir="upstream", device=device)
    segnet = net.segnet
    posenet = net.posenet
    dev = torch.device(device)
    is_mps = device == "mps"

    pose0 = torch.from_numpy(pose_np)
    seg0 = torch.from_numpy(seg_np)

    times = []
    grad_ref = None
    for i in range(warmup + iters):
        seg_x = seg0.clone().to(dev).requires_grad_(True)
        pose_x = pose0.clone().to(dev).requires_grad_(True)
        if is_mps:
            torch.mps.synchronize()
        t0 = time.perf_counter()
        seg_out = segnet(seg_x)  # (B,5,384,512)
        pose_out = posenet(pose_x)  # dict of hydra heads
        if isinstance(pose_out, dict):
            pose_term = sum(v.float().pow(2).mean() for v in pose_out.values())
        else:
            pose_term = pose_out.float().pow(2).mean()
        loss = pose_term + seg_out.float().pow(2).mean()
        loss.backward()
        if is_mps:
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
            if grad_ref is None:
                grad_ref = seg_x.grad.detach().float().cpu().numpy().flatten()
    times.sort()
    return {
        "backend": f"torch-{device}",
        "ms_per_step": round(1000 * times[len(times) // 2], 2),
        "ms_min": round(1000 * times[0], 2),
        "iters": iters,
        "_seg_grad": grad_ref,
    }


# --------------------------------------------------------------------------- #
# MLX backend (gpu, fwd+bwd or fwd-only; default or custom-backward kernel)
# --------------------------------------------------------------------------- #
def _bench_mlx(pose_np, seg_np, *, warmup: int, iters: int, fwd_only: bool, label: str):
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import (
        nchw_to_nhwc,
        temporary_mlx_device,
        torch_distortion_net_to_mlx,
    )
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    # Build the adapter on CPU torch weights then move MLX arrays to the GPU.
    dist = load_frozen_distortion_net(upstream_dir="upstream", device="cpu")
    with temporary_mlx_device("gpu"):
        adapter = torch_distortion_net_to_mlx(dist)

    seg_nhwc = mx.array(nchw_to_nhwc(seg_np))  # (B,384,512,3)
    pose_nhwc = mx.array(nchw_to_nhwc(pose_np))  # (B,192,256,12)

    def _forward(seg_in, pose_in):
        seg_logits = adapter.segnet(seg_in)  # NHWC (B,H,W,5)
        pose_out = adapter.posenet(pose_in)["pose"]
        return (seg_logits.astype(mx.float32) ** 2).mean() + (
            pose_out.astype(mx.float32) ** 2
        ).mean()

    grad_ref = None
    times = []
    with temporary_mlx_device("gpu"):
        # warmup
        for _ in range(warmup):
            if fwd_only:
                loss = _forward(seg_nhwc, pose_nhwc)
                mx.eval(loss)
            else:
                vg = mx.value_and_grad(_forward, argnums=(0, 1))
                loss, (g_seg, _g_pose) = vg(seg_nhwc, pose_nhwc)
                mx.eval(loss, g_seg, _g_pose)
        for _i in range(iters):
            mx.synchronize()
            t0 = time.perf_counter()
            if fwd_only:
                loss = _forward(seg_nhwc, pose_nhwc)
                mx.eval(loss)
            else:
                vg = mx.value_and_grad(_forward, argnums=(0, 1))
                loss, (g_seg, g_pose) = vg(seg_nhwc, pose_nhwc)
                mx.eval(loss, g_seg, g_pose)
                if grad_ref is None:
                    grad_ref = np.asarray(g_seg, dtype=np.float32).flatten()
            mx.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "backend": label,
        "ms_per_step": round(1000 * times[len(times) // 2], 2),
        "ms_min": round(1000 * times[0], 2),
        "iters": iters,
        "_seg_grad": grad_ref,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--backends",
        default="torch-cpu,torch-mps,mlx-gpu,mlx-gpu-custom,mlx-gpu-fwd",
        help="comma list subset of "
        "torch-cpu,torch-mps,mlx-gpu,mlx-gpu-custom,mlx-gpu-fwd",
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))

    load = _concurrent_load_note()
    print(f"[load] concurrent heavy python procs: {load.get('count', '?')}", flush=True)
    for p in load.get("heavy_concurrent_python", []):
        print(f"       pid={p['pid']} cpu={p['pcpu']:.0f}% {p['args']}", flush=True)

    pose_np, seg_np = _load_inputs(args.cache_dir, args.batch)
    print(
        f"[input] B={args.batch} seg={seg_np.shape} pose={pose_np.shape} "
        f"iters={args.iters} warmup={args.warmup}",
        flush=True,
    )

    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = []
    grads = {}

    for b in wanted:
        try:
            if b == "torch-cpu":
                r = _bench_torch(pose_np, seg_np, device="cpu",
                                 warmup=args.warmup, iters=args.iters)
            elif b == "torch-mps":
                import torch

                if not torch.backends.mps.is_available():
                    print("[torch-mps] UNAVAILABLE — skipped", flush=True)
                    continue
                r = _bench_torch(pose_np, seg_np, device="mps",
                                 warmup=args.warmup, iters=args.iters)
            elif b == "mlx-gpu":
                os.environ.pop("TAC_MLX_CUSTOM_GROUPED_BACKWARD", None)
                r = _bench_mlx(pose_np, seg_np, warmup=args.warmup, iters=args.iters,
                               fwd_only=False, label="mlx-gpu")
            elif b == "mlx-gpu-custom":
                os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
                r = _bench_mlx(pose_np, seg_np, warmup=args.warmup, iters=args.iters,
                               fwd_only=False, label="mlx-gpu-custom")
                os.environ.pop("TAC_MLX_CUSTOM_GROUPED_BACKWARD", None)
            elif b == "mlx-gpu-fwd":
                os.environ.pop("TAC_MLX_CUSTOM_GROUPED_BACKWARD", None)
                r = _bench_mlx(pose_np, seg_np, warmup=args.warmup, iters=args.iters,
                               fwd_only=True, label="mlx-gpu-fwd")
            else:
                print(f"[warn] unknown backend {b!r}; skipped", flush=True)
                continue
        except Exception as e:
            print(f"[{b}] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback

            traceback.print_exc()
            continue
        grads[r["backend"]] = r.pop("_seg_grad", None)
        results.append(r)
        print(
            f"[{r['backend']:16s}] {r['ms_per_step']:9.2f} ms/step  "
            f"(min {r['ms_min']:.2f})",
            flush=True,
        )

    # Relative table (vs torch-cpu and vs torch-mps).
    by = {r["backend"]: r["ms_per_step"] for r in results}
    cpu = by.get("torch-cpu")
    mps = by.get("torch-mps")
    print("\n[relative] speedup vs torch-cpu (fwd+bwd):", flush=True)
    for r in results:
        s_cpu = f"{cpu / r['ms_per_step']:6.2f}x" if cpu else "  n/a"
        s_mps = f"{mps / r['ms_per_step']:6.2f}x" if mps else "  n/a"
        print(f"   {r['backend']:16s} vs-cpu={s_cpu}  vs-mps={s_mps}", flush=True)

    # Gradient direction sanity (seg-input grad cosine vs torch-cpu).
    ref = grads.get("torch-cpu")
    cosines = {}
    if ref is not None:
        rn = ref / (np.linalg.norm(ref) + 1e-30)
        for name, g in grads.items():
            if g is None or name == "torch-cpu" or g.shape != ref.shape:
                continue
            gn = g / (np.linalg.norm(g) + 1e-30)
            cosines[name] = float(np.dot(rn, gn))
    if cosines:
        print("\n[grad-cosine vs torch-cpu] (seg-input gradient direction sanity):",
              flush=True)
        for name, c in cosines.items():
            print(f"   {name:16s} cos={c:.6f}", flush=True)

    payload = {
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "evidence_grade": "macOS-MLX/MPS advisory; torch-CPU is the only authority",
        "score_claim": False,
        "promotable": False,
        "batch": args.batch,
        "iters": args.iters,
        "warmup": args.warmup,
        "cache_dir": args.cache_dir,
        "concurrent_load": load,
        "results": results,
        "speedup_vs_cpu": {
            r["backend"]: (round(cpu / r["ms_per_step"], 3) if cpu else None)
            for r in results
        },
        "speedup_vs_mps": {
            r["backend"]: (round(mps / r["ms_per_step"], 3) if mps else None)
            for r in results
        },
        "seg_input_grad_cosine_vs_cpu": cosines,
    }
    out_json = args.out_json or (
        REPO / ".omx/research" / f"mlx_vs_mps_scorer_ceiling_{payload['stamp']}.json"
    )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"\n[json] {out_json}", flush=True)


if __name__ == "__main__":
    main()
