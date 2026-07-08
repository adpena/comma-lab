#!/usr/bin/env python3
"""Reproduce the --micro-batch-pairs bit-identity DECOMPOSITION (2026-07-08 crux finding).

Measures, live, the three components that decide whether B>1 can be bit-identical to the
serial accumulation path, and the surviving speedup at bit-identity:

  A. SCORER FORWARD batch-dependence (real EfficientNet-B2 SegNet / FastViT PoseNet):
     is segnet(f1_batch)[k] bit-identical to segnet(f1_batch[k:k+1])[0]?  (the irreducible
     root — upstream of any reduction).  cpu AND gpu.
  B. REDUCTION/accumulation ORDER (batch-INVARIANT mock scorer, isolates A away): the
     batched twin grad vs the serial left-fold mean-of-per-pair grad.
  SPEEDUP: ONE batched scorer forward over K frames vs K per-pair forwards.

Emits a JSON verdict per device (see tac.boundary_math.micro_batch_bit_identity_probe.
BitIdentityVerdict). MEANS: pointer contest-CPU 0.19110 UNMOVED; only a byte-closed n600
evaluate.py row moves it.

    .venv/bin/python tools/micro_batch_bit_identity_probe.py --devices cpu gpu --K 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _measure_real_scorer_forward(device: str, K: int, seed: int = 0):
    """Real upstream adapter: segnet/posenet batched-vs-single forward max|Δ| + argmax flips."""
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    with temporary_mlx_device(device):
        ad = load_mlx_distortion_scorer_adapter_from_upstream(Path("upstream"), device="cpu")
        rng = np.random.default_rng(seed)
        H, W = 384, 512
        f1 = mx.array(rng.random((K, H, W, 3)).astype(np.float32) * 255.0)
        f0 = mx.array(rng.random((K, H, W, 3)).astype(np.float32) * 255.0)
        mx.eval(f0, f1)
        sl_b = ad.segnet(f1)
        mx.eval(sl_b)
        seg_maxabs = 0.0
        argmax_flips = 0
        for k in range(K):
            slk = ad.segnet(f1[k:k + 1])
            mx.eval(slk)
            seg_maxabs = max(seg_maxabs, float(
                np.max(np.abs(np.asarray(sl_b[k:k + 1], np.float64) - np.asarray(slk, np.float64)))))
            am_b = np.argmax(np.asarray(sl_b[k:k + 1]), axis=-1)
            am_k = np.argmax(np.asarray(slk), axis=-1)
            argmax_flips += int(np.sum(am_b != am_k))

        def _yuv(a, b):
            pair = mx.stack([a, b], axis=1)
            yuv = rgb_to_yuv6_mlx(pair)
            _k, _t, _h2, _w2, _c6 = yuv.shape
            return mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (_k, _h2, _w2, _t * _c6))

        yb = _yuv(f0, f1)
        pb = ad.posenet(yb)["pose"]
        mx.eval(pb)
        pose_maxabs = 0.0
        for k in range(K):
            pk = ad.posenet(yb[k:k + 1])["pose"]
            mx.eval(pk)
            pose_maxabs = max(pose_maxabs, float(
                np.max(np.abs(np.asarray(pb[k:k + 1], np.float64) - np.asarray(pk, np.float64)))))
        return seg_maxabs, int(argmax_flips), pose_maxabs


def _measure_speedup(device: str, K: int, iters: int = 6, seed: int = 0) -> float:
    """ONE batched scorer forward over K vs K per-pair forwards -> wall-clock speedup."""
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    with temporary_mlx_device(device):
        ad = load_mlx_distortion_scorer_adapter_from_upstream(Path("upstream"), device="cpu")
        rng = np.random.default_rng(seed)
        H, W = 384, 512
        f1 = mx.array(rng.random((K, H, W, 3)).astype(np.float32) * 255.0)
        f0 = mx.array(rng.random((K, H, W, 3)).astype(np.float32) * 255.0)
        mx.eval(f0, f1)

        def _yuv(a, b):
            pair = mx.stack([a, b], axis=1)
            yuv = rgb_to_yuv6_mlx(pair)
            _k, _t, _h2, _w2, _c6 = yuv.shape
            return mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (_k, _h2, _w2, _t * _c6))

        mx.eval(ad.segnet(f1), ad.posenet(_yuv(f0, f1))["pose"])  # warmup
        t = time.perf_counter()
        for _ in range(iters):
            mx.eval(ad.segnet(f1), ad.posenet(_yuv(f0, f1))["pose"])
        tb = (time.perf_counter() - t) / iters
        t = time.perf_counter()
        for _ in range(iters):
            for k in range(K):
                mx.eval(ad.segnet(f1[k:k + 1]), ad.posenet(_yuv(f0[k:k + 1], f1[k:k + 1]))["pose"])
        tp = (time.perf_counter() - t) / iters
        return float(tp / tb) if tb > 0 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=["cpu"], choices=["cpu", "gpu"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    import mlx.core as mx

    from tac.boundary_math.micro_batch_bit_identity_probe import (
        classify_micro_batch_bit_identity,
        measure_reduction_order_drift,
    )

    # Source B (device-independent MATH; measure on CPU for determinism).
    mx.set_default_device(mx.cpu)
    red = measure_reduction_order_drift(K=min(args.K, 4), seg_form="ce")
    reductions = {sf: measure_reduction_order_drift(K=min(args.K, 4), seg_form=sf).grad_maxabs
                  for sf in ("ce", "tau_softplus", "margin_hinge")}

    results = []
    for dev in args.devices:
        seg, flips, pose = _measure_real_scorer_forward(dev, args.K, seed=args.seed)
        speedup = _measure_speedup(dev, args.K, iters=args.iters, seed=args.seed)
        verdict = classify_micro_batch_bit_identity(
            device=dev, scorer_fwd_seg_maxabs=seg, scorer_fwd_argmax_flips=flips,
            scorer_fwd_pose_maxabs=pose, reduction_order_grad_maxabs=red.grad_maxabs,
            scorer_fwd_speedup=speedup)
        results.append(verdict.as_dict())

    out = {
        "schema": "micro_batch_bit_identity_probe.v1",
        "K": int(args.K),
        "reduction_order_grad_maxabs_by_segform": reductions,
        "reduction_order_grad_rel_l2_ce": red.grad_rel_l2,
        "reduction_order_loss_abs_ce": red.loss_abs,
        "per_device": results,
        "note": ("scorer forward is the irreducible root of the micro-batch trajectory gate; "
                 "reduction-order is secondary. Full bit-identity at speedup>1x is impossible "
                 "on the real MLX scorer. MEANS: pointer 0.19110 UNMOVED."),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
