# SPDX-License-Identifier: MIT
"""MLX-GPU score-bridge fwd-only + fwd+bwd throughput at bs=4/8/16.

Run once per process so MLX_METAL_GPU_ARCH applies. Reuses the bridge's real-net
setup. Times loss_and_pixel_grad (the full fwd+bwd training step the trainer uses)
and a fwd-only exact_d_seg/fused path. Reports pairs/s after a warmup.

torch-CPU = authority; MLX numbers are [macOS-MLX research-signal], non-promotable.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src/tac/mlx_pr95_port/tests")
from test_mlx_gpu_score_bridge import _build_real_setup  # type: ignore

from tac.mlx_pr95_port.mlx_gpu_score_bridge import MLXGpuScorerBridge

import mlx.core as mx  # noqa: E402

ARCH = mx.device_info()["architecture"]
BATCHES = [4, 8, 16]
WARMUP, ITERS = 1, 3

# build a 24-pair real setup (cache has n24); slice idx per batch.
net, seg_t, pose_t, render16, idx16 = _build_real_setup(n_pairs=24)
bridge = MLXGpuScorerBridge(
    net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
    seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True, device_type="gpu",
)

results = {"arch": ARCH, "batches": {}}
for bs in BATCHES:
    r = render16[:bs]
    idx = idx16[:bs]
    # fwd+bwd
    for _ in range(WARMUP):
        res = bridge.loss_and_pixel_grad(r, idx)
        mx.eval(res.pixel_cotangent)
    t0 = time.time()
    for _ in range(ITERS):
        res = bridge.loss_and_pixel_grad(r, idx)
        mx.eval(res.pixel_cotangent)
    fwdbwd_s = (time.time() - t0) / ITERS
    # fwd-only
    for _ in range(WARMUP):
        bridge.fused_d_seg_d_pose(r, idx)
    t0 = time.time()
    for _ in range(ITERS):
        bridge.fused_d_seg_d_pose(r, idx)
    fwd_s = (time.time() - t0) / ITERS
    results["batches"][str(bs)] = {
        "fwdbwd_pairs_per_s": round(bs / fwdbwd_s, 4),
        "fwd_only_pairs_per_s": round(bs / fwd_s, 4),
        "fwdbwd_s_per_step": round(fwdbwd_s, 4),
    }
    print(f"  bs={bs} fwd+bwd={bs/fwdbwd_s:.4f} p/s  fwd-only={bs/fwd_s:.4f} p/s")

print("THROUGHPUT_RESULT " + json.dumps(results))
os.makedirs(".omx/tmp/arch_override_out", exist_ok=True)
tag = "g15" if ARCH == "applegpu_g15" else ARCH
with open(f".omx/tmp/arch_override_out/throughput_{tag}.json", "w") as f:
    json.dump(results, f, indent=2)
