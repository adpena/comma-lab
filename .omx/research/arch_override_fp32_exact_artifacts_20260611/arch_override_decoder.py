# SPDX-License-Identifier: MIT
"""Capstone DECODER forward throughput + numerics under MLX_METAL_GPU_ARCH override.

The override is process-wide, so it would force the decoder's MLX kernels onto the
non-NAX path too. This measures whether that hurts the decoder forward (the
renderer/VQ-NeRV decode). Run once per process. [macOS-MLX research-signal].
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src/tac/mlx_pr95_port/tests")
import mlx.core as mx  # noqa: E402

from tac.capstone_vq_nerv.vq_nerv_bundle import (  # noqa: E402
    CapstoneVqNervBundle,
    CapstoneVqNervConfig,
)

ARCH = mx.device_info()["architecture"]
N = 48
cfg = CapstoneVqNervConfig(num_pairs=N, base_channels=20, carrier="stored_latent")
bundle = CapstoneVqNervBundle(cfg)
rng = np.random.default_rng(0)
pose_store = rng.standard_normal((N, 6)).astype(np.float32)
bundle.set_pose_stats(pose_store.mean(0), pose_store.std(0))
idx = mx.array(np.arange(N, dtype=np.int32))
pose = mx.array(pose_store)

# warmup + checksum (numerics fingerprint of the decoder under this arch)
r = bundle(idx, pose)
mx.eval(r)
checksum = float(np.asarray(mx.sum(r)))
mean = float(np.asarray(mx.mean(r)))

ITERS = 5
t0 = time.time()
for _ in range(ITERS):
    r = bundle(idx, pose)
    mx.eval(r)
dt = (time.time() - t0) / ITERS

out = {
    "arch": ARCH,
    "n_pairs": N,
    "decoder_fwd_s_per_call": round(dt, 4),
    "decoder_pairs_per_s": round(N / dt, 3),
    "render_sum_checksum": checksum,
    "render_mean": mean,
    "render_shape": list(r.shape),
}
print("DECODER_RESULT " + json.dumps(out))
os.makedirs(".omx/tmp/arch_override_out", exist_ok=True)
tag = "g15" if ARCH == "applegpu_g15" else ARCH
with open(f".omx/tmp/arch_override_out/decoder_{tag}.json", "w") as f:
    json.dump(out, f, indent=2)
