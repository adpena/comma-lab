# SPDX-License-Identifier: MIT
"""Backward/VJP gradient drift of the MLX-GPU score bridge vs torch-CPU authority.

Run ONCE per process (the arch override must be set before `import mlx`), so this
is invoked twice from the shell: once with MLX_METAL_GPU_ARCH unset (g17s/NAX) and
once with MLX_METAL_GPU_ARCH=applegpu_g15 (non-NAX). It reuses the EXACT NO-FAKE
real-net setup from the bridge's own parity test (`_build_real_setup`): real
upstream SegNet/PoseNet, real 0.mkv GT targets, a real trained-init render. The
load-bearing quantity is the pixel cotangent dL/d(render) — the training gradient.

torch-CPU = authority. All MLX numbers are [macOS-MLX research-signal], non-promotable.
"""
import json
import os
import sys
import time

import numpy as np
import torch

# import the bridge test's real-setup builder (reuse, do not re-create)
sys.path.insert(0, "src/tac/mlx_pr95_port/tests")
from test_mlx_gpu_score_bridge import _build_real_setup, _cos  # type: ignore

from tac.mlx_pr95_port.mlx_gpu_score_bridge import MLXGpuScorerBridge
from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

import mlx.core as mx  # noqa: E402

ARCH = mx.device_info()["architecture"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

net, seg_t, pose_t, render, idx_t = _build_real_setup(n_pairs=N)

# torch-CPU AUTHORITY gradient.
torch_bridge = TorchScorerBridge(
    net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
    seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True,
)
tres = torch_bridge.loss_and_pixel_grad(render, idx_t)
torch_grad = np.asarray(tres.pixel_cotangent, dtype=np.float64)

# MLX-GPU gradient under the current process arch.
gpu_bridge = MLXGpuScorerBridge(
    net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
    seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True, device_type="gpu",
)
t0 = time.time()
gres = gpu_bridge.loss_and_pixel_grad(render, idx_t)
mx.eval(gres.pixel_cotangent)
dt = time.time() - t0
gpu_grad = np.asarray(gres.pixel_cotangent, dtype=np.float64)

# Drift metrics: cosine, rel-L2, abs-max delta of the GRADIENT (training signal),
# plus the loss/d_seg agreement the forward test charges.
diff = gpu_grad - torch_grad
rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(torch_grad) + 1e-12))
out = {
    "arch": ARCH,
    "n_pairs": N,
    "grad_cosine_vs_torch_cpu": float(_cos(gpu_grad, torch_grad)),
    "grad_rel_l2_error": rel_l2,
    "grad_abs_max_delta": float(np.abs(diff).max()),
    "grad_torch_abs_max": float(np.abs(torch_grad).max()),
    "grad_gpu_abs_max": float(np.abs(gpu_grad).max()),
    "grad_is_nonzero_nofake": bool(np.abs(gpu_grad).max() > 1e-6),
    "loss_rel_error": abs(gres.loss_value - tres.loss_value) / (abs(tres.loss_value) + 1e-9),
    "seg_loss_abs_delta": abs(gres.seg_loss_value - tres.seg_loss_value),
    "pose_loss_abs_delta": abs(gres.pose_loss_value - tres.pose_loss_value),
    "d_seg_gpu": gres.d_seg,
    "d_seg_torch": tres.d_seg,
    "d_seg_abs_delta": abs(gres.d_seg - tres.d_seg),
    "loss_value_gpu": gres.loss_value,
    "loss_value_torch": tres.loss_value,
    "fwdbwd_wall_s": round(dt, 3),
}
print("GRAD_RESULT " + json.dumps(out))
os.makedirs(".omx/tmp/arch_override_out", exist_ok=True)
tag = "g15" if ARCH == "applegpu_g15" else ARCH
with open(f".omx/tmp/arch_override_out/grad_{tag}.json", "w") as f:
    json.dump(out, f, indent=2)
