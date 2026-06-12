"""Full scorer (SegNet + PoseNet) end-to-end backward: custom Metal kernel vs
Python-loop fallback. The number that matters for the training step.

PoseNet ALSO has strided-grouped fallback layers (8 of them per the per-layer
profile), so the full-scorer win is the union of SegNet + PoseNet.

Authority: torch-CPU exact = d_seg/d_pose authority. All numbers
`[macOS-MLX research-signal]`. NO score claim. NO MPS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "upstream"


def _pair(v: Any) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _load_distortion() -> Any:
    import sys

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--out-json",
        default=str(REPO / "experiments" / "results" / "mlx_full_scorer_custom_backward.json"),
    )
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(UPSTREAM))

    import mlx.core as mx

    from experiments.measure_mlx_segnet_end_to_end_custom_backward import CustomKernelConvAdapter
    from tac.local_acceleration import mlx_scorer_adapters as A

    mx.set_default_device(mx.gpu)
    net = _load_distortion()

    def build(mode: str):
        orig = A.torch_conv2d_to_mlx
        if mode == "custom":
            def patched(tc):
                if int(tc.groups) > 1 and _pair(tc.stride) != (1, 1):
                    return CustomKernelConvAdapter(tc)
                return orig(tc)
            A.torch_conv2d_to_mlx = patched
        try:
            adapter = A.torch_distortion_net_to_mlx(net)
        finally:
            A.torch_conv2d_to_mlx = orig
        return adapter

    scorer_ref = build("reference")
    scorer_cust = build("custom")

    B = args.batch
    rng = np.random.default_rng(0)
    # SegNet input: frame-1 RGB (B, 384, 512, 3) NHWC
    seg_in = mx.array(rng.standard_normal((B, 384, 512, 3)).astype(np.float32))
    # PoseNet input: YUV6 pair (B, 192, 256, 12) NHWC
    pose_in = mx.array(rng.standard_normal((B, 192, 256, 12)).astype(np.float32))

    def loss(scorer, si, pi):
        out = scorer(pi, si)
        return mx.sum(out["segnet"] ** 2) + mx.sum(out["posenet"]["pose"] ** 2)

    # gradient direction parity on BOTH inputs
    gf_ref = mx.grad(lambda si, pi: loss(scorer_ref, si, pi), argnums=(0, 1))
    gf_cust = mx.grad(lambda si, pi: loss(scorer_cust, si, pi), argnums=(0, 1))
    gsi_r, gpi_r = gf_ref(seg_in, pose_in)
    gsi_c, gpi_c = gf_cust(seg_in, pose_in)
    mx.eval(gsi_r, gpi_r, gsi_c, gpi_c)

    def cos(a, b):
        a = np.asarray(a).ravel().astype(np.float64)
        b = np.asarray(b).ravel().astype(np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    seg_grad_cos = cos(gsi_r, gsi_c)
    pose_grad_cos = cos(gpi_r, gpi_c)

    def time_grad(gf):
        a, b = gf(seg_in, pose_in)
        mx.eval(a, b)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            a, b = gf(seg_in, pose_in)
            mx.eval(a, b)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    bwd_ref = time_grad(gf_ref)
    bwd_cust = time_grad(gf_cust)

    res = {
        "evidence_grade": "macOS-MLX research-signal",
        "batch": B,
        "segnet_input_grad_cosine": round(seg_grad_cos, 8),
        "posenet_input_grad_cosine": round(pose_grad_cos, 8),
        "full_scorer_backward": {
            "reference_ms": round(bwd_ref * 1e3, 1),
            "custom_ms": round(bwd_cust * 1e3, 1),
            "speedup": round(bwd_ref / max(bwd_cust, 1e-9), 2),
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print("=== FULL SCORER (SegNet+PoseNet, real weights) backward: custom vs loop ===")
    print(f"  segnet-input grad cosine:  {seg_grad_cos:.8f}")
    print(f"  posenet-input grad cosine: {pose_grad_cos:.8f}")
    print(
        f"  backward: ref {bwd_ref*1e3:8.1f} ms  custom {bwd_cust*1e3:8.1f} ms  = "
        f"{res['full_scorer_backward']['speedup']}x"
    )
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
