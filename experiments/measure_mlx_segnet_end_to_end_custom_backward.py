"""End-to-end SegNet backward: Python-loop fallback vs custom Metal backward.

The load-bearing measurement. Builds the REAL MLX SegNet two ways:
  A) reference-fallback (current main): strided-grouped layers use the
     Python-loop ``mlx_reference_conv2d_nhwc`` backward.
  B) custom-kernel: strided-grouped layers use the config-bound
     ``grouped_conv2d_nhwc`` (native fwd + Metal-kernel vjp).

Measures the full SegNet value_and_grad wall-time for each, and validates that
the FULL-SegNet pixel gradient direction matches between the two (the custom
backward must not change the descent direction).

Authority: torch-CPU exact = d_seg authority. All numbers `[macOS-MLX
research-signal]`. NO score claim. NO MPS.
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


def _load_segnet() -> Any:
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
    return net.segnet


class CustomKernelConvAdapter:
    """Strided-grouped Conv2d (NHWC) using the custom Metal backward kernel.

    Forward = native ``mx.conv2d`` (bit-exact, fast). Backward = the
    config-bound ``grouped_conv2d_nhwc`` custom_function whose vjp runs the
    two Metal kernels. Same weight layout (O,kH,kW,I/group) as the reference.
    """

    def __init__(self, torch_conv: Any):
        import mlx.core as mx

        from experiments.wire_and_measure_mlx_custom_backward import make_grouped_conv2d
        from tac.local_acceleration.mlx_scorer_adapters import _torch_tensor_to_numpy

        self.stride = _pair(torch_conv.stride)
        self.padding = _pair(torch_conv.padding)
        self.dilation = _pair(torch_conv.dilation)
        self.groups = int(torch_conv.groups)
        weight = _torch_tensor_to_numpy(torch_conv.weight)  # OIHW
        self.weight = mx.array(
            np.ascontiguousarray(weight.transpose(0, 2, 3, 1)), dtype=mx.float32
        )
        self.bias = (
            None
            if torch_conv.bias is None
            else mx.array(
                _torch_tensor_to_numpy(torch_conv.bias).reshape(1, 1, 1, -1).astype(np.float32)
            )
        )
        self._conv = make_grouped_conv2d(self.stride, self.padding, self.dilation, self.groups)

    def __call__(self, x_nhwc: Any) -> Any:
        out = self._conv(x_nhwc, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


def build_segnet(segnet_torch: Any, *, mode: str) -> Any:
    """Build the MLX SegNet, choosing the strided-grouped backward path."""
    from tac.local_acceleration import mlx_scorer_adapters as A

    orig = A.torch_conv2d_to_mlx
    if mode == "reference":
        return A.torch_segnet_to_mlx(segnet_torch)

    # mode == "custom": patch torch_conv2d_to_mlx so strided-grouped layers use
    # the custom kernel adapter; everything else uses the native fast path.
    def patched(torch_conv):
        if int(torch_conv.groups) > 1 and _pair(torch_conv.stride) != (1, 1):
            return CustomKernelConvAdapter(torch_conv)
        return orig(torch_conv)

    A.torch_conv2d_to_mlx = patched
    try:
        adapter = A.torch_segnet_to_mlx(segnet_torch)
    finally:
        A.torch_conv2d_to_mlx = orig
    return adapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--out-json",
        default=str(REPO / "experiments" / "results" / "mlx_segnet_e2e_custom_backward.json"),
    )
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(UPSTREAM))

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    segnet_torch = _load_segnet()

    B, H, W = args.batch, 384, 512
    rng = np.random.default_rng(0)
    x_nhwc = mx.array(rng.standard_normal((B, H, W, 3)).astype(np.float32))

    seg_ref = build_segnet(segnet_torch, mode="reference")
    seg_cust = build_segnet(segnet_torch, mode="custom")

    def loss_ref(x):
        out = seg_ref(x)
        return mx.sum(out * out)

    def loss_cust(x):
        out = seg_cust(x)
        return mx.sum(out * out)

    # forward parity
    o_ref = seg_ref(x_nhwc)
    o_cust = seg_cust(x_nhwc)
    mx.eval(o_ref, o_cust)
    fwd_absmax = float(np.abs(np.asarray(o_ref) - np.asarray(o_cust)).max())

    # gradient direction parity (full SegNet pixel cotangent)
    g_ref = mx.grad(loss_ref)(x_nhwc)
    g_cust = mx.grad(loss_cust)(x_nhwc)
    mx.eval(g_ref, g_cust)
    a = np.asarray(g_ref).ravel().astype(np.float64)
    b = np.asarray(g_cust).ravel().astype(np.float64)
    grad_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    grad_relmax = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-9))

    def time_grad(fn):
        gf = mx.grad(fn)
        g = gf(x_nhwc)
        mx.eval(g)  # warm (includes one-time JIT)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            g = gf(x_nhwc)
            mx.eval(g)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    bwd_ref = time_grad(loss_ref)
    bwd_cust = time_grad(loss_cust)

    # also forward-only end-to-end
    def time_fwd(net):
        o = net(x_nhwc)
        mx.eval(o)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            o = net(x_nhwc)
            mx.eval(o)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    fwd_ref = time_fwd(seg_ref)
    fwd_cust = time_fwd(seg_cust)

    res = {
        "evidence_grade": "macOS-MLX research-signal",
        "batch": B,
        "forward_parity_absmax": fwd_absmax,
        "full_segnet_grad_cosine": round(grad_cos, 8),
        "full_segnet_grad_relmax": grad_relmax,
        "forward": {
            "reference_ms": round(fwd_ref * 1e3, 1),
            "custom_ms": round(fwd_cust * 1e3, 1),
            "speedup": round(fwd_ref / max(fwd_cust, 1e-9), 2),
        },
        "backward_value_and_grad": {
            "reference_ms": round(bwd_ref * 1e3, 1),
            "custom_ms": round(bwd_cust * 1e3, 1),
            "speedup": round(bwd_ref / max(bwd_cust, 1e-9), 2),
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print("=== END-TO-END SegNet (real weights) custom Metal backward vs Python-loop ===")
    print(f"  forward parity absmax: {fwd_absmax:.2e}  (should be ~fp32 round-off)")
    print(
        f"  FULL-SegNet grad cosine: {grad_cos:.8f}  relmax {grad_relmax:.2e}  "
        f"(custom backward direction vs reference)"
    )
    print(f"  forward:  ref {fwd_ref*1e3:7.1f} ms  custom {fwd_cust*1e3:7.1f} ms  = {res['forward']['speedup']}x")
    print(
        f"  backward: ref {bwd_ref*1e3:7.1f} ms  custom {bwd_cust*1e3:7.1f} ms  = "
        f"{res['backward_value_and_grad']['speedup']}x"
    )
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
