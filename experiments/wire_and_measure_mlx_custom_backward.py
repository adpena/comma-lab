"""Wire the custom Metal grouped-conv backward into the MLX SegNet and measure
the REAL end-to-end backward speedup + validate descent-direction correctness.

The sister module ``tac.local_acceleration.metal_grouped_conv_backward`` defines
the Metal kernels but its ``@mx.custom_function`` passes stride/padding/etc. as
keyword-only args that MLX's ``.vjp`` does NOT forward (the vjp signature is
``(primals, cotangent, output)`` only). This harness fixes that with a
closure-per-config factory, validates the gradient against the trusted
Python-loop reference on the REAL strided-grouped scorer shapes, and measures
the end-to-end SegNet backward with the custom kernel swapped in for the 4
fallback layers.

Authority: torch-CPU exact = d_seg authority. All numbers here are
`[macOS-MLX research-signal]` throughput / gradient-fidelity. NO score claim.
NO MPS.
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


def make_grouped_conv2d(stride, padding, dilation, groups):
    """Return a config-bound ``@mx.custom_function`` whose vjp closes over config.

    This is the fix for the sister module's vjp-kwarg bug: MLX's
    ``custom_function.vjp`` receives only ``(primals, cotangent, output)``, so
    the conv config must be captured by closure, not passed as keyword-only args.
    """
    import mlx.core as mx

    from tac.local_acceleration.metal_grouped_conv_backward import _kernels

    s = _pair(stride)
    p = _pair(padding)
    d = _pair(dilation)
    g = int(groups)

    sh = mx.array([s[0], s[1]], dtype=mx.int32)
    pd = mx.array([p[0], p[1]], dtype=mx.int32)
    dl = mx.array([d[0], d[1]], dtype=mx.int32)
    gp = mx.array([g], dtype=mx.int32)

    @mx.custom_function
    def conv(x, weight):
        return mx.conv2d(x, weight, stride=s, padding=p, dilation=d, groups=g)

    @conv.vjp
    def conv_vjp(primals, cotangent, output):
        x, weight = primals
        grad_input_kernel, grad_weight_kernel = _kernels()
        (gx,) = grad_input_kernel(
            inputs=[x, weight, cotangent, sh, pd, dl, gp],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            grid=(int(x.size), 1, 1),
            threadgroup=(256, 1, 1),
        )
        (gw,) = grad_weight_kernel(
            inputs=[x, weight, cotangent, sh, pd, dl, gp],
            output_shapes=[weight.shape],
            output_dtypes=[weight.dtype],
            grid=(int(weight.size), 1, 1),
            threadgroup=(256, 1, 1),
        )
        return gx, gw

    return conv


# The 4 SegNet strided-depthwise fallback layers (from the profile).
SEGNET_FALLBACK = [
    # name, N, Hin, Win, Cin, Cout, kH, kW, groups, stride
    ("segnet.blocks1.0.conv_dw", 4, 192, 256, 96, 96, 3, 3, 96, 2),
    ("segnet.blocks2.0.conv_dw", 4, 96, 128, 144, 144, 5, 5, 144, 2),
    ("segnet.blocks3.0.conv_dw", 4, 48, 64, 288, 288, 3, 3, 288, 2),
    ("segnet.blocks5.0.conv_dw", 4, 24, 32, 720, 720, 5, 5, 720, 2),
]


def validate_and_time(reps: int) -> dict[str, Any]:
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import mlx_reference_conv2d_nhwc

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(0)
    rows = []
    for (name, N, Hin, Win, Cin, Cout, kH, kW, groups, stride) in SEGNET_FALLBACK:
        Ipg = Cin // groups
        pad = (kH // 2, kW // 2)
        x = mx.array(rng.standard_normal((N, Hin, Win, Cin)).astype(np.float32))
        w = mx.array(rng.standard_normal((Cout, kH, kW, Ipg)).astype(np.float32))

        conv = make_grouped_conv2d(stride, pad, 1, groups)

        def cust_loss(xx, ww):
            return mx.sum(conv(xx, ww) ** 2)

        def ref_loss(xx, ww):
            return mx.sum(
                mlx_reference_conv2d_nhwc(
                    xx, ww, None, stride=stride, padding=pad, dilation=1, groups=groups
                )
                ** 2
            )

        # --- correctness: gradient direction vs trusted reference ---
        gx_c, gw_c = mx.grad(cust_loss, argnums=(0, 1))(x, w)
        gx_r, gw_r = mx.grad(ref_loss, argnums=(0, 1))(x, w)
        mx.eval(gx_c, gw_c, gx_r, gw_r)

        def cos(a, b):
            a = np.asarray(a).ravel().astype(np.float64)
            b = np.asarray(b).ravel().astype(np.float64)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return float("nan")
            return float(np.dot(a, b) / (na * nb))

        gx_cos = cos(gx_c, gx_r)
        gw_cos = cos(gw_c, gw_r)
        gx_absdiff = float(np.abs(np.asarray(gx_c) - np.asarray(gx_r)).max())
        gw_absdiff = float(np.abs(np.asarray(gw_c) - np.asarray(gw_r)).max())
        gx_relmax = gx_absdiff / max(float(np.abs(np.asarray(gx_r)).max()), 1e-9)

        # --- speed: grad fwd+bwd of custom vs python-loop reference ---
        def time_grad(fn):
            gf = mx.grad(fn, argnums=(0, 1))
            a, b = gf(x, w)
            mx.eval(a, b)
            t0 = time.perf_counter()
            for _ in range(reps):
                a, b = gf(x, w)
                mx.eval(a, b)
            return (time.perf_counter() - t0) / reps

        t_cust = time_grad(cust_loss)
        t_ref = time_grad(ref_loss)

        rows.append(
            {
                "name": name,
                "shape": f"{N}x{Hin}x{Win}x{Cin}->{Cout} k{kH}x{kW} g{groups} s{stride}",
                "grad_input_cosine": round(gx_cos, 6),
                "grad_weight_cosine": round(gw_cos, 6),
                "grad_input_relmax": round(gx_relmax, 6),
                "grad_input_absdiff": gx_absdiff,
                "grad_weight_absdiff": gw_absdiff,
                "custom_grad_ms": round(t_cust * 1e3, 3),
                "reference_loop_grad_ms": round(t_ref * 1e3, 3),
                "speedup": round(t_ref / max(t_cust, 1e-9), 1),
            }
        )
        print(
            f"  {name}: gx_cos={gx_cos:.6f} gw_cos={gw_cos:.6f} "
            f"relmax={gx_relmax:.2e} | custom {t_cust*1e3:.2f}ms vs loop {t_ref*1e3:.1f}ms "
            f"= {t_ref/max(t_cust,1e-9):.0f}x"
        )

    return {"layers": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--out-json",
        default=str(REPO / "experiments" / "results" / "mlx_custom_backward_validation.json"),
    )
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(UPSTREAM))

    print("=== validate + time custom Metal backward on the 4 SegNet fallback shapes ===")
    res = validate_and_time(args.reps)
    res["evidence_grade"] = "macOS-MLX research-signal"

    sum_cust = sum(r["custom_grad_ms"] for r in res["layers"])
    sum_ref = sum(r["reference_loop_grad_ms"] for r in res["layers"])
    res["sum_custom_grad_ms"] = round(sum_cust, 3)
    res["sum_reference_loop_grad_ms"] = round(sum_ref, 3)
    res["aggregate_speedup"] = round(sum_ref / max(sum_cust, 1e-9), 1)
    res["all_grad_input_cosine_above_0_999"] = all(
        r["grad_input_cosine"] > 0.999 for r in res["layers"]
    )
    res["all_grad_weight_cosine_above_0_999"] = all(
        r["grad_weight_cosine"] > 0.999 for r in res["layers"]
    )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(
        f"\n=== AGGREGATE (4 fallback layers) ===\n"
        f"  python-loop grad: {sum_ref:8.1f} ms\n"
        f"  custom metal grad:{sum_cust:8.1f} ms\n"
        f"  speedup:          {res['aggregate_speedup']:8.1f}x\n"
        f"  grad_input cos>0.999 all: {res['all_grad_input_cosine_above_0_999']}\n"
        f"  grad_weight cos>0.999 all: {res['all_grad_weight_cosine_above_0_999']}\n"
    )
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
