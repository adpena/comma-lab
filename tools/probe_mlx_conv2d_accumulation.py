#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe MLX Conv2d accumulation modes against a PyTorch Conv2d reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.local_acceleration.mlx_scorer_torch_parity import (  # noqa: E402
    MLXConv2dAccumulationThresholds,
    build_mlx_conv2d_accumulation_probe_manifest,
    write_mlx_conv2d_accumulation_probe_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--allow-gpu-research-signal",
        action="store_true",
        help="Permit --device gpu as local MLX research-signal evidence only.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=6)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--width", type=int, default=13)
    parser.add_argument("--kernel-height", type=int, default=3)
    parser.add_argument("--kernel-width", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--max-optimized-abs-delta", type=float, default=1.0e-3)
    parser.add_argument("--max-fixed-fp32-abs-delta", type=float, default=1.0e-4)
    parser.add_argument("--max-kahan-fp32-abs-delta", type=float, default=1.0e-4)
    parser.add_argument("--max-fixed-fp64-abs-delta", type=float, default=1.0e-5)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument(
        "--torch-allow-tf32",
        choices=("unset", "true", "false"),
        default="unset",
    )
    parser.add_argument(
        "--torch-cudnn-benchmark",
        choices=("unset", "true", "false"),
        default="unset",
    )
    parser.add_argument(
        "--torch-cudnn-deterministic",
        choices=("unset", "true", "false"),
        default="unset",
    )
    parser.add_argument(
        "--torch-mkldnn-enabled",
        choices=("unset", "true", "false"),
        default="unset",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        x, weight, bias = _synthetic_conv2d_case(args)
        manifest = build_mlx_conv2d_accumulation_probe_manifest(
            input_nchw=x,
            weight_oihw=weight,
            bias=bias,
            stride=args.stride,
            padding=args.padding,
            dilation=args.dilation,
            groups=args.groups,
            device_type=args.device,
            torch_device_type=args.torch_device,
            thresholds=MLXConv2dAccumulationThresholds(
                max_optimized_abs_delta=args.max_optimized_abs_delta,
                max_fixed_fp32_abs_delta=args.max_fixed_fp32_abs_delta,
                max_kahan_fp32_abs_delta=args.max_kahan_fp32_abs_delta,
                max_fixed_fp64_abs_delta=args.max_fixed_fp64_abs_delta,
            ),
            run_id=args.run_id,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
            use_deterministic_algorithms=True,
            cudnn_benchmark=_tri_bool(args.torch_cudnn_benchmark),
            cudnn_deterministic=_tri_bool(args.torch_cudnn_deterministic),
            allow_tf32=_tri_bool(args.torch_allow_tf32),
            mkldnn_enabled=_tri_bool(args.torch_mkldnn_enabled),
            torch_num_threads=args.torch_num_threads,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_mlx_conv2d_accumulation_probe_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "verdict": manifest["verdict"],
                "rows": [
                    {
                        "mode": row["mode"],
                        "max_abs_delta": row["max_abs_delta"],
                        "passed": row["passed"],
                    }
                    for row in manifest["rows"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


def _synthetic_conv2d_case(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    for name in (
        "batch",
        "in_channels",
        "out_channels",
        "height",
        "width",
        "kernel_height",
        "kernel_width",
        "stride",
        "dilation",
        "groups",
    ):
        value = int(getattr(args, name))
        if value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1, got {value}")
    if int(args.padding) < 0:
        raise ValueError(f"padding must be >= 0, got {args.padding}")
    if int(args.in_channels) % int(args.groups) != 0:
        raise ValueError("in-channels must be divisible by groups")
    if int(args.out_channels) % int(args.groups) != 0:
        raise ValueError("out-channels must be divisible by groups")

    rng = np.random.default_rng(int(args.seed))
    x = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(int(args.batch), int(args.in_channels), int(args.height), int(args.width)),
    ).astype(np.float32)
    weight = rng.normal(
        loc=0.0,
        scale=0.2,
        size=(
            int(args.out_channels),
            int(args.in_channels) // int(args.groups),
            int(args.kernel_height),
            int(args.kernel_width),
        ),
    ).astype(np.float32)
    bias = (
        None
        if bool(args.no_bias)
        else rng.normal(loc=0.0, scale=0.05, size=(int(args.out_channels),)).astype(np.float32)
    )
    return x, weight, bias


def _tri_bool(value: str) -> bool | None:
    if value == "unset":
        return None
    return value == "true"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
