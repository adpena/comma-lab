#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Trace upstream PyTorch SegNet batch-vs-per-sample-loop behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_scorer_torch_parity import (
    build_torch_segnet_batch_invariance_manifest,
    write_torch_segnet_batch_invariance_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-segnet-argmax-diff-pixels", type=int, default=0)
    parser.add_argument("--max-segnet-logit-abs-delta", type=float, default=None)
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Enable torch.use_deterministic_algorithms(True) for this run.",
    )
    parser.add_argument(
        "--disable-cudnn-benchmark",
        action="store_true",
        help="Set torch.backends.cudnn.benchmark=False for this run.",
    )
    parser.add_argument(
        "--enable-cudnn-deterministic",
        action="store_true",
        help="Set torch.backends.cudnn.deterministic=True for this run.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable CUDA matmul and cuDNN TF32 for this run.",
    )
    parser.add_argument(
        "--disable-mkldnn",
        action="store_true",
        help="Set torch.backends.mkldnn.enabled=False for this run.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Temporarily set torch.set_num_threads(N) for this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = build_torch_segnet_batch_invariance_manifest(
            cache_dir=args.cache_dir,
            repo_root=args.repo_root,
            device_type=args.device,
            start_pair=args.start_pair,
            max_pairs=args.max_pairs,
            run_id=args.run_id,
            max_segnet_argmax_diff_pixels=args.max_segnet_argmax_diff_pixels,
            max_segnet_logit_abs_delta=args.max_segnet_logit_abs_delta,
            use_deterministic_algorithms=args.use_deterministic_algorithms,
            cudnn_benchmark=False if args.disable_cudnn_benchmark else None,
            cudnn_deterministic=True if args.enable_cudnn_deterministic else None,
            allow_tf32=False if args.disable_tf32 else None,
            mkldnn_enabled=False if args.disable_mkldnn else None,
            torch_num_threads=args.torch_num_threads,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_torch_segnet_batch_invariance_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "passed": manifest["passed"],
                "verdict": manifest["verdict"],
                "pair_window": manifest["pair_window"],
                "device_type": manifest["device_type"],
                "segnet_argmax_diff_pixels": manifest["deltas"][
                    "segnet_argmax_diff_pixels"
                ],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
