#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a materializer-ready Z8 per-subband detail Δ schedule from RD rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.entropy_delta_schedule import (
    build_entropy_delta_materializer_work_order,
    build_entropy_delta_schedule_from_headroom_report,
)
from tac.substrates.z8_hierarchical_predictive_coding.per_subband_rd_waterfill_solver import (
    build_rd_waterfill_schedule_from_headroom_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headroom-json", required=True, type=Path)
    parser.add_argument(
        "--max-subband-mse",
        type=float,
        default=None,
        help=(
            "Per-aggregate-subband quantization distortion ceiling used for "
            "legacy independent min-bytes selection."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=("legacy-max-subband-mse", "rd-waterfill"),
        default="legacy-max-subband-mse",
    )
    parser.add_argument(
        "--target-total-bytes",
        type=float,
        default=None,
        help="RD-waterfill total detail-byte target.",
    )
    parser.add_argument(
        "--target-detail-byte-fraction",
        type=float,
        default=None,
        help="RD-waterfill target as a fraction of measured raw-f32 detail bytes.",
    )
    parser.add_argument(
        "--max-weighted-mse",
        type=float,
        default=None,
        help="RD-waterfill weighted-mean MSE ceiling.",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=None,
        help="RD-waterfill fixed Lagrange multiplier.",
    )
    parser.add_argument(
        "--rate-field",
        default="live_codec_brotli_bytes_per_coeff",
        help="RD-waterfill per-quant-row bytes/coeff field to optimize.",
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--materializer-work-order-out",
        type=Path,
        default=None,
        help="Optional JSON work order that executes this ready schedule through the Z8 materializer.",
    )
    parser.add_argument(
        "--archive-bin",
        type=Path,
        default=None,
        help=(
            "Source Z8HPC1 0.bin for optional materializer work order. "
            "Defaults to source_archive_path from the headroom report."
        ),
    )
    parser.add_argument(
        "--materializer-output-dir",
        type=Path,
        default=None,
        help="Output directory for optional materializer work order.",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--materializer-without-receiver-proof",
        action="store_true",
        help=(
            "Emit an advisory non-executable work order without receiver proof. "
            "Queue execution stays blocked."
        ),
    )
    parser.add_argument(
        "--materializer-run-inflate-runtime-benchmark",
        action="store_true",
        help="Add --run-inflate-runtime-benchmark to the optional materializer work order.",
    )
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help=(
            "Permit schedules from sampled headroom reports to be marked materializer-ready. "
            "Default is fail-closed when pairs_measured < total_pairs_in_archive."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = json.loads(args.headroom_json.read_text())
    if args.strategy == "legacy-max-subband-mse":
        if args.max_subband_mse is None:
            raise SystemExit("--max-subband-mse is required for legacy-max-subband-mse")
        schedule = build_entropy_delta_schedule_from_headroom_report(
            report,
            max_subband_mse=float(args.max_subband_mse),
            require_full_archive_coverage=not args.allow_partial_coverage,
        )
    else:
        schedule = build_rd_waterfill_schedule_from_headroom_report(
            report,
            target_total_bytes=args.target_total_bytes,
            target_detail_byte_fraction=args.target_detail_byte_fraction,
            max_weighted_mse=args.max_weighted_mse,
            lambda_value=args.lambda_value,
            rate_field=str(args.rate_field),
            require_full_archive_coverage=not args.allow_partial_coverage,
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(schedule, indent=2, sort_keys=True))
    work_order: dict[str, object] | None = None
    if args.materializer_work_order_out is not None:
        if args.materializer_output_dir is None:
            raise SystemExit(
                "--materializer-output-dir is required with --materializer-work-order-out"
            )
        work_order = build_entropy_delta_materializer_work_order(
            schedule,
            schedule_json_path=args.out_json,
            output_dir=args.materializer_output_dir,
            archive_bin=args.archive_bin,
            repo_root=args.repo_root,
            emit_receiver_proof=not bool(args.materializer_without_receiver_proof),
            run_inflate_runtime_benchmark=bool(
                args.materializer_run_inflate_runtime_benchmark
            ),
        )
        args.materializer_work_order_out.parent.mkdir(parents=True, exist_ok=True)
        args.materializer_work_order_out.write_text(
            json.dumps(work_order, indent=2, sort_keys=True)
        )
    print(
        json.dumps(
            {
                "schema": schedule["schema"],
                "strategy": args.strategy,
                "ready_for_materializer": schedule["ready_for_materializer"],
                "schedule_sha256": schedule["schedule_sha256"],
                "step_count": len(schedule["entropy_detail_quantization_steps"]),
                "blockers": schedule["blockers"],
                "out_json": args.out_json.as_posix(),
                "materializer_work_order_out": (
                    args.materializer_work_order_out.as_posix()
                    if args.materializer_work_order_out is not None
                    else None
                ),
                "materializer_ready": (
                    work_order.get("ready_for_materializer_execution")
                    if work_order is not None
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
