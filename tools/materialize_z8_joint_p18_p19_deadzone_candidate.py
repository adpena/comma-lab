#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a Z8 joint P18/P19 coefficient dead-zone archive candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.entropy_delta_schedule import (
    load_entropy_detail_quantization_steps_json,
)
from tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill import (
    Z8JointCoefficientWaterfillConfig,
    load_joint_p18_p19_surface_file,
    materialize_joint_p18_p19_deadzone_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument(
        "--surface",
        type=Path,
        default=None,
        help="Fresh P18/P19 surface. Required unless --no-mutate-coefficients is set.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--joint-weight-quantile", type=float, default=0.35)
    parser.add_argument("--coefficient-deadzone-quantile", type=float, default=0.50)
    parser.add_argument("--quantization-step", type=float, default=1.0 / 255.0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--allow-stale-surface",
        action="store_true",
        help=(
            "Allow a surface whose embedded linearization_archive_sha does not match "
            "the archive. Exact full-video acquisition keeps this off."
        ),
    )
    parser.add_argument("--allow-without-pose-null-mask", action="store_true")
    parser.add_argument(
        "--allow-non-true-p19-surface",
        action="store_true",
        help="Allow legacy/scalar pose proxy surfaces. Exact Z8 codec work keeps this off.",
    )
    parser.add_argument(
        "--allow-broadcast-surface",
        action="store_true",
        help="Allow exploratory pair-broadcast surfaces; exact acquisition keeps this off.",
    )
    parser.add_argument("--no-archive-zip", action="store_true")
    parser.add_argument("--emit-receiver-proof", action="store_true")
    parser.add_argument(
        "--no-mutate-coefficients",
        action="store_true",
        help=(
            "Run a storage-layout-only materialization. Useful for lossless "
            "Brotli preconditioning probes where coefficient signal must not change."
        ),
    )
    parser.add_argument(
        "--entropy-code-quantized-details",
        action="store_true",
        help="Store Z8 detail subbands with the v2 quantized per-subband entropy codec.",
    )
    parser.add_argument(
        "--entropy-detail-quantization-step",
        type=float,
        default=None,
        help=(
            "Storage quantization step for the v2 detail entropy codec. "
            "Defaults to --quantization-step when omitted."
        ),
    )
    parser.add_argument(
        "--entropy-detail-quantization-steps-json",
        type=Path,
        default=None,
        help=(
            "JSON object or schedule report containing entropy_detail_quantization_steps "
            "keyed as frame_0_details:level:lh. Mutually exclusive with "
            "--entropy-detail-quantization-step."
        ),
    )
    parser.add_argument(
        "--lossless-brotli-precondition-details",
        action="store_true",
        help=(
            "Store float32 detail subbands in byte-shuffled planes before Brotli. "
            "This is reversible and preserves coefficient signal exactly."
        ),
    )
    parser.add_argument(
        "--skip-inflate-runtime-benchmark-work-order",
        action="store_true",
        help="Do not emit the advisory full inflate.sh runtime benchmark work order.",
    )
    parser.add_argument(
        "--run-inflate-runtime-benchmark",
        action="store_true",
        help=(
            "Run the advisory full inflate.sh benchmark immediately. This can "
            "write contest-sized raw output."
        ),
    )
    parser.add_argument("--inflate-runtime-benchmark-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--inflate-runtime-benchmark-auth-window-seconds", type=float, default=1800.0)
    parser.add_argument("--inflate-runtime-benchmark-device", default="cpu")
    parser.add_argument(
        "--print-full-manifest",
        action="store_true",
        help="Print the full manifest JSON to stdout instead of the compact operator summary.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.surface is None:
        if not args.no_mutate_coefficients:
            raise SystemExit("--surface is required unless --no-mutate-coefficients is set")
        surface = None
    else:
        surface = load_joint_p18_p19_surface_file(args.surface)
    entropy_detail_quantization_steps = load_entropy_detail_quantization_steps_json(
        args.entropy_detail_quantization_steps_json
    )
    config = Z8JointCoefficientWaterfillConfig(
        joint_weight_quantile=args.joint_weight_quantile,
        coefficient_deadzone_quantile=args.coefficient_deadzone_quantile,
        quantization_step=args.quantization_step,
        pose_null_required=not args.allow_without_pose_null_mask,
        max_pairs=args.max_pairs,
        emit_archive_zip=not args.no_archive_zip,
        emit_receiver_proof=args.emit_receiver_proof,
        mutate_coefficients=not args.no_mutate_coefficients,
        require_full_video_surface_coverage=not args.allow_broadcast_surface,
        require_surface_archive_freshness=not args.allow_stale_surface,
        require_true_p19_pose_surface=not args.allow_non_true_p19_surface,
        entropy_code_quantized_details=bool(args.entropy_code_quantized_details),
        entropy_detail_quantization_step=args.entropy_detail_quantization_step,
        entropy_detail_quantization_steps=entropy_detail_quantization_steps,
        lossless_brotli_precondition_details=bool(args.lossless_brotli_precondition_details),
        emit_inflate_runtime_benchmark_work_order=not args.skip_inflate_runtime_benchmark_work_order,
        run_inflate_runtime_benchmark=bool(args.run_inflate_runtime_benchmark),
        inflate_runtime_benchmark_timeout_seconds=float(
            args.inflate_runtime_benchmark_timeout_seconds
        ),
        inflate_runtime_benchmark_auth_window_seconds=float(
            args.inflate_runtime_benchmark_auth_window_seconds
        ),
        inflate_runtime_benchmark_device=str(args.inflate_runtime_benchmark_device),
    )
    manifest = materialize_joint_p18_p19_deadzone_candidate(
        args.archive_bin.read_bytes(),
        args.output_dir,
        joint_weight=surface,
        config=config,
        repo_root=args.repo_root,
    )
    if args.print_full_manifest:
        print(json.dumps(manifest, sort_keys=True))
    else:
        rate_report = (manifest.get("waterfill_result") or {}).get("rate_report") or {}
        print(
            json.dumps(
                {
                    "schema": manifest.get("schema"),
                    "manifest_path": manifest.get("manifest_path"),
                    "candidate_bin_path": manifest.get("candidate_bin_path"),
                    "candidate_bin_bytes": manifest.get("candidate_bin_bytes"),
                    "archive_zip_path": manifest.get("archive_zip_path"),
                    "archive_zip_bytes": manifest.get("archive_zip_bytes"),
                    "archive_zip_sha256": manifest.get("archive_zip_sha256"),
                    "archive_byte_delta": rate_report.get("archive_byte_delta"),
                    "archive_rate_ratio": rate_report.get("archive_rate_ratio"),
                    "wavelet_blob_byte_delta": rate_report.get("wavelet_blob_byte_delta"),
                    "receiver_proof_executed": manifest.get("receiver_proof_executed"),
                    "inflate_runtime_benchmark_executed": manifest.get(
                        "inflate_runtime_benchmark_executed"
                    ),
                    "exact_axis_blocker": manifest.get("exact_axis_blocker"),
                    "score_claim": manifest.get("score_claim"),
                    "ready_for_exact_eval_dispatch": manifest.get(
                        "ready_for_exact_eval_dispatch"
                    ),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
