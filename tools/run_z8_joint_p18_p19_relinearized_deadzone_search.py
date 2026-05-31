#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded relinearized Z8 joint P18/P19 coefficient dead-zone search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill import (
    Z8JointCoefficientRelinearizationSearchConfig,
    load_joint_p18_p19_surface_file,
    materialize_joint_p18_p19_relinearized_deadzone_search,
)


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return values


def _load_entropy_detail_steps(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    raw_steps = payload.get("entropy_detail_quantization_steps", payload)
    if not isinstance(raw_steps, dict):
        raise ValueError("entropy detail step JSON must be an object or contain entropy_detail_quantization_steps")
    return {str(key): float(value) for key, value in raw_steps.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument(
        "--surface",
        action="append",
        type=Path,
        default=[],
        help="Fresh joint surface per iteration; repeat for relinearization.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--joint-weight-quantiles",
        type=_float_tuple,
        default=_float_tuple("0.20,0.35,0.50"),
    )
    parser.add_argument(
        "--coefficient-deadzone-quantiles",
        type=_float_tuple,
        default=_float_tuple("0.25,0.50,0.75"),
    )
    parser.add_argument(
        "--quantization-steps",
        type=_float_tuple,
        default=_float_tuple("0.0039215686,0.0078431373,0.0156862745"),
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-cumulative-mse", type=float, default=None)
    parser.add_argument("--rate-weight", type=float, default=25.0)
    parser.add_argument("--distortion-weight", type=float, default=10_000.0)
    parser.add_argument("--interaction-penalty-weight", type=float, default=10_000.0)
    parser.add_argument(
        "--local-replay-prefilter-top-k",
        type=int,
        default=None,
        help=(
            "When MLX full-video replay is enabled, replay only the top-k "
            "receiver/rate proxy rows and mark skipped rows blocked."
        ),
    )
    parser.add_argument(
        "--measure-receiver-mse-proxy-with-mlx-replay",
        action="store_true",
        help=(
            "Keep the expensive receiver-MSE proxy even when full-video MLX "
            "replay will be the accept/reject authority."
        ),
    )
    parser.add_argument(
        "--skip-receiver-mse-proxy",
        action="store_true",
        help="Skip archive reconstruction MSE proxy; useful for rate-only entropy headroom probes.",
    )
    parser.add_argument("--allow-reused-surface", action="store_true")
    parser.add_argument(
        "--allow-stale-surfaces",
        action="store_true",
        help=(
            "Allow surfaces whose embedded linearization_archive_sha does not match "
            "the current archive. Exact full-video relinearization keeps this off."
        ),
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--allow-without-pose-null-mask", action="store_true")
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
            "Run storage-layout-only variants. Useful for lossless "
            "Brotli preconditioning probes where coefficient signal must not change."
        ),
    )
    parser.add_argument(
        "--entropy-code-quantized-details",
        action="store_true",
        help="Store Z8 wavelet detail bands with the v2 quantized entropy codec.",
    )
    parser.add_argument(
        "--entropy-detail-quantization-step",
        type=float,
        default=None,
        help=(
            "Storage quantization step for the v2 detail entropy codec. "
            "Defaults to the candidate quantization step when omitted."
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
            "Store float32 detail bands in byte-shuffled planes before Brotli. "
            "This is reversible and preserves coefficient signal exactly."
        ),
    )
    parser.add_argument(
        "--mlx-reference-pairs-npy",
        type=Path,
        help=(
            "Reference RGB pair grid for archive-fresh MLX VJP surface provider, "
            "shape (pairs,2,H,W,3). When set, --surface is not required."
        ),
    )
    parser.add_argument("--mlx-pair-chunk-size", type=int, default=64)
    parser.add_argument("--mlx-rgb-value-range", type=float, default=255.0)
    parser.add_argument("--mlx-scorer-hw", default="384,512")
    parser.add_argument("--mlx-seg-margin-delta", type=float, default=1.0)
    parser.add_argument("--mlx-pose-null-threshold", type=float, default=1e-8)
    parser.add_argument("--mlx-artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--mlx-replay-rate-source",
        choices=("byte_closed_zip", "payload_bytes"),
        default="byte_closed_zip",
        help="Rate byte source for full-video MLX local replay accept/reject.",
    )
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    return parser


def _parse_hw(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("expected H,W")
    return int(parts[0]), int(parts[1])


def main() -> int:
    args = build_parser().parse_args()
    surfaces = [load_joint_p18_p19_surface_file(path) for path in args.surface]
    entropy_detail_quantization_steps = _load_entropy_detail_steps(
        args.entropy_detail_quantization_steps_json
    )
    surface_provider = None
    local_replay_evaluator = None
    if args.mlx_reference_pairs_npy:
        import numpy as np

        from tac.local_acceleration.mlx_scorer_adapters import (
            load_mlx_distortion_scorer_adapter_from_upstream,
        )
        from tac.substrates.z8_hierarchical_predictive_coding.full_video_vjp_acquisition import (
            Z8FullVideoVjpAcquisitionConfig,
            build_z8_full_video_mlx_replay_evaluator,
            build_z8_full_video_mlx_surface_provider,
        )

        reference_pairs = np.load(args.mlx_reference_pairs_npy)
        mlx_scorer = load_mlx_distortion_scorer_adapter_from_upstream(args.upstream_dir, device="cpu")
        if not surfaces:
            surface_provider = build_z8_full_video_mlx_surface_provider(
                reference_pairs_rgb=reference_pairs,
                mlx_scorer=mlx_scorer,
                acquisition_config=Z8FullVideoVjpAcquisitionConfig(
                    pair_chunk_size=int(args.mlx_pair_chunk_size),
                ),
                rgb_value_range=float(args.mlx_rgb_value_range),
                scorer_hw=_parse_hw(args.mlx_scorer_hw),
                seg_margin_delta=float(args.mlx_seg_margin_delta),
                pose_null_threshold=float(args.mlx_pose_null_threshold),
                artifact_dir=args.mlx_artifact_dir,
            )
        local_replay_evaluator = build_z8_full_video_mlx_replay_evaluator(
            reference_pairs_rgb=reference_pairs,
            mlx_scorer=mlx_scorer,
            rgb_value_range=float(args.mlx_rgb_value_range),
            scorer_hw=_parse_hw(args.mlx_scorer_hw),
            pair_chunk_size=int(args.mlx_pair_chunk_size),
            rate_source=str(args.mlx_replay_rate_source),
            repo_root=args.repo_root,
            artifact_dir=args.mlx_artifact_dir,
        )
    elif not surfaces:
        raise SystemExit("--surface or --mlx-reference-pairs-npy is required")
    config = Z8JointCoefficientRelinearizationSearchConfig(
        joint_weight_quantiles=args.joint_weight_quantiles,
        coefficient_deadzone_quantiles=args.coefficient_deadzone_quantiles,
        quantization_steps=args.quantization_steps,
        max_iterations=args.max_iterations,
        max_cumulative_mse=args.max_cumulative_mse,
        rate_weight=args.rate_weight,
        distortion_weight=args.distortion_weight,
        interaction_penalty_weight=args.interaction_penalty_weight,
        require_fresh_surface_per_iteration=not args.allow_reused_surface,
        pose_null_required=not args.allow_without_pose_null_mask,
        max_pairs=args.max_pairs,
        emit_archive_zip=not args.no_archive_zip,
        emit_receiver_proof=args.emit_receiver_proof,
        require_full_video_surface_coverage=not args.allow_broadcast_surface,
        require_surface_archive_freshness=not args.allow_stale_surfaces,
        local_replay_prefilter_top_k=args.local_replay_prefilter_top_k,
        skip_receiver_mse_proxy_when_full_video_replay=not args.measure_receiver_mse_proxy_with_mlx_replay,
        mutate_coefficients=not args.no_mutate_coefficients,
        entropy_code_quantized_details=bool(args.entropy_code_quantized_details),
        entropy_detail_quantization_step=args.entropy_detail_quantization_step,
        entropy_detail_quantization_steps=entropy_detail_quantization_steps,
        measure_receiver_mse_proxy=not args.skip_receiver_mse_proxy,
        lossless_brotli_precondition_details=bool(args.lossless_brotli_precondition_details),
    )
    manifest = materialize_joint_p18_p19_relinearized_deadzone_search(
        args.archive_bin.read_bytes(),
        args.output_dir,
        surfaces=surfaces or None,
        surface_provider=surface_provider,
        local_replay_evaluator=local_replay_evaluator,
        config=config,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
