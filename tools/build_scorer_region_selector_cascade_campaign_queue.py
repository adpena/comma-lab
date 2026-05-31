#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a grouped P18/P19/P11/P15 cascade campaign experiment queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.scorer_region_selector_cascade_campaign_queue import (  # noqa: E402
    ScorerRegionSelectorCascadeCampaignQueueError,
    build_scorer_region_selector_cascade_campaign_queue,
)
from comma_lab.scheduler.scorer_region_selector_chain_queue import (  # noqa: E402
    ScorerRegionSelectorChainQueueError,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def _triple(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("delta must contain three comma-separated ints")
    return (parts[0], parts[1], parts[2])


def _csv_group(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise argparse.ArgumentTypeError("codec family set cannot be empty")
    return parts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-out", required=True, type=Path)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-waterfill-work-order", type=Path)
    parser.add_argument("--full-frame-inflate-parity-proof", type=Path)
    parser.add_argument("--pose-null-modes-artifact", required=True, type=Path)
    parser.add_argument("--segnet-softmax-16", required=True, type=Path)
    parser.add_argument("--segnet-softmax-256", required=True, type=Path)
    parser.add_argument("--null-fraction", action="append", type=float, default=[])
    parser.add_argument("--top-regions-per-pair", action="append", type=int, default=[])
    parser.add_argument("--receiver-patch-max-pairs", action="append", type=int, default=[])
    parser.add_argument("--receiver-patch-regions-per-pair", action="append", type=int, default=[])
    parser.add_argument("--receiver-patch-rgb-delta", action="append", type=_triple, default=[])
    parser.add_argument("--receiver-patch-yuv-delta", action="append", type=_triple, default=[])
    parser.add_argument(
        "--selector-codec-family-set",
        action="append",
        type=_csv_group,
        default=[],
        help="Comma-separated codec family group; may repeat.",
    )
    parser.add_argument("--scale", action="append", type=int, default=[])
    parser.add_argument("--alpha", action="append", type=int, default=[])
    parser.add_argument(
        "--repack-order",
        action="append",
        default=[],
        help="Currently supported: p11_then_p15_then_receiver_patch.",
    )
    parser.add_argument("--max-variants", type=int, default=32)
    parser.add_argument("--exhaustive-grid", action="store_true")
    parser.add_argument("--prove-receiver-patch-output-change", action="store_true")
    parser.add_argument("--receiver-patch-output-change-file-list-entry", action="append", default=[])
    parser.add_argument("--receiver-patch-output-change-expected-file-list-sha256")
    parser.add_argument("--receiver-patch-output-change-expected-entry-count", type=int)
    parser.add_argument("--receiver-patch-output-change-file-list-source")
    parser.add_argument(
        "--receiver-patch-output-change-parity-scope-kind",
        default="contest_full_sample",
    )
    parser.add_argument(
        "--receiver-patch-output-change-contest-full-sample-claim",
        action="store_true",
    )
    parser.add_argument("--include-local-component-loop", action="store_true")
    parser.add_argument("--local-component-upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument(
        "--local-component-video-names-file",
        type=Path,
        default=Path("upstream/public_test_video_names.txt"),
    )
    parser.add_argument("--local-component-inflate-timeout-seconds", type=int, default=1800)
    parser.add_argument("--local-component-evaluate-timeout-seconds", type=int, default=1800)
    parser.add_argument("--include-mlx-component-response", action="store_true")
    parser.add_argument(
        "--mlx-reference-cache-dir",
        type=Path,
        default=Path(
            "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
        ),
    )
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mlx-cache-batch-pairs", type=int, default=1)
    parser.add_argument("--mlx-batch-pairs", type=int, default=1)
    parser.add_argument("--mlx-max-pairs", type=int, default=12)
    parser.add_argument("--mlx-full-sample", action="store_true")
    parser.add_argument("--include-scorer-response-dataset", action="store_true")
    parser.add_argument("--scorer-response-baseline-score", type=float)
    parser.add_argument("--scorer-response-baseline-archive-bytes", type=int)
    parser.add_argument("--include-local-component-retention-plan", action="store_true")
    parser.add_argument("--execute-local-component-retention", action="store_true")
    parser.add_argument(
        "--local-component-retention-action",
        choices=("move", "delete"),
        default="move",
    )
    parser.add_argument("--local-component-retention-min-bytes", default="1")
    parser.add_argument(
        "--local-component-retention-cold-store-root",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--local-component-retention-cold-store-reserve-gb", type=float, default=40.0)
    parser.add_argument("--max-concurrency-local-cpu", type=int, default=2)
    parser.add_argument("--max-concurrency-local-mlx", type=int, default=1)
    parser.add_argument("--max-concurrency-local-io-heavy", type=int, default=1)
    parser.add_argument("--no-campaign-harvest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = build_scorer_region_selector_cascade_campaign_queue(
            repo_root=REPO_ROOT,
            queue_id=args.queue_id,
            source_submission_dir=args.source_submission_dir,
            output_root=args.output_root,
            source_waterfill_work_order=args.source_waterfill_work_order,
            full_frame_inflate_parity_proof=args.full_frame_inflate_parity_proof,
            pose_null_modes_artifact=args.pose_null_modes_artifact,
            segnet_softmax_16=args.segnet_softmax_16,
            segnet_softmax_256=args.segnet_softmax_256,
            null_fractions=tuple(args.null_fraction or [0.05, 0.10, 0.20]),
            top_regions_per_pair_values=tuple(args.top_regions_per_pair or [2, 4]),
            receiver_patch_max_pair_values=tuple(args.receiver_patch_max_pairs or [12, 24, 48]),
            receiver_patch_regions_per_pair_values=tuple(
                args.receiver_patch_regions_per_pair or [1, 2]
            ),
            receiver_patch_rgb_deltas=tuple(
                args.receiver_patch_rgb_delta or [(-1, -1, -1), (1, 1, 1)]
            ),
            receiver_patch_yuv_deltas=tuple(args.receiver_patch_yuv_delta),
            selector_codec_family_groups=tuple(
                args.selector_codec_family_set
                or [
                    ("fec10_adaptive_blend",),
                    ("fec8_markov_static_order1",),
                    ("fec8_markov_adaptive_order1",),
                    (
                        "fec10_adaptive_blend",
                        "fec8_markov_static_order1",
                        "fec8_markov_adaptive_order1",
                        "fec8_markov_static_order2",
                    ),
                ]
            ),
            scales=tuple(args.scale or [32, 64, 128, 256]),
            alphas=tuple(args.alpha or [1, 2, 4]),
            repack_orders=tuple(args.repack_order or ["p11_then_p15_then_receiver_patch"]),
            max_variants=None if args.exhaustive_grid else args.max_variants,
            prove_receiver_patch_output_change=args.prove_receiver_patch_output_change,
            receiver_patch_output_change_file_list_entries=tuple(
                args.receiver_patch_output_change_file_list_entry
            )
            or ("0.raw",),
            receiver_patch_output_change_expected_file_list_sha256=(
                args.receiver_patch_output_change_expected_file_list_sha256
            ),
            receiver_patch_output_change_expected_entry_count=(
                args.receiver_patch_output_change_expected_entry_count
            ),
            receiver_patch_output_change_file_list_source=(
                args.receiver_patch_output_change_file_list_source
            ),
            receiver_patch_output_change_parity_scope_kind=(
                args.receiver_patch_output_change_parity_scope_kind
            ),
            receiver_patch_output_change_contest_full_sample_claim=(
                args.receiver_patch_output_change_contest_full_sample_claim
            ),
            include_local_component_loop=args.include_local_component_loop,
            local_component_upstream_dir=args.local_component_upstream_dir,
            local_component_video_names_file=args.local_component_video_names_file,
            local_component_inflate_timeout_seconds=(
                args.local_component_inflate_timeout_seconds
            ),
            local_component_evaluate_timeout_seconds=(
                args.local_component_evaluate_timeout_seconds
            ),
            include_mlx_component_response=args.include_mlx_component_response,
            mlx_reference_cache_dir=args.mlx_reference_cache_dir,
            mlx_device=args.mlx_device,
            mlx_cache_batch_pairs=args.mlx_cache_batch_pairs,
            mlx_batch_pairs=args.mlx_batch_pairs,
            mlx_max_pairs=None if args.mlx_full_sample else args.mlx_max_pairs,
            include_scorer_response_dataset=args.include_scorer_response_dataset,
            scorer_response_baseline_score=args.scorer_response_baseline_score,
            scorer_response_baseline_archive_bytes=(
                args.scorer_response_baseline_archive_bytes
            ),
            include_local_component_retention_plan=(
                args.include_local_component_retention_plan
            ),
            execute_local_component_retention=args.execute_local_component_retention,
            local_component_retention_action=args.local_component_retention_action,
            local_component_retention_min_bytes=args.local_component_retention_min_bytes,
            local_component_retention_cold_store_roots=tuple(
                args.local_component_retention_cold_store_root
            ),
            local_component_retention_cold_store_reserve_gb=(
                args.local_component_retention_cold_store_reserve_gb
            ),
            max_concurrency_local_cpu=args.max_concurrency_local_cpu,
            max_concurrency_local_mlx=args.max_concurrency_local_mlx,
            max_concurrency_local_io_heavy=args.max_concurrency_local_io_heavy,
            append_campaign_harvest=not args.no_campaign_harvest,
        )
        queue_out = _resolve(args.queue_out)
        expected_existing_sha256 = (
            sha256_file(queue_out) if queue_out.is_file() and args.overwrite else None
        )
        write = write_json_artifact(
            queue_out,
            queue,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        ScorerRegionSelectorCascadeCampaignQueueError,
        ScorerRegionSelectorChainQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: scorer-region cascade campaign queue failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "scorer_region_selector_cascade_campaign_queue_cli_result.v1",
                "queue_out": str(args.queue_out),
                "queue_id": queue["queue_id"],
                "experiment_count": len(queue["experiments"]),
                "variant_count": queue["metadata"]["variant_count"],
                "bytes_written": write.bytes_written,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
