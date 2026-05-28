#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an experiment_queue.v1 for P18/P19 -> P11 -> P15 chains."""

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

from comma_lab.scheduler.scorer_region_selector_chain_queue import (  # noqa: E402
    ScorerRegionSelectorChainQueueError,
    build_scorer_region_selector_chain_queue,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-out", required=True, type=Path)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-waterfill-work-order", type=Path)
    parser.add_argument("--full-frame-inflate-parity-proof", type=Path)
    parser.add_argument("--posenet-null-pairs", type=Path)
    parser.add_argument("--segnet-region-masks", type=Path)
    parser.add_argument("--selector-region-bits", type=Path)
    parser.add_argument("--pose-null-modes-artifact", type=Path)
    parser.add_argument("--segnet-softmax-16", type=Path)
    parser.add_argument("--segnet-softmax-256", type=Path)
    parser.add_argument("--materialize-upstream-artifacts", action="store_true")
    parser.add_argument("--materialize-receiver-patch", action="store_true")
    parser.add_argument("--null-fraction", type=float, default=0.10)
    parser.add_argument("--top-regions-per-pair", type=int, default=4)
    parser.add_argument("--receiver-patch-max-pairs", type=int, default=12)
    parser.add_argument("--receiver-patch-regions-per-pair", type=int, default=1)
    parser.add_argument("--receiver-patch-rgb-delta", default="-1,-1,-1")
    parser.add_argument("--prove-receiver-patch-output-change", action="store_true")
    parser.add_argument(
        "--receiver-patch-output-change-file-list-entry",
        action="append",
        default=[],
    )
    parser.add_argument("--receiver-patch-output-change-expected-file-list-sha256")
    parser.add_argument(
        "--receiver-patch-output-change-expected-entry-count",
        type=int,
    )
    parser.add_argument("--receiver-patch-output-change-file-list-source")
    parser.add_argument(
        "--receiver-patch-output-change-parity-scope-kind",
        default="contest_full_sample",
    )
    parser.add_argument(
        "--receiver-patch-output-change-contest-full-sample-claim",
        action="store_true",
    )
    parser.add_argument(
        "--chain-label",
        default="cascade_c_p19_p18_to_p11_selector_context_then_p15_repack",
    )
    parser.add_argument("--codec-family", action="append", default=[])
    parser.add_argument("--scale", action="append", type=int, default=[])
    parser.add_argument("--alpha", action="append", type=int, default=[])
    parser.add_argument("--max-concurrency-local-cpu", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = build_scorer_region_selector_chain_queue(
            repo_root=REPO_ROOT,
            queue_id=args.queue_id,
            source_submission_dir=args.source_submission_dir,
            output_root=args.output_root,
            source_waterfill_work_order=args.source_waterfill_work_order,
            full_frame_inflate_parity_proof=args.full_frame_inflate_parity_proof,
            posenet_null_pairs=args.posenet_null_pairs,
            segnet_region_masks=args.segnet_region_masks,
            selector_region_bits=args.selector_region_bits,
            pose_null_modes_artifact=args.pose_null_modes_artifact,
            segnet_softmax_16=args.segnet_softmax_16,
            segnet_softmax_256=args.segnet_softmax_256,
            materialize_upstream_artifacts=args.materialize_upstream_artifacts,
            materialize_receiver_patch=args.materialize_receiver_patch,
            null_fraction=args.null_fraction,
            top_regions_per_pair=args.top_regions_per_pair,
            receiver_patch_max_pairs=args.receiver_patch_max_pairs,
            receiver_patch_regions_per_pair=args.receiver_patch_regions_per_pair,
            receiver_patch_rgb_delta=tuple(
                int(part.strip()) for part in args.receiver_patch_rgb_delta.split(",")
            ),
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
            chain_label=args.chain_label,
            codec_families=tuple(args.codec_family) if args.codec_family else (
                "fec10_adaptive_blend",
                "fec8_markov_static_order1",
                "fec8_markov_adaptive_order1",
                "fec8_markov_static_order2",
            ),
            scales=tuple(args.scale) if args.scale else (32, 64, 128, 256),
            alphas=tuple(args.alpha) if args.alpha else (1, 2, 4),
            max_concurrency_local_cpu=args.max_concurrency_local_cpu,
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
        ScorerRegionSelectorChainQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: scorer-region selector chain queue failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "scorer_region_selector_chain_queue_cli_result.v1",
                "queue_out": str(args.queue_out),
                "queue_id": queue["queue_id"],
                "experiment_count": len(queue["experiments"]),
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
