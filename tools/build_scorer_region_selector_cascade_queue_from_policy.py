#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile the next scorer-region cascade queue from an acquisition policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.scorer_region_selector_cascade_campaign_queue import (  # noqa: E402
    DEFAULT_MASTER_GRADIENT_TENSOR_PATH,
    DEFAULT_PIXEL_GRADIENT_CACHE_PATH,
    SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA,
    ScorerRegionSelectorCascadeCampaignQueueError,
    build_scorer_region_selector_cascade_campaign_queue,
)
from comma_lab.scheduler.scorer_region_selector_chain_queue import (  # noqa: E402
    ScorerRegionSelectorChainQueueError,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields  # noqa: E402
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
    parser.add_argument("--acquisition-policy", required=True, type=Path)
    parser.add_argument("--queue-out", required=True, type=Path)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-waterfill-work-order", type=Path)
    parser.add_argument("--full-frame-inflate-parity-proof", type=Path)
    parser.add_argument("--pose-null-modes-artifact", required=True, type=Path)
    parser.add_argument("--segnet-softmax-16", required=True, type=Path)
    parser.add_argument("--segnet-softmax-256", required=True, type=Path)
    parser.add_argument("--policy-null-fraction", action="append", type=float, default=[])
    parser.add_argument("--fallback-null-fraction", action="append", type=float, default=[])
    parser.add_argument("--receiver-patch-rgb-delta", action="append", type=_triple, default=[])
    parser.add_argument("--receiver-patch-yuv-delta", action="append", type=_triple, default=[])
    parser.add_argument(
        "--selector-codec-family-set",
        action="append",
        type=_csv_group,
        default=[],
        help="Override policy codec groups with comma-separated families; may repeat.",
    )
    parser.add_argument("--scale", action="append", type=int, default=[])
    parser.add_argument("--alpha", action="append", type=int, default=[])
    parser.add_argument("--max-variants", type=int, default=48)
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
    parser.add_argument("--receiver-patch-output-change-left-cache-dir", type=Path)
    parser.add_argument("--receiver-patch-output-change-right-cache-dir", type=Path)
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
    parser.add_argument("--mlx-first-acquisition", action="store_true")
    parser.add_argument("--mlx-cpu-gate-max-score-delta", type=float, default=0.0)
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
    parser.add_argument("--master-gradient-tensor", type=Path, default=DEFAULT_MASTER_GRADIENT_TENSOR_PATH)
    parser.add_argument(
        "--master-gradient-anchor-ledger-path",
        type=Path,
        default=Path(".omx/state/master_gradient_anchors.jsonl"),
    )
    parser.add_argument("--pixel-gradient-cache", type=Path, default=DEFAULT_PIXEL_GRADIENT_CACHE_PATH)
    parser.add_argument("--no-dynamic-followup-queue", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return None if str(path).strip().lower() == "none" else path


def _preferred_grid(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    next_policy = policy.get("next_queue_policy")
    if not isinstance(next_policy, Mapping):
        return {}
    preferred = next_policy.get("preferred_next_grid")
    return preferred if isinstance(preferred, Mapping) else {}


def _int_sequence(value: Any, default: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed: list[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        if parsed:
            return tuple(dict.fromkeys(parsed))
    return tuple(default)


def _codec_groups(value: Any, default: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[tuple[str, ...]] = []
        for group in value:
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes, bytearray)):
                normalized = tuple(str(item).strip() for item in group if str(item).strip())
                if normalized:
                    out.append(normalized)
            elif str(group).strip():
                out.append((str(group).strip(),))
        if out:
            return tuple(out)
    return tuple(tuple(item) for item in default)


def _repack_orders(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        orders = tuple(str(item).strip() for item in value if str(item).strip())
        if orders:
            return orders
    return ("p11_then_p15_then_receiver_patch",)


def _null_fractions(policy: Mapping[str, Any], args: argparse.Namespace) -> tuple[float, ...]:
    if args.policy_null_fraction:
        return tuple(float(item) for item in args.policy_null_fraction)
    selection = policy.get("selection_manifest")
    selection_ready = isinstance(selection, Mapping) and selection.get("selection_ready") is True
    if selection_ready:
        return (0.02, 0.05, 0.10)
    if args.fallback_null_fraction:
        return tuple(float(item) for item in args.fallback_null_fraction)
    return (0.05, 0.10)


def _rgb_deltas(args: argparse.Namespace) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        args.receiver_patch_rgb_delta
        or [
            (-1, -1, -1),
            (1, 1, 1),
            (0, -1, 1),
            (0, 1, -1),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        policy_path = _resolve(args.acquisition_policy)
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ScorerRegionSelectorCascadeCampaignQueueError(
                f"acquisition policy must be a JSON object: {policy_path}"
            )
        if policy.get("schema") != SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA:
            raise ScorerRegionSelectorCascadeCampaignQueueError(
                "acquisition policy schema mismatch"
            )
        require_no_truthy_authority_fields(
            policy,
            context="scorer_region_selector_cascade_queue_from_policy",
        )
        preferred = _preferred_grid(policy)
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
            null_fractions=_null_fractions(policy, args),
            top_regions_per_pair_values=_int_sequence(
                preferred.get("top_regions_per_pair"),
                (1, 2),
            ),
            receiver_patch_max_pair_values=_int_sequence(
                preferred.get("receiver_patch_max_pairs"),
                (4, 8, 12),
            ),
            receiver_patch_regions_per_pair_values=_int_sequence(
                preferred.get("receiver_patch_regions_per_pair"),
                (1,),
            ),
            receiver_patch_rgb_deltas=_rgb_deltas(args),
            receiver_patch_yuv_deltas=tuple(args.receiver_patch_yuv_delta),
            selector_codec_family_groups=tuple(args.selector_codec_family_set)
            or _codec_groups(
                preferred.get("selector_codec_family_groups"),
                (
                    ("fec10_adaptive_blend",),
                    ("fec8_markov_static_order1",),
                    ("fec10_adaptive_blend", "fec8_markov_static_order1"),
                ),
            ),
            scales=tuple(args.scale or [32, 64, 128, 256]),
            alphas=tuple(args.alpha or [1, 2, 4]),
            repack_orders=_repack_orders(preferred.get("repack_order")),
            max_variants=args.max_variants,
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
            receiver_patch_output_change_left_cache_dir=(
                args.receiver_patch_output_change_left_cache_dir
            ),
            receiver_patch_output_change_right_cache_dir=(
                args.receiver_patch_output_change_right_cache_dir
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
            mlx_first_acquisition=args.mlx_first_acquisition,
            mlx_cpu_gate_max_score_delta=args.mlx_cpu_gate_max_score_delta,
            mlx_reference_cache_dir=args.mlx_reference_cache_dir,
            mlx_device=args.mlx_device,
            mlx_cache_batch_pairs=args.mlx_cache_batch_pairs,
            mlx_batch_pairs=args.mlx_batch_pairs,
            mlx_max_pairs=None,
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
            append_dynamic_followup_queue=not args.no_dynamic_followup_queue,
            master_gradient_tensor_path=_optional_path(args.master_gradient_tensor),
            master_gradient_anchor_ledger_path=args.master_gradient_anchor_ledger_path,
            pixel_gradient_cache_path=_optional_path(args.pixel_gradient_cache),
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
        json.JSONDecodeError,
        ScorerRegionSelectorCascadeCampaignQueueError,
        ScorerRegionSelectorChainQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: scorer-region follow-up queue compile failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "scorer_region_selector_cascade_queue_from_policy_cli_result.v1",
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
