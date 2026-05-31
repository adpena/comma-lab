#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an experiment_queue.v1 for Z8 MLX replay and exact-auth gating."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.z8_mlx_prefilter_gate_queue import (  # noqa: E402
    Z8MlxPrefilterGateQueueError,
    build_z8_mlx_prefilter_gate_queue,
)
from comma_lab.storage_tiers import (  # noqa: E402
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402


def _parse_hw(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--scorer-hw must be H,W")
    return int(parts[0]), int(parts[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-out", required=True, type=Path)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-archive-bin", required=True, type=Path)
    parser.add_argument("--archive-zip", type=Path)
    parser.add_argument("--reference-pairs-npy", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Candidate output root. If omitted, choose the first eligible "
            "external storage tier from the operator waterfall."
        ),
    )
    parser.add_argument("--auth-frontier-score", required=True, type=float)
    parser.add_argument("--mlx-target-action", type=float)
    parser.add_argument("--lane-id", default="z8_mlx_prefilter_gate")
    parser.add_argument("--family-id", default="z8_hierarchical_predictive_coding")
    parser.add_argument("--posterior-path", type=Path)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--pair-chunk-size", type=int, default=32)
    parser.add_argument("--scorer-hw", type=_parse_hw, default=(384, 512))
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--local-cpu-concurrency", type=int, default=1)
    parser.add_argument("--local-mlx-concurrency", type=int, default=1)
    parser.add_argument("--local-io-concurrency", type=int, default=1)
    parser.add_argument("--no-learning-signal", action="store_true")
    parser.add_argument("--no-auto-cleanup", action="store_true")
    parser.add_argument("--cleanup-min-bytes", default="100MiB")
    parser.add_argument("--cleanup-cold-store-root", action="append", default=[])
    parser.add_argument("--cleanup-cold-store-reserve-gb", type=float, default=40.0)
    parser.add_argument("--storage-tier", action="append", default=[], help="name=/path storage tier override")
    parser.add_argument(
        "--storage-workload-subdir",
        default="experiments/results/z8_mlx_prefilter_gate",
    )
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument("--storage-expected-bytes", type=int, default=0)
    parser.add_argument("--storage-plan-out", type=Path)
    parser.add_argument(
        "--allow-local-output-root",
        action="store_true",
        help="Allow local-disk fallback when external tiers are unavailable.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve_output_root(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.output_root is not None:
        return args.output_root, None
    tiers = parse_storage_tier_specs(
        args.storage_tier,
        repo_root=REPO_ROOT,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=bool(args.allow_local_output_root),
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=str(args.storage_workload_subdir),
        requested_bytes=int(args.storage_expected_bytes),
        create=True,
    )
    output_root = require_selected_storage(plan)
    storage_plan_out = args.storage_plan_out or args.queue_out.with_suffix(
        args.queue_out.suffix + ".storage_plan.json"
    )
    if storage_plan_out.exists() and not args.overwrite:
        raise StorageTierError(f"refusing to overwrite storage plan: {storage_plan_out}")
    expected_sha = sha256_file(storage_plan_out) if storage_plan_out.is_file() else None
    write_json_artifact(
        storage_plan_out,
        plan.to_dict(),
        allow_overwrite=expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )
    return output_root, storage_plan_out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.queue_out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing queue: {args.queue_out}")
    output_root, storage_plan_out = _resolve_output_root(args)
    candidate = {
        "candidate_id": args.candidate_id,
        "candidate_archive_bin": args.candidate_archive_bin.as_posix(),
    }
    if args.archive_zip is not None:
        candidate["archive_zip"] = args.archive_zip.as_posix()
    queue = build_z8_mlx_prefilter_gate_queue(
        [candidate],
        queue_id=args.queue_id,
        repo_root=REPO_ROOT,
        reference_pairs_npy=args.reference_pairs_npy,
        output_root=output_root,
        auth_frontier_score=float(args.auth_frontier_score),
        mlx_target_action=args.mlx_target_action,
        lane_id=args.lane_id,
        family_id=args.family_id,
        posterior_path=args.posterior_path,
        upstream_dir=args.upstream_dir,
        pair_chunk_size=int(args.pair_chunk_size),
        scorer_hw=args.scorer_hw,
        mlx_device=str(args.mlx_device),
        local_cpu_concurrency=int(args.local_cpu_concurrency),
        local_mlx_concurrency=int(args.local_mlx_concurrency),
        local_io_concurrency=int(args.local_io_concurrency),
        enable_learning_signal=not bool(args.no_learning_signal),
        enable_auto_cleanup=not bool(args.no_auto_cleanup),
        cleanup_min_bytes=str(args.cleanup_min_bytes),
        cleanup_cold_store_roots=tuple(args.cleanup_cold_store_root),
        cleanup_cold_store_reserve_gb=float(args.cleanup_cold_store_reserve_gb),
        timeout_seconds=int(args.timeout_seconds),
    )
    expected_queue_sha = sha256_file(args.queue_out) if args.queue_out.is_file() else None
    write_json_artifact(
        args.queue_out,
        queue,
        allow_overwrite=expected_queue_sha is not None,
        expected_existing_sha256=expected_queue_sha,
    )
    print(
        json.dumps(
            {
                "queue_id": queue["queue_id"],
                "queue_out": args.queue_out.as_posix(),
                "experiment_count": len(queue["experiments"]),
                "schema": queue["schema"],
                "output_root": output_root.as_posix(),
                "storage_plan_out": None if storage_plan_out is None else storage_plan_out.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Z8MlxPrefilterGateQueueError as exc:
        print(f"build_z8_mlx_prefilter_gate_queue failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except StorageTierError as exc:
        print(f"build_z8_mlx_prefilter_gate_queue storage selection failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ArtifactWriteError as exc:
        print(f"build_z8_mlx_prefilter_gate_queue artifact write failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
