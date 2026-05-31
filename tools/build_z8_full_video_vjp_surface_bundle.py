#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned Z8 full-video VJP plan or surface bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.storage_tiers import (  # noqa: E402
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    StorageTierSpec,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402
from tac.substrates.z8_hierarchical_predictive_coding.full_video_vjp_acquisition import (  # noqa: E402
    Z8FullVideoMlxVjpShardConfig,
    Z8FullVideoVjpAcquisitionConfig,
    assemble_z8_full_video_vjp_surface_bundle,
    build_z8_full_video_mlx_vjp_surface_shard,
    load_z8_full_video_vjp_surface_shard_file,
    write_z8_full_video_vjp_acquisition_plan,
    write_z8_full_video_vjp_surface_bundle,
    write_z8_full_video_vjp_surface_shard,
)

DEFAULT_STORAGE_WORKLOAD_SUBDIR = "experiments/results/z8_full_video_vjp"
Z8_VJP_STORAGE_PLAN_SCHEMA = "z8_full_video_vjp_output_storage_plan.v1"
Z8_VJP_REPLAY_PROVENANCE_SCHEMA = "z8_full_video_vjp_replay_provenance.v1"
REPLAY_ENV_ALLOWLIST = (
    "PYTHONHASHSEED",
    "TMPDIR",
    "MLX_DEFAULT_DEVICE",
    "MLX_METAL_PATH",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. If omitted, choose the first eligible external "
            "SSD tier from the operator storage waterfall."
        ),
    )
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        default=[],
        help=(
            "Archive-pinned VJP shard NPZ/JSON. If omitted, only the shard plan "
            "is written."
        ),
    )
    parser.add_argument("--target-mode", default="contest_video_overfit")
    parser.add_argument("--pair-chunk-size", type=int, default=64)
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--corpus-manifest-path", default=None)
    parser.add_argument("--disable-minibatch-probes", action="store_true")
    parser.add_argument("--allow-partial-production-probe-surface", action="store_true")
    parser.add_argument(
        "--reference-pairs-npy",
        type=Path,
        help="Full-video reference pairs, shape (pairs,2,H,W,3). Emits one MLX VJP shard.",
    )
    parser.add_argument(
        "--candidate-pairs-npy",
        type=Path,
        help="Full-video candidate pairs, shape (pairs,2,H,W,3). Emits one MLX VJP shard.",
    )
    parser.add_argument("--pair-start", type=int, default=None)
    parser.add_argument("--pair-end", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--full-video-d-pose", type=float, default=None)
    parser.add_argument("--rgb-value-range", type=float, default=255.0)
    parser.add_argument("--scorer-hw", default="384,512")
    parser.add_argument(
        "--pose-axis-count",
        type=int,
        default=6,
        help="Number of PoseNet output axes to VJP for true P19; default is contest first-six pose axes.",
    )
    parser.add_argument(
        "--pose-inverse-variance",
        default="1,1,1,1,1,1",
        help=(
            "Comma-separated inverse-variance weights for the Mahalanobis P19 "
            "norm. Identity matches upstream first-six pose MSE."
        ),
    )
    parser.add_argument("--seg-margin-delta", type=float, default=1.0)
    parser.add_argument("--pose-null-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("upstream"),
        help="Upstream scorer model directory used when emitting MLX VJP shards.",
    )
    parser.add_argument("--storage-tier", action="append", default=[], help="name=/path storage tier override")
    parser.add_argument("--storage-workload-subdir", default=DEFAULT_STORAGE_WORKLOAD_SUBDIR)
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument(
        "--storage-expected-bytes",
        type=int,
        default=0,
        help="Expected bulky output bytes for free-space planning.",
    )
    parser.add_argument("--storage-plan-out", type=Path)
    parser.add_argument(
        "--allow-local-output-dir",
        action="store_true",
        help="Explicitly allow local-disk output when no external tier is selected.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_hw(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--scorer-hw must be formatted as H,W")
    return int(parts[0]), int(parts[1])


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError("--pose-inverse-variance must contain at least one value")
    values = tuple(float(part) for part in parts)
    if any(value <= 0.0 for value in values):
        raise ValueError("--pose-inverse-variance entries must be positive")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parsed = build_parser().parse_args(argv)
    parsed.replay_argv = list(sys.argv[1:] if argv is None else argv)
    return parsed


def _write_storage_plan(path: Path, payload: dict) -> None:
    expected_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )


def _looks_like_local_disk(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError:
        resolved = path.expanduser()
    return not str(resolved).startswith("/Volumes/")


def _git_revision_payload() -> dict:
    payload: dict[str, str | bool | None] = {
        "git_head_sha": None,
        "git_status_available": False,
        "git_worktree_has_uncommitted_changes": None,
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return payload
    payload.update(
        {
            "git_head_sha": head.stdout.strip(),
            "git_status_available": True,
            "git_worktree_has_uncommitted_changes": bool(status.stdout.strip()),
        }
    )
    return payload


def _archive_identity(path: Path, *, archive_bytes: bytes | None = None) -> dict:
    archive_path = path.expanduser()
    if not archive_path.is_absolute():
        archive_path = REPO_ROOT / archive_path
    if archive_bytes is None:
        size = archive_path.stat().st_size
        digest = sha256_file(archive_path)
    else:
        size = len(archive_bytes)
        digest = hashlib.sha256(archive_bytes).hexdigest()
    return {
        "path": archive_path.as_posix(),
        "bytes": int(size),
        "sha256": digest,
    }


def _replay_provenance(args: argparse.Namespace, *, archive_bytes: bytes | None = None) -> dict:
    return {
        "schema": Z8_VJP_REPLAY_PROVENANCE_SCHEMA,
        "repo_root": REPO_ROOT.as_posix(),
        "tool_path": Path(__file__).resolve().as_posix(),
        "python_executable": sys.executable,
        "argv": [Path(__file__).as_posix(), *list(getattr(args, "replay_argv", []))],
        "env_allowlist": {
            key: os.environ[key]
            for key in REPLAY_ENV_ALLOWLIST
            if key in os.environ
        },
        "env_allowlist_keys": list(REPLAY_ENV_ALLOWLIST),
        "archive": _archive_identity(args.archive_bin, archive_bytes=archive_bytes),
        **_git_revision_payload(),
    }


def _resolve_output_dir(args: argparse.Namespace, *, archive_bytes: bytes | None = None) -> tuple[Path, Path]:
    """Resolve bulky Z8 VJP output through SSD-first storage policy."""

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        if _looks_like_local_disk(output_dir) and not args.allow_local_output_dir:
            raise StorageTierError("local_output_dir_requires_explicit_opt_in")
        tiers = (
            StorageTierSpec(
                name="explicit_output_dir",
                root=output_dir,
                priority=0,
                reserve_free_bytes=0,
                allow_create=True,
                allow_local_disk=bool(args.allow_local_output_dir),
            ),
        )
        workload_subdir = "."
    else:
        tiers = parse_storage_tier_specs(
            args.storage_tier,
            repo_root=REPO_ROOT,
            reserve_free_gb=float(args.storage_reserve_free_gb),
            allow_local_disk=bool(args.allow_local_output_dir),
        )
        workload_subdir = str(args.storage_workload_subdir)

    plan = plan_experiment_storage(
        tiers,
        workload_subdir=workload_subdir,
        requested_bytes=int(args.storage_expected_bytes),
        create=True,
    )
    output_dir = require_selected_storage(plan)
    storage_plan_out = args.storage_plan_out or output_dir / "z8_full_video_vjp_storage_plan.json"
    if storage_plan_out.exists() and not args.overwrite:
        raise StorageTierError(f"refusing to overwrite storage plan: {storage_plan_out}")
    payload = {
        "schema": Z8_VJP_STORAGE_PLAN_SCHEMA,
        "tool": "tools/build_z8_full_video_vjp_surface_bundle.py",
        "resolved_output_dir": output_dir.as_posix(),
        "output_dir_was_explicit": args.output_dir is not None,
        "local_output_explicitly_allowed": bool(args.allow_local_output_dir),
        "storage_plan": plan.to_dict(),
        "replay_provenance": _replay_provenance(args, archive_bytes=archive_bytes),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _write_storage_plan(storage_plan_out, payload)
    return output_dir, storage_plan_out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_bytes = args.archive_bin.read_bytes()
    output_dir, storage_plan_out = _resolve_output_dir(args, archive_bytes=archive_bytes)
    config = Z8FullVideoVjpAcquisitionConfig(
        target_mode=args.target_mode,
        pair_chunk_size=args.pair_chunk_size,
        parallel_workers=args.parallel_workers,
        corpus_manifest_path=args.corpus_manifest_path,
        allow_minibatch_probe_between_full_passes=not args.disable_minibatch_probes,
        allow_partial_production_probe_surface=args.allow_partial_production_probe_surface,
    )
    if args.reference_pairs_npy or args.candidate_pairs_npy:
        if args.shard:
            raise SystemExit("--shard cannot be combined with --reference-pairs-npy/--candidate-pairs-npy")
        if not args.reference_pairs_npy or not args.candidate_pairs_npy:
            raise SystemExit("--reference-pairs-npy and --candidate-pairs-npy must be provided together")
        if args.pair_start is None or args.pair_end is None:
            raise SystemExit("--pair-start and --pair-end are required for MLX shard emission")
        if args.full_video_d_pose is None:
            raise SystemExit("--full-video-d-pose is required for exact full-video pose-term scaling")
        import numpy as np

        from tac.local_acceleration.mlx_scorer_adapters import (
            load_mlx_distortion_scorer_adapter_from_upstream,
        )

        reference_pairs = np.load(args.reference_pairs_npy)
        candidate_pairs = np.load(args.candidate_pairs_npy)
        mlx_scorer = load_mlx_distortion_scorer_adapter_from_upstream(args.upstream_dir, device="cpu")
        shard = build_z8_full_video_mlx_vjp_surface_shard(
            archive_bytes,
            reference_pairs_rgb=reference_pairs,
            candidate_pairs_rgb=candidate_pairs,
            mlx_scorer=mlx_scorer,
            config=Z8FullVideoMlxVjpShardConfig(
                shard_index=int(args.shard_index),
                pair_start=int(args.pair_start),
                pair_end=int(args.pair_end),
                full_video_pair_count=int(candidate_pairs.shape[0]),
                full_video_d_pose=float(args.full_video_d_pose),
                target_mode=args.target_mode,
                rgb_value_range=float(args.rgb_value_range),
                scorer_hw=_parse_hw(args.scorer_hw),
                pose_axis_count=int(args.pose_axis_count),
                pose_inverse_variance=_parse_float_tuple(args.pose_inverse_variance),
                seg_margin_delta=float(args.seg_margin_delta),
                pose_null_threshold=float(args.pose_null_threshold),
            ),
        )
        artifact = write_z8_full_video_vjp_surface_shard(shard, output_dir)
    elif not args.shard:
        artifact = write_z8_full_video_vjp_acquisition_plan(
            archive_bytes,
            output_dir,
            config=config,
        )
    else:
        shard_surfaces = [
            load_z8_full_video_vjp_surface_shard_file(path)
            for path in args.shard
        ]
        bundle = assemble_z8_full_video_vjp_surface_bundle(
            archive_bytes,
            shard_surfaces=shard_surfaces,
            config=config,
        )
        artifact = write_z8_full_video_vjp_surface_bundle(bundle, output_dir)
    artifact = {
        **artifact,
        "resolved_output_dir": output_dir.as_posix(),
        "storage_plan_out": storage_plan_out.as_posix(),
    }
    print(json.dumps(artifact, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except StorageTierError as exc:
        print(f"build_z8_full_video_vjp_surface_bundle storage selection failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ArtifactWriteError as exc:
        print(f"build_z8_full_video_vjp_surface_bundle artifact write failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
