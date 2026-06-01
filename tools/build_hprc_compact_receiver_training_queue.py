#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned HPRC compact-receiver train/export campaign."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.local_training_queue import (  # noqa: E402
    build_local_training_execution_queue,
)
from comma_lab.storage_tiers import (  # noqa: E402
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402
from tac.substrates.hprc.archive_candidate import (  # noqa: E402
    FALSE_AUTHORITY,
    HPRC_RECEIVER_PROOF_SCRATCH_BYTES,
)

HPRC_TRAINING_PLAN_SCHEMA = "hprc_compact_receiver_training_plan.v1"
HPRC_TRAINING_QUEUE_BUILD_SCHEMA = "hprc_compact_receiver_training_queue_build.v1"
DEFAULT_HPRC_TRAINING_QUEUE_WORKLOAD_SUBDIR = "experiments/results/hprc_compact_receiver_training_queue"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--queue-id")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--decode-pairs", type=int, default=8)
    parser.add_argument("--decode-max-pairs", type=int)
    parser.add_argument("--decode-height", type=int, default=96)
    parser.add_argument("--decode-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-pair-indices-per-step", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--basis-count", type=int, default=3)
    parser.add_argument("--residual-grid-h", type=int, default=24)
    parser.add_argument("--residual-grid-w", type=int, default=32)
    parser.add_argument("--local-cpu-concurrency", type=int, default=1)
    parser.add_argument("--local-mlx-concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--skip-runtime-consumption-proof", action="store_true")
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--storage-tier", action="append", default=[])
    parser.add_argument(
        "--storage-workload-subdir",
        default=DEFAULT_HPRC_TRAINING_QUEUE_WORKLOAD_SUBDIR,
    )
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument(
        "--storage-expected-bytes",
        type=int,
        default=HPRC_RECEIVER_PROOF_SCRATCH_BYTES,
    )
    parser.add_argument("--allow-local-output-dir", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--expected-plan-output-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    run_id = args.run_id or _utc_run_id()
    output_dir, storage_plan_path = _select_output_dir(args, repo_root=repo_root, run_id=run_id)
    output_manifest = output_dir / "hprc_compact_receiver_training_run_result.json"
    plan_output = (
        output_dir / "hprc_compact_receiver_training_plan.json"
        if args.plan_output is None
        else _resolve_path(args.plan_output, repo_root=repo_root)
    )

    plan = build_hprc_compact_receiver_training_plan(
        repo_root=repo_root,
        run_id=run_id,
        output_dir=output_dir,
        output_manifest=output_manifest,
        storage_plan_path=storage_plan_path,
        video_path=_resolve_path(args.video_path, repo_root=repo_root),
        decode_pairs=int(args.decode_pairs),
        decode_max_pairs=args.decode_max_pairs,
        decode_height=int(args.decode_height),
        decode_width=int(args.decode_width),
        epochs=int(args.epochs),
        batch_pair_indices_per_step=int(args.batch_pair_indices_per_step),
        learning_rate=float(args.learning_rate),
        basis_count=int(args.basis_count),
        residual_grid_h=int(args.residual_grid_h),
        residual_grid_w=int(args.residual_grid_w),
        skip_runtime_consumption_proof=bool(args.skip_runtime_consumption_proof),
        retain_receiver_output=bool(args.retain_receiver_output),
    )
    queue = build_local_training_execution_queue(
        [plan],
        queue_id=args.queue_id or f"hprc_compact_receiver_training_{run_id}",
        repo_root=repo_root,
        lane_id="lane_hprc_compact_receiver_training",
        local_cpu_concurrency=int(args.local_cpu_concurrency),
        local_mlx_concurrency=int(args.local_mlx_concurrency),
        timeout_seconds=int(args.timeout_seconds),
    )
    _write_json(
        plan_output,
        plan,
        allow_overwrite=bool(args.allow_overwrite) or args.expected_plan_output_sha256 is not None,
        expected_existing_sha256=args.expected_plan_output_sha256,
    )
    queue_output = _resolve_path(args.output, repo_root=repo_root)
    _write_json(
        queue_output,
        queue,
        allow_overwrite=bool(args.allow_overwrite) or args.expected_output_sha256 is not None,
        expected_existing_sha256=args.expected_output_sha256,
    )
    print(
        json.dumps(
            {
                "schema": HPRC_TRAINING_QUEUE_BUILD_SCHEMA,
                "queue_path": queue_output.as_posix(),
                "plan_path": plan_output.as_posix(),
                "queue_id": queue["queue_id"],
                "run_id": run_id,
                "output_dir": output_dir.as_posix(),
                "output_manifest": output_manifest.as_posix(),
                "storage_plan_path": storage_plan_path.as_posix(),
                "experiment_count": len(queue["experiments"]),
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0


def build_hprc_compact_receiver_training_plan(
    *,
    repo_root: Path,
    run_id: str,
    output_dir: Path,
    output_manifest: Path,
    storage_plan_path: Path,
    video_path: Path,
    decode_pairs: int,
    decode_max_pairs: int | None,
    decode_height: int,
    decode_width: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    basis_count: int,
    residual_grid_h: int,
    residual_grid_w: int,
    skip_runtime_consumption_proof: bool,
    retain_receiver_output: bool,
) -> dict[str, Any]:
    video_sha = sha256_file(video_path)
    command = [
        ".venv/bin/python",
        "tools/run_hprc_compact_receiver_training.py",
        "--video-path",
        _repo_rel_or_abs(video_path, repo_root),
        "--decode-pairs",
        str(int(decode_pairs)),
        "--decode-height",
        str(int(decode_height)),
        "--decode-width",
        str(int(decode_width)),
        "--output-dir",
        _repo_rel_or_abs(output_dir, repo_root),
        "--output-manifest",
        _repo_rel_or_abs(output_manifest, repo_root),
        "--storage-plan-path",
        _repo_rel_or_abs(storage_plan_path, repo_root),
        "--epochs",
        str(int(epochs)),
        "--batch-pair-indices-per-step",
        str(int(batch_pair_indices_per_step)),
        "--learning-rate",
        repr(float(learning_rate)),
        "--basis-count",
        str(int(basis_count)),
        "--residual-grid-h",
        str(int(residual_grid_h)),
        "--residual-grid-w",
        str(int(residual_grid_w)),
    ]
    if decode_max_pairs is not None:
        command.extend(["--decode-max-pairs", str(int(decode_max_pairs))])
    if skip_runtime_consumption_proof:
        command.append("--skip-runtime-consumption-proof")
    if retain_receiver_output:
        command.append("--retain-receiver-output")
    return {
        "schema": HPRC_TRAINING_PLAN_SCHEMA,
        "candidate_id": f"hprc_compact_receiver_{run_id}",
        "lane_id": "lane_hprc_compact_receiver_training",
        "representation_family": "hprc",
        "substrate_family": "hierarchical_predictive_coding",
        "source_dir": _repo_rel_or_abs(video_path, repo_root),
        "training_signal_kind": "real_contest_video_lowres_frame_fit",
        "candidate_params": {
            "run_id": run_id,
            "source_video_sha256": video_sha,
            "decode_pairs": int(decode_pairs),
            "decode_max_pairs": None if decode_max_pairs is None else int(decode_max_pairs),
            "decode_height": int(decode_height),
            "decode_width": int(decode_width),
            "epochs": int(epochs),
            "basis_count": int(basis_count),
            "residual_grid_h": int(residual_grid_h),
            "residual_grid_w": int(residual_grid_w),
        },
        "recommended_execution": {
            "schema": "hprc_compact_receiver_training_recommended_execution.v1",
            "tool": "tools/run_hprc_compact_receiver_training.py",
            "training_backend": "local_numpy",
            "device": "local_numpy",
            "resource_kind": "local_cpu",
            "output_manifest": output_manifest.as_posix(),
            "python_command_args": command,
            "extra_artifact_postconditions": [
                {
                    "type": "path_exists",
                    "path": (output_dir / "training_artifact.json").as_posix(),
                },
                {
                    "type": "json_false_authority",
                    "path": (output_dir / "training_artifact.json").as_posix(),
                },
                {
                    "type": "path_exists",
                    "path": (output_dir / "hprc_compact_receiver_training_export.json").as_posix(),
                },
                {
                    "type": "json_false_authority",
                    "path": (output_dir / "hprc_compact_receiver_training_export.json").as_posix(),
                },
                {
                    "type": "json_equals",
                    "path": output_manifest.as_posix(),
                    "key": "schema",
                    "equals": "hprc_compact_receiver_training_run_result.v1",
                },
                {
                    "type": "json_equals",
                    "path": output_manifest.as_posix(),
                    "key": "source_manifest.source_kind",
                    "equals": "contest_video_decode",
                },
            ],
            **FALSE_AUTHORITY,
        },
        **FALSE_AUTHORITY,
    }


def _select_output_dir(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    run_id: str,
) -> tuple[Path, Path]:
    tiers = parse_storage_tier_specs(
        list(args.storage_tier),
        repo_root=repo_root,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=bool(args.allow_local_output_dir),
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=str(args.storage_workload_subdir),
        requested_bytes=int(args.storage_expected_bytes),
        create=True,
    )
    workload_root = require_selected_storage(plan)
    output_dir = workload_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_plan_path = output_dir / "hprc_compact_receiver_training_storage_plan.json"
    _write_json(
        storage_plan_path,
        {
            "schema": "hprc_compact_receiver_training_storage_plan.v1",
            "storage_plan": plan.to_dict(),
            "selected_training_output_dir": output_dir.as_posix(),
            **FALSE_AUTHORITY,
        },
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=None,
    )
    return output_dir, storage_plan_path


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    out = Path(path).expanduser()
    return out if out.is_absolute() else repo_root / out


def _repo_rel_or_abs(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(
    path: Path,
    payload: object,
    *,
    allow_overwrite: bool,
    expected_existing_sha256: str | None,
) -> None:
    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_sha256,
    )


def _utc_run_id() -> str:
    return time.strftime("hprc_compact_receiver_%Y%m%dT%H%M%SZ", time.gmtime())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, StorageTierError, ValueError) as exc:
        print(f"build_hprc_compact_receiver_training_queue failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
