#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the HPRC compact receiver train/export adapter on low-res RGB frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact
from tac.substrates.hprc.archive_candidate import (
    FALSE_AUTHORITY,
    HPRC_RECEIVER_PROOF_SCRATCH_BYTES,
)
from tac.substrates.hprc.training_adapter import (
    HPRC_LONG_TRAINING_SUBSTRATE_ID,
    HprcCompactReceiverLongTrainingAdapter,
)
from tac.training.long_training_canonical import (
    CurriculumStage,
    LongTrainingConfig,
    run_long_training,
)

HPRC_LONG_TRAINING_STORAGE_PLAN_SCHEMA = "hprc_compact_receiver_training_storage_plan.v1"
HPRC_LONG_TRAINING_RESULT_SCHEMA = "hprc_compact_receiver_training_run_result.v1"
DEFAULT_HPRC_LONG_TRAINING_WORKLOAD_SUBDIR = "experiments/results/hprc_long_training"
DEFAULT_HPRC_LONG_TRAINING_EXPECTED_BYTES = HPRC_RECEIVER_PROOF_SCRATCH_BYTES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-pair-indices-per-step", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--basis-count", type=int, default=3)
    parser.add_argument("--residual-grid-h", type=int, default=24)
    parser.add_argument("--residual-grid-w", type=int, default=32)
    parser.add_argument("--initial-latent-gain", type=float, default=1.0)
    parser.add_argument("--initial-residual-gain", type=float, default=1.0)
    parser.add_argument("--initial-receiver-state-gain", type=float, default=0.25)
    parser.add_argument(
        "--skip-runtime-consumption-proof",
        action="store_true",
        help="Emit archive bytes without running generated inflate receiver proof.",
    )
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--storage-tier", action="append", default=[], help="name=/path storage tier override")
    parser.add_argument(
        "--storage-workload-subdir",
        default=DEFAULT_HPRC_LONG_TRAINING_WORKLOAD_SUBDIR,
    )
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument(
        "--storage-expected-bytes",
        type=int,
        default=DEFAULT_HPRC_LONG_TRAINING_EXPECTED_BYTES,
    )
    parser.add_argument(
        "--allow-local-output-dir",
        action="store_true",
        help="Allow local-disk fallback only by explicit opt-in.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    output_dir, storage_plan_path = _resolve_output_dir(args, repo_root=repo_root)
    frames_path = Path(args.frames_npy).expanduser().resolve(strict=False)
    frames = np.load(frames_path, mmap_mode="r")
    source_manifest = {
        "schema": "hprc_compact_receiver_training_source_frames.v1",
        "frames_npy_path": frames_path.as_posix(),
        "frames_npy_bytes": frames_path.stat().st_size,
        "frames_npy_sha256": sha256_file(frames_path),
        "frames_shape": [int(v) for v in frames.shape],
        "frames_dtype": str(frames.dtype),
        "score_claim": False,
        "promotion_eligible": False,
    }
    adapter = HprcCompactReceiverLongTrainingAdapter(
        frames,
        basis_count=int(args.basis_count),
        residual_grid_h=int(args.residual_grid_h),
        residual_grid_w=int(args.residual_grid_w),
        source_manifest=source_manifest,
        initial_latent_gain=float(args.initial_latent_gain),
        initial_residual_gain=float(args.initial_residual_gain),
        initial_receiver_state_gain=float(args.initial_receiver_state_gain),
        repo_root=repo_root,
        retain_receiver_proof_output=bool(args.retain_receiver_output),
        emit_archive_bound_candidate_package=not bool(args.skip_runtime_consumption_proof),
    )
    config = LongTrainingConfig(
        substrate_id=HPRC_LONG_TRAINING_SUBSTRATE_ID,
        lane_id="lane_hprc_compact_receiver_training",
        epochs=int(args.epochs),
        batch_pair_indices_per_step=int(args.batch_pair_indices_per_step),
        curriculum_stages=(
            CurriculumStage(
                name="compact_receiver_gain_fit",
                start_epoch=0,
                end_epoch=int(args.epochs),
                loss_weights={"recon": 1.0},
                notes="Fit HPRC compact receiver decode RDO gains before archive export.",
            ),
        ),
        checkpoint_interval_epochs=max(1, min(int(args.epochs), 10)),
        early_stopping_patience=max(2, int(args.epochs) * 2),
        learning_rate=float(args.learning_rate),
        output_dir=output_dir,
        notes="HPRC compact receiver train/export adapter run with storage-waterfall custody.",
    )
    artifact = run_long_training(adapter, config)
    result = {
        "schema": HPRC_LONG_TRAINING_RESULT_SCHEMA,
        "run_id": output_dir.name,
        "storage_plan_path": None if storage_plan_path is None else storage_plan_path.as_posix(),
        "source_manifest": source_manifest,
        "artifact": artifact.as_dict(),
        "runtime_consumption_proof_requested": not bool(args.skip_runtime_consumption_proof),
        "exact_axis_blocker": "contest_cpu_cuda_exact_eval_not_executed",
        **FALSE_AUTHORITY,
    }
    result_path = output_dir / "hprc_compact_receiver_training_run_result.json"
    _write_json_maybe_overwrite(result_path, result)
    print(json.dumps({**result, "result_path": result_path.as_posix()}, sort_keys=True))
    return 0


def _resolve_output_dir(
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> tuple[Path, Path | None]:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, None

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
    run_id = args.run_id or _utc_stamp()
    output_dir = workload_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_plan_path = output_dir / "hprc_compact_receiver_training_storage_plan.json"
    _write_json_maybe_overwrite(
        storage_plan_path,
        {
            "schema": HPRC_LONG_TRAINING_STORAGE_PLAN_SCHEMA,
            "storage_plan": plan.to_dict(),
            "selected_training_output_dir": output_dir.as_posix(),
            **FALSE_AUTHORITY,
        },
    )
    return output_dir, storage_plan_path


def _write_json_maybe_overwrite(path: Path, payload: object) -> None:
    expected_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )


def _utc_stamp() -> str:
    return time.strftime("hprc_compact_receiver_training_%Y%m%dT%H%M%SZ", time.gmtime())


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, StorageTierError, ValueError) as exc:
        print(f"run_hprc_compact_receiver_training failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
