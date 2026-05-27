#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan or execute PR95 MLX long-training validation runs."""

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

from tac.local_acceleration.pr95_hnerv_mlx_long_training import (  # noqa: E402
    CANONICAL_BASE_CHANNELS,
    CANONICAL_CONTEST_VIDEO_PATH,
    CANONICAL_EVAL_SIZE,
    CANONICAL_LATENT_DIM,
    LongTrainingConfig,
    MLXLongTrainingPipeline,
    build_long_training_plan_report,
    compute_video_sha256,
)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_eval_size(raw: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("eval size must be H,W")
    try:
        height, width = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("eval size must contain integers") from exc
    if height < 1 or width < 1:
        raise argparse.ArgumentTypeError("eval size dimensions must be positive")
    return height, width


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument(
        "--source-video-path",
        default=CANONICAL_CONTEST_VIDEO_PATH,
        type=Path,
    )
    parser.add_argument(
        "--checkpoint-root",
        default=Path("experiments/results/pr95_mlx_long_training_checkpoints"),
        type=Path,
    )
    parser.add_argument("--telemetry-path", type=Path)
    parser.add_argument(
        "--lane-id",
        default=(
            "lane_pr95_mlx_long_training_infrastructure_and_substrate_class_shift_"
            "candidate_validation_pipeline_20260525"
        ),
    )
    parser.add_argument("--operator-run-label", default="")
    parser.add_argument("--latent-dim", default=CANONICAL_LATENT_DIM, type=int)
    parser.add_argument("--base-channels", default=CANONICAL_BASE_CHANNELS, type=int)
    parser.add_argument(
        "--eval-size",
        default=f"{CANONICAL_EVAL_SIZE[0]},{CANONICAL_EVAL_SIZE[1]}",
        type=_parse_eval_size,
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--random-seed", default=0, type=int)
    parser.add_argument("--smoke-mode", action="store_true")
    parser.add_argument("--smoke-epochs-per-stage", default=1, type=int)
    parser.add_argument("--checkpoint-every-epochs", default=1, type=int)
    parser.add_argument(
        "--hash-source-video",
        action="store_true",
        help="Hash the source video in the plan report if it exists.",
    )
    parser.add_argument(
        "--execute-smoke",
        action="store_true",
        help="Run the MLX smoke path. Requires --smoke-mode.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the MLX training path using the configured curriculum. "
            "Without --smoke-mode this executes the full stage epoch counts."
        ),
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> LongTrainingConfig:
    return LongTrainingConfig(
        source_video_path=args.source_video_path,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        eval_size=args.eval_size,
        checkpoint_root=args.checkpoint_root,
        telemetry_path=args.telemetry_path,
        smoke_mode=bool(args.smoke_mode),
        smoke_epochs_per_stage=args.smoke_epochs_per_stage,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        max_frames=args.max_frames,
        random_seed=args.random_seed,
        lane_id=args.lane_id,
        operator_run_label=args.operator_run_label,
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv or sys.argv[1:])
    args = parse_args(raw_argv)
    if args.execute_smoke and not args.smoke_mode:
        raise SystemExit("--execute-smoke requires --smoke-mode")
    if args.execute and args.execute_smoke:
        raise SystemExit("pass only one of --execute or --execute-smoke")
    if args.execute and args.smoke_mode:
        raise SystemExit("--execute runs the configured long curriculum; use --execute-smoke with --smoke-mode")

    config = _config_from_args(args)
    source_sha = None
    source_frame_count = None
    telemetry = None
    checkpoint_artifacts = []
    mode = "plan_only"

    if args.hash_source_video and config.source_video_path.is_file():
        source_sha = compute_video_sha256(config.source_video_path)

    if args.execute or args.execute_smoke:
        pipeline = MLXLongTrainingPipeline(config)
        pipeline.setup()
        telemetry = pipeline.run_curriculum()
        source_sha = telemetry.source_video_sha256
        source_frame_count = telemetry.source_video_frame_count
        checkpoint_artifacts = pipeline.checkpoint_artifacts
        mode = "executed_smoke" if config.smoke_mode else "executed_local_mlx"

    output_report = _repo_path(args.output_report)
    queue_command = [
        ".venv/bin/python",
        "tools/run_pr95_mlx_long_training.py",
        *raw_argv,
    ]
    report = build_long_training_plan_report(
        config,
        mode=mode,
        output_report_path=args.output_report,
        source_video_sha256=source_sha,
        source_video_frame_count=source_frame_count,
        telemetry_path=config.telemetry_path,
        checkpoint_artifacts=checkpoint_artifacts,
        command=queue_command,
    )
    if telemetry is not None:
        report["telemetry_row_count"] = len(telemetry.rows)

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "output_report": output_report.as_posix(),
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
                "readiness_blockers": report["readiness_blockers"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
