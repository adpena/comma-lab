#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build an execution queue from ready 5D coverage follow-up requests."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path

from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.pair_frame_5d_coverage_acquisition_queue import (
    build_coverage_followup_execution_queue,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--queue-out", required=True, type=Path)
    parser.add_argument(
        "--queue-id",
        default="pair_frame_5d_followup_execution_queue",
    )
    parser.add_argument(
        "--lane-id",
        default="lane_pair_frame_5d_coverage_followup_execution_20260527",
    )
    parser.add_argument("--local-cpu-concurrency", type=int, default=2)
    parser.add_argument("--local-mlx-concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--status",
        choices=("queued", "frozen", "disabled"),
        default="queued",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = _repo_root()
    queue_path = _resolve(args.queue_out, repo_root)
    if queue_path.exists() and not args.overwrite:
        raise SystemExit(
            f"[build_5d_coverage_followup_execution_queue] FATAL: queue exists: {queue_path}"
        )
    try:
        queue = build_coverage_followup_execution_queue(
            repo_root=repo_root,
            readiness_report_path=args.readiness_report,
            queue_id=args.queue_id,
            lane_id=args.lane_id,
            local_cpu_concurrency=args.local_cpu_concurrency,
            local_mlx_concurrency=args.local_mlx_concurrency,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            status=args.status,
        )
    except (OSError, json.JSONDecodeError, ExperimentQueueError) as exc:
        print(f"[build_5d_coverage_followup_execution_queue] FATAL: {exc}")
        return 2
    _write_text_atomic(queue_path, json.dumps(queue, indent=2, sort_keys=True) + "\n")
    print(
        "[build_5d_coverage_followup_execution_queue] OK: "
        f"wrote {len(queue.get('experiments') or [])} experiments to {queue_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
