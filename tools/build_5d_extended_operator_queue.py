#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a local queue that fires the 8 extended 5D-canvas operators."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path

from comma_lab.scheduler.pair_frame_5d_extended_operator_queue import (
    build_pair_frame_5d_extended_operator_queue,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a false-authority queue for all 8 extended 5D operators."
    )
    parser.add_argument("--canvas-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--queue-out", required=True, type=Path)
    parser.add_argument(
        "--queue-id",
        default="pair_frame_5d_extended_operator_queue",
    )
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--local-cpu-concurrency", type=int, default=4)
    parser.add_argument(
        "--status",
        choices=("queued", "frozen", "disabled"),
        default="queued",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


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
    queue_path = args.queue_out
    if not queue_path.is_absolute():
        queue_path = repo_root / queue_path
    if queue_path.exists() and not args.overwrite:
        raise SystemExit(
            f"[build_5d_extended_operator_queue] FATAL: queue exists: {queue_path}"
        )
    queue = build_pair_frame_5d_extended_operator_queue(
        repo_root=repo_root,
        canvas_path=args.canvas_path,
        output_root=args.output_root,
        queue_id=args.queue_id,
        top_n=args.top_n,
        local_cpu_concurrency=args.local_cpu_concurrency,
        status=args.status,
    )
    _write_text_atomic(queue_path, json.dumps(queue, indent=2, sort_keys=True) + "\n")
    print(
        f"[build_5d_extended_operator_queue] OK: wrote "
        f"{len(queue.get('experiments') or [])} experiments to {queue_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
