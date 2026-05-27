#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Emit a false-authority acquisition plan for one 5D coverage work order."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path

from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.pair_frame_5d_coverage_acquisition_queue import (
    build_coverage_acquisition_plan,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-audit", required=True, type=Path)
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--canvas-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("mlx-local", "local-cpu"), default="mlx-local")
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
    audit_path = args.coverage_audit
    if not audit_path.is_absolute():
        audit_path = repo_root / audit_path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"[emit_5d_canvas_coverage_acquisition_plan] FATAL: output exists: {output_path}"
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            raise ExperimentQueueError(f"{audit_path}: expected JSON object")
        plan = build_coverage_acquisition_plan(
            coverage_audit=audit,
            work_order_id=args.work_order_id,
            coverage_audit_path=audit_path,
            repo_root=repo_root,
            canvas_path=args.canvas_path,
            output_root=args.output_root,
            mode=args.mode,
        )
    except (OSError, json.JSONDecodeError, ExperimentQueueError) as exc:
        print(f"[emit_5d_canvas_coverage_acquisition_plan] FATAL: {exc}")
        return 2
    _write_text_atomic(output_path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        "[emit_5d_canvas_coverage_acquisition_plan] OK: "
        f"wrote {args.work_order_id} plan to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
