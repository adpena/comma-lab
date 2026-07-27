#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the two-process, no-scorer G120 production-admission dry-run."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__).resolve()
ensure_repo_imports(REPO)

from tac.witness_control.g120_governed_clean_dry_run_gate_v1 import (  # noqa: E402
    run_g120_governed_clean_dry_run_v1,
)


def _absolute(value: str, *, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path.resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("checkpoint", "resume"),
        help="checkpoint must run first; resume must run in a distinct process",
    )
    parser.add_argument("--producer-run-dir", required=True)
    parser.add_argument(
        "--expected-launch-manifest-sha256",
        required=True,
    )
    parser.add_argument("--monitor-output-dir", required=True)
    parser.add_argument("--monitor-progress-dir", required=True)
    parser.add_argument("--measurement-cache-dir", required=True)
    parser.add_argument("--gate-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_g120_governed_clean_dry_run_v1(
        phase=args.phase,
        repo_root=REPO,
        producer_run_dir=_absolute(
            args.producer_run_dir,
            name="producer_run_dir",
        ),
        expected_launch_manifest_sha256=(
            args.expected_launch_manifest_sha256
        ),
        monitor_output_dir=_absolute(
            args.monitor_output_dir,
            name="monitor_output_dir",
        ),
        monitor_progress_dir=_absolute(
            args.monitor_progress_dir,
            name="monitor_progress_dir",
        ),
        measurement_cache_dir=_absolute(
            args.measurement_cache_dir,
            name="measurement_cache_dir",
        ),
        gate_dir=_absolute(args.gate_dir, name="gate_dir"),
    )
    print(
        json.dumps(
            {
                **dataclasses.asdict(result),
                "receipt_path": str(result.receipt_path),
                "next_action": (
                    "restart this tool with phase=resume and identical arguments"
                    if result.phase == "checkpoint"
                    else "supply this receipt path and SHA-256 to the G121 monitor"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
