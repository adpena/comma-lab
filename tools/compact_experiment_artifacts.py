#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan or execute compaction of certified rebuildable experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from comma_lab.artifact_retention import (  # noqa: E402
    DEFAULT_RETENTION_KINDS,
    build_retention_plan,
    dumps_json,
    execute_retention_plan,
)


def _parse_bytes(value: str) -> int:
    raw = value.strip().lower()
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * multiplier)
    return int(raw)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("experiments/results")],
        help="Experiment roots to scan.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--include-kind",
        action="append",
        dest="include_kinds",
        choices=sorted(DEFAULT_RETENTION_KINDS | {"mlx_scorer_input_cache"}),
        help="Candidate kind to include. Defaults to safe raw/extracted scratch only.",
    )
    parser.add_argument("--min-bytes", type=_parse_bytes, default=1 << 30)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--journal-output",
        type=Path,
        help=(
            "Durable JSONL execution journal. Defaults to a sibling of "
            "--json-output, or .omx/state/artifact_retention_journals when "
            "executing to stdout."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--action", choices=("delete", "move"), default="delete")
    parser.add_argument(
        "--cold-store-root",
        type=Path,
        action="append",
        default=[],
        help="cold-store root for move execution; repeat for tiered water-bucket moves",
    )
    parser.add_argument(
        "--cold-store-reserve-gb",
        type=float,
        default=0.0,
        help="free GiB to preserve on each cold-store tier after planned moves",
    )
    return parser.parse_args(argv)


def _default_execution_journal_path(args: argparse.Namespace) -> Path:
    if args.journal_output is not None:
        return args.journal_output
    if args.json_output is not None:
        return args.json_output.with_suffix(args.json_output.suffix + ".journal.jsonl")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        args.repo_root
        / ".omx"
        / "state"
        / "artifact_retention_journals"
        / f"artifact_retention_execute_{stamp}.jsonl"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    include_kinds = None if args.include_kinds is None else set(args.include_kinds)
    plan = build_retention_plan(
        args.roots,
        repo_root=args.repo_root,
        include_kinds=include_kinds,
        min_bytes=args.min_bytes,
        exclude_paths=args.exclude,
    )
    payload = {
        "plan": plan.to_dict(),
        "execution": None,
    }
    if args.execute:
        journal_path = _default_execution_journal_path(args)
        payload["execution"] = execute_retention_plan(
            plan,
            action=args.action,
            cold_store_root=args.cold_store_root[0] if len(args.cold_store_root) == 1 else None,
            cold_store_roots=args.cold_store_root if len(args.cold_store_root) != 1 else None,
            cold_store_reserve_bytes=int(max(args.cold_store_reserve_gb, 0.0) * (1024**3)),
            journal_path=journal_path,
        )
    text = dumps_json(payload)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
