#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest completed materializer chain manifests into an optimizer source queue.

This is a custody bridge, not a dispatcher. It scans explicit chain manifests,
materializer work-queue postconditions, or bounded chain roots; validates live
archive/artifact bytes; and writes a planning-only optimizer candidate queue.
Exact-ready promotion remains a separate explicit gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comma_lab.scheduler.materializer_chain_harvest import (  # noqa: E402
    harvest_materializer_chain_manifests,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--work-queue", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--chain-manifest", type=Path, action="append", default=[])
    parser.add_argument("--chain-root", type=Path, action="append", default=[])
    parser.add_argument("--source-queue-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--allow-unfinished-state",
        action="store_true",
        help=(
            "Do not require succeeded experiment_queue state for work-queue "
            "postcondition rows. Manifest custody is still validated."
        ),
    )
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Exit nonzero when no chain manifests validate into source queue rows.",
    )
    args = parser.parse_args(argv)

    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be >= 1 when provided")
    if (
        args.work_queue is None
        and not args.chain_manifest
        and not args.chain_root
    ):
        raise SystemExit(
            "provide at least one of --work-queue, --chain-manifest, or --chain-root"
        )

    result = harvest_materializer_chain_manifests(
        repo_root=args.repo_root,
        work_queue_path=args.work_queue,
        experiment_queue_state_path=args.state,
        experiment_queue_id=args.queue_id,
        chain_manifest_paths=args.chain_manifest,
        chain_roots=args.chain_root,
        require_succeeded_state=not args.allow_unfinished_state,
        top_k=args.top_k,
    )
    write_json(args.source_queue_out, result["source_queue"])
    write_json(args.report_out, result["report"])
    report = result["report"]
    print(
        f"harvested {report['accepted_manifest_count']}/"
        f"{report['unique_manifest_count']} materializer chain manifest(s); "
        f"source_queue={args.source_queue_out}; report={args.report_out}; "
        f"dispatch_ready={report['source_queue_dispatch_ready_count']}"
    )
    if args.require_accepted and int(report["accepted_manifest_count"]) < 1:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
